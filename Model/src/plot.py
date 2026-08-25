r"""
Erzeugt beschreibende Plots der Datengrundlage (kein Modelltraining).

Die Feature-Spalten in den Parquet-Dateien sind z-standardisiert; mit den in
`data/scaler.json` gespeicherten Mittelwerten/Streuungen werden A_EST und V_EST
auf ihre physikalischen Rohwerte zurueckgerechnet (raw = z * std + mean).
Das Label liegt bereits unstandardisiert in [-1, 1] vor.

Einheiten:
    - A_EST : mm/s^2 (Rohwert, keine weitere Umrechnung noetig)
    - V_EST : cm/s Rohwert -> km/h (Faktor 0.036)
    - label : Hebelstellung in [-1, 1] (positiv=Beschl., negativ=Bremsen)

Erzeugte Plots (nach reports/, 300 dpi):
    1. plot_a_est_histogram.png            Haeufigkeitsverteilung A_EST (alle Splits)
    2. plot_example_trip_{1,2,3}.png       Beispieltrips: V_EST / A_EST / Hebelstellung
    3. plot_label_overview.png             Kategorien-Anteil + Label-Verteilung
    4. plot_trip822_prediction.png         Vorhersage des besten Modells fuer trip822
                                           (wahres vs. prognostiziertes Label ueber den
                                           gesamten Trip)

Aufruf:
    .\.venv\Scripts\python.exe src\plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from model_comparison import FEATURES
from model import (
    BEST_MODEL_PATH,
    DROPOUT,
    HORIZON,
    LSTM_ARCHITECTURE,
    SEQ_LEN,
    LSTMRegressor,
    select_device,
)

# ---------------------------------------------------------------------------
# Pfade & Konstanten
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"

SPLITS = ["train", "val", "test"]
UNIT_COL = "unit"
TIME_COL = "tb"
LABEL_COL = "label"

CMS_TO_KMH = 0.036          # cm/s -> km/h
MIN_TRIP_MINUTES = 5.0      # Mindestdauer fuer Beispieltrips
MIN_A_EST_STD = 10.0        # Mindeststreuung A_EST (mm/s^2): filtert NaN-gefuellte Trips
N_EXAMPLE_TRIPS = 3
DPI = 300
SEED = 2

# Trip, den das beste Modell zur Veranschaulichung komplett vorhersagen soll.
PRED_TRIP_ID = "trip822"
PRED_BATCH_SIZE = 1024

CAT_COLORS = {"Bremsen": "#C44E52", "Halten": "#8C8C8C", "Beschleunigen": "#55A868"}


# ---------------------------------------------------------------------------
# Daten laden & de-standardisieren
# ---------------------------------------------------------------------------
def load_scaler() -> dict:
    with open(DATA_DIR / "scaler.json", "r", encoding="utf-8") as fh:
        return json.load(fh)["scaler"]


def destandardize(values: np.ndarray, scaler: dict, feature: str) -> np.ndarray:
    """Rechnet z-standardisierte Werte auf ihre Rohskala zurueck."""
    stats = scaler[feature]
    return values * stats["std"] + stats["mean"]


def load_all(scaler: dict) -> pd.DataFrame:
    """Laedt alle Splits und liefert Rohwerte fuer V_EST (km/h), A_EST (mm/s^2), label."""
    frames = []
    for split in SPLITS:
        path = DATA_DIR / f"{split}.parquet"
        if not path.exists():
            print(f"Warnung: {path} fehlt und wird uebersprungen.")
            continue
        df = pd.read_parquet(
            path, columns=[UNIT_COL, TIME_COL, "V_EST", "A_EST", LABEL_COL]
        )
        df["split"] = split
        frames.append(df)

    if not frames:
        raise SystemExit("Keine Parquet-Dateien in data/ gefunden.")

    full = pd.concat(frames, ignore_index=True)
    full["A_EST_raw"] = destandardize(full["A_EST"].to_numpy(np.float64), scaler, "A_EST")
    full["V_EST_kmh"] = (
        destandardize(full["V_EST"].to_numpy(np.float64), scaler, "V_EST") * CMS_TO_KMH
    )
    return full


# ---------------------------------------------------------------------------
# Plot 1: Haeufigkeitsverteilung A_EST
# ---------------------------------------------------------------------------
def plot_a_est_histogram(df: pd.DataFrame) -> None:
    a = df["A_EST_raw"].to_numpy(np.float64)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(a, bins=200, color="#4C72B0", edgecolor="none")
    ax.set_xlabel(r"Beschleunigung $A_{est}$ in mm/s$^2$")
    ax.set_ylabel("Anzahl (Haeufigkeit im Datensatz)")
    ax.set_title("Haeufigkeitsverteilung des Features $A_{est}$ (alle Splits)")
    ax.ticklabel_format(style="plain", axis="y")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = REPORTS_DIR / "plot_a_est_histogram.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"gespeichert: {out}")


# ---------------------------------------------------------------------------
# Plot 2: Beispieltrips (V_EST / A_EST / Hebelstellung)
# ---------------------------------------------------------------------------
def select_example_trips(df: pd.DataFrame, rng: np.random.Generator) -> list[str]:
    grouped = df.groupby(UNIT_COL)
    dur_min = grouped[TIME_COL].agg(lambda s: (s.max() - s.min()) / 60000.0)
    a_std = grouped["A_EST_raw"].std()
    # Trips mit NaN-gefuelltem (konstantem) A_EST-Signal ausschliessen.
    eligible = dur_min[(dur_min >= MIN_TRIP_MINUTES) & (a_std >= MIN_A_EST_STD)].index.to_numpy()
    if len(eligible) < N_EXAMPLE_TRIPS:
        raise SystemExit(
            f"Nur {len(eligible)} Trips >= {MIN_TRIP_MINUTES} min mit gueltigem "
            f"A_EST gefunden (benoetigt: {N_EXAMPLE_TRIPS})."
        )
    chosen = rng.choice(eligible, size=N_EXAMPLE_TRIPS, replace=False)
    return [str(u) for u in chosen]


def plot_example_trip(df: pd.DataFrame, unit: str, index: int) -> None:
    g = df[df[UNIT_COL].astype(str) == unit].sort_values(TIME_COL)
    t_min = (g[TIME_COL].to_numpy(np.float64) - g[TIME_COL].to_numpy(np.float64)[0]) / 60000.0

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(t_min, g["V_EST_kmh"], color="#4C72B0", lw=1.0)
    axes[0].set_ylabel("$V_{est}$ in km/h")
    axes[0].set_title(f"Beispieltrip {unit} (Dauer {t_min[-1]:.1f} min)")

    axes[1].plot(t_min, g["A_EST_raw"], color="#DD8452", lw=1.0)
    axes[1].axhline(0.0, color="#888", lw=0.8, ls="--")
    axes[1].set_ylabel(r"$A_{est}$ in mm/s$^2$")

    axes[2].plot(t_min, g[LABEL_COL], color="#55A868", lw=1.0)
    axes[2].axhline(0.0, color="#888", lw=0.8, ls="--")
    axes[2].set_ylim(-1.05, 1.05)
    axes[2].set_ylabel("Hebelstellung [-1, 1]")
    axes[2].set_xlabel("Zeit in min")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = REPORTS_DIR / f"plot_example_trip_{index}.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"gespeichert: {out}")


# ---------------------------------------------------------------------------
# Plot 3: Kategorien-Anteil + Label-Verteilung
# ---------------------------------------------------------------------------
def plot_label_overview(df: pd.DataFrame) -> None:
    label = df[LABEL_COL].to_numpy(np.float64)
    n = label.size
    shares = {
        "Bremsen": np.mean(label < 0.0),
        "Halten": np.mean(label == 0.0),
        "Beschleunigen": np.mean(label > 0.0),
    }

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6))

    cats = list(shares.keys())
    ax_l.bar(cats, [shares[c] for c in cats], color=[CAT_COLORS[c] for c in cats])
    ax_l.set_ylabel("Anteil an den Daten")
    ax_l.set_title("Hebelstellung nach Kategorie")
    ax_l.set_ylim(0, 1)
    for i, c in enumerate(cats):
        ax_l.text(i, shares[c] + 0.01, f"{shares[c] * 100:.1f}%", ha="center", va="bottom")
    ax_l.grid(True, axis="y", alpha=0.3)

    ax_r.hist(label, bins=200, range=(-1.0, 1.0), color="#4C72B0", edgecolor="none")
    ax_r.set_xlabel("Hebelstellung [-1, 1]")
    ax_r.set_ylabel("Anzahl (Haeufigkeit im Datensatz)")
    ax_r.set_title("Label-Verteilung")
    ax_r.set_xlim(-1.0, 1.0)
    ax_r.grid(True, alpha=0.3)

    fig.tight_layout()
    out = REPORTS_DIR / "plot_label_overview.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"gespeichert: {out} (n={n:,})")


# ---------------------------------------------------------------------------
# Plot 4: Vollstaendige Modellvorhersage fuer einen einzelnen Trip
# ---------------------------------------------------------------------------
def load_trip_features(trip_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sucht `trip_id` in allen Splits und liefert Features/Label/Zeit (chronologisch)."""
    for split in SPLITS:
        path = DATA_DIR / f"{split}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=[UNIT_COL, TIME_COL, *FEATURES, LABEL_COL])
        g = df[df[UNIT_COL].astype(str) == trip_id]
        if g.empty:
            continue
        g = g.sort_values(TIME_COL)
        feats = g[FEATURES].to_numpy(np.float32)
        label = g[LABEL_COL].to_numpy(np.float32)
        tb = g[TIME_COL].to_numpy(np.float64)
        print(f"{trip_id} gefunden in Split '{split}': {feats.shape[0]:,} Zeitschritte.")
        return feats, label, tb
    raise SystemExit(f"Trip '{trip_id}' in keinem Split gefunden.")


def load_best_model(device: torch.device) -> LSTMRegressor:
    """Laedt das beste Modell aus reports/model_checkpoints/best_model.pt."""
    source = BEST_MODEL_PATH
    if not source.exists():
        # Fallback: verschachtelter reports/reports-Pfad, falls so trainiert wurde.
        nested = REPORTS_DIR / "reports" / "model_checkpoints" / "best_model.pt"
        if nested.exists():
            source = nested
    if not source.exists():
        raise SystemExit(
            f"Kein bestes Modell gefunden: {BEST_MODEL_PATH}. "
            "Bitte zuerst src/model.py trainieren."
        )
    ckpt = torch.load(source, map_location="cpu")
    if ckpt.get("features") not in (None, FEATURES):
        raise SystemExit(
            "Checkpoint wurde nicht mit der aktuellen Feature-Auswahl trainiert. "
            "Bitte model.py erneut trainieren."
        )
    architecture = tuple(ckpt.get("architecture") or LSTM_ARCHITECTURE)
    state = ckpt.get("best_state") or ckpt.get("model_state")
    if state is None:
        raise SystemExit("Checkpoint enthaelt weder 'best_state' noch 'model_state'.")
    model = LSTMRegressor(len(FEATURES), architecture, DROPOUT)
    model.load_state_dict(state)
    model.to(device).eval()
    print(
        f"Modell geladen: LSTM {'-'.join(map(str, architecture))} "
        f"(Val-Loss={ckpt.get('best_val_loss', float('nan')):.6f}, "
        f"Epoche {ckpt.get('best_epoch', '?')})."
    )
    return model


@torch.no_grad()
def predict_full_trip(
    model: LSTMRegressor, feats: np.ndarray, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Vorhersage fuer JEDEN moeglichen Zielzeitpunkt (Stride 1) ueber den ganzen Trip.

    Fuer einen Zielzeitpunkt t wird die History [t-HORIZON-SEQ_LEN+1 .. t-HORIZON]
    verwendet (identisch zum Training). Liefert die Zielindizes und die Vorhersagen.
    """
    n = feats.shape[0]
    win = SEQ_LEN + HORIZON
    starts = np.arange(0, n - win + 1)
    if starts.size == 0:
        raise SystemExit(
            f"Trip zu kurz ({n} Schritte) fuer SEQ_LEN={SEQ_LEN} + HORIZON={HORIZON}."
        )
    target_t = starts + SEQ_LEN - 1 + HORIZON

    preds: list[np.ndarray] = []
    for i in range(0, starts.size, PRED_BATCH_SIZE):
        chunk = starts[i : i + PRED_BATCH_SIZE]
        batch = np.stack([feats[s : s + SEQ_LEN] for s in chunk])
        x = torch.from_numpy(batch).to(device)
        out = model(x).detach().cpu().numpy().reshape(-1)
        preds.append(out)
    return target_t, np.concatenate(preds).astype(np.float64)


def plot_trip_prediction(
    trip_id: str,
    tb: np.ndarray,
    label_true: np.ndarray,
    target_t: np.ndarray,
    pred_val: np.ndarray,
) -> None:
    t_min = (tb - tb[0]) / 60000.0

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(t_min, label_true, color="#55A868", lw=1.0, label="Label (wahr)")
    ax.plot(t_min[target_t], pred_val, color="#C44E52", lw=1.0,
            label="Label (Vorhersage)")
    ax.axhline(0.0, color="#888", lw=0.8, ls="--")
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("Hebelstellung [-1, 1]")
    ax.set_xlabel("Zeit in min")
    ax.set_title(
        f"Vorhersage bestes Modell fuer {trip_id} (Dauer {t_min[-1]:.1f} min)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    out = REPORTS_DIR / f"plot_{trip_id}_prediction.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"gespeichert: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    scaler = load_scaler()
    df = load_all(scaler)
    print(f"Daten geladen: {len(df):,} Zeilen, {df[UNIT_COL].nunique():,} Trips.")

    plot_a_est_histogram(df)

    trips = select_example_trips(df, rng)
    for i, unit in enumerate(trips, start=1):
        plot_example_trip(df, unit, i)

    plot_label_overview(df)

    # Plot 4: Vollstaendige Modellvorhersage fuer einen einzelnen Trip.
    device = select_device()
    model = load_best_model(device)
    feats, label_true, tb = load_trip_features(PRED_TRIP_ID)
    target_t, pred_val = predict_full_trip(model, feats, device)
    plot_trip_prediction(PRED_TRIP_ID, tb, label_true, target_t, pred_val)


if __name__ == "__main__":
    main()
