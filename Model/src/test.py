r"""
Finale Testset-Evaluation des finalen Modells (Iteration 5).

Laedt das beste Modell aus dem Training von `model.py`
(`reports/model_checkpoints/best_model.pt`; Fallback: Feld `best_state` im
Resume-Checkpoint `training_checkpoint.pt`) und erzeugt alle Auswertungen aus
Kapitel 7 der Bachelorarbeit, geordnet nach den Abschnitten 7.1 bis 7.4:

  7.1 Finale Testset-Evaluation
      (1) Ergebnistabelle Testset (finales Modell) vs. Persistenz-Baseline
      (2) Scatter-Plot Vorhersage vs. Ground Truth (+ 45-Grad + Regressionslinie)
      (3) Residuen-Histogramm (y_pred - y_true)
  7.2 Klassenweise Analyse
      (4) Richtungs-Konfusionsmatrix (zeilennormiert, bremsen/halten/beschl.)
      (5) Klassenweise Metriken (Precision / Recall / F1 pro Richtungsklasse)
      (6) Histogramm-Overlay: Verteilung y_true vs. y_pred
  7.3 Qualitative Analyse anhand von Beispielfahrten
      (7) Zeitreihen-Overlay fuer ausgewaehlte Test-Trips
          (Geschwindigkeit, wahres Label, pr-adiziertes Label)
      (8) Auswahlkriterium/Kurzbeschreibung der gewaehlten Trips (Tabelle)
  7.4 Generalisierung ueber Trip-Charakteristika
      (9) Scatter: MAE pro Trip vs. Trip-Dauer
      (10) Boxplot: Fehlerverteilung Beharrung vs. Uebergang

Datenaufbereitung/Windowing sind identisch zum Training (`model.py`):
dieselben Konstanten SEQ_LEN/HORIZON/STRIDE und dieselbe SequenceDataset-Logik.

Ausgaben in `reports/`:
    - test_evaluation_report.html   (self-contained, Plots als base64)
    - test_eval_*.png               (Einzelplots)
    - test_evaluation_metrics.csv   (Gesamt- + klassenweise + Trip-Metriken)
    - test_evaluation_metrics.json

Aufruf:
    .\.venv\Scripts\python.exe src\test.py
"""

from __future__ import annotations

import base64
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from model_comparison import (  # noqa: E402
    DATA_DIR,
    DIR_DEADBAND,
    FEATURES,
    LABEL_COL,
    TIME_COL,
    TOL,
    UNIT_COL,
    SequenceDataset,
    load_split,
    regression_metrics,
)
from model import (  # noqa: E402
    DIRECTION_CLASS_NAMES,
    DROPOUT,
    LSTM_ARCHITECTURE,
    BEST_MODEL_PATH,
    CHECKPOINT_PATH,
    Config,
    LSTMRegressor,
    direction_confusion,
    plot_direction_confusion,
    select_device,
)


# ---------------------------------------------------------------------------
# Konstanten fuer diese Auswertung
# ---------------------------------------------------------------------------
REPORTS_DIR = DATA_DIR.parent / "reports"
REPORT_NAME = "test_evaluation_report.html"
METRICS_CSV_NAME = "test_evaluation_metrics.csv"
METRICS_JSON_NAME = "test_evaluation_metrics.json"

PERIOD_S = 0.2  # 5 Hz -> 200 ms pro Schritt

# Kriterium fuer "Uebergang" (aktive Brems-/Anfahrphase) vs. "Beharrung":
# Betrag der z-standardisierten geschaetzten Beschleunigung A_EST am Zielzeitpunkt.
A_EST_INDEX = FEATURES.index("A_EST")
V_EST_INDEX = FEATURES.index("V_EST")
TRANSITION_A_EST_Z = 0.5

# Nur Trips mit mindestens so vielen Fenstern kommen fuer die Trip-Auswahl
# (Zeitreihen-Overlay, MAE-vs-Dauer) in Frage.
MIN_TRIP_WINDOWS = 30

# Anzahl zusaetzlicher, zufaellig gezogener Vorhersage-vs-Label-Zeitreihen.
N_RANDOM_TRIP_PLOTS = 10

# Zufaellige Teilstichprobe fuer den Scatter-Plot (nur Rendering-Performance).
SCATTER_MAX_POINTS = 20000

EVAL_BATCH_SIZE = 1024


# ---------------------------------------------------------------------------
# Modell laden
# ---------------------------------------------------------------------------
def load_final_model(device: torch.device) -> tuple[LSTMRegressor, dict]:
    source = BEST_MODEL_PATH if BEST_MODEL_PATH.exists() else CHECKPOINT_PATH
    if not source.exists():
        raise SystemExit(
            f"Kein Modell gefunden: weder {BEST_MODEL_PATH} noch {CHECKPOINT_PATH}. "
            "Bitte zuerst src/model.py trainieren."
        )
    ckpt = torch.load(source, map_location="cpu")
    checkpoint_features = ckpt.get("features")
    if checkpoint_features != FEATURES:
        raise SystemExit(
            "Checkpoint wurde nicht mit der aktuellen 13-Feature-Auswahl trainiert. "
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
        f"Modell geladen aus {source.name}: LSTM {'-'.join(map(str, architecture))} "
        f"(bester Val-Loss={ckpt.get('best_val_loss', float('nan')):.6f} "
        f"aus Epoche {ckpt.get('best_epoch', '?')})"
    )
    return model, ckpt


# ---------------------------------------------------------------------------
# Testdaten laden (mit Trip-IDs, unnormalisierter Geschwindigkeit)
# ---------------------------------------------------------------------------
def load_v_est_scaler() -> tuple[float, float]:
    scaler_path = DATA_DIR / "scaler.json"
    info = json.loads(scaler_path.read_text(encoding="utf-8"))
    s = info["scaler"]["V_EST"]
    return float(s["mean"]), float(s["std"])


def load_test_trips(cfg: Config):
    """Liefert pro Test-Trip Feature-/Label-/Zeit-Arrays inkl. Trip-ID.

    Kein Shuffle und keine Fraction-Auswahl: es werden ALLE Test-Trips
    ausgewertet (finale Testset-Evaluation).
    """
    df = load_split("test")
    unit_series = df[UNIT_COL].astype(str)

    v_mean, v_std = load_v_est_scaler()

    units: list[str] = []
    trips: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    speeds_kmh: list[np.ndarray] = []
    times_s: list[np.ndarray] = []

    for unit, g in df.groupby(unit_series, sort=False):
        g = g.sort_values(TIME_COL)
        feats = g[FEATURES].to_numpy(dtype=np.float32)
        lab = g[LABEL_COL].to_numpy(dtype=np.float32)
        tb = g[TIME_COL].to_numpy(dtype=np.float64)

        # V_EST ist z-standardisiert -> zurueck auf cm/s, dann km/h (1 cm/s = 0.036 km/h).
        v_cm_s = feats[:, V_EST_INDEX] * v_std + v_mean
        speed_kmh = v_cm_s * 0.036
        t_s = (tb - tb[0]) / 1000.0

        units.append(str(unit))
        trips.append(feats)
        labels.append(lab)
        speeds_kmh.append(speed_kmh.astype(np.float32))
        times_s.append(t_s.astype(np.float64))

    print(f"Test-Trips geladen: {len(units)}")
    return units, trips, labels, speeds_kmh, times_s


# ---------------------------------------------------------------------------
# Vorhersage ueber das gesamte Testset
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_dataset(model: LSTMRegressor, ds: SequenceDataset, device: torch.device):
    loader = DataLoader(ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=0)
    total = len(loader)
    yt, yp, ypr = [], [], []
    for bi, (x, y, y_prev) in enumerate(loader, start=1):
        x = x.to(device)
        pred = model(x).detach().cpu().numpy()
        yt.append(y.numpy())
        yp.append(pred)
        ypr.append(y_prev.numpy())
        if bi % 20 == 0 or bi == total:
            print(f"  Vorhersage-Batch {bi}/{total}")
    return (
        np.concatenate(yt).astype(np.float64),
        np.concatenate(yp).astype(np.float64),
        np.concatenate(ypr).astype(np.float64),
    )


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------
def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def classwise_metrics(cm: np.ndarray) -> list[dict]:
    """Precision/Recall/F1/Support je Richtungsklasse aus der Konfusionsmatrix.

    cm: Zeile = reale Klasse, Spalte = vorhergesagte Klasse.
    """
    col_sum = cm.sum(axis=0)
    row_sum = cm.sum(axis=1)
    out: list[dict] = []
    for c in range(cm.shape[0]):
        tp = float(cm[c, c])
        fp = float(col_sum[c] - tp)
        fn = float(row_sum[c] - tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        out.append({
            "class": DIRECTION_CLASS_NAMES[c],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(row_sum[c]),
        })
    return out


# ---------------------------------------------------------------------------
# (2) Scatter Vorhersage vs. Ground Truth
# ---------------------------------------------------------------------------
def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path,
                 rng: np.random.Generator) -> str:
    if y_true.size > SCATTER_MAX_POINTS:
        idx = rng.choice(y_true.size, size=SCATTER_MAX_POINTS, replace=False)
        xs, ys = y_true[idx], y_pred[idx]
    else:
        xs, ys = y_true, y_pred

    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    slope, intercept = np.polyfit(y_true, y_pred, 1)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(xs, ys, s=4, alpha=0.15, color="#4C72B0", edgecolors="none",
               label=f"Testfenster (n={y_true.size:,})")
    ax.plot([lo, hi], [lo, hi], color="#333", lw=1.5, ls="--",
            label="perfekte Vorhersage (45\u00b0)")
    line_x = np.array([lo, hi])
    ax.plot(line_x, slope * line_x + intercept, color="#C44E52", lw=1.8,
            label=f"Regression (y={slope:.2f}x{intercept:+.2f})")
    ax.set_xlabel("wahres Label")
    ax.set_ylabel("prädiziertes Label")
    ax.set_title("Vorhersage vs. Ground Truth (Test)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", "box")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64


# ---------------------------------------------------------------------------
# (3) Residuen-Histogramm
# ---------------------------------------------------------------------------
def plot_residual_hist(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path) -> str:
    resid = y_pred - y_true
    mean_r = float(resid.mean())
    median_r = float(np.median(resid))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(resid, bins=80, color="#4C72B0", alpha=0.85)
    ax.axvline(0.0, color="#333", lw=1.5, ls="--", label="0 (kein Fehler)")
    ax.axvline(mean_r, color="#C44E52", lw=1.5,
               label=f"Mittelwert={mean_r:+.4f}")
    ax.set_xlabel("Residuum  ŷ − y")
    ax.set_ylabel("Anzahl Fenster")
    ax.set_title("Residuen-Histogramm (Test)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.text(0.99, -0.02, f"Median={median_r:+.4f}", ha="right", fontsize=8, color="#666")
    fig.tight_layout()
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64


# ---------------------------------------------------------------------------
# (6) Histogramm-Overlay y_true vs. y_pred
# ---------------------------------------------------------------------------
def plot_overlay_hist(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path) -> str:
    bins = np.linspace(-1.0, 1.0, 81)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(y_true, bins=bins, color="#4C72B0", alpha=0.55, label="wahre Labels")
    ax.hist(y_pred, bins=bins, color="#C44E52", alpha=0.55, label="Vorhersagen")
    ax.axvline(-DIR_DEADBAND, color="#888", lw=0.8, ls=":")
    ax.axvline(DIR_DEADBAND, color="#888", lw=0.8, ls=":")
    ax.set_xlabel("Label")
    ax.set_ylabel("Anzahl Fenster")
    ax.set_title("Verteilung wahre Labels vs. Vorhersagen (Test)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64


# ---------------------------------------------------------------------------
# (7) Zeitreihen-Overlay pro Trip
# ---------------------------------------------------------------------------
def plot_trip_timeseries(t_full: np.ndarray, speed_kmh: np.ndarray,
                         label_true: np.ndarray, pred_t: np.ndarray,
                         pred_val: np.ndarray, title: str, save_path: Path) -> str:
    fig, ax1 = plt.subplots(figsize=(11, 4.5))
    ax2 = ax1.twinx()

    l_speed, = ax2.plot(t_full, speed_kmh, color="#999999", lw=1.0, alpha=0.8,
                        label="Geschwindigkeit")
    l_true, = ax1.plot(t_full, label_true, color="#4C72B0", lw=1.6,
                       label="Label (wahr)")
    l_pred, = ax1.plot(pred_t, pred_val, color="#C44E52", lw=1.2, marker=".",
                       ms=3, label="Label (Vorhersage)")

    ax1.axhline(DIR_DEADBAND, color="#bbb", lw=0.7, ls=":")
    ax1.axhline(-DIR_DEADBAND, color="#bbb", lw=0.7, ls=":")
    ax1.set_ylim(-1.05, 1.05)
    ax1.set_xlabel("Zeit [s]")
    ax1.set_ylabel("Stellsignal (Label)")
    ax2.set_ylabel("Geschwindigkeit [km/h]")
    ax1.set_title(title)
    ax1.grid(alpha=0.3)
    ax1.legend(handles=[l_true, l_pred, l_speed], loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64


# ---------------------------------------------------------------------------
# (9) Scatter MAE pro Trip vs. Trip-Dauer
# ---------------------------------------------------------------------------
def plot_trip_mae_vs_duration(durations_min: np.ndarray, maes: np.ndarray,
                              save_path: Path) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(durations_min, maes, s=28, alpha=0.6, color="#55A868",
               edgecolors="#2f5c3a")
    if durations_min.size >= 2:
        slope, intercept = np.polyfit(durations_min, maes, 1)
        xline = np.array([durations_min.min(), durations_min.max()])
        ax.plot(xline, slope * xline + intercept, color="#C44E52", lw=1.6,
                label=f"Trend (Steigung={slope:+.4f}/min)")
        ax.legend(fontsize=9)
    ax.set_xlabel("Trip-Dauer [min]")
    ax.set_ylabel("MAE des Trips")
    ax.set_title("Fehler pro Trip vs. Trip-Dauer (Test)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64


# ---------------------------------------------------------------------------
# (10) Boxplot Fehlerverteilung Beharrung vs. Uebergang
# ---------------------------------------------------------------------------
def plot_error_by_phase(abs_err_steady: np.ndarray, abs_err_transition: np.ndarray,
                        save_path: Path) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    data = [abs_err_steady, abs_err_transition]
    labels = [
        f"Beharrung\n(n={abs_err_steady.size:,})",
        f"Übergang\n(n={abs_err_transition.size:,})",
    ]
    bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True,
                    medianprops=dict(color="#222", lw=1.5))
    for patch, color in zip(bp["boxes"], ["#4C72B0", "#C44E52"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("absoluter Fehler  |ŷ − y|")
    ax.set_title("Fehlerverteilung nach Fahrphase (Test)")
    ax.grid(alpha=0.3, axis="y")
    fig.text(0.99, -0.02,
             f"Übergang: |A_EST(z)| > {TRANSITION_A_EST_Z} am Zielzeitpunkt",
             ha="right", fontsize=8, color="#666")
    fig.tight_layout()
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64


# ---------------------------------------------------------------------------
# Per-Trip-Aufbereitung
# ---------------------------------------------------------------------------
def build_per_trip(ds: SequenceDataset, cfg: Config, units: list[str],
                   y_true: np.ndarray, y_pred: np.ndarray):
    """Gruppiert die Fenster-Vorhersagen nach Trip und berechnet Trip-Metriken."""
    per_trip: dict[int, dict] = {}
    for k, (ti, start) in enumerate(ds.index):
        target_t = start + cfg.seq_len - 1 + cfg.horizon
        d = per_trip.setdefault(ti, {"target_t": [], "yt": [], "yp": []})
        d["target_t"].append(target_t)
        d["yt"].append(y_true[k])
        d["yp"].append(y_pred[k])

    rows = []
    for ti, d in per_trip.items():
        yt = np.asarray(d["yt"], dtype=np.float64)
        yp = np.asarray(d["yp"], dtype=np.float64)
        target_t = np.asarray(d["target_t"], dtype=np.int64)
        order = np.argsort(target_t)
        yt, yp, target_t = yt[order], yp[order], target_t[order]

        mae = float(np.mean(np.abs(yp - yt)))
        n_win = yt.size
        brake_share = float(np.mean(yt < -DIR_DEADBAND))
        accel_share = float(np.mean(yt > DIR_DEADBAND))
        hold_share = float(np.mean(np.abs(yt) <= DIR_DEADBAND))
        duration_s = float((target_t.max() - target_t.min() + 1) * PERIOD_S)

        rows.append({
            "trip_index": ti,
            "trip_id": units[ti],
            "n_windows": n_win,
            "duration_s": duration_s,
            "duration_min": duration_s / 60.0,
            "mae": mae,
            "brake_share": brake_share,
            "accel_share": accel_share,
            "hold_share": hold_share,
            "target_t": target_t,
            "yt": yt,
            "yp": yp,
        })
    return rows


def select_trips(trip_rows: list[dict]) -> list[tuple[str, dict]]:
    """Waehlt eine gut, eine schlecht und eine bremsintensive Testfahrt aus."""
    eligible = [r for r in trip_rows if r["n_windows"] >= MIN_TRIP_WINDOWS]
    if not eligible:
        eligible = list(trip_rows)

    best = min(eligible, key=lambda r: r["mae"])
    worst = max(eligible, key=lambda r: r["mae"])
    braker = max(eligible, key=lambda r: r["brake_share"])

    chosen: list[tuple[str, dict]] = []
    seen: set[int] = set()
    for reason, r in [
        ("gut vorhergesagt (min. MAE)", best),
        ("schlecht vorhergesagt (max. MAE)", worst),
        ("bremsintensiv (max. Bremsanteil)", braker),
    ]:
        if r["trip_index"] not in seen:
            chosen.append((reason, r))
            seen.add(r["trip_index"])
    return chosen


def select_random_trips(trip_rows: list[dict], rng: np.random.Generator,
                        n: int, exclude: set[int]) -> list[dict]:
    """Zieht n zufaellige Testfahrten (mit genug Fenstern), ohne die ausgeschlossenen."""
    eligible = [r for r in trip_rows
                if r["n_windows"] >= MIN_TRIP_WINDOWS and r["trip_index"] not in exclude]
    if not eligible:
        eligible = [r for r in trip_rows if r["trip_index"] not in exclude]
    take = min(n, len(eligible))
    idx = rng.choice(len(eligible), size=take, replace=False)
    return [eligible[i] for i in sorted(idx)]


# ---------------------------------------------------------------------------
# Konsolen-Ausgabe (geordnet nach Kapitel 7)
# ---------------------------------------------------------------------------
def print_console_report(test_metrics, baseline_metrics, cm, cls_metrics,
                         chosen, trip_rows, n_steady, n_transition,
                         err_steady, err_transition):
    line = "=" * 72

    print("\n" + line)
    print("7.1  FINALE TESTSET-EVALUATION")
    print(line)
    print("(1) Ergebnistabelle: finales Modell (Test) vs. Persistenz-Baseline (Test)")
    hdr = f"  {'':22s} {'R2':>9s} {'RMSE':>9s} {'MAE':>9s} {'MSE':>11s} {'Dir-Acc':>9s} {'Tol-Acc':>9s}"
    print(hdr)
    for name, m in (("Finales Modell (Test)", test_metrics),
                    ("Persistenz (Test)", baseline_metrics)):
        print(f"  {name:22s} {m['r2']:9.4f} {m['rmse']:9.4f} {m['mae']:9.4f} "
              f"{m['mse']:11.6f} {m['dir_acc_pct']:8.2f}% {m['tol_acc_pct']:8.2f}%")
    print("\n(2) Scatter Vorhersage vs. Ground Truth  -> PNG/Report")
    print("(3) Residuen-Histogramm                  -> PNG/Report")

    print("\n" + line)
    print("7.2  KLASSENWEISE ANALYSE")
    print(line)
    print("(4) Richtungs-Konfusionsmatrix (Zeile=real, Spalte=predicted, zeilennormiert):")
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm * 100.0, row_sums, out=np.zeros_like(cm, dtype=float),
                       where=row_sums > 0)
    print(f"  {'real\\pred':>14s} " + " ".join(f"{n:>16s}" for n in DIRECTION_CLASS_NAMES))
    for i, name in enumerate(DIRECTION_CLASS_NAMES):
        cells = " ".join(f"{cm_pct[i, j]:6.1f}% (n={cm[i, j]:>6,})" for j in range(3))
        print(f"  {name:>14s} {cells}")

    print("\n(5) Klassenweise Metriken (Precision / Recall / F1 / Support):")
    print(f"  {'Klasse':>16s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}")
    for row in cls_metrics:
        print(f"  {row['class']:>16s} {row['precision']:10.3f} {row['recall']:10.3f} "
              f"{row['f1']:10.3f} {row['support']:10,}")
    macro_p = np.mean([r["precision"] for r in cls_metrics])
    macro_r = np.mean([r["recall"] for r in cls_metrics])
    macro_f = np.mean([r["f1"] for r in cls_metrics])
    print(f"  {'Macro-Avg':>16s} {macro_p:10.3f} {macro_r:10.3f} {macro_f:10.3f}")
    print("\n(6) Histogramm-Overlay wahre Labels vs. Vorhersagen -> PNG/Report")

    print("\n" + line)
    print("7.3  QUALITATIVE ANALYSE ANHAND VON BEISPIELFAHRTEN")
    print(line)
    print("(7) Zeitreihen-Overlay ausgewaehlter Trips -> PNG/Report")
    print("(8) Auswahl der Trips:")
    print(f"  {'Trip-ID':>16s} {'Dauer[min]':>11s} {'#Fenster':>9s} "
          f"{'Brems%':>8s} {'Beschl%':>8s} {'MAE':>8s}  Grund")
    for reason, r in chosen:
        print(f"  {r['trip_id']:>16s} {r['duration_min']:11.2f} {r['n_windows']:9,} "
              f"{100.0 * r['brake_share']:7.1f}% {100.0 * r['accel_share']:7.1f}% "
              f"{r['mae']:8.4f}  {reason}")

    print("\n" + line)
    print("7.4  GENERALISIERUNG UEBER TRIP-CHARAKTERISTIKA")
    print(line)
    maes = np.array([r["mae"] for r in trip_rows])
    durs = np.array([r["duration_min"] for r in trip_rows])
    corr = float(np.corrcoef(durs, maes)[0, 1]) if len(trip_rows) >= 2 else float("nan")
    print(f"(9) MAE pro Trip vs. Trip-Dauer  (Trips={len(trip_rows)}, "
          f"Korrelation Dauer/MAE={corr:+.3f})  -> PNG/Report")
    print("(10) Fehlerverteilung Beharrung vs. Uebergang "
          f"(|A_EST(z)| > {TRANSITION_A_EST_Z}):")
    print(f"  Beharrung : n={n_steady:,}  MAE={err_steady.mean():.4f}  "
          f"Median={np.median(err_steady):.4f}")
    print(f"  Übergang  : n={n_transition:,}  MAE={err_transition.mean():.4f}  "
          f"Median={np.median(err_transition):.4f}")


# ---------------------------------------------------------------------------
# HTML-Report
# ---------------------------------------------------------------------------
def build_html(b64: dict, test_metrics, baseline_metrics, cm, cls_metrics,
               chosen, trip_stats, phase_stats, meta) -> str:
    def mrow(label, m, dim=False):
        style = " style='color:#888'" if dim else ""
        return (
            f"<tr{style}><td><b>{label}</b></td>"
            f"<td>{m['r2']:.4f}</td><td>{m['rmse']:.4f}</td><td>{m['mae']:.4f}</td>"
            f"<td>{m['mse']:.6f}</td><td>{m['dir_acc_pct']:.2f}%</td>"
            f"<td>{m['tol_acc_pct']:.2f}%</td></tr>"
        )

    main_rows = "\n".join([
        mrow("Finales Modell (Test)", test_metrics),
        mrow("Persistenz-Baseline (Test)", baseline_metrics, dim=True),
    ])

    cls_rows = "\n".join(
        f"<tr><td><b>{r['class']}</b></td><td>{r['precision']:.3f}</td>"
        f"<td>{r['recall']:.3f}</td><td>{r['f1']:.3f}</td><td>{r['support']:,}</td></tr>"
        for r in cls_metrics
    )
    macro_p = np.mean([r["precision"] for r in cls_metrics])
    macro_r = np.mean([r["recall"] for r in cls_metrics])
    macro_f = np.mean([r["f1"] for r in cls_metrics])
    cls_rows += (
        f"\n<tr style='color:#888'><td><b>Macro-Avg</b></td><td>{macro_p:.3f}</td>"
        f"<td>{macro_r:.3f}</td><td>{macro_f:.3f}</td><td></td></tr>"
    )

    trip_rows_html = "\n".join(
        f"<tr><td>{r['trip_id']}</td><td>{r['duration_min']:.2f}</td>"
        f"<td>{r['n_windows']:,}</td><td>{100.0 * r['brake_share']:.1f}%</td>"
        f"<td>{100.0 * r['accel_share']:.1f}%</td><td>{r['mae']:.4f}</td>"
        f"<td style='text-align:left'>{reason}</td></tr>"
        for reason, r in chosen
    )

    ts_imgs = "\n".join(
        f"<h3>{reason} &ndash; Trip {r['trip_id']}</h3>"
        f"<img src=\"data:image/png;base64,{b64['timeseries'][i]}\"/>"
        for i, (reason, r) in enumerate(chosen)
    )

    ts_random_imgs = "\n".join(
        f"<h3>Trip {r['trip_id']}</h3>"
        f"<img src=\"data:image/png;base64,{b}\"/>"
        for r, b in b64.get("timeseries_random", [])
    )

    return f"""<!DOCTYPE html>
<html lang="de"><head><meta charset="utf-8">
<title>Finale Testset-Evaluation</title>
<style>
 body {{ font-family: Segoe UI, Arial, sans-serif; margin: 32px; color:#222; max-width:1100px; }}
 h1 {{ border-bottom: 3px solid #4C72B0; padding-bottom:6px; }}
 h2 {{ margin-top: 34px; color:#333; border-bottom:1px solid #ddd; padding-bottom:4px; }}
 h3 {{ margin-top: 18px; color:#444; }}
 table {{ border-collapse: collapse; margin: 12px 0; font-size: 14px; }}
 th, td {{ border: 1px solid #ccc; padding: 6px 12px; text-align: right; }}
 th {{ background:#f0f3f7; }}
 td:first-child, th:first-child {{ text-align: left; }}
 img {{ max-width: 100%; border:1px solid #eee; border-radius:6px; margin: 6px 0 18px; }}
 .cfg {{ background:#f7f9fb; border:1px solid #e2e8f0; border-radius:6px; padding:10px 16px; font-size:13px; }}
 .note {{ color:#666; font-size:13px; }}
</style></head>
<body>
<h1>Finale Testset-Evaluation &ndash; LSTM {'-'.join(map(str, LSTM_ARCHITECTURE))}</h1>
<div class="cfg">
Modell: bestes Checkpoint (Val-Loss={meta['best_val_loss']:.6f}, Epoche {meta['best_epoch']}).<br>
History H={meta['seq_len']} Schritte ({meta['seq_len'] * PERIOD_S:.0f}&nbsp;s @ 5&nbsp;Hz)
&rarr; Label bei t+{meta['horizon']} ({int(meta['horizon'] * PERIOD_S * 1000)}&nbsp;ms Horizont).<br>
Test-Trips: {meta['n_trips']} &middot; Test-Fenster: {meta['n_windows']:,} &middot;
Stride={meta['stride']} &middot; Device: {meta['device']}.
</div>

<h2>7.1 Finale Testset-Evaluation</h2>
<h3>(1) Ergebnistabelle</h3>
<table>
<tr><th>Split</th><th>R&sup2;</th><th>RMSE</th><th>MAE</th><th>MSE</th>
<th>Richtungs-Acc</th><th>Toleranz-Acc</th></tr>
{main_rows}
</table>
<p class="note">Richtungs-Acc: 3-Klassen (bremsen/halten/beschl.) mit Deadband
&plusmn;{DIR_DEADBAND}. Toleranz-Acc: Anteil |ŷ&minus;y| &le; {TOL}.
Persistenz nutzt das letzte beobachtete Label als Vorhersage.</p>

<h3>(2) Scatter Vorhersage vs. Ground Truth</h3>
<img src="data:image/png;base64,{b64['scatter']}"/>

<h3>(3) Residuen-Histogramm</h3>
<img src="data:image/png;base64,{b64['residual']}"/>

<h2>7.2 Klassenweise Analyse</h2>
<h3>(4) Richtungs-Konfusionsmatrix (zeilennormiert)</h3>
<img src="data:image/png;base64,{b64['confusion']}"/>

<h3>(5) Klassenweise Metriken</h3>
<table>
<tr><th>Klasse</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>
{cls_rows}
</table>
<p class="note">Precision je Klasse c = Anteil korrekter unter allen als c
vorhergesagten Fenstern; Recall = Anteil korrekt erkannter unter allen wahren c.</p>

<h3>(6) Verteilung wahre Labels vs. Vorhersagen</h3>
<img src="data:image/png;base64,{b64['overlay']}"/>

<h2>7.3 Qualitative Analyse anhand von Beispielfahrten</h2>
<h3>(8) Auswahl der Trips</h3>
<table>
<tr><th>Trip-ID</th><th>Dauer [min]</th><th>#Fenster</th><th>Brems&nbsp;%</th>
<th>Beschl&nbsp;%</th><th>MAE</th><th>Grund</th></tr>
{trip_rows_html}
</table>
<h3>(7) Zeitreihen-Overlay</h3>
{ts_imgs}

<h3>Weitere Vorhersage-vs-Label-Zeitreihen</h3>
{ts_random_imgs}

<h2>7.4 Generalisierung ueber Trip-Charakteristika</h2>
<h3>(9) MAE pro Trip vs. Trip-Dauer</h3>
<p class="note">Korrelation Dauer/MAE = {trip_stats['corr']:+.3f} ueber {trip_stats['n']} Test-Trips.</p>
<img src="data:image/png;base64,{b64['mae_vs_dur']}"/>

<h3>(10) Fehlerverteilung Beharrung vs. Uebergang</h3>
<p class="note">Uebergang = |A_EST(z)| &gt; {TRANSITION_A_EST_Z} am Zielzeitpunkt.
Beharrung: n={phase_stats['n_steady']:,}, MAE={phase_stats['mae_steady']:.4f}.
Uebergang: n={phase_stats['n_transition']:,}, MAE={phase_stats['mae_transition']:.4f}.</p>
<img src="data:image/png;base64,{b64['phase']}"/>

<p class="note">Erzeugt am {meta['timestamp']}.</p>
</body></html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t_start = time.time()
    cfg = Config()  # seq_len/horizon/stride/fraction wie im Training (model.py)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    device = select_device()
    model, ckpt = load_final_model(device)

    units, trips, labels, speeds_kmh, times_s = load_test_trips(cfg)
    ds = SequenceDataset(trips, labels, cfg)
    print(f"Test-Fenster: {len(ds):,}")
    if len(ds) == 0:
        raise SystemExit("Keine Test-Fenster gebaut - SEQ_LEN/HORIZON/STRIDE pruefen.")

    print("\nVorhersage ueber das gesamte Testset ...")
    y_true, y_pred, y_prev = predict_dataset(model, ds, device)

    # --- Metriken ---
    test_metrics = regression_metrics(y_true, y_pred)
    baseline_metrics = regression_metrics(y_true, y_prev)
    cm = direction_confusion(y_true, y_pred)
    cls_metrics = classwise_metrics(cm)

    # --- Per-Trip ---
    trip_rows = build_per_trip(ds, cfg, units, y_true, y_pred)
    chosen = select_trips(trip_rows)
    maes = np.array([r["mae"] for r in trip_rows])
    durs_min = np.array([r["duration_min"] for r in trip_rows])
    corr = float(np.corrcoef(durs_min, maes)[0, 1]) if len(trip_rows) >= 2 else float("nan")

    # --- Fahrphasen-Split (Beharrung vs. Uebergang) ---
    a_est_z = np.array(
        [trips[ti][start + cfg.seq_len - 1 + cfg.horizon, A_EST_INDEX]
         for ti, start in ds.index],
        dtype=np.float64,
    )
    abs_err = np.abs(y_pred - y_true)
    is_transition = np.abs(a_est_z) > TRANSITION_A_EST_Z
    err_transition = abs_err[is_transition]
    err_steady = abs_err[~is_transition]

    # --- Plots erzeugen ---
    REPORTS_DIR.mkdir(exist_ok=True)
    b64: dict = {}
    b64["scatter"] = plot_scatter(y_true, y_pred, REPORTS_DIR / "test_eval_scatter.png", rng)
    b64["residual"] = plot_residual_hist(y_true, y_pred, REPORTS_DIR / "test_eval_residual_hist.png")
    b64["confusion"], _ = plot_direction_confusion(
        y_true, y_pred, "Richtungs-Konfusionsmatrix (Test)",
        REPORTS_DIR / "test_eval_confusion.png",
    )
    b64["overlay"] = plot_overlay_hist(y_true, y_pred, REPORTS_DIR / "test_eval_label_overlay.png")
    b64["mae_vs_dur"] = plot_trip_mae_vs_duration(
        durs_min, maes, REPORTS_DIR / "test_eval_mae_vs_duration.png")
    b64["phase"] = plot_error_by_phase(
        err_steady, err_transition, REPORTS_DIR / "test_eval_error_by_phase.png")

    b64["timeseries"] = []
    for i, (reason, r) in enumerate(chosen):
        ti = r["trip_index"]
        b = plot_trip_timeseries(
            times_s[ti], speeds_kmh[ti], labels[ti],
            r["target_t"] * PERIOD_S, r["yp"],
            f"Trip {r['trip_id']} \u2013 {reason} (MAE={r['mae']:.4f})",
            REPORTS_DIR / f"test_eval_timeseries_{i + 1}.png",
        )
        b64["timeseries"].append(b)

    # Zusaetzliche zufaellige Vorhersage-vs-Label-Zeitreihen (neutraler Titel).
    exclude = {r["trip_index"] for _, r in chosen}
    random_trips = select_random_trips(trip_rows, rng, N_RANDOM_TRIP_PLOTS, exclude)
    b64["timeseries_random"] = []
    for i, r in enumerate(random_trips):
        ti = r["trip_index"]
        b = plot_trip_timeseries(
            times_s[ti], speeds_kmh[ti], labels[ti],
            r["target_t"] * PERIOD_S, r["yp"],
            f"Trip {r['trip_id']} (MAE={r['mae']:.4f})",
            REPORTS_DIR / f"test_eval_timeseries_random_{i + 1}.png",
        )
        b64["timeseries_random"].append((r, b))

    # --- Konsolen-Report (geordnet) ---
    print_console_report(
        test_metrics, baseline_metrics, cm, cls_metrics, chosen, trip_rows,
        err_steady.size, err_transition.size, err_steady, err_transition,
    )

    # --- Report + Metrik-Dateien ---
    meta = {
        "best_val_loss": float(ckpt.get("best_val_loss", float("nan"))),
        "best_epoch": ckpt.get("best_epoch", None),
        "seq_len": cfg.seq_len,
        "horizon": cfg.horizon,
        "stride": cfg.stride,
        "n_trips": len(units),
        "n_windows": len(ds),
        "device": str(device),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    trip_stats = {"corr": corr, "n": len(trip_rows)}
    phase_stats = {
        "n_steady": int(err_steady.size),
        "n_transition": int(err_transition.size),
        "mae_steady": float(err_steady.mean()) if err_steady.size else float("nan"),
        "mae_transition": float(err_transition.mean()) if err_transition.size else float("nan"),
    }

    html = build_html(b64, test_metrics, baseline_metrics, cm, cls_metrics,
                      chosen, trip_stats, phase_stats, meta)
    (REPORTS_DIR / REPORT_NAME).write_text(html, encoding="utf-8")

    # CSV: Gesamt- und klassenweise Metriken + Trip-Metriken.
    metric_rows = [
        {"section": "overall", "name": "final_model_test", **test_metrics},
        {"section": "overall", "name": "persistence_test", **baseline_metrics},
    ]
    pd.DataFrame(metric_rows).to_csv(REPORTS_DIR / METRICS_CSV_NAME, index=False)

    trip_df = pd.DataFrame([
        {"trip_id": r["trip_id"], "n_windows": r["n_windows"],
         "duration_min": r["duration_min"], "mae": r["mae"],
         "brake_share": r["brake_share"], "accel_share": r["accel_share"],
         "hold_share": r["hold_share"]}
        for r in trip_rows
    ])
    trip_df.to_csv(REPORTS_DIR / "test_evaluation_trip_metrics.csv", index=False)

    (REPORTS_DIR / METRICS_JSON_NAME).write_text(json.dumps({
        "meta": meta,
        "overall": {
            "final_model_test": test_metrics,
            "persistence_test": baseline_metrics,
        },
        "direction_confusion": {
            "class_names": DIRECTION_CLASS_NAMES,
            "counts_row_real_col_pred": cm.tolist(),
        },
        "classwise": cls_metrics,
        "selected_trips": [
            {"reason": reason, "trip_id": r["trip_id"], "n_windows": r["n_windows"],
             "duration_min": r["duration_min"], "mae": r["mae"],
             "brake_share": r["brake_share"], "accel_share": r["accel_share"]}
            for reason, r in chosen
        ],
        "trip_duration_mae_corr": corr,
        "phase_split": phase_stats,
    }, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"Report   : {REPORTS_DIR / REPORT_NAME}")
    print(f"Metriken : {REPORTS_DIR / METRICS_CSV_NAME}")
    print(f"           {REPORTS_DIR / METRICS_JSON_NAME}")
    print(f"           {REPORTS_DIR / 'test_evaluation_trip_metrics.csv'}")
    print(f"Gesamtzeit: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
