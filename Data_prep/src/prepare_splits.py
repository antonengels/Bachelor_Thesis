#!/usr/bin/env python3
"""Erzeuge einen train/val/test-Split (75/15/10 nach aufgezeichneter Zeit) fuer das NN.

Ablauf:
1. Liest den bereinigten Trip-Master (``output/eda/_eda_trips_master.parquet``),
   der bereits sentinel-/out-of-range-maskiert auf einem gemeinsamen Zeitraster
   liegt. Fehlt der Cache, zuerst ``python src/eda.py --rebuild`` ausfuehren.
2. Feature Pruning gemaess EDA (Kap. 4.2/4.3): Data-Leak- und unbedeutende bzw.
   redundante Signale werden verworfen.
3. Feature Engineering der werthaltigen abgeleiteten Merkmale.
4. Per-Trip Imputation (ffill/bfill) der Sensorluecken.
5. Split der **Trips** so, dass die Summe der aufgezeichneten Zeit ~75/15/10
   ergibt (greedy longest-first Partitionierung).
6. Standardisierung (Clip auf robuste Perzentile + z-Score), Parameter werden
   ausschliesslich auf dem Trainingsanteil geschaetzt (kein Leakage).
7. Label-Normierung (/16384 -> [-1,1]) und milde Glaettung.
8. Schreibt ``train/val/test.parquet`` plus ``scaler.json``, ``split_manifest.csv``
   und ``feature_info.json`` in das Model-Datenverzeichnis.

Beispiel:
    python src/prepare_splits.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Konfiguration
# --------------------------------------------------------------------------- #
CACHE_FILE = Path("output/eda/_eda_trips_master.parquet")
DEFAULT_OUT = Path(r"C:\Users\Anton\Documents\Bachelorarbeit\02_Code\Model\data")

PERIOD_MS = 200
LABEL_COL = "M_ATO_RTBRq"
LABEL_SCALE = 16384.0
ROLL_WINDOW_S = 2.0          # gleitende Fenster fuer Rolling-Features
LABEL_SMOOTH_WINDOW = 5      # zentriertes Fenster (~1 s) fuer Label-Glaettung
CLIP_Q = (0.005, 0.995)      # robuste Clip-Grenzen (auf Trainingsanteil)

SPLIT_RATIOS = {"train": 0.75, "val": 0.15, "test": 0.10}

# Basissignale, die als Rohsignal in den Merkmalsraum eingehen.
BASE_FEATURES = ["V_EST", "V_PERMITTED", "A_EST", "A_GRADIENT"]

# Signale, die nur zur Ableitung engineerter Features dienen (nicht als Roh-Feature).
HELPER_COLS = ["D_STPDISTANCE"]

# Finale Feature-Reihenfolge (Eingangsvektor des NN).
FEATURES = [
    "V_EST",
    "V_PERMITTED",
    "A_EST",
    "A_GRADIENT",
    "v_ratio",
    "a_est_roll_mean_2s",
    "v_roll_std_2s",
    "stop_proximity",
]

# Dokumentation: bewusst entfernte Signale (Kap. 4.2/4.3).
DROPPED = {
    "M_RST_TBsetVal": "Data-Leak: Rueckmeldung der real gestellten Kraft (rho=0.78 ~ Label).",
    "M_RST_SlipSlide": "Praediktiv unbedeutend (rho=-0.02), quasi-konstant.",
    "TOPO_Q_RADIUS": "Praediktiv unbedeutend (rho=0.02) und nur ~35% Coverage.",
    "D_STPDISTANCE (roh)": "Ersetzt durch beschraenktes stop_proximity = 1/(|d|+1).",
    "v_headroom": "Redundant zu v_ratio (V_PERMITTED-V_EST bereits enthalten).",
    "jerk": "Stark verrauschte zweite Ableitung ohne Mehrwert.",
    "grad_x_v": "Schwacher experimenteller Kreuzterm (Gradient selbst rho=0.09).",
}

FEATURE_INFO = {
    "V_EST": "Ist-Geschwindigkeit (cm/s).",
    "V_PERMITTED": "Erlaubte Geschwindigkeit / Constraint (cm/s).",
    "A_EST": "Geschaetzte Beschleunigung (Sensor, roh).",
    "A_GRADIENT": "Streckenneigung / Gradient.",
    "v_ratio": "Ausnutzung V_EST/V_PERMITTED (0..2); normierte Geschwindigkeitsreserve.",
    "a_est_roll_mean_2s": "Gleitender Mittelwert von A_EST ueber ~2 s (entrauschter Trend).",
    "v_roll_std_2s": "Gleitende Std.-Abw. von V_EST ueber ~2 s (nichtlin. Uebergangsindikator).",
    "stop_proximity": "Haltepunkt-Naehe 1/(|D_STPDISTANCE|+1).",
}


# --------------------------------------------------------------------------- #
# Laden, Imputation, Feature Engineering
# --------------------------------------------------------------------------- #
def load_master(cache: Path) -> pd.DataFrame:
    if not cache.exists():
        raise SystemExit(
            f"Master-Cache fehlt: {cache}\n"
            "Bitte zuerst ausfuehren: python src/eda.py --rebuild"
        )
    df = pd.read_parquet(cache)
    df = df.sort_values(["unit", "tb"]).reset_index(drop=True)
    return df


def impute_and_engineer(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """Per-Trip Imputation der Sensorluecken und Ableitung der Features."""
    g = df.groupby("unit", observed=True)

    # 1) Basissignale + Helfer per Trip vorwaerts/rueckwaerts fuellen (Sensor haelt Wert).
    fill_cols = BASE_FEATURES + HELPER_COLS
    df[fill_cols] = g[fill_cols].ffill()
    df[fill_cols] = df.groupby("unit", observed=True)[fill_cols].bfill()

    # 2) Engineerte Features (aus imputierten Basissignalen).
    eps = 1e-6
    win = max(2, round(ROLL_WINDOW_S * 1000 / period))
    df["v_ratio"] = (df["V_EST"] / (df["V_PERMITTED"] + eps)).clip(0, 2)
    gg = df.groupby("unit", observed=True)
    df["a_est_roll_mean_2s"] = gg["A_EST"].transform(
        lambda s: s.rolling(win, min_periods=2).mean())
    df["v_roll_std_2s"] = gg["V_EST"].transform(
        lambda s: s.rolling(win, min_periods=2).std())
    df["stop_proximity"] = 1.0 / (df["D_STPDISTANCE"].abs() + 1.0)
    return df


def make_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Label normieren (/16384 -> [-1,1]) und mild glaetten."""
    df["label_raw"] = (df[LABEL_COL] / LABEL_SCALE).clip(-1, 1)
    df["label"] = (
        df.groupby("unit", observed=True)["label_raw"]
        .transform(lambda s: s.rolling(LABEL_SMOOTH_WINDOW, center=True, min_periods=1).mean())
    )
    df["label_raw"] = df["label_raw"].fillna(0.0)
    df["label"] = df["label"].fillna(0.0)
    return df


# --------------------------------------------------------------------------- #
# Zeitbasierter Split (75/15/10 nach aufgezeichneter Zeit)
# --------------------------------------------------------------------------- #
def time_based_split(df: pd.DataFrame, period: int, ratios: dict[str, float]) -> pd.Series:
    """Greedy longest-first: verteilt Trips so, dass die Zeitsumme ~ratios trifft."""
    dur_min = df.groupby("unit", observed=True).size() * period / 60000.0
    total = float(dur_min.sum())
    cap = {k: r * total for k, r in ratios.items()}
    cur = {k: 0.0 for k in ratios}
    assign: dict[str, str] = {}

    # Grosse Trips zuerst -> beste Balance.
    for unit, dur in dur_min.sort_values(ascending=False).items():
        k = max(ratios, key=lambda s: cap[s] - cur[s])
        assign[unit] = k
        cur[k] += float(dur)

    return df["unit"].map(assign).astype("category"), dur_min, cur, total


# --------------------------------------------------------------------------- #
# Skalierung (Fit nur auf Train)
# --------------------------------------------------------------------------- #
def fit_apply_scaler(df: pd.DataFrame, features: list[str]) -> dict:
    train = df["split"] == "train"
    scaler: dict[str, dict] = {}
    for f in features:
        xt = pd.to_numeric(df.loc[train, f], errors="coerce")
        lo = float(xt.quantile(CLIP_Q[0]))
        hi = float(xt.quantile(CLIP_Q[1]))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = float(xt.min()), float(xt.max())
            if not np.isfinite(lo) or lo == hi:
                lo, hi = 0.0, 1.0
        df[f] = df[f].clip(lo, hi)
        xt = pd.to_numeric(df.loc[train, f], errors="coerce")
        mu = float(xt.mean())
        sd = float(xt.std())
        if not np.isfinite(sd) or sd == 0.0:
            sd = 1.0
        df[f] = (df[f] - mu) / sd
        df[f] = df[f].fillna(0.0).astype("float32")   # Rest-NaN -> Mittelwert (=0)
        scaler[f] = {"clip_low": lo, "clip_high": hi, "mean": mu, "std": sd}
    return scaler


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Train/Val/Test-Split (75/15/10 nach Zeit).")
    ap.add_argument("--cache", type=Path, default=CACHE_FILE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--period-ms", type=int, default=PERIOD_MS)
    args = ap.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Lese Master: {args.cache}")
    df = load_master(args.cache)
    print(f"  {len(df):,} Zeilen, {df['unit'].nunique()} Trips")

    df = impute_and_engineer(df, args.period_ms)
    df = make_labels(df)

    split, dur_min, achieved, total = time_based_split(df, args.period_ms, SPLIT_RATIOS)
    df["split"] = split

    print("\nZeitbasierter Split (nach aufgezeichneter Zeit):")
    trip_split = df.groupby("unit", observed=True)["split"].first()
    for k in SPLIT_RATIOS:
        n_trips = int((trip_split == k).sum())
        h = achieved[k] / 60.0
        print(f"  {k:5s}: {n_trips:3d} Trips | {h:6.2f} h | "
              f"{achieved[k] / total:5.1%} (Ziel {SPLIT_RATIOS[k]:.0%})")

    scaler = fit_apply_scaler(df, FEATURES)

    keep_cols = ["unit", "tb", "rel_s", *FEATURES, "label", "label_raw"]
    written = {}
    for k in SPLIT_RATIOS:
        sub = df.loc[df["split"] == k, keep_cols].reset_index(drop=True)
        path = out_dir / f"{k}.parquet"
        sub.to_parquet(path, index=False, compression="zstd")
        written[k] = (path, len(sub))
        print(f"  -> {path} ({len(sub):,} Zeilen)")

    # Metadaten
    scaler_meta = {
        "features": FEATURES,
        "label_scale": LABEL_SCALE,
        "label_smoothing_window": LABEL_SMOOTH_WINDOW,
        "rolling_window_s": ROLL_WINDOW_S,
        "period_ms": args.period_ms,
        "clip_quantiles": CLIP_Q,
        "standardization": "z-score (fit on train only), NaN->mean(0)",
        "scaler": scaler,
    }
    (out_dir / "scaler.json").write_text(json.dumps(scaler_meta, indent=2), encoding="utf-8")

    feat_meta = {
        "features_used": FEATURE_INFO,
        "features_dropped": DROPPED,
        "label": {
            "column": "label (geglaettet) / label_raw",
            "definition": f"{LABEL_COL}/{int(LABEL_SCALE)} in [-1,1]; positiv=Beschl., negativ=Bremsen",
        },
    }
    (out_dir / "feature_info.json").write_text(json.dumps(feat_meta, indent=2, ensure_ascii=False),
                                               encoding="utf-8")

    manifest = (
        df.groupby("unit", observed=True)
        .agg(split=("split", "first"),
             recording=("recording", "first"),
             loco=("loco", "first"),
             n_rows=("tb", "size"))
        .reset_index()
    )
    manifest["dur_min"] = manifest["n_rows"] * args.period_ms / 60000.0
    manifest.to_csv(out_dir / "split_manifest.csv", index=False)

    print(f"\nMetadaten: scaler.json, feature_info.json, split_manifest.csv -> {out_dir}")
    print("Fertig.")


if __name__ == "__main__":
    main()
