#!/usr/bin/env python3
"""Trip-fokussierte EDA mit Schwerpunkt Datenqualitaet, Plausibilitaet & Sanity.

Analysiert ausschliesslich die **Trip-Daten** unter ``output/trips/tripN`` (die
stillstandsbereinigten Bewegungssegmente). Ziel ist die Beurteilung der Daten an
sich: Vollstaendigkeit, Sentinel-/Fehlwerte, Wertebereiche, zeitliche Abtastung
und physikalische Plausibilitaet (z.B. Konsistenz von Geschwindigkeit,
Beschleunigung und Label).

Alle Abbildungen werden sowohl als **SVG** (vektoriell, zoombar) als auch als
**PNG** in ``output/eda/figures/`` abgelegt und im HTML-Report inline eingebettet.
Zusaetzlich werden die Kennzahlen als CSV exportiert und ein interaktiver
Plotly-Bereich erzeugt.

Label ``M_ATO_RTBRq``: **positiv = Beschleunigen, negativ = Bremsen**.

Aufruf::

    .venv/Scripts/python.exe src/eda.py
    .venv/Scripts/python.exe src/eda.py --rebuild --resample-ms 200
"""
from __future__ import annotations

import argparse
import html
import io
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
sns.set_theme(style="whitegrid", context="notebook")

# --------------------------------------------------------------------------- #
# Konfiguration
# --------------------------------------------------------------------------- #
OUTPUT_ROOT = Path("output")
TRIPS_DIR = OUTPUT_ROOT / "trips"
EDA_DIR = OUTPUT_ROOT / "eda"
FIG_DIR = EDA_DIR / "figures"
CACHE_FILE = EDA_DIR / "_eda_trips_master.parquet"

LABEL = "M_ATO_RTBRq"
LABEL_SCALE = 16384.0     # /16384 -> normiert [-1, 1]
SPEED_TO_KMH = 0.036      # V_EST (cm/s) * 0.036 -> km/h
MOVING_THRESHOLD = 30.0   # V_EST Rohwert ~0.3 m/s
TOPO_Q_RADIUS = "TOPO_Q_RADIUS"   # aktuelle Kurvenradius-Kategorie (Kopf-Intervall)

FEATURES = [
    "V_EST", "V_PERMITTED", "A_EST", "A_GRADIENT",
    "D_STPDISTANCE", "M_RST_TBsetVal", "M_RST_SlipSlide", LABEL,
    TOPO_Q_RADIUS,
]

FEATURE_INFO: dict[str, str] = {
    "V_EST": "Ist-Geschwindigkeit (Rohwert 0..4000, /100 -> m/s, *0.036 -> km/h); Sentinel 65535",
    "V_PERMITTED": "Erlaubte Geschwindigkeit (Rohwert 0..4000); Sentinel 65535",
    "A_EST": "Geschaetzte Beschleunigung (Rohwert -2500..2500); Sentinel +/-32768",
    "A_GRADIENT": "Streckenneigung/Gradient grad[0] (Rohwert); Sentinel +/-32768",
    "D_STPDISTANCE": "Distanz bis naechster Haltepunkt (Rohwert)",
    "M_RST_TBsetVal": "Fahrzeug-Rueckmeldung: eingestellte Zug-/Bremskraft",
    "M_RST_SlipSlide": "Schleuder-/Gleitschutz-Flag (0/1/2)",
    LABEL: "LABEL: ATO Soll-Zug-/Bremskraft (positiv=Beschl., negativ=Bremsen)",
    TOPO_Q_RADIUS: "Topologie: aktuelle Kurvenradius-Kategorie (Kopf-Intervall, forward-fill; 0..21)",
}

# Sentinel-/Fehlerwerte je Feature.
SENTINELS: dict[str, tuple[float, ...]] = {
    "V_EST": (65535.0,),
    "V_PERMITTED": (65535.0,),
    "A_EST": (32768.0, 32767.0, -32768.0),
    "A_GRADIENT": (32768.0, 32767.0, -32768.0),
    "M_ATO_RTBRq": (),
    "D_STPDISTANCE": (),
    "M_RST_TBsetVal": (),
    "M_RST_SlipSlide": (),
    TOPO_Q_RADIUS: (),
}

# Physikalisch plausible Rohwert-Grenzen. Werte ausserhalb werden als ungueltig
# (NaN) maskiert und zusaetzlich als Out-of-Range gezaehlt.
BOUNDS: dict[str, tuple[float, float]] = {
    "V_EST": (0.0, 4000.0),
    "V_PERMITTED": (0.0, 4000.0),
    "A_EST": (-2500.0, 2500.0),
    "A_GRADIENT": (-1000.0, 1000.0),
    "M_ATO_RTBRq": (-16384.0, 16384.0),
    "D_STPDISTANCE": (0.0, 2e8),
    "M_RST_TBsetVal": (-16384.0, 16384.0),
    "M_RST_SlipSlide": (0.0, 3.0),
    TOPO_Q_RADIUS: (0.0, 63.0),
}

# Rohspalte im Parquet -> Feature-Name je NID-Datei.
FILE_COLS: dict[str, list[tuple[str, str]]] = {
    "nid6": [("V_EST", "V_EST"), ("V_PERMITTED", "V_PERMITTED"),
             ("A_EST", "A_EST"), ("grad_value", "A_GRADIENT")],
    "nid31": [("M_ATO_RTBRq", "M_ATO_RTBRq")],
    "nid1": [("D_STPDISTANCE", "D_STPDISTANCE")],
    "nid32": [("M_RST_TBsetVal", "M_RST_TBsetVal"),
              ("M_RST_SlipSlide", "M_RST_SlipSlide")],
}

# Topologie-/Segmentprofil-Features (ERA Subset-126); NICHT in allen Trips vorhanden.
TOPO_FILE = "features_topology_radius.parquet"
TOPO_NUM_COLS = [
    "segment_length", "segment_count", "segment_curve_change_count",
    "radius_interval_width", "q_radius_category",
]
TOPO_INFO: dict[str, str] = {
    "segment_length": "Laenge eines Streckensegments (Rohwert)",
    "segment_count": "Anzahl Segmente im Segmentprofil",
    "segment_curve_change_count": "Anzahl Kruemmungswechsel je Segment",
    "radius_interval_width": "Breite eines Radius-Intervalls (end_abs - start_abs)",
    "q_radius_category": "Kurvenradius-Kategorie (q_radius_category)",
}

C_MAIN = "#1f77b4"
C_ACC = "#2ca02c"
C_LAB = "#d62728"
C_2 = "#9467bd"


# --------------------------------------------------------------------------- #
# Laden & Ausrichten (Master auf gemeinsamem Zeitraster)
# --------------------------------------------------------------------------- #
def _read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _mask(s: pd.Series, feat: str) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").astype("float64")
    for sv in SENTINELS.get(feat, ()):
        s = s.mask(s == sv)
    lo, hi = BOUNDS.get(feat, (-np.inf, np.inf))
    s = s.mask((s < lo) | (s > hi))
    return s


def _binned(df: pd.DataFrame, cols: list[str], period: int) -> pd.DataFrame:
    tb = (df["t_timestamp"].astype("int64") // period) * period
    g = df[cols].groupby(tb).mean()
    g.index.name = "tb"
    return g


def load_trip(unit_dir: Path, period: int) -> pd.DataFrame | None:
    nid6 = _read(unit_dir / "features_nid6.parquet")
    nid31 = _read(unit_dir / "features_nid31.parquet")
    if nid6 is None or nid31 is None or nid6.empty or nid31.empty:
        return None
    nid1 = _read(unit_dir / "features_nid1.parquet")
    nid32 = _read(unit_dir / "features_nid32.parquet")

    nid6 = nid6.drop_duplicates("t_timestamp", keep="first").copy()
    nid6["V_EST"] = _mask(nid6["V_EST"], "V_EST")
    nid6["V_PERMITTED"] = _mask(nid6["V_PERMITTED"], "V_PERMITTED")
    nid6["A_EST"] = _mask(nid6["A_EST"], "A_EST")
    nid6["A_GRADIENT"] = _mask(nid6.get("grad_value"), "A_GRADIENT")
    b6 = _binned(nid6, ["V_EST", "V_PERMITTED", "A_EST", "A_GRADIENT"], period)

    nid31 = nid31.copy()
    nid31[LABEL] = pd.to_numeric(nid31[LABEL], errors="coerce").astype("float64")
    b31 = _binned(nid31, [LABEL], period)

    frames = [b6, b31]
    if nid1 is not None and not nid1.empty and "D_STPDISTANCE" in nid1:
        nid1 = nid1.copy()
        nid1["D_STPDISTANCE"] = pd.to_numeric(nid1["D_STPDISTANCE"], errors="coerce")
        frames.append(_binned(nid1, ["D_STPDISTANCE"], period))
    if nid32 is not None and not nid32.empty:
        nid32 = nid32.copy()
        cols = [c for c in ("M_RST_TBsetVal", "M_RST_SlipSlide") if c in nid32]
        for c in cols:
            nid32[c] = pd.to_numeric(nid32[c], errors="coerce")
        frames.append(_binned(nid32, cols, period))

    merged = pd.concat(frames, axis=1).sort_index()
    merged = merged[merged[LABEL].notna() | merged["V_EST"].notna()]
    if merged.empty:
        return None
    merged = merged.reset_index()
    merged["rel_s"] = (merged["tb"] - merged["tb"].min()) / 1000.0

    # Topologie: aktuelle Kurvenradius-Kategorie (Kopf-Intervall) je Segmentprofil,
    # zeitlich forward-fill auf das Raster (merge_asof, letzte Nachricht <= tb).
    topo = _read(unit_dir / TOPO_FILE)
    if (topo is not None and not topo.empty
            and {"q_radius_category", "radius_interval_index", "t_timestamp"}.issubset(topo.columns)):
        head = (topo[topo["radius_interval_index"] == 0][["t_timestamp", "q_radius_category"]]
                .dropna().drop_duplicates("t_timestamp", keep="last"))
        head["t_timestamp"] = head["t_timestamp"].astype("int64")
        head = head.sort_values("t_timestamp").rename(columns={"q_radius_category": TOPO_Q_RADIUS})
        merged = merged.sort_values("tb")
        merged = pd.merge_asof(merged, head, left_on="tb", right_on="t_timestamp",
                               direction="backward")
        merged = merged.drop(columns=["t_timestamp"])
        merged[TOPO_Q_RADIUS] = pd.to_numeric(merged[TOPO_Q_RADIUS], errors="coerce")

    for c in FEATURES:
        if c not in merged:
            merged[c] = np.nan
    return merged


def _manifest_map() -> dict[str, str]:
    mf = TRIPS_DIR / "trip_manifest.csv"
    if not mf.exists():
        return {}
    m = pd.read_csv(mf)
    return {f"trip{int(r.trip_id)}": str(r.source_recording) for r in m.itertuples()}


def _loco(rec: str) -> str:
    if rec.endswith("Loco1"):
        return "Loco1"
    if rec.endswith("Loco2"):
        return "Loco2"
    return "unknown"


def _trip_dirs() -> list[Path]:
    return sorted([d for d in TRIPS_DIR.glob("trip*") if d.is_dir()],
                  key=lambda p: int(p.name.replace("trip", "")))


def build_master(period: int) -> pd.DataFrame:
    from tqdm import tqdm

    manifest = _manifest_map()
    rows: list[pd.DataFrame] = []
    for d in tqdm(_trip_dirs(), desc="Trips"):
        u = load_trip(d, period)
        if u is None:
            continue
        rec = manifest.get(d.name, "unknown")
        u["unit"] = d.name
        u["recording"] = rec
        u["loco"] = _loco(rec)
        rows.append(u)
    master = pd.concat(rows, ignore_index=True)
    for c in FEATURES:
        master[c] = master[c].astype("float32")
    master["rel_s"] = master["rel_s"].astype("float32")
    for c in ("unit", "recording", "loco"):
        master[c] = master[c].astype("category")
    return add_kinematics(master, period)


def add_kinematics(master: pd.DataFrame, period: int) -> pd.DataFrame:
    """Numerische Beschleunigung a_num = dV/dt (cm/s^2) je Trip auf dem Zeitraster."""
    master = master.sort_values(["unit", "tb"]).reset_index(drop=True)
    g = master.groupby("unit", observed=True)
    dt = g["tb"].diff()
    dv = g["V_EST"].diff()
    ok = dt == period
    a_num = np.where(ok.to_numpy(), dv.to_numpy() / (dt.to_numpy() / 1000.0), np.nan)
    master["a_num"] = a_num.astype("float32")
    return master


# --------------------------------------------------------------------------- #
# Roh-Qualitaetsscan (vor Maskierung, direkt auf den Parquet-Dateien)
# --------------------------------------------------------------------------- #
def quality_scan() -> tuple[pd.DataFrame, pd.DataFrame]:
    from tqdm import tqdm

    manifest = _manifest_map()
    acc = {f: dict(n=0, miss=0, sent=0, oor=0, vmin=np.inf, vmax=-np.inf)
           for f in FEATURES}
    timing: list[dict] = []

    for d in tqdm(_trip_dirs(), desc="Qualitaet"):
        cache: dict[str, pd.DataFrame | None] = {}
        for nid, colmap in FILE_COLS.items():
            df = _read(d / f"features_{nid}.parquet")
            cache[nid] = df
            if df is None:
                continue
            for raw_col, feat in colmap:
                if raw_col not in df.columns:
                    acc[feat]["n"] += len(df)
                    acc[feat]["miss"] += len(df)
                    continue
                s = pd.to_numeric(df[raw_col], errors="coerce")
                n = len(s)
                miss = int(s.isna().sum())
                sent_mask = s.isin(SENTINELS.get(feat, ()))
                sent = int(sent_mask.sum())
                valid = s[~sent_mask].dropna()
                lo, hi = BOUNDS[feat]
                oor = int(((valid < lo) | (valid > hi)).sum())
                inb = valid[(valid >= lo) & (valid <= hi)]
                a = acc[feat]
                a["n"] += n
                a["miss"] += miss
                a["sent"] += sent
                a["oor"] += oor
                if len(inb):
                    a["vmin"] = min(a["vmin"], float(inb.min()))
                    a["vmax"] = max(a["vmax"], float(inb.max()))

        # Zeitliche Abtastung anhand des Labels (nid31).
        n31 = cache.get("nid31")
        if n31 is not None and not n31.empty:
            t = np.sort(n31["t_timestamp"].astype("int64").unique())
            dts = np.diff(t)
            dts = dts[dts > 0]
            if dts.size:
                timing.append({
                    "unit": d.name,
                    "recording": manifest.get(d.name, "unknown"),
                    "loco": _loco(manifest.get(d.name, "unknown")),
                    "n_label": int(len(n31)),
                    "dur_min": float((t[-1] - t[0]) / 1000.0 / 60.0),
                    "median_dt_ms": float(np.median(dts)),
                    "p99_dt_ms": float(np.percentile(dts, 99)),
                    "max_gap_ms": float(dts.max()),
                    "gaps_gt_2s": int((dts > 2000).sum()),
                    "dup_ts_frac": float(n31["t_timestamp"].duplicated().mean()),
                })

    rows = []
    for f in FEATURES:
        if f == TOPO_Q_RADIUS:
            continue  # Topologie hat kein NID-Rohfile -> Qualitaet aus dem Master.
        a = acc[f]
        n = max(a["n"], 1)
        valid_n = a["n"] - a["miss"] - a["sent"]
        rows.append({
            "Feature": f,
            "n": a["n"],
            "Fehlwerte [%]": 100 * a["miss"] / n,
            "Sentinel [%]": 100 * a["sent"] / n,
            "Out-of-Range [%]": 100 * a["oor"] / max(valid_n, 1),
            "min (gueltig)": a["vmin"] if np.isfinite(a["vmin"]) else np.nan,
            "max (gueltig)": a["vmax"] if np.isfinite(a["vmax"]) else np.nan,
            "nutzbar [%]": 100 * max(valid_n, 0) / n,
        })
    return pd.DataFrame(rows), pd.DataFrame(timing)


# --------------------------------------------------------------------------- #
# Topologie-Features (Streckenprofil / Kurvenradien, nicht in allen Trips)
# --------------------------------------------------------------------------- #
def topology_scan() -> tuple[pd.DataFrame, dict]:
    """Sammelt die (optionalen) Topologie-/Segmentprofil-Daten ueber alle Trips."""
    from tqdm import tqdm

    manifest = _manifest_map()
    trip_dirs = _trip_dirs()
    frames: list[pd.DataFrame] = []
    n_with = 0
    for d in tqdm(trip_dirs, desc="Topologie"):
        df = _read(d / TOPO_FILE)
        if df is None or df.empty:
            continue
        n_with += 1
        df = df.copy()
        df["unit"] = d.name
        df["recording"] = manifest.get(d.name, "unknown")
        df["loco"] = _loco(manifest.get(d.name, "unknown"))
        if {"radius_interval_start_abs", "radius_interval_end_abs"}.issubset(df.columns):
            df["radius_interval_width"] = (
                df["radius_interval_end_abs"].astype("float64")
                - df["radius_interval_start_abs"].astype("float64"))
        frames.append(df)

    stats_d = {
        "trips_total": len(trip_dirs),
        "trips_with_topology": n_with,
        "topology_coverage_frac": n_with / max(len(trip_dirs), 1),
    }
    if not frames:
        return pd.DataFrame(), stats_d
    topo = pd.concat(frames, ignore_index=True)
    stats_d["n_radius_intervals"] = int(len(topo))
    if {"unit", "segment_index"}.issubset(topo.columns):
        stats_d["n_segments"] = int(topo.groupby(["unit", "segment_index"]).ngroups)
    else:
        stats_d["n_segments"] = np.nan
    return topo, stats_d


def topology_describe(topo: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in TOPO_NUM_COLS if c in topo.columns]
    d = topo[cols].apply(pd.to_numeric, errors="coerce").describe(
        percentiles=[0.05, 0.5, 0.95]).T
    d.insert(0, "Beschreibung", [TOPO_INFO.get(c, c) for c in d.index])
    return d


def topo_quality_row(master: pd.DataFrame) -> dict:
    """Qualitaetszeile fuer TOPO_Q_RADIUS auf Basis des Zeitrasters (Master)."""
    s = master[TOPO_Q_RADIUS]
    n = len(s)
    miss = int(s.isna().sum())
    valid = s.dropna()
    lo, hi = BOUNDS[TOPO_Q_RADIUS]
    oor = int(((valid < lo) | (valid > hi)).sum())
    return {
        "Feature": TOPO_Q_RADIUS,
        "n": n,
        "Fehlwerte [%]": 100 * miss / max(n, 1),
        "Sentinel [%]": 0.0,
        "Out-of-Range [%]": 100 * oor / max(len(valid), 1),
        "min (gueltig)": float(valid.min()) if len(valid) else np.nan,
        "max (gueltig)": float(valid.max()) if len(valid) else np.nan,
        "nutzbar [%]": 100 * len(valid) / max(n, 1),
    }


# --------------------------------------------------------------------------- #
# Report-Helfer
# --------------------------------------------------------------------------- #
def save_fig(fig, name: str) -> str:
    """Als SVG **und** PNG speichern, Inline-SVG-Markup fuer den Report liefern."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", format="png", dpi=120, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.svg", format="svg", bbox_inches="tight")
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    svg = buf.getvalue()
    idx = svg.find("<svg")
    return svg[idx:] if idx != -1 else svg


def df_to_html(df: pd.DataFrame, **kw) -> str:
    return df.to_html(border=0, classes="tbl", float_format=lambda x: f"{x:,.3f}", **kw)


# --------------------------------------------------------------------------- #
# Kennzahlen
# --------------------------------------------------------------------------- #
def overview_stats(master: pd.DataFrame, timing: pd.DataFrame, period: int) -> pd.DataFrame:
    dur_h = len(master) * period / 1000.0 / 3600.0
    return pd.DataFrame([{
        "Kennzahl": "Wert",
        "Trips": master["unit"].nunique(),
        "Recordings (Quelle)": master["recording"].nunique(),
        "Loco1 / Loco2 Trips": f"{(timing.loco == 'Loco1').sum()} / {(timing.loco == 'Loco2').sum()}",
        "Zeilen (Raster)": len(master),
        "Fahrzeit [h]": round(dur_h, 1),
        "Zeitraster [ms]": period,
    }]).set_index("Kennzahl").T.reset_index().rename(columns={"index": "Kennzahl"})


def per_trip_summary(master: pd.DataFrame, period: int) -> pd.DataFrame:
    recs = []
    for unit, sub in master.groupby("unit", observed=True):
        lab = sub[LABEL]
        recs.append({
            "unit": unit,
            "recording": sub["recording"].iloc[0],
            "loco": sub["loco"].iloc[0],
            "n_rows": len(sub),
            "dur_min": len(sub) * period / 1000.0 / 60.0,
            "moving_frac": float((sub["V_EST"] > MOVING_THRESHOLD).mean()),
            "vmax_kmh": float(np.nanmax(sub["V_EST"]) * SPEED_TO_KMH) if sub["V_EST"].notna().any() else np.nan,
            "label_accel_frac": float((lab > 0).mean()),
            "label_brake_frac": float((lab < 0).mean()),
            "label_zero_frac": float((lab == 0).mean()),
            "vest_nan": float(sub["V_EST"].isna().mean()),
        })
    return pd.DataFrame(recs)


def describe_features(master: pd.DataFrame) -> pd.DataFrame:
    d = master[FEATURES].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
    d.insert(0, "Beschreibung", [FEATURE_INFO[c] for c in d.index])
    return d


def spearman_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURES].corr(method="spearman")


def pearson_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURES].corr(method="pearson")


def label_correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in FEATURES:
        if c == LABEL:
            continue
        pair = df[[c, LABEL]].dropna()
        if len(pair) < 100 or pair[c].nunique() < 3:
            rows.append({"Feature": c, "Spearman rho": np.nan, "p (Spearman)": np.nan,
                         "Pearson r": np.nan, "n": len(pair)})
            continue
        rho, p = stats.spearmanr(pair[c], pair[LABEL])
        r = pair[c].corr(pair[LABEL])
        rows.append({"Feature": c, "Spearman rho": rho, "p (Spearman)": p,
                     "Pearson r": r, "n": len(pair)})
    return pd.DataFrame(rows).sort_values("Spearman rho", key=lambda s: s.abs(),
                                          ascending=False)


def consistency_metrics(master: pd.DataFrame) -> dict:
    """Physikalische Sanity-Checks als Kennzahlen."""
    m: dict = {}
    # a) A_EST vs. numerische Ableitung dV/dt
    pair = master[["A_EST", "a_num"]].dropna()
    pair = pair[pair["a_num"].abs() > 5]
    if len(pair) > 200:
        rho, _ = stats.spearmanr(pair["A_EST"], pair["a_num"])
        scale = float(np.median(pair["A_EST"] / pair["a_num"]))
        m["accel_rho"] = float(rho)
        m["accel_scale"] = scale
        m["accel_n"] = int(len(pair))
    # b) Label-Vorzeichen vs. A_EST-Vorzeichen
    lp = master[[LABEL, "A_EST"]].dropna()
    lp = lp[(lp[LABEL] != 0) & (lp["A_EST"] != 0)]
    if len(lp):
        same = float((np.sign(lp[LABEL]) == np.sign(lp["A_EST"])).mean())
        m["label_accel_sign_agree"] = same
    # c) V_PERMITTED >= V_EST (Ist <= erlaubt)?
    vp = master[["V_EST", "V_PERMITTED"]].dropna()
    if len(vp):
        m["vperm_ge_vest_frac"] = float((vp["V_PERMITTED"] >= vp["V_EST"]).mean())
    # d) negative Geschwindigkeit + robuste Spitzengeschwindigkeit
    v = master["V_EST"].dropna()
    m["neg_speed_frac"] = float((v < 0).mean()) if len(v) else np.nan
    m["vmax_kmh"] = float(np.nanmax(master["V_EST"]) * SPEED_TO_KMH)
    m["vmax_kmh_p999"] = float(np.nanpercentile(v, 99.9) * SPEED_TO_KMH) if len(v) else np.nan
    m["speed_over_150kmh_frac"] = float((v * SPEED_TO_KMH > 150).mean()) if len(v) else np.nan
    # e) Label folgt Rueckmeldung
    lm = master[[LABEL, "M_RST_TBsetVal"]].dropna()
    if len(lm) > 200:
        rho, _ = stats.spearmanr(lm[LABEL], lm["M_RST_TBsetVal"])
        m["label_tb_rho"] = float(rho)
    return m


# --------------------------------------------------------------------------- #
# Engineerte Features & Wichtigkeitsbewertung
# --------------------------------------------------------------------------- #
ENGINEERED_INFO: dict[str, str] = {
    "v_headroom": "Geschwindigkeitsreserve V_PERMITTED - V_EST (Rohwert); Abstand zur erlaubten Geschwindigkeit",
    "v_ratio": "Ausnutzung V_EST / V_PERMITTED (0..1); Naehe an der erlaubten Geschwindigkeit",
    "a_num": "Numerische Beschleunigung dV/dt aus V_EST [cm/s^2] (kinematisch abgeleitet)",
    "jerk": "Ruck d(a_num)/dt [cm/s^3]; Aenderungsrate der Beschleunigung",
    "v_roll_std_2s": "Gleitende Std.-Abw. von V_EST ueber ~2 s je Trip; kurzfristige Geschwindigkeitsdynamik",
    "a_est_roll_mean_2s": "Gleitender Mittelwert von A_EST ueber ~2 s je Trip; geglaettete Beschleunigung",
    "grad_x_v": "Interaktion A_GRADIENT x V_EST; Proxy fuer neigungsabhaengigen Fahrwiderstand/-antrieb",
    "stop_proximity": "Haltepunkt-Naehe 1/(|D_STPDISTANCE|+1); gross nahe am naechsten Halt",
}


def engineer_features(master: pd.DataFrame, period: int) -> tuple[pd.DataFrame, list[str]]:
    """Leite aus den Rohsignalen physikalisch motivierte Features ab."""
    df = master.sort_values(["unit", "tb"])
    eps = 1e-6
    out = pd.DataFrame(index=df.index)
    out["v_headroom"] = df["V_PERMITTED"] - df["V_EST"]
    out["v_ratio"] = (df["V_EST"] / (df["V_PERMITTED"] + eps)).clip(0, 2)
    out["a_num"] = df["a_num"]

    g = df.groupby("unit", observed=True)
    dt = g["tb"].diff()
    ok = (dt == period).to_numpy()
    da = g["a_num"].diff().to_numpy()
    out["jerk"] = np.where(ok, da / (period / 1000.0), np.nan)

    win = max(2, round(2000 / period))
    out["v_roll_std_2s"] = g["V_EST"].transform(
        lambda s: s.rolling(win, min_periods=2).std())
    out["a_est_roll_mean_2s"] = g["A_EST"].transform(
        lambda s: s.rolling(win, min_periods=2).mean())
    out["grad_x_v"] = df["A_GRADIENT"] * df["V_EST"]
    out["stop_proximity"] = 1.0 / (df["D_STPDISTANCE"].abs() + 1.0)

    out[LABEL] = df[LABEL].to_numpy()
    feats = list(ENGINEERED_INFO.keys())
    return out, feats


def _mutual_info(x: np.ndarray, y: np.ndarray, bins: int = 32) -> float:
    """Transinformation (nats) ueber ein 2D-Histogramm; robuste Perzentilgrenzen."""
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if a.size < 1000 or np.unique(a).size < 3 or np.unique(b).size < 3:
        return np.nan
    ax = np.nanpercentile(a, [0.5, 99.5])
    bx = np.nanpercentile(b, [0.5, 99.5])
    if ax[0] == ax[1] or bx[0] == bx[1]:
        return np.nan
    c, _, _ = np.histogram2d(a, b, bins=bins, range=[list(ax), list(bx)])
    pxy = c / c.sum()
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    denom = px @ py
    mask = pxy > 0
    return float(np.sum(pxy[mask] * np.log(pxy[mask] / denom[mask])))


def _standardized_ols_importance(df: pd.DataFrame, feats: list[str], label: str,
                                 max_rows: int = 300000) -> dict:
    """Standardisierte OLS-Koeffizienten als multivariates Wichtigkeitsmass."""
    d = df[feats + [label]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) > max_rows:
        d = d.sample(max_rows, random_state=0)
    if len(d) < 500:
        return {f: np.nan for f in feats}
    X = d[feats].to_numpy(dtype=float)
    y = d[label].to_numpy(dtype=float)
    Xs = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    ys = (y - y.mean()) / (y.std() + 1e-9)
    A = np.column_stack([np.ones(len(Xs)), Xs])
    beta, *_ = np.linalg.lstsq(A, ys, rcond=None)
    return dict(zip(feats, beta[1:]))


def engineered_importance(eng: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Bewerte engineerte Features per |Spearman|, |Pearson|, MI und OLS-Koeffizient."""
    ols = _standardized_ols_importance(eng, feats, LABEL)
    rows = []
    for c in feats:
        pair = eng[[c, LABEL]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(pair) < 500 or pair[c].nunique() < 3:
            rows.append({"Feature": c, "|Spearman|": np.nan, "|Pearson|": np.nan,
                         "MI [nats]": np.nan, "|std. beta|": abs(ols.get(c, np.nan)),
                         "n": len(pair)})
            continue
        rho, _ = stats.spearmanr(pair[c], pair[LABEL])
        r = pair[c].corr(pair[LABEL])
        mi = _mutual_info(pair[c].to_numpy(), pair[LABEL].to_numpy())
        rows.append({"Feature": c, "|Spearman|": abs(rho), "|Pearson|": abs(r),
                     "MI [nats]": mi, "|std. beta|": abs(ols.get(c, np.nan)),
                     "n": len(pair)})
    out = pd.DataFrame(rows)
    rank_cols = ["|Spearman|", "MI [nats]", "|std. beta|"]
    rank_names = [f"_r_{c}" for c in rank_cols]
    for col, rn in zip(rank_cols, rank_names):
        out[rn] = out[col].rank(ascending=False)
    out["Wichtigkeit (Rang-Mittel)"] = out[rank_names].mean(axis=1)
    out = out.sort_values("Wichtigkeit (Rang-Mittel)").reset_index(drop=True)
    out = out.drop(columns=rank_names)
    out.insert(1, "Beschreibung", [ENGINEERED_INFO[c] for c in out["Feature"]])
    return out


# --------------------------------------------------------------------------- #
# Plots (SVG + PNG)
# --------------------------------------------------------------------------- #
def plot_quality(rawq: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(rawq))
    w = 0.27
    ax.bar(x - w, rawq["Fehlwerte [%]"], w, label="Fehlwerte (NaN)", color="#9e9e9e")
    ax.bar(x, rawq["Sentinel [%]"], w, label="Sentinel/Fehlerwert", color=C_LAB)
    ax.bar(x + w, rawq["Out-of-Range [%]"], w, label="Out-of-Range", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(rawq["Feature"], rotation=40, ha="right")
    ax.set_ylabel("Anteil [%]")
    ax.set_title("Datenqualitaet je Feature (Rohdaten der Trips)")
    ax.legend()
    return save_fig(fig, "01_quality")


def plot_distributions(master: pd.DataFrame) -> str:
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    for ax, c in zip(axes.ravel(), FEATURES):
        v = master[c].dropna()
        if v.empty:
            ax.set_visible(False)
            continue
        lo, hi = np.nanpercentile(v, [1, 99])
        if lo == hi:
            hi = lo + 1
        ax.hist(v.clip(lo, hi), bins=60, color=C_MAIN, alpha=0.85)
        ax.set_title(c)
        ax.set_yscale("log")
    fig.suptitle("Feature-Verteilungen (Trips, 1.–99. Perzentil, log-Anzahl)", fontsize=14)
    fig.tight_layout()
    return save_fig(fig, "02_distributions")


def plot_boxplots(master: pd.DataFrame) -> str:
    specs = [
        ("V_EST", SPEED_TO_KMH, "km/h"),
        ("A_EST", 1.0, "roh"),
        ("A_GRADIENT", 1.0, "roh"),
        ("M_RST_TBsetVal", 1.0, "roh"),
        ("D_STPDISTANCE", 1e-3, "x1000"),
        (LABEL, 1 / LABEL_SCALE, "norm"),
        (TOPO_Q_RADIUS, 1.0, "Kat."),
    ]
    fig, axes = plt.subplots(1, len(specs), figsize=(20, 5))
    for ax, (c, sc, unit) in zip(axes, specs):
        v = (master[c].dropna() * sc)
        if v.empty:
            ax.set_visible(False)
            continue
        ax.boxplot(v, showfliers=False, whis=(1, 99))
        ax.set_title(f"{c}\n[{unit}]", fontsize=9)
    fig.suptitle("Boxplots & Ausreisser (Whisker = 1./99. Perzentil)", fontsize=14)
    fig.tight_layout()
    return save_fig(fig, "03_boxplots")


def plot_sampling(timing: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].hist(timing["median_dt_ms"], bins=40, color=C_MAIN)
    axes[0].set_xlabel("Median-Abtastintervall [ms]")
    axes[0].set_ylabel("Trips")
    axes[0].set_title("Typische Abtastrate (Label)")
    axes[1].hist(np.clip(timing["max_gap_ms"] / 1000.0, 0, 60), bins=40, color="#ff7f0e")
    axes[1].set_xlabel("Max. Luecke je Trip [s] (auf 60 s gekappt)")
    axes[1].set_title("Groesste Datenluecke je Trip")
    axes[2].hist(timing["gaps_gt_2s"], bins=40, color=C_LAB)
    axes[2].set_xlabel("Anzahl Luecken > 2 s")
    axes[2].set_title("Luecken pro Trip")
    axes[2].set_yscale("log")
    fig.suptitle("Zeitliche Abtastung & Datenluecken", fontsize=14)
    fig.tight_layout()
    return save_fig(fig, "04_sampling")


def plot_spearman(master: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    sp = spearman_matrix(master)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(sp, ax=ax, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
                square=True, annot_kws={"size": 9})
    ax.set_title("Spearman-Korrelation (Trips)")
    fig.tight_layout()
    return save_fig(fig, "05_spearman"), sp


def plot_engineered_importance(imp: pd.DataFrame) -> str:
    d = imp.dropna(subset=["|Spearman|"]).copy()
    mi = d["MI [nats]"].to_numpy(dtype=float)
    mi_max = np.nanmax(mi) if np.isfinite(mi).any() and np.nanmax(mi) > 0 else 1.0
    mi_n = mi / mi_max
    fig, ax = plt.subplots(figsize=(11, 5))
    y = np.arange(len(d))[::-1]
    h = 0.26
    ax.barh(y + h, d["|Spearman|"], h, label="|Spearman| (Label)", color=C_MAIN)
    ax.barh(y, mi_n, h, label="MI (normiert)", color=C_ACC)
    ax.barh(y - h, d["|std. beta|"], h, label="|std. OLS-Koeff.|", color=C_LAB)
    ax.set_yticks(y)
    ax.set_yticklabels(d["Feature"])
    ax.set_xlabel("Wichtigkeit (|Spearman|/|beta| in [0,1], MI normiert)")
    ax.set_title("Wichtigkeit engineerter Features fuer das Label")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return save_fig(fig, "06_engineered_importance")


def plot_spearman_vs_pearson(master: pd.DataFrame) -> str:
    sp = spearman_matrix(master)
    pe = pearson_matrix(master)
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    sns.heatmap(sp, ax=axes[0], annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
                square=True, annot_kws={"size": 8})
    axes[0].set_title("Spearman (Rang)")
    sns.heatmap(pe, ax=axes[1], annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
                square=True, annot_kws={"size": 8})
    axes[1].set_title("Pearson (linear)")
    fig.tight_layout()
    return save_fig(fig, "06_spearman_vs_pearson")


def plot_label_analysis(master: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    lab = master[LABEL]
    axes[0].bar(["Bremsen (<0)", "Neutral (=0)", "Beschl. (>0)"],
                [(lab < 0).mean(), (lab == 0).mean(), (lab > 0).mean()],
                color=[C_LAB, "#9e9e9e", C_ACC])
    axes[0].set_ylabel("Anteil")
    axes[0].set_title("Label-Vorzeichen")
    ln = (lab / LABEL_SCALE)
    axes[1].hist(ln[ln != 0], bins=80, color=C_MAIN)
    axes[1].axvline(0, color="k", lw=1)
    axes[1].set_xlabel("Label / 16384")
    axes[1].set_title("Label-Verteilung (ohne 0)")
    sub = master[["V_EST", LABEL]].dropna()
    sub = sub[sub["V_EST"] <= np.nanpercentile(sub["V_EST"], 99)]
    bins = np.linspace(0, sub["V_EST"].max(), 40)
    med = sub.assign(b=pd.cut(sub["V_EST"], bins)).groupby("b", observed=True)[LABEL].median()
    centers = np.array([iv.mid for iv in med.index]) * SPEED_TO_KMH
    axes[2].plot(centers, med.values / LABEL_SCALE, marker="o", ms=3, color=C_LAB)
    axes[2].axhline(0, color="k", lw=1)
    axes[2].set_xlabel("Geschwindigkeit [km/h]")
    axes[2].set_ylabel("Median Label / 16384")
    axes[2].set_title("Label vs. Geschwindigkeit")
    fig.tight_layout()
    return save_fig(fig, "07_label_analysis")


def plot_accel_consistency(master: pd.DataFrame, m: dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    pair = master[["A_EST", "a_num"]].dropna()
    pair = pair[pair["a_num"].abs() > 5]
    if len(pair):
        lo, hi = np.nanpercentile(pair["a_num"], [1, 99])
        p = pair[(pair["a_num"] >= lo) & (pair["a_num"] <= hi)]
        hb = axes[0].hexbin(p["a_num"], p["A_EST"], gridsize=50, mincnt=1, cmap="viridis",
                            bins="log", rasterized=True)
        fig.colorbar(hb, ax=axes[0], label="log Anzahl")
        rho = m.get("accel_rho", float("nan"))
        sc = m.get("accel_scale", float("nan"))
        axes[0].set_title(f"A_EST vs. dV/dt  (Spearman rho={rho:.2f}, Skala≈{sc:.2f})")
        axes[0].set_xlabel("dV/dt aus V_EST [cm/s²]")
        axes[0].set_ylabel("A_EST (Rohwert)")
    # Label-Vorzeichen vs A_EST-Vorzeichen
    lp = master[[LABEL, "A_EST"]].dropna()
    lp = lp[(lp[LABEL] != 0) & (lp["A_EST"] != 0)]
    if len(lp):
        cm = pd.crosstab(np.sign(lp[LABEL]), np.sign(lp["A_EST"]), normalize="all")
        cm = cm.reindex(index=[-1.0, 1.0], columns=[-1.0, 1.0])
        sns.heatmap(cm, ax=axes[1], annot=True, fmt=".2f", cmap="Blues", vmin=0)
        axes[1].set_xlabel("sign(A_EST)")
        axes[1].set_ylabel("sign(Label)")
        axes[1].set_title("Vorzeichen: Label vs. Beschleunigung")
    fig.tight_layout()
    return save_fig(fig, "08_accel_consistency")


def plot_speed_plausibility(master: pd.DataFrame, m: dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    v = master["V_EST"].dropna() * SPEED_TO_KMH
    hi = np.nanpercentile(v, 99.9)
    axes[0].hist(v.clip(0, hi), bins=80, color=C_MAIN)
    axes[0].axvline(0, color="k", lw=1)
    axes[0].set_xlabel("Geschwindigkeit [km/h] (auf 99.9-Perzentil gekappt)")
    axes[0].set_ylabel("Anzahl (Raster)")
    axes[0].set_title(f"Geschwindigkeit (typ. max {m.get('vmax_kmh_p999', float('nan')):.0f} km/h, "
                      f"Ausreisser bis {m.get('vmax_kmh', float('nan')):.0f} km/h)")
    vp = master[["V_EST", "V_PERMITTED"]].dropna()
    if len(vp):
        lo, hi = 0, np.nanpercentile(vp["V_PERMITTED"], 99)
        s = vp.sample(min(15000, len(vp)), random_state=0)
        axes[1].scatter(s["V_EST"] * SPEED_TO_KMH, s["V_PERMITTED"] * SPEED_TO_KMH,
                        s=3, alpha=0.2, color=C_MAIN, rasterized=True)
        lim = hi * SPEED_TO_KMH
        axes[1].plot([0, lim], [0, lim], color=C_LAB, lw=1, label="V_EST = V_PERMITTED")
        axes[1].set_xlabel("V_EST [km/h]")
        axes[1].set_ylabel("V_PERMITTED [km/h]")
        axes[1].set_title(f"Ist vs. erlaubt (Ist≤erlaubt: {m.get('vperm_ge_vest_frac', float('nan')):.0%})")
        axes[1].legend()
    fig.tight_layout()
    return save_fig(fig, "09_speed_plausibility")


def plot_per_trip(summary: pd.DataFrame, timing: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].hist(summary["dur_min"], bins=40, color=C_MAIN)
    axes[0].set_xlabel("Dauer [min]")
    axes[0].set_ylabel("Trips")
    axes[0].set_title("Trip-Dauer")
    axes[1].scatter(summary["moving_frac"], summary["vmax_kmh"], s=12, alpha=0.6, color=C_MAIN)
    axes[1].set_xlabel("Bewegungsanteil")
    axes[1].set_ylabel("v_max [km/h]")
    axes[1].set_title("Bewegungsanteil vs. Spitzengeschwindigkeit")
    axes[2].scatter(summary["label_accel_frac"], summary["label_brake_frac"], s=12,
                    alpha=0.6, color=C_ACC)
    axes[2].set_xlabel("Anteil Beschleunigen")
    axes[2].set_ylabel("Anteil Bremsen")
    axes[2].set_title("Label-Zusammensetzung je Trip")
    fig.tight_layout()
    return save_fig(fig, "10_per_trip")


def plot_sample_trip(master: pd.DataFrame) -> tuple[str, str]:
    cand = master.dropna(subset=["A_EST"]).groupby("unit", observed=True).size()
    if cand.empty:
        cand = master.groupby("unit", observed=True).size()
    unit = cand.idxmax()
    s = master[master.unit == unit].sort_values("rel_s")
    if len(s) > 4000:
        s = s.iloc[:: max(1, len(s) // 4000)]
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    axes[0].plot(s["rel_s"] / 60, s["V_EST"] * SPEED_TO_KMH, color=C_MAIN)
    axes[0].set_ylabel("v [km/h]")
    axes[0].set_title(f"Beispiel-Trip {unit}: Geschwindigkeit")
    axes[1].plot(s["rel_s"] / 60, s[LABEL] / LABEL_SCALE, color=C_LAB)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_ylabel("Label/16384")
    axes[1].set_title("Label (+ Beschl. / − Bremsen)")
    axes[2].plot(s["rel_s"] / 60, s["A_EST"], color=C_ACC, label="A_EST")
    axes[2].plot(s["rel_s"] / 60, s["a_num"], color=C_2, alpha=0.6, label="dV/dt")
    axes[2].axhline(0, color="k", lw=0.8)
    axes[2].set_ylabel("A_EST / dV/dt")
    axes[2].set_xlabel("Zeit [min]")
    axes[2].legend()
    fig.tight_layout()
    return save_fig(fig, "11_sample_trip"), unit


def plot_topology(topo: pd.DataFrame, stats_d: dict) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    if "q_radius_category" in topo.columns:
        vc = pd.to_numeric(topo["q_radius_category"], errors="coerce").dropna() \
            .astype(int).value_counts().sort_index()
        axes[0].bar(vc.index.astype(str), vc.values, color=C_MAIN)
        axes[0].set_xlabel("q_radius_category")
        axes[0].set_ylabel("Anzahl Radius-Intervalle")
        axes[0].set_title("Kurvenradius-Kategorien")
    if {"segment_length", "unit", "segment_index"}.issubset(topo.columns):
        sl = topo.drop_duplicates(["unit", "segment_index"])["segment_length"].dropna()
        if len(sl):
            hi = np.nanpercentile(sl, 99)
            axes[1].hist(sl.clip(0, hi), bins=40, color=C_ACC)
        axes[1].set_xlabel("Segmentlaenge (roh, 99%-gekappt)")
        axes[1].set_ylabel("Anzahl Segmente")
        axes[1].set_title("Segmentlaengen")
    if "radius_interval_width" in topo.columns:
        w = topo["radius_interval_width"].dropna()
        if len(w):
            hi = np.nanpercentile(w, 99)
            axes[2].hist(w.clip(0, hi), bins=40, color="#ff7f0e")
        axes[2].set_xlabel("Radius-Intervallbreite (roh, 99%-gekappt)")
        axes[2].set_ylabel("Anzahl Intervalle")
        axes[2].set_title("Radius-Intervallbreiten")
        axes[2].set_yscale("log")
    fig.suptitle(
        f"Topologie-Features (verfuegbar in {stats_d.get('trips_with_topology', 0)}/"
        f"{stats_d.get('trips_total', 0)} Trips)", fontsize=14)
    fig.tight_layout()
    return save_fig(fig, "12_topology")


# --------------------------------------------------------------------------- #
# Interaktiver Plotly-Teil
# --------------------------------------------------------------------------- #
def _np_hist(series, bins):
    v = pd.to_numeric(series, errors="coerce").to_numpy()
    v = v[~np.isnan(v)]
    if v.size == 0:
        return np.array([]), np.array([])
    counts, edges = np.histogram(v, bins=bins, density=True)
    return (edges[:-1] + edges[1:]) / 2, counts


def _heatmap(mat):
    import plotly.graph_objects as go
    return go.Heatmap(z=mat.values, x=list(mat.columns), y=list(mat.index),
                      coloraxis="coloraxis", text=np.round(mat.values, 2),
                      texttemplate="%{text}", textfont=dict(size=9),
                      hovertemplate="%{y} / %{x}: %{z:.3f}<extra></extra>")


def build_interactive(master, summary, rawq, timing, sp, sample_unit, topo, topo_stats):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from plotly.io import to_html

    feats = [f for f in FEATURES if f != LABEL]
    figs: list[tuple[str, str, "go.Figure"]] = []

    # 1) Qualitaet
    f = go.Figure()
    f.add_bar(x=rawq["Feature"], y=rawq["Fehlwerte [%]"], name="Fehlwerte", marker_color="#9e9e9e")
    f.add_bar(x=rawq["Feature"], y=rawq["Sentinel [%]"], name="Sentinel", marker_color=C_LAB)
    f.add_bar(x=rawq["Feature"], y=rawq["Out-of-Range [%]"], name="Out-of-Range", marker_color="#ff7f0e")
    f.update_layout(barmode="group", height=460, yaxis_title="[%]",
                    title="Datenqualitaet je Feature (Rohdaten)")
    figs.append(("Qualitaet", "Fehlwerte, Sentinel- und Out-of-Range-Anteile je Feature.", f))

    # 2) Verteilungen
    f = make_subplots(rows=3, cols=3, subplot_titles=FEATURES)
    for i, c in enumerate(FEATURES):
        r, cc = divmod(i, 3)
        v = master[c].dropna()
        if v.empty:
            continue
        lo, hi = np.nanpercentile(v, [1, 99])
        if lo == hi:
            hi = lo + 1
        x, y = _np_hist(v, np.linspace(lo, hi, 60))
        f.add_trace(go.Scatter(x=x, y=y, line=dict(color=C_MAIN), showlegend=False), r + 1, cc + 1)
    f.update_layout(height=820, title="Feature-Verteilungen (Dichte)")
    figs.append(("Verteilungen", "Dichte-Histogramme je Feature.", f))

    # 3) Abtastung
    f = make_subplots(rows=1, cols=3, subplot_titles=(
        "Median-dt [ms]", "Max. Luecke [s]", "Luecken > 2 s"))
    f.add_trace(go.Histogram(x=timing["median_dt_ms"], marker_color=C_MAIN, showlegend=False), 1, 1)
    f.add_trace(go.Histogram(x=np.clip(timing["max_gap_ms"] / 1000, 0, 60),
                             marker_color="#ff7f0e", showlegend=False), 1, 2)
    f.add_trace(go.Histogram(x=timing["gaps_gt_2s"], marker_color=C_LAB, showlegend=False), 1, 3)
    f.update_layout(height=430, title="Zeitliche Abtastung & Luecken")
    figs.append(("Abtastung", "Abtastrate und Datenluecken je Trip.", f))

    # 4) Spearman-Heatmap
    f = go.Figure(_heatmap(sp))
    f.update_layout(coloraxis=dict(colorscale="RdBu_r", cmin=-1, cmax=1, colorbar=dict(title="rho")),
                    height=620, title="Spearman-Korrelation (Trips)")
    f.update_yaxes(autorange="reversed")
    figs.append(("Spearman", "Rang-Korrelationsmatrix aller Features inkl. Label.", f))

    # 5) Label-Korrelation
    lc = label_correlations(master).set_index("Feature")
    f = go.Figure(go.Bar(x=feats, y=[lc["Spearman rho"].get(c, np.nan) for c in feats],
                         marker_color=C_ACC))
    f.update_layout(height=450, yaxis_title="Spearman rho mit Label",
                    title="Korrelation der Features mit dem Label")
    figs.append(("Label-Korrelation", "Spearman-Korrelation jedes Features mit dem Label.", f))

    # 7) Label-Analyse
    f = make_subplots(rows=1, cols=3, subplot_titles=(
        "Vorzeichen", "Label-Verteilung (ohne 0)", "Median-Label ueber v"))
    lab = master[LABEL]
    f.add_trace(go.Bar(x=["Bremsen", "Neutral", "Beschl."],
                       y=[(lab < 0).mean(), (lab == 0).mean(), (lab > 0).mean()],
                       marker_color=[C_LAB, "#9e9e9e", C_ACC], showlegend=False), 1, 1)
    ln = lab / LABEL_SCALE
    x, y = _np_hist(ln[ln != 0], np.linspace(-1, 1, 80))
    f.add_trace(go.Scatter(x=x, y=y, line=dict(color=C_MAIN), showlegend=False), 1, 2)
    sub = master[["V_EST", LABEL]].dropna()
    sub = sub[sub["V_EST"] <= np.nanpercentile(sub["V_EST"], 99)]
    med = sub.assign(b=pd.cut(sub["V_EST"], np.linspace(0, sub["V_EST"].max(), 40))) \
             .groupby("b", observed=True)[LABEL].median()
    centers = np.array([iv.mid for iv in med.index]) * SPEED_TO_KMH
    f.add_trace(go.Scatter(x=centers, y=med.values / LABEL_SCALE, mode="lines+markers",
                           line=dict(color=C_LAB), showlegend=False), 1, 3)
    f.update_layout(height=440, title="Label-Analyse")
    figs.append(("Label-Analyse", "Vorzeichen, Verteilung und Median-Label ueber Geschwindigkeit.", f))

    # 8) Beschleunigungs-Konsistenz
    pair = master[["A_EST", "a_num"]].dropna()
    pair = pair[pair["a_num"].abs() > 5]
    f = go.Figure()
    if len(pair):
        lo, hi = np.nanpercentile(pair["a_num"], [1, 99])
        p = pair[(pair["a_num"] >= lo) & (pair["a_num"] <= hi)]
        p = p.sample(min(40000, len(p)), random_state=0)
        f.add_trace(go.Scattergl(x=p["a_num"], y=p["A_EST"], mode="markers",
                                 marker=dict(size=3, opacity=0.25, color=C_MAIN), showlegend=False))
        f.update_layout(xaxis_title="dV/dt aus V_EST [cm/s²]", yaxis_title="A_EST (roh)")
    f.update_layout(height=520, title="Plausibilitaet: A_EST vs. numerische Ableitung dV/dt")
    figs.append(("A_EST-Konsistenz", "Stimmt die gemeldete Beschleunigung mit dV/dt ueberein?", f))

    # 9) Beispiel-Trip
    s = master[master.unit == sample_unit].sort_values("rel_s")
    if len(s) > 4000:
        s = s.iloc[:: max(1, len(s) // 4000)]
    f = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=(
        "v [km/h]", "Label/16384", "A_EST & dV/dt"))
    f.add_trace(go.Scatter(x=s["rel_s"] / 60, y=s["V_EST"] * SPEED_TO_KMH,
                           line=dict(color=C_MAIN), name="v"), 1, 1)
    f.add_trace(go.Scatter(x=s["rel_s"] / 60, y=s[LABEL] / LABEL_SCALE,
                           line=dict(color=C_LAB), name="Label"), 2, 1)
    f.add_trace(go.Scatter(x=s["rel_s"] / 60, y=s["A_EST"], line=dict(color=C_ACC), name="A_EST"), 3, 1)
    f.add_trace(go.Scatter(x=s["rel_s"] / 60, y=s["a_num"], line=dict(color=C_2), name="dV/dt"), 3, 1)
    f.update_xaxes(title_text="Zeit [min]", row=3, col=1)
    f.update_layout(height=720, title=f"Beispiel-Trip {sample_unit}")
    figs.append(("Beispiel-Trip", f"Zeitreihe des Trips {sample_unit}.", f))

    # 10) Topologie
    if topo is not None and not topo.empty:
        f = make_subplots(rows=1, cols=3, subplot_titles=(
            "q_radius_category", "Segmentlaenge", "Radius-Intervallbreite"))
        if "q_radius_category" in topo.columns:
            vc = pd.to_numeric(topo["q_radius_category"], errors="coerce").dropna() \
                .astype(int).value_counts().sort_index()
            f.add_trace(go.Bar(x=vc.index.astype(str), y=vc.values,
                               marker_color=C_MAIN, showlegend=False), 1, 1)
        if {"segment_length", "unit", "segment_index"}.issubset(topo.columns):
            sl = topo.drop_duplicates(["unit", "segment_index"])["segment_length"].dropna()
            hi = np.nanpercentile(sl, 99) if len(sl) else 1
            f.add_trace(go.Histogram(x=sl.clip(0, hi), marker_color=C_ACC,
                                     showlegend=False), 1, 2)
        if "radius_interval_width" in topo.columns:
            w = topo["radius_interval_width"].dropna()
            hi = np.nanpercentile(w, 99) if len(w) else 1
            f.add_trace(go.Histogram(x=w.clip(0, hi), marker_color="#ff7f0e",
                                     showlegend=False), 1, 3)
        f.update_layout(height=440, title=(
            f"Topologie-Features (in {topo_stats.get('trips_with_topology', 0)}/"
            f"{topo_stats.get('trips_total', 0)} Trips)"))
        figs.append(("Topologie",
                     "Kurvenradius-Kategorien, Segmentlaengen und Radius-Intervallbreiten.", f))

    out = []
    for i, (title, cap, fig) in enumerate(figs):
        div = to_html(fig, include_plotlyjs=("cdn" if i == 0 else False), full_html=False)
        out.append((title, cap, div))
    return out


# --------------------------------------------------------------------------- #
# Interpretation
# --------------------------------------------------------------------------- #
def build_interpretation(master, overview, rawq, timing, lc, m, sample_unit, period,
                         topo_stats) -> str:
    rq = rawq.set_index("Feature")
    lab = master[LABEL]
    top = lc.dropna(subset=["Spearman rho"]).iloc[0] if len(lc.dropna(subset=["Spearman rho"])) else None
    tq_rows = lc[lc["Feature"] == TOPO_Q_RADIUS]
    tq = (tq_rows.iloc[0] if len(tq_rows) and pd.notna(tq_rows.iloc[0]["Spearman rho"]) else None)
    med_dt = timing["median_dt_ms"].median()
    gap_trips = int((timing["gaps_gt_2s"] > 0).sum())
    long_gap_trips = int((timing["max_gap_ms"] > 60000).sum())

    paras = [
        f"<b>Umfang.</b> Ausgewertet werden ausschliesslich die <b>Trips</b> "
        f"({master['unit'].nunique()} Segmente aus {master['recording'].nunique()} Recordings, "
        f"~{len(master) * period / 3.6e6:.1f} h Fahrzeit) auf einem {period}-ms-Zeitraster. "
        f"Die Trips sind die stillstandsbereinigten Bewegungsphasen der Aufzeichnungen.",

        f"<b>Vollstaendigkeit &amp; Sentinels.</b> V_EST ist zu "
        f"{rq.loc['V_EST', 'nutzbar [%]']:.0f} % nutzbar, das Label zu "
        f"{rq.loc[LABEL, 'nutzbar [%]']:.0f} %. Deutlich lueckenhaft sind D_STPDISTANCE "
        f"(nur {rq.loc['D_STPDISTANCE', 'nutzbar [%]']:.0f} % nutzbar) sowie die nur zeitweise "
        f"gesendeten Signale A_EST/A_GRADIENT/V_PERMITTED "
        f"({rq.loc['A_EST', 'Sentinel [%]']:.0f} % Sentinel bzw. "
        f"{rq.loc['A_EST', 'Fehlwerte [%]']:.0f} % Fehlwerte bei A_EST). Die Sentinel-Werte "
        f"(32767/65535) wurden konsequent auf NaN maskiert.",

        f"<b>Wertebereiche / Plausibilitaetsgrenzen.</b> Zusaetzlich zu den Sentinels "
        f"(65535 fuer Geschwindigkeiten, &plusmn;32768 fuer Beschleunigung) werden Werte "
        f"ausserhalb physikalisch plausibler Grenzen als ungueltig maskiert: Geschwindigkeit "
        f"0&ndash;4000 roh (&asymp; 0&ndash;144 km/h), A_EST &minus;2500&hellip;2500. Der Anteil "
        f"solcher Out-of-Range-Rohwerte ist gering (V_EST "
        f"{rq.loc['V_EST', 'Out-of-Range [%]']:.3f} %, A_EST "
        f"{rq.loc['A_EST', 'Out-of-Range [%]']:.3f} %, Label "
        f"{rq.loc[LABEL, 'Out-of-Range [%]']:.3f} %); dadurch werden vereinzelte Messglitches "
        f"(z.B. unrealistische Geschwindigkeitsspitzen) entfernt. Es treten "
        f"{'keine' if m.get('neg_speed_frac', 0) == 0 else 'vereinzelt'} negative "
        f"Geschwindigkeiten auf; die robuste Spitzengeschwindigkeit (99.9-Perzentil) liegt bei "
        f"~{m.get('vmax_kmh_p999', float('nan')):.0f} km/h (Maximum nach Maskierung "
        f"~{m.get('vmax_kmh', float('nan')):.0f} km/h).",

        f"<b>Zeitliche Abtastung.</b> Das typische Abtastintervall (Label) liegt bei "
        f"~{med_dt:.0f} ms (~{1000 / med_dt:.0f} Hz). {gap_trips} von {len(timing)} Trips "
        f"enthalten mindestens eine Datenluecke &gt; 2 s; {long_gap_trips} Trips weisen sogar "
        f"Luecken &gt; 60 s auf (einzelne 'Trips' erstrecken sich damit ueber mehrere Stunden). "
        f"Hier greift die Stillstands-Segmentierung nicht sauber &ndash; diese Trips sollten an "
        f"den grossen Luecken nachtraeglich getrennt und Sequenzen nicht ueber Luecken hinweg "
        f"gebildet werden.",

        f"<b>Physikalische Plausibilitaet.</b> Die vom System gemeldete Beschleunigung A_EST "
        f"korreliert mit der numerischen Ableitung dV/dt der Geschwindigkeit "
        f"(Spearman rho &asymp; {m.get('accel_rho', float('nan')):.2f}, Skalenfaktor "
        f"&asymp; {m.get('accel_scale', float('nan')):.2f}) &ndash; die beiden unabhaengig "
        f"erfassten Signale sind also konsistent. Vorzeichen von Label und Beschleunigung "
        f"stimmen in {m.get('label_accel_sign_agree', float('nan')):.0%} der Faelle ueberein "
        f"(Beschleunigen&rarr;a&gt;0, Bremsen&rarr;a&lt;0). V_EST bleibt in "
        f"{m.get('vperm_ge_vest_frac', float('nan')):.0%} der Faelle unter V_PERMITTED. Diese "
        f"Checks bestaetigen die interne Konsistenz der Daten.",

        (f"<b>Label &amp; staerkste Zusammenhaenge.</b> Bremsen/Beschleunigen/Neutral verteilen "
         f"sich auf {(lab < 0).mean():.0%} / {(lab > 0).mean():.0%} / {(lab == 0).mean():.0%}. "
         f"Am staerksten mit dem Label (rang-)korreliert "
         f"<code>{top['Feature']}</code> (rho &asymp; {top['Spearman rho']:+.2f})"
         if top is not None else "<b>Label.</b>")
        + (f"; die Fahrzeug-Rueckmeldung folgt dem ATO-Sollwert "
           f"(M_RST_TBsetVal, rho &asymp; {m.get('label_tb_rho', float('nan')):+.2f}), was den "
           f"Wirkpfad Anforderung&rarr;Umsetzung bestaetigt." if 'label_tb_rho' in m else "."),

        f"<b>Topologie-Features.</b> Streckenprofil-/Kurvenradius-Daten (ERA Subset-126) sind "
        f"nur in {topo_stats.get('trips_with_topology', 0)}/{topo_stats.get('trips_total', 0)} "
        f"Trips vorhanden ({topo_stats.get('topology_coverage_frac', 0):.0%}) und liegen "
        f"distanz-/intervallbasiert vor (nicht pro Zeitschritt). Die aktuelle Kurvenradius-"
        f"Kategorie (<code>TOPO_Q_RADIUS</code>, Kopf-Intervall, forward-gefuellt) ist als "
        f"Feature in Qualitaet, Verteilungen und Korrelationen integriert; ihre "
        f"(Rang-)Korrelation mit dem Label ist "
        + (f"{tq['Spearman rho']:+.2f} (n={int(tq['n']):,})" if tq is not None else "gering")
        + ". Die Signale eignen sich als streckenbezogene Zusatzmerkmale, muessen dafuer aber "
        "ueber die Fahrzeugposition praezise mit der Zeitreihe verknuepft werden und fehlen fuer "
        "einen Teil der Trips.",

        f"<b>Fazit.</b> Die Trip-Daten sind qualitativ solide und physikalisch konsistent. "
        f"Zu beachten sind (1) die luecken-/sentinelbehafteten Signale D_STPDISTANCE, A_EST, "
        f"A_GRADIENT und V_PERMITTED (nur teilweise verfuegbar), (2) das quasi-konstante "
        f"M_RST_SlipSlide (kaum Information), (3) vereinzelte Datenluecken innerhalb von "
        f"Trips sowie (4) die nur in einem Teil der Trips vorhandenen Topologie-Features. "
        f"Als Praediktoren fuer das Label sind primaer M_RST_TBsetVal, Geschwindigkeit "
        f"und Beschleunigung geeignet. Beispiel-Trip <code>{sample_unit}</code> illustriert die "
        f"typische Dynamik.",
    ]
    return "\n".join(f"<p>{p}</p>" for p in paras)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def build_report(master: pd.DataFrame, timing: pd.DataFrame, rawq: pd.DataFrame,
                 period: int) -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)

    overview = overview_stats(master, timing, period)
    summary = per_trip_summary(master, period)
    desc = describe_features(master)
    m = consistency_metrics(master)
    topo, topo_stats = topology_scan()
    rawq = pd.concat([rawq, pd.DataFrame([topo_quality_row(master)])], ignore_index=True)

    img_q = plot_quality(rawq)
    img_dist = plot_distributions(master)
    img_box = plot_boxplots(master)
    img_samp = plot_sampling(timing)
    img_heat, sp = plot_spearman(master)
    img_lab = plot_label_analysis(master)
    img_acc = plot_accel_consistency(master, m)
    img_speed = plot_speed_plausibility(master, m)
    img_trip = plot_per_trip(summary, timing)
    img_sample, sample_unit = plot_sample_trip(master)
    img_topo = plot_topology(topo, topo_stats) if not topo.empty else ""

    lc = label_correlations(master)
    eng, eng_feats = engineer_features(master, period)
    eng_imp = engineered_importance(eng, eng_feats)
    img_eng = plot_engineered_importance(eng_imp)
    interactive = build_interactive(master, summary, rawq, timing, sp, sample_unit,
                                    topo, topo_stats)

    # CSV-Exporte
    overview.to_csv(EDA_DIR / "overview.csv", index=False)
    rawq.to_csv(EDA_DIR / "raw_quality.csv", index=False)
    timing.to_csv(EDA_DIR / "timing.csv", index=False)
    summary.to_csv(EDA_DIR / "per_trip_summary.csv", index=False)
    desc.to_csv(EDA_DIR / "feature_describe.csv")
    sp.to_csv(EDA_DIR / "spearman.csv")
    lc.to_csv(EDA_DIR / "label_correlations.csv", index=False)
    eng_imp.to_csv(EDA_DIR / "engineered_importance.csv", index=False)
    pd.DataFrame([m]).to_csv(EDA_DIR / "consistency_metrics.csv", index=False)
    if not topo.empty:
        topology_describe(topo).to_csv(EDA_DIR / "topology_describe.csv")
        pd.DataFrame([topo_stats]).to_csv(EDA_DIR / "topology_stats.csv", index=False)

    interp = build_interpretation(master, overview, rawq, timing, lc, m, sample_unit,
                                  period, topo_stats)

    def img(svg, cap):
        return (f'<figure><div class="svgbox">{svg}</div>'
                f'<figcaption>{html.escape(cap)}</figcaption></figure>')

    p = []
    p.append(f"""<!DOCTYPE html><html lang="de"><head><meta charset="utf-8">
<title>Trip-EDA – Qualitaet & Plausibilitaet</title>
<style>
 body{{font-family:Segoe UI,Arial,sans-serif;max-width:1250px;margin:24px auto;padding:0 18px;color:#1a1a1a;line-height:1.5}}
 h1{{border-bottom:3px solid {C_MAIN};padding-bottom:8px}}
 h2{{margin-top:42px;border-bottom:1px solid #ccc;padding-bottom:4px;color:#0d3b66}}
 figure{{margin:18px 0;text-align:center}}
 .svgbox{{border:1px solid #e0e0e0;border-radius:6px;padding:6px;overflow:auto;background:#fff}}
 .svgbox svg{{width:100%;height:auto;max-width:100%}}
 figcaption{{font-size:0.9em;color:#555;margin-top:6px}}
 table.tbl{{border-collapse:collapse;font-size:0.85em;margin:10px 0}}
 table.tbl td,table.tbl th{{border:1px solid #ddd;padding:4px 8px;text-align:right}}
 table.tbl th{{background:#f0f4f8}}
 .note{{background:#f7f9fc;border-left:4px solid {C_MAIN};padding:10px 14px;margin:16px 0}}
 code{{background:#eef;padding:1px 4px;border-radius:3px}}
</style></head><body>""")

    p.append("<h1>Trip-EDA – Datenqualitaet, Plausibilitaet &amp; Sanity</h1>")
    p.append(f"""<div class="note">
    <b>Datenbasis:</b> ausschliesslich die <b>Trips</b> unter <code>output/trips/tripN</code>
    (stillstandsbereinigte Bewegungssegmente).<br>
    <b>Zeitraster:</b> {period} ms. Sentinel-Werte (32767/65535) maskiert.
    Alle Abbildungen liegen als <b>SVG und PNG</b> in <code>output/eda/figures/</code>.<br>
    <b>Label</b> <code>{LABEL}</code>: <b>positiv = Beschleunigen</b>, <b>negativ = Bremsen</b>
    (/{int(LABEL_SCALE)} &rarr; [−1, 1]).
    </div>""")

    p.append("<h2>1. Ueberblick</h2>")
    p.append(df_to_html(overview, index=False))

    p.append("<h2>2. Datenqualitaet (Rohdaten)</h2>")
    p.append(df_to_html(rawq, index=False))
    p.append(img(img_q, "Abb. 1 – Fehlwerte, Sentinel- und Out-of-Range-Anteile je Feature."))

    p.append("<h2>3. Feature-Kennzahlen</h2>")
    p.append(df_to_html(desc))
    p.append(img(img_dist, "Abb. 2 – Feature-Verteilungen (log-Anzahl, 1.–99. Perzentil)."))
    p.append(img(img_box, "Abb. 3 – Boxplots & Ausreisser je Schluessel-Signal."))

    p.append("<h2>4. Zeitliche Abtastung &amp; Luecken</h2>")
    p.append(img(img_samp, "Abb. 4 – Abtastintervall, groesste Luecke und Anzahl Luecken je Trip."))

    p.append("<h2>5. Korrelationen</h2>")
    p.append(img(img_heat, "Abb. 5 – Spearman-Korrelationsmatrix (Trips)."))
    p.append("<h3>Korrelation der Features mit dem Label</h3>")
    p.append(df_to_html(lc, index=False))

    p.append("<h2>6. Engineerte Features &amp; Wichtigkeit</h2>")
    p.append(f"""<div class="note">
    Aus den Rohsignalen abgeleitete (<b>engineerte</b>) Features und ihre Relevanz fuer das
    Label <code>{LABEL}</code>. Bewertet mit drei komplementaeren Massen:
    <b>|Spearman|</b> (monotone Assoziation), <b>MI</b> (Transinformation, erfasst auch
    nichtlineare Zusammenhaenge; im Diagramm auf das Maximum normiert) und
    <b>|std. OLS-Koeffizient|</b> (multivariater Beitrag bei gemeinsamer, standardisierter
    Regression). Die Spalte <b>Wichtigkeit (Rang-Mittel)</b> mittelt die Raenge dieser drei
    Masse (kleiner = wichtiger).
    </div>""")
    p.append(df_to_html(eng_imp, index=False))
    p.append(img(img_eng, "Abb. 6 – Wichtigkeit der engineerten Features fuer das Label "
                          "(|Spearman|, normierte MI, |standardisierter OLS-Koeffizient|)."))

    p.append("<h2>7. Label-Analyse</h2>")
    p.append(img(img_lab, "Abb. 7 – Vorzeichen-Anteile, Verteilung und Median-Label ueber v."))

    p.append("<h2>8. Physikalische Plausibilitaet / Sanity</h2>")
    p.append(f"""<div class="note">
    <b>Konsistenz-Kennzahlen:</b>
    A_EST vs. dV/dt: Spearman rho = {m.get('accel_rho', float('nan')):.2f}
    (Skalenfaktor ≈ {m.get('accel_scale', float('nan')):.2f}) &middot;
    Vorzeichen Label/Beschleunigung stimmen zu {m.get('label_accel_sign_agree', float('nan')):.0%} &middot;
    V_EST ≤ V_PERMITTED in {m.get('vperm_ge_vest_frac', float('nan')):.0%} &middot;
    negative Geschwindigkeit: {m.get('neg_speed_frac', float('nan')):.2%} &middot;
    v_max ≈ {m.get('vmax_kmh', float('nan')):.0f} km/h &middot;
    Label vs. Rueckmeldung: rho = {m.get('label_tb_rho', float('nan')):.2f}.
    </div>""")
    p.append(img(img_acc, "Abb. 8 – A_EST vs. numerische Ableitung dV/dt und Vorzeichen-Abgleich "
                          "Label/Beschleunigung."))
    p.append(img(img_speed, "Abb. 9 – Geschwindigkeits-Plausibilitaet (km/h) und Ist vs. erlaubt."))

    p.append("<h2>9. Trip-Ebene</h2>")
    p.append(img(img_trip, "Abb. 10 – Dauer, Bewegungsanteil/Spitzengeschwindigkeit und "
                           "Label-Zusammensetzung je Trip."))

    p.append("<h2>10. Beispiel-Trip (Zeitreihe)</h2>")
    p.append(img(img_sample, f"Abb. 11 – Beispiel-Trip {sample_unit}: v, Label, A_EST/dV/dt."))

    p.append("<h2>11. Topologie-Features (Streckenverlauf)</h2>")
    if not topo.empty:
        p.append(f"""<div class="note">
        Streckenprofil-/Kurvenradius-Daten (ERA Subset-126) sind nur in
        <b>{topo_stats['trips_with_topology']}/{topo_stats['trips_total']} Trips</b>
        ({topo_stats['topology_coverage_frac']:.0%}) vorhanden &ndash; insgesamt
        {topo_stats['n_radius_intervals']:,} Radius-Intervalle in
        {topo_stats['n_segments']:,} Segmenten. Die Daten sind distanz-/intervallbasiert
        (nicht pro Zeitschritt) und beschreiben den Streckenverlauf (Kurvenradien).<br>
        Zusaetzlich fliesst die <b>aktuelle Kurvenradius-Kategorie</b> (<code>{TOPO_Q_RADIUS}</code>,
        Kopf-Intervall je Segmentprofil, zeitlich forward-gefuellt) in die vorherigen Abschnitte
        ein (Qualitaet, Verteilungen, Korrelationen, Label-Korrelation).
        </div>""")
        p.append(df_to_html(topology_describe(topo)))
        p.append(img(img_topo, "Abb. 12 – Topologie: Kurvenradius-Kategorien, Segmentlaengen "
                               "und Radius-Intervallbreiten."))
    else:
        p.append('<div class="note">Keine Topologie-Daten vorhanden.</div>')

    p.append("<h2>12. Interaktive Uebersicht</h2>")
    p.append('<div class="note">Alle Auswertungen zusaetzlich interaktiv '
             '(Zoom, Hover, PNG-Export ueber die Toolbar).</div>')
    for i, (title, cap, div) in enumerate(interactive):
        p.append(f"<h3>12.{i + 1} {html.escape(title)}</h3>")
        p.append(f'<p style="color:#555;font-size:0.9em;margin:2px 0 8px">{html.escape(cap)}</p>')
        p.append(div)

    p.append("<h2>13. Interpretation &amp; Schlussfolgerungen</h2>")
    p.append(interp)
    p.append("</body></html>")

    report = EDA_DIR / "eda_report.html"
    report.write_text("\n".join(p), encoding="utf-8")

    md = (interp.replace("<b>", "**").replace("</b>", "**")
          .replace("<i>", "*").replace("</i>", "*")
          .replace("<code>", "`").replace("</code>", "`")
          .replace("<p>", "").replace("</p>", "\n")
          .replace("&ndash;", "–").replace("&rarr;", "→").replace("&asymp;", "≈")
          .replace("&plusmn;", "±").replace("&minus;", "−").replace("&hellip;", "…")
          .replace("&middot;", "·")
          .replace("&amp;", "&").replace("&gt;", ">").replace("&lt;", "<"))
    (EDA_DIR / "interpretation.md").write_text("# Trip-EDA – Interpretation\n\n" + md,
                                               encoding="utf-8")

    print(f"Report: {report}")
    print(f"Figuren (SVG+PNG): {FIG_DIR}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Trip-EDA: Qualitaet, Plausibilitaet, Sanity.")
    ap.add_argument("--resample-ms", type=int, default=200)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    EDA_DIR.mkdir(parents=True, exist_ok=True)

    if CACHE_FILE.exists() and not args.rebuild:
        print(f"Lade Cache: {CACHE_FILE}")
        master = pd.read_parquet(CACHE_FILE)
        if "a_num" not in master.columns:
            master = add_kinematics(master, args.resample_ms)
    else:
        print("Baue Trip-Master ...")
        master = build_master(args.resample_ms)
        master.to_parquet(CACHE_FILE, compression="zstd")
        print(f"Cache: {CACHE_FILE} ({len(master):,} Zeilen)")

    print("Roh-Qualitaetsscan ...")
    rawq, timing = quality_scan()

    print(f"Master: {len(master):,} Zeilen, {master['unit'].nunique()} Trips")
    build_report(master, timing, rawq, args.resample_ms)


if __name__ == "__main__":
    main()
