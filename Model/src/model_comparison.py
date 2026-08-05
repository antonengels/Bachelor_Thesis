r"""
Vergleich klassischer Sequenz-Architekturen fuer die ATO-Stellsignal-Vorhersage.

Aufgabe (Forecasting / Behavioral Cloning):
    Gegeben eine HISTORIE der letzten H Zeitschritte der 8 Beobachtungs-Features,
    sage das ATO-Stellsignal `label` in [-1,1] (pos=Beschl., neg=Bremsen) voraus.
    Standardmaessig wird das Label `horizon` Schritte nach dem letzten
    Beobachtungszeitpunkt vorhergesagt (horizon=25 -> 5 s in die Zukunft).

Verglichene Architekturen:
    - MLP  : klassisches Feed-Forward-Netz auf dem geflatteten History-Fenster.
    - LSTM : klassisches Long-Short-Term-Memory-Netz.
    - GRU  : klassisches Gated-Recurrent-Unit-Netz.

Ziel dieses Skripts:
    Auf einem KLEINEN Teil (Standard: 25 %) der Trainings- und Validierungsdaten
    trainieren, um zu entscheiden, mit welcher Architektur weitergearbeitet wird.
    Erzeugt einen self-contained HTML-Report mit den wichtigsten Plots
    (u.a. Label real vs. predicted pro Modell).

Aufruf (Schnell-Auswahl auf 25 % der Daten):
    .\.venv\Scripts\python.exe src\model_comparison.py

Weitere Optionen:
    --full            gesamte Trainings-/Val-Daten verwenden (Fraction=1.0)
    --fraction 0.25   Anteil der Trips je Split (Standard 0.25)
    --epochs 16       Anzahl Trainingsepochen
    --seq-len 30      Laenge der History (Zeitschritte; 30 = 6 s @ 5 Hz)
    --horizon 25      Vorhersagehorizont in Schritten (25 = 5 s)
    --stride 5        Schrittweite beim Fensterbau (weniger = mehr Fenster)
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Pfade & Konstanten
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"

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
LABEL_COL = "label"
UNIT_COL = "unit"
TIME_COL = "tb"

# Klassifikations-Deadband fuer die Richtungs-Genauigkeit (bremsen/halten/beschl.)
DIR_DEADBAND = 0.05
# Toleranz fuer die Toleranz-Genauigkeit (|pred-true| <= TOL gilt als "richtig")
TOL = 0.10

MODEL_COLORS = {"MLP": "#4C72B0", "LSTM": "#DD8452", "GRU": "#55A868"}


# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    seq_len: int = 30
    horizon: int = 25
    stride: int = 5
    fraction: float = 0.25
    epochs: int = 10
    batch_size: int = 128
    hidden: int = 64
    lr: float = 1e-3
    seed: int = 42
    max_train_trips: int | None = None
    max_val_trips: int | None = None
    max_test_trips: int | None = None
    max_train_windows: int | None = None
    max_val_windows: int | None = None
    max_test_windows: int | None = None


# ---------------------------------------------------------------------------
# Daten
# ---------------------------------------------------------------------------
class SequenceDataset(Dataset):
    """History-Fenster pro Trip, ohne alle Fenster im Speicher zu materialisieren.

    Ein Sample = (X, y) mit
        X: (seq_len, n_features)  History der Features [t-H+1 .. t]
        y: skalar                 label bei t + horizon
    """

    def __init__(self, trips: list[np.ndarray], labels: list[np.ndarray], cfg: Config):
        self.cfg = cfg
        self.trips = trips
        self.labels = labels
        self.index: list[tuple[int, int]] = []
        H, hor, stride = cfg.seq_len, cfg.horizon, cfg.stride
        for ti, feats in enumerate(trips):
            n = feats.shape[0]
            last_start = n - H - hor  # start so dass start+H-1+hor < n
            if last_start < 0:
                continue
            for start in range(0, last_start + 1, stride):
                self.index.append((ti, start))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        ti, start = self.index[i]
        H, hor = self.cfg.seq_len, self.cfg.horizon
        x = self.trips[ti][start : start + H]  # (H, F)
        y = self.labels[ti][start + H - 1 + hor]  # skalar bei t+horizon
        y_prev = self.labels[ti][start + H - 1]  # letztes beobachtetes label (Persistenz)
        return (
            torch.from_numpy(np.array(x, dtype=np.float32)),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(y_prev, dtype=torch.float32),
        )


def load_split(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.parquet"
    df = pd.read_parquet(path, columns=[UNIT_COL, TIME_COL, *FEATURES, LABEL_COL])
    return df


def build_trip_arrays(
    df: pd.DataFrame, cfg: Config, split: str, rng: np.random.Generator
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Waehlt `fraction` der Trips aus und liefert pro Trip Feature-/Label-Arrays."""
    unit_series = df[UNIT_COL].astype(str)
    all_units = sorted(unit_series.unique())
    rng.shuffle(all_units)

    n_keep = max(1, int(round(len(all_units) * cfg.fraction)))
    kept = all_units[:n_keep]

    if split == "train":
        cap = cfg.max_train_trips
    elif split == "val":
        cap = cfg.max_val_trips
    else:
        cap = cfg.max_test_trips
    if cap is not None:
        kept = kept[:cap]
    kept_set = set(kept)

    trips: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for unit, g in df.groupby(unit_series, sort=False):
        if unit not in kept_set:
            continue
        g = g.sort_values(TIME_COL)
        feats = g[FEATURES].to_numpy(dtype=np.float32)
        lab = g[LABEL_COL].to_numpy(dtype=np.float32)
        trips.append(feats)
        labels.append(lab)
    return trips, labels


def cap_dataset(ds: SequenceDataset, max_windows: int | None, rng: np.random.Generator):
    if max_windows is not None and len(ds.index) > max_windows:
        idx = rng.choice(len(ds.index), size=max_windows, replace=False)
        ds.index = [ds.index[i] for i in sorted(idx)]
    return ds


# ---------------------------------------------------------------------------
# Modelle
# ---------------------------------------------------------------------------
class MLPRegressor(nn.Module):
    def __init__(self, seq_len: int, n_features: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * n_features, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):  # x: (B, H, F)
        return torch.tanh(self.net(x)).squeeze(-1)


class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden: int):
        super().__init__()
        self.rnn = nn.LSTM(n_features, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):  # x: (B, H, F)
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return torch.tanh(self.head(last)).squeeze(-1)


class GRURegressor(nn.Module):
    def __init__(self, n_features: int, hidden: int):
        super().__init__()
        self.rnn = nn.GRU(n_features, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):  # x: (B, H, F)
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return torch.tanh(self.head(last)).squeeze(-1)


def build_model(name: str, cfg: Config) -> nn.Module:
    if name == "MLP":
        return MLPRegressor(cfg.seq_len, len(FEATURES), cfg.hidden)
    if name == "LSTM":
        return LSTMRegressor(len(FEATURES), cfg.hidden)
    if name == "GRU":
        return GRURegressor(len(FEATURES), cfg.hidden)
    raise ValueError(name)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Metriken
# ---------------------------------------------------------------------------
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    mse = float(np.mean(err**2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot

    def to_class(a):
        c = np.zeros_like(a, dtype=np.int64)
        c[a > DIR_DEADBAND] = 2  # beschleunigen
        c[a < -DIR_DEADBAND] = 0  # bremsen
        c[(a >= -DIR_DEADBAND) & (a <= DIR_DEADBAND)] = 1  # halten/coast
        return c

    dir_acc = float(np.mean(to_class(y_true) == to_class(y_pred)))
    tol_acc = float(np.mean(np.abs(err) <= TOL))
    dir_acc_pct = dir_acc * 100.0
    tol_acc_pct = tol_acc * 100.0
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "dir_acc": dir_acc,
        "tol_acc": tol_acc,
        "dir_acc_pct": dir_acc_pct,
        "tol_acc_pct": tol_acc_pct,
    }


def rank_models_weighted(
    results: list[RunResult], test_weight: float = 2.0, val_weight: float = 1.0
) -> tuple[list[RunResult], dict[str, float]]:
    """Gewichtetes Gesamtranking ueber alle Metriken (Test priorisiert)."""
    metrics = [
        ("r2", True),
        ("rmse", False),
        ("mae", False),
        ("dir_acc", True),
        ("tol_acc", True),
    ]
    score = {r.name: 0.0 for r in results}

    for metric_name, higher_is_better in metrics:
        ordered_test = sorted(
            results,
            key=lambda r: r.test_metrics[metric_name],
            reverse=higher_is_better,
        )
        ordered_val = sorted(
            results,
            key=lambda r: r.metrics[metric_name],
            reverse=higher_is_better,
        )
        for rank, res in enumerate(ordered_test, start=1):
            score[res.name] += test_weight * rank
        for rank, res in enumerate(ordered_val, start=1):
            score[res.name] += val_weight * rank

    ranked = sorted(
        results,
        key=lambda r: (
            score[r.name],
            -(test_weight * r.test_metrics["r2"] + val_weight * r.metrics["r2"]),
            (test_weight * r.test_metrics["rmse"] + val_weight * r.metrics["rmse"]),
        ),
    )
    return ranked, score


# ---------------------------------------------------------------------------
# Training / Evaluation
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    name: str
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    test_metrics: dict = field(default_factory=dict)
    y_true: np.ndarray | None = None
    y_pred: np.ndarray | None = None
    n_params: int = 0
    train_time_s: float = 0.0


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    crit = nn.MSELoss(reduction="sum")
    total, n = 0.0, 0
    yt, yp, ypr = [], [], []
    for x, y, y_prev in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        total += crit(pred, y).item()
        n += y.numel()
        yt.append(y.cpu().numpy())
        yp.append(pred.cpu().numpy())
        ypr.append(y_prev.numpy())
    return total / max(n, 1), np.concatenate(yt), np.concatenate(yp), np.concatenate(ypr)


def train_one(name: str, cfg: Config, train_loader, val_loader, test_loader, device) -> RunResult:
    torch.manual_seed(cfg.seed)
    model = build_model(name, cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    crit = nn.MSELoss()
    res = RunResult(name=name, n_params=count_params(model))

    print(f"\n=== {name} ({res.n_params:,} Parameter) ===")
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        run_loss, n = 0.0, 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = crit(pred, y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * y.numel()
            n += y.numel()
        tr = run_loss / max(n, 1)
        vl, yt, yp, ypr = evaluate(model, val_loader, device)
        res.train_loss.append(tr)
        res.val_loss.append(vl)
        print(f"  Epoch {epoch:02d}/{cfg.epochs}  train_mse={tr:.5f}  val_mse={vl:.5f}")

    res.train_time_s = time.time() - t0
    vl, yt, yp, ypr = evaluate(model, val_loader, device)
    tl, yt_test, yp_test, ypr_test = evaluate(model, test_loader, device)
    res.metrics = regression_metrics(yt, yp)
    res.test_metrics = regression_metrics(yt_test, yp_test)
    res.y_true = yt
    res.y_pred = yp
    res._y_true_test = yt_test  # type: ignore[attr-defined]
    res._y_prev_val = ypr  # type: ignore[attr-defined]
    res._y_prev_test = ypr_test  # type: ignore[attr-defined]
    return res


# ---------------------------------------------------------------------------
# Plots -> base64
# ---------------------------------------------------------------------------
def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def plot_loss_curves(results: list[RunResult]) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for res in results:
        c = MODEL_COLORS[res.name]
        ep = range(1, len(res.train_loss) + 1)
        axes[0].plot(ep, res.train_loss, color=c, label=res.name)
        axes[1].plot(ep, res.val_loss, color=c, label=res.name)
    axes[0].set_title("Trainings-Loss (MSE)")
    axes[1].set_title("Validierungs-Loss (MSE)")
    for ax in axes:
        ax.set_xlabel("Epoche")
        ax.set_ylabel("MSE")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_metric_bars(results: list[RunResult]) -> str:
    names = [r.name for r in results]
    metrics = ["r2", "rmse", "mae", "dir_acc_pct", "tol_acc_pct"]
    titles = ["R2 (hoeher=besser)", "RMSE (niedriger)", "MAE (niedriger)",
              "Richtungs-Acc % (hoeher)", "Toleranz-Acc % (hoeher)"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.6))
    for ax, m, t in zip(axes, metrics, titles):
        vals = [r.metrics[m] for r in results]
        bars = ax.bar(names, vals, color=[MODEL_COLORS[n] for n in names])
        ax.set_title(t, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_pred_vs_true(results: list[RunResult]) -> str:
    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 4.2))
    if len(results) == 1:
        axes = [axes]
    for ax, res in zip(axes, results):
        yt, yp = res.y_true, res.y_pred
        # Bei vielen Punkten fuer die Lesbarkeit subsampeln.
        if yt.shape[0] > 20000:
            sel = np.random.default_rng(0).choice(yt.shape[0], 20000, replace=False)
            yt, yp = yt[sel], yp[sel]
        ax.scatter(yt, yp, s=4, alpha=0.25, color=MODEL_COLORS[res.name],
                   edgecolors="none")
        ax.plot([-1, 1], [-1, 1], "k--", lw=1)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("Label real")
        ax.set_ylabel("Label predicted")
        ax.set_title(f"{res.name}  (R2={res.metrics['r2']:.3f})")
        ax.grid(alpha=0.3)
    fig.suptitle("Label real vs. Label predicted (Validierung)")
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_timeseries(results: list[RunResult], n_points: int = 600) -> str:
    fig, axes = plt.subplots(len(results), 1, figsize=(11, 2.6 * len(results)),
                             sharex=True)
    if len(results) == 1:
        axes = [axes]
    for ax, res in zip(axes, results):
        yt, yp = res.y_true, res.y_pred
        k = min(n_points, yt.shape[0])
        ax.plot(range(k), yt[:k], color="black", lw=1.2, label="real")
        ax.plot(range(k), yp[:k], color=MODEL_COLORS[res.name], lw=1.2,
                alpha=0.85, label="predicted")
        ax.set_ylabel("label")
        ax.set_title(f"{res.name}: Verlauf real vs. predicted "
                     f"(erste {k} Val-Fenster)", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Fenster-Index (Validierung)")
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_residuals(results: list[RunResult]) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    for res in results:
        resid = res.y_pred - res.y_true
        ax.hist(resid, bins=80, histtype="step", linewidth=1.5,
                color=MODEL_COLORS[res.name], label=res.name, density=True)
    ax.set_title("Residuen-Verteilung (predicted - real)")
    ax.set_xlabel("Residuum")
    ax.set_ylabel("Dichte")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def build_report(
    results: list[RunResult], cfg: Config, baseline_val: dict, baseline_test: dict, meta: dict
) -> str:
    imgs = {
        "loss": plot_loss_curves(results),
        "bars": plot_metric_bars(results),
        "scatter": plot_pred_vs_true(results),
        "timeseries": plot_timeseries(results),
        "residuals": plot_residuals(results),
    }

    # Ranking: alle Metriken, Test priorisiert (2x) gegenueber Val (1x).
    ranked, rank_score = rank_models_weighted(results, test_weight=2.0, val_weight=1.0)
    best = ranked[0]

    def row(res: RunResult, metric_key: str, show_time: bool) -> str:
        m = res.metrics if metric_key == "val" else res.test_metrics
        star = " &#11088;" if res.name == best.name else ""
        time_cell = f"<td>{res.train_time_s:.1f}s</td>" if show_time else "<td>-</td>"
        return (
            f"<tr><td><b>{res.name}{star}</b></td>"
            f"<td>{res.n_params:,}</td>"
            f"<td>{rank_score[res.name]:.1f}</td>"
            f"<td>{m['r2']:.4f}</td>"
            f"<td>{m['rmse']:.4f}</td>"
            f"<td>{m['mae']:.4f}</td>"
            f"<td>{m['dir_acc_pct']:.2f}%</td>"
            f"<td>{m['tol_acc_pct']:.2f}%</td>"
            f"{time_cell}</tr>"
        )

    rows_val = "\n".join(row(r, "val", True) for r in ranked)
    rows_test = "\n".join(row(r, "test", False) for r in ranked)
    base_row_val = (
        f"<tr style='color:#888'><td><i>Persistenz (a<sub>t</sub>)</i></td><td>-</td><td>-</td>"
        f"<td>{baseline_val['r2']:.4f}</td><td>{baseline_val['rmse']:.4f}</td>"
        f"<td>{baseline_val['mae']:.4f}</td><td>{baseline_val['dir_acc_pct']:.2f}%</td>"
        f"<td>{baseline_val['tol_acc_pct']:.2f}%</td><td>-</td></tr>"
    )
    base_row_test = (
        f"<tr style='color:#888'><td><i>Persistenz (a<sub>t</sub>)</i></td><td>-</td><td>-</td>"
        f"<td>{baseline_test['r2']:.4f}</td><td>{baseline_test['rmse']:.4f}</td>"
        f"<td>{baseline_test['mae']:.4f}</td><td>{baseline_test['dir_acc_pct']:.2f}%</td>"
        f"<td>{baseline_test['tol_acc_pct']:.2f}%</td><td>-</td></tr>"
    )

    def img_block(title, key, note=""):
        note_html = f"<p class='note'>{note}</p>" if note else ""
        return (
            f"<h2>{title}</h2>{note_html}"
            f"<img src='data:image/png;base64,{imgs[key]}'/>"
        )

    html = f"""<!DOCTYPE html>
<html lang="de"><head><meta charset="utf-8">
<title>Architektur-Vergleich MLP / LSTM / GRU</title>
<style>
 body {{ font-family: Segoe UI, Arial, sans-serif; margin: 32px; color:#222; max-width:1100px; }}
 h1 {{ border-bottom: 3px solid #4C72B0; padding-bottom:6px; }}
 h2 {{ margin-top: 34px; color:#333; }}
 table {{ border-collapse: collapse; margin: 12px 0; font-size: 14px; }}
 th, td {{ border: 1px solid #ccc; padding: 6px 12px; text-align: right; }}
 th {{ background:#f0f3f7; }}
 td:first-child, th:first-child {{ text-align: left; }}
 img {{ max-width: 100%; border:1px solid #eee; border-radius:6px; margin: 6px 0 18px; }}
 .note {{ color:#666; font-size:13px; margin:4px 0 10px; }}
 .cfg {{ background:#f7f9fb; border:1px solid #e2e8f0; border-radius:6px; padding:10px 16px; font-size:13px; }}
 .rec {{ background:#eef7ee; border-left:5px solid #55A868; padding:12px 18px; border-radius:4px; }}
 code {{ background:#f0f0f0; padding:1px 5px; border-radius:3px; }}
</style></head>
<body>
<h1>Architektur-Vergleich: MLP vs. LSTM vs. GRU</h1>
<p>ATO-Stellsignal-Vorhersage aus einer Beobachtungs-History. Trainiert auf
<b>{cfg.fraction*100:.0f}&nbsp;%</b> der Trips (Auswahl auf Validierung, finale Kennzahlen auch auf Test).</p>

<div class="rec">
<b>Empfehlung:</b> Mit <b>{best.name}</b> weiterarbeiten &ndash; bestes Gesamt-Ranking
(Test 2x, Val 1x; alle Metriken). Score={rank_score[best.name]:.1f}.<br>
Test: R&sup2;={best.test_metrics['r2']:.4f}, RMSE={best.test_metrics['rmse']:.4f}
&middot; Val: R&sup2;={best.metrics['r2']:.4f}, RMSE={best.metrics['rmse']:.4f}.
</div>

<h2>Setup</h2>
<div class="cfg">
Aufgabe: History (H={cfg.seq_len} Schritte = {cfg.seq_len*0.2:.0f}&nbsp;s @ 5&nbsp;Hz)
&rarr; label bei t+{cfg.horizon} ({cfg.horizon*200}&nbsp;ms Vorhersagehorizont).<br>
Features ({len(FEATURES)}): {", ".join(FEATURES)}.<br>
Fraction={cfg.fraction} &middot; Stride={cfg.stride} &middot; Epochen={cfg.epochs}
&middot; Batch={cfg.batch_size} &middot; Hidden={cfg.hidden} &middot; LR={cfg.lr}
.<br>
Train-Trips: {meta['n_train_trips']} &middot; Val-Trips: {meta['n_val_trips']}
&middot; Test-Trips: {meta['n_test_trips']}<br>
Train-Fenster: {meta['n_train_windows']:,} &middot; Val-Fenster: {meta['n_val_windows']:,}
&middot; Test-Fenster: {meta['n_test_windows']:,}.
</div>

<h2>Ergebnis-Tabelle (Validierung)</h2>
<table>
<tr><th>Modell</th><th>Params</th><th>Ranking-Score</th><th>R&sup2;</th><th>RMSE</th><th>MAE</th>
<th>Richtungs-Acc (%)</th><th>Toleranz-Acc (%)</th><th>Trainingszeit</th></tr>
{rows_val}
{base_row_val}
</table>

<h2>Ergebnis-Tabelle (Test)</h2>
<table>
<tr><th>Modell</th><th>Params</th><th>Ranking-Score</th><th>R&sup2;</th><th>RMSE</th><th>MAE</th>
<th>Richtungs-Acc (%)</th><th>Toleranz-Acc (%)</th><th>Trainingszeit</th></tr>
{rows_test}
{base_row_test}
</table>
<p class="note">Richtungs-Acc: 3-Klassen (bremsen/halten/beschl.) mit Deadband
&plusmn;{DIR_DEADBAND}. Toleranz-Acc: Anteil |pred&minus;real| &le; {TOL}.
Persistenz nutzt das letzte <i>wahre</i> Label a<sub>t</sub> als Vorhersage &ndash;
Referenz, aber nicht direkt fair (das Modell sieht keine vergangenen Aktionen).</p>
<p class="note">Ranking-Score: Rangsumme ueber R&sup2;, RMSE, MAE, Richtungs-Acc,
Toleranz-Acc auf Val und Test; Test wird doppelt gewichtet. Niedriger ist besser.</p>

{img_block("Label real vs. Label predicted", "scatter",
    "Kernplot der Architekturauswahl: je enger an der Diagonalen, desto besser.")}
{img_block("Zeitreihe real vs. predicted", "timeseries")}
{img_block("Loss-Kurven", "loss")}
{img_block("Metriken im Vergleich", "bars")}
{img_block("Residuen-Verteilung", "residuals")}

<p class="note">Erzeugt am {meta['timestamp']} &middot; Gesamtzeit {meta['total_time']:.0f}s
&middot; Device: {meta['device']}.</p>
</body></html>"""
    return html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> Config:
    p = argparse.ArgumentParser(description="MLP vs LSTM vs GRU fuer ATO-Stellsignal.")
    p.add_argument("--full", action="store_true", help="Alle Trips (Fraction=1.0).")
    p.add_argument("--fraction", type=float, default=0.20)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seq-len", type=int, default=100)
    p.add_argument("--horizon", type=int, default=50)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-trips", type=int, default=None)
    p.add_argument("--max-val-trips", type=int, default=None)
    p.add_argument("--max-test-trips", type=int, default=None)
    p.add_argument("--max-train-windows", type=int, default=None)
    p.add_argument("--max-val-windows", type=int, default=None)
    p.add_argument("--max-test-windows", type=int, default=None)
    a = p.parse_args()
    cfg = Config(
        seq_len=a.seq_len,
        horizon=a.horizon,
        stride=a.stride,
        fraction=1.0 if a.full else a.fraction,
        epochs=a.epochs,
        batch_size=a.batch_size,
        hidden=a.hidden,
        lr=a.lr,
        seed=a.seed,
        max_train_trips=a.max_train_trips,
        max_val_trips=a.max_val_trips,
        max_test_trips=a.max_test_trips,
        max_train_windows=a.max_train_windows,
        max_val_windows=a.max_val_windows,
        max_test_windows=a.max_test_windows,
    )
    return cfg


def main():
    cfg = parse_args()
    t_start = time.time()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Konfiguration:", cfg)
    print("Device:", device)

    rng = np.random.default_rng(cfg.seed)
    print("\nLade Daten ...")
    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    train_trips, train_labels = build_trip_arrays(train_df, cfg, "train", rng)
    val_trips, val_labels = build_trip_arrays(val_df, cfg, "val", rng)
    test_trips, test_labels = build_trip_arrays(test_df, cfg, "test", rng)
    print(
        f"  Train-Trips: {len(train_trips)}  Val-Trips: {len(val_trips)}  "
        f"Test-Trips: {len(test_trips)}"
    )

    train_ds = SequenceDataset(train_trips, train_labels, cfg)
    val_ds = SequenceDataset(val_trips, val_labels, cfg)
    test_ds = SequenceDataset(test_trips, test_labels, cfg)
    train_ds = cap_dataset(train_ds, cfg.max_train_windows, rng)
    val_ds = cap_dataset(val_ds, cfg.max_val_windows, rng)
    test_ds = cap_dataset(test_ds, cfg.max_test_windows, rng)
    print(
        f"  Train-Fenster: {len(train_ds):,}  Val-Fenster: {len(val_ds):,}  "
        f"Test-Fenster: {len(test_ds):,}"
    )

    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        raise SystemExit("Keine Fenster gebaut - seq_len/horizon/stride pruefen.")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=0, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=0, drop_last=False)

    meta_base = {
        "n_train_trips": len(train_trips),
        "n_val_trips": len(val_trips),
        "n_test_trips": len(test_trips),
        "n_train_windows": len(train_ds),
        "n_val_windows": len(val_ds),
        "n_test_windows": len(test_ds),
        "device": str(device),
    }

    results: list[RunResult] = []
    for name in ("MLP", "LSTM", "GRU"):
        results.append(train_one(name, cfg, train_loader, val_loader, test_loader, device))
        # Report/Metriken nach JEDEM Modell schreiben -> absturzsicher bei langen Laeufen.
        save_outputs(results, cfg, meta_base, t_start)


def save_outputs(results: list[RunResult], cfg: Config, meta_base: dict, t_start: float):
    """Schreibt Report + Metriken mit den bisher fertigen Modellen (inkrementell)."""
    # Persistenz-Baseline: letztes wahres Label a_t als Vorhersage fuer t+horizon.
    # y_true / y_prev sind fuer alle Modelle je Split identisch (Loader ohne Shuffle).
    y_true_ref = results[0].y_true
    y_prev_val_ref = results[0]._y_prev_val  # type: ignore[attr-defined]
    y_true_test_ref = results[0]._y_true_test  # type: ignore[attr-defined]
    y_prev_test_ref = results[0]._y_prev_test  # type: ignore[attr-defined]
    baseline_val = regression_metrics(y_true_ref, y_prev_val_ref)
    baseline_test = regression_metrics(y_true_test_ref, y_prev_test_ref)

    meta = {
        **meta_base,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_time": time.time() - t_start,
    }

    REPORTS_DIR.mkdir(exist_ok=True)
    html = build_report(results, cfg, baseline_val, baseline_test, meta)
    report_path = REPORTS_DIR / "model_comparison_report.html"
    report_path.write_text(html, encoding="utf-8")

    # Metriken als CSV/JSON ablegen.
    rows = []
    for r in results:
        rows.append(
            {
                "split": "val",
                "model": r.name,
                "n_params": r.n_params,
                "train_time_s": r.train_time_s,
                **r.metrics,
            }
        )
        rows.append(
            {
                "split": "test",
                "model": r.name,
                "n_params": r.n_params,
                "train_time_s": r.train_time_s,
                **r.test_metrics,
            }
        )
    rows.append(
        {"split": "val", "model": "Persistence", "n_params": 0, "train_time_s": 0.0, **baseline_val}
    )
    rows.append(
        {"split": "test", "model": "Persistence", "n_params": 0, "train_time_s": 0.0, **baseline_test}
    )
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(REPORTS_DIR / "metrics.csv", index=False)
    (REPORTS_DIR / "metrics.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")

    done = ", ".join(r.name for r in results)
    print("\n" + "=" * 60)
    print(f"ZWISCHENSTAND gespeichert (fertig: {done}):")
    print(metrics_df.to_string(index=False))
    ranked, rank_score = rank_models_weighted(results, test_weight=2.0, val_weight=1.0)
    best = ranked[0]
    print(
        f"Aktuell bester -> {best.name} "
        f"(Score={rank_score[best.name]:.1f}, "
        f"Test R2={best.test_metrics['r2']:.4f}, Val R2={best.metrics['r2']:.4f})"
    )
    print(f"Report: {report_path}  |  Laufzeit bisher: {meta['total_time']:.1f}s")


if __name__ == "__main__":
    main()
