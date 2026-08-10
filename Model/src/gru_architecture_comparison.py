r"""
GRU-Only Bruteforce fuer Architekturvergleich (Layeranzahl x Neuronen pro Layer).

Idee:
    Nutzt dieselbe Datenaufbereitung/Windowing-Logik wie `model_comparison.py`,
    trainiert aber nur GRU-Varianten mit unterschiedlicher Architektur.

Architekturraum (Standard):
    - Maximal 3 GRU-Layer
    - Pro Layer Neuronen aus {64, 96, 128}
    - Erlaubte Tiefen: 1, 2, 3 Layer
    -> Anzahl Kombinationen: 3 + 9 + 27 = 39

Ziel:
    - Ranking aller Architekturen nach Test-MSE
    - Aggregierte Auswertung je Layer-Tiefe (mean/median/best)
    - Self-contained HTML-Report mit den wichtigsten Plots

Beispiel:
    .\.venv\Scripts\python.exe src\gru_architecture_comparison.py

Optionen wie beim Architekturvergleich:
    --full --fraction --epochs --seq-len --horizon --stride --batch-size
    --lr --seed --max-*-trips --max-*-windows

Neue Optionen:
    --neurons 64,96,128
    --max-layers 3
    --sort-desc / --sort-asc
"""

from __future__ import annotations

import argparse
import base64
import io
import itertools
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
from torch.utils.data import DataLoader

from model_comparison import (
    FEATURES,
    SequenceDataset,
    build_trip_arrays,
    cap_dataset,
    load_split,
    regression_metrics,
)


ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"

REPORT_NAME = "gru_architecture_comparison_report.html"
METRICS_CSV_NAME = "gru_architecture_comparison_metrics.csv"
METRICS_JSON_NAME = "gru_architecture_comparison_metrics.json"

DEPTH_COLORS = {
    1: "#4C72B0",
    2: "#DD8452",
    3: "#55A868",
}


@dataclass
class Config:
    # Entspricht den Standard-Trainingsparametern aus model_comparison.py
    seq_len: int = 30
    horizon: int = 25
    stride: int = 5
    fraction: float = 0.25
    epochs: int = 10
    batch_size: int = 128
    lr: float = 1e-3
    seed: int = 42
    max_train_trips: int | None = None
    max_val_trips: int | None = None
    max_test_trips: int | None = None
    max_train_windows: int | None = None
    max_val_windows: int | None = None
    max_test_windows: int | None = None

    max_layers: int = 3
    neurons: list[int] = field(default_factory=lambda: [64, 96, 128])
    sort_desc: bool = False
    top_loss_curves: int = 12


@dataclass
class ArchResult:
    arch: tuple[int, ...]
    arch_name: str
    n_layers: int
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    test_metrics: dict = field(default_factory=dict)
    n_params: int = 0
    train_time_s: float = 0.0
    y_true: np.ndarray | None = None
    y_pred: np.ndarray | None = None
    _y_prev_val: np.ndarray | None = None
    _y_true_test: np.ndarray | None = None
    _y_prev_test: np.ndarray | None = None


class VariableGRURegressor(nn.Module):
    """Stack aus 1-layer-GRUs, damit pro Layer unterschiedliche Hidden-Size moeglich ist."""

    def __init__(self, n_features: int, hidden_sizes: tuple[int, ...]):
        super().__init__()
        if len(hidden_sizes) == 0:
            raise ValueError("hidden_sizes darf nicht leer sein")

        layers = []
        in_size = n_features
        for h in hidden_sizes:
            layers.append(nn.GRU(in_size, h, num_layers=1, batch_first=True))
            in_size = h

        self.layers = nn.ModuleList(layers)
        self.head = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for gru in self.layers:
            out, _ = gru(out)
        last = out[:, -1, :]
        return torch.tanh(self.head(last)).squeeze(-1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def arch_to_name(arch: tuple[int, ...]) -> str:
    return "-".join(str(x) for x in arch)


def build_architectures(cfg: Config) -> list[tuple[int, ...]]:
    if cfg.max_layers < 1:
        raise ValueError("max_layers muss >= 1 sein")
    if cfg.max_layers > 3:
        raise ValueError("max_layers darf fuer diesen Vergleich maximal 3 sein")
    if not cfg.neurons:
        raise ValueError("neurons darf nicht leer sein")

    for n in cfg.neurons:
        if n <= 0:
            raise ValueError("Alle Neuronenwerte muessen > 0 sein")

    archs: list[tuple[int, ...]] = []
    for depth in range(1, cfg.max_layers + 1):
        for combo in itertools.product(cfg.neurons, repeat=depth):
            archs.append(tuple(int(v) for v in combo))
    return archs


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
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


def train_one_arch(
    arch: tuple[int, ...],
    cfg: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> ArchResult:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = VariableGRURegressor(len(FEATURES), arch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    crit = nn.MSELoss()

    name = arch_to_name(arch)
    res = ArchResult(
        arch=arch,
        arch_name=name,
        n_layers=len(arch),
        n_params=count_params(model),
    )

    print(f"\n=== GRU {name} ({res.n_params:,} Parameter) ===")
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
        vl, _, _, _ = evaluate(model, val_loader, device)
        res.train_loss.append(tr)
        res.val_loss.append(vl)
        print(f"  Epoch {epoch:02d}/{cfg.epochs}  train_mse={tr:.6f}  val_mse={vl:.6f}")

    res.train_time_s = time.time() - t0

    _, yt, yp, ypr_val = evaluate(model, val_loader, device)
    _, yt_test, yp_test, ypr_test = evaluate(model, test_loader, device)

    res.metrics = regression_metrics(yt, yp)
    res.test_metrics = regression_metrics(yt_test, yp_test)
    res.y_true = yt
    res.y_pred = yp
    res._y_prev_val = ypr_val
    res._y_true_test = yt_test
    res._y_prev_test = ypr_test
    return res


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def sorted_results_by_test_mse(results: list[ArchResult], desc: bool) -> list[ArchResult]:
    return sorted(results, key=lambda r: r.test_metrics["mse"], reverse=desc)


def aggregate_by_depth(results: list[ArchResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "n_layers": r.n_layers,
                "val_mse": r.metrics["mse"],
                "val_rmse": r.metrics["rmse"],
                "val_mae": r.metrics["mae"],
                "val_r2": r.metrics["r2"],
                "test_mse": r.test_metrics["mse"],
                "test_rmse": r.test_metrics["rmse"],
                "test_r2": r.test_metrics["r2"],
                "train_time_s": r.train_time_s,
            }
        )

    df = pd.DataFrame(rows)
    grp = (
        df.groupby("n_layers", as_index=False)
        .agg(
            mean_val_mse=("val_mse", "mean"),
            median_val_mse=("val_mse", "median"),
            best_val_mse=("val_mse", "min"),
            mean_test_mse=("test_mse", "mean"),
            mean_val_r2=("val_r2", "mean"),
            mean_test_r2=("test_r2", "mean"),
            mean_train_time_s=("train_time_s", "mean"),
            n_runs=("val_mse", "count"),
        )
        .sort_values("n_layers")
    )
    return grp


def best_arch_per_depth(results: list[ArchResult]) -> pd.DataFrame:
    rows = []
    for depth in sorted(set(r.n_layers for r in results)):
        subset = [r for r in results if r.n_layers == depth]
        best = sorted(subset, key=lambda r: r.test_metrics["mse"])[0]
        rows.append(
            {
                "n_layers": depth,
                "arch": best.arch_name,
                "val_mse": best.metrics["mse"],
                "val_rmse": best.metrics["rmse"],
                "val_r2": best.metrics["r2"],
                "test_mse": best.test_metrics["mse"],
                "test_rmse": best.test_metrics["rmse"],
                "test_r2": best.test_metrics["r2"],
                "train_time_s": best.train_time_s,
                "n_params": best.n_params,
            }
        )
    return pd.DataFrame(rows).sort_values("n_layers")


def plot_arch_ranking(results: list[ArchResult], sort_desc: bool) -> str:
    ordered = sorted_results_by_test_mse(results, desc=sort_desc)
    labels = [f"{r.arch_name} ({r.n_layers}L)" for r in ordered]
    vals = [r.test_metrics["mse"] for r in ordered]
    colors = [DEPTH_COLORS.get(r.n_layers, "#808080") for r in ordered]

    fig, ax = plt.subplots(figsize=(12, max(4, 0.34 * len(ordered))))
    bars = ax.barh(labels, vals, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Test-MSE")
    ax.set_title(
        "GRU-Architekturen sortiert nach Test-MSE "
        f"({'absteigend' if sort_desc else 'aufsteigend'})"
    )
    ax.grid(alpha=0.3, axis="x")
    for b, v in zip(bars, vals):
        ax.text(v, b.get_y() + b.get_height() / 2, f" {v:.5f}", va="center", fontsize=8)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=DEPTH_COLORS[d], label=f"{d} Layer")
        for d in sorted(DEPTH_COLORS)
    ]
    ax.legend(handles=handles, loc="lower right")

    fig.tight_layout()
    return fig_to_b64(fig)


def plot_depth_mean_mse(results: list[ArchResult]) -> str:
    grp = aggregate_by_depth(results)
    x = np.arange(len(grp))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bars1 = ax.bar(
        x - width / 2,
        grp["mean_val_mse"],
        width,
        label="mean Val-MSE",
        color="#4C72B0",
    )
    bars2 = ax.bar(
        x + width / 2,
        grp["mean_test_mse"],
        width,
        label="mean Test-MSE",
        color="#55A868",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(d)} Layer" for d in grp["n_layers"]])
    ax.set_ylabel("MSE")
    ax.set_title("Mittelwerte je Layer-Tiefe")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()

    for bars in (bars1, bars2):
        for b in bars:
            v = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2,
                v,
                f"{v:.5f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    return fig_to_b64(fig)


def plot_depth_distribution(results: list[ArchResult]) -> str:
    groups = []
    labels = []
    for depth in sorted(set(r.n_layers for r in results)):
        groups.append([r.test_metrics["mse"] for r in results if r.n_layers == depth])
        labels.append(f"{depth} Layer")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bp = ax.boxplot(groups, tick_labels=labels, patch_artist=True)
    for patch, depth in zip(bp["boxes"], sorted(set(r.n_layers for r in results))):
        patch.set_facecolor(DEPTH_COLORS.get(depth, "#cccccc"))
        patch.set_alpha(0.65)

    ax.set_title("Test-MSE Verteilung je Layer-Tiefe")
    ax.set_ylabel("Test-MSE")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_loss_curves(results: list[ArchResult], top_k: int = 12) -> str:
    ordered = sorted(results, key=lambda r: r.test_metrics["mse"])[: max(1, top_k)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    for r in ordered:
        ep = range(1, len(r.train_loss) + 1)
        label = f"{r.arch_name} ({r.n_layers}L)"
        axes[0].plot(ep, r.train_loss, lw=1.2, alpha=0.85, label=label)
        axes[1].plot(ep, r.val_loss, lw=1.2, alpha=0.85, label=label)

    axes[0].set_title(f"Train-MSE Kurven (Top {len(ordered)})")
    axes[1].set_title(f"Val-MSE Kurven (Top {len(ordered)})")
    for ax in axes:
        ax.set_xlabel("Epoche")
        ax.set_ylabel("MSE")
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=8, ncol=2)

    fig.tight_layout()
    return fig_to_b64(fig)


def plot_best_per_depth(results: list[ArchResult]) -> str:
    df = best_arch_per_depth(results)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(df))
    bars = ax.bar(
        x,
        df["test_mse"],
        color=[DEPTH_COLORS.get(int(d), "#888888") for d in df["n_layers"]],
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(d)} Layer" for d in df["n_layers"]])
    ax.set_ylabel("Bestes Test-MSE")
    ax.set_title("Bestes Modell je Layer-Tiefe (nach Test-MSE)")
    ax.grid(alpha=0.3, axis="y")

    for b, arch, v in zip(bars, df["arch"], df["test_mse"]):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v,
            f"{arch}\n{v:.5f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    return fig_to_b64(fig)


def build_report(
    results: list[ArchResult],
    cfg: Config,
    baseline_val: dict,
    baseline_test: dict,
    meta: dict,
) -> str:
    imgs = {
        "ranking": plot_arch_ranking(results, cfg.sort_desc),
        "depth_mean": plot_depth_mean_mse(results),
        "depth_dist": plot_depth_distribution(results),
        "curves": plot_loss_curves(results, cfg.top_loss_curves),
        "best_depth": plot_best_per_depth(results),
    }

    ranked = sorted(results, key=lambda r: r.test_metrics["mse"])  # best first
    best = ranked[0]

    rows = []
    for rank, r in enumerate(ranked, start=1):
        star = " &#11088;" if rank == 1 else ""
        rows.append(
            "<tr>"
            f"<td>{rank}</td>"
            f"<td><b>{r.arch_name}{star}</b></td>"
            f"<td>{r.n_layers}</td>"
            f"<td>{r.n_params:,}</td>"
            f"<td>{r.metrics['mse']:.6f}</td>"
            f"<td>{r.metrics['rmse']:.6f}</td>"
            f"<td>{r.metrics['mae']:.6f}</td>"
            f"<td>{r.metrics['r2']:.4f}</td>"
            f"<td>{r.test_metrics['mse']:.6f}</td>"
            f"<td>{r.test_metrics['rmse']:.6f}</td>"
            f"<td>{r.test_metrics['r2']:.4f}</td>"
            f"<td>{r.train_time_s:.1f}s</td>"
            "</tr>"
        )
    rows_html = "\n".join(rows)

    depth_df = aggregate_by_depth(results)
    best_depth_df = best_arch_per_depth(results)

    depth_rows = []
    for _, row in depth_df.iterrows():
        depth_rows.append(
            "<tr>"
            f"<td>{int(row['n_layers'])}</td>"
            f"<td>{row['mean_val_mse']:.6f}</td>"
            f"<td>{row['median_val_mse']:.6f}</td>"
            f"<td>{row['best_val_mse']:.6f}</td>"
            f"<td>{row['mean_test_mse']:.6f}</td>"
            f"<td>{row['mean_val_r2']:.4f}</td>"
            f"<td>{row['mean_test_r2']:.4f}</td>"
            f"<td>{row['mean_train_time_s']:.1f}s</td>"
            f"<td>{int(row['n_runs'])}</td>"
            "</tr>"
        )
    depth_rows_html = "\n".join(depth_rows)

    best_depth_rows = []
    for _, row in best_depth_df.iterrows():
        best_depth_rows.append(
            "<tr>"
            f"<td>{int(row['n_layers'])}</td>"
            f"<td><b>{row['arch']}</b></td>"
            f"<td>{int(row['n_params']):,}</td>"
            f"<td>{row['val_mse']:.6f}</td>"
            f"<td>{row['val_rmse']:.6f}</td>"
            f"<td>{row['val_r2']:.4f}</td>"
            f"<td>{row['test_mse']:.6f}</td>"
            f"<td>{row['test_rmse']:.6f}</td>"
            f"<td>{row['test_r2']:.4f}</td>"
            f"<td>{row['train_time_s']:.1f}s</td>"
            "</tr>"
        )
    best_depth_rows_html = "\n".join(best_depth_rows)

    html = f"""<!DOCTYPE html>
<html lang="de"><head><meta charset="utf-8">
<title>GRU Architekturvergleich</title>
<style>
 body {{ font-family: Segoe UI, Arial, sans-serif; margin: 32px; color:#222; max-width:1250px; }}
 h1 {{ border-bottom: 3px solid #4C72B0; padding-bottom:6px; }}
 h2 {{ margin-top: 34px; color:#333; }}
 table {{ border-collapse: collapse; margin: 12px 0; font-size: 14px; width:100%; }}
 th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: right; }}
 th {{ background:#f0f3f7; }}
 td:first-child, th:first-child {{ text-align: left; }}
 img {{ max-width: 100%; border:1px solid #eee; border-radius:6px; margin: 8px 0 18px; }}
 .note {{ color:#666; font-size:13px; margin:4px 0 10px; }}
 .cfg {{ background:#f7f9fb; border:1px solid #e2e8f0; border-radius:6px; padding:10px 16px; font-size:13px; }}
 .rec {{ background:#eef7ee; border-left:5px solid #55A868; padding:12px 18px; border-radius:4px; }}
</style></head>
<body>
<h1>GRU Architekturvergleich</h1>
<p>Bruteforce ueber alle GRU-Architekturen mit 1 bis {cfg.max_layers} Layern und
Neuronen aus {{{', '.join(str(n) for n in cfg.neurons)}}} pro Layer.</p>

<div class="rec">
<b>Bestes Modell (Test-MSE):</b> <b>{best.arch_name}</b> ({best.n_layers} Layer)<br>
Val: MSE={best.metrics['mse']:.6f}, RMSE={best.metrics['rmse']:.6f}, R&sup2;={best.metrics['r2']:.4f}<br>
Test: MSE={best.test_metrics['mse']:.6f}, RMSE={best.test_metrics['rmse']:.6f}, R&sup2;={best.test_metrics['r2']:.4f}<br>
Parameter: {best.n_params:,}
</div>

<h2>Setup</h2>
<div class="cfg">
History H={cfg.seq_len} Schritte ({cfg.seq_len*0.2:.1f}s @ 5Hz), Horizon={cfg.horizon} ({cfg.horizon*200}ms),
Stride={cfg.stride}, Epochen={cfg.epochs}, Batch={cfg.batch_size}, LR={cfg.lr}<br>
Train-Trips: {meta['n_train_trips']} &middot; Val-Trips: {meta['n_val_trips']} &middot; Test-Trips: {meta['n_test_trips']}<br>
Train-Fenster: {meta['n_train_windows']:,} &middot; Val-Fenster: {meta['n_val_windows']:,} &middot; Test-Fenster: {meta['n_test_windows']:,}<br>
Architekturen: {meta['n_architectures']} (erwartet: {meta['expected_architectures']})
</div>

<h2>Vollstaendiges Ranking (sortiert nach niedrigster Test-MSE)</h2>
<table>
<tr><th>Rang</th><th>Architektur (Neuronen je Layer)</th><th>Layer</th><th>Parameter</th>
<th>Val MSE</th><th>Val RMSE</th><th>Val MAE</th><th>Val R&sup2;</th>
<th>Test MSE</th><th>Test RMSE</th><th>Test R&sup2;</th><th>Trainingszeit</th></tr>
{rows_html}
<tr style='color:#888'><td>-</td><td><i>Persistence</i></td><td>-</td><td>0</td>
<td>{baseline_val['mse']:.6f}</td><td>{baseline_val['rmse']:.6f}</td><td>{baseline_val['mae']:.6f}</td><td>{baseline_val['r2']:.4f}</td>
<td>{baseline_test['mse']:.6f}</td><td>{baseline_test['rmse']:.6f}</td><td>{baseline_test['r2']:.4f}</td><td>-</td></tr>
</table>

<h2>Mittelwerte je Layer-Tiefe (dein 1L/2L/3L Vergleich)</h2>
<table>
<tr><th>Layer</th><th>mean Val-MSE</th><th>median Val-MSE</th><th>best Val-MSE</th>
<th>mean Test-MSE</th><th>mean Val-R&sup2;</th><th>mean Test-R&sup2;</th><th>mean Trainingszeit</th><th>Runs</th></tr>
{depth_rows_html}
</table>

<h2>Bestes Modell je Layer-Tiefe (ausgewaehlt nach Test-MSE)</h2>
<table>
<tr><th>Layer</th><th>Architektur</th><th>Parameter</th><th>Val MSE</th><th>Val RMSE</th><th>Val R&sup2;</th>
<th>Test MSE</th><th>Test RMSE</th><th>Test R&sup2;</th><th>Trainingszeit</th></tr>
{best_depth_rows_html}
</table>

<h2>Plot: Alle Architekturen nach Test-MSE sortiert</h2>
<img src='data:image/png;base64,{imgs['ranking']}'/>

<h2>Plot: Mean-MSE je Layer-Tiefe</h2>
<img src='data:image/png;base64,{imgs['depth_mean']}'/>

<h2>Plot: Test-MSE Verteilung je Layer-Tiefe</h2>
<img src='data:image/png;base64,{imgs['depth_dist']}'/>

<h2>Plot: Bestes Modell je Layer-Tiefe (nach Test-MSE)</h2>
<img src='data:image/png;base64,{imgs['best_depth']}'/>

<h2>Plot: Train/Val-Kurven (Top-K nach Test-MSE)</h2>
<p class="note">Nur die Top-{cfg.top_loss_curves} zur besseren Lesbarkeit.</p>
<img src='data:image/png;base64,{imgs['curves']}'/>

<p class="note">Erzeugt am {meta['timestamp']} &middot; Gesamtzeit {meta['total_time']:.0f}s &middot; Device: {meta['device']}.</p>
</body></html>"""
    return html


def save_outputs(results: list[ArchResult], cfg: Config, meta_base: dict, t_start: float):
    if not results:
        return

    y_true_ref = results[0].y_true
    y_prev_val_ref = results[0]._y_prev_val
    y_true_test_ref = results[0]._y_true_test
    y_prev_test_ref = results[0]._y_prev_test
    assert y_true_ref is not None
    assert y_prev_val_ref is not None
    assert y_true_test_ref is not None
    assert y_prev_test_ref is not None

    baseline_val = regression_metrics(y_true_ref, y_prev_val_ref)
    baseline_test = regression_metrics(y_true_test_ref, y_prev_test_ref)

    rows = []
    for r in results:
        rows.append(
            {
                "split": "val",
                "architecture": r.arch_name,
                "n_layers": r.n_layers,
                "neurons_per_layer": ";".join(str(v) for v in r.arch),
                "n_params": r.n_params,
                "train_time_s": r.train_time_s,
                **r.metrics,
            }
        )
        rows.append(
            {
                "split": "test",
                "architecture": r.arch_name,
                "n_layers": r.n_layers,
                "neurons_per_layer": ";".join(str(v) for v in r.arch),
                "n_params": r.n_params,
                "train_time_s": r.train_time_s,
                **r.test_metrics,
            }
        )

    rows.append(
        {
            "split": "val",
            "architecture": "Persistence",
            "n_layers": 0,
            "neurons_per_layer": "a_t",
            "n_params": 0,
            "train_time_s": 0.0,
            **baseline_val,
        }
    )
    rows.append(
        {
            "split": "test",
            "architecture": "Persistence",
            "n_layers": 0,
            "neurons_per_layer": "a_t",
            "n_params": 0,
            "train_time_s": 0.0,
            **baseline_test,
        }
    )

    metrics_df = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(exist_ok=True)
    metrics_df.to_csv(REPORTS_DIR / METRICS_CSV_NAME, index=False)
    (REPORTS_DIR / METRICS_JSON_NAME).write_text(json.dumps(rows, indent=2), encoding="utf-8")

    meta = {
        **meta_base,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_time": time.time() - t_start,
    }
    html = build_report(results, cfg, baseline_val, baseline_test, meta)
    (REPORTS_DIR / REPORT_NAME).write_text(html, encoding="utf-8")

    ranked = sorted(results, key=lambda r: r.test_metrics["mse"])
    best = ranked[0]
    print("\n" + "=" * 68)
    print(f"ZWISCHENSTAND gespeichert (fertig: {len(results)} Architekturen)")
    print(
        f"Aktuell bestes Modell: {best.arch_name} ({best.n_layers}L) | "
        f"Test MSE={best.test_metrics['mse']:.6f} | Val MSE={best.metrics['mse']:.6f}"
    )
    print(f"Report: {REPORTS_DIR / REPORT_NAME}")
    print(f"Metriken: {REPORTS_DIR / METRICS_CSV_NAME}, {REPORTS_DIR / METRICS_JSON_NAME}")


def parse_int_csv(raw: str) -> list[int]:
    items = [x.strip() for x in raw.split(",") if x.strip()]
    if not items:
        raise ValueError("Leere Neuronenliste ist nicht erlaubt")

    vals: list[int] = []
    seen = set()
    for i in items:
        v = int(i)
        if v not in seen:
            vals.append(v)
            seen.add(v)
    return vals


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="GRU Bruteforce fuer Architekturvergleich.")
    p.add_argument("--full", action="store_true", help="Alle Trips (Fraction=1.0).")
    p.add_argument("--fraction", type=float, default=0.25)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seq-len", type=int, default=30)
    p.add_argument("--horizon", type=int, default=25)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-trips", type=int, default=None)
    p.add_argument("--max-val-trips", type=int, default=None)
    p.add_argument("--max-test-trips", type=int, default=None)
    p.add_argument("--max-train-windows", type=int, default=None)
    p.add_argument("--max-val-windows", type=int, default=None)
    p.add_argument("--max-test-windows", type=int, default=None)

    p.add_argument("--max-layers", type=int, default=3)
    p.add_argument("--neurons", type=str, default="64,96,128")

    sort_group = p.add_mutually_exclusive_group()
    sort_group.add_argument(
        "--sort-desc",
        dest="sort_desc",
        action="store_true",
        help="Sortierung in Plots absteigend nach Test-MSE.",
    )
    sort_group.add_argument(
        "--sort-asc",
        dest="sort_desc",
        action="store_false",
        help="Sortierung in Plots aufsteigend nach Test-MSE (Default).",
    )
    p.add_argument("--top-loss-curves", type=int, default=12)
    p.set_defaults(sort_desc=False)

    a = p.parse_args()
    cfg = Config(
        seq_len=a.seq_len,
        horizon=a.horizon,
        stride=a.stride,
        fraction=1.0 if a.full else a.fraction,
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        seed=a.seed,
        max_train_trips=a.max_train_trips,
        max_val_trips=a.max_val_trips,
        max_test_trips=a.max_test_trips,
        max_train_windows=a.max_train_windows,
        max_val_windows=a.max_val_windows,
        max_test_windows=a.max_test_windows,
        max_layers=a.max_layers,
        neurons=parse_int_csv(a.neurons),
        sort_desc=a.sort_desc,
        top_loss_curves=max(1, a.top_loss_curves),
    )
    return cfg


def main():
    cfg = parse_args()
    t_start = time.time()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    archs = build_architectures(cfg)
    expected_count = sum(len(cfg.neurons) ** d for d in range(1, cfg.max_layers + 1))

    print("Konfiguration:", cfg)
    print("Device:", device)
    print(
        f"Architekturen: {len(archs)} (Neuronenoptionen={cfg.neurons}, "
        f"max_layers={cfg.max_layers}, erwartet={expected_count})"
    )

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

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    meta_base = {
        "n_train_trips": len(train_trips),
        "n_val_trips": len(val_trips),
        "n_test_trips": len(test_trips),
        "n_train_windows": len(train_ds),
        "n_val_windows": len(val_ds),
        "n_test_windows": len(test_ds),
        "n_architectures": len(archs),
        "expected_architectures": expected_count,
        "device": str(device),
    }

    results: list[ArchResult] = []
    for i, arch in enumerate(archs, start=1):
        print("\n" + "-" * 68)
        print(f"Run {i}/{len(archs)}: architecture={arch_to_name(arch)}")
        res = train_one_arch(
            arch=arch,
            cfg=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
        )
        results.append(res)
        # Inkrementelles Speichern fuer lange Runs.
        save_outputs(results, cfg, meta_base, t_start)

    print("\nFERTIG.")
    best = sorted(results, key=lambda r: r.test_metrics["mse"])[0]
    print(
        f"Bestes Modell nach Test-MSE: {best.arch_name} ({best.n_layers}L) "
        f"(Test MSE={best.test_metrics['mse']:.6f}, Val MSE={best.metrics['mse']:.6f})"
    )


if __name__ == "__main__":
    main()
