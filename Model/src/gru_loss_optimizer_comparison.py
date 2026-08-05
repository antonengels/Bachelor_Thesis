r"""
GRU-Only Bruteforce fuer Optimizer- und Loss-Funktionsvergleich.

Idee:
    Nutzt dieselbe Datenaufbereitung/Windowing-Logik wie `model_comparison.py`,
    trainiert aber nur ein GRU-Modell und sweeped ueber alle Kombinationen aus
    Optimizer x Loss-Funktion.

Ziel:
    - Kombinations-Ranking nach Validierungs-MSE
    - Aggregierte Rankings je Optimizer und je Loss-Funktion
    - Self-contained HTML-Report mit Plots (base64)

Beispiel:
    .\.venv\Scripts\python.exe src\gru_loss_optimizer_comparison.py

Optionen wie beim Architekturvergleich:
    --full --fraction --epochs --seq-len --horizon --stride --batch-size --hidden
    --lr --seed --max-*-trips --max-*-windows

Neue Optionen:
    --optimizers adam,adamw,rmsprop,sgd,adagrad
    --losses mse,mae,smooth_l1,huber,log_cosh
    --sort-desc      (Plot-Sortierung absteigend nach Val-MSE; Default)
    --sort-asc       (Plot-Sortierung aufsteigend nach Val-MSE)
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model_comparison import (
    FEATURES,
    GRURegressor,
    SequenceDataset,
    build_trip_arrays,
    cap_dataset,
    load_split,
    regression_metrics,
)


ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"

REPORT_NAME = "gru_loss_optimizer_report.html"
METRICS_CSV_NAME = "gru_loss_optimizer_metrics.csv"
METRICS_JSON_NAME = "gru_loss_optimizer_metrics.json"


@dataclass
class Config:
    seq_len: int = 100
    horizon: int = 50
    stride: int = 5
    fraction: float = 0.20
    epochs: int = 10
    batch_size: int = 256
    hidden: int = 64
    lr: float = 1e-3
    seed: int = 42
    max_train_trips: int | None = None
    max_val_trips: int | None = None
    max_test_trips: int | None = None
    max_train_windows: int | None = None
    max_val_windows: int | None = None
    max_test_windows: int | None = None

    weight_decay: float = 1e-4
    momentum: float = 0.9
    smooth_l1_beta: float = 0.10
    huber_delta: float = 0.10
    losses: list[str] = field(default_factory=lambda: [
        "mse",
        "mae",
        "smooth_l1",
        "huber",
        "log_cosh",
    ])
    optimizers: list[str] = field(default_factory=lambda: [
        "adam",
        "adamw",
        "rmsprop",
        "sgd",
        "adagrad",
    ])
    sort_desc: bool = True
    top_loss_curves: int = 12


@dataclass
class ComboResult:
    combo_name: str
    optimizer_name: str
    loss_name: str
    train_objective: list[float] = field(default_factory=list)
    val_objective: list[float] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    test_metrics: dict = field(default_factory=dict)
    val_obj_final: float = float("nan")
    test_obj_final: float = float("nan")
    n_params: int = 0
    train_time_s: float = 0.0
    y_true: np.ndarray | None = None
    y_pred: np.ndarray | None = None
    _y_prev_val: np.ndarray | None = None
    _y_true_test: np.ndarray | None = None
    _y_prev_test: np.ndarray | None = None


class LogCoshLoss(nn.Module):
    """Numerisch stabile log-cosh Regression-Loss."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = pred - target
        # log(cosh(x)) = x + softplus(-2x) - log(2)
        return torch.mean(x + F.softplus(-2.0 * x) - math.log(2.0))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_loss(name: str, cfg: Config) -> nn.Module:
    n = name.strip().lower()
    if n == "mse":
        return nn.MSELoss(reduction="mean")
    if n in {"mae", "l1"}:
        return nn.L1Loss(reduction="mean")
    if n == "smooth_l1":
        return nn.SmoothL1Loss(beta=cfg.smooth_l1_beta, reduction="mean")
    if n == "huber":
        return nn.HuberLoss(delta=cfg.huber_delta, reduction="mean")
    if n == "log_cosh":
        return LogCoshLoss()
    raise ValueError(f"Unbekannte Loss-Funktion: {name}")


def build_optimizer(name: str, params, cfg: Config):
    n = name.strip().lower()
    if n == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if n == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if n == "rmsprop":
        return torch.optim.RMSprop(
            params,
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    if n == "sgd":
        return torch.optim.SGD(
            params,
            lr=cfg.lr,
            momentum=cfg.momentum,
            nesterov=True,
            weight_decay=cfg.weight_decay,
        )
    if n == "adagrad":
        return torch.optim.Adagrad(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unbekannter Optimizer: {name}")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    objective_fn: nn.Module,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Gibt objective_mean, mse_mean, y_true, y_pred, y_prev zurueck."""
    model.eval()
    obj_total, mse_total, n = 0.0, 0.0, 0
    yt, yp, ypr = [], [], []

    for x, y, y_prev in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        obj_batch = objective_fn(pred, y)
        obj_total += obj_batch.item() * y.numel()
        mse_total += torch.sum((pred - y) ** 2).item()
        n += y.numel()
        yt.append(y.cpu().numpy())
        yp.append(pred.cpu().numpy())
        ypr.append(y_prev.numpy())

    denom = max(n, 1)
    return (
        obj_total / denom,
        mse_total / denom,
        np.concatenate(yt),
        np.concatenate(yp),
        np.concatenate(ypr),
    )


def train_one_combo(
    optimizer_name: str,
    loss_name: str,
    cfg: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> ComboResult:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = GRURegressor(len(FEATURES), cfg.hidden).to(device)
    objective_fn = build_loss(loss_name, cfg)
    optimizer = build_optimizer(optimizer_name, model.parameters(), cfg)
    combo_name = f"{optimizer_name.upper()} + {loss_name}"

    res = ComboResult(
        combo_name=combo_name,
        optimizer_name=optimizer_name,
        loss_name=loss_name,
        n_params=count_params(model),
    )

    print(f"\n=== {combo_name} ({res.n_params:,} Parameter) ===")
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        run_obj, n = 0.0, 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = objective_fn(pred, y)
            loss.backward()
            optimizer.step()
            run_obj += loss.item() * y.numel()
            n += y.numel()

        tr_obj = run_obj / max(n, 1)
        vl_obj, vl_mse, _, _, _ = evaluate(model, val_loader, device, objective_fn)
        res.train_objective.append(tr_obj)
        res.val_objective.append(vl_obj)
        print(
            f"  Epoch {epoch:02d}/{cfg.epochs}  train_obj={tr_obj:.6f}  "
            f"val_obj={vl_obj:.6f}  val_mse={vl_mse:.6f}"
        )

    res.train_time_s = time.time() - t0

    val_obj, _, yt, yp, ypr_val = evaluate(model, val_loader, device, objective_fn)
    test_obj, _, yt_test, yp_test, ypr_test = evaluate(model, test_loader, device, objective_fn)
    res.val_obj_final = val_obj
    res.test_obj_final = test_obj
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


def sorted_results_by_val_mse(results: list[ComboResult], desc: bool) -> list[ComboResult]:
    return sorted(results, key=lambda r: r.metrics["mse"], reverse=desc)


def color_map_for_losses(losses: list[str]) -> dict[str, str]:
    cmap = plt.get_cmap("tab10")
    uniq = sorted(set(losses))
    return {name: matplotlib.colors.to_hex(cmap(i % 10)) for i, name in enumerate(uniq)}


def plot_combo_ranking(results: list[ComboResult], sort_desc: bool) -> str:
    ordered = sorted_results_by_val_mse(results, desc=sort_desc)
    labels = [r.combo_name for r in ordered]
    vals = [r.metrics["mse"] for r in ordered]
    loss_colors = color_map_for_losses([r.loss_name for r in ordered])
    colors = [loss_colors[r.loss_name] for r in ordered]

    fig, ax = plt.subplots(figsize=(12, max(4, 0.38 * len(ordered))))
    bars = ax.barh(labels, vals, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Validierungs-MSE")
    ax.set_title(
        "GRU-Kombinationen sortiert nach Val-MSE "
        f"({'absteigend' if sort_desc else 'aufsteigend'})"
    )
    ax.grid(alpha=0.3, axis="x")
    for b, v in zip(bars, vals):
        ax.text(v, b.get_y() + b.get_height() / 2, f" {v:.5f}", va="center", fontsize=8)

    fig.tight_layout()
    return fig_to_b64(fig)


def _aggregate_by_key(results: list[ComboResult], key: str) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                key: getattr(r, key),
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
        df.groupby(key, as_index=False)
        .agg(
            mean_val_mse=("val_mse", "mean"),
            median_val_mse=("val_mse", "median"),
            best_val_mse=("val_mse", "min"),
            mean_test_mse=("test_mse", "mean"),
            mean_train_time_s=("train_time_s", "mean"),
            n_runs=("val_mse", "count"),
        )
        .sort_values("mean_val_mse", ascending=True)
    )
    return grp


def plot_optimizer_ranking(results: list[ComboResult], sort_desc: bool) -> str:
    df = _aggregate_by_key(results, "optimizer_name")
    df = df.sort_values("mean_val_mse", ascending=not sort_desc)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    bars = ax.bar(df["optimizer_name"], df["mean_val_mse"], color="#4C72B0")
    ax.set_title(
        "Optimizer-Ranking (Mittelwert Val-MSE ueber alle Losses) "
        f"({'absteigend' if sort_desc else 'aufsteigend'})"
    )
    ax.set_ylabel("mean(Val-MSE)")
    ax.grid(alpha=0.3, axis="y")
    for b, v in zip(bars, df["mean_val_mse"]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.5f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_loss_ranking(results: list[ComboResult], sort_desc: bool) -> str:
    df = _aggregate_by_key(results, "loss_name")
    df = df.sort_values("mean_val_mse", ascending=not sort_desc)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    bars = ax.bar(df["loss_name"], df["mean_val_mse"], color="#55A868")
    ax.set_title(
        "Loss-Ranking (Mittelwert Val-MSE ueber alle Optimizer) "
        f"({'absteigend' if sort_desc else 'aufsteigend'})"
    )
    ax.set_ylabel("mean(Val-MSE)")
    ax.grid(alpha=0.3, axis="y")
    for b, v in zip(bars, df["mean_val_mse"]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.5f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_heatmap(results: list[ComboResult]) -> str:
    opts = sorted(set(r.optimizer_name for r in results))
    losses = sorted(set(r.loss_name for r in results))
    mat = np.full((len(opts), len(losses)), np.nan, dtype=float)

    pos_opt = {o: i for i, o in enumerate(opts)}
    pos_loss = {l: j for j, l in enumerate(losses)}
    for r in results:
        mat[pos_opt[r.optimizer_name], pos_loss[r.loss_name]] = r.metrics["mse"]

    fig, ax = plt.subplots(figsize=(1.8 * len(losses) + 1.5, 0.8 * len(opts) + 2.2))
    im = ax.imshow(mat, aspect="auto", cmap="viridis_r")
    ax.set_xticks(np.arange(len(losses)))
    ax.set_yticks(np.arange(len(opts)))
    ax.set_xticklabels(losses, rotation=30, ha="right")
    ax.set_yticklabels(opts)
    ax.set_title("Val-MSE Heatmap (Optimizer x Loss)")

    for i in range(len(opts)):
        for j in range(len(losses)):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.4f}", ha="center", va="center", color="white", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Val-MSE")
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_loss_curves(results: list[ComboResult], top_k: int = 12) -> str:
    # Fuer Lesbarkeit nur beste top_k nach Val-MSE zeigen.
    ordered = sorted(results, key=lambda r: r.metrics["mse"])[:max(1, top_k)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    for res in ordered:
        label = f"{res.optimizer_name}/{res.loss_name}"
        ep = range(1, len(res.train_objective) + 1)
        axes[0].plot(ep, res.train_objective, lw=1.2, alpha=0.85, label=label)
        axes[1].plot(ep, res.val_objective, lw=1.2, alpha=0.85, label=label)

    axes[0].set_title(f"Train-Objective (Top {len(ordered)})")
    axes[1].set_title(f"Val-Objective (Top {len(ordered)})")
    for ax in axes:
        ax.set_xlabel("Epoche")
        ax.set_ylabel("Objective")
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig_to_b64(fig)


def build_report(
    results: list[ComboResult],
    cfg: Config,
    baseline_val: dict,
    baseline_test: dict,
    meta: dict,
) -> str:
    imgs = {
        "combo": plot_combo_ranking(results, cfg.sort_desc),
        "opt": plot_optimizer_ranking(results, cfg.sort_desc),
        "loss": plot_loss_ranking(results, cfg.sort_desc),
        "heatmap": plot_heatmap(results),
        "curves": plot_loss_curves(results, cfg.top_loss_curves),
    }

    ranked = sorted(results, key=lambda r: r.metrics["mse"])  # best first
    best = ranked[0]
    best_loss_name = (
        _aggregate_by_key(results, "loss_name").sort_values("mean_val_mse", ascending=True).iloc[0]["loss_name"]
    )
    best_opt_name = (
        _aggregate_by_key(results, "optimizer_name").sort_values("mean_val_mse", ascending=True).iloc[0][
            "optimizer_name"
        ]
    )

    rows = []
    for rank, r in enumerate(ranked, start=1):
        star = " &#11088;" if rank == 1 else ""
        rows.append(
            "<tr>"
            f"<td>{rank}</td>"
            f"<td><b>{r.optimizer_name}{star}</b></td>"
            f"<td>{r.loss_name}</td>"
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

    opt_df = _aggregate_by_key(results, "optimizer_name")
    loss_df = _aggregate_by_key(results, "loss_name")

    def aggregate_table_html(df: pd.DataFrame, title_col: str) -> str:
        out = [
            "<table>",
            "<tr>"
            f"<th>{title_col}</th><th>mean Val-MSE</th><th>median Val-MSE</th>"
            "<th>best Val-MSE</th><th>mean Test-MSE</th><th>mean Trainingszeit</th><th>Runs</th>"
            "</tr>",
        ]
        for _, row in df.iterrows():
            out.append(
                "<tr>"
                f"<td>{row[title_col]}</td>"
                f"<td>{row['mean_val_mse']:.6f}</td>"
                f"<td>{row['median_val_mse']:.6f}</td>"
                f"<td>{row['best_val_mse']:.6f}</td>"
                f"<td>{row['mean_test_mse']:.6f}</td>"
                f"<td>{row['mean_train_time_s']:.1f}s</td>"
                f"<td>{int(row['n_runs'])}</td>"
                "</tr>"
            )
        out.append("</table>")
        return "\n".join(out)

    opt_table = aggregate_table_html(opt_df, "optimizer_name")
    loss_table = aggregate_table_html(loss_df, "loss_name")

    html = f"""<!DOCTYPE html>
<html lang="de"><head><meta charset="utf-8">
<title>GRU Bruteforce: Optimizer x Loss</title>
<style>
 body {{ font-family: Segoe UI, Arial, sans-serif; margin: 32px; color:#222; max-width:1200px; }}
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
<h1>GRU Bruteforce: Optimizer vs. Loss-Funktion</h1>
<p>Trainiert auf <b>{cfg.fraction*100:.0f}%</b> der Trips, alle Kombinationen aus
Optimizer und Loss-Funktion werden durchlaufen.</p>

<div class="rec">
<b>Beste Kombination (Val-MSE):</b> <b>{best.optimizer_name} + {best.loss_name}</b><br>
Val: MSE={best.metrics['mse']:.6f}, RMSE={best.metrics['rmse']:.6f}, R&sup2;={best.metrics['r2']:.4f}<br>
Test: MSE={best.test_metrics['mse']:.6f}, RMSE={best.test_metrics['rmse']:.6f}, R&sup2;={best.test_metrics['r2']:.4f}<br>
<b>Bestes Loss-Familien-Ranking (mean Val-MSE):</b> {best_loss_name}<br>
<b>Bester Optimizer (mean Val-MSE):</b> {best_opt_name}
</div>

<h2>Setup</h2>
<div class="cfg">
History H={cfg.seq_len} Schritte ({cfg.seq_len*0.2:.1f}s @ 5Hz), Horizon={cfg.horizon} ({cfg.horizon*200}ms),
Stride={cfg.stride}, Epochen={cfg.epochs}, Batch={cfg.batch_size}, Hidden={cfg.hidden}, LR={cfg.lr}<br>
Weight Decay={cfg.weight_decay}, Momentum={cfg.momentum}, smooth_l1_beta={cfg.smooth_l1_beta}, huber_delta={cfg.huber_delta}<br>
Optimizer: {", ".join(cfg.optimizers)}<br>
Losses: {", ".join(cfg.losses)}<br>
Train-Trips: {meta['n_train_trips']} &middot; Val-Trips: {meta['n_val_trips']} &middot; Test-Trips: {meta['n_test_trips']}<br>
Train-Fenster: {meta['n_train_windows']:,} &middot; Val-Fenster: {meta['n_val_windows']:,} &middot; Test-Fenster: {meta['n_test_windows']:,}
</div>

<h2>Kombinations-Ranking (sortiert nach niedrigster Val-MSE)</h2>
<table>
<tr><th>Rang</th><th>Optimizer</th><th>Loss</th><th>Val MSE</th><th>Val RMSE</th><th>Val MAE</th><th>Val R&sup2;</th>
<th>Test MSE</th><th>Test RMSE</th><th>Test R&sup2;</th><th>Trainingszeit</th></tr>
{rows_html}
<tr style='color:#888'><td>-</td><td><i>Persistence</i></td><td>a_t</td>
<td>{baseline_val['mse']:.6f}</td><td>{baseline_val['rmse']:.6f}</td><td>{baseline_val['mae']:.6f}</td><td>{baseline_val['r2']:.4f}</td>
<td>{baseline_test['mse']:.6f}</td><td>{baseline_test['rmse']:.6f}</td><td>{baseline_test['r2']:.4f}</td><td>-</td></tr>
</table>

<p class="note">Ranking erfolgt ueber Val-MSE (niedriger ist besser). Die Aggregat-Tabellen unten mitteln ueber alle Gegenkombinationen.</p>

<h2>Optimizer-Aggregat</h2>
{opt_table}

<h2>Loss-Aggregat</h2>
{loss_table}

<h2>Plot: Alle Kombinationen nach Val-MSE sortiert</h2>
<p class="note">Diese Ansicht zeigt die von dir gewuenschte Sortierung nach Loss-Wert ({'absteigend' if cfg.sort_desc else 'aufsteigend'}).</p>
<img src='data:image/png;base64,{imgs['combo']}'/>

<h2>Plot: Optimizer-Ranking</h2>
<img src='data:image/png;base64,{imgs['opt']}'/>

<h2>Plot: Loss-Ranking</h2>
<img src='data:image/png;base64,{imgs['loss']}'/>

<h2>Plot: Heatmap Optimizer x Loss</h2>
<img src='data:image/png;base64,{imgs['heatmap']}'/>

<h2>Plot: Train/Val-Objective-Kurven (Top-K)</h2>
<img src='data:image/png;base64,{imgs['curves']}'/>

<p class="note">Erzeugt am {meta['timestamp']} &middot; Gesamtzeit {meta['total_time']:.0f}s &middot; Device: {meta['device']}.</p>
</body></html>"""
    return html


def save_outputs(results: list[ComboResult], cfg: Config, meta_base: dict, t_start: float):
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
                "combo": r.combo_name,
                "optimizer": r.optimizer_name,
                "loss": r.loss_name,
                "objective": r.val_obj_final,
                "n_params": r.n_params,
                "train_time_s": r.train_time_s,
                **r.metrics,
            }
        )
        rows.append(
            {
                "split": "test",
                "combo": r.combo_name,
                "optimizer": r.optimizer_name,
                "loss": r.loss_name,
                "objective": r.test_obj_final,
                "n_params": r.n_params,
                "train_time_s": r.train_time_s,
                **r.test_metrics,
            }
        )

    rows.append(
        {
            "split": "val",
            "combo": "Persistence",
            "optimizer": "-",
            "loss": "a_t",
            "objective": baseline_val["mse"],
            "n_params": 0,
            "train_time_s": 0.0,
            **baseline_val,
        }
    )
    rows.append(
        {
            "split": "test",
            "combo": "Persistence",
            "optimizer": "-",
            "loss": "a_t",
            "objective": baseline_test["mse"],
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

    ranked = sorted(results, key=lambda r: r.metrics["mse"])
    best = ranked[0]
    print("\n" + "=" * 68)
    print(f"ZWISCHENSTAND gespeichert (fertig: {len(results)} Kombinationen)")
    print(
        f"Aktuell beste Kombi: {best.optimizer_name}+{best.loss_name} | "
        f"Val MSE={best.metrics['mse']:.6f} | Test MSE={best.test_metrics['mse']:.6f}"
    )
    print(f"Report: {REPORTS_DIR / REPORT_NAME}")
    print(f"Metriken: {REPORTS_DIR / METRICS_CSV_NAME}, {REPORTS_DIR / METRICS_JSON_NAME}")


def parse_csv_list(raw: str) -> list[str]:
    items = [x.strip().lower() for x in raw.split(",") if x.strip()]
    if not items:
        raise ValueError("Leere Liste ist nicht erlaubt.")
    # Reihenfolge behalten, Duplikate entfernen.
    seen = set()
    uniq = []
    for i in items:
        if i not in seen:
            uniq.append(i)
            seen.add(i)
    return uniq


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="GRU Bruteforce fuer Loss und Optimizer.")
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

    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--smooth-l1-beta", type=float, default=0.10)
    p.add_argument("--huber-delta", type=float, default=0.10)
    p.add_argument("--optimizers", type=str, default="adam,adamw,rmsprop,sgd,adagrad")
    p.add_argument("--losses", type=str, default="mse,mae,smooth_l1,huber,log_cosh")
    sort_group = p.add_mutually_exclusive_group()
    sort_group.add_argument(
        "--sort-desc",
        dest="sort_desc",
        action="store_true",
        help="Sortierung in Plots absteigend nach Val-MSE (Default).",
    )
    sort_group.add_argument(
        "--sort-asc",
        dest="sort_desc",
        action="store_false",
        help="Sortierung in Plots aufsteigend nach Val-MSE.",
    )
    p.add_argument("--top-loss-curves", type=int, default=12)
    p.set_defaults(sort_desc=True)

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
        weight_decay=a.weight_decay,
        momentum=a.momentum,
        smooth_l1_beta=a.smooth_l1_beta,
        huber_delta=a.huber_delta,
        optimizers=parse_csv_list(a.optimizers),
        losses=parse_csv_list(a.losses),
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

    print("Konfiguration:", cfg)
    print("Device:", device)
    print(
        f"Kombinationen: {len(cfg.optimizers)} Optimizer x {len(cfg.losses)} Losses "
        f"= {len(cfg.optimizers) * len(cfg.losses)} Runs"
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
        "device": str(device),
    }

    results: list[ComboResult] = []
    total = len(cfg.optimizers) * len(cfg.losses)
    done = 0
    for optimizer_name in cfg.optimizers:
        for loss_name in cfg.losses:
            done += 1
            print("\n" + "-" * 68)
            print(f"Run {done}/{total}: optimizer={optimizer_name}, loss={loss_name}")
            res = train_one_combo(
                optimizer_name=optimizer_name,
                loss_name=loss_name,
                cfg=cfg,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
            )
            results.append(res)
            # Inkrementell speichern fuer lange Runs.
            save_outputs(results, cfg, meta_base, t_start)

    print("\nFERTIG.")
    best = sorted(results, key=lambda r: r.metrics["mse"])[0]
    print(
        f"Beste Kombination nach Val-MSE: {best.optimizer_name}+{best.loss_name} "
        f"(Val MSE={best.metrics['mse']:.6f}, Test MSE={best.test_metrics['mse']:.6f})"
    )


if __name__ == "__main__":
    main()
