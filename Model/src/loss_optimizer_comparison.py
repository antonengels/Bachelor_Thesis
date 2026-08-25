r"""
LSTM-Only Bruteforce fuer Optimizer- und Loss-Funktionsvergleich.

Idee:
    Nutzt dieselbe Datenaufbereitung/Windowing-Logik wie `model_comparison.py`,
    trainiert aber nur ein LSTM-Modell mit fester Architektur und sweeped ueber
    alle Kombinationen aus Optimizer x Loss-Funktion.

Ziel:
    - Kombinations-Ranking ueber R2, RMSE, MAE, Richtungs- und Toleranz-Acc
    - Aggregierte Rankings je Optimizer und je Loss-Funktion
    - Self-contained HTML-Report mit Plots (base64)

Beispiel:
    .\.venv\Scripts\python.exe src\loss_optimizer_comparison.py

Optionen wie beim Architekturvergleich:
    --full --fraction --epochs --seq-len --horizon --stride --batch-size
    --num-workers --lr --seed --max-*-trips --max-*-windows

Neue Optionen:
    --lstm-architecture 128,64,256,128
    --optimizers adam,rmsprop,adagrad
    --losses mse,mae,smooth_l1,huber
    --parallel 2      (gleichzeitig trainierte Kombinationen auf CUDA)
    --sort-desc       (Sortierung absteigend nach Val-Score; Default)
    --sort-asc        (Sortierung aufsteigend nach Val-Score)

Fortsetzung:
    Nach jeder vollstaendig trainierten Kombination werden Ergebnis-Metadaten und
    Vorhersagen unter reports/lstm_loss_optimizer_checkpoints/ gespeichert. Bei
    identischer Konfiguration werden diese beim naechsten Start geladen und nicht
    erneut trainiert. Nach Abschluss wird zusaetzlich die Val-Score-Heatmap als
    reports/heatmap_val_score.png geschrieben.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
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
    SequenceDataset,
    build_trip_arrays,
    cap_dataset,
    load_split,
    rank_validation_results,
    regression_metrics,
)


ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"

REPORT_NAME = "lstm_loss_optimizer_report.html"
METRICS_CSV_NAME = "lstm_loss_optimizer_metrics.csv"
METRICS_JSON_NAME = "lstm_loss_optimizer_metrics.json"
HEATMAP_PNG_NAME = "heatmap_val_score.png"
CHECKPOINT_DIR = REPORTS_DIR / "lstm_loss_optimizer_checkpoints"


@dataclass
class Config:
    seq_len: int = 100
    horizon: int = 50
    stride: int = 5
    fraction: float = 0.5
    epochs: int = 10
    batch_size: int = 2048
    num_workers: int = 4
    parallel: int = 2
    lstm_architecture: tuple[int, ...] = (128, 64, 256, 128)
    lr: float = 1e-4
    seed: int = 2
    max_train_trips: int | None = None
    max_val_trips: int | None = None
    max_train_windows: int | None = None
    max_val_windows: int | None = None

    dropout: float = 0.45
    weight_decay: float = 1e-4
    momentum: float = 0.9
    smooth_l1_beta: float = 0.10
    huber_delta: float = 0.10
    losses: list[str] = field(default_factory=lambda: [
        "mse",
        "mae",
        "smooth_l1",
        "huber",
    ])
    optimizers: list[str] = field(default_factory=lambda: [
        "adam",
        "rmsprop",
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
    val_obj_final: float = float("nan")
    n_params: int = 0
    train_time_s: float = 0.0
    y_true: np.ndarray | None = None
    y_pred: np.ndarray | None = None
    _y_prev_val: np.ndarray | None = None


class LogCoshLoss(nn.Module):
    """Numerisch stabile log-cosh Regression-Loss."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = pred - target
        # log(cosh(x)) = x + softplus(-2x) - log(2)
        return torch.mean(x + F.softplus(-2.0 * x) - math.log(2.0))


class VariableLSTMRegressor(nn.Module):
    """Stack aus 1-layer-LSTMs fuer unterschiedliche Hidden-Sizes pro Layer."""

    def __init__(self, n_features: int, hidden_sizes: tuple[int, ...], dropout: float):
        super().__init__()
        if not hidden_sizes:
            raise ValueError("lstm_architecture darf nicht leer sein")

        layers = []
        in_size = n_features
        for h in hidden_sizes:
            layers.append(nn.LSTM(in_size, h, num_layers=1, batch_first=True))
            in_size = h

        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for index, lstm in enumerate(self.layers):
            out, _ = lstm(out)
            if index < len(self.layers) - 1:
                out = self.dropout(out)
        last = out[:, -1, :]
        last = self.dropout(last)
        return torch.tanh(self.head(last)).squeeze(-1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _signature_values(cfg: Config) -> dict:
    return {
        "model_family": "LSTM",
        "features": FEATURES,
        "seq_len": cfg.seq_len,
        "horizon": cfg.horizon,
        "stride": cfg.stride,
        "fraction": cfg.fraction,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "dropout": cfg.dropout,
        "weight_decay": cfg.weight_decay,
        "momentum": cfg.momentum,
        "smooth_l1_beta": cfg.smooth_l1_beta,
        "huber_delta": cfg.huber_delta,
        "seed": cfg.seed,
        "max_train_trips": cfg.max_train_trips,
        "max_val_trips": cfg.max_val_trips,
        "max_train_windows": cfg.max_train_windows,
        "max_val_windows": cfg.max_val_windows,
        "lstm_architecture": list(cfg.lstm_architecture),
    }


def _hash_values(values: dict) -> str:
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def checkpoint_signature(cfg: Config) -> str:
    """Bindet Kombi-Checkpoints an alle Trainings- und Datenparameter.

    num_workers und parallel sind bewusst nicht enthalten, damit Worker-Anzahl
    und Parallelitaet das Laden bestehender Checkpoints nicht beeinflussen. Die
    Signatur ist pro Kombination, unabhaengig von der Auswahl in --optimizers/
    --losses; so bleiben Checkpoints beim Erweitern des Sweeps erhalten.
    """
    return _hash_values(_signature_values(cfg))


def combo_slug(optimizer_name: str, loss_name: str) -> str:
    return f"{optimizer_name}_{loss_name}"


def checkpoint_paths(optimizer_name: str, loss_name: str) -> tuple[Path, Path]:
    stem = combo_slug(optimizer_name, loss_name)
    return CHECKPOINT_DIR / f"{stem}.json", CHECKPOINT_DIR / f"{stem}.npz"


def save_checkpoint(result: ComboResult, cfg: Config) -> None:
    """Speichert ein abgeschlossenes Kombi-Ergebnis atomar als JSON und NPZ."""
    if any(value is None for value in (result.y_true, result.y_pred, result._y_prev_val)):
        raise ValueError("Unvollstaendiges Ergebnis kann nicht checkpointed werden")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    json_path, npz_path = checkpoint_paths(result.optimizer_name, result.loss_name)
    metadata = {
        "version": 1,
        "signature": checkpoint_signature(cfg),
        "combo_name": result.combo_name,
        "optimizer_name": result.optimizer_name,
        "loss_name": result.loss_name,
        "train_objective": result.train_objective,
        "val_objective": result.val_objective,
        "metrics": result.metrics,
        "val_obj_final": result.val_obj_final,
        "n_params": result.n_params,
        "train_time_s": result.train_time_s,
    }
    json_temp = json_path.with_name(f"{json_path.name}.tmp")
    npz_temp = npz_path.with_name(f"{npz_path.name}.tmp")
    json_temp.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with npz_temp.open("wb") as file:
        np.savez_compressed(
            file,
            y_true=result.y_true,
            y_pred=result.y_pred,
            y_prev_val=result._y_prev_val,
        )
    json_temp.replace(json_path)
    npz_temp.replace(npz_path)


def load_checkpoint(
    optimizer_name: str, loss_name: str, cfg: Config
) -> ComboResult | None:
    """Laedt ein vollstaendiges Kombi-Ergebnis, falls die Signatur passt."""
    json_path, npz_path = checkpoint_paths(optimizer_name, loss_name)
    if not json_path.exists() or not npz_path.exists():
        return None

    try:
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        if (
            metadata.get("version") != 1
            or metadata.get("signature") != checkpoint_signature(cfg)
            or metadata.get("optimizer_name") != optimizer_name
            or metadata.get("loss_name") != loss_name
        ):
            return None
        with np.load(npz_path) as arrays:
            return ComboResult(
                combo_name=str(metadata["combo_name"]),
                optimizer_name=optimizer_name,
                loss_name=loss_name,
                train_objective=[float(v) for v in metadata["train_objective"]],
                val_objective=[float(v) for v in metadata["val_objective"]],
                metrics=dict(metadata["metrics"]),
                val_obj_final=float(metadata["val_obj_final"]),
                n_params=int(metadata["n_params"]),
                train_time_s=float(metadata["train_time_s"]),
                y_true=arrays["y_true"],
                y_pred=arrays["y_pred"],
                _y_prev_val=arrays["y_prev_val"],
            )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Checkpoint fuer {combo_slug(optimizer_name, loss_name)} wird ignoriert: {exc}")
        return None


def load_checkpoints(combos: list[tuple[str, str]], cfg: Config) -> list[ComboResult]:
    results = [
        result
        for optimizer_name, loss_name in combos
        if (result := load_checkpoint(optimizer_name, loss_name, cfg))
    ]
    if results:
        print(f"Checkpoints geladen: {len(results)}/{len(combos)} Kombinationen.")
    return results


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
    non_blocking = device.type == "cuda"
    obj_total, mse_total, n = 0.0, 0.0, 0
    yt, yp, ypr = [], [], []

    for x, y, y_prev in loader:
        x = x.to(device, non_blocking=non_blocking)
        y = y.to(device, non_blocking=non_blocking)
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
    device: torch.device,
    init_lock: threading.Lock | None = None,
) -> ComboResult:
    combo_name = f"{optimizer_name.upper()} + {loss_name}"

    # Modell-Init liest den globalen RNG; unter Lock bleibt die Initialisierung
    # auch bei parallelem Training deterministisch und reproduzierbar.
    init_ctx = init_lock if init_lock is not None else nullcontext()
    with init_ctx:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        model = VariableLSTMRegressor(len(FEATURES), cfg.lstm_architecture, cfg.dropout).to(device)
    objective_fn = build_loss(loss_name, cfg)
    optimizer = build_optimizer(optimizer_name, model.parameters(), cfg)

    res = ComboResult(
        combo_name=combo_name,
        optimizer_name=optimizer_name,
        loss_name=loss_name,
        n_params=count_params(model),
    )

    print(f"\n=== {combo_name} ({res.n_params:,} Parameter) ===")
    t0 = time.time()
    non_blocking = device.type == "cuda"
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        run_obj, n = 0.0, 0
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=non_blocking)
            y = y.to(device, non_blocking=non_blocking)
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
            f"  [{combo_name}] Epoch {epoch:02d}/{cfg.epochs}  train_obj={tr_obj:.6f}  "
            f"val_obj={vl_obj:.6f}  val_mse={vl_mse:.6f}"
        )

    res.train_time_s = time.time() - t0

    val_obj, _, yt, yp, ypr_val = evaluate(model, val_loader, device, objective_fn)
    res.val_obj_final = val_obj
    res.metrics = regression_metrics(yt, yp)
    res.y_true = yt
    res.y_pred = yp
    res._y_prev_val = ypr_val
    return res


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def combo_val_score(result: ComboResult) -> float:
    """Zusaetzliche Kennzahl aus Val-Richtungs- und Val-Toleranz-Acc."""
    return 0.5 * (result.metrics["dir_acc"] + result.metrics["tol_acc"])


def rank_combinations(results: list[ComboResult]) -> tuple[list[ComboResult], dict[str, float]]:
    return rank_validation_results(results, lambda result: result.combo_name)


def color_map_for_losses(losses: list[str]) -> dict[str, str]:
    cmap = plt.get_cmap("tab10")
    uniq = sorted(set(losses))
    return {name: matplotlib.colors.to_hex(cmap(i % 10)) for i, name in enumerate(uniq)}


def plot_combo_ranking(results: list[ComboResult], sort_desc: bool) -> str:
    ranked, rank_score = rank_combinations(results)
    ordered = list(reversed(ranked)) if sort_desc else ranked
    labels = [r.combo_name for r in ordered]
    vals = [rank_score[r.combo_name] for r in ordered]
    loss_colors = color_map_for_losses([r.loss_name for r in ordered])
    colors = [loss_colors[r.loss_name] for r in ordered]

    fig, ax = plt.subplots(figsize=(12, max(4, 0.38 * len(ordered))))
    bars = ax.barh(labels, vals, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Validierungs-Rangsumme (niedriger ist besser)")
    ax.set_title(
        "LSTM-Kombinationen sortiert nach fuenf Validierungsmetriken "
        f"({'absteigend' if sort_desc else 'aufsteigend'})"
    )
    ax.grid(alpha=0.3, axis="x")
    for b, v in zip(bars, vals):
        ax.text(v, b.get_y() + b.get_height() / 2, f" {v:.0f}", va="center", fontsize=8)

    fig.tight_layout()
    return fig_to_b64(fig)


def _aggregate_by_key(results: list[ComboResult], key: str) -> pd.DataFrame:
    _, rank_score = rank_combinations(results)
    rows = []
    for r in results:
        rows.append(
            {
                key: getattr(r, key),
                "val_rank_sum": rank_score[r.combo_name],
                "val_score": combo_val_score(r),
                "val_dir_acc": r.metrics["dir_acc"],
                "val_tol_acc": r.metrics["tol_acc"],
                "val_mse": r.metrics["mse"],
                "val_rmse": r.metrics["rmse"],
                "val_mae": r.metrics["mae"],
                "val_r2": r.metrics["r2"],
                "train_time_s": r.train_time_s,
            }
        )
    df = pd.DataFrame(rows)
    grp = (
        df.groupby(key, as_index=False)
        .agg(
            mean_val_rank_sum=("val_rank_sum", "mean"),
            best_val_rank_sum=("val_rank_sum", "min"),
            mean_val_score=("val_score", "mean"),
            median_val_score=("val_score", "median"),
            best_val_score=("val_score", "max"),
            mean_val_dir_acc=("val_dir_acc", "mean"),
            mean_val_tol_acc=("val_tol_acc", "mean"),
            mean_val_mse=("val_mse", "mean"),
            mean_train_time_s=("train_time_s", "mean"),
            n_runs=("val_score", "count"),
        )
        .sort_values("mean_val_rank_sum")
    )
    return grp


def plot_optimizer_ranking(results: list[ComboResult], sort_desc: bool) -> str:
    df = _aggregate_by_key(results, "optimizer_name")
    df = df.sort_values("mean_val_rank_sum", ascending=not sort_desc)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    bars = ax.bar(df["optimizer_name"], df["mean_val_rank_sum"], color="#4C72B0")
    ax.set_title(
        "Optimizer-Ranking (mittlere Validierungs-Rangsumme) "
        f"({'absteigend' if sort_desc else 'aufsteigend'})"
    )
    ax.set_ylabel("mittlere Rangsumme")
    ax.grid(alpha=0.3, axis="y")
    for b, v in zip(bars, df["mean_val_rank_sum"]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_loss_ranking(results: list[ComboResult], sort_desc: bool) -> str:
    df = _aggregate_by_key(results, "loss_name")
    df = df.sort_values("mean_val_rank_sum", ascending=not sort_desc)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    bars = ax.bar(df["loss_name"], df["mean_val_rank_sum"], color="#55A868")
    ax.set_title(
        "Loss-Ranking (mittlere Validierungs-Rangsumme) "
        f"({'absteigend' if sort_desc else 'aufsteigend'})"
    )
    ax.set_ylabel("mittlere Rangsumme")
    ax.grid(alpha=0.3, axis="y")
    for b, v in zip(bars, df["mean_val_rank_sum"]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_heatmap(results: list[ComboResult]) -> str:
    opts = sorted(set(r.optimizer_name for r in results))
    losses = sorted(set(r.loss_name for r in results))
    mat = np.full((len(opts), len(losses)), np.nan, dtype=float)

    pos_opt = {o: i for i, o in enumerate(opts)}
    pos_loss = {l: j for j, l in enumerate(losses)}
    _, rank_score = rank_combinations(results)
    for r in results:
        mat[pos_opt[r.optimizer_name], pos_loss[r.loss_name]] = rank_score[r.combo_name]

    fig, ax = plt.subplots(figsize=(1.8 * len(losses) + 1.5, 0.8 * len(opts) + 2.2))
    im = ax.imshow(mat, aspect="auto", cmap="viridis_r")
    ax.set_xticks(np.arange(len(losses)))
    ax.set_yticks(np.arange(len(opts)))
    ax.set_xticklabels(losses, rotation=30, ha="right")
    ax.set_yticklabels(opts)
    ax.set_title("Validierungs-Rangsumme (Optimizer x Loss)")

    for i in range(len(opts)):
        for j in range(len(losses)):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.4f}", ha="center", va="center", color="white", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Rangsumme")
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_loss_curves(results: list[ComboResult], top_k: int = 12) -> str:
    ordered = rank_combinations(results)[0][:max(1, top_k)]
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
    meta: dict,
) -> str:
    imgs = {
        "combo": plot_combo_ranking(results, cfg.sort_desc),
        "opt": plot_optimizer_ranking(results, cfg.sort_desc),
        "loss": plot_loss_ranking(results, cfg.sort_desc),
        "heatmap": plot_heatmap(results),
        "curves": plot_loss_curves(results, cfg.top_loss_curves),
    }

    ranked, rank_score = rank_combinations(results)
    best = ranked[0]
    best_score_pct = 100.0 * combo_val_score(best)
    best_loss_name = (
        _aggregate_by_key(results, "loss_name").iloc[0]["loss_name"]
    )
    best_opt_name = (
        _aggregate_by_key(results, "optimizer_name").iloc[0]["optimizer_name"]
    )

    rows = []
    for rank, r in enumerate(ranked, start=1):
        star = " &#11088;" if rank == 1 else ""
        rows.append(
            "<tr>"
            f"<td>{rank}</td>"
            f"<td><b>{r.optimizer_name}{star}</b></td>"
            f"<td>{r.loss_name}</td>"
            f"<td>{rank_score[r.combo_name]:.0f}</td>"
            f"<td>{r.metrics['mse']:.6f}</td>"
            f"<td>{r.metrics['rmse']:.6f}</td>"
            f"<td>{r.metrics['mae']:.6f}</td>"
            f"<td>{r.metrics['r2']:.4f}</td>"
            f"<td>{r.metrics['dir_acc_pct']:.2f}%</td>"
            f"<td>{r.metrics['tol_acc_pct']:.2f}%</td>"
            f"<td>{100.0 * combo_val_score(r):.2f}%</td>"
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
            f"<th>{title_col}</th><th>mean Rangsumme</th><th>beste Rangsumme</th><th>mean Val-Score</th><th>median Val-Score</th>"
            "<th>best Val-Score</th><th>mean Val Dir-Acc</th><th>mean Val Tol-Acc</th><th>mean Val-MSE</th><th>mean Trainingszeit</th><th>Runs</th>"
            "</tr>",
        ]
        for _, row in df.iterrows():
            out.append(
                "<tr>"
                f"<td>{row[title_col]}</td>"
                f"<td>{row['mean_val_rank_sum']:.1f}</td>"
                f"<td>{row['best_val_rank_sum']:.0f}</td>"
                f"<td>{100.0 * row['mean_val_score']:.2f}%</td>"
                f"<td>{100.0 * row['median_val_score']:.2f}%</td>"
                f"<td>{100.0 * row['best_val_score']:.2f}%</td>"
                f"<td>{100.0 * row['mean_val_dir_acc']:.2f}%</td>"
                f"<td>{100.0 * row['mean_val_tol_acc']:.2f}%</td>"
                f"<td>{row['mean_val_mse']:.6f}</td>"
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
<title>LSTM Bruteforce: Optimizer x Loss</title>
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
<h1>LSTM Bruteforce: Optimizer vs. Loss-Funktion</h1>
<p>Trainiert auf <b>{cfg.fraction*100:.0f}%</b> der Trips, alle Kombinationen aus
Optimizer und Loss-Funktion werden durchlaufen.</p>

<div class="rec">
<b>Beste Kombination (fuenf Validierungsmetriken):</b> <b>{best.optimizer_name} + {best.loss_name}</b><br>
Rangsumme={rank_score[best.combo_name]:.0f}; zusaetzlicher Richtungs-/Toleranz-Score={best_score_pct:.2f}%<br>
Val Richtungs-Acc={best.metrics['dir_acc_pct']:.2f}% &middot; Val Toleranz-Acc={best.metrics['tol_acc_pct']:.2f}%<br>
Val: MSE={best.metrics['mse']:.6f}, RMSE={best.metrics['rmse']:.6f}, R&sup2;={best.metrics['r2']:.4f}<br>
<b>Bestes Loss-Familien-Ranking (mittlere Rangsumme):</b> {best_loss_name}<br>
<b>Bester Optimizer (mittlere Rangsumme):</b> {best_opt_name}
</div>

<h2>Setup</h2>
<div class="cfg">
History H={cfg.seq_len} Schritte ({cfg.seq_len*0.2:.1f}s @ 5Hz), Horizon={cfg.horizon} ({cfg.horizon*200}ms),
Stride={cfg.stride}, Epochen={cfg.epochs}, Batch={cfg.batch_size}, LSTM-Architektur={"-".join(str(x) for x in cfg.lstm_architecture)}, LR={cfg.lr}<br>
Dropout={cfg.dropout}, Weight Decay={cfg.weight_decay}, Momentum={cfg.momentum}, smooth_l1_beta={cfg.smooth_l1_beta}, huber_delta={cfg.huber_delta}<br>
Optimizer: {", ".join(cfg.optimizers)}<br>
Losses: {", ".join(cfg.losses)}<br>
Train-Trips: {meta['n_train_trips']} &middot; Val-Trips: {meta['n_val_trips']}<br>
Train-Fenster: {meta['n_train_windows']:,} &middot; Val-Fenster: {meta['n_val_windows']:,}
</div>

<h2>Kombinations-Ranking nach fuenf Validierungsmetriken</h2>
<table>
<tr><th>Rang</th><th>Optimizer</th><th>Loss</th><th>Rangsumme</th><th>Val MSE</th><th>Val RMSE</th><th>Val MAE</th><th>Val R&sup2;</th>
<th>Val Richtungs-Acc</th><th>Val Toleranz-Acc</th><th>Val-Score</th><th>Trainingszeit</th></tr>
{rows_html}
<tr style='color:#888'><td>-</td><td><i>Persistence</i></td><td>a_t</td>
<td>-</td><td>{baseline_val['mse']:.6f}</td><td>{baseline_val['rmse']:.6f}</td><td>{baseline_val['mae']:.6f}</td><td>{baseline_val['r2']:.4f}</td>
<td>{baseline_val['dir_acc_pct']:.2f}%</td><td>{baseline_val['tol_acc_pct']:.2f}%</td>
<td>{100.0 * (0.5 * (baseline_val['dir_acc'] + baseline_val['tol_acc'])):.2f}%</td><td>-</td></tr>
</table>

<p class="note">Ranking erfolgt ausschliesslich auf der Validierung als Rangsumme ueber R2, RMSE, MAE, Richtungs-Acc und Toleranz-Acc. Niedriger ist besser; Tie-Breaker sind R2 und RMSE.</p>

<h2>Optimizer-Aggregat</h2>
{opt_table}

<h2>Loss-Aggregat</h2>
{loss_table}

<h2>Plot: Alle Kombinationen nach Validierungs-Rangsumme</h2>
<p class="note">Sortierung nach Rangsumme ({'absteigend' if cfg.sort_desc else 'aufsteigend'}).</p>
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


def save_val_score_heatmap(results: list[ComboResult]) -> Path:
    """Schreibt die Val-Score-Heatmap (Optimizer x Loss) als PNG."""
    opts = sorted(set(r.optimizer_name for r in results))
    losses = sorted(set(r.loss_name for r in results))
    mat = np.full((len(opts), len(losses)), np.nan, dtype=float)
    pos_opt = {o: i for i, o in enumerate(opts)}
    pos_loss = {l: j for j, l in enumerate(losses)}
    for r in results:
        mat[pos_opt[r.optimizer_name], pos_loss[r.loss_name]] = 100.0 * combo_val_score(r)

    fig, ax = plt.subplots(figsize=(1.8 * len(losses) + 1.5, 0.8 * len(opts) + 2.2))
    im = ax.imshow(mat, aspect="auto", cmap="viridis_r")
    ax.set_xticks(np.arange(len(losses)))
    ax.set_yticks(np.arange(len(opts)))
    ax.set_xticklabels(losses, rotation=30, ha="right")
    ax.set_yticklabels(opts)
    ax.set_title("Val-Score Heatmap [%] (Optimizer x Loss)")

    for i in range(len(opts)):
        for j in range(len(losses)):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}%", ha="center", va="center", color="white", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Val-Score [%]")
    fig.tight_layout()
    REPORTS_DIR.mkdir(exist_ok=True)
    out_path = REPORTS_DIR / HEATMAP_PNG_NAME
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_outputs(results: list[ComboResult], cfg: Config, meta_base: dict, t_start: float):
    if not results:
        return

    y_true_ref = results[0].y_true
    y_prev_val_ref = results[0]._y_prev_val
    assert y_true_ref is not None
    assert y_prev_val_ref is not None

    baseline_val = regression_metrics(y_true_ref, y_prev_val_ref)
    _, rank_score = rank_combinations(results)

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
                "validation_rank_sum": rank_score[r.combo_name],
                **r.metrics,
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
            "validation_rank_sum": None,
            **baseline_val,
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
    html = build_report(results, cfg, baseline_val, meta)
    (REPORTS_DIR / REPORT_NAME).write_text(html, encoding="utf-8")

    ranked, rank_score = rank_combinations(results)
    best = ranked[0]
    print("\n" + "=" * 68)
    print(f"ZWISCHENSTAND gespeichert (fertig: {len(results)} Kombinationen)")
    print(
        f"Aktuell beste Kombi: {best.optimizer_name}+{best.loss_name} | "
        f"Rangsumme={rank_score[best.combo_name]:.0f} | "
        f"Val Dir={best.metrics['dir_acc_pct']:.2f}% | Val Tol={best.metrics['tol_acc_pct']:.2f}%"
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


def parse_architecture(raw: str) -> tuple[int, ...]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("Leere LSTM-Architektur ist nicht erlaubt")
    vals = tuple(int(p) for p in parts)
    if any(v <= 0 for v in vals):
        raise ValueError("Alle Werte in --lstm-architecture muessen > 0 sein")
    if len(vals) > 4:
        raise ValueError("--lstm-architecture darf maximal 4 Layer enthalten")
    return vals


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="LSTM Bruteforce fuer Loss und Optimizer.")
    p.add_argument("--full", action="store_true", help="Alle Trips (Fraction=1.0).")
    p.add_argument("--fraction", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seq-len", type=int, default=100)
    p.add_argument("--horizon", type=int, default=50)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--parallel",
        type=int,
        default=2,
        help="Gleichzeitig trainierte Kombinationen auf CUDA (Standard: 2).",
    )
    p.add_argument(
        "--lstm-architecture",
        type=str,
        default="128,64,256,128",
        help="LSTM-Hidden-Sizes je Layer, z.B. 128,64,256,128.",
    )
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--max-train-trips", type=int, default=None)
    p.add_argument("--max-val-trips", type=int, default=None)
    p.add_argument("--max-train-windows", type=int, default=None)
    p.add_argument("--max-val-windows", type=int, default=None)

    p.add_argument("--dropout", type=float, default=0.45)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--smooth-l1-beta", type=float, default=0.10)
    p.add_argument("--huber-delta", type=float, default=0.10)
    p.add_argument("--optimizers", type=str, default="adam,rmsprop,adagrad")
    p.add_argument("--losses", type=str, default="mse,mae,smooth_l1,huber")
    sort_group = p.add_mutually_exclusive_group()
    sort_group.add_argument(
        "--sort-desc",
        dest="sort_desc",
        action="store_true",
        help="Sortierung in Plots absteigend nach Val-Score (Default).",
    )
    sort_group.add_argument(
        "--sort-asc",
        dest="sort_desc",
        action="store_false",
        help="Sortierung in Plots aufsteigend nach Val-Score.",
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
        num_workers=a.num_workers,
        parallel=a.parallel,
        lstm_architecture=parse_architecture(a.lstm_architecture),
        lr=a.lr,
        seed=a.seed,
        max_train_trips=a.max_train_trips,
        max_val_trips=a.max_val_trips,
        max_train_windows=a.max_train_windows,
        max_val_windows=a.max_val_windows,
        dropout=a.dropout,
        weight_decay=a.weight_decay,
        momentum=a.momentum,
        smooth_l1_beta=a.smooth_l1_beta,
        huber_delta=a.huber_delta,
        optimizers=parse_csv_list(a.optimizers),
        losses=parse_csv_list(a.losses),
        sort_desc=a.sort_desc,
        top_loss_curves=max(1, a.top_loss_curves),
    )
    if cfg.num_workers < 0:
        raise ValueError("num_workers muss >= 0 sein")
    if cfg.parallel < 1:
        raise ValueError("parallel muss >= 1 sein")
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

    train_trips, train_labels = build_trip_arrays(train_df, cfg, "train", rng)
    val_trips, val_labels = build_trip_arrays(val_df, cfg, "val", rng)
    print(
        f"  Train-Trips: {len(train_trips)}  Val-Trips: {len(val_trips)}"
    )

    train_ds = SequenceDataset(train_trips, train_labels, cfg)
    val_ds = SequenceDataset(val_trips, val_labels, cfg)
    train_ds = cap_dataset(train_ds, cfg.max_train_windows, rng)
    val_ds = cap_dataset(val_ds, cfg.max_val_windows, rng)
    print(
        f"  Train-Fenster: {len(train_ds):,}  Val-Fenster: {len(val_ds):,}"
    )

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit("Keine Fenster gebaut - seq_len/horizon/stride pruefen.")

    pin_memory = device.type == "cuda"
    if pin_memory:
        torch.backends.cudnn.benchmark = True

    def make_loaders() -> tuple[DataLoader, DataLoader]:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=pin_memory,
            persistent_workers=cfg.num_workers > 0,
            prefetch_factor=2 if cfg.num_workers > 0 else None,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=pin_memory,
            persistent_workers=cfg.num_workers > 0,
            prefetch_factor=2 if cfg.num_workers > 0 else None,
            drop_last=False,
        )
        return train_loader, val_loader

    meta_base = {
        "n_train_trips": len(train_trips),
        "n_val_trips": len(val_trips),
        "n_train_windows": len(train_ds),
        "n_val_windows": len(val_ds),
        "device": str(device),
    }

    combos = [
        (optimizer_name, loss_name)
        for optimizer_name in cfg.optimizers
        for loss_name in cfg.losses
    ]
    total = len(combos)

    results: list[ComboResult] = load_checkpoints(combos, cfg)
    completed = {(r.optimizer_name, r.loss_name) for r in results}
    if results:
        save_outputs(results, cfg, meta_base, t_start)

    pending = [
        (i, opt, loss)
        for i, (opt, loss) in enumerate(combos, start=1)
        if (opt, loss) not in completed
    ]
    for i, (opt, loss) in enumerate(combos, start=1):
        if (opt, loss) in completed:
            print(f"\nRun {i}/{total}: {opt}+{loss} bereits checkpointed.")

    # Nur auf CUDA parallel; auf CPU bringt Thread-Parallelitaet keinen Vorteil.
    n_parallel = min(cfg.parallel, max(1, len(pending))) if device.type == "cuda" else 1
    print(f"\nParalleles Training: {n_parallel} Kombination(en) gleichzeitig.")

    # Ein DataLoader-Paar pro paralleler Spur; verhindert gleichzeitiges
    # Iterieren desselben Loaders aus mehreren Threads.
    loader_pool: "queue.Queue[tuple[DataLoader, DataLoader]]" = queue.Queue()
    for _ in range(n_parallel):
        loader_pool.put(make_loaders())

    init_lock = threading.Lock()
    save_lock = threading.Lock()

    def worker(index: int, optimizer_name: str, loss_name: str) -> None:
        train_loader, val_loader = loader_pool.get()
        try:
            print("\n" + "-" * 68)
            print(f"Run {index}/{total}: optimizer={optimizer_name}, loss={loss_name}")
            res = train_one_combo(
                optimizer_name=optimizer_name,
                loss_name=loss_name,
                cfg=cfg,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                init_lock=init_lock,
            )
        finally:
            loader_pool.put((train_loader, val_loader))

        with save_lock:
            results.append(res)
            save_checkpoint(res, cfg)
            print(f"Checkpoint gespeichert: {checkpoint_paths(optimizer_name, loss_name)[0]}")
            # Inkrementell speichern fuer lange Runs.
            save_outputs(results, cfg, meta_base, t_start)

    if pending:
        with ThreadPoolExecutor(max_workers=n_parallel) as pool:
            futures = [pool.submit(worker, i, opt, loss) for i, opt, loss in pending]
            for future in futures:
                future.result()

    print("\nFERTIG.")
    heatmap_path = save_val_score_heatmap(results)
    print(f"Heatmap gespeichert: {heatmap_path}")
    ranked, rank_score = rank_combinations(results)
    best = ranked[0]
    print(
        f"Beste Kombination nach fuenf Validierungsmetriken: {best.optimizer_name}+{best.loss_name} "
        f"(Rangsumme={rank_score[best.combo_name]:.0f}, "
        f"Val Dir={best.metrics['dir_acc_pct']:.2f}%, Val Tol={best.metrics['tol_acc_pct']:.2f}%)"
    )


if __name__ == "__main__":
    main()
