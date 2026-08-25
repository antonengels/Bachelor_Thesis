r"""
LSTM-Only Bruteforce fuer Architekturvergleich (Layeranzahl x Neuronen pro Layer).

Idee:
    Nutzt dieselbe Datenaufbereitung/Windowing-Logik wie `model_comparison.py`,
    trainiert aber nur LSTM-Varianten mit unterschiedlicher Architektur.

Architekturraum (Standard):
    - Maximal 4 LSTM-Layer
    - Pro Layer Neuronen aus {64, 128, 256}
    - Erlaubte Tiefen: 1, 2, 3, 4 Layer
    -> Anzahl Kombinationen: 3 + 9 + 27 + 81 = 120

Training:
    - 50 % der Trainings- und Validierungs-Trips, 10 Epochen
    - MSE-Loss und Adam-Optimizer
    - Regularisierung: Dropout 0.45 und L2 (weight_decay=1e-4)

Ziel:
    - Ranking aller Architekturen ueber fuenf Validierungsmetriken
    - Aggregierte Auswertung je Layer-Tiefe (mean/median/best)
    - Self-contained HTML-Report mit den wichtigsten Plots
    - Resume-Checkpoints je fertig trainierter Architektur

Beispiel:
    .\.venv\Scripts\python.exe src\lstm_architecture_comparison.py

Optionen wie beim Architekturvergleich:
    --full --fraction --epochs --seq-len --horizon --stride --batch-size
    --lr --seed --max-*-trips --max-*-windows

Neue Optionen:
    --neurons 64,128,256
    --max-layers 4
    --dropout 0.45 --weight-decay 1e-4
    --sort-desc / --sort-asc
    --parallel 2   (Anzahl gleichzeitig trainierter Architekturen)

Fortsetzung:
    Nach jeder vollstaendig trainierten Architektur werden Ergebnis-Metadaten und
    Vorhersagen unter reports/lstm_architecture_comparison_checkpoints/ gespeichert.
    Bei identischer Konfiguration werden diese beim naechsten Start geladen und
    nicht erneut trainiert.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import itertools
import json
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

REPORT_NAME = "lstm_architecture_comparison_report.html"
METRICS_CSV_NAME = "lstm_architecture_comparison_metrics.csv"
METRICS_JSON_NAME = "lstm_architecture_comparison_metrics.json"
CHECKPOINT_DIR = REPORTS_DIR / "lstm_architecture_comparison_checkpoints"

DEPTH_COLORS = {
    1: "#4C72B0",
    2: "#DD8452",
    3: "#55A868",
    4: "#C44E52",
}


@dataclass
class Config:
    # Entspricht den Standard-Trainingsparametern aus model_comparison.py
    seq_len: int = 100
    horizon: int = 50
    stride: int = 5
    fraction: float = 0.5
    epochs: int = 10
    batch_size: int = 2048
    num_workers: int = 2
    parallel: int = 2
    lr: float = 1e-4
    dropout: float = 0.45
    weight_decay: float = 1e-4
    seed: int = 2
    max_train_trips: int | None = None
    max_val_trips: int | None = None
    max_train_windows: int | None = None
    max_val_windows: int | None = None

    max_layers: int = 4
    neurons: list[int] = field(default_factory=lambda: [64, 128, 256])
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
    n_params: int = 0
    train_time_s: float = 0.0
    y_true: np.ndarray | None = None
    y_pred: np.ndarray | None = None
    _y_prev_val: np.ndarray | None = None


class VariableLSTMRegressor(nn.Module):
    """Stack aus LSTMs mit variabler Hidden-Size und Dropout-Regularisierung."""

    def __init__(self, n_features: int, hidden_sizes: tuple[int, ...], dropout: float):
        super().__init__()
        if len(hidden_sizes) == 0:
            raise ValueError("hidden_sizes darf nicht leer sein")

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


def arch_to_name(arch: tuple[int, ...]) -> str:
    return "-".join(str(x) for x in arch)


def build_architectures(cfg: Config) -> list[tuple[int, ...]]:
    if cfg.max_layers < 1:
        raise ValueError("max_layers muss >= 1 sein")
    if cfg.max_layers > 4:
        raise ValueError("max_layers darf fuer diesen Vergleich maximal 4 sein")
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
        "seed": cfg.seed,
        "max_train_trips": cfg.max_train_trips,
        "max_val_trips": cfg.max_val_trips,
        "max_train_windows": cfg.max_train_windows,
        "max_val_windows": cfg.max_val_windows,
        "max_layers": cfg.max_layers,
        "neurons": cfg.neurons,
    }


def _hash_values(values: dict) -> str:
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def checkpoint_signature(cfg: Config) -> str:
    """Bindet Sweep-Checkpoints an alle Trainings- und Datenparameter.

    num_workers ist bewusst nicht enthalten, damit die Worker-Anzahl das Laden
    bestehender Checkpoints nicht beeinflusst.
    """
    return _hash_values(_signature_values(cfg))


def legacy_checkpoint_signature(cfg: Config) -> str:
    """Alte Signatur inkl. num_workers fuer Rueckwaertskompatibilitaet."""
    values = _signature_values(cfg)
    values["num_workers"] = cfg.num_workers
    return _hash_values(values)


def checkpoint_paths(arch: tuple[int, ...]) -> tuple[Path, Path]:
    stem = f"LSTM-{arch_to_name(arch)}"
    return CHECKPOINT_DIR / f"{stem}.json", CHECKPOINT_DIR / f"{stem}.npz"


def save_checkpoint(result: ArchResult, cfg: Config) -> None:
    """Speichert ein abgeschlossenes Sweep-Ergebnis atomar als JSON und NPZ."""
    if any(
        value is None
        for value in (
            result.y_true,
            result.y_pred,
            result._y_prev_val,
        )
    ):
        raise ValueError("Unvollstaendiges Ergebnis kann nicht checkpointed werden")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    json_path, npz_path = checkpoint_paths(result.arch)
    metadata = {
        "version": 2,
        "signature": checkpoint_signature(cfg),
        "arch": list(result.arch),
        "arch_name": result.arch_name,
        "n_layers": result.n_layers,
        "train_loss": result.train_loss,
        "val_loss": result.val_loss,
        "metrics": result.metrics,
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


def load_checkpoint(arch: tuple[int, ...], cfg: Config) -> ArchResult | None:
    """Laedt ein vollstaendiges Architektur-Ergebnis, falls die Signatur passt."""
    json_path, npz_path = checkpoint_paths(arch)
    if not json_path.exists() or not npz_path.exists():
        return None

    try:
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        valid_signatures = (
            checkpoint_signature(cfg),
            legacy_checkpoint_signature(cfg),
        )
        if (
            metadata.get("version") != 2
            or metadata.get("signature") not in valid_signatures
            or tuple(metadata.get("arch", ())) != arch
        ):
            return None
        with np.load(npz_path) as arrays:
            return ArchResult(
                arch=arch,
                arch_name=str(metadata["arch_name"]),
                n_layers=int(metadata["n_layers"]),
                train_loss=[float(value) for value in metadata["train_loss"]],
                val_loss=[float(value) for value in metadata["val_loss"]],
                metrics=dict(metadata["metrics"]),
                n_params=int(metadata["n_params"]),
                train_time_s=float(metadata["train_time_s"]),
                y_true=arrays["y_true"],
                y_pred=arrays["y_pred"],
                _y_prev_val=arrays["y_prev_val"],
            )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Checkpoint fuer LSTM {arch_to_name(arch)} wird ignoriert: {exc}")
        return None


def load_checkpoints(archs: list[tuple[int, ...]], cfg: Config) -> list[ArchResult]:
    results = [result for arch in archs if (result := load_checkpoint(arch, cfg))]
    if results:
        print(f"Checkpoints geladen: {len(results)}/{len(archs)} Architekturen.")
    return results


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    non_blocking = device.type == "cuda"
    crit = nn.MSELoss(reduction="sum")
    total, n = 0.0, 0
    yt, yp, ypr = [], [], []

    for x, y, y_prev in loader:
        x = x.to(device, non_blocking=non_blocking)
        y = y.to(device, non_blocking=non_blocking)
        pred = model(x)
        total += crit(pred, y).item()
        n += y.numel()
        yt.append(y.cpu().numpy())
        yp.append(pred.cpu().numpy())
        ypr.append(y_prev.numpy())

    return total / max(n, 1), np.concatenate(yt), np.concatenate(yp), np.concatenate(ypr)


@torch.no_grad()
def eval_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Schnelle Validierung: nur MSE, ohne Vorhersage-Arrays aufzubauen."""
    model.eval()
    non_blocking = device.type == "cuda"
    crit = nn.MSELoss(reduction="sum")
    total, n = 0.0, 0
    for x, y, _ in loader:
        x = x.to(device, non_blocking=non_blocking)
        y = y.to(device, non_blocking=non_blocking)
        pred = model(x)
        total += crit(pred, y).item()
        n += y.numel()
    return total / max(n, 1)


def train_one_arch(
    arch: tuple[int, ...],
    cfg: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    init_lock: threading.Lock | None = None,
) -> ArchResult:
    name = arch_to_name(arch)

    # Modell-Init liest den globalen RNG; unter Lock bleibt die Initialisierung
    # auch bei parallelem Training deterministisch und reproduzierbar.
    init_ctx = init_lock if init_lock is not None else nullcontext()
    with init_ctx:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        model = VariableLSTMRegressor(len(FEATURES), arch, cfg.dropout).to(device)
    opt = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    crit = nn.MSELoss()

    res = ArchResult(
        arch=arch,
        arch_name=name,
        n_layers=len(arch),
        n_params=count_params(model),
    )

    print(f"\n=== LSTM {name} ({res.n_params:,} Parameter) ===")
    t0 = time.time()
    non_blocking = device.type == "cuda"

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        run_loss, n = 0.0, 0
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=non_blocking)
            y = y.to(device, non_blocking=non_blocking)
            opt.zero_grad()
            pred = model(x)
            loss = crit(pred, y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * y.numel()
            n += y.numel()

        tr = run_loss / max(n, 1)
        vl = eval_loss(model, val_loader, device)
        res.train_loss.append(tr)
        res.val_loss.append(vl)
        print(f"  [{name}] Epoch {epoch:02d}/{cfg.epochs}  train_mse={tr:.6f}  val_mse={vl:.6f}")

    res.train_time_s = time.time() - t0

    _, yt, yp, ypr_val = evaluate(model, val_loader, device)

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


def rank_architectures(results: list[ArchResult]) -> tuple[list[ArchResult], dict[str, float]]:
    return rank_validation_results(results, lambda result: result.arch_name)


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
            mean_val_r2=("val_r2", "mean"),
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
        best = rank_architectures(subset)[0][0]
        rows.append(
            {
                "n_layers": depth,
                "arch": best.arch_name,
                "val_mse": best.metrics["mse"],
                "val_rmse": best.metrics["rmse"],
                "val_r2": best.metrics["r2"],
                "train_time_s": best.train_time_s,
                "n_params": best.n_params,
            }
        )
    return pd.DataFrame(rows).sort_values("n_layers")


def plot_arch_ranking(results: list[ArchResult], sort_desc: bool) -> str:
    ranked, rank_score = rank_architectures(results)
    ordered = list(reversed(ranked)) if sort_desc else ranked
    labels = [f"{r.arch_name} ({r.n_layers}L)" for r in ordered]
    vals = [rank_score[r.arch_name] for r in ordered]
    colors = [DEPTH_COLORS.get(r.n_layers, "#808080") for r in ordered]

    fig, ax = plt.subplots(figsize=(12, max(4, 0.34 * len(ordered))))
    bars = ax.barh(labels, vals, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Validierungs-Rangsumme (niedriger ist besser)")
    ax.set_title(
        "LSTM-Architekturen sortiert nach fuenf Validierungsmetriken "
        f"({'absteigend' if sort_desc else 'aufsteigend'})"
    )
    ax.grid(alpha=0.3, axis="x")
    for b, v in zip(bars, vals):
        ax.text(v, b.get_y() + b.get_height() / 2, f" {v:.0f}", va="center", fontsize=8)

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
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bars = ax.bar(x, grp["mean_val_mse"], label="mean Val-MSE", color="#4C72B0")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(d)} Layer" for d in grp["n_layers"]])
    ax.set_ylabel("MSE")
    ax.set_title("Mittelwerte je Layer-Tiefe")
    ax.grid(alpha=0.3, axis="y")
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
        groups.append([r.metrics["mse"] for r in results if r.n_layers == depth])
        labels.append(f"{depth} Layer")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bp = ax.boxplot(groups, tick_labels=labels, patch_artist=True)
    for patch, depth in zip(bp["boxes"], sorted(set(r.n_layers for r in results))):
        patch.set_facecolor(DEPTH_COLORS.get(depth, "#cccccc"))
        patch.set_alpha(0.65)

    ax.set_title("Val-MSE Verteilung je Layer-Tiefe")
    ax.set_ylabel("Val-MSE")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_loss_curves(results: list[ArchResult], top_k: int = 12) -> str:
    ordered = rank_architectures(results)[0][: max(1, top_k)]

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
        df["val_mse"],
        color=[DEPTH_COLORS.get(int(d), "#888888") for d in df["n_layers"]],
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(d)} Layer" for d in df["n_layers"]])
    ax.set_ylabel("Bestes Val-MSE")
    ax.set_title("Bestes Modell je Layer-Tiefe (Mehrmetrik-Rangsumme)")
    ax.grid(alpha=0.3, axis="y")

    for b, arch, v in zip(bars, df["arch"], df["val_mse"]):
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
    meta: dict,
) -> str:
    imgs = {
        "ranking": plot_arch_ranking(results, cfg.sort_desc),
        "depth_mean": plot_depth_mean_mse(results),
        "depth_dist": plot_depth_distribution(results),
        "curves": plot_loss_curves(results, cfg.top_loss_curves),
        "best_depth": plot_best_per_depth(results),
    }

    ranked, rank_score = rank_architectures(results)
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
            f"<td>{rank_score[r.arch_name]:.0f}</td>"
            f"<td>{r.metrics['mse']:.6f}</td>"
            f"<td>{r.metrics['rmse']:.6f}</td>"
            f"<td>{r.metrics['mae']:.6f}</td>"
            f"<td>{r.metrics['r2']:.4f}</td>"
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
            f"<td>{row['mean_val_r2']:.4f}</td>"
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
            f"<td>{row['train_time_s']:.1f}s</td>"
            "</tr>"
        )
    best_depth_rows_html = "\n".join(best_depth_rows)

    html = f"""<!DOCTYPE html>
<html lang="de"><head><meta charset="utf-8">
<title>LSTM Architekturvergleich</title>
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
<h1>LSTM Architekturvergleich</h1>
<p>Bruteforce ueber alle LSTM-Architekturen mit 1 bis {cfg.max_layers} Layern und
Neuronen aus {{{', '.join(str(n) for n in cfg.neurons)}}} pro Layer.</p>

<div class="rec">
<b>Bestes Modell (Validierungs-Rangsumme):</b> <b>{best.arch_name}</b> ({best.n_layers} Layer)<br>
Val: MSE={best.metrics['mse']:.6f}, RMSE={best.metrics['rmse']:.6f}, R&sup2;={best.metrics['r2']:.4f}<br>
Parameter: {best.n_params:,}
</div>

<h2>Setup</h2>
<div class="cfg">
History H={cfg.seq_len} Schritte ({cfg.seq_len*0.2:.1f}s @ 5Hz), Horizon={cfg.horizon} ({cfg.horizon*200}ms),
Stride={cfg.stride}, Epochen={cfg.epochs}, Batch={cfg.batch_size}, LR={cfg.lr},
Loss=MSE, Optimizer=Adam, Dropout={cfg.dropout}, L2 Weight Decay={cfg.weight_decay}<br>
Train-Trips: {meta['n_train_trips']} &middot; Val-Trips: {meta['n_val_trips']}<br>
Train-Fenster: {meta['n_train_windows']:,} &middot; Val-Fenster: {meta['n_val_windows']:,}<br>
Architekturen: {meta['n_architectures']} (erwartet: {meta['expected_architectures']})
</div>

<h2>Vollstaendiges Ranking nach fuenf Validierungsmetriken</h2>
<table>
<tr><th>Rang</th><th>Architektur (Neuronen je Layer)</th><th>Layer</th><th>Parameter</th>
<th>Rangsumme</th><th>Val MSE</th><th>Val RMSE</th><th>Val MAE</th><th>Val R&sup2;</th>
<th>Trainingszeit</th></tr>
{rows_html}
<tr style='color:#888'><td>-</td><td><i>Persistence</i></td><td>-</td><td>0</td>
<td>-</td><td>{baseline_val['mse']:.6f}</td><td>{baseline_val['rmse']:.6f}</td><td>{baseline_val['mae']:.6f}</td><td>{baseline_val['r2']:.4f}</td>
<td>-</td></tr>
</table>

<h2>Mittelwerte je Layer-Tiefe</h2>
<table>
<tr><th>Layer</th><th>mean Val-MSE</th><th>median Val-MSE</th><th>best Val-MSE</th>
<th>mean Val-R&sup2;</th><th>mean Trainingszeit</th><th>Runs</th></tr>
{depth_rows_html}
</table>

<h2>Bestes Modell je Layer-Tiefe (Mehrmetrik-Rangsumme)</h2>
<table>
<tr><th>Layer</th><th>Architektur</th><th>Parameter</th><th>Val MSE</th><th>Val RMSE</th><th>Val R&sup2;</th>
<th>Trainingszeit</th></tr>
{best_depth_rows_html}
</table>

<h2>Plot: Alle Architekturen nach Validierungs-Rangsumme</h2>
<img src='data:image/png;base64,{imgs['ranking']}'/>

<h2>Plot: Mean-MSE je Layer-Tiefe</h2>
<img src='data:image/png;base64,{imgs['depth_mean']}'/>

<h2>Plot: Val-MSE Verteilung je Layer-Tiefe</h2>
<img src='data:image/png;base64,{imgs['depth_dist']}'/>

<h2>Plot: Bestes Modell je Layer-Tiefe (Mehrmetrik-Rangsumme)</h2>
<img src='data:image/png;base64,{imgs['best_depth']}'/>

<h2>Plot: Train/Val-Kurven (Top-K nach Mehrmetrik-Rangsumme)</h2>
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
    assert y_true_ref is not None
    assert y_prev_val_ref is not None

    baseline_val = regression_metrics(y_true_ref, y_prev_val_ref)
    _, rank_score = rank_architectures(results)

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
                "validation_rank_sum": rank_score[r.arch_name],
                **r.metrics,
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

    ranked, rank_score = rank_architectures(results)
    best = ranked[0]
    print("\n" + "=" * 68)
    print(f"ZWISCHENSTAND gespeichert (fertig: {len(results)} Architekturen)")
    print(
        f"Aktuell bestes Modell: {best.arch_name} ({best.n_layers}L) | "
        f"Rangsumme={rank_score[best.arch_name]:.0f} | Val MSE={best.metrics['mse']:.6f}"
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
    p = argparse.ArgumentParser(description="LSTM Bruteforce fuer Architekturvergleich.")
    p.add_argument("--full", action="store_true", help="Alle Trips (Fraction=1.0).")
    p.add_argument("--fraction", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seq-len", type=int, default=100)
    p.add_argument("--horizon", type=int, default=50)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Parallele DataLoader-Worker (Standard: 2).",
    )
    p.add_argument(
        "--parallel",
        type=int,
        default=2,
        help="Anzahl gleichzeitig trainierter Architekturen (Standard: 2).",
    )
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.45)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--max-train-trips", type=int, default=None)
    p.add_argument("--max-val-trips", type=int, default=None)
    p.add_argument("--max-train-windows", type=int, default=None)
    p.add_argument("--max-val-windows", type=int, default=None)

    p.add_argument("--max-layers", type=int, default=4)
    p.add_argument("--neurons", type=str, default="64,128,256")

    sort_group = p.add_mutually_exclusive_group()
    sort_group.add_argument(
        "--sort-desc",
        dest="sort_desc",
        action="store_true",
        help="Sortierung in Plots absteigend nach Validierungs-Rangsumme.",
    )
    sort_group.add_argument(
        "--sort-asc",
        dest="sort_desc",
        action="store_false",
        help="Sortierung in Plots aufsteigend nach Validierungs-Rangsumme (Default).",
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
        num_workers=a.num_workers,
        parallel=a.parallel,
        lr=a.lr,
        dropout=a.dropout,
        weight_decay=a.weight_decay,
        seed=a.seed,
        max_train_trips=a.max_train_trips,
        max_val_trips=a.max_val_trips,
        max_train_windows=a.max_train_windows,
        max_val_windows=a.max_val_windows,
        max_layers=a.max_layers,
        neurons=parse_int_csv(a.neurons),
        sort_desc=a.sort_desc,
        top_loss_curves=max(1, a.top_loss_curves),
    )
    if not 0.0 <= cfg.dropout < 1.0:
        raise ValueError("dropout muss im Bereich [0, 1) liegen")
    if cfg.weight_decay < 0.0:
        raise ValueError("weight_decay muss >= 0 sein")
    if cfg.batch_size < 1:
        raise ValueError("batch_size muss >= 1 sein")
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
    pin_memory = device.type == "cuda"
    if pin_memory:
        torch.backends.cudnn.benchmark = True

    archs = build_architectures(cfg)
    expected_count = sum(len(cfg.neurons) ** d for d in range(1, cfg.max_layers + 1))

    print("Konfiguration:", cfg)
    print("Device:", device)
    if pin_memory:
        print("GPU-Datenpfad: pinned memory, non-blocking Transfers, cuDNN benchmark aktiv")
    print(
        f"Architekturen: {len(archs)} (Neuronenoptionen={cfg.neurons}, "
        f"max_layers={cfg.max_layers}, erwartet={expected_count})"
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
        "n_architectures": len(archs),
        "expected_architectures": expected_count,
        "device": str(device),
    }

    results = load_checkpoints(archs, cfg)
    completed_archs = {result.arch for result in results}
    if results:
        save_outputs(results, cfg, meta_base, t_start)

    pending = [(i, arch) for i, arch in enumerate(archs, start=1) if arch not in completed_archs]
    for i, arch in enumerate(archs, start=1):
        if arch in completed_archs:
            print(f"\nRun {i}/{len(archs)}: LSTM {arch_to_name(arch)} bereits checkpointed.")

    n_parallel = min(cfg.parallel, max(1, len(pending)))
    print(f"\nParalleles Training: {n_parallel} Architektur(en) gleichzeitig.")

    # Ein DataLoader-Paar pro paralleler Spur; verhindert gleichzeitiges
    # Iterieren desselben Loaders aus mehreren Threads.
    loader_pool: "queue.Queue[tuple[DataLoader, DataLoader]]" = queue.Queue()
    for _ in range(n_parallel):
        loader_pool.put(make_loaders())

    init_lock = threading.Lock()
    save_lock = threading.Lock()

    def worker(i: int, arch: tuple[int, ...]) -> None:
        train_loader, val_loader = loader_pool.get()
        try:
            print("\n" + "-" * 68)
            print(f"Run {i}/{len(archs)}: architecture={arch_to_name(arch)}")
            res = train_one_arch(
                arch=arch,
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
            print(f"Checkpoint gespeichert: {checkpoint_paths(arch)[0]}")
            # Inkrementelles Speichern fuer lange Runs.
            save_outputs(results, cfg, meta_base, t_start)

    if pending:
        with ThreadPoolExecutor(max_workers=n_parallel) as pool:
            futures = [pool.submit(worker, i, arch) for i, arch in pending]
            for future in futures:
                future.result()

    print("\nFERTIG.")
    best, rank_score = rank_architectures(results)
    best = best[0]
    print(
        f"Bestes Modell nach Validierungs-Rangsumme: {best.arch_name} ({best.n_layers}L) "
        f"(Rangsumme={rank_score[best.arch_name]:.0f}, Val MSE={best.metrics['mse']:.6f})"
    )


if __name__ == "__main__":
    main()
