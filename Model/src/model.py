r"""
Finales Modelltraining fuer die ATO-Stellsignal-Vorhersage.

Nutzt die besten Erkenntnisse aus den vorangegangenen Sweeps:
    - Architektur : 2-Layer-GRU mit Hidden-Sizes (96, 64)
      (bestes Ergebnis aus `gru_architecture_comparison.py`)
    - Loss        : MSE
    - Optimizer   : Adam
      (beste Kombination aus `gru_loss_optimizer_comparison.py`)

Datenaufbereitung/Windowing sind identisch zu `model_comparison.py`
(wiederverwendete Funktionen: load_split, build_trip_arrays, SequenceDataset,
cap_dataset, regression_metrics).

Alle Hyperparameter und Regularisierungsmethoden sind als globale Variablen
weiter unten einstellbar. Die Regularisierung ist vorhanden, aber initial auf
0 gesetzt (deaktiviert).

Ausgaben in `reports/`:
    - model_final_report.html   (self-contained, Plots als base64)
    - model_final_curves.png    (Train-/Val-Loss + Accuracy)
    - model_final_metrics.csv / .json

Aufruf:
    .\.venv\Scripts\python.exe src\model.py
"""

from __future__ import annotations

import base64
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model_comparison import (
    DIR_DEADBAND,
    FEATURES,
    TOL,
    SequenceDataset,
    build_trip_arrays,
    cap_dataset,
    count_params,
    load_split,
    regression_metrics,
)


# ===========================================================================
# HYPERPARAMETER  (hier alles einstellen)
# ===========================================================================
# --- Architektur (bestes Ergebnis der Sweeps: 2-Layer-GRU 96-64) ---
GRU_ARCHITECTURE: tuple[int, ...] = (96, 64)

# --- Windowing / Daten ---
SEQ_LEN = 100          # Laenge der History in Schritten (100 = 20 s @ 5 Hz)
HORIZON = 50           # Vorhersagehorizont in Schritten (50 = 10 s)
STRIDE = 5             # Schrittweite beim Fensterbau
FRACTION = 1.0         # Anteil der Trips je Split (1.0 = alle Trips)

# --- Training ---
EPOCHS = 200
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
SEED = 42

# --- Optionale Daten-Caps (None = kein Limit) ---
MAX_TRAIN_TRIPS: int | None = None
MAX_VAL_TRIPS: int | None = None
MAX_TEST_TRIPS: int | None = None
MAX_TRAIN_WINDOWS: int | None = None
MAX_VAL_WINDOWS: int | None = None
MAX_TEST_WINDOWS: int | None = None


# ===========================================================================
# REGULARISIERUNG
# ===========================================================================
DROPOUT = 0.0            # Dropout zwischen GRU-Layern und vor dem Ausgabekopf
WEIGHT_DECAY = 0.0       # L2-Regularisierung (Adam weight_decay)
GRAD_CLIP_NORM = 0.0     # Max. Gradienten-Norm (0 = deaktiviert)
EARLY_STOP_PATIENCE = 0  # Epochen ohne Val-Verbesserung bis Abbruch (0 = aus)


# ---------------------------------------------------------------------------
# Pfade & Ausgabenamen
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"

REPORT_NAME = "model_final_report.html"
CURVES_PNG_NAMES = [
    "model_final_train_loss.png",
    "model_final_val_loss.png",
    "model_final_dir_acc.png",
    "model_final_tol_acc.png",
]
METRICS_CSV_NAME = "model_final_metrics.csv"
METRICS_JSON_NAME = "model_final_metrics.json"


# ---------------------------------------------------------------------------
# Matplotlib (nicht-interaktiv, nur PNG/HTML-Ausgabe)
# ---------------------------------------------------------------------------
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# Geraet auswaehlen: GPU (CUDA) / TPU (XLA) / Apple-MPS, sonst CPU
# ---------------------------------------------------------------------------
def select_device() -> torch.device:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"Geraet: CUDA-GPU ({name})")
        return torch.device("cuda")

    # TPU via torch_xla (nur falls installiert).
    try:
        import torch_xla.core.xla_model as xm  # type: ignore

        device = xm.xla_device()
        print(f"Geraet: TPU/XLA ({device})")
        return device
    except Exception:
        pass

    # Apple Silicon GPU.
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        print("Geraet: Apple-MPS-GPU")
        return torch.device("mps")

    print("Geraet: CPU (keine GPU/TPU gefunden)")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Konfiguration (fuer die wiederverwendeten Daten-Funktionen)
# ---------------------------------------------------------------------------
@dataclass
class Config:
    seq_len: int = SEQ_LEN
    horizon: int = HORIZON
    stride: int = STRIDE
    fraction: float = FRACTION
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    lr: float = LEARNING_RATE
    seed: int = SEED
    max_train_trips: int | None = MAX_TRAIN_TRIPS
    max_val_trips: int | None = MAX_VAL_TRIPS
    max_test_trips: int | None = MAX_TEST_TRIPS
    max_train_windows: int | None = MAX_TRAIN_WINDOWS
    max_val_windows: int | None = MAX_VAL_WINDOWS
    max_test_windows: int | None = MAX_TEST_WINDOWS


# ---------------------------------------------------------------------------
# Modell: 2-Layer-GRU (96, 64) mit optionalem Dropout
# ---------------------------------------------------------------------------
class GRURegressor(nn.Module):
    """Stack aus 1-Layer-GRUs mit unterschiedlicher Hidden-Size pro Layer."""

    def __init__(self, n_features: int, hidden_sizes: tuple[int, ...], dropout: float):
        super().__init__()
        if not hidden_sizes:
            raise ValueError("hidden_sizes darf nicht leer sein")

        grus = []
        in_size = n_features
        for h in hidden_sizes:
            grus.append(nn.GRU(in_size, h, num_layers=1, batch_first=True))
            in_size = h

        self.grus = nn.ModuleList(grus)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for i, gru in enumerate(self.grus):
            out, _ = gru(out)
            # Dropout auf die Sequenz zwischen den GRU-Layern (nicht nach dem letzten).
            if i < len(self.grus) - 1:
                out = self.dropout(out)
        last = out[:, -1, :]
        last = self.dropout(last)
        return torch.tanh(self.head(last)).squeeze(-1)


# ---------------------------------------------------------------------------
# Daten laden (identisch zur Pipeline der anderen Skripte)
# ---------------------------------------------------------------------------
def build_loaders(cfg: Config, rng: np.random.Generator):
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

    train_ds = cap_dataset(SequenceDataset(train_trips, train_labels, cfg),
                           cfg.max_train_windows, rng)
    val_ds = cap_dataset(SequenceDataset(val_trips, val_labels, cfg),
                         cfg.max_val_windows, rng)
    test_ds = cap_dataset(SequenceDataset(test_trips, test_labels, cfg),
                          cfg.max_test_windows, rng)
    print(
        f"  Train-Fenster: {len(train_ds):,}  Val-Fenster: {len(val_ds):,}  "
        f"Test-Fenster: {len(test_ds):,}"
    )

    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        raise SystemExit("Keine Fenster gebaut - SEQ_LEN/HORIZON/STRIDE pruefen.")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=0, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=0, drop_last=False)

    meta = {
        "n_train_trips": len(train_trips),
        "n_val_trips": len(val_trips),
        "n_test_trips": len(test_trips),
        "n_train_windows": len(train_ds),
        "n_val_windows": len(val_ds),
        "n_test_windows": len(test_ds),
    }
    return train_loader, val_loader, test_loader, meta


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    """Gibt val_mse, y_true, y_pred, y_prev zurueck."""
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
    return (
        total / max(n, 1),
        np.concatenate(yt),
        np.concatenate(yp),
        np.concatenate(ypr),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@dataclass
class TrainHistory:
    train_loss: list[float]
    val_loss: list[float]
    val_dir_acc_pct: list[float]
    val_tol_acc_pct: list[float]


def format_eta(seconds: float) -> str:
    seconds = int(round(max(seconds, 0)))
    h, rem = divmod(seconds, 3600)
    mnt, sec = divmod(rem, 60)
    if h:
        return f"{h}h {mnt:02d}m {sec:02d}s"
    if mnt:
        return f"{mnt}m {sec:02d}s"
    return f"{sec}s"


def train(model, cfg: Config, train_loader, val_loader, device):
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=WEIGHT_DECAY)
    crit = nn.MSELoss()

    hist = TrainHistory([], [], [], [])
    best_val = float("inf")
    best_state: dict | None = None
    epochs_no_improve = 0

    print(f"\n=== GRU {'-'.join(map(str, GRU_ARCHITECTURE))} "
          f"({count_params(model):,} Parameter) ===")
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.time()
        model.train()
        run_loss, n = 0.0, 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = crit(pred, y)
            loss.backward()
            if GRAD_CLIP_NORM > 0:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()
            run_loss += loss.item() * y.numel()
            n += y.numel()

        tr = run_loss / max(n, 1)
        vl, yt, yp, _ = evaluate(model, val_loader, device)
        m = regression_metrics(yt, yp)

        hist.train_loss.append(tr)
        hist.val_loss.append(vl)
        hist.val_dir_acc_pct.append(m["dir_acc_pct"])
        hist.val_tol_acc_pct.append(m["tol_acc_pct"])

        # ETA aus mittlerer Epochendauer und verbleibenden Epochen.
        avg_epoch_s = (time.time() - t0) / epoch
        remaining = cfg.epochs - epoch
        eta_s = avg_epoch_s * remaining

        print(
            f"  Epoch {epoch:02d}/{cfg.epochs}  train_mse={tr:.6f}  val_mse={vl:.6f}  "
            f"dir_acc={m['dir_acc_pct']:.2f}%  tol_acc={m['tol_acc_pct']:.2f}%  "
            f"({time.time() - epoch_start:.1f}s/Epoche, ETA {format_eta(eta_s)})"
        )

        # Bestes Modell (nach Val-MSE) merken + optionales Early Stopping.
        if vl < best_val:
            best_val = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if EARLY_STOP_PATIENCE > 0 and epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"  Early Stopping nach Epoche {epoch} "
                      f"(keine Verbesserung seit {EARLY_STOP_PATIENCE} Epochen).")
                break

    train_time_s = time.time() - t0

    # Bestes Modell wiederherstellen.
    if best_state is not None:
        model.load_state_dict(best_state)

    return hist, train_time_s


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def plot_curves(hist: TrainHistory, save_dir: Path) -> list[str]:
    ep = range(1, len(hist.train_loss) + 1)
    specs = [
        (hist.train_loss,     "#4C72B0", "Train-Loss (MSE)",       "MSE",  None,       CURVES_PNG_NAMES[0]),
        (hist.val_loss,       "#DD8452", "Val-Loss (MSE)",          "MSE",  None,       CURVES_PNG_NAMES[1]),
        (hist.val_dir_acc_pct, "#55A868", "Val-Richtungs-Accuracy", "%",   (0, 100),   CURVES_PNG_NAMES[2]),
        (hist.val_tol_acc_pct, "#C44E52", "Val-Toleranz-Accuracy",  "%",   (0, 100),   CURVES_PNG_NAMES[3]),
    ]
    b64_list = []
    for values, color, title, ylabel, ylim, png_name in specs:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(ep, values, color=color, marker="o")
        ax.set_title(title)
        ax.set_xlabel("Epoche")
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_dir / png_name, dpi=110, bbox_inches="tight")
        b64_list.append(_fig_to_b64(fig))
        plt.close(fig)
    return b64_list


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def build_report(curves_b64: list[str], val_metrics: dict, test_metrics: dict,
                 baseline_val: dict, baseline_test: dict, meta: dict) -> str:
    def metric_row(label: str, m: dict, extra: str = "") -> str:
        return (
            f"<tr><td><b>{label}</b></td>"
            f"<td>{m['r2']:.4f}</td><td>{m['rmse']:.4f}</td><td>{m['mae']:.4f}</td>"
            f"<td>{m['mse']:.6f}</td>"
            f"<td>{m['dir_acc_pct']:.2f}%</td><td>{m['tol_acc_pct']:.2f}%</td>{extra}</tr>"
        )

    rows = "\n".join([
        metric_row("Validierung", val_metrics),
        metric_row("Test", test_metrics),
        f"<tr style='color:#888'>{metric_row('Persistenz (Val)', baseline_val)[4:]}",
        f"<tr style='color:#888'>{metric_row('Persistenz (Test)', baseline_test)[4:]}",
    ])

    reg = (
        f"Dropout={DROPOUT} &middot; Weight-Decay={WEIGHT_DECAY} &middot; "
        f"Grad-Clip={GRAD_CLIP_NORM} &middot; Early-Stop-Patience={EARLY_STOP_PATIENCE}"
    )

    return f"""<!DOCTYPE html>
<html lang="de"><head><meta charset="utf-8">
<title>Finales Modelltraining (GRU {'-'.join(map(str, GRU_ARCHITECTURE))})</title>
<style>
 body {{ font-family: Segoe UI, Arial, sans-serif; margin: 32px; color:#222; max-width:1100px; }}
 h1 {{ border-bottom: 3px solid #4C72B0; padding-bottom:6px; }}
 h2 {{ margin-top: 30px; color:#333; }}
 table {{ border-collapse: collapse; margin: 12px 0; font-size: 14px; }}
 th, td {{ border: 1px solid #ccc; padding: 6px 12px; text-align: right; }}
 th {{ background:#f0f3f7; }}
 td:first-child, th:first-child {{ text-align: left; }}
 img {{ max-width: 100%; border:1px solid #eee; border-radius:6px; margin: 6px 0 18px; }}
 .cfg {{ background:#f7f9fb; border:1px solid #e2e8f0; border-radius:6px; padding:10px 16px; font-size:13px; }}
 .note {{ color:#666; font-size:13px; }}
</style></head>
<body>
<h1>Finales Modelltraining &ndash; GRU {'-'.join(map(str, GRU_ARCHITECTURE))}</h1>
<p>Beste Konfiguration aus den Sweeps: 2-Layer-GRU (96&ndash;64), Loss=MSE, Optimizer=Adam.</p>

<h2>Setup</h2>
<div class="cfg">
Aufgabe: History (H={SEQ_LEN} Schritte = {SEQ_LEN*0.2:.0f}&nbsp;s @ 5&nbsp;Hz)
&rarr; label bei t+{HORIZON} ({HORIZON*200}&nbsp;ms Vorhersagehorizont).<br>
Features ({len(FEATURES)}): {", ".join(FEATURES)}.<br>
Fraction={FRACTION} &middot; Stride={STRIDE} &middot; Epochen={EPOCHS}
&middot; Batch={BATCH_SIZE} &middot; LR={LEARNING_RATE}.<br>
Regularisierung: {reg}.<br>
Train-Trips: {meta['n_train_trips']} &middot; Val-Trips: {meta['n_val_trips']}
&middot; Test-Trips: {meta['n_test_trips']}<br>
Train-Fenster: {meta['n_train_windows']:,} &middot; Val-Fenster: {meta['n_val_windows']:,}
&middot; Test-Fenster: {meta['n_test_windows']:,}.<br>
Parameter: {meta['n_params']:,} &middot; Trainingszeit: {meta['train_time_s']:.1f}s
&middot; Device: {meta['device']}.
</div>

<h2>Ergebnisse</h2>
<table>
<tr><th>Split</th><th>R&sup2;</th><th>RMSE</th><th>MAE</th><th>MSE</th>
<th>Richtungs-Acc</th><th>Toleranz-Acc</th></tr>
{rows}
</table>
<p class="note">Richtungs-Acc: 3-Klassen (bremsen/halten/beschl.) mit Deadband
&plusmn;{DIR_DEADBAND}. Toleranz-Acc: Anteil |pred&minus;real| &le; {TOL}.
Persistenz nutzt das letzte wahre Label als Vorhersage (Referenz).</p>

<h2>Lernkurven</h2>
<img src="data:image/png;base64,{curves_b64[0]}"/>
<img src="data:image/png;base64,{curves_b64[1]}"/>
<img src="data:image/png;base64,{curves_b64[2]}"/>
<img src="data:image/png;base64,{curves_b64[3]}"/>

<p class="note">Erzeugt am {meta['timestamp']}.</p>
</body></html>"""


# ---------------------------------------------------------------------------
# Ausgaben speichern
# ---------------------------------------------------------------------------
def save_outputs(hist, val_metrics, test_metrics, baseline_val, baseline_test, meta):
    REPORTS_DIR.mkdir(exist_ok=True)

    curves_b64 = plot_curves(hist, REPORTS_DIR)
    html = build_report(curves_b64, val_metrics, test_metrics,
                        baseline_val, baseline_test, meta)
    (REPORTS_DIR / REPORT_NAME).write_text(html, encoding="utf-8")

    rows = [
        {"split": "val", **val_metrics},
        {"split": "test", **test_metrics},
        {"split": "val_persistence", **baseline_val},
        {"split": "test_persistence", **baseline_test},
    ]
    pd.DataFrame(rows).to_csv(REPORTS_DIR / METRICS_CSV_NAME, index=False)
    (REPORTS_DIR / METRICS_JSON_NAME).write_text(
        json.dumps({"config": meta, "metrics": rows}, indent=2), encoding="utf-8")

    print(f"\nReport : {REPORTS_DIR / REPORT_NAME}")
    for name in CURVES_PNG_NAMES:
        print(f"Plot   : {REPORTS_DIR / name}")
    print(f"Metriken: {REPORTS_DIR / METRICS_CSV_NAME}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t_start = time.time()
    cfg = Config()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = select_device()

    print("Konfiguration:", cfg)
    print("Device:", device)

    rng = np.random.default_rng(cfg.seed)
    train_loader, val_loader, test_loader, meta = build_loaders(cfg, rng)

    model = GRURegressor(len(FEATURES), GRU_ARCHITECTURE, DROPOUT).to(device)

    hist, train_time_s = train(model, cfg, train_loader, val_loader, device)

    # Finale Metriken auf Val und Test mit dem besten Modell.
    _, yt_val, yp_val, yprev_val = evaluate(model, val_loader, device)
    _, yt_test, yp_test, yprev_test = evaluate(model, test_loader, device)
    val_metrics = regression_metrics(yt_val, yp_val)
    test_metrics = regression_metrics(yt_test, yp_test)
    baseline_val = regression_metrics(yt_val, yprev_val)
    baseline_test = regression_metrics(yt_test, yprev_test)

    meta.update({
        "n_params": count_params(model),
        "train_time_s": train_time_s,
        "device": str(device),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_time_s": time.time() - t_start,
    })

    print("\n" + "=" * 60)
    print("FINALE ERGEBNISSE")
    print(f"  Val : R2={val_metrics['r2']:.4f}  RMSE={val_metrics['rmse']:.4f}  "
          f"dir_acc={val_metrics['dir_acc_pct']:.2f}%  tol_acc={val_metrics['tol_acc_pct']:.2f}%")
    print(f"  Test: R2={test_metrics['r2']:.4f}  RMSE={test_metrics['rmse']:.4f}  "
          f"dir_acc={test_metrics['dir_acc_pct']:.2f}%  tol_acc={test_metrics['tol_acc_pct']:.2f}%")

    save_outputs(hist, val_metrics, test_metrics, baseline_val, baseline_test, meta)


if __name__ == "__main__":
    main()
