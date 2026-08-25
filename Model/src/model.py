r"""
Finales Modelltraining fuer die ATO-Stellsignal-Vorhersage.

Nutzt die besten Erkenntnisse aus den vorangegangenen Sweeps:
        - Architektur : 4-Layer-LSTM mit Hidden-Sizes (128, 64, 256, 128)
    - Loss        : SmoothL1
        - Optimizer   : Adam
            (beste Kombination aus `loss_optimizer_comparison.py`)
    - Regularisierung: festes Dropout, Weight-Decay, Grad-Clipping,
      additives Gauss-Rauschen auf den Trainingsfeatures, Early-Stopping
    - Initialisierung: Xavier/Glorot fuer alle LSTM-/Linear-Gewichte,
      Forget-Gate-Bias = 1.0

Datenaufbereitung/Windowing sind identisch zu `model_comparison.py`
(wiederverwendete Funktionen: load_split, build_trip_arrays, SequenceDataset,
cap_dataset, regression_metrics).

Alle Hyperparameter und Regularisierungsmethoden sind als globale Variablen
weiter unten einstellbar. Regularisierung ist aktiv (Dropout, Weight-Decay,
Grad-Clipping, Early-Stopping), da der erste Lauf ohne Regularisierung
(siehe reports/model_final_report.html) bereits ab Epoche 2 Overfitting
zeigte (Val-Loss steigt, Train-Loss faellt weiter).

Ausgaben in `reports/`:
    - model_final_report.html   (self-contained, Plots als base64)
    - model_final_curves.png    (Train-/Val-Loss + Accuracy)
    - model_final_metrics.csv / .json
    - model_checkpoints/training_checkpoint.pt (Resume-Checkpoint nach jeder Epoche)
    - model_checkpoints/best_model.pt (bestes Modell nach Val-Score, bei jeder Verbesserung)

Aufruf:
    .\.venv\Scripts\python.exe src\model.py

Fortsetzung:
    Bei einem Neustart wird ein Checkpoint mit identischer Konfiguration geladen
    und ab der naechsten Epoche weitertrainiert. Bei geaenderter Konfiguration
    wird der Checkpoint ignoriert und ein neuer Lauf begonnen.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import time
from contextlib import nullcontext
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
# --- Architektur (4-Layer-LSTM 128-64-256-128) ---
LSTM_ARCHITECTURE: tuple[int, ...] = (128, 64, 256, 128)

# --- Windowing / Daten ---
SEQ_LEN = 100          # Laenge der History in Schritten (100 = 20 s @ 5 Hz)
HORIZON = 50           # Vorhersagehorizont in Schritten (25 = 5 s)
STRIDE = 5             # Schrittweite beim Fensterbau
FRACTION = 1.0         # Anteil der Trips je Split (1.0 = alle Trips)

# --- Training ---
EPOCHS = 100
BATCH_SIZE = 2048
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
SEED = 2

# --- Speicherstabilitaet auf GPU ---
# Effektive Trainings-Batch bleibt BATCH_SIZE, wird auf CUDA aber in kleinere
# Micro-Batches aufgeteilt (Gradient Accumulation), um OOM zu vermeiden.
TRAIN_MICRO_BATCH_SIZE = 512
USE_AMP = True  # Mixed Precision nur auf CUDA

# --- LR-Scheduler (ReduceLROnPlateau auf Val-Score) ---
# ReduceLROnPlateau vergleicht jede Epoche gegen den *besten bisher gesehenen*
# Val-Score (nicht gegen die letzte Epoche). Oszilliert der Val-Score also
# 10 Epochen lang nur um den letzten Wert herum, ohne einen neuen Bestwert zu
# erreichen, zaehlt das weiterhin als "keine Verbesserung" und die LR wird
# reduziert.
LR_SCHEDULER_FACTOR = 0.5     # Faktor, um den die LR bei Plateau reduziert wird
LR_SCHEDULER_PATIENCE = 25    # Epochen ohne (neuen) Val-Bestwert bis LR reduziert wird
LR_SCHEDULER_MIN_LR = 1e-6    # Untere Grenze fuer die LR

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
# model_final_report.html (Lauf ohne Regularisierung) zeigt klares Overfitting:
# Val-Loss steigt bereits ab Epoche 2 an, waehrend Train-Loss durchgehend faellt
# (bestes Modell wurde nach Epoche 1 gespeichert). Daher hier aktiviert:
DROPOUT = 0.45           # Mittelwert des zyklischen Dropouts (auch Init-Wert des Modells)
DROPOUT_AMPLITUDE = 0.15  # Auslenkung -> Bereich [0.30, 0.60]
DROPOUT_CYCLE_EPOCHS = 10
WEIGHT_DECAY = 1e-4      # L2-Regularisierung (Optimizer weight_decay)
SMOOTH_L1_BETA = 0.10    # Beta-Parameter fuer SmoothL1-Loss
GRAD_CLIP_NORM = 1.0     # Max. Gradienten-Norm (0 = deaktiviert)
EARLY_STOP_PATIENCE = 30  # Epochen ohne Val-Verbesserung bis Abbruch (0 = aus)
INPUT_NOISE_STD = 0.02   # Mittelwert des zyklischen Gauss-Rauschens (0 = aus)
INPUT_NOISE_AMPLITUDE = 0.02  # Auslenkung -> Bereich [0.00, 0.04]
INPUT_NOISE_CYCLE_EPOCHS = 5

# --- Exponential Moving Average der Gewichte (Polyak-Averaging) ---
# Die EMA-Gewichte werden fuer Validierung, Modellauswahl und die finale
# Auswertung genutzt; trainiert wird weiterhin mit den Live-Gewichten.
EMA_ENABLED = True
EMA_DECAY = 0.999        # Zerfallsfaktor je Optimierer-Schritt

# --- Initialisierung ---
WEIGHT_INITIALIZATION = "xavier_uniform"
FORGET_GATE_BIAS = 1.0


# ---------------------------------------------------------------------------
# Pfade & Ausgabenamen
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"
CHECKPOINT_DIR = REPORTS_DIR / "model_checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "training_checkpoint.pt"
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pt"

REPORT_NAME = "model_final_report.html"
CURVES_PNG_NAMES = [
    "model_final_train_loss.png",
    "model_final_val_loss.png",
]
METRICS_CSV_NAME = "model_final_metrics.csv"
METRICS_JSON_NAME = "model_final_metrics.json"

CONFUSION_PNG_NAMES = {
    "val": "model_final_dir_confusion_val.png",
    "test": "model_final_dir_confusion_test.png",
}
DIRECTION_CLASS_NAMES = ["bremsen", "halten", "beschleunigen"]


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
    num_workers: int = NUM_WORKERS
    lr: float = LEARNING_RATE
    seed: int = SEED
    max_train_trips: int | None = MAX_TRAIN_TRIPS
    max_val_trips: int | None = MAX_VAL_TRIPS
    max_test_trips: int | None = MAX_TEST_TRIPS
    max_train_windows: int | None = MAX_TRAIN_WINDOWS
    max_val_windows: int | None = MAX_VAL_WINDOWS
    max_test_windows: int | None = MAX_TEST_WINDOWS


# ---------------------------------------------------------------------------
# Modell: 4-Layer-LSTM (128, 64, 256, 128) mit optionalem Dropout
# ---------------------------------------------------------------------------
class LSTMRegressor(nn.Module):
    """Stack aus 1-Layer-LSTMs mit unterschiedlicher Hidden-Size pro Layer."""

    def __init__(self, n_features: int, hidden_sizes: tuple[int, ...], dropout: float):
        super().__init__()
        if not hidden_sizes:
            raise ValueError("hidden_sizes darf nicht leer sein")

        lstms = []
        in_size = n_features
        for h in hidden_sizes:
            lstms.append(nn.LSTM(in_size, h, num_layers=1, batch_first=True))
            in_size = h

        self.lstms = nn.ModuleList(lstms)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_sizes[-1], 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for lstm in self.lstms:
            hidden_size = lstm.hidden_size
            for name, param in lstm.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
                    # Forget-Gate-Bias liegt im zweiten Viertel von bias_ih/bias_hh.
                    if name.startswith("bias_ih"):
                        param.data[hidden_size:2 * hidden_size].fill_(FORGET_GATE_BIAS)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for i, lstm in enumerate(self.lstms):
            out, _ = lstm(out)
            # Dropout auf die Sequenz zwischen den LSTM-Layern (nicht nach dem letzten).
            if i < len(self.lstms) - 1:
                out = self.dropout(out)
        last = out[:, -1, :]
        last = self.dropout(last)
        return torch.tanh(self.head(last)).squeeze(-1)


# ---------------------------------------------------------------------------
# Skalierungs-Check: Features z-standardisiert (scaler.json), Labels in [-1,1]
# ---------------------------------------------------------------------------
def verify_scaling(train_trips: list[np.ndarray], train_labels: list[np.ndarray]) -> None:
    feats = np.concatenate(train_trips, axis=0)
    labels = np.concatenate(train_labels, axis=0)

    mean = feats.mean(axis=0)
    std = feats.std(axis=0)
    print("\nSkalierungs-Check (Train-Split):")
    for name, mu, sigma in zip(FEATURES, mean, std):
        ok = abs(mu) < 0.5 and 0.5 < sigma < 1.5
        flag = "" if ok else "  <-- WARNUNG: kein Z-Score!"
        print(f"  {name:22s} mean={mu:+.3f}  std={sigma:.3f}{flag}")

    lab_min, lab_max = float(labels.min()), float(labels.max())
    lab_ok = -1.0001 <= lab_min and lab_max <= 1.0001
    lab_flag = "" if lab_ok else "  <-- WARNUNG: ausserhalb [-1,1]!"
    print(f"  {'label':22s} min={lab_min:+.3f}  max={lab_max:+.3f}{lab_flag}")

    if not lab_ok:
        raise SystemExit("Labels sind nicht auf [-1,1] normalisiert - Datenaufbereitung pruefen.")


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

    verify_scaling(train_trips, train_labels)

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

    loader_kwargs = {
        "num_workers": cfg.num_workers,
        "drop_last": False,
        "persistent_workers": cfg.num_workers > 0,
        "prefetch_factor": 2 if cfg.num_workers > 0 else None,
    }
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             **loader_kwargs)

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


class WeightEMA:
    """Exponentiell gleitender Mittelwert der Modellgewichte (Polyak-Averaging)."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self.extras: dict[str, torch.Tensor] = {}
        for key, value in model.state_dict().items():
            if value.dtype.is_floating_point:
                self.shadow[key] = value.detach().clone()
            else:
                self.extras[key] = value.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for key, value in model.state_dict().items():
            if key in self.shadow:
                self.shadow[key].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                self.extras[key] = value.detach().clone()

    def state_dict(self) -> dict[str, torch.Tensor]:
        merged = {k: v.clone() for k, v in self.extras.items()}
        merged.update({k: v.clone() for k, v in self.shadow.items()})
        return merged

    def load_state_dict(self, state: dict[str, torch.Tensor], device: torch.device) -> None:
        for key, value in state.items():
            target = self.shadow if key in self.shadow else self.extras
            target[key] = value.detach().clone().to(device)


def checkpoint_signature(cfg: Config) -> str:
    values = {
        "features": FEATURES,
        "model_family": "LSTM",
        "architecture": LSTM_ARCHITECTURE,
        "optimizer": "adam",
        "loss": "smooth_l1",
        "smooth_l1_beta": SMOOTH_L1_BETA,
        "seq_len": cfg.seq_len,
        "horizon": cfg.horizon,
        "stride": cfg.stride,
        "fraction": cfg.fraction,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "lr": cfg.lr,
        "seed": cfg.seed,
        "dropout": DROPOUT,
        "weight_decay": WEIGHT_DECAY,
        "input_noise_std": INPUT_NOISE_STD,
        "input_noise_amplitude": INPUT_NOISE_AMPLITUDE,
        "input_noise_cycle_epochs": INPUT_NOISE_CYCLE_EPOCHS,
        "dropout_amplitude": DROPOUT_AMPLITUDE,
        "dropout_cycle_epochs": DROPOUT_CYCLE_EPOCHS,
        "ema_enabled": EMA_ENABLED,
        "ema_decay": EMA_DECAY,
        "weight_initialization": WEIGHT_INITIALIZATION,
        "forget_gate_bias": FORGET_GATE_BIAS,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "lr_scheduler_factor": LR_SCHEDULER_FACTOR,
        "lr_scheduler_patience": LR_SCHEDULER_PATIENCE,
        "lr_scheduler_min_lr": LR_SCHEDULER_MIN_LR,
        "max_train_trips": cfg.max_train_trips,
        "max_val_trips": cfg.max_val_trips,
        "max_test_trips": cfg.max_test_trips,
        "max_train_windows": cfg.max_train_windows,
        "max_val_windows": cfg.max_val_windows,
        "max_test_windows": cfg.max_test_windows,
    }
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _move_optimizer_state_to_device(optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def save_training_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer,
    scheduler,
    history: TrainHistory,
    best_val_loss: float,
    best_epoch: int,
    best_state: dict | None,
    epochs_no_improve: int,
    lr_reduce_count: int,
    cfg: Config,
    ema_state: dict | None = None,
) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 11,
        "signature": checkpoint_signature(cfg),
        "features": FEATURES,
        "epoch": epoch,
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "history": history.__dict__,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "best_state": best_state,
        "epochs_no_improve": epochs_no_improve,
        "lr_reduce_count": lr_reduce_count,
        "ema_state": ema_state,
    }
    temporary_path = CHECKPOINT_PATH.with_suffix(".tmp.pt")
    torch.save(payload, temporary_path)
    temporary_path.replace(CHECKPOINT_PATH)


def save_best_model(
    best_state: dict,
    best_epoch: int,
    best_val_loss: float,
    cfg: Config,
) -> None:
    """Speichert das bisher beste Modell als eigenstaendige Datei (nur Gewichte + Metadaten)."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 11,
        "signature": checkpoint_signature(cfg),
        "features": FEATURES,
        "architecture": LSTM_ARCHITECTURE,
        "model_family": "LSTM",
        "seq_len": cfg.seq_len,
        "horizon": cfg.horizon,
        "best_state": best_state,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "dropout": DROPOUT,
    }
    temporary_path = BEST_MODEL_PATH.with_suffix(".tmp.pt")
    torch.save(payload, temporary_path)
    temporary_path.replace(BEST_MODEL_PATH)


def load_training_checkpoint(
    model: nn.Module, optimizer, scheduler, cfg: Config, device: torch.device
) -> tuple[int, TrainHistory, float, int, dict | None, int, int, dict | None] | None:
    if not CHECKPOINT_PATH.exists():
        return None
    try:
        payload = torch.load(CHECKPOINT_PATH, map_location="cpu")
        if payload.get("version") != 11 or payload.get("signature") != checkpoint_signature(cfg):
            print("Checkpoint ignoriert: Trainingskonfiguration ist nicht identisch.")
            return None
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        _move_optimizer_state_to_device(optimizer, device)
        scheduler.load_state_dict(payload["scheduler_state"])
        history = TrainHistory(**payload["history"])
        print(
            f"Checkpoint geladen: Epoche {payload['epoch']}/{cfg.epochs}, "
            f"bester Val-Loss={payload['best_val_loss']:.6f} "
            f"(Epoche {payload['best_epoch']})"
        )
        print(
            "Patience-Stand uebernommen: "
            f"Early-Stop {payload['epochs_no_improve']}/{EARLY_STOP_PATIENCE}, "
            f"LR-Scheduler {scheduler.num_bad_epochs}/{LR_SCHEDULER_PATIENCE}, "
            f"LR={optimizer.param_groups[0]['lr']:.2e}"
        )
        return (
            int(payload["epoch"]), history, float(payload["best_val_loss"]),
            int(payload["best_epoch"]), payload["best_state"],
            int(payload["epochs_no_improve"]), int(payload["lr_reduce_count"]),
            payload.get("ema_state"),
        )
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f"Checkpoint konnte nicht geladen werden und wird ignoriert: {exc}")
        return None


def _cycle_value(epoch: int, base: float, amplitude: float, cycle_epochs: int) -> float:
    """Sinusfoermige Oszillation um `base`; Epoche 1 startet im Mittelwert."""
    if amplitude <= 0 or cycle_epochs <= 0:
        return base
    phase = 2.0 * np.pi * ((epoch - 1) % cycle_epochs) / cycle_epochs
    return float(base + amplitude * np.sin(phase))


def input_noise_std_for_epoch(epoch: int) -> float:
    if INPUT_NOISE_STD <= 0:
        return 0.0
    return max(0.0, _cycle_value(epoch, INPUT_NOISE_STD, INPUT_NOISE_AMPLITUDE,
                                 INPUT_NOISE_CYCLE_EPOCHS))


def dropout_for_epoch(epoch: int) -> float:
    return min(1.0, max(0.0, _cycle_value(epoch, DROPOUT, DROPOUT_AMPLITUDE,
                                          DROPOUT_CYCLE_EPOCHS)))


def set_model_dropout(model: nn.Module, dropout: float) -> None:
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout


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
    opt = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
        min_lr=LR_SCHEDULER_MIN_LR,
    )
    crit = nn.SmoothL1Loss(beta=SMOOTH_L1_BETA)
    amp_enabled = USE_AMP and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    hist = TrainHistory([], [])
    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict | None = None
    epochs_no_improve = 0
    lr_reduce_count = 0
    start_epoch = 1
    ema = WeightEMA(model, EMA_DECAY) if EMA_ENABLED else None

    loaded = load_training_checkpoint(model, opt, scheduler, cfg, device)
    if loaded is not None:
        (
            completed_epoch,
            hist,
            best_val_loss,
            best_epoch,
            best_state,
            epochs_no_improve,
            lr_reduce_count,
            ema_state,
        ) = loaded
        start_epoch = completed_epoch + 1
        if ema is not None and ema_state is not None:
            ema.load_state_dict(ema_state, device)

    print(f"\n=== LSTM {'-'.join(map(str, LSTM_ARCHITECTURE))} "
          f"({count_params(model):,} Parameter) ===")
    if device.type == "cuda" and cfg.batch_size > TRAIN_MICRO_BATCH_SIZE:
        print(
            "Hinweis: CUDA-Micro-Batching aktiv "
            f"({TRAIN_MICRO_BATCH_SIZE} statt {cfg.batch_size} pro Backward-Pass)."
        )
    if amp_enabled:
        print("Hinweis: Automatic Mixed Precision (AMP) aktiv.")
    print("Scheduler: ReduceLROnPlateau")
    print(
        f"Regularisierung: dropout={DROPOUT:.2f} +/- {DROPOUT_AMPLITUDE:.2f} "
        f"(Zyklus {DROPOUT_CYCLE_EPOCHS} Epochen), weight_decay={WEIGHT_DECAY}, "
        f"input_noise_std={INPUT_NOISE_STD} +/- {INPUT_NOISE_AMPLITUDE} "
        f"(Zyklus {INPUT_NOISE_CYCLE_EPOCHS} Epochen), "
        f"grad_clip={GRAD_CLIP_NORM}, early_stop_patience={EARLY_STOP_PATIENCE}"
    )
    if ema is not None:
        print(f"Gewichts-EMA aktiv (decay={EMA_DECAY}); Validierung nutzt die EMA-Gewichte.")
    t0 = time.time()

    for epoch in range(start_epoch, cfg.epochs + 1):
        epoch_start = time.time()
        noise_std = input_noise_std_for_epoch(epoch)
        epoch_dropout = dropout_for_epoch(epoch)
        set_model_dropout(model, epoch_dropout)
        model.train()
        run_loss, n = 0.0, 0
        for x_cpu, y_cpu, _ in train_loader:
            batch_n = y_cpu.numel()
            n += batch_n
            opt.zero_grad(set_to_none=True)

            micro_bs = batch_n
            if device.type == "cuda":
                micro_bs = min(batch_n, TRAIN_MICRO_BATCH_SIZE)

            try:
                for start in range(0, batch_n, micro_bs):
                    end = min(start + micro_bs, batch_n)
                    xb = x_cpu[start:end].to(device, non_blocking=True)
                    yb = y_cpu[start:end].to(device, non_blocking=True)
                    if noise_std > 0:
                        xb = xb + torch.randn_like(xb) * noise_std

                    amp_ctx = (
                        torch.autocast(device_type="cuda", dtype=torch.float16)
                        if amp_enabled
                        else nullcontext()
                    )
                    with amp_ctx:
                        pred = model(xb)
                        loss = crit(pred, yb)

                    # Mit reduction='mean' skaliert dieser Faktor die Summe der
                    # Micro-Batch-Gradienten auf den Full-Batch-Mittelwert.
                    weight = yb.numel() / batch_n
                    scaler.scale(loss * weight).backward()
                    run_loss += loss.item() * yb.numel()
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                raise RuntimeError(
                    "CUDA OOM trotz Micro-Batching. "
                    "Setze TRAIN_MICRO_BATCH_SIZE kleiner (z.B. 256 oder 128) "
                    "oder reduziere SEQ_LEN/HORIZON/Architektur."
                ) from exc

            if GRAD_CLIP_NORM > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                ema.update(model)

        tr = run_loss / max(n, 1)
        if ema is not None:
            live_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            eval_state = ema.state_dict()
            model.load_state_dict(eval_state)
            vl, _, _, _ = evaluate(model, val_loader, device)
            model.load_state_dict(live_state)
        else:
            vl, _, _, _ = evaluate(model, val_loader, device)
            eval_state = model.state_dict()

        hist.train_loss.append(tr)
        hist.val_loss.append(vl)

        lr_before = opt.param_groups[0]["lr"]
        scheduler.step(vl)
        lr_after = opt.param_groups[0]["lr"]
        if lr_after < lr_before:
            lr_reduce_count += 1

        # ETA aus mittlerer Epochendauer und verbleibenden Epochen.
        avg_epoch_s = (time.time() - t0) / epoch
        remaining = cfg.epochs - epoch
        eta_s = avg_epoch_s * remaining

        print(
            f"  Epoch {epoch:02d}/{cfg.epochs}  train_obj={tr:.6f}  val_mse={vl:.6f}  "
            f"lr={lr_after:.2e}  dropout={epoch_dropout:.3f}  noise={noise_std:.4f}  "
            f"({time.time() - epoch_start:.1f}s/Epoche, ETA {format_eta(eta_s)})"
        )
        if lr_after < lr_before:
            print(f"  LR reduziert: {lr_before:.2e} -> {lr_after:.2e}")

        # Bestes Modell (nach Val-Loss) merken + optionales Early Stopping.
        if vl < best_val_loss:
            best_val_loss = vl
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in eval_state.items()}
            epochs_no_improve = 0
            save_best_model(best_state, best_epoch, best_val_loss, cfg)
            print(f"  Neues bestes Modell gespeichert: {BEST_MODEL_PATH}")
        else:
            epochs_no_improve += 1

        save_training_checkpoint(
            epoch,
            model,
            opt,
            scheduler,
            hist,
            best_val_loss,
            best_epoch,
            best_state,
            epochs_no_improve,
            lr_reduce_count,
            cfg,
            ema.state_dict() if ema is not None else None,
        )
        print(f"  Checkpoint gespeichert: {CHECKPOINT_PATH}")

        if EARLY_STOP_PATIENCE > 0 and epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"  Early Stopping nach Epoche {epoch} "
                  f"(keine Verbesserung seit {EARLY_STOP_PATIENCE} Epochen).")
            break

    train_time_s = time.time() - t0
    final_state = ema.state_dict() if ema is not None else model.state_dict()
    last_state = {k: v.detach().cpu().clone() for k, v in final_state.items()}
    last_epoch = len(hist.train_loss)

    # Bestes Modell wiederherstellen.
    if best_state is not None:
        model.load_state_dict(best_state)
    print(
        f"Best-Checkpoint geladen (Val-Loss): {best_val_loss:.6f} "
        f"aus Epoche {best_epoch}"
    )
    print(f"LR-Reduktionen gesamt: {lr_reduce_count}")

    return hist, train_time_s, best_epoch, best_val_loss, last_state, last_epoch


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
        (hist.train_loss, "#4C72B0", "Train-Loss (SmoothL1)", "Loss", None, CURVES_PNG_NAMES[0]),
        (hist.val_loss,   "#DD8452", "Val-Loss (MSE)",        "MSE",  None, CURVES_PNG_NAMES[1]),
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


def direction_bins(a: np.ndarray) -> np.ndarray:
    """Richtungs-Klassen analog zur Richtungs-Acc-Metrik (0/1/2)."""
    c = np.ones_like(a, dtype=np.int64)  # 1 = halten/coast
    c[a > DIR_DEADBAND] = 2   # beschleunigen
    c[a < -DIR_DEADBAND] = 0  # bremsen
    return c


def direction_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """3x3-Konfusionsmatrix der Richtungs-Klassen (Zeile=real, Spalte=predicted)."""
    t = direction_bins(y_true)
    p = direction_bins(y_pred)
    cm = np.zeros((3, 3), dtype=np.int64)
    for real, pred in zip(t, p):
        cm[real, pred] += 1
    return cm


def plot_direction_confusion(y_true: np.ndarray, y_pred: np.ndarray,
                             title: str, save_path: Path) -> tuple[str, np.ndarray]:
    """Zeilen-normierte Richtungs-Konfusionsmatrix (Prozent je realer Klasse)."""
    cm = direction_confusion(y_true, y_pred)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm * 100.0, row_sums, out=np.zeros_like(cm, dtype=float),
                       where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(DIRECTION_CLASS_NAMES)
    ax.set_yticklabels(DIRECTION_CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Real")
    ax.set_title(title)
    for i in range(3):
        for j in range(3):
            txt_color = "white" if cm_pct[i, j] > 50 else "#222"
            ax.text(j, i, f"{cm_pct[i, j]:.1f}%\n(n={cm[i, j]:,})",
                    ha="center", va="center", color=txt_color, fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Anteil je realer Klasse (%)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64, cm


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def build_report(curves_b64: list[str], val_metrics: dict, test_metrics: dict,
                 baseline_val: dict, baseline_test: dict, meta: dict,
                 confusion_b64: dict[str, str],
                 val_metrics_last: dict, test_metrics_last: dict) -> str:
    def metric_row(label: str, m: dict, extra: str = "") -> str:
        return (
            f"<tr><td><b>{label}</b></td>"
            f"<td>{m['r2']:.4f}</td><td>{m['rmse']:.4f}</td><td>{m['mae']:.4f}</td>"
            f"<td>{m['mse']:.6f}</td>"
            f"<td>{m['dir_acc_pct']:.2f}%</td><td>{m['tol_acc_pct']:.2f}%</td>{extra}</tr>"
        )

    rows = "\n".join([
        metric_row(f"Validierung &ndash; bestes Modell (Epoche {meta['best_epoch']})",
                   val_metrics),
        metric_row(f"Test &ndash; bestes Modell (Epoche {meta['best_epoch']})", test_metrics),
        metric_row(f"Validierung &ndash; letzte Epoche ({meta['last_epoch']})", val_metrics_last),
        metric_row(f"Test &ndash; letzte Epoche ({meta['last_epoch']})", test_metrics_last),
        f"<tr style='color:#888'>{metric_row('Persistenz (Val)', baseline_val)[4:]}",
        f"<tr style='color:#888'>{metric_row('Persistenz (Test)', baseline_test)[4:]}",
    ])

    reg = (
        f"Dropout zyklisch {DROPOUT}&nbsp;&plusmn;&nbsp;{DROPOUT_AMPLITUDE} "
        f"(Zyklus {DROPOUT_CYCLE_EPOCHS} Epochen) &middot; Weight-Decay={WEIGHT_DECAY} &middot; "
        f"Input-Noise (nur Training, Gauss) zyklisch {INPUT_NOISE_STD}&nbsp;&plusmn;&nbsp;"
        f"{INPUT_NOISE_AMPLITUDE} (Zyklus {INPUT_NOISE_CYCLE_EPOCHS} Epochen) &middot; "
        f"Grad-Clip={GRAD_CLIP_NORM} &middot; Early-Stop-Patience={EARLY_STOP_PATIENCE} "
        f"&middot; Gewichts-EMA={'aktiv, decay=' + str(EMA_DECAY) if EMA_ENABLED else 'aus'}"
    )

    return f"""<!DOCTYPE html>
<html lang="de"><head><meta charset="utf-8">
<title>Finales Modelltraining (LSTM {'-'.join(map(str, LSTM_ARCHITECTURE))})</title>
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
<h1>Finales Modelltraining &ndash; LSTM {'-'.join(map(str, LSTM_ARCHITECTURE))}</h1>
<p>Konfiguration: 4-Layer-LSTM (128&ndash;64&ndash;256&ndash;128), Loss=SmoothL1, Optimizer=Adam.</p>

<h2>Setup</h2>
<div class="cfg">
Aufgabe: History (H={SEQ_LEN} Schritte = {SEQ_LEN*0.2:.0f}&nbsp;s @ 5&nbsp;Hz)
&rarr; label bei t+{HORIZON} ({HORIZON*200}&nbsp;ms Vorhersagehorizont).<br>
Features ({len(FEATURES)}): {", ".join(FEATURES)}.<br>
Fraction={FRACTION} &middot; Stride={STRIDE} &middot; Epochen={EPOCHS}
&middot; Batch={BATCH_SIZE} &middot; Num-Workers={NUM_WORKERS} &middot; LR={LEARNING_RATE}
(Scheduler=ReduceLROnPlateau, Faktor={LR_SCHEDULER_FACTOR}, Patience={LR_SCHEDULER_PATIENCE},
Min-LR={LR_SCHEDULER_MIN_LR}).<br>
Loss-Setup: SmoothL1(beta={SMOOTH_L1_BETA}) &middot; Optimizer=Adam.<br>
Initialisierung: {WEIGHT_INITIALIZATION} fuer alle LSTM-/Linear-Gewichte &middot;
Forget-Gate-Bias={FORGET_GATE_BIAS}.<br>
Regularisierung: {reg}.<br>
Modellauswahl/Early-Stopping: minimaler Val-Loss (MSE), berechnet auf den
{'EMA-' if EMA_ENABLED else ''}Gewichten.<br>
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
Persistenz nutzt das letzte wahre Label als Vorhersage (Referenz).
Bestes Modell = niedrigster Val-Loss ({meta['best_val_loss']:.6f}, Epoche
{meta['best_epoch']}); letzte Epoche = Zustand am Trainingsende
(Val-Loss {meta['last_val_loss']:.6f}).</p>

<h2>Lernkurven</h2>
<img src="data:image/png;base64,{curves_b64[0]}"/>
<img src="data:image/png;base64,{curves_b64[1]}"/>
<p class="note">Waehrend des Trainings werden nur Train- und Val-Loss berechnet;
die uebrigen Metriken oben stammen aus der abschliessenden Auswertung des besten
Modells (Epoche {meta['best_epoch']}, Val-Loss {meta['best_val_loss']:.6f}) und
des Modells der letzten Epoche ({meta['last_epoch']}).</p>

<h2>Richtungs-Konfusionsmatrix (bremsen / halten / beschleunigen)</h2>
<p class="note">Zeile = reale Klasse, Spalte = vorhergesagte Klasse. Prozentwerte
sind je realer Klasse (Zeile) normiert und summieren sich pro Zeile zu 100&nbsp;%;
in Klammern die absolute Fensterzahl. Klassifikation via Deadband &plusmn;{DIR_DEADBAND}.</p>
<img src="data:image/png;base64,{confusion_b64['val']}"/>
<img src="data:image/png;base64,{confusion_b64['test']}"/>

<p class="note">Erzeugt am {meta['timestamp']}.</p>
</body></html>"""


# ---------------------------------------------------------------------------
# Ausgaben speichern
# ---------------------------------------------------------------------------
def save_outputs(hist, val_metrics, test_metrics, baseline_val, baseline_test, meta,
                 preds: dict, val_metrics_last: dict, test_metrics_last: dict):
    REPORTS_DIR.mkdir(exist_ok=True)

    curves_b64 = plot_curves(hist, REPORTS_DIR)

    confusion_b64: dict[str, str] = {}
    confusion_counts: dict[str, list] = {}
    for split, title in (("val", "Validierung"), ("test", "Test")):
        b64, cm = plot_direction_confusion(
            preds[split]["y_true"], preds[split]["y_pred"],
            f"Richtungs-Konfusionsmatrix ({title})",
            REPORTS_DIR / CONFUSION_PNG_NAMES[split],
        )
        confusion_b64[split] = b64
        confusion_counts[split] = cm.tolist()

    html = build_report(curves_b64, val_metrics, test_metrics,
                        baseline_val, baseline_test, meta, confusion_b64,
                        val_metrics_last, test_metrics_last)
    (REPORTS_DIR / REPORT_NAME).write_text(html, encoding="utf-8")

    rows = [
        {"split": "val_best", **val_metrics},
        {"split": "test_best", **test_metrics},
        {"split": "val_last_epoch", **val_metrics_last},
        {"split": "test_last_epoch", **test_metrics_last},
        {"split": "val_persistence", **baseline_val},
        {"split": "test_persistence", **baseline_test},
    ]
    pd.DataFrame(rows).to_csv(REPORTS_DIR / METRICS_CSV_NAME, index=False)
    (REPORTS_DIR / METRICS_JSON_NAME).write_text(
        json.dumps({
            "config": meta,
            "metrics": rows,
            "direction_confusion": {
                "class_names": DIRECTION_CLASS_NAMES,
                "counts_row_real_col_pred": confusion_counts,
            },
        }, indent=2), encoding="utf-8")

    print(f"\nReport : {REPORTS_DIR / REPORT_NAME}")
    for name in CURVES_PNG_NAMES:
        print(f"Plot   : {REPORTS_DIR / name}")
    for name in CONFUSION_PNG_NAMES.values():
        print(f"Konfusion: {REPORTS_DIR / name}")
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

    model = LSTMRegressor(len(FEATURES), LSTM_ARCHITECTURE, DROPOUT).to(device)

    hist, train_time_s, best_epoch, best_val_loss, last_state, last_epoch = train(
        model, cfg, train_loader, val_loader, device
    )

    # Finale Metriken auf Val und Test mit dem besten Modell.
    _, yt_val, yp_val, yprev_val = evaluate(model, val_loader, device)
    _, yt_test, yp_test, yprev_test = evaluate(model, test_loader, device)
    val_metrics = regression_metrics(yt_val, yp_val)
    test_metrics = regression_metrics(yt_test, yp_test)
    baseline_val = regression_metrics(yt_val, yprev_val)
    baseline_test = regression_metrics(yt_test, yprev_test)

    # Zusaetzlich die Metriken des Modells der letzten Epoche (ohne Best-Auswahl).
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(last_state)
    _, yt_val_last, yp_val_last, _ = evaluate(model, val_loader, device)
    _, yt_test_last, yp_test_last, _ = evaluate(model, test_loader, device)
    val_metrics_last = regression_metrics(yt_val_last, yp_val_last)
    test_metrics_last = regression_metrics(yt_test_last, yp_test_last)
    model.load_state_dict(best_state)

    meta.update({
        "n_params": count_params(model),
        "train_time_s": train_time_s,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "last_epoch": last_epoch,
        "last_val_loss": hist.val_loss[-1] if hist.val_loss else float("nan"),
        "dropout": DROPOUT,
        "device": str(device),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_time_s": time.time() - t_start,
    })

    print("\n" + "=" * 60)
    print("FINALE ERGEBNISSE")
    print(f"  Bestes Modell: Epoche {best_epoch}, Val-Loss={best_val_loss:.6f}")
    print(f"  Val : R2={val_metrics['r2']:.4f}  RMSE={val_metrics['rmse']:.4f}  "
          f"dir_acc={val_metrics['dir_acc_pct']:.2f}%  tol_acc={val_metrics['tol_acc_pct']:.2f}%")
    print(f"  Test: R2={test_metrics['r2']:.4f}  RMSE={test_metrics['rmse']:.4f}  "
          f"dir_acc={test_metrics['dir_acc_pct']:.2f}%  tol_acc={test_metrics['tol_acc_pct']:.2f}%")
    print(f"  Letzte Epoche {last_epoch}:")
    print(f"  Val : R2={val_metrics_last['r2']:.4f}  RMSE={val_metrics_last['rmse']:.4f}  "
          f"dir_acc={val_metrics_last['dir_acc_pct']:.2f}%  "
          f"tol_acc={val_metrics_last['tol_acc_pct']:.2f}%")
    print(f"  Test: R2={test_metrics_last['r2']:.4f}  RMSE={test_metrics_last['rmse']:.4f}  "
          f"dir_acc={test_metrics_last['dir_acc_pct']:.2f}%  "
          f"tol_acc={test_metrics_last['tol_acc_pct']:.2f}%")

    # Richtungs-Konfusionsmatrix (real vs. predicted) in der Konsole.
    for split, yt, yp in (("Val", yt_val, yp_val), ("Test", yt_test, yp_test)):
        cm = direction_confusion(yt, yp)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_pct = np.divide(cm * 100.0, row_sums, out=np.zeros_like(cm, dtype=float),
                           where=row_sums > 0)
        print(f"\nRichtungs-Konfusionsmatrix ({split}) - Zeile=real, Spalte=predicted:")
        header = "real\\pred"
        print(f"  {header:>14s} " + " ".join(f"{n:>14s}" for n in DIRECTION_CLASS_NAMES))
        for i, name in enumerate(DIRECTION_CLASS_NAMES):
            cells = " ".join(f"{cm_pct[i, j]:6.1f}% (n={cm[i, j]:>6,})" for j in range(3))
            print(f"  {name:>14s} {cells}")

    preds = {
        "val": {"y_true": yt_val, "y_pred": yp_val},
        "test": {"y_true": yt_test, "y_pred": yp_test},
    }
    save_outputs(hist, val_metrics, test_metrics, baseline_val, baseline_test, meta, preds,
                 val_metrics_last, test_metrics_last)


if __name__ == "__main__":
    main()
