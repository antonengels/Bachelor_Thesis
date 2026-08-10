"""Heatmap: Test-MSE je Optimizer x Loss-Funktion (Stil wie HTML-Report)."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
JSON_PATH = ROOT / "reports" / "gru_loss_optimizer_metrics.json"
OUT_PATH  = ROOT / "reports" / "heatmap_test_mse.png"

records = json.loads(JSON_PATH.read_text(encoding="utf-8"))

df = pd.DataFrame(records)
test_df = df[(df["split"] == "test") & (df["optimizer"] != "-")].copy()

opts   = sorted(test_df["optimizer"].unique())
losses = sorted(test_df["loss"].unique())

mat = np.full((len(opts), len(losses)), np.nan, dtype=float)
pos_opt  = {o: i for i, o in enumerate(opts)}
pos_loss = {l: j for j, l in enumerate(losses)}
for _, row in test_df.iterrows():
    mat[pos_opt[row["optimizer"]], pos_loss[row["loss"]]] = row["mse"]

fig, ax = plt.subplots(figsize=(1.8 * len(losses) + 1.5, 0.8 * len(opts) + 2.2))
im = ax.imshow(mat, aspect="auto", cmap="viridis_r")
ax.set_xticks(np.arange(len(losses)))
ax.set_yticks(np.arange(len(opts)))
ax.set_xticklabels(losses, rotation=30, ha="right")
ax.set_yticklabels(opts)
ax.set_title("Test-MSE Heatmap (Optimizer x Loss)")

for i in range(len(opts)):
    for j in range(len(losses)):
        if np.isfinite(mat[i, j]):
            ax.text(j, i, f"{mat[i, j]:.4f}", ha="center", va="center",
                    color="white", fontsize=8)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Test-MSE")
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Gespeichert: {OUT_PATH}")
