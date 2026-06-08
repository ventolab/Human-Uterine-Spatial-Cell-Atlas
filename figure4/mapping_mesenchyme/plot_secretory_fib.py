"""
Plot Mesen_EndoStromalFib secretory subtypes (eSec / mSec / lSec) for donor A30
using TACCO cell-type predictions.

Three side-by-side spatial panels — one per secretory subtype — with target
cells in a shaded blue and all other cells as light-grey background spots.
"""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["pdf.fonttype"] = 42

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE = Path("/lustre/scratch125/cellgen/vento/mm58/eutopic_endometrium")

TACCO_PATH = (
    BASE
    / "benchmark_knn_vs_dot/outputs/tacco/all_sc/full/Secretory/A30/tacco_predictions.csv"
)

OUT_PATH = Path(__file__).parent / "figures" / "secretory_fib_A30.pdf"

# ── Cell types & colours ──────────────────────────────────────────────────────

CELLTYPES = [
    "Mesen_EndoStromalFib_eSec",
    "Mesen_EndoStromalFib_mSec",
    "Mesen_EndoStromalFib_lSec",
]

# Blue shades from lighter → darker matching eSec → mSec → lSec progression
COLORS = [
    plt.cm.Blues(0.50),   # eSec  — medium-light blue
    plt.cm.Blues(0.70),   # mSec  — medium-dark blue
    plt.cm.Blues(0.90),   # lSec  — dark blue
]

BG_COLOR = "#DDDDDD"
BG_SIZE  = 0.3
FG_SIZE  = 1.5
DPI      = 200

# ── Load TACCO predictions ────────────────────────────────────────────────────

print("Loading TACCO predictions …", flush=True)
tacco = pd.read_csv(TACCO_PATH)

xy        = tacco[["spatial.1", "spatial.2"]].values
celltypes = tacco["celltype"].values

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, ct, color in zip(axes, CELLTYPES, COLORS):
    mask = celltypes == ct
    n_fg = mask.sum()

    ax.scatter(
        xy[~mask, 0], xy[~mask, 1],
        s=BG_SIZE, c=BG_COLOR, rasterized=True, linewidths=0,
    )
    ax.scatter(
        xy[mask, 0], xy[mask, 1],
        s=FG_SIZE, c=[color], rasterized=True, linewidths=0,
        label=ct,
    )

    label = ct.replace("Mesen_EndoStromalFib_", "")
    ax.set_title(f"{label}  (n={n_fg:,})", fontsize=9)
    ax.axis("off")
    ax.set_aspect("equal")

fig.suptitle(
    "A30 (Secretory) — Endometrial stromal fibroblast secretory subtypes\nTACCO predictions",
    fontsize=10, y=1.01,
)
fig.tight_layout()

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {OUT_PATH}")
