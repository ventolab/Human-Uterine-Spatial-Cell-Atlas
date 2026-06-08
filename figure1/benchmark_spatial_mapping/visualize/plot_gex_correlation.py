"""
compute_gex_correlation.py

Two-panel summary figure from outputs/gex_correlation/summary.csv.

  Left  Mean GEX correlation (mean_corr_matched) per method, value annotated.
  Right Number of cell types recovered (n_celltypes) per method, count annotated.

Usage
-----
python visualize/compute_gex_correlation.py
"""
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.rcParams["pdf.fonttype"] = 42

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OUTPUT_DIR

GEX_DIR  = OUTPUT_DIR / "gex_correlation"
OUT_PATH = GEX_DIR / "gex_correlation_overview.pdf"

# ── Aesthetics ────────────────────────────────────────────────────────────────
METHOD_ORDER = ["tacco", "iss_full", "iss_ds1k"]

METHOD_LABELS = {
    "iss_full": "ISS full",
    "iss_ds1k": "ISS 1k",
    "tacco":    "TACCO",
}

METHOD_COLORS = {
    "iss_full": "lightblue",
    "iss_ds1k": "lightgray",
    "tacco":    "darkred",
}

FONT_TICK  = 10
FONT_LABEL = 13
FONT_TITLE = 14
FONT_ANNOT = 9


def main() -> None:
    df = pd.read_csv(GEX_DIR / "summary.csv")
    present = [m for m in METHOD_ORDER if m in df["method"].values]
    df = df[df["method"].isin(present)].set_index("method").reindex(present)

    labels = [METHOD_LABELS[m] for m in present]
    colors = [METHOD_COLORS[m] for m in present]
    y      = range(len(present))

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(9, 3.5))
    fig.subplots_adjust(wspace=0.45)

    # ── Left: mean correlation ────────────────────────────────────────────────
    vals_corr = df["mean_corr_matched"].values
    bars = ax_left.barh(list(y), vals_corr, color=colors, height=0.55)
    ax_left.set_yticks(list(y))
    ax_left.set_yticklabels(labels, fontsize=FONT_TICK)
    ax_left.set_xlabel("Mean Pearson r", fontsize=FONT_LABEL)
    ax_left.set_title("Mean GEX correlation", fontsize=FONT_TITLE)
    ax_left.tick_params(axis="x", labelsize=FONT_TICK)
    ax_left.spines[["top", "right"]].set_visible(False)
    ax_left.invert_yaxis()

    x_max = max(vals_corr) * 1.15
    ax_left.set_xlim(0, x_max)
    for bar, v in zip(bars, vals_corr):
        ax_left.text(
            bar.get_width() + x_max * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{round(v, 2):.2f}",
            va="center", ha="left", fontsize=FONT_ANNOT,
        )

    # ── Right: n_celltypes ────────────────────────────────────────────────────
    vals_n = df["n_celltypes"].values
    bars2 = ax_right.barh(list(y), vals_n, color=colors, height=0.55)
    ax_right.set_yticks(list(y))
    ax_right.set_yticklabels(labels, fontsize=FONT_TICK)
    ax_right.set_xlabel("Number of cell types", fontsize=FONT_LABEL)
    ax_right.set_title("Cell types recovered", fontsize=FONT_TITLE)
    ax_right.tick_params(axis="x", labelsize=FONT_TICK)
    ax_right.spines[["top", "right"]].set_visible(False)
    ax_right.invert_yaxis()

    x_max2 = max(vals_n) * 1.15
    ax_right.set_xlim(0, x_max2)
    for bar, v in zip(bars2, vals_n):
        ax_right.text(
            bar.get_width() + x_max2 * 0.02,
            bar.get_y() + bar.get_height() / 2,
            str(int(v)),
            va="center", ha="left", fontsize=FONT_ANNOT,
        )

    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
