"""
tacco_plot_fig1_v2.py

Generate summary heatmap figure from TACCO spatial predictions.

Layout matches target: one sub-heatmap per cell type (columns = stages,
rows = annotation zones), all sharing a single colorbar. A legend box on
the right shows total adata cells per stage.

Usage
-----
python tacco/tacco_plot_fig1_v2.py
"""
import re
import sys
from pathlib import Path

import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OUTPUT_DIR, SPATIAL_FILES

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["pdf.use14corefonts"] = False
matplotlib.rcParams["path.simplify"] = False  # keep all vector path points exact

# ── Cell type lists ────────────────────────────────────────────────────────────
# These strings must exactly match what appears in tacco_predictions.csv
# EPI_MESEN_BASAL_CELLTYPES = [
#     "Mesen_EndoGlandBas",
#     "Epi_EndoGlandBas",
#     "Epi_EndoMucinous",
#     "Immune_B",
# ]

EPI_MESEN_BASAL_CELLTYPES = [
    "Epi_EndoGlandBas",
    "Mesen_EndoGlandBas",
    "Immune_B",    
]

# Display names shown as column headers in the figure (mapped from data strings)
CT_DISPLAY_NAMES = {
    "Mesen_EndoGlandBas": "Mesen_GlandBas",
    "Epi_EndoGlandBas":   "Epi_GlandBas",
    "Epi_EndoMucinous":   "Epi_EndoMucinous",
    "Immune_B":           "Immune_B",
}

# ── Stage abbreviations ───────────────────────────────────────────────────────
STAGE_ABBREV = {
    "Proliferative": "Pro",
    "Secretory":     "Sec",
    "Menstrual":     "Men",
    "Hormones":      "ExHor",
}

# ── Constants ─────────────────────────────────────────────────────────────────
ANNOT_KEY        = "annotation"
ANNOT_CATEGORIES = ["basalis", "functionalis", "lumen"]

TACCO_DIR        = OUTPUT_DIR / "tacco" / "all_sc" / "full"
TACCO_THRESHOLD  = 0.4
MIN_CELLS_PER_CT = 10

STAGE_ORDER_ALL = ["Proliferative", "Secretory", "Menstrual", "Hormones"]

OUT_DIR = OUTPUT_DIR / "tacco"


# ── TACCO predictions ─────────────────────────────────────────────────────────

def _load_tacco(stage: str, donor: str) -> pd.Series:
    """Series(index=cell_id, values=celltype); NaN for low-confidence cells."""
    d      = TACCO_DIR / stage / donor
    csv    = d / "tacco_predictions.csv"
    scores = d / "tacco_scores.parquet"

    if not csv.exists():
        return pd.Series(dtype=str)

    preds = pd.read_csv(csv).set_index("cell_id")["celltype"].copy()
    if scores.exists():
        sc_df = pq.read_table(scores).to_pandas(ignore_metadata=True).set_index("cell_id")
        low   = sc_df.max(axis=1) <= TACCO_THRESHOLD
        preds[low.reindex(preds.index, fill_value=False)] = np.nan

    return preds


# ── Per-stage data collection ─────────────────────────────────────────────────

def collect_stage_data(
    stage: str, celltypes: list[str]
) -> dict | None:
    """
    Returns dict with:
      - frac:    DataFrame (celltype × annotation) — mean fraction across donors
      - ct_counts: Series (celltype → n cells in relevant cell types)
      - n_donors: int
      - total_cells: int (ALL cells in adata across donors, not just relevant CTs)
    """
    frames:       list[pd.DataFrame] = []
    count_frames: list[pd.Series]    = []
    total_adata_cells = 0
    donor_set: set[str] = set()

    for sample_id, sp_path_str in SPATIAL_FILES.items():
        if re.sub(r"_\d+$", "", sample_id) != stage:
            continue
        try:
            sp    = ad.read_h5ad(Path(sp_path_str), backed="r")
            donor = str(sp.obs["sample"].iloc[0])

            # Count ALL cells in this adata (total, regardless of annotation/ct)
            total_adata_cells += sp.n_obs
            donor_set.add(donor)

            obs = sp.obs[[ANNOT_KEY]].copy()
            obs = obs[obs[ANNOT_KEY].isin(ANNOT_CATEGORIES)].dropna(subset=[ANNOT_KEY])
            if obs.empty:
                continue

            preds = _load_tacco(stage, donor)
            if preds.dropna().empty:
                continue

            in_set = preds[preds.isin(celltypes)]
            if in_set.empty:
                continue

            joined = (
                in_set.to_frame("celltype")
                .join(obs[[ANNOT_KEY]], how="inner")
                .dropna()
            )
            if joined.empty:
                continue

            counts = joined["celltype"].value_counts()
            keep   = counts[counts >= MIN_CELLS_PER_CT].index
            joined = joined[joined["celltype"].isin(keep)]
            if joined.empty:
                continue

            frac = pd.crosstab(joined["celltype"], joined[ANNOT_KEY], normalize="index")
            frac = frac.reindex(columns=ANNOT_CATEGORIES, fill_value=0.0)
            frames.append(frac)
            count_frames.append(joined.groupby("celltype").size())
            print(f"    {donor}: {len(frac)} cell types, {len(joined):,} cells", flush=True)
        except Exception as e:
            print(f"    [warn] {sample_id}: {e}", flush=True)

    if not frames:
        print(f"  No data for stage '{stage}' — skipping", flush=True)
        return None

    combined  = pd.concat(frames)
    mean_frac = combined.groupby(combined.index).mean()
    mean_frac = mean_frac.reindex([ct for ct in celltypes if ct in mean_frac.index])

    combined_counts = pd.concat(count_frames).groupby(level=0).sum()
    combined_counts = combined_counts.reindex(mean_frac.index, fill_value=0)

    return {
        "frac":        mean_frac,
        "ct_counts":   combined_counts,
        "n_donors":    len(donor_set),
        "total_cells": total_adata_cells,
    }


# ── Main figure ───────────────────────────────────────────────────────────────

def plot_epi_mesen_basal(
    celltypes: list[str],
    stages: list[str],
    out_path: Path,
) -> None:
    """
    One heatmap per cell type (rows = annotation, cols = stages).
    Single shared colorbar fixed 0–1. Legend box on right with total cells/stage.
    """
    # ── Collect data ──────────────────────────────────────────────────────────
    stage_data: dict[str, dict] = {}
    for stage in stages:
        print(f"\n── {stage} ──", flush=True)
        result = collect_stage_data(stage, celltypes)
        if result is not None:
            stage_data[stage] = result

    abbrevs    = [STAGE_ABBREV.get(s, s) for s in stages]
    n_ct       = len(celltypes)
    n_stage    = len(stages)
    annot_rows = ["lumen", "functionalis", "basalis"]

    # ── Build per-cell-type heatmap matrices ──────────────────────────────────
    # Shape for each CT: (3 annot rows × n_stage cols)
    ct_matrices: dict[str, pd.DataFrame] = {}
    ct_stage_counts: dict[str, dict[str, int]] = {}  # ct → {stage_abbrev → n}

    for ct in celltypes:
        mat = pd.DataFrame(
            np.nan,
            index=annot_rows,
            columns=abbrevs,
        )
        counts: dict[str, int] = {}
        for stage, abbrev in zip(stages, abbrevs):
            if stage not in stage_data:
                counts[abbrev] = 0
                continue
            sd = stage_data[stage]
            if ct in sd["frac"].index:
                row = sd["frac"].loc[ct]
                for ar in annot_rows:
                    mat.loc[ar, abbrev] = row.get(ar, np.nan)
                counts[abbrev] = int(sd["ct_counts"].get(ct, 0))
            else:
                counts[abbrev] = 0
        ct_matrices[ct] = mat
        ct_stage_counts[ct] = counts

    # ── Figure layout ─────────────────────────────────────────────────────────
    # Columns: n_ct heatmaps + 1 narrow colorbar + 1 legend text box
    panel_w   = 1.0 + n_stage * 0.35   # width per heatmap panel
    cb_w      = 0.25                    # colorbar width
    legend_w  = 2.2                     # legend text box width
    fig_w     = n_ct * panel_w + cb_w + legend_w + 0.4
    fig_h     = 2.6

    fig = plt.figure(figsize=(fig_w, fig_h))

    # GridSpec: n_ct panels | 1 cbar col | 1 legend col
    gs = gridspec.GridSpec(
        1, n_ct + 2,
        width_ratios=[panel_w] * n_ct + [cb_w, legend_w],
        wspace=0.08,
        left=0.08, right=0.98, top=0.82, bottom=0.28,
    )

    axes     = [fig.add_subplot(gs[0, i]) for i in range(n_ct)]
    cbar_ax  = fig.add_subplot(gs[0, n_ct])
    leg_ax   = fig.add_subplot(gs[0, n_ct + 1])

    cmap  = plt.cm.Reds
    vmin, vmax = 0.0, 1.0
    norm  = Normalize(vmin=vmin, vmax=vmax)

    # ── Draw heatmaps ─────────────────────────────────────────────────────────
    for idx, ct in enumerate(celltypes):
        ax  = axes[idx]
        mat = ct_matrices[ct]

        # Draw heatmap as vector rectangles (one patch per cell) so that
        # Illustrator sees individual filled paths rather than a rasterized image.
        # imshow embeds a bitmap in the PDF which Illustrator resamples/flattens.
        data = mat.values.astype(float)
        n_rows, n_cols = data.shape   # 3 annotation rows × n_stage cols

        from matplotlib.patches import Rectangle
        for ri in range(n_rows):
            for ci in range(n_cols):
                val = data[ri, ci]
                color = cmap(norm(val)) if not np.isnan(val) else (1, 1, 1, 1)
                rect = Rectangle(
                    (ci - 0.5, ri - 0.5), 1, 1,
                    facecolor=color, edgecolor="white", linewidth=0.8,
                )
                ax.add_patch(rect)

        ax.set_xlim(-0.5, n_cols - 0.5)
        ax.set_ylim(n_rows - 0.5, -0.5)  # invert y so row 0 (lumen) is at top
        ax.set_aspect("auto")
        ax.tick_params(which="minor", bottom=False, left=False)

        # Y-axis: annotation labels only on leftmost panel
        ax.set_yticks(range(3))
        if idx == 0:
            ax.set_yticklabels(annot_rows, fontsize=8.5)
            ax.set_ylabel("", fontsize=9)
        else:
            ax.set_yticklabels([])
            ax.tick_params(left=False)

        # X-axis: hide default ticks; draw custom labels with coloured bbox below
        ax.set_xticks(range(n_stage))
        ax.set_xticklabels([], fontsize=0)   # hide default labels
        ax.tick_params(axis="x", length=0)

        STAGE_COLORS = {
            "Pro":   "#d7e8d5",
            "Sec":   "#ebf5da",
            "Men":   "#ffe8d6",
            "ExHor": "#ddfbf9",
        }

        counts_here = ct_stage_counts[ct]
        for col_i, abbrev in enumerate(abbrevs):
            n   = counts_here.get(abbrev, 0)
            col = STAGE_COLORS.get(abbrev, "#cccccc")
            x_ax = (col_i + 0.5) / n_stage
            ax.text(
                x_ax, -0.08,
                abbrev,
                transform=ax.transAxes,
                ha="center", va="top", fontsize=7.5, color="black",
                bbox=dict(
                    facecolor=col, edgecolor="none",
                    boxstyle="round,pad=0.15", alpha=0.85,
                ),
            )
            ax.text(
                x_ax, -0.22,
                f"n={n:,}",
                transform=ax.transAxes,
                ha="center", va="top", fontsize=7, color="black",
            )

        # Column header (cell type name)
        display_name = ct
        if CT_DISPLAY_NAMES and ct in CT_DISPLAY_NAMES:
            display_name = CT_DISPLAY_NAMES[ct]
        ax.set_title(display_name, fontsize=9, fontweight="bold", pad=5)

        # Border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.6)

    # ── Colorbar ──────────────────────────────────────────────────────────────
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("Fraction", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    cb.set_ticks([0.0, 0.5, 1.0])

    # ── Legend box (total adata cells per stage) ──────────────────────────────
    leg_ax.axis("off")

    # Build legend text
    legend_lines = []
    stage_color_map = {
        "Proliferative": "#d7e8d5",
        "Secretory":     "#ebf5da",
        "Menstrual":     "#ffe8d6",
        "Hormones":      "#ddfbf9",
    }
    for stage in stages:
        abbrev = STAGE_ABBREV.get(stage, stage)
        if stage in stage_data:
            sd      = stage_data[stage]
            n_don   = sd["n_donors"]
            n_cells = sd["total_cells"]
            legend_lines.append((abbrev, n_don, n_cells, stage_color_map.get(stage, "black")))
        else:
            legend_lines.append((abbrev, 0, 0, stage_color_map.get(stage, "black")))

    # Draw a light-green rounded outer box
    from matplotlib.patches import FancyBboxPatch
    box = FancyBboxPatch(
        (0.02, 0.05), 0.96, 0.92,
        boxstyle="round,pad=0.05",
        facecolor="#e8f5e9", edgecolor="#a5d6a7", linewidth=1.0,
        transform=leg_ax.transAxes, zorder=0,
    )
    leg_ax.add_patch(box)

    n_lines = len(legend_lines)
    for i, (abbrev, n_don, n_cells, color) in enumerate(legend_lines):
        y = 0.80 - i * (0.68 / max(n_lines - 1, 1))
        # Abbrev badge with coloured background
        leg_ax.text(
            0.10, y,
            abbrev,
            transform=leg_ax.transAxes,
            fontsize=8, va="center", color="black", fontweight="bold",
            bbox=dict(
                facecolor=color, edgecolor="none",
                boxstyle="round,pad=0.18", alpha=0.85,
            ),
        )
        # Donor/cell count in plain black text
        leg_ax.text(
            0.30, y,
            f"{n_don} donors, {n_cells:,} cells",
            transform=leg_ax.transAxes,
            fontsize=8, va="center", color="black",
        )

    # ── Save ──────────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved -> {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "━" * 60, flush=True)
    print("Figure 1: epi_mesen_basal_reds_v2.pdf", flush=True)
    plot_epi_mesen_basal(
        celltypes=EPI_MESEN_BASAL_CELLTYPES,
        stages=STAGE_ORDER_ALL,
        out_path=OUT_DIR / "epi_mesen_basal_reds_v2.pdf",
    )


if __name__ == "__main__":
    main()