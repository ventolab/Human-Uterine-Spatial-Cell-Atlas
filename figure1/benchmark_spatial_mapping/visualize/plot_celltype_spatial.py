"""
Spatial grid plots of predicted cell types for both ISS-Patcher and TACCO.

One panel per cell type: background cells grey, foreground cells coloured.
Output saved as celltype_spatial_grid.pdf next to each input file.

Usage:
  python plot_celltype_spatial.py                        # all results (both methods)
  python plot_celltype_spatial.py path/to/iss_patched.h5ad
"""
import sys
import math
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OUTPUT_DIR

NCOLS    = 6
BG_COLOR = "#DDDDDD"
BG_SIZE  = 0.1
FG_SIZE  = 0.5
DPI      = 150

# ISS-Patcher: only keep high-confidence predictions
ISS_FRACTION_THRESHOLD = 0.5


def _make_palette(celltypes: list[str]) -> dict[str, np.ndarray]:
    cmap = plt.cm.get_cmap("tab20", 20)
    return {ct: cmap(i % 20) for i, ct in enumerate(sorted(celltypes))}


def _plot_grid(xy: np.ndarray, cts: np.ndarray, title: str, out_path: Path) -> None:
    unique_cts = sorted(set(cts))
    palette    = _make_palette(unique_cts)
    n          = len(unique_cts)
    nrows      = math.ceil(n / NCOLS)

    fig, axes = plt.subplots(
        nrows, NCOLS,
        figsize=(3 * NCOLS, 2.5 * nrows),
        squeeze=False,
    )

    for idx, ct in enumerate(unique_cts):
        ax   = axes[idx // NCOLS][idx % NCOLS]
        mask = cts == ct
        ax.scatter(xy[~mask, 0], xy[~mask, 1],
                   s=BG_SIZE, c=BG_COLOR, rasterized=True)
        ax.scatter(xy[mask, 0],  xy[mask, 1],
                   s=FG_SIZE, c=[palette[ct]], rasterized=True)
        ax.set_title(ct, fontsize=6, pad=2)
        ax.axis("off")

    for idx in range(n, nrows * NCOLS):
        axes[idx // NCOLS][idx % NCOLS].axis("off")

    fig.suptitle(title, fontsize=10, y=1.002)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def _title_from_path(path: Path, method_dir: str) -> str:
    parts = path.parts
    try:
        idx = parts.index(method_dir)
        return " / ".join((method_dir,) + parts[idx + 1: idx + 5])
    except ValueError:
        return str(path.parent)


# ── ISS-Patcher ──────────────────────────────────────────────────────────────

def plot_iss(h5ad_path: Path) -> None:
    out_path = h5ad_path.parent / "celltype_spatial_grid.pdf"
    print(f"  [ISS] Loading {h5ad_path.name} ...", flush=True)

    adata = ad.read_h5ad(h5ad_path, backed="r")
    print(f"    Loaded adata with obs columns: {adata.obs.columns.tolist()}")
    adata = adata[adata.obs["fine_celltype_fraction"] > ISS_FRACTION_THRESHOLD]
    xy  = adata.obsm["spatial"]
    cts = adata.obs["fine_celltype"].astype(str).values

    _plot_grid(xy, cts, _title_from_path(h5ad_path, "iss_patcher"), out_path)


# ── TACCO ─────────────────────────────────────────────────────────────────────

TACCO_SCORE_THRESHOLD = 0.4

def plot_tacco(csv_path: Path) -> None:
    out_path = csv_path.parent / "celltype_spatial_grid.pdf"
    print(f"  [TACCO] Loading {csv_path.name} ...", flush=True)

    df = pd.read_csv(csv_path)

    # Filter to high-confidence predictions using the scores parquet
    scores_path = csv_path.parent / "tacco_scores.parquet"
    if scores_path.exists():
        scores = pq.read_table(scores_path).to_pandas(ignore_metadata=True).set_index("cell_id")
        max_scores = scores.max(axis=1)
        high_conf = max_scores[max_scores > TACCO_SCORE_THRESHOLD].index
        df = df[df["cell_id"].isin(high_conf)]
        print(f"    High-confidence cells (score>{TACCO_SCORE_THRESHOLD}): "
              f"{len(df):,} / {len(max_scores):,}", flush=True)

    xy  = df[["spatial.1", "spatial.2"]].values
    cts = df["celltype"].astype(str).values

    _plot_grid(xy, cts, _title_from_path(csv_path, "tacco"), out_path)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if p.suffix == ".h5ad":
            plot_iss(p)
        elif p.name == "tacco_predictions.csv":
            plot_tacco(p)
        return

    iss_paths   = sorted(OUTPUT_DIR.glob("iss_patcher/*/*/*/*/iss_patched.h5ad"))
    tacco_paths = sorted(OUTPUT_DIR.glob("tacco/*/*/*/*/tacco_predictions.csv"))
    
    print(f"Found {len(iss_paths)} ISS-Patcher, and "
          f"{len(tacco_paths)} TACCO result directories")

    for p in iss_paths:
        print(f"\n{p.parent.relative_to(OUTPUT_DIR)}")
        plot_iss(p)

    for p in tacco_paths:
        print(f"\n{p.parent.relative_to(OUTPUT_DIR)}")
        plot_tacco(p)


if __name__ == "__main__":
    main()
