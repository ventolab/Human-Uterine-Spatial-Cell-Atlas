"""
Plot Mesen_MyoFib predicted locations and SFRP5 expression for donor A66.

Produces a 4-panel figure:
  - ISS-Patcher full
  - ISS-Patcher downsampled (1k)
  - TACCO full
  - SFRP5 log-normalised expression

Output: sfrp5_mesen_myofib_A66.pdf  (next to this script)
"""
from pathlib import Path

import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp

matplotlib.rcParams["pdf.fonttype"] = 42

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE = Path("/lustre/scratch125/cellgen/vento/mm58/eutopic_endometrium/benchmark_knn_vs_dot/outputs")

ISS_FULL_PATH  = BASE / "iss_patcher/all_sc/full/Proliferative/A66/iss_annots.parquet"
ISS_DS_PATH    = BASE / "iss_patcher/all_sc/downsampled_1k/Proliferative/A66/iss_annots.parquet"
TACCO_PATH     = BASE / "tacco/all_sc/full/Proliferative/A66/tacco_predictions.csv"
# H5AD_PATH      = Path("/nfs/team292/vl6/Endometriosis/Xenium/A13-UTR-0-TL4-1-S50/A13_annotated_new_axis.h5ad")
H5AD_PATH      = Path("/nfs/team292/vl6/Endometriosis/Xenium/A66-RPT-8-FO-1-S40/A66_annotated_new_axis.h5ad")

OUT_PATH       = Path(__file__).parent / "sfrp5_mesen_myofib_A66.pdf"

TARGET_CT      = "Mesen_EndoGlandBas"
ISS_THRESHOLD  = 0.5

BG_COLOR = "#DDDDDD"
FG_COLOR = "#D62728"
BG_SIZE  = 0.3
FG_SIZE  = 1.5
DPI      = 200

# ── Load reference h5ad ───────────────────────────────────────────────────────

print("Loading h5ad …", flush=True)
adata = ad.read_h5ad(H5AD_PATH, backed="r")

xy_all = adata.obsm["spatial"]                        # (N, 2)  x / y
barcodes = adata.obs.index.values                     # cell IDs shared across files

sfrp5_idx = list(adata.var_names).index("SFRP5")
x_col = adata.X[:, sfrp5_idx]
if sp.issparse(x_col):
    sfrp5_expr = x_col.toarray().ravel()
else:
    sfrp5_expr = np.asarray(x_col).ravel()

# Fast barcode → row-index lookup
bc2idx = {bc: i for i, bc in enumerate(barcodes)}

# ── Load ISS-Patcher annotations ──────────────────────────────────────────────

def load_iss(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (xy, is_target_mask) after applying fraction threshold."""
    df = pd.read_parquet(path)
    df = df[df["fine_celltype_fraction"] > ISS_THRESHOLD]
    rows = np.array([bc2idx[bc] for bc in df["barcode"] if bc in bc2idx])
    # Re-filter to only barcodes found in h5ad
    mask_found = df["barcode"].isin(bc2idx)
    df = df[mask_found].copy()
    rows = np.array([bc2idx[bc] for bc in df["barcode"]])
    xy   = xy_all[rows]
    is_target = df["fine_celltype"].values == TARGET_CT
    return xy, is_target


print("Loading ISS-Patcher full …", flush=True)
xy_iss_full, mask_iss_full = load_iss(ISS_FULL_PATH)

print("Loading ISS-Patcher downsampled …", flush=True)
xy_iss_ds, mask_iss_ds = load_iss(ISS_DS_PATH)

# ── Load TACCO annotations ────────────────────────────────────────────────────

print("Loading TACCO …", flush=True)
tacco = pd.read_csv(TACCO_PATH)
xy_tacco    = tacco[["spatial.1", "spatial.2"]].values
mask_tacco  = tacco["celltype"].values == TARGET_CT

# ── Plot ──────────────────────────────────────────────────────────────────────

def scatter_panel(ax, xy, mask, title):
    ax.scatter(xy[~mask, 0], xy[~mask, 1],
               s=BG_SIZE, c=BG_COLOR, rasterized=True, linewidths=0)
    ax.scatter(xy[mask,  0], xy[mask,  1],
               s=FG_SIZE, c=FG_COLOR, rasterized=True, linewidths=0,
               label=TARGET_CT)
    n_fg = mask.sum()
    ax.set_title(f"{title}\n{TARGET_CT}  (n={n_fg:,})", fontsize=8)
    ax.axis("off")
    ax.set_aspect("equal")


print("Plotting …", flush=True)
fig, axes = plt.subplots(1, 4, figsize=(18, 5))

scatter_panel(axes[0], xy_iss_full, mask_iss_full, "ISS-Patcher  (full)")
scatter_panel(axes[1], xy_iss_ds,   mask_iss_ds,   "ISS-Patcher  (downsampled 1k)")
scatter_panel(axes[2], xy_tacco,    mask_tacco,     "TACCO  (full)")

# SFRP5 expression panel
ax = axes[3]
order = np.argsort(sfrp5_expr)
sc = ax.scatter(
    xy_all[order, 0], xy_all[order, 1],
    c=sfrp5_expr[order],
    s=0.3,
    cmap="Reds",
    rasterized=True,
    linewidths=0,
    vmin=0,
    vmax=np.percentile(sfrp5_expr[sfrp5_expr > 0], 99) if (sfrp5_expr > 0).any() else 1,
)
plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02, label="log-norm expr")
ax.set_title("SFRP5  (log-normalised)", fontsize=8)
ax.axis("off")
ax.set_aspect("equal")

fig.suptitle("A66 — Mesen_MyoFib predictions & SFRP5 expression", fontsize=10, y=1.01)
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {OUT_PATH}")
