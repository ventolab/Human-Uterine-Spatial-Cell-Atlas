"""Utilities for the ISS-Patcher vs DOT benchmark (ISS-Patcher side)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

sys.path.insert(0, "/lustre/scratch125/cellgen/vento/mm58/eutopic_endometrium/iss_patcher")
import iss_patcher as ip


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sc_reference(
    h5ad_path: str,
    annot_path: str,
    config: str,
    stage: str = None,
    downsample: bool = False,
    downsample_n: int = 100,
    seed: int = 42,
) -> ad.AnnData:
    """Load scRNA reference from concat_uterus_inner.h5ad + fine_celltype annotation parquet.

    Parameters
    ----------
    annot_path : parquet with barcode index and fine_celltype / lineage columns
    config : {"all_sc", "stage_matched"}
    stage : menstrual stage short label (e.g. "Proliferative"); used when config=="stage_matched"
    downsample : if True, cap at downsample_n cells per celltype
    """
    print(f"\n[SC] Loading: {Path(h5ad_path).name}  config={config}  stage={stage}  downsample={downsample}", flush=True)
    adata = sc.read(h5ad_path)
    print(f"  Initial shape: {adata.shape}", flush=True)

    # Restore raw counts
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
        print("  Restored raw counts from layers['counts']", flush=True)

    # Join fine_celltype / broad_celltype / lineage (not stored in the h5ad itself)
    ann = pd.read_csv(annot_path, index_col=0)[["fine_celltype", "broad_celltype", "lineage"]]
    adata.obs = adata.obs.join(ann, how="left")
    n_annotated = adata.obs["fine_celltype"].notna().sum()
    print(f"  Annotated cells: {n_annotated:,} / {adata.n_obs:,}", flush=True)

    # Derive short stage label from first word of Menstrual_stage
    adata.obs["Menstrual_stage_short"] = (
        adata.obs["Menstrual_stage"].astype(str).str.split(" ").str[0]
    )

    # Stage filter
    if config == "stage_matched":
        if stage is None:
            raise ValueError("stage must be provided when config=='stage_matched'")
        adata = adata[adata.obs["Menstrual_stage_short"] == stage].copy()
        print(f"  After stage filter ({stage}): {adata.shape}", flush=True)

    # Drop unwanted datasets
    datasets_to_drop = {"uterus_menopause_sanger-denoised", "uterus_adult_menstrualfluid_sanger-denoised"}
    adata = adata[~adata.obs["Dataset"].isin(datasets_to_drop)].copy()
    print(f"  After dropping datasets: {adata.shape}", flush=True)

    # Drop unannotated cells (no match in annotation parquet)
    adata = adata[adata.obs["fine_celltype"].notna()].copy()
    print(f"  After dropping unannotated: {adata.shape}", flush=True)

    # Drop low-quality cells
    if "fine_celltype" in adata.obs.columns:
        exclude = {"lowQC", "doublet", "unknown"}
        mask = ~adata.obs["fine_celltype"].isin(exclude)
        adata = adata[mask].copy()
        print(f"  After removing lowQC/doublet/unknown: {adata.shape}", flush=True)

    # Downsample
    if downsample:
        rng = np.random.default_rng(seed)
        keep = []
        for _, grp in adata.obs.groupby("fine_celltype"):
            idx = grp.index.tolist()
            if len(idx) > downsample_n:
                idx = rng.choice(idx, size=downsample_n, replace=False).tolist()
            keep.extend(idx)
        adata = adata[keep].copy()
        print(f"  After downsampling (max {downsample_n}/celltype): {adata.shape}", flush=True)

    return adata


def load_spatial(h5ad_path: str) -> ad.AnnData:
    """Load Xenium h5ad, rename axis column typo, restore raw counts."""
    print(f"\n[SPATIAL] Loading: {Path(h5ad_path).name}", flush=True)
    adata = sc.read(h5ad_path)
    print(f"  Shape: {adata.shape}", flush=True)

    # Normalise axis column to "universal_axis" (legacy names → canonical)
    for old in ["myometrial_luminal_axis", "myometrial_lumina_axis"]:
        if old in adata.obs.columns and "universal_axis" not in adata.obs.columns:
            adata.obs.rename(columns={old: "universal_axis"}, inplace=True)

    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()

    if "universal_axis" in adata.obs.columns:
        adata.obs["universal_axis"] = pd.to_numeric(
            adata.obs["universal_axis"], errors="coerce"
        )

    return adata


# ---------------------------------------------------------------------------
# Marker computation
# ---------------------------------------------------------------------------

_PANATLAS_PATH = "/nfs/team292/projects/PanTissue/code/working/lg18/utils/panatlas_utils.py"

def _load_quick_markers():
    spec = importlib.util.spec_from_file_location("panatlas_utils", _PANATLAS_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["panatlas_utils"] = mod
    spec.loader.exec_module(mod)
    return mod.quick_markers


def compute_markers(
    sc: ad.AnnData,
    shared_genes: list[str],
    n_markers: int = 20,
) -> pd.DataFrame | None:
    """Find TF-IDF markers on the shared-gene subset of the scRNA reference.

    Returns a long-format DataFrame with columns [celltype, gene], top
    n_markers per cell type, sorted by tf_idf descending.  Returns None if
    panatlas_utils is unavailable or quick_markers fails.
    """
    try:
        quick_markers = _load_quick_markers()
    except Exception as e:
        print(f"  [MARKERS] panatlas_utils unavailable: {e}", flush=True)
        return None

    avail = [g for g in shared_genes if g in sc.var_names]
    sc_shared = sc[:, avail].copy()
    sc_shared.obs["fine_celltype"] = sc_shared.obs["fine_celltype"].astype("category")
    print(
        f"  [MARKERS] Running quick_markers on "
        f"{sc_shared.n_obs} cells × {sc_shared.n_vars} shared genes...",
        flush=True,
    )
    try:
        markers = quick_markers(sc_shared, cluster_key="fine_celltype", n_markers=n_markers)
        markers = markers.rename(columns={"cluster": "celltype"})[["celltype", "gene"]]
        print(
            f"  [MARKERS] Found markers for {markers['celltype'].nunique()} cell types",
            flush=True,
        )
        return markers
    except Exception as e:
        print(f"  [MARKERS] quick_markers failed: {e}", flush=True)
        return None


# ---------------------------------------------------------------------------
# Patching + evaluation
# ---------------------------------------------------------------------------

def run_patch_and_evaluate(
    sc: ad.AnnData,
    spatial: ad.AnnData,
    out_dir: str | Path,
    neighbours: int = 10,
    computation: str = "annoy",
) -> None:
    """Run ISS-Patcher for cell-type prediction and axis imputation, evaluate quality.

    Outputs (all saved to out_dir)
    --------------------------------
    iss_patched.h5ad                : spatial AnnData with obs['fine_celltype'], obs['fine_celltype_fraction']
    axis.csv                        : scRNA cells × {universal_axis, universal_axis_std}
    similarity_per_cell.csv         : per spatial-cell Pearson/Spearman with matched SC neighbour
    similarity_metrics.csv          : per-celltype + weighted-average summary (all shared genes)
    markers.csv                     : top-20 TF-IDF markers per celltype (shared genes only)
    similarity_metrics_markers.csv  : same summary restricted to top-15 markers per celltype
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Cell-type prediction: transfer celltype from SC → spatial ──────────
    print("\n[PATCH 1/2] Cell-type prediction (spatial ← SC celltype)...", flush=True)
    out_ct = ip.patch(
        iss=spatial,
        gex=sc,
        obs_to_take="fine_celltype",
        neighbours=neighbours,
        computation=computation,
    )
    if "sample" in out_ct.obs.columns:
        out_ct.obs["sample"] = out_ct.obs["sample"].astype(str)
    out_ct.write(out_dir / "iss_patched.h5ad")
    print(f"  Saved iss_patched.h5ad  shape={out_ct.shape}", flush=True)

    # # ── 2. Axis imputation: transfer axis from spatial → SC ──────────────────
    # print("\n[PATCH 2/2] Axis imputation (SC ← spatial axis)...", flush=True)
    # out_axis = ip.patch(
    #     iss=sc,
    #     gex=spatial,
    #     cont_obs_to_take="universal_axis",
    #     neighbours=neighbours,
    #     computation=computation,
    # )
    # axis_df = out_axis.obs[["universal_axis", "universal_axis_std"]].copy()
    # axis_df.index.name = "cell_id"
    # axis_df.to_csv(out_dir / "axis.csv")
    # print(f"  Saved axis.csv  n_cells={len(axis_df)}", flush=True)

    # # ── 3. GEX similarity evaluation ─────────────────────────────────────────
    # print("\n[EVAL] Computing GEX similarity...", flush=True)
    # iss_scaled, gex_scaled, _ = ip.split_and_normalise_objects(spatial, sc)

    # # All shared genes
    # per_cell, summary = ip.evaluate_imputation_quality(
    #     iss_scaled, gex_scaled, annot_key="fine_celltype", computation=computation
    # )
    # per_cell.to_csv(out_dir / "similarity_per_cell.csv", index=False)
    # summary.to_csv(out_dir / "similarity_metrics.csv", index=False)
    # print(f"  Saved similarity_metrics.csv  n_celltypes={len(summary)-1}", flush=True)

    # # Marker genes (top-20 TF-IDF on shared genes, evaluate with top-15)
    # shared_genes = iss_scaled.var_names.tolist()
    # markers_df = compute_markers(sc, shared_genes, n_markers=20)
    # if markers_df is not None:
    #     markers_df.to_csv(out_dir / "markers.csv", index=False)
    #     markers_15 = markers_df.groupby("celltype", group_keys=False).head(15)
    #     _, summary_m = ip.evaluate_imputation_quality(
    #         iss_scaled, gex_scaled, annot_key="fine_celltype",
    #         markers_df=markers_15, computation=computation,
    #     )
    #     summary_m.to_csv(out_dir / "similarity_metrics_markers.csv", index=False)
    #     print(f"  Saved similarity_metrics_markers.csv", flush=True)

    print(f"  Done → {out_dir}", flush=True)
