"""
Shared utilities for ISS patcher pipelines (Xenium, Visium HD, ...).
"""

from math import ceil
from pathlib import Path
from typing import Literal, Optional

import anndata as ad
import iss_patcher as ip
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Visium HD QC thresholds
VHD_MIN_BINS  = 5    # keep bins with count > this
VHD_MIN_CELLS = 3    # sc.pp.filter_genes min_cells
VHD_MIN_GENES = 100  # sc.pp.filter_cells min_genes


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_xenium_counts(xenium_dir: str | Path) -> ad.AnnData:
    """
    Load a Xenium output folder into an AnnData object with raw counts
    and spatial coordinates.

    Supports both:
      - New format: cell_feature_matrix.h5  + cells.parquet
      - Old format: matrix.csv              + cells.csv.gz

    Parameters
    ----------
    xenium_dir : str or Path
        Path to the Xenium output folder (contains experiment.xenium).

    Returns
    -------
    AnnData with:
        .X / .layers["counts"]  — raw sparse counts
        .obsm["spatial"]        — (x, y) centroids
        .raw                    — frozen raw state
    """
    xenium_dir = Path(xenium_dir)

    # 1. Load counts
    h5_path     = xenium_dir / "cell_feature_matrix.h5"
    matrix_path = xenium_dir / "matrix.csv"

    if h5_path.exists():
        print("Loading counts from cell_feature_matrix.h5 ...", flush=True)
        adata = sc.read_10x_h5(str(h5_path))
        adata.var_names_make_unique()
    elif matrix_path.exists():
        print("Loading counts from matrix.csv ...", flush=True)
        counts = pd.read_csv(matrix_path, index_col=0).T
        adata = ad.AnnData(
            X=sp.csr_matrix(counts.values),
            obs=pd.DataFrame(index=counts.index),
            var=pd.DataFrame(index=counts.columns),
        )
        adata.var_names_make_unique()
    else:
        raise FileNotFoundError(
            f"No counts file found in {xenium_dir}. "
            "Expected cell_feature_matrix.h5 or matrix.csv."
        )

    print(f"  Counts shape (cells × genes): {adata.shape}", flush=True)

    # 2. Load cell metadata with spatial coordinates
    parquet_path = xenium_dir / "cells.parquet"
    csv_path     = xenium_dir / "cells.csv.gz"

    if parquet_path.exists():
        print("Loading cell metadata from cells.parquet ...", flush=True)
        cells_df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        print("Loading cell metadata from cells.csv.gz ...", flush=True)
        cells_df = pd.read_csv(csv_path, compression="gzip")
    else:
        raise FileNotFoundError(
            f"No cell metadata file found in {xenium_dir}. "
            "Expected cells.parquet or cells.csv.gz."
        )

    print(f"  Cells columns: {cells_df.columns.tolist()}", flush=True)

    # Detect cell ID column
    cell_id_col = next(
        (c for c in ["cell_id", "Unnamed: 0", "barcode"] if c in cells_df.columns),
        cells_df.columns[0],
    )
    cells_df = cells_df.set_index(cell_id_col)
    cells_df.index = cells_df.index.astype(str)
    adata.obs.index = adata.obs.index.astype(str)

    # Detect x/y columns
    x_col = next(c for c in ["x_centroid", "x_location", "x"] if c in cells_df.columns)
    y_col = next(c for c in ["y_centroid", "y_location", "y"] if c in cells_df.columns)

    # 3. Align cells
    shared = adata.obs.index.intersection(cells_df.index)
    print(
        f"  Shared cells: {len(shared)} "
        f"(counts: {adata.n_obs}, cells file: {len(cells_df)})",
        flush=True,
    )
    if len(shared) == 0:
        raise ValueError(
            "No shared cell IDs between counts and cells file. "
            "Check index formatting."
        )

    adata = adata[shared].copy()
    cells_df = cells_df.loc[shared]

    # 4. Attach metadata and coordinates
    for col in cells_df.columns:
        adata.obs[col] = cells_df[col].values
    adata.obsm["spatial"] = cells_df[[x_col, y_col]].values.astype(float)

    # 5. Store raw counts
    adata.layers["counts"] = adata.X.copy()
    adata.raw = adata

    # 6. Sample metadata
    adata.uns["spatial"] = {"sample": xenium_dir.name}

    print(f"\nOutput adata {adata}", flush=True)
    return adata


# ---------------------------------------------------------------------------
# QC
# ---------------------------------------------------------------------------

def qc_visium_hd(adata: ad.AnnData) -> ad.AnnData:
    """Filter and normalise a Visium HD AnnData ready for iss_patcher."""
    adata.var_names_make_unique()
    adata = adata[adata.obs["bin_count"] > VHD_MIN_BINS].copy()
    adata.X.data = np.round(adata.X.data)   # integers required for Seurat v3 HVGs
    adata.raw = adata.copy()
    sc.pp.filter_genes(adata, min_cells=VHD_MIN_CELLS)
    sc.pp.filter_cells(adata, min_genes=VHD_MIN_GENES)
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    return adata


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_celltype_grid(adata: ad.AnnData, output_png: Path, ncols: int = 4) -> None:
    """Save a spatial grid figure with one panel per cell type."""
    cell_types = sorted(adata.obs["celltype"].astype(str).unique().tolist())
    n_panels = len(cell_types)
    ncols = max(1, ncols)
    nrows = ceil(n_panels / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.5 * ncols, 4.5 * nrows),
        squeeze=False,
    )

    for idx, cell_type in enumerate(cell_types):
        ax = axes[idx // ncols][idx % ncols]
        sc.pl.embedding(
            adata,
            basis="spatial",
            color="celltype",
            groups=[cell_type],
            size=20,
            frameon=False,
            na_color="#E8E8E8",
            ax=ax,
            show=False,
            title=cell_type,
        )

    for idx in range(n_panels, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.tight_layout()
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Patching
# ---------------------------------------------------------------------------

def run_patch(
    sample_id: str,
    adata: ad.AnnData,
    gex: ad.AnnData,
    output_dir: Path,
    obs_to_take: str = "celltype",
    cont_obs_to_take: Optional[str] = None,
    neighbours: int = 10,
    computation: str = "cKDTree",
    ncols: int = 4,
    twostep: bool = False,
    annot_key: str = "celltype",
    neighbours_annot: int = 15,
) -> None:
    """Run iss_patcher.patch (or patch_twostep) on a single sample and write outputs.

    Parameters
    ----------
    twostep : bool
        If True, use ip.patch_twostep() instead of ip.patch(). Recommended
        for large datasets such as Visium HD, where a two-pass KNN (coarse
        annotation first, then within-cell-type refinement) is much faster.
    annot_key : str
        obs column used as the first-pass annotation in patch_twostep.
    neighbours_annot : int
        Number of neighbours for the first-pass annotation in patch_twostep.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if twostep:
        print(f"[{sample_id}] Running iss_patcher.patch_twostep (computation={computation}) ...", flush=True)
        out = ip.patch_twostep(
            iss=adata,
            gex=gex,
            annot_key=annot_key,
            obs_to_take=obs_to_take,
            cont_obs_to_take = cont_obs_to_take,
            neighbours=neighbours,
            neighbours_annot=neighbours_annot,
            computation=computation,
        )
    else:
        print(f"[{sample_id}] Running iss_patcher.patch (computation={computation}) ...", flush=True)
        out = ip.patch(
            iss=adata,
            gex=gex,
            obs_to_take=obs_to_take,
            cont_obs_to_take = cont_obs_to_take,
            neighbours=neighbours,
            computation=computation,
        )

    # avoid errors writing to disk
    if 'sample' in out.obs.columns:
        out.obs['sample'] = out.obs['sample'].astype(str)
    if 'Postnatal_age_years' in out.obs.columns:
        out.obs['Postnatal_age_years'] = pd.to_numeric(out.obs['Postnatal_age_years'], errors='coerce')
    if 'predicted_doublets' in out.obs.columns:
        out.obs['predicted_doublets'] = out.obs['predicted_doublets'].astype(str)
    if 'predicted_doublet' in out.obs.columns:
        out.obs['predicted_doublet'] = out.obs['predicted_doublet'].astype(str)

    out.write(output_dir / "iss_patched.h5ad")
    axis_cols = [c for c in ['universal_axis', 'universal_axis_std'] if c in out.obs.columns]
    out.obs[axis_cols].to_csv(output_dir / "axis.csv")
    # not plotting the spatial distribution in this version
    # plot_celltype_grid(out, output_dir / "celltype_spatial_grid.png", ncols=ncols)
    print(f"[{sample_id}] Done → {output_dir}", flush=True)


# ---------------------------------------------------------------------------
# Axis mapping: spatial to scRNA-seq
# ---------------------------------------------------------------------------

# Maps lineage name used in this codebase to the prefix used in TACCO celltype labels
_LINEAGE_TO_TACCO_PREFIX: dict[str, str] = {
    "epithelium": "Epi",
    "mesenchyme": "Mesen",
}


def load_spatial_data(
    h5ad_path: str | Path,
    lineage: Literal["epithelium", "mesenchyme"],
    tacco_pred_path: str | Path,
) -> ad.AnnData:
    """
    Load and preprocess spatial (Xenium) data for axis mapping.

    Filters cells to the specified lineage using pre-computed TACCO fine-celltype
    predictions (celltype prefix 'Epi' for epithelium, 'Mesen' for mesenchyme).

    Parameters
    ----------
    h5ad_path : str or Path
        Path to the Xenium h5ad file.
    lineage : {'epithelium', 'mesenchyme'}
        Cell lineage to keep.
    tacco_pred_path : str or Path
        Path to a TACCO predictions CSV with a 'celltype' column whose values
        are prefixed by lineage (e.g. 'Epi_luminal', 'Mesen_stromal').

    Returns
    -------
    AnnData
        Preprocessed spatial data filtered to the requested lineage.
    """
    h5ad_path = Path(h5ad_path)
    print(f"\n[SPATIAL] Loading: {h5ad_path.name}", flush=True)
    adata = sc.read(h5ad_path)
    print(f"  Initial shape: {adata.shape}", flush=True)

    # Remove myometrium and unannotated cells
    adata = adata[adata.obs['annotation'].notna()].copy()
    adata = adata[adata.obs['annotation'] != 'myometrium'].copy()

    # Ensure universal_axis is numeric, then drop NaN rows
    adata.obs['universal_axis'] = pd.to_numeric(adata.obs['universal_axis'], errors='coerce')
    n_before_axis = adata.shape[0]
    adata = adata[adata.obs['universal_axis'].notna()].copy()
    if adata.shape[0] < n_before_axis:
        print(f"  Dropped {n_before_axis - adata.shape[0]} cells with NaN universal_axis", flush=True)
    print(f"  Shape after annotation + axis filters: {adata.shape}", flush=True)

    # Restore raw counts
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
        print("  Restored raw counts from layers['counts']", flush=True)

    # Filter to lineage using TACCO predictions
    prefix = _LINEAGE_TO_TACCO_PREFIX[lineage]
    print(f"  Loading TACCO predictions for lineage filter: {Path(tacco_pred_path).name}", flush=True)
    preds = pd.read_csv(tacco_pred_path, index_col=0)
    preds["_lineage_prefix"] = preds["celltype"].str.split("_", n=1).str[0]
    lineage_ids = preds.index[preds["_lineage_prefix"] == prefix]

    n_before = adata.shape[0]
    adata = adata[adata.obs_names.isin(lineage_ids)].copy()
    print(
        f"  After {lineage} filter ({prefix}_*): {adata.shape[0]:,} cells "
        f"({100 * adata.shape[0] / n_before:.1f}% of slide)",
        flush=True,
    )
    if adata.shape[0] == 0:
        raise ValueError(
            f"No cells remain after filtering to lineage '{lineage}' "
            f"(TACCO prefix '{prefix}'). Check that {Path(tacco_pred_path).name} "
            "covers this slide."
        )

    # Prefix obs_names with sample so barcodes stay unique after concat
    adata.obs_names = adata.obs["sample"].astype(str) + "_" + adata.obs_names.astype(str)

    return adata


# Maps lineage names used in this codebase to the values in adata.obs["lineage"]
_LINEAGE_TO_OBS_VALUE: dict[str, str] = {
    "epithelium": "epithelial",
    "mesenchyme": "mesenchymal",
}


def load_scrna(
    h5ad_path: str | Path,
    lineage: Literal["epithelium", "mesenchyme"],
    menstrual_stage: Literal["Proliferative", "Secretory", "Hormones", "Menstrual", "Postmenopause"],
) -> ad.AnnData:
    """
    Load and preprocess scRNA-seq data for axis mapping.

    Parameters
    ----------
    h5ad_path : str or Path
        Path to integrated scRNA-seq h5ad.
    lineage : {'epithelium', 'mesenchyme'}
        Lineage to keep. Translated to obs column values via _LINEAGE_TO_OBS_VALUE.
    menstrual_stage : {'Proliferative', 'Secretory', 'Hormones', 'Menstrual'}
        Menstrual stage to filter to (case-sensitive).

    Returns
    -------
    AnnData
        Filtered scRNA-seq data for specified lineage and stage.
    """
    lineage_obs_value = _LINEAGE_TO_OBS_VALUE[lineage]

    h5ad_path = Path(h5ad_path)
    print(f"\n[SCRNA-SEQ] Loading: {h5ad_path.name}", flush=True)
    adata = sc.read(h5ad_path)
    print(f"  Initial shape: {adata.shape}", flush=True)
    print(f"  lineage col unique values: {sorted(adata.obs['lineage'].unique().tolist())}", flush=True)

    mask = (
        (adata.obs["Menstrual_stage_short"] == menstrual_stage)
        & (adata.obs["lineage"] == lineage_obs_value)
        & ~adata.obs["Tissue_ROI"].isin(["Menstrual fluid", "Mentrual fluid"])
        & (adata.obs["Donor_id"] != "GSM7277298")
        & adata.obs["Menstrual_stage_short"].notna()
    )
    adata = adata[mask].copy()
    print(
        f"  After filtering (stage='{menstrual_stage}', lineage='{lineage}' "
        f"→ obs value='{lineage_obs_value}'): {adata.shape}",
        flush=True,
    )
    if adata.shape[0] == 0:
        raise ValueError(
            f"No scRNA-seq cells found for stage='{menstrual_stage}', "
            f"lineage='{lineage}' (obs value='{lineage_obs_value}'). "
            "Check the lineage column values above."
        )

    return adata



def map_spatial_to_scrna(
    spatial_h5ad: tuple[str, str] | list[tuple[str, str]],
    scrna_h5ad: str | Path,
    lineage: Literal["epithelium", "mesenchyme"],
    menstrual_stage: Literal["Proliferative", "Secretory", "Hormones", "Menstrual", "Postmenopause"],
    output_dir: str | Path,
    neighbours: int = 10,
    computation: str = "cKDTree",
    scrna_annot_csv: str | Path | None = None,
) -> None:
    """
    Core function to map spatial axis values onto scRNA-seq reference.

    Parameters
    ----------
    spatial_h5ad : (h5ad_path, tacco_pred_path) or list thereof
        Each element is a tuple of (path to Xenium h5ad, path to TACCO predictions
        CSV for that slide). When a list is provided all slides are loaded,
        filtered to the specified lineage, and concatenated before mapping.
    scrna_h5ad : str or Path
        Path to scRNA-seq reference h5ad file.
    lineage : {'epithelium', 'mesenchyme'}
        Cell lineage to process.
    menstrual_stage : {'Proliferative', 'Secretory', 'Hormones', 'Menstrual'}
        Menstrual stage (case-sensitive).
    output_dir : str or Path
        Where to save outputs.
    neighbours : int
        Number of neighbours for iss_patcher.
    computation : str
        Computation method for iss_patcher ('cKDTree', 'annoy', 'pynndescent').
    scrna_annot_csv : str or Path or None
        Unused — kept for backwards compatibility.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}", flush=True)
    print(f"Mapping {lineage.upper()} | {menstrual_stage}", flush=True)
    print(f"{'='*70}", flush=True)

    # Load spatial data (single slide or concatenate multiple)
    try:
        if isinstance(spatial_h5ad, list):
            parts = [
                load_spatial_data(h5ad_path=h5ad, lineage=lineage, tacco_pred_path=pred)
                for h5ad, pred in spatial_h5ad
            ]
            iss = ad.concat(parts, join="inner", label="batch", keys=[Path(h5ad).stem for h5ad, _ in spatial_h5ad])
            print(f"  Concatenated {len(parts)} spatial datasets → {iss.shape}", flush=True)
        else:
            h5ad, pred = spatial_h5ad
            iss = load_spatial_data(h5ad_path=h5ad, lineage=lineage, tacco_pred_path=pred)
    except Exception as e:
        print(f"ERROR loading spatial data: {e}", flush=True)
        raise

    # Load scRNA-seq reference
    gex = load_scrna(
        h5ad_path=scrna_h5ad,
        lineage=lineage,
        menstrual_stage=menstrual_stage,
    )

    # Initialise universal_axis column in reference (target) so iss_patcher can fill it
    gex.obs["universal_axis"] = np.nan
    print("[axis_copy] Added universal_axis column to reference (initialized with NaN)", flush=True)

    # Run iss_patcher with axis information
    run_patch(
        sample_id=f"{lineage}_{menstrual_stage}",
        adata=gex,  # reference data (target for patching)
        gex=iss,    # spatial data (source with axis values)
        output_dir=output_dir,
        obs_to_take="annotation",
        cont_obs_to_take="universal_axis",
        neighbours=neighbours,
        computation=computation,
    )
    
    
def map_scrna_to_spatial(
    spatial_h5ad: tuple[str, str] | list[tuple[str, str]],
    scrna_h5ad: str | Path,
    lineage: Literal["epithelium", "mesenchyme"],
    menstrual_stage: Literal["Proliferative", "Secretory", "Hormones", "Menstrual", "Postmenopause"],
    output_dir: str | Path,
    neighbours: int = 10,
    computation: str = "annoy",
    scrna_annot_csv: str | Path | None = None,
) -> None:
    """
    Core function to map cell type labels from scRNA-seq onto spatial data.

    Parameters
    ----------
    spatial_h5ad : (h5ad_path, tacco_pred_path) or list thereof
        Each element is a tuple of (path to Xenium h5ad, path to TACCO predictions
        CSV for that slide).
    scrna_h5ad : str or Path
        Path to scRNA-seq reference h5ad file.
    lineage : {'epithelium', 'mesenchyme'}
        Cell lineage to process.
    menstrual_stage : {'Proliferative', 'Secretory', 'Hormones', 'Menstrual'}
        Menstrual stage (case-sensitive).
    output_dir : str or Path
        Where to save outputs.
    neighbours : int
        Number of neighbours for iss_patcher.
    computation : str
        Computation method for iss_patcher ('cKDTree', 'annoy', 'pynndescent').
    scrna_annot_csv : str or Path or None
        Unused — kept for backwards compatibility.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}", flush=True)
    print(f"Mapping {lineage.upper()} | {menstrual_stage}", flush=True)
    print(f"{'='*70}", flush=True)

    # Load spatial data (single slide or concatenate multiple)
    try:
        if isinstance(spatial_h5ad, list):
            parts = [
                load_spatial_data(h5ad_path=h5ad, lineage=lineage, tacco_pred_path=pred)
                for h5ad, pred in spatial_h5ad
            ]
            iss = ad.concat(parts, join="inner", label="batch", keys=[Path(h5ad).stem for h5ad, _ in spatial_h5ad])
            print(f"  Concatenated {len(parts)} spatial datasets → {iss.shape}", flush=True)
        else:
            h5ad, pred = spatial_h5ad
            iss = load_spatial_data(h5ad_path=h5ad, lineage=lineage, tacco_pred_path=pred)
    except Exception as e:
        print(f"ERROR loading spatial data: {e}", flush=True)
        raise

    # Load scRNA-seq reference
    gex = load_scrna(
        h5ad_path=scrna_h5ad,
        lineage=lineage,
        menstrual_stage=menstrual_stage,
    )

    # Initialise universal_axis column in reference (target) so iss_patcher can fill it
    gex.obs["universal_axis"] = np.nan
    print("[axis_copy] Added universal_axis column to reference (initialized with NaN)", flush=True)

    # Run iss_patcher with axis information
    run_patch(
        sample_id=f"{lineage}_{menstrual_stage}",
        adata=iss,  # reference data (target for patching)
        gex=gex,    # spatial data (source with axis values)
        output_dir=output_dir,
        obs_to_take="celltype",
        neighbours=neighbours,
        computation=computation,
    )