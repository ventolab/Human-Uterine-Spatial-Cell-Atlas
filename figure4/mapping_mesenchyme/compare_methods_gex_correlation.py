"""
compare_methods_gex_correlation.py

QC comparison of axis-bin TACCO / ISS-Patcher configurations via
gene-expression Pearson correlation (scRNA vs spatial, per bin).

For each method/configuration:
  1. Assign axis bins to scRNA cells (via predictions or ISS-Patcher axis output)
  2. Bin spatial cells by their actual annotation (basalis → 2 bins,
     functionalis → 3 bins, lumen → 1 bin)  — identical logic for both methods
  3. Compute Pearson(mean_sc[bin], mean_sp[bin]) for each bin across shared genes
  4. Average across bins and spatial slides per stage; average across stages

QC filters
----------
  TACCO:       discard scRNA cells where max(tacco_scores) < 0.5
  ISS patcher: discard scRNA cells where universal_axis_std > 0.15

Outputs  gex_correlation_methods/
  per_bin_stage.parquet   (lineage, method, stage, slide, bin, n_sc, n_sp, corr)
  per_stage.parquet       (lineage, method, stage, mean_corr_across_bins)
  summary.parquet         (lineage, method, mean_corr, n_stages)
  figure.pdf

Usage
-----
cd mapping_mesenchyme
python compare_methods_gex_correlation.py
"""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sps
from scipy.stats import pearsonr

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = Path("/lustre/scratch125/cellgen/vento/mm58/eutopic_endometrium")
OUT_DIR  = Path("gex_correlation_methods")

TACCO_PREDS_DIR = BASE / "benchmark_knn_vs_dot/outputs/tacco/all_sc/full"

# Spatial slides per stage (shared by both lineages)
SPATIAL_BY_STAGE: dict[str, list[tuple[str, str]]] = {
    "Proliferative": [
        (
            "/nfs/team292/vl6/Endometriosis/Xenium/A13-UTR-0-TL4-1-S50/A13_annotated_new_axis.h5ad",
            str(TACCO_PREDS_DIR / "Proliferative/A13/tacco_predictions.csv"),
        ),
        (
            "/nfs/team292/vl6/Endometriosis/Xenium/A66-RPT-8-FO-1-S40/A66_annotated_new_axis.h5ad",
            str(TACCO_PREDS_DIR / "Proliferative/A66/tacco_predictions.csv"),
        ),
        (
            "/nfs/team292/vl6/Endometriosis/Xenium/DA72-END-0-FO-2-S2-ii/DA72_endo_annotated_new_axis.h5ad",
            str(TACCO_PREDS_DIR / "Proliferative/DA72_endo/tacco_predictions.csv"),
        ),
    ],
    "Secretory": [
        (
            "/nfs/team292/vl6/Endometriosis/Xenium/A30-UTR-2-FO-1-S48/A30_annotated_new_axis.h5ad",
            str(TACCO_PREDS_DIR / "Secretory/A30/tacco_predictions.csv"),
        ),
    ],
    "Hormones": [
        (
            "/nfs/team292/vl6/Endometriosis/Xenium/BZ99-END-0-FO-1-S3/BZ99_annotated_new_axis.h5ad",
            str(TACCO_PREDS_DIR / "Hormones/BZ99/tacco_predictions.csv"),
        ),
        (
            "/nfs/team292/vl6/Endometriosis/Xenium/DA45-END-0-FO-2-S2-i/DA45_annotated_new_axis.h5ad",
            str(TACCO_PREDS_DIR / "Hormones/DA45/tacco_predictions.csv"),
        ),
        (
            "/nfs/team292/vl6/Endometriosis/Xenium/DA46-END-0-FO-1-S4-i/DA46_annotated_new_axis.h5ad",
            str(TACCO_PREDS_DIR / "Hormones/DA46/tacco_predictions.csv"),
        ),
    ],
    "Menstrual": [
        (
            "/nfs/team292/vl6/Endometriosis/Xenium/DA39-END-0-FO-4-S8b/DA39_S8b_annotated_new_axis.h5ad",
            str(TACCO_PREDS_DIR / "Menstrual/DA39_S8b/tacco_predictions.csv"),
        ),
        (
            "/nfs/team292/vl6/Endometriosis/Xenium/DA50-END-0-FO-1-S2-i/DA50_annotated_new_axis.h5ad",
            str(TACCO_PREDS_DIR / "Menstrual/DA50/tacco_predictions.csv"),
        ),
        (
            "/nfs/team292/vl6/Endometriosis/Xenium/DA63-END-0-FO-1-S2-ii/DA63_annotated_new_axis.h5ad",
            str(TACCO_PREDS_DIR / "Menstrual/DA63/tacco_predictions.csv"),
        ),
    ],
}

# Lineage-specific configs — 2 TACCO + 2 ISS-Patcher methods each
LINEAGE_CONFIGS = {
    "Mesenchyme": {
        "scrna_h5ad":      str(BASE / "build_eutopic_object/integrated_scvi_uterus.h5ad"),
        "use_raw":         False,
        "lineage_obs_val": "mesenchymal",  # value in obs["lineage"]
        "lineage_prefix":  "Mesen",        # TACCO celltype prefix for spatial filter
        "methods": [
            {
                "name":      "tacco_concat",
                "type":      "tacco",
                "pred_dir":  BASE / "mapping_mesenchyme_all/tacco_annotation_bins",
                "scores_fn": "tacco_scores.parquet",
            },
            {
                "name":      "tacco_ensemble",
                "type":      "tacco",
                "pred_dir":  BASE / "mapping_mesenchyme_all/tacco_annotation_ensembl",
                "scores_fn": "tacco_scores_ensemble.parquet",
            },
            {
                "name":         "iss_concat",
                "type":         "iss_concat",
                "axis_dir":     BASE / "mapping_mesenchyme_all/axis_mapping_outputs",
                "stage_prefix": "concat_mesenchyme",
                "k":            10,
            },
            {
                "name":         "iss_ensemble",
                "type":         "iss_ensemble",
                "axis_dir":     BASE / "mapping_mesenchyme_all/axis_mapping_outputs",
                "stage_prefix": "ensemble_mesenchyme",
                "k":            10,
            },
        ],
    },
    "Epithelium": {
        "scrna_h5ad":      "/nfs/team292/projects/PanTissue/results/temp/02_annotation/annotated_postnatal_epithelial_endometrium.h5ad",
        "use_raw":         True,   # main X is normalised; raw.X has counts
        "lineage_obs_val": "epithelial",
        "lineage_prefix":  "Epi",
        "methods": [
            {
                "name":      "tacco_concat",
                "type":      "tacco",
                "pred_dir":  BASE / "mapping_epithelial_all/tacco_annotation_bins",
                "scores_fn": "tacco_scores.parquet",
            },
            {
                "name":      "tacco_ensemble",
                "type":      "tacco",
                "pred_dir":  BASE / "mapping_epithelial_all/tacco_annotation_ensembl",
                "scores_fn": "tacco_scores_ensemble.parquet",
            },
            {
                "name":         "iss_concat",
                "type":         "iss_concat",
                "axis_dir":     BASE / "mapping_epithelial_all/axis_mapping_outputs",
                "stage_prefix": "concat_epithelium",
                "k":            30,
            },
            {
                "name":         "iss_ensemble",
                "type":         "iss_ensemble",
                "axis_dir":     BASE / "mapping_epithelial_all/axis_mapping_outputs",
                "stage_prefix": "ensemble_epithelium",
                "k":            30,
            },
        ],
    },
}

BIN_CATEGORIES = [
    "basalis_1", "basalis_2",
    "functionalis_1", "functionalis_2", "functionalis_3",
    "lumen_1",
]
MIN_CELLS = 10      # min cells per bin to include
TACCO_SCORE_THRESH  = 0.5
ISS_STD_THRESH      = 0.15


# ── Preprocessing ──────────────────────────────────────────────────────────────

def _preprocess(X_counts, gene_names: list[str]) -> np.ndarray:
    """normalize_total → log1p → scale(max_value=10). Returns dense float32."""
    a = ad.AnnData(X=sps.csr_matrix(X_counts, dtype=np.float32))
    a.var_names = gene_names
    sc.pp.normalize_total(a)
    sc.pp.log1p(a)
    sc.pp.scale(a, max_value=10)
    return np.asarray(a.X, dtype=np.float32)


# ── Axis-bin assignment (shared logic for spatial and ISS-patcher) ─────────────

def _assign_axis_bins(axis: pd.Series, annot: pd.Series) -> pd.Series:
    """Map continuous universal_axis + compartment annotation → axis_bin label."""
    bins = pd.Series(pd.NA, index=axis.index, dtype=object)

    m_bas = annot == "basalis"
    if m_bas.any():
        bins[m_bas] = pd.cut(
            axis[m_bas], bins=2, labels=["basalis_1", "basalis_2"]
        ).astype(object)

    m_fun = annot == "functionalis"
    if m_fun.any():
        bins[m_fun] = pd.cut(
            axis[m_fun], bins=3,
            labels=["functionalis_1", "functionalis_2", "functionalis_3"],
        ).astype(object)

    m_lum = annot == "lumen"
    if m_lum.any():
        bins[m_lum] = "lumen_1"

    return pd.Categorical(bins, categories=BIN_CATEGORIES, ordered=True)


# ── scRNA loading ──────────────────────────────────────────────────────────────

def _load_scrna_obs_mesen(cfg: dict) -> pd.DataFrame:
    """Load mesenchyme scRNA obs from the integrated object, filtered by lineage."""
    adata = ad.read_h5ad(cfg["scrna_h5ad"], backed="r")
    obs = adata.obs[["Menstrual_stage_short", "Donor_id", "Tissue_ROI", "lineage"]].copy()
    obs = obs[obs["Donor_id"] != "GSM7277298"]
    obs = obs[~obs["Tissue_ROI"].isin(["Menstrual fluid", "Mentrual fluid"])]
    obs = obs[obs["Menstrual_stage_short"].notna()]
    obs = obs[obs["lineage"] == cfg["lineage_obs_val"]]
    return obs


def _load_scrna_obs_epi(cfg: dict) -> pd.DataFrame:
    """Load epithelium scRNA obs via h5py (backed mode unsupported)."""
    path = cfg["scrna_h5ad"]
    with h5py.File(path, "r") as f:
        idx = np.array([x.decode() for x in f["obs/_index"][:]])

        def _cat(grp):
            codes = grp["codes"][:]
            cats  = np.array([x.decode() for x in grp["categories"][:]])
            return cats[codes]

        ms_vals = _cat(f["obs/Menstrual_stage_short"])
        ct_vals = _cat(f["obs/celltype"])

    obs = pd.DataFrame(
        {"Menstrual_stage_short": ms_vals, "celltype": ct_vals},
        index=idx,
    )
    obs = obs[obs["celltype"].notna()]
    obs = obs[obs["celltype"].str.startswith("Epi")]
    return obs


def _read_scrna_counts_backed(
    h5ad_path: str,
    obs_names_ordered: list[str],
    gene_idx: list[int],
) -> sps.csr_matrix:
    """Read scRNA count rows for obs_names_ordered via backed mode (mesenchyme)."""
    adata = ad.read_h5ad(h5ad_path, backed="r")
    pos_map = {b: i for i, b in enumerate(adata.obs_names)}
    row_idx = np.array([pos_map[b] for b in obs_names_ordered])

    chunks = []
    n = len(row_idx)
    for s in range(0, n, 50_000):
        e   = min(s + 50_000, n)
        blk = adata.X[row_idx[s:e]][:, gene_idx]
        if not sps.issparse(blk):
            blk = sps.csr_matrix(blk)
        chunks.append(blk.astype(np.float32))
    return sps.vstack(chunks, format="csr")


def _read_scrna_counts_h5py(
    h5ad_path: str,
    obs_names_ordered: list[str],
    gene_idx: list[int],
    use_raw: bool,
) -> sps.csr_matrix:
    """Read scRNA counts via h5py (epithelium, where backed mode fails).

    Loads the full sparse matrix once, then subsets rows and columns.
    """
    with h5py.File(h5ad_path, "r") as f:
        xkey    = "raw/X" if use_raw else "X"
        idx_key = "raw/var/_index" if use_raw else "var/_index"

        all_barcodes = np.array([x.decode() for x in f["obs/_index"][:]])
        n_cols       = len(f[idx_key])

        data    = f[f"{xkey}/data"][:].astype(np.float32)
        indices = f[f"{xkey}/indices"][:].astype(np.int32)
        indptr  = f[f"{xkey}/indptr"][:].astype(np.int64)

    full_csr = sps.csr_matrix((data, indices, indptr), shape=(len(all_barcodes), n_cols))

    pos_map = {b: i for i, b in enumerate(all_barcodes)}
    row_idx = np.array([pos_map[b] for b in obs_names_ordered])
    return full_csr[row_idx][:, gene_idx]


# ── Spatial loading ────────────────────────────────────────────────────────────

def _load_spatial_slide(sp_path: str, pred_path: str, lineage_prefix: str) -> ad.AnnData:
    """Load spatial slide filtered to lineage, assign axis_bins, use raw counts."""
    adata = sc.read(sp_path)

    preds = pd.read_csv(pred_path, index_col=0)
    preds["lineage"] = preds["celltype"].str.split("_", n=1).str[0]
    keep_ids = preds.index[preds["lineage"] == lineage_prefix]

    adata = adata[adata.obs_names.isin(keep_ids)].copy()
    adata = adata[adata.obs["annotation"] != "myometrium"].copy()
    adata = adata[adata.obs["universal_axis"].notna()].copy()

    axis_bin = _assign_axis_bins(
        adata.obs["universal_axis"].astype(float),
        adata.obs["annotation"],
    )
    adata.obs["axis_bin"] = axis_bin
    adata = adata[adata.obs["axis_bin"].notna()].copy()

    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()

    return adata


# # ── Bin assignment: TACCO ──────────────────────────────────────────────────────

def _load_tacco_bins(pred_dir: Path, stage: str, scores_fn: str) -> pd.Series:
    """Return Series(index=cell_id, values=axis_bin) after QC score filter."""
    pred_file   = pred_dir / stage / "tacco_predictions.csv"
    scores_file = pred_dir / stage / scores_fn

    if not pred_file.exists():
        return pd.Series(dtype=str)

    preds = pd.read_csv(pred_file).set_index("cell_id")["axis_bin"].copy()

    if scores_file.exists():
        sc_df  = pd.read_parquet(scores_file)
        sc_df.index = sc_df.index.astype(str)
        low    = sc_df.max(axis=1) < TACCO_SCORE_THRESH
        preds[low.reindex(preds.index, fill_value=False)] = np.nan

    return preds.dropna()


# ── Bin assignment: ISS patcher ────────────────────────────────────────────────

def _load_iss_concat_bins(axis_dir: Path, stage: str, stage_prefix: str, k: int) -> pd.Series:
    """
    ISS-Patcher concat: single run per stage.
    Loads axis.csv (universal_axis + std) and iss_patched.h5ad (annotation).
    Applies universal_axis_std QC filter.
    """
    subdir   = axis_dir / f"{stage_prefix}_{stage.lower()}_k{k}"
    axis_csv = subdir / "axis.csv"
    iss_h5ad = subdir / "iss_patched.h5ad"

    if not axis_csv.exists() or not iss_h5ad.exists():
        return pd.Series(dtype=str)

    axis_df = pd.read_csv(axis_csv, index_col=0)
    iss_obs = ad.read_h5ad(iss_h5ad, backed="r").obs[["annotation"]]

    merged = axis_df.join(iss_obs, how="inner")
    merged = merged[merged["universal_axis_std"] <= ISS_STD_THRESH]
    merged = merged[merged["annotation"].isin(["basalis", "functionalis", "lumen"])]

    bins = _assign_axis_bins(merged["universal_axis"].astype(float), merged["annotation"])
    return pd.Series(bins, index=merged.index, dtype=object).dropna()


def _load_iss_ensemble_bins(axis_dir: Path, stage: str, stage_prefix: str, k: int) -> pd.Series:
    """
    ISS-Patcher ensemble: averaged universal_axis across per-sample runs.
    Loads ensemble_axis.csv and takes annotation from the first available
    per-sample iss_patched.h5ad (no std filter — not available after averaging).
    """
    subdir       = axis_dir / f"{stage_prefix}_{stage.lower()}_k{k}"
    ensemble_csv = subdir / "ensemble_axis.csv"

    if not ensemble_csv.exists():
        return pd.Series(dtype=str)

    axis_df = pd.read_csv(ensemble_csv, index_col=0)

    # Get annotation from the first available per-sample iss_patched.h5ad
    annot = None
    for sample_dir in sorted(p for p in subdir.iterdir() if p.is_dir()):
        h5ad_path = sample_dir / "iss_patched.h5ad"
        if h5ad_path.exists():
            annot = ad.read_h5ad(h5ad_path, backed="r").obs[["annotation"]]
            break

    if annot is None:
        return pd.Series(dtype=str)

    merged = axis_df.join(annot, how="inner")
    merged = merged[merged["annotation"].isin(["basalis", "functionalis", "lumen"])]

    bins = _assign_axis_bins(merged["universal_axis"].astype(float), merged["annotation"])
    return pd.Series(bins, index=merged.index, dtype=object).dropna()


# ── Correlation computation ────────────────────────────────────────────────────

def _bin_means(X_scaled: np.ndarray, bins: np.ndarray) -> dict[str, tuple[np.ndarray, int]]:
    """Mean expression vector per bin. Only bins with >= MIN_CELLS cells.
    Returns {bin: (mean_vector, n_cells)}.
    """
    result = {}
    for b in BIN_CATEGORIES:
        mask = bins == b
        n = mask.sum()
        if n >= MIN_CELLS:
            result[b] = (X_scaled[mask].mean(axis=0), int(n))
    return result


def _correlate_bins(
    sc_means: dict[str, tuple[np.ndarray, int]],
    sp_means: dict[str, tuple[np.ndarray, int]],
) -> tuple[list[dict], float | None]:
    """Per-bin Pearson correlation; return rows and mean correlation."""
    shared = [b for b in BIN_CATEGORIES if b in sc_means and b in sp_means]
    if not shared:
        return [], None

    rows = []
    for b in shared:
        sc_vec, n_sc = sc_means[b]
        sp_vec, n_sp = sp_means[b]
        r, _ = pearsonr(sc_vec, sp_vec)
        rows.append({"bin": b, "corr": float(r), "n_sc": n_sc, "n_sp": n_sp})
    mean_corr = float(np.mean([row["corr"] for row in rows]))
    return rows, mean_corr


def _cross_correlate_bins(
    sc_means: dict[str, tuple[np.ndarray, int]],
    sp_means: dict[str, tuple[np.ndarray, int]],
) -> pd.DataFrame:
    """Full cross-correlation matrix: rows = sc bins, cols = sp bins."""
    sc_bins = [b for b in BIN_CATEGORIES if b in sc_means]
    sp_bins = [b for b in BIN_CATEGORIES if b in sp_means]
    mat = pd.DataFrame(np.nan, index=sc_bins, columns=sp_bins)
    for bi in sc_bins:
        for bj in sp_bins:
            r, _ = pearsonr(sc_means[bi][0], sp_means[bj][0])
            mat.loc[bi, bj] = float(r)
    return mat


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    ensemble_cross_corrs: dict[str, list[pd.DataFrame]] = {}

    for lineage, cfg in LINEAGE_CONFIGS.items():
        print(f"\n{'='*60}", flush=True)
        print(f"Lineage: {lineage}", flush=True)
        print(f"{'='*60}", flush=True)

        # ── Load scRNA obs ────────────────────────────────────────────
        print("  Loading scRNA obs ...", flush=True)
        if lineage == "Mesenchyme":
            sc_obs = _load_scrna_obs_mesen(cfg)
        else:
            sc_obs = _load_scrna_obs_epi(cfg)
        print(f"  scRNA cells: {len(sc_obs):,}", flush=True)

        # ── Get shared genes with any spatial slide ───────────────────
        any_sp  = ad.read_h5ad(
            SPATIAL_BY_STAGE["Proliferative"][0][0], backed="r"
        )
        sp_gene_set  = set(any_sp.var_names)
        sp_genes_ord = list(any_sp.var_names)

        # Find scRNA gene indices
        with h5py.File(cfg["scrna_h5ad"], "r") as f:
            idx_key   = "raw/var/_index" if cfg["use_raw"] else "var/_index"
            sc_genes  = np.array([g.decode() for g in f[idx_key][:]])
        sc_gene_map = {g: i for i, g in enumerate(sc_genes)}
        gene_idx    = [sc_gene_map[g] for g in sp_genes_ord if g in sc_gene_map]
        shared_gene_names = [g for g in sp_genes_ord if g in sc_gene_map]
        print(f"  Shared genes: {len(shared_gene_names)}", flush=True)

        # ── Load & scale scRNA counts once per lineage ────────────────
        print("  Loading scRNA counts ...", flush=True)
        obs_names = sc_obs.index.tolist()
        if lineage == "Mesenchyme":
            X_sc_raw = _read_scrna_counts_backed(
                cfg["scrna_h5ad"], obs_names, gene_idx
            )
        else:
            X_sc_raw = _read_scrna_counts_h5py(
                cfg["scrna_h5ad"], obs_names, gene_idx, cfg["use_raw"]
            )
        print(f"  Preprocessing scRNA ({X_sc_raw.shape}) ...", flush=True)
        X_sc = _preprocess(X_sc_raw, shared_gene_names)
        sc_obs = sc_obs.loc[obs_names]  # ensure alignment

        # ── For each method, load bin assignments ─────────────────────
        method_bins: dict[str, pd.Series] = {}
        for meth in cfg["methods"]:
            print(f"\n  Method: {meth['name']}", flush=True)
            all_stage_bins = []
            for stage in SPATIAL_BY_STAGE:
                if meth["type"] == "tacco":
                    b = _load_tacco_bins(meth["pred_dir"], stage, meth["scores_fn"])
                elif meth["type"] == "iss_concat":
                    b = _load_iss_concat_bins(meth["axis_dir"], stage, meth["stage_prefix"], meth["k"])
                elif meth["type"] == "iss_ensemble":
                    b = _load_iss_ensemble_bins(meth["axis_dir"], stage, meth["stage_prefix"], meth["k"])
                else:
                    b = pd.Series(dtype=str)
                if len(b):
                    all_stage_bins.append(b)
            if all_stage_bins:
                method_bins[meth["name"]] = pd.concat(all_stage_bins)
                print(
                    f"    {len(method_bins[meth['name']]):,} cells with valid bins",
                    flush=True,
                )
            else:
                print("    No bins found!", flush=True)

        # ── Load spatial slides (once each), compute correlations ──────
        print("\n  Computing correlations ...", flush=True)
        for stage, slide_list in SPATIAL_BY_STAGE.items():
            print(f"\n  Stage: {stage}", flush=True)

            # Indices of scRNA cells in this stage
            sc_stage_mask = sc_obs["Menstrual_stage_short"] == stage
            sc_stage_obs  = sc_obs[sc_stage_mask]
            sc_stage_idx  = np.where(sc_stage_mask.values)[0]
            X_sc_stage    = X_sc[sc_stage_idx]

            if sc_stage_idx.size == 0:
                print(f"    No scRNA cells for stage {stage}", flush=True)
                continue

            for sp_path, pred_path in slide_list:
                slide_name = Path(pred_path).parent.name
                print(f"    Slide: {slide_name}", flush=True)

                sp_adata = _load_spatial_slide(
                    sp_path, pred_path, cfg["lineage_prefix"]
                )
                if sp_adata.shape[0] == 0:
                    print("      No spatial cells after filtering", flush=True)
                    continue

                # Subset spatial to shared genes (in the same order as sc)
                sp_gene_map_local = {g: i for i, g in enumerate(sp_adata.var_names)}
                sp_gene_idx_local = [
                    sp_gene_map_local[g] for g in shared_gene_names
                    if g in sp_gene_map_local
                ]
                shared_for_slide = [
                    g for g in shared_gene_names if g in sp_gene_map_local
                ]
                if not shared_for_slide:
                    continue

                X_sp_raw = sp_adata.X[:, sp_gene_idx_local]
                X_sp     = _preprocess(X_sp_raw, shared_for_slide)
                sp_bins  = sp_adata.obs["axis_bin"].values
                sp_means = _bin_means(X_sp, sp_bins)
                if not sp_means:
                    continue

                # sc gene index into shared_gene_names → shared_for_slide
                sc_shared_idx = [
                    i for i, g in enumerate(shared_gene_names)
                    if g in sp_gene_map_local
                ]

                for meth_name, bins_series in method_bins.items():
                    # Intersect scRNA cells in this stage with valid bins
                    valid_stage = bins_series.reindex(sc_stage_obs.index).dropna()
                    if len(valid_stage) == 0:
                        continue

                    # Align to scRNA matrix order
                    sc_obs_idx_map = {b: i for i, b in enumerate(sc_stage_obs.index)}
                    cell_row_idx   = np.array([
                        sc_obs_idx_map[c] for c in valid_stage.index
                        if c in sc_obs_idx_map
                    ])
                    cell_bin_vals  = valid_stage.reindex(
                        [sc_stage_obs.index[i] for i in cell_row_idx]
                    ).values

                    X_sc_meth = X_sc_stage[cell_row_idx][:, sc_shared_idx]
                    sc_means  = _bin_means(X_sc_meth, cell_bin_vals)

                    bin_rows, mean_corr = _correlate_bins(sc_means, sp_means)
                    if mean_corr is None:
                        continue

                    if meth_name == "tacco_ensemble":
                        cross_mat = _cross_correlate_bins(sc_means, sp_means)
                        ensemble_cross_corrs.setdefault(lineage, []).append(cross_mat)

                    for br in bin_rows:
                        all_rows.append({
                            "lineage": lineage,
                            "method":  meth_name,
                            "stage":   stage,
                            "slide":   slide_name,
                            **br,
                        })
                    print(
                        f"      [{meth_name}] {len(bin_rows)} bins | "
                        f"mean corr = {mean_corr:.3f}",
                        flush=True,
                    )

    # ── Save & aggregate ───────────────────────────────────────────────────────
    df_bins = pd.DataFrame(all_rows)
    df_bins.to_parquet(OUT_DIR / "per_bin_stage.parquet", index=False)
    print(f"\nSaved {len(df_bins):,} rows → {OUT_DIR}/per_bin_stage.parquet")

    per_stage = (
        df_bins.groupby(["lineage", "method", "stage", "slide"])
        ["corr"].mean()
        .reset_index()
        .rename(columns={"corr": "mean_corr"})
    )
    per_stage.to_parquet(OUT_DIR / "per_stage.parquet", index=False)

    summary = (
        per_stage.groupby(["lineage", "method"])
        .agg(mean_corr=("mean_corr", "mean"), n_stages=("stage", "nunique"))
        .reset_index()
        .sort_values(["lineage", "mean_corr"], ascending=[True, False])
    )
    summary.to_parquet(OUT_DIR / "summary.parquet", index=False)
    print(summary.to_string(index=False))

    # ── Figure ─────────────────────────────────────────────────────────────────
    _make_figure(summary, per_stage)
    _make_ensemble_heatmap(ensemble_cross_corrs)


METHOD_ORDER = ["tacco_concat", "tacco_ensemble", "iss_concat", "iss_ensemble"]
METHOD_LABELS = {
    "tacco_concat":   "TACCO\n(concat)",
    "tacco_ensemble": "TACCO\n(ensemble)",
    "iss_concat":     "ISS-Patcher\n(concat)",
    "iss_ensemble":   "ISS-Patcher\n(ensemble)",
}
COLORS = {
    "tacco_concat":   "#2E74C0",
    "tacco_ensemble": "#1A4A7A",
    "iss_concat":     "#E07B39",
    "iss_ensemble":   "#A84E1A",
}


def _make_figure(summary: pd.DataFrame, per_stage: pd.DataFrame) -> None:
    lineages = list(LINEAGE_CONFIGS.keys())
    fig, axes = plt.subplots(1, len(lineages), figsize=(4 * len(lineages), 5),
                             sharey=False)
    if len(lineages) == 1:
        axes = [axes]

    for ax, lineage in zip(axes, lineages):
        sub = summary[summary["lineage"] == lineage].copy()
        methods = [m for m in METHOD_ORDER if m in sub["method"].values]
        xs      = np.arange(len(methods))
        heights = [sub.loc[sub["method"] == m, "mean_corr"].values[0] for m in methods]
        colors  = [COLORS[m] for m in methods]

        # Stage-level dots for individual estimates
        ps_sub = per_stage[per_stage["lineage"] == lineage]

        bars = ax.bar(xs, heights, color=colors, width=0.6, zorder=2)
        for xi, m in enumerate(methods):
            stage_vals = ps_sub[ps_sub["method"] == m]["mean_corr"].values
            ax.scatter(
                np.full(len(stage_vals), xi),
                stage_vals,
                color="k", s=20, zorder=3, alpha=0.6,
            )

        ax.set_xticks(xs)
        ax.set_xticklabels([METHOD_LABELS[m] for m in methods], fontsize=9)
        ax.set_ylabel("Mean Pearson correlation (scRNA vs spatial)", fontsize=9)
        ax.set_title(lineage, fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(max(heights) * 1.2, 0.5))
        ax.axhline(0, color="k", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig_path = OUT_DIR / "figure.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    print(f"Saved figure → {fig_path}")


def _make_ensemble_heatmap(
    ensemble_cross_corrs: dict[str, list[pd.DataFrame]],
) -> None:
    """Heatmap of mean Pearson(sc_bin_i, sp_bin_j) for tacco_ensemble."""
    lineages = [lin for lin in LINEAGE_CONFIGS if ensemble_cross_corrs.get(lin)]
    if not lineages:
        return

    fig, axes = plt.subplots(1, len(lineages), figsize=(5.5 * len(lineages), 4.5))
    if len(lineages) == 1:
        axes = [axes]

    for ax, lineage in zip(axes, lineages):
        mats = ensemble_cross_corrs[lineage]
        aligned = [
            m.reindex(index=BIN_CATEGORIES, columns=BIN_CATEGORIES)
            for m in mats
        ]
        avg = np.nanmean(np.stack([m.values for m in aligned], axis=0), axis=0)

        im = ax.imshow(avg, aspect="auto", vmin=-1, vmax=1, cmap="RdBu_r")
        ticks = range(len(BIN_CATEGORIES))
        ax.set_xticks(ticks)
        ax.set_xticklabels(BIN_CATEGORIES, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(ticks)
        ax.set_yticklabels(BIN_CATEGORIES, fontsize=8)
        ax.set_xlabel("Spatial bin", fontsize=9)
        ax.set_ylabel("scRNA bin", fontsize=9)
        ax.set_title(f"{lineage} — TACCO ensemble", fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax, label="Pearson r", fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig_path = OUT_DIR / "ensemble_bin_heatmap.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    print(f"Saved ensemble heatmap → {fig_path}")


if __name__ == "__main__":
    main()
