
# # 03 · Integration with scVI — uterus, mesenchymal lineage
# 
# Subset of the uterus integration restricted to cells annotated as **mesenchymal** in `uterus_02_annotation.ipynb`.
# 
# **Pipeline:**
# 1. Load `obs_annotation_uterus.csv` → extract mesenchymal barcodes
# 2. For each uterus h5ad: load → subset to mesenchymal barcodes → save to a dedicated temp folder
# 3. Concatenate subsetted files on-disk: `join='outer'` (union of genes) and `join='inner'` (shared genes)
# 4. Inner concat → HVGs stratified by menstrual stage (`seurat_v3`) → scVI (`Dataset` batch, `Donor_id` covariate)
# 5. Transfer `X_scVI` to outer concat → KNN → UMAP → Leiden
# 6. Save integrated object


# ## Configuration


import os

# ── Paths ─────────────────────────────────────────────────────────────────────
MAIN_DIR   = "/nfs/team292/projects/PanTissue/"
INPUT_DIR  = os.path.join(MAIN_DIR, "results/temp/anndata_copy_freeze/")

# Annotation CSV 
ANNOTATION_CONCAT = "/nfs/team292/projects/PanTissue/results/freeze/annotations/concatenated_annotations_postnatal_v2.csv"

# for all, remove unknown, lowQC, doublet
    
TISSUE_COL = "Organ"
TISSUE_TARGET = 'Uterus'

# Integration outputs
OUTPUT_DIR        = '/lustre/scratch125/cellgen/vento/mm58/eutopic_endometrium/build_eutopic_object'
CONCAT_H5AD_OUTER = os.path.join(OUTPUT_DIR, "concat_uterus_outer.h5ad")
CONCAT_H5AD_INNER = os.path.join(OUTPUT_DIR, "concat_uterus_inner.h5ad")
INTEGRATED_H5AD   = os.path.join(OUTPUT_DIR, "integrated_scvi_uterus.h5ad")
SCVI_MODEL_DIR    = os.path.join(OUTPUT_DIR, "scvi_model_uterus")

# Temporary folder for per-dataset subsetted h5ads (created here, used for concat)
SUBSET_DIR = os.path.join(OUTPUT_DIR, "results/temp/00_preprocessing_uterus/")

os.makedirs(SUBSET_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Column names in obs ───────────────────────────────────────────────────────
DATASET_COL   = "Dataset"  # column used as batch key for scVI
DONOR_COL     = "Donor_id" # column used as covariate key for scVI
CELL_CYCLE_PHASE = 'phase' # column used as covariate key for scVI
MENSTRUAL_COL = "Menstrual_stage"


# ── HVG parameters ───────────────────────────────────────────────────────────
N_HVG      = 3000   # fewer cells → fewer HVGs to avoid over-fitting
HVG_FLAVOR = "seurat_v3"
HVG_BATCH_KEY = None

# ── scVI parameters ──────────────────────────────────────────────────────────
N_LATENT        = 60
N_LAYERS        = 1
GENE_LIKELIHOOD = "nb"
MAX_EPOCHS      = 200
EARLY_STOPPING  = True
DISPERSION      = 'gene-batch'

# ── Neighbours / UMAP parameters ─────────────────────────────────────────────
N_NEIGHBORS       = 20
LEIDEN_RESOLUTION = 2

# ── Misc ─────────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
N_CPUS      = 4


# ## Libraries


import warnings
warnings.filterwarnings("ignore")

import gc
import glob as glob_mod

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc
import scvi
from anndata.experimental import concat_on_disk

sc.settings.verbosity = 3
sc.settings.n_jobs    = N_CPUS
sc.set_figure_params(dpi=100, frameon=False, figsize=(6, 5))
ad.settings.allow_write_nullable_strings = True

print(f"anndata  : {ad.__version__}")
print(f"scanpy   : {sc.__version__}")
print(f"scvi     : {scvi.__version__}")


# ## 1 · Load annotation and identify mesenchymal barcodes


QUALITY_LABELS = {'unknown', 'lowqc', 'doublet', 'soup', 'donor_specific'}

def _is_quality_fail(series):
    return series.str.lower().isin(QUALITY_LABELS)

frames = []

# ── Shared annotation tables (loaded once) ────────────────────────────────────────────
ann = pd.read_csv(ANNOTATION_CONCAT, index_col="barcode")
ann = ann[ann[TISSUE_COL] == TISSUE_TARGET]

ann = ann[ann['fine_celltype'].notna()]
ann = ann[~ann['fine_celltype'].isin(QUALITY_LABELS)]

valid_barcodes = set(ann.index)

# check against cells to exclude
exclude = ann[ann['cell_to_exclude']].index.tolist()
print('valid barcodes before exclusion:', len(valid_barcodes))
print('example barcodes to exclude:', exclude[:5])
print('example barcodes in annotation:', list(valid_barcodes)[:5])

valid_barcodes = valid_barcodes - set(exclude)
print(f"Valid barcodes after exclusion: {len(valid_barcodes):,}")

print(f"Valid barcodes across all lineages: {len(valid_barcodes):,}")
print("\nLineage distribution:")
print(ann['lineage'].value_counts())
print("\nTop cell types:")
print(ann['fine_celltype'].value_counts().head(20))


# ## 2 · Subset individual h5ads to mesenchymal cells
# 
# Each uterus dataset is loaded individually, subset to mesenchymal barcodes, and saved to `SUBSET_DIR`.  
# This keeps peak memory proportional to a single dataset rather than the full atlas.


# ── Discover source uterus h5ads ──────────────────────────────────────────────
source_paths = sorted(glob_mod.glob(os.path.join(INPUT_DIR, "uterus_*.h5ad")))

if len(source_paths) == 0:
    raise FileNotFoundError(f"No uterus_*.h5ad files found in {INPUT_DIR}")

print(f"Source datasets ({len(source_paths)}):")
for p in source_paths:
    print(f"  {os.path.basename(p)}")


skipped = []

for src_path in source_paths:
    name     = os.path.splitext(os.path.basename(src_path))[0]
    out_path = os.path.join(SUBSET_DIR, f"{name}.h5ad")

    if os.path.exists(out_path):
        print(f"  {name}: subset already exists — skipping.")
        continue

    print(f"\n  {name}: loading …", end=" ", flush=True)
    adata = ad.read_h5ad(src_path)
    mask  = adata.obs_names.isin(valid_barcodes)
    n_valid = int(mask.sum())
    print(f"{adata.n_obs:,} cells → {n_valid:,} valid", end=" ", flush=True)

    if n_valid == 0:
        print("\n— no valid cells, skipping.")
        print('Example barcodes in adata.obs_names:', adata.obs_names[0],)
        skipped.append(name)
        del adata
        gc.collect()
        continue

    adata_sub = adata[mask].copy()
    del adata
    gc.collect()

    if adata_sub.n_obs == 0:
        print("\n— no cells remaining after filters, skipping.")
        skipped.append(name)
        del adata_sub
        gc.collect()
        continue

    adata_sub.write_h5ad(out_path)
    print(f"→ saved {name} ({adata_sub.n_obs:,} cells).")
    del adata_sub
    gc.collect()

print(f"\nDone. Skipped (no mesenchymal cells): {skipped if skipped else 'none'}")


# ## 3 · Concatenate subsetted files on disk


# ── Collect the subsetted h5ads ───────────────────────────────────────────────
subset_paths = sorted(glob_mod.glob(os.path.join(SUBSET_DIR, "*.h5ad")))

if len(subset_paths) == 0:
    raise FileNotFoundError(f"No subsetted h5ad files found in {SUBSET_DIR}")

in_files = {
    os.path.splitext(os.path.basename(p))[0]: p
    for p in subset_paths
}

print(f"Datasets to concatenate ({len(in_files)}):")
for k, v in in_files.items():
    print(f"  {k:50s}  {v}")


# ── Outer concat (union of genes) ────────────────────────────────────────────
if os.path.exists(CONCAT_H5AD_OUTER):
    print(f"Outer concat already exists: {CONCAT_H5AD_OUTER}\nSkipping.")
else:
    print("Concatenating (outer — union of genes) …")
    concat_on_disk(
        in_files = in_files,
        out_file = CONCAT_H5AD_OUTER,
        join     = "outer",
        label    = DATASET_COL,
    )
    print(f"Saved → {CONCAT_H5AD_OUTER}")

# ── Inner concat (intersection of genes) ─────────────────────────────────────
if os.path.exists(CONCAT_H5AD_INNER):
    print(f"Inner concat already exists: {CONCAT_H5AD_INNER}\nSkipping.")
else:
    print("Concatenating (inner — intersection of genes) …")
    concat_on_disk(
        in_files = in_files,
        out_file = CONCAT_H5AD_INNER,
        join     = "inner",
        label    = DATASET_COL,
    )
    print(f"Saved → {CONCAT_H5AD_INNER}")


print("Loading inner concat (intersection of genes) …")
adata_inner = ad.read_h5ad(CONCAT_H5AD_INNER)

# ── Guard against duplicate barcodes ─────────────────────────────────────────
dup_mask = adata_inner.obs_names.duplicated()
if dup_mask.any():
    n_dup = int(dup_mask.sum())
    print(f"WARNING: {n_dup} duplicate barcodes — deduplicating (keeping first).")
    adata_inner = adata_inner[~dup_mask].copy()
    
# # remove duplicate {sample}_{barcode}
# adata_inner.obs['sample_barcode'] = adata_inner.obs_names.map(
#     lambda x: x.rsplit('_', 1)[0] if x.count('_') > 1 else x
# )

# dup_mask = adata_inner.obs['sample_barcode'].duplicated()
# if dup_mask.any():
#     n_dup = int(dup_mask.sum())
#     print(f"WARNING: {n_dup} duplicate sample_barcode entries — deduplicating (keeping first).")
#     adata_inner = adata_inner[~dup_mask].copy()
#     del adata_inner.obs['sample_barcode']

# ── Fix sparse matrix dtype mismatch (concat_on_disk can mix int32/int64) ────
if sp.issparse(adata_inner.X) and adata_inner.X.indptr.dtype != adata_inner.X.indices.dtype:
    target_dtype = np.result_type(adata_inner.X.indptr.dtype, adata_inner.X.indices.dtype)
    print(f"Fixing sparse dtype mismatch: indptr={adata_inner.X.indptr.dtype}, "
          f"indices={adata_inner.X.indices.dtype} → {target_dtype}")
    adata_inner.X.indptr  = adata_inner.X.indptr.astype(target_dtype)
    adata_inner.X.indices = adata_inner.X.indices.astype(target_dtype)

print(f"inner : {adata_inner.n_obs:,} cells × {adata_inner.n_vars:,} genes")
print("\nobs columns:", adata_inner.obs.columns.tolist())


# ## 4 · Highly variable genes stratified by menstrual stage
adata_inner.obs = adata_inner.obs.join(ann[['lineage']])

sc.pp.filter_genes(adata_inner, min_cells=5)


adata_inner.obs["Menstrual_stage_short"] = (
    adata_inner.obs[MENSTRUAL_COL].astype(str).str.split(" ").str[0]
)
print(adata_inner.obs["Menstrual_stage_short"].value_counts())


print(f"Identifying {N_HVG} HVGs on inner concat "
      f"({adata_inner.n_vars:,} shared genes) …")

sc.pp.highly_variable_genes(
    adata_inner,
    n_top_genes = N_HVG,
    flavor      = HVG_FLAVOR,
    batch_key   = HVG_BATCH_KEY,
    subset      = False,
)

inner_genes_set = set(adata_inner.var_names)
hvg_set         = set(adata_inner.var_names[adata_inner.var.highly_variable])
print(f"HVGs selected: {len(hvg_set)}")
sc.pl.highly_variable_genes(adata_inner)


adata_hvg = adata_inner[:, adata_inner.var.highly_variable].copy()
del adata_inner
gc.collect()
print(f"adata_hvg (HVGs only): {adata_hvg}")


# ## 5 · Batch correction with scVI


adata_hvg.obs[DONOR_COL] = adata_hvg.obs[DONOR_COL].astype(str)


scvi.settings.seed = RANDOM_SEED

if DONOR_COL not in adata_hvg.obs.columns:
    raise ValueError(f"Column '{DONOR_COL}' not found. Available: {adata_hvg.obs.columns.tolist()}")

print(f"Number of donors: {adata_hvg.obs[DONOR_COL].nunique()}")
print(f"Number of datasets: {adata_hvg.obs[DATASET_COL].nunique()}")

scvi.model.SCVI.setup_anndata(
    adata_hvg,
    batch_key                  = DATASET_COL,
    categorical_covariate_keys = [DONOR_COL, CELL_CYCLE_PHASE,],
)

model = scvi.model.SCVI(
    adata_hvg,
    n_latent        = N_LATENT,
    n_layers        = N_LAYERS,
    gene_likelihood = GENE_LIKELIHOOD,
    dispersion      = DISPERSION,
)
print('scVI model', model)


model.train(
    max_epochs     = MAX_EPOCHS,
    early_stopping = EARLY_STOPPING,
)
model.save(SCVI_MODEL_DIR, overwrite=True)
print(f"Model saved → {SCVI_MODEL_DIR}")


X_scVI = model.get_latent_representation()

del adata_hvg
gc.collect()

print("Loading outer concat (union of genes) …")
adata = ad.read_h5ad(CONCAT_H5AD_OUTER)

# ── Guard against duplicate barcodes ─────────────────────────────────────────
dup_mask = adata.obs_names.duplicated()
if dup_mask.any():
    n_dup = int(dup_mask.sum())
    print(f"WARNING: {n_dup} duplicate barcodes — deduplicating (keeping first).")
    adata = adata[~dup_mask].copy()
    
# # remove duplicate {sample}_{barcode}
# adata.obs['sample_barcode'] = adata.obs_names.map(
#     lambda x: x.rsplit('_', 1)[0] if x.count('_') > 1 else x
# )

# dup_mask = adata.obs['sample_barcode'].duplicated()
# if dup_mask.any():
#     n_dup = int(dup_mask.sum())
#     print(f"WARNING: {n_dup} duplicate sample_barcode entries — deduplicating (keeping first).")
#     adata = adata[~dup_mask].copy()
#     del adata.obs['sample_barcode']

# ── Fix sparse matrix dtype mismatch ─────────────────────────────────────────
if sp.issparse(adata.X) and adata.X.indptr.dtype != adata.X.indices.dtype:
    target_dtype = np.result_type(adata.X.indptr.dtype, adata.X.indices.dtype)
    print(f"Fixing sparse dtype mismatch: indptr={adata.X.indptr.dtype}, "
          f"indices={adata.X.indices.dtype} → {target_dtype}")
    adata.X.indptr  = adata.X.indptr.astype(target_dtype)
    adata.X.indices = adata.X.indices.astype(target_dtype)

print(f"outer : {adata.n_obs:,} cells × {adata.n_vars:,} genes")

adata.obsm["X_scVI"]         = X_scVI
adata.var["in_inner"]        = adata.var_names.isin(inner_genes_set)
adata.var["in_hvg"]          = adata.var_names.isin(hvg_set)
adata.var["highly_variable"] = adata.var["in_hvg"]
adata.obs["Menstrual_stage_short"] = (
    adata.obs[MENSTRUAL_COL].astype(str).str.split(" ").str[0]
)
del X_scVI
print(f"X_scVI transferred: shape {adata.obsm['X_scVI'].shape}")
print(f"in_inner: {adata.var['in_inner'].sum():,} genes  |  in_hvg: {adata.var['in_hvg'].sum():,} genes")

model.history["elbo_train"].plot()
plt.title("ELBO - training")
plt.ylabel("ELBO")
plt.xlabel("Epoch")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "scvi_uterus_mesenchymal_elbo.png"), dpi=150)
plt.show()


# ## 6 · KNN graph, UMAP and Leiden clustering

_use_rapids = False
try:
    import rapids_singlecell as rsc
    import cupy as cp
    rsc.get.anndata_to_GPU(adata)
    rsc.pp.neighbors(
        adata,
        n_neighbors = N_NEIGHBORS+1,
        use_rep     = "X_scVI",
        algorithm   = "cagra",
        metric      = "euclidean",
    )
    rsc.get.anndata_to_CPU(adata)
    _use_rapids = True
    print("KNN computed with CAGRA (rapids_singlecell, GPU).")
except ImportError:
    print("rapids_singlecell not available — falling back to scanpy (pynndescent).")

if not _use_rapids:
    sc.pp.neighbors(
        adata,
        n_neighbors  = N_NEIGHBORS,
        use_rep      = "X_scVI",
        method       = "umap",
        metric       = "euclidean",
        random_state = RANDOM_SEED,
    )
    print("KNN computed with pynndescent (scanpy, CPU).")


sc.tl.umap(adata, random_state=RANDOM_SEED)
print("UMAP coordinates stored in adata.obsm['X_umap'].")


if _use_rapids:
    rsc.tl.leiden(
        adata,
        resolution   = LEIDEN_RESOLUTION,
        random_state = RANDOM_SEED,
        key_added    = "leiden",
    )
else:
    sc.tl.leiden(
        adata,
        resolution   = LEIDEN_RESOLUTION,
        random_state = RANDOM_SEED,
        key_added    = "leiden",
        flavor       = "igraph",
        n_iterations = 2,
        directed     = False,
    )
print(f"Leiden clusters (res={LEIDEN_RESOLUTION}): {adata.obs['leiden'].nunique()} clusters")
print(adata.obs["leiden"].value_counts().sort_index())


# ## 7 · Save integrated object


# ── Transfer celltype / lineage from combined annotation ─────────────────────
if 'lineage' in adata.obs.columns:
    del adata.obs['lineage']
adata.obs = adata.obs.join(ann[["fine_celltype", "broad_celltype", "lineage"]])

adata.write_h5ad(INTEGRATED_H5AD)
print(f"Integrated object saved → {INTEGRATED_H5AD}")
print(adata)


# adata = ad.read_h5ad(INTEGRATED_H5AD)


# ## 8 · UMAP visualisation


sc.pl.umap(
    adata,
    color           = DATASET_COL,
    title           = "UMAP - dataset",
    legend_loc      = "right margin",
    legend_fontsize = 7,
    save            = "_uterus_dataset.png",
)


sc.pl.umap(
    adata,
    color           = "Menstrual_stage_short",
    title           = "UMAP - menstrual stage",
    legend_loc      = "right margin",
    legend_fontsize = 8,
    save            = "_uterus_menstrual.png",
)


sc.pl.umap(
    adata,
    color           = "Tissue_ROI",
    title           = "UMAP - Tissue_ROI",
    legend_fontsize = 7,
    save            = "_uterus_tissue_roi.png",
)


sc.pl.umap(
    adata,
    color           = "leiden",
    title           = f"UMAP - Leiden (res={LEIDEN_RESOLUTION})",
    legend_loc      = "on data",
    legend_fontsize = 8,
    save            = "_uterus_leiden.png",
)


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

_coords   = adata.obsm["X_umap"]
_cat_cols = [DATASET_COL, DONOR_COL, "Menstrual_stage_short"]
_titles   = ["Dataset", "Donor", "Menstrual stage"]

for ax, col, title in zip(axes, _cat_cols, _titles):
    cats = adata.obs[col].astype("category").cat
    palette = plt.cm.get_cmap("tab20", len(cats.categories))
    for i, cat in enumerate(cats.categories):
        mask = adata.obs[col] == cat
        ax.scatter(
            _coords[mask, 0], _coords[mask, 1],
            s=0.5, alpha=0.5,
            color=palette(i),
            label=cat,
            rasterized=True,
        )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.axis("off")
    if len(cats.categories) <= 20:
        ax.legend(markerscale=8, fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "umap_uterus_panel.png"), dpi=150, bbox_inches="tight")
plt.show()



