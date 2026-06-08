import os
import scanpy as sc
import anndata as ad
import scvi
import matplotlib.pyplot as plt
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_H5AD   = '/nfs/team292/projects/PanTissue/results/temp/anndata_copy_freeze/uterus_adult_menstrualfluid_sanger-denoised.h5ad'
OUTPUT_DIR   = '/lustre/scratch125/cellgen/vento/mm58/eutopic_endometrium/flex_umap'
ANNOTATIONS_CSV = "/nfs/team292/projects/PanTissue/results/freeze/annotations/concatenated_annotations_postnatal_v2.csv"
OUTPUT_H5AD  = os.path.join(OUTPUT_DIR, 'scvi_output.h5ad')
MODEL_DIR    = os.path.join(OUTPUT_DIR, 'scvi_model')

BATCH_KEY    = 'Donor_id'
N_HVG        = 2000
N_LATENT     = 30
N_LAYERS     = 2
MAX_EPOCHS   = 200
RANDOM_SEED  = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
scvi.settings.seed = RANDOM_SEED

# ── Load data ─────────────────────────────────────────────────────────────────
adata = sc.read_h5ad(INPUT_H5AD)
print(f"Loaded: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

# Add annotations
annot = pd.read_csv(ANNOTATIONS_CSV, index_col=0)
annot = annot[['lineage', 'fine_celltype', 'broad_celltype', 'cell_to_exclude']]
adata.obs = adata.obs.join(annot)

# Remove low-quality cells and those without annotation
adata = adata[~adata.obs['cell_to_exclude']]
adata = adata[~adata.obs['fine_celltype'].isin(['lowQC', 'nan', 'unknown', 'soup', 'donor_specific'])]

# ── HVGs ──────────────────────────────────────────────────────────────────────
sc.pp.highly_variable_genes(
    adata,
    n_top_genes = N_HVG,
    flavor      = 'seurat_v3',
    batch_key   = BATCH_KEY,
    subset      = False,
)
print(f"HVGs: {adata.var.highly_variable.sum()}")

adata_hvg = adata[:, adata.var.highly_variable].copy()

# ── scVI ──────────────────────────────────────────────────────────────────────
scvi.model.SCVI.setup_anndata(adata_hvg, batch_key=BATCH_KEY)

model = scvi.model.SCVI(
    adata_hvg,
    n_latent        = N_LATENT,
    n_layers        = N_LAYERS,
    gene_likelihood = 'nb',
)
print(model)

model.train(max_epochs=MAX_EPOCHS, early_stopping=True)
model.save(MODEL_DIR, overwrite=True)

model.history['elbo_train'].plot()
plt.title('ELBO - training')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'scvi_elbo.png'), dpi=150)
plt.close()

# ── Transfer latent to full object ────────────────────────────────────────────
adata.obsm['X_scVI'] = model.get_latent_representation()

# ── Neighbors, UMAP, Leiden ───────────────────────────────────────────────────
sc.pp.neighbors(adata, use_rep='X_scVI', n_neighbors=20, random_state=RANDOM_SEED)
sc.tl.umap(adata, random_state=RANDOM_SEED)

# ── Save ──────────────────────────────────────────────────────────────────────
adata.write_h5ad(OUTPUT_H5AD)
print(f"Saved → {OUTPUT_H5AD}")
