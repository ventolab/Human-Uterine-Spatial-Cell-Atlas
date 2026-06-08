# Assign TACCO-predicted axis bins and create pseudobulk profiles for mesenchymal
# cells, to be used for gradient modelling in R.
import scanpy as sc
import numpy as np
import pandas as pd
from pathlib import Path

# TACCO bin ordering (basalis → lumen)
BIN_ORDER = [
    "basalis_1", "basalis_2",
    "functionalis_1", "functionalis_2", "functionalis_3",
    "lumen_1",
]
N_BINS    = len(BIN_ORDER)
bin_to_idx = {b: i for i, b in enumerate(BIN_ORDER)}

# ── Load GEX ─────────────────────────────────────────────────────────────────

SCRNA_H5AD = "/lustre/scratch125/cellgen/vento/mm58/eutopic_endometrium/build_eutopic_object/integrated_scvi_uterus.h5ad"

adata = sc.read_h5ad(SCRNA_H5AD)

adata_mesenchyme = adata[adata.obs['fine_celltype'].str.contains('stroma', case = False, na = False)]

adata_mesenchyme.obs["sample"] = adata_mesenchyme.obs["Donor_id"].astype(str)

# ── Load TACCO predictions and merge ─────────────────────────────────────────

TACCO_DIR  = Path("../tacco_annotation_ensembl")
pred_files = sorted(TACCO_DIR.glob("*/tacco_predictions.csv"))
print(f"Found {len(pred_files)} TACCO prediction files: "
      f"{[f.parent.name for f in pred_files]}")

tacco_df = pd.concat(
    [pd.read_csv(f) for f in pred_files], ignore_index=True
).set_index("cell_id")
print(f"Loaded {len(tacco_df):,} TACCO predictions")

adata_mesenchyme.obs = adata_mesenchyme.obs.join(tacco_df[["axis_bin"]], how="left")
# Drop cells without a TACCO prediction
n_before = adata_mesenchyme.shape[0]
adata_mesenchyme = adata_mesenchyme[adata_mesenchyme.obs["axis_bin"].notna()].copy()
print(f"Dropped {n_before - adata_mesenchyme.shape[0]:,} cells without TACCO predictions; "
      f"{adata_mesenchyme.shape[0]:,} cells retained")

# Enforce ordered categorical so groupby respects anatomical direction
adata_mesenchyme.obs["axis_bin"] = pd.Categorical(
    adata_mesenchyme.obs["axis_bin"], categories=BIN_ORDER, ordered=True
)

# ── Pseudobulk per sample × axis_bin ─────────────────────────────────────────

MIN_CELLS_PER_BIN  = 25
MIN_BINS_PER_DONOR = 3

groups = adata_mesenchyme.obs.groupby(["sample", "axis_bin"], observed=True)
pb_counts, pb_meta, col_names = [], [], []

for (sid, binn), group_df in groups:
    raw    = adata_mesenchyme[group_df.index].X
    counts = np.asarray(raw.sum(axis=0)).flatten()
    pb_counts.append(counts)

    col_names.append(f"{sid}_{binn}")
    pb_meta.append({
        "sample"         : sid,
        "axis_bin"     : str(binn),
        "axis_bin_mid" : (bin_to_idx[str(binn)] + 0.5) / N_BINS,
        "menstrual_phase": group_df["Menstrual_stage_short"].iloc[0],
        "n_cells"        : len(group_df),
    })

pb_meta   = pd.DataFrame(pb_meta)
pb_matrix = np.vstack(pb_counts).T    # genes × pseudobulk_samples

# Filter bins with too few cells
n_before  = len(pb_meta)
keep_bins = pb_meta["n_cells"] >= MIN_CELLS_PER_BIN
pb_meta   = pb_meta[keep_bins].reset_index(drop=True)
pb_matrix = pb_matrix[:, keep_bins.values]
col_names = [c for c, k in zip(col_names, keep_bins) if k]
print(f"Removed {n_before - len(pb_meta)} bins with < {MIN_CELLS_PER_BIN} cells; "
      f"{len(pb_meta)} bins retained")

# Remove donors with too few bins remaining
donor_bin_counts_pre = pb_meta.groupby("sample")["axis_bin"].count()
donors_keep          = donor_bin_counts_pre[donor_bin_counts_pre >= MIN_BINS_PER_DONOR].index
n_donors_removed     = (donor_bin_counts_pre < MIN_BINS_PER_DONOR).sum()
keep_donors          = pb_meta["sample"].isin(donors_keep)
pb_meta   = pb_meta[keep_donors].reset_index(drop=True)
pb_matrix = pb_matrix[:, keep_donors.values]
col_names = [c for c, k in zip(col_names, keep_donors) if k]
print(f"Removed {n_donors_removed} donors with < {MIN_BINS_PER_DONOR} bins; "
      f"{pb_meta['sample'].nunique()} donors retained")


# ── QC summary ────────────────────────────────────────────────────────────────

print("=" * 70)
print("QC SUMMARY")
print("=" * 70)

print("\n[1] Menstrual stage distribution (pseudobulk observations)")
print(pb_meta["menstrual_phase"].value_counts().to_string())

donor_bin_counts = pb_meta.groupby("sample")["axis_bin"].count().rename("n_bins")
donor_phase      = pb_meta.groupby("sample")["menstrual_phase"].first()
donor_qc         = pd.concat([donor_bin_counts, donor_phase], axis=1)

print("\n[2] Bin coverage per donor")
print(f"  Donors total:               {donor_qc.shape[0]}")
print(f"  Donors with all {N_BINS} bins:      {(donor_qc['n_bins'] == N_BINS).sum()}")
print(f"  Donors with <5 bins:        {(donor_qc['n_bins'] <  5).sum()}")
print(f"  Donors with <3 bins:        {(donor_qc['n_bins'] <  3).sum()}")
print(f"  Donors with 1 bin:          {(donor_qc['n_bins'] == 1).sum()}")
print(f"\n  Distribution of n_bins per donor:")
print(donor_qc["n_bins"].value_counts().sort_index().to_string())

print("\n[3] Median bin coverage per donor, by menstrual phase")
phase_bin_summary = (
    donor_qc.groupby("menstrual_phase")["n_bins"]
    .agg(n_donors="count", median_bins="median", mean_bins="mean",
         min_bins="min", max_bins="max")
    .round(2)
)
print(phase_bin_summary.to_string())

sparse_donors = donor_qc[donor_qc["n_bins"] < 3]
if len(sparse_donors) > 0:
    print(f"\n  *** WARNING: {len(sparse_donors)} donors have <3 bins. "
          f"Phase breakdown of sparse donors:")
    print(sparse_donors["menstrual_phase"].value_counts().to_string())
else:
    print("\n  No donors with <3 bins detected.")

print("\n[4] Cells per bin-donor pseudobulk (n_cells)")
print(pb_meta["n_cells"].describe().round(1).to_string())
n_low = (pb_meta["n_cells"] < MIN_CELLS_PER_BIN).sum()
print(f"\n  Pseudobulks with <{MIN_CELLS_PER_BIN} cells: {n_low} "
      f"({100 * n_low / len(pb_meta):.1f}%) [should be 0 after filtering]")

print("\n[5] Mean cells per bin across the axis")
axis_cell_means = (
    pb_meta.groupby("axis_bin")["n_cells"]
    .agg(mean="mean", median="median", std="std", n_pseudobulks="count")
    .reindex(BIN_ORDER)
    .round(1)
)
print(axis_cell_means.to_string())

cv_across_bins = (
    pb_meta.groupby("axis_bin")["n_cells"].mean().std()
    / pb_meta.groupby("axis_bin")["n_cells"].mean().mean()
)
print(f"\n  CV of mean n_cells across bins: {cv_across_bins:.3f}")
if cv_across_bins > 0.3:
    print("  *** WARNING: High CV suggests strong axis-wise cell density gradient.")

pb_meta = pb_meta.merge(
    donor_bin_counts.reset_index().rename(columns={"sample": "sample"}),
    on="sample", how="left"
)
pb_meta.rename(columns={"n_bins": "donor_n_bins"}, inplace=True)

print("\n[6] Extended metadata columns saved:")
print(f"  {list(pb_meta.columns)}")
print("\n" + "=" * 70)
print("END QC SUMMARY")
print("=" * 70)

# ── Save for R ────────────────────────────────────────────────────────────────

gene_names = adata_mesenchyme.var_names
pb_df      = pd.DataFrame(pb_matrix, index=gene_names, columns=col_names)
pb_df.to_csv("mesenchymal_pb_matrix_tacco_ensemble.tsv", sep="\t")
pb_meta.to_csv("mesenchymal_pb_meta_tacco_ensemble.tsv", sep="\t", index=False)

donor_qc_out  = donor_qc.reset_index().rename(columns={"sample": "donor"})
cells_per_bin = pb_meta[["sample", "axis_bin", "n_cells"]].rename(columns={"sample": "donor"})
donor_qc_out  = donor_qc_out.merge(cells_per_bin, on="donor", how="left")
donor_qc_out.to_csv("mesenchymal_pb_donor_qc_tacco_ensemble.tsv", sep="\t", index=False)

print("\nOutputs written:")
print("  mesenchymal_pb_matrix_tacco.tsv")
print("  mesenchymal_pb_meta_tacco.tsv  (includes donor_n_bins column)")
print("  mesenchymal_pb_donor_qc_tacco.tsv")
