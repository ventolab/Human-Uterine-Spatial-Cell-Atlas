"""
compute_gex_correlation.py

Compute GEX correlation between scRNA cell-type profiles and spatial
cell-type profiles predicted by each method.

Preprocessing (matches ISS-Patcher, applied after subsetting to shared genes):
  sc.pp.normalize_total -> sc.pp.log1p -> sc.pp.scale(max_value=10)
  Applied INDEPENDENTLY to scRNA and to each spatial donor.

Metric
------
  corr_matched         = mean over donors of Pearson(mean_sc[A], mean_sp[A])
  corr_random_baseline = mean over donors of
                         mean Pearson(mean_sc[A], mean_sp[B!=A]) for 5 random B

Only (method, donor) pairs where all five methods have outputs are included.
Cell types with fewer than MIN_CELLS_SP spatial predictions per donor are skipped.

Outputs  outputs/gex_correlation/
  per_donor.parquet    — raw per-donor correlations
      method, stage, donor, celltype, n_sp, corr, corr_random
  per_celltype.parquet — aggregated across donors (mean correlation)
      method, celltype, n_sc, n_donors, mean_n_sp,
      corr_matched, corr_random_baseline
  summary.csv          — method-level summary
      method, mean_corr_matched, mean_corr_random, n_celltypes

Usage
-----
python visualize/compute_gex_correlation.py
"""
import re
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scanpy as sc
import scipy.sparse as sps
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OUTPUT_DIR, SC_REF_PATH, SC_ANNOT_PATH, SPATIAL_FILES

# ── Parameters ────────────────────────────────────────────────────────────────
N_RANDOM_BASELINE  = 5
MIN_CELLS_SP       = 10
TACCO_SCORE_THRESH = 0.4
RANDOM_SEED        = 42

BAD_DATASETS = [
    "uterus_menopause_sanger-denoised",
    "uterus_adult_menstrualfluid_sanger-denoised",
]

METHODS = {
    "iss_full":  ("iss_patcher", "all_sc", "full"),
    "iss_ds1k":  ("iss_patcher", "all_sc", "downsampled_1k"),
    # "dot_ds1k":  ("dot",         "all_sc", "downsampled_1k"),
    # "dot_full":  ("dot",         "all_sc", "full"),
    # "dot_ds":    ("dot",         "all_sc", "downsampled"),
    "tacco":     ("tacco",       "all_sc", "full"),
}

OUT_DIR = OUTPUT_DIR / "gex_correlation"


# ── Preprocessing ─────────────────────────────────────────────────────────────

def _preprocess(X_counts, gene_names):
    """normalize_total -> log1p -> scale(max_value=10). Returns dense ndarray."""
    adata = ad.AnnData(X=X_counts.astype(np.float32))
    adata.var_names = gene_names
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=10)
    return np.asarray(adata.X, dtype=np.float32)


# ── Donor / method discovery ──────────────────────────────────────────────────

def _donor_spatial_map():
    result = {}
    for sample_id, path_str in SPATIAL_FILES.items():
        p = Path(path_str)
        m = re.match(r"(.+)_annotated", p.stem)
        if m:
            donor = m.group(1)
            stage = re.sub(r"_\d+$", "", sample_id)
            result[donor] = (p, stage)
    return result


def _method_donors(method_key):
    base, config, label = METHODS[method_key]
    if base == "iss_patcher":
        donors  = {p.parts[-2] for p in OUTPUT_DIR.glob(
            f"{base}/{config}/{label}/*/*/iss_annots.parquet")}
        donors |= {p.parts[-2] for p in OUTPUT_DIR.glob(
            f"{base}/{config}/{label}/*/*/iss_patched.h5ad")}
    elif base == "dot":
        donors = {p.parts[-2] for p in OUTPUT_DIR.glob(
            f"{base}/{config}/{label}/*/*/dot_predictions.csv")}
    else:
        donors = {p.parts[-2] for p in OUTPUT_DIR.glob(
            f"{base}/{config}/{label}/*/*/tacco_predictions.csv")}
    return donors


def _common_donors(dsmap):
    print(f"Donors from SPATIAL_FILES: {sorted(dsmap)}", flush=True)
    method_sets = {}
    for m in METHODS:
        s = _method_donors(m)
        method_sets[m] = s
        print(f"  {m:12s}: {sorted(s)}", flush=True)
    common = set(dsmap) & set.intersection(*method_sets.values())
    print(f"Common donors ({len(common)}): {sorted(common)}")
    return {d: dsmap[d] for d in common}


# ── scRNA reference ───────────────────────────────────────────────────────────

def _load_sc_reference(sp_genes):
    """
    Load filtered scRNA, subset to shared genes, preprocess.
    Returns (sc_means, sc_counts, genes_ord).
      sc_means  : {celltype: mean_scaled_vector}
      sc_counts : {celltype: n_cells}
      genes_ord : list of shared gene names (in scRNA var order)
    """
    print("\nLoading scRNA reference ...", flush=True)
    raw = ad.read_h5ad(SC_REF_PATH, backed="r")
    ann = pd.read_csv(SC_ANNOT_PATH, index_col=0)[["fine_celltype"]]

    valid_mask = ~raw.obs["dataset"].isin(BAD_DATASETS)
    valid_obs  = raw.obs_names[valid_mask]

    sp_gene_set = set(sp_genes)
    gene_idx    = [i for i, g in enumerate(raw.var_names) if g in sp_gene_set]
    genes_ord   = [raw.var_names[i] for i in gene_idx]
    print(f"  {len(valid_obs):,} cells | {len(genes_ord)} shared genes", flush=True)

    # Load into memory in chunks (backed random-access is slow)
    sc_pos    = {b: i for i, b in enumerate(raw.obs_names)}
    valid_idx = np.array([sc_pos[b] for b in valid_obs])
    chunks    = []
    n         = len(valid_idx)
    for s in range(0, n, 50_000):
        e   = min(s + 50_000, n)
        blk = raw.X[valid_idx[s:e]][:, gene_idx]
        if not sps.issparse(blk):
            blk = sps.csr_matrix(blk)
        chunks.append(blk.astype(np.float32))
    X = sps.vstack(chunks, format="csr")

    ct_labels = ann.reindex(valid_obs)["fine_celltype"].values
    has_label = ~pd.isna(ct_labels)
    X         = X[has_label]
    ct_labels = ct_labels[has_label].astype(str)
    print(f"  {X.shape[0]:,} labelled cells | {len(np.unique(ct_labels))} cell types",
          flush=True)

    print("  Preprocessing: normalize_total -> log1p -> scale(max_value=10) ...",
          flush=True)
    X_scaled = _preprocess(X, genes_ord)

    sc_means  = {ct: X_scaled[ct_labels == ct].mean(axis=0)
                 for ct in np.unique(ct_labels)}
    sc_counts = {ct: int((ct_labels == ct).sum())
                 for ct in np.unique(ct_labels)}
    return sc_means, sc_counts, genes_ord


# ── Spatial loading ───────────────────────────────────────────────────────────

def _load_scale_spatial(sp_path, genes_ord):
    """
    Load all spatial cells, subset to shared genes (in genes_ord order),
    preprocess independently.
    Returns (X_scaled, obs_names_array, shared_sc_idx).
      shared_sc_idx : positions in genes_ord present in spatial var_names
    """
    sp_adata    = ad.read_h5ad(sp_path, backed="r")
    sp_gene_map = {g: i for i, g in enumerate(sp_adata.var_names)}

    shared_sc_idx = [i          for i, g in enumerate(genes_ord) if g in sp_gene_map]
    shared_sp_idx = [sp_gene_map[g] for g in genes_ord            if g in sp_gene_map]
    shared_genes  = [genes_ord[i] for i in shared_sc_idx]

    X_sp = sp_adata.layers["counts"][:, shared_sp_idx]
    if not sps.issparse(X_sp):
        X_sp = sps.csr_matrix(X_sp)

    print(f"    Preprocessing spatial ({X_sp.shape[0]:,} cells × "
          f"{len(shared_genes)} genes) ...", flush=True)
    X_scaled = _preprocess(X_sp, shared_genes)
    return X_scaled, np.array(sp_adata.obs_names), shared_sc_idx


# ── Prediction loaders ────────────────────────────────────────────────────────

def _load_predictions(method_key, stage, donor):
    """Returns Series(index=barcode, values=fine_celltype or NaN)."""
    base, config, label = METHODS[method_key]
    d = OUTPUT_DIR / base / config / label / stage / donor

    if base == "iss_patcher":
        parquet = d / "iss_annots.parquet"
        h5ad    = d / "iss_patched.h5ad"
        if parquet.exists():
            return pd.read_parquet(parquet).set_index("barcode")["fine_celltype"]
        elif h5ad.exists():
            return ad.read_h5ad(h5ad, backed="r").obs["fine_celltype"]
        return pd.Series(dtype=str)

    elif base == "dot":
        csv = d / "dot_predictions.csv"
        if not csv.exists():
            return pd.Series(dtype=str)
        return pd.read_csv(csv, index_col=0)["celltype"]

    else:  # tacco
        csv    = d / "tacco_predictions.csv"
        scores = d / "tacco_scores.parquet"
        if not csv.exists():
            return pd.Series(dtype=str)
        preds = pd.read_csv(csv).set_index("cell_id")["celltype"].copy()
        if scores.exists():
            sc_df = pq.read_table(scores).to_pandas(ignore_metadata=True).set_index("cell_id")
            low   = sc_df.max(axis=1) <= TACCO_SCORE_THRESH
            preds[low.reindex(preds.index, fill_value=False)] = np.nan
        return preds


# ── Random baseline ───────────────────────────────────────────────────────────

def _random_baseline(sc_means_sub, sp_means, rng):
    cts    = list(sp_means)
    result = {}
    for ct in cts:
        if ct not in sc_means_sub:
            continue
        others  = [x for x in cts if x != ct]
        if not others:
            continue
        sampled = rng.choice(others, size=min(N_RANDOM_BASELINE, len(others)), replace=False)
        result[ct] = float(np.mean([pearsonr(sc_means_sub[ct], sp_means[b])[0]
                                    for b in sampled]))
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rng    = np.random.default_rng(RANDOM_SEED)
    dsmap  = _donor_spatial_map()
    common = _common_donors(dsmap)

    # Use any spatial file to determine shared gene list
    sp_genes   = list(ad.read_h5ad(next(iter(common.values()))[0], backed="r").var_names)
    sc_means, sc_counts, genes_ord = _load_sc_reference(sp_genes)

    per_donor_rows = []

    for donor, (sp_path, stage) in sorted(common.items()):
        print(f"\n== {donor}  ({stage}) ==", flush=True)

        # Load and scale spatial once per donor (reused across all methods)
        X_sp, sp_obs, shared_sc_idx = _load_scale_spatial(sp_path, genes_ord)
        sp_obs_index = pd.Index(sp_obs)

        # Subset scRNA means to shared genes for this donor
        sc_means_sub = {ct: v[shared_sc_idx] for ct, v in sc_means.items()}

        for method_key in METHODS:
            print(f"  [{method_key}]", flush=True)
            preds = _load_predictions(method_key, stage, donor).dropna()
            if preds.empty:
                print("    no predictions -- skip", flush=True)
                continue

            # Align predictions to spatial obs order
            in_sp    = sp_obs_index.isin(preds.index)
            cell_idx = np.where(in_sp)[0]
            if len(cell_idx) == 0:
                continue
            ct_sp = preds.reindex(sp_obs[cell_idx]).values.astype(str)

            # Spatial mean per predicted cell type
            sp_means = {
                ct: X_sp[cell_idx][ct_sp == ct].mean(axis=0)
                for ct in np.unique(ct_sp)
                if (ct_sp == ct).sum() >= MIN_CELLS_SP
            }
            if not sp_means:
                continue

            baseline = _random_baseline(sc_means_sub, sp_means, rng)

            n_types = 0
            for ct, sp_vec in sp_means.items():
                if ct not in sc_means_sub:
                    continue
                r, _ = pearsonr(sc_means_sub[ct], sp_vec)
                per_donor_rows.append({
                    "method":      method_key,
                    "stage":       stage,
                    "donor":       donor,
                    "celltype":    ct,
                    "n_sp":        int((ct_sp == ct).sum()),
                    "corr":        float(r),
                    "corr_random": baseline.get(ct, np.nan),
                })
                n_types += 1

            mean_r = np.mean([r["corr"] for r in per_donor_rows[-n_types:]])
            print(f"    {n_types} types | mean corr = {mean_r:.3f}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Save per-donor raw results ────────────────────────────────────────────
    df_donor = pd.DataFrame(per_donor_rows)
    donor_path = OUT_DIR / "per_donor.parquet"
    df_donor.to_parquet(donor_path, index=False)
    print(f"\nSaved {len(df_donor):,} rows -> {donor_path}")

    # ── Aggregate across donors per (method, celltype) ────────────────────────
    per_ct = (
        df_donor.groupby(["method", "celltype"])
        .agg(
            n_donors             =("donor",       "nunique"),
            mean_n_sp            =("n_sp",        "mean"),
            corr_matched         =("corr",        "mean"),
            corr_random_baseline =("corr_random", "mean"),
        )
        .reset_index()
    )
    per_ct["n_sc"] = per_ct["celltype"].map(sc_counts).fillna(0).astype(int)

    ct_path = OUT_DIR / "per_celltype.parquet"
    per_ct.to_parquet(ct_path, index=False)
    print(f"Saved {len(per_ct):,} rows -> {ct_path}")

    # ── Method-level summary ──────────────────────────────────────────────────
    summary = (
        per_ct.groupby("method")
        .agg(
            mean_corr_matched         =("corr_matched",         "mean"),
            mean_corr_random_baseline =("corr_random_baseline", "mean"),
            n_celltypes               =("celltype",             "count"),
        )
        .reset_index()
        .sort_values("mean_corr_matched", ascending=False)
    )
    summary_path = OUT_DIR / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary -> {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
