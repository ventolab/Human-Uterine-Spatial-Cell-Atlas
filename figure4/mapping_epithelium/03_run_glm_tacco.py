"""
Mixed linear model analysis of pseudobulk gene expression along TACCO-predicted axis bins.

Models (ML estimation, random intercept per donor):
  M_reduced: expr ~ axis_bin + C(menstrual_phase),           groups = sample
  M_full:    expr ~ axis_bin * C(menstrual_phase),           groups = sample

axis_bin is encoded as an integer 1–6 (basalis_1=1 … lumen_1=6) so the
axis_bin coefficient is interpretable as a per-bin monotonic slope.

Test A (axis gradient):      Wald p-value for axis_bin from M_reduced
                             beta_axis = slope (log2-CPM per bin, basalis→lumen)
Test B (phase×axis):         LRT comparing M_full vs M_reduced

Gene selection for individual-gene and matrix plots:
  - FDR < FDR_THRESH
  - Top N genes by largest +ve beta  (luminal-high)
  - Top N genes by largest -ve beta  (basalis-high)
  - Dynamic range (max − min mean log2-CPM across bins) > MIN_CPM_RANGE

Output:
  glm_outputs_tacco[_<phases>]/glm_full_results.tsv
  glm_outputs_tacco[_<phases>]/TestA_axis_genes.tsv
  glm_outputs_tacco[_<phases>]/TestB_interact_genes.tsv
  glm_outputs_tacco[_<phases>]/figures/
"""

import os
import random
import warnings
from scipy import stats

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm


# ── 0. Configuration ──────────────────────────────────────────────────────────

def set_seed(seed=0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

set_seed(0)

BIN_ORDER = [
    "basalis_1", "basalis_2",
    "functionalis_1", "functionalis_2", "functionalis_3",
    "lumen_1",
]
BIN_NUMERIC = {b: i + 1 for i, b in enumerate(BIN_ORDER)}

MATRIX_FILE = "epithelial_pb_matrix_tacco_ensemble.tsv"
META_FILE   = "epithelial_pb_meta_tacco_ensemble.tsv"

CONDITION = "menstrual_phase"
GROUPING  = "axis_bin"
BATCH     = "sample"

PHASE_ORDER  = ["Menstrual", "Proliferative", "Secretory", "Hormones"]
PHASE_COLORS = {
    "Menstrual"    : "indianred",
    "Proliferative": "lightblue",
    "Secretory"    : "navy",
    "Hormones"     : "slategrey",
}

PHASES_TO_INCLUDE = ["Proliferative", "Secretory"]

if len(PHASES_TO_INCLUDE) == 4:
    OUTPUT_DIR = "glm_outputs_tacco_ensemble"
else:
    OUTPUT_DIR = "glm_outputs_tacco_ensemble_" + "_".join(PHASES_TO_INCLUDE)

REUSE_RESULTS = False

MIN_CPM       = 1
MIN_SAMPLES   = 10
PRIOR_COUNT   = 1
VAR_QUANTILE  = 0.25
FDR_THRESH    = 0.05
N_TOP         = 10
MIN_CPM_RANGE = 4.0   # minimum dynamic range in log2-CPM units for plot gene selection

os.makedirs(f"{OUTPUT_DIR}/figures", exist_ok=True)


# ── 1. Load and preprocess ────────────────────────────────────────────────────

def load_pseudobulk(matrix_file, meta_file, prior_count=1):
    print("Loading pseudobulk data...")
    counts = pd.read_csv(matrix_file, sep="\t", index_col=0)
    meta   = pd.read_csv(meta_file,   sep="\t")

    lib_sizes = counts.sum(axis=0)
    cpm       = counts.div(lib_sizes, axis=1) * 1e6
    logcpm    = np.log2(cpm + prior_count)

    meta[GROUPING] = pd.Categorical(
        meta[GROUPING], categories=BIN_ORDER, ordered=True
    )

    print(f"  {counts.shape[0]} genes × {counts.shape[1]} pseudobulk samples loaded")
    print(f"  Menstrual phase distribution:\n"
          f"{meta['menstrual_phase'].value_counts().to_string()}\n")
    return logcpm, meta


def filter_genes(logcpm, min_cpm=1, min_samples=10, var_quantile=0.25):
    cpm_thresh = np.log2(min_cpm + 1)
    expressed  = (logcpm >= cpm_thresh).sum(axis=1)
    logcpm     = logcpm[expressed >= min_samples]
    print(f"  {logcpm.shape[0]} genes after low-expression filter "
          f"(CPM ≥ {min_cpm} in ≥ {min_samples} samples)")

    mt_mask = logcpm.index.str.upper().str.startswith("MT-")
    logcpm  = logcpm[~mt_mask]
    print(f"  Removed {mt_mask.sum()} mitochondrial genes")

    ribo_mask = logcpm.index.str.upper().str.match(r"^RP[LS]\d")
    logcpm    = logcpm[~ribo_mask]
    print(f"  Removed {ribo_mask.sum()} ribosomal genes")

    pseudo_mask = logcpm.index.str.match(r".*\.\d+$")
    logcpm      = logcpm[~pseudo_mask]
    print(f"  Removed {pseudo_mask.sum()} likely pseudogenes")

    gene_var   = logcpm.var(axis=1)
    var_thresh = gene_var.quantile(var_quantile)
    var_mask   = gene_var >= var_thresh
    logcpm     = logcpm[var_mask]
    print(f"  Removed {(~var_mask).sum()} low-variance genes "
          f"(bottom {int(var_quantile*100)}%, threshold={var_thresh:.3f})")

    print(f"  {logcpm.shape[0]} genes retained for modelling\n")
    return logcpm


# ── 2. Helpers ────────────────────────────────────────────────────────────────

def _bin_order_present(meta, grouping):
    present = set(meta[grouping].astype(str).unique())
    return [b for b in BIN_ORDER if b in present]


# ── 3. Mixed linear models ────────────────────────────────────────────────────

def _fit_mixedlm(formula, data, groups):
    """Fit MixedLM with fallback optimisers; returns result even if not converged."""
    md = mixedlm(formula, data=data, groups=groups)
    for method, maxiter in [("lbfgs", 300), ("nm", 1000), ("powell", 1000)]:
        try:
            res = md.fit(reml=False, method=method, maxiter=maxiter)
            if res.converged:
                return res
        except Exception:
            continue
    return md.fit(reml=False, method="nm", maxiter=2000)


def run_glm(logcpm, meta, condition, grouping, batch, genes=None):
    """
    Per-gene random-intercept mixed linear model.

    M_reduced: expr ~ axis_bin_num + C(condition), groups=batch
    M_full:    expr ~ axis_bin_num * C(condition), groups=batch

    Returns DataFrame with columns:
      gene, beta_axis, p_axis (Test A Wald), p_interact (Test B LRT), converged
    """
    if genes is None:
        genes = logcpm.index.tolist()

    base = meta[[condition, grouping, batch]].copy().reset_index(drop=True)
    # astype(str) before map ensures Categorical dtype doesn't make patsy treat this as a factor
    base["axis_bin_num"] = base[grouping].astype(str).map(BIN_NUMERIC).astype(float)

    formula_red  = f"expr ~ axis_bin_num + C({condition})"
    formula_full = f"expr ~ axis_bin_num * C({condition})"

    results = []
    warnings.filterwarnings("ignore")

    for gene in tqdm(genes, desc="GLM"):
        df         = base.copy()
        df["expr"] = logcpm.loc[gene].values

        try:
            res_red  = _fit_mixedlm(formula_red,  df, df[batch])
            res_full = _fit_mixedlm(formula_full, df, df[batch])

            beta_axis = res_red.fe_params.get("axis_bin_num", np.nan)
            p_axis    = res_red.pvalues.get("axis_bin_num", np.nan)

            lrt_stat = 2 * (res_full.llf - res_red.llf)
            df_diff  = len(res_full.params) - len(res_red.params)
            if df_diff > 0 and np.isfinite(lrt_stat) and lrt_stat >= 0:
                p_interact = stats.chi2.sf(lrt_stat, df_diff)
            else:
                p_interact = np.nan

            results.append({
                "gene"      : gene,
                "beta_axis" : beta_axis,
                "p_axis"    : p_axis,
                "p_interact": p_interact,
                "converged" : bool(res_red.converged and res_full.converged),
            })

        except Exception:
            results.append({
                "gene"      : gene,
                "beta_axis" : np.nan,
                "p_axis"    : np.nan,
                "p_interact": np.nan,
                "converged" : False,
            })

    warnings.filterwarnings("default")
    return pd.DataFrame(results)


# ── 4. FDR correction ─────────────────────────────────────────────────────────

def apply_fdr(glm_df, fdr_level=0.05):
    df = glm_df.copy()
    for p_col, fdr_col, sig_col in [
        ("p_axis",     "FDR_A", "sig_A"),
        ("p_interact", "FDR_B", "sig_B"),
    ]:
        valid = df[p_col].notna()
        fdr   = np.full(len(df), np.nan)
        if valid.sum() > 0:
            _, fdr_vals, _, _ = multipletests(
                df.loc[valid, p_col], method="fdr_bh", alpha=fdr_level
            )
            fdr[valid.values] = fdr_vals
        df[fdr_col] = fdr
        df[sig_col] = df[fdr_col] < fdr_level
    return df


# ── 5. Dynamic range & gene selection ─────────────────────────────────────────

def compute_dynamic_range(logcpm, meta, genes, grouping):
    """Max – min mean log2-CPM across axis bins (all samples pooled per bin)."""
    bin_order = _bin_order_present(meta, grouping)
    genes     = [g for g in genes if g in logcpm.index]

    profile = pd.DataFrame(index=genes, columns=bin_order, dtype=float)
    for b in bin_order:
        idx = meta[meta[grouping].astype(str) == b].index
        if len(idx) > 0:
            cols       = logcpm.columns[idx]
            profile[b] = logcpm.loc[genes, cols].mean(axis=1).values
        else:
            profile[b] = np.nan

    return (profile.max(axis=1) - profile.min(axis=1)).rename("dynamic_range")


def select_plot_genes(glm_fdr, drange, fdr_col, fdr_thresh, min_range, n_top):
    """
    Return (pos_genes, neg_genes): top N significant genes by +ve and -ve
    beta_axis that also pass the dynamic-range filter.
    """
    df  = glm_fdr.copy()
    df  = df.merge(drange.rename("dynamic_range"), left_on="gene",
                   right_index=True, how="left")
    sig = df[(df[fdr_col] < fdr_thresh) & (df["dynamic_range"] > min_range)]

    pos_genes = sig.nlargest(n_top,  "beta_axis")["gene"].tolist()
    neg_genes = sig.nsmallest(n_top, "beta_axis")["gene"].tolist()
    return pos_genes, neg_genes


# ── 6. Plotting ───────────────────────────────────────────────────────────────

def build_plot_df(logcpm, meta, genes, grouping, condition):
    df = logcpm.loc[logcpm.index.isin(genes)].T.copy()
    df.index      = meta.index
    df[grouping]  = meta[grouping].astype(str).values
    df[condition] = meta[condition].values
    return df.melt(id_vars=[grouping, condition], var_name="Gene", value_name="Expression")


def plot_box_groups(df_melted, genes, grouping, condition, ct_order,
                    palette=None, save_dir=None):
    for gene in genes:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(
            data=df_melted[df_melted["Gene"] == gene],
            x=grouping, y="Expression", hue=condition,
            order=ct_order, palette=palette, ax=ax,
        )
        ax.set_title(f"Expression of $\\it{{{gene}}}$ along {grouping}")
        ax.set_xlabel(grouping)
        ax.set_ylabel("log2-CPM")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(title=condition, loc="upper right", fontsize=8)
        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/{gene}_{grouping}_{condition}.pdf", dpi=100)
        plt.show()
        plt.close()


def plot_matrixplot_per_phase(logcpm, meta, genes_dict, grouping, condition,
                              phase_order=None, phase_colors=None,
                              cmap="viridis", save_dir=None, file_prefix=""):
    bin_order      = _bin_order_present(meta, grouping)
    all_genes      = list(dict.fromkeys(g for gl in genes_dict.values() for g in gl))
    phases_present = set(meta[condition].unique())
    ordered_phases = [p for p in (phase_order or sorted(phases_present))
                      if p in phases_present]

    if not all_genes:
        print("  No genes to plot in matrixplot — skipping.")
        return

    obs = meta[[condition, grouping]].copy()
    obs[grouping] = pd.Categorical(
        obs[grouping].astype(str), categories=bin_order, ordered=True
    )

    adata_pb = anndata.AnnData(
        X   = logcpm.loc[all_genes].T.values.astype(float),
        obs = obs.reset_index(drop=True),
        var = pd.DataFrame(index=all_genes),
    )

    for phase in ordered_phases:
        adata_phase = adata_pb[adata_pb.obs[condition] == phase].copy()
        color       = (phase_colors or {}).get(phase, "black")

        for label, genes in genes_dict.items():
            genes_in = [g for g in genes if g in adata_phase.var_names]
            if not genes_in:
                continue

            sc.pl.matrixplot(
                adata_phase, genes_in, groupby=grouping,
                cmap=cmap, standard_scale="var",
                title=f"{phase} — {label}",
                colorbar_title="scaled\nexpression",
                show=False,
            )
            plt.gcf().axes[0].set_title(f"{phase} — {label}",
                                        color=color, fontweight="bold")
            plt.tight_layout()
            if save_dir:
                prefix = f"{file_prefix}_" if file_prefix else ""
                plt.savefig(f"{save_dir}/matrixplot_{prefix}{phase}_{label}.pdf",
                            dpi=100, bbox_inches="tight")
            plt.show()
            plt.close()


# ── 7. Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/figures", exist_ok=True)

    logcpm, meta = load_pseudobulk(MATRIX_FILE, META_FILE, prior_count=PRIOR_COUNT)

    phase_mask = meta[CONDITION].isin(PHASES_TO_INCLUDE)
    n_dropped  = (~phase_mask).sum()
    meta       = meta[phase_mask].reset_index(drop=True)
    logcpm     = logcpm.loc[:, phase_mask.values]
    print(f"Phase filter: kept {PHASES_TO_INCLUDE}, dropped {n_dropped} samples → "
          f"{meta.shape[0]} samples remaining\n")

    print("Filtering genes...")
    logcpm = filter_genes(logcpm, min_cpm=MIN_CPM, min_samples=MIN_SAMPLES,
                          var_quantile=VAR_QUANTILE)

    if REUSE_RESULTS:
        print("Loading precomputed GLM results...")
        glm_fdr = pd.read_csv(f"{OUTPUT_DIR}/glm_full_results.tsv", sep="\t")
    else:
        glm_df  = run_glm(logcpm, meta, condition=CONDITION,
                          grouping=GROUPING, batch=BATCH)
        glm_fdr = apply_fdr(glm_df, fdr_level=FDR_THRESH)

        n_conv = glm_fdr["converged"].sum()
        sig_A  = glm_fdr["sig_A"].sum()
        sig_B  = glm_fdr["sig_B"].sum()
        print(f"\nConverged models: {n_conv} / {len(glm_fdr)} genes")
        print(f"Test A (axis gradient):      {sig_A} genes at FDR < {FDR_THRESH}")
        print(f"Test B (phase × axis):       {sig_B} genes at FDR < {FDR_THRESH}")

        glm_fdr.to_csv(f"{OUTPUT_DIR}/glm_full_results.tsv", sep="\t", index=False)
        glm_fdr[glm_fdr["sig_A"]].sort_values("beta_axis", key=abs, ascending=False)\
            .to_csv(f"{OUTPUT_DIR}/TestA_axis_genes.tsv", sep="\t", index=False)
        glm_fdr[glm_fdr["sig_B"]].sort_values("p_interact")\
            .to_csv(f"{OUTPUT_DIR}/TestB_interact_genes.tsv", sep="\t", index=False)
        print(f"\nResults saved to {OUTPUT_DIR}/")

    # Dynamic range for all modelled genes
    drange = compute_dynamic_range(logcpm, meta, glm_fdr["gene"].tolist(), GROUPING)
    ct_order = _bin_order_present(meta, GROUPING)

    # ── Test A: axis gradient ────────────────────────────────────────────────
    pos_A, neg_A = select_plot_genes(
        glm_fdr, drange,
        fdr_col="FDR_A", fdr_thresh=FDR_THRESH,
        min_range=MIN_CPM_RANGE, n_top=N_TOP,
    )
    print(f"\nTest A luminal-high (top {N_TOP} +ve beta): {pos_A}")
    print(f"Test A basalis-high (top {N_TOP} -ve beta): {neg_A}")

    all_A = list(dict.fromkeys(pos_A + neg_A))
    if all_A:
        df_melted_A = build_plot_df(logcpm, meta, all_A, GROUPING, CONDITION)
        plot_box_groups(df_melted_A, all_A, grouping=GROUPING, condition=CONDITION,
                        ct_order=ct_order, palette=PHASE_COLORS,
                        save_dir=f"{OUTPUT_DIR}/figures")
        plot_matrixplot_per_phase(
            logcpm, meta,
            genes_dict={"luminal_high": pos_A, "basalis_high": neg_A},
            grouping=GROUPING, condition=CONDITION,
            phase_order=PHASES_TO_INCLUDE, phase_colors=PHASE_COLORS,
            cmap="viridis", save_dir=f"{OUTPUT_DIR}/figures", file_prefix="TestA",
        )

    # ── Test B: phase × axis interaction ─────────────────────────────────────
    pos_B, neg_B = select_plot_genes(
        glm_fdr, drange,
        fdr_col="FDR_B", fdr_thresh=FDR_THRESH,
        min_range=MIN_CPM_RANGE, n_top=N_TOP,
    )
    print(f"\nTest B luminal-high (top {N_TOP} +ve beta): {pos_B}")
    print(f"Test B basalis-high (top {N_TOP} -ve beta): {neg_B}")

    all_B = list(dict.fromkeys(pos_B + neg_B))
    if all_B:
        df_melted_B = build_plot_df(logcpm, meta, all_B, GROUPING, CONDITION)
        plot_box_groups(df_melted_B, all_B, grouping=GROUPING, condition=CONDITION,
                        ct_order=ct_order, palette=PHASE_COLORS,
                        save_dir=f"{OUTPUT_DIR}/figures")
        plot_matrixplot_per_phase(
            logcpm, meta,
            genes_dict={"luminal_high": pos_B, "basalis_high": neg_B},
            grouping=GROUPING, condition=CONDITION,
            phase_order=PHASES_TO_INCLUDE, phase_colors=PHASE_COLORS,
            cmap="viridis", save_dir=f"{OUTPUT_DIR}/figures", file_prefix="TestB",
        )
