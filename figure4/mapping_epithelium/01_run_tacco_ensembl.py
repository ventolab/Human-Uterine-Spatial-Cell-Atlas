"""
Map spatial axis bins onto epithelial scRNA-seq cells using TACCO,
running one spatial sample at a time and ensembling predictions by averaging
scores across samples before taking argmax.

Reference = spatial (Xenium), filtered to Epi cells per sample
Query     = epithelial scRNA-seq cells for the same menstrual stage

Output layout:
  tacco_annotation_ensembl/{stage}/{sample}/tacco_scores.parquet  — per-sample scores
  tacco_annotation_ensembl/{stage}/tacco_scores_ensemble.parquet  — averaged scores
  tacco_annotation_ensembl/{stage}/tacco_predictions.csv          — argmax of ensemble
"""
from pathlib import Path

import anndata as ad
import pandas as pd
import scanpy as sc
import tacco as tc

# ── Spatial files and their pre-computed fine-celltype TACCO predictions ──────
TACCO_PREDS_DIR = Path(
    "/lustre/scratch125/cellgen/vento/mm58/eutopic_endometrium"
    "/benchmark_knn_vs_dot/outputs/tacco/all_sc/full"
)

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
        # (
        #     "/nfs/team292/vl6/Endometriosis/Xenium/DA64-END-0-FO-1-S2-i/DA64_annotated_new_axis.h5ad",
        #     str(TACCO_PREDS_DIR / "Proliferative/DA64/tacco_predictions.csv"),
        # ),
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

SCRNA_H5AD = "/nfs/team292/projects/PanTissue/results/temp/02_annotation/annotated_postnatal_epithelial_endometrium.h5ad"

ANNOT_KEY  = "axis_bin"
OUTPUT_DIR = Path("tacco_annotation_ensembl")

BIN_CATEGORIES = [
    "basalis_1", "basalis_2",
    "functionalis_1", "functionalis_2", "functionalis_3",
    "lumen_1",
]


def _sample_name_from_pred_path(pred_path: str) -> str:
    """Extract sample name from the tacco predictions CSV path."""
    return Path(pred_path).parent.name


def load_spatial_epi_reference(spatial_path: str, tacco_pred_path: str) -> ad.AnnData:
    """Load one spatial slide filtered to epithelial cells."""
    print(f"    Loading spatial: {Path(spatial_path).name}", flush=True)
    adata = sc.read(spatial_path)

    preds = pd.read_csv(tacco_pred_path, index_col=0)
    preds["lineage"] = preds["celltype"].str.split("_", n=1).str[0]
    epi_ids = preds.index[preds["lineage"] == "Epi"]

    adata = adata[adata.obs_names.isin(epi_ids)].copy()
    print(
        f"    After Epi filter: {adata.shape[0]:,} cells "
        f"({100 * adata.shape[0] / len(preds):.1f}% of slide)",
        flush=True,
    )

    adata = adata[adata.obs["annotation"] != "myometrium"].copy()

    n_before = adata.shape[0]
    adata = adata[adata.obs["universal_axis"].notna()].copy()
    if adata.shape[0] < n_before:
        print(f"    Dropped {n_before - adata.shape[0]} cells with NaN universal_axis", flush=True)

    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()

    return adata


def assign_axis_bins(adata: ad.AnnData) -> ad.AnnData:
    """Add axis_bin column to adata.obs and drop NaN rows."""
    axis  = adata.obs["universal_axis"].copy()
    annot = adata.obs["annotation"].copy()
    bins  = pd.Series(pd.NA, index=adata.obs_names, dtype=object)

    mask_bas = annot == "basalis"
    if mask_bas.any():
        bins[mask_bas] = pd.cut(
            axis[mask_bas], bins=2, labels=["basalis_1", "basalis_2"]
        ).astype(object)

    mask_fun = annot == "functionalis"
    if mask_fun.any():
        bins[mask_fun] = pd.cut(
            axis[mask_fun], bins=3,
            labels=["functionalis_1", "functionalis_2", "functionalis_3"],
        ).astype(object)

    mask_lum = annot == "lumen"
    if mask_lum.any():
        bins[mask_lum] = "lumen_1"

    adata.obs[ANNOT_KEY] = pd.Categorical(bins, categories=BIN_CATEGORIES, ordered=True)
    print(
        f"    axis_bin distribution:\n{adata.obs[ANNOT_KEY].value_counts().sort_index()}",
        flush=True,
    )

    adata = adata[adata.obs[ANNOT_KEY].notna()].copy()
    return adata


def load_scrna_query(stage: str) -> ad.AnnData:
    """Load epithelial scRNA-seq cells for the given menstrual stage."""
    print(f"\n  Loading scRNA-seq: {Path(SCRNA_H5AD).name}", flush=True)
    adata = sc.read(str(SCRNA_H5AD))
    print(f"  Initial shape: {adata.shape}", flush=True)

    if "Tissue_ROI" in adata.obs.columns:
        adata = adata[~adata.obs["Tissue_ROI"].isin(["Menstrual fluid", "Mentrual fluid"])]

    adata = adata[adata.obs["celltype"].notna()].copy()
    adata = adata[~adata.obs["celltype"].isin(["lowQC", "doublet", "unknown"])].copy()

    adata = adata[adata.obs["Menstrual_stage_short"] == stage].copy()
    print(f"  After stage filter ('{stage}'): {adata.shape}", flush=True)

    adata = adata[adata.obs["celltype"].str.startswith("Epi")].copy()
    print(f"  After Epi filter: {adata.shape}", flush=True)

    return adata


def run_tacco_single(
    query: ad.AnnData,
    reference: ad.AnnData,
    sample_name: str,
    sample_out_dir: Path,
) -> pd.DataFrame:
    """
    Run TACCO with a single spatial reference slide.
    Saves per-sample scores to sample_out_dir.
    Returns the scores DataFrame (cells × bins).
    """
    sample_out_dir.mkdir(parents=True, exist_ok=True)
    out_path = sample_out_dir / "tacco_scores.parquet"

    if out_path.exists():
        print(f"    Found cached scores for {sample_name}, loading.", flush=True)
        return pd.read_parquet(out_path)

    print(f"    Running tc.tl.annotate with reference={sample_name} ...", flush=True)
    tc.tl.annotate(
        query,
        reference,
        annotation_key=ANNOT_KEY,
        result_key="predicted_annotation",
        bisections=0,
        assume_valid_counts=True,
    )

    scores: pd.DataFrame = query.obsm["predicted_annotation"].copy()
    scores.columns    = scores.columns.astype(str)
    scores.index      = query.obs_names.astype(str)
    scores.index.name = "cell_id"

    scores.to_parquet(out_path)
    print(
        f"    Saved tacco_scores.parquet  "
        f"({scores.shape[0]:,} cells × {scores.shape[1]} bins)",
        flush=True,
    )
    return scores


def ensemble_and_save(
    scores_list: list[pd.DataFrame],
    sample_names: list[str],
    query: ad.AnnData,
    stage_out_dir: Path,
) -> None:
    """Average scores across samples, assign argmax, write ensemble outputs."""
    print(f"\n  Ensembling {len(scores_list)} sample(s): {sample_names}", flush=True)

    aligned         = [df.reindex(columns=BIN_CATEGORIES, fill_value=0.0) for df in scores_list]
    ensemble_scores = pd.concat(aligned).groupby(level=0).mean()
    ensemble_scores = ensemble_scores.reindex(query.obs_names.astype(str))

    ensemble_scores.columns    = ensemble_scores.columns.astype(str)
    ensemble_scores.index.name = "cell_id"
    ensemble_scores.to_parquet(stage_out_dir / "tacco_scores_ensemble.parquet")
    print(
        f"  Saved tacco_scores_ensemble.parquet  "
        f"({ensemble_scores.shape[0]:,} cells × {ensemble_scores.shape[1]} bins)",
        flush=True,
    )

    preds = pd.DataFrame({"cell_id": ensemble_scores.index})
    preds[ANNOT_KEY] = ensemble_scores.idxmax(axis=1).values
    for col in ["celltype", "Menstrual_stage_short", "Donor_id", "sample"]:
        if col in query.obs.columns:
            preds[col] = query.obs[col].values
    preds.to_csv(stage_out_dir / "tacco_predictions.csv", index=False)
    print(f"  Saved tacco_predictions.csv  ({len(preds):,} cells)", flush=True)


def main() -> None:
    print(f"\n{'='*70}", flush=True)
    print("TACCO  ensemble  spatial annotation  →  epithelial scRNA-seq", flush=True)
    print(f"{'='*70}", flush=True)

    for stage, slide_list in SPATIAL_BY_STAGE.items():
        print(f"\n{'─'*60}", flush=True)
        print(f"Stage: {stage}  ({len(slide_list)} spatial sample(s))", flush=True)
        print(f"{'─'*60}", flush=True)

        stage_out_dir = OUTPUT_DIR / stage
        stage_out_dir.mkdir(parents=True, exist_ok=True)

        if (stage_out_dir / "tacco_predictions.csv").exists():
            print(f"  Skipping — ensemble outputs already exist in {stage_out_dir}", flush=True)
            continue

        query = load_scrna_query(stage)
        if query.shape[0] == 0:
            print(f"  WARNING: no scRNA-seq cells found for stage '{stage}', skipping.", flush=True)
            continue

        scores_list: list[pd.DataFrame] = []
        sample_names: list[str] = []

        for spatial_path, pred_path in slide_list:
            sample = _sample_name_from_pred_path(pred_path)
            print(f"\n  ── Sample: {sample} ──", flush=True)

            sample_out_dir = stage_out_dir / sample

            cached = sample_out_dir / "tacco_scores.parquet"
            if cached.exists():
                print(f"    Found cached scores for {sample}, loading.", flush=True)
                scores = pd.read_parquet(cached)
            else:
                print(f"  Building spatial reference for {sample} ...", flush=True)
                reference = load_spatial_epi_reference(spatial_path, pred_path)
                reference = assign_axis_bins(reference)
                print(f"  Reference shape after binning: {reference.shape}", flush=True)

                scores = run_tacco_single(query, reference, sample, sample_out_dir)

            scores_list.append(scores)
            sample_names.append(sample)

        ensemble_and_save(scores_list, sample_names, query, stage_out_dir)
        print(f"\n  Done → {stage_out_dir}", flush=True)

    print(f"\n{'='*70}", flush=True)
    print("TACCO ensemble annotation complete.", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
