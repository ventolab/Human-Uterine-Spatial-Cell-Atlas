"""
Map spatial tissue-zone annotation (functionalis/basalis/lumen/myometrium) onto
mesenchymal scRNA-seq cells using TACCO.

Direction:
  reference = spatial (Xenium, annotation col), filtered to Mesen cells
               using existing fine-celltype TACCO predictions
  query     = mesenchymal scRNA-seq cells for the same menstrual stage

Run per menstrual stage; outputs written to:
  tacco_annotation_from_spatial/{stage}/
    tacco_predictions.csv  — cell_id, annotation (argmax)
    tacco_scores.parquet   — cells × annotation-categories probability matrix
"""
from pathlib import Path

import anndata as ad
import pandas as pd
import scanpy as sc
import tacco as tc

# ── Spatial files and their pre-computed fine-celltype TACCO predictions ──────
# dict: spatial_h5ad_path → tacco_predictions_csv_path (relative to project root)
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

SCRNA_H5AD  = "/lustre/scratch125/cellgen/vento/mm58/eutopic_endometrium/build_eutopic_object/integrated_scvi_uterus.h5ad"

# ANNOT_KEY  = "annotation"
ANNOT_KEY = 'axis_bin'
OUTPUT_DIR = Path("tacco_annotation_bins")


def load_spatial_mesen_reference(spatial_path: str, tacco_pred_path: str) -> ad.AnnData:
    """Load one spatial slide filtered to mesenchymal cells."""
    print(f"    Loading spatial: {Path(spatial_path).name}", flush=True)
    adata = sc.read(spatial_path)

    # Identify mesenchymal cells from existing fine-celltype predictions
    preds = pd.read_csv(tacco_pred_path, index_col=0)
    preds["lineage"] = preds["celltype"].str.split("_", n=1).str[0]
    mesen_ids = preds.index[preds["lineage"] == "Mesen"]

    adata = adata[adata.obs_names.isin(mesen_ids)].copy()
    print(
        f"    After Mesen filter: {adata.shape[0]:,} cells "
        f"({100 * adata.shape[0] / len(preds):.1f}% of slide)",
        flush=True,
    )

    # Drop cells with missing annotation — TACCO's OT fails on NaN priors
    n_before = adata.shape[0]
    
    adata = adata[adata.obs['universal_axis'].notna()].copy()
    if adata.shape[0] < n_before:
        print(f"    Dropped {n_before - adata.shape[0]} cells with NaN annotation", flush=True)

    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()

    # Drop myometrium
    adata = adata[adata.obs['annotation'] != 'myometrium']

    return adata


def build_spatial_reference(stage: str) -> ad.AnnData:
    """Concatenate all Mesen-filtered spatial slides for a stage."""
    parts = []
    for spatial_path, pred_path in SPATIAL_BY_STAGE[stage]:
        slide = load_spatial_mesen_reference(spatial_path, pred_path)
        parts.append(slide)
    ref = ad.concat(parts, label="slide", join="inner")
    ref.obs_names_make_unique()
    print(f"  Concatenated reference: {ref.shape}", flush=True)
    print(f"  Annotation categories: {sorted(ref.obs['annotation'].dropna().unique().tolist())}", flush=True)

    # Define bin names and their order
    BIN_CATEGORIES = [
        "basalis_1", "basalis_2",           # basalis → 2 bins
        "functionalis_1", "functionalis_2", "functionalis_3",  # functionalis → 3 bins
        "lumen_1",                           # lumen → 1 bin
    ]
    print("   Spatial ref cols", ref.obs.columns, flush = True)
    axis = ref.obs["universal_axis"].copy()
    annot = ref.obs["annotation"].copy()
    bins = pd.Series(pd.NA, index=ref.obs_names, dtype=object)

    # Basalis: cut into 2 equal-quantile bins
    mask_bas = annot == "basalis"
    if mask_bas.any():
        bins[mask_bas] = pd.cut(
            axis[mask_bas],
            bins=2,
            labels=["basalis_1", "basalis_2"],
        ).astype(object)

    # Functionalis: cut into 3 equal-quantile bins
    mask_fun = annot == "functionalis"
    if mask_fun.any():
        bins[mask_fun] = pd.cut(
            axis[mask_fun],
            bins=3,
            labels=["functionalis_1", "functionalis_2", "functionalis_3"],
        ).astype(object)

    # Lumen: single bin
    mask_lum = annot == "lumen"
    if mask_lum.any():
        bins[mask_lum] = "lumen_1"

    ref.obs["axis_bin"] = pd.Categorical(
        bins,
        categories=BIN_CATEGORIES,
        ordered=True,
    )

    print(f"  axis_bin distribution:\n{ref.obs['axis_bin'].value_counts().sort_index()}", flush=True)

    # drop na
    ref = ref[ref.obs['axis_bin'].notna()]

    return ref


def load_scrna_query(stage: str) -> ad.AnnData:
    """Load mesenchymal scRNA-seq cells for the given menstrual stage."""
    print(f"\n  Loading scRNA-seq: {Path(SCRNA_H5AD).name}", flush=True)
    adata = sc.read(str(SCRNA_H5AD))
    print(f"  Initial shape: {adata.shape}", flush=True)

    mask = (
        ~adata.obs["Tissue_ROI"].isin(["Menstrual fluid", "Mentrual fluid"])
        & (adata.obs["Menstrual_stage_short"] == stage)
        & (adata.obs["lineage"] == 'mesenchymal')
    )

    adata = adata[mask].copy()  
    print(f"  After all filters ('{stage}', mesenchymal): {adata.shape}", flush=True)

    return adata


def main() -> None:
    print(f"\n{'='*70}", flush=True)
    print("TACCO  spatial annotation  →  mesenchymal scRNA-seq  (per stage)", flush=True)
    print(f"{'='*70}", flush=True)

    for stage in SPATIAL_BY_STAGE:
        print(f"\n{'─'*60}", flush=True)
        print(f"Stage: {stage}", flush=True)
        print(f"{'─'*60}", flush=True)

        out_dir = OUTPUT_DIR / stage
        out_dir.mkdir(parents=True, exist_ok=True)

        if (out_dir / "tacco_predictions.csv").exists():
            print(f"  Skipping — outputs already exist in {out_dir}", flush=True)
            continue

        print(f"\n  Building spatial reference (Mesen cells)...", flush=True)
        reference = build_spatial_reference(stage)

        query = load_scrna_query(stage)

        if query.shape[0] == 0:
            print(f"  WARNING: no scRNA-seq cells found for stage '{stage}', skipping.", flush=True)
            continue

        print(f"\n  Running tc.tl.annotate ...", flush=True)
        tc.tl.annotate(
            query,
            reference,
            annotation_key=ANNOT_KEY,
            result_key="predicted_annotation",
            bisections=0,
        )
        print("  Annotation complete.", flush=True)

        scores: pd.DataFrame = query.obsm["predicted_annotation"]
        scores.index.name = "cell_id"

        preds = pd.DataFrame({
            "cell_id":   query.obs_names,
            ANNOT_KEY:   scores.idxmax(axis=1).values,
        })
        # Carry along useful metadata
        for col in ["fine_celltype", "Menstrual_stage_short", "Donor_id", "sample"]:
            if col in query.obs.columns:
                preds[col] = query.obs[col].values
        preds.to_csv(out_dir / "tacco_predictions.csv", index=False)
        print(f"  Saved tacco_predictions.csv  ({len(preds):,} cells)", flush=True)

        scores.columns = scores.columns.astype(str)
        scores.index   = scores.index.astype(str)
        scores.to_parquet(out_dir / "tacco_scores.parquet")
        print(
            f"  Saved tacco_scores.parquet  "
            f"({scores.shape[0]:,} cells × {scores.shape[1]} categories)",
            flush=True,
        )

        print(f"  Done → {out_dir}", flush=True)

    print(f"\n{'='*70}", flush=True)
    print("TACCO annotation complete.", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
