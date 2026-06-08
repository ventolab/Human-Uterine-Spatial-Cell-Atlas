"""
Map spatial axis values from Xenium onto scRNA-seq reference data using iss_patcher.

Two multi-sample aggregation flavors (--mode):

  concat   — concatenate all spatial slides for a stage, then run iss_patcher once.
             Mirrors 01_run_tacco.py.

  ensemble — run iss_patcher per spatial slide separately, then average the
             universal_axis predictions across slides before saving.
             Mirrors 01_run_tacco_ensembl.py.

  unique   — single representative slide per stage (quick baseline).

Output layout
─────────────
concat / unique:
  axis_mapping_outputs/{mode}_{lineage}_{stage}_k{k}/
    iss_patched.h5ad
    axis.csv

ensemble:
  axis_mapping_outputs/ensemble_{lineage}_{stage}_k{k}/
    {sample}/
      iss_patched.h5ad
      axis.csv          ← per-sample predictions
    ensemble_axis.csv   ← averaged universal_axis across samples
"""

import argparse
import faulthandler
from pathlib import Path

import pandas as pd

from utils import map_spatial_to_scrna

# ── Spatial data files ────────────────────────────────────────────────────────

TACCO_PREDS_DIR = Path(
    "/lustre/scratch125/cellgen/vento/mm58/eutopic_endometrium"
    "/benchmark_knn_vs_dot/outputs/tacco/all_sc/full"
)

# Each entry is (h5ad_path, tacco_pred_path) — used by utils.load_spatial_data
# for lineage filtering via TACCO celltype prefix (Mesen_*)
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

SPATIAL_UNIQUE: dict[str, tuple[str, str]] = {
    stage: slides[0] for stage, slides in SPATIAL_BY_STAGE.items()
}

# ── scRNA-seq reference ───────────────────────────────────────────────────────

SCRNA_H5AD = "/lustre/scratch125/cellgen/vento/mm58/eutopic_endometrium/build_eutopic_object/integrated_scvi_uterus.h5ad"

OUTPUT_BASE = Path("axis_mapping_outputs")

STAGES = ["Proliferative", "Secretory", "Hormones", "Menstrual"]

LINEAGE = "mesenchyme"


# ── Ensemble helper ───────────────────────────────────────────────────────────

def run_ensemble_stage(
    stage: str,
    stage_out_dir: Path,
    neighbours: int,
    computation: str,
) -> None:
    """
    Run iss_patcher once per spatial sample, then average universal_axis
    predictions across samples and save ensemble_axis.csv.
    """
    ensemble_out = stage_out_dir / "ensemble_axis.csv"
    if ensemble_out.exists():
        print(f"  Skipping — ensemble output already exists: {ensemble_out}", flush=True)
        return

    slide_list = SPATIAL_BY_STAGE[stage]
    print(f"\n  Ensemble: {len(slide_list)} spatial sample(s) for {stage}", flush=True)

    per_sample_axes: list[pd.Series] = []
    sample_names: list[str] = []

    for h5ad, pred in slide_list:
        sample = Path(pred).parent.name
        sample_out_dir = stage_out_dir / sample
        axis_csv = sample_out_dir / "axis.csv"

        if axis_csv.exists():
            print(f"    Found cached axis.csv for {sample}, loading.", flush=True)
        else:
            print(f"\n  ── Sample: {sample} ──", flush=True)
            map_spatial_to_scrna(
                spatial_h5ad=(h5ad, pred),
                scrna_h5ad=SCRNA_H5AD,
                lineage=LINEAGE,
                menstrual_stage=stage,
                output_dir=sample_out_dir,
                neighbours=neighbours,
                computation=computation,
            )

        axis_df = pd.read_csv(axis_csv, index_col=0)
        per_sample_axes.append(axis_df["universal_axis"])
        sample_names.append(sample)

    print(f"\n  Averaging universal_axis across {sample_names} ...", flush=True)
    stacked = pd.concat(per_sample_axes, axis=1, keys=sample_names)
    ensemble_axis = stacked.mean(axis=1).rename("universal_axis")
    ensemble_axis.index.name = per_sample_axes[0].index.name

    ensemble_axis.to_frame().to_csv(ensemble_out)
    print(f"  Saved ensemble_axis.csv  ({len(ensemble_axis):,} cells)", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map continuous axis values from spatial to scRNA-seq data."
    )
    parser.add_argument(
        "--mode",
        choices=["unique", "concat", "ensemble"],
        default="unique",
        help=(
            "'unique': one representative slide per stage. "
            "'concat': concatenate all slides per stage, run once. "
            "'ensemble': run per slide separately, average predictions."
        ),
    )
    parser.add_argument(
        "--stage",
        choices=STAGES,
        default=None,
        help="Process only specified menstrual stage. If None, process all.",
    )
    parser.add_argument(
        "--neighbours",
        type=int,
        default=10,
        help="Number of neighbours for iss_patcher",
    )
    parser.add_argument(
        "--computation",
        choices=["cKDTree", "annoy", "pynndescent"],
        default="annoy",
        help="Computation method for iss_patcher",
    )
    return parser.parse_args()


def main() -> None:
    faulthandler.enable()
    args = parse_args()

    stages = [args.stage] if args.stage else STAGES

    print(f"\n{'='*70}", flush=True)
    print("Spatial-to-scRNA Axis Mapping Pipeline  [mesenchyme]", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Mode: {args.mode}", flush=True)
    print(f"Menstrual stages: {stages}", flush=True)
    print(f"Output base: {OUTPUT_BASE}", flush=True)
    print(f"{'='*70}\n", flush=True)

    for stage in stages:
        print(f"\n{'─'*60}", flush=True)
        print(f"Stage: {stage}  |  Lineage: {LINEAGE}  |  Mode: {args.mode}", flush=True)
        print(f"{'─'*60}", flush=True)

        out_dir = OUTPUT_BASE / f"{args.mode}_{LINEAGE}_{stage.lower()}_k{args.neighbours}"

        if args.mode == "unique":
            map_spatial_to_scrna(
                spatial_h5ad=SPATIAL_UNIQUE[stage],
                scrna_h5ad=SCRNA_H5AD,
                lineage=LINEAGE,
                menstrual_stage=stage,
                output_dir=out_dir,
                neighbours=args.neighbours,
                computation=args.computation,
            )

        elif args.mode == "concat":
            map_spatial_to_scrna(
                spatial_h5ad=SPATIAL_BY_STAGE[stage],
                scrna_h5ad=SCRNA_H5AD,
                lineage=LINEAGE,
                menstrual_stage=stage,
                output_dir=out_dir,
                neighbours=args.neighbours,
                computation=args.computation,
            )

        elif args.mode == "ensemble":
            run_ensemble_stage(
                stage=stage,
                stage_out_dir=out_dir,
                neighbours=args.neighbours,
                computation=args.computation,
            )

    print(f"\n{'='*70}", flush=True)
    print("Pipeline complete", flush=True)
    print(f"{'='*70}\n", flush=True)


if __name__ == "__main__":
    main()
