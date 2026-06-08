"""ISS-Patcher benchmark: all_sc and stage_matched configs.

Output structure: outputs/iss_patcher/{config}/{downsample_label}/{stage}/{donor}/

Usage
-----
python run_benchmark.py --config all_sc                       # default: 100 cells/type
python run_benchmark.py --config all_sc --downsample_n 1000  # 1 000 cells/type
"""
import argparse
import re
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    SC_REF_PATH, SC_ANNOT_PATH, SPATIAL_FILES, OUTPUT_DIR,
    DOWNSAMPLE_N, KNN_NEIGHBOURS, KNN_COMPUTATION,
)
from utils import load_sc_reference, load_spatial, run_patch_and_evaluate


def _stage(sample_id: str) -> str:
    """'Proliferative_1' → 'Proliferative',  'Secretory' → 'Secretory'"""
    return re.sub(r"_\d+$", "", sample_id)


def main(config: str, downsample_n: int, full_only: bool = False, sample_id: str | None = None) -> None:
    # Directory label for the downsampled run:
    #   100  → "downsampled"   (default, keeps existing outputs)
    #   1000 → "downsampled_1k"
    #   N    → "downsampled_N"
    if downsample_n == 100:
        ds_label = "downsampled"
    elif downsample_n % 1000 == 0:
        ds_label = f"downsampled_{downsample_n // 1000}k"
    else:
        ds_label = f"downsampled_{downsample_n}"

    if sample_id is not None:
        if sample_id not in SPATIAL_FILES:
            print(f"ERROR: unknown sample_id '{sample_id}'. Valid: {list(SPATIAL_FILES)}", flush=True)
            sys.exit(1)
        spatial_files = {sample_id: SPATIAL_FILES[sample_id]}
    else:
        spatial_files = SPATIAL_FILES

    print(f"\n{'='*70}", flush=True)
    print(f"ISS-Patcher benchmark  config={config}  downsample_n={downsample_n}  label={ds_label}  full_only={full_only}  sample_id={sample_id}", flush=True)
    print(f"{'='*70}", flush=True)

    runs = [(False, "full")] if full_only else [(True, ds_label), (False, "full")]
    for downsample, label in runs:
        print(f"\n{'─'*60}", flush=True)
        print(f"Run: {label}", flush=True)
        print(f"{'─'*60}", flush=True)

        if config == "all_sc":
            sc = load_sc_reference(
                SC_REF_PATH, SC_ANNOT_PATH, config="all_sc", stage=None,
                downsample=downsample, downsample_n=downsample_n,
            )
            for sample_id, spatial_path in spatial_files.items():
                stage = _stage(sample_id)
                spatial = load_spatial(spatial_path)
                donor = str(spatial.obs["sample"].iloc[0])
                print(f"\n── {sample_id}  (donor: {donor}) ──", flush=True)
                out_dir = OUTPUT_DIR / "iss_patcher" / config / label / stage / donor
                run_patch_and_evaluate(
                    sc, spatial, out_dir,
                    neighbours=KNN_NEIGHBOURS, computation=KNN_COMPUTATION,
                )

        else:  # stage_matched — group samples by stage to load SC once per stage
            by_stage = defaultdict(list)
            for sample_id, spatial_path in spatial_files.items():
                by_stage[_stage(sample_id)].append((sample_id, spatial_path))

            for stage, samples in by_stage.items():
                sc = load_sc_reference(
                    SC_REF_PATH, SC_ANNOT_PATH, config="stage_matched", stage=stage,
                    downsample=downsample, downsample_n=downsample_n,
                )
                for sample_id, spatial_path in samples:
                    spatial = load_spatial(spatial_path)
                    donor = str(spatial.obs["sample"].iloc[0])
                    print(f"\n── {sample_id}  (donor: {donor}) ──", flush=True)
                    out_dir = OUTPUT_DIR / "iss_patcher" / config / label / stage / donor
                    run_patch_and_evaluate(
                        sc, spatial, out_dir,
                        neighbours=KNN_NEIGHBOURS, computation=KNN_COMPUTATION,
                    )

    print(f"\n{'='*70}", flush=True)
    print("ISS-Patcher benchmark complete.", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["all_sc", "stage_matched"], required=True)
    parser.add_argument(
        "--downsample_n", type=int, default=DOWNSAMPLE_N,
        help=f"Max cells per celltype for the downsampled run (default: {DOWNSAMPLE_N})",
    )
    parser.add_argument(
        "--full_only", action="store_true",
        help="Skip the downsampled pass and run only the full (no downsampling) version",
    )
    parser.add_argument(
        "--sample_id", default=None,
        help="Run only this sample_id (e.g. 'Proliferative_1'). Default: all samples.",
    )
    args = parser.parse_args()
    main(args.config, args.downsample_n, full_only=args.full_only, sample_id=args.sample_id)
