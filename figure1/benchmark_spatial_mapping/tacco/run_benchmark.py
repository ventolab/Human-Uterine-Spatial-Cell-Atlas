"""TACCO benchmark: all_sc config or annotation-granularity variants.

Output structure: outputs/tacco/{annot_config}/full/{stage}/{donor}/
  tacco_predictions.csv  — cell_id, spatial.1, spatial.2, celltype (argmax)
  tacco_scores.parquet   — cells × cell_types probability matrix

The baseline (fine_celltype, no exclusions) keeps its existing output path
under outputs/tacco/all_sc/full/ for backwards compatibility.

Usage
-----
python tacco/run_benchmark.py                               # baseline (all_sc)
python tacco/run_benchmark.py --annot_config broad_celltype
python tacco/run_benchmark.py --annot_config fine_no_oLAM --sample_id Proliferative_1
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import scanpy as sc
import tacco as tc

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SPATIAL_FILES, OUTPUT_DIR, ANNOT_CONFIGS

REFERENCE_PATH = Path(__file__).parent / "../dot/concat_uterus_inner_annotated.h5ad"
LABEL          = "full"


def _stage(sample_id: str) -> str:
    return re.sub(r"_\d+$", "", sample_id)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--annot_config", default=None, choices=list(ANNOT_CONFIGS.keys()),
        help="Annotation config to use. Default: baseline fine_celltype (all_sc).",
    )
    ap.add_argument(
        "--sample_id", default=None,
        help="Run only this sample_id (e.g. 'Proliferative_1'). Default: all.",
    )
    args = ap.parse_args()

    if args.annot_config is None:
        # Baseline — preserve existing behaviour and output path
        config_label   = "all_sc"
        annotation_key = "fine_celltype"
        exclude        = set()
        out_base       = OUTPUT_DIR / "tacco"
    else:
        config_label   = args.annot_config
        annotation_key = ANNOT_CONFIGS[args.annot_config]["annotation_key"]
        exclude        = set(ANNOT_CONFIGS[args.annot_config]["exclude_celltypes"])
        out_base       = OUTPUT_DIR.parent / "outputs_tacco_granularity"

    if args.sample_id is not None and args.sample_id not in SPATIAL_FILES:
        print(f"ERROR: unknown sample_id '{args.sample_id}'. "
              f"Valid: {list(SPATIAL_FILES)}", flush=True)
        sys.exit(1)

    spatial_files = (
        {args.sample_id: SPATIAL_FILES[args.sample_id]}
        if args.sample_id else SPATIAL_FILES
    )

    print(f"\n{'='*70}", flush=True)
    print(f"TACCO benchmark  config={config_label}  "
          f"annotation_key={annotation_key}  "
          f"exclude={exclude or 'none'}  "
          f"sample_id={args.sample_id or 'all'}", flush=True)
    print(f"{'='*70}", flush=True)

    # ── Load reference once ───────────────────────────────────────────────────
    print(f"\nLoading reference: {REFERENCE_PATH} ...", flush=True)
    reference = sc.read(str(REFERENCE_PATH))
    print(f"  Reference shape: {reference.shape}", flush=True)

    if exclude:
        mask = ~reference.obs["fine_celltype"].isin(exclude)
        reference = reference[mask] #.copy()
        print(f"  After exclusion of {exclude}: {reference.shape}", flush=True)

    # ── Process each spatial sample ───────────────────────────────────────────
    for sample_id, spatial_path in spatial_files.items():
        stage = _stage(sample_id)

        print(f"\n{'─'*60}", flush=True)
        print(f"Sample: {sample_id}  stage: {stage}", flush=True)
        print(f"{'─'*60}", flush=True)

        print(f"  Loading spatial: {Path(spatial_path).name} ...", flush=True)
        query = sc.read(spatial_path)
        donor = str(query.obs["sample"].iloc[0])
        print(f"  Spatial shape: {query.shape}  donor: {donor}", flush=True)

        out_dir = out_base / config_label / LABEL / stage / donor
        out_dir.mkdir(parents=True, exist_ok=True)

        if (out_dir / "tacco_predictions.csv").exists():
            print(f"  Skipping — outputs already exist in {out_dir}", flush=True)
            continue

        if "counts" in query.layers:
            query.X = query.layers["counts"].copy()

        # ── Run TACCO annotation ──────────────────────────────────────────────
        print("  Running tc.tl.annotate ...", flush=True)
        tc.tl.annotate(
            query,
            reference,
            annotation_key=annotation_key,
            result_key="predicted_celltype",
            bisections=0,
        )
        print("  Annotation complete.", flush=True)

        scores: pd.DataFrame = query.obsm["predicted_celltype"]
        scores.index.name = "cell_id"

        preds = pd.DataFrame({
            "cell_id":   query.obs_names,
            "spatial.1": query.obsm["spatial"][:, 0],
            "spatial.2": query.obsm["spatial"][:, 1],
            "celltype":  scores.idxmax(axis=1).values,
        })
        preds.to_csv(out_dir / "tacco_predictions.csv", index=False)
        print(f"  Saved tacco_predictions.csv  ({len(preds):,} cells)", flush=True)

        scores.columns = scores.columns.astype(str)
        scores.index   = scores.index.astype(str)
        scores.to_parquet(out_dir / "tacco_scores.parquet")
        print(f"  Saved tacco_scores.parquet  "
              f"({scores.shape[0]:,} cells × {scores.shape[1]} cell types)", flush=True)

        print(f"  Done → {out_dir}", flush=True)

    print(f"\n{'='*70}", flush=True)
    print("TACCO benchmark complete.", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
