"""
Extract ISS-Patcher cell type predictions from iss_patched.h5ad files into
parquet, then delete the (large) h5ad to reclaim disk space.

Output per directory: iss_annots.parquet
  Columns: barcode, fine_celltype, fine_celltype_fraction

Only deletes the h5ad if the parquet was written successfully.
Already-converted directories (parquet exists, h5ad absent) are skipped.

Usage
-----
python save_iss_annots.py            # dry run — shows what would happen
python save_iss_annots.py --confirm  # actually write parquet and delete h5ad
"""
import argparse
import sys
from pathlib import Path

import anndata as ad
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import OUTPUT_DIR


def process(h5ad_path: Path, dry_run: bool) -> None:
    parquet_path = h5ad_path.parent / "iss_annots.parquet"

    if parquet_path.exists():
        print(f"  [skip]    {parquet_path.relative_to(OUTPUT_DIR)} already exists")
        return

    size_mb = h5ad_path.stat().st_size / 1e6
    if dry_run:
        print(f"  [dry-run] would extract → {parquet_path.name}  "
              f"then delete iss_patched.h5ad  ({size_mb:.0f} MB)")
        return

    print(f"  [export]  {h5ad_path.relative_to(OUTPUT_DIR)}  ({size_mb:.0f} MB) ...",
          flush=True)
    adata = ad.read_h5ad(h5ad_path, backed="r")
    df = (
        adata.obs[["fine_celltype", "fine_celltype_fraction"]]
        .copy()
        .reset_index()
        .rename(columns={"index": "barcode"})
    )
    df.to_parquet(parquet_path, index=False)
    print(f"           saved {len(df):,} rows → {parquet_path.name}", flush=True)

    h5ad_path.unlink()
    print(f"           deleted {h5ad_path.name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", action="store_true",
                        help="Actually write parquet and delete h5ad files")
    args = parser.parse_args()
    dry_run = not args.confirm

    h5ad_paths = sorted(OUTPUT_DIR.glob("iss_patcher/*/*/*/*/iss_patched.h5ad"))
    print(f"Found {len(h5ad_paths)} iss_patched.h5ad files")
    if dry_run:
        print("(dry run — pass --confirm to execute)\n")

    total_mb = sum(p.stat().st_size for p in h5ad_paths) / 1e6
    print(f"Total disk usage: {total_mb:.0f} MB\n")

    for p in h5ad_paths:
        process(p, dry_run)

    if dry_run:
        print(f"\nRe-run with --confirm to free ~{total_mb:.0f} MB")


if __name__ == "__main__":
    main()
