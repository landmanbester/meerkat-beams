#!/usr/bin/env python
"""Stage local MdV mean-beam zarrs into the meerkat-beams cache.

The MdV-2026 and MeerKAT+ products are not published yet, so
``cache.ensure_product_bds`` refuses to download them. Copy them into
``<cache>/inputs/<product>.zarr`` with this script and the normal
``BeamWizard(band=..., group=...)`` path works unchanged -- the BDS
conversion still runs locally on first use.

Example:

    python scripts/stage_group_cache.py \
        --product MeerKAT_L_mdv2026 ~/data/mkat_beams/MeerKAT_L_mdv2026.zarr \
        --product MKE_L            ~/data/mkat_beams/MKE_L.zarr
"""

import argparse
import shutil
from pathlib import Path

from meerkat_beams import cache


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--product",
        nargs=2,
        action="append",
        required=True,
        metavar=("NAME", "PATH"),
        help="product name and the local mean-beam zarr to stage as it",
    )
    ap.add_argument("--force", action="store_true", help="replace an already-staged input")
    args = ap.parse_args()

    for name, src in args.product:
        cache.product_gdrive_id(name)  # raises ValueError for an unknown product
        source = Path(src).expanduser().resolve()
        if not (source / ".zgroup").exists():
            raise SystemExit(f"{source} does not look like a zarr store (no .zgroup)")
        dest = cache.input_zarr_path_for_product(name)
        if dest.exists():
            if not args.force:
                print(f"{dest} already staged; pass --force to replace")
                continue
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"staging {source} -> {dest}")
        shutil.copytree(source, dest)


if __name__ == "__main__":
    main()
