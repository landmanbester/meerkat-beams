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


def stage_product(name: str, src, force: bool = False) -> Path:
    """Copy the mean-beam zarr at ``src`` into the cache as product ``name``.

    A forced restage also drops the BDS already built from the previous input:
    ``ensure_product_bds`` returns early when the BDS exists, so leaving it
    behind would silently keep serving a beam derived from the input that was
    just replaced.
    """
    cache.product_gdrive_id(name)  # raises ValueError for an unknown product
    source = Path(src).expanduser().resolve()
    if not (source / ".zgroup").exists():
        raise SystemExit(f"{source} does not look like a zarr store (no .zgroup)")
    dest = cache.input_zarr_path_for_product(name)
    if dest.exists():
        if not force:
            print(f"{dest} already staged; pass --force to replace")
            return dest
        shutil.rmtree(dest)
        stale_bds = cache.bds_path_for_product(name)
        if stale_bds.exists():
            print(f"dropping {stale_bds}, built from the input being replaced")
            shutil.rmtree(stale_bds)
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"staging {source} -> {dest}")
    shutil.copytree(source, dest)
    return dest


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
        stage_product(name, src, force=args.force)


if __name__ == "__main__":
    main()
