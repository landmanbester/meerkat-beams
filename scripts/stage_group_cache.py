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
import os
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
    resolved_dest = dest.resolve() if dest.exists() else dest

    # Staging a product from its own cache entry (or from somewhere inside it)
    # would delete the source before copying it. These products are
    # unpublished, so there is nothing to re-download: refuse instead.
    if source == resolved_dest or resolved_dest in source.parents:
        raise SystemExit(
            f"{source} is already the cache entry for {name!r} (or lives inside it); "
            f"there is nothing to stage. Point --product at the original MdV zarr."
        )

    if dest.exists() and not force:
        print(f"{dest} already staged; pass --force to replace")
        return dest

    # Copy beside the destination first and swap it in, so a copy that dies
    # partway leaves the previous entry serving. Mirrors the .partial +
    # os.replace pattern cache.py uses for downloads.
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    print(f"staging {source} -> {dest}")
    try:
        shutil.copytree(source, partial)
        if dest.exists():
            shutil.rmtree(dest)
        os.replace(partial, dest)
    finally:
        if partial.exists():
            shutil.rmtree(partial, ignore_errors=True)

    # Only once the new input is in place: ensure_product_bds returns early on
    # bds.exists(), so a BDS built from the replaced input would keep serving.
    stale_bds = cache.bds_path_for_product(name)
    if stale_bds.exists():
        print(f"dropping {stale_bds}, built from the input just replaced")
        shutil.rmtree(stale_bds)
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
