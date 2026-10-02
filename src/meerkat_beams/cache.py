"""
On-demand download + cache of MdV mean-beam zarrs and the BDS files
built from them.

The cache lives under ``cache_root()``:

    <root>/inputs/<product>.zarr/       # mean-beam zarr from gdrive
    <root>/bds/<product>.bds.zarr/      # compressed BDS, built locally

where <product> is MeerKAT_<BAND> for the legacy bands, or one of the
MdV-2026 products (MeerKAT_L_mdv2026, MKE_L) used by the baseline groups.

Cache root resolution:

    MBEAMS_CACHE_DIR              if set and non-empty
    $XDG_CACHE_HOME/meerkat-beams if XDG_CACHE_HOME set and non-empty
    $HOME/.cache/meerkat-beams    otherwise

Concurrent first-time downloads of the same band from multiple processes
are not guarded. Warm the cache from a single process.
"""

import os
import shutil
import sys
import tarfile
from pathlib import Path

BAND_GDRIVE_IDS: dict[str, str] = {
    "U": "105JWCFo4R-Qo6wHCCkhPm7ZhOSlUaoPx",
    "L": "1dAVD5sE-9fL1kGTjlpaXtI1lOBHJH19K",
    "S0": "1UN5slkHYfXD_MGUZaKFH-UBalgqiepfP",
    "S4": "1-8eg7cCZO4HwTdXW5F55ftmJPOSj3qFV",
}
SUPPORTED_BANDS: tuple[str, ...] = tuple(BAND_GDRIVE_IDS.keys())

# Sentinel for a product whose tarball has not been published yet. Such a
# product can still be used by staging its input zarr into the cache by hand;
# see ensure_product_bds().
PLACEHOLDER_GDRIVE_ID = "PLACEHOLDER"

# MdV-2026 generation. These are 64x64 / 63-channel stores, a different grid
# from the legacy BAND_GDRIVE_IDS products, and the only generation for which
# a matched MeerKAT/MeerKAT+ pair exists. Selected by asking for a group.
MDV2026_GDRIVE_IDS: dict[str, str] = {
    "MeerKAT_L_mdv2026": PLACEHOLDER_GDRIVE_ID,
    "MKE_L": PLACEHOLDER_GDRIVE_ID,
}

SUPPORTED_GROUPS: tuple[str, ...] = ("MM", "MPM", "MPMP")

# (band, group) -> (product for the FIRST antenna, product for the SECOND).
# The MPM convention is p = MeerKAT, q = MeerKAT+; see docs/wiki/data-model.md.
# Only L band has a matched pair: MKE's S3 product shares no grid with any
# MeerKAT S-band product (S04 aligns in frequency but is on a half-size
# +-0.5 deg grid; S0+S4 match spatially but leave a 27 MHz hole).
GROUP_PRODUCTS: dict[tuple[str, str], tuple[str, str]] = {
    ("L", "MM"): ("MeerKAT_L_mdv2026", "MeerKAT_L_mdv2026"),
    ("L", "MPM"): ("MeerKAT_L_mdv2026", "MKE_L"),
    ("L", "MPMP"): ("MKE_L", "MKE_L"),
}

SUPPORTED_GROUP_BANDS: tuple[str, ...] = tuple(dict.fromkeys(band for band, _ in GROUP_PRODUCTS))


def cache_root() -> Path:
    explicit = os.environ.get("MBEAMS_CACHE_DIR")
    if explicit:
        root = Path(explicit)
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        root = Path(xdg) / "meerkat-beams" if xdg else Path.home() / ".cache" / "meerkat-beams"
    root.mkdir(parents=True, exist_ok=True)
    return root


def product_gdrive_id(product: str) -> str:
    """Return the gdrive id for a beam product, legacy or MdV-2026."""
    if product in MDV2026_GDRIVE_IDS:
        return MDV2026_GDRIVE_IDS[product]
    if product.startswith("MeerKAT_") and product[len("MeerKAT_") :] in BAND_GDRIVE_IDS:
        return BAND_GDRIVE_IDS[product[len("MeerKAT_") :]]
    known = sorted(set(MDV2026_GDRIVE_IDS) | {f"MeerKAT_{b}" for b in BAND_GDRIVE_IDS})
    raise ValueError(f"unknown beam product {product!r}; known products are {', '.join(known)}")


def input_zarr_path_for_product(product: str) -> Path:
    return cache_root() / "inputs" / f"{product}.zarr"


def bds_path_for_product(product: str) -> Path:
    return cache_root() / "bds" / f"{product}.bds.zarr"


def input_zarr_path(band: str) -> Path:
    return input_zarr_path_for_product(f"MeerKAT_{band}")


def bds_path(band: str) -> Path:
    return bds_path_for_product(f"MeerKAT_{band}")


def ensure_band_bds(band: str) -> str:
    """Return a local BDS path for a legacy ``band``, downloading and converting as needed."""
    if band not in SUPPORTED_BANDS:
        raise ValueError(f"band must be one of {SUPPORTED_BANDS}, got {band!r}")
    return ensure_product_bds(f"MeerKAT_{band}")


def ensure_product_bds(product: str) -> str:
    """Return a local BDS path for ``product``, downloading and converting as needed."""
    gid = product_gdrive_id(product)  # raises ValueError for an unknown product

    _clear_partials(product)

    bds = bds_path_for_product(product)
    if bds.exists():
        return str(bds)

    inp = input_zarr_path_for_product(product)
    if not inp.exists():
        if gid == PLACEHOLDER_GDRIVE_ID:
            raise RuntimeError(
                f"beam product {product!r} is not yet published, so it cannot be downloaded. "
                f"Stage it manually by copying the MdV mean-beam zarr to {inp} and retry."
            )
        _download_and_extract(product)

    _convert_to_bds(product)
    return str(bds)


def ensure_group_bds(band: str, group: str) -> tuple[str, str]:
    """Return ``(bds_p, bds_q)`` BDS paths for a baseline group.

    ``bds_p`` is the FIRST antenna of the baseline and ``bds_q`` the second.
    For ``MM`` and ``MPMP`` the two are the same path, converted once.
    """
    if group not in SUPPORTED_GROUPS:
        raise ValueError(f"group must be one of {SUPPORTED_GROUPS}, got {group!r}")
    if (band, group) not in GROUP_PRODUCTS:
        raise ValueError(
            f"band {band!r} has no matched MeerKAT/MeerKAT+ beam counterpart, so group "
            f"{group!r} is unavailable. Group bands are {SUPPORTED_GROUP_BANDS}."
        )
    product_p, product_q = GROUP_PRODUCTS[(band, group)]
    bds_p = ensure_product_bds(product_p)
    bds_q = bds_p if product_q == product_p else ensure_product_bds(product_q)
    return bds_p, bds_q


def _gdown_download(id: str, output: str, quiet: bool) -> None:  # noqa: A002
    """Thin wrapper around gdown.download so tests can monkeypatch it."""
    import gdown  # local import: gdown is a [full] extra

    gdown.download(id=id, output=output, quiet=quiet)


def _download_and_extract(product: str) -> None:
    from meerkat_beams.utils import log

    inp = input_zarr_path_for_product(product)
    partial = _partial(inp)
    inp.parent.mkdir(parents=True, exist_ok=True)
    partial.mkdir(parents=True, exist_ok=True)

    tarball = partial.parent / f"{product}.zarr.tgz"
    gid = product_gdrive_id(product)
    try:
        try:
            log.info(f"downloading {product}.zarr.tgz from gdrive id {gid}")
            _gdown_download(id=gid, output=str(tarball), quiet=False)
        except ImportError as e:
            raise ImportError(
                f"meerkat-beams was installed without the [full] extra; "
                f"install meerkat-beams[full] to use product={product!r}"
            ) from e

        log.info(f"extracting {tarball} into {partial}")
        # Python 3.12+ tarfile expects an explicit `filter`; "data" strips
        # setuid/setgid bits and absolute paths. Default becomes mandatory in 3.14.
        extract_kwargs = {"filter": "data"} if sys.version_info >= (3, 12) else {}
        with tarfile.open(tarball, "r:gz") as tar:
            tar.extractall(path=partial, **extract_kwargs)

        # Tarball contains a top-level <product>.zarr/ directory; promote it.
        extracted = partial / f"{product}.zarr"
        if not extracted.is_dir():
            raise RuntimeError(f"expected {extracted.name}/ inside tarball but did not find it")
        os.replace(extracted, inp)
    finally:
        if tarball.exists():
            tarball.unlink()
        if partial.exists():
            shutil.rmtree(partial, ignore_errors=True)


def _convert_to_bds(product: str) -> None:
    from meerkat_beams.core.mdv_beams_to_bds import mdv_beams_to_bds
    from meerkat_beams.utils import log

    inp = input_zarr_path_for_product(product)
    out = bds_path_for_product(product)
    partial = _partial(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if partial.exists():
        shutil.rmtree(partial, ignore_errors=True)

    log.info(f"converting {inp} -> {out} (via .partial)")
    try:
        mdv_beams_to_bds(mdv_beams=str(inp), bds=str(partial), compress=True)
        os.replace(partial, out)
    finally:
        if partial.exists():
            shutil.rmtree(partial, ignore_errors=True)


def _partial(path: Path) -> Path:
    """Sibling .partial directory next to ``path``."""
    return path.with_name(path.name + ".partial")


def _clear_partials(product: str) -> None:
    from meerkat_beams.utils import log

    for p in (_partial(input_zarr_path_for_product(product)), _partial(bds_path_for_product(product))):
        if p.exists():
            log.warning(f"removing stale partial cache dir {p}")
            shutil.rmtree(p, ignore_errors=True)

    tarball = input_zarr_path_for_product(product).parent / f"{product}.zarr.tgz"
    if tarball.exists():
        log.warning(f"removing stale download tarball {tarball}")
        tarball.unlink()
