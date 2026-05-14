"""
On-demand download + cache of MdV mean-beam zarrs and the BDS files
built from them.

The cache lives under ``cache_root()``:

    <root>/inputs/MeerKAT_<BAND>.zarr/   # mean-beam zarr from gdrive
    <root>/bds/MeerKAT_<BAND>.bds.zarr/  # compressed BDS, built locally

Cache root resolution:

    MBEAMS_CACHE_DIR              if set and non-empty
    $XDG_CACHE_HOME/meerkat-beams if XDG_CACHE_HOME set and non-empty
    $HOME/.cache/meerkat-beams    otherwise

Concurrent first-time downloads of the same band from multiple processes
are not guarded. Warm the cache from a single process.
"""

import os
import shutil
import tarfile
from pathlib import Path

BAND_GDRIVE_IDS: dict[str, str] = {
    "U": "105JWCFo4R-Qo6wHCCkhPm7ZhOSlUaoPx",
    "L": "1dAVD5sE-9fL1kGTjlpaXtI1lOBHJH19K",
    "S0": "1UN5slkHYfXD_MGUZaKFH-UBalgqiepfP",
    "S4": "1-8eg7cCZO4HwTdXW5F55ftmJPOSj3qFV",
}
SUPPORTED_BANDS: tuple[str, ...] = tuple(BAND_GDRIVE_IDS.keys())


def cache_root() -> Path:
    explicit = os.environ.get("MBEAMS_CACHE_DIR")
    if explicit:
        root = Path(explicit)
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        root = Path(xdg) / "meerkat-beams" if xdg else Path.home() / ".cache" / "meerkat-beams"
    root.mkdir(parents=True, exist_ok=True)
    return root


def input_zarr_path(band: str) -> Path:
    return cache_root() / "inputs" / f"MeerKAT_{band}.zarr"


def bds_path(band: str) -> Path:
    return cache_root() / "bds" / f"MeerKAT_{band}.bds.zarr"


def ensure_band_bds(band: str) -> str:
    """Return a local BDS path for ``band``, downloading and converting as needed."""
    if band not in SUPPORTED_BANDS:
        raise ValueError(f"band must be one of {SUPPORTED_BANDS}, got {band!r}")

    _clear_partials(band)

    bds = bds_path(band)
    if bds.exists():
        return str(bds)

    if not input_zarr_path(band).exists():
        _download_and_extract(band)

    _convert_to_bds(band)
    return str(bds)


def _gdown_download(id: str, output: str, quiet: bool) -> None:  # noqa: A002
    """Thin wrapper around gdown.download so tests can monkeypatch it."""
    import gdown  # local import: gdown is a [full] extra

    gdown.download(id=id, output=output, quiet=quiet)


def _download_and_extract(band: str) -> None:
    from meerkat_beams.utils import log

    inp = input_zarr_path(band)
    partial = _partial(inp)
    inp.parent.mkdir(parents=True, exist_ok=True)
    partial.mkdir(parents=True, exist_ok=True)

    tarball = partial.parent / f"MeerKAT_{band}.zarr.tgz"
    gid = BAND_GDRIVE_IDS[band]
    try:
        try:
            log.info(f"downloading MeerKAT_{band}.zarr.tgz from gdrive id {gid}")
            _gdown_download(id=gid, output=str(tarball), quiet=False)
        except ImportError as e:
            raise ImportError(
                f"meerkat-beams was installed without the [full] extra; "
                f"install meerkat-beams[full] to use band={band!r}"
            ) from e

        log.info(f"extracting {tarball} into {partial}")
        with tarfile.open(tarball, "r:gz") as tar:
            tar.extractall(path=partial)

        # Tarball contains a top-level MeerKAT_<BAND>.zarr/ directory; promote it.
        extracted = partial / f"MeerKAT_{band}.zarr"
        if not extracted.is_dir():
            raise RuntimeError(f"expected {extracted.name}/ inside tarball but did not find it")
        os.replace(extracted, inp)
    finally:
        if tarball.exists():
            tarball.unlink()
        if partial.exists():
            shutil.rmtree(partial, ignore_errors=True)


def _convert_to_bds(band: str) -> None:
    raise NotImplementedError


def _partial(path: Path) -> Path:
    """Sibling .partial directory next to ``path``."""
    return path.with_name(path.name + ".partial")


def _clear_partials(band: str) -> None:
    from meerkat_beams.utils import log

    for p in (_partial(input_zarr_path(band)), _partial(bds_path(band))):
        if p.exists():
            log.warning(f"removing stale partial cache dir {p}")
            shutil.rmtree(p, ignore_errors=True)
