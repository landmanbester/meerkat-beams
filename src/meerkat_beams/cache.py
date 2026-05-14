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
