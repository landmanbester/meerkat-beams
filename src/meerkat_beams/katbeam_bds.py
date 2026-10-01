"""Synthesize a BDS-shaped dataset from katbeam's analytic MeerKAT beams.

katbeam (https://github.com/ska-sa/katbeam) fits a cosine-aperture taper to
holography measurements, giving a cheap approximate beam with no download. This
module samples it onto a grid and assembles an in-memory, dask-backed Dataset
carrying the same schema as an MdV BDS, so BeamWizard can consume either one
without changes. See docs/wiki/data-model.md.

katbeam is imported lazily throughout: it lives in the [full] extra, and the
base install must stay importable without it.
"""

from typing import Optional, Tuple

import numpy as np

# katbeam ships one table per receiver band. MdV splits S into five sub-bands
# (S0..S4) but katbeam has a single S model spanning 1750-3450 MHz, so all five
# map to it.
KATBEAM_MODEL_FOR_BAND: dict[str, str] = {
    "U": "MKAT-AA-UHF-JIM-2020",  # 550-1050 MHz
    "L": "MKAT-AA-L-JIM-2020",  # 856-1712 MHz
    "S0": "MKAT-AA-S-JIM-2020",  # 1750-3450 MHz
    "S1": "MKAT-AA-S-JIM-2020",
    "S2": "MKAT-AA-S-JIM-2020",
    "S3": "MKAT-AA-S-JIM-2020",
    "S4": "MKAT-AA-S-JIM-2020",
}

# Default (npix, fov_deg) per band, where fov_deg is a HALF-width: the grid
# spans -fov_deg to +fov_deg - dx. Values are read off the real MdV BDSs so a
# synthesized katbeam BDS and an MdV BDS for the same band are pixel-identical
# and directly comparable. S-band MdV geometry has not been measured yet, so
# those bands have no default and require explicit npix/fov_deg.
BAND_GEOMETRY: dict[str, Tuple[int, float]] = {
    "L": (128, 4.0),  # dx = 0.0625 deg
    "U": (128, 6.0),  # dx = 0.09375 deg
}


def _import_jimbeam():
    """Import JimBeam lazily so this module stays importable without katbeam."""
    try:
        from katbeam import JimBeam
    except ImportError as e:  # pragma: no cover - exercised only without katbeam
        raise ImportError(
            "the katbeam beam model needs the katbeam package, which is not installed. "
            "It lives in this project's [full] extra and its dev/test groups: run "
            "`uv sync --group dev --group test --extra full`."
        ) from e
    return JimBeam


def require_model(model_name: str):
    """Return a ``JimBeam`` for ``model_name``, or raise an actionable error.

    ``JimBeam`` treats an unrecognised name as a *filename* and fails deep
    inside ``np.loadtxt``, which is an unhelpful way to discover that the
    installed katbeam is too old. The only PyPI release (0.1) has no S-band
    model, so this is a live failure mode; check the name up front and name the
    models that are actually available.
    """
    JimBeam = _import_jimbeam()
    from katbeam.jimbeam import KNOWN_MODELS

    if model_name not in KNOWN_MODELS:
        raise ValueError(
            f"katbeam model {model_name!r} is not available in the installed katbeam. "
            f"Available models: {sorted(KNOWN_MODELS)}. "
            "The PyPI release (0.1) predates the S-band model; this project pins "
            "katbeam from git main in its dev and test dependency groups."
        )
    return JimBeam(model_name)


def katbeam_freq_table(model_name: str) -> np.ndarray:
    """Frequencies (MHz) at which ``model_name``'s squint/FWHM table is defined."""
    return np.asarray(require_model(model_name).freqMHzlist, dtype=float)


def resolve_geometry(band: str, npix: Optional[int], fov_deg: Optional[float]) -> Tuple[int, float]:
    """Resolve the (npix, fov_deg) grid geometry for ``band``.

    Explicit arguments win. Missing ones fall back to BAND_GEOMETRY, which
    mirrors MdV. A band with no default raises rather than guessing: a wrong
    field of view silently produces a wrong beam.
    """
    if band not in KATBEAM_MODEL_FOR_BAND:
        raise ValueError(f"unknown band {band!r}; expected one of {sorted(KATBEAM_MODEL_FOR_BAND)}")
    if npix is not None and fov_deg is not None:
        return int(npix), float(fov_deg)
    if band not in BAND_GEOMETRY:
        raise ValueError(
            f"no default grid geometry for band {band!r}, so npix and fov_deg must both be given. "
            f"Defaults exist for {sorted(BAND_GEOMETRY)} only (read off their MdV BDSs); "
            "S-band MdV geometry has not been measured yet."
        )
    default_npix, default_fov = BAND_GEOMETRY[band]
    return int(npix if npix is not None else default_npix), float(fov_deg if fov_deg is not None else default_fov)
