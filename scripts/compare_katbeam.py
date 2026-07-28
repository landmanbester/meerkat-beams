#!/usr/bin/env python
"""
Compare katbeam's analytic JimBeam primary beam against the holography-derived
beam this repo builds from MdV data.

katbeam models the co-pol beams as an elliptical, squinted cosine aperture
taper whose only parameters are per-axis FWHM and squint, interpolated in
frequency from a table measured at 60 degrees elevation. Its Stokes I is
0.5 * (|HH|**2 + |VV|**2), with no cross-polarisation at all. Our BDS beams
come from the full MdV Jones matrix, so they carry cross-pol, complex phase,
and real sidelobe structure.

Three consequences shape the comparison:

  1. Cross-pol is a term katbeam sets to exactly zero, so it is reported on
     its own rather than buried inside a residual.
  2. katbeam's taper keeps ringing past the mainlobe with a 1/r**2 envelope,
     and katbeam's own docstring disclaims sidelobe accuracy. The BDS grid
     spans roughly 8 HWHM at L-band, so metrics are partitioned by radius and
     never aggregated into one field-wide number.
  3. cos(pi*rr)/(1 - 4*rr**2) is 0/0 at rr = 0.5 (r ~ 0.42053 in FWHM units),
     so a grid point landing there yields NaN. Non-finite katbeam samples are
     counted and reported, not silently propagated.

Our side is read directly from BDS variables at native pixel centres and
native frequency channels -- deliberately NOT through
BeamWizard.interpolate_beam -- so spline and frequency-interpolation error do
not contaminate a model-vs-model question.

Usage:
    python scripts/compare_katbeam.py --band L
    python scripts/compare_katbeam.py --band L --freqs 900 1284 1650

Requires the dev dependency group: `uv sync --group dev --extra full`.
"""

import numpy as np

# --------------------------------------------------------------------------
# Band -> katbeam model
# --------------------------------------------------------------------------

# katbeam ships one table per receiver band. MdV splits S into five
# sub-bands (S0..S4) but katbeam has a single S model spanning 1750-3450 MHz,
# so all five map to it.
KATBEAM_MODEL_FOR_BAND: dict[str, str] = {
    "U": "MKAT-AA-UHF-JIM-2020",  # 550-1050 MHz
    "L": "MKAT-AA-L-JIM-2020",  # 856-1712 MHz
    "S0": "MKAT-AA-S-JIM-2020",  # 1750-3450 MHz
    "S1": "MKAT-AA-S-JIM-2020",
    "S2": "MKAT-AA-S-JIM-2020",
    "S3": "MKAT-AA-S-JIM-2020",
    "S4": "MKAT-AA-S-JIM-2020",
}


def _import_jimbeam():
    """Import JimBeam lazily so the pure helpers stay importable without katbeam."""
    try:
        from katbeam import JimBeam
    except ImportError as e:  # pragma: no cover - exercised only without katbeam
        raise ImportError(
            "scripts/compare_katbeam.py needs katbeam, which is not installed. "
            "It lives in this project's dev and test dependency groups: run "
            "`uv sync --group dev --extra full`."
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
