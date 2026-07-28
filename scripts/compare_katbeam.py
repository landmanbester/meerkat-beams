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

from meerkat_beams.utils import log

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


# --------------------------------------------------------------------------
# Geometry and shape helpers (pure)
# --------------------------------------------------------------------------

# Maps are (..., NY, NX): axis -2 is Y/m/north, axis -1 is X/l/east. The
# sweep's whole interpretation rests on flip_x touching X and flip_y Y, so
# tests pin that explicitly.
ORIENTATIONS = {
    "none": lambda a: a,
    "flip_x": lambda a: a[..., :, ::-1],
    "flip_y": lambda a: a[..., ::-1, :],
    "swap_xy": lambda a: np.swapaxes(a, -1, -2),
}


def apply_orientation(arr: np.ndarray, name: str) -> np.ndarray:
    """Apply a named axis perturbation to the trailing (Y, X) axes of ``arr``."""
    try:
        transform = ORIENTATIONS[name]
    except KeyError:
        raise ValueError(f"unknown orientation {name!r}, expected one of {sorted(ORIENTATIONS)}") from None
    return np.ascontiguousarray(transform(np.asarray(arr)))


def measure_fwhm(coord: np.ndarray, profile: np.ndarray) -> float:
    """Full width at half maximum of ``profile``, in ``coord`` units.

    Locates the peak, then linearly interpolates the half-power crossing on
    each side and returns their separation. Returns NaN when half power is not
    bracketed on both sides (e.g. a monotonic slice, or a beam wider than the
    sampled field).
    """
    coord = np.asarray(coord, dtype=float)
    profile = np.asarray(profile, dtype=float)
    if coord.shape != profile.shape:
        raise ValueError(f"coord and profile must have the same shape, got {coord.shape} and {profile.shape}")
    if not np.any(np.isfinite(profile)):
        return float("nan")

    ipk = int(np.nanargmax(profile))
    half = profile[ipk] / 2.0

    left = float("nan")
    for i in range(ipk, 0, -1):
        if profile[i - 1] <= half <= profile[i]:
            # np.interp needs ascending xp; profile[i-1] <= profile[i] here.
            left = float(np.interp(half, [profile[i - 1], profile[i]], [coord[i - 1], coord[i]]))
            break

    right = float("nan")
    for i in range(ipk, profile.size - 1):
        if profile[i + 1] <= half <= profile[i]:
            right = float(np.interp(half, [profile[i + 1], profile[i]], [coord[i + 1], coord[i]]))
            break

    if not (np.isfinite(left) and np.isfinite(right)):
        return float("nan")
    return right - left


def _radius_grid(l_deg: np.ndarray, m_deg: np.ndarray) -> np.ndarray:
    """Radial distance from the field centre on the (NY, NX) grid."""
    ll, mm = np.meshgrid(np.asarray(l_deg, dtype=float), np.asarray(m_deg, dtype=float))
    return np.sqrt(ll**2 + mm**2)


def azimuthal_profile(arr: np.ndarray, l_deg: np.ndarray, m_deg: np.ndarray, nbins: int = 64):
    """Azimuthally averaged radial profile of a single ``(NY, NX)`` map.

    Returns ``(r_centres, mean, std, count)``. ``std`` is the azimuthal scatter
    within each annulus: for a radially symmetric model it is ~0, so it
    measures how much real azimuthal structure a symmetric model discards.
    Non-finite samples are excluded rather than propagated, since katbeam's
    removable singularity can inject NaN.
    """
    arr = np.asarray(arr, dtype=float)
    r = _radius_grid(l_deg, m_deg)
    edges = np.linspace(0.0, float(r.max()), int(nbins) + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    bin_of = np.clip(np.digitize(r.ravel(), edges) - 1, 0, int(nbins) - 1)

    flat = arr.ravel()
    finite = np.isfinite(flat)
    mean = np.full(int(nbins), np.nan)
    std = np.full(int(nbins), np.nan)
    count = np.zeros(int(nbins), dtype=int)
    for b in range(int(nbins)):
        sel = finite & (bin_of == b)
        n = int(sel.sum())
        count[b] = n
        if n:
            vals = flat[sel]
            mean[b] = vals.mean()
            std[b] = vals.std()
    return centres, mean, std, count


def region_masks(l_deg: np.ndarray, m_deg: np.ndarray, hwhm_deg: float) -> dict[str, np.ndarray]:
    """Partition the field into mainlobe / near-sidelobe / far-sidelobe.

    katbeam only claims mainlobe fidelity, and the BDS field spans roughly
    8 HWHM at L-band, so a single field-wide residual would be dominated by a
    region the model makes no claim about. The three regions are disjoint and
    cover the field.
    """
    r = _radius_grid(l_deg, m_deg)
    hwhm = float(hwhm_deg)
    return {
        "mainlobe": r < hwhm,
        "near": (r >= hwhm) & (r < 3.0 * hwhm),
        "far": r >= 3.0 * hwhm,
    }


# --------------------------------------------------------------------------
# Frequency selection (pure)
# --------------------------------------------------------------------------


def overlap_indices(bds_freqs_hz: np.ndarray, model_freqs_mhz: np.ndarray, stride: int = 1) -> np.ndarray:
    """BDS channel indices lying inside the katbeam table's frequency range.

    katbeam interpolates its squint/FWHM table with ``np.interp``, which
    silently clamps (extrapolates flat) outside the table. Restricting to the
    overlap keeps that from being mistaken for a model difference.
    """
    bds = np.asarray(bds_freqs_hz, dtype=float)
    model = np.asarray(model_freqs_mhz, dtype=float)
    lo, hi = model.min() * 1e6, model.max() * 1e6
    idx = np.flatnonzero((bds >= lo) & (bds <= hi))
    if idx.size == 0:
        raise ValueError(
            f"no frequency overlap: BDS spans [{bds.min() * 1e-6:.1f}, {bds.max() * 1e-6:.1f}] MHz "
            f"but the katbeam model table spans [{model.min():.1f}, {model.max():.1f}] MHz"
        )
    return idx[:: int(stride)]


def select_native_freqs(
    bds_freqs_hz: np.ndarray,
    model_freqs_mhz: np.ndarray,
    requested_mhz=None,
    n: int = 5,
):
    """Pick BDS channels to compare at, snapping to native channel frequencies.

    Returns ``(indices, freqs_hz)``. Requested frequencies are clipped to the
    BDS/katbeam overlap and snapped to the nearest real channel, so no
    frequency interpolation enters the comparison. Duplicates are collapsed.
    """
    bds = np.asarray(bds_freqs_hz, dtype=float)
    usable = overlap_indices(bds, model_freqs_mhz)
    lo, hi = bds[usable].min(), bds[usable].max()

    if requested_mhz is None:
        targets = np.linspace(lo, hi, int(n))
    else:
        targets = np.clip(np.atleast_1d(np.asarray(requested_mhz, dtype=float)) * 1e6, lo, hi)

    chosen = np.unique([int(usable[np.argmin(np.abs(bds[usable] - t))]) for t in targets])
    return chosen, bds[chosen]


# --------------------------------------------------------------------------
# Residual statistics (pure)
# --------------------------------------------------------------------------


def residual_stats(
    ours: np.ndarray,
    theirs: np.ndarray,
    masks: dict[str, np.ndarray],
    frac_floor: float = 0.1,
) -> dict[str, dict[str, float]]:
    """Residual statistics of ``ours - theirs``, per named region.

    ``rms_diff_peaknorm`` repeats the RMS after dividing each map by its own
    peak. Our ``njones`` is normalised to exactly 1 on axis while katbeam's
    on-axis value is slightly below 1 because of squint, so the peak-normalised
    figure separates that constant offset from a genuine shape difference.
    ``median_frac_diff`` is gated to pixels where ``theirs > frac_floor`` so
    ratios are not taken through katbeam's nulls.
    """
    ours = np.asarray(ours, dtype=float)
    theirs = np.asarray(theirs, dtype=float)
    diff = ours - theirs

    def _peak_norm(arr):
        peak = np.nanmax(np.abs(arr))
        return arr / peak if np.isfinite(peak) and peak > 0 else arr

    diff_norm = _peak_norm(ours) - _peak_norm(theirs)

    out: dict[str, dict[str, float]] = {}
    for name, mask in masks.items():
        valid = mask & np.isfinite(diff)
        d = diff[valid]
        valid_norm = mask & np.isfinite(diff_norm)
        dn = diff_norm[valid_norm]
        gate = valid & (theirs > frac_floor)

        out[name] = {
            "n_pixels": int(valid.sum()),
            "max_abs_diff": float(np.max(np.abs(d))) if d.size else float("nan"),
            "rms_diff": float(np.sqrt(np.mean(d**2))) if d.size else float("nan"),
            "rms_diff_peaknorm": float(np.sqrt(np.mean(dn**2))) if dn.size else float("nan"),
            "n_pixels_frac": int(gate.sum()),
            "median_frac_diff": (float(np.median(diff[gate] / theirs[gate])) if gate.any() else float("nan")),
        }
    return out


# --------------------------------------------------------------------------
# Products
# --------------------------------------------------------------------------

# "xpol" is our cross-pol power. katbeam models it as identically zero, so it
# is carried as its own product rather than folded into a residual.
PRODUCTS: tuple[str, ...] = ("I", "HH", "VV", "xpol")


def resolve_bds(band, bds) -> str:
    """Resolve the BDS to compare against, from a band name or an explicit path.

    ``band`` goes through ``cache.ensure_band_bds`` so the BDS is always built
    by this repo's current ``mdv_beams_to_bds``. That matters: the orientation
    sweep asks a question about the converter's axis convention, and a legacy
    BDS (e.g. the ``MBEAMS_REFERENCE_BDS_*`` regression references, which
    predate commit 616906b) may carry a stale convention of its own, which
    would make the sweep's answer confidently wrong.
    """
    if (band is None) == (bds is None):
        raise ValueError("exactly one of --band or --bds must be given")

    if bds is not None:
        log.warning(
            "using an explicit --bds path: if this BDS was not produced by this repo's "
            "current mdv_beams_to_bds, its axis convention may be stale and the "
            "orientation sweep result must not be trusted. Prefer --band."
        )
        return str(bds)

    if band not in KATBEAM_MODEL_FOR_BAND:
        raise ValueError(f"no katbeam model for band {band!r}, known bands: {sorted(KATBEAM_MODEL_FOR_BAND)}")

    from meerkat_beams import cache

    return cache.ensure_band_bds(band)


def load_ours(xds, chan_indices) -> dict[str, np.ndarray]:
    """Read our four beam products straight out of the BDS at native channels.

    Read directly rather than through ``BeamWizard.interpolate_beam``: the
    question is how the two *models* differ, so spline error must not enter.
    Returns ``(NFREQ, NY, NX)`` float64 arrays.
    """
    chans = [int(c) for c in chan_indices]
    stokes_i = xds["nstokes"].isel(FREQ=chans).sel(stokes_i="I", stokes_j="I").values
    njones = xds["njones"].isel(FREQ=chans).values  # (2, 2, NFREQ, NY, NX)

    return {
        "I": np.asarray(stokes_i, dtype=float),
        "HH": np.abs(njones[0, 0]).astype(float) ** 2,
        "VV": np.abs(njones[1, 1]).astype(float) ** 2,
        "xpol": np.abs(njones[0, 1]).astype(float) ** 2 + np.abs(njones[1, 0]).astype(float) ** 2,
    }


def eval_katbeam(model_name: str, l_deg, m_deg, freqs_mhz) -> dict[str, np.ndarray]:
    """Evaluate katbeam on the same grid, as power beams.

    katbeam's ``HH``/``VV`` are voltage patterns, so they are squared here to
    match our ``|njones|**2``. ``meshgrid(l, m)`` gives ``(NY, NX)``, matching
    the BDS map order. Returns ``(NFREQ, NY, NX)`` arrays.
    """
    jb = require_model(model_name)
    ll, mm = np.meshgrid(np.asarray(l_deg, dtype=float), np.asarray(m_deg, dtype=float))

    hh_planes, vv_planes = [], []
    for f in np.atleast_1d(np.asarray(freqs_mhz, dtype=float)):
        hh = np.asarray(jb.HH(ll, mm, float(f)), dtype=float)
        vv = np.asarray(jb.VV(ll, mm, float(f)), dtype=float)
        hh_planes.append(np.abs(hh) ** 2)
        vv_planes.append(np.abs(vv) ** 2)

    hh_arr = np.stack(hh_planes)
    vv_arr = np.stack(vv_planes)
    return {
        "I": 0.5 * (hh_arr + vv_arr),
        "HH": hh_arr,
        "VV": vv_arr,
        "xpol": np.zeros_like(hh_arr),
    }


def count_non_finite(maps: dict[str, np.ndarray]) -> dict[str, int]:
    """Per-product count of non-finite samples.

    katbeam's ``cos(pi*rr)/(1 - 4*rr**2)`` is 0/0 at ``rr = 0.5``
    (``r ~ 0.42053`` in FWHM units). A grid point landing there yields NaN,
    which must be reported rather than silently skewing statistics.
    """
    return {name: int(np.count_nonzero(~np.isfinite(arr))) for name, arr in maps.items()}
