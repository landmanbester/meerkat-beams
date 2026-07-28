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

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402  headless: this script only ever writes PNGs

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from meerkat_beams.utils import log  # noqa: E402

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


# --------------------------------------------------------------------------
# HWHM, orientation sweep, metrics
# --------------------------------------------------------------------------

# Products used to score the orientation sweep. HH and VV carry the
# discriminating power (katbeam's per-axis FWHM differ by ~5% and its squint is
# one-sided); I is included as a control that is expected to be weakly
# discriminating, because a nearly-symmetric beam barely changes under a
# transpose. That degeneracy is exactly what let the earlier (X, Y)/(Y, X) bug
# hide -- see docs/wiki/beam-orientation.md.
SWEEP_PRODUCTS: tuple[str, ...] = ("HH", "VV", "I")


def beam_hwhm(plane: np.ndarray, l_deg, m_deg, x0: int, y0: int) -> float:
    """Half width at half maximum of a ``(NY, NX)`` beam plane, in degrees.

    Averages the FWHM measured along the l and m cuts through the field centre
    and halves it. Used to set the mainlobe/near/far region boundaries.
    """
    plane = np.asarray(plane, dtype=float)
    fwhm_l = measure_fwhm(np.asarray(l_deg, dtype=float), plane[int(y0), :])
    fwhm_m = measure_fwhm(np.asarray(m_deg, dtype=float), plane[:, int(x0)])
    return 0.5 * float(np.nanmean([fwhm_l, fwhm_m]))


def orientation_sweep(ours, theirs, l_deg, m_deg, hwhm_deg, products=SWEEP_PRODUCTS) -> dict:
    """Score each axis perturbation of *our* maps against fixed katbeam maps.

    katbeam's ``(ll_deg, mm_deg)`` contract is unambiguous and documented (SIN
    projection, degrees), so it is the reference frame and the perturbation is
    applied to our side. The winning label therefore reads directly as a
    statement about the BDS axis convention.

    Score is the mainlobe RMS residual averaged over the compared frequencies:
    lower is better. Returns per-product scores, the per-product winner, and
    the overall winner (summed across products).
    """
    l_arr = np.asarray(l_deg, dtype=float)
    m_arr = np.asarray(m_deg, dtype=float)
    if l_arr.size != m_arr.size:
        raise ValueError(
            f"orientation_sweep needs a square grid to score swap_xy, got len(l)={l_arr.size} and len(m)={m_arr.size}"
        )

    masks = region_masks(l_arr, m_arr, hwhm_deg)
    per_product: dict[str, dict[str, float]] = {}
    for product in products:
        scores: dict[str, float] = {}
        for name in ORIENTATIONS:
            moved = apply_orientation(ours[product], name)
            per_freq = [
                residual_stats(moved[k], theirs[product][k], masks)["mainlobe"]["rms_diff"]
                for k in range(moved.shape[0])
            ]
            scores[name] = float(np.nanmean(per_freq))
        per_product[product] = scores

    best = {p: min(s, key=lambda n: s[n]) for p, s in per_product.items()}
    totals = {name: float(np.nansum([per_product[p][name] for p in per_product])) for name in ORIENTATIONS}
    return {
        "per_product": per_product,
        "best": best,
        "best_overall": min(totals, key=lambda n: totals[n]),
        "totals": totals,
    }


def build_metrics(ours, theirs, l_deg, m_deg, freqs_hz, x0: int, y0: int) -> dict:
    """Assemble the full per-frequency, per-product, per-region metric tree."""
    l_arr = np.asarray(l_deg, dtype=float)
    m_arr = np.asarray(m_deg, dtype=float)
    freqs = np.asarray(freqs_hz, dtype=float)

    per_freq = []
    for k, f in enumerate(freqs):
        hwhm = beam_hwhm(ours["I"][k], l_arr, m_arr, x0, y0)
        masks = region_masks(l_arr, m_arr, hwhm)

        fwhm_ours_l = measure_fwhm(l_arr, ours["I"][k][int(y0), :])
        fwhm_ours_m = measure_fwhm(m_arr, ours["I"][k][:, int(x0)])
        fwhm_kb_l = measure_fwhm(l_arr, theirs["I"][k][int(y0), :])
        fwhm_kb_m = measure_fwhm(m_arr, theirs["I"][k][:, int(x0)])

        per_freq.append(
            {
                "freq_mhz": float(f * 1e-6),
                "hwhm_deg": float(hwhm),
                "fwhm_deg": {
                    "ours_l": float(fwhm_ours_l),
                    "ours_m": float(fwhm_ours_m),
                    "katbeam_l": float(fwhm_kb_l),
                    "katbeam_m": float(fwhm_kb_m),
                    "ratio_l": float(fwhm_ours_l / fwhm_kb_l),
                    "ratio_m": float(fwhm_ours_m / fwhm_kb_m),
                },
                "peak": {
                    p: {"ours": float(np.nanmax(ours[p][k])), "katbeam": float(np.nanmax(theirs[p][k]))}
                    for p in PRODUCTS
                },
                "katbeam_non_finite": count_non_finite({p: theirs[p][k] for p in PRODUCTS}),
                "residuals": {p: residual_stats(ours[p][k], theirs[p][k], masks) for p in PRODUCTS},
            }
        )

    return {"per_freq": per_freq}


def format_summary_table(metrics: dict) -> str:
    """Fixed-width per-frequency residual table for stdout and summary.md."""
    header = f"{'freq/MHz':>9} {'product':>8} {'region':>9} {'rms':>11} {'max|d|':>11} {'med frac':>10}"
    lines = [header, "-" * len(header)]
    for entry in metrics["per_freq"]:
        for product, regions in entry["residuals"].items():
            for region in ("mainlobe", "near", "far"):
                s = regions[region]
                lines.append(
                    f"{entry['freq_mhz']:>9.1f} {product:>8} {region:>9} "
                    f"{s['rms_diff']:>11.3e} {s['max_abs_diff']:>11.3e} {s['median_frac_diff']:>10.3f}"
                )
    return "\n".join(lines)


def _json_safe(obj):
    """Recursively replace non-finite floats with None so json stays valid."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    return obj


def write_outputs(metrics: dict, sweep: dict, out_dir) -> None:
    """Write metrics.json and summary.md into ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    payload = dict(metrics)
    payload["orientation_sweep"] = sweep
    # allow_nan=False would raise; non-finite values become null instead.
    (out / "metrics.json").write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n")

    table = format_summary_table(metrics)
    body = [
        "# compare_katbeam summary",
        "",
        "Residuals of our MdV-derived beam minus katbeam's analytic JimBeam,",
        "partitioned by radius. katbeam only claims mainlobe fidelity, so the",
        "`near` and `far` rows are context, not a verdict on the model.",
        "",
        "```",
        table,
        "```",
        "",
        "## Orientation sweep",
        "",
        "Mainlobe RMS residual with each axis perturbation applied to *our* maps",
        "(katbeam held fixed). Lower is better; `none` winning is consistent with",
        "the BDS axis convention being correct as written.",
        "",
        f"- best overall: **{sweep.get('best_overall', 'n/a')}**",
    ]
    for product, scores in sweep.get("per_product", {}).items():
        rendered = ", ".join(f"{n}={v:.3e}" for n, v in scores.items())
        body.append(f"- {product}: {rendered} -> best `{sweep['best'][product]}`")
    (out / "summary.md").write_text("\n".join(body) + "\n")


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

DPI = 120

# Palette: Okabe-Ito, validated with the dataviz skill's validate_palette.js
# (all-pairs, light AND dark surfaces). matplotlib's default C0-C3 FAILS that
# check -- #2ca02c green vs #ff7f0e orange are deltaE 0.7 apart under
# protanopia, i.e. indistinguishable -- so the orientation bars must not fall
# back to the default cycle. Pinned by test_orientation_colours_are_the_cvd_safe_set.
BLUE = "#0072B2"
VERMILLION = "#D55E00"
SKY = "#56B4E9"
AMBER = "#E69F00"

# ours vs katbeam are distinguished by colour AND linestyle, never colour alone.
STYLE_OURS = dict(color=BLUE, linestyle="-", label="ours (MdV)")
STYLE_KATBEAM = dict(color=VERMILLION, linestyle="--", label="katbeam")

ORIENTATION_COLOURS = {
    "none": BLUE,
    "flip_x": VERMILLION,
    "flip_y": SKY,
    "swap_xy": AMBER,
}

# Sequential (magnitude) and diverging (polarity) colormaps. "viridis" is a
# single perceptually-uniform ramp; "RdBu_r" is two hues about a neutral
# midpoint. Never a rainbow, and never a hue at the diverging midpoint.
CMAP_MAGNITUDE = "viridis"
CMAP_RESIDUAL = "RdBu_r"


def _extent(l_deg, m_deg):
    """imshow extent for an (NY, NX) map with l on x and m on y."""
    l_arr = np.asarray(l_deg, dtype=float)
    m_arr = np.asarray(m_deg, dtype=float)
    return [l_arr[0], l_arr[-1], m_arr[0], m_arr[-1]]


def _symmetric_limit(arr):
    """Largest finite absolute value, for a symmetric diverging norm."""
    finite = np.abs(np.asarray(arr, dtype=float))
    finite = finite[np.isfinite(finite)]
    v = float(finite.max()) if finite.size else 1.0
    return v if v > 0 else 1.0


def _recessive_grid(ax):
    """Grid and spines stay behind the data, never competing with it."""
    ax.grid(alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_maps(ours_plane, theirs_plane, l_deg, m_deg, out_path, *, title, cbar_label):
    """Three panels: ours, katbeam, and their difference."""
    diff = np.asarray(ours_plane, dtype=float) - np.asarray(theirs_plane, dtype=float)
    v = _symmetric_limit(diff)
    extent = _extent(l_deg, m_deg)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, data, label in (
        (axes[0], ours_plane, "ours (MdV)"),
        (axes[1], theirs_plane, "katbeam"),
    ):
        im = ax.imshow(data, origin="lower", extent=extent, interpolation="nearest", cmap=CMAP_MAGNITUDE)
        ax.set_title(label)
        fig.colorbar(im, ax=ax, label=cbar_label)

    im = axes[2].imshow(
        diff, origin="lower", extent=extent, interpolation="nearest", cmap=CMAP_RESIDUAL, vmin=-v, vmax=v
    )
    axes[2].set_title("ours - katbeam")
    fig.colorbar(im, ax=axes[2], label="difference")

    for ax in axes:
        ax.set_xlabel("l (deg)")
    axes[0].set_ylabel("m (deg)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_cuts(ours_plane, theirs_plane, l_deg, m_deg, x0, y0, out_path, *, title):
    """Axis cuts through the field centre, log-y, with a residual subpanel."""
    ours_plane = np.asarray(ours_plane, dtype=float)
    theirs_plane = np.asarray(theirs_plane, dtype=float)
    l_arr = np.asarray(l_deg, dtype=float)
    m_arr = np.asarray(m_deg, dtype=float)

    cuts = (
        ("l cut (m = 0)", l_arr, ours_plane[int(y0), :], theirs_plane[int(y0), :], "l (deg)"),
        ("m cut (l = 0)", m_arr, ours_plane[:, int(x0)], theirs_plane[:, int(x0)], "m (deg)"),
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex="col", gridspec_kw={"height_ratios": [3, 1]})
    for col, (label, coord, ours_cut, theirs_cut, xlabel) in enumerate(cuts):
        top, bottom = axes[0, col], axes[1, col]
        top.plot(coord, np.abs(ours_cut), **STYLE_OURS)
        top.plot(coord, np.abs(theirs_cut), **STYLE_KATBEAM)
        top.set_yscale("log")
        top.set_title(label)
        top.set_ylabel("beam power")
        top.legend()
        _recessive_grid(top)

        bottom.plot(coord, ours_cut - theirs_cut, color="k", linewidth=1)
        bottom.axhline(0.0, color="grey", linewidth=0.8)
        bottom.set_xlabel(xlabel)
        bottom.set_ylabel("residual")
        _recessive_grid(bottom)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_radial(ours_plane, theirs_plane, l_deg, m_deg, out_path, *, title, nbins=64):
    """Azimuthally averaged profiles, with our azimuthal scatter as a shaded band.

    katbeam's own band is essentially zero-width by construction, so the width
    of our band is a direct picture of the azimuthal structure the analytic
    model discards.
    """
    r, mean_ours, std_ours, count = azimuthal_profile(ours_plane, l_deg, m_deg, nbins=nbins)
    _, mean_kb, std_kb, _ = azimuthal_profile(theirs_plane, l_deg, m_deg, nbins=nbins)
    ok = count > 0

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    top.plot(r[ok], mean_ours[ok], **STYLE_OURS)
    top.fill_between(
        r[ok],
        mean_ours[ok] - std_ours[ok],
        mean_ours[ok] + std_ours[ok],
        color=STYLE_OURS["color"],
        alpha=0.25,
        label="ours: +/-1 sigma azimuthal scatter",
    )
    top.plot(r[ok], mean_kb[ok], **STYLE_KATBEAM)
    top.fill_between(
        r[ok],
        mean_kb[ok] - std_kb[ok],
        mean_kb[ok] + std_kb[ok],
        color=STYLE_KATBEAM["color"],
        alpha=0.25,
        label="katbeam: +/-1 sigma",
    )
    top.set_yscale("log")
    top.set_ylabel("azimuthally averaged beam power")
    top.legend()
    _recessive_grid(top)

    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(mean_kb > 0.1, (mean_ours - mean_kb) / mean_kb, np.nan)
    bottom.plot(r[ok], frac[ok], color="k", linewidth=1)
    bottom.axhline(0.0, color="grey", linewidth=0.8)
    bottom.set_xlabel("radius (deg)")
    bottom.set_ylabel("frac. diff\n(where katbeam > 0.1)")
    _recessive_grid(bottom)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_hhvv(ours, theirs, k, l_deg, m_deg, out_path, *, title):
    """HH and VV power: ours / katbeam / difference, one row each.

    This is where squint and per-axis ellipticity show up, and it is the panel
    that carries the orientation sweep's discriminating power.
    """
    extent = _extent(l_deg, m_deg)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for row, product in enumerate(("HH", "VV")):
        ours_plane = np.asarray(ours[product][k], dtype=float)
        theirs_plane = np.asarray(theirs[product][k], dtype=float)
        diff = ours_plane - theirs_plane
        v = _symmetric_limit(diff)

        im = axes[row, 0].imshow(
            ours_plane, origin="lower", extent=extent, interpolation="nearest", cmap=CMAP_MAGNITUDE
        )
        axes[row, 0].set_title(f"{product} ours (MdV)")
        fig.colorbar(im, ax=axes[row, 0], label="power")

        im = axes[row, 1].imshow(
            theirs_plane, origin="lower", extent=extent, interpolation="nearest", cmap=CMAP_MAGNITUDE
        )
        axes[row, 1].set_title(f"{product} katbeam")
        fig.colorbar(im, ax=axes[row, 1], label="power")

        im = axes[row, 2].imshow(
            diff, origin="lower", extent=extent, interpolation="nearest", cmap=CMAP_RESIDUAL, vmin=-v, vmax=v
        )
        axes[row, 2].set_title(f"{product} ours - katbeam")
        fig.colorbar(im, ax=axes[row, 2], label="difference")

        axes[row, 0].set_ylabel("m (deg)")
        for ax in axes[row]:
            ax.set_xlabel("l (deg)")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_crosspol(ours_plane, l_deg, m_deg, out_path, *, title):
    """Our cross-pol power, which katbeam models as identically zero."""
    arr = np.asarray(ours_plane, dtype=float)
    positive = arr[np.isfinite(arr) & (arr > 0)]
    extent = _extent(l_deg, m_deg)

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.6))
    if positive.size:
        norm = LogNorm(vmin=max(positive.min(), positive.max() * 1e-6), vmax=positive.max())
        im = left.imshow(arr, origin="lower", extent=extent, interpolation="nearest", norm=norm, cmap=CMAP_MAGNITUDE)
    else:
        # A perfectly diagonal Jones gives identically zero cross-pol; LogNorm
        # cannot render that, so fall back to a linear scale.
        im = left.imshow(arr, origin="lower", extent=extent, interpolation="nearest", cmap=CMAP_MAGNITUDE)
    left.set_title("ours: |Jhv|^2 + |Jvh|^2")
    left.set_xlabel("l (deg)")
    left.set_ylabel("m (deg)")
    fig.colorbar(im, ax=left, label="cross-pol power")

    r, mean, _, count = azimuthal_profile(arr, l_deg, m_deg, nbins=48)
    ok = count > 0
    right.plot(r[ok], mean[ok], color=BLUE, linestyle="-", label="ours")
    right.axhline(0.0, color=VERMILLION, linestyle="--", label="katbeam (identically 0)")
    if positive.size:
        right.set_yscale("log")
    right.set_xlabel("radius (deg)")
    right.set_ylabel("azimuthally averaged cross-pol power")
    right.legend()
    _recessive_grid(right)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_fwhm_vs_freq(freqs_hz, fwhm_table, out_path):
    """Measured FWHM along each axis for both models across the whole overlap."""
    freq_mhz = np.asarray(freqs_hz, dtype=float) * 1e-6
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    top.plot(freq_mhz, fwhm_table["ours_l"], color=BLUE, linestyle="-", marker="o", ms=4, label="ours l")
    top.plot(freq_mhz, fwhm_table["ours_m"], color=BLUE, linestyle=":", marker="s", ms=4, label="ours m")
    top.plot(freq_mhz, fwhm_table["katbeam_l"], color=VERMILLION, linestyle="--", marker="o", ms=4, label="katbeam l")
    top.plot(freq_mhz, fwhm_table["katbeam_m"], color=VERMILLION, linestyle="-.", marker="s", ms=4, label="katbeam m")
    top.set_ylabel("FWHM (deg)")
    top.legend(ncol=2)
    _recessive_grid(top)

    with np.errstate(divide="ignore", invalid="ignore"):
        bottom.plot(
            freq_mhz,
            np.asarray(fwhm_table["ours_l"]) / np.asarray(fwhm_table["katbeam_l"]),
            color=BLUE,
            linestyle="-",
            label="l",
        )
        bottom.plot(
            freq_mhz,
            np.asarray(fwhm_table["ours_m"]) / np.asarray(fwhm_table["katbeam_m"]),
            color=VERMILLION,
            linestyle="--",
            label="m",
        )
    bottom.axhline(1.0, color="grey", linewidth=0.8)
    bottom.set_xlabel("frequency (MHz)")
    bottom.set_ylabel("ours / katbeam")
    bottom.legend()
    _recessive_grid(bottom)

    fig.suptitle("Stokes I FWHM vs frequency")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_orientation_residuals(sweep, out_path):
    """Two views of the sweep: the decision (left) and the magnitudes (right).

    Left panel is *ratio to that product's best score* on a LINEAR axis. Bars
    encode magnitude by length, so a log axis would break the proportionality
    they exist for and put the baseline somewhere arbitrary. The ratio is the
    quantity the sweep actually asks about -- "which perturbation wins, and by
    how much" -- it is dimensionless so products are comparable, and a
    non-discriminating result is immediately visible as every bar sitting at
    1.0.

    Right panel carries the absolute residuals, which do span decades, as a dot
    plot on a log axis. Dots mark a position rather than a length, so a log
    axis is legitimate there.
    """
    per_product = sweep["per_product"]
    products = list(per_product)
    names = list(ORIENTATIONS)
    if not products:
        products, names = ["(none run)"], list(ORIENTATIONS)
        per_product = {products[0]: dict.fromkeys(names, float("nan"))}

    width = 0.8 / max(len(names), 1)
    positions = np.arange(len(products), dtype=float)

    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.2))

    for i, name in enumerate(names):
        raw = np.array([per_product[p].get(name, np.nan) for p in products], dtype=float)
        best = np.array(
            [np.nanmin(list(per_product[p].values())) if per_product[p] else np.nan for p in products],
            dtype=float,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(best > 0, raw / best, np.nan)
        bars = left.bar(
            positions + i * width,
            ratio,
            width=width * 0.92,  # surface gap between adjacent bars
            color=ORIENTATION_COLOURS[name],
            label=name,
        )
        # A contrast WARN on the lighter steps obligates visible labels.
        left.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
        right.plot(
            positions + i * width,
            raw,
            marker="o",
            markersize=9,
            linestyle="none",
            color=ORIENTATION_COLOURS[name],
            markeredgecolor="white",
            markeredgewidth=1.2,  # surface ring on overlapping marks
            label=name,
        )

    left.axhline(1.0, color="grey", linewidth=0.8)
    left.set_xticks(positions + 0.4 - width / 2)
    left.set_xticklabels(products)
    left.set_ylabel("mainlobe RMS residual / best for that product")
    left.set_xlabel("product")
    left.set_title("Relative: 1.0 = winner, all ~1.0 = not discriminating")
    left.grid(alpha=0.3, axis="y", linewidth=0.6)
    left.set_axisbelow(True)
    # Headroom so the winning bars and their labels are not flush with the frame.
    left.set_ylim(0.0, float(np.nanmax([1.0, *left.get_ylim()])) * 1.18)

    right.set_xticks(positions + 0.4 - width / 2)
    right.set_xticklabels(products)
    right.set_yscale("log")
    right.set_ylabel("mainlobe RMS residual (absolute)")
    right.set_xlabel("product")
    right.set_title("Absolute magnitudes")
    right.grid(alpha=0.3, axis="y", linewidth=0.6)
    right.set_axisbelow(True)
    right.margins(y=0.18)  # keep markers off the frame on the log axis

    # Legend below the figure: inside the axes it collided with the bars and
    # their value labels.
    handles, labels = left.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="perturbation applied to our maps",
        loc="lower center",
        ncol=len(names),
        fontsize=9,
        frameon=False,
    )
    fig.suptitle(f"Orientation sweep -- best overall: {sweep.get('best_overall', 'n/a')}")
    fig.tight_layout(rect=(0, 0.10, 1, 0.96))
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Compare katbeam's analytic JimBeam against this repo's MdV-derived beam.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--band",
        default=None,
        help="Receiver band; the BDS is built from scratch via the on-demand cache. "
        f"One of {sorted(KATBEAM_MODEL_FOR_BAND)}.",
    )
    parser.add_argument(
        "--bds",
        default=None,
        help="Explicit BDS path instead of --band. Escape hatch only: a BDS not built by "
        "this repo's current mdv_beams_to_bds may carry a stale axis convention, which "
        "would invalidate the orientation sweep.",
    )
    parser.add_argument(
        "--band-model",
        default=None,
        help="Band whose katbeam model to use. Required with --bds; defaults to --band.",
    )
    parser.add_argument(
        "--freqs", type=float, nargs="+", default=None, help="Frequencies in MHz. Default: 5 across the overlap."
    )
    parser.add_argument("--n-freqs", type=int, default=5, help="How many frequencies when --freqs is omitted.")
    parser.add_argument("--nbins", type=int, default=64, help="Radial bins for azimuthal profiles.")
    parser.add_argument("--fwhm-stride", type=int, default=16, help="Channel stride for the FWHM-vs-frequency sweep.")
    parser.add_argument("--output-dir", default=None, help="Default: outputs/compare_katbeam/<band>/")
    parser.add_argument("--no-orientation-sweep", action="store_true", help="Skip the axis perturbation sweep.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    bds_path = resolve_bds(args.band, args.bds)

    model_band = args.band_model or args.band
    if model_band is None:
        raise ValueError("--band-model is required when --bds is used, since a path carries no band information")
    if model_band not in KATBEAM_MODEL_FOR_BAND:
        raise ValueError(f"no katbeam model for band {model_band!r}, known bands: {sorted(KATBEAM_MODEL_FOR_BAND)}")
    model_name = KATBEAM_MODEL_FOR_BAND[model_band]
    log.info(f"comparing BDS {bds_path} against katbeam model {model_name}")

    out_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / "compare_katbeam" / str(model_band)
    out_dir.mkdir(parents=True, exist_ok=True)

    xds = xarray.open_zarr(bds_path)
    l_deg = np.asarray(xds.coords["X"].values, dtype=float)
    m_deg = np.asarray(xds.coords["Y"].values, dtype=float)
    bds_freqs = np.asarray(xds.coords["FREQ"].values, dtype=float)
    x0 = int(xds.attrs["x0"])
    y0 = int(xds.attrs["y0"])

    model_freqs = katbeam_freq_table(model_name)
    chans, freqs = select_native_freqs(bds_freqs, model_freqs, requested_mhz=args.freqs, n=args.n_freqs)
    log.info(f"comparing at {[f'{f * 1e-6:.1f}' for f in freqs]} MHz (native BDS channels {list(chans)})")

    ours = load_ours(xds, chans)
    theirs = eval_katbeam(model_name, l_deg, m_deg, freqs * 1e-6)

    non_finite = count_non_finite(theirs)
    if any(non_finite.values()):
        log.warning(
            f"katbeam produced non-finite samples {non_finite} -- this is the removable "
            "0/0 singularity at rr = 0.5; those pixels are excluded from statistics"
        )

    metrics = build_metrics(ours, theirs, l_deg, m_deg, freqs, x0, y0)

    if args.no_orientation_sweep:
        sweep = {"per_product": {}, "best": {}, "best_overall": "not run"}
    else:
        hwhm = float(np.nanmean([e["hwhm_deg"] for e in metrics["per_freq"]]))
        sweep = orientation_sweep(
            {p: ours[p] for p in SWEEP_PRODUCTS},
            {p: theirs[p] for p in SWEEP_PRODUCTS},
            l_deg,
            m_deg,
            hwhm,
        )
        log.info(f"orientation sweep best overall: {sweep['best_overall']} (per product: {sweep['best']})")
        plot_orientation_residuals(sweep, out_dir / "orientation_residuals.png")

    write_outputs(metrics, sweep, out_dir)
    print(format_summary_table(metrics))

    for k, f in enumerate(freqs):
        tag = f"{f * 1e-6:.0f}MHz"
        plot_maps(
            ours["I"][k],
            theirs["I"][k],
            l_deg,
            m_deg,
            out_dir / f"stokesI_maps_{tag}.png",
            title=f"Stokes I at {tag}",
            cbar_label="beam power",
        )
        plot_cuts(
            ours["I"][k],
            theirs["I"][k],
            l_deg,
            m_deg,
            x0,
            y0,
            out_dir / f"stokesI_cuts_{tag}.png",
            title=f"Stokes I cuts at {tag}",
        )
        plot_radial(
            ours["I"][k],
            theirs["I"][k],
            l_deg,
            m_deg,
            out_dir / f"radial_{tag}.png",
            title=f"Stokes I radial profile at {tag}",
            nbins=args.nbins,
        )
        plot_hhvv(ours, theirs, k, l_deg, m_deg, out_dir / f"hhvv_maps_{tag}.png", title=f"HH / VV power at {tag}")
        plot_crosspol(
            ours["xpol"][k],
            l_deg,
            m_deg,
            out_dir / f"crosspol_{tag}.png",
            title=f"Cross-pol power at {tag} (katbeam models this as zero)",
        )

    sweep_chans = overlap_indices(bds_freqs, model_freqs, stride=max(int(args.fwhm_stride), 1))
    sweep_freqs = bds_freqs[sweep_chans]
    ours_sweep = load_ours(xds, sweep_chans)
    theirs_sweep = eval_katbeam(model_name, l_deg, m_deg, sweep_freqs * 1e-6)
    fwhm_table = {
        "ours_l": np.array([measure_fwhm(l_deg, ours_sweep["I"][k][y0, :]) for k in range(sweep_chans.size)]),
        "ours_m": np.array([measure_fwhm(m_deg, ours_sweep["I"][k][:, x0]) for k in range(sweep_chans.size)]),
        "katbeam_l": np.array([measure_fwhm(l_deg, theirs_sweep["I"][k][y0, :]) for k in range(sweep_chans.size)]),
        "katbeam_m": np.array([measure_fwhm(m_deg, theirs_sweep["I"][k][:, x0]) for k in range(sweep_chans.size)]),
    }
    plot_fwhm_vs_freq(sweep_freqs, fwhm_table, out_dir / "fwhm_vs_freq.png")

    log.info(f"wrote metrics, summary, and plots to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
