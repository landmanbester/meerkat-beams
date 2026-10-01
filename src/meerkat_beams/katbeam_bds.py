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

from meerkat_beams.utils import log

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


# ---------------------------------------------------------------------------
# Beam synthesis
# ---------------------------------------------------------------------------

# katbeam's taper is cos(pi*rr)/(1 - 4*rr**2) with rr = r*1.1889647809329453,
# which is 0/0 at rr = 0.5. The L'Hopital limit there is pi/4: differentiating
# gives -pi*sin(pi*rr) / -8*rr -> -pi / -4. A grid point landing on this radius
# returns NaN from katbeam, and a NaN reaching spline_filter smears across the
# entire slab, so synthesis replaces it.
SINGULAR_TAPER_RADIUS = 0.4205339031217265  # 0.5 / 1.1889647809329453
SINGULAR_TAPER_LIMIT = float(np.pi / 4)

# Frequency chunk size for the dask arrays. The analytic evaluation is cheap and
# vectorised over the full spatial plane, so the arrays are chunked in FREQ
# only -- spatial chunking would multiply the number of JimBeam calls for no
# benefit. 256 matches the MdV BDS frequency chunking.
FREQ_CHUNK = 256


def _sanitize(arr: np.ndarray, label: str) -> np.ndarray:
    """Replace katbeam's 0/0 singularity with its analytic limit."""
    bad = ~np.isfinite(arr)
    n_bad = int(bad.sum())
    if n_bad:
        log.warning(
            f"replacing {n_bad} non-finite {label} sample(s) at the cosine-taper "
            f"singularity (r = {SINGULAR_TAPER_RADIUS:.6f}) with the analytic "
            f"limit {SINGULAR_TAPER_LIMIT:.6f}"
        )
        arr = np.where(bad, SINGULAR_TAPER_LIMIT, arr)
    return arr


def _eval_block(model_name: str, ll: np.ndarray, mm: np.ndarray, freqs_mhz: np.ndarray):
    """Evaluate one frequency block, returning (njones, nmueller, nstokes).

    Shapes are (2, 2, NF, NY, NX), (4, 4, NF, NY, NX), (4, 4, NF, NY, NX).
    """
    from meerkat_beams.utils import jones_to_mueller, mueller_to_stokes

    jb = require_model(model_name)
    nf = len(freqs_mhz)
    ny, nx = ll.shape

    # Build Jones in (FREQ, Y, X, ROW, COL) order, which is what
    # jones_to_mueller expects.
    jones = np.zeros((nf, ny, nx, 2, 2), dtype=np.complex128)
    for k, f in enumerate(freqs_mhz):
        jones[k, :, :, 0, 0] = _sanitize(np.asarray(jb.HH(ll, mm, float(f)), dtype=float), "HH")
        jones[k, :, :, 1, 1] = _sanitize(np.asarray(jb.VV(ll, mm, float(f)), dtype=float), "VV")

    mueller = jones_to_mueller(jones)
    stokes = mueller_to_stokes(mueller)

    # Transpose back to the BDS (i, j, FREQ, Y, X) layout.
    njones = jones.transpose((3, 4, 0, 1, 2)).astype(np.complex64)
    nmueller = mueller.transpose((3, 4, 0, 1, 2)).astype(np.complex64)
    nstokes = stokes.transpose((3, 4, 0, 1, 2)).real.astype(np.float32)
    return njones, nmueller, nstokes


def _resolve_katbeam_freqs(model_name: str, freq: Optional[np.ndarray], num_freq: Optional[int]) -> np.ndarray:
    """Resolve the frequency axis in Hz and validate it against the model table.

    katbeam interpolates its squint/FWHM table with np.interp, which *clamps*
    outside the table rather than raising, so an out-of-band request would
    silently return the nearest in-band beam. Refuse instead.
    """
    if freq is not None and num_freq is not None:
        raise ValueError("freq and num_freq are mutually exclusive")

    table_hz = katbeam_freq_table(model_name) * 1e6
    fmin, fmax = float(table_hz.min()), float(table_hz.max())

    if freq is None:
        if num_freq is not None:
            freqs = np.linspace(fmin, fmax, int(num_freq))
        else:
            freqs = table_hz
    else:
        freqs = np.atleast_1d(np.asarray(freq, dtype=float))

    if freqs.size and (freqs.min() < fmin or freqs.max() > fmax):
        raise ValueError(
            f"requested frequencies [{freqs.min() * 1e-6:.3f}, {freqs.max() * 1e-6:.3f}] MHz "
            f"fall outside the katbeam model {model_name!r} range "
            f"[{fmin * 1e-6:.3f}, {fmax * 1e-6:.3f}] MHz. katbeam clamps silently "
            "outside its table, which would return the nearest in-band beam."
        )
    return freqs


def synthesize_katbeam_bds(
    band: str,
    *,
    npix: Optional[int] = None,
    fov_deg: Optional[float] = None,
    freq: Optional[np.ndarray] = None,
    num_freq: Optional[int] = None,
):
    """Build an in-memory, BDS-shaped Dataset from katbeam's analytic beams.

    Args:
        band: MdV band code -- 'U', 'L', or 'S0'..'S4'.
        npix: pixels per spatial axis; must be even. Defaults to the band's MdV
            value (see BAND_GEOMETRY).
        fov_deg: half-width of the grid in degrees, so it spans -fov_deg to
            +fov_deg - dx. Defaults to the band's MdV value.
        freq: explicit frequencies in Hz. Mutually exclusive with num_freq.
        num_freq: number of frequencies linearly spaced across the katbeam
            model's range. Mutually exclusive with freq.

    Returns:
        A dask-backed xarray.Dataset with njones/nstokes/nmueller and the BDS
        attrs. The unnormalised jones/stokes/mueller variables are absent:
        katbeam beams are on-axis normalised by construction.
    """
    import dask
    import dask.array as da
    import xarray

    if band not in KATBEAM_MODEL_FOR_BAND:
        raise ValueError(f"unknown band {band!r}; expected one of {sorted(KATBEAM_MODEL_FOR_BAND)}")
    model_name = KATBEAM_MODEL_FOR_BAND[band]

    npix, fov_deg = resolve_geometry(band, npix, fov_deg)
    if npix % 2 != 0:
        raise ValueError(
            f"npix must be even, got {npix}: the grid centre x0 = npix//2 lands on "
            "exactly 0 degrees only for an even npix, and an odd npix would offset "
            "the beam centre by half a pixel."
        )

    freqs = _resolve_katbeam_freqs(model_name, freq, num_freq)

    delta = 2.0 * fov_deg / npix
    degs = -fov_deg + np.arange(npix) * delta
    i0 = npix // 2
    ll, mm = np.meshgrid(degs, degs)  # (NY, NX): axis 0 is Y, axis 1 is X

    log.info(
        f"synthesizing katbeam BDS for band {band} (model {model_name}): "
        f"{npix}x{npix} pixels, {delta:.5f} deg/pixel, fov +/-{fov_deg} deg, "
        f"{len(freqs)} frequencies {freqs[0] * 1e-6:.1f}-{freqs[-1] * 1e-6:.1f} MHz"
    )

    # One delayed call per frequency chunk; nout=3 makes the three outputs
    # separate Delayed objects that share a single evaluation per compute.
    jones_blocks, mueller_blocks, stokes_blocks = [], [], []
    for start in range(0, len(freqs), FREQ_CHUNK):
        fchunk = freqs[start : start + FREQ_CHUNK] / 1e6  # katbeam wants MHz
        nf = len(fchunk)
        blk = dask.delayed(_eval_block, nout=3)(model_name, ll, mm, fchunk)
        jones_blocks.append(da.from_delayed(blk[0], shape=(2, 2, nf, npix, npix), dtype=np.complex64))
        mueller_blocks.append(da.from_delayed(blk[1], shape=(4, 4, nf, npix, npix), dtype=np.complex64))
        stokes_blocks.append(da.from_delayed(blk[2], shape=(4, 4, nf, npix, npix), dtype=np.float32))

    njones = da.concatenate(jones_blocks, axis=2)
    nmueller = da.concatenate(mueller_blocks, axis=2)
    nstokes = da.concatenate(stokes_blocks, axis=2)

    jcoords = dict(receptor_i=[0, 1], receptor_j=[0, 1], X=degs, Y=degs, FREQ=freqs)
    scoords = dict(stokes_i=list("IQUV"), stokes_j=list("IQUV"), X=degs, Y=degs, FREQ=freqs)

    xds = xarray.Dataset(
        dict(
            njones=xarray.DataArray(njones, dims=("receptor_i", "receptor_j", "FREQ", "Y", "X"), coords=jcoords),
            nstokes=xarray.DataArray(nstokes, dims=("stokes_i", "stokes_j", "FREQ", "Y", "X"), coords=scoords),
            nmueller=xarray.DataArray(nmueller, dims=("stokes_i", "stokes_j", "FREQ", "Y", "X"), coords=scoords),
        )
    )

    # Synthesized FITS header, matching the keys mdv_beams_to_bds writes.
    hdr = {
        "CDELT1": delta,
        "CDELT2": delta,
        "CRPIX1": i0 + 1,  # FITS is 1-based
        "CRPIX2": i0 + 1,
        "CRVAL1": 0,
        "CRVAL2": 0,
        "CTYPE1": "X",
        "CTYPE2": "Y",
        "CRPIX3": 1,
        "CRVAL3": float(freqs[0]),
        "CDELT3": float(freqs[1] - freqs[0]) if len(freqs) > 1 else 0.0,
        "CTYPE3": "FREQ",
        "CUNIT3": "Hz",
    }
    xds.attrs["fits_header"] = hdr
    xds.attrs.update(
        x0=i0,
        y0=i0,
        dx=delta,
        dy=delta,
        freqs=list(freqs),
        npix=npix,
        fov_deg=fov_deg,
        beam_model="katbeam",
        katbeam_model=model_name,
        band=band,
    )
    return xds
