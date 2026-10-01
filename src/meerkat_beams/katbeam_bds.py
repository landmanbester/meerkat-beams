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
# whose denominator vanishes at rr = 0.5. The L'Hopital limit there is pi/4:
# differentiating gives -pi*sin(pi*rr) / -8*rr -> -pi / -4.
#
# A grid point landing on this radius returns a non-finite value -- in practice
# inf, NOT NaN, because cos(pi*rr) is ~6.1e-17 rather than exactly 0 there, so it
# is a divide-by-zero. _sanitize tests ~np.isfinite and catches either. Something
# non-finite reaching spline_filter smears across the entire slab, so synthesis
# replaces it.
SINGULAR_TAPER_RADIUS = 0.4205339031217265  # 0.5 / 1.1889647809329453
SINGULAR_TAPER_LIMIT = float(np.pi / 4)

# Frequency chunking for the dask arrays. The analytic evaluation is cheap and
# vectorised over the full spatial plane, so the arrays are chunked in FREQ
# only -- spatial chunking would multiply the number of JimBeam calls for no
# benefit.
#
# The chunk is sized by an *element budget*, not a fixed frequency count.
# _eval_block holds complex128 jones (2x2), mueller (4x4) and stokes (4x4) for
# a whole chunk before the astype downcasts, so its transient scales as
# nfreq * npix**2. A fixed 256-frequency chunk peaked at 3.3 GiB at npix=128,
# which would make this "lighter alternative" heavier than the MdV path it
# replaces. 300k elements holds the peak near 250 MiB at any npix.
FREQ_CHUNK_ELEMENTS = 300_000


def _freq_chunk_size(npix: int) -> int:
    """Frequencies per dask block for an ``npix`` x ``npix`` grid.

    At least 1, so a grid too large for the budget still makes progress one
    frequency at a time rather than failing.
    """
    return max(1, FREQ_CHUNK_ELEMENTS // (npix * npix))


def _sanitize(arr: np.ndarray, label: str) -> np.ndarray:
    """Replace katbeam's taper singularity with its analytic limit.

    The mask is unconditional on radius rather than gated near
    SINGULAR_TAPER_RADIUS. That is safe only because the singularity is the sole
    source of non-finite values here: the taper is finite at every other radius,
    and every model's FWHM table is strictly positive (min 0.421 deg), so the
    normalised radius is always well defined. If either ceases to hold, gate this
    on radius -- pi/4 is a plausible-looking beam value, so a masked unrelated
    non-finite would be hard to spot.
    """
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

    Outputs are normalised the same way an MdV BDS normalises its ``n*``
    variables: pre-multiplied by the inverse of the centre-pixel Jones, so
    ``njones`` on axis is exactly the identity. katbeam's raw HH/VV are only
    *approximately* unity at (0, 0) -- each beam's peak is offset by its squint,
    giving 0.9950 at 1712 MHz -- so without this the ``n`` prefix would mean
    something different here than it does for MdV, and a caller correcting data
    with ``njones`` would carry an on-axis error of up to 0.5%.
    """
    from meerkat_beams.utils import jones_to_mueller, mueller_to_stokes

    jb = require_model(model_name)
    nf = len(freqs_mhz)
    ny, nx = ll.shape

    # Build Jones in (FREQ, Y, X, ROW, COL) order, which is what
    # jones_to_mueller expects. complex64 throughout: the 4x4 Mueller array
    # dominates this block's transient, and the outputs are cast to
    # complex64/float32 anyway, so complex128 here would double peak memory for
    # precision that is discarded. Measured difference in the Stokes result is
    # ~2e-8, against a float32 output.
    jones = np.zeros((nf, ny, nx, 2, 2), dtype=np.complex64)
    for k, f in enumerate(freqs_mhz):
        hh = _sanitize(np.asarray(jb.HH(ll, mm, float(f)), dtype=float), "HH")
        vv = _sanitize(np.asarray(jb.VV(ll, mm, float(f)), dtype=float), "VV")
        # Normalise by the on-axis value, matching mdv_beams_to_bds. The Jones
        # matrix is diagonal, so inv(J(centre)) @ J reduces to dividing each
        # receptor by its own centre sample -- no matrix inverse needed.
        hh0 = _sanitize(np.asarray(jb.HH(0.0, 0.0, float(f)), dtype=float), "HH")
        vv0 = _sanitize(np.asarray(jb.VV(0.0, 0.0, float(f)), dtype=float), "VV")
        if hh0 == 0 or vv0 == 0:  # pragma: no cover - the taper is ~1 on axis
            raise ValueError(
                f"katbeam model {model_name!r} has a zero on-axis response at {f} MHz, "
                "so the beam cannot be normalised."
            )
        jones[k, :, :, 0, 0] = hh / hh0
        jones[k, :, :, 1, 1] = vv / vv0

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

    # Validate the axis before it becomes a FREQ coordinate: BeamWizard builds
    # interp1d index<->freq mappings from it, which silently misbehave on a
    # non-finite or non-monotonic axis, and an empty axis used to die in the log
    # line below with a bare IndexError.
    if freqs.size == 0:
        raise ValueError("at least one frequency is required, got an empty array")
    if not np.all(np.isfinite(freqs)):
        raise ValueError(f"frequencies must all be finite, got {freqs}")
    if freqs.size > 1 and not np.all(np.diff(freqs) > 0):
        raise ValueError(
            "frequencies must be strictly increasing (no duplicates): index<->frequency "
            f"interpolation is built from this axis. Got {freqs}"
        )

    if freqs.min() < fmin or freqs.max() > fmax:
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
    if npix < 2:
        raise ValueError(f"npix must be at least 2, got {npix}")
    if npix % 2 != 0:
        raise ValueError(
            f"npix must be even, got {npix}: the grid centre x0 = npix//2 lands on "
            "exactly 0 degrees only for an even npix, and an odd npix would offset "
            "the beam centre by half a pixel."
        )
    if fov_deg <= 0:
        # A negative half-width silently MIRRORS the beam: dx comes out negative,
        # the grid runs +fov -> -fov, and x0 still lands on 0.0, so nothing
        # downstream complains while xp = l/dx + x0 flips east for west. In a repo
        # whose axis sign conventions are still unsettled (see
        # docs/wiki/beam-orientation.md) that is the worst available failure mode,
        # so refuse rather than produce a wrong beam. fov_deg == 0 gives dx == 0.
        raise ValueError(
            f"fov_deg must be positive, got {fov_deg}: it is a half-width in degrees. "
            "A negative value silently mirrors the beam east-west, and zero gives a "
            "zero pixel scale."
        )
    if num_freq is not None and num_freq < 1:
        raise ValueError(f"num_freq must be at least 1, got {num_freq}")

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
    freq_chunk = _freq_chunk_size(npix)
    for start in range(0, len(freqs), freq_chunk):
        fchunk = freqs[start : start + freq_chunk] / 1e6  # katbeam wants MHz
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
        "CTYPE3": "FREQ",
        "CUNIT3": "Hz",
    }
    # CDELT3 only when the axis really is uniform. The default axis is the katbeam
    # table, which is not (... 1600, 1650, 1670, 1712 MHz), and a single increment
    # would make a WCS reader assign the wrong frequency to every later plane. The
    # exact axis is always in attrs["freqs"] and the FREQ coordinate.
    if len(freqs) > 1:
        diffs = np.diff(freqs)
        if np.allclose(diffs, diffs[0], rtol=1e-9, atol=0.0):
            hdr["CDELT3"] = float(diffs[0])
        else:
            log.info(
                "frequency axis is non-uniform, so CDELT3 is omitted from the synthesized "
                "FITS header; use attrs['freqs'] or the FREQ coordinate for the exact axis"
            )
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
