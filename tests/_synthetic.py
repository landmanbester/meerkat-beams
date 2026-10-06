"""Synthetic BDS + image builders for hermetic tests.

Used by tests/test_beam_wizard.py and the core/* unit tests. Single source of
truth so the synthetic-fixture shape can evolve in one place.
"""

from pathlib import Path

import numpy as np
import xarray
from astropy.io import fits

N_XY = 41
I0 = N_XY // 2
DELTA = 0.05
FREQS = np.array([1.0e9, 1.1e9, 1.2e9, 1.3e9])
SIGMA_PIX = 5.0
RA0 = 0.0
DEC0 = -30.0


def gaussian_plane() -> np.ndarray:
    """Separable radial Gaussian, peak 1.0 at (I0, I0), replicated across freq."""
    y, x = np.indices((N_XY, N_XY), dtype=np.float64)
    r2 = (x - I0) ** 2 + (y - I0) ** 2
    plane = np.exp(-0.5 * r2 / SIGMA_PIX**2)
    return np.broadcast_to(plane, (len(FREQS), N_XY, N_XY)).astype(np.float32).copy()


def build_synthetic_bds(path: Path) -> Path:
    degs = (np.arange(N_XY) - I0) * DELTA
    gauss = gaussian_plane()
    zeros = np.zeros_like(gauss)

    njones = np.stack(
        [np.stack([gauss, zeros], axis=0), np.stack([zeros, gauss], axis=0)],
        axis=0,
    ).astype(np.float32)
    jones = njones.copy()

    nstokes_arr = np.zeros((4, 4, len(FREQS), N_XY, N_XY), dtype=np.float32)
    for s in range(4):
        nstokes_arr[s, s] = gauss
    stokes_arr = nstokes_arr.copy()

    # Complex coherency Mueller (mueller/nmueller). Synthetic, not physically
    # derived from `jones`: it exists purely to exercise the complex
    # interpolation paths. Identity on-axis (diagonal=gauss, off-diagonal=0 at
    # centre) with a purely-imaginary cross-hand (U<->V) term that grows with an
    # x-ramp vanishing at the centre pixel, so tests can pin a known nonzero
    # imaginary component off-axis. The (V,U) element is the conjugate (-cross).
    _, x_idx = np.indices((N_XY, N_XY), dtype=np.float64)
    x_ramp = np.broadcast_to(((x_idx - I0) / I0).astype(np.float32), gauss.shape).copy()
    cross = (0.5j * x_ramp * gauss).astype(np.complex64)  # 0 at centre, imaginary
    nmueller_arr = np.zeros((4, 4, len(FREQS), N_XY, N_XY), dtype=np.complex64)
    for s in range(4):
        nmueller_arr[s, s] = gauss
    nmueller_arr[2, 3] = cross  # (U, V)
    nmueller_arr[3, 2] = -cross  # (V, U) = conj(cross), cross is imaginary
    mueller_arr = nmueller_arr.copy()

    fits_header = {
        "SIMPLE": "T",
        "NAXIS1": N_XY,
        "NAXIS2": N_XY,
        "NAXIS3": len(FREQS),
        "CRPIX1": I0 + 1,
        "CRPIX2": I0 + 1,
        "CRPIX3": 1,
        "CRVAL1": 0,
        "CRVAL2": 0,
        "CRVAL3": float(FREQS[0]),
        "CDELT1": DELTA,
        "CDELT2": DELTA,
        "CDELT3": float(FREQS[1] - FREQS[0]),
        "CTYPE1": "X",
        "CTYPE2": "Y",
        "CTYPE3": "FREQ",
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        "CUNIT3": "Hz",
    }

    jcoords = dict(receptor_i=[0, 1], receptor_j=[0, 1], X=degs, Y=degs, FREQ=FREQS)
    scoords = dict(stokes_i=list("IQUV"), stokes_j=list("IQUV"), X=degs, Y=degs, FREQ=FREQS)

    xds = xarray.Dataset(
        {
            "jones": xarray.DataArray(jones, dims=("receptor_i", "receptor_j", "FREQ", "Y", "X"), coords=jcoords),
            "njones": xarray.DataArray(njones, dims=("receptor_i", "receptor_j", "FREQ", "Y", "X"), coords=jcoords),
            "stokes": xarray.DataArray(stokes_arr, dims=("stokes_i", "stokes_j", "FREQ", "Y", "X"), coords=scoords),
            "nstokes": xarray.DataArray(nstokes_arr, dims=("stokes_i", "stokes_j", "FREQ", "Y", "X"), coords=scoords),
            "mueller": xarray.DataArray(mueller_arr, dims=("stokes_i", "stokes_j", "FREQ", "Y", "X"), coords=scoords),
            "nmueller": xarray.DataArray(nmueller_arr, dims=("stokes_i", "stokes_j", "FREQ", "Y", "X"), coords=scoords),
        }
    )
    xds.attrs["fits_header"] = fits_header
    xds.attrs.update(x0=I0, y0=I0, dx=DELTA, dy=DELTA, freqs=FREQS)
    xds.to_zarr(str(path), mode="w")
    return path


def build_synthetic_image(path: Path) -> Path:
    """Minimal 2-axis FITS image with SIN-projection WCS centred at (RA0, DEC0)."""
    nx = ny = 64
    data = np.zeros((ny, nx), dtype=np.float32)
    hdr = fits.Header()
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = nx
    hdr["NAXIS2"] = ny
    hdr["CRPIX1"] = nx // 2 + 1
    hdr["CRPIX2"] = ny // 2 + 1
    hdr["CRVAL1"] = RA0
    hdr["CRVAL2"] = DEC0
    hdr["CDELT1"] = -0.01
    hdr["CDELT2"] = 0.01
    hdr["CTYPE1"] = "RA---SIN"
    hdr["CTYPE2"] = "DEC--SIN"
    hdr["CUNIT1"] = "deg"
    hdr["CUNIT2"] = "deg"
    fits.PrimaryHDU(data=data, header=hdr).writeto(str(path), overwrite=True)
    return path


def jones_beam_cube(scale=1.0, leak=0.0, phase=0.0, n_xy=N_XY, n_freq=None, sigma=None):
    """MdV-ordered [HH, HV, VH, VV] complex beam cube, shape (4, NFREQ, NY, NX).

    A Gaussian co-pol envelope of width ``sigma`` pixels (default ``SIGMA_PIX``)
    scaled by ``scale``, a radial leakage term of peak amplitude ``leak`` in the
    cross-hands, and a phase ramp of ``phase`` radians per half-width across x.
    All three are zero/unity at the centre pixel, so the on-axis Jones is
    ``scale * identity`` and normalising by its inverse gives exactly the
    identity on axis.

    ``scale`` is divided out by normalisation, so two stores differing only in
    ``scale`` have near-identical ``nstokes``. Use ``sigma`` when a test needs
    two telescopes whose *normalised* beams actually differ -- physically the
    right knob too, since a larger dish gives a narrower beam.
    """
    n_freq = len(FREQS) if n_freq is None else n_freq
    sigma = SIGMA_PIX if sigma is None else sigma
    i0 = n_xy // 2
    y, x = np.indices((n_xy, n_xy), dtype=np.float64)
    r2 = (x - i0) ** 2 + (y - i0) ** 2
    envelope = np.exp(-0.5 * r2 / sigma**2)
    ramp = (x - i0) / max(i0, 1)
    co = scale * envelope * np.exp(1j * phase * ramp)
    # HV and VH must differ, or a swap of the Jones receptor axes is a no-op
    # and no test can detect a transposed or p/q-reversed group assembly.
    cross_hv = leak * envelope * (r2 / (i0**2))  # zero at centre
    cross_vh = 0.4j * cross_hv + 0.6 * leak * envelope * (ramp**2)
    plane = np.stack([co, cross_hv, cross_vh, co], axis=0)  # HH, HV, VH, VV
    return np.broadcast_to(plane[:, None], (4, n_freq, n_xy, n_xy)).astype(np.complex64).copy()


def build_mean_beam_zarr(
    path,
    *,
    scale=1.0,
    leak=0.0,
    phase=0.0,
    telescope=None,
    antenna=None,
    band="L",
    freqs=None,
    degs=None,
    sigma=None,
):
    """MdV-shaped mean-beam zarr, i.e. exactly what mdv_beams_to_bds consumes."""
    freqs = FREQS if freqs is None else np.asarray(freqs, dtype=float)
    degs = (np.arange(N_XY) - I0) * DELTA if degs is None else np.asarray(degs, dtype=float)
    beam = jones_beam_cube(scale=scale, leak=leak, phase=phase, n_xy=len(degs), n_freq=len(freqs), sigma=sigma)
    ds = xarray.Dataset(
        {"BEAM": xarray.DataArray(beam, dims=("corr", "chan", "l_beam", "m_beam"))},
        coords={"corr": ["XX", "XY", "YX", "YY"], "chan": freqs, "l_beam": degs, "m_beam": degs},
    )
    if telescope is not None:
        ds.attrs["telescope"] = telescope
    if antenna is not None:
        ds.attrs["antenna"] = antenna
    ds.attrs["band"] = band
    ds.to_zarr(str(path), mode="w")
    return path


def build_telescope_bds(zarr_path, bds_path, **kwargs):
    """Write a mean-beam zarr then convert it with the real mdv_beams_to_bds."""
    from meerkat_beams.core.mdv_beams_to_bds import mdv_beams_to_bds

    build_mean_beam_zarr(zarr_path, **kwargs)
    mdv_beams_to_bds(str(zarr_path), str(bds_path))
    return bds_path
