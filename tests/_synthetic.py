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
