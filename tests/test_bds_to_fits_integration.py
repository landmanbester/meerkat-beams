"""Integration test: render the real L-band BDS to FITS and check it end to end.

Skips unless ``MBEAMS_BDS_PATH`` points at a legacy MeerKAT L BDS (128 px over
+-4 deg, dx 0.0625 deg).

The geometry is chosen so the output grid falls exactly on stored samples: a
64 px grid over 8 deg has cdelt 0.125 deg, twice the native spacing, so output
pixel ``p`` sits on BDS pixel ``2p`` with no fractional offset. The rendered
planes can therefore be compared against the stored ones directly, which is
what pins the whole path (grid, header, sampling) against real data rather than
a synthetic stand-in.
"""

import os

import numpy as np
import pytest
import xarray
from astropy.io import fits

from meerkat_beams.core.bds_to_fits import bds_to_fits

pytestmark = pytest.mark.integration

BDS_ENV = "MBEAMS_BDS_PATH"
PIXELS = 64
DIAMETER = 8.0


def test_real_l_bds_renders_to_fits(tmp_path):
    path = os.environ.get(BDS_ENV)
    if not path:
        pytest.skip(f"set {BDS_ENV} to run the FITS render on the real L-band BDS")

    stored = xarray.open_zarr(path)["njones"]  # (receptor_i, receptor_j, FREQ, Y, X)
    chan = stored.coords["FREQ"].values
    freqs = [float(chan[100]), float(chan[chan.size - 100])]

    prefix = str(tmp_path / "lband")
    written = bds_to_fits(path, prefix, pixels=PIXELS, diameter=DIAMETER, freqs=freqs)
    assert written == [prefix + "_re.fits", prefix + "_im.fits"]

    with fits.open(prefix + "_re.fits") as h:
        hdr, data = h[0].header, h[0].data
    with fits.open(prefix + "_im.fits") as h:
        imag = h[0].data

    assert data.shape == (2, 2, 2, PIXELS, PIXELS)
    assert hdr["CRPIX1"] == PIXELS / 2 + 1  # eidos's centre rule
    assert hdr["CRPIX2"] == PIXELS / 2 + 1
    assert hdr["CDELT1"] == pytest.approx(DIAMETER / PIXELS)
    assert hdr["CRVAL5"] == pytest.approx(freqs[0])
    assert hdr["CDELT5"] == pytest.approx(freqs[1] - freqs[0])

    # output pixel p is stored pixel 2p, so the rendered planes are the stored
    # ones decimated -- same values, not an approximation of them.
    for out_ch, bds_ch in ((0, 100), (1, chan.size - 100)):
        plane = stored.values[:, :, bds_ch]  # (receptor_i, receptor_j, Y, X)
        np.testing.assert_allclose(data[out_ch], plane.real[:, :, ::2, ::2], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(imag[out_ch], plane.imag[:, :, ::2, ::2], rtol=1e-5, atol=1e-6)

    # on axis the normalised Jones is the identity at every channel
    on_axis = data[:, :, :, PIXELS // 2, PIXELS // 2]
    np.testing.assert_allclose(on_axis[:, 0, 0], 1.0, atol=1e-5)
    np.testing.assert_allclose(on_axis[:, 0, 1], 0.0, atol=1e-5)
    # and well off axis the response has died away
    assert abs(data[0, 0, 0, -1, -1]) < abs(data[0, 0, 0, PIXELS // 2, PIXELS // 2])
