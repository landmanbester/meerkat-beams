"""Hermetic unit tests for core/bds_to_fits.py.

The geometry assertions mirror eidos's own ``tests/test_fits_centre.py``: the
sub-pixel beam peak must sit on the header's ``CRPIX1 - 1``, and the header
cards must describe the cube the way eidos's ``write_fits``/``write_fits_cube``
do. eidos itself is deliberately not a dependency -- parity is pinned against
the documented convention instead (docs/wiki/data-model.md).
"""

import logging
from pathlib import Path

import numpy as np
import pytest
import xarray
from astropy.io import fits
from astropy.time import Time

from meerkat_beams.core.bds_to_fits import bds_to_fits
from tests._synthetic import DELTA, FREQS, I0, N_XY, build_synthetic_bds

pytestmark = pytest.mark.unit

# A 2 deg output grid on the synthetic BDS's own +-1 deg extent: cdelt 0.0625 deg
# for 32 px, so every target pixel lands inside the beam model.
PIXELS = 32
DIAMETER = 2.0
# (N_XY - 1) * DELTA is exactly the coordinate span of the synthetic grid.
NATIVE_SPAN = (N_XY - 1) * DELTA


@pytest.fixture(scope="module")
def synthetic_bds(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bds_fits")
    return build_synthetic_bds(tmp / "synth.bds.zarr")


def read(path):
    with fits.open(str(path)) as h:
        return h[0].header, h[0].data


def peak_x(plane):
    """Sub-pixel x (last-axis) position of max |plane|, by a 3-point parabola.

    Lifted from eidos's test_fits_centre.py so both suites measure the same thing.
    """
    a = np.abs(plane)
    j, i = np.unravel_index(np.argmax(a), a.shape)
    left, mid, right = a[j, i - 1], a[j, i], a[j, i + 1]
    return i + 0.5 * (left - right) / (left - 2 * mid + right)


def test_hypercube_header_matches_eidos_geometry(synthetic_bds, tmp_path):
    prefix = str(tmp_path / "beam")
    written = bds_to_fits(str(synthetic_bds), prefix, pixels=PIXELS, diameter=DIAMETER)
    assert written == [prefix + "_re.fits", prefix + "_im.fits"]

    hdr, data = read(prefix + "_re.fits")
    # numpy (FREQ, p0, p1, Y, X) -> FITS axes 5, 4, 3, 2, 1.
    assert data.shape == (len(FREQS), 2, 2, PIXELS, PIXELS)
    assert [hdr[f"NAXIS{i}"] for i in range(1, 6)] == [PIXELS, PIXELS, 2, 2, len(FREQS)]
    # px/py are the correction: eidos writes py/px on the 5-axis hypercube.
    assert (hdr["CTYPE1"], hdr["CTYPE2"], hdr["CTYPE3"], hdr["CTYPE4"], hdr["CTYPE5"]) == (
        "px",
        "py",
        "V",
        "H",
        "FREQ",
    )
    # eidos's centre rule: 0-based n/2, 1-based n/2 + 1.
    assert (hdr["CRPIX1"], hdr["CRPIX2"]) == (PIXELS / 2 + 1, PIXELS / 2 + 1)
    assert (hdr["CRPIX3"], hdr["CRPIX4"], hdr["CRPIX5"]) == (1.0, 1.0, 1.0)
    assert (hdr["CRVAL1"], hdr["CRVAL2"]) == (0.0, 0.0)
    assert hdr["CDELT1"] == pytest.approx(DIAMETER / PIXELS)
    assert hdr["CDELT2"] == pytest.approx(DIAMETER / PIXELS)
    assert (hdr["CUNIT1"], hdr["CUNIT2"]) == ("deg", "deg")
    assert hdr["CRVAL5"] == pytest.approx(FREQS[0])
    assert hdr["CDELT5"] == pytest.approx(FREQS[1] - FREQS[0])
    assert hdr["CUNIT5"] == "Hz"
    # bookkeeping
    assert hdr["TELESCOP"] == "MeerKAT"
    assert hdr["BEAMTYPE"] == "njones"
    assert hdr["ELEMNT"] == "XX,XY,YX,YY"
    assert hdr["PART"] == "RE"
    Time(hdr["DATE"])  # ISO-8601, unlike eidos's time.ctime()


@pytest.mark.parametrize("pixels", [32, 33], ids=["even", "odd"])
def test_beam_peak_sits_at_crpix(synthetic_bds, tmp_path, pixels):
    prefix = str(tmp_path / f"beam{pixels}")
    bds_to_fits(str(synthetic_bds), prefix, pixels=pixels, diameter=DIAMETER)
    hdr, data = read(prefix + "_re.fits")
    _, imag = read(prefix + "_im.fits")
    data = data + 1j * imag
    plane = data.reshape(-1, pixels, pixels)[0]  # first channel, XX element
    offset = peak_x(plane) - (hdr["CRPIX1"] - 1)
    assert abs(offset) < 0.05, f"beam peak is {offset:+.3f} px from CRPIX1"


def test_output_eight_writes_eight_files(synthetic_bds, tmp_path):
    prefix = str(tmp_path / "beam8")
    written = bds_to_fits(str(synthetic_bds), prefix, output_eight=True, pixels=PIXELS, diameter=DIAMETER)
    expected = [f"{prefix}_{el}_{part}.fits" for el in ("xx", "xy", "yx", "yy") for part in ("re", "im")]
    assert written == expected

    hdr, data = read(expected[0])
    assert data.shape == (len(FREQS), PIXELS, PIXELS)
    assert (hdr["CTYPE1"], hdr["CTYPE2"], hdr["CTYPE3"]) == ("px", "py", "FREQ")
    assert hdr["CRPIX1"] == pytest.approx(PIXELS / 2 + 1)
    assert hdr["CDELT1"] == pytest.approx(DIAMETER / PIXELS)
    assert hdr["CUNIT3"] == "Hz"
    assert hdr["CRVAL3"] == pytest.approx(FREQS[0])
    assert hdr["CDELT3"] == pytest.approx(FREQS[1] - FREQS[0])
    assert hdr["ELEMNT"] == "XX"
    assert hdr["PART"] == "RE"

    # the element files are exactly the planes of the hypercube
    cube_prefix = str(tmp_path / "beammulti")
    bds_to_fits(str(synthetic_bds), cube_prefix, pixels=PIXELS, diameter=DIAMETER)
    _, hyper_re = read(cube_prefix + "_re.fits")
    _, hyper_im = read(cube_prefix + "_im.fits")
    for idx, element in enumerate(("xx", "xy", "yx", "yy")):
        i, j = divmod(idx, 2)
        _, re_data = read(f"{prefix}_{element}_re.fits")
        _, im_data = read(f"{prefix}_{element}_im.fits")
        np.testing.assert_array_equal(re_data, hyper_re[:, i, j])
        np.testing.assert_array_equal(im_data, hyper_im[:, i, j])


def test_native_grid_copies_bds_and_keeps_its_header(synthetic_bds, tmp_path):
    prefix = str(tmp_path / "native")
    bds_to_fits(str(synthetic_bds), prefix)
    hdr, data = read(prefix + "_re.fits")
    _, imag = read(prefix + "_im.fits")

    bds = xarray.open_zarr(str(synthetic_bds))
    expected = bds["njones"].values  # (receptor_i, receptor_j, FREQ, Y, X)
    assert data.shape == (len(FREQS), 2, 2, N_XY, N_XY)
    # native geometry takes the copy path: bit-identical to the stored planes
    np.testing.assert_array_equal(data, expected.transpose(2, 0, 1, 3, 4))
    np.testing.assert_array_equal(imag, np.zeros_like(imag))
    # header agrees with the BDS's own synthesized header
    assert hdr["CRPIX1"] == bds.attrs["fits_header"]["CRPIX1"] == I0 + 1
    assert hdr["CRPIX2"] == bds.attrs["fits_header"]["CRPIX2"] == I0 + 1
    assert hdr["CDELT1"] == pytest.approx(DELTA)
    assert hdr["CRVAL5"] == pytest.approx(FREQS[0])


def test_stokes_hypercube_writes_real_part_only(synthetic_bds, tmp_path):
    prefix = str(tmp_path / "stokes")
    written = bds_to_fits(str(synthetic_bds), prefix, beam_type="nstokes")
    assert written == [prefix + "_re.fits"]
    assert not Path(prefix + "_im.fits").exists()

    hdr, data = read(prefix + "_re.fits")
    assert data.shape == (len(FREQS), 1, 1, N_XY, N_XY)
    assert (hdr["CTYPE3"], hdr["CTYPE4"]) == ("STOKES", "STOKES")
    assert (hdr["CRVAL3"], hdr["CRVAL4"]) == (1.0, 1.0)  # I is Stokes code 1
    assert hdr["BEAMTYPE"] == "nstokes"
    assert hdr["ELEMNT"] == "II"

    bds = xarray.open_zarr(str(synthetic_bds))
    expected = bds["nstokes"].sel(stokes_i="I", stokes_j="I").values
    np.testing.assert_array_equal(data[:, 0, 0], expected)


def test_mueller_element_selects_and_labels_stokes_axes(synthetic_bds, tmp_path):
    prefix = str(tmp_path / "mueller")
    written = bds_to_fits(str(synthetic_bds), prefix, beam_type="nmueller", elements=["UV"])
    assert written == [prefix + "_re.fits", prefix + "_im.fits"]

    hdr, data = read(prefix + "_re.fits")
    # row of the element goes to axis 4, column to axis 3: U=3, V=4
    assert (hdr["CRVAL4"], hdr["CRVAL3"]) == (3.0, 4.0)
    assert hdr["ELEMNT"] == "UV"

    bds = xarray.open_zarr(str(synthetic_bds))
    expected = bds["nmueller"].sel(stokes_i="U", stokes_j="V").values
    np.testing.assert_array_equal(data[:, 0, 0], expected.real)
    _, imag = read(prefix + "_im.fits")
    np.testing.assert_array_equal(imag[:, 0, 0], expected.imag)
    assert np.any(expected.imag != 0)  # the synthetic cross-hand is complex


def test_requested_grid_agrees_with_native_where_pixels_share(synthetic_bds, tmp_path):
    native_prefix = str(tmp_path / "native")
    resamp_prefix = str(tmp_path / "resamp")
    bds_to_fits(str(synthetic_bds), native_prefix)
    # 40 px over the 2 deg span keeps cdelt at the native 0.05 deg and the centre
    # at index 20, so target pixel p is native pixel p for every p < 40.
    bds_to_fits(str(synthetic_bds), resamp_prefix, pixels=N_XY - 1, diameter=NATIVE_SPAN)

    _, native_re = read(native_prefix + "_re.fits")
    _, native_im = read(native_prefix + "_im.fits")
    resamp_hdr, resamp_re = read(resamp_prefix + "_re.fits")
    _, resamp_im = read(resamp_prefix + "_im.fits")

    assert resamp_re.shape == (len(FREQS), 2, 2, N_XY - 1, N_XY - 1)
    assert resamp_hdr["CDELT1"] == pytest.approx(DELTA)
    np.testing.assert_allclose(resamp_re, native_re[..., : N_XY - 1, : N_XY - 1], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(resamp_im, native_im[..., : N_XY - 1, : N_XY - 1], rtol=1e-6, atol=1e-6)


def test_output_beyond_bds_fov_is_zero_and_warns(synthetic_bds, tmp_path, caplog):
    prefix = str(tmp_path / "wide")
    with caplog.at_level(logging.WARNING, logger="meerkat_beams"):
        bds_to_fits(str(synthetic_bds), prefix, pixels=64, diameter=8.0)
    assert any("fall outside the beam model" in r.message for r in caplog.records)

    _, data = read(prefix + "_re.fits")
    assert data[0, 0, 0, 0, 0] == 0.0  # corner, 4 deg off, is outside the model
    assert data[0, 0, 0, 32, 32] == pytest.approx(1.0)  # on axis the beam is unity


def test_num_freq_selects_a_uniform_channel_axis(synthetic_bds, tmp_path):
    prefix = str(tmp_path / "nfreq")
    bds_to_fits(str(synthetic_bds), prefix, num_freq=3)
    hdr, data = read(prefix + "_re.fits")
    expected = np.linspace(FREQS[0], FREQS[-1], 3)
    assert data.shape[0] == 3
    assert hdr["CRVAL5"] == pytest.approx(expected[0])
    # a linear axis must reproduce every requested channel
    for k, want in enumerate(expected):
        got = hdr["CRVAL5"] + k * hdr["CDELT5"]
        assert got == pytest.approx(want, rel=1e-12)


def test_non_uniform_freqs_warn_and_use_mean_spacing(synthetic_bds, tmp_path, caplog):
    prefix = str(tmp_path / "gap")
    freqs = [1.0e9, 1.05e9, 1.3e9]
    with caplog.at_level(logging.WARNING, logger="meerkat_beams"):
        bds_to_fits(str(synthetic_bds), prefix, freqs=freqs)
    assert any("not uniformly spaced" in r.message for r in caplog.records)

    hdr, data = read(prefix + "_re.fits")
    assert data.shape[0] == 3
    assert hdr["CRVAL5"] == pytest.approx(freqs[0])
    assert hdr["CDELT5"] == pytest.approx((freqs[-1] - freqs[0]) / 2)


def test_single_frequency_keeps_eidos_cdelta(synthetic_bds, tmp_path):
    prefix = str(tmp_path / "one")
    bds_to_fits(str(synthetic_bds), prefix, freqs=[1.2e9])
    hdr, data = read(prefix + "_re.fits")
    assert data.shape[0] == 1
    assert hdr["CRVAL5"] == pytest.approx(1.2e9)
    assert hdr["CDELT5"] == pytest.approx(1e6)  # eidos's fallback for one channel


def test_out_of_band_freq_raises_unless_scaling(synthetic_bds, tmp_path):
    with pytest.raises(ValueError, match="outside the BDS frequency range"):
        bds_to_fits(str(synthetic_bds), str(tmp_path / "far"), freqs=[2.0e9])

    prefix = str(tmp_path / "edge")
    # one BDS channel past the top edge (1.3e9 + 1e8 margin) is a lambda/D job
    bds_to_fits(str(synthetic_bds), prefix, freqs=[1.35e9], freq_extrapolation="scale")
    hdr, data = read(prefix + "_re.fits")
    assert data.shape[0] == 1
    assert hdr["CRVAL5"] == pytest.approx(1.35e9)


def test_freqs_and_num_freq_are_mutually_exclusive(synthetic_bds, tmp_path):
    with pytest.raises(ValueError, match="mutually exclusive"):
        bds_to_fits(str(synthetic_bds), str(tmp_path / "both"), freqs=[1.1e9], num_freq=2)


def test_output_eight_refuses_stokes(synthetic_bds, tmp_path):
    with pytest.raises(ValueError, match="not a Jones variable"):
        bds_to_fits(str(synthetic_bds), str(tmp_path / "p"), beam_type="nstokes", output_eight=True)


def test_hypercube_refuses_jones_subset(synthetic_bds, tmp_path):
    with pytest.raises(ValueError, match="full 2x2 Jones matrix"):
        bds_to_fits(str(synthetic_bds), str(tmp_path / "p"), elements=["XX"])


def test_hypercube_refuses_multiple_stokes_elements(synthetic_bds, tmp_path):
    with pytest.raises(ValueError, match="single Stokes/Mueller element"):
        bds_to_fits(str(synthetic_bds), str(tmp_path / "p"), beam_type="nstokes", elements=["II", "QQ"])


def test_unknown_beam_type_raises(synthetic_bds, tmp_path):
    with pytest.raises(ValueError, match="Unknown beam_type"):
        bds_to_fits(str(synthetic_bds), str(tmp_path / "p"), beam_type="bogus")


def test_invalid_element_raises(synthetic_bds, tmp_path):
    with pytest.raises(ValueError, match="Invalid Jones matrix element"):
        bds_to_fits(str(synthetic_bds), str(tmp_path / "p"), beam_type="jones", elements=["II"])
    with pytest.raises(ValueError, match="Invalid Stokes matrix element"):
        bds_to_fits(str(synthetic_bds), str(tmp_path / "q"), beam_type="nstokes", elements=["XX"])


def test_missing_prefix_raises(synthetic_bds):
    # the generated CLI's output parameter defaults to None
    with pytest.raises(ValueError, match="prefix is required"):
        bds_to_fits(str(synthetic_bds), None)


def test_bad_pixels_and_diameter_raise(synthetic_bds, tmp_path):
    with pytest.raises(ValueError, match="pixels must be at least 2"):
        bds_to_fits(str(synthetic_bds), str(tmp_path / "p"), pixels=1)
    with pytest.raises(ValueError, match="diameter must be positive"):
        bds_to_fits(str(synthetic_bds), str(tmp_path / "q"), diameter=0.0)


def test_missing_parent_directory_is_created(synthetic_bds, tmp_path):
    prefix = str(tmp_path / "new" / "deeper" / "beam")
    bds_to_fits(str(synthetic_bds), prefix, pixels=PIXELS, diameter=DIAMETER)
    assert Path(prefix + "_re.fits").exists()


def test_writes_through_a_file_uri(synthetic_bds, tmp_path):
    """The remote-prefix path (UPath handle) is exercised through file://."""
    prefix = f"file://{tmp_path}/beam"
    written = bds_to_fits(str(synthetic_bds), prefix, pixels=PIXELS, diameter=DIAMETER)
    assert [p[len("file://") :] for p in written] == [str(tmp_path / "beam_re.fits"), str(tmp_path / "beam_im.fits")]
    hdr, data = read(tmp_path / "beam_re.fits")
    assert data.shape == (len(FREQS), 2, 2, PIXELS, PIXELS)
    assert hdr["CTYPE1"] == "px"
