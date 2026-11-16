"""Phase centre vs pointing centre: the geometry and the render methods' phase_centre= (issue #57)."""

import astropy.units as u
import numpy as np
import pytest
import xarray
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.wcs import WCS

from meerkat_beams.core.bds_to_xradio import bds_to_xradio, parse_pointing_centre
from meerkat_beams.utils import (
    OFF_SKY_DEG,
    BeamWizard,
    phase_to_pointing_lm,
    sky_rotation_matrix,
)
from tests._synthetic import DEC0, RA0, build_synthetic_bds, build_synthetic_image, build_synthetic_image_zarr

TIMES = Time("2024-01-01T00:00:00") + np.linspace(0, 2, 5) / 24.0
FREQ = np.array([1.1e9])
PHASE = SkyCoord(ra=RA0, dec=DEC0, unit="deg")


def _offset(pa_deg, sep_deg):
    return PHASE.directional_offset_by(pa_deg * u.deg, sep_deg * u.deg)


# east of and a little south of the phase centre, inside the synthetic image (+-0.32 deg)
POINTING = _offset(110.0, 0.15)


def _sin_lm(direction, about):
    """SIN direction cosines of ``direction`` about ``about``, in degrees (East, North positive)."""
    a, d = direction.icrs.ra.rad, direction.icrs.dec.rad
    a0, d0 = about.icrs.ra.rad, about.icrs.dec.rad
    l = np.cos(d) * np.sin(a - a0)
    m = np.sin(d) * np.cos(d0) - np.cos(d) * np.sin(d0) * np.cos(a - a0)
    return np.rad2deg(l), np.rad2deg(m)


def _sin_wcs(centre, cdelt=0.01):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    w.wcs.cunit = ["deg", "deg"]
    w.wcs.cdelt = [-cdelt, cdelt]
    w.wcs.crpix = [1.0, 1.0]
    w.wcs.crval = [centre.icrs.ra.deg, centre.icrs.dec.deg]
    return w


@pytest.fixture(scope="module")
def bds_path(tmp_path_factory):
    return str(build_synthetic_bds(tmp_path_factory.mktemp("phasecentre") / "bds.zarr"))


@pytest.fixture(scope="module")
def image_path(tmp_path_factory):
    return str(build_synthetic_image(tmp_path_factory.mktemp("phasecentreimg") / "image.fits"))


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_rotation_is_exactly_the_identity_for_equal_centres():
    assert np.array_equal(sky_rotation_matrix(PHASE, PHASE), np.eye(3))
    # an equal direction in another frame is taken to ICRS first
    assert np.array_equal(sky_rotation_matrix(PHASE, SkyCoord(PHASE.icrs.ra, PHASE.icrs.dec)), np.eye(3))


@pytest.mark.unit
def test_rotation_is_proper_and_orthogonal():
    rot = sky_rotation_matrix(POINTING, PHASE)
    np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-15)
    assert np.linalg.det(rot) == pytest.approx(1.0, abs=1e-14)


@pytest.mark.unit
def test_identity_returns_l_and_m_unchanged():
    l = np.linspace(-2, 2, 7)
    m = np.linspace(-1, 3, 7)
    lp, mp, valid = phase_to_pointing_lm(l, m, np.eye(3))
    assert np.array_equal(lp, l) and np.array_equal(mp, m) and valid.all()


@pytest.mark.unit
def test_pointing_direction_maps_to_the_origin():
    """The pointing centre, seen from the phase centre, is the beam axis."""
    l, m = _sin_lm(POINTING, PHASE)
    assert l > 0  # pointing is east of the phase centre: l is East positive
    lp, mp, valid = phase_to_pointing_lm(l, m, sky_rotation_matrix(POINTING, PHASE))
    assert valid
    np.testing.assert_allclose([lp, mp], [0.0, 0.0], atol=1e-12)


@pytest.mark.unit
def test_matches_astropy_sin_to_sin_reprojection():
    """Oracle: phase-centre SIN pixel -> world -> pointing-centre SIN pixel through astropy WCS."""
    pointing = _offset(-35.0, 1.7)  # an offset large enough that SIN != flat-sky
    rng = np.random.default_rng(57)
    l = rng.uniform(-3, 3, 200)
    m = rng.uniform(-3, 3, 200)
    lp, mp, valid = phase_to_pointing_lm(l, m, sky_rotation_matrix(pointing, PHASE))
    assert valid.all()

    w_phase, w_point = _sin_wcs(PHASE), _sin_wcs(pointing)
    # 0-based pixel of intermediate coordinate x is x / cdelt (crpix = 1); RA cdelt is
    # negative, so x = (p - 0) * cdelt is l (East positive) itself
    world = w_phase.wcs_pix2world(l / w_phase.wcs.cdelt[0], m / w_phase.wcs.cdelt[1], 0)
    px, py = w_point.wcs_world2pix(world[0], world[1], 0)
    np.testing.assert_allclose(lp, px * w_point.wcs.cdelt[0], atol=1e-9)
    np.testing.assert_allclose(mp, py * w_point.wcs.cdelt[1], atol=1e-9)


@pytest.mark.unit
def test_off_sky_and_far_hemisphere_are_invalid():
    rot = sky_rotation_matrix(POINTING, PHASE)
    lp, mp, valid = phase_to_pointing_lm(np.array([70.0, 0.0]), np.array([0.0, 0.0]), rot)
    assert valid.tolist() == [False, True]  # 70 deg is deg2rad(70) = 1.22 > 1: off the sky
    assert lp[0] == OFF_SKY_DEG and mp[0] == OFF_SKY_DEG

    far = phase_to_pointing_lm(0.0, 0.0, sky_rotation_matrix(SkyCoord(0, 90, unit="deg"), PHASE))
    assert not far[2]  # dec -30 is on the far side of the north pole's hemisphere


# ---------------------------------------------------------------------------
# render methods
# ---------------------------------------------------------------------------


def _avg(bw, **kwargs):
    kw = dict(times=TIMES, freq=FREQ, time_stepping=1, pixel_stepping=1, verbose=0)
    kw.update(kwargs)
    return bw.get_rotation_averaged_beam(**kw)[0]


@pytest.mark.unit
def test_phase_centre_equal_to_pointing_is_bit_identical(bds_path):
    bw = BeamWizard(bds_path)
    l = np.linspace(-0.3, 0.3, 9)
    without = _avg(bw, l=l, m=l, pointing_centre=POINTING)
    with_phase = _avg(bw, l=l, m=l, pointing_centre=POINTING, phase_centre=POINTING)
    assert np.array_equal(without, with_phase)


@pytest.mark.unit
@pytest.mark.parametrize("opts", [dict(), dict(pixel_stepping=3), dict(pa_tolerance=5.0)])
def test_phase_centre_is_the_mapped_2d_grid(bds_path, opts):
    """phase_centre= is exactly the existing 2D path fed the mapped coordinates."""
    bw = BeamWizard(bds_path)
    l = np.linspace(-0.3, 0.3, 10)
    m = np.linspace(-0.25, 0.35, 7)
    mapped = _avg(bw, l=l, m=m, pointing_centre=POINTING, phase_centre=PHASE, **opts)
    ll, mm = np.meshgrid(l, m)
    lp, mp, _ = phase_to_pointing_lm(ll, mm, sky_rotation_matrix(POINTING, PHASE))
    direct = _avg(bw, l=lp, m=mp, pointing_centre=POINTING, **opts)
    assert mapped.shape == (len(m), len(l))
    assert np.array_equal(mapped, direct)


@pytest.mark.unit
def test_beam_peak_lands_on_the_pointing_direction(bds_path, image_path):
    """On the image grid, an offset pointing moves the beam peak to the pointing's own pixel.

    The synthetic beam is circular, so this pins the translation (including the
    sign of l) and not the parallactic rotation, which beam-orientation.md owns.
    """
    bw = BeamWizard(bds_path, image_path)
    assert bw.phase_centre.separation(PHASE).deg < 1e-9
    centred = _avg(bw)
    peak = np.unravel_index(np.argmax(centred), centred.shape)
    assert bw.m_grid[peak[0]] == pytest.approx(0.0, abs=0.006)
    assert bw.l_grid[peak[1]] == pytest.approx(0.0, abs=0.006)

    offset = _avg(bw, pointing_centre=POINTING)
    peak = np.unravel_index(np.argmax(offset), offset.shape)
    l_expected, m_expected = _sin_lm(POINTING, PHASE)
    assert bw.l_grid[peak[1]] == pytest.approx(l_expected, abs=0.006)
    assert bw.m_grid[peak[0]] == pytest.approx(m_expected, abs=0.006)

    # set_pointing_centre on the wizard is the same as passing it per call
    bw.set_pointing_centre(POINTING)
    assert np.array_equal(_avg(bw), offset)
    assert bw.phase_centre.separation(PHASE).deg < 1e-9


@pytest.mark.unit
def test_azimuth_average_needs_no_pointing_without_a_phase_centre(bds_path):
    bw = BeamWizard(bds_path)
    l = np.linspace(-0.3, 0.3, 5)
    mean, _ = bw.get_rotation_averaged_beam(
        l=l, m=l, freq=FREQ, average="azimuth", num_angles=8, pixel_stepping=1, verbose=0
    )
    assert np.isfinite(mean).all()
    with pytest.raises(RuntimeError, match="pointing_centre"):
        bw.get_rotation_averaged_beam(
            l=l, m=l, freq=FREQ, average="azimuth", num_angles=8, pixel_stepping=1, verbose=0, phase_centre=PHASE
        )


@pytest.mark.unit
def test_partition_mueller_passes_phase_centre_through(bds_path):
    bw = BeamWizard(bds_path)
    l = np.linspace(-0.3, 0.3, 6)
    kw = dict(pointing_centre=POINTING, times=TIMES, freq=float(FREQ[0]), l=l, m=l, stokes_out="IQ", stokes_in="I")
    assert np.array_equal(bw.partition_mueller(**kw), bw.partition_mueller(**kw, phase_centre=POINTING))
    block = bw.partition_mueller(**kw, phase_centre=PHASE)
    expected = _avg(bw, l=l, m=l, pointing_centre=POINTING, phase_centre=PHASE, i="Q", j="I", var="nstokes")
    assert np.array_equal(block[1, 0], expected.astype(np.float32))
    assert not np.allclose(block, bw.partition_mueller(**kw))


@pytest.mark.unit
def test_bds_to_xradio_pointing_centre_moves_the_peak(tmp_path):
    bds = build_synthetic_bds(tmp_path / "bds.zarr")
    img = build_synthetic_image_zarr(tmp_path / "img.zarr")
    out_c = bds_to_xradio(bds, img, tmp_path / "centred.zarr", pixel_stepping=1)
    text = f"{POINTING.ra.deg} {POINTING.dec.deg}"
    out_o = bds_to_xradio(bds, img, tmp_path / "offset.zarr", pointing_centre=text, pixel_stepping=1)
    sky_c = xarray.open_zarr(out_c)["SKY"]
    sky_o = xarray.open_zarr(out_o)["SKY"]
    # the direction reference is the image's phase centre either way
    assert sky_o.attrs == sky_c.attrs
    plane = sky_o.isel(time=0, frequency=0, polarization=0).values  # (l, m)
    il, im = np.unravel_index(np.argmax(plane), plane.shape)
    l_expected, m_expected = _sin_lm(POINTING, PHASE)
    assert np.rad2deg(sky_o["l"].values[il]) == pytest.approx(l_expected, abs=0.006)
    assert np.rad2deg(sky_o["m"].values[im]) == pytest.approx(m_expected, abs=0.006)


@pytest.mark.unit
def test_parse_pointing_centre_refuses_colons():
    assert parse_pointing_centre("05h21m09.9s +16d38m22s").ra.deg == pytest.approx(80.29125)
    assert parse_pointing_centre("80.29 16.64").dec.deg == pytest.approx(16.64)
    with pytest.raises(ValueError, match="colons"):
        parse_pointing_centre("05:21:09.9 +16:38:22")
