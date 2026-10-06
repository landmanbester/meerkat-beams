"""BeamWizard.partition_mueller and the centre= override that serves it."""

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time

from meerkat_beams.utils import BeamWizard
from tests._synthetic import DEC0, RA0, build_synthetic_image, build_telescope_bds

TIMES = Time("2024-01-01T00:00:00") + np.linspace(0, 2, 5) / 24.0
FREQ = 1.1e9


@pytest.fixture(scope="module")
def bds_path(tmp_path_factory):
    """A real BDS, built by running mdv_beams_to_bds on a synthetic mean-beam zarr."""
    tmp = tmp_path_factory.mktemp("partmueller")
    return str(build_telescope_bds(tmp / "mk.zarr", tmp / "mk.bds.zarr", scale=1.0, leak=0.05, phase=0.3))


@pytest.fixture(scope="module")
def image_path(tmp_path_factory):
    return str(build_synthetic_image(tmp_path_factory.mktemp("partmuellerimg") / "image.fits"))


@pytest.fixture
def bw(bds_path):
    """BDS-only wizard: no image, so self.centre raises until a centre is supplied."""
    return BeamWizard(bds_path)


FIELD_CENTRE = SkyCoord(ra=RA0, dec=DEC0, unit="deg")
L = np.linspace(-0.3, 0.3, 5)
M = np.linspace(-0.3, 0.3, 5)


def _avg(bw, **kwargs):
    """get_rotation_averaged_beam with this file's standard partition arguments."""
    defaults = dict(
        l=L, m=M, times=TIMES, freq=np.array([FREQ]), time_stepping=1, pixel_stepping=1, var="nstokes", verbose=0
    )
    defaults.update(kwargs)
    return bw.get_rotation_averaged_beam(**defaults)


@pytest.mark.unit
def test_centre_override_works_without_an_image(bw):
    """A BDS-only wizard has no self.centre; centre= must make PA averaging possible."""
    mean, var = _avg(bw, centre=FIELD_CENTRE)
    assert mean.shape == (len(M), len(L))
    assert np.isfinite(mean).all()


@pytest.mark.unit
def test_centre_override_does_not_mutate_the_wizard(bw):
    assert bw._centre is None
    _avg(bw, centre=FIELD_CENTRE)
    assert bw._centre is None


@pytest.mark.unit
def test_centre_none_matches_the_attached_image_centre(bds_path, image_path):
    """centre=None must reproduce today's behaviour bit for bit."""
    bw = BeamWizard(bds_path, image_path)
    implicit, _ = _avg(bw)
    explicit, _ = _avg(bw, centre=bw.centre)
    assert np.array_equal(implicit, explicit)


@pytest.mark.unit
def test_different_centres_give_different_parallactic_averages(bw):
    """The override must actually reach the PA computation, not be ignored."""
    south, _ = _avg(bw, centre=SkyCoord(ra=0.0, dec=-30.0, unit="deg"))
    pole, _ = _avg(bw, centre=SkyCoord(ra=0.0, dec=-75.0, unit="deg"))
    assert not np.allclose(south, pole)


@pytest.mark.unit
def test_centre_is_ignored_for_azimuth_averaging(bw):
    """average='azimuth' has no centre dependence; supplying one must not change it."""
    without = bw.get_rotation_averaged_beam(
        l=L, m=M, freq=np.array([FREQ]), average="azimuth", num_angles=8, pixel_stepping=1, verbose=0
    )[0]
    with_centre = bw.get_rotation_averaged_beam(
        l=L,
        m=M,
        freq=np.array([FREQ]),
        average="azimuth",
        num_angles=8,
        pixel_stepping=1,
        centre=SkyCoord(ra=0.0, dec=-75.0, unit="deg"),
        verbose=0,
    )[0]
    assert np.array_equal(without, with_centre)


@pytest.mark.unit
def test_default_block_shape(bw):
    """stokes_out='IQUV', stokes_in='I' is the apparent-I-from-intrinsic-I column."""
    M_blk = bw.partition_mueller(field_centre=FIELD_CENTRE, times=TIMES, freq=FREQ, l=L, m=M)
    assert M_blk.shape == (4, 1, len(M), len(L))


@pytest.mark.unit
def test_requested_subsets_set_the_block_shape(bw):
    M_blk = bw.partition_mueller(
        field_centre=FIELD_CENTRE, times=TIMES, freq=FREQ, l=L, m=M, stokes_out="IQ", stokes_in="IQUV"
    )
    assert M_blk.shape == (2, 4, len(M), len(L))


@pytest.mark.unit
def test_elements_equal_get_rotation_averaged_beam(bw):
    """partition_mueller assembles; it must not reimplement the averaging."""
    M_blk = bw.partition_mueller(
        field_centre=FIELD_CENTRE, times=TIMES, freq=FREQ, l=L, m=M, stokes_out="IQ", stokes_in="IU"
    )
    for a, i in enumerate("IQ"):
        for b, j in enumerate("IU"):
            expected, _ = _avg(bw, i=i, j=j, centre=FIELD_CENTRE)
            assert np.array_equal(M_blk[a, b], expected.astype(np.float32))


@pytest.mark.unit
def test_block_follows_caller_stokes_order(bw):
    """Column 0 of stokes_in='VI' is V, not the dataset's first Stokes parameter."""
    reversed_blk = bw.partition_mueller(
        field_centre=FIELD_CENTRE, times=TIMES, freq=FREQ, l=L, m=M, stokes_out="VI", stokes_in="VI"
    )
    forward_blk = bw.partition_mueller(
        field_centre=FIELD_CENTRE, times=TIMES, freq=FREQ, l=L, m=M, stokes_out="IV", stokes_in="IV"
    )
    # Reversing both axes of a 2x2 block is the same as [::-1, ::-1] of the other.
    assert np.array_equal(reversed_blk, forward_blk[::-1, ::-1])


@pytest.mark.unit
def test_on_axis_block_is_the_identity(bw):
    """nstokes is normalised so the on-axis Mueller is the identity; rotation fixes
    the centre pixel, so the PA average there is the on-axis value."""
    zero = np.array([0.0])
    M_blk = bw.partition_mueller(
        field_centre=FIELD_CENTRE, times=TIMES, freq=FREQ, l=zero, m=zero, stokes_out="IQUV", stokes_in="IQUV"
    )
    assert M_blk.shape == (4, 4, 1, 1)
    np.testing.assert_allclose(M_blk[:, :, 0, 0], np.eye(4), rtol=0, atol=1e-4)


@pytest.mark.unit
def test_non_square_grid_pins_the_y_x_order(bw):
    """Axis -2 is m/north and axis -1 is l/east. A non-square grid makes a
    transpose a shape error rather than a silent wrong answer."""
    l7 = np.linspace(-0.3, 0.3, 7)
    m5 = np.linspace(-0.2, 0.2, 5)
    M_blk = bw.partition_mueller(field_centre=FIELD_CENTRE, times=TIMES, freq=FREQ, l=l7, m=m5)
    assert M_blk.shape == (4, 1, 5, 7)


@pytest.mark.unit
def test_two_dimensional_lm_grids_pass_through(bw):
    """A caller holding meshgrid output passes it straight in."""
    ll, mm = np.meshgrid(L, M)
    from_2d = bw.partition_mueller(field_centre=FIELD_CENTRE, times=TIMES, freq=FREQ, l=ll, m=mm)
    from_1d = bw.partition_mueller(field_centre=FIELD_CENTRE, times=TIMES, freq=FREQ, l=L, m=M)
    assert np.array_equal(from_2d, from_1d)


@pytest.mark.unit
def test_pixel_stepping_returns_full_grid(bw):
    """pixel_stepping=3 on a 7-pixel axis must still return 7 pixels."""
    l7 = np.linspace(-0.3, 0.3, 7)
    m7 = np.linspace(-0.3, 0.3, 7)
    M_blk = bw.partition_mueller(field_centre=FIELD_CENTRE, times=TIMES, freq=FREQ, l=l7, m=m7, pixel_stepping=3)
    assert M_blk.shape == (4, 1, 7, 7)
    assert np.isfinite(M_blk).all()


@pytest.mark.unit
def test_dtype_is_float32_for_a_single_telescope(bw):
    M_blk = bw.partition_mueller(field_centre=FIELD_CENTRE, times=TIMES, freq=FREQ, l=L, m=M)
    assert M_blk.dtype == np.float32


@pytest.mark.unit
def test_normalised_false_selects_the_raw_stokes_beam(bw):
    """normalised=False must read 'stokes', not 'nstokes'."""
    raw = bw.partition_mueller(field_centre=FIELD_CENTRE, times=TIMES, freq=FREQ, l=L, m=M, normalised=False)
    expected, _ = _avg(bw, var="stokes", i="I", j="I", centre=FIELD_CENTRE)
    assert np.array_equal(raw[0, 0], expected.astype(np.float32))


@pytest.mark.unit
def test_field_centre_does_not_mutate_the_wizard(bw):
    assert bw._centre is None
    bw.partition_mueller(field_centre=FIELD_CENTRE, times=TIMES, freq=FREQ, l=L, m=M)
    assert bw._centre is None


@pytest.mark.unit
def test_an_attached_image_centre_survives_a_different_field_centre(bds_path, image_path):
    """A wizard built from an image keeps that centre even after answering for
    a partition pointed somewhere else."""
    bw = BeamWizard(bds_path, image_path)
    before = bw.centre
    bw.partition_mueller(field_centre=SkyCoord(ra=10.0, dec=-45.0, unit="deg"), times=TIMES, freq=FREQ, l=L, m=M)
    assert bw.centre is before


def _call(bw, **kwargs):
    """partition_mueller with valid defaults, so each test overrides one thing."""
    defaults = dict(field_centre=FIELD_CENTRE, times=TIMES, freq=FREQ, l=L, m=M)
    defaults.update(kwargs)
    return bw.partition_mueller(**defaults)


@pytest.mark.unit
def test_katbeam_is_refused_as_a_scope_decision():
    bw = BeamWizard(band="L", beam_model="katbeam", npix=16, fov_deg=2.0, num_freq=3)
    with pytest.raises(ValueError, match="no MeerKAT. model"):
        _call(bw)


@pytest.mark.unit
def test_weights_are_reserved(bw):
    with pytest.raises(NotImplementedError, match="weighted time averaging"):
        _call(bw, weights=np.ones(len(TIMES)))


@pytest.mark.unit
@pytest.mark.parametrize("value", [np.float64(1.1e9), np.array(1.1e9)])
def test_freq_accepts_numpy_scalars(bw, value):
    """np.float64 and 0-d arrays are scalars and must not be refused."""
    block = _call(bw, freq=value)
    reference = _call(bw, freq=FREQ)
    assert np.array_equal(block, reference)


@pytest.mark.unit
@pytest.mark.parametrize("value", [np.array([1.1e9]), [1.1e9], np.array([1.0e9, 1.1e9])])
def test_freq_array_raises(bw, value):
    """A frequency axis is a shape ambiguity; the message must say what to pass."""
    with pytest.raises(ValueError, match=r"float\(freq\)"):
        _call(bw, freq=value)


@pytest.mark.unit
def test_scalar_time_is_accepted(bw):
    """One timestamp, not wrapped in a list, is a reasonable partition."""
    single = Time("2024-01-01T00:00:00")
    block = _call(bw, times=single)
    wrapped = _call(bw, times=single.reshape(1))
    assert block.shape == (4, 1, len(M), len(L))
    assert np.array_equal(block, wrapped)


@pytest.mark.unit
def test_empty_times_raises(bw):
    """Rather than divide by zero and return NaN."""
    with pytest.raises(ValueError, match="times is empty"):
        _call(bw, times=Time([], format="isot"))


@pytest.mark.unit
@pytest.mark.parametrize("field", ["stokes_out", "stokes_in"])
def test_empty_stokes_selection_raises(bw, field):
    with pytest.raises(ValueError, match="at least one Stokes parameter"):
        _call(bw, **{field: ""})


@pytest.mark.unit
@pytest.mark.parametrize("field", ["stokes_out", "stokes_in"])
def test_unknown_stokes_character_raises(bw, field):
    with pytest.raises(ValueError, match="not in 'IQUV'"):
        _call(bw, **{field: "IX"})


@pytest.mark.unit
@pytest.mark.parametrize("field", ["stokes_out", "stokes_in"])
def test_lower_case_stokes_raises(bw, field):
    """Lower case is refused, not coerced: the BDS labels are upper case."""
    with pytest.raises(ValueError, match="upper case"):
        _call(bw, **{field: "iq"})


@pytest.mark.unit
@pytest.mark.parametrize("field", ["stokes_out", "stokes_in"])
def test_repeated_stokes_character_raises(bw, field):
    with pytest.raises(ValueError, match="repeats a Stokes parameter"):
        _call(bw, **{field: "IQI"})


@pytest.mark.unit
def test_frequency_outside_the_bds_range_is_left_to_interpolate_beam(bw):
    """Already guarded downstream; duplicating it would mean two messages to keep
    in step, so the existing one must still be what surfaces."""
    with pytest.raises(ValueError, match="fall outside the BDS frequency range"):
        _call(bw, freq=5.0e9)
