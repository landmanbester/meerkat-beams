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
