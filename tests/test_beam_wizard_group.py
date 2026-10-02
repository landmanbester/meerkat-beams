"""BeamWizard's group= constructor path."""

import astropy.units as u
import numpy as np
import pytest
from astropy.time import Time

from meerkat_beams.utils import BeamWizard
from tests._synthetic import I0, build_synthetic_image, build_telescope_bds

MK = "MeerKAT"
MKE = "MeerKAT Extension"


@pytest.fixture(scope="module")
def paired_cache(tmp_path_factory):
    """Two BDS paths plus a FITS image, as cache.ensure_group_bds would return."""
    tmp = tmp_path_factory.mktemp("wizgroup")
    p = build_telescope_bds(tmp / "mk.zarr", tmp / "mk.bds.zarr", scale=1.0, leak=0.05, telescope=MK)
    q = build_telescope_bds(tmp / "mke.zarr", tmp / "mke.bds.zarr", scale=0.8, leak=0.12, phase=0.4, telescope=MKE)
    image = build_synthetic_image(tmp / "image.fits")
    return str(p), str(q), str(image)


@pytest.fixture
def patched_cache(paired_cache, monkeypatch):
    """Route ensure_group_bds at the two local stores, recording the call."""
    from meerkat_beams import cache

    p, q, _ = paired_cache
    calls = []

    def stub(band, group):
        calls.append((band, group))
        return {"MM": (p, p), "MPM": (p, q), "MPMP": (q, q)}[group]

    monkeypatch.setattr(cache, "ensure_group_bds", stub)
    return calls


@pytest.mark.unit
@pytest.mark.parametrize("group", ["MM", "MPM", "MPMP"])
def test_group_wizard_constructs(patched_cache, group):
    bw = BeamWizard(band="L", group=group)
    assert bw.group == group
    assert "nstokes" in bw.bds.data_vars
    assert patched_cache == [("L", group)]


@pytest.mark.unit
def test_group_wizard_records_telescopes(patched_cache):
    bw = BeamWizard(band="L", group="MPM")
    assert bw.telescope_p == MK
    assert bw.telescope_q == MKE


@pytest.mark.unit
def test_group_none_leaves_telescopes_none(paired_cache):
    p, _, _ = paired_cache
    bw = BeamWizard(p)
    assert bw.group is None
    assert bw.telescope_p is None and bw.telescope_q is None


@pytest.mark.unit
def test_group_wizard_interpolates_on_axis(patched_cache):
    """The assembled dataset works through the ordinary interpolation path."""
    bw = BeamWizard(band="L", group="MPM")
    xpyp = np.array([[float(I0)], [float(I0)]])
    val = bw.interpolate_beam(xpyp, bw.bds.coords["FREQ"].values[:1], var="nstokes", i="I", j="I")
    np.testing.assert_allclose(val.real, 1.0, rtol=1e-4, atol=1e-4)


@pytest.mark.unit
def test_group_wizard_with_image_rotation_averages(patched_cache, paired_cache):
    """group= must compose with image_name=: downstream methods are unchanged."""
    _, _, image = paired_cache
    bw = BeamWizard(band="L", group="MPM", image_name=image)
    times = Time("2024-01-01T00:00:00") + np.linspace(0, 1, 3) * u.hour
    l = m = np.linspace(-0.2, 0.2, 5)
    mean, var = bw.get_rotation_averaged_beam(
        l=l,
        m=m,
        times=times,
        freq=bw.bds.coords["FREQ"].values[:1],
        time_stepping=1,
        pixel_stepping=1,
        verbose=0,
    )
    assert mean.shape == (5, 5)
    assert np.isfinite(mean).all()


@pytest.mark.unit
def test_mpm_njones_raises_with_group_hint(patched_cache):
    bw = BeamWizard(band="L", group="MPM")
    with pytest.raises(ValueError, match="cross baseline has no single Jones"):
        bw._get_prefilter("njones", 0, 0)


@pytest.mark.unit
def test_group_rejects_bds_name(paired_cache):
    p, _, _ = paired_cache
    with pytest.raises(ValueError, match="mutually exclusive"):
        BeamWizard(p, group="MPM")


@pytest.mark.unit
def test_group_requires_band():
    with pytest.raises(ValueError, match="requires band"):
        BeamWizard(group="MPM")


@pytest.mark.unit
def test_group_rejects_katbeam():
    with pytest.raises(ValueError, match="not available for beam_model='katbeam'"):
        BeamWizard(band="L", group="MPM", beam_model="katbeam")


@pytest.mark.unit
def test_unknown_group_raises():
    with pytest.raises(ValueError, match="group must be one of"):
        BeamWizard(band="L", group="MPX")


@pytest.mark.unit
def test_unsupported_group_band_raises():
    """S band has no matched counterpart; the error must say so."""
    with pytest.raises(ValueError, match="no matched MeerKAT/MeerKAT. beam counterpart"):
        BeamWizard(band="S0", group="MPM")
