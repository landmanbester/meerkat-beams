"""
Unit tests for BeamWizard's frequency-extrapolation policy (issue #41).

Hermetic: builds a synthetic BDS in tmp_path (FREQS = 1.0-1.3 GHz, one channel
= 100 MHz). The synthetic beam is the same Gaussian in every plane, so any
frequency dependence in a "scale" result comes from the lambda/D scaling alone.
"""

import logging

import astropy.units as u
import numpy as np
import pytest
import xarray
from astropy.coordinates import SkyCoord
from astropy.time import Time

from meerkat_beams.utils import BeamWizard
from tests._synthetic import DEC0, FREQS, I0, RA0, build_synthetic_bds, build_synthetic_image

CHANNEL = 1.0e8  # FREQS spacing, Hz


@pytest.fixture(scope="module")
def bds_path(tmp_path_factory):
    return str(build_synthetic_bds(tmp_path_factory.mktemp("freqextrap") / "synthetic.bds.zarr"))


def _wizard(bds_path, **kwargs):
    return BeamWizard(bds_path, **kwargs)


def _relabel(src, dst, freqs):
    """Copy the BDS at ``src`` to ``dst`` with its FREQ axis replaced by ``freqs`` (same length or a prefix)."""
    ds = xarray.open_zarr(src).isel(FREQ=slice(0, len(freqs))).load()
    ds = ds.assign_coords(FREQ=np.asarray(freqs, dtype=float))
    ds.attrs["freqs"] = np.asarray(freqs, dtype=float)
    for var in ds.variables.values():
        var.encoding = {}
    ds.to_zarr(str(dst), mode="w")
    return str(dst)


# ---------------------------------------------------------------------------
# policy state and freq_range
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_default_policy_is_raise(bds_path):
    bw = _wizard(bds_path)
    assert bw.freq_extrapolation == "raise"
    assert bw.freq_range == (1.0e9, 1.3e9)


@pytest.mark.unit
def test_default_margin_is_one_channel(bds_path):
    bw = _wizard(bds_path, freq_extrapolation="scale")
    assert bw.max_freq_extrapolation == CHANNEL
    assert bw.freq_range == pytest.approx((0.9e9, 1.4e9))


@pytest.mark.unit
def test_explicit_margin_overrides_default(bds_path):
    bw = _wizard(bds_path, freq_extrapolation="scale", max_freq_extrapolation=2.5e7)
    assert bw.freq_range == pytest.approx((0.975e9, 1.325e9))


@pytest.mark.unit
def test_invalid_policy_raises(bds_path):
    with pytest.raises(ValueError, match="'raise' or 'scale'"):
        _wizard(bds_path, freq_extrapolation="clamp")


@pytest.mark.unit
def test_negative_margin_raises(bds_path):
    with pytest.raises(ValueError, match="max_freq_extrapolation"):
        _wizard(bds_path, freq_extrapolation="scale", max_freq_extrapolation=-1.0)


@pytest.mark.unit
def test_katbeam_scale_is_refused():
    pytest.importorskip("katbeam")
    with pytest.raises(ValueError, match="freq_extrapolation") as exc:
        BeamWizard(band="L", beam_model="katbeam", freq_extrapolation="scale")
    assert "katbeam" in str(exc.value)


# ---------------------------------------------------------------------------
# check_freqs
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_freqs_raise_message_unchanged(bds_path):
    with pytest.raises(ValueError, match="fall outside the BDS frequency range"):
        _wizard(bds_path).check_freqs([0.95e9])


@pytest.mark.unit
def test_check_freqs_returns_float_array(bds_path):
    out = _wizard(bds_path).check_freqs([1.1e9, 1.2e9])
    assert isinstance(out, np.ndarray) and out.dtype == float
    np.testing.assert_array_equal(out, [1.1e9, 1.2e9])


@pytest.mark.unit
@pytest.mark.parametrize("freq", [0.85e9, 1.45e9])
def test_scale_beyond_margin_raises(bds_path, freq):
    with pytest.raises(ValueError, match="extrapolation margin"):
        _wizard(bds_path, freq_extrapolation="scale").check_freqs([freq])


@pytest.mark.unit
def test_scale_accepts_frequency_exactly_at_margin(bds_path):
    """The 2026 bottom edge lands exactly on the margin (869.375 - 13.375 = 856);
    round-off of up to 1 kHz must not reject it."""
    _wizard(bds_path, freq_extrapolation="scale").check_freqs([0.9e9 - 500.0, 1.4e9 + 500.0])


@pytest.mark.unit
def test_scale_units_mistake_raises(bds_path):
    """Frequencies passed in MHz instead of Hz must still fail loudly."""
    with pytest.raises(ValueError, match="fall outside the BDS frequency range"):
        _wizard(bds_path, freq_extrapolation="scale").check_freqs(FREQS * 1e-6)


@pytest.mark.unit
def test_scale_covers_top_channel_of_finer_ms(bds_path, tmp_path):
    """Legacy-L labelling (856 + k * 856/1024 MHz): the top channel of a 32k MS
    (the analogue of 1711.97 MHz against 1711.164) is within the margin."""
    dbds = 856e6 / 1024
    path = _relabel(bds_path, tmp_path / "legacy.bds.zarr", 856e6 + np.arange(len(FREQS)) * dbds)
    bw = _wizard(path, freq_extrapolation="scale")
    fmax = bw.bds.coords["FREQ"].values.max()
    bw.check_freqs([fmax + dbds - 856e6 / 32768])
    with pytest.raises(ValueError, match="extrapolation margin"):
        bw.check_freqs([fmax + dbds + 2e3])


@pytest.mark.unit
def test_single_channel_bds_default_margin_is_zero(bds_path, tmp_path):
    path = _relabel(bds_path, tmp_path / "one.bds.zarr", FREQS[:1])
    bw = _wizard(path, freq_extrapolation="scale")
    assert bw.max_freq_extrapolation == 0.0
    with pytest.raises(ValueError, match="fall outside the BDS frequency range"):
        bw.check_freqs([1.0e9 + 1.0e6])


# ---------------------------------------------------------------------------
# "scale" evaluation
# ---------------------------------------------------------------------------

XPYP = np.array([[I0, I0 + 4.0, I0 - 3.0], [I0, I0 + 2.0, I0 + 5.0]])


@pytest.mark.unit
@pytest.mark.parametrize("nu, edge", [(0.95e9, 1.0e9), (1.35e9, 1.3e9)])
def test_scale_matches_scaled_offsets_at_edge_plane(bds_path, nu, edge):
    s = nu / edge
    expected = _wizard(bds_path).interpolate_beam(I0 + (XPYP - I0) * s, [edge])
    got = _wizard(bds_path, freq_extrapolation="scale").interpolate_beam(XPYP, [nu])
    np.testing.assert_allclose(got, expected, rtol=1e-6)


@pytest.mark.unit
def test_scale_on_axis_equals_edge_plane(bds_path):
    on_axis = np.array([[float(I0)], [float(I0)]])
    expected = _wizard(bds_path).interpolate_beam(on_axis, [1.0e9])
    got = _wizard(bds_path, freq_extrapolation="scale").interpolate_beam(on_axis, [0.95e9])
    np.testing.assert_array_equal(got, expected)


@pytest.mark.unit
def test_scale_narrows_beam_above_band(bds_path):
    """The synthetic Gaussian is the same in every plane, so only the scaling can change it."""
    bw = _wizard(bds_path, freq_extrapolation="scale")
    xpyp = np.array([[I0 + 6.0], [float(I0)]])
    assert bw.interpolate_beam(xpyp, [1.35e9])[0, 0] < bw.interpolate_beam(xpyp, [1.3e9])[0, 0]
    assert bw.interpolate_beam(xpyp, [0.95e9])[0, 0] > bw.interpolate_beam(xpyp, [1.0e9])[0, 0]


@pytest.mark.unit
def test_scale_mixed_frequencies_in_range_rows_bit_identical(bds_path):
    scale_bw = _wizard(bds_path, freq_extrapolation="scale")
    got = scale_bw.interpolate_beam(XPYP, [0.95e9, 1.05e9, 1.3e9, 1.35e9])
    np.testing.assert_array_equal(got[1:3], _wizard(bds_path).interpolate_beam(XPYP, [1.05e9, 1.3e9]))
    np.testing.assert_array_equal(got[0], scale_bw.interpolate_beam(XPYP, [0.95e9])[0])
    np.testing.assert_array_equal(got[3], scale_bw.interpolate_beam(XPYP, [1.35e9])[0])


@pytest.mark.unit
@pytest.mark.parametrize("var, i, j", [("nstokes", "I", "I"), ("nmueller", "U", "V")])
@pytest.mark.parametrize("freqs", [FREQS, [1.17e9]])
def test_in_range_is_bit_identical_across_policies(bds_path, var, i, j, freqs):
    a = _wizard(bds_path).interpolate_beam(XPYP, freqs, var=var, i=i, j=j)
    b = _wizard(bds_path, freq_extrapolation="scale").interpolate_beam(XPYP, freqs, var=var, i=i, j=j)
    np.testing.assert_array_equal(a, b)


@pytest.mark.unit
def test_scale_complex_var_preserves_dtype(bds_path):
    got = _wizard(bds_path, freq_extrapolation="scale").interpolate_beam(XPYP, [1.35e9], var="nmueller", i="U", j="V")
    assert got.dtype == np.complex64
    assert np.any(got.imag != 0)


@pytest.mark.unit
def test_scale_warns_once_per_wizard(bds_path, caplog):
    bw = _wizard(bds_path, freq_extrapolation="scale")
    with caplog.at_level(logging.WARNING, logger="meerkat_beams"):
        bw.interpolate_beam(XPYP, [0.95e9])
        bw.interpolate_beam(XPYP, [1.35e9])
    hits = [r for r in caplog.records if "extrapolat" in r.message and "scale" in r.message]
    assert len(hits) == 1


@pytest.mark.unit
def test_scale_in_range_does_not_warn(bds_path, caplog):
    bw = _wizard(bds_path, freq_extrapolation="scale")
    with caplog.at_level(logging.WARNING, logger="meerkat_beams"):
        bw.interpolate_beam(XPYP, FREQS)
    assert not [r for r in caplog.records if "extrapolat" in r.message]


# ---------------------------------------------------------------------------
# public entry points honour the wizard's policy
# ---------------------------------------------------------------------------

TIMES = Time("2024-01-01T00:00:00") + np.linspace(0, 1, 5) * u.hour
FIELD_CENTRE = SkyCoord(ra=RA0, dec=DEC0, unit="deg")


@pytest.fixture(scope="module")
def image_path(tmp_path_factory):
    return str(build_synthetic_image(tmp_path_factory.mktemp("freqextrapimg") / "image.fits"))


@pytest.mark.unit
def test_partition_mueller_scale_evaluates_band_edge(bds_path):
    kw = dict(
        field_centre=FIELD_CENTRE, times=TIMES, freq=0.95e9, l=np.linspace(-0.3, 0.3, 5), m=np.linspace(-0.3, 0.3, 5)
    )
    block = _wizard(bds_path, freq_extrapolation="scale").partition_mueller(**kw)
    assert block.shape == (4, 1, 5, 5)
    assert np.isfinite(block).all()
    with pytest.raises(ValueError, match="fall outside the BDS frequency range"):
        _wizard(bds_path).partition_mueller(**kw)


@pytest.mark.unit
def test_rotation_averaged_scale_matches_edge_plane_on_axis(bds_path, image_path):
    kw = dict(l=np.array([0.0]), m=np.array([0.0]), times=TIMES, var="nstokes", verbose=0)
    scaled, _ = BeamWizard(bds_path, image_path, freq_extrapolation="scale").get_rotation_averaged_beam(
        freq=np.array([1.35e9]), **kw
    )
    edge, _ = BeamWizard(bds_path, image_path).get_rotation_averaged_beam(freq=np.array([1.3e9]), **kw)
    np.testing.assert_allclose(scaled, edge, rtol=1e-6)


@pytest.mark.unit
def test_time_freq_beam_scale_renders_band_edge(bds_path, image_path, tmp_path):
    kw = dict(
        var_name="BEAM",
        dim_names=("time", "frequency", "polarization", "l", "m"),
        l=np.array([-0.05, 0.0, 0.05]),
        m=np.array([-0.05, 0.0, 0.05]),
        times=TIMES,
        freq=np.array([0.95e9, 1.0e9]),
        pixel_stepping=1,
        time_stepping=1,
        ij_list=[("I", "I")],
        var="nstokes",
        verbose=0,
    )
    out = tmp_path / "scaled.zarr"
    BeamWizard(bds_path, image_path, freq_extrapolation="scale").get_time_freq_beam(filename=str(out), **kw)
    ds = xarray.open_zarr(str(out))
    np.testing.assert_allclose(ds.coords["frequency"].values, [0.95e9, 1.0e9])
    assert np.isfinite(ds["BEAM"].values).all()
    with pytest.raises(ValueError, match="fall outside the BDS frequency range"):
        BeamWizard(bds_path, image_path).get_time_freq_beam(filename=str(tmp_path / "raised.zarr"), **kw)


@pytest.mark.unit
def test_get_time_variable_beamgain_scale(bds_path, image_path):
    scale_bw = BeamWizard(bds_path, image_path, freq_extrapolation="scale")
    raise_bw = BeamWizard(bds_path, image_path)
    on_axis = scale_bw.get_time_variable_beamgain(scale_bw.centre, freq=np.array([1.35e9]), times=TIMES)
    np.testing.assert_allclose(
        on_axis, raise_bw.get_time_variable_beamgain(raise_bw.centre, freq=np.array([1.3e9]), times=TIMES)
    )
    off = SkyCoord(ra=RA0 + 0.3, dec=DEC0, unit="deg")
    above = scale_bw.get_time_variable_beamgain(off, freq=np.array([1.35e9]), times=TIMES)
    edge = raise_bw.get_time_variable_beamgain(off, freq=np.array([1.3e9]), times=TIMES)
    assert (above < edge).all()  # narrower beam above the band


@pytest.mark.unit
@pytest.mark.parametrize("policy", ["raise", "scale"])
def test_non_finite_frequency_raises(bds_path, policy):
    """NaN makes min()/max() comparisons False; it must not smuggle 5 GHz past the guard."""
    with pytest.raises(ValueError, match="finite"):
        _wizard(bds_path, freq_extrapolation=policy).interpolate_beam(XPYP, [np.nan, 5e9])
