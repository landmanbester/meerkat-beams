"""
Unit tests for BeamWizard's frequency-extrapolation policy (issue #41).

Hermetic: builds a synthetic BDS in tmp_path (FREQS = 1.0-1.3 GHz, one channel
= 100 MHz). The synthetic beam is the same Gaussian in every plane, so any
frequency dependence in a "scale" result comes from the lambda/D scaling alone.
"""

import numpy as np
import pytest
import xarray

from meerkat_beams.utils import BeamWizard
from tests._synthetic import FREQS, build_synthetic_bds

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
