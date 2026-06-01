"""Unit tests for beam_orientation.calibrator and beam_orientation.plots."""

import numpy as np
import pytest
from beam_orientation import calibrator

from tests.conftest import CALIBRATOR_SPECTRUM


@pytest.mark.unit
def test_calibrator_spectrum_at_reference_frequency():
    nu0 = CALIBRATOR_SPECTRUM["nu0"]
    I0 = CALIBRATOR_SPECTRUM["I0"]
    val = calibrator.evaluate(np.array([nu0]))
    np.testing.assert_allclose(val, [I0], rtol=1e-12)


@pytest.mark.unit
def test_calibrator_spectrum_returns_positive_at_l_band():
    freqs = np.linspace(0.9e9, 1.7e9, 32)
    val = calibrator.evaluate(freqs)
    assert val.shape == freqs.shape
    assert np.all(val > 0)
    # Spectrum should be falling across L-band.
    assert val[0] > val[-1]


import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless backend for CI

from beam_orientation import plots  # noqa: E402


@pytest.fixture
def fake_2d():
    rng = np.random.default_rng(7)
    Nt, Nf = 8, 16
    times = np.linspace(0.0, 3600.0, Nt)  # seconds
    freq = np.linspace(0.9e9, 1.7e9, Nf)
    data = (1.0 + 0.01 * rng.standard_normal((Nt, Nf))) + 0.01j * rng.standard_normal((Nt, Nf))
    return times, freq, data


@pytest.mark.unit
def test_plot_dyn_spectrum_writes_png(tmp_path, fake_2d):
    times, freq, data = fake_2d
    out = tmp_path / "dyn.png"
    plots.dyn_spectrum(times, freq, data, out, title="t", cbar_label="Jy")
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.unit
def test_plot_time_profile_writes_png(tmp_path, fake_2d):
    times, freq, data = fake_2d
    out = tmp_path / "tp.png"
    plots.time_profile(times, data, out, title="t", ylabel="Jy")
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.unit
def test_plot_freq_profile_writes_png(tmp_path, fake_2d):
    times, freq, data = fake_2d
    out = tmp_path / "fp.png"
    plots.freq_profile(freq, data, out, title="t", ylabel="Jy")
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.unit
def test_profiles_ignore_nan_bins(tmp_path, fake_2d):
    times, freq, data = fake_2d
    data = data.copy()
    data[0, :] = np.nan  # blank one time row
    data[:, 0] = np.nan  # blank one freq channel
    out = tmp_path / "tp_nan.png"
    plots.time_profile(times, data, out, title="t", ylabel="Jy")
    assert out.exists() and out.stat().st_size > 0
    # The reduction must not propagate the blanked row into every channel.
    prof = plots._profile(data, axis=0)
    assert np.isfinite(prof[1:]).all()


@pytest.mark.unit
def test_profile_excludes_fully_flagged_zeros():
    # Exact complex zeros (fully-flagged fill) are dropped from the mean; a
    # slice with no surviving bins reduces to NaN (so nothing is plotted there).
    data = np.array(
        [
            [1.0 + 0j, 0.0 + 0j, 3.0 + 0j],  # mean over surviving (1, 3) = 2.0
            [0.0 + 0j, 0.0 + 0j, 0.0 + 0j],  # all flagged -> NaN
        ]
    )
    prof = plots._profile(data, axis=1)
    assert prof[0] == pytest.approx(2.0)
    assert np.isnan(prof[1])
