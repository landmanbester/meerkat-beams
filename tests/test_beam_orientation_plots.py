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
def fake_dyn_spec():
    rng = np.random.default_rng(2)
    Nt, Nf = 8, 16
    times = np.linspace(0.0, 3600.0, Nt)  # seconds
    freq = np.linspace(0.9e9, 1.7e9, Nf)
    B = (1.0 + 0.01 * rng.standard_normal((Nt, Nf, 4))).astype(complex)
    cond = np.ones((Nt, Nf), dtype=float) * 1.5
    return times, freq, B, cond


@pytest.mark.unit
def test_plot_waterfall_writes_png(tmp_path, fake_dyn_spec):
    times, freq, B, cond = fake_dyn_spec
    out = tmp_path / "waterfall_I.png"
    plots.waterfall(times, freq, B, cond, stokes="I", out_path=out)
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.unit
def test_plot_mean_spectrum_writes_png(tmp_path, fake_dyn_spec):
    times, freq, B, cond = fake_dyn_spec
    out = tmp_path / "mean_I_spectrum.png"
    plots.mean_spectrum(freq, B, cond, out_path=out)
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.unit
def test_plot_time_variation_writes_png(tmp_path, fake_dyn_spec):
    times, freq, B, cond = fake_dyn_spec
    out = tmp_path / "time_variation.png"
    plots.time_variation(freq, B, cond, out_path=out)
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.unit
def test_plot_control_overlay_writes_png(tmp_path, fake_dyn_spec):
    times, freq, B, cond = fake_dyn_spec
    runs = {
        "none": (B, cond),
        "flip_x": (B * 1.1, cond),
        "flip_y": (B * 1.2, cond),
        "swap_xy": (B * 1.3, cond),
    }
    out = tmp_path / "control_overlay.png"
    plots.control_overlay(freq, runs, out_path=out)
    assert out.exists() and out.stat().st_size > 0
