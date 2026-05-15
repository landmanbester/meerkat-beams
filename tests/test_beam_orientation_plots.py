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
