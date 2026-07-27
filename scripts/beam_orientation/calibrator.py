"""
Evaluate the PKS 1934-638 spectrum.

Coefficients live in tests/conftest.py:CALIBRATOR_SPECTRUM. Formula:

    I(ν) = I0 * (ν/ν0) ** (a + b*x + c*x**2 + d*x**3 + e*x**4)
    x    = log10(ν/ν0)
"""

import numpy as np

from tests.conftest import CALIBRATOR_SPECTRUM


def evaluate(freq_hz: np.ndarray) -> np.ndarray:
    """Return Stokes I in Jy at each frequency in ``freq_hz`` (Hz)."""
    freq_hz = np.asarray(freq_hz, dtype=float)
    nu0 = CALIBRATOR_SPECTRUM["nu0"]
    I0 = CALIBRATOR_SPECTRUM["I0"]
    a = CALIBRATOR_SPECTRUM["a"]
    b = CALIBRATOR_SPECTRUM["b"]
    c = CALIBRATOR_SPECTRUM["c"]
    d = CALIBRATOR_SPECTRUM["d"]
    e = CALIBRATOR_SPECTRUM["e"]
    x = np.log10(freq_hz / nu0)
    exponent = a + b * x + c * x * x + d * x**3 + e * x**4
    return I0 * (freq_hz / nu0) ** exponent
