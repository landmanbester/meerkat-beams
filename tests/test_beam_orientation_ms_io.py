"""Unit tests for scripts/beam_orientation/ms_io.py."""

import numpy as np
import pytest
from beam_orientation import ms_io


@pytest.mark.unit
def test_original_pointing_table_values():
    """ORIGINAL_POINTING carries the five fields with the documented coords."""
    expected = {
        0: (5.146178203219011, -1.1119958085589738),
        1: (5.146178203219011, -1.1364304180868943),
        2: (5.146178203219011, -1.0875611990310532),
        3: (5.201372059151767, -1.1119958085589738),
        4: (5.090979983963126, -1.1119958085589738),
    }
    assert ms_io.ORIGINAL_POINTING == expected


@pytest.mark.unit
def test_pointing_from_direction_time_window_average():
    """Rows inside [t0, t1] are selected and their (ra, dec) averaged."""
    ptime = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    # shape (row, npoly=1, 2) with last axis = [ra, dec]
    direction = np.zeros((5, 1, 2))
    direction[:, 0, 0] = [10.0, 20.0, 30.0, 40.0, 50.0]  # ra
    direction[:, 0, 1] = [-1.0, -2.0, -3.0, -4.0, -5.0]  # dec
    ra, dec = ms_io._pointing_from_direction(ptime, direction, 1.0, 3.0)
    assert ra == pytest.approx(30.0)  # mean(20, 30, 40)
    assert dec == pytest.approx(-3.0)  # mean(-2, -3, -4)


@pytest.mark.unit
def test_pointing_from_direction_no_rows_in_window_returns_none():
    """A window that matches no POINTING rows returns None (-> fallback)."""
    ptime = np.array([0.0, 1.0, 2.0])
    direction = np.zeros((3, 1, 2))
    assert ms_io._pointing_from_direction(ptime, direction, 100.0, 200.0) is None


@pytest.mark.unit
def test_pointing_from_direction_accepts_2d_direction():
    """A (row, 2) DIRECTION (no polynomial axis) is handled too."""
    ptime = np.array([5.0, 6.0])
    direction = np.array([[1.5, -0.5], [2.5, -1.5]])  # (row, 2)
    ra, dec = ms_io._pointing_from_direction(ptime, direction, 0.0, 10.0)
    assert ra == pytest.approx(2.0)
    assert dec == pytest.approx(-1.0)
