"""Unit tests for scripts/beam_orientation/mueller.py."""

import numpy as np
import pytest
from beam_orientation import mueller  # noqa: E402


@pytest.mark.unit
def test_T_maps_unpolarized_stokes_to_equal_parallel_hands():  # noqa: N802
    T = mueller.linear_to_stokes_matrix()
    S_unpol = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)  # I=1, Q=U=V=0
    V_lin = T @ S_unpol
    # XX = YY = I/2, XY = YX = 0
    np.testing.assert_allclose(V_lin, [0.5, 0.0, 0.0, 0.5], atol=1e-12)


@pytest.mark.unit
def test_T_maps_pure_Q_to_parallel_hand_difference():  # noqa: N802
    T = mueller.linear_to_stokes_matrix()
    S = np.array([0.0, 1.0, 0.0, 0.0], dtype=complex)  # pure Q
    V_lin = T @ S
    # XX = +Q/2, YY = -Q/2, XY = YX = 0
    np.testing.assert_allclose(V_lin, [0.5, 0.0, 0.0, -0.5], atol=1e-12)


@pytest.mark.unit
def test_T_maps_pure_V_to_imaginary_cross_hands():  # noqa: N802
    T = mueller.linear_to_stokes_matrix()
    S = np.array([0.0, 0.0, 0.0, 1.0], dtype=complex)  # pure V
    V_lin = T @ S
    # XY = +i V/2, YX = -i V/2, XX = YY = 0
    np.testing.assert_allclose(V_lin, [0.0, 0.5j, -0.5j, 0.0], atol=1e-12)


@pytest.mark.unit
def test_T_and_inverse_compose_to_identity():  # noqa: N802
    T = mueller.linear_to_stokes_matrix()
    T_inv = mueller.stokes_to_linear_matrix()
    np.testing.assert_allclose(T @ T_inv, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(T_inv @ T, np.eye(4), atol=1e-12)
