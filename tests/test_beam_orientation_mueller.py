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


@pytest.mark.unit
def test_solve_recovers_known_B_from_synthetic_M():  # noqa: N802
    rng = np.random.default_rng(0)
    Nt, Nf = 3, 5  # noqa: N806
    # Random invertible 4×4 Mueller per bin
    M = rng.standard_normal((Nt, Nf, 4, 4)) + 1j * rng.standard_normal((Nt, Nf, 4, 4))  # noqa: N806
    B_true = rng.standard_normal((Nt, Nf, 4)) + 1j * rng.standard_normal((Nt, Nf, 4))  # noqa: N806
    V = np.einsum("tfij,tfj->tfi", M, B_true)  # noqa: N806

    B_rec, cond = mueller.solve_per_bin(M, V)  # noqa: N806

    np.testing.assert_allclose(B_rec, B_true, atol=1e-10)  # noqa: N806
    assert cond.shape == (Nt, Nf)
    assert np.all(cond > 0)


@pytest.mark.unit
def test_solve_flags_ill_conditioned_bins():  # noqa: N802
    # M with a near-singular bin should produce a large condition number.
    M = np.tile(np.eye(4, dtype=complex), (1, 1, 1, 1))  # (1, 1, 4, 4)  # noqa: N806
    M[0, 0, 1, 1] = 1e-15  # near-singular
    V = np.ones((1, 1, 4), dtype=complex)  # noqa: N806

    _, cond = mueller.solve_per_bin(M, V)  # noqa: N806
    assert cond[0, 0] > 1e10
