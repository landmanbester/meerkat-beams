"""Unit tests for katbeam BDS synthesis and the shared Stokes helpers."""

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def test_stokes_basis_matrices_are_inverses():
    from meerkat_beams.utils import COHERENCY_TO_STOKES, STOKES_TO_COHERENCY

    assert STOKES_TO_COHERENCY.shape == (4, 4)
    assert COHERENCY_TO_STOKES.shape == (4, 4)
    np.testing.assert_allclose(COHERENCY_TO_STOKES @ STOKES_TO_COHERENCY, np.eye(4), atol=1e-12)


def test_jones_to_mueller_shape_and_identity():
    from meerkat_beams.utils import jones_to_mueller

    # Identity Jones at every (F, Y, X) point -> identity Mueller.
    jones = np.zeros((2, 3, 4, 2, 2), dtype=np.complex128)
    jones[..., 0, 0] = 1.0
    jones[..., 1, 1] = 1.0

    mueller = jones_to_mueller(jones)

    assert mueller.shape == (2, 3, 4, 4, 4)
    np.testing.assert_allclose(mueller[0, 0, 0], np.eye(4), atol=1e-12)


def test_mueller_to_stokes_of_identity_is_identity():
    from meerkat_beams.utils import mueller_to_stokes

    mueller = np.broadcast_to(np.eye(4, dtype=np.complex128), (2, 3, 4, 4, 4)).copy()

    stokes = mueller_to_stokes(mueller)

    assert stokes.shape == (2, 3, 4, 4, 4)
    np.testing.assert_allclose(stokes[0, 0, 0], np.eye(4), atol=1e-12)


def test_diagonal_jones_gives_the_expected_stokes_beam_matrix():
    """Pin the full Stokes beam matrix for a real diagonal Jones diag(h, v).

    With HH = I+Q and VV = I-Q, a diagonal Jones gives HH_out = h^2(I+Q) and
    VV_out = v^2(I-Q), so:
        I_out = I(h^2+v^2)/2 + Q(h^2-v^2)/2
        Q_out = I(h^2-v^2)/2 + Q(h^2+v^2)/2
    i.e. the gain sits on the diagonal and the I<->Q leakage off it, with U and
    V both scaled by h*v. Nothing leaks into or out of U/V.
    """
    from meerkat_beams.utils import jones_to_mueller, mueller_to_stokes

    h, v = 0.8, 0.6
    jones = np.zeros((1, 1, 1, 2, 2), dtype=np.complex128)
    jones[..., 0, 0] = h
    jones[..., 1, 1] = v

    stokes = mueller_to_stokes(jones_to_mueller(jones)).real

    gain = 0.5 * (h**2 + v**2)
    leak = 0.5 * (h**2 - v**2)
    expected = np.array(
        [
            [gain, leak, 0.0, 0.0],
            [leak, gain, 0.0, 0.0],
            [0.0, 0.0, h * v, 0.0],
            [0.0, 0.0, 0.0, h * v],
        ]
    )
    np.testing.assert_allclose(stokes[0, 0, 0], expected, atol=1e-12)
