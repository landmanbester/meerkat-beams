"""Unit tests for the Jones -> Mueller conversions in utils."""

import numpy as np
import pytest

from meerkat_beams.utils import jones_to_mueller, jones_to_mueller_cross


def _random_jones(rng, nf=2, ny=3, nx=4):
    return (rng.normal(size=(nf, ny, nx, 2, 2)) + 1j * rng.normal(size=(nf, ny, nx, 2, 2))).astype(np.complex64)


@pytest.mark.unit
def test_cross_with_itself_matches_jones_to_mueller():
    rng = np.random.default_rng(0)
    j = _random_jones(rng)
    np.testing.assert_allclose(jones_to_mueller_cross(j, j), jones_to_mueller(j), rtol=1e-5, atol=1e-6)


@pytest.mark.unit
def test_cross_matches_explicit_kronecker():
    """The cross Mueller is kron(J1, conj(J2)) at every pixel."""
    rng = np.random.default_rng(1)
    j1, j2 = _random_jones(rng), _random_jones(rng)
    out = jones_to_mueller_cross(j1, j2)
    for f in range(j1.shape[0]):
        for y in range(j1.shape[1]):
            for x in range(j1.shape[2]):
                expected = np.kron(j1[f, y, x], np.conj(j2[f, y, x]))
                np.testing.assert_allclose(out[f, y, x], expected, rtol=1e-5, atol=1e-6)


@pytest.mark.unit
def test_cross_swap_is_conjugate_under_coherency_permutation():
    """Baseline reversal: conj(M(q, p)) is M(p, q) with the coherency indices swapped.

    The coherency index is ``2 * receptor_1 + receptor_2``, so exchanging the
    two antennas permutes rows and columns by [0, 2, 1, 3]. This is NOT a plain
    conjugate transpose: kron(A, B) and kron(B, A) differ by that permutation.
    """
    rng = np.random.default_rng(2)
    j1, j2 = _random_jones(rng), _random_jones(rng)
    a = jones_to_mueller_cross(j1, j2)
    b = jones_to_mueller_cross(j2, j1)
    perm = [0, 2, 1, 3]
    np.testing.assert_allclose(np.conj(b), a[..., perm, :][..., :, perm], rtol=1e-5, atol=1e-6)


@pytest.mark.unit
def test_cross_rejects_mismatched_shapes():
    rng = np.random.default_rng(3)
    j1 = _random_jones(rng, nx=4)
    j2 = _random_jones(rng, nx=5)
    with pytest.raises(ValueError, match="same shape"):
        jones_to_mueller_cross(j1, j2)
