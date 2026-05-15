"""Unit tests for scripts/beam_orientation/phase_rotate.py."""

import numpy as np
import pytest
from beam_orientation import phase_rotate

C = 299_792_458.0  # m/s


@pytest.mark.unit
def test_phase_rotate_zero_offset_is_identity():
    rng = np.random.default_rng(0)
    Nb, Nt, Nf = 10, 4, 6
    vis = (rng.standard_normal((Nb, Nt, Nf, 4)) + 1j * rng.standard_normal((Nb, Nt, Nf, 4))).astype(complex)
    uvw = rng.standard_normal((Nb, Nt, 3)) * 100.0
    freq = np.linspace(1.0e9, 1.2e9, Nf)

    out = phase_rotate.phase_rotate(vis, uvw, freq, dl=0.0, dm=0.0)
    np.testing.assert_allclose(out, vis, atol=1e-12)


@pytest.mark.unit
def test_phase_rotate_removes_known_offset_phase():
    """A point source at (l0, m0) has visibility V = exp(-2πi (ul + vm + w(n-1))/λ).
    Phase-rotating to (l0, m0) should produce a flat constant visibility."""
    rng = np.random.default_rng(1)
    Nb, Nt, Nf = 50, 3, 4
    uvw = rng.standard_normal((Nb, Nt, 3)) * 200.0
    freq = np.linspace(1.0e9, 1.2e9, Nf)
    l0, m0 = 0.01, -0.005  # radians, small angle
    n0 = np.sqrt(1.0 - l0**2 - m0**2)

    lmbda = C / freq  # (Nf,)
    arg = uvw[..., 0:1] * l0 + uvw[..., 1:2] * m0 + uvw[..., 2:3] * (n0 - 1.0)  # (Nb, Nt, 1)
    phase = -2j * np.pi * arg / lmbda[None, None, :]  # (Nb, Nt, Nf)
    vis = np.broadcast_to(np.exp(phase)[..., None], (Nb, Nt, Nf, 4)).astype(complex).copy()

    out = phase_rotate.phase_rotate(vis, uvw, freq, dl=l0, dm=m0)
    # After rotation every visibility should be 1+0j (within float tolerance).
    np.testing.assert_allclose(out, np.ones_like(out), atol=1e-8)
