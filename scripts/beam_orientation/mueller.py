"""
Linear↔Stokes transforms, Mueller assembly, and per-(t,ν) solve
for the beam-orientation validation experiment.
"""

import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

from meerkat_beams.utils import BeamWizard

STOKES_LABELS = ("I", "Q", "U", "V")


def linear_to_stokes_matrix() -> np.ndarray:
    """4×4 complex matrix T mapping Stokes (I,Q,U,V) to linear (XX,XY,YX,YY).

    V_lin = T @ V_S.
    Ordering: rows = (XX, XY, YX, YY); columns = (I, Q, U, V).
    """
    return 0.5 * np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0j],
            [0.0, 0.0, 1.0, -1.0j],
            [1.0, -1.0, 0.0, 0.0],
        ],
        dtype=complex,
    )


def stokes_to_linear_matrix() -> np.ndarray:
    """Inverse of :func:`linear_to_stokes_matrix`."""
    return linear_to_stokes_matrix().conj().T


def solve_per_bin(M: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  # noqa: N803
    """Solve M(t,ν) · B(t,ν) = V(t,ν) for every (t, ν) bin.

    Parameters
    ----------
    M : (Nt, Nf, 4, 4) complex
        Per-bin 4×4 Stokes Mueller matrices.
    V : (Nt, Nf, 4) complex
        Per-bin observed Stokes vectors.

    Returns
    -------
    B : (Nt, Nf, 4) complex
        Per-bin solved Stokes vectors.
    cond : (Nt, Nf) float
        Per-bin 2-norm condition numbers of M (for downstream masking).

    Notes
    -----
    Why a plain inverse and not the quadratic form of eq. (5)?

    The maximum-likelihood solution in ``scratch/Dynamic_Spectra.pdf`` is

        B = W⁻¹ Mˢᵗᵃᶜᵏ† Σ⁻¹ R,     W = Mˢᵗᵃᶜᵏ† Σ⁻¹ Mˢᵗᵃᶜᵏ     (eqs. 5, 6)

    where ``Mˢᵗᵃᶜᵏ`` and ``R`` are *stacked over all baselines*. That full
    quadratic ``W`` is not formed here because it analytically collapses for
    this experiment. Two facts make it collapse:

    1. The geometric phase Kᵦ is removed beforehand by ``phase_rotate``, and a
       single array-average beam is used, so the per-baseline Mueller term is
       the same matrix M for every baseline b (``assemble_mueller`` returns one
       4×4 per (t, ν), not one per baseline).
    2. M is square (4×4) and invertible.

    With Mᵦ = M and scalar per-baseline weights Σᵦ⁻¹ = wᵦ I,

        W                = (Σᵦ wᵦ) M† M
        Mˢᵗᵃᶜᵏ† Σ⁻¹ R    = M† Σᵦ wᵦ Rᵦ
        ⇒ B = (M† M)⁻¹ M† · (Σᵦ wᵦ Rᵦ) / (Σᵦ wᵦ) = M⁻¹ R̄,

    using (M† M)⁻¹ M† = M⁻¹ for square invertible M. The Σ⁻¹-weighted baseline
    average R̄ is computed in the caller (test_beam_orientation.py); this
    function applies the remaining M⁻¹ per bin. ``cond`` reports cond(M); the
    conditioning of the un-collapsed W would be its square, cond(M)².

    This collapse relies on M being baseline-independent. If per-antenna /
    per-baseline beams were ever modelled (Mᵦ differing per baseline), W would
    no longer factor out and the full eq. (5) quadratic would be required.
    """
    B = np.linalg.solve(M, V[..., None])[..., 0]
    cond = np.linalg.cond(M)
    return B, cond


def assemble_mueller(
    bw: BeamWizard,
    srcpos: SkyCoord,
    times: Time,
    freq: np.ndarray,
    loc: EarthLocation | None = None,
    signs: tuple[int, int] = (1, 1),
    swap: bool = False,
) -> np.ndarray:
    """Build the per-(t, ν) Stokes Mueller tensor for a source.

    Calls ``bw.get_source_coordinates`` once with the supplied ``signs`` /
    ``swap`` knobs, then loops over the 16 (i, j) Stokes-index pairs calling
    ``bw.interpolate_beam(..., var='nstokes', ...)``. With the default knobs
    this matches the composition wrapped by
    ``BeamWizard.get_time_variable_beamgain(..., spi=None)``.

    Returns
    -------
    M : (Nt, Nf, 4, 4) complex
        ``M[t, f, i, j]`` is the (i, j) Stokes Mueller element at the
        source's beam-frame position at time ``t`` and frequency ``f``.
    """
    xpyp, _, _ = bw.get_source_coordinates(srcpos, times=times, loc=loc, signs=signs, swap=swap)
    freq = np.asarray(freq, dtype=float)
    Nt = xpyp.shape[1]
    Nf = freq.size
    M = np.empty((Nt, Nf, 4, 4), dtype=complex)
    for ii, i in enumerate(STOKES_LABELS):
        for jj, j in enumerate(STOKES_LABELS):
            beam_ij = bw.interpolate_beam(xpyp, freq, var="nstokes", i=i, j=j)
            # interpolate_beam returns (Nf, Nt); transpose to (Nt, Nf).
            M[:, :, ii, jj] = beam_ij.T
    return M
