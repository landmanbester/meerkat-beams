"""
Coherency↔Stokes transform, Mueller assembly, and per-(t,ν) solve
for the beam-orientation validation experiment.
"""

import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

from meerkat_beams.utils import BeamWizard

# Beam-element labels for the BDS stokes_i/stokes_j axes. These index the
# coherency correlations (XX, XY, YX, YY) for the `mueller`/`nmueller` vars and
# the Stokes parameters (I, Q, U, V) for `stokes`/`nstokes` — the BDS reuses the
# same coordinate labels for both. Enumeration order is what fixes the 4×4 axis
# ordering in assemble_mueller, matching the MS correlation order.
STOKES_LABELS = ("I", "Q", "U", "V")


def coherency_to_stokes_matrix() -> np.ndarray:
    """4×4 matrix mapping a coherency vector to Stokes.

    This is ``inv(S)`` for the same Stokes→coherency matrix ``S`` used by
    ``meerkat_beams.core.mdv_beams_to_bds`` to build the Stokes beams, so the
    convention matches the BDS exactly:

        coherency (XX, XY, YX, YY)  ->  Stokes (I, Q, U, V),    I = (XX + YY)/2.
    """
    S = np.array(  # noqa: N806  (S is the conventional symbol)
        [[1, 1, 0, 0], [0, 0, 1, 1j], [0, 0, 1, -1j], [1, -1, 0, 0]],
        dtype=complex,
    )
    return np.linalg.inv(S)


def solve_per_bin(M: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  # noqa: N803
    """Solve M(t,ν) · B(t,ν) = V(t,ν) for every (t, ν) bin.

    Parameters
    ----------
    M : (Nt, Nf, 4, 4) complex
        Per-bin 4×4 Mueller matrices (coherency or Stokes basis).
    V : (Nt, Nf, 4) complex
        Per-bin observed visibility vectors (same basis as M).

    Returns
    -------
    B : (Nt, Nf, 4) complex
        Per-bin solved vectors (same basis as M/V).
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
    var: str = "nmueller",
) -> np.ndarray:
    """Build the per-(t, ν) Mueller tensor for a source.

    Calls ``bw.get_source_coordinates`` once with the supplied ``signs`` /
    ``swap`` knobs, then loops over the 16 (i, j) index pairs calling
    ``bw.interpolate_beam(..., var=var, ...)``.

    ``var`` selects the beam representation:

    - ``"nmueller"`` (default): the complex coherency Mueller. The four indices
      are the correlations (XX, XY, YX, YY) in the BDS label order (I, Q, U, V),
      matching the MS correlation order, so ``M`` can be inverted directly
      against the per-(t, ν) averaged coherency visibility.
    - ``"nstokes"``: the real Stokes Mueller (legacy Stokes-frame path).

    Returns
    -------
    M : (Nt, Nf, 4, 4) complex
        ``M[t, f, i, j]`` is the (i, j) Mueller element at the source's
        beam-frame position at time ``t`` and frequency ``f``.
    """
    xpyp, _, _ = bw.get_source_coordinates(srcpos, times=times, loc=loc, signs=signs, swap=swap)
    freq = np.asarray(freq, dtype=float)
    Nt = xpyp.shape[1]
    Nf = freq.size
    M = np.empty((Nt, Nf, 4, 4), dtype=complex)
    for ii, i in enumerate(STOKES_LABELS):
        for jj, j in enumerate(STOKES_LABELS):
            beam_ij = bw.interpolate_beam(xpyp, freq, var=var, i=i, j=j)
            # interpolate_beam returns (Nf, Nt); transpose to (Nt, Nf).
            M[:, :, ii, jj] = beam_ij.T
    return M
