"""
Phase-rotate visibilities to a new phase centre.

Convention being validated:
    V'(u, v, w; ν) = V(u, v, w; ν) * exp(+2πi (u·Δl + v·Δm + w·(Δn - 1)) / λ),
where (Δl, Δm) is the direction-cosine offset of the new phase centre from
the original, and Δn = sqrt(1 − Δl² − Δm²).

The sign of the exponent and of the w-term are listed in the spec as
candidate convention knobs (Section 6) — if the parallactic-angle
controls don't isolate an orientation issue, this sign is the next
fallback to flip.
"""

import numpy as np

C = 299_792_458.0  # m/s


def phase_rotate(
    vis: np.ndarray,
    uvw: np.ndarray,
    freq: np.ndarray,
    dl: float,
    dm: float,
) -> np.ndarray:
    """Phase-rotate visibilities to a new phase centre at (dl, dm).

    Parameters
    ----------
    vis : (Nb, Nt, Nf, Ncorr) complex
        Input visibilities at the original phase centre.
    uvw : (Nb, Nt, 3) float, metres
        Baseline coordinates for each (baseline, time) sample.
    freq : (Nf,) float, Hz
        Channel frequencies.
    dl, dm : float, radians
        Direction-cosine offset of the new phase centre from the original.

    Returns
    -------
    vis_rot : (Nb, Nt, Nf, Ncorr) complex
    """
    dn = np.sqrt(1.0 - dl * dl - dm * dm)
    lmbda = C / np.asarray(freq, dtype=float)  # (Nf,)
    arg = uvw[..., 0:1] * dl + uvw[..., 1:2] * dm + uvw[..., 2:3] * (dn - 1.0)  # (Nb, Nt, 1)
    phase = 2j * np.pi * arg / lmbda[None, None, :]  # (Nb, Nt, Nf)
    return vis * np.exp(phase)[..., None]
