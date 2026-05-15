"""
Linear↔Stokes transforms, Mueller assembly, and per-(t,ν) solve
for the beam-orientation validation experiment.
"""

import numpy as np


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
    return np.linalg.inv(linear_to_stokes_matrix())
