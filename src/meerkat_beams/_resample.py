"""numba kernel behind ``utils.beam_on_grid``.

Kept in its own module so importing ``meerkat_beams.utils`` does not import
numba (0.15 s, and an LLVM runtime): only a call to ``beam_on_grid`` does. The
kernel is compiled per beam dtype on first use and cached on disk.
"""

import numba
import numpy as np

_DEG = np.pi / 180.0


@numba.njit(parallel=True, cache=True, nogil=True)
def bilinear_on_rotated_grid(beam, l0b, dlb, m0b, dmb, rot, lr, mr, out):  # pragma: no cover - compiled
    """Fill ``out[p, i, j]`` with ``beam[p]`` bilinearly sampled at output pixel ``(i, j)``.

    ``beam`` is ``(P, nyb, nxb)`` on the uniform axes ``l0b + dlb * k`` and
    ``m0b + dmb * k`` (degrees, about the pointing centre). ``lr``/``mr`` are the
    output axes in radians of direction cosine, about the phase centre; ``rot``
    is ``sky_rotation_matrix(pointing, phase)``. Pixels off the sky, on the
    pointing's far hemisphere, or whose fractional beam index leaves
    ``[0, n - 1]`` on either axis get 0. Rows run in parallel.
    """
    nplane, nyb, nxb = beam.shape
    for i in numba.prange(mr.size):
        m = mr[i]
        for j in range(lr.size):
            l = lr[j]
            r2 = l * l + m * m
            inside = r2 < 1.0
            fx = fy = 0.0
            if inside:
                n = np.sqrt(1.0 - r2)
                sz = rot[2, 0] * l + rot[2, 1] * m + rot[2, 2] * n
                fx = ((rot[0, 0] * l + rot[0, 1] * m + rot[0, 2] * n) / _DEG - l0b) / dlb
                fy = ((rot[1, 0] * l + rot[1, 1] * m + rot[1, 2] * n) / _DEG - m0b) / dmb
                inside = sz > 0.0 and fx >= 0.0 and fx <= nxb - 1 and fy >= 0.0 and fy <= nyb - 1
            if not inside:
                for p in range(nplane):
                    out[p, i, j] = 0
                continue
            ix = min(int(fx), nxb - 2)
            iy = min(int(fy), nyb - 2)
            tx = fx - ix
            ty = fy - iy
            w00 = (1.0 - tx) * (1.0 - ty)
            w01 = tx * (1.0 - ty)
            w10 = (1.0 - tx) * ty
            w11 = tx * ty
            for p in range(nplane):
                out[p, i, j] = (
                    w00 * beam[p, iy, ix]
                    + w01 * beam[p, iy, ix + 1]
                    + w10 * beam[p, iy + 1, ix]
                    + w11 * beam[p, iy + 1, ix + 1]
                )
