"""numba kernels behind ``utils.beam_on_grid``.

Kept in its own module so importing ``meerkat_beams.utils`` does not import
numba (0.15 s, and an LLVM runtime): only a call to ``beam_on_grid`` does. The
kernels are compiled per beam dtype on first use and cached on disk.

There are two entry points over one row function, so there is one source for
the arithmetic:

- ``bilinear_serial``: plain ``njit(nogil=True)``. Safe to call from any number
  of Python threads at once (dask workers, thread pools), whatever numba
  threading layer is installed.
- ``bilinear_parallel``: ``parallel=True`` over output rows. numba's
  ``workqueue`` layer -- what numba falls back to without TBB or OpenMP, e.g.
  on ``python:3.11-slim``, this package's container base -- aborts the process
  if two threads launch a parallel region at once. ``PARALLEL_LOCK`` serialises
  those launches; each still uses its ``nthreads``.
"""

import threading

import numba
import numpy as np

_DEG = np.pi / 180.0

PARALLEL_LOCK = threading.Lock()


@numba.njit(nogil=True, cache=True, inline="always")
def _fill_row(i, beam, l0b, dlb, m0b, dmb, rot, lr, mr, out):  # pragma: no cover - compiled
    """Fill output row ``i``: ``out[p, i, j]`` is ``beam[p]`` bilinearly sampled at pixel ``(i, j)``.

    ``beam`` is ``(P, nyb, nxb)`` on the uniform axes ``l0b + dlb * k`` and
    ``m0b + dmb * k`` (degrees, about the pointing centre). ``lr``/``mr`` are the
    output axes in radians of direction cosine, about the phase centre; ``rot``
    is ``sky_rotation_matrix(pointing, phase)``. Pixels off the sky, on the
    pointing's far hemisphere, or whose fractional beam index leaves
    ``[0, n - 1]`` on either axis get 0.
    """
    nplane, nyb, nxb = beam.shape
    # loop invariants, hoisted by hand: the serial kernel gets no parallel=True passes
    r00, r01, r02 = rot[0, 0], rot[0, 1], rot[0, 2]
    r10, r11, r12 = rot[1, 0], rot[1, 1], rot[1, 2]
    r20, r21, r22 = rot[2, 0], rot[2, 1], rot[2, 2]
    sx, sy = 1.0 / (_DEG * dlb), 1.0 / (_DEG * dmb)  # radians of cosine -> beam pixels
    ox, oy = l0b / dlb, m0b / dmb
    m = mr[i]
    m2 = m * m
    for j in range(lr.size):
        l = lr[j]
        r2 = l * l + m2
        inside = r2 < 1.0
        fx = fy = 0.0
        if inside:
            n = np.sqrt(1.0 - r2)
            sz = r20 * l + r21 * m + r22 * n
            fx = (r00 * l + r01 * m + r02 * n) * sx - ox
            fy = (r10 * l + r11 * m + r12 * n) * sy - oy
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


@numba.njit(nogil=True, cache=True)
def bilinear_serial(beam, l0b, dlb, m0b, dmb, rot, lr, mr, out):  # pragma: no cover - compiled
    """All rows on the calling thread; see ``_fill_row``."""
    for i in range(mr.size):
        _fill_row(i, beam, l0b, dlb, m0b, dmb, rot, lr, mr, out)


@numba.njit(parallel=True, nogil=True, cache=True)
def bilinear_parallel(beam, l0b, dlb, m0b, dmb, rot, lr, mr, out):  # pragma: no cover - compiled
    """Rows in parallel; call under ``PARALLEL_LOCK``. See ``_fill_row``."""
    for i in numba.prange(mr.size):
        _fill_row(i, beam, l0b, dlb, m0b, dmb, rot, lr, mr, out)
