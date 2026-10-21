"""
Integration tests: bound the lambda/D frequency-extrapolation error, and the
in-chunk beam variation, on the real L-band beam datasets (issue #41, D19).

Each product skips unless its env var points at a BDS:

- MBEAMS_BDS_PATH          legacy MeerKAT L (1024 channels, 856.000-1711.164 MHz)
- MBEAMS_MK2026_BDS_PATH   MdV-2026 MeerKAT L (63 channels, 869.375-1698.625 MHz)
- MBEAMS_MKE_BDS_PATH      MdV-2026 MeerKAT+ L (same grid)

Metric: RMS error over BDS pixels within 1 deg of the pointing centre, one value
per predicted plane (A) or per chunk (B), summarised by the median. The max over
planes/chunks is dominated by a few outlier channels and the top-of-band chunk,
where every predictor does about equally badly, so it is not pinned. Only the
centre-normalised variables are measured: the ~17 MHz standing-wave ripple is
mostly an on-axis gain ripple that bandpass calibration absorbs and that
normalising by the central Jones divides out.

Bounds are the 2026-10-21 measurement x 1.5, rounded up to 2 significant figures.
"""

import os

import numpy as np
import pytest

from meerkat_beams.utils import BeamWizard

pytestmark = pytest.mark.integration

ENV = {"legacy": "MBEAMS_BDS_PATH", "mk2026": "MBEAMS_MK2026_BDS_PATH", "mke": "MBEAMS_MKE_BDS_PATH"}
RADIUS_DEG = 1.0
BAND_START = 856e6  # L-band receiver edge; chunks tile from here, as an MS's would
ELEMENTS = [("nstokes", "I", "I"), ("nstokes", "I", "Q"), ("nstokes", "I", "V"), ("njones", 0, 0), ("njones", 0, 1)]

# (product, channels extrapolated) -> bound on median RMS of "scale", per ELEMENTS
SCALE_BOUND = {
    ("mk2026", 1): (3.2e-3, 3.3e-3, 1.8e-3, 4.4e-3, 2.6e-3),
    ("mke", 1): (2.0e-3, 2.5e-3, 1.8e-3, 5.4e-3, 4.5e-3),
    ("legacy", 1): (3.7e-4, 2.9e-4, 1.5e-4, 5.0e-4, 3.3e-4),
    ("legacy", 16): (3.2e-3, 2.6e-3, 1.4e-3, 4.7e-3, 2.5e-3),
}
# (product, chunk width MHz) -> bound on median RMS of the lambda/D slope, per ELEMENTS
LAMD_BOUND = {
    ("mk2026", 53.5): (3.5e-3, 3.3e-3, 1.5e-3, 5.0e-3, 2.9e-3),
    ("mke", 53.5): (2.9e-3, 2.6e-3, 1.7e-3, 6.2e-3, 4.7e-3),
    ("legacy", 13.375): (1.6e-3, 1.2e-3, 7.0e-4, 2.0e-3, 1.3e-3),
    ("legacy", 53.5): (4.0e-3, 3.2e-3, 1.5e-3, 5.2e-3, 2.8e-3),
}

_WIZARDS = {}


def _wizard(product):
    path = os.environ.get(ENV[product])
    if not path:
        pytest.skip(f"set {ENV[product]} to run the {product} extrapolation measurements")
    if product not in _WIZARDS:
        _WIZARDS[product] = BeamWizard(path, freq_extrapolation="scale")
    return _WIZARDS[product]


def _grid(bw):
    n_y, n_x = bw.bds.sizes["Y"], bw.bds.sizes["X"]
    x0, y0, dx = bw.bds.attrs["x0"], bw.bds.attrs["y0"], bw.bds.attrs["dx"]
    yy, xx = np.indices((n_y, n_x), dtype=float)
    inside = np.hypot(xx - x0, yy - y0) * dx <= RADIUS_DEG
    return xx, yy, inside, np.array([x0, y0], dtype=float)[:, None]


def _rms(x):
    return float(np.sqrt(np.mean(np.abs(x) ** 2)))


def _margin_errors(bw, d, var, i, j):
    """Median RMS of (scaled, clamped) predictions of plane k from planes k-d and k+d."""
    xx, yy, inside, c = _grid(bw)
    xpyp = np.array([xx[inside], yy[inside]])
    f = bw.bds.coords["FREQ"].values
    scaled, clamped = [], []
    for k in range(d, len(f) - d, 8):
        truth = bw.interpolate_beam(xpyp, [f[k]], var=var, i=i, j=j)[0]
        for src in (k - d, k + d):
            s = f[k] / f[src]
            scaled.append(_rms(bw.interpolate_beam(c + (xpyp - c) * s, [f[src]], var=var, i=i, j=j)[0] - truth))
            clamped.append(_rms(bw.interpolate_beam(xpyp, [f[src]], var=var, i=i, j=j)[0] - truth))
    return np.median(scaled), np.median(clamped)


def _chunk_errors(bw, width, var, i, j):
    """Median RMS of (flat, chord, lambda/D) predictions of the BDS planes inside each chunk.

    The chord runs through the chunk's extreme planes, which are themselves scored (zero error there),
    so on chunks with few planes (4 on the 2026 grids) it is flattered; legacy L (~64 planes) is unbiased."""
    xx, yy, inside, c = _grid(bw)
    xpyp = np.array([xx.ravel(), yy.ravel()])
    f = bw.bds.coords["FREQ"].values
    lo, hi = bw.freq_range
    flat, chord, lamd = [], [], []
    for a in np.arange(BAND_START, f.max() + 1, width):
        nu0 = a + width / 2
        planes = f[(f >= a) & (f < a + width)]
        if len(planes) < 2 or not lo <= nu0 <= hi:
            continue
        m0 = bw.interpolate_beam(xpyp, [nu0], var=var, i=i, j=j)[0].reshape(xx.shape)
        dm_dy, dm_dx = np.gradient(m0)
        slope = ((xx - c[0]) * dm_dx + (yy - c[1]) * dm_dy) / nu0
        truth = bw.interpolate_beam(xpyp, planes, var=var, i=i, j=j).reshape((len(planes),) + xx.shape)
        dnu = (planes - nu0)[:, None, None]
        secant = (truth[-1] - truth[0]) / (planes[-1] - planes[0])
        chord.append(_rms((m0 + dnu * secant - truth)[:, inside]))
        flat.append(_rms((m0 - truth)[:, inside]))
        lamd.append(_rms((m0 + dnu * slope - truth)[:, inside]))
    return np.median(flat), np.median(chord), np.median(lamd)


@pytest.mark.parametrize("element", range(len(ELEMENTS)), ids=[f"{v}[{i},{j}]" for v, i, j in ELEMENTS])
@pytest.mark.parametrize("product, d", list(SCALE_BOUND), ids=[f"{p}-d{d}" for p, d in SCALE_BOUND])
def test_scale_error_bound(product, d, element):
    var, i, j = ELEMENTS[element]
    scaled, clamped = _margin_errors(_wizard(product), d, var, i, j)
    assert scaled <= SCALE_BOUND[(product, d)][element]
    if (var, i, j) == ("nstokes", "I", "I"):
        assert scaled < clamped


@pytest.mark.parametrize("element", range(len(ELEMENTS)), ids=[f"{v}[{i},{j}]" for v, i, j in ELEMENTS])
@pytest.mark.parametrize("product, width", list(LAMD_BOUND), ids=[f"{p}-W{w}" for p, w in LAMD_BOUND])
def test_chunk_slope_error_bound(product, width, element):
    var, i, j = ELEMENTS[element]
    flat, chord, lamd = _chunk_errors(_wizard(product), width * 1e6, var, i, j)
    assert lamd <= LAMD_BOUND[(product, width)][element]
    if (var, i, j) == ("nstokes", "I", "I"):
        assert lamd < flat
        assert chord < lamd  # D19 recommends the chord to pfb-model-spec#39
