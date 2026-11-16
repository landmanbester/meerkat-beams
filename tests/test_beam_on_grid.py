"""beam_on_grid: stored beam maps about the pointing centre onto a grid about the phase centre (#57)."""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_pixel
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates

from meerkat_beams.utils import BeamWizard, beam_on_grid
from tests._synthetic import DEC0, RA0, build_synthetic_bds

pytest.importorskip("numba")

PHASE = SkyCoord(ra=RA0, dec=DEC0, unit="deg")
POINTING = PHASE.directional_offset_by(-35 * u.deg, 0.8 * u.deg)
NB = 48
L_BEAM = np.linspace(-1.5, 1.5, NB)
M_BEAM = np.linspace(-1.2, 1.8, NB)  # deliberately not symmetric about 0
L_OUT = -(np.arange(97) - 48) * 0.03  # descending, like an RA axis with CDELT1 < 0
M_OUT = (np.arange(81) - 40) * 0.03


def _lopsided(nplanes=3, dtype=np.float32):
    """Asymmetric maps, so a transposed or mirrored placement cannot pass."""
    rng = np.random.default_rng(57)
    mm, ll = np.meshgrid(M_BEAM, L_BEAM, indexing="ij")
    base = np.exp(-((ll - 0.3) ** 2) / 0.5 - (mm + 0.2) ** 2 / 0.3) + 0.2 * ll + 0.05 * mm**2
    planes = np.stack([base * (k + 1) + 0.01 * rng.standard_normal(base.shape) for k in range(nplanes)])
    if np.dtype(dtype).kind == "c":
        planes = planes + 1j * planes[::-1] * 0.5
    return planes.astype(dtype)


def _sin_wcs(centre, axis_l, axis_m):
    """A SIN WCS whose pixel (j, i) is (axis_l[j], axis_m[i]) degrees about centre."""
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    w.wcs.cunit = ["deg", "deg"]
    w.wcs.cdelt = [axis_l[1] - axis_l[0], axis_m[1] - axis_m[0]]
    w.wcs.crpix = [1.0 - axis_l[0] / w.wcs.cdelt[0], 1.0 - axis_m[0] / w.wcs.cdelt[1]]
    w.wcs.crval = [centre.icrs.ra.deg, centre.icrs.dec.deg]
    return w


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [np.float32, np.complex64])
def test_matches_an_astropy_reprojection_oracle(dtype):
    """Independent of meerkat-beams' geometry: astropy maps output pixels to beam pixels, scipy interpolates."""
    beam = _lopsided(dtype=dtype)
    got = beam_on_grid(beam, L_BEAM, M_BEAM, POINTING, PHASE, L_OUT, M_OUT)
    assert got.shape == (3, M_OUT.size, L_OUT.size) and got.dtype == dtype

    rows, cols = np.meshgrid(np.arange(M_OUT.size), np.arange(L_OUT.size), indexing="ij")
    px, py = pixel_to_pixel(_sin_wcs(PHASE, L_OUT, M_OUT), _sin_wcs(POINTING, L_BEAM, M_BEAM), cols, rows)
    inside = (px >= 0) & (px <= NB - 1) & (py >= 0) & (py <= NB - 1)
    assert 0.2 < inside.mean() < 0.95  # both covered and uncovered pixels are exercised
    for p in range(3):
        expect = map_coordinates(beam[p].real, [py, px], order=1)
        if np.iscomplexobj(beam):
            expect = expect + 1j * map_coordinates(beam[p].imag, [py, px], order=1)
        np.testing.assert_allclose(got[p][inside], expect[inside], rtol=0, atol=2e-5)
        assert np.all(got[p][~inside] == 0)


@pytest.mark.unit
def test_a_beam_linear_in_the_pointing_frame_is_reproduced_exactly():
    """Bilinear interpolation is exact on a linear field; needs no interpolation library to check."""
    a, b, c = 0.7, -1.3, 0.25
    mm, ll = np.meshgrid(M_BEAM, L_BEAM, indexing="ij")
    beam = (a * ll + b * mm + c)[None].astype(np.float64)
    got = beam_on_grid(beam, L_BEAM, M_BEAM, POINTING, PHASE, L_OUT, M_OUT)[0]

    # the same pixels' pointing-frame coordinates, by the closed-form SIN formulae
    lo, mo = np.meshgrid(np.deg2rad(L_OUT), np.deg2rad(M_OUT))
    a0, d0 = PHASE.ra.rad, PHASE.dec.rad
    d = np.arcsin(mo * np.cos(d0) + np.sqrt(1 - lo**2 - mo**2) * np.sin(d0))
    ra = a0 + np.arctan2(lo, np.sqrt(1 - lo**2 - mo**2) * np.cos(d0) - mo * np.sin(d0))
    ap, dp = POINTING.ra.rad, POINTING.dec.rad
    lp = np.rad2deg(np.cos(d) * np.sin(ra - ap))
    mp = np.rad2deg(np.sin(d) * np.cos(dp) - np.cos(d) * np.sin(dp) * np.cos(ra - ap))
    inside = (lp >= L_BEAM[0]) & (lp <= L_BEAM[-1]) & (mp >= M_BEAM[0]) & (mp <= M_BEAM[-1])
    np.testing.assert_allclose(got[inside], (a * lp + b * mp + c)[inside], rtol=0, atol=1e-9)
    assert np.all(got[~inside] == 0)


@pytest.mark.unit
def test_equal_centres_is_plain_separable_bilinear_interpolation():
    beam = _lopsided(nplanes=1, dtype=np.float64)
    got = beam_on_grid(beam, L_BEAM, M_BEAM, PHASE, PHASE, L_OUT, M_OUT)[0]
    interp = RegularGridInterpolator((M_BEAM, L_BEAM), beam[0], method="linear", bounds_error=False, fill_value=0.0)
    mo, lo = np.meshgrid(M_OUT, L_OUT, indexing="ij")
    np.testing.assert_allclose(got, interp((mo, lo)), rtol=0, atol=1e-12)


@pytest.mark.unit
def test_descending_beam_axis_is_the_same_beam():
    beam = _lopsided()
    asc = beam_on_grid(beam, L_BEAM, M_BEAM, POINTING, PHASE, L_OUT, M_OUT)
    desc = beam_on_grid(beam[..., ::-1], L_BEAM[::-1], M_BEAM, POINTING, PHASE, L_OUT, M_OUT)
    np.testing.assert_allclose(asc, desc, rtol=0, atol=1e-6)


@pytest.mark.unit
def test_leading_dims_out_buffer_and_threads():
    beam = _lopsided(nplanes=6).reshape(2, 3, NB, NB)
    serial = beam_on_grid(beam, L_BEAM, M_BEAM, POINTING, PHASE, L_OUT, M_OUT)
    assert serial.shape == (2, 3, M_OUT.size, L_OUT.size)
    buf = np.full_like(serial, np.nan)
    returned = beam_on_grid(beam, L_BEAM, M_BEAM, POINTING, PHASE, L_OUT, M_OUT, out=buf, nthreads=4)
    assert returned is buf
    assert np.array_equal(buf, serial)  # threads split rows; each pixel is computed identically


@pytest.mark.unit
def test_refuses_what_it_cannot_do_correctly():
    beam = _lopsided()
    call = dict(pointing_centre=POINTING, phase_centre=PHASE, l_out=L_OUT, m_out=M_OUT)
    with pytest.raises(ValueError, match="uniformly spaced"):
        beam_on_grid(beam, np.r_[L_BEAM[:-1], L_BEAM[-1] + 0.5], M_BEAM, **call)
    with pytest.raises(ValueError, match=r"\(Y, X\)"):
        beam_on_grid(beam[..., :-1], L_BEAM, M_BEAM, **call)
    with pytest.raises(ValueError, match="expected"):
        beam_on_grid(beam, L_BEAM, M_BEAM, **call, out=np.empty((3, 2, 2), np.float32))
    with pytest.raises(ValueError, match="complex"):
        out = np.empty((3, M_OUT.size, L_OUT.size), np.float32)
        beam_on_grid(beam.astype(np.complex64), L_BEAM, M_BEAM, **call, out=out)
    with pytest.raises(ValueError, match="nthreads"):
        beam_on_grid(beam, L_BEAM, M_BEAM, **call, nthreads=0)


@pytest.mark.unit
def test_stored_map_placement_tracks_the_exact_wizard_path(tmp_path):
    """A densely stored pointing-centred map, placed with beam_on_grid, agrees with evaluating
    the wizard on the image grid with phase_centre= to within the bilinear error."""
    bw = BeamWizard(str(build_synthetic_bds(tmp_path / "bds.zarr")))
    times = Time("2024-01-01T00:00:00") + np.linspace(0, 2, 5) / 24.0
    kw = dict(times=times, freq=np.array([1.1e9]), time_stepping=1, pixel_stepping=1, verbose=0)
    pointing = PHASE.directional_offset_by(70 * u.deg, 0.2 * u.deg)
    l_out = -(np.arange(41) - 20) * 0.02
    m_out = (np.arange(41) - 20) * 0.02
    exact, _ = bw.get_rotation_averaged_beam(l=l_out, m=m_out, pointing_centre=pointing, phase_centre=PHASE, **kw)
    axis = np.linspace(-0.9, 0.9, 181)  # 0.01 deg, five times the 0.05 deg BDS spacing
    stored, _ = bw.get_rotation_averaged_beam(l=axis, m=axis, pointing_centre=pointing, **kw)
    placed = beam_on_grid(stored, axis, axis, pointing, PHASE, l_out, m_out)
    assert np.abs(placed - exact).max() < 2e-3 * np.abs(exact).max()


@pytest.mark.unit
def test_the_far_hemisphere_gets_no_beam():
    """Seen from the antipode, the phase-centre grid has |l'|, |m'| small but n' < 0: without the
    hemisphere test it would land on the beam grid's centre."""
    antipode = SkyCoord(ra=PHASE.ra + 180 * u.deg, dec=-PHASE.dec)
    beam = _lopsided()
    assert not np.any(beam_on_grid(beam, L_BEAM, M_BEAM, antipode, PHASE, L_OUT, M_OUT))
