"""Unit tests for scripts/beam_orientation/mueller.py."""

import astropy.units as u
import numpy as np
import pytest
import xarray
from astropy.coordinates import SkyCoord
from astropy.time import Time
from beam_orientation import mueller  # noqa: E402

from meerkat_beams.utils import BeamWizard
from tests._synthetic import DEC0, DELTA, FREQS, I0, N_XY, RA0, build_synthetic_bds, build_synthetic_image


@pytest.mark.unit
def test_T_maps_unpolarized_stokes_to_equal_parallel_hands():  # noqa: N802
    T = mueller.linear_to_stokes_matrix()
    S_unpol = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)  # I=1, Q=U=V=0
    V_lin = T @ S_unpol
    # XX = YY = I/2, XY = YX = 0
    np.testing.assert_allclose(V_lin, [0.5, 0.0, 0.0, 0.5], atol=1e-12)


@pytest.mark.unit
def test_T_maps_pure_Q_to_parallel_hand_difference():  # noqa: N802
    T = mueller.linear_to_stokes_matrix()
    S = np.array([0.0, 1.0, 0.0, 0.0], dtype=complex)  # pure Q
    V_lin = T @ S
    # XX = +Q/2, YY = -Q/2, XY = YX = 0
    np.testing.assert_allclose(V_lin, [0.5, 0.0, 0.0, -0.5], atol=1e-12)


@pytest.mark.unit
def test_T_maps_pure_V_to_imaginary_cross_hands():  # noqa: N802
    T = mueller.linear_to_stokes_matrix()
    S = np.array([0.0, 0.0, 0.0, 1.0], dtype=complex)  # pure V
    V_lin = T @ S
    # XY = +i V/2, YX = -i V/2, XX = YY = 0
    np.testing.assert_allclose(V_lin, [0.0, 0.5j, -0.5j, 0.0], atol=1e-12)


@pytest.mark.unit
def test_solve_recovers_known_B_from_synthetic_M():  # noqa: N802
    rng = np.random.default_rng(0)
    Nt, Nf = 3, 5  # noqa: N806
    # Random invertible 4×4 Mueller per bin
    M = rng.standard_normal((Nt, Nf, 4, 4)) + 1j * rng.standard_normal((Nt, Nf, 4, 4))  # noqa: N806
    B_true = rng.standard_normal((Nt, Nf, 4)) + 1j * rng.standard_normal((Nt, Nf, 4))  # noqa: N806
    V = np.einsum("tfij,tfj->tfi", M, B_true)  # noqa: N806

    B_rec, cond = mueller.solve_per_bin(M, V)  # noqa: N806

    np.testing.assert_allclose(B_rec, B_true, atol=1e-10)  # noqa: N806
    assert cond.shape == (Nt, Nf)
    assert np.all(cond > 0)


@pytest.mark.unit
def test_solve_flags_ill_conditioned_bins():  # noqa: N802
    # M with a near-singular bin should produce a large condition number.
    M = np.tile(np.eye(4, dtype=complex), (1, 1, 1, 1))  # (1, 1, 4, 4)  # noqa: N806
    M[0, 0, 1, 1] = 1e-15  # near-singular
    V = np.ones((1, 1, 4), dtype=complex)  # noqa: N806

    _, cond = mueller.solve_per_bin(M, V)  # noqa: N806
    assert cond[0, 0] > 1e10


def _build_asymmetric_bds(path):
    """BDS whose nstokes diagonal is a Gaussian modulated by a tilted linear ramp.

    The ramp (1 + 0.1*(x - I0) + 0.05*(y - I0)) breaks the even-function symmetry
    so that sign flips and axis swaps produce measurably different beam values.
    """
    SIGMA = 6.0
    y_idx, x_idx = np.indices((N_XY, N_XY), dtype=np.float64)
    r2 = (x_idx - I0) ** 2 + (y_idx - I0) ** 2
    ramp = 1.0 + 0.1 * (x_idx - I0) + 0.05 * (y_idx - I0)
    plane = np.exp(-0.5 * r2 / SIGMA**2) * ramp
    plane = np.clip(plane, 0, None)  # keep non-negative
    plane = np.broadcast_to(plane, (len(FREQS), N_XY, N_XY)).astype(np.float32).copy()
    zeros = np.zeros_like(plane)

    njones = np.stack([np.stack([plane, zeros], axis=0), np.stack([zeros, plane], axis=0)], axis=0).astype(np.float32)
    nstokes_arr = np.zeros((4, 4, len(FREQS), N_XY, N_XY), dtype=np.float32)
    for s in range(4):
        nstokes_arr[s, s] = plane
    stokes_arr = nstokes_arr.copy()
    jones = njones.copy()

    degs = (np.arange(N_XY) - I0) * DELTA
    fits_header = {
        "SIMPLE": "T",
        "NAXIS1": N_XY,
        "NAXIS2": N_XY,
        "NAXIS3": len(FREQS),
        "CRPIX1": I0 + 1,
        "CRPIX2": I0 + 1,
        "CRPIX3": 1,
        "CRVAL1": 0,
        "CRVAL2": 0,
        "CRVAL3": float(FREQS[0]),
        "CDELT1": DELTA,
        "CDELT2": DELTA,
        "CDELT3": float(FREQS[1] - FREQS[0]),
        "CTYPE1": "X",
        "CTYPE2": "Y",
        "CTYPE3": "FREQ",
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        "CUNIT3": "Hz",
    }
    jcoords = dict(receptor_i=[0, 1], receptor_j=[0, 1], X=degs, Y=degs, FREQ=FREQS)
    scoords = dict(stokes_i=list("IQUV"), stokes_j=list("IQUV"), X=degs, Y=degs, FREQ=FREQS)
    xds = xarray.Dataset(
        {
            "jones": xarray.DataArray(jones, dims=("receptor_i", "receptor_j", "FREQ", "Y", "X"), coords=jcoords),
            "njones": xarray.DataArray(njones, dims=("receptor_i", "receptor_j", "FREQ", "Y", "X"), coords=jcoords),
            "stokes": xarray.DataArray(stokes_arr, dims=("stokes_i", "stokes_j", "FREQ", "Y", "X"), coords=scoords),
            "nstokes": xarray.DataArray(nstokes_arr, dims=("stokes_i", "stokes_j", "FREQ", "Y", "X"), coords=scoords),
        }
    )
    xds.attrs["fits_header"] = fits_header
    xds.attrs.update(x0=I0, y0=I0, dx=DELTA, dy=DELTA, freqs=FREQS)
    xds.to_zarr(str(path), mode="w")
    return path


@pytest.fixture(scope="module")
def synthetic_bw(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mueller")
    build_synthetic_bds(tmp / "synthetic.bds.zarr")
    build_synthetic_image(tmp / "synthetic.fits")
    return BeamWizard(str(tmp / "synthetic.bds.zarr"), str(tmp / "synthetic.fits"))


@pytest.fixture(scope="module")
def asymmetric_bw(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mueller_asym")
    _build_asymmetric_bds(tmp / "asym.bds.zarr")
    build_synthetic_image(tmp / "synthetic.fits")
    return BeamWizard(str(tmp / "asym.bds.zarr"), str(tmp / "synthetic.fits"))


@pytest.fixture
def short_times():
    return Time("2024-01-01T00:00:00") + np.linspace(0, 1, 3) * u.hour


@pytest.mark.unit
def test_assemble_mueller_shape_and_dtype(synthetic_bw, short_times):
    src = SkyCoord(ra=RA0 * u.deg, dec=DEC0 * u.deg)
    M = mueller.assemble_mueller(synthetic_bw, src, short_times, FREQS)
    assert M.shape == (len(short_times), len(FREQS), 4, 4)
    assert M.dtype == complex


@pytest.mark.unit
def test_assemble_mueller_on_axis_is_identity(synthetic_bw, short_times):
    # Source at field centre → on-axis → normalised nstokes is the identity.
    src = SkyCoord(ra=RA0 * u.deg, dec=DEC0 * u.deg)
    M = mueller.assemble_mueller(synthetic_bw, src, short_times, FREQS)
    eye = np.broadcast_to(np.eye(4, dtype=complex), M.shape)
    np.testing.assert_allclose(M, eye, atol=1e-6)


@pytest.mark.unit
def test_assemble_mueller_signs_swap_propagate(asymmetric_bw, short_times):
    # Pick a source offset in BOTH RA and DEC so the ramp-modulated beam returns
    # different values under sign-flip (x → -x) and under axis swap (x↔y).
    src = SkyCoord(ra=(RA0 + 0.3) * u.deg, dec=(DEC0 + 0.2) * u.deg)
    M_default = mueller.assemble_mueller(asymmetric_bw, src, short_times, FREQS)
    M_flipx = mueller.assemble_mueller(asymmetric_bw, src, short_times, FREQS, signs=(-1, 1), swap=False)
    M_swap = mueller.assemble_mueller(asymmetric_bw, src, short_times, FREQS, signs=(1, 1), swap=True)
    assert not np.allclose(M_default, M_flipx)
    assert not np.allclose(M_default, M_swap)
