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
def test_coherency_to_stokes_matches_inv_S():  # noqa: N802
    """coherency_to_stokes_matrix() is inv(S) for the BDS mdv convention."""
    S = np.array([[1, 1, 0, 0], [0, 0, 1, 1j], [0, 0, 1, -1j], [1, -1, 0, 0]], dtype=complex)
    np.testing.assert_allclose(mueller.coherency_to_stokes_matrix(), np.linalg.inv(S), atol=1e-12)


@pytest.mark.unit
def test_coherency_to_stokes_unpolarized():  # noqa: N802
    """Equal parallel hands, zero cross hands -> Stokes I only."""
    M = mueller.coherency_to_stokes_matrix()
    coh = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex)  # XX=YY=1, XY=YX=0
    np.testing.assert_allclose(M @ coh, [1.0, 0.0, 0.0, 0.0], atol=1e-12)


@pytest.mark.unit
def test_coherency_to_stokes_pure_Q():  # noqa: N802
    """XX = -YY -> pure Q."""
    M = mueller.coherency_to_stokes_matrix()
    coh = np.array([1.0, 0.0, 0.0, -1.0], dtype=complex)
    np.testing.assert_allclose(M @ coh, [0.0, 1.0, 0.0, 0.0], atol=1e-12)


@pytest.mark.unit
def test_coherency_to_stokes_pure_V():  # noqa: N802
    """Imaginary cross hands -> pure V."""
    M = mueller.coherency_to_stokes_matrix()
    coh = np.array([0.0, 1.0j, -1.0j, 0.0], dtype=complex)
    np.testing.assert_allclose(M @ coh, [0.0, 0.0, 0.0, 1.0], atol=1e-12)


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


@pytest.mark.unit
def test_coherency_and_stokes_frames_are_equivalent():  # noqa: N802
    """Solving in the coherency frame then converting to Stokes equals solving
    directly in the Stokes frame, for a consistent Mueller pair M_S = Sinv·M_C·S.

    This pins the refactor: the script may move to the coherency frame without
    changing the recovered Stokes spectrum.
    """
    rng = np.random.default_rng(3)
    coh2s = mueller.coherency_to_stokes_matrix()  # = inv(S)
    S = np.linalg.inv(coh2s)  # noqa: N806
    M_C = rng.standard_normal((2, 3, 4, 4)) + 1j * rng.standard_normal((2, 3, 4, 4))  # noqa: N806
    V = rng.standard_normal((2, 3, 4)) + 1j * rng.standard_normal((2, 3, 4))  # noqa: N806
    M_S = np.einsum("ij,tfjk,kl->tfil", coh2s, M_C, S)  # noqa: N806

    B_coh, _ = mueller.solve_per_bin(M_C, V)  # noqa: N806
    B_stokes_frame = np.einsum("ij,tfj->tfi", coh2s, B_coh)  # noqa: N806
    B_direct, _ = mueller.solve_per_bin(M_S, np.einsum("ij,tfj->tfi", coh2s, V))  # noqa: N806

    np.testing.assert_allclose(B_stokes_frame, B_direct, atol=1e-10)


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

    # Complex coherency Mueller: diagonal = the (ramp-broken) plane, plus a
    # purely-imaginary U<->V cross term so var="nmueller" is exercised on an
    # asymmetric, complex beam.
    x_ramp = np.broadcast_to(((x_idx - I0) / I0).astype(np.float32), plane.shape).copy()
    cross = (0.5j * x_ramp * plane).astype(np.complex64)
    nmueller_arr = np.zeros((4, 4, len(FREQS), N_XY, N_XY), dtype=np.complex64)
    for s in range(4):
        nmueller_arr[s, s] = plane
    nmueller_arr[2, 3] = cross
    nmueller_arr[3, 2] = -cross
    mueller_arr = nmueller_arr.copy()

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
            "mueller": xarray.DataArray(mueller_arr, dims=("stokes_i", "stokes_j", "FREQ", "Y", "X"), coords=scoords),
            "nmueller": xarray.DataArray(nmueller_arr, dims=("stokes_i", "stokes_j", "FREQ", "Y", "X"), coords=scoords),
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
    # Source at field centre → on-axis → normalised nmueller is the identity.
    src = SkyCoord(ra=RA0 * u.deg, dec=DEC0 * u.deg)
    M = mueller.assemble_mueller(synthetic_bw, src, short_times, FREQS)
    eye = np.broadcast_to(np.eye(4, dtype=complex), M.shape)
    np.testing.assert_allclose(M, eye, atol=1e-6)


@pytest.mark.unit
def test_assemble_mueller_nmueller_offaxis_is_complex(synthetic_bw, short_times):
    """var='nmueller' (the default) yields the complex coherency Mueller: the
    off-axis cross-hand (U,V) element carries a nonzero imaginary part."""
    src = SkyCoord(ra=(RA0 + 0.3) * u.deg, dec=(DEC0 + 0.2) * u.deg)
    M = mueller.assemble_mueller(synthetic_bw, src, short_times, FREQS, var="nmueller")
    assert M.dtype == complex
    # (U, V) is index (2, 3); the synthetic nmueller cross term is purely imaginary.
    assert np.any(np.abs(M[:, :, 2, 3].imag) > 1e-6)


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
