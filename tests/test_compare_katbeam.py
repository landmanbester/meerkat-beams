"""
Unit tests for scripts/compare_katbeam.py.

Hermetic: pure helpers only. No katbeam import at module scope, no BDS on
disk, no network, no env vars. Runs under MBEAMS_OFFLINE=1.
"""

import json
import logging

import compare_katbeam as ck
import numpy as np
import pytest
import xarray

from meerkat_beams.cache import SUPPORTED_BANDS
from tests._synthetic import I0, N_XY, build_synthetic_bds


@pytest.mark.unit
def test_katbeam_model_covers_every_supported_band():
    """Every band the cache can produce must map to a katbeam model."""
    missing = [b for b in SUPPORTED_BANDS if b not in ck.KATBEAM_MODEL_FOR_BAND]
    assert not missing, f"bands with no katbeam model: {missing}"


@pytest.mark.unit
def test_katbeam_s_subbands_share_one_model():
    """MdV splits S into sub-bands; katbeam has a single S table."""
    s_models = {b: m for b, m in ck.KATBEAM_MODEL_FOR_BAND.items() if b.startswith("S")}
    assert len(s_models) >= 2
    assert len(set(s_models.values())) == 1


@pytest.mark.unit
def test_katbeam_model_names_are_known_to_katbeam():
    """Guard against typos, and against a katbeam too old to have a model.

    PyPI only ever released katbeam 0.1, which has no S-band model at all;
    JimBeam would fall through to treating the name as a filename and die in
    np.loadtxt. The dev/test groups pin git main for this reason.
    """
    pytest.importorskip("katbeam")
    for name in ck.KATBEAM_MODEL_FOR_BAND.values():
        ck.require_model(name)


@pytest.mark.unit
def test_require_model_error_names_the_installed_models():
    """The failure mode is an outdated katbeam, so the error must be actionable."""
    pytest.importorskip("katbeam")
    with pytest.raises(ValueError, match="not available in the installed katbeam"):
        ck.require_model("MKAT-AA-NOSUCH-JIM-2020")


@pytest.mark.unit
def test_katbeam_freq_table_is_ascending_and_in_mhz():
    pytest.importorskip("katbeam")
    table = ck.katbeam_freq_table("MKAT-AA-L-JIM-2020")
    assert table.ndim == 1
    assert np.all(np.diff(table) > 0)
    # L band, so hundreds-to-low-thousands of MHz rather than Hz.
    assert 500.0 < table[0] < 2000.0


# ---------------------------------------------------------------------------
# apply_orientation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_orientation_names_are_exactly_the_sweep_set():
    assert set(ck.ORIENTATIONS) == {"none", "flip_x", "flip_y", "swap_xy"}


@pytest.mark.unit
def test_orientation_none_is_identity():
    arr = np.arange(12.0).reshape(3, 4)
    np.testing.assert_array_equal(ck.apply_orientation(arr, "none"), arr)


@pytest.mark.unit
def test_flip_x_reverses_last_axis_and_flip_y_the_second_last():
    """Pins which named perturbation touches which axis.

    The entire orientation sweep is interpreted through these names: if
    flip_x silently started reversing Y, the sweep would report the opposite
    conclusion about the BDS axis convention. Axis -1 is X/l, axis -2 is Y/m.
    """
    arr = np.arange(12.0).reshape(3, 4)
    np.testing.assert_array_equal(ck.apply_orientation(arr, "flip_x"), arr[:, ::-1])
    np.testing.assert_array_equal(ck.apply_orientation(arr, "flip_y"), arr[::-1, :])


@pytest.mark.unit
@pytest.mark.parametrize("name", ["flip_x", "flip_y", "swap_xy"])
def test_orientations_are_involutions(name):
    arr = np.arange(16.0).reshape(4, 4)  # square so swap_xy round-trips
    once = ck.apply_orientation(arr, name)
    twice = ck.apply_orientation(once, name)
    np.testing.assert_array_equal(twice, arr)


@pytest.mark.unit
def test_orientation_applies_to_trailing_axes_of_a_stack():
    """Sweeps run on (NFREQ, NY, NX) stacks, so only trailing axes may move."""
    stack = np.arange(2 * 3 * 4.0).reshape(2, 3, 4)
    out = ck.apply_orientation(stack, "flip_x")
    assert out.shape == stack.shape
    for k in range(stack.shape[0]):
        np.testing.assert_array_equal(out[k], stack[k][:, ::-1])


@pytest.mark.unit
def test_swap_xy_transposes_trailing_axes_of_a_stack():
    stack = np.arange(2 * 3 * 4.0).reshape(2, 3, 4)
    out = ck.apply_orientation(stack, "swap_xy")
    assert out.shape == (2, 4, 3)
    for k in range(stack.shape[0]):
        np.testing.assert_array_equal(out[k], stack[k].T)


@pytest.mark.unit
def test_apply_orientation_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown orientation"):
        ck.apply_orientation(np.zeros((2, 2)), "rotate_90")


# ---------------------------------------------------------------------------
# measure_fwhm
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_measure_fwhm_recovers_gaussian_analytic_value():
    sigma = 0.4
    coord = np.linspace(-4.0, 4.0, 4001)
    profile = np.exp(-0.5 * (coord / sigma) ** 2)
    expected = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
    assert ck.measure_fwhm(coord, profile) == pytest.approx(expected, rel=1e-4)


@pytest.mark.unit
def test_measure_fwhm_handles_offset_peak():
    """Squint shifts the peak off centre; the width must be unaffected."""
    sigma = 0.3
    coord = np.linspace(-4.0, 4.0, 4001)
    profile = np.exp(-0.5 * ((coord - 0.7) / sigma) ** 2)
    expected = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
    assert ck.measure_fwhm(coord, profile) == pytest.approx(expected, rel=1e-4)


@pytest.mark.unit
def test_measure_fwhm_returns_nan_when_half_power_not_bracketed():
    """A monotonic ramp never crosses half power on both sides."""
    coord = np.linspace(0.0, 1.0, 11)
    assert np.isnan(ck.measure_fwhm(coord, coord))


@pytest.mark.unit
def test_measure_fwhm_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        ck.measure_fwhm(np.zeros(5), np.zeros(6))


# ---------------------------------------------------------------------------
# azimuthal_profile
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_azimuthal_profile_of_a_constant_map_has_zero_scatter():
    """Pins the binning mechanics: a constant map has no scatter of any kind.

    Note that a radially symmetric *Gaussian* would NOT give zero scatter here.
    Each annulus has finite width, so the radial gradient shows up as in-bin
    spread. Only a constant map isolates the mechanics.
    """
    coord = np.linspace(-2.0, 2.0, 81)
    r, mean, std, count = ck.azimuthal_profile(np.full((81, 81), 3.0), coord, coord, nbins=20)
    ok = count > 0
    np.testing.assert_allclose(mean[ok], 3.0)
    np.testing.assert_allclose(std[ok], 0.0, atol=1e-12)
    assert np.all(np.diff(r) > 0)


@pytest.mark.unit
def test_azimuthal_profile_captures_pure_azimuthal_variation():
    """The scatter band is the whole point of the radial plot.

    It must measure the azimuthal structure a radially symmetric model cannot
    represent. A map varying only with azimuth -- no radial dependence at all --
    must therefore give a flat mean with large scatter.
    """
    coord = np.linspace(-2.0, 2.0, 81)
    ll, mm = np.meshgrid(coord, coord)
    arr = 1.0 + 0.5 * np.cos(2.0 * np.arctan2(mm, ll))
    r, mean, std, count = ck.azimuthal_profile(arr, coord, coord, nbins=20)
    # Only annuli inscribed in the sampled square see a full circle of azimuth;
    # outer bins sample the corners only. Bin 0 holds too few pixels to average.
    inside = (count > 0) & (r > 0.25) & (r < 1.9)
    assert inside.sum() >= 5
    np.testing.assert_allclose(mean[inside], 1.0, atol=0.1)
    assert np.all(std[inside] > 0.2)


@pytest.mark.unit
def test_azimuthal_profile_ignores_non_finite_samples():
    """katbeam's removable singularity can inject NaN; it must not poison bins."""
    coord = np.linspace(-2.0, 2.0, 41)
    arr = np.ones((41, 41))
    arr[20, 20] = np.nan
    _, mean, _, count = ck.azimuthal_profile(arr, coord, coord, nbins=10)
    assert np.all(np.isfinite(mean[count > 0]))
    np.testing.assert_allclose(mean[count > 0], 1.0)


# ---------------------------------------------------------------------------
# region_masks
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_region_masks_are_disjoint_and_cover_the_field():
    coord = np.linspace(-4.0, 4.0, 65)
    masks = ck.region_masks(coord, coord, hwhm_deg=0.5)
    assert set(masks) == {"mainlobe", "near", "far"}
    stacked = np.stack([masks[k] for k in ("mainlobe", "near", "far")])
    np.testing.assert_array_equal(stacked.sum(axis=0), np.ones((65, 65), dtype=int))


@pytest.mark.unit
def test_region_masks_split_at_hwhm_and_three_hwhm():
    coord = np.linspace(-4.0, 4.0, 65)
    masks = ck.region_masks(coord, coord, hwhm_deg=1.0)
    centre = 32  # coord[32] == 0.0
    assert masks["mainlobe"][centre, centre]
    # (l, m) = (2.0, 0.0) -> r = 2.0, inside [1.0, 3.0)
    j = int(np.argmin(np.abs(coord - 2.0)))
    assert masks["near"][centre, j]
    # (l, m) = (4.0, 0.0) -> r = 4.0, beyond 3*hwhm
    assert masks["far"][centre, -1]


# ---------------------------------------------------------------------------
# frequency selection
# ---------------------------------------------------------------------------

# Stand-in for katbeam's L-band table bounds (856-1712 MHz).
_MODEL_MHZ = np.array([856.0, 1200.0, 1712.0])


@pytest.mark.unit
def test_overlap_indices_excludes_channels_outside_the_model_table():
    bds = np.array([700.0, 900.0, 1200.0, 1700.0, 1900.0]) * 1e6
    idx = ck.overlap_indices(bds, _MODEL_MHZ)
    np.testing.assert_array_equal(idx, [1, 2, 3])


@pytest.mark.unit
def test_overlap_indices_applies_stride():
    bds = np.linspace(900.0, 1700.0, 41) * 1e6
    idx = ck.overlap_indices(bds, _MODEL_MHZ, stride=10)
    np.testing.assert_array_equal(idx, [0, 10, 20, 30, 40])


@pytest.mark.unit
def test_overlap_indices_raises_when_bands_do_not_overlap():
    bds = np.array([2500.0, 2600.0]) * 1e6
    with pytest.raises(ValueError, match="no frequency overlap"):
        ck.overlap_indices(bds, _MODEL_MHZ)


@pytest.mark.unit
def test_select_native_freqs_returns_native_channel_values():
    """Requested frequencies snap to real channels: no frequency interpolation."""
    bds = np.linspace(900.0, 1700.0, 41) * 1e6
    idx, freqs = ck.select_native_freqs(bds, _MODEL_MHZ, requested_mhz=[1000.0, 1500.0])
    assert np.all(np.isin(freqs, bds))
    np.testing.assert_array_equal(bds[idx], freqs)


@pytest.mark.unit
def test_select_native_freqs_clips_requests_to_the_overlap():
    """katbeam's np.interp would silently extrapolate flat outside its table."""
    bds = np.linspace(700.0, 1900.0, 121) * 1e6
    _, freqs = ck.select_native_freqs(bds, _MODEL_MHZ, requested_mhz=[100.0, 5000.0])
    assert freqs.min() >= 856.0e6
    assert freqs.max() <= 1712.0e6


@pytest.mark.unit
def test_select_native_freqs_spreads_n_across_overlap_by_default():
    bds = np.linspace(900.0, 1700.0, 401) * 1e6
    idx, freqs = ck.select_native_freqs(bds, _MODEL_MHZ, n=5)
    assert idx.size == 5
    assert np.all(np.diff(freqs) > 0)
    assert freqs[0] == pytest.approx(900.0e6, abs=5e6)
    assert freqs[-1] == pytest.approx(1700.0e6, abs=5e6)


@pytest.mark.unit
def test_select_native_freqs_deduplicates_collapsed_requests():
    """Two requests landing on one channel must not yield a duplicate entry."""
    bds = np.linspace(900.0, 1700.0, 9) * 1e6  # 100 MHz spacing
    idx, freqs = ck.select_native_freqs(bds, _MODEL_MHZ, requested_mhz=[1000.0, 1001.0])
    assert idx.size == 1
    assert freqs[0] == pytest.approx(1000.0e6)


# ---------------------------------------------------------------------------
# residual_stats
# ---------------------------------------------------------------------------


@pytest.fixture
def flat_masks():
    """Three disjoint masks over a 4x4 field, for residual_stats tests."""
    mainlobe = np.zeros((4, 4), dtype=bool)
    mainlobe[1:3, 1:3] = True
    far = np.zeros((4, 4), dtype=bool)
    far[0, :] = True
    near = ~(mainlobe | far)
    return {"mainlobe": mainlobe, "near": near, "far": far}


@pytest.mark.unit
def test_residual_stats_reports_zero_for_identical_maps(flat_masks):
    arr = np.linspace(0.2, 1.0, 16).reshape(4, 4)
    stats = ck.residual_stats(arr, arr, flat_masks)
    assert set(stats) == {"mainlobe", "near", "far"}
    for region in stats.values():
        assert region["max_abs_diff"] == pytest.approx(0.0)
        assert region["rms_diff"] == pytest.approx(0.0)
        assert region["rms_diff_peaknorm"] == pytest.approx(0.0)


@pytest.mark.unit
def test_residual_stats_computes_rms_and_max_per_region(flat_masks):
    theirs = np.ones((4, 4))
    ours = np.ones((4, 4))
    ours[1, 1] = 1.5  # inside mainlobe only
    stats = ck.residual_stats(ours, theirs, flat_masks)
    assert stats["mainlobe"]["max_abs_diff"] == pytest.approx(0.5)
    assert stats["mainlobe"]["rms_diff"] == pytest.approx(np.sqrt(0.25 / 4))
    assert stats["near"]["max_abs_diff"] == pytest.approx(0.0)
    assert stats["far"]["max_abs_diff"] == pytest.approx(0.0)


@pytest.mark.unit
def test_residual_stats_frac_gate_excludes_null_pixels(flat_masks):
    """Fractional differences must not be taken through katbeam's nulls."""
    theirs = np.full((4, 4), 1.0)
    theirs[1, 1] = 1e-6  # a null: dividing by this is meaningless
    ours = np.full((4, 4), 1.1)
    stats = ck.residual_stats(ours, theirs, flat_masks, frac_floor=0.1)
    assert stats["mainlobe"]["n_pixels"] == 4
    assert stats["mainlobe"]["n_pixels_frac"] == 3
    assert stats["mainlobe"]["median_frac_diff"] == pytest.approx(0.1)


@pytest.mark.unit
def test_residual_stats_peaknorm_removes_a_pure_scale_offset(flat_masks):
    """njones is exactly 1 on axis; katbeam is ~0.9999 because of squint.

    That constant scale offset must not masquerade as a shape difference.
    """
    theirs = np.linspace(0.2, 1.0, 16).reshape(4, 4)
    ours = 0.9 * theirs
    stats = ck.residual_stats(ours, theirs, flat_masks)
    assert stats["mainlobe"]["rms_diff"] > 1e-3
    assert stats["mainlobe"]["rms_diff_peaknorm"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.unit
def test_residual_stats_ignores_non_finite_pixels(flat_masks):
    theirs = np.ones((4, 4))
    ours = np.ones((4, 4))
    ours[1, 1] = np.nan
    stats = ck.residual_stats(ours, theirs, flat_masks)
    assert stats["mainlobe"]["n_pixels"] == 3
    assert stats["mainlobe"]["rms_diff"] == pytest.approx(0.0)


@pytest.mark.unit
def test_residual_stats_returns_nan_for_an_empty_region():
    masks = {"empty": np.zeros((4, 4), dtype=bool)}
    stats = ck.residual_stats(np.ones((4, 4)), np.ones((4, 4)), masks)
    assert stats["empty"]["n_pixels"] == 0
    assert np.isnan(stats["empty"]["rms_diff"])


# ---------------------------------------------------------------------------
# load_ours (hermetic: synthetic BDS in tmp_path)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_bds(tmp_path_factory):
    """Synthetic BDS whose njones diagonal is a known Gaussian and off-diagonal 0."""
    path = tmp_path_factory.mktemp("ck") / "synthetic.bds.zarr"
    build_synthetic_bds(path)
    return xarray.open_zarr(str(path))


@pytest.mark.unit
def test_load_ours_returns_all_four_products(synthetic_bds):
    got = ck.load_ours(synthetic_bds, [0, 2])
    assert set(got) == set(ck.PRODUCTS)
    for name, arr in got.items():
        assert arr.shape == (2, N_XY, N_XY), name


@pytest.mark.unit
def test_load_ours_squares_the_jones_terms(synthetic_bds):
    """HH/VV must be |njones|**2 (power), not the voltage Jones itself."""
    got = ck.load_ours(synthetic_bds, [0])
    stokes_i = got["I"][0]
    # The synthetic fixture sets njones[0,0] = njones[1,1] = nstokes[I,I] = gauss,
    # so the power beams are exactly the square of the Stokes I map.
    np.testing.assert_allclose(got["HH"][0], stokes_i**2, rtol=1e-6)
    np.testing.assert_allclose(got["VV"][0], stokes_i**2, rtol=1e-6)


@pytest.mark.unit
def test_load_ours_crosspol_is_zero_for_a_diagonal_jones(synthetic_bds):
    got = ck.load_ours(synthetic_bds, [0])
    np.testing.assert_allclose(got["xpol"][0], 0.0, atol=1e-12)


@pytest.mark.unit
def test_load_ours_is_unity_on_axis(synthetic_bds):
    """njones is normalised so the on-axis Jones matrix is the identity."""
    got = ck.load_ours(synthetic_bds, [0])
    assert got["I"][0][I0, I0] == pytest.approx(1.0, abs=1e-6)
    assert got["HH"][0][I0, I0] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.unit
def test_load_ours_selects_the_requested_channels(synthetic_bds):
    got = ck.load_ours(synthetic_bds, [1, 3])
    assert got["I"].shape[0] == 2
    direct = synthetic_bds["nstokes"].isel(FREQ=[1, 3]).sel(stokes_i="I", stokes_j="I").values
    np.testing.assert_allclose(got["I"], direct, rtol=1e-6)


# ---------------------------------------------------------------------------
# eval_katbeam (needs katbeam; skipped if absent)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_eval_katbeam_returns_ny_nx_for_non_square_grids():
    """Pins the (NY, NX) convention: axis -2 is m, axis -1 is l.

    A square grid would hide a transposed evaluation, and a transposed katbeam
    map would invert the orientation sweep's conclusion.
    """
    pytest.importorskip("katbeam")
    l_deg = np.linspace(-1.0, 1.0, 5)
    m_deg = np.linspace(-1.0, 1.0, 7)
    got = ck.eval_katbeam("MKAT-AA-L-JIM-2020", l_deg, m_deg, [1284.0])
    assert got["I"].shape == (1, 7, 5)


@pytest.mark.unit
def test_eval_katbeam_stokes_i_is_the_mean_of_the_power_beams():
    pytest.importorskip("katbeam")
    coord = np.linspace(-1.0, 1.0, 21)
    got = ck.eval_katbeam("MKAT-AA-L-JIM-2020", coord, coord, [1284.0])
    np.testing.assert_allclose(got["I"], 0.5 * (got["HH"] + got["VV"]), rtol=1e-10)


@pytest.mark.unit
def test_eval_katbeam_crosspol_is_identically_zero():
    """katbeam models no cross-pol at all; the product exists so plots can say so."""
    pytest.importorskip("katbeam")
    coord = np.linspace(-1.0, 1.0, 11)
    got = ck.eval_katbeam("MKAT-AA-L-JIM-2020", coord, coord, [1284.0])
    assert set(got) == set(ck.PRODUCTS)
    np.testing.assert_array_equal(got["xpol"], np.zeros_like(got["I"]))


@pytest.mark.unit
def test_eval_katbeam_is_near_unity_on_axis():
    """On axis the taper is 1 minus a small squint offset."""
    pytest.importorskip("katbeam")
    coord = np.array([0.0])
    got = ck.eval_katbeam("MKAT-AA-L-JIM-2020", coord, coord, [1284.0])
    assert got["I"][0, 0, 0] == pytest.approx(1.0, abs=5e-3)
    assert got["I"][0, 0, 0] <= 1.0


@pytest.mark.unit
def test_eval_katbeam_hh_and_vv_differ_somewhere():
    """The HH/VV anisotropy is what makes the orientation sweep discriminating.

    If these were identical the sweep would be degenerate and the whole
    experiment would be uninformative.
    """
    pytest.importorskip("katbeam")
    coord = np.linspace(-1.5, 1.5, 61)
    got = ck.eval_katbeam("MKAT-AA-L-JIM-2020", coord, coord, [1400.0])
    assert np.max(np.abs(got["HH"] - got["VV"])) > 1e-3


@pytest.mark.unit
def test_eval_katbeam_is_anisotropic_between_l_and_m():
    """The L-band table gives Hx fwhm 59.58' vs Hy fwhm 62.70' at 1400 MHz.

    So a cut along l must be measurably narrower than a cut along m. This is
    the signal the orientation sweep exploits.
    """
    pytest.importorskip("katbeam")
    coord = np.linspace(-2.0, 2.0, 801)
    got = ck.eval_katbeam("MKAT-AA-L-JIM-2020", coord, coord, [1400.0])
    plane = got["HH"][0]
    mid = coord.size // 2
    fwhm_l = ck.measure_fwhm(coord, plane[mid, :])
    fwhm_m = ck.measure_fwhm(coord, plane[:, mid])
    assert np.isfinite(fwhm_l) and np.isfinite(fwhm_m)
    assert abs(fwhm_l - fwhm_m) > 0.01  # degrees


# ---------------------------------------------------------------------------
# count_non_finite
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_count_non_finite_counts_nan_and_inf_per_product():
    """katbeam's cos(pi*rr)/(1-4rr**2) is 0/0 at rr=0.5; NaN must be reported."""
    maps = {
        "I": np.array([[1.0, np.nan], [np.inf, 2.0]]),
        "HH": np.ones((2, 2)),
    }
    assert ck.count_non_finite(maps) == {"I": 2, "HH": 0}


# ---------------------------------------------------------------------------
# resolve_bds
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_resolve_bds_requires_exactly_one_source():
    with pytest.raises(ValueError, match="exactly one"):
        ck.resolve_bds(None, None)
    with pytest.raises(ValueError, match="exactly one"):
        ck.resolve_bds("L", "/some/path.bds")


@pytest.mark.unit
def test_resolve_bds_rejects_unknown_band():
    with pytest.raises(ValueError, match="no katbeam model"):
        ck.resolve_bds("X9", None)


@pytest.mark.unit
def test_resolve_bds_explicit_path_warns_about_stale_conventions(tmp_path, caplog):
    """--bds may point at a legacy BDS whose axis labelling predates 616906b.

    The orientation sweep would then be confidently wrong, so the escape hatch
    must warn loudly.
    """
    path = tmp_path / "legacy.bds.zarr"
    path.mkdir()
    with caplog.at_level(logging.WARNING):
        assert ck.resolve_bds(None, str(path)) == str(path)
    assert "orientation sweep" in caplog.text.lower()


# ---------------------------------------------------------------------------
# beam_hwhm
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_beam_hwhm_recovers_half_the_gaussian_fwhm():
    sigma = 0.35
    coord = np.linspace(-2.0, 2.0, 401)
    ll, mm = np.meshgrid(coord, coord)
    plane = np.exp(-0.5 * (ll**2 + mm**2) / sigma**2)
    centre = coord.size // 2
    expected = 0.5 * 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
    assert ck.beam_hwhm(plane, coord, coord, centre, centre) == pytest.approx(expected, rel=1e-3)


@pytest.mark.unit
def test_beam_hwhm_averages_the_two_axes_for_an_elliptical_beam():
    coord = np.linspace(-2.0, 2.0, 801)
    ll, mm = np.meshgrid(coord, coord)
    plane = np.exp(-0.5 * ((ll / 0.3) ** 2 + (mm / 0.6) ** 2))
    centre = coord.size // 2
    k = 2.0 * np.sqrt(2.0 * np.log(2.0))
    expected = 0.5 * 0.5 * (k * 0.3 + k * 0.6)
    assert ck.beam_hwhm(plane, coord, coord, centre, centre) == pytest.approx(expected, rel=1e-3)


# ---------------------------------------------------------------------------
# orientation_sweep
# ---------------------------------------------------------------------------


@pytest.fixture
def sweep_grid():
    """Square grid for sweep tests; swap_xy needs matching axis lengths."""
    return np.linspace(-1.0, 1.0, 41)


@pytest.mark.unit
def test_orientation_sweep_picks_none_when_maps_already_agree(sweep_grid):
    coord = sweep_grid
    ll, mm = np.meshgrid(coord, coord)
    # Deliberately asymmetric in BOTH axes so every perturbation is a real change.
    plane = np.exp(-0.5 * ((ll - 0.15) ** 2 / 0.3**2 + (mm - 0.35) ** 2 / 0.6**2))
    ours = {"HH": plane[None], "VV": plane[None], "I": plane[None]}
    theirs = {"HH": plane[None], "VV": plane[None], "I": plane[None]}
    sweep = ck.orientation_sweep(ours, theirs, coord, coord, hwhm_deg=0.5)
    assert sweep["best"]["HH"] == "none"
    assert sweep["best_overall"] == "none"
    assert sweep["per_product"]["HH"]["none"] == pytest.approx(0.0)


@pytest.mark.unit
def test_orientation_sweep_detects_a_transposed_map(sweep_grid):
    """If our map were transposed relative to katbeam, swap_xy must win."""
    coord = sweep_grid
    ll, mm = np.meshgrid(coord, coord)
    plane = np.exp(-0.5 * ((ll / 0.3) ** 2 + (mm / 0.6) ** 2))
    theirs = {"HH": plane[None], "VV": plane[None], "I": plane[None]}
    swapped = plane.T[None]
    ours = {"HH": swapped, "VV": swapped, "I": swapped}
    sweep = ck.orientation_sweep(ours, theirs, coord, coord, hwhm_deg=0.5)
    assert sweep["best"]["HH"] == "swap_xy"
    assert sweep["best_overall"] == "swap_xy"


@pytest.mark.unit
def test_orientation_sweep_detects_a_y_flipped_map(sweep_grid):
    coord = sweep_grid
    ll, mm = np.meshgrid(coord, coord)
    plane = np.exp(-0.5 * ((ll / 0.4) ** 2 + (mm - 0.4) ** 2 / 0.3**2))
    theirs = {"HH": plane[None], "VV": plane[None], "I": plane[None]}
    flipped = plane[::-1, :][None]
    ours = {"HH": flipped, "VV": flipped, "I": flipped}
    sweep = ck.orientation_sweep(ours, theirs, coord, coord, hwhm_deg=0.5)
    assert sweep["best"]["HH"] == "flip_y"


@pytest.mark.unit
def test_orientation_sweep_reports_every_orientation_for_every_product(sweep_grid):
    coord = sweep_grid
    plane = np.ones((coord.size, coord.size))
    ours = {p: plane[None] for p in ("HH", "VV", "I")}
    sweep = ck.orientation_sweep(ours, dict(ours), coord, coord, hwhm_deg=0.5)
    assert set(sweep["per_product"]) == {"HH", "VV", "I"}
    for scores in sweep["per_product"].values():
        assert set(scores) == set(ck.ORIENTATIONS)


@pytest.mark.unit
def test_orientation_sweep_rejects_a_non_square_grid():
    """swap_xy cannot be scored against a rectangular field; fail loudly."""
    l_deg = np.linspace(-1.0, 1.0, 9)
    m_deg = np.linspace(-1.0, 1.0, 13)
    plane = np.ones((13, 9))
    ours = {p: plane[None] for p in ("HH", "VV", "I")}
    with pytest.raises(ValueError, match="square"):
        ck.orientation_sweep(ours, dict(ours), l_deg, m_deg, hwhm_deg=0.5)


# ---------------------------------------------------------------------------
# build_metrics + outputs
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_metrics_inputs():
    coord = np.linspace(-1.0, 1.0, 41)
    ll, mm = np.meshgrid(coord, coord)
    plane = np.exp(-0.5 * (ll**2 + mm**2) / 0.3**2)
    ours = {p: np.stack([plane, plane]) for p in ck.PRODUCTS}
    ours["xpol"] = np.zeros_like(ours["xpol"])
    theirs = {p: np.stack([plane, plane]) for p in ck.PRODUCTS}
    theirs["xpol"] = np.zeros_like(theirs["xpol"])
    freqs = np.array([1.0e9, 1.2e9])
    return ours, theirs, coord, freqs


@pytest.mark.unit
def test_build_metrics_covers_every_frequency_product_and_region(tiny_metrics_inputs):
    ours, theirs, coord, freqs = tiny_metrics_inputs
    centre = coord.size // 2
    metrics = ck.build_metrics(ours, theirs, coord, coord, freqs, centre, centre)
    assert len(metrics["per_freq"]) == 2
    entry = metrics["per_freq"][0]
    assert entry["freq_mhz"] == pytest.approx(1000.0)
    assert set(entry["residuals"]) == set(ck.PRODUCTS)
    assert set(entry["residuals"]["I"]) == {"mainlobe", "near", "far"}
    assert "hwhm_deg" in entry
    assert "fwhm_deg" in entry


@pytest.mark.unit
def test_build_metrics_records_fwhm_for_both_models_and_axes(tiny_metrics_inputs):
    ours, theirs, coord, freqs = tiny_metrics_inputs
    centre = coord.size // 2
    metrics = ck.build_metrics(ours, theirs, coord, coord, freqs, centre, centre)
    fwhm = metrics["per_freq"][0]["fwhm_deg"]
    assert set(fwhm) == {"ours_l", "ours_m", "katbeam_l", "katbeam_m", "ratio_l", "ratio_m"}
    assert fwhm["ratio_l"] == pytest.approx(1.0, rel=1e-6)


@pytest.mark.unit
def test_build_metrics_records_non_finite_counts(tiny_metrics_inputs):
    ours, theirs, coord, freqs = tiny_metrics_inputs
    theirs = {k: v.copy() for k, v in theirs.items()}
    theirs["I"][0, 0, 0] = np.nan
    centre = coord.size // 2
    metrics = ck.build_metrics(ours, theirs, coord, coord, freqs, centre, centre)
    assert metrics["per_freq"][0]["katbeam_non_finite"]["I"] == 1


@pytest.mark.unit
def test_write_outputs_emits_json_and_markdown(tmp_path, tiny_metrics_inputs):
    ours, theirs, coord, freqs = tiny_metrics_inputs
    centre = coord.size // 2
    metrics = ck.build_metrics(ours, theirs, coord, coord, freqs, centre, centre)
    sweep = ck.orientation_sweep(
        {p: ours[p] for p in ("HH", "VV", "I")},
        {p: theirs[p] for p in ("HH", "VV", "I")},
        coord,
        coord,
        hwhm_deg=0.35,
    )
    ck.write_outputs(metrics, sweep, tmp_path)

    payload = json.loads((tmp_path / "metrics.json").read_text())
    assert payload["per_freq"][0]["freq_mhz"] == pytest.approx(1000.0)
    assert payload["orientation_sweep"]["best_overall"] == "none"
    assert "compare_katbeam" in (tmp_path / "summary.md").read_text()


@pytest.mark.unit
def test_write_outputs_json_is_finite_safe(tmp_path, tiny_metrics_inputs):
    """NaN is not valid JSON; it must serialise as null, not crash a reader."""
    ours, theirs, coord, freqs = tiny_metrics_inputs
    centre = coord.size // 2
    metrics = ck.build_metrics(ours, theirs, coord, coord, freqs, centre, centre)
    metrics["per_freq"][0]["fwhm_deg"]["ours_l"] = float("nan")
    sweep = {"per_product": {}, "best": {}, "best_overall": "none"}
    ck.write_outputs(metrics, sweep, tmp_path)
    text = (tmp_path / "metrics.json").read_text()
    assert "NaN" not in text
    assert json.loads(text)["per_freq"][0]["fwhm_deg"]["ours_l"] is None


@pytest.mark.unit
def test_format_summary_table_mentions_each_region(tiny_metrics_inputs):
    ours, theirs, coord, freqs = tiny_metrics_inputs
    centre = coord.size // 2
    metrics = ck.build_metrics(ours, theirs, coord, coord, freqs, centre, centre)
    table = ck.format_summary_table(metrics)
    for region in ("mainlobe", "near", "far"):
        assert region in table
    assert "1000.0" in table


# ---------------------------------------------------------------------------
# palette (validated with the dataviz skill's validate_palette.js)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_orientation_colours_are_the_cvd_safe_set():
    """Pins the CVD-validated palette against an accidental revert to mpl defaults.

    matplotlib's default C0-C3 FAILS colourblind separation: #2ca02c green vs
    #ff7f0e orange sit at deltaE 0.7 under protanopia, i.e. identical. The
    Okabe-Ito subset below was validated at worst-all-pairs deltaE 13.1
    (deuteranopia) in both light and dark mode.
    """
    assert list(ck.ORIENTATION_COLOURS) == list(ck.ORIENTATIONS)
    assert set(ck.ORIENTATION_COLOURS.values()) == {"#0072B2", "#D55E00", "#56B4E9", "#E69F00"}
    forbidden = {"#2ca02c", "#ff7f0e"}
    assert not (set(ck.ORIENTATION_COLOURS.values()) & forbidden)


@pytest.mark.unit
def test_ours_and_katbeam_differ_by_linestyle_not_only_colour():
    """Identity must never be carried by colour alone."""
    assert ck.STYLE_OURS["color"] != ck.STYLE_KATBEAM["color"]
    assert ck.STYLE_OURS["linestyle"] != ck.STYLE_KATBEAM["linestyle"]


# ---------------------------------------------------------------------------
# plots (smoke tests: each writes a non-trivial PNG)
# ---------------------------------------------------------------------------


@pytest.fixture
def plot_inputs():
    coord = np.linspace(-1.0, 1.0, 41)
    ll, mm = np.meshgrid(coord, coord)
    ours_plane = np.exp(-0.5 * (ll**2 + mm**2) / 0.3**2)
    theirs_plane = np.exp(-0.5 * ((ll / 0.31) ** 2 + (mm / 0.29) ** 2))
    return coord, ours_plane, theirs_plane


def _assert_png(path):
    assert path.exists(), f"{path} was not written"
    assert path.stat().st_size > 1000, f"{path} looks empty ({path.stat().st_size} bytes)"
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.unit
def test_plot_maps_writes_png(tmp_path, plot_inputs):
    coord, ours_plane, theirs_plane = plot_inputs
    out = tmp_path / "maps.png"
    ck.plot_maps(ours_plane, theirs_plane, coord, coord, out, title="t", cbar_label="c")
    _assert_png(out)


@pytest.mark.unit
def test_plot_cuts_writes_png(tmp_path, plot_inputs):
    coord, ours_plane, theirs_plane = plot_inputs
    out = tmp_path / "cuts.png"
    ck.plot_cuts(ours_plane, theirs_plane, coord, coord, 20, 20, out, title="t")
    _assert_png(out)


@pytest.mark.unit
def test_plot_radial_writes_png(tmp_path, plot_inputs):
    coord, ours_plane, theirs_plane = plot_inputs
    out = tmp_path / "radial.png"
    ck.plot_radial(ours_plane, theirs_plane, coord, coord, out, title="t", nbins=16)
    _assert_png(out)


@pytest.mark.unit
def test_plot_hhvv_writes_png(tmp_path, plot_inputs):
    coord, ours_plane, theirs_plane = plot_inputs
    ours = {"HH": ours_plane[None], "VV": ours_plane[None]}
    theirs = {"HH": theirs_plane[None], "VV": theirs_plane[None]}
    out = tmp_path / "hhvv.png"
    ck.plot_hhvv(ours, theirs, 0, coord, coord, out, title="t")
    _assert_png(out)


@pytest.mark.unit
def test_plot_crosspol_writes_png(tmp_path, plot_inputs):
    coord, ours_plane, _ = plot_inputs
    out = tmp_path / "xpol.png"
    ck.plot_crosspol(ours_plane * 1e-3, coord, coord, out, title="t")
    _assert_png(out)


@pytest.mark.unit
def test_plot_crosspol_survives_an_all_zero_map(tmp_path, plot_inputs):
    """A perfectly diagonal Jones gives identically zero cross-pol; log scale must cope."""
    coord, _, _ = plot_inputs
    out = tmp_path / "xpol_zero.png"
    ck.plot_crosspol(np.zeros((41, 41)), coord, coord, out, title="t")
    _assert_png(out)


@pytest.mark.unit
def test_plot_fwhm_vs_freq_writes_png(tmp_path):
    freqs = np.linspace(0.9e9, 1.7e9, 12)
    table = {
        "ours_l": np.linspace(1.2, 0.6, 12),
        "ours_m": np.linspace(1.25, 0.63, 12),
        "katbeam_l": np.linspace(1.18, 0.58, 12),
        "katbeam_m": np.linspace(1.24, 0.62, 12),
    }
    out = tmp_path / "fwhm.png"
    ck.plot_fwhm_vs_freq(freqs, table, out)
    _assert_png(out)


@pytest.mark.unit
def test_plot_orientation_residuals_writes_png(tmp_path):
    sweep = {
        "per_product": {
            "HH": {"none": 1e-3, "flip_x": 2e-3, "flip_y": 3e-3, "swap_xy": 4e-3},
            "VV": {"none": 1.1e-3, "flip_x": 2.1e-3, "flip_y": 3.1e-3, "swap_xy": 4.1e-3},
            "I": {"none": 1.2e-3, "flip_x": 2.2e-3, "flip_y": 3.2e-3, "swap_xy": 4.2e-3},
        },
        "best": {"HH": "none", "VV": "none", "I": "none"},
        "best_overall": "none",
    }
    out = tmp_path / "orient.png"
    ck.plot_orientation_residuals(sweep, out)
    _assert_png(out)


# ---------------------------------------------------------------------------
# main / argument parsing
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_main_rejects_band_and_bds_together(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        ck.main(["--band", "L", "--bds", str(tmp_path), "--output-dir", str(tmp_path)])


@pytest.mark.unit
def test_main_end_to_end_on_a_synthetic_bds(tmp_path):
    """Full run against a synthetic BDS: no network, no cache, no real beam data."""
    pytest.importorskip("katbeam")
    bds = tmp_path / "synthetic.bds.zarr"
    build_synthetic_bds(bds)
    out = tmp_path / "out"
    rc = ck.main(
        [
            "--bds",
            str(bds),
            "--band-model",
            "L",
            "--freqs",
            "1000",
            "1200",
            "--output-dir",
            str(out),
            "--nbins",
            "16",
            "--fwhm-stride",
            "2",
        ]
    )
    assert rc == 0
    assert (out / "metrics.json").exists()
    assert (out / "summary.md").exists()
    assert (out / "fwhm_vs_freq.png").exists()
    assert (out / "orientation_residuals.png").exists()
    assert (out / "stokesI_maps_1000MHz.png").exists()
    assert (out / "stokesI_cuts_1000MHz.png").exists()
    assert (out / "radial_1000MHz.png").exists()
    assert (out / "hhvv_maps_1000MHz.png").exists()
    assert (out / "crosspol_1000MHz.png").exists()


@pytest.mark.unit
def test_main_no_orientation_sweep_skips_that_plot(tmp_path):
    pytest.importorskip("katbeam")
    bds = tmp_path / "synthetic.bds.zarr"
    build_synthetic_bds(bds)
    out = tmp_path / "out"
    rc = ck.main(
        [
            "--bds",
            str(bds),
            "--band-model",
            "L",
            "--freqs",
            "1000",
            "--output-dir",
            str(out),
            "--no-orientation-sweep",
        ]
    )
    assert rc == 0
    assert not (out / "orientation_residuals.png").exists()
    assert (out / "metrics.json").exists()
