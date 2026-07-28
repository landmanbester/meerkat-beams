"""
Unit tests for scripts/compare_katbeam.py.

Hermetic: pure helpers only. No katbeam import at module scope, no BDS on
disk, no network, no env vars. Runs under MBEAMS_OFFLINE=1.
"""

import compare_katbeam as ck
import numpy as np
import pytest

from meerkat_beams.cache import SUPPORTED_BANDS


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
