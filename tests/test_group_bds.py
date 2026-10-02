"""Unit tests for baseline-group BDS assembly (_build_group_bds)."""

import logging

import numpy as np
import pytest
import xarray

from meerkat_beams.utils import GROUP_TELESCOPES, _build_group_bds
from tests._synthetic import FREQS, I0, build_telescope_bds

MK = "MeerKAT"
MKE = "MeerKAT Extension"


@pytest.fixture(scope="module")
def stores(tmp_path_factory):
    """One MeerKAT-like and one MKE-like BDS, with genuinely different Jones."""
    tmp = tmp_path_factory.mktemp("group")
    p = build_telescope_bds(
        tmp / "mk.zarr", tmp / "mk.bds.zarr", scale=1.0, leak=0.05, phase=0.0, telescope=MK, antenna="mavg"
    )
    q = build_telescope_bds(
        tmp / "mke.zarr", tmp / "mke.bds.zarr", scale=0.8, leak=0.12, phase=0.4, telescope=MKE, antenna="eavg"
    )
    return xarray.open_zarr(str(p)), xarray.open_zarr(str(q))


@pytest.fixture(scope="module")
def relabelled(tmp_path_factory):
    """The same two Jones cubes, carrying each other's telescope labels.

    The group API fixes p = MeerKAT and q = MeerKAT+, so probing a maths
    property that needs the roles exchanged means exchanging the labels, not
    bypassing the validation.
    """
    tmp = tmp_path_factory.mktemp("relabelled")
    # MKE label, MeerKAT's Jones.
    mke_like_mk = build_telescope_bds(
        tmp / "a.zarr", tmp / "a.bds.zarr", scale=1.0, leak=0.05, phase=0.0, telescope=MKE
    )
    # MeerKAT label, MKE's Jones.
    mk_like_mke = build_telescope_bds(tmp / "b.zarr", tmp / "b.bds.zarr", scale=0.8, leak=0.12, phase=0.4, telescope=MK)
    return xarray.open_zarr(str(mke_like_mk)), xarray.open_zarr(str(mk_like_mke))


@pytest.mark.unit
def test_group_telescopes_table():
    assert GROUP_TELESCOPES == {"MM": (MK, MK), "MPM": (MK, MKE), "MPMP": (MKE, MKE)}


@pytest.mark.unit
def test_mm_matches_single_telescope_nstokes(stores):
    """MM against the same store must reproduce its own nstokes exactly."""
    p, _ = stores
    grp = _build_group_bds(p, p, "MM")
    np.testing.assert_allclose(grp["nstokes"].values, p["nstokes"].values, rtol=1e-5, atol=1e-6)


@pytest.mark.unit
def test_mpm_reduces_to_mm_for_identical_jones(stores, relabelled):
    """With both halves carrying the same Jones, the cross collapses to the auto case."""
    p, _ = stores
    mke_like_mk, _ = relabelled
    mm = _build_group_bds(p, p, "MM")
    mpm = _build_group_bds(p, mke_like_mk, "MPM")
    np.testing.assert_allclose(mpm["nstokes"].values.real, mm["nstokes"].values, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(mpm["nstokes"].values.imag, 0.0, rtol=0, atol=1e-6)


@pytest.mark.unit
def test_mpm_is_complex_and_has_imaginary_part(stores):
    p, q = stores
    mpm = _build_group_bds(p, q, "MPM")
    assert mpm["nstokes"].dtype == np.complex64
    assert np.abs(mpm["nstokes"].values.imag).max() > 1e-4, "cross block should not be real"


@pytest.mark.unit
def test_auto_groups_are_real_float32(stores):
    p, q = stores
    assert _build_group_bds(p, p, "MM")["nstokes"].dtype == np.float32
    assert _build_group_bds(q, q, "MPMP")["nstokes"].dtype == np.float32
    assert _build_group_bds(p, p, "MM")["nmueller"].dtype == np.complex64


@pytest.mark.unit
def test_swapping_p_and_q_conjugates_the_stokes_block(stores, relabelled):
    """Baseline reversal conjugates the Stokes-basis block, with no transpose.

    Reversal Hermitian-conjugates the coherency matrix, which in the Stokes
    basis is plain complex conjugation: P @ S == conj(S) for the standard
    linear-feed basis, so conj(Sinv) @ conj(K) @ conj(S) == conj(Sinv @ K @ S).
    """
    p, q = stores
    mke_like_mk, mk_like_mke = relabelled
    a = _build_group_bds(p, q, "MPM")["nstokes"].values
    b = _build_group_bds(mk_like_mke, mke_like_mk, "MPM")["nstokes"].values
    np.testing.assert_allclose(b, np.conj(a), rtol=1e-4, atol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("group", ["MM", "MPM", "MPMP"])
def test_on_axis_nstokes_is_identity(stores, group):
    """Normalise-then-cross means every group is the identity on axis."""
    p, q = stores
    a, b = {"MM": (p, p), "MPM": (p, q), "MPMP": (q, q)}[group]
    grp = _build_group_bds(a, b, group)
    on_axis = grp["nstokes"].values[:, :, :, I0, I0]
    for f in range(on_axis.shape[2]):
        np.testing.assert_allclose(on_axis[:, :, f], np.eye(4), rtol=1e-4, atol=1e-5)


@pytest.mark.unit
def test_jones_absent_for_mpm_present_for_auto(stores):
    p, q = stores
    assert "njones" not in _build_group_bds(p, q, "MPM").data_vars
    assert "jones" not in _build_group_bds(p, q, "MPM").data_vars
    assert "njones" in _build_group_bds(p, p, "MM").data_vars
    assert "njones" in _build_group_bds(q, q, "MPMP").data_vars


@pytest.mark.unit
def test_group_attrs_are_carried(stores):
    p, q = stores
    grp = _build_group_bds(p, q, "MPM")
    assert grp.attrs["group"] == "MPM"
    assert grp.attrs["telescope_p"] == MK
    assert grp.attrs["telescope_q"] == MKE
    for key in ("x0", "y0", "dx", "dy", "fits_header"):
        assert key in grp.attrs


@pytest.mark.unit
def test_cross_group_drops_single_telescope_provenance(stores):
    """Side p's telescope/antenna would be misleading on a cross block."""
    p, q = stores
    grp = _build_group_bds(p, q, "MPM")
    assert "telescope" not in grp.attrs
    assert "antenna" not in grp.attrs


@pytest.mark.unit
def test_wrong_telescope_raises(stores):
    p, q = stores
    with pytest.raises(ValueError, match="expects telescope"):
        _build_group_bds(q, p, "MPMP")  # side p is MeerKAT, MPMP wants MKE


@pytest.mark.unit
def test_missing_telescope_attr_warns_and_proceeds(tmp_path, caplog):
    """A hand-built or legacy BDS carries no telescope attr: warn, don't raise."""
    anon = xarray.open_zarr(
        str(build_telescope_bds(tmp_path / "anon.zarr", tmp_path / "anon.bds.zarr", telescope=None))
    )
    with caplog.at_level(logging.WARNING, logger="meerkat_beams"):
        grp = _build_group_bds(anon, anon, "MM")
    assert "nstokes" in grp.data_vars
    assert any("telescope" in r.message for r in caplog.records)


@pytest.mark.unit
def test_frequency_subset_is_intersected(tmp_path):
    """One store covering more channels is sliced down to the common set."""
    wide = np.concatenate([FREQS, FREQS[-1] + np.diff(FREQS)[0] * np.arange(1, 3)])
    a = xarray.open_zarr(
        str(build_telescope_bds(tmp_path / "a.zarr", tmp_path / "a.bds.zarr", telescope=MK, freqs=wide))
    )
    b = xarray.open_zarr(
        str(build_telescope_bds(tmp_path / "b.zarr", tmp_path / "b.bds.zarr", telescope=MKE, freqs=FREQS))
    )
    grp = _build_group_bds(a, b, "MPM")
    np.testing.assert_allclose(grp.coords["FREQ"].values, FREQS, rtol=0, atol=1.0e3)


@pytest.mark.unit
def test_frequency_match_tolerates_float32_roundtrip(tmp_path):
    """Channel centres must not be compared with ==; float32 shifts them."""
    nudged = FREQS.astype(np.float32).astype(np.float64)
    a = xarray.open_zarr(str(build_telescope_bds(tmp_path / "a.zarr", tmp_path / "a.bds.zarr", telescope=MK)))
    b = xarray.open_zarr(
        str(build_telescope_bds(tmp_path / "b.zarr", tmp_path / "b.bds.zarr", telescope=MKE, freqs=nudged))
    )
    grp = _build_group_bds(a, b, "MPM")
    assert len(grp.coords["FREQ"]) == len(FREQS)


@pytest.mark.unit
def test_disjoint_frequencies_raise(tmp_path):
    a = xarray.open_zarr(str(build_telescope_bds(tmp_path / "a.zarr", tmp_path / "a.bds.zarr", telescope=MK)))
    b = xarray.open_zarr(
        str(build_telescope_bds(tmp_path / "b.zarr", tmp_path / "b.bds.zarr", telescope=MKE, freqs=FREQS + 5.0e9))
    )
    with pytest.raises(ValueError, match="no overlapping channels"):
        _build_group_bds(a, b, "MPM")


@pytest.mark.unit
def test_spatial_grid_mismatch_raises(tmp_path):
    a = xarray.open_zarr(str(build_telescope_bds(tmp_path / "a.zarr", tmp_path / "a.bds.zarr", telescope=MK)))
    coarse = (np.arange(21) - 10) * 0.1
    b = xarray.open_zarr(
        str(build_telescope_bds(tmp_path / "b.zarr", tmp_path / "b.bds.zarr", telescope=MKE, degs=coarse))
    )
    with pytest.raises(ValueError, match="spatial grids differ"):
        _build_group_bds(a, b, "MPM")


@pytest.mark.unit
def test_nmueller_is_kron_of_p_with_conj_q(stores):
    """Pins both the receptor-axis transposes and the p = MeerKAT ordering.

    Without this, swapping transpose(2,3,4,0,1) to transpose(2,3,4,1,0), or
    passing (jq, jp) instead of (jp, jq), leaves the whole group suite green:
    an on-axis identity check is symmetric under both, and the conjugation
    test is symmetric under a p/q swap by construction.
    """
    p, q = stores
    grp = _build_group_bds(p, q, "MPM")
    f, y, x = 1, I0 + 3, I0 - 5  # off-axis, where the asymmetry lives
    jp = p["njones"].values[:, :, f, y, x]
    jq = q["njones"].values[:, :, f, y, x]
    expected = np.kron(jp, np.conj(jq))
    np.testing.assert_allclose(grp["nmueller"].values[:, :, f, y, x], expected, rtol=1e-4, atol=1e-6)
    # the reversed ordering must NOT also satisfy it, or the test proves nothing
    assert not np.allclose(np.kron(jq, np.conj(jp)), expected, rtol=1e-4, atol=1e-6)


@pytest.mark.unit
def test_ambiguous_channel_match_raises(tmp_path):
    """Two of p's channels matching one of q's must raise, not silently alias.

    np.nonzero on the pairwise-match matrix does not require an injective
    match, so without a guard q's plane would be used twice against two
    different p channels and both sides would still come out the same length.
    """
    near = np.array([FREQS[0], FREQS[0] + 100.0, FREQS[1], FREQS[2], FREQS[3]])
    a = xarray.open_zarr(
        str(build_telescope_bds(tmp_path / "a.zarr", tmp_path / "a.bds.zarr", telescope=MK, freqs=near))
    )
    b = xarray.open_zarr(
        str(build_telescope_bds(tmp_path / "b.zarr", tmp_path / "b.bds.zarr", telescope=MKE, freqs=FREQS))
    )
    with pytest.raises(ValueError, match="ambiguous channel match"):
        _build_group_bds(a, b, "MPM")


@pytest.mark.unit
def test_fits_header_follows_a_frequency_slice(tmp_path):
    """fits_header is part of the BDS contract: it must describe the sliced data."""
    wide = np.concatenate([FREQS, FREQS[-1] + np.diff(FREQS)[0] * np.arange(1, 3)])
    a = xarray.open_zarr(
        str(build_telescope_bds(tmp_path / "a.zarr", tmp_path / "a.bds.zarr", telescope=MK, freqs=wide))
    )
    b = xarray.open_zarr(
        str(build_telescope_bds(tmp_path / "b.zarr", tmp_path / "b.bds.zarr", telescope=MKE, freqs=FREQS))
    )
    grp = _build_group_bds(a, b, "MPM")
    freqs = grp.coords["FREQ"].values
    hdr = grp.attrs["fits_header"]
    assert hdr["NAXIS3"] == len(freqs)
    np.testing.assert_allclose(hdr["CRVAL3"], freqs[0], rtol=0, atol=1.0)
    np.testing.assert_allclose(hdr["CDELT3"], freqs[1] - freqs[0], rtol=0, atol=1.0)


@pytest.mark.unit
def test_unevenly_spaced_intersection_raises(tmp_path):
    """A FITS linear FREQ axis cannot describe a gapped channel set.

    If q lacks an interior channel the intersection is non-uniform, and a
    single CRVAL3/CDELT3 pair then misplaces every plane after the gap --
    100 MHz out, for the synthetic spacing used here.
    """
    a = xarray.open_zarr(
        str(build_telescope_bds(tmp_path / "a.zarr", tmp_path / "a.bds.zarr", telescope=MK, freqs=FREQS))
    )
    b = xarray.open_zarr(
        str(build_telescope_bds(tmp_path / "b.zarr", tmp_path / "b.bds.zarr", telescope=MKE, freqs=FREQS[[0, 1, 3]]))
    )
    with pytest.raises(ValueError, match="unevenly spaced"):
        _build_group_bds(a, b, "MPM")
