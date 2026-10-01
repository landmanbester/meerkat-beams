"""Unit tests for katbeam BDS synthesis and the shared Stokes helpers."""

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def test_stokes_basis_matrices_are_inverses():
    from meerkat_beams.utils import COHERENCY_TO_STOKES, STOKES_TO_COHERENCY

    assert STOKES_TO_COHERENCY.shape == (4, 4)
    assert COHERENCY_TO_STOKES.shape == (4, 4)
    np.testing.assert_allclose(COHERENCY_TO_STOKES @ STOKES_TO_COHERENCY, np.eye(4), atol=1e-12)


def test_jones_to_mueller_shape_and_identity():
    from meerkat_beams.utils import jones_to_mueller

    # Identity Jones at every (F, Y, X) point -> identity Mueller.
    jones = np.zeros((2, 3, 4, 2, 2), dtype=np.complex128)
    jones[..., 0, 0] = 1.0
    jones[..., 1, 1] = 1.0

    mueller = jones_to_mueller(jones)

    assert mueller.shape == (2, 3, 4, 4, 4)
    np.testing.assert_allclose(mueller[0, 0, 0], np.eye(4), atol=1e-12)


def test_mueller_to_stokes_of_identity_is_identity():
    from meerkat_beams.utils import mueller_to_stokes

    mueller = np.broadcast_to(np.eye(4, dtype=np.complex128), (2, 3, 4, 4, 4)).copy()

    stokes = mueller_to_stokes(mueller)

    assert stokes.shape == (2, 3, 4, 4, 4)
    np.testing.assert_allclose(stokes[0, 0, 0], np.eye(4), atol=1e-12)


def test_diagonal_jones_gives_the_expected_stokes_beam_matrix():
    """Pin the full Stokes beam matrix for a real diagonal Jones diag(h, v).

    With HH = I+Q and VV = I-Q, a diagonal Jones gives HH_out = h^2(I+Q) and
    VV_out = v^2(I-Q), so:
        I_out = I(h^2+v^2)/2 + Q(h^2-v^2)/2
        Q_out = I(h^2-v^2)/2 + Q(h^2+v^2)/2
    i.e. the gain sits on the diagonal and the I<->Q leakage off it, with U and
    V both scaled by h*v. Nothing leaks into or out of U/V.
    """
    from meerkat_beams.utils import jones_to_mueller, mueller_to_stokes

    h, v = 0.8, 0.6
    jones = np.zeros((1, 1, 1, 2, 2), dtype=np.complex128)
    jones[..., 0, 0] = h
    jones[..., 1, 1] = v

    stokes = mueller_to_stokes(jones_to_mueller(jones)).real

    gain = 0.5 * (h**2 + v**2)
    leak = 0.5 * (h**2 - v**2)
    expected = np.array(
        [
            [gain, leak, 0.0, 0.0],
            [leak, gain, 0.0, 0.0],
            [0.0, 0.0, h * v, 0.0],
            [0.0, 0.0, 0.0, h * v],
        ]
    )
    np.testing.assert_allclose(stokes[0, 0, 0], expected, atol=1e-12)


# --------------------------------------------------------------------------
# Band mapping, model validation, grid geometry
# --------------------------------------------------------------------------


def test_band_to_model_map_covers_all_mdv_bands():
    from meerkat_beams.katbeam_bds import KATBEAM_MODEL_FOR_BAND

    assert set(KATBEAM_MODEL_FOR_BAND) == {"U", "L", "S0", "S1", "S2", "S3", "S4"}
    assert KATBEAM_MODEL_FOR_BAND["L"] == "MKAT-AA-L-JIM-2020"
    assert KATBEAM_MODEL_FOR_BAND["U"] == "MKAT-AA-UHF-JIM-2020"
    # MdV splits S into five sub-bands; katbeam has a single S model.
    for band in ("S0", "S1", "S2", "S3", "S4"):
        assert KATBEAM_MODEL_FOR_BAND[band] == "MKAT-AA-S-JIM-2020"


def test_require_model_rejects_unknown_name_with_actionable_error():
    from meerkat_beams.katbeam_bds import require_model

    with pytest.raises(ValueError, match="not available in the installed katbeam"):
        require_model("MKAT-AA-NOPE-JIM-2020")


def test_require_model_returns_usable_jimbeam():
    from meerkat_beams.katbeam_bds import require_model

    jb = require_model("MKAT-AA-L-JIM-2020")
    assert jb.HH(0.0, 0.0, 1000.0) == pytest.approx(1.0, abs=2e-3)


def test_freq_table_is_ascending_mhz():
    from meerkat_beams.katbeam_bds import katbeam_freq_table

    table = katbeam_freq_table("MKAT-AA-L-JIM-2020")
    assert table[0] == pytest.approx(856.0)
    assert table[-1] == pytest.approx(1712.0)
    assert np.all(np.diff(table) > 0)


def test_resolve_geometry_defaults_match_the_mdv_bds():
    """Defaults mirror the real MdV BDS so the two models are pixel-identical."""
    from meerkat_beams.katbeam_bds import resolve_geometry

    assert resolve_geometry("L", None, None) == (128, 4.0)
    assert resolve_geometry("U", None, None) == (128, 6.0)


def test_resolve_geometry_honours_explicit_overrides():
    from meerkat_beams.katbeam_bds import resolve_geometry

    assert resolve_geometry("L", 64, 2.5) == (64, 2.5)
    assert resolve_geometry("S2", 32, 1.5) == (32, 1.5)


def test_resolve_geometry_raises_for_band_without_defaults():
    """S-band MdV geometry has not been measured, so do not guess it."""
    from meerkat_beams.katbeam_bds import resolve_geometry

    with pytest.raises(ValueError, match="npix"):
        resolve_geometry("S2", None, None)


def test_resolve_geometry_rejects_unknown_band():
    from meerkat_beams.katbeam_bds import resolve_geometry

    with pytest.raises(ValueError, match="unknown band"):
        resolve_geometry("X", 128, 4.0)


# --------------------------------------------------------------------------
# synthesize_katbeam_bds
# --------------------------------------------------------------------------

# A small grid and few frequencies keep these tests fast; the schema and the
# maths do not depend on resolution.
SMALL = dict(npix=16, fov_deg=4.0, num_freq=3)


def test_synthesized_dataset_matches_the_bds_schema():
    from meerkat_beams.katbeam_bds import synthesize_katbeam_bds

    ds = synthesize_katbeam_bds("L", **SMALL)

    assert set(ds.data_vars) == {"njones", "nstokes", "nmueller"}
    assert ds.njones.dims == ("receptor_i", "receptor_j", "FREQ", "Y", "X")
    assert ds.nstokes.dims == ("stokes_i", "stokes_j", "FREQ", "Y", "X")
    assert ds.nmueller.dims == ("stokes_i", "stokes_j", "FREQ", "Y", "X")
    assert ds.njones.shape == (2, 2, 3, 16, 16)
    assert ds.nstokes.shape == (4, 4, 3, 16, 16)
    assert ds.njones.dtype == np.complex64
    assert ds.nmueller.dtype == np.complex64
    assert ds.nstokes.dtype == np.float32
    assert list(ds.coords["receptor_i"].values) == [0, 1]
    assert list(ds.coords["stokes_i"].values) == list("IQUV")


def test_synthesized_attrs_match_the_bds_contract():
    from meerkat_beams.katbeam_bds import synthesize_katbeam_bds

    ds = synthesize_katbeam_bds("L", **SMALL)

    assert ds.attrs["npix"] == 16
    assert ds.attrs["dx"] == pytest.approx(0.5)  # 2*4.0/16
    assert ds.attrs["dy"] == pytest.approx(0.5)
    assert ds.attrs["x0"] == 8
    assert ds.attrs["y0"] == 8
    assert len(ds.attrs["freqs"]) == 3
    assert ds.attrs["beam_model"] == "katbeam"
    assert ds.attrs["katbeam_model"] == "MKAT-AA-L-JIM-2020"
    hdr = ds.attrs["fits_header"]
    for key in ("CDELT1", "CDELT2", "CRPIX1", "CRPIX2", "CTYPE1", "CTYPE2", "CTYPE3", "CUNIT3"):
        assert key in hdr
    assert hdr["CTYPE1"] == "X"
    assert hdr["CTYPE2"] == "Y"
    assert hdr["CTYPE3"] == "FREQ"
    assert hdr["CUNIT3"] == "Hz"
    # FITS CRPIX is 1-based, so it is x0 + 1 -- matching mdv_beams_to_bds.
    assert hdr["CRPIX1"] == ds.attrs["x0"] + 1


def test_grid_centre_is_exactly_on_axis():
    from meerkat_beams.katbeam_bds import synthesize_katbeam_bds

    ds = synthesize_katbeam_bds("L", **SMALL)

    assert float(ds.X[ds.attrs["x0"]]) == pytest.approx(0.0, abs=1e-12)
    assert float(ds.Y[ds.attrs["y0"]]) == pytest.approx(0.0, abs=1e-12)
    assert float(ds.X[0]) == pytest.approx(-4.0)


def test_construction_is_lazy():
    """Nothing is evaluated until a slice is pulled -- otherwise the 'lighter
    alternative' costs several GB at MdV resolution."""
    import dask.array as dask_array

    from meerkat_beams.katbeam_bds import synthesize_katbeam_bds

    ds = synthesize_katbeam_bds("L", **SMALL)

    for name in ("njones", "nstokes", "nmueller"):
        assert isinstance(ds[name].data, dask_array.Array), f"{name} is not dask-backed"


def test_jones_is_diagonal_with_exactly_zero_off_diagonals():
    from meerkat_beams.katbeam_bds import synthesize_katbeam_bds

    ds = synthesize_katbeam_bds("L", **SMALL)
    njones = ds.njones.values

    assert np.all(njones[0, 1] == 0)
    assert np.all(njones[1, 0] == 0)
    # katbeam has no phase information, so the diagonal is purely real.
    np.testing.assert_array_equal(njones[0, 0].imag, 0)
    np.testing.assert_array_equal(njones[1, 1].imag, 0)


def test_on_axis_jones_is_katbeam_evaluated_on_axis():
    """The centre pixel must be katbeam sampled at exactly (0, 0).

    It is near 1 but not equal to it: katbeam's squint offsets each beam centre
    (up to 0.052 deg in the L table's Hx column at 1712 MHz), so the value on
    the grid's axis sits slightly off each beam's own peak. Pinning it against
    JimBeam directly is stricter than a tolerance around 1, and it is what
    actually has to hold.
    """
    from meerkat_beams.katbeam_bds import require_model, synthesize_katbeam_bds

    ds = synthesize_katbeam_bds("L", **SMALL)
    x0, y0 = ds.attrs["x0"], ds.attrs["y0"]
    jb = require_model("MKAT-AA-L-JIM-2020")

    for k, f_mhz in enumerate(np.asarray(ds.attrs["freqs"]) / 1e6):
        assert ds.njones[0, 0, k, y0, x0].values.real == pytest.approx(jb.HH(0.0, 0.0, float(f_mhz)))
        assert ds.njones[1, 1, k, y0, x0].values.real == pytest.approx(jb.VV(0.0, 0.0, float(f_mhz)))

    # Still close to the identity: this is a normalised beam.
    np.testing.assert_allclose(ds.njones[0, 0, :, y0, x0].values.real, 1.0, atol=1e-2)
    np.testing.assert_allclose(ds.njones[1, 1, :, y0, x0].values.real, 1.0, atol=1e-2)


def test_stokes_u_and_v_do_not_mix_with_i():
    """A real diagonal Jones can produce no I<->U or I<->V leakage."""
    from meerkat_beams.katbeam_bds import synthesize_katbeam_bds

    ds = synthesize_katbeam_bds("L", **SMALL)
    nstokes = ds.nstokes.values

    assert np.allclose(nstokes[0, 2], 0, atol=1e-7)
    assert np.allclose(nstokes[0, 3], 0, atol=1e-7)
    assert np.allclose(nstokes[3, 0], 0, atol=1e-7)


def test_derived_stokes_i_matches_katbeam_own_i():
    """The issue's Jones-to-Stokes cross-check.

    JimBeam.I() is 0.5*(|HH|^2 + |VV|^2). Our nstokes[I, I] gets there by a
    completely different route -- diag(HH, VV) -> Mueller -> Stokes basis -- so
    agreement exercises jones_to_mueller and mueller_to_stokes.
    """
    from meerkat_beams.katbeam_bds import require_model, synthesize_katbeam_bds

    ds = synthesize_katbeam_bds("L", **SMALL)
    freqs_mhz = np.asarray(ds.attrs["freqs"]) / 1e6
    ll, mm = np.meshgrid(ds.X.values, ds.Y.values)

    jb = require_model("MKAT-AA-L-JIM-2020")
    for k, f in enumerate(freqs_mhz):
        expected = jb.I(ll, mm, float(f))
        np.testing.assert_allclose(ds.nstokes[0, 0, k].values, expected, rtol=1e-5, atol=1e-6)


def test_singularity_is_replaced_by_the_analytic_limit():
    """katbeam returns NaN at rr = 0.5. A NaN reaching spline_filter would
    smear across the whole slab, so synthesis must remove it."""
    from meerkat_beams.katbeam_bds import (
        SINGULAR_TAPER_LIMIT,
        SINGULAR_TAPER_RADIUS,
        require_model,
        synthesize_katbeam_bds,
    )

    # Confirm the singularity is real in the installed katbeam, so this test
    # cannot silently pass against a version that fixed it upstream.
    jb = require_model("MKAT-AA-L-JIM-2020")
    squintdeg, fwhmdeg = jb._interp_squint_fwhm_deg(1000.0)
    singular_l = squintdeg[0] + SINGULAR_TAPER_RADIUS * fwhmdeg[0]
    raw = jb.HH(np.array([singular_l]), np.array([squintdeg[1]]), 1000.0)
    assert not np.isfinite(raw[0]), "installed katbeam no longer has the 0/0 singularity"

    # A grid placed to land on it must still come out finite.
    ds = synthesize_katbeam_bds("L", npix=16, fov_deg=float(singular_l), num_freq=1)
    assert np.all(np.isfinite(ds.njones.values))
    assert np.all(np.isfinite(ds.nstokes.values))
    assert SINGULAR_TAPER_LIMIT == pytest.approx(np.pi / 4)


def test_default_frequencies_are_the_model_table():
    from meerkat_beams.katbeam_bds import katbeam_freq_table, synthesize_katbeam_bds

    ds = synthesize_katbeam_bds("L", npix=8, fov_deg=4.0)

    expected_hz = katbeam_freq_table("MKAT-AA-L-JIM-2020") * 1e6
    np.testing.assert_allclose(ds.coords["FREQ"].values, expected_hz)


def test_num_freq_spans_the_model_range():
    from meerkat_beams.katbeam_bds import synthesize_katbeam_bds

    ds = synthesize_katbeam_bds("L", npix=8, fov_deg=4.0, num_freq=5)

    freqs = ds.coords["FREQ"].values
    assert len(freqs) == 5
    assert freqs[0] == pytest.approx(856e6)
    assert freqs[-1] == pytest.approx(1712e6)


def test_freq_and_num_freq_are_mutually_exclusive():
    from meerkat_beams.katbeam_bds import synthesize_katbeam_bds

    with pytest.raises(ValueError, match="mutually exclusive"):
        synthesize_katbeam_bds("L", npix=8, fov_deg=4.0, freq=np.array([1e9]), num_freq=3)


def test_frequency_outside_the_model_table_raises():
    """np.interp inside katbeam clamps silently, so 500 MHz on the L model
    would return the 856 MHz beam. Refuse instead of lying."""
    from meerkat_beams.katbeam_bds import synthesize_katbeam_bds

    with pytest.raises(ValueError, match="outside"):
        synthesize_katbeam_bds("L", npix=8, fov_deg=4.0, freq=np.array([500e6]))
    with pytest.raises(ValueError, match="outside"):
        synthesize_katbeam_bds("L", npix=8, fov_deg=4.0, freq=np.array([2.0e9]))


def test_odd_npix_is_rejected():
    """With dx = 2*fov/npix and x0 = npix//2, the centre pixel lands on 0 only
    for even npix; odd npix would offset the beam centre by dx/2."""
    from meerkat_beams.katbeam_bds import synthesize_katbeam_bds

    with pytest.raises(ValueError, match="even"):
        synthesize_katbeam_bds("L", npix=15, fov_deg=4.0, num_freq=1)


def test_single_frequency_works():
    from meerkat_beams.katbeam_bds import synthesize_katbeam_bds

    ds = synthesize_katbeam_bds("L", npix=8, fov_deg=4.0, num_freq=1)

    assert ds.njones.shape == (2, 2, 1, 8, 8)
    assert np.all(np.isfinite(ds.nstokes.values))
