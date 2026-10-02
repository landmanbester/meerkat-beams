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

    ``JimBeam.I()`` is ``0.5*(|HH|^2 + |VV|^2)``. Our ``nstokes[I, I]`` reaches
    the same place by a completely different route -- ``diag(HH, VV)`` ->
    Mueller -> Stokes basis -- so agreement exercises ``jones_to_mueller`` and
    ``mueller_to_stokes``, and now the on-axis normalisation as well.

    The expected value is built only from ``JimBeam`` calls, never from our own
    pipeline, so this stays a genuine cross-check rather than a restatement: the
    normalised Stokes I of a diagonal Jones is
    ``0.5*((HH/HH0)^2 + (VV/VV0)^2)``.
    """
    from meerkat_beams.katbeam_bds import require_model, synthesize_katbeam_bds

    ds = synthesize_katbeam_bds("L", **SMALL)
    freqs_mhz = np.asarray(ds.attrs["freqs"]) / 1e6
    ll, mm = np.meshgrid(ds.X.values, ds.Y.values)

    jb = require_model("MKAT-AA-L-JIM-2020")
    for k, f in enumerate(freqs_mhz):
        hh = jb.HH(ll, mm, float(f)) / jb.HH(0.0, 0.0, float(f))
        vv = jb.VV(ll, mm, float(f)) / jb.VV(0.0, 0.0, float(f))
        expected = 0.5 * (np.abs(hh) ** 2 + np.abs(vv) ** 2)
        np.testing.assert_allclose(ds.nstokes[0, 0, k].values, expected, rtol=1e-5, atol=1e-6)


def test_our_conversion_reproduces_katbeam_i_on_raw_jones():
    """The purest form of the issue's cross-check, independent of the dataset.

    Feeding katbeam's RAW diagonal Jones through our shared Jones->Stokes helpers
    must reproduce ``JimBeam.I()`` exactly. Unlike the normalised comparison this
    involves no scaling at all, so it isolates ``jones_to_mueller`` and
    ``mueller_to_stokes``.

    Note the normalised dataset values do *not* relate to ``I()`` by a single
    scalar: ``0.25*((HH/HH0)^2+(VV/VV0)^2)*(HH0^2+VV0^2)`` equals
    ``0.5*(HH^2+VV^2)`` only when ``HH0 == VV0``, which katbeam's squint makes
    false. Hence this test works from raw Jones rather than rescaling the dataset.
    """
    from meerkat_beams.katbeam_bds import require_model
    from meerkat_beams.utils import jones_to_mueller, mueller_to_stokes

    jb = require_model("MKAT-AA-L-JIM-2020")
    degs = np.linspace(-4.0, 4.0, 16)
    ll, mm = np.meshgrid(degs, degs)

    for f in (856.0, 1284.0, 1712.0):
        hh = np.asarray(jb.HH(ll, mm, f), dtype=float)
        vv = np.asarray(jb.VV(ll, mm, f), dtype=float)
        jones = np.zeros((1,) + ll.shape + (2, 2), dtype=np.complex128)
        jones[0, :, :, 0, 0] = hh
        jones[0, :, :, 1, 1] = vv

        derived = mueller_to_stokes(jones_to_mueller(jones)).real[0, :, :, 0, 0]

        np.testing.assert_allclose(derived, jb.I(ll, mm, f), rtol=1e-10, atol=1e-12)


def test_sanitize_replaces_non_finite_with_the_analytic_limit(caplog):
    """Pin the replacement directly.

    Reaching the singularity through a real grid needs ``1 - 4*rr**2`` to be
    exactly ``0.0`` in floating point, which is not reliably constructible, so
    the sanitizer gets its own test rather than relying on an integration path
    to land on it.
    """
    import logging

    from meerkat_beams.katbeam_bds import SINGULAR_TAPER_LIMIT, _sanitize

    arr = np.array([np.inf, np.nan, -np.inf, 1.0, 0.5])

    with caplog.at_level(logging.WARNING, logger="meerkat_beams"):
        out = _sanitize(arr, "HH")

    np.testing.assert_allclose(out, [SINGULAR_TAPER_LIMIT, SINGULAR_TAPER_LIMIT, SINGULAR_TAPER_LIMIT, 1.0, 0.5])
    assert SINGULAR_TAPER_LIMIT == pytest.approx(np.pi / 4)
    # The replacement is reported, not silent: it is a departure from what
    # katbeam returned.
    assert "replacing 3 non-finite HH sample(s)" in caplog.text


def test_sanitize_leaves_finite_input_untouched_and_silent(caplog):
    import logging

    from meerkat_beams.katbeam_bds import _sanitize

    arr = np.array([1.0, 0.5, -0.25])

    with caplog.at_level(logging.WARNING, logger="meerkat_beams"):
        out = _sanitize(arr, "VV")

    np.testing.assert_array_equal(out, arr)
    assert "replacing" not in caplog.text


def test_upstream_katbeam_still_has_the_zero_over_zero_singularity():
    """Guards the reason _sanitize exists.

    If katbeam ever fixes this upstream, this test fails and the sanitizer can
    be reconsidered -- rather than it silently becoming dead code.
    """
    from meerkat_beams.katbeam_bds import SINGULAR_TAPER_RADIUS, require_model

    jb = require_model("MKAT-AA-L-JIM-2020")
    squintdeg, fwhmdeg = jb._interp_squint_fwhm_deg(1000.0)
    singular_l = squintdeg[0] + SINGULAR_TAPER_RADIUS * fwhmdeg[0]

    raw = jb.HH(np.array([singular_l]), np.array([squintdeg[1]]), 1000.0)

    assert not np.isfinite(raw[0]), "installed katbeam no longer has the 0/0 singularity"


def test_synthesis_output_is_all_finite():
    """Whatever katbeam returns, nothing non-finite reaches the Dataset.

    A NaN surviving into spline_filter smears across the entire slab, so this is
    asserted over a grid and frequency range wide enough to be representative.
    """
    from meerkat_beams.katbeam_bds import synthesize_katbeam_bds

    ds = synthesize_katbeam_bds("L", npix=32, fov_deg=4.0, num_freq=4)

    assert np.all(np.isfinite(ds.njones.values))
    assert np.all(np.isfinite(ds.nstokes.values))
    assert np.all(np.isfinite(ds.nmueller.values))


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


def test_freq_chunk_size_adapts_to_npix():
    """Frequency chunking must bound the per-block transient, not the frequency
    count.

    ``_eval_block`` holds complex128 ``jones`` (2x2), ``mueller`` (4x4) and
    ``stokes`` (4x4) for a whole chunk before the astype downcasts, so the
    transient scales as ``nfreq * npix**2``. A fixed 256-frequency chunk peaked
    at 3.3 GiB at npix=128 -- heavier than the MdV path this is meant to be a
    lighter alternative to.
    """
    from meerkat_beams.katbeam_bds import FREQ_CHUNK_ELEMENTS, _freq_chunk_size

    # Coarse grids get many frequencies per block, fine grids get few.
    assert _freq_chunk_size(16) > _freq_chunk_size(128)
    assert _freq_chunk_size(128) > _freq_chunk_size(512)

    # The element budget is respected at every size, and never drops below 1.
    for npix in (8, 16, 32, 64, 128, 256, 512, 1024):
        nf = _freq_chunk_size(npix)
        assert nf >= 1
        assert nf * npix * npix <= FREQ_CHUNK_ELEMENTS or nf == 1


def test_eval_block_transient_stays_bounded():
    """The per-block peak must stay far below the 3.3 GiB a 256-frequency chunk
    cost at npix=128."""
    import tracemalloc

    from meerkat_beams.katbeam_bds import _eval_block, _freq_chunk_size

    npix = 128
    nf = _freq_chunk_size(npix)
    ll, mm = np.meshgrid(np.linspace(-4, 4, npix), np.linspace(-4, 4, npix))
    freqs = np.linspace(856.0, 1712.0, nf)

    tracemalloc.start()
    try:
        _eval_block("MKAT-AA-L-JIM-2020", ll, mm, freqs)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    peak_mib = peak / 2**20
    # 144 MiB as measured; 250 leaves headroom without letting a regression to
    # the original 3328 MiB (fixed 256-frequency chunk, complex128) slip through.
    assert peak_mib < 250, f"per-block peak {peak_mib:.0f} MiB is too large"


@pytest.mark.parametrize("fov_deg", [-4.0, 0.0])
def test_non_positive_fov_is_rejected(fov_deg):
    """A negative fov_deg silently MIRRORS the beam, which is the worst failure
    mode available in a repo whose axis sign conventions are still unsettled.

    With fov_deg=-4.0 the grid runs +4.0 -> -3.9375, dx is negative, and x0 still
    lands on 0.0 -- so nothing downstream complains, while xp = l/dx + x0 flips
    east for west. fov_deg=0.0 gives dx=0 and inf.
    """
    from meerkat_beams.katbeam_bds import synthesize_katbeam_bds

    with pytest.raises(ValueError, match="fov_deg"):
        synthesize_katbeam_bds("L", npix=16, fov_deg=fov_deg, num_freq=1)


@pytest.mark.parametrize("npix", [0, -2])
def test_non_positive_npix_is_rejected(npix):
    """npix=0 passes the evenness check and dies in a raw ZeroDivisionError."""
    from meerkat_beams.katbeam_bds import synthesize_katbeam_bds

    with pytest.raises(ValueError, match="npix"):
        synthesize_katbeam_bds("L", npix=npix, fov_deg=4.0, num_freq=1)


@pytest.mark.parametrize("num_freq", [0, -1])
def test_non_positive_num_freq_is_rejected(num_freq):
    """num_freq=0 dies in the log line with an IndexError."""
    from meerkat_beams.katbeam_bds import synthesize_katbeam_bds

    with pytest.raises(ValueError, match="num_freq"):
        synthesize_katbeam_bds("L", npix=16, fov_deg=4.0, num_freq=num_freq)


def test_mirrored_grid_is_what_the_fov_guard_prevents():
    """Documents the concrete harm, so the guard is not relaxed casually.

    Builds the grid arithmetic by hand with a negative half-width and shows the
    axis reverses while the centre index still lands on zero -- undetectable
    downstream.
    """
    npix, fov_deg = 16, -4.0
    delta = 2.0 * fov_deg / npix
    degs = -fov_deg + np.arange(npix) * delta

    assert delta < 0
    assert degs[0] > degs[-1], "axis is reversed"
    assert degs[npix // 2] == pytest.approx(0.0), "centre still on axis, so nothing complains"


def test_stokes_conversion_preserves_complex64():
    """The conversion must not promote complex64 to complex128.

    The basis matrices are complex128 literals, so a bare `Sinv @ M @ S`
    promotes, doubling the largest intermediate in the pipeline -- the
    (nfreq, ny, nx, 4, 4) Mueller array. Both callers cast their output down to
    complex64/float32 anyway, so the promotion buys nothing and costs 2x memory.
    """
    from meerkat_beams.utils import jones_to_mueller, mueller_to_stokes

    jones64 = np.zeros((2, 4, 4, 2, 2), dtype=np.complex64)
    jones64[..., 0, 0] = 0.9
    jones64[..., 1, 1] = 0.8

    mueller64 = jones_to_mueller(jones64)
    stokes64 = mueller_to_stokes(mueller64)

    assert mueller64.dtype == np.complex64
    assert stokes64.dtype == np.complex64

    # complex128 must still work, and give the same answer to float32 precision.
    stokes128 = mueller_to_stokes(jones_to_mueller(jones64.astype(np.complex128)))
    assert stokes128.dtype == np.complex128
    np.testing.assert_allclose(stokes64, stokes128, atol=1e-6)


def test_eval_block_does_not_use_complex128():
    """The per-block transient is dominated by the 4x4 Mueller array, so the whole
    block pipeline stays in complex64."""
    from meerkat_beams.katbeam_bds import _eval_block

    ll, mm = np.meshgrid(np.linspace(-4, 4, 8), np.linspace(-4, 4, 8))
    njones, nmueller, nstokes = _eval_block("MKAT-AA-L-JIM-2020", ll, mm, np.array([1000.0]))

    assert njones.dtype == np.complex64
    assert nmueller.dtype == np.complex64
    assert nstokes.dtype == np.float32


def test_njones_is_exactly_identity_on_axis():
    """The `n` prefix must mean the same thing as it does for an MdV BDS.

    MdV's n* variables are pre-multiplied by the inverse of the centre-pixel
    Jones, so njones on axis is exactly the identity and nstokes[I,I] exactly 1.
    katbeam's raw HH/VV are only *approximately* unity at (0,0) -- squint offsets
    each beam's peak off the grid centre, giving 0.9950 at 1712 MHz -- so the same
    normalisation has to be applied here, or a caller correcting data with
    njones carries a 0.5% on-axis error that the name denies.
    """
    from meerkat_beams.katbeam_bds import synthesize_katbeam_bds

    ds = synthesize_katbeam_bds("L", npix=16, fov_deg=4.0, num_freq=5)
    x0, y0 = ds.attrs["x0"], ds.attrs["y0"]

    np.testing.assert_allclose(ds.njones[0, 0, :, y0, x0].values.real, 1.0, atol=1e-6)
    np.testing.assert_allclose(ds.njones[1, 1, :, y0, x0].values.real, 1.0, atol=1e-6)
    np.testing.assert_allclose(ds.nstokes[0, 0, :, y0, x0].values, 1.0, atol=1e-6)
    # Q/U/V leakage on axis stays zero.
    np.testing.assert_allclose(ds.nstokes[0, 1, :, y0, x0].values, 0.0, atol=1e-6)


def test_normalisation_is_a_per_frequency_scalar_on_each_receptor():
    """Normalisation must divide each receptor by its own centre value and nothing
    else -- it may not reshape the beam.

    Pinned against JimBeam directly, so this does not restate the implementation:
    njones[0,0] * HH(0,0,f) must reproduce raw HH everywhere.
    """
    from meerkat_beams.katbeam_bds import require_model, synthesize_katbeam_bds

    ds = synthesize_katbeam_bds("L", npix=16, fov_deg=4.0, num_freq=3)
    jb = require_model("MKAT-AA-L-JIM-2020")
    ll, mm = np.meshgrid(ds.X.values, ds.Y.values)

    for k, f in enumerate(np.asarray(ds.attrs["freqs"]) / 1e6):
        raw_hh = jb.HH(ll, mm, float(f))
        raw_vv = jb.VV(ll, mm, float(f))
        np.testing.assert_allclose(
            ds.njones[0, 0, k].values.real * jb.HH(0.0, 0.0, float(f)), raw_hh, rtol=1e-5, atol=1e-7
        )
        np.testing.assert_allclose(
            ds.njones[1, 1, k].values.real * jb.VV(0.0, 0.0, float(f)), raw_vv, rtol=1e-5, atol=1e-7
        )


@pytest.mark.parametrize(
    "bad, match",
    [
        (np.array([]), "at least one frequency"),
        (np.array([1.0e9, np.nan]), "finite"),
        (np.array([1.0e9, np.inf]), "finite"),
        (np.array([1.2e9, 1.0e9]), "increasing"),
        (np.array([1.0e9, 1.0e9]), "increasing"),
    ],
)
def test_invalid_explicit_freq_is_rejected(bad, match):
    """An empty axis died in the log line with IndexError; NaN, inf, unsorted and
    duplicate values were accepted outright and became FREQ coordinates, after
    which BeamWizard builds interp1d index<->freq mappings on a non-monotonic or
    non-finite axis."""
    from meerkat_beams.katbeam_bds import synthesize_katbeam_bds

    with pytest.raises(ValueError, match=match):
        synthesize_katbeam_bds("L", npix=8, fov_deg=4.0, freq=bad)


def test_cdelt3_is_only_set_for_a_uniform_frequency_axis():
    """A single CDELT3 cannot describe the default L table, which is non-uniform
    (... 1600, 1650, 1670, 1712 MHz). Software reading it as a linear axis would
    assign the wrong frequency to later planes, so it is omitted instead -- the
    exact axis is always available in attrs['freqs'] and the FREQ coordinate.
    """
    from meerkat_beams.katbeam_bds import synthesize_katbeam_bds

    non_uniform = synthesize_katbeam_bds("L", npix=8, fov_deg=4.0)
    assert "CDELT3" not in non_uniform.attrs["fits_header"]

    uniform = synthesize_katbeam_bds("L", npix=8, fov_deg=4.0, num_freq=5)
    hdr = uniform.attrs["fits_header"]
    freqs = uniform.coords["FREQ"].values
    assert hdr["CDELT3"] == pytest.approx(freqs[1] - freqs[0])

    single = synthesize_katbeam_bds("L", npix=8, fov_deg=4.0, num_freq=1)
    assert "CDELT3" not in single.attrs["fits_header"]
