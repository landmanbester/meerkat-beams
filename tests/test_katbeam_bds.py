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
