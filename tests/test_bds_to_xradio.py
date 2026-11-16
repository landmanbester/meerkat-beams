"""Hermetic unit tests for core/bds_to_xradio.py.

The element grammar is tested directly on ``resolve_elements``; the end-to-end
render runs through the real CLI in tests/test_cli_native.py.
"""

from pathlib import Path

import pytest

from meerkat_beams.core.bds_to_xradio import bds_to_xradio
from meerkat_beams.utils import resolve_elements
from tests._synthetic import build_synthetic_bds, build_synthetic_image


@pytest.fixture(scope="module")
def synthetic_bds_image(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bds_xradio")
    bds_path = build_synthetic_bds(tmp / "synth.bds.zarr")
    image_path = build_synthetic_image(tmp / "synth.fits")
    return Path(bds_path), Path(image_path)


@pytest.mark.unit
def test_resolve_elements_mueller_accepts_iquv_and_labels_by_element():
    """mueller/nmueller share IQUV labels with stokes, but a 16-term cube must be
    labelled by the full element (e.g. 'IQ') to avoid polarization-label collisions."""
    ij_list, pol = resolve_elements("nmueller", ["IQ", "UV"], [])
    assert ij_list == [("I", "Q"), ("U", "V")]
    assert pol == ["IQ", "UV"]


@pytest.mark.unit
def test_resolve_elements_mueller_rejects_jones_element():
    with pytest.raises(ValueError, match="Invalid Stokes matrix element"):
        resolve_elements("mueller", ["XX"], [])


@pytest.mark.unit
def test_resolve_elements_stokes_backward_compatible_label():
    """stokes/nstokes keep labelling by the output Stokes (e[1])."""
    ij_list, pol = resolve_elements("nstokes", ["IQ"], [])
    assert ij_list == [("I", "Q")]
    assert pol == ["Q"]


@pytest.mark.unit
def test_resolve_elements_rejects_unknown_beam_type():
    with pytest.raises(ValueError, match="Unknown beam_type"):
        resolve_elements("bogus", [], [])


@pytest.mark.unit
def test_resolve_elements_stokes_rejects_jones_element():
    with pytest.raises(ValueError, match="Invalid Stokes matrix element"):
        resolve_elements("nstokes", ["XX"], [])


@pytest.mark.unit
def test_resolve_elements_jones_rejects_stokes_element():
    with pytest.raises(ValueError, match="Invalid Jones matrix element"):
        resolve_elements("jones", ["II"], [])


@pytest.mark.unit
def test_resolve_elements_output_pol_length_mismatch_raises():
    with pytest.raises(ValueError, match="output_pol"):
        resolve_elements("nstokes", ["II", "QQ"], ["I"])


@pytest.mark.unit
def test_bds_to_xradio_validates_elements_before_writing(synthetic_bds_image, tmp_path):
    """The core function runs the element grammar itself, and before it writes anything."""
    bds_path, image_path = synthetic_bds_image
    out = tmp_path / "out.zarr"
    with pytest.raises(ValueError, match="Invalid Stokes matrix element"):
        bds_to_xradio(
            bds_path=str(bds_path),
            image_path=str(image_path),
            output=str(out),
            beam_type="nmueller",
            elements=["XX"],
        )
    assert not out.exists()
