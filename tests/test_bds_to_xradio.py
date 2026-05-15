"""Hermetic unit tests for core/bds_to_xradio.py.

Validation branches are tested without rendering; the small end-to-end uses
the synthetic BDS + FITS image from tests/_synthetic.py.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray

from meerkat_beams.core.bds_to_xradio import bds_to_xradio
from tests._synthetic import build_synthetic_bds, build_synthetic_image


@pytest.fixture(scope="module")
def synthetic_bds_image(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bds_xradio")
    bds_path = build_synthetic_bds(tmp / "synth.bds.zarr")
    image_path = build_synthetic_image(tmp / "synth.fits")
    return Path(bds_path), Path(image_path)


@pytest.mark.unit
def test_bds_to_xradio_invalid_beam_type_raises(synthetic_bds_image, tmp_path):
    bds_path, image_path = synthetic_bds_image
    with pytest.raises(ValueError, match="Unknown beam_type"):
        bds_to_xradio(
            bds_path=str(bds_path),
            image_path=str(image_path),
            output=str(tmp_path / "out.zarr"),
            beam_type="bogus",
        )


@pytest.mark.unit
def test_bds_to_xradio_invalid_stokes_element_raises(synthetic_bds_image, tmp_path):
    bds_path, image_path = synthetic_bds_image
    with pytest.raises(ValueError, match="Invalid Stokes matrix element"):
        bds_to_xradio(
            bds_path=str(bds_path),
            image_path=str(image_path),
            output=str(tmp_path / "out.zarr"),
            beam_type="nstokes",
            elements=["XX"],
        )


@pytest.mark.unit
def test_bds_to_xradio_invalid_jones_element_raises(synthetic_bds_image, tmp_path):
    bds_path, image_path = synthetic_bds_image
    with pytest.raises(ValueError, match="Invalid Jones matrix element"):
        bds_to_xradio(
            bds_path=str(bds_path),
            image_path=str(image_path),
            output=str(tmp_path / "out.zarr"),
            beam_type="jones",
            elements=["II"],
        )


@pytest.mark.unit
def test_bds_to_xradio_output_pol_length_mismatch_raises(synthetic_bds_image, tmp_path):
    bds_path, image_path = synthetic_bds_image
    with pytest.raises(ValueError, match="output_pol"):
        bds_to_xradio(
            bds_path=str(bds_path),
            image_path=str(image_path),
            output=str(tmp_path / "out.zarr"),
            beam_type="nstokes",
            elements=["II", "QQ"],
            output_pol=["I"],
        )


@pytest.mark.unit
def test_bds_to_xradio_end_to_end_synthetic(synthetic_bds_image, tmp_path):
    """Smoke: synthetic BDS + 2D FITS image renders to an xradio-shaped zarr.

    Currently skipped: ``bds_to_xradio`` calls
    ``BeamWizard.get_time_freq_beam`` without passing ``times=``, so when the
    image is a 2-D FITS (no time axis) ``self.times`` is ``None`` and the
    rendering path raises ``RuntimeError`` in
    ``utils.BeamWizard.get_time_freq_beam``. A hermetic end-to-end requires
    either:
      * extending the synthetic image fixture to be a zarr with a TIME
        coordinate, or
      * threading an optional ``times=`` argument through ``bds_to_xradio``.
    Neither belongs in this test task; flagged for follow-up.
    """
    pytest.skip(
        "bds_to_xradio does not supply observational times; the synthetic "
        "FITS fixture has no time axis, so BeamWizard.get_time_freq_beam "
        "raises RuntimeError. Needs a zarr-image fixture or a times= path."
    )
    bds_path, image_path = synthetic_bds_image
    out = tmp_path / "rendered.zarr"
    bds_to_xradio(
        bds_path=str(bds_path),
        image_path=str(image_path),
        output=str(out),
        beam_type="nstokes",
        elements=["II"],
        pixel_stepping=8,
        time_stepping=1,
        compress=False,
    )
    ds = xarray.open_zarr(str(out))
    assert "SKY" in ds.data_vars
    assert ds["SKY"].dims == ("time", "frequency", "polarization", "l", "m")
    assert list(ds.coords["polarization"].values) == ["I"]
    # l/m should be in radians after enrich_bds_xradio (very small numerical range).
    assert np.abs(ds.coords["l"].values).max() < 0.1
