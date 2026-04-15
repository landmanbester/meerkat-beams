"""
Regression tests for mdv_beams_to_bds().

Runs the refactored conversion on input zarr data in tests/data/,
then compares the output BDS against a reference BDS produced by
the original suricat-beams.

Reference BDS paths are read from environment variables:
    MBEAMS_REFERENCE_BDS_U   - U (UHF) band
    MBEAMS_REFERENCE_BDS_L   - L band
    MBEAMS_REFERENCE_BDS_S0  - S0 sub-band
    ... (S1-S4 follow the same pattern)

Tests skip when the env var is not set or the input data is missing.
"""

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import xarray

from tests.conftest import BAND_INPUT_ZARR, test_data_path

# Bands to test -- add new bands here as data becomes available
ALL_BANDS = ["U", "L", "S0", "S1", "S2", "S3", "S4"]

# Tolerance for floating-point comparisons
ATOL = 1e-6
RTOL = 1e-6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ref_bds_path(band: str) -> str | None:
    """Return the reference BDS path from env, or None."""
    return os.environ.get(f"MBEAMS_REFERENCE_BDS_{band}")


def _input_zarr_path(band: str) -> Path | None:
    """Return the input zarr path in tests/data/, or None if missing."""
    name = BAND_INPUT_ZARR.get(band)
    if name is None:
        return None
    p = test_data_path / name
    return p if p.exists() else None


def _run_conversion(input_zarr: Path, output_dir: Path) -> Path:
    """Run mdv_beams_to_bds and return the output BDS path."""
    from meerkat_beams.core.mdv_beams_to_bds import mdv_beams_to_bds

    bds_path = output_dir / "test_output.bds"
    mdv_beams_to_bds(
        mdv_beams=str(input_zarr),
        bds=str(bds_path),
        compress=False,
    )
    return bds_path


# ---------------------------------------------------------------------------
# Parametrized fixture
# ---------------------------------------------------------------------------


@pytest.fixture(params=ALL_BANDS, scope="session")
def bds_pair(request):
    """
    For each band, run the conversion and yield (new_ds, ref_ds, band).
    Skips if reference BDS env var is unset or input data is missing.
    """
    band = request.param
    ref_path = _ref_bds_path(band)
    if ref_path is None:
        pytest.skip(f"MBEAMS_REFERENCE_BDS_{band} not set")
    if not Path(ref_path).exists():
        pytest.skip(f"Reference BDS not found at {ref_path}")

    input_zarr = _input_zarr_path(band)
    if input_zarr is None:
        pytest.skip(f"Input zarr for band {band} not in tests/data/")

    tmpdir = Path(tempfile.mkdtemp(prefix=f"mbeams_test_{band}_"))
    try:
        new_bds_path = _run_conversion(input_zarr, tmpdir)
        new_ds = xarray.open_zarr(str(new_bds_path), chunks=None)
        ref_ds = xarray.open_zarr(ref_path, chunks=None)
        yield new_ds, ref_ds, band
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Tests: coordinates and attributes
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestBdsCoordinatesAndAttributes:
    def test_data_variables_present(self, bds_pair):
        new_ds, ref_ds, band = bds_pair
        assert set(new_ds.data_vars) == set(ref_ds.data_vars)

    def test_freq_coordinate(self, bds_pair):
        new_ds, ref_ds, band = bds_pair
        np.testing.assert_allclose(
            new_ds.coords["FREQ"].values,
            ref_ds.coords["FREQ"].values,
            rtol=1e-12,
        )

    def test_spatial_coordinates(self, bds_pair):
        new_ds, ref_ds, band = bds_pair
        np.testing.assert_allclose(
            new_ds.coords["X"].values,
            ref_ds.coords["X"].values,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            new_ds.coords["Y"].values,
            ref_ds.coords["Y"].values,
            rtol=1e-12,
        )

    def test_receptor_coordinates(self, bds_pair):
        new_ds, ref_ds, band = bds_pair
        np.testing.assert_array_equal(
            new_ds.coords["receptor_i"].values,
            ref_ds.coords["receptor_i"].values,
        )
        np.testing.assert_array_equal(
            new_ds.coords["receptor_j"].values,
            ref_ds.coords["receptor_j"].values,
        )

    def test_stokes_coordinates(self, bds_pair):
        new_ds, ref_ds, band = bds_pair
        np.testing.assert_array_equal(
            new_ds.coords["stokes_i"].values,
            ref_ds.coords["stokes_i"].values,
        )
        np.testing.assert_array_equal(
            new_ds.coords["stokes_j"].values,
            ref_ds.coords["stokes_j"].values,
        )

    def test_scalar_attributes(self, bds_pair):
        new_ds, ref_ds, band = bds_pair
        for key in ("x0", "y0", "dx", "dy"):
            assert new_ds.attrs[key] == ref_ds.attrs[key], (
                f"[{band}] attr {key}: new={new_ds.attrs[key]}, ref={ref_ds.attrs[key]}"
            )

    def test_fits_header(self, bds_pair):
        new_ds, ref_ds, band = bds_pair
        new_hdr = new_ds.attrs["fits_header"]
        ref_hdr = ref_ds.attrs["fits_header"]
        assert set(new_hdr.keys()) == set(ref_hdr.keys())
        for key in ref_hdr:
            assert new_hdr[key] == ref_hdr[key], f"[{band}] fits_header[{key}]: new={new_hdr[key]}, ref={ref_hdr[key]}"

    def test_freqs_attribute(self, bds_pair):
        new_ds, ref_ds, band = bds_pair
        np.testing.assert_allclose(
            np.array(new_ds.attrs["freqs"]),
            np.array(ref_ds.attrs["freqs"]),
            rtol=1e-12,
        )


# ---------------------------------------------------------------------------
# Tests: data variables
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestBdsDataVariables:
    @pytest.mark.parametrize("var", ["jones", "njones", "stokes", "nstokes"])
    def test_shape(self, bds_pair, var):
        new_ds, ref_ds, band = bds_pair
        assert new_ds[var].shape == ref_ds[var].shape, (
            f"[{band}] {var} shape: new={new_ds[var].shape}, ref={ref_ds[var].shape}"
        )

    @pytest.mark.parametrize("var", ["jones", "njones", "stokes", "nstokes"])
    def test_values(self, bds_pair, var):
        new_ds, ref_ds, band = bds_pair
        np.testing.assert_allclose(
            new_ds[var].values,
            ref_ds[var].values,
            atol=ATOL,
            rtol=RTOL,
            err_msg=f"[{band}] {var} values differ",
        )
