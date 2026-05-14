# BDS Regression Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify that `mdv_beams_to_bds()` in the refactored meerkat-beams produces output identical (within tolerance) to the original suricat-beams, for every supported band.

**Architecture:** A single test module runs `mdv_beams_to_bds()` on input zarr data already in `tests/data/`, writes BDS output to a temp directory, then opens both the new and reference BDS as xarray Datasets and compares coordinates, data variables, and attributes element-by-element. Reference BDS paths come from env vars `MBEAMS_REFERENCE_BDS_<BAND>` (U, L, S0-S4). Tests skip when the env var is unset.

**Tech Stack:** pytest, xarray, numpy, zarr (< 3)

---

## File Structure

| File | Responsibility |
|------|---------------|
| `tests/test_mdv_beams_to_bds.py` (create) | Regression tests comparing BDS output against reference |
| `tests/conftest.py` (modify) | Add shared paths/constants for test data lookup |

---

### Task 1: Add band-to-path mapping in conftest.py

**Files:**
- Modify: `tests/conftest.py`

The existing conftest only knows about `MeerKAT_UHF.zarr`. We need a mapping from band codes to input zarr paths, and from env vars to reference BDS paths.

- [ ] **Step 1: Read current conftest and add band mappings**

Add these constants after the existing `beam_path` definition (around line 14):

```python
# Band code -> input zarr filename in tests/data/
BAND_INPUT_ZARR = {
    "U": "MeerKAT_UHF.zarr",
    "L": "MeerKAT_L.zarr",
    # S-band sub-bands can be added as data becomes available
    # "S0": "MeerKAT_S0.zarr",
    # "S1": "MeerKAT_S1.zarr",
    # "S2": "MeerKAT_S2.zarr",
    # "S3": "MeerKAT_S3.zarr",
    # "S4": "MeerKAT_S4.zarr",
}
```

- [ ] **Step 2: Verify conftest loads cleanly**

Run: `python -c "import tests.conftest; print(tests.conftest.BAND_INPUT_ZARR)"`
Expected: `{'U': 'MeerKAT_UHF.zarr', 'L': 'MeerKAT_L.zarr'}`

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "chore: add band-to-input-zarr mapping in conftest"
```

---

### Task 2: Write the regression test module (U-band, coordinates and attributes)

**Files:**
- Create: `tests/test_mdv_beams_to_bds.py`

Start with tests that verify the structural aspects: coordinates match, attributes match, and the right data variables exist.

- [ ] **Step 1: Write the test file with fixtures and structural tests**

Create `tests/test_mdv_beams_to_bds.py`:

```python
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
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pytest
import xarray

from tests.conftest import test_data_path, BAND_INPUT_ZARR

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
# Fixtures
# ---------------------------------------------------------------------------

def _make_bds_pair(band: str):
    """
    Run conversion and open both datasets.
    Returns (new_ds, ref_ds, tmpdir) or skips the test.
    """
    ref_path = _ref_bds_path(band)
    if ref_path is None:
        pytest.skip(f"MBEAMS_REFERENCE_BDS_{band} not set")
    if not Path(ref_path).exists():
        pytest.skip(f"Reference BDS not found at {ref_path}")

    input_zarr = _input_zarr_path(band)
    if input_zarr is None:
        pytest.skip(f"Input zarr for band {band} not found in tests/data/")

    tmpdir = Path(tempfile.mkdtemp(prefix=f"mbeams_test_{band}_"))
    new_bds_path = _run_conversion(input_zarr, tmpdir)
    new_ds = xarray.open_zarr(str(new_bds_path), chunks=None)
    ref_ds = xarray.open_zarr(ref_path, chunks=None)
    return new_ds, ref_ds, tmpdir


@pytest.fixture
def u_band_bds_pair():
    """Produce (new_ds, ref_ds) for U band, clean up after."""
    new_ds, ref_ds, tmpdir = _make_bds_pair("U")
    yield new_ds, ref_ds
    shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Tests: U-band coordinates and attributes
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestUBandCoordinatesAndAttributes:
    """Verify BDS coordinates and attributes match the reference."""

    def test_data_variables_present(self, u_band_bds_pair):
        new_ds, ref_ds = u_band_bds_pair
        assert set(new_ds.data_vars) == set(ref_ds.data_vars), (
            f"Data vars differ: new={set(new_ds.data_vars)}, ref={set(ref_ds.data_vars)}"
        )

    def test_freq_coordinate(self, u_band_bds_pair):
        new_ds, ref_ds = u_band_bds_pair
        np.testing.assert_allclose(
            new_ds.coords["FREQ"].values,
            ref_ds.coords["FREQ"].values,
            rtol=1e-12,
            err_msg="FREQ coordinates differ",
        )

    def test_x_coordinate(self, u_band_bds_pair):
        new_ds, ref_ds = u_band_bds_pair
        np.testing.assert_allclose(
            new_ds.coords["X"].values,
            ref_ds.coords["X"].values,
            rtol=1e-12,
            err_msg="X coordinates differ",
        )

    def test_y_coordinate(self, u_band_bds_pair):
        new_ds, ref_ds = u_band_bds_pair
        np.testing.assert_allclose(
            new_ds.coords["Y"].values,
            ref_ds.coords["Y"].values,
            rtol=1e-12,
            err_msg="Y coordinates differ",
        )

    def test_receptor_coordinates(self, u_band_bds_pair):
        new_ds, ref_ds = u_band_bds_pair
        np.testing.assert_array_equal(
            new_ds.coords["receptor_i"].values,
            ref_ds.coords["receptor_i"].values,
        )
        np.testing.assert_array_equal(
            new_ds.coords["receptor_j"].values,
            ref_ds.coords["receptor_j"].values,
        )

    def test_stokes_coordinates(self, u_band_bds_pair):
        new_ds, ref_ds = u_band_bds_pair
        np.testing.assert_array_equal(
            new_ds.coords["stokes_i"].values,
            ref_ds.coords["stokes_i"].values,
        )
        np.testing.assert_array_equal(
            new_ds.coords["stokes_j"].values,
            ref_ds.coords["stokes_j"].values,
        )

    def test_scalar_attributes(self, u_band_bds_pair):
        new_ds, ref_ds = u_band_bds_pair
        for key in ("x0", "y0", "dx", "dy"):
            assert new_ds.attrs[key] == ref_ds.attrs[key], (
                f"Attribute {key} differs: new={new_ds.attrs[key]}, ref={ref_ds.attrs[key]}"
            )

    def test_fits_header(self, u_band_bds_pair):
        new_ds, ref_ds = u_band_bds_pair
        new_hdr = new_ds.attrs["fits_header"]
        ref_hdr = ref_ds.attrs["fits_header"]
        assert set(new_hdr.keys()) == set(ref_hdr.keys()), (
            f"FITS header keys differ"
        )
        for key in ref_hdr:
            assert new_hdr[key] == ref_hdr[key], (
                f"FITS header {key} differs: new={new_hdr[key]}, ref={ref_hdr[key]}"
            )

    def test_freqs_attribute(self, u_band_bds_pair):
        new_ds, ref_ds = u_band_bds_pair
        np.testing.assert_allclose(
            np.array(new_ds.attrs["freqs"]),
            np.array(ref_ds.attrs["freqs"]),
            rtol=1e-12,
            err_msg="freqs attribute differs",
        )
```

- [ ] **Step 2: Run the structural tests to verify they work**

Run: `MBEAMS_REFERENCE_BDS_U=$HOME/data/mkat_beams/meerkat_U.bds pytest tests/test_mdv_beams_to_bds.py::TestUBandCoordinatesAndAttributes -v`

Expected: All tests PASS (or FAIL if the zarr code path has a bug, which is valuable information).

- [ ] **Step 3: Commit**

```bash
git add tests/test_mdv_beams_to_bds.py
git commit -m "test: add BDS regression tests for coordinates and attributes (U-band)"
```

---

### Task 3: Add data variable comparison tests (U-band)

**Files:**
- Modify: `tests/test_mdv_beams_to_bds.py`

Add tests that compare the actual beam data arrays element-by-element.

- [ ] **Step 1: Add data variable tests to the test file**

Append to `tests/test_mdv_beams_to_bds.py`:

```python
# ---------------------------------------------------------------------------
# Tests: U-band data variables
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestUBandDataVariables:
    """Verify BDS data arrays match the reference within tolerance."""

    ATOL = 1e-6
    RTOL = 1e-6

    def test_jones_shape(self, u_band_bds_pair):
        new_ds, ref_ds = u_band_bds_pair
        assert new_ds["jones"].shape == ref_ds["jones"].shape, (
            f"jones shape: new={new_ds['jones'].shape}, ref={ref_ds['jones'].shape}"
        )

    def test_jones_values(self, u_band_bds_pair):
        new_ds, ref_ds = u_band_bds_pair
        np.testing.assert_allclose(
            new_ds["jones"].values,
            ref_ds["jones"].values,
            atol=self.ATOL,
            rtol=self.RTOL,
            err_msg="jones values differ",
        )

    def test_njones_shape(self, u_band_bds_pair):
        new_ds, ref_ds = u_band_bds_pair
        assert new_ds["njones"].shape == ref_ds["njones"].shape, (
            f"njones shape: new={new_ds['njones'].shape}, ref={ref_ds['njones'].shape}"
        )

    def test_njones_values(self, u_band_bds_pair):
        new_ds, ref_ds = u_band_bds_pair
        np.testing.assert_allclose(
            new_ds["njones"].values,
            ref_ds["njones"].values,
            atol=self.ATOL,
            rtol=self.RTOL,
            err_msg="njones values differ",
        )

    def test_stokes_shape(self, u_band_bds_pair):
        new_ds, ref_ds = u_band_bds_pair
        assert new_ds["stokes"].shape == ref_ds["stokes"].shape, (
            f"stokes shape: new={new_ds['stokes'].shape}, ref={ref_ds['stokes'].shape}"
        )

    def test_stokes_values(self, u_band_bds_pair):
        new_ds, ref_ds = u_band_bds_pair
        np.testing.assert_allclose(
            new_ds["stokes"].values,
            ref_ds["stokes"].values,
            atol=self.ATOL,
            rtol=self.RTOL,
            err_msg="stokes values differ",
        )

    def test_nstokes_shape(self, u_band_bds_pair):
        new_ds, ref_ds = u_band_bds_pair
        assert new_ds["nstokes"].shape == ref_ds["nstokes"].shape, (
            f"nstokes shape: new={new_ds['nstokes'].shape}, ref={ref_ds['nstokes'].shape}"
        )

    def test_nstokes_values(self, u_band_bds_pair):
        new_ds, ref_ds = u_band_bds_pair
        np.testing.assert_allclose(
            new_ds["nstokes"].values,
            ref_ds["nstokes"].values,
            atol=self.ATOL,
            rtol=self.RTOL,
            err_msg="nstokes values differ",
        )
```

- [ ] **Step 2: Run all U-band tests**

Run: `MBEAMS_REFERENCE_BDS_U=$HOME/data/mkat_beams/meerkat_U.bds pytest tests/test_mdv_beams_to_bds.py -v`

Expected: All tests PASS. If a shape mismatch or value mismatch appears, it indicates a real bug in the zarr code path of `mdv_beams_to_bds()` that needs fixing before proceeding.

- [ ] **Step 3: Commit**

```bash
git add tests/test_mdv_beams_to_bds.py
git commit -m "test: add BDS data variable regression tests (U-band)"
```

---

### Task 4: Parametrize for L-band and future bands

**Files:**
- Modify: `tests/test_mdv_beams_to_bds.py`

Replace the U-band-specific fixture with a parametrized approach so L-band (and future S-bands) work automatically when their env vars and input data are present.

- [ ] **Step 1: Refactor to parametrized fixtures**

Replace the `u_band_bds_pair` fixture and class structure with a parametrized approach. The full updated file:

```python
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
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pytest
import xarray

from tests.conftest import test_data_path, BAND_INPUT_ZARR

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

@pytest.fixture(params=ALL_BANDS)
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
            new_ds.coords["X"].values, ref_ds.coords["X"].values, rtol=1e-12,
        )
        np.testing.assert_allclose(
            new_ds.coords["Y"].values, ref_ds.coords["Y"].values, rtol=1e-12,
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
            assert new_hdr[key] == ref_hdr[key], (
                f"[{band}] fits_header[{key}]: new={new_hdr[key]}, ref={ref_hdr[key]}"
            )

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
```

- [ ] **Step 2: Run with U-band env var**

Run: `MBEAMS_REFERENCE_BDS_U=$HOME/data/mkat_beams/meerkat_U.bds pytest tests/test_mdv_beams_to_bds.py -v`

Expected: U-band tests run (PASS or informative FAIL). L and S-band tests skip with "not set" message.

- [ ] **Step 3: Run with both U and L band env vars (if L reference exists)**

Run: `MBEAMS_REFERENCE_BDS_U=$HOME/data/mkat_beams/meerkat_U.bds MBEAMS_REFERENCE_BDS_L=$HOME/data/mkat_beams/meerkat_L.bds pytest tests/test_mdv_beams_to_bds.py -v`

Expected: U and L tests run. S-band tests skip.

- [ ] **Step 4: Commit**

```bash
git add tests/test_mdv_beams_to_bds.py
git commit -m "test: parametrize BDS regression tests across all bands"
```

---

### Task 5: Fix any bugs surfaced by the tests

This task only applies if earlier tests revealed failures. The most likely issue is in the zarr code path of `mdv_beams_to_bds()`: the input zarr at `tests/data/MeerKAT_UHF.zarr` has `BEAM` with shape `[4, 1024, 128, 128]` (no antenna dimension — already the mean beam), but the code applies `bm = bm[:, -1]` unconditionally, which would incorrectly select the last frequency channel instead of the last antenna.

**Files:**
- Modify: `src/meerkat_beams/core/mdv_beams_to_bds.py:29-33`

- [ ] **Step 1: Diagnose the failure**

If tests show a shape mismatch (e.g. output jones has wrong shape), check whether `bm[:, -1]` is the cause. The zarr BEAM array is `[4, NFREQ, NY, NX]` while the npz beam is `[4, NANT, NFREQ, NY, NX]`. The `[:, -1]` indexing only makes sense for the 5D npz case.

- [ ] **Step 2: Fix the zarr branch**

In `src/meerkat_beams/core/mdv_beams_to_bds.py`, the zarr branch should skip the antenna selection since the zarr already contains only the mean beam:

```python
    if mdv_beams.endswith(".npz"):
        mdv = np.load(mdv_beams)
        bm = mdv["beam"]
        degs = mdv["margin_deg"]
        freqs = mdv["freq_MHz"] * 1e6
        bm = bm[:, -1]  # select average beam (last antenna index)
    elif (Path(mdv_beams) / ".zgroup").exists():
        xds = xarray.open_zarr(mdv_beams, chunks=None)
        bm = xds.BEAM.values  # already mean beam: [4, NFREQ, NY, NX]
        degs = xds.l_beam.values
        freqs = xds.chan.values
    else:
        raise ValueError(f"input mdv_beams {mdv_beams} is not a valid npz or zarr dataset")
```

Move `bm = bm[:, -1]` into the npz branch only.

- [ ] **Step 3: Re-run the tests**

Run: `MBEAMS_REFERENCE_BDS_U=$HOME/data/mkat_beams/meerkat_U.bds pytest tests/test_mdv_beams_to_bds.py -v`

Expected: All U-band tests PASS.

- [ ] **Step 4: Commit the fix**

```bash
git add src/meerkat_beams/core/mdv_beams_to_bds.py
git commit -m "fix: skip antenna selection for zarr input in mdv_beams_to_bds"
```

---

## Notes

### Data needed per band

| Band | Input zarr in tests/data/ | Reference BDS (from suricat) | Env var |
|------|--------------------------|------------------------------|---------|
| U    | `MeerKAT_UHF.zarr` (present) | `~/data/mkat_beams/meerkat_U.bds` (present) | `MBEAMS_REFERENCE_BDS_U` |
| L    | `MeerKAT_L.zarr` (present) | `~/data/mkat_beams/meerkat_L.bds` (present) | `MBEAMS_REFERENCE_BDS_L` |
| S0   | Not yet in tests/data/ | Not yet produced | `MBEAMS_REFERENCE_BDS_S0` |
| S1   | Not yet in tests/data/ | Not yet produced | `MBEAMS_REFERENCE_BDS_S1` |
| S2   | Not yet in tests/data/ | Not yet produced | `MBEAMS_REFERENCE_BDS_S2` |
| S3   | Not yet in tests/data/ | Not yet produced | `MBEAMS_REFERENCE_BDS_S3` |
| S4   | Not yet in tests/data/ | Not yet produced | `MBEAMS_REFERENCE_BDS_S4` |

To add S-band coverage:
1. Download the S-band npz files (e.g. via `suricat download S0`)
2. Convert to zarr if needed, place in `tests/data/`
3. Run `suricat mdv2bds` on the npz to produce reference BDS files
4. Add the zarr filenames to `BAND_INPUT_ZARR` in `conftest.py`
5. Set the `MBEAMS_REFERENCE_BDS_S0` (etc.) env vars when running tests
