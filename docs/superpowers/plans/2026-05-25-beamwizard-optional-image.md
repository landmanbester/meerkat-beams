# BeamWizard Optional `image_name` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `BeamWizard`'s `image_name` optional so it can be used for beam values at explicit source positions, with clear errors when image-derived state is used unset and methods to attach an image or a bare field centre after construction.

**Architecture:** `__init__` stores image-derived state (`wcs`, `centre`, `times`, `l_grid`, `m_grid`) in private backing fields initialised to `None`, exposed via read-only properties. `centre`/`wcs`/`l_grid`/`m_grid` raise a clear `RuntimeError` when unset; `times` returns `None` (already a valid state). The image-parsing block becomes `attach_image()`; a lightweight `set_field_centre()` covers the no-file path.

**Tech Stack:** Python 3.10+, numpy, astropy (`SkyCoord`, `WCS`, `Time`), xarray, zarr<3, pytest. Spec: `docs/superpowers/specs/2026-05-25-beamwizard-optional-image-design.md`.

---

## File Structure

- Modify: `src/meerkat_beams/utils.py` — `BeamWizard.__init__`, new `attach_image()` / `set_field_centre()`, guarded properties.
- Modify: `tests/test_beam_wizard.py` — remove obsolete `test_beam_wizard_requires_image_name`, add no-image / guard / setter / attach tests.
- Modify: `scripts/test_beam_orientation.py` — call `set_field_centre` from `bundle.phase_centre`.
- Modify: `CLAUDE.md` — document optional image in the `BeamWizard` section.

---

### Task 1: Optional `image_name` + `attach_image` + guarded properties

**Files:**
- Modify: `src/meerkat_beams/utils.py:95-155` (`__init__`; add properties + `attach_image`)
- Test: `tests/test_beam_wizard.py` (remove lines 331-334; add new tests + fixture)

- [ ] **Step 1: Replace the obsolete requirement test with no-image tests**

In `tests/test_beam_wizard.py`, first add `SkyCoord` to the astropy import near the top:

```python
from astropy.coordinates import SkyCoord
```

Delete this existing test (currently `tests/test_beam_wizard.py:331-334`):

```python
@pytest.mark.unit
def test_beam_wizard_requires_image_name():
    with pytest.raises(ValueError, match="image_name is required"):
        BeamWizard(bds_name="some.bds.zarr")
```

Add this fixture and four tests in its place:

```python
@pytest.fixture
def no_image_paths(tmp_path):
    """Synthetic BDS + FITS on disk; returns (bds_path, fits_path) as strings."""
    bds = tmp_path / "synthetic.bds.zarr"
    img = tmp_path / "synthetic.fits"
    build_synthetic_bds(bds)
    build_synthetic_image(img)
    return str(bds), str(img)


@pytest.mark.unit
def test_construct_without_image(no_image_paths):
    """A BDS-only wizard constructs and supports BDS-only interpolation."""
    bds, _ = no_image_paths
    bw = BeamWizard(bds_name=bds)
    xpyp = np.array([[float(bw.bds.attrs["x0"])], [float(bw.bds.attrs["y0"])]])
    vals = bw.interpolate_beam(xpyp, FREQS, var="nstokes", i="I", j="I")
    np.testing.assert_allclose(vals[:, 0], 1.0, atol=1e-5)


@pytest.mark.unit
def test_no_image_attrs_raise(no_image_paths):
    """Image-derived attrs raise a clear RuntimeError; times is None."""
    bds, _ = no_image_paths
    bw = BeamWizard(bds_name=bds)
    for attr in ("centre", "wcs", "l_grid", "m_grid"):
        with pytest.raises(RuntimeError, match="without an image"):
            getattr(bw, attr)
    assert bw.times is None


@pytest.mark.unit
def test_no_image_get_source_coordinates_raises(no_image_paths, times):
    """get_source_coordinates needs a centre; without one it raises RuntimeError."""
    bds, _ = no_image_paths
    bw = BeamWizard(bds_name=bds)
    src = SkyCoord(RA0, DEC0, unit="deg", frame="icrs")
    with pytest.raises(RuntimeError, match="without an image"):
        bw.get_source_coordinates(src, times=times)


@pytest.mark.unit
def test_attach_image_unblocks_grid_methods(no_image_paths, times):
    """attach_image populates centre/l_grid/m_grid and unblocks grid-default methods."""
    bds, img = no_image_paths
    bw = BeamWizard(bds_name=bds)
    bw.attach_image(img)
    assert bw.l_grid is not None and bw.m_grid is not None
    xpyp, seps, _ = bw.get_source_coordinates(bw.centre, times=times)
    np.testing.assert_allclose(xpyp[0], bw.bds.attrs["x0"], atol=1e-6)
    np.testing.assert_allclose(seps.deg, 0.0, atol=1e-6)
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_beam_wizard.py -k "construct_without_image or no_image or attach_image_unblocks" -v`
Expected: FAIL — `BeamWizard(bds_name=...)` currently raises `ValueError("image_name is required")`.

- [ ] **Step 3: Rewrite `__init__` to make the image optional**

In `src/meerkat_beams/utils.py`, replace the body of `__init__` (current lines 95-155, from the signature through `self._prefilters = {}`) with:

```python
    def __init__(
        self,
        bds_name: Optional[str] = None,
        image_name: Optional[str] = None,
        *,
        band: Optional[str] = None,
    ):
        if (bds_name is None) == (band is None):
            raise ValueError("exactly one of bds_name or band must be provided")
        if band is not None:
            from meerkat_beams import cache

            bds_name = cache.ensure_band_bds(band)
        self.log = log
        log.info(f"opening BDS {bds_name}")
        self.bds = xarray.open_zarr(bds_name)
        freqs = self.bds.coords["FREQ"].values
        log.info(f"frequency range is {freqs[0] * 1e-6:.0f} to {freqs[-1] * 1e-6:.0f} MHz")
        self.index_to_freq = scipy.interpolate.interp1d(np.arange(len(freqs)), freqs)
        self.freq_to_index = scipy.interpolate.interp1d(freqs, np.arange(len(freqs)))

        # location could be made configurable
        self.default_location = EarthLocation.of_site("MeerKAT")
        log.info(f"location is MeerKAT ({self.default_location})")
        self._prefilters = {}

        # Image-derived state; unset until an image is attached or a field
        # centre is supplied. See attach_image() / set_field_centre().
        self._wcs = None
        self._centre = None
        self._times = None
        self._l_grid = None
        self._m_grid = None

        if image_name is not None:
            self.attach_image(image_name)
        else:
            log.warning(
                "BeamWizard constructed without an image: operating in BDS-only mode. "
                "Methods needing a field centre, time axis, or default l/m grid "
                "(get_source_coordinates, get_rotation_averaged_beam, get_time_freq_beam) "
                "will raise until you call attach_image(image_name) or set_field_centre(centre=...)."
            )

    @property
    def centre(self):
        if self._centre is None:
            raise RuntimeError(
                "BeamWizard.centre is unavailable: this wizard was constructed without an image. "
                "Call attach_image(image_name) or set_field_centre(centre=...) first."
            )
        return self._centre

    @property
    def wcs(self):
        if self._wcs is None:
            raise RuntimeError(
                "BeamWizard.wcs is unavailable: this wizard was constructed without an image. "
                "Call attach_image(image_name) first."
            )
        return self._wcs

    @property
    def l_grid(self):
        if self._l_grid is None:
            raise RuntimeError(
                "BeamWizard.l_grid is unavailable: this wizard was constructed without an image. "
                "Call attach_image(image_name) first, or pass l/m explicitly."
            )
        return self._l_grid

    @property
    def m_grid(self):
        if self._m_grid is None:
            raise RuntimeError(
                "BeamWizard.m_grid is unavailable: this wizard was constructed without an image. "
                "Call attach_image(image_name) first, or pass l/m explicitly."
            )
        return self._m_grid

    @property
    def times(self):
        # None is a valid state (FITS images and BDS-only wizards have no time
        # axis). Callers handle None by requiring an explicit times= argument.
        return self._times

    def attach_image(self, image_name: str) -> None:
        """Attach an image (FITS or xradio zarr), populating the field centre,
        time axis, and default l/m grid from its WCS.

        May be called after construction to upgrade a BDS-only wizard, or to
        swap the image on an existing wizard.
        """
        if image_name.endswith(".fits"):
            log.info(f"obtaining WCS from FITS image {image_name}")
            fitshdr = fits.open(image_name)[0].header
            wcs = WCS(fitshdr)
            self._times = None
        elif (Path(image_name) / ".zgroup").exists():
            log.info(f"obtaining WCS from dataset {image_name}")
            ds = xarray.open_zarr(image_name)
            fitshdr = fits.Header(dict(ds.attrs["fits_header"]))
            wcs = WCS(fitshdr)
            self._times = Time(ds.coords["TIME"].values / (24 * 3600), format="mjd")
            log.info(f"time axis is {self._times[0].iso} to {self._times[-1].iso}")
        else:
            raise RuntimeError(f"unable to determine type of image {image_name}")
        # drop WCS axes >2
        while len(wcs.axis_type_names) > 2:
            log.debug(f"dropping WCS axis {wcs.axis_type_names[-1]}")
            wcs = wcs.dropaxis(len(wcs.axis_type_names) - 1)
        self._wcs = wcs
        self._centre = wcs.pixel_to_world(fitshdr["CRPIX1"] - 1, fitshdr["CRPIX2"] - 1)
        log.info(f"image centre is at {self._centre}")

        # Construct default l/m grid from image pixels
        nx, ny = fitshdr["NAXIS1"], fitshdr["NAXIS2"]
        crpix1, crpix2 = fitshdr["CRPIX1"], fitshdr["CRPIX2"]
        cdelt1, cdelt2 = fitshdr["CDELT1"], fitshdr["CDELT2"]
        # l/m are offsets from center in degrees (l increases east, m north)
        self._l_grid = (np.arange(nx) - (crpix1 - 1)) * cdelt1
        self._m_grid = (np.arange(ny) - (crpix2 - 1)) * cdelt2
        log.info(
            f"default l/m grid: {nx}x{ny} pixels, "
            f"l=[{self._l_grid[0]:.4f}, {self._l_grid[-1]:.4f}], "
            f"m=[{self._m_grid[0]:.4f}, {self._m_grid[-1]:.4f}] deg"
        )
```

Note: the old `__init__` set `self.wcs`, `self.times`, `self.centre`, `self.l_grid`, `self.m_grid` directly. Those names are now read-only properties backed by `_wcs`/`_times`/`_centre`/`_l_grid`/`_m_grid`, so all internal reads (`self.centre`, `self.times`, `self.l_grid`, `self.m_grid` at the existing call sites) and external reads (`bw.l_grid` in `core/bds_to_xradio.py`, `bw.centre`/`bw.wcs` in tests) resolve through the properties unchanged. The two `getattr(self, "times", None)` sites still work (property always returns).

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `uv run pytest tests/test_beam_wizard.py -k "construct_without_image or no_image or attach_image_unblocks" -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Run the full BeamWizard suite for regressions**

Run: `uv run pytest tests/test_beam_wizard.py -v`
Expected: PASS — all pre-existing image-attached tests (`test_source_coordinates_at_field_centre`, `test_fits_branch_sets_times_none_raises_runtimeerror`, `test_beam_wizard_band_routes_through_cache`, etc.) still pass; `test_beam_wizard_requires_image_name` is gone.

- [ ] **Step 6: Lint**

Run: `uv run ruff check src/meerkat_beams/utils.py tests/test_beam_wizard.py && uv run ruff format --check src/meerkat_beams/utils.py tests/test_beam_wizard.py`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add src/meerkat_beams/utils.py tests/test_beam_wizard.py
git commit -m "feat(utils): make BeamWizard image_name optional with attach_image + guarded properties"
```

---

### Task 2: `set_field_centre` — file-free centre

**Files:**
- Modify: `src/meerkat_beams/utils.py` (add method after `attach_image`)
- Test: `tests/test_beam_wizard.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_beam_wizard.py`:

```python
@pytest.mark.unit
def test_set_field_centre_unblocks_source_coordinates(no_image_paths, times):
    """After set_field_centre, get_source_coordinates works without an image;
    the l/m grid stays unavailable."""
    bds, _ = no_image_paths
    bw = BeamWizard(bds_name=bds)
    centre = SkyCoord(RA0, DEC0, unit="deg", frame="icrs")
    bw.set_field_centre(centre)
    xpyp, seps, _ = bw.get_source_coordinates(centre, times=times)
    assert xpyp.shape == (2, len(times))
    np.testing.assert_allclose(xpyp[0], bw.bds.attrs["x0"], atol=1e-6)
    np.testing.assert_allclose(xpyp[1], bw.bds.attrs["y0"], atol=1e-6)
    np.testing.assert_allclose(seps.deg, 0.0, atol=1e-6)
    with pytest.raises(RuntimeError, match="without an image"):
        bw.l_grid
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_beam_wizard.py::test_set_field_centre_unblocks_source_coordinates -v`
Expected: FAIL — `AttributeError: 'BeamWizard' object has no attribute 'set_field_centre'`.

- [ ] **Step 3: Implement `set_field_centre`**

In `src/meerkat_beams/utils.py`, add this method directly after `attach_image`:

```python
    def set_field_centre(self, centre: SkyCoord, times: Optional[Time] = None) -> None:
        """Set the field (pointing) centre — and optionally the time axis —
        without attaching an image.

        Use this for workflows that only need beam values at explicit source
        positions (e.g. beam gain at a calibrator). ``centre`` is an astropy
        ``SkyCoord``; ``times`` an astropy ``Time``. The WCS and default l/m
        grid stay unavailable, so methods that fall back to the image grid
        still require explicit l/m.
        """
        self._centre = centre
        if times is not None:
            self._times = times
        log.info(f"field centre set to {centre}")
```

Confirm `SkyCoord` is imported in `utils.py` (it is used by `get_source_coordinates`; check the `from astropy.coordinates import ...` line near the top and add `SkyCoord` if missing).

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_beam_wizard.py::test_set_field_centre_unblocks_source_coordinates -v`
Expected: PASS.

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff check src/meerkat_beams/utils.py tests/test_beam_wizard.py
git add src/meerkat_beams/utils.py tests/test_beam_wizard.py
git commit -m "feat(utils): add BeamWizard.set_field_centre for image-free pointing"
```

---

### Task 3: Wire up `scripts/test_beam_orientation.py`

**Files:**
- Modify: `scripts/test_beam_orientation.py:121` (after `bw = BeamWizard(band="L")`)

- [ ] **Step 1: Add the `set_field_centre` call**

In `scripts/test_beam_orientation.py`, replace:

```python
    bw = BeamWizard(band="L")
    runs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
```

with:

```python
    bw = BeamWizard(band="L")
    # Beam pointing centre = original MS phase centre (radians from MSBundle).
    bw.set_field_centre(SkyCoord(ra_pc, dec_pc, unit="rad", frame="icrs"))
    runs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
```

`SkyCoord` is already imported (line 29) and `ra_pc, dec_pc = bundle.phase_centre` is defined earlier (line 96).

- [ ] **Step 2: Verify it parses and lints**

Run: `uv run python -c "import ast; ast.parse(open('scripts/test_beam_orientation.py').read())" && uv run ruff check scripts/test_beam_orientation.py && uv run ruff format --check scripts/test_beam_orientation.py`
Expected: no output / no errors (the script needs MS data to run end-to-end; parse + lint is the hermetic check here).

- [ ] **Step 3: Commit**

```bash
git add scripts/test_beam_orientation.py
git commit -m "feat(script): set BeamWizard field centre from MS phase centre"
```

---

### Task 4: Documentation

**Files:**
- Modify: `src/meerkat_beams/utils.py` (`BeamWizard` class docstring, line ~83)
- Modify: `CLAUDE.md` (`### BeamWizard` section)

- [ ] **Step 1: Expand the class docstring**

In `src/meerkat_beams/utils.py`, replace the `BeamWizard` docstring:

```python
    """Attaches to a BDS and provides various convenienece functions"""
```

with:

```python
    """Attaches to a BDS and provides beam-interpolation conveniences.

    ``image_name`` is optional. Without an image the wizard runs in BDS-only
    mode: ``interpolate_beam`` and the prefilter/frequency helpers work, but
    ``centre``/``wcs``/``l_grid``/``m_grid`` raise ``RuntimeError`` and
    ``times`` is ``None``. Call ``attach_image(image_name)`` to populate all
    image-derived state from a FITS/xradio image, or ``set_field_centre(centre)``
    to supply just a pointing centre (e.g. for beam gain at a fixed source).
    """
```

- [ ] **Step 2: Update CLAUDE.md**

In `CLAUDE.md`, in the `### BeamWizard` subsection, replace this opening sentence:

```
Attaches to a BDS (zarr) + an image (FITS or xradio zarr) and provides beam interpolation. The image supplies WCS and (optionally) a time axis.
```

with:

```
Attaches to a BDS (zarr) and, optionally, an image (FITS or xradio zarr) and provides beam interpolation. The image supplies WCS and (optionally) a time axis.

`image_name` is optional. Without it the wizard is BDS-only: `interpolate_beam` works, but `centre`/`wcs`/`l_grid`/`m_grid` raise `RuntimeError` and `times` is `None`. Use `attach_image(image_name)` to populate image-derived state after construction, or `set_field_centre(centre, times=None)` to supply just a pointing centre (the image-free path used by `scripts/test_beam_orientation.py`).
```

- [ ] **Step 3: Verify docs render / no stray lint**

Run: `uv run ruff check src/meerkat_beams/utils.py`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add src/meerkat_beams/utils.py CLAUDE.md
git commit -m "docs: document optional image_name, attach_image, set_field_centre"
```

---

## Final verification

- [ ] Run the unit suite: `uv run pytest -m unit -v` — all pass.
- [ ] Lint + format: `uv run ruff check . && uv run ruff format --check .` — clean.
