# meerkat-beams

MeerKAT primary-beam model handling: download MdV beam files from the SARAO archive, convert them to a beam dataset (BDS), and render time/frequency-resolved primary beams to `xradio`-compatible zarr stores. See https://doi.org/10.48479/wdb0-h061 for the upstream MdV beam data.

## Architecture — three layers + cabs

```
src/meerkat_beams/
├── utils.py                # shared: BeamWizard, PowerBeam, logging, zarr constants
├── cache.py                # on-demand download + BDS cache (per band)
├── cabs/<cmd>.yml          # Stimela cab definitions (one per command)
├── cli/<cmd>.py            # thin Typer wrappers (one per command)
└── core/<cmd>.py           # plain-Python implementations (one per command)
```

One-to-one mapping: `cabs/foo.yml` ↔ `cli/foo.py` ↔ `core/foo.py`. Commands wired through `cli/__init__.py` into the `mbeams` Typer app.

**Rule:** code used by more than one core module lives in `utils.py`. `core/__init__.py` stays minimal. CLI wrappers only do Typer-specific conversions (`Path → str`, `ListStr → list`, `None` handling) and lazy-import the core function. See `docs/progress.md` for the full rationale behind this layout.

### Commands

| command | core | purpose |
|---|---|---|
| `mbeams download-mdv-beams` | `core/download_mdv_beams.py` | fetch an MdV `.npz` from SARAO mirrors (band code `L`/`U`/`S0`…`S4`, filename, or full URL) |
| `mbeams mdv-beams-to-bds` | `core/mdv_beams_to_bds.py` | convert MdV `.npz` (or per-antenna mean-beam zarr) to a BDS (zarr) containing `jones`, `njones`, `stokes`, `nstokes` |
| `mbeams bds-to-xradio` | `core/bds_to_xradio.py` | render a BDS into a full `(time, frequency, polarization, l, m)` xradio zarr by interpolating the beam along parallactic-angle-rotated tracks, using a WCS image for pointing/time |
| `mbeams mdv-to-xradio` | `core/mdv_to_xradio.py` | shortcut: dump one Jones element / component of an MdV `.npz` directly into an xradio-shaped zarr (no time axis, no rotation) |

## Key abstractions in `utils.py`

### `BeamWizard`
Attaches to a BDS (zarr) + an image (FITS or xradio zarr) and provides beam interpolation. The image supplies WCS and (optionally) a time axis.

- `get_source_coordinates(srcpos, times, loc)` → (xpyp, seps, angles). Transforms sky → AltAz per time, computes sep/position-angle from field centre, converts to BDS pixel coords via `dx`/`dy`/`x0`/`y0` attrs.
- `interpolate_beam(xpyp, freq, var, i, j)` → cubic-B-spline interpolation of the beam at (freq, y, x). Uses `_get_prefilter` to cache `spline_filter(var[i,j])`; the `map_coordinates` call passes `prefilter=False` because the input is already prefiltered. **Do not flip this back to `True`** — it silently double-filters.
- `get_time_variable_beamgain(coord, times, freq, spi, …)` → beam gain along a source's track through the beam as parallactic angle rotates. With `spi`, returns frequency-averaged gain weighted by `(f/f0)**spi`.
- `get_rotation_averaged_beam(l, m, times, freq, spi, time_stepping, pixel_stepping, chunk_size, …)` → `(mean, var)` over the parallactic-angle rotations for a grid of l/m offsets. Grid is in degrees; internal upsampling uses linear `map_coordinates` when `pixel_stepping > 1`.
- `get_time_freq_beam(filename, var_name, dim_names, …, ij_list, …)` → writes the full `(ij, time, freq, x, y)` cube to a zarr store; handles dim ordering and optional pixel subsampling + upsample.

Convention: the beam cube in BDS is dim-ordered `(i, j, FREQ, Y, X)`; variable name picks Jones (`njones`/`jones`, dims `receptor_{i,j}` ∈ {0,1}) or Stokes (`nstokes`/`stokes`, dims `stokes_{i,j}` ∈ {I,Q,U,V}). `dx`, `dy` are degrees/pixel; `x0`, `y0` are the center pixel index.

### `enrich_bds_xradio(zarr_path, bw, output_var, polarizations)`
Post-processes a zarr store written by `get_time_freq_beam` into xradio schema: converts `l`/`m` from degrees to radians, sets polarization labels, adds a `direction` attribute block (`icrs`, SIN projection, reference = field centre), and re-consolidates metadata.

### Cache (`cache.py`)

`BeamWizard(band="L", image_name=...)` auto-downloads the MeerKAT
mean-beam zarr for the named band from Google Drive and builds a
compressed BDS locally, caching both under

  `$MBEAMS_CACHE_DIR` or `$XDG_CACHE_HOME/meerkat-beams` or `~/.cache/meerkat-beams`

as `inputs/MeerKAT_<BAND>.zarr/` and `bds/MeerKAT_<BAND>.bds.zarr/`.
Subsequent constructions of `BeamWizard(band=...)` reuse the cached
BDS. Supported bands: `U`, `L`, `S0`, `S4` (S1/S2/S3 have no published
gdrive ID — request the band explicitly via `bds_name=` instead).
Concurrent first-time downloads of the same band from multiple
processes are not guarded; warm the cache from a single process.

### Logging
`LOGGER` / `log` (same object). Console handler is kept on the module-level `CONSOLE`; change level via `set_console_logging_level(level)`. Don't create additional handlers.

### Zarr compression
`ZARR_COMPRESSOR` = Blosc zstd/clevel=5/BITSHUFFLE, `ZARR_FILTERS` = `[Delta(float32)]`. Only applied when a core function is called with `compress=True`.

## Domain concepts

- **MdV beams**: raw voltage beams from the SARAO archive, `.npz` with `beam` (pol, ant, freq, y, x) complex64, `freq_MHz`, `margin_deg`, `pols` (`HH`/`HV`/`VH`/`VV`), `antnames`. Last antenna index (`-1`) is `array_average`.
- **BDS (beam dataset)**: zarr holding normalised & unnormalised Jones and Stokes (Mueller-row) beams + a synthesized FITS header in `.attrs["fits_header"]` and scalar attrs `x0`, `y0`, `dx`, `dy`, `freqs`. Produced by `mdv-beams-to-bds`.
- **xradio zarr**: schema-compatible primary-beam image `(time, frequency, polarization, l, m)` with `l`/`m` in radians and a `direction` attribute block. Produced by `bds-to-xradio` or `mdv-to-xradio`.
- **Normalised vs not**: `njones` / `nstokes` are pre-multiplied by the inverse of the central-pixel Jones matrix so the on-axis beam is the identity; use these unless you specifically need raw voltage beams.

## Conventions

- Python `>=3.10`. Runtime is `hip-cargo >= 0.2.0`; the scientific stack (`xarray`, `zarr<3`, `astropy`, `scipy`, `numpy`, `matplotlib`, `dask-ms`, `wget`) is under the `[full]` extra.
- Ruff: `line-length=120`, `target-version=py310`, rules `E,F,I,N,W` with `E741`/`N806` ignored (domain names: `l`, `m`, `I`, `S`, `Sinv`). Pre-commit runs `ruff-check --fix` and `ruff-format`.
- Core function signatures mirror their CLI signatures (same names/defaults) with plain types. Don't move Typer conversions into core.
- Commits: conventional prefixes (`build:`, `chore:`, `feat:`, `fix:`…) based on recent history.
- Version bumping via `tbump` (config in `tbump.toml`).

## Running things

```bash
uv sync --group dev --group test      # install dev + test deps
uv run ruff format --check .          # format check (CI)
uv run ruff check .                   # lint (CI)
uv run pytest -v                      # full test run
uv run pytest -m unit                 # unit tests only
uv run mbeams --help                  # CLI
```

CI (`.github/workflows/ci.yml`) runs ruff + pytest on Python 3.10–3.13. Commit message containing `[skip checks]` skips the CI job. Docker image built from `Dockerfile` (python:3.11-slim, installs `.[full]`).

## Tests

Located in `tests/`. Markers: `unit`, `integration`, `slow`.

- `test_install.py`, `test_cli.py` — always run; import + `--help` smoke tests.
- `test_mdv_beams_to_bds.py` — integration; parametrised over bands `U`/`L`/`S0`–`S4`. Compares a freshly-converted BDS against a reference BDS. Needs `MBEAMS_REFERENCE_BDS_<BAND>` env var to point at a reference, **and** input zarr `tests/data/MeerKAT_<BAND>.zarr` to exist (auto-downloaded once by `conftest.py` from a Google Drive tarball for the U band).
- `test_beam_consistency.py`, `test_rendered_beam.py` — integration + slow; need `MBEAMS_BDS_PATH` and `MBEAMS_IMAGE_PATH` env vars. Cross-checks `get_time_variable_beamgain`, `get_rotation_averaged_beam`, and rendered-zarr values against each other.

Tests silently skip when env vars / data are unavailable — if you expect a test to run, check the skip reason first.

## Known minor issues

Flagged while reviewing `BeamWizard.interpolate_beam` and `get_time_freq_beam`. Unit coverage for `BeamWizard` lives in `tests/test_beam_wizard.py` (hermetic — synthetic BDS + FITS in `tmp_path`); those tests are what pin the fixes below.

**Fixed:**
- **Double-prefiltering in `interpolate_beam`.** `_get_prefilter` already runs `spline_filter`, so the inner `map_coordinates` call now passes `prefilter=False`. Pinned by `test_subpixel_matches_direct_scipy`.
- **Implicit off-cube policy.** `map_coordinates` in `interpolate_beam` now passes `mode="constant", cval=0.0` explicitly; out-of-X/Y coordinates still return 0 (pinned by `test_out_of_range_xy_returns_zero`).
- **Spline order in the prefilter cache.** `_get_prefilter` now takes `order: int = 3` and includes it in the cache key; `interpolate_beam` accepts and forwards `order` to both `spline_filter` and `map_coordinates`.
- **Silent out-of-range frequency.** `interpolate_beam` now raises `ValueError` with the requested vs. BDS frequency ranges (in MHz) before calling `freq_to_index`. Pinned by `test_out_of_range_freq_raises`.
- **FITS-image branch `self.time` → `self.times` typo.** The FITS branch of `BeamWizard.__init__` now sets `self.times = None`, so `get_source_coordinates` with no explicit `times` raises the documented `RuntimeError` instead of an `AttributeError`. Pinned by `test_fits_branch_sets_times_none_raises_runtimeerror`.

**Still open:**
1. **Redundant meshgrid in `interpolate_beam`** — `fx` and `fy` both recompute the identical freq grid; only `fy[0]` and `fx[1]` are used. Functionally correct, just wasteful and confusing.
2. **`spline_filter` default output dtype is float64** even on float32 inputs — doubles the memory per cached prefilter entry. Fix: pass `output=np.float32` (or inherit the input dtype) in `_get_prefilter`.
3. **`get_time_freq_beam` zarr fill_value pitfall.** `zarr.Group.create_dataset` is called for the beam variable with `fill_value=0`, and the coord datasets are created with zarr's default fill (also 0). When the store is re-opened with `xarray.open_zarr(...)` (default `mask_and_scale=True`), every genuine `0.0` — including a `l=0` or `m=0` coord, or a beam pixel that happens to be 0 — comes back as `NaN`. The `test_time_freq_beam_writes_zarr` test works around this with `mask_and_scale=False`. Proper fix: pass `fill_value=None` for the coord datasets (values are always valid), and for the beam variable either use `fill_value=None` or a sentinel that can't collide with real data.

## Current branch

`dev001` — the version of the package still in the hip-cargo transition (see `docs/progress.md`, "Remaining" section). CI matrix is 3.10–3.13 but `requires-python = ">=3.10"`.
