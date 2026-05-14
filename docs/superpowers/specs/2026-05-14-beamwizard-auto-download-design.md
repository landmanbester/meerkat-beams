# BeamWizard auto-download design

Date: 2026-05-14
Branch: dev001
Status: approved (ready for implementation plan)

## Goal

`BeamWizard` should be usable from a fresh checkout without pre-running
`mbeams download-mdv-beams` and `mbeams mdv-beams-to-bds`. Specifying a
primary-beam band code (e.g. `"L"`, `"U"`) is enough; data is fetched and
converted on demand and cached for reuse in later sessions.

## Non-goals

- Concurrent first-time downloads of the same band from multiple processes
  (locking is deferred).
- Bands without a published Google Drive ID (S1, S2, S3) — these continue to
  raise.
- New CLI commands. Warming the cache is done by calling
  `BeamWizard(band=...)` once.
- Pre-built BDS artifacts hosted on Google Drive. The cached BDS is always
  built locally from the mean-beam input zarr.

## High-level approach

1. New module `src/meerkat_beams/cache.py` owns:
   - the band → Google Drive ID registry,
   - cache-root resolution,
   - `ensure_band_bds(band: str) -> str`, which idempotently downloads the
     mean-beam zarr and converts it to a BDS, caching both on disk.
2. `BeamWizard.__init__` gains a keyword-only `band=` parameter. When set,
   `bds_name` is resolved via `ensure_band_bds(band)`; otherwise the
   constructor behaves as today.
3. `tests/conftest.py` imports from `cache.py` (single source of truth for
   gdrive IDs) and uses `ensure_band_bds("L")` to populate the cache at
   session start.

## Module layout

```
src/meerkat_beams/
  cache.py          # NEW
  utils.py          # BeamWizard.__init__ gets `band=` kwarg
  core/             # unchanged
  cli/              # unchanged
tests/
  conftest.py       # rewritten to delegate to cache.ensure_band_bds
  test_cache.py     # NEW, unit-marked, hermetic
  test_beam_wizard.py  # adds XOR-validation tests, optional online test
  test_mdv_beams_to_bds.py  # reads inputs from cache.input_zarr_path
```

## Public surface of `cache.py`

```python
BAND_GDRIVE_IDS: dict[str, str] = {
    "U":  "105JWCFo4R-Qo6wHCCkhPm7ZhOSlUaoPx",
    "L":  "1dAVD5sE-9fL1kGTjlpaXtI1lOBHJH19K",
    "S0": "1UN5slkHYfXD_MGUZaKFH-UBalgqiepfP",
    "S4": "1-8eg7cCZO4HwTdXW5F55ftmJPOSj3qFV",
}
SUPPORTED_BANDS: tuple[str, ...] = ("U", "L", "S0", "S4")

def cache_root() -> Path: ...
def input_zarr_path(band: str) -> Path: ...   # <root>/inputs/MeerKAT_<BAND>.zarr
def bds_path(band: str) -> Path: ...          # <root>/bds/MeerKAT_<BAND>.bds.zarr
def ensure_band_bds(band: str) -> str: ...    # returns str(bds_path(band))
```

Internal helpers (underscore-prefixed):

- `_download_and_extract(band)` — `gdown.download(id=BAND_GDRIVE_IDS[band])`
  fetches the `MeerKAT_<BAND>.zarr.tgz` tarball into a sibling
  `inputs/MeerKAT_<BAND>.zarr.partial/` directory, extracts it in place,
  then `os.replace()`s the directory to its final name and deletes the
  tarball.
- `_convert_to_bds(band)` — calls `meerkat_beams.core.mdv_beams_to_bds`
  with `compress=True`, writing to a sibling `.partial` directory then
  renaming.
- `_clear_partials(band)` — removes any leftover `*.partial` directories
  for this band (called at the top of `ensure_band_bds`).

## Cache-root resolution

```
MBEAMS_CACHE_DIR              if set and non-empty
else $XDG_CACHE_HOME/meerkat-beams   if $XDG_CACHE_HOME set and non-empty
else $HOME/.cache/meerkat-beams
```

`cache_root()` calls `Path.mkdir(parents=True, exist_ok=True)` on the
resolved directory and returns it.

## Cache layout

```
<root>/
  inputs/
    MeerKAT_<BAND>.zarr/        # mean-beam zarr, decompressed gdrive payload
  bds/
    MeerKAT_<BAND>.bds.zarr/    # compressed BDS, output of mdv_beams_to_bds
```

Sibling `*.partial/` directories may exist transiently during
downloads/conversions and are cleaned up automatically.

## `ensure_band_bds` algorithm

```python
def ensure_band_bds(band: str) -> str:
    if band not in SUPPORTED_BANDS:
        raise ValueError(f"band must be one of {SUPPORTED_BANDS}, got {band!r}")

    _clear_partials(band)   # belt-and-braces recovery

    bds = bds_path(band)
    if bds.exists():
        return str(bds)

    if not input_zarr_path(band).exists():
        _download_and_extract(band)

    _convert_to_bds(band)
    return str(bds)
```

Atomicity is provided by `os.replace()` on the `.partial` directory, which
is atomic on POSIX same-filesystem. Both `_download_and_extract` and
`_convert_to_bds` wrap their work in `try/finally` that removes the
`.partial` dir on failure with `shutil.rmtree(..., ignore_errors=True)`.

## `BeamWizard` API change

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
    if image_name is None:
        raise ValueError("image_name is required")
    if band is not None:
        from meerkat_beams.cache import ensure_band_bds
        bds_name = ensure_band_bds(band)
    # ... existing __init__ body unchanged ...
```

The `cache` import is local to keep `utils.py` import-light.

## Backward compatibility

- All existing positional callers — `bds_to_xradio.py:52`,
  `utils.py:870` (`collect_beam_gain_to_source`),
  `tests/test_rendered_beam.py:39`,
  `tests/test_beam_consistency.py:27`,
  `tests/test_beam_wizard.py:126` — keep working unchanged.
- The new `image_name is None` check is a tightening of an existing latent
  failure (`None.endswith(".fits")` → `AttributeError`). No real callers
  pass `None`.
- `mbeams download-mdv-beams` and `mbeams mdv-beams-to-bds` are not
  modified.

## Failure modes

| Condition | Behaviour |
|---|---|
| `band` not in `SUPPORTED_BANDS` | `ValueError` listing supported bands, before any IO |
| Both/neither of `band` and `bds_name` | `ValueError` from `BeamWizard.__init__` |
| `image_name` missing | `ValueError` from `BeamWizard.__init__` |
| `gdown` not importable | `ImportError` with "install meerkat-beams[full]" message |
| gdrive download fails | exception propagates; `.partial` dir removed in `finally`; cached input zarr (if any) preserved |
| Tarball extraction fails | same as above |
| BDS conversion fails | `.partial` BDS dir removed in `finally`; cached input zarr preserved so retry skips re-download |
| Disk full mid-write | covered by the `.partial`/`finally` pattern |
| Cache root un-writable | `cache_root()` propagates `mkdir` error; message shows resolved path so user can set `MBEAMS_CACHE_DIR` |
| Stale `*.partial` from prior killed process | swept by `_clear_partials` at the top of `ensure_band_bds`, with `log.warning` |
| Two processes warming same band concurrently | not guarded — documented limitation, locking deferred |

## Dependencies

- `gdown` moves from `[dependency-groups.test]` to
  `[project.optional-dependencies.full]`. It is now a runtime dependency
  when `BeamWizard(band=...)` is used.
- No other new dependencies.

## Tests

### New: `tests/test_cache.py` (hermetic, unit-marked)

- `cache_root` env precedence: monkeypatch `MBEAMS_CACHE_DIR`,
  `XDG_CACHE_HOME`, and `HOME`; assert correct resolution and fallback.
- `SUPPORTED_BANDS == tuple(BAND_GDRIVE_IDS.keys())` typo guard.
- `ensure_band_bds` rejects unknown band with `ValueError`.
- `ensure_band_bds` short-circuits when BDS exists: pre-create
  `bds/MeerKAT_U.bds.zarr/.zgroup`, monkeypatch `_download_and_extract`
  and `_convert_to_bds` to raise; assert they aren't called.
- `ensure_band_bds` skips download when input zarr exists: pre-create
  `inputs/MeerKAT_U.zarr/`, monkeypatch download to raise and conversion
  to a stub; assert conversion ran and no download.
- Stale `.partial` cleanup: pre-create both `*.partial/` dirs with junk;
  call `ensure_band_bds`; assert they are removed and a warning was
  logged.
- `_convert_to_bds` failure cleans `.partial`: monkeypatch
  `mdv_beams_to_bds` to raise; assert partial BDS dir is gone and the
  cached input zarr is preserved.

### Updated: `tests/test_beam_wizard.py`

- `BeamWizard(band=..., bds_name=...)` raises `ValueError` (XOR).
- `BeamWizard()` with neither raises `ValueError` (XOR).
- `BeamWizard(bds_name=..., image_name=None)` raises `ValueError`.
- New integration-marked test: `BeamWizard(band="L", image_name=...)` runs
  `ensure_band_bds("L")` for real and opens the result. Skipped when
  `MBEAMS_OFFLINE=1` is set, for air-gapped CI.

### Updated: `tests/conftest.py`

- Drop the bespoke `gdown.download` block and the `BAND_INPUT_ZARR` map.
- `pytest_sessionstart` calls `cache.ensure_band_bds("L")` if
  `cache.bds_path("L")` is missing. The L-band test MS is what the
  integration tests use.

### Updated: `tests/test_mdv_beams_to_bds.py`

- `_input_zarr_path(band)` reads from `cache.input_zarr_path(band)` and
  returns `None` if absent.
- No longer imports `BAND_INPUT_ZARR` from `tests/conftest.py`.

The existing `tests/data/MeerKAT_U.zarr` on disk (if present from prior
runs) becomes orphaned — users can delete it manually. No test-time
cleanup is added.

## Documentation

- `CLAUDE.md` "Architecture" section: add `cache.py` to the module list.
- `CLAUDE.md` "Domain concepts": brief note that `BeamWizard(band=...)`
  is the typical entry point and that the cache lives at
  `~/.cache/meerkat-beams` by default.
- Docstring on `cache.py` covers the cache layout, env vars, and the
  deferred-locking limitation.

## Out of scope (could be added later)

- `fcntl.flock`-based per-band locking for concurrent warm-ups.
- A `mbeams fetch-band <BAND>` CLI command for explicit pre-warming.
- A `mbeams cache-clear` / `mbeams cache-path` CLI surface.
- SARAO archive fallback for S1, S2, S3 (currently unsupported in
  `SUPPORTED_BANDS`).
- Pre-built BDS artifacts hosted on Google Drive (would let us skip the
  local conversion step entirely).
