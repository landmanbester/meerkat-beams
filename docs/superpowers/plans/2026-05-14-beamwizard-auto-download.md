# BeamWizard auto-download Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `BeamWizard(band=..., image_name=...)` download an MdV mean-beam zarr from Google Drive on demand, build a compressed BDS locally, and cache both in a user cache dir for reuse in later sessions.

**Architecture:** New `src/meerkat_beams/cache.py` owns the band registry, cache layout, and `ensure_band_bds(band) -> str`. `BeamWizard.__init__` gains a keyword-only `band=` parameter that resolves through it. `tests/conftest.py` is rewritten to use the same machinery (single source of truth for gdrive IDs). The cache root is `MBEAMS_CACHE_DIR` or `XDG_CACHE_HOME/meerkat-beams` or `~/.cache/meerkat-beams`. Atomicity comes from writing to sibling `.partial` directories and `os.replace`-ing on success; `.partial` leftovers are swept at the top of `ensure_band_bds`.

**Tech Stack:** Python ≥3.10, `gdown` (now a runtime dep under the `[full]` extra), `tarfile`, `pathlib`, `shutil`, the existing `mdv_beams_to_bds` core function. Tests use pytest with the existing `unit` / `integration` markers.

**Spec:** `docs/superpowers/specs/2026-05-14-beamwizard-auto-download-design.md`

---

## File Map

**Create:**
- `src/meerkat_beams/cache.py` — band registry, path helpers, `ensure_band_bds`, internal `_download_and_extract` / `_convert_to_bds` / `_clear_partials`.
- `tests/test_cache.py` — hermetic unit tests for `cache.py` (mocked download/convert).

**Modify:**
- `src/meerkat_beams/utils.py` — `BeamWizard.__init__` gets a keyword-only `band=` parameter and XOR validation.
- `pyproject.toml` — move `gdown` from `[dependency-groups.test]` to `[project.optional-dependencies.full]`.
- `tests/conftest.py` — drop bespoke `gdown` block and `BAND_INPUT_ZARR`; delegate to `cache.ensure_band_bds("L")`.
- `tests/test_mdv_beams_to_bds.py` — read inputs from `cache.input_zarr_path(band)`; drop import of `BAND_INPUT_ZARR`.
- `tests/test_beam_wizard.py` — add XOR validation tests and an optional online integration test.
- `CLAUDE.md` — note `cache.py` in the architecture section and the `BeamWizard(band=...)` entry point.

---

## Task 1: cache.py band registry and path helpers (TDD)

**Files:**
- Create: `src/meerkat_beams/cache.py`
- Create: `tests/test_cache.py`

- [ ] **Step 1: Write failing tests for the registry and path helpers**

Create `tests/test_cache.py`:

```python
"""
Hermetic unit tests for meerkat_beams.cache.

No network, no real conversion. ensure_band_bds is exercised with the
download+convert internals monkeypatched.
"""

from pathlib import Path

import pytest

from meerkat_beams import cache


@pytest.mark.unit
def test_supported_bands_matches_registry():
    assert cache.SUPPORTED_BANDS == tuple(cache.BAND_GDRIVE_IDS.keys())


@pytest.mark.unit
def test_registry_contains_expected_bands():
    assert set(cache.BAND_GDRIVE_IDS) == {"U", "L", "S0", "S4"}
    for band, gid in cache.BAND_GDRIVE_IDS.items():
        assert isinstance(gid, str) and gid, f"empty gdrive id for {band}"


@pytest.mark.unit
def test_input_zarr_path_under_cache_root(tmp_path, monkeypatch):
    monkeypatch.setenv("MBEAMS_CACHE_DIR", str(tmp_path))
    assert cache.input_zarr_path("U") == tmp_path / "inputs" / "MeerKAT_U.zarr"


@pytest.mark.unit
def test_bds_path_under_cache_root(tmp_path, monkeypatch):
    monkeypatch.setenv("MBEAMS_CACHE_DIR", str(tmp_path))
    assert cache.bds_path("L") == tmp_path / "bds" / "MeerKAT_L.bds.zarr"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cache.py -v`
Expected: ImportError or collection error — `meerkat_beams.cache` does not exist.

- [ ] **Step 3: Create the minimal cache.py module**

Create `src/meerkat_beams/cache.py`:

```python
"""
On-demand download + cache of MdV mean-beam zarrs and the BDS files
built from them.

The cache lives under ``cache_root()``:

    <root>/inputs/MeerKAT_<BAND>.zarr/   # mean-beam zarr from gdrive
    <root>/bds/MeerKAT_<BAND>.bds.zarr/  # compressed BDS, built locally

Cache root resolution:

    MBEAMS_CACHE_DIR              if set and non-empty
    $XDG_CACHE_HOME/meerkat-beams if XDG_CACHE_HOME set and non-empty
    $HOME/.cache/meerkat-beams    otherwise

Concurrent first-time downloads of the same band from multiple processes
are not guarded. Warm the cache from a single process.
"""

import os
from pathlib import Path

BAND_GDRIVE_IDS: dict[str, str] = {
    "U": "105JWCFo4R-Qo6wHCCkhPm7ZhOSlUaoPx",
    "L": "1dAVD5sE-9fL1kGTjlpaXtI1lOBHJH19K",
    "S0": "1UN5slkHYfXD_MGUZaKFH-UBalgqiepfP",
    "S4": "1-8eg7cCZO4HwTdXW5F55ftmJPOSj3qFV",
}
SUPPORTED_BANDS: tuple[str, ...] = tuple(BAND_GDRIVE_IDS.keys())


def cache_root() -> Path:
    explicit = os.environ.get("MBEAMS_CACHE_DIR")
    if explicit:
        root = Path(explicit)
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        root = Path(xdg) / "meerkat-beams" if xdg else Path.home() / ".cache" / "meerkat-beams"
    root.mkdir(parents=True, exist_ok=True)
    return root


def input_zarr_path(band: str) -> Path:
    return cache_root() / "inputs" / f"MeerKAT_{band}.zarr"


def bds_path(band: str) -> Path:
    return cache_root() / "bds" / f"MeerKAT_{band}.bds.zarr"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cache.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/meerkat_beams/cache.py tests/test_cache.py
git commit -m "feat(cache): add band registry and path helpers

Introduce meerkat_beams.cache with BAND_GDRIVE_IDS, SUPPORTED_BANDS,
cache_root(), input_zarr_path(), and bds_path(). Cache root resolves
from MBEAMS_CACHE_DIR, then XDG_CACHE_HOME/meerkat-beams, then
~/.cache/meerkat-beams."
```

---

## Task 2: cache_root env precedence

**Files:**
- Modify: `tests/test_cache.py`

- [ ] **Step 1: Add failing precedence tests**

Append to `tests/test_cache.py`:

```python
@pytest.mark.unit
def test_cache_root_prefers_mbeams_cache_dir(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit"
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("MBEAMS_CACHE_DIR", str(explicit))
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg))
    assert cache.cache_root() == explicit
    assert explicit.is_dir()


@pytest.mark.unit
def test_cache_root_falls_back_to_xdg(tmp_path, monkeypatch):
    xdg = tmp_path / "xdg"
    monkeypatch.delenv("MBEAMS_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg))
    assert cache.cache_root() == xdg / "meerkat-beams"
    assert (xdg / "meerkat-beams").is_dir()


@pytest.mark.unit
def test_cache_root_falls_back_to_home(tmp_path, monkeypatch):
    monkeypatch.delenv("MBEAMS_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert cache.cache_root() == tmp_path / ".cache" / "meerkat-beams"


@pytest.mark.unit
def test_cache_root_empty_env_vars_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv("MBEAMS_CACHE_DIR", "")
    monkeypatch.setenv("XDG_CACHE_HOME", "")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert cache.cache_root() == tmp_path / ".cache" / "meerkat-beams"
```

- [ ] **Step 2: Run and verify pass**

Run: `uv run pytest tests/test_cache.py -v`
Expected: 8 passed. (`cache_root` already implements the precedence; these tests just pin it.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_cache.py
git commit -m "test(cache): pin cache_root env-var precedence"
```

---

## Task 3: ensure_band_bds — validation and short-circuit when BDS exists

**Files:**
- Modify: `src/meerkat_beams/cache.py`
- Modify: `tests/test_cache.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_cache.py`:

```python
@pytest.mark.unit
def test_ensure_band_bds_rejects_unknown_band(tmp_path, monkeypatch):
    monkeypatch.setenv("MBEAMS_CACHE_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="band must be one of"):
        cache.ensure_band_bds("Q")


@pytest.mark.unit
def test_ensure_band_bds_short_circuits_when_bds_exists(tmp_path, monkeypatch):
    monkeypatch.setenv("MBEAMS_CACHE_DIR", str(tmp_path))
    bds = cache.bds_path("U")
    bds.mkdir(parents=True)
    (bds / ".zgroup").write_text("{}")

    def boom(*a, **kw):
        raise AssertionError("must not be called when BDS already exists")

    monkeypatch.setattr(cache, "_download_and_extract", boom)
    monkeypatch.setattr(cache, "_convert_to_bds", boom)
    assert cache.ensure_band_bds("U") == str(bds)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cache.py -v`
Expected: `AttributeError: module 'meerkat_beams.cache' has no attribute 'ensure_band_bds'` (or similar).

- [ ] **Step 3: Add `ensure_band_bds` and internal stubs to cache.py**

Append to `src/meerkat_beams/cache.py`:

```python
def ensure_band_bds(band: str) -> str:
    """Return a local BDS path for ``band``, downloading and converting as needed."""
    if band not in SUPPORTED_BANDS:
        raise ValueError(f"band must be one of {SUPPORTED_BANDS}, got {band!r}")

    bds = bds_path(band)
    if bds.exists():
        return str(bds)

    if not input_zarr_path(band).exists():
        _download_and_extract(band)

    _convert_to_bds(band)
    return str(bds)


def _download_and_extract(band: str) -> None:
    raise NotImplementedError


def _convert_to_bds(band: str) -> None:
    raise NotImplementedError
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cache.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add src/meerkat_beams/cache.py tests/test_cache.py
git commit -m "feat(cache): add ensure_band_bds with validation + short-circuit

Adds ensure_band_bds(band) that returns the cached BDS path when it
already exists. Download and conversion are stubbed (NotImplementedError)
and filled in by subsequent commits."
```

---

## Task 4: Skip download when input zarr already exists

**Files:**
- Modify: `tests/test_cache.py`

- [ ] **Step 1: Add failing test**

Append to `tests/test_cache.py`:

```python
@pytest.mark.unit
def test_ensure_band_bds_skips_download_when_input_exists(tmp_path, monkeypatch):
    monkeypatch.setenv("MBEAMS_CACHE_DIR", str(tmp_path))
    inp = cache.input_zarr_path("U")
    inp.mkdir(parents=True)
    (inp / ".zgroup").write_text("{}")

    def must_not_download(*a, **kw):
        raise AssertionError("download must not run when input already exists")

    convert_calls = []

    def stub_convert(band):
        convert_calls.append(band)
        out = cache.bds_path(band)
        out.mkdir(parents=True)
        (out / ".zgroup").write_text("{}")

    monkeypatch.setattr(cache, "_download_and_extract", must_not_download)
    monkeypatch.setattr(cache, "_convert_to_bds", stub_convert)

    result = cache.ensure_band_bds("U")
    assert result == str(cache.bds_path("U"))
    assert convert_calls == ["U"]
```

- [ ] **Step 2: Run to verify pass**

Run: `uv run pytest tests/test_cache.py::test_ensure_band_bds_skips_download_when_input_exists -v`
Expected: PASS. (`ensure_band_bds` already implements this branch; this test pins it.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_cache.py
git commit -m "test(cache): pin download-skip when input zarr is already cached"
```

---

## Task 5: _clear_partials sweep

**Files:**
- Modify: `src/meerkat_beams/cache.py`
- Modify: `tests/test_cache.py`

- [ ] **Step 1: Add failing test**

Append to `tests/test_cache.py`:

```python
@pytest.mark.unit
def test_ensure_band_bds_clears_stale_partials(tmp_path, monkeypatch, caplog):
    import logging

    monkeypatch.setenv("MBEAMS_CACHE_DIR", str(tmp_path))

    inp = cache.input_zarr_path("U")
    out = cache.bds_path("U")
    stale_input = inp.with_name(inp.name + ".partial")
    stale_bds = out.with_name(out.name + ".partial")
    stale_input.mkdir(parents=True)
    (stale_input / "junk").write_text("x")
    stale_bds.mkdir(parents=True)
    (stale_bds / "junk").write_text("x")

    def stub_download(band):
        inp = cache.input_zarr_path(band)
        inp.mkdir(parents=True)
        (inp / ".zgroup").write_text("{}")

    def stub_convert(band):
        out = cache.bds_path(band)
        out.mkdir(parents=True)
        (out / ".zgroup").write_text("{}")

    monkeypatch.setattr(cache, "_download_and_extract", stub_download)
    monkeypatch.setattr(cache, "_convert_to_bds", stub_convert)

    with caplog.at_level(logging.WARNING, logger="meerkat_beams"):
        cache.ensure_band_bds("U")

    assert not stale_input.exists()
    assert not stale_bds.exists()
    assert any("partial" in r.message.lower() for r in caplog.records)
```

Note: `Path.with_suffix` only replaces the *last* suffix. `cache.input_zarr_path("U")` is `.../MeerKAT_U.zarr`, so `.with_suffix(".zarr.partial")` gives `.../MeerKAT_U.zarr.partial`. Similarly for `MeerKAT_U.bds.zarr` → `MeerKAT_U.bds.zarr.partial`.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_cache.py::test_ensure_band_bds_clears_stale_partials -v`
Expected: FAIL — partials still exist after `ensure_band_bds`.

- [ ] **Step 3: Add `_clear_partials` and wire it into `ensure_band_bds`**

Edit `src/meerkat_beams/cache.py` — add `shutil` import at the top:

```python
import os
import shutil
from pathlib import Path
```

Add at the bottom of the file:

```python
def _partial(path: Path) -> Path:
    """Sibling .partial directory next to ``path``."""
    return path.with_name(path.name + ".partial")


def _clear_partials(band: str) -> None:
    from meerkat_beams.utils import log
    for p in (_partial(input_zarr_path(band)), _partial(bds_path(band))):
        if p.exists():
            log.warning(f"removing stale partial cache dir {p}")
            shutil.rmtree(p, ignore_errors=True)
```

Replace `ensure_band_bds`:

```python
def ensure_band_bds(band: str) -> str:
    """Return a local BDS path for ``band``, downloading and converting as needed."""
    if band not in SUPPORTED_BANDS:
        raise ValueError(f"band must be one of {SUPPORTED_BANDS}, got {band!r}")

    _clear_partials(band)

    bds = bds_path(band)
    if bds.exists():
        return str(bds)

    if not input_zarr_path(band).exists():
        _download_and_extract(band)

    _convert_to_bds(band)
    return str(bds)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_cache.py -v`
Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add src/meerkat_beams/cache.py tests/test_cache.py
git commit -m "feat(cache): sweep stale .partial directories on entry

Adds _clear_partials, run at the top of ensure_band_bds, removing
any leftover *.partial cache directories from previously killed
processes and logging a warning."
```

---

## Task 6: _download_and_extract — gdown + tar + atomic rename

**Files:**
- Modify: `src/meerkat_beams/cache.py`
- Modify: `tests/test_cache.py`

- [ ] **Step 1: Add failing test for the failure path (no network needed)**

Append to `tests/test_cache.py`:

```python
import io
import tarfile


def _make_fake_tarball(tar_path: Path, member_name: str):
    """Build a real .tar.gz on disk containing a single zarr-shaped directory."""
    payload_dir = tar_path.parent / "_stage"
    payload_dir.mkdir(parents=True, exist_ok=True)
    zarr_dir = payload_dir / member_name
    zarr_dir.mkdir(parents=True, exist_ok=True)
    (zarr_dir / ".zgroup").write_text("{}")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(zarr_dir, arcname=member_name)


@pytest.mark.unit
def test_download_and_extract_writes_atomic(tmp_path, monkeypatch):
    monkeypatch.setenv("MBEAMS_CACHE_DIR", str(tmp_path))

    def fake_gdown_download(id, output, quiet):  # noqa: A002
        _make_fake_tarball(Path(output), "MeerKAT_U.zarr")

    monkeypatch.setattr(cache, "_gdown_download", fake_gdown_download)
    cache._download_and_extract("U")

    inp = cache.input_zarr_path("U")
    assert inp.is_dir()
    assert (inp / ".zgroup").exists()
    assert not cache._partial(inp).exists()


@pytest.mark.unit
def test_download_and_extract_failure_cleans_partial(tmp_path, monkeypatch):
    monkeypatch.setenv("MBEAMS_CACHE_DIR", str(tmp_path))

    def boom(id, output, quiet):  # noqa: A002
        Path(output).write_text("not a tarball")  # download "succeeds" with junk

    monkeypatch.setattr(cache, "_gdown_download", boom)
    with pytest.raises(Exception):
        cache._download_and_extract("U")

    assert not cache.input_zarr_path("U").exists()
    assert not cache._partial(cache.input_zarr_path("U")).exists()


@pytest.mark.unit
def test_download_and_extract_missing_gdown(tmp_path, monkeypatch):
    monkeypatch.setenv("MBEAMS_CACHE_DIR", str(tmp_path))

    def raise_import(*a, **kw):
        raise ImportError("gdown not available")

    monkeypatch.setattr(cache, "_gdown_download", raise_import)
    with pytest.raises(ImportError, match=r"meerkat-beams\[full\]"):
        cache._download_and_extract("U")
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_cache.py -v`
Expected: 3 new tests fail with `NotImplementedError` (and `_gdown_download` attribute missing).

- [ ] **Step 3: Implement `_download_and_extract`**

Add `import tarfile` near the existing imports at the top of
`src/meerkat_beams/cache.py`:

```python
import os
import shutil
import tarfile
from pathlib import Path
```

Replace the `_download_and_extract` stub at the bottom of
`src/meerkat_beams/cache.py`:

```python
def _gdown_download(id: str, output: str, quiet: bool) -> None:  # noqa: A002
    """Thin wrapper around gdown.download so tests can monkeypatch it."""
    import gdown  # local import: gdown is a [full] extra
    gdown.download(id=id, output=output, quiet=quiet)


def _download_and_extract(band: str) -> None:
    from meerkat_beams.utils import log

    inp = input_zarr_path(band)
    partial = _partial(inp)
    inp.parent.mkdir(parents=True, exist_ok=True)
    partial.mkdir(parents=True, exist_ok=True)

    tarball = partial.parent / f"MeerKAT_{band}.zarr.tgz"
    gid = BAND_GDRIVE_IDS[band]
    try:
        try:
            log.info(f"downloading MeerKAT_{band}.zarr.tgz from gdrive id {gid}")
            _gdown_download(id=gid, output=str(tarball), quiet=False)
        except ImportError as e:
            raise ImportError(
                f"meerkat-beams was installed without the [full] extra; "
                f"install meerkat-beams[full] to use band={band!r}"
            ) from e

        log.info(f"extracting {tarball} into {partial}")
        with tarfile.open(tarball, "r:gz") as tar:
            tar.extractall(path=partial)

        # Tarball contains a top-level MeerKAT_<BAND>.zarr/ directory; promote it.
        extracted = partial / f"MeerKAT_{band}.zarr"
        if not extracted.is_dir():
            raise RuntimeError(
                f"expected {extracted.name}/ inside tarball but did not find it"
            )
        os.replace(extracted, inp)
    finally:
        if tarball.exists():
            tarball.unlink()
        if partial.exists():
            shutil.rmtree(partial, ignore_errors=True)
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_cache.py -v`
Expected: 15 passed.

- [ ] **Step 5: Commit**

```bash
git add src/meerkat_beams/cache.py tests/test_cache.py
git commit -m "feat(cache): implement _download_and_extract with atomic rename

Downloads MeerKAT_<BAND>.zarr.tgz via gdown into a sibling .partial
directory, extracts it, then os.replace()s the inner zarr to the
final cache path. On any failure the .partial dir and tarball are
removed in finally. A missing gdown surfaces as ImportError pointing
at the [full] extra."
```

---

## Task 7: _convert_to_bds — call mdv_beams_to_bds, atomic rename

**Files:**
- Modify: `src/meerkat_beams/cache.py`
- Modify: `tests/test_cache.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_cache.py`:

```python
@pytest.mark.unit
def test_convert_to_bds_atomic(tmp_path, monkeypatch):
    monkeypatch.setenv("MBEAMS_CACHE_DIR", str(tmp_path))

    inp = cache.input_zarr_path("U")
    inp.mkdir(parents=True)
    (inp / ".zgroup").write_text("{}")

    calls = []

    def stub_mdv(mdv_beams, bds, compress):
        calls.append((mdv_beams, bds, compress))
        Path(bds).mkdir(parents=True)
        (Path(bds) / ".zgroup").write_text("{}")

    monkeypatch.setattr("meerkat_beams.core.mdv_beams_to_bds.mdv_beams_to_bds", stub_mdv)
    cache._convert_to_bds("U")

    out = cache.bds_path("U")
    assert out.is_dir()
    assert (out / ".zgroup").exists()
    assert not cache._partial(out).exists()
    assert calls[0][0] == str(inp)
    assert calls[0][1].endswith(".partial")
    assert calls[0][2] is True


@pytest.mark.unit
def test_convert_to_bds_failure_preserves_input(tmp_path, monkeypatch):
    monkeypatch.setenv("MBEAMS_CACHE_DIR", str(tmp_path))

    inp = cache.input_zarr_path("U")
    inp.mkdir(parents=True)
    (inp / ".zgroup").write_text("{}")

    def boom(mdv_beams, bds, compress):
        Path(bds).mkdir(parents=True)
        (Path(bds) / "half").write_text("x")
        raise RuntimeError("conversion failed")

    monkeypatch.setattr("meerkat_beams.core.mdv_beams_to_bds.mdv_beams_to_bds", boom)
    with pytest.raises(RuntimeError, match="conversion failed"):
        cache._convert_to_bds("U")

    assert not cache.bds_path("U").exists()
    assert not cache._partial(cache.bds_path("U")).exists()
    assert inp.exists(), "input zarr must be preserved on conversion failure"
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_cache.py -v`
Expected: 2 new tests fail with `NotImplementedError`.

- [ ] **Step 3: Implement `_convert_to_bds`**

Replace the `_convert_to_bds` stub in `src/meerkat_beams/cache.py`:

```python
def _convert_to_bds(band: str) -> None:
    from meerkat_beams.core.mdv_beams_to_bds import mdv_beams_to_bds
    from meerkat_beams.utils import log

    inp = input_zarr_path(band)
    out = bds_path(band)
    partial = _partial(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if partial.exists():
        shutil.rmtree(partial, ignore_errors=True)

    log.info(f"converting {inp} -> {out} (via .partial)")
    try:
        mdv_beams_to_bds(mdv_beams=str(inp), bds=str(partial), compress=True)
        os.replace(partial, out)
    finally:
        if partial.exists():
            shutil.rmtree(partial, ignore_errors=True)
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_cache.py -v`
Expected: 17 passed.

- [ ] **Step 5: Commit**

```bash
git add src/meerkat_beams/cache.py tests/test_cache.py
git commit -m "feat(cache): implement _convert_to_bds with atomic rename

Runs mdv_beams_to_bds() into a .partial directory and os.replace()s
it into the final cache slot on success. The cached input zarr is
preserved on failure so retries skip re-downloading."
```

---

## Task 8: BeamWizard band= keyword-only parameter

**Files:**
- Modify: `src/meerkat_beams/utils.py`
- Modify: `tests/test_beam_wizard.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_beam_wizard.py`:

```python
@pytest.mark.unit
def test_beam_wizard_requires_one_of_bds_or_band(tmp_path):
    with pytest.raises(ValueError, match="exactly one of bds_name or band"):
        BeamWizard(image_name=str(tmp_path / "x.fits"))


@pytest.mark.unit
def test_beam_wizard_rejects_both_bds_and_band(tmp_path):
    with pytest.raises(ValueError, match="exactly one of bds_name or band"):
        BeamWizard(bds_name="some.bds.zarr", image_name=str(tmp_path / "x.fits"), band="U")


@pytest.mark.unit
def test_beam_wizard_requires_image_name():
    with pytest.raises(ValueError, match="image_name is required"):
        BeamWizard(bds_name="some.bds.zarr")


@pytest.mark.unit
def test_beam_wizard_band_routes_through_cache(tmp_path, monkeypatch):
    """band='U' must call ensure_band_bds and feed the result to the existing init."""
    from meerkat_beams import cache

    monkeypatch.setenv("MBEAMS_CACHE_DIR", str(tmp_path))

    fake_bds = tmp_path / "fake.bds.zarr"
    fake_image = tmp_path / "synthetic.fits"
    _build_bds(fake_bds)
    _build_image(fake_image)

    calls = []

    def stub_ensure(band):
        calls.append(band)
        return str(fake_bds)

    monkeypatch.setattr(cache, "ensure_band_bds", stub_ensure)
    bw = BeamWizard(image_name=str(fake_image), band="U")
    assert calls == ["U"]
    assert bw.bds is not None
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_beam_wizard.py -v -k "band or image_name"`
Expected: 4 failures (TypeError for missing positional or unexpected `band` kwarg).

- [ ] **Step 3: Modify BeamWizard.__init__ in utils.py**

In `src/meerkat_beams/utils.py`, replace the existing signature and the first few lines of `BeamWizard.__init__` (currently at `utils.py:96-99`):

Old:

```python
    def __init__(self, bds_name: str, image_name: str):
        self.log = log
        log.info(f"opening BDS {bds_name}")
        self.bds = xarray.open_zarr(bds_name)
```

New:

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
            from meerkat_beams import cache
            bds_name = cache.ensure_band_bds(band)
        self.log = log
        log.info(f"opening BDS {bds_name}")
        self.bds = xarray.open_zarr(bds_name)
```

(`Optional` is already imported on `utils.py:17`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_beam_wizard.py -v`
Expected: all original tests still pass + 4 new pass.

Then run the wider test suite to confirm no positional-call regressions:

```bash
uv run pytest -v
```

Expected: all previously passing tests still pass; integration tests that skip due to env vars continue to skip.

- [ ] **Step 5: Commit**

```bash
git add src/meerkat_beams/utils.py tests/test_beam_wizard.py
git commit -m "feat(BeamWizard): add band= kwarg routed through cache.ensure_band_bds

BeamWizard now accepts a keyword-only band= argument. Exactly one of
band or bds_name must be supplied; image_name becomes a required kwarg.
When band is set, cache.ensure_band_bds(band) populates and returns
the cached BDS path before the existing init body runs."
```

---

## Task 9: Move gdown to the [full] extra

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Edit pyproject.toml**

In `pyproject.toml`:

1. Add `"gdown>=5.0.0",` to the `[project.optional-dependencies.full]` list (`pyproject.toml:24-36`), e.g. after `"wget",`.
2. Remove `"gdown>=5.0.0",` from `[dependency-groups.test]` (`pyproject.toml:88-91`).

After: `[project.optional-dependencies.full]` should contain `gdown>=5.0.0`, and `[dependency-groups.test]` should only contain pytest.

- [ ] **Step 2: Refresh the lockfile**

Run: `uv sync --group dev --group test --extra full`
Expected: lockfile updated; no errors.

- [ ] **Step 3: Sanity-check the import is reachable at runtime**

Run: `uv run python -c "from meerkat_beams.cache import _gdown_download; _gdown_download.__doc__"`
Expected: no error.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: move gdown from [test] to [full] extra

gdown is now a runtime dependency when BeamWizard(band=...) auto-downloads
mean-beam zarrs, so it belongs in the [full] extra alongside wget."
```

---

## Task 10: Rewrite tests/conftest.py to use the cache

**Files:**
- Modify: `tests/conftest.py`

- [ ] **Step 1: Read current conftest.py**

The file currently downloads a tarball into `tests/data/` and defines `BAND_INPUT_ZARR`. Some `gdrive_id_*` variables and a `test_ms_gdrive_id` have been added by the user. Preserve `test_ms_gdrive_id` (orthogonal to this work).

- [ ] **Step 2: Replace conftest.py**

Rewrite `tests/conftest.py`:

```python
"""
pytest session-level setup.

Ensures the L-band BDS cache is populated before tests run so the
integration suite (which uses an L-band test MS) has data available.
"""

from pathlib import Path

from meerkat_beams import cache

test_root_path = Path(__file__).resolve().parent
test_data_path = test_root_path / "data"
test_data_path.mkdir(parents=True, exist_ok=True)

# Kept for fixtures that pair the BDS with a test measurement set.
# https://drive.google.com/file/d/1mCTrC3IbMUqu0Adu1DWOjhwvzQS6gseo/view?usp=drive_link
test_ms_gdrive_id = "1mCTrC3IbMUqu0Adu1DWOjhwvzQS6gseo"


def pytest_sessionstart(session):
    """Populate the L-band cache once per session if it isn't there yet."""
    if cache.bds_path("L").exists():
        print(f"L-band BDS already cached at {cache.bds_path('L')}.")
        return
    print("L-band BDS not in cache - downloading and converting...")
    cache.ensure_band_bds("L")
    print(f"L-band BDS ready at {cache.bds_path('L')}.")
```

- [ ] **Step 3: Smoke-check pytest collection**

Run: `uv run pytest --collect-only tests/test_cache.py tests/test_beam_wizard.py -q`
Expected: collection succeeds (this only loads conftest; it does not trigger the actual L-band download because `pytest_sessionstart` only fires on a real test run, not `--collect-only`).

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py
git commit -m "test: route conftest session setup through cache.ensure_band_bds

Drops the bespoke gdown block and BAND_INPUT_ZARR map. The L-band BDS
cache is populated at session start if missing, matching the band the
integration tests actually use."
```

---

## Task 11: Update tests/test_mdv_beams_to_bds.py to read from cache

**Files:**
- Modify: `tests/test_mdv_beams_to_bds.py`

- [ ] **Step 1: Replace the import and helper**

In `tests/test_mdv_beams_to_bds.py`:

Replace this import block (`test_mdv_beams_to_bds.py:26`):

```python
from tests.conftest import BAND_INPUT_ZARR, test_data_path
```

with:

```python
from meerkat_beams import cache
```

Replace `_input_zarr_path` (`test_mdv_beams_to_bds.py:46-52`):

```python
def _input_zarr_path(band: str) -> Path | None:
    """Return the cached input zarr path, or None if absent."""
    if band not in cache.SUPPORTED_BANDS:
        return None
    p = cache.input_zarr_path(band)
    return p if p.exists() else None
```

- [ ] **Step 2: Smoke-check pytest collection**

Run: `uv run pytest --collect-only tests/test_mdv_beams_to_bds.py -q`
Expected: collection succeeds; tests will skip at runtime for bands without cached input zarrs and without `MBEAMS_REFERENCE_BDS_<BAND>` set.

- [ ] **Step 3: Commit**

```bash
git add tests/test_mdv_beams_to_bds.py
git commit -m "test(mdv_beams_to_bds): read input zarrs from the cache

Use meerkat_beams.cache.input_zarr_path() instead of the
tests/data/<file> location, so the regression suite and the runtime
auto-download path share a single source of truth."
```

---

## Task 12: Optional online integration test

**Files:**
- Modify: `tests/test_beam_wizard.py`

- [ ] **Step 1: Add an integration-marked test**

Append to `tests/test_beam_wizard.py`:

```python
@pytest.mark.integration
def test_beam_wizard_band_l_end_to_end(tmp_path):
    """End-to-end: BeamWizard(band='L', ...) opens a real cached BDS.

    Skipped when MBEAMS_OFFLINE=1 (air-gapped CI). Reuses whatever the
    cache already contains; populates it via ensure_band_bds if needed.
    """
    import os

    if os.environ.get("MBEAMS_OFFLINE") == "1":
        pytest.skip("MBEAMS_OFFLINE=1 set")

    fits_path = tmp_path / "synthetic.fits"
    _build_image(fits_path)
    bw = BeamWizard(image_name=str(fits_path), band="L")
    assert "FREQ" in bw.bds.coords
    assert bw.bds.attrs["dx"] > 0
```

- [ ] **Step 2: Run the integration test once (network)**

Run: `uv run pytest tests/test_beam_wizard.py::test_beam_wizard_band_l_end_to_end -v`
Expected: PASS on a networked host (uses cache if already warmed by conftest). If first time and slow, that is expected.

Run also: `MBEAMS_OFFLINE=1 uv run pytest tests/test_beam_wizard.py::test_beam_wizard_band_l_end_to_end -v`
Expected: SKIPPED.

- [ ] **Step 3: Commit**

```bash
git add tests/test_beam_wizard.py
git commit -m "test(BeamWizard): add online end-to-end test for band='L'

Integration-marked test that exercises the full cache pipeline.
Skipped when MBEAMS_OFFLINE=1 so it can be disabled in air-gapped CI."
```

---

## Task 13: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update the architecture tree**

In `CLAUDE.md`, find the architecture block (around line 9-14):

```
src/meerkat_beams/
├── utils.py                # shared: BeamWizard, PowerBeam, logging, zarr constants
├── cabs/<cmd>.yml          # Stimela cab definitions (one per command)
├── cli/<cmd>.py            # thin Typer wrappers (one per command)
└── core/<cmd>.py           # plain-Python implementations (one per command)
```

Replace with:

```
src/meerkat_beams/
├── utils.py                # shared: BeamWizard, PowerBeam, logging, zarr constants
├── cache.py                # on-demand download + BDS cache (per band)
├── cabs/<cmd>.yml          # Stimela cab definitions (one per command)
├── cli/<cmd>.py            # thin Typer wrappers (one per command)
└── core/<cmd>.py           # plain-Python implementations (one per command)
```

- [ ] **Step 2: Add a "Cache" subsection under "Key abstractions in `utils.py`"**

Insert a new top-level section after the `enrich_bds_xradio` paragraph (before `### Logging`):

```markdown
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
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(CLAUDE.md): document the cache.py module and BeamWizard band= kwarg"
```

---

## Final verification

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -v`
Expected: same set of skips/passes as before this change, plus the new unit tests under `tests/test_cache.py` and the four new BeamWizard validation tests.

- [ ] **Step 2: Lint and format check**

Run: `uv run ruff check .`
Expected: no issues.

Run: `uv run ruff format --check .`
Expected: no formatting changes needed.

- [ ] **Step 3: Smoke-test the new entry point against a real image**

Only if you have a local FITS image handy:

```bash
uv run python -c "
from meerkat_beams.utils import BeamWizard
bw = BeamWizard(image_name='/path/to/image.fits', band='L')
print('cached BDS at', bw.bds.attrs.get('freqs', 'no freqs')[:3])
"
```

Expected: prints first three frequency values from the cached L-band BDS.
