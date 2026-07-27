# meerkat-beams transition progress

## Status: In Progress

Transitioning suricat-beams to hip-cargo format as meerkat-beams.

## Completed

### Scaffold (hip-cargo init)
- [x] Project scaffolded with `hip-cargo init`
- [x] CI/CD workflows, Dockerfile, pre-commit hooks, tbump config

### Cab YAML splitting
- [x] Split monolithic `suricat.yml` into per-cab files:
  - `cabs/download_mdv_beams.yml`
  - `cabs/mdv_beams_to_bds.yml`
  - `cabs/bds_to_xradio.yml`
  - `cabs/mdv_to_xradio.yml`

### CLI generation
- [x] Generated CLI wrappers via `hip-cargo generate-function`
- [x] Registered all commands in `cli/__init__.py`

### Core module restructuring
- [x] Created `utils.py` with shared utilities:
  - Logging (log, LOGGER, create_logger, set_console_logging_level)
  - PowerBeam dataclass
  - BeamWizard class (full beam interpolation)
  - enrich_bds_xradio helper
  - collect_beam_gain_to_source (beamplots)
  - ZARR_COMPRESSOR / ZARR_FILTERS constants
- [x] Created one-to-one core modules:
  - `core/download_mdv_beams.py` - plain function, imports from utils
  - `core/mdv_beams_to_bds.py` - plain function, imports from utils
  - `core/bds_to_xradio.py` - plain function, imports BeamWizard from utils
  - `core/mdv_to_xradio.py` - plain function, imports from utils
- [x] CLI wrappers updated to import from one-to-one core modules
- [x] Old monolithic `core/beams.py`, `core/xradio_util.py`, `core/beamplots.py` removed

### Verification
- [x] `uv run ruff check src/ tests/` passes
- [x] `uv run ruff format --check src/ tests/` passes
- [x] `mbeams --help` works
- [x] All subcommand `--help` works
- [x] `uv run pytest tests/test_install.py tests/test_cli.py -v` — 7/7 pass

## Remaining

- [ ] Regenerate cab YAML from CLI wrappers (`hip-cargo generate-cabs`)
- [ ] Port recipes (deferred)
- [ ] Add Ray-based parallelism (future)
- [ ] Integration tests (require beam data files)

## Architecture

```
src/meerkat_beams/
├── __init__.py              # package version
├── _container_image.py      # container image config
├── utils.py                 # shared: BeamWizard, PowerBeam, logging, constants
├── cabs/
│   ├── __init__.py
│   ├── download_mdv_beams.yml
│   ├── mdv_beams_to_bds.yml
│   ├── bds_to_xradio.yml
│   └── mdv_to_xradio.yml
├── cli/
│   ├── __init__.py          # registers all commands on app
│   ├── download_mdv_beams.py
│   ├── mdv_beams_to_bds.py
│   ├── bds_to_xradio.py
│   └── mdv_to_xradio.py
└── core/
    ├── __init__.py
    ├── download_mdv_beams.py
    ├── mdv_beams_to_bds.py
    ├── bds_to_xradio.py
    └── mdv_to_xradio.py
```

The one-to-one mapping: `cabs/foo.yml` <-> `cli/foo.py` <-> `core/foo.py`

---

## Lessons for hip-ify skill refinement

### 1. Three-layer architecture needs explicit guidance

The hip-ify skill currently describes a two-layer model (cli <-> core with cabs alongside).
In practice there are **three layers**:

- **`utils.py`** — Shared classes, constants, logging, helper functions. These are the
  building blocks that core modules compose. Heavy imports (numpy, scipy, astropy, xarray)
  live here.
- **`core/`** — One module per command. Each contains a single plain Python function with
  the same signature as the CLI function but without Typer/Annotated type hints. These
  functions are what users call from Python scripts. They import from `utils.py`.
- **`cli/`** — Thin Typer wrappers. Each uses lazy import to call its corresponding core
  function. Handles type conversions (ListStr -> list, Path -> str, etc.).

The skill should describe this three-layer split explicitly and explain what goes where.

### 2. Core function signatures should mirror CLI signatures

The core functions should have **the same parameter names and defaults** as the CLI
functions, just with plain Python types:
- `Annotated[str, typer.Option(...)]` -> `str`
- `Annotated[ListStr, ...]` -> `Optional[List[str]]`
- `Annotated[File, ...]` -> `str`
- `Annotated[Directory, ...]` -> `str`
- `Annotated[int | None, ...]` -> `Optional[int]`

This makes it trivial to verify the CLI wrapper passes through all args correctly.

### 3. Where to put shared code: utils.py vs core/__init__.py

The previous attempt put logging in `core/__init__.py` and heavy classes in `core/beams.py`.
This broke the one-to-one mapping because `core/beams.py` contained both:
- Implementations of two commands (download_mdv_beams, mdv_beams_to_bds)
- The BeamWizard class used by other commands

**Rule**: If code is used by more than one core module, it belongs in `utils.py`, not in
any core module. `core/__init__.py` should be minimal (just a docstring).

### 4. Type conversions happen in CLI layer only

The CLI wrappers handle all Typer-specific type conversions:
- `ListStr` (comma-separated string) -> `list` via `.split(",")`
- `Path` objects -> `str` via `str(path)`
- `None` handling for optional Path types

Core functions receive plain Python types and don't need to handle Typer conventions.

### 5. Mutable default arguments in core functions

Be careful with `List[str]` defaults in function signatures. Use `None` as default and
set the actual default inside the function body, or accept that ruff/linting may flag
mutable defaults. The CLI layer uses string defaults for ListStr which avoids this issue.

### 6. Naming: _enrich_bds_xradio moved to utils as enrich_bds_xradio

Private helper functions that support a core function but are substantial enough to be
shared should be promoted to utils. Drop the underscore prefix when moving to utils since
it's now part of the public utility API.

### 7. The hip-ify skill should specify the order of operations

1. Scaffold with `hip-cargo init`
2. Split cab YAML into per-cab files in `cabs/`
3. Generate CLI wrappers with `hip-cargo generate-function`
4. Create `utils.py` with shared code (classes, logging, constants)
5. Create one-to-one core modules with plain Python functions
6. Wire CLI -> core imports (lazy imports inside function body)
7. Update dependencies
8. Verify (ruff, pytest, CLI --help)
9. Regenerate cabs (deferred until CLI wrappers are finalized)

Step 4 (create utils.py) was missing from the original skill and is critical.
