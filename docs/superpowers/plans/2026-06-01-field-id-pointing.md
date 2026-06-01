# FIELD_ID-driven field selection with POINTING-table pointing centre — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `scripts/test_beam_orientation.py` select an MS field by a single `--field-id`, deriving the beam pointing centre from the POINTING table (with a manual fallback) and keeping the existing rephasing step correct for a pre-rephased MS.

**Architecture:** `scripts/beam_orientation/ms_io.py` gains a `field_id` parameter, a module-level `ORIGINAL_POINTING` fallback dict, and a two-tier pointing-centre resolver (POINTING-table time-window match → dict fallback) exposed as small, pure-where-possible helpers so the resolution logic is unit-testable without a real MS. `MSBundle` carries a new `pointing_centre`. The script reads the rephasing origin from `bundle.phase_centre` (correct for the pre-rephased MS) and the beam centre from `bundle.pointing_centre`.

**Tech Stack:** Python, numpy, dask-ms (lazy import), astropy, pytest (`unit` marker), ruff.

Spec: `docs/superpowers/specs/2026-06-01-field-id-pointing-design.md`

---

### Task 1: `ORIGINAL_POINTING` fallback dict

**Files:**
- Modify: `scripts/beam_orientation/ms_io.py` (top of module, replacing the comment block)
- Test: `tests/test_beam_orientation_ms_io.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_beam_orientation_ms_io.py`:

```python
"""Unit tests for scripts/beam_orientation/ms_io.py."""

import numpy as np
import pytest
from beam_orientation import ms_io  # noqa: E402


@pytest.mark.unit
def test_original_pointing_table_values():
    """ORIGINAL_POINTING carries the five fields with the documented coords."""
    expected = {
        0: (5.146178203219011, -1.1119958085589738),
        1: (5.146178203219011, -1.1364304180868943),
        2: (5.146178203219011, -1.0875611990310532),
        3: (5.201372059151767, -1.1119958085589738),
        4: (5.090979983963126, -1.1119958085589738),
    }
    assert ms_io.ORIGINAL_POINTING == expected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_beam_orientation_ms_io.py::test_original_pointing_table_values -v`
Expected: FAIL with `AttributeError: module 'beam_orientation.ms_io' has no attribute 'ORIGINAL_POINTING'`

- [ ] **Step 3: Write minimal implementation**

In `scripts/beam_orientation/ms_io.py`, replace the comment block (the `# FIELD_ID, FIELD_NAME, ...` lines, currently lines 16-21) with a real dict plus a short explanatory comment. Keep it directly under the existing imports:

```python
# Original pointing directions (ra_rad, dec_rad) per FIELD_ID for this MS.
# The MS was rephased to the source by an earlier step, which did NOT preserve
# the original pointing directions in the FIELD table; these are the recovered
# values used as a fallback when the POINTING table is unavailable. Specific to
# this MS — see docs/superpowers/specs/2026-06-01-field-id-pointing-design.md.
#   FIELD_ID  name        ra            dec
#   0         Offset1     19:39:25.03   -63:42:45.60
#   1         J1939-6342  19:39:25.03   -65:06:45.60
#   2         Offset2     19:39:25.03   -62:18:45.60
#   3         Offset3     19:52:04.00   -63:42:45.60
#   4         Offset4     19:26:46.00   -63:42:45.60
ORIGINAL_POINTING: dict[int, tuple[float, float]] = {
    0: (5.146178203219011, -1.1119958085589738),
    1: (5.146178203219011, -1.1364304180868943),
    2: (5.146178203219011, -1.0875611990310532),
    3: (5.201372059151767, -1.1119958085589738),
    4: (5.090979983963126, -1.1119958085589738),
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_beam_orientation_ms_io.py::test_original_pointing_table_values -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/beam_orientation/ms_io.py tests/test_beam_orientation_ms_io.py
git commit -m "feat(beam-orientation): add ORIGINAL_POINTING fallback dict to ms_io"
```

---

### Task 2: `_pointing_from_direction` pure helper

**Files:**
- Modify: `scripts/beam_orientation/ms_io.py`
- Test: `tests/test_beam_orientation_ms_io.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_beam_orientation_ms_io.py`:

```python
@pytest.mark.unit
def test_pointing_from_direction_time_window_average():
    """Rows inside [t0, t1] are selected and their (ra, dec) averaged."""
    ptime = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    # shape (row, npoly=1, 2) with last axis = [ra, dec]
    direction = np.zeros((5, 1, 2))
    direction[:, 0, 0] = [10.0, 20.0, 30.0, 40.0, 50.0]  # ra
    direction[:, 0, 1] = [-1.0, -2.0, -3.0, -4.0, -5.0]  # dec
    ra, dec = ms_io._pointing_from_direction(ptime, direction, 1.0, 3.0)
    assert ra == pytest.approx(30.0)   # mean(20, 30, 40)
    assert dec == pytest.approx(-3.0)  # mean(-2, -3, -4)


@pytest.mark.unit
def test_pointing_from_direction_no_rows_in_window_returns_none():
    """A window that matches no POINTING rows returns None (-> fallback)."""
    ptime = np.array([0.0, 1.0, 2.0])
    direction = np.zeros((3, 1, 2))
    assert ms_io._pointing_from_direction(ptime, direction, 100.0, 200.0) is None


@pytest.mark.unit
def test_pointing_from_direction_accepts_2d_direction():
    """A (row, 2) DIRECTION (no polynomial axis) is handled too."""
    ptime = np.array([5.0, 6.0])
    direction = np.array([[1.5, -0.5], [2.5, -1.5]])  # (row, 2)
    ra, dec = ms_io._pointing_from_direction(ptime, direction, 0.0, 10.0)
    assert ra == pytest.approx(2.0)
    assert dec == pytest.approx(-1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_beam_orientation_ms_io.py -k pointing_from_direction -v`
Expected: FAIL with `AttributeError: ... has no attribute '_pointing_from_direction'`

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/beam_orientation/ms_io.py` (below `ORIGINAL_POINTING`, above `read_ms`):

```python
def _pointing_from_direction(
    ptime: np.ndarray,
    direction: np.ndarray,
    t0: float,
    t1: float,
) -> tuple[float, float] | None:
    """Average POINTING DIRECTION over rows whose TIME falls in [t0, t1].

    ``direction`` may be (row, 2) or (row, npoly, 2) with the last axis
    ordered (ra, dec) in radians. Returns ``None`` when no row matches.
    """
    ptime = np.asarray(ptime)
    mask = (ptime >= t0) & (ptime <= t1)
    if not mask.any():
        return None
    d = np.asarray(direction, dtype=float)[mask]
    if d.ndim == 3:  # (row, npoly, 2) -> constant (zeroth-order) term
        d = d[:, 0, :]
    d = d.reshape(-1, 2)
    return float(d[:, 0].mean()), float(d[:, 1].mean())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_beam_orientation_ms_io.py -k pointing_from_direction -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/beam_orientation/ms_io.py tests/test_beam_orientation_ms_io.py
git commit -m "feat(beam-orientation): add _pointing_from_direction time-window helper"
```

---

### Task 3: POINTING-table read + two-tier resolver

**Files:**
- Modify: `scripts/beam_orientation/ms_io.py` (add `log` import, `_read_pointing_table`, `_resolve_pointing_centre`)
- Test: `tests/test_beam_orientation_ms_io.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_beam_orientation_ms_io.py`:

```python
@pytest.mark.unit
def test_resolve_pointing_centre_falls_back_to_dict():
    """An unreadable POINTING table falls back to ORIGINAL_POINTING[field_id]."""
    times = np.array([1.0, 2.0, 3.0])
    pc = ms_io._resolve_pointing_centre("/no/such/ms.ms", 3, times)
    assert pc == ms_io.ORIGINAL_POINTING[3]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_beam_orientation_ms_io.py::test_resolve_pointing_centre_falls_back_to_dict -v`
Expected: FAIL with `AttributeError: ... has no attribute '_resolve_pointing_centre'`

- [ ] **Step 3: Write minimal implementation**

At the top of `scripts/beam_orientation/ms_io.py`, add to the imports (after `import numpy as np`):

```python
from meerkat_beams.utils import log
```

Then add below `_pointing_from_direction`:

```python
def _read_pointing_table(path: str, times: np.ndarray) -> tuple[float, float] | None:
    """Read the original pointing centre from the MS POINTING table.

    Selects POINTING rows whose TIME lies within the selected field's scan
    window (POINTING has no FIELD_ID column) and averages their DIRECTION.
    Returns ``None`` on any failure or when no rows match, so the caller can
    fall back to ORIGINAL_POINTING.
    """
    try:
        from daskms import xds_from_table

        groups = xds_from_table(f"{path}::POINTING")
        if not groups:
            return None
        pnt = groups[0].compute()
        if "DIRECTION" not in pnt or "TIME" not in pnt:
            return None
        return _pointing_from_direction(
            np.asarray(pnt.TIME.values),
            np.asarray(pnt.DIRECTION.values),
            float(np.min(times)),
            float(np.max(times)),
        )
    except Exception as exc:  # noqa: BLE001 - any read failure -> fallback
        log.warning(f"could not read POINTING table ({exc}); using fallback")
        return None


def _resolve_pointing_centre(
    path: str,
    field_id: int,
    times: np.ndarray,
) -> tuple[float, float]:
    """Pointing centre for ``field_id``: POINTING table first, dict fallback."""
    pc = _read_pointing_table(path, times)
    if pc is not None:
        log.info(f"pointing centre for field {field_id} from POINTING table: {pc}")
        return pc
    pc = ORIGINAL_POINTING[field_id]
    log.info(f"pointing centre for field {field_id} from ORIGINAL_POINTING: {pc}")
    return pc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_beam_orientation_ms_io.py::test_resolve_pointing_centre_falls_back_to_dict -v`
Expected: PASS

- [ ] **Step 5: Run the full ms_io unit file**

Run: `uv run pytest tests/test_beam_orientation_ms_io.py -v`
Expected: PASS (all tasks so far — 5 tests)

- [ ] **Step 6: Commit**

```bash
git add scripts/beam_orientation/ms_io.py tests/test_beam_orientation_ms_io.py
git commit -m "feat(beam-orientation): resolve pointing centre from POINTING table with fallback"
```

---

### Task 4: Wire `field_id` and `pointing_centre` into `read_ms`

**Files:**
- Modify: `scripts/beam_orientation/ms_io.py` (`MSBundle` dataclass + `read_ms`)

No new hermetic test: `read_ms` requires a real MS, exercised by the smoke run in Task 6. Tasks 1-3 already pin the resolver logic.

- [ ] **Step 1: Add the `pointing_centre` dataclass field**

In `MSBundle`, add the field immediately after `phase_centre`:

```python
    phase_centre: tuple[float, float]  # (ra_rad, dec_rad), MS phase centre
    pointing_centre: tuple[float, float]  # (ra_rad, dec_rad), original dish pointing
```

- [ ] **Step 2: Parameterize `read_ms` and the taql clauses**

Change the signature:

```python
def read_ms(path: str | Path, field_id: int = 0) -> MSBundle:
```

In the main-table read, change the taql clause to use the parameter:

```python
        taql_where=(f"FIELD_ID == {field_id}"),
```

In the FIELD-subtable read, remove the line `field_id = int(xds.attrs.get("FIELD_ID", 0))` and parameterize the existing clause so it uses the function argument:

```python
    field = xds_from_table(
        f"{path}::FIELD",
        taql_where=(f"SOURCE_ID == {field_id}"),
    )[0].compute()
```

- [ ] **Step 3: Resolve and return the pointing centre**

After `times = np.unique(time_row)` is computed (the `times` array exists before the `return`), resolve the pointing centre. Add, just before the `return MSBundle(`:

```python
    pointing_centre = _resolve_pointing_centre(path, field_id, times)
```

Then add the field to the `MSBundle(...)` construction, right after `phase_centre=(ra_rad, dec_rad),`:

```python
        phase_centre=(ra_rad, dec_rad),
        pointing_centre=pointing_centre,
```

- [ ] **Step 4: Lint**

Run: `uv run ruff check scripts/beam_orientation/ms_io.py && uv run ruff format --check scripts/beam_orientation/ms_io.py`
Expected: no errors. (If format check fails, run `uv run ruff format scripts/beam_orientation/ms_io.py`.)

- [ ] **Step 5: Import smoke check**

Run: `uv run python -c "from beam_orientation import ms_io; import inspect; print('field_id' in inspect.signature(ms_io.read_ms).parameters); print([f for f in ms_io.MSBundle.__dataclass_fields__])"`
Expected: prints `True` and a field list that includes `pointing_centre`.

- [ ] **Step 6: Commit**

```bash
git add scripts/beam_orientation/ms_io.py
git commit -m "feat(beam-orientation): add field_id param and pointing_centre to read_ms"
```

---

### Task 5: Update `test_beam_orientation.py` to use field_id + pointing_centre

**Files:**
- Modify: `scripts/test_beam_orientation.py`

- [ ] **Step 1: Add the `--field-id` argument**

In `parse_args`, add after the `--ms` argument block:

```python
    p.add_argument(
        "--field-id",
        type=int,
        default=0,
        help="FIELD_ID to select; also chooses the original pointing direction.",
    )
```

- [ ] **Step 2: Pass `field_id` to `read_ms`**

Change the `read_ms` call in `main`:

```python
    bundle = read_ms(ms_path, field_id=args.field_id)
```

- [ ] **Step 3: Read the rephasing origin from the MS phase centre**

Replace the hardcoded pointing block (the lines from `# ra_pc, dec_pc = bundle.phase_centre` through `# ra_pc, dec_pc = 5.090979983963126, ...  # Offset4`) with:

```python
    # SIN-projection direction-cosine offset from the MS phase centre. For the
    # pre-rephased MS this centre is already the source, so (dl, dm) ~= 0 and
    # phase_rotate is a no-op; for a non-rephased MS it rephases correctly.
    ra_pc, dec_pc = bundle.phase_centre
```

Leave the `dl = ...`, `dm = ...`, `log.info(...)`, and `vis_rot = phase_rotate.phase_rotate(...)` lines unchanged.

- [ ] **Step 4: Set the beam centre from the pointing centre**

Change the `set_field_centre` call so the beam pointing centre is the original dish pointing rather than the rephasing origin:

```python
    bw = BeamWizard(band="L")
    # Beam pointing centre = original dish pointing for this field (radians).
    bw.set_field_centre(SkyCoord(*bundle.pointing_centre, unit="rad", frame="icrs"))
```

- [ ] **Step 5: Lint**

Run: `uv run ruff check scripts/test_beam_orientation.py && uv run ruff format --check scripts/test_beam_orientation.py`
Expected: no errors. (If format check fails, run `uv run ruff format scripts/test_beam_orientation.py`.)

- [ ] **Step 6: Argument-parsing smoke check**

Run: `uv run python scripts/test_beam_orientation.py --help`
Expected: help text lists `--field-id`.

- [ ] **Step 7: Commit**

```bash
git add scripts/test_beam_orientation.py
git commit -m "feat(beam-orientation): drive field selection and beam centre by --field-id"
```

---

### Task 6: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Lint and format the whole tree**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: no errors.

- [ ] **Step 2: Run the unit-test subset**

Run: `uv run pytest -m unit -v`
Expected: PASS, including the 5 new `tests/test_beam_orientation_ms_io.py` tests; no regressions.

- [ ] **Step 3: Smoke-run the script if an MS is available**

If a calibrator MS is reachable (cached download or `--ms`), run for the on-axis field and one offset:

Run:
```bash
uv run python scripts/test_beam_orientation.py --field-id 0 --perturbations none --out-dir scratch/ftest0
uv run python scripts/test_beam_orientation.py --field-id 1 --perturbations none --out-dir scratch/ftest1
```
Expected: both complete; the log shows the pointing-centre source (POINTING table or fallback) and `(dl, dm) ~= (0, 0)` for the pre-rephased MS. Field 0 should give a near-identity on-axis beam; field 1 should show ~1.4° off-axis structure. (Skip this step if no MS is available; note it as skipped.)

---

## Notes for the implementer

- `scripts/` is on the pytest path (`pythonpath = ["scripts"]` in `pyproject.toml`), so `from beam_orientation import ms_io` works in tests.
- `daskms` is a lazy import inside the helper functions; the fallback test deliberately uses a non-existent path so it passes whether or not `daskms` is installed.
- Do not touch the Mueller assembly, `solve_per_bin`, Stokes conversion, plots, or zarr output — they are out of scope.
