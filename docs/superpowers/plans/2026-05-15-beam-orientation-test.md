# Beam-Orientation Validation Experiment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone script that recovers the dynamic spectrum of an off-axis-pointed PKS 1934-638 observation and produces diagnostic plots that falsify (or confirm) the parallactic-angle convention encoded in `BeamWizard.get_source_coordinates`.

**Architecture:** A `scripts/beam_orientation/` package of small pure-Python helpers (linear↔Stokes transforms, per-bin Mueller solve, phase rotation, MS I/O, plotting), driven by a top-level entrypoint `scripts/test_beam_orientation.py`. Unit-test the helpers hermetically against the synthetic BDS already used in `tests/test_beam_wizard.py`; smoke-test the end-to-end pipeline against the real MS. The "wrong-orientation" controls reuse the existing `signs=(±1,±1)` and `swap=bool` knobs on `get_source_coordinates`.

**Tech Stack:** Python ≥3.10, NumPy, SciPy, `astropy` (SkyCoord/Time), `dask-ms` (MS I/O), `gdown` (calibrator-MS download), `matplotlib` (plots), `zarr<3` (output dynamic-spectrum store). All under the existing `[full]` extra. Pytest with `unit` markers for helper coverage.

**Spec:** `docs/superpowers/specs/2026-05-15-beam-orientation-test-design.md`.

---

## File Structure

```
src/meerkat_beams/                     # untouched
tests/
├── conftest.py                        # MODIFY: lift commented calibrator
│                                      #         coefficients into a dict
├── _synthetic.py                      # untouched (reused)
├── test_beam_orientation_mueller.py        # CREATE
├── test_beam_orientation_phase_rotate.py   # CREATE
├── test_beam_orientation_plots.py          # CREATE
scripts/
├── test_beam_orientation.py           # CREATE: main entrypoint
└── beam_orientation/                  # CREATE: package
    ├── __init__.py
    ├── calibrator.py                  # CREATE: spectrum-polynomial evaluator
    ├── mueller.py                     # CREATE: T matrices, Mueller assembly, solve
    ├── phase_rotate.py                # CREATE: visibility phase rotation
    ├── ms_io.py                       # CREATE: dask-ms read helper
    ├── download.py                    # CREATE: MS GDrive download (mirrors cache.py)
    └── plots.py                       # CREATE: waterfalls, spectrum, control overlay
pyproject.toml                         # MODIFY: pythonpath = ["src", "scripts"]
```

Why split: each helper has one responsibility and one file. `mueller.py` owns the linear-algebra layer, `phase_rotate.py` owns the visibility geometry, `ms_io.py` owns the data-access boundary, `plots.py` owns presentation. Hosting under `scripts/` keeps it out of the shipped package surface.

The MS is single-scan and fits in memory; everything is dense NumPy. No dask/xarray-zarr indirection in the helpers themselves — we read once via `dask-ms` and call `.compute()` immediately.

---

## Task 1: Lift calibrator coefficients into conftest.py as a Python dict

**Files:**
- Modify: `tests/conftest.py:26-34`

- [ ] **Step 1: Write the failing test**

Append to `tests/conftest.py` is not the right home for a test — instead, add a tiny smoke test at the bottom of `tests/test_beam_orientation_plots.py` once that file exists. For this task, just verify by `python -c` import after editing. **Skip the formal TDD step for this constants-only change**; the dict is exercised by the plots tests in Task 7.

- [ ] **Step 2: Edit `tests/conftest.py`**

Replace the existing block:

```python
# Primary calibrator coefficients and formula
# I0 = 15.088731791006047
# a = -1.2369319597991164
# b = -7.995603882017982
# c = 11.605973123430397
# d = -15.787559501497967
# e = -3.928824456855068
# Reference frequency = 1283791015.625 Hz (1.283791015625
# I(nu) = I(nu0) (nu/nu0) ** (a + b * log10(nu/nu0)) + c * log10(nu/nu0)**2 + d * log10(nu/nu0)**3 + e * log10(nu/nu0)**4)
```

with:

```python
# Primary calibrator (PKS 1934-638) spectral model.
# I(nu) = I0 * (nu/nu0) ** (a + b*x + c*x**2 + d*x**3 + e*x**4)   where x = log10(nu/nu0)
CALIBRATOR_SPECTRUM = {
    "I0": 15.088731791006047,
    "nu0": 1283791015.625,
    "a": -1.2369319597991164,
    "b": -7.995603882017982,
    "c": 11.605973123430397,
    "d": -15.787559501497967,
    "e": -3.928824456855068,
}
```

- [ ] **Step 3: Verify the dict imports cleanly**

```bash
uv run python -c "from tests.conftest import CALIBRATOR_SPECTRUM; print(CALIBRATOR_SPECTRUM['I0'])"
```

Expected: `15.088731791006047`.

- [ ] **Step 4: Verify nothing else broke**

```bash
uv run pytest -m unit -q
```

Expected: same pass/skip counts as before this task.

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py
git commit -m "test(conftest): expose PKS 1934-638 spectrum as CALIBRATOR_SPECTRUM dict"
```

---

## Task 2: Create scripts/beam_orientation package skeleton + wire pytest pythonpath

**Files:**
- Create: `scripts/beam_orientation/__init__.py`
- Modify: `pyproject.toml:65-79`

- [ ] **Step 1: Create the package marker file**

`scripts/beam_orientation/__init__.py`:

```python
"""
Helpers for the beam-orientation validation experiment.

See docs/superpowers/specs/2026-05-15-beam-orientation-test-design.md.
This package lives under scripts/ on purpose — it is not part of the
shipped meerkat-beams distribution.
"""
```

- [ ] **Step 2: Add scripts to pytest's import path**

Edit the `[tool.pytest.ini_options]` block in `pyproject.toml` to add a `pythonpath` entry:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["scripts"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--strict-config",
    "--verbose",
]
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "slow: Tests that take more time to run",
]
```

- [ ] **Step 3: Verify the package imports**

```bash
uv run python -c "import beam_orientation; print(beam_orientation.__doc__)"
```

This will fail (`ModuleNotFoundError`) because `scripts/` isn't on the runtime sys.path. That's expected; only pytest will use it. Verify via:

```bash
uv run pytest --collect-only -q 2>&1 | head -5
```

Expected: collection succeeds.

- [ ] **Step 4: Commit**

```bash
git add scripts/beam_orientation/__init__.py pyproject.toml
git commit -m "feat(scripts): scaffold beam_orientation package + wire pytest pythonpath"
```

---

## Task 3: Implement linear↔Stokes transforms in mueller.py (TDD)

**Files:**
- Create: `scripts/beam_orientation/mueller.py`
- Create: `tests/test_beam_orientation_mueller.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_beam_orientation_mueller.py`:

```python
"""Unit tests for scripts/beam_orientation/mueller.py."""

import numpy as np
import pytest

from beam_orientation import mueller


@pytest.mark.unit
def test_T_maps_unpolarized_stokes_to_equal_parallel_hands():
    T = mueller.linear_to_stokes_matrix()
    S_unpol = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)  # I=1, Q=U=V=0
    V_lin = T @ S_unpol
    # XX = YY = I/2, XY = YX = 0
    np.testing.assert_allclose(V_lin, [0.5, 0.0, 0.0, 0.5], atol=1e-12)


@pytest.mark.unit
def test_T_maps_pure_Q_to_parallel_hand_difference():
    T = mueller.linear_to_stokes_matrix()
    S = np.array([0.0, 1.0, 0.0, 0.0], dtype=complex)  # pure Q
    V_lin = T @ S
    # XX = +Q/2, YY = -Q/2, XY = YX = 0
    np.testing.assert_allclose(V_lin, [0.5, 0.0, 0.0, -0.5], atol=1e-12)


@pytest.mark.unit
def test_T_maps_pure_V_to_imaginary_cross_hands():
    T = mueller.linear_to_stokes_matrix()
    S = np.array([0.0, 0.0, 0.0, 1.0], dtype=complex)  # pure V
    V_lin = T @ S
    # XY = +i V/2, YX = -i V/2, XX = YY = 0
    np.testing.assert_allclose(V_lin, [0.0, 0.5j, -0.5j, 0.0], atol=1e-12)


@pytest.mark.unit
def test_T_and_inverse_compose_to_identity():
    T = mueller.linear_to_stokes_matrix()
    T_inv = mueller.stokes_to_linear_matrix()
    np.testing.assert_allclose(T @ T_inv, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(T_inv @ T, np.eye(4), atol=1e-12)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_beam_orientation_mueller.py -v
```

Expected: collection-time `ModuleNotFoundError: No module named 'beam_orientation.mueller'`.

- [ ] **Step 3: Implement the minimal code**

`scripts/beam_orientation/mueller.py`:

```python
"""
Linear↔Stokes transforms, Mueller assembly, and per-(t,ν) solve
for the beam-orientation validation experiment.
"""

import numpy as np


def linear_to_stokes_matrix() -> np.ndarray:
    """4×4 complex matrix T mapping Stokes (I,Q,U,V) to linear (XX,XY,YX,YY).

    V_lin = T @ V_S.
    Ordering: rows = (XX, XY, YX, YY); columns = (I, Q, U, V).
    """
    return 0.5 * np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0j],
            [0.0, 0.0, 1.0, -1.0j],
            [1.0, -1.0, 0.0, 0.0],
        ],
        dtype=complex,
    )


def stokes_to_linear_matrix() -> np.ndarray:
    """Inverse of :func:`linear_to_stokes_matrix`."""
    return np.linalg.inv(linear_to_stokes_matrix())
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_beam_orientation_mueller.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/beam_orientation/mueller.py tests/test_beam_orientation_mueller.py
git commit -m "feat(mueller): linear↔Stokes T and inverse, with round-trip tests"
```

---

## Task 4: Implement per-bin Mueller solve in mueller.py (TDD)

**Files:**
- Modify: `scripts/beam_orientation/mueller.py`
- Modify: `tests/test_beam_orientation_mueller.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/test_beam_orientation_mueller.py`:

```python
@pytest.mark.unit
def test_solve_recovers_known_B_from_synthetic_M():
    rng = np.random.default_rng(0)
    Nt, Nf = 3, 5
    # Random invertible 4×4 Mueller per bin
    M = rng.standard_normal((Nt, Nf, 4, 4)) + 1j * rng.standard_normal((Nt, Nf, 4, 4))
    B_true = rng.standard_normal((Nt, Nf, 4)) + 1j * rng.standard_normal((Nt, Nf, 4))
    V = np.einsum("tfij,tfj->tfi", M, B_true)

    B_rec, cond = mueller.solve_per_bin(M, V)

    np.testing.assert_allclose(B_rec, B_true, atol=1e-10)
    assert cond.shape == (Nt, Nf)
    assert np.all(cond > 0)


@pytest.mark.unit
def test_solve_flags_ill_conditioned_bins():
    # M with a near-singular bin should produce a large condition number.
    M = np.tile(np.eye(4, dtype=complex), (1, 1, 1, 1))  # (1, 1, 4, 4)
    M[0, 0, 1, 1] = 1e-15  # near-singular
    V = np.ones((1, 1, 4), dtype=complex)

    _, cond = mueller.solve_per_bin(M, V)
    assert cond[0, 0] > 1e10
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_beam_orientation_mueller.py::test_solve_recovers_known_B_from_synthetic_M -v
```

Expected: `AttributeError: module 'beam_orientation.mueller' has no attribute 'solve_per_bin'`.

- [ ] **Step 3: Append implementation**

Append to `scripts/beam_orientation/mueller.py`:

```python
def solve_per_bin(M: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve M(t,ν) · B(t,ν) = V(t,ν) for every (t, ν) bin.

    Parameters
    ----------
    M : (Nt, Nf, 4, 4) complex
        Per-bin 4×4 Stokes Mueller matrices.
    V : (Nt, Nf, 4) complex
        Per-bin observed Stokes vectors.

    Returns
    -------
    B : (Nt, Nf, 4) complex
        Per-bin solved Stokes vectors.
    cond : (Nt, Nf) float
        Per-bin 2-norm condition numbers of M (for downstream masking).
    """
    # NumPy 2.x's np.linalg.solve treats a (..., M) RHS as a stack of
    # (M, K) matrices rather than (M,)-vectors (numpy 2.0 release notes).
    # Add a trailing singleton, solve, then squeeze it back out.
    B = np.linalg.solve(M, V[..., None])[..., 0]
    cond = np.linalg.cond(M)
    return B, cond
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_beam_orientation_mueller.py -v
```

Expected: all 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/beam_orientation/mueller.py tests/test_beam_orientation_mueller.py
git commit -m "feat(mueller): per-bin 4×4 Stokes Mueller solve with cond reporting"
```

---

## Task 5: Implement assemble_mueller in mueller.py (TDD with synthetic BDS)

**Files:**
- Modify: `scripts/beam_orientation/mueller.py`
- Modify: `tests/test_beam_orientation_mueller.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/test_beam_orientation_mueller.py`:

```python
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from meerkat_beams.utils import BeamWizard
from tests._synthetic import DEC0, FREQS, RA0, build_synthetic_bds, build_synthetic_image


@pytest.fixture(scope="module")
def synthetic_bw(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mueller")
    build_synthetic_bds(tmp / "synthetic.bds.zarr")
    build_synthetic_image(tmp / "synthetic.fits")
    return BeamWizard(str(tmp / "synthetic.bds.zarr"), str(tmp / "synthetic.fits"))


@pytest.fixture
def short_times():
    return Time("2024-01-01T00:00:00") + np.linspace(0, 1, 3) * u.hour


@pytest.mark.unit
def test_assemble_mueller_shape_and_dtype(synthetic_bw, short_times):
    src = SkyCoord(ra=RA0 * u.deg, dec=DEC0 * u.deg)
    M = mueller.assemble_mueller(synthetic_bw, src, short_times, FREQS)
    assert M.shape == (len(short_times), len(FREQS), 4, 4)
    assert M.dtype == complex


@pytest.mark.unit
def test_assemble_mueller_on_axis_is_identity(synthetic_bw, short_times):
    # Source at field centre → on-axis → normalised nstokes is the identity.
    src = SkyCoord(ra=RA0 * u.deg, dec=DEC0 * u.deg)
    M = mueller.assemble_mueller(synthetic_bw, src, short_times, FREQS)
    eye = np.broadcast_to(np.eye(4, dtype=complex), M.shape)
    np.testing.assert_allclose(M, eye, atol=1e-6)


@pytest.mark.unit
def test_assemble_mueller_signs_swap_propagate(synthetic_bw, short_times):
    # Pick a source slightly off-axis so the perturbation actually moves the
    # lookup to a different beam pixel. RA0 + 0.5°, DEC0.
    src = SkyCoord(ra=(RA0 + 0.5) * u.deg, dec=DEC0 * u.deg)
    M_default = mueller.assemble_mueller(synthetic_bw, src, short_times, FREQS)
    M_flipx = mueller.assemble_mueller(
        synthetic_bw, src, short_times, FREQS, signs=(-1, 1), swap=False
    )
    M_swap = mueller.assemble_mueller(
        synthetic_bw, src, short_times, FREQS, signs=(1, 1), swap=True
    )
    assert not np.allclose(M_default, M_flipx)
    assert not np.allclose(M_default, M_swap)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_beam_orientation_mueller.py::test_assemble_mueller_shape_and_dtype -v
```

Expected: `AttributeError: module 'beam_orientation.mueller' has no attribute 'assemble_mueller'`.

- [ ] **Step 3: Append implementation**

Append to `scripts/beam_orientation/mueller.py`:

```python
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

from meerkat_beams.utils import BeamWizard

STOKES_LABELS = ("I", "Q", "U", "V")


def assemble_mueller(
    bw: BeamWizard,
    srcpos: SkyCoord,
    times: Time,
    freq: np.ndarray,
    loc: EarthLocation | None = None,
    signs: tuple[int, int] = (1, 1),
    swap: bool = False,
) -> np.ndarray:
    """Build the per-(t, ν) Stokes Mueller tensor for a source.

    Calls ``bw.get_source_coordinates`` once with the supplied ``signs`` /
    ``swap`` knobs, then loops over the 16 (i, j) Stokes-index pairs calling
    ``bw.interpolate_beam(..., var='nstokes', ...)``. With the default knobs
    this matches the composition wrapped by
    ``BeamWizard.get_time_variable_beamgain(..., spi=None)``.

    Returns
    -------
    M : (Nt, Nf, 4, 4) complex
        ``M[t, f, i, j]`` is the (i, j) Stokes Mueller element at the
        source's beam-frame position at time ``t`` and frequency ``f``.
    """
    xpyp, _, _ = bw.get_source_coordinates(srcpos, times=times, loc=loc, signs=signs, swap=swap)
    freq = np.asarray(freq, dtype=float)
    Nt = xpyp.shape[1]
    Nf = freq.size
    M = np.empty((Nt, Nf, 4, 4), dtype=complex)
    for ii, i in enumerate(STOKES_LABELS):
        for jj, j in enumerate(STOKES_LABELS):
            beam_ij = bw.interpolate_beam(xpyp, freq, var="nstokes", i=i, j=j)
            # interpolate_beam returns (Nf, Nt); transpose to (Nt, Nf).
            M[:, :, ii, jj] = beam_ij.T
    return M
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_beam_orientation_mueller.py -v
```

Expected: all tests in the file pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/beam_orientation/mueller.py tests/test_beam_orientation_mueller.py
git commit -m "feat(mueller): assemble 4x4 Stokes Mueller via get_source_coordinates+interpolate_beam"
```

---

## Task 6: Implement phase_rotate in phase_rotate.py (TDD)

**Files:**
- Create: `scripts/beam_orientation/phase_rotate.py`
- Create: `tests/test_beam_orientation_phase_rotate.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_beam_orientation_phase_rotate.py`:

```python
"""Unit tests for scripts/beam_orientation/phase_rotate.py."""

import numpy as np
import pytest

from beam_orientation import phase_rotate

C = 299_792_458.0  # m/s


@pytest.mark.unit
def test_phase_rotate_zero_offset_is_identity():
    rng = np.random.default_rng(0)
    Nb, Nt, Nf = 10, 4, 6
    vis = (rng.standard_normal((Nb, Nt, Nf, 4)) + 1j * rng.standard_normal((Nb, Nt, Nf, 4))).astype(complex)
    uvw = rng.standard_normal((Nb, Nt, 3)) * 100.0
    freq = np.linspace(1.0e9, 1.2e9, Nf)

    out = phase_rotate.phase_rotate(vis, uvw, freq, dl=0.0, dm=0.0)
    np.testing.assert_allclose(out, vis, atol=1e-12)


@pytest.mark.unit
def test_phase_rotate_removes_known_offset_phase():
    """A point source at (l0, m0) has visibility V = exp(-2πi (ul + vm + w(n-1))/λ).
    Phase-rotating to (l0, m0) should produce a flat constant visibility."""
    rng = np.random.default_rng(1)
    Nb, Nt, Nf = 50, 3, 4
    uvw = rng.standard_normal((Nb, Nt, 3)) * 200.0
    freq = np.linspace(1.0e9, 1.2e9, Nf)
    l0, m0 = 0.01, -0.005  # radians, small angle
    n0 = np.sqrt(1.0 - l0**2 - m0**2)

    lmbda = C / freq                            # (Nf,)
    arg = (
        uvw[..., 0:1] * l0 + uvw[..., 1:2] * m0 + uvw[..., 2:3] * (n0 - 1.0)
    )                                           # (Nb, Nt, 1)
    phase = -2j * np.pi * arg / lmbda[None, None, :]   # (Nb, Nt, Nf)
    vis = np.broadcast_to(np.exp(phase)[..., None], (Nb, Nt, Nf, 4)).astype(complex).copy()

    out = phase_rotate.phase_rotate(vis, uvw, freq, dl=l0, dm=m0)
    # After rotation every visibility should be 1+0j (within float tolerance).
    np.testing.assert_allclose(out, np.ones_like(out), atol=1e-8)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_beam_orientation_phase_rotate.py -v
```

Expected: `ModuleNotFoundError: No module named 'beam_orientation.phase_rotate'`.

- [ ] **Step 3: Implement the minimal code**

`scripts/beam_orientation/phase_rotate.py`:

```python
"""
Phase-rotate visibilities to a new phase centre.

Convention being validated:
    V'(u, v, w; ν) = V(u, v, w; ν) * exp(+2πi (u·Δl + v·Δm + w·(Δn - 1)) / λ),
where (Δl, Δm) is the direction-cosine offset of the new phase centre from
the original, and Δn = sqrt(1 − Δl² − Δm²).

The sign of the exponent and of the w-term are listed in the spec as
candidate convention knobs (Section 6) — if the parallactic-angle
controls don't isolate an orientation issue, this sign is the next
fallback to flip.
"""

import numpy as np

C = 299_792_458.0  # m/s


def phase_rotate(
    vis: np.ndarray,
    uvw: np.ndarray,
    freq: np.ndarray,
    dl: float,
    dm: float,
) -> np.ndarray:
    """Phase-rotate visibilities to a new phase centre at (dl, dm).

    Parameters
    ----------
    vis : (Nb, Nt, Nf, Ncorr) complex
        Input visibilities at the original phase centre.
    uvw : (Nb, Nt, 3) float, metres
        Baseline coordinates for each (baseline, time) sample.
    freq : (Nf,) float, Hz
        Channel frequencies.
    dl, dm : float, radians
        Direction-cosine offset of the new phase centre from the original.

    Returns
    -------
    vis_rot : (Nb, Nt, Nf, Ncorr) complex
    """
    dn = np.sqrt(1.0 - dl * dl - dm * dm)
    lmbda = C / np.asarray(freq, dtype=float)                # (Nf,)
    arg = (
        uvw[..., 0:1] * dl
        + uvw[..., 1:2] * dm
        + uvw[..., 2:3] * (dn - 1.0)
    )                                                        # (Nb, Nt, 1)
    phase = 2j * np.pi * arg / lmbda[None, None, :]           # (Nb, Nt, Nf)
    return vis * np.exp(phase)[..., None]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_beam_orientation_phase_rotate.py -v
```

Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/beam_orientation/phase_rotate.py tests/test_beam_orientation_phase_rotate.py
git commit -m "feat(phase_rotate): visibility phase rotation to a new phase centre"
```

---

## Task 7: Implement calibrator_spectrum evaluator (TDD)

**Files:**
- Create: `scripts/beam_orientation/calibrator.py`
- Create: `tests/test_beam_orientation_plots.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_beam_orientation_plots.py`:

```python
"""Unit tests for beam_orientation.calibrator and beam_orientation.plots."""

import numpy as np
import pytest

from beam_orientation import calibrator
from tests.conftest import CALIBRATOR_SPECTRUM


@pytest.mark.unit
def test_calibrator_spectrum_at_reference_frequency():
    nu0 = CALIBRATOR_SPECTRUM["nu0"]
    I0 = CALIBRATOR_SPECTRUM["I0"]
    val = calibrator.evaluate(np.array([nu0]))
    np.testing.assert_allclose(val, [I0], rtol=1e-12)


@pytest.mark.unit
def test_calibrator_spectrum_returns_positive_at_l_band():
    freqs = np.linspace(0.9e9, 1.7e9, 32)
    val = calibrator.evaluate(freqs)
    assert val.shape == freqs.shape
    assert np.all(val > 0)
    # Spectrum should be falling across L-band.
    assert val[0] > val[-1]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_beam_orientation_plots.py -v
```

Expected: `ModuleNotFoundError: No module named 'beam_orientation.calibrator'`.

- [ ] **Step 3: Implement the minimal code**

`scripts/beam_orientation/calibrator.py`:

```python
"""
Evaluate the PKS 1934-638 spectrum.

Coefficients live in tests/conftest.py:CALIBRATOR_SPECTRUM. Formula:

    I(ν) = I0 * (ν/ν0) ** (a + b*x + c*x**2 + d*x**3 + e*x**4)
    x    = log10(ν/ν0)
"""

import numpy as np

from tests.conftest import CALIBRATOR_SPECTRUM


def evaluate(freq_hz: np.ndarray) -> np.ndarray:
    """Return Stokes I in Jy at each frequency in ``freq_hz`` (Hz)."""
    freq_hz = np.asarray(freq_hz, dtype=float)
    nu0 = CALIBRATOR_SPECTRUM["nu0"]
    I0 = CALIBRATOR_SPECTRUM["I0"]
    a = CALIBRATOR_SPECTRUM["a"]
    b = CALIBRATOR_SPECTRUM["b"]
    c = CALIBRATOR_SPECTRUM["c"]
    d = CALIBRATOR_SPECTRUM["d"]
    e = CALIBRATOR_SPECTRUM["e"]
    x = np.log10(freq_hz / nu0)
    exponent = a + b * x + c * x * x + d * x**3 + e * x**4
    return I0 * (freq_hz / nu0) ** exponent
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_beam_orientation_plots.py -v
```

Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/beam_orientation/calibrator.py tests/test_beam_orientation_plots.py
git commit -m "feat(calibrator): PKS 1934-638 polynomial spectrum evaluator"
```

---

## Task 8: Implement plot functions in plots.py (smoke tests)

**Files:**
- Create: `scripts/beam_orientation/plots.py`
- Modify: `tests/test_beam_orientation_plots.py`

- [ ] **Step 1: Append failing smoke tests**

Append to `tests/test_beam_orientation_plots.py`:

```python
import matplotlib

matplotlib.use("Agg")  # headless backend for CI

from beam_orientation import plots


@pytest.fixture
def fake_dyn_spec():
    rng = np.random.default_rng(2)
    Nt, Nf = 8, 16
    times = np.linspace(0.0, 3600.0, Nt)  # seconds
    freq = np.linspace(0.9e9, 1.7e9, Nf)
    B = (1.0 + 0.01 * rng.standard_normal((Nt, Nf, 4))).astype(complex)
    cond = np.ones((Nt, Nf), dtype=float) * 1.5
    return times, freq, B, cond


@pytest.mark.unit
def test_plot_waterfall_writes_png(tmp_path, fake_dyn_spec):
    times, freq, B, cond = fake_dyn_spec
    out = tmp_path / "waterfall_I.png"
    plots.waterfall(times, freq, B, cond, stokes="I", out_path=out)
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.unit
def test_plot_mean_spectrum_writes_png(tmp_path, fake_dyn_spec):
    times, freq, B, cond = fake_dyn_spec
    out = tmp_path / "mean_I_spectrum.png"
    plots.mean_spectrum(freq, B, cond, out_path=out)
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.unit
def test_plot_time_variation_writes_png(tmp_path, fake_dyn_spec):
    times, freq, B, cond = fake_dyn_spec
    out = tmp_path / "time_variation.png"
    plots.time_variation(freq, B, cond, out_path=out)
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.unit
def test_plot_control_overlay_writes_png(tmp_path, fake_dyn_spec):
    times, freq, B, cond = fake_dyn_spec
    runs = {
        "none": (B, cond),
        "flip_x": (B * 1.1, cond),
        "flip_y": (B * 1.2, cond),
        "swap_xy": (B * 1.3, cond),
    }
    out = tmp_path / "control_overlay.png"
    plots.control_overlay(freq, runs, out_path=out)
    assert out.exists() and out.stat().st_size > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_beam_orientation_plots.py -v
```

Expected: `ModuleNotFoundError` or `AttributeError` for missing plot functions.

- [ ] **Step 3: Implement the plot module**

`scripts/beam_orientation/plots.py`:

```python
"""
Diagnostic plots for the beam-orientation validation experiment.

All functions take fully-resolved NumPy arrays (no xarray/zarr), apply
``cond > threshold`` masking, and write a single PNG via matplotlib's
Agg backend. They do not display anything interactively.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from beam_orientation.calibrator import evaluate as catalog_spectrum

STOKES_INDEX = {"I": 0, "Q": 1, "U": 2, "V": 3}
DEFAULT_COND_THRESHOLD = 1e6


def _mask(arr: np.ndarray, cond: np.ndarray, threshold: float) -> np.ndarray:
    """Return arr with cond>threshold bins set to NaN (broadcast over trailing axes)."""
    bad = cond > threshold
    bad = np.broadcast_to(bad[..., None], arr.shape) if arr.ndim > cond.ndim else bad
    return np.where(bad, np.nan, arr)


def waterfall(
    times: np.ndarray,
    freq: np.ndarray,
    B: np.ndarray,
    cond: np.ndarray,
    stokes: str,
    out_path: Path,
    cond_threshold: float = DEFAULT_COND_THRESHOLD,
) -> None:
    """Waterfall (time × freq) plot of one Stokes component of B."""
    idx = STOKES_INDEX[stokes]
    val = _mask(B[..., idx].real, cond, cond_threshold)
    fig, ax = plt.subplots(figsize=(8, 4))
    extent = [freq[0] * 1e-9, freq[-1] * 1e-9, times[-1], times[0]]
    im = ax.imshow(val, aspect="auto", extent=extent, interpolation="nearest")
    ax.set_xlabel("frequency (GHz)")
    ax.set_ylabel("time (s)")
    ax.set_title(f"Stokes {stokes}")
    fig.colorbar(im, ax=ax, label="Jy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def mean_spectrum(
    freq: np.ndarray,
    B: np.ndarray,
    cond: np.ndarray,
    out_path: Path,
    cond_threshold: float = DEFAULT_COND_THRESHOLD,
) -> None:
    """Time-averaged recovered I(ν) with catalog polynomial overlaid; residuals subplot."""
    val = _mask(B[..., 0].real, cond, cond_threshold)
    mean_I = np.nanmean(val, axis=0)
    cat = catalog_spectrum(freq)
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax_top.plot(freq * 1e-9, mean_I, label="recovered ⟨I⟩_t")
    ax_top.plot(freq * 1e-9, cat, "--", label="PKS 1934-638 catalog")
    ax_top.set_ylabel("Jy")
    ax_top.legend()
    ax_bot.plot(freq * 1e-9, mean_I - cat)
    ax_bot.axhline(0, color="k", linewidth=0.5)
    ax_bot.set_xlabel("frequency (GHz)")
    ax_bot.set_ylabel("residual (Jy)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def time_variation(
    freq: np.ndarray,
    B: np.ndarray,
    cond: np.ndarray,
    out_path: Path,
    cond_threshold: float = DEFAULT_COND_THRESHOLD,
) -> None:
    """Per-channel std_t(B) / median_t(|B|) for each Stokes."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, idx in STOKES_INDEX.items():
        val = _mask(B[..., idx], cond, cond_threshold)
        std = np.nanstd(val.real, axis=0)
        med = np.nanmedian(np.abs(val), axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            metric = np.where(med > 0, std / med, np.nan)
        ax.plot(freq * 1e-9, metric, label=f"Stokes {label}")
    ax.set_xlabel("frequency (GHz)")
    ax.set_ylabel("std_t(B) / median_t(|B|)")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def control_overlay(
    freq: np.ndarray,
    runs: dict[str, tuple[np.ndarray, np.ndarray]],
    out_path: Path,
    cond_threshold: float = DEFAULT_COND_THRESHOLD,
) -> None:
    """Overlay the Stokes-I time-variation metric across multiple runs."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, (B, cond) in runs.items():
        val = _mask(B[..., 0], cond, cond_threshold)
        std = np.nanstd(val.real, axis=0)
        med = np.nanmedian(np.abs(val), axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            metric = np.where(med > 0, std / med, np.nan)
        ax.plot(freq * 1e-9, metric, label=name)
    ax.set_xlabel("frequency (GHz)")
    ax.set_ylabel("std_t(I) / median_t(|I|)")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("Residual time variation: unperturbed vs controls")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_beam_orientation_plots.py -v
```

Expected: all 4 smoke tests pass (PNG files created with non-zero size).

- [ ] **Step 5: Commit**

```bash
git add scripts/beam_orientation/plots.py tests/test_beam_orientation_plots.py
git commit -m "feat(plots): waterfall, mean spectrum, time variation, control overlay"
```

---

## Task 9: Implement MS reader in ms_io.py

**Files:**
- Create: `scripts/beam_orientation/ms_io.py`

This task has no TDD step — `dask-ms` needs an MS to read, and we don't have a synthetic-MS fixture. Coverage comes from the end-to-end run in Task 12.

- [ ] **Step 1: Create the module**

`scripts/beam_orientation/ms_io.py`:

```python
"""
Read a single-scan MeerKAT MS into a flat dense NumPy bundle.

Assumptions (per spec Section 5 step 2):
  * Single scan, single field on the calibrator.
  * Linear feeds, 4 correlations (XX, XY, YX, YY).
  * WEIGHT_SPECTRUM present.
  * Fits in memory.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class MSBundle:
    vis: np.ndarray              # (Nb, Nt, Nf, 4) complex64
    weight_spectrum: np.ndarray  # (Nb, Nt, Nf, 4) float32
    flag: np.ndarray             # (Nb, Nt, Nf, 4) bool
    uvw: np.ndarray              # (Nb, Nt, 3) float64, metres
    time: np.ndarray             # (Nt,) MJD seconds
    freq: np.ndarray             # (Nf,) Hz
    phase_centre: tuple[float, float]  # (ra_rad, dec_rad)
    ant1: np.ndarray             # (Nb,) int
    ant2: np.ndarray             # (Nb,) int


def read_ms(path: str | Path) -> MSBundle:
    from daskms import xds_from_ms, xds_from_table

    path = str(path)
    # Main-table groups by DDID + FIELD_ID + SCAN; we expect exactly one group.
    main = xds_from_ms(
        path,
        columns=("DATA", "WEIGHT_SPECTRUM", "FLAG", "UVW", "TIME", "ANTENNA1", "ANTENNA2"),
        group_cols=("DATA_DESC_ID", "FIELD_ID", "SCAN_NUMBER"),
    )
    if len(main) != 1:
        raise RuntimeError(f"{path}: expected one DDID/FIELD/SCAN group, got {len(main)}")
    xds = main[0].compute()

    # Spectral window for the freq axis.
    spw = xds_from_table(f"{path}::SPECTRAL_WINDOW")[0].compute()
    ddid = int(xds.attrs.get("DATA_DESC_ID", 0))
    pol_id = xds_from_table(f"{path}::DATA_DESCRIPTION")[0].compute()
    spw_idx = int(pol_id.SPECTRAL_WINDOW_ID.values[ddid])
    freq = np.asarray(spw.CHAN_FREQ.values[spw_idx], dtype=float)

    # Field table for the phase centre.
    field = xds_from_table(f"{path}::FIELD")[0].compute()
    field_id = int(xds.attrs.get("FIELD_ID", 0))
    phase_dir = np.asarray(field.PHASE_DIR.values[field_id])  # (1, 2) usually
    ra_rad, dec_rad = float(phase_dir.flat[0]), float(phase_dir.flat[1])

    # Reshape (row, chan, corr) -> (Nb, Nt, Nf, Ncorr).
    ant1 = np.asarray(xds.ANTENNA1.values)
    ant2 = np.asarray(xds.ANTENNA2.values)
    time_row = np.asarray(xds.TIME.values)

    times = np.unique(time_row)
    Nt = times.size
    pairs, inv = np.unique(np.stack([ant1, ant2], axis=1), axis=0, return_inverse=True)
    Nb = pairs.shape[0]
    Nf = freq.size

    def _reshape(col, fill):
        Ncorr = col.shape[-1] if col.ndim == 3 else 1
        out = np.full((Nb, Nt, Nf, Ncorr), fill, dtype=col.dtype)
        # Row-by-row scatter. Single scan so this is small.
        time_idx = np.searchsorted(times, time_row)
        out[inv, time_idx] = col
        return out

    vis = _reshape(np.asarray(xds.DATA.values), 0.0 + 0.0j)
    ws = _reshape(np.asarray(xds.WEIGHT_SPECTRUM.values), 0.0)
    flag = _reshape(np.asarray(xds.FLAG.values), True)

    uvw = np.zeros((Nb, Nt, 3), dtype=float)
    uvw_row = np.asarray(xds.UVW.values)
    time_idx = np.searchsorted(times, time_row)
    uvw[inv, time_idx] = uvw_row

    return MSBundle(
        vis=vis,
        weight_spectrum=ws,
        flag=flag,
        uvw=uvw,
        time=times,
        freq=freq,
        phase_centre=(ra_rad, dec_rad),
        ant1=pairs[:, 0],
        ant2=pairs[:, 1],
    )
```

- [ ] **Step 2: Verify the module imports cleanly**

```bash
uv run python -c "from beam_orientation.ms_io import MSBundle, read_ms; print(MSBundle.__doc__)"
```

This will fail outside a pytest context. Verify via:

```bash
uv run pytest --collect-only tests/test_beam_orientation_plots.py -q
```

(Collection should still succeed; the module is only imported transitively from `__init__.py` indirectly via the test files.)

- [ ] **Step 3: Run ruff and the existing unit tests**

```bash
uv run ruff check scripts/beam_orientation/ms_io.py
uv run pytest -m unit -q
```

Expected: ruff passes; unit tests unchanged.

- [ ] **Step 4: Commit**

```bash
git add scripts/beam_orientation/ms_io.py
git commit -m "feat(ms_io): single-scan MS reader returning dense NumPy MSBundle"
```

---

## Task 10: Implement MS download helper in download.py

**Files:**
- Create: `scripts/beam_orientation/download.py`

This mirrors `src/meerkat_beams/cache.py` for the calibrator MS. The MS itself is treated as a tarball on Google Drive (the same pattern as the band BDS).

- [ ] **Step 1: Create the module**

`scripts/beam_orientation/download.py`:

```python
"""
On-demand download + cache of the PKS 1934-638 calibrator MS used by the
beam-orientation validation experiment.

Cache layout under ``meerkat_beams.cache.cache_root()``:

    <root>/test_ms/<MS_BASENAME>/

The GDrive ID is taken from ``tests.conftest.test_ms_gdrive_id``. The
download is treated as a tarball and unpacked under a sibling
``.partial`` directory before being promoted atomically.

Concurrent first-time downloads are not guarded. Same caveat as
``meerkat_beams.cache``: warm the cache from a single process.
"""

import os
import shutil
import sys
import tarfile
from pathlib import Path

from meerkat_beams.cache import cache_root

MS_BASENAME = "pks1934_offset.ms"  # promoted directory name; matches tarball top-level


def ms_path() -> Path:
    return cache_root() / "test_ms" / MS_BASENAME


def ensure_ms() -> Path:
    target = ms_path()
    if target.exists():
        return target

    from tests.conftest import test_ms_gdrive_id

    _download_and_extract(test_ms_gdrive_id, target)
    return target


def _download_and_extract(gdrive_id: str, target: Path) -> None:
    import gdown
    from meerkat_beams.utils import log

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)

    tarball = partial.parent / f"{target.name}.tgz"
    try:
        log.info(f"downloading calibrator MS from gdrive id {gdrive_id}")
        gdown.download(id=gdrive_id, output=str(tarball), quiet=False)

        log.info(f"extracting {tarball} into {partial}")
        extract_kwargs = {"filter": "data"} if sys.version_info >= (3, 12) else {}
        with tarfile.open(tarball, "r:gz") as tar:
            tar.extractall(path=partial, **extract_kwargs)

        # Expect a single top-level directory inside the tarball.
        entries = [p for p in partial.iterdir() if p.is_dir()]
        if len(entries) != 1:
            raise RuntimeError(
                f"expected exactly one top-level dir inside calibrator MS tarball, got {entries!r}"
            )
        os.replace(entries[0], target)
    finally:
        if tarball.exists():
            tarball.unlink()
        if partial.exists():
            shutil.rmtree(partial, ignore_errors=True)
```

- [ ] **Step 2: Verify the module imports**

```bash
uv run python -c "import sys; sys.path.insert(0, 'scripts'); from beam_orientation.download import ms_path; print(ms_path())"
```

Expected: prints a path inside `~/.cache/meerkat-beams/test_ms/pks1934_offset.ms` (or `MBEAMS_CACHE_DIR`-relative).

- [ ] **Step 3: Run ruff**

```bash
uv run ruff check scripts/beam_orientation/download.py
```

Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add scripts/beam_orientation/download.py
git commit -m "feat(download): GDrive download/cache for calibrator MS"
```

---

## Task 11: Wire up the main entrypoint script

**Files:**
- Create: `scripts/test_beam_orientation.py`

- [ ] **Step 1: Create the script**

`scripts/test_beam_orientation.py`:

```python
#!/usr/bin/env python
"""
Beam-orientation validation experiment.

End-to-end pipeline:
  1. Download the calibrator MS (cached) if not provided via --ms.
  2. Read visibilities, weights, UVW, freq, phase centre.
  3. Phase-rotate to PKS 1934-638 using a SIN-projection (Δl, Δm) offset
     computed from the (ra, dec) in tests/conftest.py.
  4. Noise-weighted average over baselines.
  5. Convert observed linear visibilities to observed Stokes.
  6. For each perturbation in {"none", "flip_x", "flip_y", "swap_xy"}:
       - assemble M_S(t, ν) from the cached L-band BDS
       - solve B(t, ν) = M_S⁻¹ V̄_S
       - write dynamic_spectrum.zarr
       - write 6 PNG plots
  7. Write a control_overlay.png across the four runs.

The spec for this script is in
docs/superpowers/specs/2026-05-15-beam-orientation-test-design.md.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import zarr
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

# Ensure scripts/ is on sys.path when the script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from beam_orientation import mueller, phase_rotate, plots  # noqa: E402
from beam_orientation.download import ensure_ms  # noqa: E402
from beam_orientation.ms_io import read_ms  # noqa: E402
from meerkat_beams.utils import BeamWizard, log  # noqa: E402
from tests.conftest import dec as DEC_STR  # noqa: E402
from tests.conftest import ra as RA_STR

PERTURBATIONS: dict[str, tuple[tuple[int, int], bool]] = {
    "none": ((1, 1), False),
    "flip_x": ((-1, 1), False),
    "flip_y": ((1, -1), False),
    "swap_xy": ((1, 1), True),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("scratch/orientation_test"),
        help="output directory; one subdir per perturbation will be created here.",
    )
    p.add_argument(
        "--ms",
        type=Path,
        default=None,
        help="override path to the calibrator MS; defaults to the cached download.",
    )
    p.add_argument(
        "--perturbations",
        nargs="+",
        choices=list(PERTURBATIONS),
        default=list(PERTURBATIONS),
        help="which perturbation runs to execute (default: all four).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ms_path = args.ms or ensure_ms()
    log.info(f"reading MS from {ms_path}")
    bundle = read_ms(ms_path)
    log.info(
        f"MS shape Nb={bundle.vis.shape[0]} Nt={bundle.vis.shape[1]} "
        f"Nf={bundle.vis.shape[2]} corr={bundle.vis.shape[3]}"
    )

    # Source position from conftest constants (HMS/DMS strings).
    srcpos = SkyCoord(RA_STR, DEC_STR.replace(".", ":", 2), unit=("hourangle", "deg"))
    ra_src_rad = float(srcpos.ra.rad)
    dec_src_rad = float(srcpos.dec.rad)

    # SIN-projection direction-cosine offset from the original phase centre.
    ra_pc, dec_pc = bundle.phase_centre
    dl = np.cos(dec_src_rad) * np.sin(ra_src_rad - ra_pc)
    dm = np.sin(dec_src_rad) * np.cos(dec_pc) - np.cos(dec_src_rad) * np.sin(dec_pc) * np.cos(
        ra_src_rad - ra_pc
    )
    log.info(f"phase-rotating to (dl, dm) = ({dl:.6e}, {dm:.6e}) rad")

    vis_rot = phase_rotate.phase_rotate(bundle.vis, bundle.uvw, bundle.freq, dl=dl, dm=dm)

    # Mask flagged samples by zeroing their weight.
    w = bundle.weight_spectrum.astype(float).copy()
    w[bundle.flag] = 0.0

    # Noise-weighted average over baselines: V̄_lin(t, ν, corr).
    num = np.einsum("btfc,btfc->tfc", w, vis_rot)
    den = np.einsum("btfc->tfc", w)
    with np.errstate(invalid="ignore", divide="ignore"):
        V_lin = np.where(den > 0, num / den, 0.0 + 0.0j)

    # Linear → Stokes (per spec Section 5 step 5).
    T_inv = mueller.stokes_to_linear_matrix()
    V_S = np.einsum("ij,tfj->tfi", T_inv, V_lin)

    # Astropy Time vector and MeerKAT location for the Mueller assembly.
    times = Time(bundle.time / 86400.0, format="mjd", scale="utc")
    loc = EarthLocation.from_geodetic(lon=21.4 * np.pi / 180, lat=-30.7 * np.pi / 180, height=1054.0)

    bw = BeamWizard(band="L")
    runs: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for name in args.perturbations:
        signs, swap = PERTURBATIONS[name]
        run_dir = args.out_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"=== perturbation '{name}': signs={signs}, swap={swap} ===")

        M_S = mueller.assemble_mueller(
            bw, srcpos, times, bundle.freq, loc=loc, signs=signs, swap=swap
        )
        B, cond = mueller.solve_per_bin(M_S, V_S)

        _write_zarr(run_dir / "dynamic_spectrum.zarr", B, cond, bundle.time, bundle.freq, name)
        plots.waterfall(bundle.time, bundle.freq, B, cond, "I", run_dir / "dyn_spec_I.png")
        plots.waterfall(bundle.time, bundle.freq, B, cond, "Q", run_dir / "dyn_spec_Q.png")
        plots.waterfall(bundle.time, bundle.freq, B, cond, "U", run_dir / "dyn_spec_U.png")
        plots.waterfall(bundle.time, bundle.freq, B, cond, "V", run_dir / "dyn_spec_V.png")
        plots.mean_spectrum(bundle.freq, B, cond, run_dir / "mean_I_spectrum.png")
        plots.time_variation(bundle.freq, B, cond, run_dir / "time_variation.png")
        runs[name] = (B, cond)

    if len(runs) > 1:
        plots.control_overlay(bundle.freq, runs, args.out_dir / "control_overlay.png")
        log.info(f"control overlay → {args.out_dir / 'control_overlay.png'}")


def _write_zarr(
    path: Path,
    B: np.ndarray,
    cond: np.ndarray,
    times_sec: np.ndarray,
    freq: np.ndarray,
    perturbation: str,
) -> None:
    root = zarr.open(str(path), mode="w")
    root.create_dataset("B", data=B.astype(np.complex64), chunks=False)
    root.create_dataset("cond_M", data=cond.astype(np.float32), chunks=False)
    root.create_dataset("time", data=times_sec.astype(np.float64), chunks=False)
    root.create_dataset("frequency", data=freq.astype(np.float64), chunks=False)
    root.attrs["source"] = "PKS 1934-638"
    root.attrs["polarization"] = ["I", "Q", "U", "V"]
    root.attrs["band"] = "L"
    root.attrs["perturbation"] = perturbation


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make the script executable**

```bash
chmod +x scripts/test_beam_orientation.py
```

- [ ] **Step 3: Lint-only smoke check**

```bash
uv run ruff check scripts/test_beam_orientation.py
```

Expected: no errors.

- [ ] **Step 4: --help works (no network access)**

```bash
uv run python scripts/test_beam_orientation.py --help
```

Expected: prints the argparse usage block.

- [ ] **Step 5: Commit**

```bash
git add scripts/test_beam_orientation.py
git commit -m "feat(script): wire up beam-orientation validation entrypoint"
```

---

## Task 12: End-to-end run and visual inspection

This task produces no committed code — it executes the script against the real MS and inspects the resulting plots. Results stay under `scratch/orientation_test/` (already in `.gitignore` via `scratch/`).

- [ ] **Step 1: Warm the L-band BDS cache (skip if already populated)**

```bash
uv run python -c "from meerkat_beams import cache; cache.ensure_band_bds('L')"
```

Expected: prints "L-band BDS already cached at …" or downloads + converts.

- [ ] **Step 2: Run the full pipeline (all four perturbations)**

```bash
uv run python scripts/test_beam_orientation.py --out-dir scratch/orientation_test
```

Expected output (paraphrased):
- "downloading calibrator MS from gdrive id …" on first run, else "reading MS from …".
- "MS shape Nb=… Nt=… Nf=… corr=4".
- "phase-rotating to (dl, dm) = …".
- 4 sections, one per perturbation.
- "control overlay → scratch/orientation_test/control_overlay.png".

Runtime is dominated by the BDS spline interpolation; expect O(minutes) for a typical small MS.

- [ ] **Step 3: Inspect the four plots in `scratch/orientation_test/none/`**

Open them in an image viewer:

```bash
ls scratch/orientation_test/none/
xdg-open scratch/orientation_test/none/mean_I_spectrum.png &
xdg-open scratch/orientation_test/none/time_variation.png &
xdg-open scratch/orientation_test/none/dyn_spec_I.png &
xdg-open scratch/orientation_test/control_overlay.png &
```

Expected (if the orientation convention is correct):
- `mean_I_spectrum.png`: top panel — recovered ⟨I⟩_t lies on top of the PKS 1934-638 catalog polynomial; bottom panel — residual is flat near zero, scattered around the thermal noise.
- `time_variation.png`: Stokes I curve sits near the thermal-noise floor; Q/U/V are similar (no orientation-driven swing).
- `dyn_spec_I.png`: featureless waterfall, no diagonal/bowtie banding.
- `control_overlay.png`: `none` is visibly below `flip_x`, `flip_y`, `swap_xy` — the controls show clearly worse residual time variation.

If those checks fail, the convention is wrong (or one of the fallback knobs from Section 6 of the spec is). Do **not** silently "fix" the script — bring the failure mode back to the user so we can decide which knob to flip next.

- [ ] **Step 4: (Optional) Record the plot directory in a doc note**

If results are interesting and you want to keep them for posterity, copy the `none/` and `control_overlay.png` outputs into a dated subdirectory and reference it from `docs/progress.md`. The git-ignored `scratch/` is fine for routine iteration.

- [ ] **Step 5: No commit for this task** — it's a manual validation step, not a code change.

---

## Self-review notes

Spec coverage:
- Section 4 data flow → Task 11 (main script orchestration).
- Section 5 stages 1–9 → Tasks 5, 6, 8, 9, 10, 11.
- Section 6 controls → `PERTURBATIONS` dict + assemble_mueller signs/swap (Tasks 5, 11).
- Section 7 code layout → matches Tasks 2–10 exactly.
- Section 8 constants → Tasks 1, 3, 7.
- Section 9 outputs → Task 11 `_write_zarr` + Task 8 plot functions.
- Section 10 pass/fail interpretation → Task 12 inspection checklist.
- Section 11 risks/open items → documented in code comments (phase_rotate.py) and in Task 12 step 3 "if these checks fail" callout.

Placeholders: none. Every code block is complete.

Type consistency: `B` is consistently `(Nt, Nf, 4)`; `cond` is `(Nt, Nf)`; `M_S` is `(Nt, Nf, 4, 4)`. The `STOKES_LABELS = ("I","Q","U","V")` constant is used by `assemble_mueller`; `STOKES_INDEX` is the corresponding label→index map used by `plots`. Both files keep the same ordering.
