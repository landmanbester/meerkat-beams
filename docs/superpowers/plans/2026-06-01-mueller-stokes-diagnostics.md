# Per-Stokes source/beam dynamic-spectrum diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure `scripts/test_beam_orientation.py` output into per-perturbation, per-Stokes folders, each holding the dynamic spectrum, time profile, and frequency profile of the apparent source, the recovered source, and the beam — all real-part, absolute-valued, unnormalised.

**Architecture:** Replace the special-purpose plot functions in `scripts/beam_orientation/plots.py` with three generic single-array helpers (`dyn_spectrum`, `time_profile`, `freq_profile`). The script computes three per-Stokes `(Nt, Nf)` complex quantities — `apparent = coh_to_stokes·V̄_coh`, `recovered = B`, `beam = diag(coh_to_stokes·M_C·S)` — and drives the helpers into a `run_dir/<Stokes>/` tree. Only the recovered quantity is `cond`-masked.

**Tech Stack:** NumPy, matplotlib (Agg), pytest. Spec: `docs/superpowers/specs/2026-06-01-mueller-stokes-diagnostics-design.md`.

**Note on current working tree:** `plots.py`, `scripts/test_beam_orientation.py`, and `tests/test_beam_orientation_plots.py` contain *uncommitted* edits adding `mueller_dynamic_spectrum` / `mueller_time_variation` / `mueller_time_profile`. This plan **replaces** those edits; do not preserve them.

---

### Task 1: Generic plot helpers + tests

**Files:**
- Modify (full rewrite): `scripts/beam_orientation/plots.py`
- Modify (rewrite plot tests, keep calibrator tests): `tests/test_beam_orientation_plots.py`

- [ ] **Step 1: Replace the test file with calibrator tests (unchanged) + new helper tests**

Write `tests/test_beam_orientation_plots.py`:

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


import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless backend for CI

from beam_orientation import plots  # noqa: E402


@pytest.fixture
def fake_2d():
    rng = np.random.default_rng(7)
    Nt, Nf = 8, 16
    times = np.linspace(0.0, 3600.0, Nt)  # seconds
    freq = np.linspace(0.9e9, 1.7e9, Nf)
    data = (1.0 + 0.01 * rng.standard_normal((Nt, Nf))) + 0.01j * rng.standard_normal((Nt, Nf))
    return times, freq, data


@pytest.mark.unit
def test_plot_dyn_spectrum_writes_png(tmp_path, fake_2d):
    times, freq, data = fake_2d
    out = tmp_path / "dyn.png"
    plots.dyn_spectrum(times, freq, data, out, title="t", cbar_label="Jy")
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.unit
def test_plot_time_profile_writes_png(tmp_path, fake_2d):
    times, freq, data = fake_2d
    out = tmp_path / "tp.png"
    plots.time_profile(times, data, out, title="t", ylabel="Jy")
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.unit
def test_plot_freq_profile_writes_png(tmp_path, fake_2d):
    times, freq, data = fake_2d
    out = tmp_path / "fp.png"
    plots.freq_profile(freq, data, out, title="t", ylabel="Jy")
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.unit
def test_profiles_ignore_nan_bins(tmp_path, fake_2d):
    times, freq, data = fake_2d
    data = data.copy()
    data[0, :] = np.nan  # blank one time row
    data[:, 0] = np.nan  # blank one freq channel
    out = tmp_path / "tp_nan.png"
    plots.time_profile(times, data, out, title="t", ylabel="Jy")
    assert out.exists() and out.stat().st_size > 0
    # nanmean over time must not propagate the blanked row into every channel.
    prof = np.nanmean(np.real(data), axis=0)
    assert np.isfinite(prof[1:]).all()
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_beam_orientation_plots.py -q -m unit`
Expected: FAIL — `AttributeError: module 'beam_orientation.plots' has no attribute 'dyn_spectrum'` (calibrator tests still pass).

- [ ] **Step 3: Replace `scripts/beam_orientation/plots.py` in full**

```python
"""
Diagnostic plots for the beam-orientation validation experiment.

Each function takes a single fully-resolved ``(Nt, Nf)`` complex NumPy array,
plots its real part via matplotlib's Agg backend, and writes one PNG. Profile
means use ``nanmean`` so blanked (NaN) bins are ignored. Nothing is displayed
interactively and nothing is normalised.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def dyn_spectrum(
    times: np.ndarray,
    freq: np.ndarray,
    data: np.ndarray,
    out_path: Path,
    *,
    title: str,
    cbar_label: str,
) -> None:
    """Dynamic spectrum (time × freq) of ``Re(data)`` for one ``(Nt, Nf)`` array."""
    val = np.real(data)
    t0 = times[0]
    extent = [freq[0] * 1e-9, freq[-1] * 1e-9, times[-1] - t0, times[0] - t0]
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(val, aspect="auto", extent=extent, interpolation="nearest")
    ax.set_xlabel("frequency (GHz)")
    ax.set_ylabel("time (s)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def time_profile(
    times: np.ndarray,
    data: np.ndarray,
    out_path: Path,
    *,
    title: str,
    ylabel: str,
) -> None:
    """``Re(data)`` averaged over frequency, plotted as a function of time."""
    prof = np.nanmean(np.real(data), axis=1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times - times[0], prof)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def freq_profile(
    freq: np.ndarray,
    data: np.ndarray,
    out_path: Path,
    *,
    title: str,
    ylabel: str,
) -> None:
    """``Re(data)`` averaged over time, plotted as a function of frequency."""
    prof = np.nanmean(np.real(data), axis=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(freq * 1e-9, prof)
    ax.set_xlabel("frequency (GHz)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_beam_orientation_plots.py -q -m unit`
Expected: PASS (all calibrator + helper tests).

- [ ] **Step 5: Lint/format**

Run: `uv run ruff check scripts/beam_orientation/plots.py tests/test_beam_orientation_plots.py && uv run ruff format scripts/beam_orientation/plots.py tests/test_beam_orientation_plots.py`
Expected: no errors; files formatted.

- [ ] **Step 6: Commit**

```bash
git add scripts/beam_orientation/plots.py tests/test_beam_orientation_plots.py
git commit -m "refactor(plots): generic per-array dyn_spectrum/time_profile/freq_profile helpers"
```

Note: this commit leaves `scripts/test_beam_orientation.py` temporarily referencing the removed plot functions. The script is not imported by the test suite, so CI (ruff + pytest) stays green; Task 2 fixes the script.

---

### Task 2: Restructure the experiment script into per-Stokes folders

**Files:**
- Modify: `scripts/test_beam_orientation.py`

- [ ] **Step 1: Add module-level constants below `PERTURBATIONS`**

Find:

```python
PERTURBATIONS: dict[str, tuple[tuple[int, int], bool]] = {
    "none": ((1, 1), False),
    "flip_x": ((-1, 1), False),
    "flip_y": ((1, -1), False),
    "swap_xy": ((1, 1), True),
}
```

Add immediately after it:

```python
STOKES = ("I", "Q", "U", "V")
COND_THRESHOLD = 1e6  # recovered bins with cond above this are blanked (NaN)
```

- [ ] **Step 2: Replace the BeamWizard/loop/overlay block**

Find (current body from the `coh_to_stokes` line through the end of `main`'s overlay calls):

```python
    # Coherency (XX,XY,YX,YY) → Stokes (I,Q,U,V), matching the BDS convention.
    coh_to_stokes = mueller.coherency_to_stokes_matrix()

    bw = BeamWizard(band="L")
    # Beam pointing centre = original dish pointing for this field (radians).
    bw.set_field_centre(SkyCoord(*bundle.pointing_centre, unit="rad", frame="icrs"))
    runs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    mueller_runs: dict[str, np.ndarray] = {}

    for name in args.perturbations:
        signs, swap = PERTURBATIONS[name]
        run_dir = args.out_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"=== perturbation '{name}': signs={signs}, swap={swap} ===")

        # Solve in the coherency frame, then convert the recovered spectrum to Stokes.
        M_C = mueller.assemble_mueller(bw, srcpos, times, bundle.freq, signs=signs, swap=swap)
        B_C, cond = mueller.solve_per_bin(M_C, V_coh)
        B = np.einsum("ij,tfj->tfi", coh_to_stokes, B_C)

        _write_zarr(run_dir / "dynamic_spectrum.zarr", B, cond, bundle.time, bundle.freq, name)
        plots.waterfall(bundle.time, bundle.freq, B, cond, "I", run_dir / "dyn_spec_I.png")
        plots.waterfall(bundle.time, bundle.freq, B, cond, "Q", run_dir / "dyn_spec_Q.png")
        plots.waterfall(bundle.time, bundle.freq, B, cond, "U", run_dir / "dyn_spec_U.png")
        plots.waterfall(bundle.time, bundle.freq, B, cond, "V", run_dir / "dyn_spec_V.png")
        plots.mean_spectrum(bundle.freq, B, cond, run_dir / "mean_I_spectrum.png")
        plots.time_variation(bundle.freq, B, cond, run_dir / "time_variation.png")
        plots.mueller_dynamic_spectrum(bundle.time, bundle.freq, M_C, run_dir / "mueller_dyn_spec.png")
        runs[name] = (B, cond)
        mueller_runs[name] = M_C

    plots.mueller_time_variation(bundle.freq, mueller_runs, args.out_dir / "mueller_time_variation.png")
    log.info(f"mueller time variability → {args.out_dir / 'mueller_time_variation.png'}")
    plots.mueller_time_profile(bundle.time, mueller_runs, args.out_dir / "mueller_time_profile.png")
    log.info(f"mueller time profile → {args.out_dir / 'mueller_time_profile.png'}")

    if len(runs) > 1:
        plots.control_overlay(bundle.freq, runs, args.out_dir / "control_overlay.png")
        log.info(f"control overlay → {args.out_dir / 'control_overlay.png'}")
```

Replace with:

```python
    # Coherency (XX,XY,YX,YY) → Stokes (I,Q,U,V), matching the BDS convention.
    coh_to_stokes = mueller.coherency_to_stokes_matrix()
    S = np.linalg.inv(coh_to_stokes)  # noqa: N806  Stokes → coherency

    # Observed apparent Stokes from the data — the beam-modulated spectrum the
    # model should reproduce. Depends only on the data, not the perturbation, so
    # it is identical in every perturbation folder.
    apparent = np.einsum("ij,tfj->tfi", coh_to_stokes, V_coh)

    bw = BeamWizard(band="L")
    # Beam pointing centre = original dish pointing for this field (radians).
    bw.set_field_centre(SkyCoord(*bundle.pointing_centre, unit="rad", frame="icrs"))

    for name in args.perturbations:
        signs, swap = PERTURBATIONS[name]
        run_dir = args.out_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"=== perturbation '{name}': signs={signs}, swap={swap} ===")

        # Solve in the coherency frame, then convert the recovered spectrum to Stokes.
        M_C = mueller.assemble_mueller(bw, srcpos, times, bundle.freq, signs=signs, swap=swap)
        B_C, cond = mueller.solve_per_bin(M_C, V_coh)
        B = np.einsum("ij,tfj->tfi", coh_to_stokes, B_C)

        # Stokes Mueller and its per-Stokes diagonal (the beam gain per Stokes).
        M_S = np.einsum("ij,tfjk,kl->tfil", coh_to_stokes, M_C, S)  # noqa: N806
        beam_diag = np.einsum("tfpp->tfp", M_S)

        _write_zarr(run_dir / "dynamic_spectrum.zarr", B, cond, bundle.time, bundle.freq, name)
        _write_stokes_plots(run_dir, bundle.time, bundle.freq, apparent, B, cond, beam_diag)
```

- [ ] **Step 3: Add the `_write_stokes_plots` helper above `_write_zarr`**

Find:

```python
def _write_zarr(
```

Insert immediately before it:

```python
def _write_stokes_plots(
    run_dir: Path,
    times: np.ndarray,
    freq: np.ndarray,
    apparent: np.ndarray,  # (Nt, Nf, 4) complex, observed apparent Stokes
    B: np.ndarray,  # noqa: N803  (Nt, Nf, 4) complex, recovered Stokes
    cond: np.ndarray,  # (Nt, Nf) float
    beam_diag: np.ndarray,  # (Nt, Nf, 4) complex, Stokes-Mueller diagonal
) -> None:
    """Write the 9 per-Stokes diagnostic PNGs (apparent / recovered / beam) per folder."""
    for p_idx, p in enumerate(STOKES):
        sdir = run_dir / p
        sdir.mkdir(parents=True, exist_ok=True)
        app = apparent[..., p_idx]
        # Blank ill-conditioned recovered bins so solve outliers don't dominate.
        rec = np.where(cond > COND_THRESHOLD, np.nan, B[..., p_idx])
        bem = beam_diag[..., p_idx]

        plots.dyn_spectrum(times, freq, app, sdir / "apparent_dyn_spec.png", title=f"apparent Stokes {p}", cbar_label="Jy")
        plots.time_profile(times, app, sdir / "apparent_time_profile.png", title=f"apparent Stokes {p} — time profile", ylabel="Jy")
        plots.freq_profile(freq, app, sdir / "apparent_freq_profile.png", title=f"apparent Stokes {p} — freq profile", ylabel="Jy")

        plots.dyn_spectrum(times, freq, rec, sdir / "recovered_dyn_spec.png", title=f"recovered Stokes {p}", cbar_label="Jy")
        plots.time_profile(times, rec, sdir / "recovered_time_profile.png", title=f"recovered Stokes {p} — time profile", ylabel="Jy")
        plots.freq_profile(freq, rec, sdir / "recovered_freq_profile.png", title=f"recovered Stokes {p} — freq profile", ylabel="Jy")

        plots.dyn_spectrum(times, freq, bem, sdir / "beam_dyn_spec.png", title=f"beam Stokes {p}", cbar_label="gain")
        plots.time_profile(times, bem, sdir / "beam_time_profile.png", title=f"beam Stokes {p} — time profile", ylabel="gain")
        plots.freq_profile(freq, bem, sdir / "beam_freq_profile.png", title=f"beam Stokes {p} — freq profile", ylabel="gain")
```

- [ ] **Step 4: Verify the script parses and imports cleanly**

Run: `uv run python -c "import ast; ast.parse(open('scripts/test_beam_orientation.py').read()); print('parse-ok')"`
Expected: `parse-ok`

Run: `uv run python scripts/test_beam_orientation.py --help`
Expected: argparse help text prints (confirms imports resolve and no reference to removed plot functions remains).

- [ ] **Step 5: Lint/format**

Run: `uv run ruff check scripts/test_beam_orientation.py && uv run ruff format scripts/test_beam_orientation.py`
Expected: no errors (line-length 120 — the `plots.*` calls are long; if ruff reflows them, re-stage). If E501 fires on a call, let `ruff format` wrap it.

- [ ] **Step 6: Commit**

```bash
git add scripts/test_beam_orientation.py
git commit -m "feat(beam-orientation): per-Stokes apparent/recovered/beam dynamic-spectrum outputs"
```

---

### Task 3: Whole-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the unit suite**

Run: `uv run pytest -m unit -q`
Expected: PASS (no references to removed plot functions; new helper tests green).

- [ ] **Step 2: Lint + format check across the repo**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: all checks pass; all files formatted.

- [ ] **Step 3 (optional, user-driven smoke run)**

The MS is cached at `~/.cache/meerkat-beams/test_ms/pks1934_offset.ms`. Example:

Run: `uv run python scripts/test_beam_orientation.py --field-id 1 --out-dir scratch/stokes_diag1`
Expected: `scratch/stokes_diag1/<perturbation>/<Stokes>/` each contains the 9 PNGs (`apparent_*`, `recovered_*`, `beam_*`) and `dynamic_spectrum.zarr`. Left to the user.

---

## Self-review

- **Spec coverage:** apparent/recovered/beam quantities (Task 2 Steps 2–3) ✓; three plot types via generic helpers (Task 1) ✓; per-Stokes folder tree (Task 2 Step 3) ✓; real part only (helpers use `np.real`) ✓; no median normalisation (helpers plot raw means) ✓; `cond` mask on recovered only (Task 2 Step 3 `np.where`) ✓; `apparent` duplicated per folder (loop re-emits it) ✓; removal of `mueller_*`/`waterfall`/`mean_spectrum`/`time_variation`/`control_overlay` (Task 1 full rewrite drops them) ✓; calibrator module/tests kept (Task 1 Step 1) ✓.
- **Placeholder scan:** none — every code step contains full content.
- **Type consistency:** helper signatures (`dyn_spectrum(times, freq, data, out_path, *, title, cbar_label)`, `time_profile(times, data, out_path, *, title, ylabel)`, `freq_profile(freq, data, out_path, *, title, ylabel)`) are identical between Task 1 definitions and Task 2 call sites. `_write_stokes_plots` parameter order matches its single call site in Task 2 Step 2.
