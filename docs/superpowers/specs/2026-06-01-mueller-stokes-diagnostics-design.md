# Per-Stokes source/beam dynamic-spectrum diagnostics

Date: 2026-06-01
Status: approved (design)

## Problem

`scripts/test_beam_orientation.py` recovers an unpolarized calibrator's Stokes
dynamic spectrum through a perturbed beam Mueller term and should yield a flat,
unpolarized result. The current results are not as expected, so we need richer
diagnostics to track down the cause.

The diagnostics added so far collapse the 4×4 Mueller term by averaging over the
Stokes / correlation axes and normalise by the per-bin median. Neither is useful
here: the average over Stokes/correlation hides exactly the per-Stokes structure
we need, and median normalisation destroys the absolute values we want to read
off. We need per-Stokes, absolute-valued time and frequency behaviour for both
the beam and the source so they can be matched up directly.

## Design

For each perturbation and each Stokes parameter `p ∈ {I, Q, U, V}`, emit the
**dynamic spectrum**, **time profile**, and **frequency profile** of three
quantities: the observed apparent source, the recovered source, and the beam.
All plots show the **real part** and are **not normalised** (absolute values).

### Quantities

Let `coh_to_stokes = mueller.coherency_to_stokes_matrix()` (= `S⁻¹`, the
coherency→Stokes map) and `S = inv(coh_to_stokes)` (Stokes→coherency).

| name | definition | units |
|---|---|---|
| **apparent** | `(coh_to_stokes · V̄_coh)[p]` — measured beam-modulated Stokes spectrum from the baseline-averaged coherency `V̄_coh` | Jy |
| **recovered** | `B[p]` — the solve output (`B = coh_to_stokes · M_C⁻¹ · V̄_coh`) | Jy |
| **beam** | `diag(M_S)[p]`, where `M_S = coh_to_stokes · M_C · S` is the Stokes Mueller | dimensionless gain |

- `apparent` depends only on the data, **not** on the perturbation, so it is
  identical in every perturbation folder. It is duplicated into each folder so
  each folder is self-contained for matching.
- `recovered` and `beam` are per-perturbation (they depend on `M_C`, which the
  `signs`/`swap` knobs alter via `assemble_mueller`).

### Profiles

- **dynamic spectrum** — `imshow` of `Re(·)` over (frequency [GHz] × time [s]).
- **time profile** — mean over the **frequency** axis → line vs time [s].
- **frequency profile** — mean over the **time** axis → line vs frequency [GHz].

Means use `nanmean` so blanked bins (see `cond` masking) are ignored.

### `cond` masking

Only the **recovered** quantity is masked: bins with `cond > 1e6` are set to
`NaN` before plotting, so ill-conditioned solve outliers do not wreck the colour
scale or the profile means. `apparent` and `beam` are never masked.

### Output tree

Replaces the current flat per-perturbation plots:

```
out_dir/<perturbation>/                 # none, flip_x, flip_y, swap_xy
    dynamic_spectrum.zarr               # unchanged: recovered Stokes B + cond
    I/  apparent_dyn_spec.png   apparent_time_profile.png   apparent_freq_profile.png
        recovered_dyn_spec.png  recovered_time_profile.png  recovered_freq_profile.png
        beam_dyn_spec.png       beam_time_profile.png       beam_freq_profile.png
    Q/  … (same 9)   U/ …   V/ …
```

9 plots × 4 Stokes = 36 PNGs per perturbation.

### `scripts/beam_orientation/plots.py`

**Add** three generic helpers, each taking a single `(Nt, Nf)` complex array and
plotting its real part:

- `dyn_spectrum(times, freq, data, out_path, *, title, cbar_label)`
- `time_profile(times, data, out_path, *, title, ylabel)` — `nanmean` over axis 1
- `freq_profile(freq, data, out_path, *, title, ylabel)` — `nanmean` over axis 0

**Remove** the superseded / explicitly-unwanted functions and their tests:

- `mueller_dynamic_spectrum`, `mueller_time_variation`, `mueller_time_profile`
  (average over Stokes/correlation and/or normalise by median),
- `waterfall`, `mean_spectrum`, `time_variation`, `control_overlay`,
- the now-unused `_mask` helper and the
  `from beam_orientation.calibrator import evaluate` import.

The `beam_orientation.calibrator` module itself is kept (its own unit tests in
`tests/test_beam_orientation_plots.py` stay).

### `scripts/test_beam_orientation.py`

- Before the perturbation loop: compute
  `apparent = np.einsum("ij,tfj->tfi", coh_to_stokes, V_coh)` and
  `S = np.linalg.inv(coh_to_stokes)`.
- Inside the loop, after solving: form
  `M_S = np.einsum("ij,tfjk,kl->tfil", coh_to_stokes, M_C, S)` and
  `beam_diag = np.einsum("tfpp->tfp", M_S)`.
- For each Stokes `p`, create `run_dir/<p>/` and emit the 9 plots, masking only
  the recovered array via a small `np.where(cond > 1e6, np.nan, B[..., p])`
  helper.
- Remove the old per-perturbation plot calls (`waterfall`, `mean_spectrum`,
  `time_variation`), the `mueller_*` calls, the cross-perturbation
  `control_overlay` call, and the now-unused `runs` / `mueller_runs` dicts.

### Unchanged

`read_ms`, phase rotation, the baseline-weighted average `V_coh`, Mueller
assembly, `solve_per_bin`, the coherency→Stokes conversion, `_write_zarr`, and
`srcpos` (PKS 1934 from `tests/conftest.py`).

## Verification

- `uv run ruff check .` and `uv run ruff format --check .` pass.
- `uv run pytest -m unit` passes (plot helper tests rewritten; calibrator tests
  unchanged).
- Smoke run per field where the MS is available (user-driven): each
  `out_dir/<perturbation>/<Stokes>/` contains the 9 expected PNGs.

## Out of scope

- Writing `apparent` / `M_S` to zarr (plots only).
- Cross-perturbation overlay plots (the per-folder layout supports matching by
  eye across folders with shared axes).
- Catalog-spectrum overlay on the recovered-I frequency profile.
