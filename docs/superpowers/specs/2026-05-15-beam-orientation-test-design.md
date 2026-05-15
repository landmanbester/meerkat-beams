# Beam orientation validation via dynamic-spectrum reconstruction

**Status:** draft — design only, no code yet.
**Author:** brainstormed with Claude, 2026-05-15.
**Branch context:** `dev001`.

## 1. Motivation

`scratch/beam_orientation_diagram.pdf` pins how the supplied MdV beam pixels are
meant to be oriented relative to the sky (azimuth N=0°, E=+90°; the "as
supplied" pattern is flipped vertically relative to the physical sky frame).
The current code in `meerkat_beams.utils.BeamWizard` encodes a specific
parallactic-angle rotation when evaluating the beam along a source's track
through the field (`get_source_coordinates` → `interpolate_beam`, wrapped by
`get_time_variable_beamgain`). We do not have a direct end-to-end check that
this convention is correct.

The dynamic-spectrum formulation in `scratch/Dynamic_Spectra.pdf` gives us a
clean falsifiable predication: if `get_time_variable_beamgain` returns the
correct time-varying Stokes Mueller `M_S(t,ν)` for a source at a known
off-boresight position, then solving the per-bin linear inverse
`M_S(t,ν) · B(t,ν) = V̄_S(t,ν)` against an MS of a stable primary calibrator
should recover the calibrator's known spectrum, flat in time. Any orientation
error leaves residual time structure correlated with parallactic angle.

PKS 1934-638 (RA 19:39:25.027, Dec −63°42′45.626″ — already in
`tests/conftest.py`) is the target. Spectrum is stable and well-modelled, and
the source is essentially unpolarised so Q/U/V should come out near zero.

## 2. Scope and non-goals

**In scope.**
- A standalone validation experiment under `scripts/`, not a pytest test (yet).
- Off-axis-pointed L-band observation of PKS 1934-638; the calibrator is held
  off boresight so parallactic-angle rotation modulates the beam value at the
  source position.
- Full-Stokes dynamic-spectrum solve `B(t,ν) = (I,Q,U,V)`.
- Array-average Jones used uniformly across all baselines (the cached L-band
  BDS is already array-averaged).
- Single-source field assumption: `R_b = V_b`, no `X_b` subtraction.
- Three diagnostic outputs: catalog-spectrum match, time-flatness metric, and
  a side-by-side comparison against deliberately-wrong-orientation control
  runs.

**Out of scope.**
- Per-antenna Jones (defer until the array-averaged result is trusted).
- Any new CLI command or core function under `src/meerkat_beams/`.
- Promotion to a pytest test (deferred until thresholds are calibrated against
  real output).
- Bandpass calibration. The MS is assumed pre-corrected for the bandpass; the
  on-axis frequency structure has already been absorbed, so `nstokes` (the
  center-normalised Stokes Mueller-row beam) is the correct lookup variable.
  On-axis `nstokes` is the identity by construction.
- Frequency/time averaging upfront. The MS is small enough to run unaveraged.
- Per-antenna and per-scan complications: assume a single scan on the
  calibrator field.

## 3. Mathematical setup

Following `scratch/Dynamic_Spectra.pdf` and absorbing the geometric phase
factor `K_b` into a per-baseline phase rotation done outside the solve:

```
R_b(t,ν) = M_b(t,ν) · B(t,ν) + ϵ_b,    ϵ_b ~ N(0, Σ_b)
```

For a single-site array with the BDS pre-collapsed to the array-average
Jones, `M_b(t,ν) = M(t,ν)` is independent of baseline. The closed-form ML
solution `B̄ = W⁻¹ M† Σ⁻¹ R`, `W = M† Σ⁻¹ M` then collapses to:

```
B̄(t,ν) = M(t,ν)⁻¹ · V̄(t,ν),
```

where `V̄(t,ν)` is the noise-weighted baseline-average of the
phase-rotated residuals (= visibilities, since `X_b ≡ 0` by assumption).

`mdv_beams_to_bds` already bakes the Jones → Stokes mapping into the
`nstokes` variable of the BDS. So:

- We convert the **observed** linear-feed visibilities to observed Stokes
  via a fixed 4×4 transform `T⁻¹` applied once after the baseline average:
  `V̄_S = T⁻¹ V̄_lin`.
- We look up `M_S(t,ν)` directly from `nstokes` via
  `BeamWizard.get_time_variable_beamgain(coord, times, freq, spi=None,
  var='nstokes', i, j)` for each `(i, j) ∈ {I, Q, U, V}²` and stack the 16
  results into a `(Nt, Nν, 4, 4)` complex tensor.
- We solve `M_S(t,ν) · B(t,ν) = V̄_S(t,ν)` per bin.

`get_time_variable_beamgain` is the function under test. The script does not
modify it; the wrong-orientation controls in Section 6 perturb the inputs or
the parallactic angle returned by `get_source_coordinates`, leaving the
function's code untouched.

## 4. Data flow

```
MS (V_lin, UVW, t, freq, WEIGHT_SPECTRUM, FLAG, phase_centre)
  │
  │  phase-rotate visibilities to (RA, Dec)_PKS1934-638
  ▼
  V'_lin(b, t, ν, corr)
  │
  │  noise-weighted average over unflagged baselines
  ▼
  V̄_lin(t, ν, corr)
  │
  │  V̄_S = T⁻¹ V̄_lin   (fixed 4×4 linear→Stokes)
  ▼
  V̄_S(t, ν)
  │
  │  solve 4×4 system per (t, ν)        ◀──── M_S(t, ν)
  ▼                                              ▲
  B(t, ν) = (I, Q, U, V)                          │
  │                                              │
  │                                       loop over (i, j) ∈ {I,Q,U,V}²:
  ▼                                       M_S[:,:, i, j] =
  zarr + diagnostic plots                   bw.get_time_variable_beamgain(
                                              coord=(ra, dec),
                                              times=times,
                                              freq=freq,
                                              spi=None,
                                              var='nstokes',
                                              i=i, j=j,
                                            )
                                            ▲
                                            │
                                  BeamWizard(band='L') reads the cached
                                  L-band BDS produced by mdv_beams_to_bds.
```

## 5. Pipeline stages

| # | Stage | Inputs | Outputs | Notes |
|---|---|---|---|---|
| 1 | Fetch MS | GDrive ID from `tests/conftest.py` (`test_ms_gdrive_id`) | Local MS path | Reuse the gdown/tarball pattern from `cache.py`; cache under `$MBEAMS_CACHE_DIR/test_ms/`. Idempotent — skip if already extracted. |
| 2 | Read MS | MS path | `vis[b,t,ν,corr]`, `weight_spectrum[b,t,ν,corr]`, `flag[b,t,ν,corr]`, `uvw[b,t]`, `time`, `freq`, phase centre | `dask-ms` (already a project dep). Assume single scan on the calibrator field. |
| 3 | Phase-rotate to source | `vis`, `uvw`, `freq`, source (RA, Dec) | `vis'` with PKS 1934-638 at phase centre | Direct implementation: `vis' = vis · exp(±2πi (uΔl + vΔm + w(Δn − 1)) / λ)`. Sign of the exponent and of the w-term are noted as candidate convention knobs (Section 6). |
| 4 | Baseline average | `vis'`, `weight_spectrum`, `flag` | `V̄_lin(t, ν, corr)` | `V̄ = Σ_b w_b v_b / Σ_b w_b` over unflagged baselines. Equivalent to `W⁻¹ M† Σ⁻¹ R` because `M` is baseline-independent. |
| 5 | Linear → Stokes | `V̄_lin` | `V̄_S(t, ν)` | Fixed `T⁻¹` (Section 8). |
| 6 | **Call function under test** | `BeamWizard(band='L')`, source coords, `times`, `freq` | `M_S(t, ν)` as `(Nt, Nν, 4, 4)` | Loop over 16 `(i, j)` pairs calling `bw.get_time_variable_beamgain(..., spi=None, var='nstokes', i=i, j=j)`. The unit under test. |
| 7 | Per-bin solve | `M_S`, `V̄_S` | `B(t, ν) = (I, Q, U, V)`; `cond_M(t, ν)` | `np.linalg.solve(M_S, V̄_S)` vectorised over (t, ν); flag bins where `cond_M > threshold`. |
| 8 | Write outputs | `B`, `cond_M`, `freq`, `time` | `dynamic_spectrum.zarr` | Minimal store, only what plotting needs. |
| 9 | Plots | DS zarr, catalog polynomial, control-run DS | PNGs in same dir | Spectra, waterfalls, control overlay (Section 9). |

## 6. Wrong-orientation controls

Three perturbations to `get_source_coordinates`'s returned position
angle, applied outside `BeamWizard` (so its code is untouched):

1. **`identity`** — pretend the beam is fixed on the sky (no parallactic
   rotation).
2. **`angle_sign_flip`** — replace `angle(t)` with `-angle(t)`.
3. **`angle_plus_pi`** — replace `angle(t)` with `angle(t) + π`.

(Names match the `perturbation` attr values listed in Section 9; the
unperturbed run is `perturbation = "none"`.)

For v1 these are the only controls we run. Additional convention knobs
held at their "expected correct" values during v1, to be systematically
flipped one at a time if the unperturbed run still looks bad:

- Sign of the phase-rotation exponent (step 3).
- Sign of the w-term in the same.
- `T` versus `T*` for the linear↔Stokes transform (step 5).
- Y-axis flip on the BDS beam pixels (would require rebuilding the BDS).

## 7. Code layout

```
scripts/
├── test_beam_orientation.py     # main entrypoint
└── beam_orientation/
    ├── __init__.py
    ├── ms_io.py                 # dask-ms read; returns named tuple
    ├── phase_rotate.py          # uvw + freq → per-baseline phase shift
    ├── mueller.py               # 16-pair loop wrapping
                                 # get_time_variable_beamgain;
                                 # linear↔Stokes T and T_inv; per-bin solve
    └── plots.py                 # PKS 1934-638 catalog polynomial,
                                 # waterfalls, control overlay
```

Why split: each helper has a single small purpose; the main script is
glue. Hosting under `scripts/` (rather than `src/meerkat_beams/`)
signals this is a validation experiment, not a shipped feature.
Promotion to a CLI command or pytest test is deferred to a follow-up.

## 8. Conventions and constants

**Linear ↔ Stokes transform.** For correlation ordering `(XX, XY, YX, YY)`
and Stokes ordering `(I, Q, U, V)`:

```
        ⎡ 1   1    0    0 ⎤
T = ½ · ⎢ 0   0    1   +i ⎥
        ⎢ 0   0    1   −i ⎥
        ⎣ 1  −1    0    0 ⎦
```

so that `V_lin = T · V_S`, and `V̄_S = T⁻¹ · V̄_lin` with `T⁻¹` precomputed.

**`nstokes` indexing.** Per `CLAUDE.md`, `nstokes` has dims `(stokes_i,
stokes_j, FREQ, Y, X)` with `stokes_i, stokes_j ∈ {I, Q, U, V}`. The
script's 16-pair loop uses the same ordering.

**Frequency.** L-band BDS, frequencies taken from the MS. No averaging.

**Noise weights.** `WEIGHT_SPECTRUM` is present in this MS (confirmed);
use it directly. No fallback path needed for v1.

**Source position constant.** Pull `ra` and `dec` from
`tests/conftest.py` (already defined there).

## 9. Outputs

All under `scratch/orientation_test/<run-name>/`:

- `dynamic_spectrum.zarr` with variables
  - `B`: `(time, frequency, polarization)` complex64; polarization =
    `("I", "Q", "U", "V")`.
  - `cond_M`: `(time, frequency)` float32 condition number.
  - and attrs: `source = "PKS 1934-638"`, `ra`, `dec`, `band = "L"`,
    `perturbation ∈ {"none", "identity", "angle_sign_flip",
    "angle_plus_pi"}`, plus the catalog polynomial coefficients used
    for comparison.
- `dyn_spec_I.png`, `dyn_spec_Q.png`, `dyn_spec_U.png`,
  `dyn_spec_V.png` — waterfalls per Stokes (with `cond_M`-bad bins
  masked).
- `mean_I_spectrum.png` — time-averaged recovered `I(ν)` with the
  PKS 1934-638 polynomial overlaid; residuals subplot below.
- `time_variation.png` — per-channel
  `std_t(B) / median_t(|B|)` for each Stokes; the floor sets the noise
  baseline.
- `control_overlay.png` — same "residual time variation" metric for the
  unperturbed run vs the three controls; the unperturbed run should be
  the clear winner if the convention is correct.

## 10. Pass / fail interpretation

The script does not assert pass/fail. Visual inspection of the four plots
is the decision procedure for v1. The intended reading:

- **Convention correct:** `I(ν)` matches the catalog polynomial to within
  the per-channel noise; Q/U/V flat near zero; per-channel time variation
  near the thermal-noise floor; all three control runs visibly worse on
  the control-overlay plot.
- **Convention wrong:** at least one of the above fails. Common failure
  modes to look for:
  - Bowtie/sinusoidal time structure in `I(t,ν)` that tracks parallactic
    angle → wrong rotation sign or sense.
  - Constant offset in `I(ν)` from catalog → on-axis normalisation off
    (unlikely since `nstokes` is normalised, but worth catching).
  - Q/U/V picking up real signal in correlation with time → cross-pol
    leakage convention off (would re-direct us to the `T` vs `T*`
    perturbation).

Thresholds for promotion to a pytest assertion are deferred until we have
the v1 plots and can pick numbers grounded in real output.

## 11. Risks and open items

- **`get_time_variable_beamgain` signature.** This design assumes
  `spi=None` returns a `(Nt, Nν)` complex array per `(i, j)` pair. To be
  verified before writing the script; if the actual return is reduced
  along an axis the call shape changes but the data flow doesn't.
- **Phase-rotation sign convention.** Held at the "expected correct"
  value for v1; flagged as a fallback perturbation if the parallactic-
  angle controls don't isolate the problem.
- **Concurrent MS download.** Same caveat as the BDS cache in `cache.py`:
  if two processes warm the MS cache simultaneously, the second may see
  a half-extracted directory. Mitigated by running this validation
  experiment from a single process — same posture as the rest of the
  cache.
- **No `X_b` subtraction.** Fine for PKS 1934-638 at L-band (~15 Jy
  dominates any sub-Jy field source) but documented as an assumption.
  Any orientation-correlated bias from confusing sources would be a
  failure mode worth ruling out via the controls.
