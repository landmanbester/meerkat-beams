---
type: Subsystem Notes
title: Beam orientation and (Y, X) map-order conventions
description: The settled (Y, X) rotation-averaged map order, parallactic-angle rotation averaging, and the still-provisional beam-orientation convention with its open M1 validation.
tags: [beam, orientation, conventions, rotation, parallactic, yx-order]
timestamp: 2026-07-27T09:48:17Z
last_verified_commit: 7dd4e67
---

# Beam orientation and (Y, X) map-order conventions

Two independent questions live under "beam orientation" in this repo, and
they must not be conflated:

1. **Index order of the rotation-averaged map arrays** — SETTLED. `(Y, X)`,
   pinned by tests, landed at commit `616906b`.
2. **Physical orientation convention of the sky→beam-pixel transform**
   (`get_source_coordinates`'s "0 is up, +90 is right") — PROVISIONAL. Not
   yet physically verified; the validation experiment currently does not
   recover a flat/unpolarised calibrator spectrum through the off-axis
   beam. Do not treat this as settled just because (1) is.

## Settled: `(Y, X)` map order

`get_rotation_averaged_beam` (`src/meerkat_beams/utils.py:426`) returns
`(mean, variance)` maps in `(Y, X)` index order — FITS convention, axis 0 =
m/north, axis 1 = l/east:

```python
if l.ndim == 1 and m.ndim == 1:
    ll, mm = np.meshgrid(l, m)          # utils.py:511, default indexing="xy"
elif l.ndim == 2 and m.ndim == 2:
    ...
    ll, mm = l, m                        # 2D inputs passed through as-is
```

For 1D `l`/`m` inputs the default `np.meshgrid(l, m)` (`"xy"` indexing, not
`"ij"`) gives shape `(len(m), len(l)) = (NY, NX)` directly — no transpose
needed. With `spi` given (or a single frequency) the returned shape is
`(NY, NX)`; otherwise `(NFREQ, NY, NX)`. 2D `l`/`m` inputs must already be
`(Y, X)`-shaped grids and are passed through in that orientation
(`utils.py:472-478` docstring).

Downstream consumers — breifast and the pfb-imaging `hci` command — consume
these maps as `(Y, X)` with **no on-receipt transpose**. See
`docs/wiki/beamwizard.md` for the surrounding `BeamWizard` API.

**Post-mortem (why this took a while to notice):** the array order used to
be `(X, Y)` (`np.meshgrid(l, m, indexing="ij")`). killick found the bug via
a covariance-map reference pixel — `rho[x, y] = 1.000` vs `rho[y, x] =
-0.15` — and patched breifast to transpose on receipt
(`killick-polishes-silver` @ `0a898bb`) as a stopgap. The bug was invisible
almost everywhere else because the MeerKAT rotation-averaged beam is nearly
circularly symmetric, so a transpose of the map barely changes it. Fixed at
commit `616906b`; breifast's on-receipt transpose is now stale and tracked
for removal by issue #14 (would double-transpose if left in place once
breifast adopts this package).

Pinned by (`tests/test_beam_wizard.py`):
`test_rotation_averaged_beam_1d_lm_returns_y_x_order`,
`test_rotation_averaged_beam_map_indexes_as_y_x`,
`test_rotation_averaged_beam_2d_lm`,
`test_rotation_averaged_beam_mismatched_2d_raises`.

## Rotation averaging

Two `BeamWizard` methods rotate through parallactic angle:

- `get_time_variable_beamgain` — beam gain along one source's track through
  the beam as parallactic angle rotates over the supplied `times`. With
  `spi` given, returns a frequency-averaged gain weighted by
  `(f/f0)**spi`.
- `get_rotation_averaged_beam` (`utils.py:426`, described above) — mean +
  variance over the same parallactic-angle rotations, but for a full l/m
  grid (degrees) instead of a single source track. `time_stepping`
  subsamples the time axis; `pixel_stepping > 1` computes on a coarser
  spatial grid and linearly upsamples back (`map_coordinates`) to the full
  grid; `chunk_size` bounds memory for large grids.

Pinned by `test_time_variable_beamgain_at_centre`,
`test_rotation_averaged_beam_on_axis`,
`test_rotation_averaged_beam_spi_collapses_freq`
(`tests/test_beam_wizard.py`).

## PROVISIONAL: the orientation convention

`get_source_coordinates` (`utils.py:265`) converts a source's separation
and position angle from the field centre into beam-pixel offsets:

```python
# convert to pixel position
# confused about angles, but experiments show that 0 is up and +90 is right
x = signs[0] * seps.deg * np.sin(angles.rad)
y = signs[1] * seps.deg * np.cos(angles.rad)
if swap:
    x, y = y, x
```

(`utils.py:292-296`). That inline comment is original and accurate: this
convention has **never been rigorously verified**, only assumed. It is not
equivalent in status to the `(Y, X)` map order above — do not present it as
settled.

**The M1 validation experiment.** The intended check is to recover PKS
1934-638's known-flat, unpolarised spectrum through the beam's off-axis
Mueller matrix as parallactic angle rotates (see the design context folded
into `docs/wiki/design-decisions.md`). As of this branch, the recovered
Stokes dynamic spectra for the offset pointings are **not flat/unpolarised**
as expected — the convention has not passed validation. This is called out
as the top-priority blocking item in PR #8's merge note.

**Falsification method.** `scripts/test_beam_orientation.py` runs the full
pipeline (download calibrator MS → phase-rotate → noise-weighted baseline
average → assemble Mueller → solve for the source spectrum → plot) once per
perturbation in a fixed dict (`scripts/test_beam_orientation.py:51-56`):

```python
PERTURBATIONS: dict[str, tuple[tuple[int, int], bool]] = {
    "none": ((1, 1), False),
    "flip_x": ((-1, 1), False),
    "flip_y": ((1, -1), False),
    "swap_xy": ((1, 1), True),
}
```

Each `(signs, swap)` pair is fed straight through to
`get_source_coordinates`'s `signs`/`swap` keyword arguments via
`scripts/beam_orientation/mueller.py::assemble_mueller`, which calls
`bw.get_source_coordinates(srcpos, times=times, loc=loc, signs=signs,
swap=swap)` before looping over the 16 `(i, j)` Mueller index pairs
(`mueller.py:91-121`). The success criterion for validating the convention
is that `flip_x`/`flip_y`/`swap_xy` must look **visibly worse** than `none`
(less flat, more spuriously polarised) for every offset field — that has
not yet been confirmed for all fields. Pinned (propagation of `signs`/`swap`
through `assemble_mueller`, not the physical-correctness question):
`tests/test_beam_orientation_mueller.py::test_assemble_mueller_signs_swap_propagate`.

**Convention knobs still under investigation** (per PR #8's M1 checklist):
phase-rotation exponent sign, w-term sign (see the docstring in
`scripts/beam_orientation/phase_rotate.py`, which validates
`V' = V * exp(+2πi(uΔl + vΔm + w(Δn-1))/λ)` as the convention being tested),
`T` vs `T*`, and a possible BDS Y-axis flip.

**Deferred to:**
- issue #9 — diagnose why the recovered offset-field spectra aren't flat
- issue #10 — run the `flip_x`/`flip_y`/`swap_xy` controls for all offset
  fields (Offset3/Offset4 still outstanding)
- issue #11 — document the verified convention (CLAUDE.md + code comments,
  replacing the "confused about angles" note) once resolved
- issue #12 — add an `integration`/`slow` regression test pinning the
  verified convention so it can't silently regress

## Validation tooling (implemented as designed)

The pieces below are implemented and tested; only the *physical-correctness
conclusion* they're meant to produce is still open.

- **MS selection + pointing-centre resolution**
  (`scripts/beam_orientation/ms_io.py`). `read_ms(path, field_id)` selects
  a single `FIELD_ID` group from the MS main table
  (`taql_where=f"FIELD_ID == {field_id}"`, `ms_io.py:121-131`).
  `_resolve_pointing_centre` (`ms_io.py:92-104`) is a two-tier resolver:
  first tries `_read_pointing_table`, which averages the MS `POINTING`
  table's `DIRECTION` over rows whose `TIME` falls in the selected field's
  scan window (`ms_io.py:64-89`); on any failure (missing table, missing
  columns, no matching rows) it falls back to a hardcoded
  `ORIGINAL_POINTING` dict keyed by `FIELD_ID` (`ms_io.py:31-37`).
  **Note:** that fallback dict's comment block records that the field names
  for `Offset1` and `J1939-6342` were originally swapped and have since
  been corrected in `ms_io.py:18-27` — cite `ms_io.py`, not the design
  spec, for the correct per-field `(ra, dec)` values.
- **Plots** (`scripts/beam_orientation/plots.py`): per-Stokes
  dynamic-spectrum (`dyn_spectrum`), time-profile (`time_profile`), and
  frequency-profile (`freq_profile`) plots of the apparent source, the
  recovered source, and the beam (Stokes-Mueller diagonal).
- **Calibrator MS caching** (`scripts/beam_orientation/download.py`): the
  PKS 1934-638 calibrator MS is cached at
  `cache.cache_root()/test_ms/<MS_BASENAME>/`
  (`MS_BASENAME = "pks1934_offset.ms"`, `download.py:25-39`) — same
  unlocked, single-process download posture as the per-band BDS cache (see
  `docs/wiki/beamwizard.md` / `CLAUDE.md` cache section).
- **Single source of truth for calibrator facts** (`tests/conftest.py`):
  `test_ms_gdrive_id` (line 20), the calibrator `ra`/`dec` strings (lines
  23-24), and the `CALIBRATOR_SPECTRUM` polynomial coefficients for PKS
  1934-638's known flux model (lines 28-36). Both
  `scripts/test_beam_orientation.py` and the test suite import these
  directly rather than duplicating them.

Pinned by `tests/test_beam_orientation_ms_io.py`,
`tests/test_beam_orientation_plots.py`.

## Cross-refs

- pfb-imaging `docs/wiki/image-and-beam-orientation.md` documents the
  downstream `hci` consumer: it fills `(ny, nx)` buffers directly from
  `get_rotation_averaged_beam`'s `(Y, X)`-ordered output (that page's §3
  cites this repo's `616906b`). It also independently pins that the
  MdV-derived beam grid (`bds.X`/`bds.Y`) is degrees, centred on 0,
  ascending — useful context if you're chasing the "BDS Y-axis flip" knob
  above. That page does **not** cover the physical beam-orientation
  convention question above; that is owned here (PR #8's M1).
- issue #13 — make pfb-imaging `hci` fully conformant with the `(ny, nx)`
  order this package now returns (some call sites there still assume the
  pre-`616906b` `(nx, ny)` order).
- issue #14 — drop breifast's on-receipt transpose once it adopts this
  package (see post-mortem above).
- issue #15 — port killick's per-pixel time-covariance accumulation and
  chase an outstanding ~24x amplitude discrepancy between per-channel and
  frequency-averaged beam tracks.

Sources: `src/meerkat_beams/utils.py:265-299` (`get_source_coordinates`),
`src/meerkat_beams/utils.py:426-511` (`get_rotation_averaged_beam`),
`tests/test_beam_wizard.py` (tests listed above),
`scripts/test_beam_orientation.py:1-56`,
`scripts/beam_orientation/mueller.py:91-131` (`assemble_mueller`),
`scripts/beam_orientation/ms_io.py:18-131`,
`scripts/beam_orientation/plots.py`, `scripts/beam_orientation/download.py`,
`scripts/beam_orientation/phase_rotate.py:1-13`, `tests/conftest.py:18-36`,
`tests/test_beam_orientation_mueller.py::test_assemble_mueller_signs_swap_propagate`,
`tests/test_beam_orientation_ms_io.py`, `tests/test_beam_orientation_plots.py`,
commit `616906b`, PR #8 (merge note, M1 checklist), issues #9, #10, #11,
#12, #13, #14, #15, `~/software/pfb-imaging/docs/wiki/image-and-beam-orientation.md`.
