---
type: Subsystem Notes
title: Beam orientation and (Y, X) map-order conventions
description: The settled (Y, X) rotation-averaged map order, parallactic-angle rotation averaging, the still-provisional beam-orientation convention with its open M1 validation, and the katbeam comparison probe that corroborates the BDS transpose but not the sign flips.
tags: [beam, orientation, conventions, rotation, parallactic, yx-order, katbeam]
timestamp: 2026-07-28T14:35:00Z
last_verified_commit: a62c7e1
---

# Beam orientation and (Y, X) map-order conventions

Three questions live under "beam orientation" in this repo, and they must not
be conflated. The first two are independent of each other:

1. **Index order of the rotation-averaged map arrays** — SETTLED. `(Y, X)`,
   pinned by tests, landed at commit `616906b`.
2. **Physical orientation convention of the sky→beam-pixel transform**
   (`get_source_coordinates`'s "0 is up, +90 is right") — PROVISIONAL. Not
   yet physically verified; the validation experiment currently does not
   recover a flat/unpolarised calibrator spectrum through the off-axis
   beam. Do not treat this as settled just because (1) is.

A third, narrower question sits underneath both: **whether the BDS's on-disk
trailing axes really are `(Y, X)`** as `mdv_beams_to_bds` labels them. That is
the converter-level question, and it now has independent evidence — see
"Independent probe: the katbeam comparison" below. Its transpose half is
corroborated; its sign-flip half is not addressed. Keep it distinct from (2),
which is about the sky→pixel transform, not the array layout.

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

The pfb-imaging `hci` command consumes these maps as `(Y, X)` with **no
on-receipt transpose** (confirmed by pfb-imaging's own
`docs/wiki/image-and-beam-orientation.md`). See `docs/wiki/beamwizard.md`
for the surrounding `BeamWizard` API.

breifast is a separate case: it has **not adopted this package yet**.
breifast's own beam-fetch path still produces the old `(X, Y)` order, so
its existing on-receipt transpose (`killick-polishes-silver` @ `0a898bb`)
is **currently correct** and must not be touched. Removing it now would
introduce a bug, not fix one.

**Post-mortem (why this took a while to notice):** the array order used to
be `(X, Y)` (`np.meshgrid(l, m, indexing="ij")`) in this package. killick
found the bug via a covariance-map reference pixel — `rho[x, y] = 1.000` vs
`rho[y, x] = -0.15` — and patched breifast to transpose on receipt
(`killick-polishes-silver` @ `0a898bb`) as a stopgap. The bug was invisible
almost everywhere else because the MeerKAT rotation-averaged beam is nearly
circularly symmetric, so a transpose of the map barely changes it. Fixed in
this package at commit `616906b`. Per issue #14, breifast's on-receipt
transpose will become stale — and should be removed — **once breifast
switches to consuming this package's `(Y, X)` maps directly**; until then
the transpose remains correct and issue #14 explicitly says not to action
it yet.

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
phase-rotation exponent sign, w-term sign — `scripts/beam_orientation/phase_rotate.py`'s
docstring records the sign convention currently being validated (a `+2πi`
exponent, `w·(Δn − 1)` w-term) and flags both signs as candidate flips if
the parallactic-angle controls above don't isolate the issue — `T` vs
`T*`, and a possible BDS Y-axis flip.

**Deferred to:**
- issue #9 — diagnose why the recovered offset-field spectra aren't flat
- issue #10 — run the `flip_x`/`flip_y`/`swap_xy` controls for all offset
  fields (Offset3/Offset4 still outstanding)
- issue #11 — document the verified convention (CLAUDE.md + code comments,
  replacing the "confused about angles" note) once resolved
- issue #12 — add an `integration`/`slow` regression test pinning the
  verified convention so it can't silently regress

## Independent probe: the katbeam comparison

`scripts/compare_katbeam.py` compares this repo's MdV-derived BDS beams
against `katbeam`'s analytic JimBeam model, and its orientation sweep is a
second probe of the axis question above. It is independent of the M1
experiment *in method* — an image-plane model comparison rather than a
visibility-based calibrator solve — but the two beam models are **not**
independent in data provenance; see "How well the two models actually agree"
below before leaning on it. Re-run with:

```bash
uv sync --group dev --extra full          # katbeam is a dev/test dependency
uv run python scripts/compare_katbeam.py --band L --freqs 900 1070 1284 1500 1650
```

It perturbs **our** maps (`none`/`flip_x`/`flip_y`/`swap_xy`) while holding
katbeam fixed, so the winning label reads directly as a statement about the
BDS convention. Scored by mainlobe RMS residual, averaged over frequency.

**Verdict is split. Read both halves.**

**1. The transpose is decisively rejected — `(Y, X)` corroborated.** Measured
on a cache-built L-band BDS across 900–1650 MHz:

| product | `none` | `swap_xy` | penalty |
|---|---|---|---|
| HH | 2.535e-3 | 1.440e-2 | ×5.68 |
| VV | 2.822e-3 | 2.829e-2 | ×10.03 |
| I | 2.544e-3 | 1.767e-2 | ×6.95 |

All three products agree and the margin is large, so the BDS trailing axes
are labelled the right way round — the `mdv_beams_to_bds.py:26`/`:130`
`Y,X` assumption (whose inline comment flags itself as unverified) is
corroborated. `fwhm_vs_freq.png` shows the same thing directly and more
simply: **both** models put the m-axis FWHM above the l-axis, and `ours_l`
tracks `katbeam_l` rather than `katbeam_m`. A transposed BDS would swap that
pairing.

**2. The sign flips are NOT discriminated. This probe says nothing about
them.** `flip_x` on VV scores ×1.05 against `none` — indistinguishable.
The reason is physical: MeerKAT's squint is ~0.05–1 arcmin while the BDS
pixel is 0.0625 deg = 3.75 arcmin, so the beam is effectively even in both
axes at this sampling and a mirror is nearly a no-op. **The `flip_x`/`flip_y`
sign conventions remain entirely with the M1 experiment**; nothing here
closes them.

### The one-pixel mirror artifact (fixed; do not reintroduce)

The sweep initially reported `flip_x`/`flip_y` penalties of ×17–21, which
looked like strong evidence and was entirely spurious. An even-sized grid
centred on a pixel is **not** symmetric about zero: the L-band X axis runs
`-4.0 .. +3.9375` with 64 negative samples, one zero, and 63 positive, so
reversing it displaces the zero point by a whole pixel. The naive mirror was
therefore a mirror *plus a translation*, and the translation dominated —
at 1284 MHz naive `flip_x` scored 5.21e-2 while a pure one-pixel roll with no
mirror at all scored 5.14e-2. Corrected, `flip_x` scores 2.37e-3 against
2.75e-3 for `none`.

`registration_roll()` (`scripts/compare_katbeam.py`) now rolls by
`2*i0 - n + 1` after a reverse; `orientation_sweep` always passes the
coordinate arrays. `swap_xy` needs no correction, because it displaces no
coordinate (the X and Y grids are identical) — which is why it stayed the
only trustworthy control throughout. Pinned by
`test_registration_roll_is_one_pixel_for_the_bds_style_even_grid`,
`test_apply_orientation_with_coords_reregisters_a_mirror_symmetric_map`, and
`test_orientation_sweep_does_not_penalise_a_symmetric_beam_for_flips`
(`tests/test_compare_katbeam.py`).

**Any future orientation tooling on this grid inherits the same trap.**

### How well the two models actually agree

Mainlobe (r < HWHM) Stokes I residuals are 1.4–3.5e-3 of peak, with
`median_frac_diff` inside 0.2%, and FWHM agrees to 0.1–0.3% on both axes
across 900–1650 MHz (±0.5% channel-to-channel, degrading to ±2–4% only at
the extreme band edges). **This agreement is not fully independent**:
katbeam's squint/FWHM tables were themselves measured by MeerKAT holography
at 60 deg elevation, so both models trace back to the same measurement
programme. Treat it as a consistency check, not as two independent models
converging.

Where they genuinely part company:

- **Cross-polarisation.** katbeam models `|Jhv|²+|Jvh|²` as exactly zero. Ours
  shows the expected four-lobe clover-leaf — nulling on axis, peaking off axis
  diagonally at ~±0.7 deg — reaching 6.8e-4 of peak at 900 MHz and rising to
  3.8e-3 at 1650 MHz.
- **Sidelobes.** katbeam's cosine taper keeps ringing with a 1/r² envelope and
  its own docstring disclaims sidelobe accuracy, so residuals are reported per
  radial region (`mainlobe` / `near` / `far`) and never aggregated field-wide.
  The BDS field spans ~8 HWHM, so a single number would be dominated by a
  region katbeam makes no claim about.
- **Azimuthal structure.** `radial_*.png` shades our ±1σ azimuthal scatter;
  katbeam's is essentially zero-width by construction.

katbeam caveats worth knowing: `cos(pi*rr)/(1-4*rr**2)` is 0/0 at `rr = 0.5`
(`r ≈ 0.42053` in FWHM units) so a grid point landing exactly there yields
NaN — non-finite samples are counted and reported (zero occurrences on this
grid). And PyPI's only release (0.1) predates the S-band model and carries a
narrower L table (900–1650 MHz vs 856–1712), which is why the dev/test groups
pin katbeam from git main.

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
- issue #13 (this repo) — track pfb-imaging `hci` conformance with the
  `(ny, nx)` order this package now returns. Per pfb-imaging's own
  `docs/wiki/image-and-beam-orientation.md` §7, the corrected `(Y, X)`
  construction is already implemented and validated there as of its
  `last_verified_commit`; treat issue #13 as a cross-repo tracking item to
  re-check, not evidence of a known-open bug in pfb-imaging today.
- issue #14 — breifast has not adopted this package yet; its current
  on-receipt transpose remains correct (do not touch it). The transpose
  will become stale, and should be dropped, once breifast adopts this
  package — see post-mortem above and issue #14's explicit "do not action
  until breifast actually switches" note.
- issue #15 — port killick's per-pixel time-covariance accumulation and
  chase an outstanding ~24x amplitude discrepancy between per-channel and
  frequency-averaged beam tracks.

Sources: `scripts/compare_katbeam.py` (band->model map, `registration_roll`,
`apply_orientation`, `orientation_sweep`, `residual_stats`),
`tests/test_compare_katbeam.py` (84 hermetic unit tests, including the
one-pixel-mirror regression guards),
`outputs/compare_katbeam/L/metrics.json` (generated, untracked -- regenerate
with the command above), `pyproject.toml` (katbeam git-main pin in the `dev`
and `test` groups), `src/meerkat_beams/core/mdv_beams_to_bds.py:26,130` (the
`Y,X` assumption this probe tests),
`src/meerkat_beams/utils.py:265-299` (`get_source_coordinates`),
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
#12, #13, #14 (`gh issue view 14`: breifast not yet adopted, transpose
currently correct, "do not action until breifast actually switches"), #15,
`~/software/pfb-imaging/docs/wiki/image-and-beam-orientation.md`.
