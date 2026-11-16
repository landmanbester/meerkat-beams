---
type: reference
title: BeamWizard interpolation and rendering internals
description: beam_model (mdv/katbeam), group (MM/MPM/MPMP) and average (pa/azimuth) selectors, interpolate_beam prefilter/off-cube/spline-order/freq-guard semantics and the freq_extrapolation policy, get_source_coordinates transforms, the lm_to_pixel rotation kernel, optional-image paths, get_time_freq_beam canonical dim_names, enrich_bds_xradio, partition_mueller and the pointing_centre= override, phase centre vs pointing centre (phase_centre=, sky_rotation_matrix, phase_to_pointing_lm), the numba beam_on_grid resampler for stored maps with its accuracy-vs-spacing table, and the nthreads/pa_tolerance/order speed options of get_rotation_averaged_beam.
tags: [beamwizard, phase-centre, pointing-centre, interpolation, scipy, zarr, xradio, utils, katbeam, baseline-groups, meerkat+, partition-mueller, mueller]
timestamp: 2026-11-16T17:55:14Z
last_verified_commit: 0935424
---

# BeamWizard interpolation and rendering internals

`BeamWizard` (`src/meerkat_beams/utils.py`) attaches to a beam dataset and,
optionally, an image (FITS or xradio zarr), and provides spline-based beam
interpolation, time-variable beam gain, rotation-averaged beam maps, and
full time/frequency zarr rendering. The beam dataset is either an MdV BDS zarr
(`beam_model="mdv"`, the default) or one synthesized from katbeam
(`beam_model="katbeam"`) — see [`## beam_model`](#beam_model) below. This page documents the interpolation
contract and the plumbing around it — the parts most likely to bite a
caller who changes a default without reading the inline comments.

## `interpolate_beam` — the prefilter contract

`_get_prefilter` (`src/meerkat_beams/utils.py:1043-1080`) runs
`scipy.ndimage.spline_filter` once per `(var, i, j, order)` key and caches
the result in `self._prefilters`. The cache's output dtype is **chosen by
variable kind, not unconditionally `float32`**:

```python
out_dtype = np.complex64 if np.iscomplexobj(da) else np.float32
```

(`utils.py:1069-1073`). Complex variables (`jones`, `njones`, `mueller`,
`nmueller`) cache as `complex64`; real variables (`stokes`, `nstokes`)
cache as `float32`. Passing a real output dtype for complex input would
make scipy implicitly promote it (a version-dependent `UserWarning`), so
the dtype is picked explicitly instead of relying on that promotion. Note
that an older revision of this project's notes described the cache as
unconditionally `float32` — that was true before the Mueller-term work
landed (commit `0f2a4f4`, "test: add tests for complex Mueller term") and
is no longer accurate; treat the dtype-aware rule above as current.

`interpolate_beam` (`utils.py:1194-1241`) calls `map_coordinates` against this
cached, already-filtered array with **`prefilter=False`**
(`utils.py:1238`) — an inline comment at `utils.py:1232` spells out why:
`_get_prefilter` already applied `spline_filter`, so flipping
`prefilter` back to `True` would double-filter the data. Do not do this.

Pinned by `test_subpixel_matches_direct_scipy`,
`test_prefilter_cached_dtype_is_float32`,
`test_prefilter_complex_var_is_complex64_without_warning`, and
`test_prefilter_is_cached` (`tests/test_beam_wizard.py`).

### Complex vars preserve the imaginary part end-to-end

The complex-dtype cache above only matters if the imaginary part survives
the full pipeline. It does: prefilter → `map_coordinates` → the
`pixel_stepping` upsample branch → the zarr write in `get_time_freq_beam`
all keep values complex when the source variable is complex. Pinned by
`test_interpolate_beam_complex_var_preserves_imaginary`,
`test_time_freq_beam_complex_var_preserves_imaginary`, and
`test_time_freq_beam_complex_var_upsamples`.

### Off-cube policy

`map_coordinates` passes `mode="constant", cval=0.0` explicitly
(`utils.py:1239-1240`) — coordinates outside the X/Y cube return `0`, not a
nearest-edge value or a NaN. Pinned by `test_out_of_range_xy_returns_zero`.

### Spline order

`_get_prefilter(order=3)` includes `order` in its cache key (`key = var,
i, j, order`, `utils.py:1046`), since `spline_filter` coefficients depend
on the requested order and callers requesting different orders must not
collide in the cache. `interpolate_beam` forwards its own `order`
parameter to both `_get_prefilter` and the `map_coordinates` call.

`spline_filter` accepts orders 2-5 only. For orders 0 and 1 `_get_prefilter`
caches the raw cube at the same dtype instead (nearest and linear
interpolation read the samples themselves), so `interpolate_beam(order=1)`
works; before #39 it raised "spline order not supported". Pinned by
`test_low_orders_skip_the_spline_prefilter`.

### Frequency guard and the extrapolation policy

`interpolate_beam` first calls `check_freqs` (`utils.py:1171-1192`), which
raises `ValueError` for any non-finite frequency, and — reporting both the
requested and available ranges in MHz — for any frequency outside `freq_range` (`utils.py:1162-1169`). Both are
public so callers can validate up front (pfb-imaging checks every chunk on its
driver before starting workers). What `freq_range` is depends on the wizard's
policy, set once in the constructor:

- `freq_extrapolation="raise"` (default): `freq_range` is the BDS `FREQ`
  range and nothing outside it is evaluated. Pinned by
  `test_out_of_range_freq_raises` and `test_check_freqs_raise_message_unchanged`.
- `freq_extrapolation="scale"`: `freq_range` widens by `max_freq_extrapolation`
  Hz each side (default one BDS channel spacing, `0.0` for a single-channel
  BDS), with 1 kHz (`FREQ_MATCH_ATOL_HZ`) of slack on the widened bounds
  because the 2026 bottom edge lands exactly on the margin. A frequency `nu`
  beyond the edge plane `nu_e` is evaluated on that plane with the pixel
  offsets from `(x0, y0)` scaled by `nu / nu_e` (`utils.py:1206-1228`), and one
  `log.warning` per wizard names the requested range, the BDS range and the
  margin. The error message beyond the margin gains
  `(extrapolation margin ±X MHz)`.

Only out-of-range rows of the coordinate array are rewritten, so in-range
results are **bit-identical** under both policies (`x0 + (xp - x0) * 1.0` is
not bit-exact, which is why the scaling is not applied unconditionally) —
pinned by `test_in_range_is_bit_identical_across_policies`. Scaling outward
at the top edge pushes pixels near the cube boundary off-cube, where D2's zero
applies. `"scale"` is refused for `beam_model="katbeam"`, whose own table
limits are deliberate, and any value other than `"raise"`/`"scale"` is
refused. Every public entry point (`get_time_variable_beamgain`,
`get_rotation_averaged_beam`, `get_time_freq_beam`, `partition_mueller`)
inherits the policy through `interpolate_beam`; no method takes a per-call
override. Rationale and measured accuracy: `design-decisions.md` D19. Tests:
`tests/test_freq_extrapolation.py` (unit) and
`tests/test_freq_extrapolation_integration.py` (real products).

## `get_source_coordinates` — sky-to-beam-pixel transform

```python
def get_source_coordinates(self, srcpos, times=None, loc=None, signs=(1, 1), swap=False):
```

(`utils.py:1133-1139`). For each requested time, transforms `srcpos` and the
pointing centre to `AltAz`, then computes the source's separation and
position angle relative to the centre. Those are converted to beam-pixel
offsets via the BDS's `dx`/`dy`/`x0`/`y0` attrs:

```python
x = signs[0] * seps.deg * np.sin(angles.rad)
y = signs[1] * seps.deg * np.cos(angles.rad)
if swap:
    x, y = y, x
return np.array(self.lm_to_pixel(x, y)), seps, angles
```

(`utils.py:1156-1160`); the degrees-to-pixel step is `lm_to_pixel`, below. The `signs`/`swap` keyword arguments are not dead
parameters — they are the exact lever the beam-orientation validation
tooling's `flip_x`/`flip_y`/`swap_xy` perturbations twiddle (see
`beam-orientation.md`); mirroring or transposing the sky→pixel map this
way is how that tooling falsifies candidate orientation conventions.

If `times` is omitted, the method falls back to `self.times`. The
FITS-image construction branch of `BeamWizard.__init__` sets
`self.times = None` (FITS headers carry no time axis), so calling
`get_source_coordinates` with no explicit `times` on a FITS-backed wizard
raises the documented `RuntimeError` — not an `AttributeError` from a
missing attribute. Pinned by
`test_fits_branch_sets_times_none_raises_runtimeerror` and
`test_source_coordinates_at_pointing_centre` (the latter also confirms a
source exactly at the pointing centre resolves to `(x0, y0)` with `sep=0`).

## `lm_to_pixel` — the rotation kernel

```python
def lm_to_pixel(self, l, m, angle=None) -> Tuple[np.ndarray, np.ndarray]:
    if angle is not None:
        l, m = m * np.sin(angle) - l * np.cos(angle), l * np.sin(angle) + m * np.cos(angle)
    xp = l / self.bds.attrs["dx"] + self.bds.attrs["x0"]
    yp = m / self.bds.attrs["dy"] + self.bds.attrs["y0"]
    return xp, yp
```

(`utils.py:1082-1106`). Public. It maps `(l, m)` offsets from the pointing
centre, in degrees, to BDS pixel coordinates. With `angle` (the parallactic
angle, radians) the sky-frame offsets are first taken into the beam frame.
That transform is a rotation **combined with a flip of `l`** (determinant
−1): `angle=0` gives `(-l, m)`, so it is not the same as `angle=None`, which
applies no transform. The two renders (`get_rotation_averaged_beam`,
`get_time_freq_beam`) use the `angle` path; `core/bds_to_fits.py` and
`get_source_coordinates` use the plain scaling.

This is the only place the renders' sky-to-beam transform and the
degrees-to-pixel scaling are written. It is **not** the whole orientation
convention: `get_source_coordinates` derives its beam-frame offsets
separately, from a separation and position angle with its own
`signs`/`swap`, so a change of convention (still provisional —
`beam-orientation.md`) is two edits, here and there, rather than four. It
returns a tuple rather than a stacked array because `bds_to_fits` passes 1-D
`l` and `m` of different lengths; callers stack with `np.array(...)` to get
`interpolate_beam`'s `xpyp`.

Three private helpers complete the kernel:

- `_parallactic_angles(times, loc=None, pointing_centre=None)` (`utils.py:1118-1131`) —
  the position angle from the pointing centre to the north celestial pole in
  `AltAz`, in radians. `pointing_centre=None` is the wizard's own; a supplied
  one is used for that call only (see `pointing_centre=` below).
- `_resolve_times(times)` (`utils.py:1108-1116`) — `times`, else the attached
  image's time axis, else the documented `RuntimeError`.
- `_coarsen_grid` / `_upsample_planes` (`utils.py:177-205`, module level) —
  `pixel_stepping`: subsample the l/m grid, then bilinearly interpolate the
  coarse planes back (`order=1, mode="nearest"`).

The extraction was behaviour-preserving: 55 output arrays from every caller
(real, complex, stepped, `pa_tolerance`, `azimuth`, `centre=`, FITS native
and resampled) were bit-identical before and after. What each caller still
owns is its `meshgrid` indexing — `"xy"` in `get_rotation_averaged_beam`
(D4), `"ij"` in `get_time_freq_beam` (xradio `l`, `m` order).

Pinned by `test_lm_to_pixel_maps_degrees_onto_the_bds_grid` (the scaling),
`test_lm_to_pixel_with_an_angle_pins_the_sky_to_beam_transform` (literal
values of the transform, including the `l` flip) and
`test_rotation_angle_is_the_textbook_parallactic_angle` (the angle against
the closed form, at two sites). Before these, negating the angle, swapping
the rotated axes or substituting a proper rotation passed the whole unit
suite. The bilinear order of `_upsample_planes` is pinned by
`test_pixel_stepping_upsample_is_bilinear` (a skipped pixel is exactly the
mean of its two evaluated neighbours; orders 0 and 3 both fail it), and its
coordinates by `test_time_freq_beam_pixel_stepping_upsamples`.

One error message changed: with no times available,
`get_rotation_averaged_beam` and `get_time_freq_beam` now raise "explicit
times must be supplied, since …" (was "times must be supplied, since …"),
the text `get_source_coordinates` already used.

`get_time_freq_beam` still lacks the `order=` and `nthreads` options of
`get_rotation_averaged_beam`; with one kernel they are a matter of passing
arguments through. It has `pointing_centre=` and `phase_centre=` (see
[Phase centre and pointing centre](#phase-centre-and-pointing-centre)).

## Optional-image construction

`image_name` is optional at construction time. Without it, `BeamWizard` is
BDS-only: `interpolate_beam` still works (it only needs the BDS), but the
`pointing_centre`, `phase_centre`, `wcs`, `l_grid`, and `m_grid` properties raise `RuntimeError`,
and `times` returns `None` rather than raising. Two methods populate
image-derived state after the fact:

- `attach_image(image_name)` (`utils.py:944`) — attaches a FITS or xradio
  zarr image (any path-like, including the CLI's `UPath`, since #38) after construction (or swaps the image on an existing
  wizard), populating the **phase centre** (the image's reference
  direction), WCS, default l/m grid, and time axis (the latter only for
  xradio-zarr images; FITS images still leave `times` as `None`). The
  pointing centre follows the phase centre unless set explicitly.
- `set_pointing_centre(pointing_centre, times=None)` (`utils.py:993`) — supplies just
  a pointing centre (and optionally a time axis). Without an image this is
  the image-free path used by `scripts/test_beam_orientation.py`; the WCS
  and default l/m grid remain unavailable, so methods that fall back to the
  image grid still need explicit `l`/`m`. With an image it overrides the
  default pointing (the image's phase centre), and the image grid is then
  mapped onto the pointing's SIN plane.

Construction also requires exactly one of `bds`/`band` — passing neither
or both raises.

Pinned by `test_construct_without_image`, `test_no_image_attrs_raise`,
`test_set_pointing_centre_unblocks_source_coordinates`,
`test_attach_image_unblocks_grid_methods`,
`test_beam_wizard_requires_one_of_bds_or_band`, and
`test_beam_wizard_rejects_both_bds_and_band`.

## `get_time_freq_beam` — canonical `dim_names` only

`get_time_freq_beam` (`utils.py:1802`) writes the full `(ij, time, freq,
x, y)` beam cube to a zarr store, computing each time/ij plane by rotating
the l/m grid by the parallactic angle and calling `interpolate_beam`.

Its `dim_names` parameter is **positionally interpreted**: index 0 is the
time-axis name, 1 the frequency-axis name, 2 the polarization/ij-axis
name, 3 the x/l-axis name, 4 the y/m-axis name. Only the canonical xradio
order is currently accepted:

```python
_CANONICAL_DIM_NAMES = ("time", "frequency", "polarization", "l", "m")
if tuple(dim_names) != _CANONICAL_DIM_NAMES:
    raise ValueError(...)
```

(`utils.py:1875-1882`). Real permutation of the data layout is not
implemented — passing a differently-ordered tuple would relabel the dims
without reordering the underlying array (silent corruption), so any
non-canonical tuple raises `ValueError` instead. Pinned by
`test_time_freq_beam_rejects_non_canonical_dim_names`.

Both the beam-variable dataset and the coordinate datasets are created
with `fill_value=None` (`utils.py:1986` and `:1681`) rather than zarr's
default `fill_value=0`. With the default, `xarray.open_zarr`'s
`mask_and_scale=True` treats stored `0.0` as "unwritten" and masks it to
`NaN` on read — which corrupts genuine zero coordinates (e.g. `l=0.0`) or
genuine zero beam pixels. `fill_value=None` means a default `open_zarr`
call keeps real zeros intact, no `mask_and_scale=False` workaround
required. Pinned by `test_time_freq_beam_open_default_keeps_real_zeros`
and `test_time_freq_beam_writes_zarr` (dims/shape/coords round-trip).

The store is created through `zarr_open_group`/`zarr_create_array`, not
zarr's own API, so it is written as zarr format 2 under zarr-python 2 or 3
(see `data-model.md`, "On-disk zarr format"). The helper keeps zarr 2's
`create_dataset` semantics, which the output depends on: `compressor=None`
(the `compress=False` path) writes the beam variable **uncompressed**, while
the coordinate arrays, which pass no compressor, get zarr's default.

The output dtype mirrors the source variable the same way the prefilter
cache does: `complex64` for complex beams, `float32` for real ones — see
"Complex vars preserve the imaginary part end-to-end" above.

## `enrich_bds_xradio`

`enrich_bds_xradio(zarr_path, bw, output_var, polarizations)`
(`utils.py:2034`) post-processes a zarr store already written by
`get_time_freq_beam` into xradio schema:

- converts the `l`/`m` coordinate arrays from degrees to radians in place;
- replaces the polarization coordinate with the given single-letter Stokes
  labels (`get_time_freq_beam` itself writes two-letter labels like `"II"`,
  `"QQ"`), recreated through `zarr_create_array` so it stays zarr v2;
- adds a dataset-level `direction` attribute block (`icrs` frame, `SIN`
  projection, reference position = the phase centre the `l`/`m` axes are about,
  not the pointing — `lonpole`/`pc` filled with xradio-schema defaults). That
  centre is the `phase_centre=` argument if given, else the one
  `get_time_freq_beam` recorded in the store's `phase_centre` attribute (its
  per-call `phase_centre=`, else the stored one, else the pointing), else
  `bw.phase_centre`. The record is consumed, so the output carries only the
  `direction` block. Before this, a per-call `phase_centre=` was mislabelled with
  the wizard's, and a bare wizard raised (Greptile review of #58; pinned by
  `test_enrich_labels_the_grid_with_the_phase_centre_it_was_rendered_about`);
- sets `image_type`/`units` attrs on the beam variable;
- re-consolidates zarr metadata so the result opens with
  `xarray.open_zarr` without `consolidated=False`.

Pinned by `test_enrich_bds_xradio_writes_xradio_schema`.

## `beam_model` — MdV holography or katbeam

`beam_model` selects where the beams come from. Everything downstream of
`self.bds` is identical either way: that is the whole design of the katbeam
support (see `design-decisions.md`).

| | `"mdv"` (default) | `"katbeam"` |
|---|---|---|
| source | BDS zarr on disk, via `bds_name` or `band` (cache) | `synthesize_katbeam_bds(band, ...)`, in memory |
| accepts `bds_name` | yes | **no** — raises; there is no BDS to read |
| requires `band` | only if no `bds_name` | yes |
| `npix`/`fov_deg`/`num_freq` | **no** — raises | yes, optional |
| variables | all six | `njones`/`nstokes`/`nmueller` only |
| on-axis `njones` | identity | identity (same normalisation applied) |
| download | yes, on cache miss | never |

`npix`, `fov_deg` and `num_freq` are rejected for `"mdv"` rather than silently
ignored, since a caller passing them to the MdV path has misunderstood which
model they are configuring.

### Unnormalised variables are rejected for katbeam

A katbeam dataset has no `jones`/`stokes`/`mueller` (katbeam beams are on-axis
normalised by construction). `_get_prefilter` guards on this before touching
`self.bds[var]`, so the caller gets an actionable message instead of a bare
`KeyError`:

> `var='stokes' is not available for beam_model='katbeam' (katbeam beams are
> on-axis normalised by construction; use 'nstokes'). Available: njones,
> nmueller, nstokes.`

An unknown `var` on either model raises from the same guard, naming what is
available. Pinned by `test_katbeam_wizard_rejects_unnormalised_variables`.

### The off-cube policy still applies

`interpolate_beam`'s `mode="constant", cval=0.0` is unchanged for a synthesized
dataset, so a katbeam beam falls to a hard `0` beyond `fov_deg` even though
`JimBeam` would happily evaluate there. This is deliberate — the two models
behave identically off-cube — and is pinned by
`test_beam_falls_to_zero_outside_the_synthesized_grid`.

## `get_rotation_averaged_beam` — `average="pa"` or `"azimuth"`

`average` chooses which angles the beam is averaged over. The `(mean, variance)`
return contract, the `(Y, X)` output order, chunking, `pixel_stepping` and `spi`
handling are identical in both modes; only the angle array differs.

| | `average="pa"` (default) | `average="azimuth"` |
|---|---|---|
| angles | parallactic angles at `times`, via `AltAz` + position angle to NCP | `linspace(0, 2π, num_angles, endpoint=False)` |
| needs `times` | **yes** — raises `RuntimeError` without them | no |
| uses `time_stepping`, `loc` | yes | no, both irrelevant |
| result | reflects the observation's real PA coverage | circularly symmetric |

`num_angles` (default 64) applies only to `"azimuth"` and must be at least 2 —
a single angle is not an average, so `0`, `1` and negatives all raise. Pinned by
`test_degenerate_num_angles_raises`.

Use `"azimuth"` when full PA tracking is overkill, which it usually is for
image-space mosaicing. The two agree when the supplied times give PA coverage
spanning a full turn (`test_azimuthal_matches_pa_average_given_full_pa_coverage`),
and the azimuthal map is invariant under both axis flips and transposition
(`test_azimuthal_average_is_circularly_symmetric`).

### Speed: `nthreads`, `pa_tolerance`, `order` (#39)

The cost is one `map_coordinates` over the whole grid per angle. For
pfb-imaging's call (1024² grid, one frequency, a 37-integration piece) that
made this method 90% of `pfb imager` pass-1 task time, on one core. Three
arguments trade it down. The defaults (`1`, `0.0`, `3`) reproduce the old
result exactly, and `partition_mueller` forwards all three.

| argument | what it does | effect on the result |
|---|---|---|
| `nthreads` | splits the pixels into at least `nthreads` chunks and runs the per-chunk angle loop on a thread pool; `map_coordinates` releases the GIL. `chunk_size` bounds the pixels in flight **across all threads** (each gets `chunk_size // nthreads`), so the working set does not grow with `nthreads` | none: each pixel's sum is untouched, so the result is **bitwise identical** (`test_rotation_averaged_beam_threads_are_bitwise_identical`; `test_threads_really_use_a_pool` checks the pool is actually used) |
| `pa_tolerance` | bins angles on `round(angle / tol)` (degrees, at most 180), evaluates each group once at its **mean** angle and weights it by its size. A group spans less than `tol`, but two angles closer than `tol` can still land in neighbouring bins | second order in a group's spread: evaluating at the mean cancels the first-order term |
| `order` | spline order 0-5, as on `interpolate_beam`; anything else is refused before `_get_prefilter` caches a cube | order 1 is 0.5% of the peak off cubic, order 2 is 5e-5 |

Measured on the MdV-2026 L band, 1024² over the cube, 37 × 8 s integrations:

| | MPM (complex) | MM |
|---|---|---|
| defaults | 12.7 s | 6.5 s |
| `nthreads=4` | 3.8 s (8: 2.3 s, 16: 1.7 s) | |
| `nthreads=4, pa_tolerance=0.1` | 1.07 s | 0.47 s |

Merging at tolerances up to 1° changes the map by ~1e-7 of the peak within 1°
of the centre, at every hour angle tried, which is the float32 floor. The
larger differences (up to ~2e-3) occur only at the cube corners, where the
`cval=0` off-cube policy leaves a hard edge that a rotation crosses. With
bins of at most 180°, angles either side of the 0/2π wrap land in different
groups; that costs one extra evaluation and no accuracy. A wider bin could
hold 8° and 351° together, whose mean is ~180°: at 721° that was a 1.8e-2
error, which is why the cap exists. A tolerance so small that the binning
overflows (1e-310) is refused rather than collapsing every angle into one
group. With merging, the variance is computed from the representative angles,
so it is approximate in the same way; a merged group's weighted sum is
widened to the accumulator's float64 first, so it does not depend on numpy's
casting rules.

`order=1` is 3-8× faster than cubic (more single-threaded) but is the one option whose error
shows inside the field, so cubic stays the default.

## `group` — baseline-group beams

`group` selects a MeerKAT/MeerKAT+ baseline group instead of a single
telescope. It is a third selector alongside `beam_model` and `average`, and
it changes only *what `self.bds` holds* — every method below works against a
group dataset unchanged.

| condition | behaviour |
|---|---|
| `group=None` (default) | exactly today's behaviour; legacy MdV generation |
| `group` with `bds_name` | `ValueError` — a group needs two stores; pass `band` |
| `group` without `band` | `ValueError` |
| `group` with `beam_model="katbeam"` | `ValueError` — katbeam has no MKE model and gives only power beams |
| `group` not in `{MM, MPM, MPMP}` | `ValueError` naming the three |
| `group` with a band other than `"L"` | `ValueError` naming the supported group bands |

`cache.ensure_group_bds(band, group)` returns the `(p, q)` BDS paths —
the same path twice for `MM`/`MPMP`, which is then opened once —
and `_build_group_bds` cross-multiplies their Jones cubes into an
**in-memory** `xarray.Dataset`. There is no file behind a group wizard, so
nothing reopens it; at the MdV-2026 grid (64 x 64 x 63) the four Mueller/Stokes
variables come to roughly 25 MB. `_get_prefilter` caches per
`(var, i, j, order)` against that in-memory dataset exactly as it does for a
file-backed one, so a full 16-element Stokes assembly holds 16 prefiltered
cubes on top of it.

`jones`/`njones` are absent for `MPM`, and `_get_prefilter` raises a
group-specific message saying a cross baseline has no single Jones matrix,
mirroring the katbeam hint. The schema, the `p = MeerKAT` convention and the
conjugation rule for the reversed ordering are in
[data-model.md](data-model.md).

## `partition_mueller` — one partition's Stokes Mueller block

`partition_mueller` answers "what is the average beam over this chunk of data, on
this image grid?" and returns
`(len(stokes_out), len(stokes_in), NY, NX)` so a caller can turn an intrinsic
Stokes model into an apparent one: `apparent[i] = sum_j M[i, j] * model[j]`.

It exists so `pfb-imaging` and `QuartiCal` stop each carrying the same loop over
`get_rotation_averaged_beam`, the same `(Y, X)` orientation convention, and the
same `stokes`-vs-`nstokes` decision (issue #27, consumed by ratt-ru/pfb-imaging#278
and landmanbester/pfb-model-spec#22).

**It is a method, not a free function, and that is the point.** `_get_prefilter`
caches spline coefficients on `(var, i, j, order)` on the wizard, so the first call
pays for `len(stokes_out) * len(stokes_in)` filtered cubes and every later
partition pays nothing. Hold one wizard per worker and reuse it. A free function
would either rebuild the wizard per chunk — fatal in a Ray worker's inner loop — or
need a module-level cache.

The beam source is whatever the wizard was built with. There is no `group=`
argument: partitioning visibilities into baseline groups happens outside this
package, so a mixed array means one wizard per group.

Contract, all four parts pinned by `tests/test_partition_mueller.py`:

| | |
|---|---|
| orientation | `(Y, X)` — axis -2 is m/north, axis -1 is l/east, inherited from `get_rotation_averaged_beam`. A non-square-grid test turns a transposed *output array* into a shape error. Transposed *content* — an `xp`/`yp` swap, which keeps the shape — is pinned one level down, by `tests/test_beam_wizard.py` plus this method's exact-equality-with-`get_rotation_averaged_beam` test; it is not directly testable on the synthetic fixture, whose beam is near-radially-symmetric (see the trap noted in `tests/test_beam_wizard.py`). |
| normalisation | `normalised=True` (the default) reads `nstokes` and is what apparent-flux prediction wants: it is pre-multiplied by the inverse of the central-pixel Jones matrix, so the on-axis response is the identity and the model's flux scale survives. `normalised=False` reads `stokes`, which additionally folds in the absolute voltage gain — degenerate with the flux scale that calibration has already set. |
| no `1/n` | The bare beam. The wgridder's geometric n-term stays on the caller's side (pfb-imaging D22). |
| dtype | `float32` for a single telescope and for `MM`/`MPMP`; `complex64` for `MPM`. See [D13](design-decisions.md) — taking `.real` here would undo the fixed `p = MeerKAT` convention. |

Pure numpy: no dask, no Ray. The two consumers distribute differently, so
distribution stays in the applications.

Memory, worth knowing before a worker loops on it — one prefiltered cube per
`(var, i, j, order)` key, held for the wizard's lifetime:

| beam dataset | one cube | `stokes_in="I"` (4) | full 4×4 (16) |
|---|---|---|---|
| legacy L, 128×128×1024 float32 | ~67 MB | ~270 MB | ~1.1 GB |
| MdV-2026 L, 64×64×63 float32 | ~1 MB | ~4 MB | ~16 MB |
| MdV-2026 L group `MPM`, complex64 | ~2 MB | ~8 MB | ~32 MB |

`weights=` is accepted and must be `None`: it is reserved for a per-timestamp
weight in the parallactic-angle average (the visibilities in a chunk are not
uniformly weighted — flagging and varying integration time see to that) and raises
`NotImplementedError` otherwise. Accepting the keyword now means adding it later is
not a change consumers have to track.

A scalar `Time` (one timestamp, not wrapped in a list) is reshaped and accepted; an
empty `Time` raises, rather than dividing by zero and returning a NaN map. `freq`
must be scalar — a numpy scalar or 0-d array is fine, a 1-element array raises and
says to pass `float(freq)`, because `float()` on an array with an axis is a
`TypeError` that tells the caller nothing.

### `pointing_centre=` on `get_rotation_averaged_beam`

`get_rotation_averaged_beam` grew `pointing_centre: Optional[SkyCoord] = None` (it
was `centre=` until 0.4.0) to serve this. `None` uses `self.pointing_centre`,
raising if there is none; a supplied `SkyCoord` is used for the parallactic-angle
computation and **not stored**, so one wizard serves many pointings. The
alternative — `set_pointing_centre` per chunk — mutates an object shared across
chunks, which is why it is not what `partition_mueller` does. `partition_mueller`
takes `pointing_centre=` (was `field_centre=`) and `phase_centre=` and passes both
through.

## Phase centre and pointing centre

Issue #57. The **pointing centre** is where the dishes track: the beam is centred on
it and turns about it with parallactic angle. The **phase centre** is the tangent
point of an image or model grid, the origin of its `l`/`m` (both in `GLOSSARY.md`).
They differ for a model imaged at a shifted phase centre and for a mosaic field.
Until 0.4.0 the code had one `centre` for both; 0.4.0 renamed it (hard, no aliases):

| until 0.3 | from 0.4.0 |
|---|---|
| `BeamWizard.centre` | `BeamWizard.pointing_centre`; new `BeamWizard.phase_centre` |
| `set_field_centre(centre, times)` | `set_pointing_centre(pointing_centre, times)` |
| `get_rotation_averaged_beam(centre=)` | `pointing_centre=`, plus new `phase_centre=` |
| `partition_mueller(field_centre=)` | `pointing_centre=`, plus new `phase_centre=` |
| `get_time_freq_beam` (wizard's centre only) | `pointing_centre=`, `phase_centre=` |
| `bds-to-xradio` (image centre only) | `--pointing-centre` |

**Defaults, so the single-field case is bit-identical to 0.3.** `attach_image` stores
the image's reference direction as the phase centre; `pointing_centre` returns the
explicitly set one, else that phase centre. A method's `phase_centre=None` means the
stored phase centre, else the pointing centre. When phase and pointing are the same
direction `sky_rotation_matrix` returns exactly `np.eye(3)` and nothing is mapped.
A wizard with no stored phase centre needs no pointing at all for
`average="azimuth"`; passing `phase_centre=` there makes it need one.

One consequence worth knowing: on a wizard **with an attached image**, passing
`pointing_centre=` per call means "the image grid seen from that pointing" (the grid
stays about the image's phase centre). In 0.3, `centre=` simply moved the whole
grid's origin. On a BDS-only wizard (what `partition_mueller` callers hold) nothing
changes.

**The geometry** is two module-level functions in `utils.py`, the only place it is
written (as `lm_to_pixel` is for the parallactic rotation):

- `sky_rotation_matrix(pointing_centre, phase_centre)` (`utils.py:230`) —
  `M = R(pointing)ᵀ R(phase)`, `R` holding the local east, north and radial unit
  vectors as columns; both centres go to ICRS first.
- `phase_to_pointing_lm(l, m, M)` (`utils.py:249`) — `s_f = (l, m, √(1−l²−m²))`,
  `s_0 = M s_f`, `(l', m') = s_0[:2]`, in float64. Returns a `valid` mask, False off
  the sky (`l²+m² ≥ 1`) or on the pointing's far hemisphere (`s_0[2] ≤ 0`); there
  `l'`, `m'` are `OFF_SKY_DEG` (1e6 deg), which `interpolate_beam` treats as
  off-cube, so the beam is 0 (D2).

`l`/`m` here are degrees in the sense of `rad2deg` of the SIN direction cosine. That
is also what the BDS grids are: MdV confirmed they are feed-plane direction cosines
expressed in degrees (ska-sa/katbeam#5). The feed plane is the pointing's SIN
tangent plane turned by parallactic angle, so phase → pointing on the sky followed
by `lm_to_pixel`'s existing rotation is exact, not a flat-sky approximation.

The render methods apply the map **after** `_coarsen_grid`
(`BeamWizard._phase_rotation`, `_to_pointing_frame`). It is pointwise, so mapping
the coarse points is exact, and the upsampling still runs in output-grid index
space, where the beam is smooth. The parallactic angles, `pa_tolerance` binning and
threading are untouched.

Why evaluate at mapped coordinates rather than reproject a pointing-centred map:
degridding consumers (pfb-imaging `degrid`, QuartiCal via pfb-model-spec's
`BeamTerm`) already take a coarse block on the model's own separable axes from
`partition_mueller`. Evaluating the beam directly at the mapped coordinates is one
spline evaluation with no second interpolation, and `BeamTerm` needs no change. A
fast resampler for beam maps that are already stored (pfb-imaging `hci`) is the
second half of #57, `beam_on_grid` below. See D21.

Pinned by `tests/test_phase_centre.py`: the identity is exact for equal centres; the
rotation is proper and orthogonal; the pointing direction maps to the origin with
`l > 0` for a pointing east of the phase centre; `phase_to_pointing_lm` matches an
astropy SIN→SIN `WCS` reprojection to 1e-9 deg at a 1.7° offset; off-sky and
far-hemisphere directions are masked; `phase_centre=` equals the 2D path fed the
mapped coordinates exactly (also with `pixel_stepping=3` and `pa_tolerance=5`); and
the beam peak lands on the pointing's pixel in the image grid, in
`get_rotation_averaged_beam` and through `bds_to_xradio(pointing_centre=...)`.
Transposing `M`, flipping the sign of `l`, flipping the east vector, or dropping the
hemisphere test each fails at least one of them. The synthetic beam is circular, so
these pin the translation; orientation under parallactic rotation stays with
`beam-orientation.md`.

## `beam_on_grid` — stored maps onto a phase-centre grid

```python
beam_on_grid(beam, l_beam, m_beam, pointing_centre, phase_centre, l_out, m_out, out=None, nthreads=1)
```

Module-level in `utils.py`, not a method: it uses no wizard state, so a caller
holding only stored arrays (a per-partition beam kept by pfb-imaging#401, or a
small field-centred map `hci` evaluates once) does not open a BDS to place them.
It is the fast path; `get_rotation_averaged_beam(phase_centre=...)` on the image
grid is the exact one.

- `beam` is `(..., NYb, NXb)` in (Y, X) order, real or complex, on uniform
  `l_beam` (ascending or descending) and `m_beam`, degrees about the pointing
  centre. Every leading plane is resampled with one set of indices and weights
  per pixel.
- `l_out`/`m_out` are 1-D, degrees about the phase centre, any order — the
  caller's own convention (pfb-model-spec `pixel_lm` with `flip_u` and `--target`
  offsets, in degrees). No `flip_u`, `x0` or `l0` lives in this package.
- Per output pixel: `phase_to_pointing_lm`'s map (inlined), fractional beam
  index, bilinear gather. **0** off the sky, on the pointing's far hemisphere,
  or when the fractional index leaves `[0, n − 1]` on either axis by more than
  `_resample.EDGE_TOLERANCE_PIX` (1e-6 pixel) — so no half-pixel fringe (D2),
  but rounding never drops an edge sample: resampling onto a `linspace` beam
  grid itself lands its last sample at index `47.000000000000014`, which a
  strict bound zeroed. Indices inside the tolerance are snapped onto the bound.
  No `1/n` term.
- `out=` takes a caller-owned C-contiguous `(..., NY, NX)` buffer (refused if
  real for a complex beam) and is returned; `nthreads` splits output rows.
  `out` may be `beam` itself: overlapping memory is detected
  (`np.shares_memory`) and the input copied first, since a reordered output
  grid would otherwise read samples it has already overwritten.
- **Concurrency.** `nthreads=1` (the default) runs `_resample.bilinear_serial`,
  plain `njit(nogil=True)`, on the calling thread: any number of Python threads
  (dask workers, thread pools) may call at once. `nthreads > 1` runs
  `bilinear_parallel` (`parallel=True`) under the process-wide
  `_resample.PARALLEL_LOCK`, because numba's `workqueue` threading layer — what
  numba falls back to without TBB or OpenMP, including on `python:3.11-slim`,
  this package's container base — aborts the process (exit 134) when two threads
  launch parallel regions at once. Concurrent threaded calls therefore queue;
  each still uses its `nthreads`.
- Refuses non-uniform axes and a beam whose trailing shape is not
  `(len(m_beam), len(l_beam))`. "Uniform" is judged against the line through
  the axis's end points (the step is `(last − first)/(n − 1)`, so both end
  points land on integer indices), with a tolerance of `1e-6·step` plus the
  axis's own storage precision: a float32 `linspace` passes, a moved point
  does not.

Both kernels share one row function (`_resample._fill_row`, inlined), so the
arithmetic has one source; its loop invariants (`rot` entries, the
cosine-to-pixel scales) are hoisted by hand, without which the serial kernel was
~35% slower than the parallel one at one thread. They are `cache=True` and
imported only when `beam_on_grid` is called — importing `utils` does not load
numba. First call per dtype compiles in ~0.7 s, or loads from the on-disk cache
in ~0.2 s. At 8748², 4 float32 planes: 0.86 s serial, 0.17 s on 8 threads. Why
numba and not numpy: D22.

**How dense a stored map must be.** Bilinear error is second order in the stored
step `h`. Measured on the legacy L-band BDS at 1284 MHz, a 4 h track, the pointing
0.5° from the phase centre, output 0.01° cells over ±1.6°, against
`get_rotation_averaged_beam(phase_centre=...)` on that grid (max error over the
whole output, relative to the on-axis peak for I←I and to its own peak for Q←I):

| stored step (deg) | grid over ±2.4° | I←I max | Q←I max | I←I rms |
|---|---|---|---|---|
| 0.0625 (BDS native) | 78² | 4.2e-3 | 6.7e-3 | 6.2e-4 |
| 0.03125 | 155² | 1.0e-3 | 1.7e-3 | 1.5e-4 |
| 0.015625 | 308² | 2.6e-4 | 4.2e-4 | 3.9e-5 |
| 0.0078125 | 615² | 6.6e-5 | 1.1e-4 | 9.7e-6 |
| 0.00390625 | 1230² | 1.6e-5 | 2.7e-5 | 2.4e-6 |

Each halving of `h` quarters the error. Measured again at 1700 MHz the I←I
column is 6.7e-3, 1.6e-3, 4.2e-4, 1.1e-4, 2.6e-5 (Q←I 1.0e-2 … 4.0e-5). In units
of the beam width the two frequencies collapse onto one rule:

    max error (fraction of peak) ≈ (h / (λ/D))²     (prefactor 1.05 at 1284 MHz, 0.96 at 1700 MHz)

This is exactly the sizing rule pfb-imaging already uses for its stored beam
axis (`utils/beam.py: beam_grid`, cell `= sqrt(tol)·λ_min/D`, default
`BEAM_CELL_TOL = 1e-4`, i.e. h ≈ λ_min/(100 D)), now confirmed against the exact
path. The issue's guess that the native BDS grid "should be enough" holds only at
the ~0.4–0.7% level. Pick `h` from the error budget, not from the BDS. Reproduce with
`scripts/beam_on_grid_accuracy.py` (a study, not a test: it needs the cached
L-band BDS; flags change the offset, frequency and grids).

Pinned by `tests/test_beam_on_grid.py`: an independent oracle (astropy
`pixel_to_pixel` between two SIN `WCS`s plus scipy `map_coordinates(order=1)`,
float32 and complex64, ~half the pixels uncovered) to 2e-5; a beam linear in the
pointing frame reproduced to 1e-9 by closed-form SIN formulae with no
interpolation library; equal centres equal to scipy's separable
`RegularGridInterpolator` to 1e-12; a descending beam axis; leading dims, `out=`
and threads (bit-identical to serial); the refusals; the far hemisphere (the
phase-centre grid seen from its antipode is all zero); and a densely stored
wizard map placed with `beam_on_grid` agreeing with the exact `phase_centre=`
path to 2e-3 of peak; and concurrent callers (48 calls over 8 Python threads,
mixed `nthreads`, in a subprocess under both the default and the `workqueue`
threading layer) exit cleanly with identical results. Swapping an interpolation
neighbour, negating `l`, loosening the edge to `n` or dropping the hemisphere
test each fails at least one; removing the lock and the serial path fails the
`workqueue` case. The built container (`python:3.11-slim`, `workqueue`) runs
concurrent threaded calls cleanly. From the Greptile review of #60:
`test_resampling_onto_the_beam_grid_itself_keeps_every_sample`,
`test_out_may_be_the_input` and `test_float32_axes_are_uniform`, each failing
on the code before the fix.

## Sources

- `src/meerkat_beams/utils.py:208-279` (`OFF_SKY_DEG`, `_local_frame`,
  `sky_rotation_matrix`, `phase_to_pointing_lm`), `utils.py:890-1041`
  (`pointing_centre`/`phase_centre` properties, `attach_image`,
  `set_pointing_centre`, `_phase_rotation`, `_to_pointing_frame`);
  `src/meerkat_beams/core/bds_to_xradio.py` (`parse_pointing_centre`);
  `tests/test_phase_centre.py`; ska-sa/katbeam#5 (grid units)
- `src/meerkat_beams/utils.py` (`beam_on_grid`, `_uniform_axis`),
  `src/meerkat_beams/_resample.py` (`_fill_row`, `bilinear_serial`,
  `bilinear_parallel`, `PARALLEL_LOCK`),
  `tests/test_beam_on_grid.py`, `scripts/beam_on_grid_accuracy.py`

- `src/meerkat_beams/utils.py:1043-1080` (`_get_prefilter`)
- `src/meerkat_beams/utils.py:1133-1160` (`get_source_coordinates`)
- `src/meerkat_beams/utils.py:177-205,1082-1131` (`_coarsen_grid`, `_upsample_planes`,
  `lm_to_pixel`, `_resolve_times`, `_parallactic_angles`)
- `src/meerkat_beams/utils.py:1194-1241` (`interpolate_beam`)
- `src/meerkat_beams/utils.py:1162-1192` (`freq_range`, `check_freqs`)
- `src/meerkat_beams/utils.py:944-991` (`attach_image`)
- `src/meerkat_beams/utils.py` (`GROUP_TELESCOPES`, `_build_group_bds`,
  `_align_group_freqs`, `_check_group_grids`, the `group` branch of `__init__`)
- `src/meerkat_beams/cache.py` (`ensure_group_bds`, `GROUP_PRODUCTS`)
- `tests/test_beam_wizard_group.py`, `tests/test_group_bds.py`
- `src/meerkat_beams/utils.py:993-1008` (`set_pointing_centre`)
- `src/meerkat_beams/utils.py:1802-2018` (`get_time_freq_beam`)
- `src/meerkat_beams/utils.py:1875-1882` (`_CANONICAL_DIM_NAMES` guard)
- `src/meerkat_beams/utils.py:2034-2123` (`enrich_bds_xradio`)
- commit `0f2a4f4` ("test: add tests for complex Mueller term" — landed the
  dtype-aware prefilter cache)
- `tests/test_beam_wizard.py`: `test_prefilter_is_cached`,
  `test_prefilter_cached_dtype_is_float32`,
  `test_prefilter_complex_var_is_complex64_without_warning`,
  `test_subpixel_matches_direct_scipy`,
  `test_out_of_range_xy_returns_zero`, `test_out_of_range_freq_raises`,
  `test_interpolate_beam_complex_var_preserves_imaginary`,
  `test_source_coordinates_at_pointing_centre`,
  `test_fits_branch_sets_times_none_raises_runtimeerror`,
  `test_construct_without_image`, `test_no_image_attrs_raise`,
  `test_set_pointing_centre_unblocks_source_coordinates`,
  `test_attach_image_unblocks_grid_methods`,
  `test_beam_wizard_requires_one_of_bds_or_band`,
  `test_beam_wizard_rejects_both_bds_and_band`,
  `test_time_freq_beam_rejects_non_canonical_dim_names`,
  `test_time_freq_beam_open_default_keeps_real_zeros`,
  `test_time_freq_beam_writes_zarr`,
  `test_time_freq_beam_complex_var_preserves_imaginary`,
  `test_time_freq_beam_complex_var_upsamples`,
  `test_enrich_bds_xradio_writes_xradio_schema`
