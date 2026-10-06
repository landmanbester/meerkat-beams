---
type: reference
title: BeamWizard interpolation and rendering internals
description: beam_model (mdv/katbeam), group (MM/MPM/MPMP) and average (pa/azimuth) selectors, interpolate_beam prefilter/off-cube/spline-order/freq-guard semantics, get_source_coordinates transforms, optional-image paths, get_time_freq_beam canonical dim_names, enrich_bds_xradio, and partition_mueller with the centre= override.
tags: [beamwizard, interpolation, scipy, zarr, xradio, utils, katbeam, baseline-groups, meerkat+, partition-mueller, mueller]
timestamp: 2026-10-06T10:15:21Z
last_verified_commit: 6f59aa9
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

`_get_prefilter` (`src/meerkat_beams/utils.py:247`) runs
`scipy.ndimage.spline_filter` once per `(var, i, j, order)` key and caches
the result in `self._prefilters`. The cache's output dtype is **chosen by
variable kind, not unconditionally `float32`**:

```python
out_dtype = np.complex64 if np.iscomplexobj(da) else np.float32
```

(`utils.py:257-262`). Complex variables (`jones`, `njones`, `mueller`,
`nmueller`) cache as `complex64`; real variables (`stokes`, `nstokes`)
cache as `float32`. Passing a real output dtype for complex input would
make scipy implicitly promote it (a version-dependent `UserWarning`), so
the dtype is picked explicitly instead of relying on that promotion. Note
that an older revision of this project's notes described the cache as
unconditionally `float32` — that was true before the Mueller-term work
landed (commit `0f2a4f4`, "test: add tests for complex Mueller term") and
is no longer accurate; treat the dtype-aware rule above as current.

`interpolate_beam` (`utils.py:301`) calls `map_coordinates` against this
cached, already-filtered array with **`prefilter=False`**
(`utils.py:332`) — an inline comment at `utils.py:326-327` spells out why:
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
(`utils.py:333`) — coordinates outside the X/Y cube return `0`, not a
nearest-edge value or a NaN. Pinned by `test_out_of_range_xy_returns_zero`.

### Spline order

`_get_prefilter(order=3)` includes `order` in its cache key (`key = var,
i, j, order`, `utils.py:250`), since `spline_filter` coefficients depend
on the requested order and callers requesting different orders must not
collide in the cache. `interpolate_beam` forwards its own `order`
parameter to both `_get_prefilter` and the `map_coordinates` call.

### Frequency guard

`interpolate_beam` checks the requested frequency array against the BDS's
`FREQ` coordinate range and raises `ValueError` — reporting both the
requested and available ranges in MHz — before calling `freq_to_index`.
Pinned by `test_out_of_range_freq_raises`.

## `get_source_coordinates` — sky-to-beam-pixel transform

```python
def get_source_coordinates(self, srcpos, times=None, loc=None, signs=(1, 1), swap=False):
```

(`utils.py:265-272`). For each requested time, transforms `srcpos` and the
field centre to `AltAz`, then computes the source's separation and
position angle relative to the centre. Those are converted to beam-pixel
offsets via the BDS's `dx`/`dy`/`x0`/`y0` attrs:

```python
x = signs[0] * seps.deg * np.sin(angles.rad)
y = signs[1] * seps.deg * np.cos(angles.rad)
if swap:
    x, y = y, x
xp = x / self.bds.attrs["dx"] + self.bds.attrs["x0"]
yp = y / self.bds.attrs["dy"] + self.bds.attrs["y0"]
```

(`utils.py:293-298`). The `signs`/`swap` keyword arguments are not dead
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
`test_source_coordinates_at_field_centre` (the latter also confirms a
source exactly at the field centre resolves to `(x0, y0)` with `sep=0`).

## Optional-image construction

`image_name` is optional at construction time. Without it, `BeamWizard` is
BDS-only: `interpolate_beam` still works (it only needs the BDS), but the
`centre`, `wcs`, `l_grid`, and `m_grid` properties raise `RuntimeError`,
and `times` returns `None` rather than raising. Two methods populate
image-derived state after the fact:

- `attach_image(image_name)` (`utils.py:189`) — attaches a FITS or xradio
  zarr image after construction (or swaps the image on an existing
  wizard), populating the field centre, WCS, default l/m grid, and time
  axis (the latter only for xradio-zarr images; FITS images still leave
  `times` as `None`).
- `set_field_centre(centre, times=None)` (`utils.py:232`) — supplies just
  a pointing centre (and optionally a time axis) without attaching an
  image at all. This is the image-free path used by
  `scripts/test_beam_orientation.py`; the WCS and default l/m grid remain
  unavailable, so methods that fall back to the image grid still need
  explicit `l`/`m`.

Construction also requires exactly one of `bds`/`band` — passing neither
or both raises.

Pinned by `test_construct_without_image`, `test_no_image_attrs_raise`,
`test_set_field_centre_unblocks_source_coordinates`,
`test_attach_image_unblocks_grid_methods`,
`test_beam_wizard_requires_one_of_bds_or_band`, and
`test_beam_wizard_rejects_both_bds_and_band`.

## `get_time_freq_beam` — canonical `dim_names` only

`get_time_freq_beam` (`utils.py:655`) writes the full `(ij, time, freq,
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

(`utils.py:722-728`). Real permutation of the data layout is not
implemented — passing a differently-ordered tuple would relabel the dims
without reordering the underlying array (silent corruption), so any
non-canonical tuple raises `ValueError` instead. Pinned by
`test_time_freq_beam_rejects_non_canonical_dim_names`.

Both the beam-variable dataset and the coordinate datasets are created
with `fill_value=None` (`utils.py:869` and `:877`) rather than zarr's
default `fill_value=0`. With the default, `xarray.open_zarr`'s
`mask_and_scale=True` treats stored `0.0` as "unwritten" and masks it to
`NaN` on read — which corrupts genuine zero coordinates (e.g. `l=0.0`) or
genuine zero beam pixels. `fill_value=None` means a default `open_zarr`
call keeps real zeros intact, no `mask_and_scale=False` workaround
required. Pinned by `test_time_freq_beam_open_default_keeps_real_zeros`
and `test_time_freq_beam_writes_zarr` (dims/shape/coords round-trip).

The output dtype mirrors the source variable the same way the prefilter
cache does: `complex64` for complex beams, `float32` for real ones — see
"Complex vars preserve the imaginary part end-to-end" above.

## `enrich_bds_xradio`

`enrich_bds_xradio(zarr_path, bw, output_var, polarizations)`
(`utils.py:924`) post-processes a zarr store already written by
`get_time_freq_beam` into xradio schema:

- converts the `l`/`m` coordinate arrays from degrees to radians in place;
- replaces the polarization coordinate with the given single-letter Stokes
  labels (`get_time_freq_beam` itself writes two-letter labels like `"II"`,
  `"QQ"`);
- adds a dataset-level `direction` attribute block (`icrs` frame, `SIN`
  projection, reference position = `bw.centre`, `lonpole`/`pc` filled with
  xradio-schema defaults);
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

### `centre=` on `get_rotation_averaged_beam`

`get_rotation_averaged_beam` grew `centre: Optional[SkyCoord] = None` to serve this.
`None` is exactly the old behaviour (`self.centre`, raising if no image is
attached); a supplied `SkyCoord` is used for the parallactic-angle computation and
**not stored**, so one wizard serves many pointings. Ignored for
`average="azimuth"`, which has no centre dependence. The alternative —
`set_field_centre` per chunk — mutates an object shared across chunks, which is why
it is not what `partition_mueller` does.

## Sources

- `src/meerkat_beams/utils.py:247-263` (`_get_prefilter`)
- `src/meerkat_beams/utils.py:265-299` (`get_source_coordinates`)
- `src/meerkat_beams/utils.py:301-335` (`interpolate_beam`)
- `src/meerkat_beams/utils.py:189-230` (`attach_image`)
- `src/meerkat_beams/utils.py` (`GROUP_TELESCOPES`, `_build_group_bds`,
  `_align_group_freqs`, `_check_group_grids`, the `group` branch of `__init__`)
- `src/meerkat_beams/cache.py` (`ensure_group_bds`, `GROUP_PRODUCTS`)
- `tests/test_beam_wizard_group.py`, `tests/test_group_bds.py`
- `src/meerkat_beams/utils.py:232-245` (`set_field_centre`)
- `src/meerkat_beams/utils.py:655-916` (`get_time_freq_beam`)
- `src/meerkat_beams/utils.py:722-728` (`_CANONICAL_DIM_NAMES` guard)
- `src/meerkat_beams/utils.py:924-995` (`enrich_bds_xradio`)
- commit `0f2a4f4` ("test: add tests for complex Mueller term" — landed the
  dtype-aware prefilter cache)
- `tests/test_beam_wizard.py`: `test_prefilter_is_cached`,
  `test_prefilter_cached_dtype_is_float32`,
  `test_prefilter_complex_var_is_complex64_without_warning`,
  `test_subpixel_matches_direct_scipy`,
  `test_out_of_range_xy_returns_zero`, `test_out_of_range_freq_raises`,
  `test_interpolate_beam_complex_var_preserves_imaginary`,
  `test_source_coordinates_at_field_centre`,
  `test_fits_branch_sets_times_none_raises_runtimeerror`,
  `test_construct_without_image`, `test_no_image_attrs_raise`,
  `test_set_field_centre_unblocks_source_coordinates`,
  `test_attach_image_unblocks_grid_methods`,
  `test_beam_wizard_requires_one_of_bds_or_band`,
  `test_beam_wizard_rejects_both_bds_and_band`,
  `test_time_freq_beam_rejects_non_canonical_dim_names`,
  `test_time_freq_beam_open_default_keeps_real_zeros`,
  `test_time_freq_beam_writes_zarr`,
  `test_time_freq_beam_complex_var_preserves_imaginary`,
  `test_time_freq_beam_complex_var_upsamples`,
  `test_enrich_bds_xradio_writes_xradio_schema`
