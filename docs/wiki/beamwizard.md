---
type: reference
title: BeamWizard interpolation and rendering internals
description: interpolate_beam prefilter/off-cube/spline-order/freq-guard semantics, get_source_coordinates transforms, optional-image paths, get_time_freq_beam canonical dim_names, and enrich_bds_xradio.
tags: [beamwizard, interpolation, scipy, zarr, xradio, utils]
timestamp: 2026-07-27T09:38:10Z
last_verified_commit: 3bb9b3d
---

# BeamWizard interpolation and rendering internals

`BeamWizard` (`src/meerkat_beams/utils.py`) attaches to a BDS (zarr) and,
optionally, an image (FITS or xradio zarr), and provides spline-based beam
interpolation, time-variable beam gain, rotation-averaged beam maps, and
full time/frequency zarr rendering. This page documents the interpolation
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

(`utils.py:722-726`). Real permutation of the data layout is not
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

## Sources

- `src/meerkat_beams/utils.py:247-263` (`_get_prefilter`)
- `src/meerkat_beams/utils.py:265-299` (`get_source_coordinates`)
- `src/meerkat_beams/utils.py:301-335` (`interpolate_beam`)
- `src/meerkat_beams/utils.py:189-230` (`attach_image`)
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
