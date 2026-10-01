---
type: reference
title: Data model — MdV npz, BDS zarr, xradio zarr
description: The beam formats and their conversions — MdV .npz structure, the BDS zarr schema (jones/njones/stokes/nstokes/mueller/nmueller, fits_header, scalar attrs), and the xradio primary-beam schema.
tags: [mdv, bds, xradio, zarr, schema, data-model, katbeam]
timestamp: 2026-10-01T00:00:00Z
last_verified_commit: ad10c54
---

# Data model — MdV npz, BDS zarr, xradio zarr

Three formats, three conversions: MdV `.npz` (raw archive data) →
`mdv-beams-to-bds` → BDS zarr (the package's working format) →
`bds-to-xradio` → xradio zarr (schema-compatible primary-beam image).
`mdv-to-xradio` shortcuts directly from `.npz` to xradio, skipping the BDS
and the time axis. This page pins the field/variable names, dtypes, dims,
and one input-handling asymmetry that is easy to get backwards.

## MdV `.npz`

Raw voltage beams from the SARAO archive
(<https://doi.org/10.48479/wdb0-h061>). Fields:

- `beam` — complex64, `(pol, ant, freq, y, x)`.
- `freq_MHz` — frequency axis in MHz (converted to Hz on load: `* 1e6`).
- `margin_deg` — spatial axis in degrees (same grid for x and y).
- `pols` — `HH`/`HV`/`VH`/`VV`, ordered so `beam.reshape([2, 2, ...])` gives
  the Jones matrix `[[HH, HV], [VH, VV]]`.
- `antnames` — antenna index `-1` is `array_average`.

### Two input branches in `mdv_beams_to_bds`, different antenna handling

`mdv_beams_to_bds` (`src/meerkat_beams/core/mdv_beams_to_bds.py:22-35`)
accepts either an `.npz` file or a per-antenna mean-beam zarr, and the two
branches disagree on whether antenna selection is needed:

```python
if mdv_beams.endswith(".npz"):
    mdv = np.load(mdv_beams)
    bm = mdv["beam"]
    ...
    bm = bm[:, -1]  # select average beam (last antenna index)
elif (Path(mdv_beams) / ".zgroup").exists():
    xds = xarray.open_zarr(mdv_beams, chunks=None)
    bm = xds.BEAM.values  # already mean beam: [4, NFREQ, NY, NX]
    ...
```

The `.npz` branch's `beam` array still carries a per-antenna axis, so
`bm[:, -1]` selects the array-average antenna out of the 5D
`(pol, ant, freq, y, x)` array. The zarr branch's `BEAM` variable is
**already** the 4D mean beam `[4, NFREQ, NY, NX]` — there is no antenna
axis left to index. Applying the same `bm[:, -1]` slice to the zarr branch
was a real bug: with no antenna axis, `[:, -1]` silently selected the last
*frequency* channel instead, corrupting the conversion without raising.
Fixed in commit `a4c8df7` ("fix: skip antenna selection for zarr input in
mdv_beams_to_bds"), which is why the zarr branch above has no `[:, -1]`
and the `.npz` branch still does.

## BDS zarr

Produced by `mdv-beams-to-bds`. Holds **six** data variables
(`src/meerkat_beams/core/mdv_beams_to_bds.py:110-121`):

| variable | dtype | dims | normalisation |
|---|---|---|---|
| `jones` | `complex64` | `receptor_i, receptor_j, FREQ, Y, X` | raw |
| `njones` | `complex64` | `receptor_i, receptor_j, FREQ, Y, X` | normalised |
| `stokes` | `float32` | `stokes_i, stokes_j, FREQ, Y, X` | raw |
| `nstokes` | `float32` | `stokes_i, stokes_j, FREQ, Y, X` | normalised |
| `mueller` | `complex64` | `stokes_i, stokes_j, FREQ, Y, X` | raw |
| `nmueller` | `complex64` | `stokes_i, stokes_j, FREQ, Y, X` | normalised |

`receptor_{i,j}` coordinates are `[0, 1]`; `stokes_{i,j}` coordinates are
`["I", "Q", "U", "V"]`. Beam-cube dim order (dropping the matrix indices)
is `(FREQ, Y, X)`, i.e. the full variable is `(i, j, FREQ, Y, X)`.

`mueller`/`nmueller` are the coherency Mueller matrix
`Jones ⊗ conj(Jones)` (`jones_to_mueller`, `utils.py` — moved there from
`mdv_beams_to_bds` so the katbeam synthesizer shares one conversion path),
kept in complex coherency form (not converted through `Sinv @ M @ S` to
real Stokes like `stokes`/`nstokes` are) — hence `complex64` rather than
`float32` despite sharing the `stokes_i`/`stokes_j` dims. Added in commit
`2d6d0dc` ("feat(bds): add mueller term to bds").

Additional dataset-level attrs (`mdv_beams_to_bds.py:120-121`):

- `.attrs["fits_header"]` — a synthesized FITS header dict (`SIMPLE`,
  `NAXIS{1,2,3}`, `CRPIX{1,2,3}`, `CRVAL{1,2,3}`, `CDELT{1,2,3}`,
  `CTYPE{1,2,3}`, `CUNIT{1,2,3}`).
- `x0`, `y0` — centre pixel index (both equal `len(degs) // 2`).
- `dx`, `dy` — degrees/pixel (both equal `degs[1] - degs[0]`).
- `freqs` — the frequency axis in Hz.

**Normalised vs raw:** the `n…` variants (`njones`, `nstokes`, `nmueller`)
are pre-multiplied by the inverse of the central-pixel Jones matrix so the
on-axis beam is the identity — use these unless raw voltage beams are
specifically needed.

When `compress=True`, all six variables get Delta(float32) + Blosc
zstd/clevel=5 encoding (`mdv_beams_to_bds.py:123-130`) — note the filter
is `Delta(dtype="float32")` even for the `complex64` variables (`jones`,
`njones`, `mueller`, `nmueller`); this matches the module-level
`ZARR_FILTERS` convention documented in `utils.py`.

## Synthesized katbeam BDS

`BeamWizard(band=..., beam_model="katbeam")` does not read a BDS at all. It
calls `katbeam_bds.synthesize_katbeam_bds(band, ...)`, which samples katbeam's
analytic `JimBeam` onto a grid and assembles a Dataset carrying the BDS schema
in memory. Nothing is downloaded and nothing is written to disk.

It holds **three** data variables, not six:

| variable | dtype | dims | present |
|---|---|---|---|
| `njones` | `complex64` | `receptor_i, receptor_j, FREQ, Y, X` | yes — `diag(HH, VV)` |
| `nstokes` | `float32` | `stokes_i, stokes_j, FREQ, Y, X` | yes |
| `nmueller` | `complex64` | `stokes_i, stokes_j, FREQ, Y, X` | yes |
| `jones`, `stokes`, `mueller` | — | — | **no** |

The unnormalised variables are deliberately absent: katbeam beams are on-axis
normalised by construction, so there is no raw counterpart to expose, and
aliasing them to the normalised ones would misrepresent them. Requesting one
raises from `BeamWizard._get_prefilter` with an actionable message rather than
a bare `KeyError` — see [`beamwizard.md`](beamwizard.md).

Structural consequences of katbeam supplying only two real co-polarisation
patterns:

- `njones` off-diagonals are **exactly** zero, and the diagonal is purely real
  (katbeam carries no phase information).
- `nstokes` has no I↔U or I↔V leakage. `U` and `V` are *not* zero: they carry
  gain `HH·VV` on the diagonal. The I↔Q leakage is `(HH²−VV²)/2`, off-diagonal.
- Everything past `njones` is derived through the shared `jones_to_mueller` /
  `mueller_to_stokes` helpers in `utils.py` — the same code path
  `mdv_beams_to_bds` uses. This is what makes katbeam's own `JimBeam.I()` a
  genuine cross-check of our Jones→Stokes conversion
  (`tests/test_katbeam_bds.py::test_derived_stokes_i_matches_katbeam_own_i`).

Attrs are a superset of the MdV BDS's. The shared ones (`x0`, `y0`, `dx`, `dy`,
`freqs`, `fits_header`) carry the same meaning, plus:

- `npix` — pixels per spatial axis. Must be **even**: the grid is
  `X = -fov_deg + arange(npix)*dx` with `x0 = npix//2`, so the centre pixel
  lands on exactly 0° only for even `npix`. Odd values are rejected.
- `fov_deg` — grid **half**-width in degrees; the grid spans `-fov_deg` to
  `+fov_deg - dx`.
- `beam_model` — `"katbeam"`.
- `katbeam_model` — the `JimBeam` model name actually used.
- `band` — the MdV band code requested.

Defaults mirror the real MdV BDSs so the two models are pixel-identical and
directly comparable (`katbeam_bds.BAND_GEOMETRY`): L is `npix=128, fov_deg=4.0`
(`dx=0.0625`), U is `npix=128, fov_deg=6.0` (`dx=0.09375`). S-band MdV geometry
has not been measured, so `S0`–`S4` have no default and require explicit
`npix`/`fov_deg` rather than a guess. All five S sub-bands map to katbeam's
single `MKAT-AA-S-JIM-2020`.

Frequencies default to the katbeam model's own table (`freqMHzlist`) — 19
entries for L — rather than an invented axis. Frequencies outside that table are
**refused**: katbeam interpolates its squint/FWHM table with `np.interp`, which
clamps silently, so asking the L model for 500 MHz would otherwise hand back the
856 MHz beam with no warning.

Variables are dask-backed, chunked in `FREQ` only (`FREQ_CHUNK = 256`). The
analytic evaluation is vectorised over the whole spatial plane, so spatial
chunking would only multiply the number of `JimBeam` calls. Eager construction
at MdV L-band resolution would cost several GB, which would make the "lighter
alternative" heavier than what it replaces.

**The cosine-taper singularity.** katbeam's pattern is
`cos(π·rr)/(1 − 4·rr²)` with `rr = r·1.1889647809329453`, which is `0/0` at
`rr = 0.5` — normalised radius `r = 0.4205339031217265`. A grid point landing
there returns `NaN`. Synthesis replaces non-finite samples with the L'Hôpital
limit `π/4 = 0.7853981633974483` and logs the count. This is not optional: a
`NaN` reaching `spline_filter` smears across the entire slab.

## xradio zarr

Produced by `bds-to-xradio` (from a BDS) or `mdv-to-xradio` (directly from
an `.npz`, no time axis, no parallactic-angle rotation). Schema:
`(time, frequency, polarization, l, m)`, with `l`/`m` in **radians** (BDS
values are in degrees; `enrich_bds_xradio` converts) and a `direction`
attribute block (`icrs` frame, `SIN` projection, reference = field
centre). Rendering internals (interpolation, dim-name canonicalisation,
`enrich_bds_xradio`) live in `beamwizard.md`.

`bds_to_xradio`'s `beam_type` parameter selects the source BDS variable,
resolved through `_resolve_elements`
(`src/meerkat_beams/core/bds_to_xradio.py:11-58`). It accepts **all six**
BDS variable names:

```python
if beam_type in ("nstokes", "stokes", "mueller", "nmueller"):
    ...
    label_by_element = beam_type in ("mueller", "nmueller")
elif beam_type in ("njones", "jones"):
    ...
else:
    raise ValueError(
        f"Unknown beam_type '{beam_type}', expected 'nstokes', 'stokes', "
        f"'mueller', 'nmueller', 'njones', or 'jones'"
    )
```

(`bds_to_xradio.py:31-52`). `stokes`/`nstokes` and `mueller`/`nmueller`
share the same IQUV element-pair validation (`elements` like `"II"`,
`"QQ"`), but differ in output polarization labelling: `stokes`/`nstokes`
label by the output Stokes only (`e[1]`, for backward compatibility),
while `mueller`/`nmueller` label by the full 2-character element (e.g.
`"IQ"`) so a 16-term Mueller cube gets unique polarization labels
(`bds_to_xradio.py:31-39`). Note the `bds_to_xradio` docstring
(`bds_to_xradio.py:94`) still lists only `'nstokes', 'stokes', 'njones',
'jones'` — that's stale; the code path (`_resolve_elements`) is the source
of truth and accepts `mueller`/`nmueller` too.

## BDS regression testing — coverage nuance

`tests/test_mdv_beams_to_bds.py` (marker `integration`,
`ALL_BANDS = ["U", "L", "S0", "S1", "S2", "S3", "S4"]`) converts input
data fresh and compares against a reference BDS produced by the original
suricat-beams. It needs `MBEAMS_REFERENCE_BDS_<BAND>` env vars and the
matching input zarr under `tests/data/MeerKAT_<BAND>.zarr` (auto-
downloaded once for L by `tests/conftest.py`; only `MeerKAT_L.zarr` and
`MeerKAT_UHF.zarr` are present locally, consistent with S1/S2/S3 having no
published gdrive ID in `cache.py`).

This regression test's variable-parametrised checks
(`tests/test_mdv_beams_to_bds.py:184`, `:191`) cover only
`["jones", "njones", "stokes", "nstokes"]` — `mueller`/`nmueller` (added
later, commit `2d6d0dc`) are **not** in this cross-reference suite.
Mueller coverage instead comes from unit tests: `test_bds_to_xradio.py`
(`_resolve_elements`/`beam_type` behaviour for `mueller`/`nmueller`),
`test_beam_wizard.py` (prefilter dtype, interpolation, rotation-averaging,
and rendering of the complex `nmueller` variable), and
`test_beam_orientation_mueller.py` (unit tests for
`scripts/beam_orientation/mueller.py`, the standalone Mueller-solve
script — not the BDS `mueller` variable itself).

## Sources

- `src/meerkat_beams/core/mdv_beams_to_bds.py:12-131` (`mdv_beams_to_bds`;
  antenna-branch asymmetry at `:22-35`, six-variable write at `:110-121`)
- `src/meerkat_beams/core/bds_to_xradio.py:11-130` (`bds_to_xradio`,
  `_resolve_elements`)
- commit `2d6d0dc` ("feat(bds): add mueller term to bds")
- commit `a4c8df7` ("fix: skip antenna selection for zarr input in
  mdv_beams_to_bds")
- `tests/test_mdv_beams_to_bds.py` (`ALL_BANDS`, `MBEAMS_REFERENCE_BDS_*`,
  jones/stokes-only parametrisation at `:184`, `:191`)
- `tests/test_bds_to_xradio.py`, `tests/test_beam_wizard.py`,
  `tests/test_beam_orientation_mueller.py` (mueller/nmueller coverage)
- `tests/conftest.py` (L-band cache warm-up, `MBEAMS_OFFLINE`)
