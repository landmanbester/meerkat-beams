---
type: reference
title: Data model — MdV npz, BDS zarr, xradio zarr
description: The three beam formats and their conversions — MdV .npz structure, the BDS zarr schema (jones/njones/stokes/nstokes/mueller/nmueller, fits_header, scalar attrs), and the xradio primary-beam schema.
tags: [mdv, bds, xradio, zarr, schema, data-model]
timestamp: 2026-07-27T10:19:03Z
last_verified_commit: 56a57b7
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
`Jones ⊗ conj(Jones)` (`mueller_func`, `mdv_beams_to_bds.py:81-84`),
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
- `src/meerkat_beams/core/bds_to_xradio.py:11-131` (`bds_to_xradio`,
  `_resolve_elements`)
- commit `2d6d0dc` ("feat(bds): add mueller term to bds")
- commit `a4c8df7` ("fix: skip antenna selection for zarr input in
  mdv_beams_to_bds")
- `tests/test_mdv_beams_to_bds.py` (`ALL_BANDS`, `MBEAMS_REFERENCE_BDS_*`,
  jones/stokes-only parametrisation at `:184`, `:191`)
- `tests/test_bds_to_xradio.py`, `tests/test_beam_wizard.py`,
  `tests/test_beam_orientation_mueller.py` (mueller/nmueller coverage)
- `tests/conftest.py` (L-band cache warm-up, `MBEAMS_OFFLINE`)
