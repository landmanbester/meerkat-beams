---
type: reference
title: Data model — MdV npz, BDS zarr, xradio zarr
description: The beam formats and their conversions — MdV .npz structure, the BDS zarr schema (jones/njones/stokes/nstokes/mueller/nmueller, fits_header, scalar attrs), the MM/MPM/MPMP baseline-group datasets, and the xradio primary-beam schema.
tags: [mdv, bds, xradio, zarr, schema, data-model, katbeam, meerkat+, mke, baseline-groups, mueller]
timestamp: 2026-10-02T09:00:59Z
last_verified_commit: eac4bd0
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

**Normalisation matches the MdV BDS exactly.** `njones` is pre-multiplied by the
inverse of the centre-pixel Jones, so on axis it is the identity and
`nstokes[I, I]` is 1 — the same meaning the `n` prefix has for MdV. This is not
free: katbeam's raw `HH`/`VV` are only *approximately* unity at (0, 0), because
each beam's peak is offset by its squint (0.9950 at 1712 MHz, where the L table's
Hx squint reaches 0.052°). Without the normalisation a caller correcting data
with `njones` would carry an on-axis error of up to 0.5% that the variable's name
denies. Since the Jones matrix is diagonal, `inv(J(centre)) @ J` reduces to
dividing each receptor by its own centre sample.

The unnormalised variables are deliberately absent: there is no second,
independent normalisation for katbeam to expose (the raw voltage patterns differ
from the normalised ones only by that per-frequency scalar), and aliasing them to
the normalised ones would misrepresent them. Requesting one raises from
`BeamWizard._get_prefilter` with an actionable message rather than a bare
`KeyError` — see [`beamwizard.md`](beamwizard.md).

One consequence worth knowing when comparing against katbeam directly: the
normalised `nstokes[I, I]` does **not** relate to `JimBeam.I()` by a single
scalar, since `0.25·((HH/HH₀)² + (VV/VV₀)²)·(HH₀² + VV₀²)` equals
`0.5·(HH² + VV²)` only when `HH₀ == VV₀`, which the squint makes false. The
cross-check test therefore compares normalised against normalised, and a separate
test feeds *raw* katbeam Jones through our helpers to reproduce `I()` exactly.

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
entries for L — rather than an invented axis. An explicit `freq` array must be
non-empty, finite and strictly increasing, since `BeamWizard` builds its
index↔frequency `interp1d` mappings from this axis. Frequencies outside the
model's table are **refused**: katbeam interpolates its squint/FWHM table with
`np.interp`, which clamps silently, so asking the L model for 500 MHz would
otherwise hand back the 856 MHz beam with no warning.

Because the default axis is the model table and that table is not uniformly
spaced (… 1600, 1650, 1670, 1712 MHz), `CDELT3` is **omitted** from the
synthesized `fits_header` unless the axis really is uniform. A single increment
would make a WCS reader assign the wrong frequency to every later plane; the
exact axis is always available in `attrs["freqs"]` and the `FREQ` coordinate.

**Which katbeam you have changes the default L axis.** `uv.lock` resolves both
the `[full]` extra and the dev/test groups to the git pin, so a `uv sync` gets
git main (856–1712 MHz for L, and the S model). A non-lock install —
`pip install meerkat-beams[full]`, or the `Dockerfile`'s `.[full]` — gets PyPI
0.1 instead, whose L table spans only 900–1650 MHz and which has no S model at
all. So on that path the default frequency axis differs and the
"pixel-identical with MdV" property does not extend to the frequency axis.

Variables are dask-backed and chunked in `FREQ` only — the analytic evaluation
is vectorised over the whole spatial plane, so spatial chunking would only
multiply the number of `JimBeam` calls.

**Memory, measured rather than assumed.** The cost is not the stored arrays (at
default L geometry all three total only ~70 MB) but the *transient* inside
`_eval_block`: it holds `jones` (2×2), `mueller` (4×4) and `stokes` (4×4) for a
whole chunk before the `astype` downcasts, so the peak scales as
`nfreq · npix²` with a 16-entry 4×4 matrix per pixel. Two things bound it:

- **An element budget** (`FREQ_CHUNK_ELEMENTS = 300_000`, via
  `_freq_chunk_size(npix)`) rather than a fixed frequency count, so the chunk
  shrinks as the grid grows: L-band `npix=128` gets 18 frequencies per block,
  `npix=32` gets 292.
- **complex64 throughout**, including `mueller_to_stokes`, which casts its
  complex128 basis matrices down to the input dtype rather than promoting. The
  outputs are `complex64`/`float32` regardless, so complex128 intermediates cost
  2× memory for precision that is then discarded (measured difference in the
  Stokes result: ~2e-8).

Together these took the peak for pulling one `nstokes[I,I]` slab at `npix=128`
from 3.3 GiB to 144 MiB at default L, 846 MiB at `num_freq=128` and 1.7 GiB at
`num_freq=256`. Pinned by
`tests/test_katbeam_bds.py::test_eval_block_transient_stays_bounded` (per-block,
scheduler-independent) and `test_stokes_conversion_preserves_complex64`.

**The remaining scaling is dask's, not ours.** The threaded scheduler holds every
live block, so process peak is roughly `n_workers × per-block transient` —
reducing the chunk size alone does *not* help, since more smaller chunks simply
means more in flight. A caller rendering a few hundred frequencies at full
spatial resolution should either accept ~1-2 GB or restrict the scheduler
(`dask.config.set(num_workers=...)`), which drops it back to the per-block
figure.

**The cosine-taper singularity.** katbeam's pattern is
`cos(π·rr)/(1 − 4·rr²)` with `rr = r·1.1889647809329453`, whose denominator
vanishes at `rr = 0.5` — normalised radius `r = 0.4205339031217265`. A grid point
landing there returns a **non-finite value — `inf` in practice, not `NaN`**,
since `cos(π·rr)` is ~6.1e-17 rather than exactly 0 there, so it is a
divide-by-zero rather than a true `0/0` in floating point. `_sanitize` tests with
`~np.isfinite`, so it catches either. Synthesis replaces such samples with the
L'Hôpital limit `π/4 = 0.7853981633974483` and logs the count. This is not
optional: a non-finite value reaching `spline_filter` smears across the entire
slab.

## Baseline-group beam datasets

A MeerKAT+ array mixes two dish types, so a baseline's beam depends on
*which pair* of antennas forms it. `BeamWizard(band="L", group=...)`
assembles that per-pair beam on the fly from two single-telescope BDSs; no
group product is ever written to disk.

### The three groups

| group | p (first antenna) | q (second antenna) | `nstokes` dtype | `jones`/`njones` |
|---|---|---|---|---|
| `MM` | MeerKAT | MeerKAT | `float32` | present |
| `MPM` | MeerKAT | MeerKAT Extension | `complex64` | **absent** |
| `MPMP` | MeerKAT Extension | MeerKAT Extension | `float32` | present |

`mueller`/`nmueller` are `complex64` for all three, as in a
single-telescope BDS.

For a baseline between antennas p and q the visibility is
`V_pq = J_p X J_q^H`, so the coherency-basis block is
`kron(J_p, conj(J_q))` (`utils.jones_to_mueller_cross`). When p and q are
the same telescope this is real up to numerical noise and the Stokes
variables take `.real`, exactly as `mdv_beams_to_bds` does. For `MPM` it
is genuinely complex and is kept that way: discarding the imaginary part
would discard a real part of the mixed-baseline response.

**The `p = MeerKAT` convention is fixed**, matching MS antenna indexing
where MKE antennas follow the MeerKAT ones. A caller needing the opposite
ordering conjugates: baseline reversal Hermitian-conjugates the coherency
matrix, which in the Stokes basis is plain complex conjugation with *no*
transpose, because `P @ S == conj(S)` for the standard linear-feed basis
(`tests/test_group_bds.py::test_swapping_p_and_q_conjugates_the_stokes_block`).
At the coherency level the relation is instead a permutation of the
coherency index `2*receptor_1 + receptor_2` by `[0, 2, 1, 3]`
(`tests/test_jones_mueller.py`) — `kron(A, B)` and `kron(B, A)` differ by
that permutation, not by a transpose.

### Normalisation

Both sides are normalised by their *own* on-axis Jones inverse first (the
stored `njones`), then crossed. The product of two on-axis-normalised
Jones matrices is the identity on axis, so `MPM`'s `nstokes` behaves
exactly like `MM`'s and `MPMP`'s. **A caller predicting apparent flux uses
`nstokes`/`nmueller`**; `stokes`/`mueller` are the raw voltage products.

### Extra attrs

A group dataset carries the usual `fits_header`, `x0`, `y0`, `dx`, `dy`,
`freqs`, plus `group`, `telescope_p` and `telescope_q`. For `MPM` the
single-telescope provenance attrs (`telescope`, `antenna`, `source_file`,
`source_doc`) are dropped, since side p's values would misdescribe a cross
block.

### Provenance attrs on a single-telescope BDS

`mdv_beams_to_bds` copies `PROVENANCE_ATTRS` — `telescope`, `antenna`,
`band`, `source_file`, `source_doc` — from a zarr input's attrs onto the
BDS. Keys absent from the input are not written, so a legacy BDS carries
none of them. `_build_group_bds` reads `telescope` to verify it paired the
right two stores; a BDS missing it **warns and proceeds** rather than
failing, so hand-built and pre-2026 stores stay usable.

### Two MdV generations

| generation | grid | channels | extent | products |
|---|---|---|---|---|
| legacy | 128 x 128 | 1024 | +-4 deg (L) | `MeerKAT_{U,L,S0,S4}` |
| MdV-2026 | 64 x 64 | 63 | +-2 deg (L) | `MeerKAT_L_mdv2026`, `MKE_L` |

`group=None` (the default) resolves the legacy generation, unchanged.
Asking for **any** group switches to MdV-2026, so all three groups are
mutually consistent; the two generations are never mixed within one
wizard.

### L band only

Only L band has a matched pair. In L band `beam_eavg_L` (MKE) and
`beam_mavg_L` (MeerKAT) are both 64 x 64, `CDELT1 = 0.0625` deg, 63
channels from 869.375 MHz — identical grids, no resampling. MKE's
`beam_eavg_S3` (2420-3268 MHz, 0.03125 deg, +-1 deg) has no counterpart:
`beam_mavg_S04` aligns exactly in frequency (MKE S3 is its channels
48-110) but sits on a half-size +-0.5 deg grid, while `beam_mavg_S0` +
`beam_mavg_S4` match spatially but leave a 27 MHz hole at 2611-2639 MHz,
straddling the middle of MKE S3. S-band group requests therefore raise.
Single-telescope MKE S3 is unaffected.

`_build_group_bds` requires exactly matching `X`/`Y` axes and `x0`, `y0`,
`dx`, `dy` and performs **no spatial resampling**. Channel centres are
matched within `FREQ_MATCH_ATOL_HZ = 1 kHz` rather than for equality (a
1e9 Hz centre round-tripped through float32 moves by ~64 Hz, against a
13.375 MHz narrowest channel), and both sides are sliced to the
intersection when one covers more channels than the other.

The match must be **injective**: if two of p's channels fall within the
tolerance of one of q's, the assembly raises rather than using q's plane
twice. Nothing downstream could detect that aliasing, since both sides
would still come out the same length. It is unreachable with MdV spacing
and is treated as a corrupt input, not something to resolve by picking a
nearest match. When a slice does happen, `fits_header`'s `NAXIS3`,
`CRVAL3` and `CDELT3` are refreshed to describe the sliced cube — the
header is part of the BDS contract, so it must not keep describing p's
unsliced one.

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
- `src/meerkat_beams/utils.py` (`jones_to_mueller_cross`, `GROUP_TELESCOPES`,
  `FREQ_MATCH_ATOL_HZ`, `_build_group_bds`)
- `tests/test_group_bds.py`, `tests/test_jones_mueller.py`,
  `tests/test_mdv_beams_to_bds_attrs.py`, `tests/test_group_integration.py`
- issue #30 (MeerKAT+ Mueller blocks per baseline group)
