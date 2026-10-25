# meerkat-beams

This is a port of [suricat-beams](https://github.com/ratt-ru/suricat-beams) to the [hip-cargo](https://github.com/landmanbester/hip-cargo) format.

MeerKAT primary-beam modelling based on the holographic beam models of [M. de Villiers](https://doi.org/10.48479/wdb0-h061) (see [paper](https://ui.adsabs.harvard.edu/abs/2023AJ....165...78D/abstract)). Only the array mean beam is currently exposed.

## Installation

```bash
pip install meerkat-beams[full]
```

The `[full]` extra pulls in the scientific stack (`xarray`, `xarray-fits`, `zarr`, `numcodecs`, `astropy`, `scipy`, `numpy`, `matplotlib`, `dask`, `wget`, `gdown`, `requests`, `katbeam` for the analytic beam model, plus `hip-cargo[s3]` for `s3://` paths). Use plain `meerkat-beams` only if you need the CLI metadata for Stimela without running the implementations locally. Every store the package writes is zarr format 2, whether zarr-python 2 or 3 is installed, so its output stays readable by either.

### Python support

| Python | Support |
|---|---|
| 3.11 – 3.13 | Full: `[full]` scientific stack, tested in CI |
| 3.10 | Lightweight only: base install (CLI + hip-cargo container dispatch via `--backend docker/podman/...`). The `[full]` stack is neither tested nor supported on 3.10. |

On 3.10 the base install lets you run every `mbeams` command by dispatching the actual computation to the container image — nothing scientific runs natively. CI pins this with a dedicated lightweight-install job; the test matrix itself starts at 3.11 (do not add 3.10 to it).

## Quick start

`BeamWizard` is the main entry point. Give it a MeerKAT band code and the mean-beam zarr for that band is downloaded from Google Drive on first use, converted to a beam dataset (BDS), and cached locally:

```python
import numpy as np
import astropy.units as u
from astropy.time import Time
from meerkat_beams.utils import BeamWizard

bw = BeamWizard(band="L", image_name="my_image.fits")
src = bw.centre.directional_offset_by(45 * u.deg, 0.5 * u.deg)  # a source 0.5 deg from the pointing centre
times = Time("2024-01-01T18:00:00") + np.linspace(0, 8, 33) * u.hour
gain = bw.get_time_variable_beamgain(src, times=times, freq=np.array([1.0e9, 1.4e9]))
# gain.shape == (2, 33): centre-normalised Stokes I beam per frequency and time
```

The image supplies the pointing centre and the l/m grid from its WCS. It can be either of two things:

- **A FITS image** carries no time axis, so pass `times=` to every method that tracks the parallactic angle, as above. Without it they raise.
- **A zarr image cube** with a `TIME` coordinate and a `fits_header` attribute, such as a pfb-imaging `hci` output, gives the wizard its times, so `times=` becomes optional. `bds-to-xradio` needs this kind of image.

The image is optional. Without one, pass the pointing centre and times explicitly (`set_field_centre`, or `centre=` and `times=` on the methods that take them). Frequencies default to every BDS channel.

Supported bands: `U`, `L`, `S0`, `S4`. There are no S1, S2 or S3 beams: neither the mean-beam mirror nor the SARAO archive has them. To use a BDS you built yourself, pass `bds_name=` instead of `band=`.

### What else the wizard does

- `get_rotation_averaged_beam`: the beam averaged over the observation's parallactic angles, or over a full turn with `average="azimuth"`.
- `partition_mueller`: the rotation-averaged Stokes-basis Mueller block for one data partition, for imagers and calibrators that apply the full polarised beam.
- `get_time_freq_beam`: render a full `(time, frequency, polarization, l, m)` xradio zarr; `bds-to-xradio` wraps it together with the module-level `enrich_bds_xradio`.
- `beam_model="katbeam"`: the analytic katbeam model instead of the MdV holographic beams. Nothing is downloaded.
- `group="MM" | "MPM" | "MPMP"` (L band only): the baseline-group beams for a mixed MeerKAT / MeerKAT+ array, built from the 2026 MdV MeerKAT and MeerKAT+ products. These are not on the download mirrors yet, so stage them first with `scripts/stage_group_cache.py`.
- `freq_extrapolation="scale"`: evaluate frequencies up to one BDS channel past the BDS range by scaling the edge plane with λ/D, so data out to the receiver band edges can be modelled. `freq_range` and `check_freqs` let you validate frequencies up front. The default, `"raise"`, refuses anything outside the BDS range.

[`docs/wiki/beamwizard.md`](docs/wiki/beamwizard.md) documents every method, and [`docs/wiki/data-model.md`](docs/wiki/data-model.md) the BDS and xradio schemas.

### Cache layout

```
$MBEAMS_CACHE_DIR/  (or $XDG_CACHE_HOME/meerkat-beams, or ~/.cache/meerkat-beams)
├── inputs/<product>.zarr/      # downloaded (or staged) mean-beam zarr
└── bds/<product>.bds.zarr/     # BDS built from it locally, Blosc-compressed
```

`<product>` is `MeerKAT_<BAND>` for the band codes, and `MeerKAT_L_mdv2026` or `MKE_L` for the 2026 products behind `group=`. Later constructions reuse the cached BDS, so to warm a band ahead of time, construct a wizard once.

**Concurrency:** the first-time download is not locked. Warm each band from a single process before fanning out.

**Upgrading from 0.2.0 or earlier:** those releases compressed the cached BDS with a lossy filter that also wrote NaNs into the MdV-2026 products. Delete `bds/` from the cache once, and it is rebuilt on next use.

## CLI

The `mbeams` command wraps each stage of the pipeline. The CLI is generated by [`hip-cargo`](https://github.com/landmanbester/hip-cargo). With the default `--backend auto`, each command runs natively and falls back to the container image only when the scientific stack is not installed, for example on a base install. Pass `--backend apptainer|singularity|docker|podman` to choose the container runtime explicitly.

```bash
# Convert a mean-beam zarr (or a raw MdV .npz) to a BDS
mbeams mdv-beams-to-bds --mdv-beams MeerKAT_L.zarr --bds MeerKAT_L.bds.zarr --compress

# Render a BDS to an xradio-compatible zarr along an image cube's WCS and time axis
mbeams bds-to-xradio --bds-path MeerKAT_L.bds.zarr --image-path image_cube.zarr --output beam_xradio.zarr
```

`bds-to-xradio` needs a zarr image cube with a `TIME` coordinate; a FITS image has no time axis to render along. Each command accepts `--help`.

Two commands are **deprecated** and will be removed in a future release. Both work only with the raw MdV `.npz` files (about 35 GB per band), which the package no longer needs:

- `download-mdv-beams` fetches a raw `.npz` from the SARAO archive. `BeamWizard(band=...)` downloads the mean-beam zarr (about 420 MB) itself.
- `mdv-to-xradio` dumps one Jones element of a raw `.npz` straight to an xradio zarr. Use `mdv-beams-to-bds` followed by `bds-to-xradio` instead.

## Development

```bash
uv sync --group dev --group test --extra full   # always pass --extra full: without it uv removes the scientific stack
pre-commit install && pre-commit install --hook-type commit-msg
uv run ruff check . && uv run ruff format --check .
MBEAMS_OFFLINE=1 uv run pytest                  # what CI runs: hermetic, ~20 s
uv run pytest                                   # online: L-band cache warm-up and live IERS tables
bash scripts/genfuncs.sh                        # regenerate cli/*.py from cabs/*.yml
```

`CLAUDE.md` and the wiki under `docs/wiki/` hold the full architecture, conventions and test layout.

### CLI and cab generation

`src/meerkat_beams/cli/*.py` and `src/meerkat_beams/cabs/*.yml` are generated by [`hip-cargo`](https://github.com/landmanbester/hip-cargo), not written by hand. The `generate-cabs` pre-commit hook regenerates the cabs from the CLI on every commit (re-stage and commit again if it changes them), `scripts/genfuncs.sh` goes the other way, and `tests/test_roundtrip.py` fails if the two drift apart. Edit the plain-Python implementations in `core/` freely. See the [hip-cargo README](https://github.com/landmanbester/hip-cargo#readme) for the decorators, the generators and how native-then-container dispatch works.

### Tests

There are three tiers.

**CI** runs `MBEAMS_OFFLINE=1 uv run pytest` on Python 3.11–3.13 and on zarr-python 2. `MBEAMS_OFFLINE=1` makes the run hermetic:

- it skips the L-band cache warm-up in `tests/conftest.py`, so tests that need real beams skip;
- it stops astropy downloading IERS Earth-orientation tables and uses the copy bundled with astropy, which is ample for parallactic angles. Without this, a failed download is retried by every test that computes a parallactic angle, at 30–75 s each, so an IERS outage stretches a one-minute CI run to hours.

**Online local runs** (`uv run pytest` without `MBEAMS_OFFLINE`) download the L-band input (~420 MB) from Google Drive and convert it into the cache on first use, and astropy uses live IERS tables. If the IERS servers are down the run stalls as described above; set `MBEAMS_OFFLINE=1` to fall back to the bundled table.

**Manual data tests** need beam products that are not public or are too big for CI, and skip silently unless their environment variables are set. Run them before a release, or when you touch the code they cover:

| tests | environment variables | data | cost (measured locally) |
|---|---|---|---|
| `test_mdv_beams_to_bds.py`: convert the legacy U and L inputs and compare against reference BDSs | `MBEAMS_REFERENCE_BDS_U`, `MBEAMS_REFERENCE_BDS_L` | Reference BDSs (5.4 GB each; not published, ask the maintainer). The inputs must already be in the cache: L is fetched by the warm-up, U by `meerkat_beams.cache.ensure_band_bds("U")`. | ~80 s, **~27 GB peak RAM**, ~5.4 GB scratch per band in `$TMPDIR` |
| `test_beam_consistency.py`, `test_rendered_beam.py`: time-variable gain vs rotation-averaged beam vs rendered zarr | `MBEAMS_BDS_PATH`, `MBEAMS_IMAGE_PATH` | Legacy L BDS, plus an image zarr with a `TIME` coordinate and a `fits_header` attr, such as a pfb-imaging `hci` cube. A FITS image does not work: it has no time axis. | ~20 s, ~2 GB |
| `test_freq_extrapolation_integration.py`: λ/D extrapolation and in-chunk error bounds (D19) | `MBEAMS_BDS_PATH`, `MBEAMS_MK2026_BDS_PATH`, `MBEAMS_MKE_BDS_PATH` | Legacy L BDS, plus MdV-2026 MeerKAT and MeerKAT+ L BDSs (see below) | ~60 s, ~1 GB |
| `test_group_integration.py`: baseline-group assembly on the real 2026 beams | `MBEAMS_MK2026_BDS_PATH`, `MBEAMS_MKE_BDS_PATH` | MdV-2026 MeerKAT and MeerKAT+ L BDSs | a few seconds |

The MdV-2026 products are not on the download mirrors yet. Stage their mean-beam zarrs with `scripts/stage_group_cache.py`, then construct `BeamWizard(band="L", group="MPM")` once (or call `mdv_beams_to_bds` directly) to build `<cache>/bds/MeerKAT_L_mdv2026.bds.zarr` and `<cache>/bds/MKE_L.bds.zarr`.

The conversion and rendering tests write multi-GB stores through `tempfile`, so point `TMPDIR` at a disk with room if `/tmp` is small or RAM-backed. A full manual run looks like:

```bash
export TMPDIR=/path/with/space
export MBEAMS_REFERENCE_BDS_U=/path/to/meerkat_U.bds MBEAMS_REFERENCE_BDS_L=/path/to/meerkat_L.bds
export MBEAMS_BDS_PATH=/path/to/meerkat_L.bds MBEAMS_IMAGE_PATH=/path/to/image_with_time.zarr
export MBEAMS_MK2026_BDS_PATH=~/.cache/meerkat-beams/bds/MeerKAT_L_mdv2026.bds.zarr
export MBEAMS_MKE_BDS_PATH=~/.cache/meerkat-beams/bds/MKE_L.bds.zarr
uv run pytest -rs        # -rs lists every skip with its reason
```

Even with everything set, some tests always skip:

- the S0–S4 conversion regressions: S0 and S4 have published inputs but no reference BDS, and S1–S3 have no inputs;
- `test_bds_to_xradio_end_to_end_synthetic`: `bds_to_xradio` cannot render a FITS image, which has no time axis;
- one `test_zarr_format.py` test, which applies only under the other zarr-python major.
