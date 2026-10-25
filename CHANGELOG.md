# Changelog

All notable changes to meerkat-beams are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-10-25

### Added

- Evaluate out-of-range frequencies by lambda/D scaling of the edge plane
- Add freq_extrapolation policy, freq_range and check_freqs to BeamWizard

### CI

- Bump astral-sh/setup-uv

### Documentation

- README review for release; document lossless compression, native CLI paths and the npz deprecation
- Point README at suricat-beams+MdV paper etc.
- Document manual data tests, their resources, and the offline IERS guard
- D19 points the linear term at pfb-imaging's --spectral linear
- D19 chord bias note, B-table maxima, non-finite frequency guard
- D19 frequency extrapolation policy and BeamWizard freq guard

### Fixed

- Accept the CLI's UPath arguments natively; deprecate the raw-npz commands
- Drop the lossy Delta filter from compressed zarr writes
- Reject non-finite frequencies in check_freqs; pin the chord in the chunk measurement

### Testing

- Skip IERS downloads under MBEAMS_OFFLINE so CI does not stall on dead IERS mirrors
- Bound lambda/D extrapolation and in-chunk beam error on the real L-band products
- Pin freq_extrapolation through the public BeamWizard entry points


## [0.2.0] - 2026-10-18

### CI

- Run the test suite against zarr-python 2

### Changed

- Thread and merge angles in get_rotation_averaged_beam

### Dependencies

- Unpin zarr, writing zarr v2 under zarr-python 2 or 3
- Drop dask-ms from the [full] extra

### Documentation

- List the speed options in the wiki index and re-stamp at 9f91975
- Record the hardened speed options, fix stale citations, re-stamp
- Record the rotation-averaging speed options and re-stamp the wiki
- Record the zarr unpin review fixes and re-stamp the wiki
- Record the zarr unpin and re-stamp the wiki
- Re-stamp beam-orientation at the ms_io TaQL fix
- Stamp wiki at the dask-ms removal
- Re-point stale wiki line citations and fix stamps after review
- Stamp design-decisions at the Dockerfile change and settle D8-D10
- Record #13/#21 closures and correct D8's stale #16 status

### Fixed

- Harden the rotation-averaging speed options (#40 review)
- Address review of the zarr unpin
- Select MS rows in NumPy instead of TaQL in ms_io

### Other

- Drop the Dockerfile git layer left over from the hip-cargo git pin

### Testing

- Mark the merge-to-mean-angle test unit, not its helper


## [0.1.0] - 2026-10-06

### Added

- Validate partition_mueller arguments with actionable messages
- Add BeamWizard.partition_mueller for per-partition Mueller blocks
- Let get_rotation_averaged_beam take an explicit pointing centre
- Add group= selector to BeamWizard for MM/MPM/MPMP baseline groups
- Assemble baseline-group beam datasets from two single-telescope BDSs
- **bds**: Carry telescope and provenance attrs from the input zarr
- **cache**: Key the beam cache on product names and add ensure_group_bds
- Add jones_to_mueller_cross for mixed-antenna baselines
- Add full azimuthal averaging to get_rotation_averaged_beam
- Add a beam_model option to BeamWizard
- Synthesize a BDS-shaped dataset from katbeam beams
- Add katbeam band mapping, model validation and grid geometry
- **scripts**: Add plots and CLI entry point to compare_katbeam
- **scripts**: Add orientation sweep and metrics assembly to compare_katbeam
- **scripts**: Load BDS products and evaluate katbeam in compare_katbeam
- **scripts**: Add frequency selection and residual stats to compare_katbeam
- **scripts**: Add geometry and shape helpers to compare_katbeam
- **scripts**: Add compare_katbeam skeleton and band-to-model map

### CI

- Bump astral-sh/setup-uv
- Bump astral-sh/setup-uv in the github-actions group

### Changed

- Extract Jones to Stokes helpers into utils

### Dependencies

- Update uv-build requirement
- Add katbeam and dask to the full extra

### Documentation

- Fix the wiki stamping rule at the source, in CLAUDE.md
- Stamp the wiki pages at the commit their claims were checked against
- Document partition_mueller and the centre= override
- Record the post-review group fixes
- Document MeerKAT+ beams and MM/MPM/MPMP baseline groups
- **wiki**: Document the katbeam beam model and azimuthal averaging
- Ignore generated outputs/ and correct the uv sync incantation
- **wiki**: Record katbeam comparison findings and orientation probe

### Fixed

- Make the partition_mueller fixtures discriminating after review
- Harden group channel alignment and cache staging after review
- Pin the p=MeerKAT group contract and guard ambiguous channel matches
- **scripts**: Commit the plot_tag helper that f452ba1 left behind
- Normalise katbeam beams on axis and validate the frequency axis
- Reject non-positive geometry and halve block memory
- **scripts**: Re-register mirrors in compare_katbeam orientation sweep

### Other

- Upgrade hip-cargo >= 0.4.0

### Testing

- Pin partition_mueller dtypes for the MM/MPM/MPMP groups
- Verify baseline groups against the real MKE and MdV-2026 beams
- Give three vacuous tests teeth, and bound block memory


## [0.0.1] - 2026-07-27

### Added

- Add BAND_INPUT_ZARR mapping to conftest for regression tests
- Port tests as pytest with skip markers
- Update CLI wrappers and add scientific dependencies
- Port core modules from suricat-beams

### Fixed

- Repair update-cabs workflow token and cab generation
- Skip antenna selection for zarr input in mdv_beams_to_bds

### Miscellaneous

- Remove scaffold onboard artifacts, update README
- Initial project scaffold

### Other

- PEP 440 versioning + git-cliff changelog (repairs tbump) ([#24](https://github.com/landmanbester/meerkat-beams/pull/24))
- Hip-cargo transition + beam-orientation validation (dev001 → main) ([#8](https://github.com/landmanbester/meerkat-beams/pull/8))

### Testing

- Test against suricat outputs. skip cab generation in pre-commits for time being
- Add parametrized BDS regression tests for mdv_beams_to_bds


[0.3.0]: https://github.com/landmanbester/meerkat-beams/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/landmanbester/meerkat-beams/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/landmanbester/meerkat-beams/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/landmanbester/meerkat-beams/releases/tag/v0.0.1

