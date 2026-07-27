# Changelog

All notable changes to meerkat-beams are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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


[0.0.1]: https://github.com/landmanbester/meerkat-beams/releases/tag/v0.0.1

