# Changelog

All notable changes to meerkat-beams are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Add BAND_INPUT_ZARR mapping to conftest for regression tests
- Port tests as pytest with skip markers
- Update CLI wrappers and add scientific dependencies
- Port core modules from suricat-beams

### Fixed

- Adopt PEP 440 versions and repair tbump
- Repair update-cabs workflow token and cab generation
- Skip antenna selection for zarr input in mdv_beams_to_bds

### Miscellaneous

- Correct mbeans->hip-cargo in tbump. remove stale docs/progress.md
- Remove scaffold onboard artifacts, update README
- Initial project scaffold

### Other

- Hip-cargo transition + beam-orientation validation (dev001 → main) ([#8](https://github.com/landmanbester/meerkat-beams/pull/8))

### Testing

- Test against suricat outputs. skip cab generation in pre-commits for time being
- Add parametrized BDS regression tests for mdv_beams_to_bds



