#!/bin/bash

hip-cargo generate-function --cab-file src/meerkat_beams/cabs/bds_to_fits.yml --config-file pyproject.toml --output-file src/meerkat_beams/cli/bds_to_fits.py
hip-cargo generate-function --cab-file src/meerkat_beams/cabs/bds_to_xradio.yml --config-file pyproject.toml --output-file src/meerkat_beams/cli/bds_to_xradio.py
hip-cargo generate-function --cab-file src/meerkat_beams/cabs/mdv_beams_to_bds.yml --config-file pyproject.toml --output-file src/meerkat_beams/cli/mdv_beams_to_bds.py
