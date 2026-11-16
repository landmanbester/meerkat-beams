"""Core implementations for meerkat-beams.

Each module in this package corresponds 1:1 to a CLI command and cab definition:

    core/mdv_beams_to_bds.py    <->  cli/mdv_beams_to_bds.py    <->  cabs/mdv_beams_to_bds.yml
    core/bds_to_xradio.py       <->  cli/bds_to_xradio.py       <->  cabs/bds_to_xradio.yml
    core/bds_to_fits.py         <->  cli/bds_to_fits.py         <->  cabs/bds_to_fits.yml

Shared utilities (BeamWizard, resolve_elements, logging, etc.) live in meerkat_beams.utils.
"""
