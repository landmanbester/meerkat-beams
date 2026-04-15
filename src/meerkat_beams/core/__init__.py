"""Core implementations for meerkat-beams.

Each module in this package corresponds 1:1 to a CLI command and cab definition:

    core/download_mdv_beams.py  <->  cli/download_mdv_beams.py  <->  cabs/download_mdv_beams.yml
    core/mdv_beams_to_bds.py    <->  cli/mdv_beams_to_bds.py    <->  cabs/mdv_beams_to_bds.yml
    core/bds_to_xradio.py       <->  cli/bds_to_xradio.py       <->  cabs/bds_to_xradio.yml
    core/mdv_to_xradio.py       <->  cli/mdv_to_xradio.py       <->  cabs/mdv_to_xradio.yml

Shared utilities (BeamWizard, logging, etc.) live in meerkat_beams.utils.
"""
