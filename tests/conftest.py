from pathlib import Path

test_root_path = Path(__file__).resolve().parent
test_data_path = Path(test_root_path, "data")
test_data_path.mkdir(parents=True, exist_ok=True)

_data_tar_name = "MeerKAT_U.zarr.tgz"
_beam_name = "MeerKAT_U.zarr"

data_tar_path = Path(test_data_path, _data_tar_name)
beam_path = Path(test_data_path, _beam_name)

# Band code -> input zarr filename in tests/data/
BAND_INPUT_ZARR = {
    "U": "MeerKAT_U.zarr",
    "L": "MeerKAT_L.zarr",
    # S-band sub-bands can be added as data becomes available
    "S0": "MeerKAT_S0.zarr",
    # "S1": "MeerKAT_S1.zarr",
    # "S2": "MeerKAT_S2.zarr",
    # "S3": "MeerKAT_S3.zarr",
    "S4": "MeerKAT_S4.zarr",
}

# https://drive.google.com/file/d/13k5WyyQFdcNG8FqsBAz3mAvhuaVZ2FlW/view?usp=sharing
# gdown handles the Google Drive "large file" confirm-token flow that a plain
# requests.get(...) is unable to negotiate (it returns the HTML interstitial).
gdrive_id = "13k5WyyQFdcNG8FqsBAz3mAvhuaVZ2FlW"


def pytest_sessionstart(session):
    """Called after Session object has been created, before run test loop."""
    # WIP: replaced in Task 10 of docs/superpowers/plans/2026-05-14-beamwizard-auto-download.md
    return
