import tarfile
from pathlib import Path

import gdown

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

    if beam_path.exists():
        print("Test data already present - not downloading.")
        return

    print("Test data not found - downloading...")
    gdown.download(id=gdrive_id, output=str(data_tar_path), quiet=False)
    with tarfile.open(data_tar_path, "r:gz") as tar:
        tar.extractall(path=test_data_path)
    data_tar_path.unlink()
    print("Test data successfully downloaded.")
