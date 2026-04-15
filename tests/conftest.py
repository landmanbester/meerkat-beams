import tarfile
from pathlib import Path

import requests

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

gdrive_id = "13k5WyyQFdcNG8FqsBAz3mAvhuaVZ2FlW"

url = "https://drive.google.com/uc?id={id}".format(id=gdrive_id)


def pytest_sessionstart(session):
    """Called after Session object has been created, before run test loop."""

    if beam_path.exists():
        print("Test data already present - not downloading.")
    else:
        print("Test data not found - downloading...")
        download = requests.get(url)  # , params={"dl": 1}
        with open(data_tar_path, "wb") as f:
            f.write(download.content)
        with tarfile.open(data_tar_path, "r:gz") as tar:
            tar.extractall(path=test_data_path)
        data_tar_path.unlink()
        print("Test data successfully downloaded.")


# def download_file_from_google_drive(id, destination):
#     URL = "https://docs.google.com/uc?export=download"
#     session = requests.Session()

#     # First request to check for the 'large file' warning
#     response = session.get(URL, params={'id': id}, stream=True)
#     token = get_confirm_token(response)

#     # Second request with the confirmation token
#     if token:
#         params = {'id': id, 'confirm': token}
#         response = session.get(URL, params=params, stream=True)

#     save_response_content(response, destination)

# def get_confirm_token(response):
#     for key, value in response.cookies.items():
#         if key.startswith('download_warning'):
#             return value
#     return None

# def save_response_content(response, destination):
#     CHUNK_SIZE = 32768
#     with open(destination, "wb") as f:
#         for chunk in response.iter_content(CHUNK_SIZE):
#             if chunk: # filter out keep-alive new chunks
#                 f.write(chunk)
