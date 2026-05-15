"""
pytest session-level setup.

Ensures the L-band BDS cache is populated before tests run so the
integration suite (which uses an L-band test MS) has data available.
Set MBEAMS_OFFLINE=1 to skip the download (for air-gapped CI).
"""

import os
from pathlib import Path

from meerkat_beams import cache

test_root_path = Path(__file__).resolve().parent
test_data_path = test_root_path / "data"
test_data_path.mkdir(parents=True, exist_ok=True)

# Kept for fixtures that pair the BDS with a test measurement set.
# https://drive.google.com/file/d/1mCTrC3IbMUqu0Adu1DWOjhwvzQS6gseo/view?usp=drive_link
test_ms_gdrive_id = "1mCTrC3IbMUqu0Adu1DWOjhwvzQS6gseo"

# Primary calibrator location
ra = "19:39:25.027"
dec = "-63.42.45.626"

# Primary calibrator (PKS 1934-638) spectral model.
# I(nu) = I0 * (nu/nu0) ** (a + b*x + c*x**2 + d*x**3 + e*x**4)   where x = log10(nu/nu0)
CALIBRATOR_SPECTRUM = {
    "I0": 15.088731791006047,
    "nu0": 1283791015.625,
    "a": -1.2369319597991164,
    "b": -7.995603882017982,
    "c": 11.605973123430397,
    "d": -15.787559501497967,
    "e": -3.928824456855068,
}


def pytest_sessionstart(session):
    """Populate the L-band cache once per session if it isn't there yet."""
    if os.environ.get("MBEAMS_OFFLINE") == "1":
        print("MBEAMS_OFFLINE=1 - skipping L-band cache warm-up.")
        return
    if cache.bds_path("L").exists():
        print(f"L-band BDS already cached at {cache.bds_path('L')}.")
        return
    print("L-band BDS not in cache - downloading and converting...")
    cache.ensure_band_bds("L")
    print(f"L-band BDS ready at {cache.bds_path('L')}.")
