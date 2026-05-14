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
