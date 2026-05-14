"""
Hermetic unit tests for meerkat_beams.cache.

No network, no real conversion. As later tasks build out the module,
ensure_band_bds is exercised with the download+convert internals
monkeypatched.
"""

from pathlib import Path  # noqa: F401  -- used by later-task tests

import pytest

from meerkat_beams import cache


@pytest.mark.unit
def test_supported_bands_matches_registry():
    assert cache.SUPPORTED_BANDS == tuple(cache.BAND_GDRIVE_IDS.keys())


@pytest.mark.unit
def test_registry_contains_expected_bands():
    assert set(cache.BAND_GDRIVE_IDS) == {"U", "L", "S0", "S4"}
    for band, gid in cache.BAND_GDRIVE_IDS.items():
        assert isinstance(gid, str) and gid, f"empty gdrive id for {band}"


@pytest.mark.unit
def test_input_zarr_path_under_cache_root(tmp_path, monkeypatch):
    monkeypatch.setenv("MBEAMS_CACHE_DIR", str(tmp_path))
    assert cache.input_zarr_path("U") == tmp_path / "inputs" / "MeerKAT_U.zarr"


@pytest.mark.unit
def test_bds_path_under_cache_root(tmp_path, monkeypatch):
    monkeypatch.setenv("MBEAMS_CACHE_DIR", str(tmp_path))
    assert cache.bds_path("L") == tmp_path / "bds" / "MeerKAT_L.bds.zarr"
