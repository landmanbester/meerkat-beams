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


@pytest.mark.unit
def test_cache_root_prefers_mbeams_cache_dir(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit"
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("MBEAMS_CACHE_DIR", str(explicit))
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg))
    assert cache.cache_root() == explicit
    assert explicit.is_dir()


@pytest.mark.unit
def test_cache_root_falls_back_to_xdg(tmp_path, monkeypatch):
    xdg = tmp_path / "xdg"
    monkeypatch.delenv("MBEAMS_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg))
    assert cache.cache_root() == xdg / "meerkat-beams"
    assert (xdg / "meerkat-beams").is_dir()


@pytest.mark.unit
def test_cache_root_falls_back_to_home(tmp_path, monkeypatch):
    monkeypatch.delenv("MBEAMS_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert cache.cache_root() == tmp_path / ".cache" / "meerkat-beams"


@pytest.mark.unit
def test_cache_root_empty_env_vars_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv("MBEAMS_CACHE_DIR", "")
    monkeypatch.setenv("XDG_CACHE_HOME", "")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert cache.cache_root() == tmp_path / ".cache" / "meerkat-beams"
