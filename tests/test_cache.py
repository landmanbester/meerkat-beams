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


@pytest.mark.unit
def test_ensure_band_bds_rejects_unknown_band(tmp_path, monkeypatch):
    monkeypatch.setenv("MBEAMS_CACHE_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="band must be one of"):
        cache.ensure_band_bds("Q")


@pytest.mark.unit
def test_ensure_band_bds_short_circuits_when_bds_exists(tmp_path, monkeypatch):
    monkeypatch.setenv("MBEAMS_CACHE_DIR", str(tmp_path))
    bds = cache.bds_path("U")
    bds.mkdir(parents=True)
    (bds / ".zgroup").write_text("{}")

    def boom(*a, **kw):
        raise AssertionError("must not be called when BDS already exists")

    monkeypatch.setattr(cache, "_download_and_extract", boom)
    monkeypatch.setattr(cache, "_convert_to_bds", boom)
    assert cache.ensure_band_bds("U") == str(bds)


@pytest.mark.unit
def test_ensure_band_bds_skips_download_when_input_exists(tmp_path, monkeypatch):
    monkeypatch.setenv("MBEAMS_CACHE_DIR", str(tmp_path))
    inp = cache.input_zarr_path("U")
    inp.mkdir(parents=True)
    (inp / ".zgroup").write_text("{}")

    def must_not_download(*a, **kw):
        raise AssertionError("download must not run when input already exists")

    convert_calls = []

    def stub_convert(band):
        convert_calls.append(band)
        out = cache.bds_path(band)
        out.mkdir(parents=True)
        (out / ".zgroup").write_text("{}")

    monkeypatch.setattr(cache, "_download_and_extract", must_not_download)
    monkeypatch.setattr(cache, "_convert_to_bds", stub_convert)

    result = cache.ensure_band_bds("U")
    assert result == str(cache.bds_path("U"))
    assert convert_calls == ["U"]


@pytest.mark.unit
def test_ensure_band_bds_clears_stale_partials(tmp_path, monkeypatch, caplog):
    import logging

    monkeypatch.setenv("MBEAMS_CACHE_DIR", str(tmp_path))

    inp = cache.input_zarr_path("U")
    out = cache.bds_path("U")
    stale_input = inp.with_name(inp.name + ".partial")
    stale_bds = out.with_name(out.name + ".partial")
    stale_input.mkdir(parents=True)
    (stale_input / "junk").write_text("x")
    stale_bds.mkdir(parents=True)
    (stale_bds / "junk").write_text("x")

    def stub_download(band):
        inp = cache.input_zarr_path(band)
        inp.mkdir(parents=True)
        (inp / ".zgroup").write_text("{}")

    def stub_convert(band):
        out = cache.bds_path(band)
        out.mkdir(parents=True)
        (out / ".zgroup").write_text("{}")

    monkeypatch.setattr(cache, "_download_and_extract", stub_download)
    monkeypatch.setattr(cache, "_convert_to_bds", stub_convert)

    with caplog.at_level(logging.WARNING, logger="meerkat_beams"):
        cache.ensure_band_bds("U")

    assert not stale_input.exists()
    assert not stale_bds.exists()
    assert any("partial" in r.message.lower() for r in caplog.records)
