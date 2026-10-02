"""
Hermetic unit tests for meerkat_beams.cache.

No network, no real conversion. As later tasks build out the module,
ensure_band_bds is exercised with the download+convert internals
monkeypatched.
"""

import tarfile
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

    def stub_convert(product):
        convert_calls.append(product)
        out = cache.bds_path_for_product(product)
        out.mkdir(parents=True)
        (out / ".zgroup").write_text("{}")

    monkeypatch.setattr(cache, "_download_and_extract", must_not_download)
    monkeypatch.setattr(cache, "_convert_to_bds", stub_convert)

    result = cache.ensure_band_bds("U")
    assert result == str(cache.bds_path("U"))
    assert convert_calls == ["MeerKAT_U"]


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

    def stub_download(product):
        inp = cache.input_zarr_path_for_product(product)
        inp.mkdir(parents=True)
        (inp / ".zgroup").write_text("{}")

    def stub_convert(product):
        out = cache.bds_path_for_product(product)
        out.mkdir(parents=True)
        (out / ".zgroup").write_text("{}")

    monkeypatch.setattr(cache, "_download_and_extract", stub_download)
    monkeypatch.setattr(cache, "_convert_to_bds", stub_convert)

    with caplog.at_level(logging.WARNING, logger="meerkat_beams"):
        cache.ensure_band_bds("U")

    assert not stale_input.exists()
    assert not stale_bds.exists()
    assert any("partial" in r.message.lower() for r in caplog.records)


def _make_fake_tarball(tar_path: Path, member_name: str):
    """Build a real .tar.gz on disk containing a single zarr-shaped directory."""
    payload_dir = tar_path.parent / "_stage"
    payload_dir.mkdir(parents=True, exist_ok=True)
    zarr_dir = payload_dir / member_name
    zarr_dir.mkdir(parents=True, exist_ok=True)
    (zarr_dir / ".zgroup").write_text("{}")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(zarr_dir, arcname=member_name)


@pytest.mark.unit
def test_download_and_extract_writes_atomic(tmp_path, monkeypatch):
    monkeypatch.setenv("MBEAMS_CACHE_DIR", str(tmp_path))

    def fake_gdown_download(id, output, quiet):  # noqa: A002
        _make_fake_tarball(Path(output), "MeerKAT_U.zarr")

    monkeypatch.setattr(cache, "_gdown_download", fake_gdown_download)
    cache._download_and_extract("MeerKAT_U")

    inp = cache.input_zarr_path("U")
    assert inp.is_dir()
    assert (inp / ".zgroup").exists()
    assert not cache._partial(inp).exists()


@pytest.mark.unit
def test_download_and_extract_failure_cleans_partial(tmp_path, monkeypatch):
    monkeypatch.setenv("MBEAMS_CACHE_DIR", str(tmp_path))

    def boom(id, output, quiet):  # noqa: A002
        Path(output).write_text("not a tarball")  # download "succeeds" with junk

    monkeypatch.setattr(cache, "_gdown_download", boom)
    with pytest.raises(Exception):
        cache._download_and_extract("MeerKAT_U")

    assert not cache.input_zarr_path("U").exists()
    assert not cache._partial(cache.input_zarr_path("U")).exists()


@pytest.mark.unit
def test_download_and_extract_missing_gdown(tmp_path, monkeypatch):
    monkeypatch.setenv("MBEAMS_CACHE_DIR", str(tmp_path))

    def raise_import(*a, **kw):
        raise ImportError("gdown not available")

    monkeypatch.setattr(cache, "_gdown_download", raise_import)
    with pytest.raises(ImportError, match=r"meerkat-beams\[full\]"):
        cache._download_and_extract("MeerKAT_U")


@pytest.mark.unit
def test_convert_to_bds_atomic(tmp_path, monkeypatch):
    monkeypatch.setenv("MBEAMS_CACHE_DIR", str(tmp_path))

    inp = cache.input_zarr_path("U")
    inp.mkdir(parents=True)
    (inp / ".zgroup").write_text("{}")

    calls = []

    def stub_mdv(mdv_beams, bds, compress):
        calls.append((mdv_beams, bds, compress))
        Path(bds).mkdir(parents=True)
        (Path(bds) / ".zgroup").write_text("{}")

    monkeypatch.setattr("meerkat_beams.core.mdv_beams_to_bds.mdv_beams_to_bds", stub_mdv)
    cache._convert_to_bds("MeerKAT_U")

    out = cache.bds_path("U")
    assert out.is_dir()
    assert (out / ".zgroup").exists()
    assert not cache._partial(out).exists()
    assert calls[0][0] == str(inp)
    assert calls[0][1].endswith(".partial")
    assert calls[0][2] is True


@pytest.mark.unit
def test_convert_to_bds_failure_preserves_input(tmp_path, monkeypatch):
    monkeypatch.setenv("MBEAMS_CACHE_DIR", str(tmp_path))

    inp = cache.input_zarr_path("U")
    inp.mkdir(parents=True)
    (inp / ".zgroup").write_text("{}")

    def boom(mdv_beams, bds, compress):
        Path(bds).mkdir(parents=True)
        (Path(bds) / "half").write_text("x")
        raise RuntimeError("conversion failed")

    monkeypatch.setattr("meerkat_beams.core.mdv_beams_to_bds.mdv_beams_to_bds", boom)
    with pytest.raises(RuntimeError, match="conversion failed"):
        cache._convert_to_bds("MeerKAT_U")

    assert not cache.bds_path("U").exists()
    assert not cache._partial(cache.bds_path("U")).exists()
    assert inp.exists(), "input zarr must be preserved on conversion failure"


@pytest.mark.unit
def test_clear_partials_removes_stale_tarball(tmp_path, monkeypatch, caplog):
    import logging

    monkeypatch.setenv("MBEAMS_CACHE_DIR", str(tmp_path))
    inp = cache.input_zarr_path("U")
    inp.parent.mkdir(parents=True, exist_ok=True)
    stale_tarball = inp.parent / "MeerKAT_U.zarr.tgz"
    stale_tarball.write_text("garbage from a killed download")

    def stub_download(product):
        out = cache.input_zarr_path_for_product(product)
        out.mkdir(parents=True)
        (out / ".zgroup").write_text("{}")

    def stub_convert(product):
        out = cache.bds_path_for_product(product)
        out.mkdir(parents=True)
        (out / ".zgroup").write_text("{}")

    monkeypatch.setattr(cache, "_download_and_extract", stub_download)
    monkeypatch.setattr(cache, "_convert_to_bds", stub_convert)

    with caplog.at_level(logging.WARNING, logger="meerkat_beams"):
        cache.ensure_band_bds("U")

    assert not stale_tarball.exists(), "stale tarball should have been removed"
    assert any("tarball" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# Product keying and baseline groups
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_product_paths_under_cache_root(tmp_path, monkeypatch):
    monkeypatch.setenv("MBEAMS_CACHE_DIR", str(tmp_path))
    assert cache.input_zarr_path_for_product("MKE_L") == tmp_path / "inputs" / "MKE_L.zarr"
    assert cache.bds_path_for_product("MKE_L") == tmp_path / "bds" / "MKE_L.bds.zarr"


@pytest.mark.unit
def test_legacy_band_paths_delegate_to_product_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("MBEAMS_CACHE_DIR", str(tmp_path))
    assert cache.input_zarr_path("U") == cache.input_zarr_path_for_product("MeerKAT_U")
    assert cache.bds_path("U") == cache.bds_path_for_product("MeerKAT_U")


@pytest.mark.unit
def test_group_products_table():
    assert cache.SUPPORTED_GROUPS == ("MM", "MPM", "MPMP")
    assert cache.SUPPORTED_GROUP_BANDS == ("L",)
    assert cache.GROUP_PRODUCTS[("L", "MM")] == ("MeerKAT_L_mdv2026", "MeerKAT_L_mdv2026")
    assert cache.GROUP_PRODUCTS[("L", "MPM")] == ("MeerKAT_L_mdv2026", "MKE_L")
    assert cache.GROUP_PRODUCTS[("L", "MPMP")] == ("MKE_L", "MKE_L")


@pytest.mark.unit
def test_ensure_group_bds_rejects_unknown_group(tmp_path, monkeypatch):
    monkeypatch.setenv("MBEAMS_CACHE_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="group must be one of"):
        cache.ensure_group_bds("L", "XX")


@pytest.mark.unit
def test_ensure_group_bds_rejects_unsupported_band(tmp_path, monkeypatch):
    monkeypatch.setenv("MBEAMS_CACHE_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="no matched MeerKAT/MeerKAT. beam counterpart"):
        cache.ensure_group_bds("S3", "MPM")


@pytest.mark.unit
def test_ensure_group_bds_converts_each_product_once(tmp_path, monkeypatch):
    """MM names the same product twice: it must be converted once, not twice."""
    monkeypatch.setenv("MBEAMS_CACHE_DIR", str(tmp_path))
    for product in ("MeerKAT_L_mdv2026",):
        inp = cache.input_zarr_path_for_product(product)
        inp.mkdir(parents=True)
        (inp / ".zgroup").write_text("{}")

    convert_calls = []

    def stub_convert(product):
        convert_calls.append(product)
        out = cache.bds_path_for_product(product)
        out.mkdir(parents=True)
        (out / ".zgroup").write_text("{}")

    monkeypatch.setattr(cache, "_convert_to_bds", stub_convert)
    p, q = cache.ensure_group_bds("L", "MM")
    assert p == q == str(cache.bds_path_for_product("MeerKAT_L_mdv2026"))
    assert convert_calls == ["MeerKAT_L_mdv2026"]


@pytest.mark.unit
def test_ensure_group_bds_returns_two_paths_for_mpm(tmp_path, monkeypatch):
    monkeypatch.setenv("MBEAMS_CACHE_DIR", str(tmp_path))
    for product in ("MeerKAT_L_mdv2026", "MKE_L"):
        inp = cache.input_zarr_path_for_product(product)
        inp.mkdir(parents=True)
        (inp / ".zgroup").write_text("{}")

    def stub_convert(product):
        out = cache.bds_path_for_product(product)
        out.mkdir(parents=True)
        (out / ".zgroup").write_text("{}")

    monkeypatch.setattr(cache, "_convert_to_bds", stub_convert)
    p, q = cache.ensure_group_bds("L", "MPM")
    assert p == str(cache.bds_path_for_product("MeerKAT_L_mdv2026"))
    assert q == str(cache.bds_path_for_product("MKE_L"))


@pytest.mark.unit
def test_placeholder_product_raises_instead_of_downloading(tmp_path, monkeypatch):
    """An unpublished product must not reach gdown with a bogus id."""
    monkeypatch.setenv("MBEAMS_CACHE_DIR", str(tmp_path))

    def must_not_download(*a, **kw):
        raise AssertionError("gdown must not be called for a placeholder id")

    monkeypatch.setattr(cache, "_gdown_download", must_not_download)
    with pytest.raises(RuntimeError, match="not yet published"):
        cache.ensure_product_bds("MKE_L")


@pytest.mark.unit
def test_placeholder_error_names_the_cache_path(tmp_path, monkeypatch):
    monkeypatch.setenv("MBEAMS_CACHE_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match=str(cache.input_zarr_path_for_product("MKE_L"))):
        cache.ensure_product_bds("MKE_L")


@pytest.mark.unit
def test_ensure_product_bds_rejects_unknown_product(tmp_path, monkeypatch):
    monkeypatch.setenv("MBEAMS_CACHE_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="unknown beam product"):
        cache.ensure_product_bds("NotAProduct")
