"""Unit tests for scripts/stage_group_cache.py."""

import importlib.util
from pathlib import Path

import pytest

from meerkat_beams import cache

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "stage_group_cache.py"


@pytest.fixture(scope="module")
def stage_mod():
    spec = importlib.util.spec_from_file_location("stage_group_cache", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_zarr(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / ".zgroup").write_text("{}")
    return path


@pytest.mark.unit
def test_stage_product_copies_input(stage_mod, tmp_path, monkeypatch):
    monkeypatch.setenv("MBEAMS_CACHE_DIR", str(tmp_path / "cache"))
    src = _fake_zarr(tmp_path / "src.zarr")
    stage_mod.stage_product("MKE_L", src)
    assert (cache.input_zarr_path_for_product("MKE_L") / ".zgroup").exists()


@pytest.mark.unit
def test_stage_product_rejects_unknown_product(stage_mod, tmp_path, monkeypatch):
    monkeypatch.setenv("MBEAMS_CACHE_DIR", str(tmp_path / "cache"))
    src = _fake_zarr(tmp_path / "src.zarr")
    with pytest.raises(ValueError, match="unknown beam product"):
        stage_mod.stage_product("NotAProduct", src)


@pytest.mark.unit
def test_stage_product_rejects_non_zarr(stage_mod, tmp_path, monkeypatch):
    monkeypatch.setenv("MBEAMS_CACHE_DIR", str(tmp_path / "cache"))
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(SystemExit, match="does not look like a zarr"):
        stage_mod.stage_product("MKE_L", plain)


@pytest.mark.unit
def test_force_restage_invalidates_the_built_bds(stage_mod, tmp_path, monkeypatch):
    """Re-staging a corrected input must not keep serving the old BDS.

    ensure_product_bds returns early on bds.exists(), so a --force restage that
    leaves the built BDS in place silently serves a beam derived from the input
    the user just replaced -- during exactly the pre-publication window this
    script exists for.
    """
    monkeypatch.setenv("MBEAMS_CACHE_DIR", str(tmp_path / "cache"))
    src = _fake_zarr(tmp_path / "src.zarr")
    stage_mod.stage_product("MKE_L", src)
    stale_bds = _fake_zarr(cache.bds_path_for_product("MKE_L"))
    (stale_bds / "marker").write_text("built from the old input")

    stage_mod.stage_product("MKE_L", src, force=True)

    assert not stale_bds.exists(), "a forced restage must drop the BDS built from the old input"


@pytest.mark.unit
def test_without_force_existing_input_is_left_alone(stage_mod, tmp_path, monkeypatch):
    monkeypatch.setenv("MBEAMS_CACHE_DIR", str(tmp_path / "cache"))
    src = _fake_zarr(tmp_path / "src.zarr")
    stage_mod.stage_product("MKE_L", src)
    marker = cache.input_zarr_path_for_product("MKE_L") / "marker"
    marker.write_text("original")
    stage_mod.stage_product("MKE_L", src)
    assert marker.exists(), "without --force an already-staged input must be kept"


@pytest.mark.unit
def test_force_restage_from_the_cache_entry_itself_is_refused(stage_mod, tmp_path, monkeypatch):
    """Staging a product from its own cache entry must not delete its source.

    rmtree(dest) before copytree(source, dest) destroys the only copy, and an
    unpublished product cannot be re-downloaded to recover it.
    """
    monkeypatch.setenv("MBEAMS_CACHE_DIR", str(tmp_path / "cache"))
    src = _fake_zarr(tmp_path / "src.zarr")
    stage_mod.stage_product("MKE_L", src)
    dest = cache.input_zarr_path_for_product("MKE_L")

    with pytest.raises(SystemExit, match="already the cache entry"):
        stage_mod.stage_product("MKE_L", dest, force=True)

    assert (dest / ".zgroup").exists(), "the cache entry must survive a refused restage"


@pytest.mark.unit
def test_force_restage_from_inside_the_cache_entry_is_refused(stage_mod, tmp_path, monkeypatch):
    monkeypatch.setenv("MBEAMS_CACHE_DIR", str(tmp_path / "cache"))
    src = _fake_zarr(tmp_path / "src.zarr")
    stage_mod.stage_product("MKE_L", src)
    dest = cache.input_zarr_path_for_product("MKE_L")
    nested = _fake_zarr(dest / "nested.zarr")

    with pytest.raises(SystemExit, match="already the cache entry"):
        stage_mod.stage_product("MKE_L", nested, force=True)

    assert (dest / ".zgroup").exists()


@pytest.mark.unit
def test_failed_force_restage_leaves_the_old_entry_intact(stage_mod, tmp_path, monkeypatch):
    """A copy that dies partway must not leave the cache with nothing."""
    monkeypatch.setenv("MBEAMS_CACHE_DIR", str(tmp_path / "cache"))
    src = _fake_zarr(tmp_path / "src.zarr")
    stage_mod.stage_product("MKE_L", src)
    dest = cache.input_zarr_path_for_product("MKE_L")
    (dest / "marker").write_text("the good old input")
    old_bds = _fake_zarr(cache.bds_path_for_product("MKE_L"))

    def boom(*a, **kw):
        raise OSError("no space left on device")

    monkeypatch.setattr(stage_mod.shutil, "copytree", boom)
    with pytest.raises(OSError, match="no space left"):
        stage_mod.stage_product("MKE_L", src, force=True)

    assert (dest / "marker").exists(), "the previous input must survive a failed restage"
    assert old_bds.exists(), "the previous BDS must survive a failed restage"
    assert not stage_mod.cache.input_zarr_path_for_product("MKE_L").with_name("MKE_L.zarr.partial").exists()
