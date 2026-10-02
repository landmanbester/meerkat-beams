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
