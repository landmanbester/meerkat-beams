"""Hermetic unit tests for core/mdv_to_xradio.py.

Synthetic NPZ in tmp_path; no network IO.
"""

import pytest
import xarray

from meerkat_beams.core.mdv_to_xradio import mdv_to_xradio
from tests._synthetic import build_synthetic_npz


@pytest.mark.unit
@pytest.mark.parametrize("part", ["real", "imag", "abs", "phase"])
def test_mdv_to_xradio_part_dispatch(part, tmp_path):
    npz = build_synthetic_npz(tmp_path)
    out = tmp_path / f"{part}.zarr"
    mdv_to_xradio(str(npz), str(out), antenna=-1, jones="HH", part=part)
    ds = xarray.open_zarr(str(out))
    assert ds["SKY"].dims == ("time", "frequency", "polarization", "l", "m")
    assert ds.attrs.get("component") == part


@pytest.mark.unit
@pytest.mark.parametrize("jones", ["HH", "HV", "VH", "VV"])
def test_mdv_to_xradio_jones_selection(jones, tmp_path):
    npz = build_synthetic_npz(tmp_path)
    out = tmp_path / f"{jones}.zarr"
    mdv_to_xradio(str(npz), str(out), antenna=-1, jones=jones, part="real")
    ds = xarray.open_zarr(str(out))
    assert list(ds.coords["polarization"].values) == [jones]
    assert ds.attrs.get("jones_element") == jones


@pytest.mark.unit
def test_mdv_to_xradio_antenna_zero_selects_first_antenna(tmp_path):
    npz = build_synthetic_npz(tmp_path)
    out = tmp_path / "ant0.zarr"
    mdv_to_xradio(str(npz), str(out), antenna=0, jones="HH", part="real")
    ds = xarray.open_zarr(str(out))
    assert ds.attrs.get("antenna") == "ant0"


@pytest.mark.unit
def test_mdv_to_xradio_invalid_part_raises(tmp_path):
    npz = build_synthetic_npz(tmp_path)
    with pytest.raises(ValueError, match="Unknown part"):
        mdv_to_xradio(str(npz), str(tmp_path / "bad.zarr"), part="bogus")


@pytest.mark.unit
def test_mdv_to_xradio_compress_true(tmp_path):
    import json

    npz = build_synthetic_npz(tmp_path)
    out = tmp_path / "compressed.zarr"
    mdv_to_xradio(str(npz), str(out), antenna=-1, jones="HH", part="real", compress=True)
    # On-disk v2 metadata, not zarr's array API (its codec attributes differ
    # between zarr-python 2 and 3).
    meta = json.loads((out / "SKY" / ".zarray").read_text())
    assert meta["compressor"]["id"] == "blosc"
