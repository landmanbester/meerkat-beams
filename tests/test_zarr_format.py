"""Every zarr store the package writes is zarr format 2, whichever zarr-python is installed.

zarr is unpinned so downstream applications can choose zarr-python 2 or 3. We
keep writing v2 so our output stays readable by consumers still on zarr 2, which
cannot read v3. These tests run under whichever zarr the environment resolved;
CI runs them under both.
"""

import json

import astropy.units as u
import numpy as np
import pytest
import xarray
import zarr
from astropy.time import Time
from numcodecs import Shuffle

from meerkat_beams.core.mdv_beams_to_bds import mdv_beams_to_bds
from meerkat_beams.core.mdv_to_xradio import mdv_to_xradio
from meerkat_beams.utils import (
    ZARR_COMPRESSOR,
    BeamWizard,
    enrich_bds_xradio,
    is_zarr_group,
    require_readable_zarr,
    zarr_create_array,
    zarr_open_group,
)
from tests._synthetic import DELTA, FREQS, build_synthetic_bds, build_synthetic_image, build_synthetic_npz


def assert_zarr_v2(path, compressed_vars=()):
    """Store at ``path`` is zarr v2 with consolidated metadata; ``compressed_vars`` use our codecs."""
    assert (path / ".zgroup").exists(), f"{path} has no .zgroup"
    assert (path / ".zmetadata").exists(), f"{path} metadata is not consolidated"
    assert not (path / "zarr.json").exists(), f"{path} is zarr v3"
    for v in compressed_vars:
        meta = json.loads((path / v / ".zarray").read_text())
        assert meta["zarr_format"] == 2
        assert meta["compressor"]["id"] == "blosc", f"{v}: {meta['compressor']}"
        assert meta["compressor"]["cname"] == "zstd"
        assert meta["filters"] is None, f"{v}: {meta['filters']}"


@pytest.mark.unit
@pytest.mark.parametrize("compress", [False, True])
def test_get_time_freq_beam_and_enrich_write_v2(tmp_path, compress):
    bds = build_synthetic_bds(tmp_path / "in.bds.zarr")
    img = build_synthetic_image(tmp_path / "in.fits")
    bw = BeamWizard(str(bds), str(img))
    out = tmp_path / "tf.zarr"
    bw.get_time_freq_beam(
        filename=str(out),
        var_name="SKY",
        dim_names=("time", "frequency", "polarization", "l", "m"),
        l=np.array([-DELTA, 0.0, DELTA]),
        m=np.array([-DELTA, 0.0, DELTA]),
        times=Time("2024-01-01T00:00:00") + np.linspace(0, 1, 3) * u.hour,
        freq=FREQS[:2],
        pixel_stepping=1,
        ij_list=[("I", "I")],
        var="nstokes",
        compressor=ZARR_COMPRESSOR if compress else None,
        verbose=0,
    )
    enrich_bds_xradio(str(out), bw, "SKY", ["I"])
    assert_zarr_v2(out, compressed_vars=("SKY",) if compress else ())
    # The polarization coordinate is rewritten by enrich_bds_xradio; it must stay v2 too.
    assert json.loads((out / "polarization" / ".zarray").read_text())["zarr_format"] == 2


@pytest.mark.unit
@pytest.mark.parametrize("compress", [False, True])
def test_mdv_to_xradio_writes_v2(tmp_path, compress):
    out = tmp_path / "mx.zarr"
    mdv_to_xradio(str(build_synthetic_npz(tmp_path)), str(out), part="abs", compress=compress)
    assert_zarr_v2(out, compressed_vars=("SKY",) if compress else ())


@pytest.mark.unit
@pytest.mark.parametrize("compress", [False, True])
def test_mdv_beams_to_bds_writes_v2(tmp_path, compress):
    out = tmp_path / "mb.bds.zarr"
    mdv_beams_to_bds(mdv_beams=str(build_synthetic_npz(tmp_path)), bds=str(out), compress=compress)
    beams = ("jones", "njones", "stokes", "nstokes", "mueller", "nmueller")
    assert_zarr_v2(out, compressed_vars=beams if compress else ())


@pytest.mark.unit
def test_zarr_helpers_round_trip(tmp_path):
    """zarr_open_group/zarr_create_array give a v2 store with the requested codecs."""
    path = tmp_path / "h.zarr"
    g = zarr_open_group(str(path), mode="w")
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    zarr_create_array(g, "a", data=data, chunks=(3, 2), compressor=ZARR_COMPRESSOR, filters=[Shuffle(elementsize=4)])
    zarr_create_array(g, "b", shape=(2,), dtype="f8", fill_value=None)
    np.testing.assert_array_equal(zarr_open_group(str(path), mode="r")["a"][:], data)
    meta = json.loads((path / "a" / ".zarray").read_text())
    assert meta["zarr_format"] == 2
    assert meta["chunks"] == [3, 2]
    assert meta["compressor"]["id"] == "blosc"
    assert [f["id"] for f in meta["filters"]] == ["shuffle"]
    assert (path / ".zgroup").exists() and not (path / "zarr.json").exists()
    assert json.loads((path / "b" / ".zarray").read_text())["fill_value"] is None


@pytest.mark.unit
def test_is_zarr_group_accepts_both_formats(tmp_path):
    """Inputs (mean-beam zarrs, image zarrs) may be either format; zarr 3 writes v3 by default."""
    (tmp_path / "v2").mkdir()
    (tmp_path / "v2" / ".zgroup").write_text("{}")
    (tmp_path / "v3").mkdir()
    (tmp_path / "v3" / "zarr.json").write_text("{}")
    (tmp_path / "neither").mkdir()
    assert is_zarr_group(tmp_path / "v2")
    assert is_zarr_group(str(tmp_path / "v3"))
    assert not is_zarr_group(tmp_path / "neither")
    assert not is_zarr_group(tmp_path / "missing")


@pytest.mark.unit
def test_zarr_create_array_compressor_none_vs_omitted(tmp_path):
    """compressor=None means uncompressed; omitting it means zarr's default, under zarr 2 and 3 alike."""
    g = zarr_open_group(str(tmp_path / "c.zarr"), mode="w")
    zarr_create_array(g, "none", data=np.zeros(4), compressor=None)
    zarr_create_array(g, "omitted", data=np.zeros(4))
    assert json.loads((tmp_path / "c.zarr" / "none" / ".zarray").read_text())["compressor"] is None
    assert json.loads((tmp_path / "c.zarr" / "omitted" / ".zarray").read_text())["compressor"] is not None


ZARR3 = int(zarr.__version__.split(".")[0]) >= 3


@pytest.mark.unit
@pytest.mark.skipif(not ZARR3, reason="writing a v3 store needs zarr-python 3")
def test_zarr_open_group_reads_existing_v3_store(tmp_path):
    """Only creation modes force ZARR_FORMAT; reading opens a store in the format it has."""
    path = tmp_path / "v3.zarr"
    g = zarr.open_group(str(path), mode="w", zarr_format=3)
    g.attrs["k"] = 1
    assert dict(zarr_open_group(str(path), mode="r").attrs) == {"k": 1}
    assert dict(zarr_open_group(str(path), mode="r+").attrs) == {"k": 1}
    require_readable_zarr(path)  # zarr 3 reads v3: no error


@pytest.mark.unit
@pytest.mark.skipif(ZARR3, reason="the error is for zarr-python 2 only")
def test_v3_input_under_zarr2_raises_clearly(tmp_path):
    path = tmp_path / "v3.zarr"
    path.mkdir()
    (path / "zarr.json").write_text("{}")
    with pytest.raises(ValueError, match="needs zarr-python>=3"):
        require_readable_zarr(path)
    with pytest.raises(ValueError, match="needs zarr-python>=3"):
        mdv_beams_to_bds(mdv_beams=str(path), bds=str(tmp_path / "out.bds.zarr"))


@pytest.mark.unit
def test_compressed_bds_is_lossless(tmp_path):
    """compress=True must not change a single value of any BDS variable."""
    plain, packed = tmp_path / "plain.bds.zarr", tmp_path / "packed.bds.zarr"
    npz = str(build_synthetic_npz(tmp_path))
    mdv_beams_to_bds(mdv_beams=npz, bds=str(plain), compress=False)
    mdv_beams_to_bds(mdv_beams=npz, bds=str(packed), compress=True)
    a, b = xarray.open_zarr(plain), xarray.open_zarr(packed)
    for v in ("jones", "njones", "stokes", "nstokes", "mueller", "nmueller"):
        np.testing.assert_array_equal(b[v].values, a[v].values, err_msg=v)


@pytest.mark.unit
def test_compressed_partial_edge_chunk_round_trips(tmp_path):
    """A partly-filled edge chunk with a NaN fill value must decode without NaNs.

    The 2026 products have 63 channels in chunks of 32, so the last chunk is
    padded with the fill value; a Delta filter carried that NaN into every later
    value of the chunk (issue found on #42)."""
    g = zarr_open_group(str(tmp_path / "e.zarr"), mode="w")
    data = np.linspace(0.5, 1.0, 2 * 63, dtype=np.float32).reshape(2, 63)
    zarr_create_array(g, "a", data=data, chunks=(2, 32), fill_value=np.nan, compressor=ZARR_COMPRESSOR)
    np.testing.assert_array_equal(zarr_open_group(str(tmp_path / "e.zarr"), mode="r")["a"][:], data)
