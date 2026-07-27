"""Hermetic unit tests for core/mdv_to_xradio.py.

Synthetic NPZ in tmp_path; no network IO.
"""

import numpy as np
import pytest
import xarray

from meerkat_beams.core.mdv_to_xradio import mdv_to_xradio


def _make_synthetic_npz(tmp_path):
    """Write a tiny MdV-shaped NPZ. Returns the path."""
    n_pol, n_ant, n_freq, n_y, n_x = 4, 2, 3, 7, 7
    # complex64 with both real and imag content so part dispatch is non-trivial.
    beam = (
        np.linspace(1.0, 2.0, n_pol * n_ant * n_freq * n_y * n_x, dtype=np.float32).reshape(
            n_pol, n_ant, n_freq, n_y, n_x
        )
        + 1j
        * np.linspace(0.5, 1.5, n_pol * n_ant * n_freq * n_y * n_x, dtype=np.float32).reshape(
            n_pol, n_ant, n_freq, n_y, n_x
        )
    ).astype(np.complex64)

    path = tmp_path / "synth.npz"
    np.savez(
        str(path),
        beam=beam,
        freq_MHz=np.linspace(1000.0, 1200.0, n_freq),
        margin_deg=np.linspace(-0.5, 0.5, n_x),
        pols=np.array([b"HH", b"HV", b"VH", b"VV"]),
        antnames=np.array([b"ant0", b"array_average"]),
    )
    return path


@pytest.mark.unit
@pytest.mark.parametrize("part", ["real", "imag", "abs", "phase"])
def test_mdv_to_xradio_part_dispatch(part, tmp_path):
    npz = _make_synthetic_npz(tmp_path)
    out = tmp_path / f"{part}.zarr"
    mdv_to_xradio(str(npz), str(out), antenna=-1, jones="HH", part=part)
    ds = xarray.open_zarr(str(out))
    assert ds["SKY"].dims == ("time", "frequency", "polarization", "l", "m")
    assert ds.attrs.get("component") == part


@pytest.mark.unit
@pytest.mark.parametrize("jones", ["HH", "HV", "VH", "VV"])
def test_mdv_to_xradio_jones_selection(jones, tmp_path):
    npz = _make_synthetic_npz(tmp_path)
    out = tmp_path / f"{jones}.zarr"
    mdv_to_xradio(str(npz), str(out), antenna=-1, jones=jones, part="real")
    ds = xarray.open_zarr(str(out))
    assert list(ds.coords["polarization"].values) == [jones]
    assert ds.attrs.get("jones_element") == jones


@pytest.mark.unit
def test_mdv_to_xradio_antenna_zero_selects_first_antenna(tmp_path):
    npz = _make_synthetic_npz(tmp_path)
    out = tmp_path / "ant0.zarr"
    mdv_to_xradio(str(npz), str(out), antenna=0, jones="HH", part="real")
    ds = xarray.open_zarr(str(out))
    assert ds.attrs.get("antenna") == "ant0"


@pytest.mark.unit
def test_mdv_to_xradio_invalid_part_raises(tmp_path):
    npz = _make_synthetic_npz(tmp_path)
    with pytest.raises(ValueError, match="Unknown part"):
        mdv_to_xradio(str(npz), str(tmp_path / "bad.zarr"), part="bogus")


@pytest.mark.unit
def test_mdv_to_xradio_compress_true(tmp_path):
    import zarr

    npz = _make_synthetic_npz(tmp_path)
    out = tmp_path / "compressed.zarr"
    mdv_to_xradio(str(npz), str(out), antenna=-1, jones="HH", part="real", compress=True)
    z = zarr.open(str(out), mode="r")
    assert z["SKY"].compressor is not None
