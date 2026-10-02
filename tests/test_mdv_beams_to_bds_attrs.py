"""Provenance attrs carried from an MdV mean-beam zarr onto the BDS."""

import numpy as np
import pytest
import xarray

from meerkat_beams.core.mdv_beams_to_bds import PROVENANCE_ATTRS, mdv_beams_to_bds

N_XY = 9
N_FREQ = 3


def _write_mean_beam_zarr(path, **attrs):
    """Minimal MdV-shaped mean-beam zarr: BEAM (corr, chan, l_beam, m_beam)."""
    degs = (np.arange(N_XY) - N_XY // 2) * 0.1
    chans = np.linspace(1.0e9, 1.2e9, N_FREQ)
    beam = np.zeros((4, N_FREQ, N_XY, N_XY), dtype=np.complex64)
    beam[0] = 1.0  # HH
    beam[3] = 1.0  # VV
    ds = xarray.Dataset(
        {"BEAM": xarray.DataArray(beam, dims=("corr", "chan", "l_beam", "m_beam"))},
        coords={"corr": ["XX", "XY", "YX", "YY"], "chan": chans, "l_beam": degs, "m_beam": degs},
    )
    ds.attrs.update(attrs)
    ds.to_zarr(str(path), mode="w")
    return path


@pytest.mark.unit
def test_provenance_attrs_are_copied(tmp_path):
    inp = _write_mean_beam_zarr(
        tmp_path / "in.zarr",
        telescope="MeerKAT Extension",
        antenna="eavg",
        band="L",
        source_file="beam_eavg_L.fits",
        source_doc="SSA-0004B-0002 Rev 01",
        pol="linear",
    )
    out = tmp_path / "out.bds.zarr"
    mdv_beams_to_bds(str(inp), str(out))
    bds = xarray.open_zarr(str(out))
    assert bds.attrs["telescope"] == "MeerKAT Extension"
    assert bds.attrs["antenna"] == "eavg"
    assert bds.attrs["band"] == "L"
    assert bds.attrs["source_file"] == "beam_eavg_L.fits"
    assert bds.attrs["source_doc"] == "SSA-0004B-0002 Rev 01"


@pytest.mark.unit
def test_non_provenance_attrs_are_not_copied(tmp_path):
    """Only the named keys travel; the BDS does not inherit everything."""
    inp = _write_mean_beam_zarr(tmp_path / "in.zarr", telescope="MeerKAT", pol="linear")
    out = tmp_path / "out.bds.zarr"
    mdv_beams_to_bds(str(inp), str(out))
    bds = xarray.open_zarr(str(out))
    assert "pol" not in bds.attrs


@pytest.mark.unit
def test_missing_provenance_attrs_are_skipped(tmp_path):
    """A legacy input carrying none of them must still convert cleanly."""
    inp = _write_mean_beam_zarr(tmp_path / "in.zarr")
    out = tmp_path / "out.bds.zarr"
    mdv_beams_to_bds(str(inp), str(out))
    bds = xarray.open_zarr(str(out))
    for key in PROVENANCE_ATTRS:
        assert key not in bds.attrs
    assert "nstokes" in bds.data_vars  # conversion itself unaffected


@pytest.mark.unit
def test_npz_input_carries_no_provenance(tmp_path):
    """The .npz path has no attrs to copy and must not invent any."""
    degs = (np.arange(N_XY) - N_XY // 2) * 0.1
    beam = np.zeros((4, 2, N_FREQ, N_XY, N_XY), dtype=np.complex64)
    beam[0] = 1.0
    beam[3] = 1.0
    npz = tmp_path / "in.npz"
    np.savez(npz, beam=beam, margin_deg=degs, freq_MHz=np.linspace(1000.0, 1200.0, N_FREQ))
    out = tmp_path / "out.bds.zarr"
    mdv_beams_to_bds(str(npz), str(out))
    bds = xarray.open_zarr(str(out))
    for key in PROVENANCE_ATTRS:
        assert key not in bds.attrs
