"""Run each mbeams command through the real CLI with --backend native (issue #38).

The CLI hands core functions UPath objects; these pin that every command copes
with them end to end, rather than only through its Python core."""

import subprocess
import sys
from pathlib import Path

import pytest
import xarray
from astropy.io import fits

from tests._synthetic import build_mean_beam_zarr, build_synthetic_bds, build_synthetic_image_zarr, build_synthetic_npz

pytestmark = pytest.mark.unit


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-c", "from meerkat_beams.cli import app; app()", *args, "--backend", "native"],
        capture_output=True,
        text=True,
    )


def assert_ok(result, out):
    assert result.returncode == 0, result.stdout + result.stderr
    assert out.exists(), result.stdout + result.stderr


def test_mdv_beams_to_bds_from_npz(tmp_path):
    out = tmp_path / "npz.bds.zarr"
    assert_ok(run_cli("mdv-beams-to-bds", "--mdv-beams", str(build_synthetic_npz(tmp_path)), "--bds", str(out)), out)
    assert "nstokes" in xarray.open_zarr(out)


def test_mdv_beams_to_bds_from_zarr(tmp_path):
    src = build_mean_beam_zarr(tmp_path / "mean.zarr")
    out = tmp_path / "z.bds.zarr"
    assert_ok(run_cli("mdv-beams-to-bds", "--mdv-beams", str(src), "--bds", str(out), "--compress"), out)
    assert "nstokes" in xarray.open_zarr(out)


def test_bds_to_xradio(tmp_path):
    bds = build_synthetic_bds(tmp_path / "in.bds.zarr")
    img = build_synthetic_image_zarr(tmp_path / "img.zarr")
    out = tmp_path / "rendered.zarr"
    args = ("--bds-path", str(bds), "--image-path", str(img), "--output", str(out), "--pixel-stepping", "1")
    assert_ok(run_cli("bds-to-xradio", *args), out)
    assert xarray.open_zarr(out)["SKY"].sizes["time"] == 3


def test_download_mdv_beams_to_dest(tmp_path):
    src = tmp_path / "MeerKAT_L_band_primary_beam.npz"
    src.write_bytes(b"not really a beam")
    out = tmp_path / "fetched.npz"
    result = run_cli("download-mdv-beams", "--source", src.as_uri(), "--dest", str(out))
    assert_ok(result, out)
    assert out.read_bytes() == b"not really a beam"
    assert "deprecated" in (result.stdout + result.stderr).lower()


def test_mdv_to_xradio(tmp_path):
    out = tmp_path / "mx.zarr"
    result = run_cli("mdv-to-xradio", "--npz-path", str(build_synthetic_npz(tmp_path)), "--output", str(out))
    assert_ok(result, out)
    assert "deprecated" in (result.stdout + result.stderr).lower()


def test_bds_to_fits(tmp_path):
    bds = build_synthetic_bds(tmp_path / "in.bds.zarr")
    prefix = tmp_path / "beam"
    args = ("--bds-path", str(bds), "--prefix", str(prefix), "--pixels", "32", "--diameter", "2")
    result = run_cli("bds-to-fits", *args)
    out = Path(str(prefix) + "_re.fits")
    assert_ok(result, out)
    assert Path(str(prefix) + "_im.fits").exists()
    with fits.open(out) as h:
        assert h[0].header["CTYPE1"] == "px"
        assert h[0].data.shape == (4, 2, 2, 32, 32)
