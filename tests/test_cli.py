"""Smoke tests for CLI commands."""

import subprocess
import sys

import pytest


def run_cli(*args):
    result = subprocess.run(
        [sys.executable, "-c", "from meerkat_beams.cli import app; app()", *args],
        capture_output=True,
        text=True,
    )
    return result


@pytest.mark.unit
def test_help():
    result = run_cli("--help")
    assert result.returncode == 0
    assert "MeerKAT beam interpolator" in result.stdout


@pytest.mark.unit
def test_download_mdv_beams_help():
    result = run_cli("download-mdv-beams", "--help")
    assert result.returncode == 0
    assert "Downloads MdV-format primary beams" in result.stdout


@pytest.mark.unit
def test_mdv_beams_to_bds_help():
    result = run_cli("mdv-beams-to-bds", "--help")
    assert result.returncode == 0
    assert "Converts MdV-format primary beams" in result.stdout


@pytest.mark.unit
def test_bds_to_xradio_help():
    result = run_cli("bds-to-xradio", "--help")
    assert result.returncode == 0
    assert "Renders a beam dataset" in result.stdout


@pytest.mark.unit
def test_mdv_to_xradio_help():
    result = run_cli("mdv-to-xradio", "--help")
    assert result.returncode == 0
    assert "Converts raw MdV beam npz" in result.stdout
