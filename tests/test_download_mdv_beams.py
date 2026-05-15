"""Hermetic unit tests for core/download_mdv_beams.py.

We monkeypatch wget.download so no network IO ever happens.
"""

from urllib.error import HTTPError

import pytest

from meerkat_beams.core import download_mdv_beams as dl_mod
from meerkat_beams.core.download_mdv_beams import download_mdv_beams


def _stub_wget_factory(calls, *, fail_first=False):
    """Build a stub for wget.download that records (url, dest) and optionally fails first."""
    state = {"called": 0}

    def stub(url, out):
        state["called"] += 1
        calls.append((url, out))
        if fail_first and state["called"] == 1:
            raise HTTPError(url, 404, "Not Found", hdrs=None, fp=None)

    return stub


@pytest.mark.unit
def test_download_full_url_uses_url_verbatim(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(dl_mod.wget, "download", _stub_wget_factory(calls))
    dest = str(tmp_path / "out.npz")
    rc = download_mdv_beams("https://example.org/foo.npz", dest=dest)
    assert rc == 0
    assert calls == [("https://example.org/foo.npz", dest)]


@pytest.mark.unit
def test_download_filename_resolves_via_base_url(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(dl_mod.wget, "download", _stub_wget_factory(calls))
    dest = str(tmp_path / "out.npz")
    download_mdv_beams(
        "MeerKAT_U_band_primary_beam.npz",
        dest=dest,
        base_url=["https://mirror-a/", "https://mirror-b/"],
    )
    assert calls == [("https://mirror-a/MeerKAT_U_band_primary_beam.npz", dest)]


@pytest.mark.unit
@pytest.mark.parametrize("band", ["L", "U", "S0", "S1", "S2", "S3", "S4"])
def test_download_band_code_expands_to_canonical_filename(band, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(dl_mod.wget, "download", _stub_wget_factory(calls))
    dest = str(tmp_path / "out.npz")
    download_mdv_beams(band, dest=dest, base_url=["https://mirror/"])
    assert calls == [(f"https://mirror/MeerKAT_{band}_band_primary_beam.npz", dest)]


@pytest.mark.unit
def test_download_unknown_source_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(dl_mod.wget, "download", lambda *a, **kw: None)
    with pytest.raises(RuntimeError, match="unrecognized source"):
        download_mdv_beams("garbage")


@pytest.mark.unit
def test_download_falls_back_to_second_mirror(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(dl_mod.wget, "download", _stub_wget_factory(calls, fail_first=True))
    dest = str(tmp_path / "out.npz")
    rc = download_mdv_beams(
        "MeerKAT_U_band_primary_beam.npz",
        dest=dest,
        base_url=["https://broken/", "https://working/"],
    )
    assert rc == 0
    assert len(calls) == 2
    assert calls[1][0] == "https://working/MeerKAT_U_band_primary_beam.npz"


@pytest.mark.unit
def test_download_all_mirrors_fail_exit_on_error_none_raises(tmp_path, monkeypatch):
    def always_fail(url, out):
        raise HTTPError(url, 500, "Bad", hdrs=None, fp=None)

    monkeypatch.setattr(dl_mod.wget, "download", always_fail)
    with pytest.raises(RuntimeError, match="all download"):
        download_mdv_beams(
            "U",
            dest=str(tmp_path / "out.npz"),
            base_url=["https://a/", "https://b/"],
            exit_on_error=None,
        )


@pytest.mark.unit
def test_download_all_mirrors_fail_exit_on_error_int_sysexits(tmp_path, monkeypatch):
    def always_fail(url, out):
        raise HTTPError(url, 500, "Bad", hdrs=None, fp=None)

    monkeypatch.setattr(dl_mod.wget, "download", always_fail)
    with pytest.raises(SystemExit) as exc:
        download_mdv_beams(
            "U",
            dest=str(tmp_path / "out.npz"),
            base_url=["https://a/"],
            exit_on_error=2,
        )
    assert exc.value.code == 2


@pytest.mark.unit
def test_download_default_dest_uses_url_basename(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(dl_mod.wget, "download", _stub_wget_factory(calls))
    monkeypatch.chdir(tmp_path)
    download_mdv_beams("U", base_url=["https://mirror/"])
    assert calls[0][1] == "MeerKAT_U_band_primary_beam.npz"
