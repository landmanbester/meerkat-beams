def test_import():
    import meerkat_beams

    assert hasattr(meerkat_beams, "__version__")


def test_version_is_string():
    from meerkat_beams import __version__

    assert isinstance(__version__, str)


def test_katbeam_bds_module_imports_without_katbeam(monkeypatch):
    """The base install must stay importable without katbeam: it is in the
    [full] extra, and 3.10 lightweight installs have neither."""
    import builtins
    import importlib
    import sys

    import pytest

    real_import = builtins.__import__

    def _no_katbeam(name, *args, **kwargs):
        if name == "katbeam" or name.startswith("katbeam."):
            raise ImportError("katbeam is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "meerkat_beams.katbeam_bds", raising=False)
    monkeypatch.setattr(builtins, "__import__", _no_katbeam)

    module = importlib.import_module("meerkat_beams.katbeam_bds")

    # Module-level lookups work; only evaluation needs katbeam.
    assert module.KATBEAM_MODEL_FOR_BAND["L"] == "MKAT-AA-L-JIM-2020"
    with pytest.raises(ImportError, match="uv sync"):
        module.require_model("MKAT-AA-L-JIM-2020")
