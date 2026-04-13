def test_import():
    import meerkat_beams

    assert hasattr(meerkat_beams, "__version__")


def test_version_is_string():
    from meerkat_beams import __version__

    assert isinstance(__version__, str)
