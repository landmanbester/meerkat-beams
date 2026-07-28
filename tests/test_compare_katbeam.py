"""
Unit tests for scripts/compare_katbeam.py.

Hermetic: pure helpers only. No katbeam import at module scope, no BDS on
disk, no network, no env vars. Runs under MBEAMS_OFFLINE=1.
"""

import compare_katbeam as ck
import numpy as np
import pytest

from meerkat_beams.cache import SUPPORTED_BANDS


@pytest.mark.unit
def test_katbeam_model_covers_every_supported_band():
    """Every band the cache can produce must map to a katbeam model."""
    missing = [b for b in SUPPORTED_BANDS if b not in ck.KATBEAM_MODEL_FOR_BAND]
    assert not missing, f"bands with no katbeam model: {missing}"


@pytest.mark.unit
def test_katbeam_s_subbands_share_one_model():
    """MdV splits S into sub-bands; katbeam has a single S table."""
    s_models = {b: m for b, m in ck.KATBEAM_MODEL_FOR_BAND.items() if b.startswith("S")}
    assert len(s_models) >= 2
    assert len(set(s_models.values())) == 1


@pytest.mark.unit
def test_katbeam_model_names_are_known_to_katbeam():
    """Guard against typos, and against a katbeam too old to have a model.

    PyPI only ever released katbeam 0.1, which has no S-band model at all;
    JimBeam would fall through to treating the name as a filename and die in
    np.loadtxt. The dev/test groups pin git main for this reason.
    """
    pytest.importorskip("katbeam")
    for name in ck.KATBEAM_MODEL_FOR_BAND.values():
        ck.require_model(name)


@pytest.mark.unit
def test_require_model_error_names_the_installed_models():
    """The failure mode is an outdated katbeam, so the error must be actionable."""
    pytest.importorskip("katbeam")
    with pytest.raises(ValueError, match="not available in the installed katbeam"):
        ck.require_model("MKAT-AA-NOSUCH-JIM-2020")


@pytest.mark.unit
def test_katbeam_freq_table_is_ascending_and_in_mhz():
    pytest.importorskip("katbeam")
    table = ck.katbeam_freq_table("MKAT-AA-L-JIM-2020")
    assert table.ndim == 1
    assert np.all(np.diff(table) > 0)
    # L band, so hundreds-to-low-thousands of MHz rather than Hz.
    assert 500.0 < table[0] < 2000.0
