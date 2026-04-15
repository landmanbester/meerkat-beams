"""
Test consistency between different beam computation methods.

Tests that get_time_variable_beamgain() and get_rotation_averaged_beam()
produce consistent results. Requires beam data files specified via
environment variables BDS_PATH and IMAGE_PATH.
"""

import os

import numpy as np
import pytest
from astropy.coordinates import SkyCoord

BDS_PATH = os.environ.get("MBEAMS_BDS_PATH")
IMAGE_PATH = os.environ.get("MBEAMS_IMAGE_PATH")

needs_data = pytest.mark.skipif(
    BDS_PATH is None or IMAGE_PATH is None,
    reason="MBEAMS_BDS_PATH and MBEAMS_IMAGE_PATH env vars not set",
)


def _make_beam_wizard():
    from meerkat_beams.utils import BeamWizard

    return BeamWizard(BDS_PATH, IMAGE_PATH)


def time_variable_vs_rotation_averaged(
    beam_wizard,
    srcpos: SkyCoord,
    times=None,
    loc=None,
    freq=None,
    num_freq: int = 1,
    spi=None,
    time_stepping: int = 4,
    var: str = "nstokes",
    i: str = "I",
    j: str = "I",
):
    """
    Compare get_time_variable_beamgain() and get_rotation_averaged_beam()
    at a single sky position. Returns dict with results.
    """
    if loc is None:
        loc = beam_wizard.default_location
    if times is None:
        times = beam_wizard.times
    if time_stepping > 1:
        times = times[::time_stepping]

    # Method 1: time-variable beam gain
    time_variable_beam = beam_wizard.get_time_variable_beamgain(
        coord=srcpos, times=times, loc=loc, freq=freq, num_freq=num_freq, spi=spi, var=var, i=i, j=j
    )

    if spi is not None:
        tv_mean = np.array([np.mean(time_variable_beam)])
    else:
        tv_mean = np.mean(time_variable_beam, axis=1)

    # Method 2: rotation-averaged beam at source's l/m
    angle = beam_wizard.centre.position_angle(srcpos)
    sep = beam_wizard.centre.separation(srcpos)
    l_src = sep.deg * np.sin(angle.rad)
    m_src = sep.deg * np.cos(angle.rad)

    ra_beam, ra_var = beam_wizard.get_rotation_averaged_beam(
        l=np.array([l_src]),
        m=np.array([m_src]),
        times=times,
        loc=loc,
        freq=freq,
        num_freq=num_freq,
        spi=spi,
        time_stepping=1,
        var=var,
        i=i,
        j=j,
    )

    if spi is not None:
        ra_mean = np.array([ra_beam[0, 0]])
    else:
        if ra_beam.ndim == 3:
            ra_mean = ra_beam[:, 0, 0]
        else:
            ra_mean = np.array([ra_beam[0, 0]])

    rel_diff = np.abs(tv_mean - ra_mean) / ((tv_mean + ra_mean) / 2) * 100
    return {
        "time_variable_mean": tv_mean,
        "rotation_averaged": ra_mean,
        "relative_difference": rel_diff,
        "consistent": bool(np.all(rel_diff < 1.0)),
    }


@needs_data
@pytest.mark.integration
@pytest.mark.slow
def test_beam_consistency_at_center():
    """Test consistency at field center."""
    bw = _make_beam_wizard()
    srcpos = bw.centre
    results = time_variable_vs_rotation_averaged(bw, srcpos, time_stepping=4, num_freq=1)
    assert results["consistent"], (
        f"Inconsistent beam results: max relative difference {np.max(results['relative_difference']):.3f}%"
    )


@needs_data
@pytest.mark.integration
@pytest.mark.slow
def test_beam_consistency_multi_freq():
    """Test consistency with multiple frequencies."""
    bw = _make_beam_wizard()
    srcpos = bw.centre
    results = time_variable_vs_rotation_averaged(bw, srcpos, time_stepping=4, num_freq=3)
    assert results["consistent"], (
        f"Inconsistent beam results: max relative difference {np.max(results['relative_difference']):.3f}%"
    )
