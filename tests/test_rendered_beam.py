"""
Test consistency between a rendered beam zarr dataset and get_time_variable_beamgain().

Renders the beam to a zarr dataset via bds_to_xradio, then picks an off-centre
l,m pixel and compares the time/frequency slice from the zarr with the result of
get_time_variable_beamgain() at the same sky coordinate.

Requires beam data files specified via environment variables
MBEAMS_BDS_PATH and MBEAMS_IMAGE_PATH.
"""

import os
import shutil
import tempfile

import numpy as np
import pytest
from astropy.time import Time

BDS_PATH = os.environ.get("MBEAMS_BDS_PATH")
IMAGE_PATH = os.environ.get("MBEAMS_IMAGE_PATH")

needs_data = pytest.mark.skipif(
    BDS_PATH is None or IMAGE_PATH is None,
    reason="MBEAMS_BDS_PATH and MBEAMS_IMAGE_PATH env vars not set",
)


@needs_data
@pytest.mark.integration
@pytest.mark.slow
def test_rendered_vs_beamgain():
    """Test that rendered beam zarr matches get_time_variable_beamgain()."""
    import xarray

    from meerkat_beams.core.bds_to_xradio import bds_to_xradio
    from meerkat_beams.utils import BeamWizard

    bw = BeamWizard(BDS_PATH, IMAGE_PATH)

    # Pick off-centre pixel
    l_index = len(bw.l_grid) // 4
    m_index = len(bw.m_grid) // 4

    output = tempfile.mkdtemp(suffix=".zarr")
    try:
        # Render beam to zarr
        bds_to_xradio(
            bds_path=BDS_PATH,
            image_path=IMAGE_PATH,
            output=output,
            output_var="SKY",
            pixel_stepping=4,
            time_stepping=4,
            num_freq=3,
            chunks_time=1,
            elements=["II"],
            beam_type="nstokes",
        )

        # Extract rendered beam slice at test pixel
        ds = xarray.open_zarr(output)
        rendered = ds["SKY"].isel(polarization=0, l=l_index, m=m_index).values
        zarr_times = Time(ds.coords["time"].values, format="mjd")
        zarr_freq = ds.coords["frequency"].values

        # Compute sky coordinate from l,m pixel
        srcpos = bw.wcs.pixel_to_world(l_index, m_index)

        # Get beam gain at same position
        beamgain = bw.get_time_variable_beamgain(
            coord=srcpos, times=zarr_times, freq=zarr_freq, var="nstokes", i="I", j="I"
        )
        beamgain = beamgain.T  # (nfreq, ntime) -> (ntime, nfreq)

        # Compare
        full_rel_diff = np.abs(rendered - beamgain) / ((np.abs(rendered) + np.abs(beamgain)) / 2 + 1e-30) * 100
        max_rel_diff = full_rel_diff.max()

        assert max_rel_diff < 1.0, f"Max relative difference {max_rel_diff:.3f}% exceeds 1% threshold"

    finally:
        shutil.rmtree(output, ignore_errors=True)
