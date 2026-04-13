from typing import List, Optional, Union

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.units import Quantity

from meerkat_beams.core.beams import BeamWizard


def collect_beam_gain_to_source(
    bds_name,
    image_name,
    coord: Union[SkyCoord, str],
    freq: Union[float, str, List[float], List[str]],
    time: Optional[Union[str, Time]] = None,
):
    from meerkat_beams.core import log

    bw = BeamWizard(bds_name, image_name)
    if type(coord) is str:
        coord = SkyCoord(coord)
    log.info(f"computing beam gain towards {coord}")
    if time is not None:
        if type(time) is str:
            time = Time(time)
        log.info(f"explicit time specified as {time}")
    # compute freqs
    if isinstance(freq, (list, tuple)):
        freq = [Quantity(f).to_value(u.Hz) if type(f) is str else f for f in freq]
    elif isinstance(freq, str):
        freq = [Quantity(freq).to_value(u.Hz)]
    else:
        freq = [freq]
    log.info(f"{len(freq)} channels from {min(freq)} to {max(freq)}")

    xpyp, seps, angles = bw.get_source_coordinates(coord, time)
    log.info(f"coordinates are {xpyp}")
    log.info(f"distances are {seps}")
    log.info(f"angles are {angles}")

    beams = {}
    beams["I beam"] = bw.interpolate_beam(xpyp, freq, i="I", j="I")
    beams["V beam"] = bw.interpolate_beam(xpyp, freq, i="V", j="V")
    beams["I->Q leakage"] = bw.interpolate_beam(xpyp, freq, i="Q", j="I")
    beams["I->U leakage"] = bw.interpolate_beam(xpyp, freq, i="U", j="I")
    beams["I->V leakage"] = bw.interpolate_beam(xpyp, freq, i="V", j="I")

    return beams
