#!/usr/bin/env python
"""
How dense must a stored beam map be for ``beam_on_grid``?

Evaluates the exact beam on an output grid about a phase centre with
``get_rotation_averaged_beam(phase_centre=...)``, then stores pointing-centred
maps at a range of grid steps and places each with ``beam_on_grid``. Prints one
JSON line per (Stokes element, step) with the max and rms error relative to the
on-axis peak (I<-I) or to the element's own peak (Q<-I).

Needs the L-band BDS in the cache (``BeamWizard(band="L")`` downloads it once).
The table in docs/wiki/beamwizard.md ("beam_on_grid") came from the defaults.

    uv run python scripts/beam_on_grid_accuracy.py [--offset-deg 0.5] [--freq 1.284e9]
"""

import argparse
import json

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time

from meerkat_beams.utils import BeamWizard, beam_on_grid


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--band", default="L")
    ap.add_argument("--freq", type=float, default=1.284e9, help="Hz")
    ap.add_argument("--offset-deg", type=float, default=0.5, help="pointing to phase centre separation")
    ap.add_argument("--half-width-deg", type=float, default=1.6, help="output grid half width")
    ap.add_argument("--cell-deg", type=float, default=0.01, help="output grid cell")
    ap.add_argument("--nthreads", type=int, default=16)
    args = ap.parse_args()

    bw = BeamWizard(band=args.band)
    phase = SkyCoord(ra=0.0, dec=-30.0, unit="deg")
    pointing = phase.directional_offset_by(60 * u.deg, args.offset_deg * u.deg)
    times = Time("2024-01-01T20:00:00") + np.linspace(0, 4, 25) / 24.0
    kw = dict(times=times, freq=np.array([args.freq]), time_stepping=1, pixel_stepping=1, verbose=0)
    kw["nthreads"] = args.nthreads
    n_half = int(round(args.half_width_deg / args.cell_deg))
    out = (np.arange(2 * n_half + 1) - n_half) * args.cell_deg
    # stored maps must cover the output grid as seen from the pointing
    reach = args.half_width_deg + args.offset_deg + 0.3

    for stokes in ("I", "Q"):
        exact, _ = bw.get_rotation_averaged_beam(
            l=-out, m=out, pointing_centre=pointing, phase_centre=phase, i=stokes, j="I", **kw
        )
        peak = 1.0 if stokes == "I" else float(np.abs(exact).max())
        for step in (0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625):
            axis = np.arange(-reach, reach + step / 2, step)
            stored, _ = bw.get_rotation_averaged_beam(l=axis, m=axis, pointing_centre=pointing, i=stokes, j="I", **kw)
            placed = beam_on_grid(stored, axis, axis, pointing, phase, -out, out, nthreads=args.nthreads)
            err = np.abs(placed - exact) / peak
            row = dict(element=f"{stokes}<-I", step_deg=step, npix=axis.size)
            row.update(max_err=float(err.max()), rms_err=float(np.sqrt((err**2).mean())))
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
