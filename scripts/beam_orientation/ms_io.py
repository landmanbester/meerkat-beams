"""
Read a single-scan MeerKAT MS into a flat dense NumPy bundle.

Assumptions (per spec Section 5 step 2):
  * Single scan, single field on the calibrator.
  * Linear feeds, 4 correlations (XX, XY, YX, YY).
  * WEIGHT_SPECTRUM present.
  * Fits in memory.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# FIELD_ID, FIELD_NAME
# 0    Offset1
# 1    J1939-6342
# 2    Offset2
# 3    Offset3
# 4    Offset4


@dataclass
class MSBundle:
    vis: np.ndarray  # (Nb, Nt, Nf, 4) complex64
    weight_spectrum: np.ndarray  # (Nb, Nt, Nf, 4) float32
    flag: np.ndarray  # (Nb, Nt, Nf, 4) bool
    uvw: np.ndarray  # (Nb, Nt, 3) float64, metres
    time: np.ndarray  # (Nt,) MJD seconds
    freq: np.ndarray  # (Nf,) Hz
    phase_centre: tuple[float, float]  # (ra_rad, dec_rad)
    ant1: np.ndarray  # (Nb,) int
    ant2: np.ndarray  # (Nb,) int


def read_ms(path: str | Path) -> MSBundle:
    from daskms import xds_from_ms, xds_from_table

    path = str(path)
    # Main-table groups by DDID + FIELD_ID + SCAN; we expect exactly one group.
    main = xds_from_ms(
        path,
        columns=("CORRECTED_DATA", "WEIGHT_SPECTRUM", "FLAG", "UVW", "TIME", "ANTENNA1", "ANTENNA2", "FLAG_ROW"),
        group_cols=("DATA_DESC_ID", "FIELD_ID"),
        taql_where=("FIELD_ID == 0"),
    )
    # make sure only a single DDID is present
    assert len(main) == 1, f"expected exactly one group in main table, got {len(main)}"
    xds = main[0].compute()

    # Spectral window for the freq axis.
    spw = xds_from_table(f"{path}::SPECTRAL_WINDOW")[0].compute()
    ddid = int(xds.attrs.get("DATA_DESC_ID", 0))
    pol_id = xds_from_table(f"{path}::DATA_DESCRIPTION")[0].compute()
    spw_idx = int(pol_id.SPECTRAL_WINDOW_ID.values[ddid])
    freq = np.asarray(spw.CHAN_FREQ.values[spw_idx], dtype=float)

    # Field table for the phase centre.
    field_id = int(xds.attrs.get("FIELD_ID", 0))
    field = xds_from_table(
        f"{path}::FIELD",
        taql_where=(f"SOURCE_ID == {field_id}"),
    )[0].compute()
    phase_dir = field.PHASE_DIR.values.squeeze()
    ra_rad, dec_rad = phase_dir[0], phase_dir[1]

    # Reshape (row, chan, corr) -> (Nb, Nt, Nf, Ncorr).
    ant1 = np.asarray(xds.ANTENNA1.values)
    ant2 = np.asarray(xds.ANTENNA2.values)
    time_row = np.asarray(xds.TIME.values)

    times = np.unique(time_row)
    Nt = times.size
    pairs, inv = np.unique(np.stack([ant1, ant2], axis=1), axis=0, return_inverse=True)
    Nb = pairs.shape[0]
    Nf = freq.size

    def _reshape(col, fill):
        Ncorr = col.shape[-1] if col.ndim == 3 else 1
        out = np.full((Nb, Nt, Nf, Ncorr), fill, dtype=col.dtype)
        # Row-by-row scatter. Single scan so this is small.
        time_idx = np.searchsorted(times, time_row)
        out[inv, time_idx] = col
        return out

    # get flag and apply FLAG_ROW + autocorrs
    flag = xds.FLAG.values
    flag_row = xds.FLAG_ROW.values | (ant1 == ant2)

    print(flag.sum() / flag.size, flag_row.sum() / flag_row.size)
    flag = flag | flag_row[:, None, None]

    vis = _reshape(np.asarray(xds.CORRECTED_DATA.values), 0.0 + 0.0j)
    ws = _reshape(np.asarray(xds.WEIGHT_SPECTRUM.values), 0.0)
    flag = _reshape(np.asarray(flag), True)

    uvw = np.zeros((Nb, Nt, 3), dtype=float)
    uvw_row = np.asarray(xds.UVW.values)
    time_idx = np.searchsorted(times, time_row)
    uvw[inv, time_idx] = uvw_row

    return MSBundle(
        vis=vis,
        weight_spectrum=ws,
        flag=flag,
        uvw=uvw,
        time=times,
        freq=freq,
        phase_centre=(ra_rad, dec_rad),
        ant1=pairs[:, 0],
        ant2=pairs[:, 1],
    )
