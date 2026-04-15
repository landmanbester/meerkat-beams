"""Core implementation for mdv-to-xradio command."""

import numpy as np

from meerkat_beams.utils import LOGGER, ZARR_COMPRESSOR, ZARR_FILTERS


def mdv_to_xradio(
    npz_path: str,
    output: str,
    antenna: int = -1,
    jones: str = "HH",
    part: str = "real",
    output_var: str = "SKY",
    chunks_freq: int = 64,
    chunks_x: int = 128,
    chunks_y: int = 128,
    compress: bool = False,
):
    """
    Convert a raw MdV beam npz file to an xradio-compatible zarr image.

    Args:
        npz_path: Path to MdV .npz beam file
        output: Output zarr path
        antenna: Antenna index (default -1 = array_average, the last entry)
        jones: Jones element to render ('HH', 'HV', 'VH', or 'VV')
        part: 'real', 'imag', 'abs', or 'phase'
        output_var: Data variable name (default: SKY)
        chunks_freq, chunks_x, chunks_y: Zarr chunk sizes
        compress: Apply Delta+Blosc compression to zarr output (default: False)
    """
    import xarray
    import zarr

    mdv = np.load(npz_path)
    beam = mdv["beam"]  # (4, N_ant, N_freq, N_y, N_x) complex64
    pols = [p.decode() for p in mdv["pols"]]
    antnames = [a.decode() for a in mdv["antnames"]]
    freqs = mdv["freq_MHz"] * 1e6  # Hz
    margin_deg = mdv["margin_deg"]

    # Select polarization
    pol_idx = pols.index(jones)

    # Select antenna
    ant_name = antnames[antenna]
    LOGGER.info(f"Antenna: {ant_name} (index {antenna})")
    LOGGER.info(f"Jones element: {jones}, part: {part}")

    # Extract: (N_freq, N_y, N_x)
    data = beam[pol_idx, antenna]

    if part == "real":
        data = data.real
    elif part == "imag":
        data = data.imag
    elif part == "abs":
        data = np.abs(data)
    elif part == "phase":
        data = np.angle(data)
    else:
        raise ValueError(f"Unknown part '{part}', expected real/imag/abs/phase")

    data = data.astype(np.float32)

    # Build xradio-compatible dataset
    # Dimensions: (time, frequency, polarization, l, m)
    l_rad = np.deg2rad(margin_deg)
    m_rad = np.deg2rad(margin_deg)
    pol_label = jones

    # Add singleton time dimension, value 0 (no time info in MdV beams)
    # data shape: (freq, y, x) -> transpose to (freq, x, y) so that
    # l corresponds to the x-axis and m to the y-axis:
    # (freq, x, y) -> (1, freq, 1, x, y) = (time, freq, pol, l, m)
    data = data.transpose(0, 2, 1)
    data_5d = data[np.newaxis, :, np.newaxis, :, :]

    ds = xarray.Dataset(
        {
            output_var: xarray.DataArray(
                data_5d,
                dims=["time", "frequency", "polarization", "l", "m"],
                coords={
                    "time": [0.0],
                    "frequency": freqs,
                    "polarization": [pol_label],
                    "l": l_rad,
                    "m": m_rad,
                },
            )
        }
    )

    # Attributes
    ds[output_var].attrs.update(
        {
            "image_type": f"jones_{jones}_{part}",
            "units": "dimensionless",
        }
    )
    ds.attrs["direction"] = {
        "reference": {
            "attrs": {"frame": "beam", "type": "beam_coordinates", "units": "rad"},
            "data": [0.0, 0.0],
            "dims": ["l", "m"],
        },
        "projection": "SIN",
        "projection_parameters": {
            "_dtype": "float64",
            "_type": "numpy.ndarray",
            "_value": [0.0, 0.0],
        },
        "pc": {
            "_dtype": "float64",
            "_type": "numpy.ndarray",
            "_value": [[1.0, 0.0], [0.0, 1.0]],
        },
    }
    ds.attrs["antenna"] = ant_name
    ds.attrs["jones_element"] = jones
    ds.attrs["component"] = part

    # Write with chunking and optional compression
    time_len, freq_len, pol_len, l_len, m_len = data_5d.shape
    chunk_time = min(1, time_len)
    chunk_freq = min(chunks_freq, freq_len)
    chunk_pol = min(1, pol_len)
    chunk_l = min(chunks_x, l_len)
    chunk_m = min(chunks_y, m_len)
    enc = {"chunks": (chunk_time, chunk_freq, chunk_pol, chunk_l, chunk_m)}
    if compress:
        enc["compressor"] = ZARR_COMPRESSOR
        enc["filters"] = ZARR_FILTERS
    encoding = {output_var: enc}
    ds.to_zarr(output, mode="w", encoding=encoding)
    zarr.consolidate_metadata(output)

    LOGGER.info(f"Written {output}")
    LOGGER.info(f"  Shape: {data_5d.shape} (time, frequency, polarization, l, m)")
    LOGGER.info(f"  Frequencies: {len(freqs)} channels, {freqs[0] / 1e6:.1f} to {freqs[-1] / 1e6:.1f} MHz")
    LOGGER.info(
        f"  Spatial: {len(margin_deg)}x{len(margin_deg)} pixels, {margin_deg[0]:.2f} to {margin_deg[-1]:.2f} deg"
    )

    return output
