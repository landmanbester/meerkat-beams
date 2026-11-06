"""Core implementation for bds-to-xradio command."""

from typing import List, Optional

from meerkat_beams.utils import ZARR_COMPRESSOR, BeamWizard, enrich_bds_xradio, resolve_elements


def bds_to_xradio(
    bds_path: str,
    image_path: str,
    output: str,
    output_var: str = "SKY",
    pixel_stepping: int = 4,
    time_stepping: int = 1,
    num_freq: Optional[int] = None,
    chunks_time: int = 1,
    chunks_freq: Optional[int] = None,
    chunks_x: int = 256,
    chunks_y: int = 256,
    elements: Optional[List[str]] = None,
    beam_type: str = "nstokes",
    output_pol: Optional[List[str]] = None,
    compress: bool = False,
):
    """
    Render a beam dataset (BDS) to an xradio-compatible zarr image.

    Args:
        bds_path: Path to beam dataset (.bds.zarr)
        image_path: Path to image/dataset for WCS and time info
        output: Output path for the zarr dataset
        output_var: Name of the output data variable (default: SKY)
        pixel_stepping: Compute every Nth pixel, interpolate back (default 4)
        time_stepping: Use every Nth timeslot (default 1)
        num_freq: Number of frequency channels (None = use beam dataset freqs)
        chunks_time, chunks_freq, chunks_x, chunks_y: Zarr chunk sizes
        elements: List of Jones or Mueller elements to render, e.g.
          "II", "QQ", "XX", "YY" etc.
        output_pol: corresponding list of output polarization labels for each element,
          default is chosen automatically
        beam_type: Beam variable ('nstokes', 'stokes', 'mueller', 'nmueller', 'njones', 'jones')
        compress: Apply lossless Blosc compression to zarr output (default: False)

    Returns:
        Path to the output zarr dataset.
    """
    # Validate beam_type/elements (and resolve pol labels) before opening files.
    ij_list, output_pol = resolve_elements(beam_type, elements, output_pol)

    bds_path, image_path, output = str(bds_path), str(image_path), str(output)  # the CLI passes UPaths (#38)
    bw = BeamWizard(bds_path, image_path)

    bw.log.info(f"Rendering {beam_type} elements {ij_list} using pol labels {output_pol}")
    bw.log.info(f"Using ij_list: {ij_list}")

    bw.get_time_freq_beam(
        filename=output,
        var_name=output_var,
        dim_names=("time", "frequency", "polarization", "l", "m"),
        l=bw.l_grid,
        m=bw.m_grid,
        pixel_stepping=pixel_stepping,
        time_stepping=time_stepping,
        num_freq=num_freq,
        chunks_time=chunks_time,
        chunks_freq=chunks_freq,
        chunks_x=chunks_x,
        chunks_y=chunks_y,
        var=beam_type,
        ij_list=ij_list,
        compressor=ZARR_COMPRESSOR if compress else None,
    )

    enrich_bds_xradio(output, bw, output_var, output_pol)

    bw.log.info(f"xradio-compatible {output_var} written to {output}")
    return output
