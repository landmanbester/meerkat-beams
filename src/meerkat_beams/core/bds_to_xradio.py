"""Core implementation for bds-to-xradio command."""

from typing import List, Optional

from meerkat_beams.utils import ZARR_COMPRESSOR, BeamWizard, enrich_bds_xradio, resolve_elements


def parse_pointing_centre(text: str):
    """Parse ``--pointing-centre``: sexagesimal with unit letters, or two decimal degrees, in ICRS.

    ``"05h21m09.9s +16d38m22s"``, ``"80.29deg 16.64deg"`` and ``"80.29 16.64"``
    are accepted. Anything else with a component lacking its unit is refused --
    ``"05:21:09.9 +16:38:22"``, ``"05 21 09.9 +16 38 22"``, and mixed forms such
    as ``"05:21:09.9 +16d38m22s"`` -- because astropy would read a unitless RA as
    degrees, a 75 degree error that would pass silently.
    """
    from astropy.coordinates import SkyCoord

    hint = 'Write it with unit letters ("05h21m09.9s +16d38m22s") or as two decimal degrees ("80.29 16.64").'
    if not any(c.isalpha() for c in text):
        tokens = text.replace(",", " ").split()
        if ":" in text or len(tokens) != 2:
            raise ValueError(f"pointing centre {text!r} has no unit letters, so its RA unit is ambiguous. {hint}")
        try:
            ra, dec = (float(t) for t in tokens)
        except ValueError:
            raise ValueError(f"pointing centre {text!r} is not two decimal numbers. {hint}") from None
        return SkyCoord(ra, dec, unit="deg", frame="icrs")
    # Unit letters present: give astropy no default unit, so a component without its
    # own ("05:21:09.9 +16d38m22s", "80.29 +16d38m22s") is refused rather than read as degrees.
    try:
        return SkyCoord(text, frame="icrs")
    except ValueError as err:
        raise ValueError(f"pointing centre {text!r}: every component needs a unit. {hint}") from err


def bds_to_xradio(
    bds_path: str,
    image_path: str,
    output: str,
    pointing_centre: Optional[str] = None,
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
        image_path: Path to image/dataset for WCS and time info. Its reference
          direction is the phase centre of the rendered l/m grid.
        pointing_centre: Where the dishes point, as a string (see
          ``parse_pointing_centre``). None (default) means the image's phase
          centre; otherwise the image grid is mapped onto the pointing
          centre's SIN plane before the beam is evaluated.
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
    pointing = parse_pointing_centre(pointing_centre) if pointing_centre is not None else None
    bw = BeamWizard(bds_path, image_path)
    if pointing is not None:
        bw.set_pointing_centre(pointing)

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
