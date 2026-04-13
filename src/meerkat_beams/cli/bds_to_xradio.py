from pathlib import Path
from typing import Annotated, NewType

import typer
from hip_cargo import ListStr, parse_list_str, stimela_cab, stimela_output

Directory = NewType("Directory", Path)


@stimela_cab(
    name="bds-to-xradio",
    info="Renders a beam dataset (BDS) to an xradio-compatible zarr image",
)
@stimela_output(
    dtype="Directory",
    name="output",
    info="output xradio zarr dataset",
    policies={"positional": True},
    mkdir=False,
)
def bds_to_xradio(
    bds_path: Annotated[
        Directory,
        typer.Option(
            ...,
            parser=Path,
            help="input beam dataset (.bds.zarr)",
        ),
        {
            "stimela": {
                "metavar": "BDS_PATH",
            },
        },
    ],
    image_path: Annotated[
        Directory,
        typer.Option(
            ...,
            parser=Path,
            help="image/dataset for WCS and time info",
        ),
        {
            "stimela": {
                "metavar": "IMAGE_PATH",
            },
        },
    ],
    output_var: Annotated[
        str,
        typer.Option(
            help="name of the output data variable",
        ),
    ] = "SKY",
    pixel_stepping: Annotated[
        int,
        typer.Option(
            help="compute every Nth pixel, interpolate back",
        ),
    ] = 4,
    time_stepping: Annotated[
        int,
        typer.Option(
            help="use every Nth timeslot",
        ),
    ] = 1,
    num_freq: Annotated[
        int | None,
        typer.Option(
            help="number of frequency channels (default uses beam dataset freqs)",
        ),
    ] = None,
    chunks_time: Annotated[
        int,
        typer.Option(
            help="zarr chunk size along time axis",
        ),
    ] = 1,
    chunks_freq: Annotated[
        int | None,
        typer.Option(
            help="zarr chunk size along frequency axis",
        ),
    ] = None,
    chunks_x: Annotated[
        int,
        typer.Option(
            help="zarr chunk size along l axis",
        ),
    ] = 256,
    chunks_y: Annotated[
        int,
        typer.Option(
            help="zarr chunk size along m axis",
        ),
    ] = 256,
    elements: Annotated[
        ListStr | None,
        typer.Option(
            parser=parse_list_str,
            help="Jones/Mueller elements to render (e.g. II, QQ for Stokes; XX, YY for Jones)",
        ),
    ] = None,
    output_pol: Annotated[
        ListStr | None,
        typer.Option(
            parser=parse_list_str,
            help="output polarization labels for each element (default: auto)",
        ),
    ] = None,
    beam_type: Annotated[
        str,
        typer.Option(
            help="beam variable to use (nstokes, stokes, njones, jones)",
        ),
    ] = "nstokes",
    compress: Annotated[
        bool,
        typer.Option(
            help="apply Delta+Blosc compression to zarr output",
        ),
    ] = False,
    output: Annotated[
        Directory | None,
        typer.Option(
            parser=Path,
            help="output xradio zarr dataset",
        ),
        {
            "stimela": {
                "metavar": "OUTPUT",
                "mkdir": False,
            },
        },
    ] = None,
):
    """
    Renders a beam dataset (BDS) to an xradio-compatible zarr image
    """
    # Lazy import the core implementation
    from meerkat_beams.core.xradio_util import bds_to_xradio as bds_to_xradio_core

    # Convert ListStr to list
    elements_list = elements.split(",") if isinstance(elements, str) else (list(elements) if elements else [])
    output_pol_list = output_pol.split(",") if isinstance(output_pol, str) else (list(output_pol) if output_pol else [])

    bds_to_xradio_core(
        str(bds_path),
        str(image_path),
        str(output),
        output_var=output_var,
        pixel_stepping=pixel_stepping,
        time_stepping=time_stepping,
        num_freq=num_freq,
        chunks_time=chunks_time,
        chunks_freq=chunks_freq,
        chunks_x=chunks_x,
        chunks_y=chunks_y,
        elements=elements_list,
        output_pol=output_pol_list,
        beam_type=beam_type,
        compress=compress,
    )
