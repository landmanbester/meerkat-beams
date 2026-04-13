from pathlib import Path
from typing import Annotated, NewType

import typer
from hip_cargo import stimela_cab, stimela_output

Directory = NewType("Directory", Path)
File = NewType("File", Path)


@stimela_cab(
    name="mdv-to-xradio",
    info="Converts raw MdV beam npz to an xradio-compatible zarr image",
)
@stimela_output(
    dtype="Directory",
    name="output",
    info="output xradio zarr dataset",
    policies={"positional": True},
    mkdir=False,
)
def mdv_to_xradio(
    npz_path: Annotated[
        File,
        typer.Option(
            ...,
            parser=Path,
            help="input MdV beam npz file",
        ),
        {
            "stimela": {
                "metavar": "NPZ_PATH",
            },
        },
    ],
    antenna: Annotated[
        int,
        typer.Option(
            help="antenna index (-1 = array_average)",
        ),
    ] = -1,
    jones: Annotated[
        str,
        typer.Option(
            help="Jones element (HH, HV, VH, VV)",
        ),
    ] = "HH",
    part: Annotated[
        str,
        typer.Option(
            help="component to render (real, imag, abs, phase)",
        ),
    ] = "real",
    output_var: Annotated[
        str,
        typer.Option(
            help="data variable name",
        ),
    ] = "SKY",
    chunks_freq: Annotated[
        int,
        typer.Option(
            help="zarr chunk size along frequency axis",
        ),
    ] = 64,
    chunks_x: Annotated[
        int,
        typer.Option(
            help="zarr chunk size along l axis",
        ),
    ] = 128,
    chunks_y: Annotated[
        int,
        typer.Option(
            help="zarr chunk size along m axis",
        ),
    ] = 128,
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
    Converts raw MdV beam npz to an xradio-compatible zarr image
    """
    # Lazy import the core implementation
    from meerkat_beams.core.xradio_util import mdv_to_xradio as mdv_to_xradio_core

    mdv_to_xradio_core(
        str(npz_path),
        str(output),
        antenna=antenna,
        jones=jones,
        part=part,
        output_var=output_var,
        chunks_freq=chunks_freq,
        chunks_x=chunks_x,
        chunks_y=chunks_y,
        compress=compress,
    )
