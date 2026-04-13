from pathlib import Path
from typing import Annotated, NewType

import typer
from hip_cargo import stimela_cab, stimela_output

Directory = NewType("Directory", Path)
File = NewType("File", Path)


@stimela_cab(
    name="mdv-beams-to-bds",
    info="Converts MdV-format primary beams to a beam dataset (BDS)",
)
@stimela_output(
    dtype="Directory",
    name="bds",
    info="output beam dataset (BDS) file",
    policies={"positional": True},
    mkdir=False,
)
def mdv_beams_to_bds(
    mdv_beams: Annotated[
        File,
        typer.Option(
            ...,
            parser=Path,
            help="input MdV beams npz file (see https://doi.org/10.48479/wdb0-h061)",
        ),
        {
            "stimela": {
                "metavar": "MDV_FILE",
            },
        },
    ],
    compress: Annotated[
        bool,
        typer.Option(
            help="apply Delta+Blosc compression to zarr output",
        ),
    ] = False,
    bds: Annotated[
        Directory | None,
        typer.Option(
            parser=Path,
            help="output beam dataset (BDS) file",
        ),
        {
            "stimela": {
                "metavar": "BDS_FILE",
                "mkdir": False,
            },
        },
    ] = None,
):
    """
    Converts MdV-format primary beams to a beam dataset (BDS)
    """
    # Lazy import the core implementation
    from meerkat_beams.core.beams import mdv_beams_to_bds as mdv_beams_to_bds_core

    mdv_beams_to_bds_core(
        str(mdv_beams),
        str(bds),
        compress=compress,
    )
