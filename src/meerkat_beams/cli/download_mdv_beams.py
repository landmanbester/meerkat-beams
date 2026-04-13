from pathlib import Path
from typing import Annotated, NewType

import typer
from hip_cargo import ListStr, parse_list_str, stimela_cab, stimela_output

File = NewType("File", Path)


@stimela_cab(
    name="download-mdv-beams",
    info="Downloads MdV-format primary beams from SARAO archive",
)
@stimela_output(
    dtype="File",
    name="dest",
    info="destination file, default uses filename from URL",
    policies={"positional": True},
)
def download_mdv_beams(
    source: Annotated[
        str,
        typer.Option(
            ...,
            help="full URL, or filename (e.g. MeerKAT_U_band_primary_beam.npz), or band (e.g. U)",
        ),
        {
            "stimela": {
                "metavar": "BAND",
            },
        },
    ],
    base_url: Annotated[
        ListStr,
        typer.Option(
            parser=parse_list_str,
            help="download locations (mirrors), multiple may be given",
        ),
        {
            "stimela": {
                "metavar": "URL",
            },
        },
    ] = "https://ratt-public-data.s3.af-south-1.amazonaws.com/MeerKATbeams/,https://archive-gw-1.kat.ac.za/public/repository/10.48479/wdb0-h061/data/",
    dest: Annotated[
        File | None,
        typer.Option(
            parser=Path,
            help="destination file, default uses filename from URL",
        ),
        {
            "stimela": {
                "metavar": "[DESTINATION]",
            },
        },
    ] = None,
):
    """
    Downloads MdV-format primary beams from SARAO archive
    """
    # Lazy import the core implementation
    from meerkat_beams.core.beams import download_mdv_beams as download_mdv_beams_core

    # Convert ListStr to list
    url_list = base_url.split(",") if isinstance(base_url, str) else list(base_url)
    download_mdv_beams_core(
        source,
        dest=str(dest) if dest is not None else None,
        base_url=url_list,
    )
