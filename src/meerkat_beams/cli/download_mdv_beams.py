from pathlib import Path
from typing import Annotated, Literal, NewType

import typer
from hip_cargo import (
    ListStr,
    StimelaMeta,
    parse_list_str,
    parse_upath,
    stimela_cab,
    stimela_output,
)

File = NewType("File", Path)


@stimela_cab(
    name="download_mdv_beams",
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
        StimelaMeta(
            metavar="BAND",
        ),
    ],
    base_url: Annotated[
        ListStr,
        typer.Option(
            parser=parse_list_str,
            help="download locations (mirrors), multiple may be given",
        ),
        StimelaMeta(
            metavar="URL",
        ),
    ] = [
        "https://ratt-public-data.s3.af-south-1.amazonaws.com/MeerKATbeams/",
        "https://archive-gw-1.kat.ac.za/public/repository/10.48479/wdb0-h061/data/",
    ],
    dest: Annotated[
        File | None,
        typer.Option(
            parser=parse_upath,
            help="destination file, default uses filename from URL",
        ),
    ] = None,
    backend: Annotated[
        Literal["auto", "native", "apptainer", "singularity", "docker", "podman"],
        typer.Option(
            help="Execution backend.",
        ),
        StimelaMeta(
            skip=True,
        ),
    ] = "auto",
    always_pull_images: Annotated[
        bool,
        typer.Option(
            help="Always pull container images, even if cached locally.",
        ),
        StimelaMeta(
            skip=True,
        ),
    ] = False,
):
    """
    Downloads MdV-format primary beams from SARAO archive
    """
    if backend == "native" or backend == "auto":
        try:
            # Pre-flight must_exist for remote URIs before dispatching.
            from hip_cargo.utils.runner import preflight_remote_must_exist  # noqa: E402

            preflight_remote_must_exist(
                download_mdv_beams,
                dict(
                    source=source,
                    base_url=base_url,
                    dest=dest,
                ),
            )

            # Lazy import the core implementation
            from meerkat_beams.core.download_mdv_beams import (
                download_mdv_beams as download_mdv_beams_core,  # noqa: E402
            )

            # Call the core function with all parameters
            download_mdv_beams_core(
                source,
                dest,
                base_url=base_url,
            )
            return
        except ImportError:
            if backend == "native":
                raise

    # Resolve container image from installed package metadata
    from hip_cargo.utils.config import get_container_image  # noqa: E402
    from hip_cargo.utils.runner import run_in_container  # noqa: E402

    image = get_container_image("meerkat-beams")
    if image is None:
        raise RuntimeError("No Container URL in meerkat-beams metadata.")

    run_in_container(
        download_mdv_beams,
        dict(
            source=source,
            base_url=base_url,
            dest=dest,
        ),
        image=image,
        backend=backend,
        always_pull_images=always_pull_images,
    )
