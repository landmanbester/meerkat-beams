from pathlib import Path
from typing import Annotated, Literal, NewType

import typer
from hip_cargo import StimelaMeta, parse_upath, stimela_cab, stimela_output

Directory = NewType("Directory", Path)
File = NewType("File", Path)


@stimela_cab(
    name="mdv_beams_to_bds",
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
            parser=parse_upath,
            help="input MdV beams npz file (see https://doi.org/10.48479/wdb0-h061)",
        ),
        StimelaMeta(
            metavar="MDV_FILE",
        ),
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
            parser=parse_upath,
            help="output beam dataset (BDS) file",
        ),
        StimelaMeta(
            mkdir=False,
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
    Converts MdV-format primary beams to a beam dataset (BDS)
    """
    if backend == "native" or backend == "auto":
        try:
            # Pre-flight must_exist for remote URIs before dispatching.
            from hip_cargo.utils.runner import preflight_remote_must_exist  # noqa: E402

            preflight_remote_must_exist(
                mdv_beams_to_bds,
                dict(
                    mdv_beams=mdv_beams,
                    compress=compress,
                    bds=bds,
                ),
            )

            # Lazy import the core implementation
            from meerkat_beams.core.mdv_beams_to_bds import mdv_beams_to_bds as mdv_beams_to_bds_core  # noqa: E402

            # Call the core function with all parameters
            mdv_beams_to_bds_core(
                mdv_beams,
                bds,
                compress=compress,
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
        mdv_beams_to_bds,
        dict(
            mdv_beams=mdv_beams,
            compress=compress,
            bds=bds,
        ),
        image=image,
        backend=backend,
        always_pull_images=always_pull_images,
    )
