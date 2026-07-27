from pathlib import Path
from typing import Annotated, Literal, NewType

import typer
from hip_cargo import StimelaMeta, parse_upath, stimela_cab, stimela_output

Directory = NewType("Directory", Path)
File = NewType("File", Path)


@stimela_cab(
    name="mdv_to_xradio",
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
            parser=parse_upath,
            help="input MdV beam npz file",
        ),
        StimelaMeta(
            metavar="NPZ_PATH",
        ),
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
            parser=parse_upath,
            help="output xradio zarr dataset",
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
    Converts raw MdV beam npz to an xradio-compatible zarr image
    """
    if backend == "native" or backend == "auto":
        try:
            # Pre-flight must_exist for remote URIs before dispatching.
            from hip_cargo.utils.runner import preflight_remote_must_exist  # noqa: E402

            preflight_remote_must_exist(
                mdv_to_xradio,
                dict(
                    npz_path=npz_path,
                    antenna=antenna,
                    jones=jones,
                    part=part,
                    output_var=output_var,
                    chunks_freq=chunks_freq,
                    chunks_x=chunks_x,
                    chunks_y=chunks_y,
                    compress=compress,
                    output=output,
                ),
            )

            # Lazy import the core implementation
            from meerkat_beams.core.mdv_to_xradio import mdv_to_xradio as mdv_to_xradio_core  # noqa: E402

            # Call the core function with all parameters
            mdv_to_xradio_core(
                npz_path,
                output,
                antenna=antenna,
                jones=jones,
                part=part,
                output_var=output_var,
                chunks_freq=chunks_freq,
                chunks_x=chunks_x,
                chunks_y=chunks_y,
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
        mdv_to_xradio,
        dict(
            npz_path=npz_path,
            antenna=antenna,
            jones=jones,
            part=part,
            output_var=output_var,
            chunks_freq=chunks_freq,
            chunks_x=chunks_x,
            chunks_y=chunks_y,
            compress=compress,
            output=output,
        ),
        image=image,
        backend=backend,
        always_pull_images=always_pull_images,
    )
