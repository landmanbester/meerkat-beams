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

Directory = NewType("Directory", Path)


@stimela_cab(
    name="bds_to_xradio",
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
            parser=parse_upath,
            help="input beam dataset (.bds.zarr)",
        ),
        StimelaMeta(
            metavar="BDS_PATH",
        ),
    ],
    image_path: Annotated[
        Directory,
        typer.Option(
            ...,
            parser=parse_upath,
            help="image/dataset for WCS and time info",
        ),
        StimelaMeta(
            metavar="IMAGE_PATH",
        ),
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
    Renders a beam dataset (BDS) to an xradio-compatible zarr image
    """
    if backend == "native" or backend == "auto":
        try:
            # Pre-flight must_exist for remote URIs before dispatching.
            from hip_cargo.utils.runner import preflight_remote_must_exist  # noqa: E402

            preflight_remote_must_exist(
                bds_to_xradio,
                dict(
                    bds_path=bds_path,
                    image_path=image_path,
                    output_var=output_var,
                    pixel_stepping=pixel_stepping,
                    time_stepping=time_stepping,
                    num_freq=num_freq,
                    chunks_time=chunks_time,
                    chunks_freq=chunks_freq,
                    chunks_x=chunks_x,
                    chunks_y=chunks_y,
                    elements=elements,
                    output_pol=output_pol,
                    beam_type=beam_type,
                    compress=compress,
                    output=output,
                ),
            )

            # Lazy import the core implementation
            from meerkat_beams.core.bds_to_xradio import bds_to_xradio as bds_to_xradio_core  # noqa: E402

            # Call the core function with all parameters
            bds_to_xradio_core(
                bds_path,
                image_path,
                output,
                output_var=output_var,
                pixel_stepping=pixel_stepping,
                time_stepping=time_stepping,
                num_freq=num_freq,
                chunks_time=chunks_time,
                chunks_freq=chunks_freq,
                chunks_x=chunks_x,
                chunks_y=chunks_y,
                elements=elements,
                output_pol=output_pol,
                beam_type=beam_type,
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
        bds_to_xradio,
        dict(
            bds_path=bds_path,
            image_path=image_path,
            output_var=output_var,
            pixel_stepping=pixel_stepping,
            time_stepping=time_stepping,
            num_freq=num_freq,
            chunks_time=chunks_time,
            chunks_freq=chunks_freq,
            chunks_x=chunks_x,
            chunks_y=chunks_y,
            elements=elements,
            output_pol=output_pol,
            beam_type=beam_type,
            compress=compress,
            output=output,
        ),
        image=image,
        backend=backend,
        always_pull_images=always_pull_images,
    )
