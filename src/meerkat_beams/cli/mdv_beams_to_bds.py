from typing import Annotated, Literal

import typer
from hip_cargo import Directory, File, StimelaMeta, parse_upath, stimela_cab, stimela_output


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
            help="apply lossless Blosc compression to zarr output",
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
    params = dict(
        mdv_beams=mdv_beams,
        compress=compress,
        bds=bds,
    )

    if backend == "native" or backend == "auto":
        # Only a missing install falls back to the container, so only the
        # imports and the pre-flight are guarded; the core call is not.
        try:
            from hip_cargo.utils.runner import preflight_remote_must_exist  # noqa: E402

            from meerkat_beams.core.mdv_beams_to_bds import mdv_beams_to_bds as mdv_beams_to_bds_core  # noqa: E402

            # Pre-flight must_exist for remote URIs before dispatching.
            preflight_remote_must_exist(mdv_beams_to_bds, params)
        except ImportError as exc:
            # A failed import from this package is a bug, not a missing install.
            if backend == "native" or (exc.name or "").split(".")[0] == "meerkat_beams":
                raise
            import typer  # a parameter may be named typer

            typer.echo(f"Cannot run natively ({exc!r}); falling back to a container.", err=True)
        else:
            mdv_beams_to_bds_core(**params)
            return

    # Resolve container image from installed package metadata
    from hip_cargo.utils.config import get_container_image  # noqa: E402
    from hip_cargo.utils.runner import run_in_container  # noqa: E402

    image = get_container_image("meerkat-beams")
    if image is None:
        raise RuntimeError("No Container URL in meerkat-beams metadata.")

    run_in_container(
        mdv_beams_to_bds,
        params,
        image=image,
        backend=backend,
        always_pull_images=always_pull_images,
    )
