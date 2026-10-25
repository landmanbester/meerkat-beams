from pathlib import Path
from typing import Annotated, Literal, NewType

import typer
from hip_cargo import (
    ListFloat,
    ListStr,
    StimelaMeta,
    parse_list_float,
    parse_list_str,
    parse_upath,
    stimela_cab,
    stimela_output,
)

Directory = NewType("Directory", Path)
File = NewType("File", Path)


@stimela_cab(
    name="bds_to_fits",
    info="Renders a beam dataset (BDS) to eidos-compatible FITS beam cubes",
)
@stimela_output(
    dtype="File",
    name="prefix",
    info="output path prefix (files are written as prefix_re.fits and friends)",
    policies={"positional": True},
)
def bds_to_fits(
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
    beam_type: Annotated[
        str,
        typer.Option(
            help="beam variable to render (njones, jones, nstokes, stokes, nmueller, mueller)",
        ),
    ] = "njones",
    elements: Annotated[
        ListStr | None,
        typer.Option(
            parser=parse_list_str,
            help="matrix elements to render, e.g. "
            "XX,XY,YX,YY for Jones or II for Stokes (default is the full Jones matrix, or a single II element)",
        ),
    ] = None,
    output_eight: Annotated[
        bool,
        typer.Option(
            help="write one file per Jones element per part (eidos -o8)",
        ),
    ] = False,
    pixels: Annotated[
        int | None,
        typer.Option(
            help="pixels per side of the output grid, eidos -p (default uses the beam dataset grid)",
        ),
    ] = None,
    diameter: Annotated[
        float | None,
        typer.Option(
            help="field of view of the output grid in degrees, eidos -d (default uses the beam dataset grid)",
        ),
    ] = None,
    freqs: Annotated[
        ListFloat | None,
        typer.Option(
            parser=parse_list_float,
            help="explicit output frequencies in Hz, comma separated (default uses the beam dataset frequencies)",
        ),
        StimelaMeta(
            metavar="HZ",
        ),
    ] = None,
    num_freq: Annotated[
        int | None,
        typer.Option(
            help="number of evenly spaced output channels (default uses the beam dataset frequencies)",
        ),
    ] = None,
    freq_extrapolation: Annotated[
        Literal["raise", "scale"],
        typer.Option(
            help="what to do with frequencies outside the beam dataset range",
        ),
    ] = "raise",
    prefix: Annotated[
        File | None,
        typer.Option(
            parser=parse_upath,
            help="output path prefix (files are written as prefix_re.fits and friends)",
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
    Renders a beam dataset (BDS) to eidos-compatible FITS beam cubes
    """
    if backend == "native" or backend == "auto":
        try:
            # Pre-flight must_exist for remote URIs before dispatching.
            from hip_cargo.utils.runner import preflight_remote_must_exist  # noqa: E402

            preflight_remote_must_exist(
                bds_to_fits,
                dict(
                    bds_path=bds_path,
                    beam_type=beam_type,
                    elements=elements,
                    output_eight=output_eight,
                    pixels=pixels,
                    diameter=diameter,
                    freqs=freqs,
                    num_freq=num_freq,
                    freq_extrapolation=freq_extrapolation,
                    prefix=prefix,
                ),
            )

            # Lazy import the core implementation
            from meerkat_beams.core.bds_to_fits import bds_to_fits as bds_to_fits_core  # noqa: E402

            # Call the core function with all parameters
            bds_to_fits_core(
                bds_path,
                prefix,
                beam_type=beam_type,
                elements=elements,
                output_eight=output_eight,
                pixels=pixels,
                diameter=diameter,
                freqs=freqs,
                num_freq=num_freq,
                freq_extrapolation=freq_extrapolation,
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
        bds_to_fits,
        dict(
            bds_path=bds_path,
            beam_type=beam_type,
            elements=elements,
            output_eight=output_eight,
            pixels=pixels,
            diameter=diameter,
            freqs=freqs,
            num_freq=num_freq,
            freq_extrapolation=freq_extrapolation,
            prefix=prefix,
        ),
        image=image,
        backend=backend,
        always_pull_images=always_pull_images,
    )
