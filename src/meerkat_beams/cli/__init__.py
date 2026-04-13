"""CLI for meerkat-beams."""

import typer

app = typer.Typer(
    name="mbeams",
    help="MeerKAT beam interpolator",
    no_args_is_help=True,
)


@app.callback()
def callback() -> None:
    """MeerKAT beam interpolator"""
    pass


# Register subcommands below. Imports go here (bottom) to avoid circular imports.
from meerkat_beams.cli.onboard import onboard  # noqa: E402

app.command(name="onboard")(onboard)

from meerkat_beams.cli.download_mdv_beams import download_mdv_beams  # noqa: E402

app.command(name="download-mdv-beams")(download_mdv_beams)

from meerkat_beams.cli.mdv_to_xradio import mdv_to_xradio  # noqa: E402

app.command(name="mdv-to-xradio")(mdv_to_xradio)

from meerkat_beams.cli.mdv_beams_to_bds import mdv_beams_to_bds  # noqa: E402

app.command(name="mdv-beams-to-bds")(mdv_beams_to_bds)

from meerkat_beams.cli.bds_to_xradio import bds_to_xradio  # noqa: E402

app.command(name="bds-to-xradio")(bds_to_xradio)


__all__ = ["app"]
