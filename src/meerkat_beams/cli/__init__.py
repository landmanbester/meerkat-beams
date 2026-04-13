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

__all__ = ["app"]
