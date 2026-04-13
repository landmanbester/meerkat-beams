"""Core implementations for meerkat-beams."""

import logging
import sys

CONSOLE = None


def create_logger():
    """Create a console logger"""
    log = logging.getLogger("meerkat_beams")
    cfmt = logging.Formatter("%(name)s - %(asctime)s %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    log.setLevel(logging.DEBUG)
    global CONSOLE
    CONSOLE = logging.StreamHandler(sys.stdout)
    CONSOLE.setLevel(logging.INFO)
    CONSOLE.setFormatter(cfmt)
    log.addHandler(CONSOLE)
    return log


def set_console_logging_level(level: int):
    CONSOLE.setLevel(level)


log = LOGGER = create_logger()
