"""Core implementation for download-mdv-beams command."""

import os.path
import sys
from typing import List, Optional
from urllib.error import HTTPError

import wget

from meerkat_beams.utils import log


def download_mdv_beams(
    source: str,
    dest: Optional[str] = None,
    base_url: List[str] = ["https://archive-gw-1.kat.ac.za/public/repository/10.48479/wdb0-h061/data/"],
    exit_on_error: Optional[int] = 1,
):
    """Downloads MdV beams from SARAO archive

    Args:
        source: Full URL, or filename (e.g. MeerKAT_U_band_primary_beam.npz), or band (e.g. U)
        dest: destination file, defaults to basename of filename
        base_url: download locations (mirrors), multiple may be given
        exit_on_error: exit code on failure, or None to raise instead
    """
    urls = []
    if "://" in source:
        urls = [source]
    elif source.endswith(".npz"):
        urls = [f"{url.rstrip('/')}/{source}" for url in base_url]
    elif source in ("L", "U", "S0", "S1", "S2", "S3", "S4"):
        urls = [f"{url.rstrip('/')}/MeerKAT_{source}_band_primary_beam.npz" for url in base_url]
    else:
        raise RuntimeError(f"unrecognized source argument: {source}")

    if not urls:
        raise RuntimeError("no download paths -- did you specify base_url?")

    if dest is None:
        dest = os.path.basename(urls[0])

    for url in urls:
        log.info(f"downloading {url} to {dest}")
        try:
            wget.download(url, out=dest)
            log.info("download complete")
            return 0
        except HTTPError as exc:
            log.warning(f"download failed: {exc}")

    # if we got here, all downloads failed
    log.error("all download atempts failed")
    if exit_on_error is not None:
        sys.exit(exit_on_error)
    else:
        raise RuntimeError("all download atempts failed")
