"""
On-demand download + cache of the PKS 1934-638 calibrator MS used by the
beam-orientation validation experiment.

Cache layout under ``meerkat_beams.cache.cache_root()``:

    <root>/test_ms/<MS_BASENAME>/

The GDrive ID is taken from ``tests.conftest.test_ms_gdrive_id``. The
download is treated as a tarball and unpacked under a sibling
``.partial`` directory before being promoted atomically.

Concurrent first-time downloads are not guarded. Same caveat as
``meerkat_beams.cache``: warm the cache from a single process.
"""

import os
import shutil
import sys
import tarfile
from pathlib import Path

from meerkat_beams.cache import cache_root

MS_BASENAME = "pks1934_offset.ms"  # promoted directory name; matches tarball top-level


def ms_path() -> Path:
    return cache_root() / "test_ms" / MS_BASENAME


def ensure_ms() -> Path:
    target = ms_path()
    if target.exists():
        return target

    from tests.conftest import test_ms_gdrive_id

    _download_and_extract(test_ms_gdrive_id, target)
    return target


def _download_and_extract(gdrive_id: str, target: Path) -> None:
    import gdown

    from meerkat_beams.utils import log

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)

    tarball = partial.parent / f"{target.name}.tgz"
    try:
        log.info(f"downloading calibrator MS from gdrive id {gdrive_id}")
        gdown.download(id=gdrive_id, output=str(tarball), quiet=False)

        log.info(f"extracting {tarball} into {partial}")
        extract_kwargs = {"filter": "data"} if sys.version_info >= (3, 12) else {}
        with tarfile.open(tarball, "r:gz") as tar:
            tar.extractall(path=partial, **extract_kwargs)

        # Expect a single top-level directory inside the tarball.
        entries = [p for p in partial.iterdir() if p.is_dir()]
        if len(entries) != 1:
            raise RuntimeError(f"expected exactly one top-level dir inside calibrator MS tarball, got {entries!r}")
        os.replace(entries[0], target)
    finally:
        if tarball.exists():
            tarball.unlink()
        if partial.exists():
            shutil.rmtree(partial, ignore_errors=True)
