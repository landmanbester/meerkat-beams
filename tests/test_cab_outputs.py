"""Pin the bds-to-fits cab output contract (PR #46 review).

The cab declares ``prefix`` as ``dtype: str`` — not ``File`` — plus two
*implicit* File outputs (``<prefix>_re.fits`` / ``<prefix>_im.fits``):

* ``prefix`` as ``File`` lied (a prefix is not a file) and made Stimela's
  ``skip_if_outputs`` unsatisfiable; as ``str`` it is no longer path-typed.
* The implicit outputs carry the real file names so downstream steps can
  reference them and skip checks can see them. They are NOT function
  parameters (hip-cargo excludes implicit outputs from the signature), so
  the container-dispatch mount for the output parent comes from rendering
  those templates, not from the ``prefix`` param.

These tests pin both halves: the declared contract, and the mount behaviour
a ``str`` prefix depends on.
"""

from pathlib import Path

import pytest
from hip_cargo.utils.runner import _resolve_mounts

from meerkat_beams.cli.bds_to_fits import bds_to_fits


def _outputs_by_name():
    return {o["name"]: o for o in bds_to_fits.__stimela_outputs__}


@pytest.mark.unit
def test_prefix_is_str_and_not_a_file_output():
    outputs = _outputs_by_name()
    assert outputs["prefix"]["dtype"] == "str"
    # A prefix is not a file: no path typing, no parse_upath, no skip trap.
    assert outputs["prefix"]["dtype"] != "File"


@pytest.mark.unit
def test_implicit_hypercube_outputs_are_declared():
    outputs = _outputs_by_name()
    assert outputs["re-fits"]["implicit"] == "{prefix}_re.fits"
    assert outputs["im-fits"]["implicit"] == "{prefix}_im.fits"
    assert outputs["re-fits"]["dtype"] == "File"
    assert outputs["im-fits"]["dtype"] == "File"
    # Implicit outputs must not become function parameters: the core signature
    # is (bds_path, prefix, ...) and must not grow re-fits/im-fits kwargs.
    assert "re_fits" not in bds_to_fits.__code__.co_varnames
    assert "im_fits" not in bds_to_fits.__code__.co_varnames


@pytest.mark.unit
def test_str_prefix_mounts_output_parent_read_write(tmp_path):
    """Container dispatch must still mount the output directory rw.

    ``prefix`` is no longer path-typed, so the mount has to come from the
    implicit output templates. Missing parent -> nearest existing ancestor,
    matching the old File-typed behaviour.
    """
    bds = tmp_path / "in.bds.zarr"
    prefix = tmp_path / "out" / "beam"  # parent does not exist yet

    mounts = _resolve_mounts(bds_to_fits, {"bds_path": bds, "prefix": str(prefix)})

    ancestor = _nearest_existing(prefix.parent)
    assert mounts.get(str(ancestor)) is True, f"expected rw mount of {ancestor}, got {mounts}"


@pytest.mark.unit
def test_str_prefix_mounts_existing_parent_read_write(tmp_path):
    mounts = _resolve_mounts(bds_to_fits, {"bds_path": tmp_path / "in.bds.zarr", "prefix": str(tmp_path / "beam")})

    assert mounts.get(str(tmp_path)) is True, f"expected rw mount of {tmp_path}, got {mounts}"


def _nearest_existing(path: Path) -> Path:
    """Walk up to the first existing ancestor (mirrors the runner's helper)."""
    while not path.exists() and path != path.parent:
        path = path.parent
    return path
