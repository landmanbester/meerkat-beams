"""Group assembly against the real MdV-2026 MeerKAT and MeerKAT+ beams.

Gated on MBEAMS_MK2026_BDS_PATH and MBEAMS_MKE_BDS_PATH, each pointing at a
BDS built from the corresponding mean-beam zarr. Skips silently otherwise,
like the other integration tests in this suite.
"""

import os

import numpy as np
import pytest
import xarray

from meerkat_beams.utils import _build_group_bds

MK_BDS = os.environ.get("MBEAMS_MK2026_BDS_PATH")
MKE_BDS = os.environ.get("MBEAMS_MKE_BDS_PATH")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (MK_BDS and MKE_BDS),
        reason="set MBEAMS_MK2026_BDS_PATH and MBEAMS_MKE_BDS_PATH to run group integration tests",
    ),
]


@pytest.fixture(scope="module")
def real_stores():
    return xarray.open_zarr(MK_BDS), xarray.open_zarr(MKE_BDS)


def test_real_stores_share_a_grid(real_stores):
    p, q = real_stores
    np.testing.assert_allclose(p.coords["X"].values, q.coords["X"].values, rtol=0, atol=1e-9)
    np.testing.assert_allclose(p.coords["Y"].values, q.coords["Y"].values, rtol=0, atol=1e-9)
    assert p.attrs["dx"] == q.attrs["dx"]


def test_real_telescope_attrs(real_stores):
    p, q = real_stores
    assert p.attrs["telescope"] == "MeerKAT"
    assert q.attrs["telescope"] == "MeerKAT Extension"


@pytest.mark.parametrize("group", ["MM", "MPM", "MPMP"])
def test_real_group_on_axis_identity(real_stores, group):
    p, q = real_stores
    a, b = {"MM": (p, p), "MPM": (p, q), "MPMP": (q, q)}[group]
    grp = _build_group_bds(a, b, group)
    x0, y0 = int(grp.attrs["x0"]), int(grp.attrs["y0"])
    on_axis = grp["nstokes"].values[:, :, :, y0, x0]
    mid = on_axis.shape[2] // 2
    np.testing.assert_allclose(on_axis[:, :, mid], np.eye(4), rtol=0, atol=1e-3)


def test_real_mpm_is_complex(real_stores):
    p, q = real_stores
    grp = _build_group_bds(p, q, "MPM")
    assert grp["nstokes"].dtype == np.complex64
    assert np.abs(grp["nstokes"].values.imag).max() > 1e-4


def test_real_mpm_lies_between_the_auto_groups(real_stores):
    """The cross I->I beam should track the geometric mean of the two autos."""
    p, q = real_stores
    mm = _build_group_bds(p, p, "MM")["nstokes"].values[0, 0]
    pp = _build_group_bds(q, q, "MPMP")["nstokes"].values[0, 0]
    mpm = np.abs(_build_group_bds(p, q, "MPM")["nstokes"].values[0, 0])
    expected = np.sqrt(np.clip(mm, 0, None) * np.clip(pp, 0, None))
    inner = np.s_[:, mm.shape[1] // 4 : 3 * mm.shape[1] // 4, mm.shape[2] // 4 : 3 * mm.shape[2] // 4]
    np.testing.assert_allclose(mpm[inner], expected[inner], rtol=0.35, atol=0.02)
