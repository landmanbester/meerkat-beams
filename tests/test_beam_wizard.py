"""
Unit tests for BeamWizard.

Hermetic: builds a synthetic BDS zarr + a minimal FITS image in tmp_path.
No external data, no env vars required.
"""

from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
import xarray
from astropy.io import fits
from astropy.time import Time
from scipy.ndimage import map_coordinates, spline_filter

from meerkat_beams.utils import BeamWizard

# ---------------------------------------------------------------------------
# Synthetic-beam parameters
# ---------------------------------------------------------------------------

N_XY = 41
I0 = N_XY // 2  # centre pixel index
DELTA = 0.05  # degrees/pixel
FREQS = np.array([1.0e9, 1.1e9, 1.2e9, 1.3e9])
SIGMA_PIX = 5.0
RA0 = 0.0
DEC0 = -30.0


def _gaussian_plane() -> np.ndarray:
    """Separable radial Gaussian, peak 1.0 at (I0, I0), replicated across freq."""
    y, x = np.indices((N_XY, N_XY), dtype=np.float64)
    r2 = (x - I0) ** 2 + (y - I0) ** 2
    plane = np.exp(-0.5 * r2 / SIGMA_PIX**2)
    return np.broadcast_to(plane, (len(FREQS), N_XY, N_XY)).astype(np.float32).copy()


def _build_bds(path: Path) -> Path:
    degs = (np.arange(N_XY) - I0) * DELTA
    gauss = _gaussian_plane()  # (nfreq, ny, nx)
    zeros = np.zeros_like(gauss)

    # Jones: [[G, 0], [0, G]]  — identity at centre
    njones = np.stack(
        [np.stack([gauss, zeros], axis=0), np.stack([zeros, gauss], axis=0)],
        axis=0,
    ).astype(np.float32)
    jones = njones.copy()

    # Stokes: diag(G, G, G, G) — identity Mueller at centre
    nstokes_arr = np.zeros((4, 4, len(FREQS), N_XY, N_XY), dtype=np.float32)
    for s in range(4):
        nstokes_arr[s, s] = gauss
    stokes_arr = nstokes_arr.copy()

    fits_header = {
        "SIMPLE": "T",
        "NAXIS1": N_XY,
        "NAXIS2": N_XY,
        "NAXIS3": len(FREQS),
        "CRPIX1": I0 + 1,
        "CRPIX2": I0 + 1,
        "CRPIX3": 1,
        "CRVAL1": 0,
        "CRVAL2": 0,
        "CRVAL3": float(FREQS[0]),
        "CDELT1": DELTA,
        "CDELT2": DELTA,
        "CDELT3": float(FREQS[1] - FREQS[0]),
        "CTYPE1": "X",
        "CTYPE2": "Y",
        "CTYPE3": "FREQ",
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        "CUNIT3": "Hz",
    }

    jcoords = dict(receptor_i=[0, 1], receptor_j=[0, 1], X=degs, Y=degs, FREQ=FREQS)
    scoords = dict(stokes_i=list("IQUV"), stokes_j=list("IQUV"), X=degs, Y=degs, FREQ=FREQS)

    xds = xarray.Dataset(
        {
            "jones": xarray.DataArray(jones, dims=("receptor_i", "receptor_j", "FREQ", "Y", "X"), coords=jcoords),
            "njones": xarray.DataArray(njones, dims=("receptor_i", "receptor_j", "FREQ", "Y", "X"), coords=jcoords),
            "stokes": xarray.DataArray(stokes_arr, dims=("stokes_i", "stokes_j", "FREQ", "Y", "X"), coords=scoords),
            "nstokes": xarray.DataArray(nstokes_arr, dims=("stokes_i", "stokes_j", "FREQ", "Y", "X"), coords=scoords),
        }
    )
    xds.attrs["fits_header"] = fits_header
    xds.attrs.update(x0=I0, y0=I0, dx=DELTA, dy=DELTA, freqs=FREQS)
    xds.to_zarr(str(path), mode="w")
    return path


def _build_image(path: Path) -> Path:
    """Minimal 2-axis FITS image with SIN-projection WCS centred at (RA0, DEC0)."""
    nx = ny = 64
    data = np.zeros((ny, nx), dtype=np.float32)
    hdr = fits.Header()
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = nx
    hdr["NAXIS2"] = ny
    hdr["CRPIX1"] = nx // 2 + 1
    hdr["CRPIX2"] = ny // 2 + 1
    hdr["CRVAL1"] = RA0
    hdr["CRVAL2"] = DEC0
    hdr["CDELT1"] = -0.01
    hdr["CDELT2"] = 0.01
    hdr["CTYPE1"] = "RA---SIN"
    hdr["CTYPE2"] = "DEC--SIN"
    hdr["CUNIT1"] = "deg"
    hdr["CUNIT2"] = "deg"
    fits.PrimaryHDU(data=data, header=hdr).writeto(str(path), overwrite=True)
    return path


@pytest.fixture(scope="module")
def bw(tmp_path_factory):
    """BeamWizard over a synthetic BDS + FITS image."""
    tmp = tmp_path_factory.mktemp("bw")
    _build_bds(tmp / "synthetic.bds.zarr")
    _build_image(tmp / "synthetic.fits")
    return BeamWizard(str(tmp / "synthetic.bds.zarr"), str(tmp / "synthetic.fits"))


@pytest.fixture
def times():
    """5 samples spanning one hour — enough to exercise parallactic-angle rotation."""
    return Time("2024-01-01T00:00:00") + np.linspace(0, 1, 5) * u.hour


# ---------------------------------------------------------------------------
# _get_prefilter + interpolate_beam
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_prefilter_is_cached(bw):
    bw._prefilters.clear()
    a = bw._get_prefilter("nstokes", "I", "I")
    b = bw._get_prefilter("nstokes", "I", "I")
    assert a is b
    assert len(bw._prefilters) == 1
    c = bw._get_prefilter("nstokes", "Q", "Q")
    assert c is not a
    assert len(bw._prefilters) == 2


@pytest.mark.unit
def test_on_axis_normalised_beam_is_one(bw):
    """At the BDS centre, the normalised beam equals 1 across all frequencies."""
    xpyp = np.array([[float(bw.bds.attrs["x0"])], [float(bw.bds.attrs["y0"])]])
    vals = bw.interpolate_beam(xpyp, FREQS, var="nstokes", i="I", j="I")
    assert vals.shape == (len(FREQS), 1)
    np.testing.assert_allclose(vals[:, 0], 1.0, atol=1e-5)


@pytest.mark.unit
def test_integer_pixel_matches_raw(bw):
    """At integer (x, y) pixels, interpolation returns the raw array values."""
    xi = np.array([I0 - 2, I0, I0 + 3], dtype=float)
    yi = np.array([I0 + 1, I0, I0 - 4], dtype=float)
    xpyp = np.array([xi, yi])
    vals = bw.interpolate_beam(xpyp, FREQS[:1], var="nstokes", i="I", j="I")
    raw = bw.bds["nstokes"].sel(stokes_i="I", stokes_j="I").values
    expected = raw[0, yi.astype(int), xi.astype(int)]
    np.testing.assert_allclose(vals[0], expected, atol=1e-5)


@pytest.mark.unit
def test_subpixel_matches_direct_scipy(bw):
    """Sub-pixel interpolation matches a direct scipy call with prefilter=False.

    Pins the prefilter=False fix: flipping it back to True would double-filter
    and fail this test.
    """
    xp = np.array([I0 + 0.37, I0 - 1.8, I0 + 2.42])
    yp = np.array([I0 - 0.91, I0 + 0.5, I0 - 3.13])
    xpyp = np.array([xp, yp])

    freq_subset = FREQS[:2]
    vals = bw.interpolate_beam(xpyp, freq_subset, var="nstokes", i="I", j="I")

    raw = bw.bds["nstokes"].sel(stokes_i="I", stokes_j="I").values
    filtered = spline_filter(raw)
    freq_idx = bw.freq_to_index(freq_subset)
    fx = np.meshgrid(freq_idx, xp, indexing="ij")
    fy = np.meshgrid(freq_idx, yp, indexing="ij")
    coords = np.vstack([fy] + [fx[1:]])
    expected = map_coordinates(filtered, coords, prefilter=False)

    np.testing.assert_allclose(vals, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.unit
def test_out_of_range_xy_returns_zero(bw):
    """Coords outside the X/Y cube return 0 (pins current cval=0.0 behaviour)."""
    xpyp = np.array([[-5.0, N_XY + 10.0], [-5.0, N_XY + 10.0]])
    vals = bw.interpolate_beam(xpyp, FREQS[:1], var="nstokes", i="I", j="I")
    np.testing.assert_array_equal(vals[0], 0.0)


@pytest.mark.unit
def test_out_of_range_freq_raises(bw):
    """A frequency outside the BDS freq range raises ValueError."""
    xpyp = np.array([[float(I0)], [float(I0)]])
    bad_freq = np.array([FREQS[0] - 1.0e8])
    with pytest.raises(ValueError):
        bw.interpolate_beam(xpyp, bad_freq, var="nstokes", i="I", j="I")


# ---------------------------------------------------------------------------
# get_source_coordinates, get_time_variable_beamgain,
# get_rotation_averaged_beam, get_time_freq_beam
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_source_coordinates_at_field_centre(bw, times):
    """Source at the field centre → xpyp equals (x0, y0) and sep = 0."""
    xpyp, seps, _ = bw.get_source_coordinates(bw.centre, times=times)
    assert xpyp.shape == (2, len(times))
    np.testing.assert_allclose(xpyp[0], bw.bds.attrs["x0"], atol=1e-6)
    np.testing.assert_allclose(xpyp[1], bw.bds.attrs["y0"], atol=1e-6)
    np.testing.assert_allclose(seps.deg, 0.0, atol=1e-6)


@pytest.mark.unit
def test_time_variable_beamgain_at_centre(bw, times):
    """Normalised beam gain at field centre is 1 at every time/frequency."""
    vals = bw.get_time_variable_beamgain(bw.centre, times=times, var="nstokes", i="I", j="I")
    assert vals.shape == (len(FREQS), len(times))
    np.testing.assert_allclose(vals, 1.0, atol=1e-5)


@pytest.mark.unit
def test_rotation_averaged_beam_on_axis(bw, times):
    """At (l=0, m=0) the rotated track collapses to the centre pixel → mean=1, var=0."""
    l = np.array([0.0])
    m = np.array([0.0])
    mean, var = bw.get_rotation_averaged_beam(
        l=l,
        m=m,
        times=times,
        freq=FREQS,
        time_stepping=1,
        pixel_stepping=1,
        var="nstokes",
        i="I",
        j="I",
        verbose=0,
    )
    # len(freq) > 1 and spi is None → (NFREQ, NL, NM)
    assert mean.shape == (len(FREQS), 1, 1)
    assert var.shape == (len(FREQS), 1, 1)
    np.testing.assert_allclose(mean, 1.0, atol=1e-5)
    np.testing.assert_allclose(var, 0.0, atol=1e-10)


@pytest.mark.unit
def test_rotation_averaged_beam_spi_collapses_freq(bw, times):
    """With spi set, the frequency axis collapses and output is 2D."""
    l = np.array([0.0])
    m = np.array([0.0])
    mean, var = bw.get_rotation_averaged_beam(
        l=l,
        m=m,
        times=times,
        freq=FREQS,
        spi=-0.7,
        time_stepping=1,
        pixel_stepping=1,
        var="nstokes",
        i="I",
        j="I",
        verbose=0,
    )
    assert mean.shape == (1, 1)
    assert var.shape == (1, 1)
    np.testing.assert_allclose(mean, 1.0, atol=1e-5)


@pytest.mark.unit
def test_time_freq_beam_writes_zarr(bw, times, tmp_path):
    """get_time_freq_beam writes a zarr with declared dims/shape/coords."""
    out = tmp_path / "tfbeam.zarr"
    l_grid = np.array([-DELTA, 0.0, DELTA])
    m_grid = np.array([-DELTA, 0.0, DELTA])
    bw.get_time_freq_beam(
        filename=str(out),
        var_name="BEAM",
        dim_names=("time", "frequency", "polarization", "l", "m"),
        l=l_grid,
        m=m_grid,
        times=times,
        freq=FREQS[:2],
        pixel_stepping=1,
        time_stepping=1,
        ij_list=[("I", "I")],
        var="nstokes",
        verbose=0,
    )
    # mask_and_scale=False: the zarr is written with default fill_value=0, so
    # xarray would otherwise mask genuine 0 values (e.g. l=0.0) as NaN.
    ds = xarray.open_zarr(str(out), mask_and_scale=False)
    assert ds["BEAM"].dims == ("time", "frequency", "polarization", "l", "m")
    assert ds["BEAM"].shape == (len(times), 2, 1, 3, 3)
    np.testing.assert_allclose(ds.coords["frequency"].values, FREQS[:2])
    np.testing.assert_allclose(ds.coords["l"].values, l_grid)
    np.testing.assert_allclose(ds.coords["m"].values, m_grid)
    # at (l=0, m=0) the rotated track collapses to the centre pixel → 1.0 everywhere
    np.testing.assert_allclose(ds["BEAM"].isel(l=1, m=1, polarization=0).values, 1.0, atol=1e-5)


@pytest.mark.unit
def test_beam_wizard_requires_one_of_bds_or_band(tmp_path):
    with pytest.raises(ValueError, match="exactly one of bds_name or band"):
        BeamWizard(image_name=str(tmp_path / "x.fits"))


@pytest.mark.unit
def test_beam_wizard_rejects_both_bds_and_band(tmp_path):
    with pytest.raises(ValueError, match="exactly one of bds_name or band"):
        BeamWizard(bds_name="some.bds.zarr", image_name=str(tmp_path / "x.fits"), band="U")


@pytest.mark.unit
def test_beam_wizard_requires_image_name():
    with pytest.raises(ValueError, match="image_name is required"):
        BeamWizard(bds_name="some.bds.zarr")


@pytest.mark.unit
def test_beam_wizard_band_routes_through_cache(tmp_path, monkeypatch):
    """band='U' must call ensure_band_bds and feed the result to the existing init."""
    from meerkat_beams import cache

    monkeypatch.setenv("MBEAMS_CACHE_DIR", str(tmp_path))

    fake_bds = tmp_path / "fake.bds.zarr"
    fake_image = tmp_path / "synthetic.fits"
    _build_bds(fake_bds)
    _build_image(fake_image)

    calls = []

    def stub_ensure(band):
        calls.append(band)
        return str(fake_bds)

    monkeypatch.setattr(cache, "ensure_band_bds", stub_ensure)
    bw = BeamWizard(image_name=str(fake_image), band="U")
    assert calls == ["U"]
    assert bw.bds is not None


@pytest.mark.integration
def test_beam_wizard_band_l_end_to_end(tmp_path):
    """End-to-end: BeamWizard(band='L', ...) opens a real cached BDS.

    Skipped when MBEAMS_OFFLINE=1 (air-gapped CI). Reuses whatever the
    cache already contains; populates it via ensure_band_bds if needed.
    """
    import os

    if os.environ.get("MBEAMS_OFFLINE") == "1":
        pytest.skip("MBEAMS_OFFLINE=1 set")

    fits_path = tmp_path / "synthetic.fits"
    _build_image(fits_path)
    bw = BeamWizard(image_name=str(fits_path), band="L")
    assert "FREQ" in bw.bds.coords
    assert bw.bds.attrs["dx"] > 0
