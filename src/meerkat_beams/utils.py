"""
Shared utilities for meerkat-beams.

Contains:
- Logging setup (log, LOGGER, set_console_logging_level)
- PowerBeam dataclass
- BeamWizard class for beam interpolation
- beamplots utilities
- xradio zarr enrichment helpers
- Zarr compression defaults
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import astropy.units as u
import numpy as np
import numpy.linalg
import scipy.interpolate
import xarray
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from numcodecs import Blosc, Delta
from scipy.ndimage import map_coordinates, spline_filter

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Zarr compression defaults
# ---------------------------------------------------------------------------

ZARR_COMPRESSOR = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
ZARR_FILTERS = [Delta(dtype="float32")]

# ---------------------------------------------------------------------------
# PowerBeam dataclass
# ---------------------------------------------------------------------------


@dataclass
class PowerBeam(object):
    """Power beam info"""

    I: np.ndarray  # Stokes I beam, of shape NFREQ x NDEG x NDEG
    deg: np.ndarray  # coordinates in beam
    freq: np.ndarray  # frequencies


# ---------------------------------------------------------------------------
# BeamWizard
# ---------------------------------------------------------------------------


class BeamWizard(object):
    """Attaches to a BDS and provides various convenienece functions"""

    Eband: np.ndarray  # per-band power beam
    Emean: np.ndarray  # mean MFS beam
    band_weights: np.ndarray  # per-band weights
    freqs: np.ndarray  # band frequencies
    x0: int  # center pixel of beam
    y0: int
    delta: float  # degrees per pixel
    ra0: float  # field centre in degrees
    dec0: float

    def __init__(
        self,
        bds_name: Optional[str] = None,
        image_name: Optional[str] = None,
        *,
        band: Optional[str] = None,
    ):
        if (bds_name is None) == (band is None):
            raise ValueError("exactly one of bds_name or band must be provided")
        if image_name is None:
            raise ValueError("image_name is required")
        if band is not None:
            from meerkat_beams import cache

            bds_name = cache.ensure_band_bds(band)
        self.log = log
        log.info(f"opening BDS {bds_name}")
        self.bds = xarray.open_zarr(bds_name)
        freqs = self.bds.coords["FREQ"].values
        log.info(f"frequency range is {freqs[0] * 1e-6:.0f} to {freqs[-1] * 1e-6:.0f} MHz")
        self.index_to_freq = scipy.interpolate.interp1d(np.arange(len(freqs)), freqs)
        self.freq_to_index = scipy.interpolate.interp1d(freqs, np.arange(len(freqs)))

        if image_name.endswith(".fits"):
            log.info(f"obtaining WCS from FITS image {image_name}")
            fitshdr = fits.open(image_name)[0].header
            self.wcs = WCS(fitshdr)
            self.times = None
        elif (Path(image_name) / ".zgroup").exists():
            log.info(f"obtaining WCS from dataset {image_name}")
            ds = xarray.open_zarr(image_name)
            fitshdr = fits.Header(dict(ds.attrs["fits_header"]))
            self.wcs = WCS(fitshdr)
            self.times = Time(ds.coords["TIME"].values / (24 * 3600), format="mjd")
            log.info(f"time axis is {self.times[0].iso} to {self.times[-1].iso}")
        else:
            raise RuntimeError(f"unable to determine type of image {image_name}")
        # drop WCS axes >2
        while len(self.wcs.axis_type_names) > 2:
            log.debug(f"dropping WCS axis {self.wcs.axis_type_names[-1]}")
            self.wcs = self.wcs.dropaxis(len(self.wcs.axis_type_names) - 1)
        self.centre = self.wcs.pixel_to_world(fitshdr["CRPIX1"] - 1, fitshdr["CRPIX2"] - 1)
        log.info(f"image centre is at {self.centre}")

        # Construct default l/m grid from image pixels
        nx, ny = fitshdr["NAXIS1"], fitshdr["NAXIS2"]
        crpix1, crpix2 = fitshdr["CRPIX1"], fitshdr["CRPIX2"]
        cdelt1, cdelt2 = fitshdr["CDELT1"], fitshdr["CDELT2"]
        # l/m are offsets from center in degrees (l increases to the east, m to the north)
        self.l_grid: np.ndarray = (np.arange(nx) - (crpix1 - 1)) * cdelt1
        self.m_grid: np.ndarray = (np.arange(ny) - (crpix2 - 1)) * cdelt2
        log.info(
            f"default l/m grid: {nx}x{ny} pixels, "
            f"l=[{self.l_grid[0]:.4f}, {self.l_grid[-1]:.4f}], "
            f"m=[{self.m_grid[0]:.4f}, {self.m_grid[-1]:.4f}] deg"
        )

        # location could be made configurable
        self.default_location = EarthLocation.of_site("MeerKAT")
        log.info(f"location is MeerKAT ({self.default_location})")
        self._prefilters = {}

    def _get_prefilter(self, var: str, i: Union[str, int], j: Union[str, int], order: int = 3, verbose=1):
        # order is included in the cache key: spline_filter coefficients depend on
        # the spline order, so callers requesting different orders must not collide.
        key = var, i, j, order
        if key not in self._prefilters:
            if verbose > 0:
                self.log.debug(f"computing spline prefilter for {var}[{i},{j}] (order={order})")
            da = self.bds[var]
            # Use the variable's actual first two dims (receptor_i/j or stokes_i/j)
            sel = {da.dims[0]: i, da.dims[1]: j}
            self._prefilters[key] = spline_filter(da.sel(**sel), order=order, output=np.float32)
        return self._prefilters[key]

    def get_source_coordinates(
        self,
        srcpos: SkyCoord,
        times: Optional[Time] = None,
        loc: Optional[EarthLocation] = None,
        signs=(1, 1),
        swap=False,
    ):
        """
        Given a sky position and a list of times, derives the in-beam coordinates of the source (in beam pixels)
        """
        if loc is None:
            loc = self.default_location
        if times is None:
            if self.times is None:
                raise RuntimeError(
                    "explicit times must be supplied, since BeamWizard was constructed without observational time info"
                )
            times = self.times
        # convert positions to alt-az
        frame = AltAz(obstime=times, location=loc)
        altaz_src = srcpos.transform_to(frame)
        altaz_centre = self.centre.transform_to(frame)
        # get angle and separation of source w.r.t. centre
        angles = altaz_centre.position_angle(altaz_src)
        seps = altaz_centre.separation(altaz_src)
        # convert to pixel position
        # confused about angles, but experiments show that 0 is up and +90 is right
        x = signs[0] * seps.deg * np.sin(angles.rad)
        y = signs[1] * seps.deg * np.cos(angles.rad)
        if swap:
            x, y = y, x
        xp = x / self.bds.attrs["dx"] + self.bds.attrs["x0"]
        yp = y / self.bds.attrs["dy"] + self.bds.attrs["y0"]
        return np.array([xp, yp]), seps, angles

    def interpolate_beam(
        self,
        xpyp: np.ndarray,
        freq: Union[List[float], np.ndarray],
        var: str = "nstokes",
        i="I",
        j="I",
        order: int = 3,
    ):
        # beam is I,J,FREQ,Y,X
        freq = np.asarray(freq, dtype=float)
        bds_freqs = self.bds.coords["FREQ"].values
        fmin, fmax = float(bds_freqs.min()), float(bds_freqs.max())
        if freq.size and (freq.min() < fmin or freq.max() > fmax):
            raise ValueError(
                f"requested frequencies [{freq.min() * 1e-6:.3f}, {freq.max() * 1e-6:.3f}] MHz "
                f"fall outside the BDS frequency range [{fmin * 1e-6:.3f}, {fmax * 1e-6:.3f}] MHz"
            )
        freq = self.freq_to_index(freq)
        nfreq = len(freq)
        n_xy = xpyp.shape[1]
        freq_grid = np.broadcast_to(freq[:, None], (nfreq, n_xy))
        x_grid = np.broadcast_to(xpyp[0], (nfreq, n_xy))
        y_grid = np.broadcast_to(xpyp[1], (nfreq, n_xy))
        coords = np.array([freq_grid, y_grid, x_grid])
        # prefilter=False because _get_prefilter already applied spline_filter;
        # mode="constant", cval=0.0 makes the off-cube extrapolation policy explicit.
        return map_coordinates(
            self._get_prefilter(var, i, j, order=order),
            coords,
            order=order,
            prefilter=False,
            mode="constant",
            cval=0.0,
        )

    def _resolve_freqs(
        self, freq: Optional[np.ndarray] = None, num_freq: Optional[int] = None, spi: Optional[float] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Resolve frequency array and compute spectral weights.

        Helper method that determines which frequencies to use and optionally
        computes spectral weights based on a spectral index.

        Args:
            freq: Explicit frequency array in Hz. If None, uses beam dataset frequencies.
            num_freq: Number of linearly spaced frequencies between min and max
                      beam frequencies. Mutually exclusive with freq.
            spi: Spectral index for computing frequency weights as (freq/freq[0])^spi.
                 If None, no weights are computed.

        Returns:
            Tuple of (frequencies, weights) where:
                - frequencies: 1D array of frequencies in Hz
                - weights: 1D array of normalized spectral weights, or None if spi is None

        Raises:
            ValueError: If both freq and num_freq are specified.
        """
        if freq is not None and num_freq is not None:
            raise ValueError("freq and num_freq are mutually exclusive")
        if freq is None:
            bds_freqs = self.bds.coords["FREQ"].values
            if num_freq is not None:
                freq = np.linspace(bds_freqs[0], bds_freqs[-1], num_freq)
            else:
                freq = bds_freqs
        if spi is not None:
            norm_weights = (freq / freq[0]) ** spi
            norm_weights /= norm_weights.sum()
        else:
            norm_weights = None
        return freq, norm_weights

    def get_time_variable_beamgain(
        self,
        coord: SkyCoord,
        times: Optional[Time] = None,
        loc: Optional[EarthLocation] = None,
        freq: Optional[np.ndarray] = None,
        num_freq: Optional[int] = None,
        spi: Optional[float] = None,
        var: str = "nstokes",
        i: str = "I",
        j: str = "I",
    ) -> np.ndarray:
        """
        Compute time-variable beam gain for a source at fixed sky coordinates.

        Given a source position, computes the beam gain at that source as a function
        of time. As the parallactic angle changes, the source traces a path through
        the beam, and this method returns the beam value along that path.

        Args:
            coord: Source sky coordinate (RA/Dec)
            times: Times to sample. If None, uses times from image/dataset.
            loc: Observer location. If None, uses MeerKAT.
            freq: Explicit frequency array in Hz. If None, uses beam dataset frequencies.
            num_freq: Number of linearly spaced frequencies. Mutually exclusive with freq.
            spi: Spectral index. If provided, returns frequency-averaged beam gain
                 weighted by (freq/freq[0])^spi.
            var: Beam variable to interpolate ('nstokes', 'stokes', 'njones', 'jones')
            i, j: Stokes or Jones indices (e.g., "I", "Q", 0, 1)

        Returns:
            Beam gain values as np.ndarray:
                - If spi is None: shape (NFREQ, NTIME) - beam gain per frequency and time
                - If spi is not None: shape (NTIME,) - frequency-averaged beam gain per time

        Raises:
            RuntimeError: If times are not available and not provided.
            ValueError: If both freq and num_freq are specified.
        """
        xpyp, seps, angles = self.get_source_coordinates(coord, times=times, loc=loc)

        freq, norm_weights = self._resolve_freqs(freq, num_freq, spi)

        beam_vals = self.interpolate_beam(xpyp, freq, var=var, i=i, j=j)

        if spi is not None:
            beam_vals = (beam_vals * norm_weights[:, np.newaxis]).sum(axis=0)

        return beam_vals

    def get_rotation_averaged_beam(
        self,
        l: Optional[np.ndarray] = None,
        m: Optional[np.ndarray] = None,
        times: Optional[Time] = None,
        loc: Optional[EarthLocation] = None,
        freq: Optional[np.ndarray] = None,
        num_freq: Optional[int] = None,
        spi: Optional[float] = None,
        time_stepping: int = 4,
        pixel_stepping: int = 4,
        chunk_size: int = 1024**2,
        var: str = "nstokes",
        i: str = "I",
        j: str = "I",
        verbose: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the rotation-averaged beam at specified l/m coordinates.

        Given a grid of l/m coordinates (in degrees, relative to field center),
        computes the average beam value at each position by rotating through
        the parallactic angles corresponding to the given times.

        Args:
            l: 1D or 2D array of l coordinates in degrees (east offset from center).
               If None, uses image grid. If 1D, will be meshed with m.
            m: 1D or 2D array of m coordinates in degrees (north offset from center).
               If None, uses image grid. If 1D, will be meshed with l.
            times: Times to average over. If None, uses times from image/dataset.
            loc: Observer location. If None, uses MeerKAT.
            freq: Explicit frequency array in Hz. If None, uses beam dataset frequencies.
            num_freq: Number of linearly spaced frequencies. Mutually exclusive with freq.
            spi: Spectral index. If provided, averages over frequency with weights
                 (freq/freq[0])^spi, returning 2D spatial arrays.
            time_stepping: Use every Nth timeslot (default 4) to reduce computation
                           while maintaining representative parallactic angle coverage.
            pixel_stepping: Compute on every Nth pixel in l and m (default 4), then
                            interpolate back to the full grid. Reduces computation for
                            large images; use 1 to disable.
            chunk_size: Number of spatial pixels to process at once (default 1024^2).
                        Controls memory usage for large grids.
            var: Beam variable to interpolate ('nstokes', 'stokes', 'njones', 'jones')
            i, j: Stokes or Jones indices (e.g., "I", "Q", 0, 1)

        Returns:
            Tuple of (mean_beam, variance_beam) as np.ndarray:
                - If spi is None and len(freq) > 1: both arrays have shape (NFREQ, NL, NM)
                - If spi is not None or len(freq) == 1: both arrays have shape (NL, NM)
                Where NL and NM are the dimensions of the l/m grid, corresponding to
                the lengths of the l and m axes respectively (matching indexing='ij').

        Raises:
            RuntimeError: If times are not available and not provided.
            ValueError: If both freq and num_freq are specified.
        """
        if loc is None:
            loc = self.default_location
        if times is None:
            available_times = getattr(self, "times", None)
            if available_times is None:
                raise RuntimeError(
                    "times must be supplied, since BeamWizard was constructed without observational time info"
                )
            times = available_times
        if time_stepping > 1:
            times = times[::time_stepping]

        freq, norm_weights = self._resolve_freqs(freq, num_freq, spi)

        if l is None:
            l = self.l_grid
        if m is None:
            m = self.m_grid

        # Set up l/m grid:
        # - if both are 1D, create a meshgrid;
        # - if both are 2D, use them directly (shapes must match);
        # - otherwise, raise an error.
        if l.ndim == 1 and m.ndim == 1:
            ll, mm = np.meshgrid(l, m, indexing="ij")
        elif l.ndim == 2 and m.ndim == 2:
            if l.shape != m.shape:
                raise ValueError(
                    f"Inconsistent shapes for 2D l and m grids: l.shape={l.shape}, m.shape={m.shape}. "
                    "When both l and m are 2D, they must have the same shape."
                )
            ll, mm = l, m
        else:
            raise ValueError(
                f"Inconsistent dimensions for l and m: l.ndim={l.ndim}, m.ndim={m.ndim}. "
                "Both must be either 1D (to form a meshgrid) or 2D (pre-constructed grid with matching shapes)."
            )

        full_shape = ll.shape

        # Apply pixel stepping: subsample the grid for cheaper computation
        if pixel_stepping > 1:
            ll_compute = ll[::pixel_stepping, ::pixel_stepping]
            mm_compute = mm[::pixel_stepping, ::pixel_stepping]
        else:
            ll_compute = ll
            mm_compute = mm

        shape = ll_compute.shape
        ll_flat = ll_compute.ravel()
        mm_flat = mm_compute.ravel()

        # Compute parallactic angles at each time for the field center
        frame = AltAz(obstime=times, location=loc)
        altaz_centre = self.centre.transform_to(frame)

        # Get position angle to NCP (north celestial pole) to determine parallactic angle
        ncp = SkyCoord(ra=0 * u.deg, dec=90 * u.deg)
        altaz_ncp = ncp.transform_to(frame)
        pa = altaz_centre.position_angle(altaz_ncp)

        n_times = len(times)
        n_pixels = len(ll_flat)
        n_chunks = (n_pixels + chunk_size - 1) // chunk_size
        stepping_info = f", pixel_stepping={pixel_stepping}" if pixel_stepping > 1 else ""
        if verbose > 0:
            self.log.info(
                f"computing rotation-averaged beam over {n_times} times, "
                f"PA range {pa.min().deg:.1f} to {pa.max().deg:.1f} deg, "
                f"{len(freq)} frequency planes, {n_pixels} pixels in {n_chunks} chunks"
                f"{stepping_info}"
            )

        # Precompute the spline filter to ensure it's cached
        self._get_prefilter(var, i, j)

        # Allocate output arrays
        out_shape = (n_pixels,) if spi is not None else (len(freq), n_pixels)
        beam_sum = np.zeros(out_shape)
        beam_sum_sq = np.zeros(out_shape)

        # Process in spatial chunks to limit memory
        for chunk_idx in range(n_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, n_pixels)
            if verbose > 0:
                self.log.info(f"processing chunk {chunk_idx + 1}/{n_chunks} (pixels {chunk_start}-{chunk_end})")
            ll_chunk = ll_flat[chunk_start:chunk_end]
            mm_chunk = mm_flat[chunk_start:chunk_end]

            # Accumulate over time for this chunk
            chunk_sum = np.zeros(
                (chunk_end - chunk_start,) if spi is not None else (len(freq), chunk_end - chunk_start)
            )
            chunk_sum_sq = np.zeros_like(chunk_sum)

            for t_idx in range(n_times):
                pa_t = pa[t_idx].rad
                l_rot = mm_chunk * np.sin(pa_t) - ll_chunk * np.cos(pa_t)
                m_rot = ll_chunk * np.sin(pa_t) + mm_chunk * np.cos(pa_t)

                # Convert to beam pixel coordinates
                xp = l_rot / self.bds.attrs["dx"] + self.bds.attrs["x0"]
                yp = m_rot / self.bds.attrs["dy"] + self.bds.attrs["y0"]

                # Interpolate beam at these coordinates
                xpyp = np.array([xp, yp])
                beam_vals = self.interpolate_beam(xpyp, freq, var=var, i=i, j=j)

                # Average over frequency if spectral index is given
                if spi is not None:
                    beam_vals = (beam_vals * norm_weights[:, np.newaxis]).sum(axis=0)

                chunk_sum += beam_vals
                chunk_sum_sq += beam_vals**2

            # Store chunk results
            if spi is not None:
                beam_sum[chunk_start:chunk_end] = chunk_sum
                beam_sum_sq[chunk_start:chunk_end] = chunk_sum_sq
            else:
                beam_sum[:, chunk_start:chunk_end] = chunk_sum
                beam_sum_sq[:, chunk_start:chunk_end] = chunk_sum_sq

        # Compute mean and variance over time
        beam_mean = beam_sum / n_times
        beam_var = beam_sum_sq / n_times - beam_mean**2

        # Reshape to coarse grid
        if spi is not None or len(freq) == 1:
            beam_mean = beam_mean.reshape(shape)
            beam_var = beam_var.reshape(shape)
        else:
            beam_mean = beam_mean.reshape((len(freq),) + shape)
            beam_var = beam_var.reshape((len(freq),) + shape)

        # Interpolate back to full resolution if pixel_stepping was applied
        if pixel_stepping > 1 and shape != full_shape:
            # Fractional coarse-grid coordinates for each full-resolution pixel
            fi = np.arange(full_shape[0]) / pixel_stepping
            fj = np.arange(full_shape[1]) / pixel_stepping
            fi2d, fj2d = np.meshgrid(fi, fj, indexing="ij")
            coords = np.array([fi2d.ravel(), fj2d.ravel()])
            if beam_mean.ndim == 2:
                beam_mean = map_coordinates(beam_mean, coords, order=1, mode="nearest").reshape(full_shape)
                beam_var = map_coordinates(beam_var, coords, order=1, mode="nearest").reshape(full_shape)
            else:
                mean_full = np.empty((len(freq),) + full_shape)
                var_full = np.empty((len(freq),) + full_shape)
                for f_idx in range(len(freq)):
                    mean_full[f_idx] = map_coordinates(beam_mean[f_idx], coords, order=1, mode="nearest").reshape(
                        full_shape
                    )
                    var_full[f_idx] = map_coordinates(beam_var[f_idx], coords, order=1, mode="nearest").reshape(
                        full_shape
                    )
                beam_mean = mean_full
                beam_var = var_full

        return beam_mean, beam_var

    def get_time_freq_beam(
        self,
        filename: str,
        var_name: str,
        dim_names: Tuple[str, str, str, str, str] = ("time", "freq", "ij", "x", "y"),
        ds: Optional[xarray.Dataset] = None,
        l: Optional[np.ndarray] = None,
        m: Optional[np.ndarray] = None,
        times: Optional[Time] = None,
        loc: Optional[EarthLocation] = None,
        freq: Optional[np.ndarray] = None,
        num_freq: Optional[int] = None,
        pixel_stepping: int = 4,
        time_stepping: int = 1,
        chunks_time: int = 1,
        chunks_freq: Optional[int] = None,
        chunks_x: int = 256,
        chunks_y: int = 256,
        var: str = "nstokes",
        ij_list: Optional[List[Tuple]] = None,
        compressor=None,
        filters=None,
        verbose: int = 1,
    ):
        """
        Compute the beam per time and frequency and write to a zarr dataset.

        Similar to get_rotation_averaged_beam, but instead of averaging over time,
        writes the full (ij, time, freq, x, y) beam cube to a zarr dataset.

        Args:
            filename: Path for the output zarr store.
            var_name: Name of the beam variable in the dataset.
            dim_names: Tuple of five dimension names for the output axes
                       (time, freq, ij, x, y).
            ds: Optional existing xarray Dataset.
                If provided, its coordinates are used as defaults for l, m, times,
                freq; explicitly provided values are checked for consistency.
            l: 1D array of l coordinates in degrees. If None, uses ds coords or image grid.
            m: 1D array of m coordinates in degrees. If None, uses ds coords or image grid.
            times: Times to compute at. If None, uses ds coords or image/dataset times.
            loc: Observer location. If None, uses MeerKAT.
            freq: Explicit frequency array in Hz. If None, uses ds coords or beam
                  dataset frequencies.
            num_freq: Number of linearly spaced frequencies. Mutually exclusive with freq.
            pixel_stepping: Compute on every Nth pixel in l and m (default 4), then
                            interpolate back to the full grid.
            time_stepping: Use every Nth timeslot (default 1).
            chunks_time: Zarr chunk size along the time axis (default 1).
            chunks_freq: Zarr chunk size along the frequency axis (default: all freqs).
            chunks_x: Zarr chunk size along the x axis (default 256).
            chunks_y: Zarr chunk size along the y axis (default 256).
            var: Beam variable to interpolate ('nstokes', 'stokes', 'njones', 'jones').
            ij_list: List of (i, j) tuples specifying which matrix elements to compute.
                     Default is [("I", "I")].
            compressor: Zarr compressor (e.g. numcodecs.Blosc). If None, uses zarr default.
            filters: List of zarr filters (e.g. [numcodecs.Delta]). If None, no filters.

        """
        import zarr

        dim_time, dim_freq, dim_ij, dim_x, dim_y = dim_names

        if loc is None:
            loc = self.default_location
        if ij_list is None:
            ij_list = [("I", "I")]

        # Resolve parameters from ds or defaults
        if ds is not None:

            def _check_or_default(value, coord_name, label):
                coord = ds.coords[coord_name].values
                if value is not None:
                    if not np.allclose(value, coord):
                        raise ValueError(f"{label} inconsistent with dataset coordinate {coord_name}")
                    return value
                return coord

            l = _check_or_default(l, dim_x, "l")
            m = _check_or_default(m, dim_y, "m")
            freq = _check_or_default(freq, dim_freq, "freq")
            if times is not None:
                ds_times = Time(ds.coords[dim_time].values, format="mjd")
                if not np.allclose(times.mjd, ds_times.mjd):
                    raise ValueError("times inconsistent with dataset coordinate")
            else:
                times = Time(ds.coords[dim_time].values, format="mjd")
        else:
            if times is None:
                available_times = getattr(self, "times", None)
                if available_times is None:
                    raise RuntimeError(
                        "times must be supplied, since BeamWizard was constructed without observational time info"
                    )
                times = available_times
            if l is None:
                l = self.l_grid
            if m is None:
                m = self.m_grid
            freq, _ = self._resolve_freqs(freq, num_freq, spi=None)

        if time_stepping > 1:
            times = times[::time_stepping]

        # Create meshgrid if l and m are 1D
        if l.ndim == 1 and m.ndim == 1:
            ll, mm = np.meshgrid(l, m, indexing="ij")
        elif l.ndim == 2 and m.ndim == 2:
            if l.shape != m.shape:
                raise ValueError(f"Inconsistent shapes for l and m: l.shape={l.shape}, m.shape={m.shape}.")
            ll, mm = l, m
        else:
            raise ValueError(f"Inconsistent dimensions for l and m: l.ndim={l.ndim}, m.ndim={m.ndim}.")

        full_shape = ll.shape
        nx, ny = full_shape

        # Apply pixel stepping
        if pixel_stepping > 1:
            ll_compute = ll[::pixel_stepping, ::pixel_stepping]
            mm_compute = mm[::pixel_stepping, ::pixel_stepping]
        else:
            ll_compute = ll
            mm_compute = mm

        compute_shape = ll_compute.shape
        ll_flat = ll_compute.ravel()
        mm_flat = mm_compute.ravel()

        # Precompute upsampling coordinates
        if pixel_stepping > 1 and compute_shape != full_shape:
            fi = np.arange(full_shape[0]) / pixel_stepping
            fj = np.arange(full_shape[1]) / pixel_stepping
            fi2d, fj2d = np.meshgrid(fi, fj, indexing="ij")
            upsample_coords = np.array([fi2d.ravel(), fj2d.ravel()])
        else:
            upsample_coords = None

        # Compute parallactic angles
        frame = AltAz(obstime=times, location=loc)
        altaz_centre = self.centre.transform_to(frame)
        ncp = SkyCoord(ra=0 * u.deg, dec=90 * u.deg)
        altaz_ncp = ncp.transform_to(frame)
        pa = altaz_centre.position_angle(altaz_ncp)

        n_times = len(times)
        n_freq = len(freq)
        n_ij = len(ij_list)

        if chunks_freq is None:
            chunks_freq = n_freq

        if verbose > 0:
            self.log.info(
                f"computing time-freq beam: {n_ij} ij elements, {n_times} times, "
                f"PA range {pa.min().deg:.1f} to {pa.max().deg:.1f} deg, "
                f"{n_freq} freqs, {nx}x{ny} pixels"
                f"{f', pixel_stepping={pixel_stepping}' if pixel_stepping > 1 else ''}"
            )

        # Precompute spline filters for all ij pairs
        for ii, jj in ij_list:
            self._get_prefilter(var, ii, jj)

        # Build shape, chunks and coords in dim_names
        dim_sizes = {dim_ij: n_ij, dim_time: n_times, dim_freq: n_freq, dim_x: nx, dim_y: ny}
        dim_chunks = {dim_ij: 1, dim_time: chunks_time, dim_freq: chunks_freq, dim_x: chunks_x, dim_y: chunks_y}
        ij_labels = [f"{ii}{jj}" for ii, jj in ij_list]
        dim_coords = {
            dim_ij: ij_labels,
            dim_time: times.mjd,
            dim_freq: freq,
            dim_x: l if l.ndim == 1 else np.arange(nx),
            dim_y: m if m.ndim == 1 else np.arange(ny),
        }

        shape = tuple(dim_sizes[d] for d in dim_names)
        # Clamp chunk sizes so they do not exceed the corresponding dimension lengths
        clamped_dim_chunks = {d: min(int(dim_chunks[d]), int(dim_sizes[d])) for d in dim_names}
        chunks = tuple(clamped_dim_chunks[d] for d in dim_names)
        coords = {d: dim_coords[d] for d in dim_names}

        # Determine axis positions for ij and time dimensions in output layout
        ij_axis = list(dim_names).index(dim_ij)
        time_axis = list(dim_names).index(dim_time)

        # Create zarr store directly (no dask dependency)
        if ds is None:
            store = zarr.open(filename, mode="w")
            store.create_dataset(
                var_name,
                shape=shape,
                chunks=chunks,
                dtype="float32",
                fill_value=None,
                compressor=compressor,
                filters=filters,
            )
            store[var_name].attrs["_ARRAY_DIMENSIONS"] = list(dim_names)
            # Write coordinate arrays
            for dim_idx, dim in enumerate(dim_names):
                coord_data = np.asarray(coords[dim])
                store.create_dataset(dim, data=coord_data, overwrite=True, fill_value=None)
                store[dim].attrs["_ARRAY_DIMENSIONS"] = [dim]
            zarr.consolidate_metadata(filename)

        def compute_plane(pa_t, ii, jj):
            """Compute beam for the full spatial plane at one time/ij, with upsampling."""
            l_rot = mm_flat * np.sin(pa_t) - ll_flat * np.cos(pa_t)
            m_rot = ll_flat * np.sin(pa_t) + mm_flat * np.cos(pa_t)
            xp = l_rot / self.bds.attrs["dx"] + self.bds.attrs["x0"]
            yp = m_rot / self.bds.attrs["dy"] + self.bds.attrs["y0"]
            xpyp = np.array([xp, yp])
            beam_vals = self.interpolate_beam(xpyp, freq, var=var, i=ii, j=jj)
            beam_2d = beam_vals.reshape((n_freq,) + compute_shape)

            if upsample_coords is not None:
                beam_full = np.empty((n_freq,) + full_shape, dtype=np.float32)
                for f_idx in range(n_freq):
                    beam_full[f_idx] = map_coordinates(
                        beam_2d[f_idx], upsample_coords, order=1, mode="nearest"
                    ).reshape(full_shape)
                return beam_full

            return beam_2d.astype(np.float32)

        # Sequential computation across (time, ij); write to zarr after each plane
        done_count = 0
        total = n_times * n_ij
        zarr_arr = zarr.open(filename, mode="r+")[var_name]
        for t_idx in range(n_times):
            pa_t = pa[t_idx].rad
            for ij_idx, (ii, jj) in enumerate(ij_list):
                plane = compute_plane(pa_t, ii, jj)  # shape: (n_freq, nx, ny)
                # Build index tuple to write plane into correct position
                idx = [slice(None)] * 5
                idx[time_axis] = t_idx
                idx[ij_axis] = ij_idx
                zarr_arr[tuple(idx)] = plane
                done_count += 1
                if (done_count % max(1, total // 10) == 0 or done_count == total) and verbose > 0:
                    self.log.info(f"  written {done_count}/{total} planes")


# ---------------------------------------------------------------------------
# xradio zarr enrichment helper
# ---------------------------------------------------------------------------


def enrich_bds_xradio(zarr_path: str, bw: BeamWizard, output_var: str, polarizations: List[str]):
    """Post-process zarr store for xradio compatibility.

    - Converts l/m from degrees to radians
    - Fixes polarization labels
    - Reorders dimensions to (time, frequency, polarization, l, m)
    - Adds direction attributes matching xradio schema
    """
    import zarr

    store = zarr.open(zarr_path, mode="r+")

    # Convert l/m from degrees to radians (xradio convention)
    l_deg = store["l"][:]
    m_deg = store["m"][:]
    store["l"][:] = np.deg2rad(l_deg)
    store["m"][:] = np.deg2rad(m_deg)

    # Fix polarization labels: get_time_freq_beam writes "II", "QQ", etc.
    # xradio expects single-letter Stokes labels "I", "Q", "U", "V"
    pol_arr = store.create_dataset("polarization", data=np.array(polarizations), overwrite=True)
    pol_arr.attrs["_ARRAY_DIMENSIONS"] = ["polarization"]

    # Dimensions are already in xradio order (time, frequency, polarization, l, m)
    # from get_time_freq_beam, just ensure attrs are set correctly
    store[output_var].attrs["_ARRAY_DIMENSIONS"] = ["time", "frequency", "polarization", "l", "m"]

    # Dataset-level attributes: direction block matching reference schema
    ra0 = float(bw.centre.ra.rad)
    dec0 = float(bw.centre.dec.rad)

    existing_attrs = dict(store.attrs)
    existing_attrs["direction"] = {
        "reference": {
            "attrs": {"frame": "icrs", "type": "sky_coord", "units": "rad"},
            "data": [ra0, dec0],
            "dims": ["l", "m"],
        },
        "latpole": {
            "attrs": {"type": "quantity", "units": "rad"},
            "data": dec0,
            "dims": ["l", "m"],
        },
        "lonpole": {
            "attrs": {"type": "quantity", "units": "rad"},
            "data": float(np.pi),
            "dims": ["l", "m"],
        },
        "projection": "SIN",
        "projection_parameters": {
            "_dtype": "float64",
            "_type": "numpy.ndarray",
            "_value": [0.0, 0.0],
        },
        "pc": {
            "_dtype": "float64",
            "_type": "numpy.ndarray",
            "_value": [[1.0, 0.0], [0.0, 1.0]],
        },
    }
    store.attrs.put(existing_attrs)

    # Variable-level attributes
    store[output_var].attrs.update(
        {
            "image_type": "primary_beam",
            "units": "dimensionless",
        }
    )

    # Re-consolidate metadata so open_zarr works without consolidated=False
    zarr.consolidate_metadata(zarr_path)


# ---------------------------------------------------------------------------
# beamplots utilities
# ---------------------------------------------------------------------------


def collect_beam_gain_to_source(
    bds_name,
    image_name,
    coord: Union[SkyCoord, str],
    freq: Union[float, str, List[float], List[str]],
    time: Optional[Union[str, Time]] = None,
):
    from astropy.units import Quantity

    bw = BeamWizard(bds_name, image_name)
    if type(coord) is str:
        coord = SkyCoord(coord)
    log.info(f"computing beam gain towards {coord}")
    if time is not None:
        if type(time) is str:
            time = Time(time)
        log.info(f"explicit time specified as {time}")
    # compute freqs
    if isinstance(freq, (list, tuple)):
        freq = [Quantity(f).to_value(u.Hz) if type(f) is str else f for f in freq]
    elif isinstance(freq, str):
        freq = [Quantity(freq).to_value(u.Hz)]
    else:
        freq = [freq]
    log.info(f"{len(freq)} channels from {min(freq)} to {max(freq)}")

    xpyp, seps, angles = bw.get_source_coordinates(coord, time)
    log.info(f"coordinates are {xpyp}")
    log.info(f"distances are {seps}")
    log.info(f"angles are {angles}")

    beams = {}
    beams["I beam"] = bw.interpolate_beam(xpyp, freq, i="I", j="I")
    beams["V beam"] = bw.interpolate_beam(xpyp, freq, i="V", j="V")
    beams["I->Q leakage"] = bw.interpolate_beam(xpyp, freq, i="Q", j="I")
    beams["I->U leakage"] = bw.interpolate_beam(xpyp, freq, i="U", j="I")
    beams["I->V leakage"] = bw.interpolate_beam(xpyp, freq, i="V", j="I")

    return beams
