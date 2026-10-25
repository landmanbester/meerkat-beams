"""Core implementation for bds-to-fits command.

Writes a beam dataset (BDS) as FITS beam cubes in the layout ``../eidos``
produces (``eidos.util.write_fits`` / ``write_fits_cube``), so a consumer that
reads an eidos beam cube -- DDFacet / killMS with ``Beam.Model: FITS`` -- reads
ours the same way. What we copy and what we correct is pinned in
``docs/wiki/data-model.md``; the short version:

* FITS axis order is the numpy order reversed: ``(FREQ, p0, p1, Y, X)`` becomes
  axes 5, 4, 3, 2, 1, so axis 1 is the X (east) pixel axis and axis 3/4 carry
  the matrix (polarization) indices;
* ``CRPIX = centre0 + 1`` with the 0-based centre at ``pixels / 2`` -- the rule
  ``eidos.util._crpix`` uses (fixed in ratt-ru/eidos#1) -- or, on the native
  grid, at the BDS centre index ``x0``/``y0`` (identical for even grids);
* ``CDELT1/2`` are the pixel scale in degrees, ``CRVAL`` 0 on both sky axes,
  and the last axis is ``FREQ`` in Hz with ``CRVAL5 = freq[0]``;
* eidos swaps ``CTYPE1``/``CTYPE2`` (and ``CDELT1``/``CDELT2``) on the 5-axis
  hypercube and leaves the polarization ``CTYPEs`` blank unless the channel
  count happens to be 2 (then ``H``/``V``) or 4 (then ``Mx``/``My``). We write
  ``px``/``py`` and real polarization labels.
  World coordinates are identical either way: both sky axes share ``CRVAL 0``
  and the same ``CDELT`` on the square grids these files use.

Two layouts, both eidos's:

* hypercube (default): ``<prefix>_re.fits`` and ``<prefix>_im.fits`` (the
  imaginary part is written for Jones and Mueller variables; eidos writes it
  for Jones only, because its Mueller output is auto-correlation and real,
  while ours is a genuinely complex coherency-basis Mueller) holding every
  selected element, shaped ``(FREQ, p0, p1, Y, X)`` -- eidos's Jones
  hypercube, or one Stokes/Mueller element. eidos's own ``-S M`` writes the
  full 4x4 Mueller; this command deliberately takes a single element, so a
  hypercube's matrix axes stay 2x2 (Jones) or 1x1 (Stokes/Mueller);
* ``output_eight`` (eidos's ``-o8``): ``<prefix>_{xx,xy,yx,yy}_{re,im}.fits``,
  one ``(FREQ, Y, X)`` real cube per Jones element per part. Jones only, as in
  eidos.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from astropy.io import fits
from upath import UPath

from meerkat_beams.utils import FREQ_MATCH_ATOL_HZ, BeamWizard, log, resolve_elements

_JONES_TYPES = ("jones", "njones")
# eidos's -o8 default: every element of the 2x2 Jones matrix, row-major.
_JONES_FULL = ["XX", "XY", "YX", "YY"]
_STOKES_DEFAULT = "II"
# Which variables get an imaginary-part file. eidos always writes both parts
# for its Jones cube and only the real part for its Stokes/Mueller output, so
# this follows the variable rather than the stored dtype: a store that happens
# to carry real Jones values still gets a (zero) _im file, which keeps the file
# layout a function of --beam-type alone.
_IM_TYPES = ("jones", "njones", "mueller", "nmueller")

# Resampling runs in frequency blocks. The interpolation coordinates alone are
# three float64 arrays of (block x npix), so this budget keeps them near 24 MB
# at the cap. Same idiom as katbeam_bds.FREQ_CHUNK_ELEMENTS.
_BLOCK_ELEMENTS = 1_000_000

# eidos writes 1 MHz when a single frequency is requested -- there is no df to
# read off a one-element axis -- and so do we, so a one-channel cube is
# header-identical to eidos's.
_SINGLE_FREQ_CDELT_HZ = 1.0e6

# IAU Stokes axis codes: axis value = CRVAL + (pixel - CRPIX) * CDELT.
_STOKES_CODE = {c: i for i, c in enumerate("IQUV", start=1)}


def _target_grid(
    bds, pixels: Optional[int], diameter: Optional[float]
) -> Tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """Target axes in degrees, pixel scales, and the 1-based FITS reference pixel.

    The native grid (neither ``pixels`` nor ``diameter``) reproduces the BDS
    ``X``/``Y`` coordinate exactly and puts ``CRPIX`` on the BDS centre index
    ``x0``/``y0``, so the header agrees with the BDS's own ``fits_header``.

    Requesting either rebuilds the grid with eidos's centre rule: the 0-based
    centre is ``n / 2``, so an odd pixel count puts it *between* two pixels and
    ``CRPIX`` lands on ``x.5`` -- which is what ``eidos.util._crpix`` writes and
    what its ``test_fits_centre.py`` checks against the sub-pixel beam peak.
    """
    nx, ny = int(bds.sizes["X"]), int(bds.sizes["Y"])
    dx, dy = float(bds.attrs["dx"]), float(bds.attrs["dy"])
    x0, y0 = int(bds.attrs["x0"]), int(bds.attrs["y0"])
    if pixels is None and diameter is None:
        l = (np.arange(nx) - x0) * dx
        m = (np.arange(ny) - y0) * dy
        return l, m, dx, dy, x0 + 1.0, y0 + 1.0
    n_x = pixels if pixels is not None else nx
    n_y = pixels if pixels is not None else ny
    cdelt_x = diameter / n_x if diameter is not None else dx
    cdelt_y = diameter / n_y if diameter is not None else dy
    centre_x, centre_y = n_x / 2.0, n_y / 2.0
    l = (np.arange(n_x) - centre_x) * cdelt_x
    m = (np.arange(n_y) - centre_y) * cdelt_y
    return l, m, cdelt_x, cdelt_y, centre_x + 1.0, centre_y + 1.0


def _freq_cdelta(freq: np.ndarray) -> float:
    """CDELT for the FREQ axis, following eidos's rule, warning if it cannot be exact.

    A FITS FREQ axis is linear (``CRVAL + k * CDELT``), so a gapped channel set
    cannot be described by any single value: the mean spacing is written and a
    warning says so, rather than eidos's ``f[1] - f[0]``, which would misplace
    every plane after the first gap by the gap size.
    """
    if freq.size == 1:
        return _SINGLE_FREQ_CDELT_HZ
    span = float((freq[-1] - freq[0]) / (freq.size - 1))
    if np.allclose(np.diff(freq), span, rtol=0.0, atol=FREQ_MATCH_ATOL_HZ):
        # eidos writes the first spacing; on a uniform axis that is this value.
        return float(freq[1] - freq[0])
    log.warning(
        f"requested frequencies are not uniformly spaced ({np.diff(freq).min():.6g} to "
        f"{np.diff(freq).max():.6g} Hz): writing the mean spacing {span:.6g} Hz as CDELT, so the "
        "FREQ WCS positions are approximate. Select uniform channels for exact per-plane frequencies."
    )
    return span


def _sky_cards(hdr: fits.Header, cdelt_x: float, cdelt_y: float, crpix_x: float, crpix_y: float) -> None:
    """Sky axes: direction offsets in degrees, exactly as eidos writes them."""
    hdr["CTYPE1"] = "px"
    hdr["CRPIX1"] = float(crpix_x)
    hdr["CRVAL1"] = 0.0
    hdr["CDELT1"] = float(cdelt_x)
    hdr["CUNIT1"] = "deg"
    hdr["CTYPE2"] = "py"
    hdr["CRPIX2"] = float(crpix_y)
    hdr["CRVAL2"] = 0.0
    hdr["CDELT2"] = float(cdelt_y)
    hdr["CUNIT2"] = "deg"


def _provenance_cards(
    hdr: fits.Header, beam_type: str, elements: List[str], part: str, telescope: str = "MeerKAT"
) -> None:
    """Non-WCS bookkeeping: what this plane is and which part it holds.

    ``DATE`` is ISO-8601 UTC rather than eidos's ``time.ctime()``: FITS wants
    ISO-8601 and astropy can parse it back, which it cannot do with ctime.
    ``TELESCOP`` follows the BDS provenance attr (MeerKAT for the legacy
    products, MeerKAT Extension for the 2026 MKE store); eidos hard-codes
    MeerKAT because it only models that telescope.
    """
    hdr["TELESCOP"] = telescope
    hdr["DATE"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    hdr["ORIGIN"] = "meerkat-beams"
    hdr["BEAMTYPE"] = beam_type
    hdr["ELEMNT"] = ",".join(elements)
    hdr["PART"] = part.upper()


def _write_fits(path: str, data: np.ndarray, hdr: fits.Header) -> None:
    """Write one real plane: local paths get their parent created, URIs a file handle."""
    hdu = fits.PrimaryHDU(data, header=hdr)
    if "://" in path:
        # astropy wants a seekable binary handle; UPath gives one for s3/gs/…
        with UPath(path).open("wb") as fh:
            hdu.writeto(fh)
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        hdu.writeto(path, overwrite=True)


def _exact_channel_indices(bw: BeamWizard, freq: np.ndarray) -> Optional[np.ndarray]:
    """Indices of ``freq`` in the BDS axis when every value sits exactly on a channel, else None.

    Native grid *and* native channels is the common case, and there the write
    path can copy the stored planes instead of interpolating them back onto the
    grid they already sit on: bit-identical output, and far cheaper than a
    spline evaluation per sample.
    """
    bds_freqs = bw.bds.coords["FREQ"].values
    if freq.size == 0:
        return None
    # freq_to_index is a plain interp1d, which raises for *any* value outside
    # the BDS range -- so the bounds must be strict here, with no
    # FREQ_MATCH_ATOL_HZ slack. A frequency within the extrapolation margin
    # (even one only a few Hz outside the edge) must go through
    # interpolate_beam, which clips to the edge plane and scales it by nu/nu_e.
    # The tolerance is still used below for the exact-channel match itself.
    if freq.min() < bds_freqs.min() or freq.max() > bds_freqs.max():
        return None
    idx = np.rint(bw.freq_to_index(freq)).astype(int)
    if np.all(np.abs(bw.index_to_freq(idx) - freq) <= FREQ_MATCH_ATOL_HZ):
        return idx
    return None


def bds_to_fits(
    bds_path: str,
    prefix: str,
    beam_type: str = "njones",
    elements: Optional[List[str]] = None,
    output_eight: bool = False,
    pixels: Optional[int] = None,
    diameter: Optional[float] = None,
    freqs: Optional[List[float]] = None,
    num_freq: Optional[int] = None,
    freq_extrapolation: str = "raise",
) -> List[str]:
    """
    Render a beam dataset (BDS) to eidos-compatible FITS beam cubes.

    Args:
        bds_path: path to the beam dataset (.bds.zarr)
        prefix: output path prefix; files land as <prefix>_re.fits and
          <prefix>_im.fits (the latter for Jones/Mueller variables), or with
          output_eight as <prefix>_{ij}_{re,im}.fits -- eidos's naming
        beam_type: BDS variable to render: 'njones' (default), 'jones',
          'nstokes', 'stokes', 'nmueller' or 'mueller'
        elements: matrix elements to render. Default: all four Jones elements
          for a Jones variable (eidos's default full-matrix hypercube), or a
          single 'II' otherwise (a single Stokes element, as eidos's -S
          selects). An empty list is treated as the default.
        output_eight: write one file per Jones element per part (eidos's -o8);
          refuses non-Jones beam types, as eidos refuses -o8 with --Stokes
        pixels: pixels per side of the output grid (eidos's -p)
        diameter: field of view of the output grid in degrees (eidos's -d)
        freqs: explicit frequencies in Hz; mutually exclusive with num_freq
        num_freq: number of evenly spaced channels spanning the BDS range
        freq_extrapolation: 'raise' (default) or 'scale'; see BeamWizard

    Returns:
        The list of FITS files written, in write order.
    """
    bds_path, prefix = str(bds_path), str(prefix)  # the CLI passes UPaths (#38)
    if prefix in ("", "None"):
        # The generated CLI's output parameter defaults to None; refuse it here
        # rather than writing a file called None_re.fits.
        raise ValueError("prefix is required: files are written as <prefix>_re.fits etc.")
    if pixels is not None and pixels < 2:
        raise ValueError(f"pixels must be at least 2, got {pixels}")
    if diameter is not None and diameter <= 0:
        raise ValueError(f"diameter must be positive (degrees), got {diameter}")

    # --- element selection and layout checks -------------------------------
    if not elements:
        # eidos's defaults: the full Jones matrix, or a single Stokes element.
        # An explicit empty list means the same as omitting it, matching
        # resolve_elements()'s own default and avoiding a silent no-op with
        # --output-eight (which would otherwise write no files at all).
        els = list(_JONES_FULL) if beam_type in _JONES_TYPES else [_STOKES_DEFAULT]
    else:
        els = list(elements)
    # Validates the grammar (and rejects an unknown beam_type) before anything is opened.
    label_list, _ = resolve_elements(beam_type, els, None)
    is_jones = beam_type in _JONES_TYPES
    if output_eight and not is_jones:
        raise ValueError(
            f"output_eight writes the eight Jones element files, but beam_type={beam_type!r} is not "
            "a Jones variable; drop --output-eight to write one hypercube pair instead"
        )
    if not output_eight:
        if is_jones:
            if sorted(els) != sorted(_JONES_FULL):
                raise ValueError(
                    f"hypercube output writes the full 2x2 Jones matrix: elements must be {','.join(_JONES_FULL)}, "
                    f"got {','.join(els)}; pass --output-eight to write individual Jones element files"
                )
        elif len(els) != 1:
            raise ValueError(
                f"hypercube output takes a single Stokes/Mueller element (as eidos's -S does), got {len(els)}: "
                + ", ".join(els)
            )

    # --- open the BDS and resolve frequencies ------------------------------
    bw = BeamWizard(bds_path, freq_extrapolation=freq_extrapolation)
    telescope = str(bw.bds.attrs.get("telescope", "MeerKAT"))
    try:
        da = bw.bds[beam_type]
    except KeyError:
        # Reuse BeamWizard's actionable message (typo, or a variable this kind
        # of BDS does not carry) instead of a bare xarray KeyError.
        bw._get_prefilter(beam_type, 0, 0, order=3, verbose=0)
        raise
    freq, _ = bw._resolve_freqs(freq=freqs, num_freq=num_freq)
    freq = bw.check_freqs(np.asarray(freq, dtype=float))
    if freq.size == 0:
        raise ValueError("no frequencies selected: give --freqs and/or --num-freq")

    # --- output geometry ---------------------------------------------------
    l, m, cdelt_x, cdelt_y, crpix_x, crpix_y = _target_grid(bw.bds, pixels, diameter)
    n_x, n_y, nfreq = l.size, m.size, freq.size
    native = pixels is None and diameter is None
    x0_b, y0_b = float(bw.bds.attrs["x0"]), float(bw.bds.attrs["y0"])
    dx_b, dy_b = float(bw.bds.attrs["dx"]), float(bw.bds.attrs["dy"])
    nx_b, ny_b = int(bw.bds.sizes["X"]), int(bw.bds.sizes["Y"])
    # BeamWizard works in BDS pixel coordinates: x = l/dx + x0.
    xp, yp = l / dx_b + x0_b, m / dy_b + y0_b
    grid_x, grid_y = np.meshgrid(xp, yp)  # (n_y, n_x), x fastest -- C order below
    xpyp = np.stack([grid_x.ravel(), grid_y.ravel()])

    # interpolate_beam zero-fills pixels outside the model cube (mode="constant",
    # cval=0.0). Under freq_extrapolation="scale" an out-of-band plane is the
    # edge plane with its offsets scaled by nu/nu_e, so its footprint is not the
    # nominal one: count each distinct scale, otherwise a scaled native-grid
    # plane that leaves the cube is zero-filled without being reported.
    bds_freqs = bw.bds.coords["FREQ"].values
    scales, scale_counts = np.unique(freq / np.clip(freq, bds_freqs.min(), bds_freqs.max()), return_counts=True)

    def _outside_count(scale: float) -> int:
        """Number of target pixels outside the cube at one nu/nu_e scale.

        xp/yp are 1-D, so the 2-D mask is separable: the union of the x and y
        out-of-range columns is nx_out * n_y + ny_out * n_x - nx_out * ny_out.
        """
        xs = x0_b + (xp - x0_b) * scale
        ys = y0_b + (yp - y0_b) * scale
        nx_out = int(((xs < -0.5) | (xs > nx_b - 0.5)).sum())
        ny_out = int(((ys < -0.5) | (ys > ny_b - 0.5)).sum())
        return nx_out * n_y + ny_out * n_x - nx_out * ny_out

    outside_counts = [_outside_count(float(s)) for s in scales]
    outside_total = int(np.dot(outside_counts, scale_counts))
    if outside_total:
        l_b = (np.arange(nx_b) - x0_b) * dx_b
        m_b = (np.arange(ny_b) - y0_b) * dy_b
        scale_note = ""
        out_scales = scales[scales != 1.0]
        if out_scales.size:
            eff_l = [float(l_b.min() / s) for s in out_scales], [float(l_b.max() / s) for s in out_scales]
            eff_m = [float(m_b.min() / s) for s in out_scales], [float(m_b.max() / s) for s in out_scales]
            scale_note = (
                "; out-of-band channels are evaluated on the edge plane with offsets scaled by nu/nu_e, "
                f"so their model covers l in [{min(eff_l[0]):.3f}, {max(eff_l[1]):.3f}] deg and "
                f"m in [{min(eff_m[0]):.3f}, {max(eff_m[1]):.3f}] deg"
            )
        log.warning(
            f"{outside_total} of {nfreq * n_x * n_y} output pixels fall outside the beam model "
            f"(nominal l in [{l_b.min():.3f}, {l_b.max():.3f}] deg, "
            f"m in [{m_b.min():.3f}, {m_b.max():.3f}] deg) and are set to 0{scale_note}"
        )
    freq_cdelt = _freq_cdelta(freq)
    exact_idx = _exact_channel_indices(bw, freq) if native else None
    block = max(1, min(nfreq, _BLOCK_ELEMENTS // max(1, n_x * n_y)))

    log.info(
        f"rendering {beam_type} elements {els} at {n_x}x{n_y} px over {nfreq} channel(s), "
        f"pixel scale {cdelt_x:.6g} deg, CRPIX ({crpix_x:.6g}, {crpix_y:.6g})"
        + (" [native grid, copying stored planes]" if exact_idx is not None else "")
    )

    def sample_element(i, j, out: np.ndarray) -> None:
        """Fill ``out`` (nfreq, n_y, n_x) for matrix element labelled (i, j)."""
        if exact_idx is not None:
            out[...] = da.sel({da.dims[0]: i, da.dims[1]: j}).isel(FREQ=exact_idx).values
            return
        for start in range(0, nfreq, block):
            fb = freq[start : start + block]
            vals = bw.interpolate_beam(xpyp, fb, var=beam_type, i=i, j=j, order=3)
            out[start : start + len(fb)] = vals.reshape(len(fb), n_y, n_x)

    # The matrix axes are 2x2 for a Jones hypercube and 1x1 for the single
    # Stokes/Mueller element a Stokes hypercube carries; the *labels* sampled
    # stay the BDS's own (0/1 receptors, IQUV stokes), only the storage slot
    # collapses to (0, 0).
    ni = nj = 2 if is_jones else 1
    slots = label_list if is_jones else [(0, 0)] * len(label_list)
    dtype = da.dtype
    # Bytes of one real part of one element cube -- complex64 stores as float32.
    plane_bytes = nfreq * n_y * n_x * np.empty(0, dtype=dtype).real.dtype.itemsize

    def cube_header(part: str) -> fits.Header:
        hdr = fits.Header()
        _sky_cards(hdr, cdelt_x, cdelt_y, crpix_x, crpix_y)
        if is_jones:
            # eidos maps xy[0]='H' to the matrix row (FITS axis 4) and xy[1]='V'
            # to the column (axis 3); it only writes them when the channel count
            # happens to be 2 (H/V) or 4 (Mx/My), and blanks them otherwise.
            hdr["CTYPE4"], hdr["CRPIX4"], hdr["CRVAL4"], hdr["CDELT4"] = "H", 1.0, 0.0, 1.0
            hdr["CTYPE3"], hdr["CRPIX3"], hdr["CRVAL3"], hdr["CDELT3"] = "V", 1.0, 0.0, 1.0
        else:
            # Label both matrix axes with the FITS Stokes code of the element
            # they carry (row -> axis 4, column -> axis 3).
            e0, e1 = label_list[0]
            hdr["CTYPE4"], hdr["CRPIX4"], hdr["CRVAL4"], hdr["CDELT4"] = "STOKES", 1.0, float(_STOKES_CODE[e0]), 1.0
            hdr["CTYPE3"], hdr["CRPIX3"], hdr["CRVAL3"], hdr["CDELT3"] = "STOKES", 1.0, float(_STOKES_CODE[e1]), 1.0
        hdr["CTYPE5"], hdr["CRPIX5"] = "FREQ", 1.0
        hdr["CRVAL5"], hdr["CDELT5"], hdr["CUNIT5"] = float(freq[0]), float(freq_cdelt), "Hz"
        _provenance_cards(hdr, beam_type, els, part, telescope)
        return hdr

    def element_header(element: str, part: str) -> fits.Header:
        hdr = fits.Header()
        _sky_cards(hdr, cdelt_x, cdelt_y, crpix_x, crpix_y)
        hdr["CTYPE3"], hdr["CRPIX3"] = "FREQ", 1.0
        hdr["CRVAL3"], hdr["CDELT3"], hdr["CUNIT3"] = float(freq[0]), float(freq_cdelt), "Hz"
        _provenance_cards(hdr, beam_type, [element], part, telescope)
        return hdr

    written: List[str] = []
    if output_eight:
        log.info(f"writing {len(els)} element cubes, about {plane_bytes / 1e6:.0f} MB per part")
        for element, (i, j) in zip(els, label_list):
            plane = np.empty((nfreq, n_y, n_x), dtype=dtype)
            sample_element(i, j, plane)
            for part, arr in (("re", plane.real), ("im", plane.imag)):
                if part == "im" and beam_type not in _IM_TYPES:
                    continue
                path = f"{prefix}_{element.lower()}_{part}.fits"
                _write_fits(path, np.ascontiguousarray(arr), element_header(element, part))
                log.info(f"wrote {path}")
                written.append(path)
    else:
        log.info(f"writing the {ni}x{nj} element hypercube, about {ni * nj * plane_bytes / 1e6:.0f} MB per part")
        cube = np.empty((nfreq, ni, nj, n_y, n_x), dtype=dtype)
        for (i, j), (slot_i, slot_j) in zip(label_list, slots):
            sample_element(i, j, cube[:, slot_i, slot_j])
        for part in ("re", "im"):
            if part == "im" and beam_type not in _IM_TYPES:
                # stokes/nstokes are real: eidos writes no _im file for them either.
                continue
            arr = np.ascontiguousarray(cube.real if part == "re" else cube.imag)
            path = f"{prefix}_{part}.fits"
            _write_fits(path, arr, cube_header(part))
            log.info(f"wrote {path}")
            written.append(path)
    return written
