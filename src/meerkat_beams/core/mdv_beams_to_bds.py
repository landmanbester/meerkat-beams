"""Core implementation for mdv-beams-to-bds command."""

from pathlib import Path

import numpy as np
import numpy.linalg
import xarray

from meerkat_beams.utils import LOGGER


def mdv_beams_to_bds(mdv_beams: str, bds: str, compress: bool = False):
    """
    Converts MdV's npz beamset into a Stokes I power beam.

    Args:
        mdv_beams: input MdV beams. This can be an npz file or the zarr file containing only the data for the mean beam.
        bds: output beam dataset (BDS) path
        compress: apply Delta+Blosc compression to zarr output
    """
    LOGGER.info(f"loading MdV beams from {mdv_beams}")
    if mdv_beams.endswith(".npz"):
        # load from zarr (probably only the mean beam data, but should work with the full dataset too)
        mdv = np.load(mdv_beams)
        bm = mdv["beam"]
        degs = mdv["margin_deg"]
        freqs = mdv["freq_MHz"] * 1e6
        bm = bm[:, -1]  # select average beam (last antenna index)
    elif (Path(mdv_beams) / ".zgroup").exists():
        xds = xarray.open_zarr(mdv_beams, chunks=None)
        bm = xds.BEAM.values  # already mean beam: [4, NFREQ, NY, NX]
        degs = xds.l_beam.values
        freqs = xds.chan.values
    else:
        raise ValueError(f"input mdv_beams {mdv_beams} is not a valid npz or zarr dataset")

    delta = degs[1] - degs[0]
    i0 = len(degs) // 2

    # form up fits header
    hdr = {}
    hdr["SIMPLE"] = "T"
    hdr["NAXIS1"] = len(degs)
    hdr["NAXIS2"] = len(degs)
    hdr["NAXIS3"] = len(freqs)
    hdr["CRPIX1"] = i0 + 1
    hdr["CRPIX2"] = i0 + 1
    hdr["CRPIX3"] = 1
    hdr["CRVAL1"] = 0
    hdr["CRVAL2"] = 0
    hdr["CRVAL3"] = freqs[0]
    hdr["CDELT1"] = delta
    hdr["CDELT2"] = delta
    hdr["CDELT3"] = freqs[1] - freqs[0]
    hdr["CTYPE1"] = "X"
    hdr["CTYPE2"] = "Y"
    hdr["CTYPE3"] = "FREQ"
    hdr["CUNIT1"] = "deg"
    hdr["CUNIT2"] = "deg"
    hdr["CUNIT3"] = "Hz"

    # See also https://archive-gw-1.kat.ac.za/public/repository/10.48479/wdb0-h061/beam_orientation_diagram.pdf
    # MdV pols are HH, HV, VH, VV, so I think that corresponds to [[HH, HV],[VH,VV]] in the Jones matrix

    LOGGER.info("computing normalized beams")
    jj = bm.reshape([2, 2] + list(bm.shape[1:]))  # reshape to 2x2 to get Jones matrix
    # MdV axes are FREQ,Y,X (probably worth double-checking), so now ROW,COL,FREQ,Y,X
    jjt = jj.transpose((2, 3, 4, 0, 1))  # now FREQ,Y,X,ROW,COLUMN
    jj0 = jjt[:, i0, i0, :, :]  # centre beam: FREQ,ROW,COLUMN
    # linalg.inv() wants last two axes to be matrix row and column, so transpose
    jj0inv = numpy.linalg.inv(jj0)
    # normalized Jones matrix (Jnorm.J)
    jnorm = jj0inv[:, np.newaxis, np.newaxis, :, :] @ jjt

    LOGGER.info("computing Stokes beams")
    # S converts Stokes to coherency
    S = np.array([[1, 1, 0, 0], [0, 0, 1, 1j], [0, 0, 1, -1j], [1, -1, 0, 0]])
    # Sinv converts coherency to Stokes
    Sinv = numpy.linalg.inv(S)

    def mueller_func(jones):
        mshape = list(jones.shape[:-2]) + [4, 4]
        mueller = np.einsum("fyxij,fyxkl->fyxikjl", jones, np.conj(jones)).reshape(mshape)
        return mueller

    # compute Stokes matrices from FREQ,Y,X,ROW,COLUMN Jones matrices
    def stokes_func(mueller):
        return Sinv @ mueller @ S

    # compute Mueller and and normalized Mueller
    mueller = mueller_func(jjt)
    muellernorm = mueller_func(jnorm)
    # convert to Stokes and transpose back to FREQ,Y,X,STOKES_i,STOKES_j
    stokes = stokes_func(mueller).transpose((3, 4, 0, 1, 2)).astype(np.float32)
    stokesnorm = stokes_func(muellernorm).transpose((3, 4, 0, 1, 2)).astype(np.float32)
    # transpose Mueller and Kones
    mueller = mueller.transpose((3, 4, 0, 1, 2)).astype(np.complex64)
    muellernorm = muellernorm.transpose((3, 4, 0, 1, 2)).astype(np.complex64)
    jj = jj.transpose((3, 4, 0, 1, 2)).astype(np.complex64)
    jnorm = jnorm.transpose((3, 4, 0, 1, 2)).astype(np.complex64)

    LOGGER.info(f"saving output dataset {bds}")
    # write to dataset
    # Jones and Stokes have different matrix sizes (2x2 vs 4x4), so use
    # separate dimension names to avoid xarray coordinate conflicts
    jcoords = dict(receptor_i=[0, 1], receptor_j=[0, 1], X=degs, Y=degs, FREQ=freqs)
    scoords = dict(stokes_i=list("IQUV"), stokes_j=list("IQUV"), X=degs, Y=degs, FREQ=freqs)

    xds = xarray.Dataset(
        dict(
            jones=xarray.DataArray(jj, dims=("receptor_i", "receptor_j", "FREQ", "Y", "X"), coords=jcoords),
            njones=xarray.DataArray(jnorm, dims=["receptor_i", "receptor_j", "FREQ", "Y", "X"], coords=jcoords),
            stokes=xarray.DataArray(stokes, dims=["stokes_i", "stokes_j", "FREQ", "Y", "X"], coords=scoords),
            nstokes=xarray.DataArray(stokesnorm, dims=["stokes_i", "stokes_j", "FREQ", "Y", "X"], coords=scoords),
            mueller=xarray.DataArray(mueller, dims=["stokes_i", "stokes_j", "FREQ", "Y", "X"], coords=scoords),
            nmueller=xarray.DataArray(muellernorm, dims=["stokes_i", "stokes_j", "FREQ", "Y", "X"], coords=scoords),
        )
    )
    xds.attrs["fits_header"] = hdr
    xds.attrs.update(x0=i0, y0=i0, dx=delta, dy=delta, freqs=freqs)

    encoding = {}
    if compress:
        from numcodecs import Blosc, Delta

        compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
        filters = [Delta(dtype="float32")]
        for var in ["jones", "njones", "stokes", "nstokes", "mueller", "nmueller"]:
            encoding[var] = dict(compressor=compressor, filters=filters)
    xds.to_zarr(bds, mode="w", encoding=encoding)
