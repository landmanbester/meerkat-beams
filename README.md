# meerkat-beams

MeerKAT primary beam model handling utilities. Downloads, converts, and interpolates primary beam models from the SARAO archive (see https://doi.org/10.48479/wdb0-h061).

## Installation

```bash
pip install meerkat-beams
```

## Usage

```bash
# Download U-band beam model
mbeams download-mdv-beams --source U

# Convert MdV npz to beam dataset
mbeams mdv-beams-to-bds --mdv-beams input.npz --bds output.bds.zarr

# Render BDS to xradio-compatible zarr
mbeams bds-to-xradio --bds-path beam.bds.zarr --image-path image.zarr --output beam_xradio.zarr

# Convert MdV npz directly to xradio zarr
mbeams mdv-to-xradio --npz-path input.npz --output beam_xradio.zarr
```
