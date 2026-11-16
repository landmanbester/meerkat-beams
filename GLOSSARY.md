# meerkat-beams

Primary-beam models for the MeerKAT and MeerKAT+ dishes: the measured beams, the forms they are stored in, and the forms applications ask for.

## Language

### Beam sources

**MdV beams**:
The holographic primary-beam measurements of MeerKAT by M. de Villiers, published by SARAO as voltage beams per band.
_Avoid_: holography beams, SARAO beams

**Mean beam**:
The array-average MdV beam, one Jones matrix per frequency and direction. It is the only beam this package exposes.
_Avoid_: array beam, average beam

**MdV generation**:
A release of MdV beams with its own grid and channelisation. The legacy generation covers MeerKAT only; the MdV-2026 generation covers MeerKAT and MeerKAT+ on a matched grid.

**katbeam model**:
The analytic MeerKAT beam from the `katbeam` package, used as an alternative to the MdV beams.
_Avoid_: analytic beam, JimBeam

**MeerKAT+**:
The MeerKAT Extension dishes, abbreviated MKE in product names.
_Avoid_: MK+, extension array

### Beam representations

**Jones beam**:
The 2×2 complex voltage response of one dish, per frequency and direction.
_Avoid_: voltage beam, E-Jones

**Mueller beam**:
The 4×4 response of a baseline to the coherencies, formed from the Jones beams of its two dishes.

**Stokes beam**:
The Mueller beam expressed in the Stokes basis: how each intrinsic Stokes parameter appears in each apparent one.
_Avoid_: power beam

**Normalised beam**:
A beam divided by its on-axis response, so that it is the identity at the pointing centre. This is the default; the unnormalised beam is called raw.

**Baseline group**:
The set of baselines sharing one pair of dish types: `MM` (MeerKAT–MeerKAT), `MPM` (MeerKAT–MeerKAT+) or `MPMP` (MeerKAT+–MeerKAT+). Each group has its own Mueller beam.
_Avoid_: baseline type, sub-array

**Rotation-averaged beam**:
A beam averaged over the parallactic angles of an observation (or over a full turn), as seen in the sky frame.
_Avoid_: time-averaged beam, effective beam

**Partition Mueller block**:
The rotation-averaged Stokes beam for one partition of the data, used to turn an intrinsic Stokes model into an apparent one.

### Geometry

**Pointing centre**:
The sky direction the dishes track. The beam is centred on it and turns about it with parallactic angle.
_Avoid_: field centre, beam centre

**Phase centre**:
The tangent point of an image or model grid, from which its `l` and `m` are measured. It may differ from the pointing centre.
_Avoid_: field centre, image centre

**Feed-plane coordinates**:
The coordinates of the MdV beam grids: direction cosines in the plane of the feed, expressed in degrees. They turn on the sky with parallactic angle because of the dish's alt-az mount.

### Stored forms

**BDS (beam dataset)**:
The package's working store of one telescope's beam: Jones, Mueller and Stokes beams, raw and normalised, on one grid.
_Avoid_: beam cube, beam file

**Product**:
A named, downloadable mean beam for one telescope, band and MdV generation, from which a BDS is built and cached.
_Avoid_: band (a band may have several products)

**xradio beam**:
A beam rendered along time and frequency for one observation, in the xradio image schema.

**eidos FITS cube**:
A beam written as FITS in the layout the `eidos` package produces, for consumers that read FITS beam models.
