# BeamWizard — optional `image_name`

**Date:** 2026-05-25
**Status:** Approved (design)

## Motivation

`BeamWizard.__init__` currently raises `ValueError("image_name is required")`.
Some workflows only need beam values at one or more explicit source positions
and never use the image's WCS grid or time axis. The motivating example is
`scripts/test_beam_orientation.py`, which constructs `BeamWizard(band="L")`
with no image and asks for the beam gain at the PKS 1934-638 position via
`assemble_mueller` → `bw.get_source_coordinates(...)` + `bw.interpolate_beam(...)`.
As written, that script cannot run.

Make `image_name` optional, give clear errors when an image-requiring method
is used on a BDS-only wizard, and provide methods to attach an image — or just
a field centre — after construction.

## Background: what the image provides

When `image_name` is given, `__init__` derives four things the BDS alone does
not provide:

- `self.wcs` / `self.centre` — the field (pointing) centre.
- `self.times` — observation time axis (`None` for a FITS image).
- `self.l_grid` / `self.m_grid` — default l/m grid from image pixels.

Method dependency on this image-derived state:

| Method | Needs image state? |
|---|---|
| `interpolate_beam`, `_resolve_freqs`, `_get_prefilter` | No — BDS only |
| `get_source_coordinates` | `centre` (and `times` only if `times=` omitted) |
| `get_rotation_averaged_beam` | `centre`; `times`/`l_grid`/`m_grid` only if omitted |
| `get_time_freq_beam` | `centre`; `times`/`l_grid`/`m_grid` only if omitted |

All existing external consumers of these attributes — `core/bds_to_xradio.py`
(`bw.l_grid`, `bw.m_grid`) and the integration/unit tests
(`bw.centre`, `bw.wcs`, `bw.l_grid`, `bw.m_grid`, `bw.times`) — only run in
image-attached contexts, so read-only properties that return the value when set
are transparent to them.

Key consequence: even with no image, `get_source_coordinates` still needs a
**field/pointing centre**. In the orientation test that centre is the MS
`phase_centre`. So "no image" must still offer a way to supply that centre.

## Design

### Construction behaviour

- `__init__(bds_name=None, image_name=None, *, band=None)` keeps the existing
  XOR check (exactly one of `bds_name` / `band` must be given — a BDS source is
  always required) and drops the `image_name is required` raise.
- BDS-derived state is always set: `bds`, `index_to_freq` / `freq_to_index`,
  `default_location`, `_prefilters`.
- Image-derived state is stored in private backing fields, all initialised to
  `None`: `_wcs`, `_centre`, `_times`, `_l_grid`, `_m_grid`.
- If `image_name is not None`, `__init__` calls `self.attach_image(image_name)`.
- Otherwise it emits a single `log.warning`: the wizard is in BDS-only mode and
  image-requiring methods will error until `attach_image()` or
  `set_field_centre()` is called.

### Guard mechanism — guarded properties

- `centre`, `wcs`, `l_grid`, `m_grid` are read-only properties. If the backing
  field is `None`, the getter raises `RuntimeError` with an actionable message
  naming both remedies, e.g.:

  > `BeamWizard.centre is unavailable: this wizard was constructed without an
  > image. Call attach_image(image_name) or set_field_centre(centre=...) first.`

  Centralising the check in the property means every current and future method
  that reads `self.centre` (etc.) gets the clear error for free, with no risk of
  forgetting a guard.

- `times` is a property that returns `None` when unset — **not** an error.
  `None` is already a valid state (FITS images have no time axis) and every
  method already handles `times is None` by demanding an explicit `times=`
  argument with its own `RuntimeError`. This preserves existing behaviour,
  including the two `getattr(self, "times", None)` call sites (which can be
  simplified to read `self.times` directly, since the property now always
  exists).

*Alternative considered and rejected:* explicit `if self._centre is None: raise`
checks at the top of each public method. More boilerplate, easy to miss when
adding methods, and scatters the message text.

### New and refactored methods

- `attach_image(self, image_name)` — the existing image-parsing block
  (current `utils.py` lines 118–150: FITS/zarr WCS, axis dropping, `centre`,
  `l_grid`/`m_grid`, logging) lifted verbatim into a method that writes the
  backing fields (`_wcs`, `_centre`, `_times`, `_l_grid`, `_m_grid`).
  `__init__` calls it when `image_name` is provided. This is the method to
  "set image_name and related parameters after instantiation".

- `set_field_centre(self, centre: SkyCoord, times: Optional[Time] = None)` —
  the file-free path for the single-source use case. Sets `_centre` to the
  supplied `SkyCoord` and, if given, `_times` to the supplied `Time`. It does
  **not** populate `wcs`/`l_grid`/`m_grid`; those remain unavailable, so
  `get_rotation_averaged_beam` / `get_time_freq_beam` called without explicit
  `l`/`m` will correctly hit the `l_grid` guard.

`centre` is an astropy `SkyCoord` (no unit ambiguity; `get_source_coordinates`
already uses `self.centre` as a `SkyCoord` via `.transform_to(frame)`).

### Script wiring — `scripts/test_beam_orientation.py`

The script already builds `BeamWizard(band="L")` and computes
`ra_pc, dec_pc = bundle.phase_centre` (radians). After constructing `bw`, add:

```python
bw.set_field_centre(SkyCoord(ra_pc, dec_pc, unit="rad", frame="icrs"))
```

The script already passes explicit `times` and `freq` everywhere, so no `times`
argument to `set_field_centre` is needed.

### Tests — `tests/test_beam_wizard.py` (hermetic)

Reuse the synthetic BDS + FITS built in `tmp_path` by the existing fixtures.
Add cases pinning:

1. No-image construction (`bds_name=<synthetic BDS>`, no `image_name`) succeeds
   and `interpolate_beam` returns finite values.
2. `bw.centre` and `bw.l_grid` raise `RuntimeError` with the actionable
   message; `bw.times` returns `None`.
3. `get_source_coordinates(srcpos, times=...)` on a no-image wizard raises the
   centre `RuntimeError`.
4. After `set_field_centre(centre)`, `get_source_coordinates(srcpos, times=...)`
   returns finite pixel coordinates.
5. `attach_image(image_name)` on a no-image wizard populates
   `centre`/`l_grid`/`m_grid` and unblocks the grid-default methods.

### Docs

- Update the `BeamWizard` docstring to document optional image / BDS-only mode,
  `attach_image`, and `set_field_centre`.
- Update the CLAUDE.md "Key abstractions in `utils.py` → `BeamWizard`" section
  to the same effect.

## Out of scope

- No method-signature changes other than the two new methods.
- No per-call `centre=` argument on `get_source_coordinates` et al.
- No changes to `interpolate_beam`, `_resolve_freqs`, or prefilter logic.
