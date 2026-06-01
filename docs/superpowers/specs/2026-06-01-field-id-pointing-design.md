# FIELD_ID-driven field selection with a pre-rephased MS

Date: 2026-06-01
Status: approved (design)

## Problem

`scripts/test_beam_orientation.py` runs the beam-orientation validation
against a calibrator MS. That MS has already been **rephased to the source**
(PKS 1934-638) by an earlier processing step, and that step did **not**
preserve the original pointing directions in the FIELD table. The original
pointings are currently recorded only as a comment block at the top of
`scripts/beam_orientation/ms_io.py`.

Two consequences:

1. **The script's rephasing origin is hardcoded and wrong.** It computes the
   SIN-projection `(dl, dm)` offset from a hardcoded `Offset1` position rather
   than from the MS's actual phase centre. For a pre-rephased MS this
   double-rotates relative to what the data already encodes.

2. **Field selection and beam centre are inconsistent.** `read_ms` hardcodes
   `FIELD_ID == 1` in its taql clauses, while the script uses `Offset1`
   (FIELD 0) coordinates as the beam centre. The selected visibilities and the
   assumed pointing direction therefore disagree.

The experiment is a beam-orientation test: the dish points on-source (FIELD 0,
0° separation) and at four offsets (FIELDS 1–4, ~1.4° each), so the
unpolarized calibrator lands at different positions in the primary beam. The
beam centre for `BeamWizard` must be the **pointing direction** of the selected
field; the source position (`srcpos`, PKS 1934 from `tests/conftest.py`) is
fixed.

## Design

Make field selection consistent and general by keying everything off a single
`FIELD_ID`, while keeping the rephasing step in place and correct for both
pre-rephased and non-rephased MSs.

### `scripts/beam_orientation/ms_io.py`

- **`read_ms(path, field_id: int = 0)`** — replace the two hardcoded `== 1`
  taql clauses (main-table `FIELD_ID == 1` and FIELD-subtable
  `SOURCE_ID == 1`) with the `field_id` parameter.

- **`phase_centre`** — unchanged. Keep reading it from the FIELD table. For the
  rephased MS this is the source centre; for a non-rephased MS it is the
  original phase centre. Either way it is the correct origin for rephasing.

- **`pointing_centre: tuple[float, float]`** — new field on `MSBundle`,
  resolved by a defensive two-tier helper:

  1. **POINTING table.** Open `{path}::POINTING`. Select rows whose `TIME`
     lies within the selected field's main-table time window
     (`times.min() … times.max()`), average `DIRECTION` over those rows
     (and over antennas / the leading poly term), and return `(ra, dec)` in
     radians. The POINTING table carries no `FIELD_ID`, so matching is by scan
     time window. The POINTING table is the authoritative original pointing and
     survives rephasing (which touches DATA / UVW / FIELD, not POINTING).

  2. **Fallback dict.** `ORIGINAL_POINTING: dict[int, tuple[float, float]]`,
     a module-level dict promoted from the existing comment block, used when
     the POINTING table is absent, empty, or yields no rows in the window
     (the likely case for this particular MS).

  The helper wraps the POINTING read in `try/except` and falls through to the
  dict on any failure or empty result. It logs which source supplied the
  pointing.

`ORIGINAL_POINTING` values (from the current comment block):

| FIELD_ID | name        | ra (rad)            | dec (rad)            |
|----------|-------------|---------------------|----------------------|
| 0        | Offset1     | 5.146178203219011   | -1.1119958085589738  |
| 1        | J1939-6342  | 5.146178203219011   | -1.1364304180868943  |
| 2        | Offset2     | 5.146178203219011   | -1.0875611990310532  |
| 3        | Offset3     | 5.201372059151767   | -1.1119958085589738  |
| 4        | Offset4     | 5.090979983963126   | -1.1119958085589738  |

### `scripts/test_beam_orientation.py`

- Add `--field-id` (`type=int`, `default=0`), passed to `read_ms`.

- **Rephasing origin** — replace the hardcoded `ra_pc, dec_pc = … # Offset1`
  block with `ra_pc, dec_pc = bundle.phase_centre`. The `(dl, dm)` SIN math and
  the `phase_rotate.phase_rotate(...)` call are unchanged. For the pre-rephased
  MS, `srcpos ≈ bundle.phase_centre`, so `(dl, dm) ≈ 0` and the rotation is a
  no-op; for a non-rephased MS the same code rephases correctly.

- **Beam centre** — `bw.set_field_centre(SkyCoord(*bundle.pointing_centre,
  unit="rad", frame="icrs"))`, replacing the previous reuse of the hardcoded
  pointing.

### Unchanged

Mueller assembly, `solve_per_bin`, coherency→Stokes conversion, plotting,
zarr output, and `srcpos` (PKS 1934 from `tests/conftest.py`).

## Verification

- `uv run ruff check .` and `uv run ruff format --check .` pass.
- Smoke run per field where the MS is available:
  - `--field-id 0` → source on-axis, beam gain ≈ identity.
  - `--field-id 1..4` → source ~1.4° off-axis, with the expected
    orientation-dependent structure across the four perturbation runs.

## Out of scope

- Reading a non-standard `FIELD_ID`/`TARGET` column from the POINTING table
  (time-window matching is sufficient here).
- Any change to the Mueller / solve / plotting path.
