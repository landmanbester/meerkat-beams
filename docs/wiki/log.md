---
type: log
title: Wiki changelog
description: Chronological record of wiki updates.
timestamp: 2026-10-02T12:02:46Z
---

# Wiki changelog

## 2026-10-01 — katbeam as a BeamWizard beam model (verified at `ad10c54`)

Issue #22. `BeamWizard` can now build its beams from katbeam's analytic model as
well as the MdV holographic BDS, and `get_rotation_averaged_beam` gained full
azimuthal averaging.

- New `src/meerkat_beams/katbeam_bds.py`: `synthesize_katbeam_bds` samples
  `JimBeam` onto a grid and assembles a lazy, dask-backed `xarray.Dataset`
  carrying the BDS schema, so `BeamWizard` consumes it unchanged. The band to
  model map and `require_model` moved here out of `scripts/compare_katbeam.py`.
- Jones to Stokes conversion extracted from `core/mdv_beams_to_bds.py` into
  `utils.py` (`jones_to_mueller`, `mueller_to_stokes`). Both models share it,
  which makes katbeam's own `JimBeam.I()` a genuine cross-check of our
  conversion rather than a tautology.
- `data-model.md` gains "Synthesized katbeam BDS": three variables not six, the
  structural zeros, the even-`npix` requirement, the per-band geometry defaults,
  the refusal of out-of-table frequencies, and the `pi/4` singularity fix. Also
  corrects the now-stale `mueller_func, mdv_beams_to_bds.py:81-84` reference.
- `beamwizard.md` gains "`beam_model`" and "`average=\"pa\"` or `\"azimuth\"`",
  including that the hard `cval=0.0` off-cube policy still applies to a
  synthesized grid even though `JimBeam` would evaluate beyond it.
- `design-decisions.md` gains **D11**, recording that the synthesized-BDS
  approach was chosen over a beam-source abstraction and that the scipy
  interpolators are deliberately **retained** — so issue #22's "without the need
  to use the scipy interpolators" is explicitly not delivered, with the costs
  listed.
- `CLAUDE.md`: resolves the open "whether katbeam becomes a runtime dep is an
  issue #22 decision" note. It is now in the `[full]` extra from PyPI, with the
  dev/test git pin kept, so **S-band katbeam works only in a dev checkout**.

## 2026-07-28 — katbeam comparison probe (verified at `a62c7e1`)

- New `scripts/compare_katbeam.py` + `tests/test_compare_katbeam.py` (84
  hermetic unit tests): compares the MdV-derived BDS beams against katbeam's
  analytic JimBeam across four products (Stokes I, HH power, VV power,
  cross-pol power), with radially-partitioned residuals and an orientation
  sweep.
- `beam-orientation.md` gains "Independent probe: the katbeam comparison".
  **Split verdict, deliberately stated as such:** the transpose is decisively
  rejected (`swap_xy` x5.7-10.0 worse than `none`, all three products
  agreeing), corroborating the `(Y, X)` labelling in
  `mdv_beams_to_bds.py:26,130`; the **sign flips are not discriminated**
  (`flip_x` on VV scores x1.05), because MeerKAT's ~0.05-1 arcmin squint is
  under one 3.75 arcmin BDS pixel. Nothing here closes the M1 question.
- Caught and fixed a one-pixel mirror artifact in the sweep. An even-sized
  grid centred on a pixel is not symmetric about zero (L-band X runs
  `-4.0..+3.9375`: 64 negatives, one zero, 63 positives), so a naive axis
  reverse is a mirror *plus* a one-pixel translation. It reported spurious
  x17-21 flip penalties; a pure one-pixel roll with no mirror reproduced
  5.14e-2 against the naive flip's 5.21e-2. `registration_roll()` now
  corrects it, with three regression guards. Any future orientation tooling on
  this grid inherits the same trap.
- Agreement where the models overlap: mainlobe Stokes I residual 1.4-3.5e-3 of
  peak, FWHM to 0.1-0.3% on both axes over 900-1650 MHz. Recorded with the
  caveat that this is **not independent** — katbeam's tables were themselves
  fit to MeerKAT holography, so it is a consistency check, not two independent
  models converging.
- Real divergences quantified: cross-pol (katbeam models it as exactly zero;
  ours reaches 3.8e-3 of peak at 1650 MHz, with the expected four-lobe
  clover-leaf nulling on axis), sidelobes, and azimuthal structure.
- `katbeam` added to the `dev` **and** `test` dependency groups, pinned to git
  main: PyPI's only release (0.1) has no S-band model and a narrower L table
  (900-1650 vs 856-1712 MHz). Duplicated into `test` because CI's test job
  syncs `--group test --extra full` without `--group dev`, which would have
  made every katbeam test skip silently. This reintroduces the git-pin pattern
  D8 retired for hip-cargo — a deliberate, dev/test-only choice.

## 2026-07-27 — PEP 440 tbump + git-cliff changelog (verified at `0b4e799`)

- `design-decisions.md`: added **D10**. `tbump` was refusing every bump — a
  `[[file]]` entry searched `_container_image.py` for a versioned tag while the
  file holds `:latest`. Entry deleted (the `before_commit` regex hook already
  owns that tag); version regex moved to PEP 440; `message_template` is now
  `chore(release): ...`.
- New `cliff.toml` + generated `CHANGELOG.md`; `conventional-pre-commit` now
  gates messages at the `commit-msg` stage.
- `publish-container.yml` moved `type=semver` → `type=pep440` so pre-release
  tags still produce container images.
- Two `commit_preprocessors` in `cliff.toml` deviate from hip-cargo's config
  (strip commit bodies, drop GitHub's `(#N)` squash suffix) — see D10.
- No release cut — version is still `0.0.0`, issue #17 stays open.

## 2026-07-27 — bundle created (verified at `967be4d`)

- Initial six-page bundle: `beamwizard.md`, `beam-orientation.md`,
  `data-model.md`, `design-decisions.md`, `index.md`, `log.md`.
- Folded the durable content of, and retired, `docs/superpowers/specs/*` and
  `docs/superpowers/plans/*` (point-in-time process artifacts; recoverable
  from git history — last tracked at commit `c79f590`).
- Context captured in `design-decisions.md` (D8/D9): the original "no PyPI
  release" plan from PR #8's merge note has been reversed — a release is now
  being cut, tracked by issue #17, which names the hip-cargo git dependency
  (D8) as its hard blocker. That blocker landed first: `hip-cargo>=0.3.0` now
  resolves from PyPI (commit `3bb9b3d`) instead of a `git+...@main` pin,
  unblocking issue #17, which itself remains open.

## 2026-10-02 — MeerKAT+ beams and MM/MPM/MPMP baseline groups (issue #30)

- `data-model.md`: new "Baseline-group beam datasets" section — the three
  groups and their dtypes, the `p = MeerKAT` convention and the conjugation
  rule for the reversed ordering, normalise-then-cross, the group attrs, the
  `PROVENANCE_ATTRS` carried by `mdv_beams_to_bds`, the two MdV generations,
  and why groups are L-band only.
- `beamwizard.md`: new "`group` — baseline-group beams" section — the
  constructor-contract table, the in-memory dataset and its ~25 MB footprint,
  and the `MPM` `jones`/`njones` absence.
- `design-decisions.md`: D12 (on-the-fly assembly, not stored), D13 (complex
  `MPM` with p = MeerKAT, and the two easily-confused reversal identities),
  D14 (normalise each Jones first, then cross), D15 (generation follows group;
  L band only, blocked on data not code).
- `index.md`: `data-model.md` and `beamwizard.md` entries extended.
- Code: `jones_to_mueller_cross`, `GROUP_TELESCOPES`, `_build_group_bds`,
  `BeamWizard(group=...)`, `cache` re-keyed on product names with
  `ensure_group_bds`, `scripts/stage_group_cache.py`.
- The MdV-2026 and MKE cache entries carry `PLACEHOLDER` gdrive IDs: the
  tarballs are not public yet, so `ensure_product_bds` refuses to download
  them and points at manual staging. Fill in `MDV2026_GDRIVE_IDS` when they
  are published.
- `partition_mueller` (issue #27) is deliberately **not** in this change; it
  is a thin wrapper to be built on this foundation.
- Post-review fixes (eac4bd0): the channel match is now required to be
  injective (ambiguous matches raise instead of aliasing a plane), a
  frequency slice refreshes `fits_header`, `stage_product --force` drops the
  stale BDS as well as the input, and the group fixtures were made asymmetric
  so the `p = MeerKAT` contract is actually pinned by a test.
- Greptile review fixes: a gapped (unevenly spaced) channel intersection now
  raises instead of emitting a header whose linear FREQ axis misplaces every
  plane after the gap; `stage_product` refuses to stage a product from its own
  cache entry and stages through a `.partial` + `os.replace` so a failed copy
  leaves the previous entry serving; D8 and CLAUDE.md updated for the
  `hip-cargo>=0.4.0` floor (commit `4b07822`, which predates this branch).

## 2026-10-06 — partition_mueller

- `BeamWizard.partition_mueller` documented in `beamwizard.md`: one per-partition
  Stokes Mueller block for `pfb-imaging` and `QuartiCal` (issue #27), with the
  four contract points — `(Y, X)` orientation, `nstokes` for apparent flux, no
  `1/n`, and dtype following the data — plus the prefilter memory table.
- `centre=` on `get_rotation_averaged_beam` documented in the same section: it is
  how a partition's pointing centre reaches the parallactic-angle computation
  without mutating a wizard shared across chunks.
- D16 records that the method assembles rather than fuses the 16 element calls,
  and that the amortisation worth having is the per-wizard prefilter cache.
- D17 records that katbeam is excluded by scope, and corrects the premise in
  issue #27: `katbeam_bds.py` does produce a full 4×4 Stokes beam.
- Post-review fixes: the `partition_mueller` fixtures were not discriminating.
  The single-telescope BDS used `scale=1.0`, which makes `stokes` and `nstokes`
  bit-identical (normalising by an identity on-axis Jones is a no-op), so every
  assertion about `normalised=False` passed for the wrong reason; it now uses
  `scale=0.7`. The two group BDSs differed only in `scale`, which normalisation
  divides out, leaving their `nstokes` 3e-5 apart — so the cross-group
  geometric-mean check accepted either auto group in the cross's place; the MKE
  fixture now has a narrower beam (`sigma=3.5`, the physical case for 15 m
  dishes) and the tolerance is tight enough to reject both substitutions.
  `jones_beam_cube`/`build_mean_beam_zarr` grew a `sigma` argument for this.
  The normalisation row in this page's `partition_mueller` table had the
  relationship inverted ("divided by the Jones inverse" is multiplying by the
  matrix) and is corrected to match `data-model.md` and D14; the orientation row
  no longer implies the shape test covers transposed content.
