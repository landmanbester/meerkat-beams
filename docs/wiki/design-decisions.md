---
type: Design Ledger
title: Design decisions, conventions, and recurring gotchas
description: Context/Decision/Rationale/Consequences ledger for meerkat-beams' load-bearing choices, plus the interpolation gotchas and the settled/reversed conventions.
tags: [design, decisions, conventions, gotchas, cache, hip-cargo, release, versioning, changelog]
timestamp: 2026-07-27T13:10:48Z
last_verified_commit: 0b4e799
---

# Design decisions, conventions, and recurring gotchas

Context/Decision/Rationale/Consequences ledger for load-bearing choices in
`meerkat-beams`. Entries are numbered D1–D9 and are not chronological;
cite the entry ID plus its Source when referring to one of these decisions
elsewhere.

## D1 — `interpolate_beam` prefilter is applied exactly once, cached dtype-aware

**Context:** `scipy.ndimage.map_coordinates` can prefilter its input itself
(`prefilter=True`, the scipy default) or accept an already-filtered array.
`BeamWizard.interpolate_beam` needs the same prefiltered cube repeatedly
(once per source/time), so filtering on every call would be wasteful, and
double-filtering would silently corrupt the interpolated values.

**Decision:** `_get_prefilter` (`src/meerkat_beams/utils.py:247`) runs
`scipy.ndimage.spline_filter` once per `(var, i, j, order)` key and caches
the result; `interpolate_beam` (`utils.py:301`) then calls
`map_coordinates` against that cached array with `prefilter=False`
(`utils.py:326-332`). The cache's output dtype is **not** unconditionally
`float32` — it is chosen per variable kind: `np.complex64 if
np.iscomplexobj(da) else np.float32` (`utils.py:257-262`). Complex
variables (`jones`, `njones`, `mueller`, `nmueller`) cache as `complex64`;
real variables (`stokes`, `nstokes`) cache as `float32`. Passing a real
dtype for complex input would make scipy implicitly promote it (a
version-dependent `UserWarning`), so the dtype is picked explicitly
instead.

**Rationale:** `_get_prefilter` already applied `spline_filter`; flipping
`prefilter` back to `True` in the `map_coordinates` call would re-filter
already-filtered coefficients (double-filter trap). The dtype-aware cache
exists because an earlier, simpler `float32`-only cache silently dropped
the imaginary part of complex beams.

**Consequences:** Do not flip `prefilter` back to `True`. Do not hardcode
the `_get_prefilter` output dtype to `float32` when adding new variable
kinds — check `np.iscomplexobj` first, or complex variables will lose
their imaginary part without raising. See `beamwizard.md` for the full
prefilter contract, including how the dtype rule propagates through
`get_time_freq_beam`'s zarr writes.

**Source:** `utils.py:247,257-262,326-332`; `test_subpixel_matches_direct_scipy`,
`test_prefilter_complex_var_is_complex64_without_warning`
(`tests/test_beam_wizard.py`); commit `0f2a4f4` ("test: add tests for
complex Mueller term" — landed the dtype-aware cache).

## D2 — Explicit off-cube policy (`mode=constant, cval=0`)

**Context:** `map_coordinates` needs an explicit policy for coordinates
that fall outside the beam cube's X/Y extent; the scipy default
(`mode="constant", cval=0.0`) happens to match what's wanted here, but
leaving it implicit means a future scipy version or a copy-pasted call
site could silently change behavior (e.g. to nearest-edge extrapolation).

**Decision:** `interpolate_beam` passes `mode="constant", cval=0.0`
explicitly (`utils.py:333`) rather than relying on the scipy default.
Out-of-cube coordinates return `0`, not a nearest-edge value or `NaN`.

**Consequences:** Callers requesting positions well outside the beam
support get zero gain, not an error or an extrapolated guess — relevant
for sources near the edge of a wide field.

**Source:** `utils.py:333`; `test_out_of_range_xy_returns_zero`
(`tests/test_beam_wizard.py`).

## D3 — `get_time_freq_beam` accepts only canonical `dim_names`

**Context:** `get_time_freq_beam`'s `dim_names` parameter is positionally
interpreted (index 0 = time axis name, 1 = freq, 2 =
polarization/ij, 3 = x/l, 4 = y/m). Real permutation of the underlying
array to match an arbitrary `dim_names` ordering is not implemented.

**Decision:** Only the canonical xradio order
`("time", "frequency", "polarization", "l", "m")` is accepted; any other
tuple raises `ValueError` (`utils.py:722-726`).

**Rationale:** A non-canonical tuple would relabel the dims without
reordering the underlying array — silent data corruption, not a
formatting choice. Raising is strictly safer than partial support.

**Consequences:** Do not add "convenience" reordering via `dim_names`
without implementing real permutation of the array first; a caller
wanting a different axis order must transpose the opened dataset
themselves after the fact.

**Source:** `utils.py:722-726`; `test_time_freq_beam_rejects_non_canonical_dim_names`
(`tests/test_beam_wizard.py`).

## D4 — Rotation-averaged maps are `(Y, X)` (FITS) order

**Context:** `get_rotation_averaged_beam` used to build its 1D `l`/`m`
meshgrid with `indexing="ij"`, producing `(X, Y)`-indexed mean/variance
maps — transposed relative to FITS-convention `(Y, X)` consumers
(breifast, pfb-imaging `hci`). killick found the bug via a covariance-map
reference pixel (`rho[x, y] = 1.000` vs `rho[y, x] = -0.15`); it stayed
hidden elsewhere because the MeerKAT rotation-averaged beam is nearly
circularly symmetric, so a transpose barely changes it.

**Decision:** `np.meshgrid(l, m)` (default `"xy"` indexing, not `"ij"`,
`utils.py:511`) now returns `(NY, NX)`-shaped maps natively, matching
FITS convention (axis 0 = m/north, axis 1 = l/east). Settled at commit
`616906b`.

**Consequences:** This is the settled convention — downstream consumers
must consume the maps as `(Y, X)` with **no** on-receipt transpose. Do not
flip this back to `(X, Y)`. breifast is a separate, still-open case: it
has not adopted this package yet, so its own stopgap on-receipt transpose
(`killick-polishes-silver` @ `0a898bb`) remains correct until it does —
see `beam-orientation.md` for the full post-mortem and the distinction
between this settled index-order question and the still-provisional
physical orientation convention (D9-adjacent, not in this ledger's
numbered list — see `beam-orientation.md`).

**Source:** `utils.py:511`; `test_rotation_averaged_beam_1d_lm_returns_y_x_order`,
`test_rotation_averaged_beam_map_indexes_as_y_x` (`tests/test_beam_wizard.py`);
`beam-orientation.md`.

## D5 — Python support policy: 3.11–3.13 full, 3.10 lightweight-only

**Context:** The `[full]` extra pulls in the full scientific stack
(`xarray`, `zarr<3`, `astropy`, `scipy`, `numpy`, `matplotlib`, `dask-ms`,
`wget`, `gdown`); some of that stack's transitive deps lag on 3.10
support or aren't worth pinning for a base install.

**Decision:** 3.10 is supported only for the lightweight base install
(CLI + hip-cargo container dispatch, no `[full]`); it is deliberately
excluded from the CI test matrix (`.github/workflows/ci.yml`
`matrix.python-version = ["3.11", "3.12", "3.13"]`) and instead pinned by
a dedicated `lightweight` CI job that installs only the base package and
checks `mbeams --help` on 3.10. Test-group deps in `pyproject.toml` carry
`python_version >= '3.11'` markers for the same reason
(`xarray-ms`, `msv4-utils`).

**Consequences:** Do not add 3.10 to the main test matrix. A CLI-only
change intended to work on 3.10 should be smoke-tested against the base
install, not the full test suite.

**Source:** `pyproject.toml:6-9,93-98`; `.github/workflows/ci.yml:68-146`
(the `lightweight` job); CLAUDE.md ("Conventions" section).

## D6 — Band cache under `MBEAMS_CACHE_DIR`

**Context:** `BeamWizard(band=...)` needs a beam dataset (BDS) to attach
to; building one from scratch means downloading a per-band mean-beam
zarr from gdrive and converting it, which is too slow to repeat on every
construction.

**Decision:** `BeamWizard(band=...)` routes through `cache.ensure_band_bds`
(`src/meerkat_beams/cache.py`), which auto-downloads the mean-beam zarr
tarball from gdrive and converts it to a compressed BDS, caching both
under `cache_root()` (`$MBEAMS_CACHE_DIR`, else
`$XDG_CACHE_HOME/meerkat-beams`, else `~/.cache/meerkat-beams`) as
`inputs/MeerKAT_<BAND>.zarr/` and `bds/MeerKAT_<BAND>.bds.zarr/`.
Downloads and conversions land in a sibling `.partial` directory first and
are atomically promoted (`os.replace`) on success; a stale `.partial` or
stale download tarball from a killed process is cleared on the next call
(`_clear_partials`). Supported bands are exactly the keys of
`BAND_GDRIVE_IDS` — `U`, `L`, `S0`, `S4` (S1/S2/S3 have no published
gdrive ID; request those explicitly via `bds_name=` instead of
`band=`). Concurrent first-time downloads of the same band from multiple
processes are **not** guarded — warm the cache from a single process.

**Consequences:** A second consumer of `cache.cache_root()` is the
beam-orientation validation tooling's calibrator-MS cache
(`test_ms/<MS_BASENAME>/`, `scripts/beam_orientation/download.py`) — same
unlocked, single-process download posture. Any new "supported bands" work
must update `BAND_GDRIVE_IDS`, not a separate hardcoded list — code that
special-cases band names elsewhere risks drifting from this registry.

**Source:** `cache.py:26-32` (`BAND_GDRIVE_IDS`/`SUPPORTED_BANDS`),
`cache.py:1-18` (module docstring: cache layout, resolution order,
concurrency posture), `cache.py:54-69` (`ensure_band_bds`);
`test_beam_wizard_band_routes_through_cache` (`tests/test_beam_wizard.py`),
`test_registry_contains_expected_bands`,
`test_ensure_band_bds_rejects_unknown_band`,
`test_ensure_band_bds_clears_stale_partials` (`tests/test_cache.py`).

## D7 — hip-cargo three-layer + cab round-trip discipline

**Context:** `mbeams` commands need a Typer CLI, a Stimela cab definition,
and a plain-Python implementation, kept in lockstep. Hand-maintaining all
three risked drift between the CLI's declared interface and what the cab
actually described.

**Decision:** One-to-one layout: `cabs/foo.yml` ↔ `cli/foo.py` ↔
`core/foo.py`. `cli/*.py` is generated by `hip-cargo generate-function`
from the cab YAML — it is **not** hand-written and must not be
hand-edited. The pre-commit `generate-cabs` hook runs
`hip-cargo generate-cabs` on every commit to regenerate cab YAML from
`cli/*.py`; the inverse direction (`cab → cli`) is manual via
`scripts/genfuncs.sh`.

**Rationale:** `cli → cab → cli` must compose to the identity, or the
generated CLI and the cab YAML silently diverge over time.

**Consequences:** If a CLI signature needs to change, edit the cab YAML
and run `scripts/genfuncs.sh` (or edit `cli/*.py`'s core-adjacent logic
only in `core/*.py`, which is plain Python and free to hand-edit).
`tests/test_roundtrip.py` fails the build if `cli/*.py` drifts from what
hip-cargo would regenerate from `cabs/*.yml` — treat that failure as "cab
YAML needs a fix" or "hip-cargo regen is non-idempotent (upstream bug)",
never as license to hand-patch `cli/*.py`.

**Source:** CLAUDE.md ("CLI ↔ cab generation" section);
`tests/test_roundtrip.py` (confirmed present via
`ls tests/test_roundtrip.py`); `test_cli_cab_cli_roundtrip`.

## D8 — hip-cargo git dependency retired

**Context:** For most of the `dev001` branch's life, `meerkat-beams`
depended on `hip-cargo @ git+https://github.com/landmanbester/hip-cargo.git@main`
because recent cli↔cab fixes (list-default round-tripping, `image:`
persistence) were not yet in a tagged release. A git dependency blocks a
clean PyPI publish and is why the Dockerfile installed `git` at all.

**Decision:** Now `hip-cargo>=0.3.0`, resolved from PyPI (`pyproject.toml:22`,
commit `3bb9b3d`); `uv.lock` confirms the resolved source is
`{ registry = "https://pypi.org/simple" }`, not a git ref. The transitional
git-main pin is gone from `pyproject.toml`.

**Consequences:** The `git` install layer in the Dockerfile (present
*solely* for the git dependency, per its own inline comment) is now
unnecessary and can be dropped — but has **not** been removed yet as of
this commit; that cleanup, plus updating CLAUDE.md's "Conventions" note
and re-confirming `test_roundtrip.py` against the tagged release, is
tracked by issue #16 (still open — its checklist items beyond the
version bump remain outstanding). Do not assume the Dockerfile has
already been updated just because the dependency has.

**Source:** `pyproject.toml:22`; commit `3bb9b3d` ("build: depend on
hip-cargo>=0.3.0"); `uv.lock` (`hip-cargo` package entry, `source =
{ registry = ... }`); issue #16 ("M2: Re-pin from hip-cargo git main to
the next tagged release").

## D9 — "No PyPI release" decision reversed

**Context:** PR #8's original merge note treated `meerkat-beams` as a
demonstrator branch not intended for its own release — consumers would
either vendor it or wait for the upstream `suricat-beams` port.
`pfb-imaging#237` now needs an installable, versioned `meerkat-beams` to
depend on; pointing a merged pfb-imaging feature at a git fork is not
acceptable.

**Decision:** Reversed — a release is now being cut. Tracked by issue #17
("Cut the first PyPI release of meerkat-beams"), which names D8 (the
hip-cargo git dependency) as its hard blocker: a git dependency in
`pyproject.toml` cannot be published to PyPI as-is.

**Consequences:** Issue #17's scope (version bump via `tbump`, packaging
metadata verification, a `publish` GitHub Actions workflow preferring
PyPI Trusted Publishing, TestPyPI dry run, then tag + publish) is
unblocked now that D8 has landed, but issue #17 itself is still open —
do not assume a release has already shipped.

**Source:** PR #8 merge note ("Update (2026-07-24): the original 'no
PyPI release' plan has been reversed..."); issue #16; issue #17.

## D10 — PEP 440 versions and a generated changelog

**Context:** The release machinery was `hip-cargo init` scaffold that had
drifted. `tbump.toml` declared a `[[file]]` entry searching for
`ghcr.io/landmanbester/meerkat-beams:{current_version}` in
`_container_image.py`, but that file holds `:latest` — and tbump validates
every `[[file]]` search string before bumping. `tbump` therefore refused
every version, failing with `Error: Current version string: (0.0.0) not
found in src/meerkat_beams/_container_image.py`. Separately there was no
changelog machinery, and the version regex matched bare semver only.

**Decision:** Ported hip-cargo's arrangement. `tbump.toml` takes the PEP 440
regex (optional `(a|b|rc)N`, optional `.postN`) plus `channel`/`release`/`post`
fields defaulting to `""`; `message_template` becomes `chore(release): bump
version to {new_version}`; the `[[file]]` entry for `_container_image.py` is
**deleted** rather than repaired. `cliff.toml` (new) drives git-cliff from two
`before_commit` hooks. `conventional-pre-commit` gates messages at
`commit-msg`.

**Rationale:** The deleted `[[file]]` entry was redundant, not just wrong —
the tag is already rewritten by the `before_commit` regex hook, and
`update-cabs.yml` resets it to `:latest` on `main` afterwards. That
`:latest`-on-main / pinned-on-tag cycle is deliberate and was already correct;
the `[[file]]` entry fought it. hip-cargo has the hook and no such entry.

**Consequences:**

- **`message_template` and the commit-msg hook are coupled.** The old
  `"Bump version to {new_version}"` is not conventional; installing the hook
  without changing it would make tbump's own release commit fail the gate it
  just added. Verified directly: that exact string is rejected by
  `conventional-pre-commit`, the new one passes. Do not revert one without
  the other.
- **`publish-container.yml` had to move from `type=semver` to `type=pep440`.**
  `v0.1.0rc1` is not valid semver, so a pre-release tag would have produced no
  container tags at all — silent until the first rc.
- **Do not restore the `[[file]]` entry for `_container_image.py`,** and do not
  "fix" that file to a version number on `main`. Both re-break `tbump`.
- **`CHANGELOG.md` is generated and hand edits do not survive.** `git-cliff -o`
  rewrites the file from full git history on every release, so it is not a
  prepend. Changelog defects must be fixed in `cliff.toml`.
- **Two `commit_preprocessors` deviate from hip-cargo's `cliff.toml`,** both
  forced by this repo's pre-enforcement history. (1) `(?s)\r?\n.*` → `""`
  strips commit bodies: conventional commits already expose only their subject
  as `commit.message`, but `filter_unconventional = false` keeps unconventional
  ones whole, and PR #8's squash body alone rendered 600 of the seeded
  changelog's 640 lines. (2) ` \(#[0-9]+\)$` → `""` drops GitHub's trailing
  squash suffix, which the body template already renders as a linked
  reference. Both are no-ops for well-formed conventional commits, so they can
  stay indefinitely.
- Commits predating enforcement land under `### Other` in the changelog. This
  is cosmetic and self-limiting — it cannot be tidied by hand (see above), and
  new commits are gated.
- git-cliff is invoked via `uvx`, not declared as a dependency — cutting a
  release on a cold machine needs network access.
- Issue #17 remains open: this lands the machinery only. No tag was cut, and
  the version is still `0.0.0`.

**Source:** `tbump.toml`; `cliff.toml`; `.pre-commit-config.yaml`;
`.github/workflows/publish-container.yml`; `~/software/hip-cargo` at v0.3.0
(reference implementation).

## Recurring gotchas

- **Don't flip `interpolate_beam`'s `prefilter` back to `True`.** The
  cached array from `_get_prefilter` is already filtered; re-filtering it
  double-filters and silently corrupts interpolated values. See D1.
- **Don't hardcode the `_get_prefilter` output dtype to `float32`.** It's
  `np.iscomplexobj`-selected — complex variables (`jones`, `njones`,
  `mueller`, `nmueller`) need `complex64`, or the imaginary part is
  silently dropped rather than raising. See D1; `beamwizard.md`.
- **Don't re-transpose `get_rotation_averaged_beam`'s `(Y, X)` maps.**
  That index order is the settled convention as of commit `616906b`;
  a downstream on-receipt transpose (breifast's stopgap) is only correct
  because breifast hasn't adopted this package's output yet — don't add a
  new one against this package's maps. See D4.
- **Don't hand-edit `cli/*.py`.** All CLI-signature changes flow through
  the cab YAML plus `scripts/genfuncs.sh`; `test_roundtrip.py` exists to
  catch exactly this mistake. See D7.
- **The physical beam-orientation convention is provisional, not
  settled.** `get_source_coordinates`'s "0 is up, +90 is right" convention
  (distinct from D4's map-index order, which *is* settled) has not passed
  the M1 validation experiment as of this commit — see `beam-orientation.md`
  for the full status and open issues (#9–#12).

## Sources

- `src/meerkat_beams/utils.py:247-263,301-335` (`_get_prefilter`,
  `interpolate_beam`)
- `src/meerkat_beams/utils.py:426-511` (`get_rotation_averaged_beam`)
- `src/meerkat_beams/utils.py:655-728` (`get_time_freq_beam`,
  `_CANONICAL_DIM_NAMES` guard)
- `src/meerkat_beams/cache.py` (cache layout, `BAND_GDRIVE_IDS`,
  `ensure_band_bds`, `.partial` atomicity)
- `pyproject.toml:6-9,22,93-98` (Python support policy, hip-cargo
  dependency, test-group markers)
- `.github/workflows/ci.yml:68-146` (test matrix + `lightweight` job)
- `Dockerfile` (`git` install layer, inline comment noting it is
  transitional)
- `tests/test_beam_wizard.py`: `test_subpixel_matches_direct_scipy`,
  `test_prefilter_complex_var_is_complex64_without_warning`,
  `test_out_of_range_xy_returns_zero`,
  `test_time_freq_beam_rejects_non_canonical_dim_names`,
  `test_rotation_averaged_beam_1d_lm_returns_y_x_order`,
  `test_rotation_averaged_beam_map_indexes_as_y_x`,
  `test_beam_wizard_band_routes_through_cache`
- `tests/test_cache.py`: `test_registry_contains_expected_bands`,
  `test_ensure_band_bds_rejects_unknown_band`,
  `test_ensure_band_bds_clears_stale_partials`
- `tests/test_roundtrip.py`: `test_cli_cab_cli_roundtrip`
- commit `0f2a4f4` ("test: add tests for complex Mueller term")
- commit `616906b` ("fix(utils)!: get_rotation_averaged_beam returns
  (Y, X)-ordered maps")
- commit `3bb9b3d` ("build: depend on hip-cargo>=0.3.0")
- issue #16 ("M2: Re-pin from hip-cargo git main to the next tagged
  release")
- issue #17 ("Cut the first PyPI release of meerkat-beams")
- PR #8 (merge note, M1/M2 checklists)
- `docs/wiki/beamwizard.md`, `docs/wiki/beam-orientation.md`,
  `docs/wiki/data-model.md`
- CLAUDE.md ("Architecture", "CLI ↔ cab generation", "Conventions"
  sections)
