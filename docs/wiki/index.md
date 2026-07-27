---
type: index
title: meerkat-beams LLM wiki
description: Progressive-disclosure listing of the in-repo knowledge bundle.
timestamp: 2026-07-27T10:35:37Z
last_verified_commit: 967be4d
---

# meerkat-beams LLM wiki

In-repo knowledge bundle in the [Open Knowledge Format](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md)
style: plain markdown + YAML frontmatter, readable by humans without tools and
by agents without SDKs. The primary reader is an LLM agent working in this
repo; humans are a secondary audience. Start here, read only the page(s) the
table below points you at, and treat the rest of the codebase as ground truth
when a page and the code disagree.

**Verification contract:** every page's frontmatter carries
`last_verified_commit` — the commit its claims were last checked against. To
assess staleness: `git diff <stamp>..HEAD -- <files the page covers>`; a
non-empty diff means re-verify the page before trusting it. Maintenance rule:
if your change invalidates or extends a page, update the page and refresh its
`timestamp`/`last_verified_commit` **in the same session**. Specs and plans
under `docs/superpowers/` are ephemeral point-in-time process artifacts — they
are not retained as sources and MUST NOT be cited from wiki pages or from
agent output; cite code, tests, commits, or issues instead.

## Pages

| Page | Covers | Read when |
|------|--------|-----------|
| [beamwizard.md](beamwizard.md) | `interpolate_beam` prefilter/off-cube/spline-order/freq-guard semantics, `get_source_coordinates` transforms, optional-image paths, `get_time_freq_beam` canonical `dim_names`, and `enrich_bds_xradio`. | Before touching `BeamWizard` in `utils.py`, or when interpolation/rendering output looks wrong. |
| [beam-orientation.md](beam-orientation.md) | The settled `(Y, X)` rotation-averaged map order, parallactic-angle rotation averaging, and the still-provisional beam-orientation convention with its open M1 validation. | Before touching orientation-sensitive code (`get_rotation_averaged_beam`, pointing-angle transforms), or when a beam map looks transposed/flipped. |
| [data-model.md](data-model.md) | The three beam formats and their conversions — MdV `.npz` structure, the BDS zarr schema (`jones`/`njones`/`stokes`/`nstokes`/`mueller`/`nmueller`, `fits_header`, scalar attrs), and the xradio primary-beam schema. | Before touching `core/mdv_beams_to_bds.py`, `core/bds_to_xradio.py`, `core/mdv_to_xradio.py`, or when a schema field's meaning is unclear. |
| [design-decisions.md](design-decisions.md) | Context/Decision/Rationale/Consequences ledger for meerkat-beams' load-bearing choices, plus the interpolation gotchas and the settled/reversed conventions. | Asking "why is it built this way", before "fixing" something that looks wrong, or before re-litigating a past decision (e.g. the release policy or the hip-cargo dependency pin). |
| [log.md](log.md) | Chronological record of wiki updates. | Checking what changed in the wiki and when. |

## Not covered here

- **How to work in this codebase** (harness/edit rules, `cli/*.py` generation
  and round-trip discipline, commit message format, hip-cargo mechanics,
  Python support policy, test markers): `CLAUDE.md` at the repo root.
- **Install and usage** (package install, `mbeams` CLI walkthrough): `README.md`
  at the repo root.
