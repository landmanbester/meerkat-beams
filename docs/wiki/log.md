---
type: log
title: Wiki changelog
description: Chronological record of wiki updates.
timestamp: 2026-07-27T13:10:48Z
---

# Wiki changelog

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
