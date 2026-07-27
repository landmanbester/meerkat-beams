---
type: log
title: Wiki changelog
description: Chronological record of wiki updates.
timestamp: 2026-07-27T10:35:37Z
---

# Wiki changelog

## 2026-07-27 — bundle created (verified at `967be4d`)

- Initial six-page bundle: `beamwizard.md`, `beam-orientation.md`,
  `data-model.md`, `design-decisions.md`, `index.md`, `log.md`.
- Folded the durable content of, and retired, `docs/superpowers/specs/*` and
  `docs/superpowers/plans/*` (point-in-time process artifacts; recoverable
  from git history — last tracked at the pre-Task-8 commit).
- Context captured in `design-decisions.md` (D8/D9): the original "no PyPI
  release" plan from PR #8's merge note has been reversed — a release is now
  being cut, tracked by issue #17, which names the hip-cargo git dependency
  (D8) as its hard blocker. That blocker landed first: `hip-cargo>=0.3.0` now
  resolves from PyPI (commit `3bb9b3d`) instead of a `git+...@main` pin,
  unblocking issue #17, which itself remains open.
