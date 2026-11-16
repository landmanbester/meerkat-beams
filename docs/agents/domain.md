# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

This repo is single-context. It keeps its decisions in the LLM wiki's design ledger, not in `docs/adr/`.

## Before exploring, read these

- **`GLOSSARY.md`** at the repo root: the domain terms. It holds definitions only; schemas, dtypes and other implementation detail belong in `docs/wiki/data-model.md`.
- **`docs/wiki/index.md`**, then the pages it routes to for the area you're about to work in.
- **`docs/wiki/design-decisions.md`**: this repo's ADRs, a Context/Decision/Rationale/Consequences ledger of `D<n>` entries. Read the entries that touch the area you're about to work in. There is no `docs/adr/`; do not create one.

## Recording a decision

Where a skill says "write an ADR", add an entry to `docs/wiki/design-decisions.md` instead:

- Add `## D<n> — <title>` at the **end** of the file, using the next free D-number. Entries are not in file order (the `Recurring gotchas` and `Sources` sections sit between them), so find the highest existing number rather than reading the last heading's neighbours.
- Use the Context / Decision / Rationale / Consequences form of the existing entries.
- Update the "numbered D1–D<n>" line in the page intro.
- Follow the wiki maintenance rules in `CLAUDE.md` ("LLM wiki"): refresh the page's `timestamp` and `last_verified_commit` as that section defines, and add a line to `docs/wiki/log.md`.

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in `GLOSSARY.md`. Don't drift to synonyms the glossary explicitly avoids.

If the concept you need isn't in the glossary yet, that's a signal: either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/domain-modeling`).

## Flag decision conflicts

If your output contradicts an existing ledger entry, surface it explicitly rather than silently overriding:

> _Contradicts D15 (generation follows group; L band only), but worth reopening because…_
