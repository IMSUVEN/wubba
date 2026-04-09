# 003 Adopt Anima Seed Structure

## Decision

`wubba` adopts the `anima` seed pattern at the repository knowledge layer:

- `AGENTS.md` becomes the state, map, conventions, and cultivation entrypoint
- `docs/ARCHITECTURE.md` becomes the architecture mirror
- `docs/decisions/` becomes the persistent decision record

## Why

Before this change, the repository had code and a public README, but almost no
persistent project memory beyond those files. That made it easy for future
sessions to understand what `wubba` does, but harder to understand how it
should grow or where important decisions belong.

The seed structure gives the project a minimal way to accumulate identity
without forcing a heavyweight template onto an early-stage library.

## Alternatives Considered

- Keep only the README and a command-focused `AGENTS.md`:
  Minimal, but too weak as a long-term memory system.
- Add a large documentation tree immediately:
  More complete, but too heavy for the current project size and maturity.
