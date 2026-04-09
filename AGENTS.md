# wubba

> This project is past germination but still early enough that architecture,
> conventions, and research priorities should be kept explicit as they evolve.
> The harness should grow from actual model work, not from generic ML boilerplate.

## State

**Phase: Early Growth.**

`wubba` is a self-supervised HTML representation learning library built around a
Transformer encoder, behavioral augmentation, contrastive-style losses, and an
inference layer for embeddings, similarity, quantization, and ONNX export.

What already exists:

- A packaged `src/wubba/` library with public training and inference APIs
- A central `Config` dataclass for model, data, and training hyperparameters
- Example scripts covering quickstart, similarity search, classification,
  embedding analysis, custom training, deployment, and batch processing
- Basic quality tooling via `ruff` and `pyright`
- A growing `pytest` suite covering config, data processing, inference,
  utilities, metrics, lightweight training entrypoint behavior, and example
  surface contracts
- External maintenance feedback loops via GitHub Actions CI and Dependabot

What is still taking shape:

- The long-term architecture map beyond the current single-package layout
- Which modeling and augmentation choices are core identity versus experiments
- The project's durable decision record and maintenance conventions
- How far the test suite should expand into model- and trainer-level execution
  coverage

## Map

| Path | Purpose |
|---|---|
| `README.md` | Product framing, installation, examples, public-facing overview |
| `docs/ARCHITECTURE.md` | Architecture mirror: what exists now and why |
| `docs/decisions/` | Significant technical and project decisions |
| `.github/` | External feedback loop configuration: CI and dependency updates |
| `src/wubba/` | Library source code |
| `examples/` | Usage and integration examples |
| `pyproject.toml` | Packaging, dependencies, and quality tool configuration |

## Conventions

- Keep the public API small and explicit. New capabilities should fit the
  current package shape unless there is a clear architectural reason not to.
- Keep hyperparameters centralized in `Config` rather than scattering them
  across modules.
- Treat examples as part of the product surface. If the workflow changes,
  examples should stay honest.
- Keep examples aligned with the public API and `Config`; lightweight tests may
  guard that surface even when examples are not executed end-to-end.
- Treat dependency update PRs as normal maintenance: let CI run first, then
  review according to how much of the public surface they can affect.
- Treat core runtime upgrades, especially `torch`, as environment decisions as
  well as version bumps; importability and resolved wheel families matter.
- Prefer repository memory over chat memory. Stable decisions belong in
  `docs/decisions/`, not only in commit context or conversation.

## Cultivation

This project's harness should continue to grow from practice:

- **Decisions**: When a technical choice would be costly to reverse or its
  reasoning is non-obvious, record it in `docs/decisions/`.
- **Architecture**: When responsibilities move between modules or the package
  layout changes meaningfully, update `docs/ARCHITECTURE.md`.
- **Conventions**: When a repeated pattern proves itself, promote it into this
  file rather than re-explaining it ad hoc.
- **State**: Keep this file honest about what is stable, what is exploratory,
  and what still needs to be decided.

## Quality Loop

Run these before shipping meaningful changes:

```bash
uv sync --group dev
uv run ruff format src examples tests
uv run ruff check src examples tests
uv run pytest tests
uv run pyright
uv run python -c "import wubba; print(wubba.__version__)"
```
