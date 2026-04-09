# 008 Make uv Resolution CPU-First For PyTorch

## Decision

For `uv`-managed environments, `wubba` should resolve `torch` from the
PyTorch CPU wheel index by default.

This is a development and CI stability choice. It gives the repository a
predictable default runtime path for installs, tests, type checking, and ONNX
export validation. GPU-specific installs remain possible later, but they should
be treated as explicit environment choices rather than accidental outcomes of
default dependency resolution.

## Why

Recent evaluation of a `torch 2.11.0` upgrade showed that unconstrained
resolution can drift into a different CUDA-family packaging path and fail
before tests even start. That is too unstable for the repository's default
maintenance loop.

The project already runs its quality loop in CPU-oriented CI, and most current
tests exercise importability, preprocessing, training entrypoints, inference
contracts, and export behavior rather than GPU throughput. A CPU-first default
matches the repository's real day-to-day operating environment better than
implicitly following whichever accelerator wheel family resolves on a machine.

## Consequences

- `uv sync` and CI will use the PyTorch CPU wheel index for `torch`
- Local development becomes more predictable across machines that do not share
  the same CUDA runtime layout
- GPU enablement remains possible, but it is no longer the accidental default
- If the project later wants an official CUDA-first or dual-path strategy, that
  should be documented as a separate decision

## Alternatives Considered

- Keep unconstrained default resolution:
  Simpler on paper, but too dependent on host-specific runtime packaging
- Immediately adopt a specific CUDA-family default:
  Premature while the repository still uses CPU-first CI and has not committed
  to a supported GPU runtime matrix
