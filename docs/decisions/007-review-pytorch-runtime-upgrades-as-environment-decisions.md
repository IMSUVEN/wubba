# 007 Review PyTorch Runtime Upgrades As Environment Decisions

## Decision

`wubba` should treat major PyTorch runtime upgrades as environment decisions,
not only package-version updates.

The default handling pattern is:

- Let CI run first, but do not rely on CI alone
- Check whether the resolved wheel set still imports cleanly in the working
  environment
- Distinguish core runtime upgrades from ordinary indirect dependency bumps
- Avoid merging runtime upgrades that require an unchosen packaging strategy
  such as a new CUDA family, GPU-only assumptions, or a CPU-only override path

## Why

PyTorch is not just another Python dependency in this repository. It is the
 core runtime for training, inference, quantization, and ONNX export.

During evaluation of the `torch 2.11.0` Dependabot PR, the updated environment
resolved to a different CUDA-family package set locally and failed at import
time with `libcusparseLt.so.0` missing. That means a runtime upgrade can be
green in one environment and still be operationally wrong in another if the
packaging path has shifted underneath the project.

This is materially different from updates such as `ruff`, `rich`, or other
lockfile-only maintenance. The risk is not just API behavior. It is whether the
project can even start in the environments it claims to support.

## Consequences

- Runtime upgrades should be reviewed with explicit attention to importability
  and resolved wheel families, not only test results
- The project should not silently drift into a new CUDA packaging assumption
  without a recorded decision
- If future runtime upgrades keep hitting this boundary, the repository should
  choose and document one of these paths:
  CPU-first installs for development and CI, explicit CUDA-family support, or
  a tighter version/marker constraint for `torch`

## Alternatives Considered

- Treat PyTorch like any other dependency bump:
  Simpler, but too weak for a package whose core behavior depends on PyTorch
  importing and running correctly
- Block all runtime upgrades until a full packaging strategy is designed:
  Safer, but overly rigid for the current stage of the project
