# 006 Handle Dependency Update PRs Through CI And Targeted Review

## Decision

`wubba` should treat dependency update pull requests, including Dependabot PRs,
as normal maintenance work that is gated by CI and reviewed with targeted
judgment rather than merged blindly or ignored indefinitely.

The default handling pattern is:

- Let CI run first
- Review the dependency's role in the repository surface
- Merge low-risk updates when checks are green and no public-surface regression
  is evident
- Investigate manually when updates touch training, inference, packaging, or
  optional deployment paths in ways that could change behavior

## Why

The repository now has an active CI feedback loop and already receives
Dependabot PRs. That means dependency maintenance is no longer hypothetical
external noise; it is part of the real operating environment of the project.

Ignoring these PRs would let the dependency surface drift without judgment.
Merging them automatically would be too loose for a library that mixes training,
inference, optional ONNX paths, and a still-settling public surface.

CI plus targeted review is the right middle path for the current project stage:
lightweight enough to keep maintenance moving, but explicit enough to catch
meaningful regressions.

## Alternatives Considered

- Auto-merge all dependency PRs:
  Fast, but too permissive for the current maturity and mixed runtime surface.
- Ignore dependency PRs until breakage occurs:
  Lower immediate effort, but turns maintenance into reactive cleanup.
