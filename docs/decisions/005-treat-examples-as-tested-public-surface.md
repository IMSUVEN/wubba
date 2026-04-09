# 005 Treat Examples As Tested Public Surface

## Decision

`wubba` treats the `examples/` directory as part of the public product surface,
not as disposable sample code. Examples should stay aligned with the current
public API, `Config` fields, and documented workflows, and that alignment
should be protected by lightweight tests.

## Why

Examples are one of the first places users look to understand how the project
is meant to be used. In this repository, they do more than advertise features:
they define the intended shape of quick training, inference, deployment,
analysis, and batch processing workflows.

That makes example drift costly. A stale example can misrepresent supported
configuration fields, monitoring names, or public imports even when the library
itself is still healthy. Protecting examples with lightweight surface tests is
a low-cost way to keep the repository's public story honest without requiring
every example to execute end-to-end in CI.

## Alternatives Considered

- Treat examples as informal sketches:
  Lower maintenance overhead, but too likely to drift away from the actual
  public API and confuse users.
- Require every example to run end-to-end in tests:
  Stronger validation, but too heavy for the current project stage because most
  examples depend on checkpoints, optional dependencies, or training data.
