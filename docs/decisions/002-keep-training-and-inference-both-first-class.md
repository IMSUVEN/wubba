# 002 Keep Training And Inference Both First-Class

## Decision

`wubba` treats both training-time workflows and inference/deployment workflows
as first-class parts of the package.

## Why

The project is not only a research prototype. It also positions itself as a way
to produce reusable HTML embeddings for downstream tasks. That makes inference,
similarity search, quantization, and ONNX export part of the product identity,
not just utilities attached later.

## Alternatives Considered

- Focus only on training and leave inference to external scripts:
  Simpler at first, but weakens the library's usability and product coherence.
- Focus only on inference wrappers around a fixed checkpoint:
  Easier to market, but breaks the project's self-supervised training identity.
