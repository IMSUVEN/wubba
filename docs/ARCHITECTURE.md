# Architecture

This document describes what currently exists in `wubba` and why it is shaped
that way. It should describe the real package, not an aspirational future one.

## Current Shape

`wubba` is a single-package Python library under `src/wubba/` with three main
centers of gravity:

- Configuration and constants
- Training-time model and data pipeline
- Inference-time embedding and deployment surface

## Module Map

- `src/wubba/config.py`
  Central configuration dataclass for model, data, augmentation, training,
  monitoring, and path settings.
- `src/wubba/const.py`
  Shared vocabularies and semantic-group constants used across parsing and
  feature construction.
- `src/wubba/data.py`
  HTML preprocessing, feature extraction, augmentation, and data module logic.
- `src/wubba/model.py`
  Core model definition, projection heads, loss setup, and multi-task heads.
- `src/wubba/train.py`
  Training entrypoints and callback wiring.
- `src/wubba/inference.py`
  Inference API, similarity operations, quantization, and ONNX export.
- `src/wubba/metrics.py`
  Embedding quality metrics, collapse detection, EMA support, and related
  training utilities.
- `src/wubba/utils.py`
  DOM-oriented helpers and lower-level utilities.
- `src/wubba/__init__.py`
  Public API surface and package exports.

## Operating Pattern

The current design keeps most project behavior inside one cohesive package
rather than splitting into many subpackages early. This keeps navigation and
API discovery simple while the project is still settling.

The main flow is:

1. Define behavior through `Config`.
2. Process HTML and generate augmented training views in `data.py`.
3. Encode documents and compute losses in `model.py`.
4. Train through `train.py` callbacks and trainer wiring.
5. Serve embeddings and deployment exports through `inference.py`.

## Stable Architectural Biases

- Configuration is centralized instead of distributed.
- Inference is a first-class package concern, not an afterthought.
- Raw-HTML-to-embedding flow is the architectural center of the package.
- DOM-aware augmentation and contrastive representation learning are core,
  while specific loss recipes and training tricks remain swappable.
- Examples are part of the public surface and are lightly protected against API
  drift.
- The repository currently favors a compact library layout over aggressive
  decomposition.

## Questions To Revisit Later

- Whether the current single-package layout should split as the project gains
  more training recipes or deployment targets
- Which experimental training add-ons should graduate into stable defaults
  after more evidence
- How far the test suite should extend from utility-level regression tests into
  data, model, and training-path coverage
