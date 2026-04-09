# 004 Clarify Core Identity Versus Experimental Choices

## Decision

`wubba` should treat the following as core project identity:

- Learning reusable HTML embeddings from raw HTML structure
- Using a Transformer encoder over DOM-derived node features
- Creating training signal through behavioral augmentation and contrastive-style
  objectives
- Keeping both training and inference as first-class library surfaces
- Supporting embedding truncation, similarity, quantization, and ONNX export as
  part of the product surface

`wubba` should currently treat the following as experimental or replaceable:

- The specific default loss recipe (`enhanced_hybrid`) and its extra terms such
  as spectral loss, synthesized hard negatives, and alignment-uniformity
- Auxiliary multitask heads such as masked node prediction and structure
  prediction
- Phase-2 feature and augmentation additions such as extended features,
  contextual augmentation, and tree mixup
- Training orchestration strategies such as curriculum learning, self-paced
  learning, EMA checkpointing, collapse-response heuristics, and progressive
  Matryoshka unlocking

## Why

The core set is what the repository repeatedly commits to across package shape,
README framing, and public API:

- `README.md` describes `wubba` as self-supervised HTML representation learning
  that turns raw HTML into embeddings for downstream use
- `src/wubba/data.py`, `src/wubba/model.py`, `src/wubba/train.py`, and
  `src/wubba/inference.py` all depend on the same main flow of HTML parsing,
  DOM feature construction, Transformer encoding, contrastive training, and
  embedding serving
- `src/wubba/inference.py` exposes similarity, Matryoshka truncation,
  quantization, and ONNX export directly as product-facing behavior rather than
  script-local helpers

The experimental set is important work, but the code presents it as layered
options rather than irreducible identity:

- Several features are explicitly marked as later additions in `src/wubba/data.py`
  with `Phase 2` comments
- Many advanced behaviors are controlled by booleans or loss-type switches in
  `Config`, which means the package can still function coherently without
  committing to one permanent recipe
- `train_quick()` disables much of this stack and still preserves a meaningful
  training path, which is a practical sign that these choices are enhancements
  rather than the library's minimal identity

Matryoshka needs a split judgment:

- Matryoshka-style truncated embeddings are core to the inference surface
  because they are exposed directly to users and shape deployment behavior
- Progressive Matryoshka training schedules and weighting schemes are still
  experimental because they are optimization strategy, not the product promise

## Alternatives Considered

- Treat the full current default stack as core:
  This would freeze too many early research choices before there is evidence
  they are durable.
- Treat only "Transformer over HTML" as core and everything else as
  experimental:
  This is too weak and ignores that augmentation-driven contrastive learning and
  deployment-ready embeddings are already central to the package's public
  identity.
