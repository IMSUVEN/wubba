# 001 Centralize Hyperparameters In Config

## Decision

`wubba` keeps model, augmentation, training, monitoring, and path
hyperparameters centralized in the `Config` dataclass instead of scattering them
through multiple modules or training scripts.

## Why

The project already exposes many interacting options: Transformer depth and
width, Matryoshka settings, loss variants, EMA, self-paced learning,
augmentation probabilities, and data limits. A single configuration surface
makes the training system easier to reason about, easier to serialize in user
code, and easier to extend without hiding behavior in many places.

## Alternatives Considered

- Module-local configuration:
  Reduces dataclass size, but hides the system's true control surface.
- Script-only flags:
  Convenient for isolated experiments, but weak as a stable library interface.
