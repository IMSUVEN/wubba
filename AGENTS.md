# AGENTS.md

## Project

Self-supervised HTML representation learning with PyTorch Lightning.

- **Python**: >=3.10
- **Package Manager**: uv
- **Layout**: `src/wubba/`

## Commands

```bash
# Setup
uv sync                         # Install dependencies
uv sync --group dev             # With dev tools

# Code quality (run before committing)
uv run ruff format src examples # Format
uv run ruff check src examples  # Lint (must pass)
uv run pyright                  # Type check (0 errors)

# Quick test
uv run python -c "import wubba; print(wubba.__version__)"
```

## Code Style

- Line length: 100
- Quotes: double
- Type hints: `T | None` not `Optional[T]`
- Imports: sorted by ruff
- Docstrings: Google style

## Architecture

```
src/wubba/
├── config.py      # ALL hyperparameters here
├── model.py       # Encoder, loss functions, multi-task heads
├── data.py        # Data processing, augmentation
├── train.py       # Training pipeline, callbacks
├── inference.py   # Inference, quantization, ONNX
├── metrics.py     # Embedding quality, collapse detection
├── const.py       # Vocabulary, semantic groups
└── utils.py       # DOM utilities
```

## Key Patterns

- All hyperparameters → `Config` dataclass
- Training logic → Lightning callbacks
- New loss → add to `model.py`, register in `_setup_loss_function`
- New augmentation → add to `utils.py`, register in `const.py` tiers

## PR Checklist

```bash
uv run ruff format src examples  # 1. Format
uv run ruff check src examples   # 2. Lint (must pass)
uv run pyright                   # 3. Type check
```

Title format: `[component] description` (e.g., `[model] add GQA support`)

## Tips

- Use `train_quick()` for fast iteration
- Check `AGENTS.md` patterns before adding new components
- Update `__all__` in `__init__.py` when adding public API
