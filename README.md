<h1 align="center">🧪 Wubba</h1>

<p align="center">
  <strong>Web Understanding By Behavioral Augmentation</strong><br>
  <sub>Self-supervised representation learning for HTML documents</sub>
</p>

<p align="center">
  English | <a href="./README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  <a href="#-installation">Installation</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-features">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-license">License</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-3776ab?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.7+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Lightning-2.5+-792ee5?logo=lightning&logoColor=white" alt="Lightning">
  <a href="https://github.com/IMSUVEN/wubba/actions/workflows/ci.yml"><img src="https://github.com/IMSUVEN/wubba/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/code%20style-ruff-000000?logo=ruff&logoColor=white" alt="Ruff">
  <img src="https://img.shields.io/badge/types-pyright-blue?logo=python&logoColor=white" alt="Pyright">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
</p>

---

Wubba learns **layout-invariant embeddings** from raw HTML using contrastive learning. Convert any HTML document into a fixed-size vector for similarity search, clustering, or classification.

The stable center of the project is the raw-HTML-to-embedding pipeline plus the
inference surface around those embeddings. Several advanced training recipes
and augmentation variants are available, but they should still be read as
research-oriented options rather than permanent identity claims.

```python
from wubba import WubbaInference

model = WubbaInference("model.ckpt")
embeddings = model.predict(["<html>...</html>", "<html>...</html>"])
similarity = model.compute_similarity(html1, html2)
```

## 📦 Installation

```bash
uv sync                # or: pip install .
uv sync --group dev    # with dev tools
uv sync --extra onnx   # with ONNX export and validation deps
uv sync --extra examples  # with example-only dependencies
```

`uv` resolves `torch` from the PyTorch CPU wheel index by default so local
development and CI stay on a predictable runtime path. If you need a GPU build,
choose that environment explicitly instead of relying on the default resolver.
The simplest way to do that is to create a dedicated environment and install
PyTorch from the appropriate official index selected via pytorch.org, then
install `wubba` on top of that environment.

## 🚀 Quick Start

### 🔍 Generate Embeddings

```python
from wubba import WubbaInference

model = WubbaInference("models/best.ckpt")

# Full embeddings
embeddings = model.predict(html_docs)  # (N, 256)

# Matryoshka: truncate for speed/size tradeoff
embeddings = model.predict(html_docs, dim=64)  # (N, 64)
```

### 🎯 Train a Model

```python
from wubba import Config, train

config = Config(
    data_dir="data/",
    num_epochs=100,
    batch_size=1024,
    loss_type="enhanced_hybrid",
)

model, trainer = train(config)
```

### 🚢 Deploy

```python
model = WubbaInference("model.ckpt")
model.quantize()                    # INT8 for faster CPU inference
model.export_onnx("model.onnx")     # Cross-platform deployment
```

`quantize()` works with the base install. `export_onnx()` and ONNX validation
require the `onnx` optional dependency group, which installs `onnx`,
`onnxscript`, and `onnxruntime`.

## ✨ Features

| Category | Features |
|----------|----------|
| ✅ **Core Surface** | Raw HTML to embeddings, Transformer encoder, behavioral augmentation, contrastive training, similarity search |
| ✅ **Deployment** | Matryoshka truncation, INT8 quantization, ONNX export |
| 🧪 **Advanced Training Options** | VICReg, InfoNCE, hybrid losses, EMA, collapse detection, curriculum learning |
| 🧪 **Experimental Extensions** | Spectral loss, hard negative mining, multitask heads, contextual augmentation, Tree Mixup |

## 🏗️ Architecture

```
HTML → Parser → Node Features → Transformer Encoder → CLS Pooling → Embedding
                     ↑                    ↑
              10/15 dims          Hierarchical RoPE
                                  Flash Attention
                                  RMSNorm + SwiGLU
```

📊 **Input features per node:** tag_id, semantic_group, depth, position, num_children, sibling_count, is_leaf, parent_tag_id, tag_role, subtree_depth

## ⚙️ Configuration

Key options in `Config`:

```python
Config(
    # 🧠 Model
    transformer_dim=256,
    transformer_layers=6,
    matryoshka_dims=[32, 64, 128, 256],
    
    # 🎯 Training
    loss_type="enhanced_hybrid",  # advanced recipe, not the only intended path
    use_ema=True,
    enable_multitask=True,
    
    # 📊 Data
    use_extended_features=True,   # 15-dim features
    use_contextual_aug=True,
)
```

## 📚 Examples

| Example | Description |
|---------|-------------|
| [01_quickstart.py](examples/01_quickstart.py) | Train and generate embeddings in minutes |
| [02_web_deduplication.py](examples/02_web_deduplication.py) | Detect duplicate/similar pages for crawlers |
| [03_similarity_search.py](examples/03_similarity_search.py) | Build a search index for HTML documents |
| [04_page_classification.py](examples/04_page_classification.py) | Classify pages using embeddings as features |
| [05_production_deployment.py](examples/05_production_deployment.py) | Quantization and ONNX export |
| [06_embedding_analysis.py](examples/06_embedding_analysis.py) | Analyze embedding quality and visualize |
| [07_custom_training.py](examples/07_custom_training.py) | Advanced training with custom callbacks |
| [08_batch_processing.py](examples/08_batch_processing.py) | Process millions of HTML documents |

## 📁 Project Structure

```
src/wubba/
├── config.py      # All hyperparameters
├── model.py       # Encoder and loss functions
├── data.py        # Data processing and augmentation
├── train.py       # Training pipeline
├── inference.py   # Inference and export
├── metrics.py     # Embedding quality metrics
└── utils.py       # DOM utilities
```

## 🛠️ Development

```bash
uv run ruff format src examples tests    # 🎨 Format
uv run ruff check src examples tests     # 🔍 Lint
uv run pytest tests                      # ✅ Regression tests
uv run pyright            # 📝 Type check
```

📖 See [AGENTS.md](AGENTS.md) for the project state, conventions, and growth
rules.

📚 Persistent project memory lives in:

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/decisions/README.md](docs/decisions/README.md)

## 📄 License

MIT
