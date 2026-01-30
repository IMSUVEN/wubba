# Web Understanding By Behavioral Augmentation (wubba)

> Self-supervised representation learning for HTML documents.

A PyTorch + Lightning implementation for learning layout-invariant embeddings from raw HTML using self-supervised learning and behavioral augmentation.

---

## Features

- **Transformer Encoder**: Captures complex relationships within the DOM structure.
- **Self-Supervised**: Learns from unlabeled HTML using the VICReg loss, no manual labeling required.
- **Behavioral Augmentation**: Creates robust representations by applying structural perturbations to the DOM tree.
- **Simple & Efficient**: Built with PyTorch Lightning for clean and scalable training.

## Quick Start

### 1. Installation

Install the package and its dependencies from the project root:

```bash
pip install .
```

### 2. Get Embeddings

Use a pretrained model to generate embeddings for your HTML documents.

```python
from wubba.config import Config
from wubba.inference import WubbaInference

# Initialize with a model checkpoint and configuration
# Ensure you have a trained model at 'models/best.ckpt'
model = WubbaInference("models/best.ckpt", Config())

# Get a 256-dimensional embedding for a list of HTML strings
html_docs = [
    "<body><h1>Hello</h1><p>This is a test.</p></body>",
    "<body><nav><a>Home</a></nav></body>"
]
embeddings = model.predict(html_docs)

print(f"Generated {len(embeddings)} embeddings.")
print(f"First embedding (first 4 dims): {embeddings[0][:4]}")
```

## License

This project is licensed under the MIT License.