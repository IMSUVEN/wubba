#!/usr/bin/env python3
"""Quickstart: Train a model and generate embeddings in minutes."""

from pathlib import Path

from wubba import Config, WubbaInference, train_quick

# --- Quick Training (minimal config, fast iteration) ---
config = Config(data_dir=Path("data/html_samples/"))
model, trainer = train_quick(
    config=config,
    num_epochs=10,
    batch_size=256,
)

# --- Inference ---
inference = WubbaInference("models/best.ckpt", Config())

html_docs = [
    "<body><nav><a>Home</a><a>About</a></nav><main><h1>Welcome</h1></main></body>",
    "<body><header><nav><a>Home</a></nav></header><article><h1>Hello</h1></article></body>",
]

# Get embeddings (256-dim by default)
embeddings = inference.predict(html_docs)
print(f"Shape: {embeddings.shape}")  # (2, 256)

# Compute similarity
sim = inference.compute_similarity(html_docs[0], html_docs[1])
print(f"Similarity: {sim:.4f}")

# Use smaller dimensions for faster retrieval (Matryoshka)
embeddings_64 = inference.predict(html_docs, dim=64)
print(f"64-dim shape: {embeddings_64.shape}")  # (2, 64)
