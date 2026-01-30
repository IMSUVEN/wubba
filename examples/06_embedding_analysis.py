#!/usr/bin/env python3
"""Embedding Analysis: Analyze embedding quality and properties.

Use case: Researchers analyzing learned representations, debugging
training issues, or comparing different model configurations.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from wubba import Config, EmbeddingMetrics, WubbaInference


def analyze_embedding_quality(embeddings: np.ndarray) -> dict:
    """Computes comprehensive embedding quality metrics."""
    import torch

    tensor = torch.from_numpy(embeddings)
    metrics = EmbeddingMetrics.compute_all(tensor)

    return {
        "effective_rank": metrics["effective_rank"],
        "rank_ratio": metrics["rank_ratio"],
        "std_min": metrics["std_min"],
        "std_mean": metrics["std_mean"],
        "std_max": metrics["std_max"],
        "uniformity": metrics["uniformity"],
        "sim_mean": metrics["sim_mean"],
        "sim_std": metrics["sim_std"],
    }


def compare_matryoshka_dimensions(
    html_docs: list[str],
    model: WubbaInference,
) -> dict[int, dict]:
    """Compares embedding quality across Matryoshka dimensions."""
    results = {}

    for dim in model.available_dims:
        embeddings = model.predict(html_docs, dim=dim).numpy()
        metrics = analyze_embedding_quality(embeddings)
        results[dim] = metrics
        print(
            f"dim={dim}: rank_ratio={metrics['rank_ratio']:.3f}, uniformity={metrics['uniformity']:.3f}"
        )

    return results


def visualize_embeddings(
    embeddings: np.ndarray,
    labels: list[str] | None = None,
    method: str = "pca",
    output_path: str | None = None,
):
    """Visualizes embeddings in 2D using PCA or t-SNE."""
    if method == "pca":
        reducer = PCA(n_components=2, random_state=42)
    else:
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))

    coords = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))

    if labels:
        unique_labels = list(set(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        label_to_color = dict(zip(unique_labels, colors, strict=True))

        for label in unique_labels:
            mask = [lbl == label for lbl in labels]
            plt.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=[label_to_color[label]],
                label=label,
                alpha=0.7,
            )
        plt.legend()
    else:
        plt.scatter(coords[:, 0], coords[:, 1], alpha=0.7)

    plt.title(f"Embedding Visualization ({method.upper()})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()


def analyze_similarity_distribution(embeddings: np.ndarray) -> dict:
    """Analyzes the distribution of pairwise similarities."""
    sim_matrix = embeddings @ embeddings.T
    n = len(embeddings)

    # Get upper triangle (excluding diagonal)
    upper_triangle = sim_matrix[np.triu_indices(n, k=1)]

    return {
        "mean": float(upper_triangle.mean()),
        "std": float(upper_triangle.std()),
        "min": float(upper_triangle.min()),
        "max": float(upper_triangle.max()),
        "percentile_25": float(np.percentile(upper_triangle, 25)),
        "percentile_50": float(np.percentile(upper_triangle, 50)),
        "percentile_75": float(np.percentile(upper_triangle, 75)),
    }


def plot_similarity_histogram(
    embeddings: np.ndarray,
    output_path: str | None = None,
):
    """Plots histogram of pairwise similarities."""
    sim_matrix = embeddings @ embeddings.T
    n = len(embeddings)
    upper_triangle = sim_matrix[np.triu_indices(n, k=1)]

    plt.figure(figsize=(10, 6))
    plt.hist(upper_triangle, bins=50, edgecolor="black", alpha=0.7)
    plt.axvline(
        upper_triangle.mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {upper_triangle.mean():.3f}",
    )
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.title("Pairwise Similarity Distribution")
    plt.legend()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    # Sample pages with different types
    pages = [
        # E-commerce
        (
            "<body><nav><a>Shop</a></nav><main><div class='product'><h1>Laptop</h1><p>$999</p></div></main></body>",
            "ecommerce",
        ),
        (
            "<body><nav><a>Shop</a></nav><main><div class='product'><h1>Phone</h1><p>$599</p></div></main></body>",
            "ecommerce",
        ),
        (
            "<body><nav><a>Store</a></nav><section><div class='item'><h1>Camera</h1></div></section></body>",
            "ecommerce",
        ),
        # Blog
        (
            "<body><header><h1>Blog</h1></header><article><h2>Post 1</h2><p>Content...</p></article></body>",
            "blog",
        ),
        (
            "<body><header><h1>Blog</h1></header><article><h2>Post 2</h2><p>More content...</p></article></body>",
            "blog",
        ),
        (
            "<body><nav><a>Home</a></nav><main><article><h1>News</h1><p>Text...</p></article></main></body>",
            "blog",
        ),
        # Forms
        ("<body><main><form><input/><input/><button>Login</button></form></main></body>", "form"),
        (
            "<body><div><form><input/><textarea></textarea><button>Send</button></form></div></body>",
            "form",
        ),
    ]

    html_docs = [p for p, _ in pages]
    labels = [label for _, label in pages]

    model = WubbaInference("models/best.ckpt", Config())

    print("=== Embedding Quality Analysis ===")
    embeddings = model.predict(html_docs, dim=128).numpy()
    quality = analyze_embedding_quality(embeddings)

    for key, value in quality.items():
        print(f"  {key}: {value:.4f}")

    print("\n=== Matryoshka Dimension Comparison ===")
    compare_matryoshka_dimensions(html_docs, model)

    print("\n=== Similarity Distribution ===")
    sim_stats = analyze_similarity_distribution(embeddings)
    for key, value in sim_stats.items():
        print(f"  {key}: {value:.4f}")

    # Visualization (requires matplotlib)
    print("\n=== Visualization ===")
    try:
        visualize_embeddings(embeddings, labels, method="pca", output_path="embeddings_pca.png")
        plot_similarity_histogram(embeddings, output_path="similarity_hist.png")
    except Exception as e:
        print(f"  Visualization skipped: {e}")
