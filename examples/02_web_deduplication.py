#!/usr/bin/env python3
"""Web Deduplication: Identify duplicate or near-duplicate pages.

Use case: Web crawlers need to detect when pages share the same template
or are exact/near duplicates, even when content differs.
"""

from sklearn.cluster import DBSCAN

from wubba import Config, WubbaInference


def find_duplicates(
    html_pages: list[str],
    model: WubbaInference,
    threshold: float = 0.95,
    dim: int = 64,
) -> list[set[int]]:
    """Finds groups of duplicate pages based on embedding similarity.

    Args:
        html_pages: List of HTML strings.
        model: Wubba inference model.
        threshold: Cosine similarity threshold for duplicates.
        dim: Embedding dimension (smaller = faster).

    Returns:
        List of sets, each containing indices of duplicate pages.
    """
    embeddings = model.predict(html_pages, dim=dim).numpy()

    # Compute pairwise cosine similarity
    similarity_matrix = embeddings @ embeddings.T

    # Find duplicate pairs
    duplicate_groups: list[set[int]] = []
    visited = set()

    for i in range(len(html_pages)):
        if i in visited:
            continue

        group = {i}
        for j in range(i + 1, len(html_pages)):
            if similarity_matrix[i, j] >= threshold:
                group.add(j)
                visited.add(j)

        if len(group) > 1:
            duplicate_groups.append(group)
        visited.add(i)

    return duplicate_groups


def cluster_by_template(
    html_pages: list[str],
    model: WubbaInference,
    eps: float = 0.15,
    min_samples: int = 2,
    dim: int = 64,
) -> dict[int, list[int]]:
    """Clusters pages by template using DBSCAN.

    Args:
        html_pages: List of HTML strings.
        model: Wubba inference model.
        eps: DBSCAN epsilon (max distance between samples).
        min_samples: Minimum samples per cluster.
        dim: Embedding dimension.

    Returns:
        Dict mapping cluster_id -> list of page indices.
    """
    embeddings = model.predict(html_pages, dim=dim).numpy()

    # DBSCAN with cosine distance
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = clustering.fit_predict(embeddings)

    clusters: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(idx)

    return clusters


if __name__ == "__main__":
    # Example pages with similar templates
    pages = [
        # Product pages (same template)
        "<body><nav><a>Shop</a></nav><main><div class='product'><h1>iPhone</h1><p>$999</p></div></main></body>",
        "<body><nav><a>Shop</a></nav><main><div class='product'><h1>MacBook</h1><p>$1299</p></div></main></body>",
        "<body><nav><a>Shop</a></nav><main><div class='product'><h1>iPad</h1><p>$599</p></div></main></body>",
        # Blog pages (different template)
        "<body><header><h1>Blog</h1></header><article><h2>Post 1</h2><p>Content...</p></article></body>",
        "<body><header><h1>Blog</h1></header><article><h2>Post 2</h2><p>More content...</p></article></body>",
        # Unique page
        "<body><form><input type='text'/><button>Submit</button></form></body>",
    ]

    model = WubbaInference("models/best.ckpt", Config())

    # Find exact/near duplicates
    print("=== Duplicate Detection ===")
    duplicates = find_duplicates(pages, model, threshold=0.9)
    for group in duplicates:
        print(f"Duplicate group: {group}")

    # Cluster by template
    print("\n=== Template Clustering ===")
    clusters = cluster_by_template(pages, model, eps=0.2)
    for cluster_id, indices in clusters.items():
        label = "Noise" if cluster_id == -1 else f"Template {cluster_id}"
        print(f"{label}: pages {indices}")
