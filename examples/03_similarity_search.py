#!/usr/bin/env python3
"""Similarity Search: Find pages similar to a query page.

Use case: Given a page, find other pages with similar layout/structure
for template matching, competitive analysis, or content discovery.
"""

import numpy as np

from wubba import Config, WubbaInference


class HTMLSearchIndex:
    """Simple in-memory search index for HTML documents."""

    def __init__(self, model: WubbaInference, dim: int = 64):
        self.model = model
        self.dim = dim
        self.embeddings: np.ndarray | None = None
        self.documents: list[str] = []
        self.metadata: list[dict] = []

    def add_documents(
        self,
        documents: list[str],
        metadata: list[dict] | None = None,
    ) -> None:
        """Adds documents to the index."""
        new_embeddings = self.model.predict(documents, dim=self.dim).numpy()

        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        self.documents.extend(documents)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(documents))

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Finds the top-k most similar documents to the query."""
        if self.embeddings is None or len(self.documents) == 0:
            return []

        query_embedding = self.model.predict([query], dim=self.dim).numpy()

        # Cosine similarity
        similarities = (self.embeddings @ query_embedding.T).flatten()

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append(
                {
                    "index": int(idx),
                    "similarity": float(similarities[idx]),
                    "metadata": self.metadata[idx],
                }
            )

        return results

    def find_similar_pairs(
        self,
        threshold: float = 0.8,
    ) -> list[tuple[int, int, float]]:
        """Finds all pairs of documents with similarity above threshold."""
        if self.embeddings is None:
            return []

        sim_matrix = self.embeddings @ self.embeddings.T
        pairs = []

        for i in range(len(self.documents)):
            for j in range(i + 1, len(self.documents)):
                if sim_matrix[i, j] >= threshold:
                    pairs.append((i, j, float(sim_matrix[i, j])))

        return sorted(pairs, key=lambda x: x[2], reverse=True)


if __name__ == "__main__":
    model = WubbaInference("models/best.ckpt", Config())
    index = HTMLSearchIndex(model, dim=64)

    # Index some pages with metadata
    pages = [
        "<body><nav><a>Home</a></nav><main><h1>Welcome</h1><p>Content</p></main></body>",
        "<body><nav><a>Products</a></nav><main><div class='grid'><div>Item 1</div><div>Item 2</div></div></main></body>",
        "<body><header><h1>Blog</h1></header><article><h2>Post</h2><p>Text</p></article></body>",
        "<body><nav><a>Home</a></nav><section><h1>About Us</h1><p>Description</p></section></body>",
        "<body><nav><a>Shop</a></nav><main><div class='products'><div>Product A</div></div></main></body>",
    ]

    metadata = [
        {"url": "https://example.com/", "type": "landing"},
        {"url": "https://example.com/products", "type": "listing"},
        {"url": "https://example.com/blog/post-1", "type": "article"},
        {"url": "https://example.com/about", "type": "landing"},
        {"url": "https://example.com/shop", "type": "listing"},
    ]

    index.add_documents(pages, metadata)

    # Search for similar pages
    print("=== Similarity Search ===")
    query = "<body><nav><a>Menu</a></nav><main><h1>Hello World</h1><p>Welcome!</p></main></body>"
    results = index.search(query, top_k=3)

    print("Query: landing page with nav + main content")
    for r in results:
        print(f"  #{r['index']} sim={r['similarity']:.3f} - {r['metadata'].get('url', 'N/A')}")

    # Find all similar pairs
    print("\n=== Similar Pairs (threshold=0.7) ===")
    pairs = index.find_similar_pairs(threshold=0.7)
    for i, j, sim in pairs:
        print(f"  Page {i} <-> Page {j}: similarity={sim:.3f}")
