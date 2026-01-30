#!/usr/bin/env python3
"""Batch Processing: Process large HTML datasets efficiently.

Use case: Processing millions of HTML documents for indexing,
analysis, or feature extraction at scale.
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wubba import Config, WubbaInference


@dataclass
class ProcessingStats:
    """Statistics for batch processing."""

    total_docs: int = 0
    processed_docs: int = 0
    failed_docs: int = 0
    total_time_sec: float = 0.0

    @property
    def throughput(self) -> float:
        if self.total_time_sec > 0:
            return self.processed_docs / self.total_time_sec
        return 0.0


def html_file_iterator(
    input_dir: str | Path,
    pattern: str = "*.html",
) -> Iterator[tuple[str, str]]:
    """Iterates over HTML files in a directory."""
    input_dir = Path(input_dir)
    for html_file in input_dir.glob(pattern):
        try:
            content = html_file.read_text(encoding="utf-8", errors="ignore")
            yield str(html_file), content
        except Exception:
            continue


def jsonl_iterator(
    input_file: str | Path,
    html_field: str = "html",
    id_field: str = "id",
) -> Iterator[tuple[str, str]]:
    """Iterates over JSONL file with HTML content."""
    with open(input_file, encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                doc_id = str(record.get(id_field, ""))
                html_content = record.get(html_field, "")
                if html_content:
                    yield doc_id, html_content
            except json.JSONDecodeError:
                continue


class BatchProcessor:
    """Efficient batch processor for HTML embeddings."""

    def __init__(
        self,
        model_path: str,
        dim: int = 64,
        batch_size: int = 512,
        num_workers: int = 4,
    ):
        self.model = WubbaInference(model_path, Config())
        self.dim = dim
        self.batch_size = batch_size
        self.num_workers = num_workers

    def process_iterator(
        self,
        doc_iterator: Iterator[tuple[str, str]],
        output_path: str | Path | None = None,
    ) -> tuple[dict[str, np.ndarray], ProcessingStats]:
        """Processes documents from an iterator."""
        import time

        stats = ProcessingStats()
        all_embeddings: dict[str, np.ndarray] = {}
        start_time = time.time()

        batch_ids: list[str] = []
        batch_docs: list[str] = []

        for doc_id, html_content in doc_iterator:
            batch_ids.append(doc_id)
            batch_docs.append(html_content)
            stats.total_docs += 1

            if len(batch_docs) >= self.batch_size:
                embeddings = self._process_batch(batch_docs)

                for i, doc_id in enumerate(batch_ids):
                    all_embeddings[doc_id] = embeddings[i]

                stats.processed_docs += len(batch_docs)
                batch_ids = []
                batch_docs = []

                # Progress update
                if stats.processed_docs % 10000 == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"Processed {stats.processed_docs} docs ({stats.processed_docs / elapsed:.1f} docs/sec)"
                    )

        # Process remaining
        if batch_docs:
            embeddings = self._process_batch(batch_docs)
            for i, doc_id in enumerate(batch_ids):
                all_embeddings[doc_id] = embeddings[i]
            stats.processed_docs += len(batch_docs)

        stats.total_time_sec = time.time() - start_time

        # Save if output path provided
        if output_path:
            self._save_embeddings(all_embeddings, output_path)

        return all_embeddings, stats

    def _process_batch(self, html_docs: list[str]) -> np.ndarray:
        """Processes a batch of documents."""
        try:
            embeddings = self.model.predict(html_docs, dim=self.dim)
            return embeddings.numpy()
        except Exception as e:
            print(f"Batch processing error: {e}")
            return np.zeros((len(html_docs), self.dim))

    def _save_embeddings(
        self,
        embeddings: dict[str, np.ndarray],
        output_path: str | Path,
    ) -> None:
        """Saves embeddings to disk."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == ".npz":
            np.savez_compressed(
                output_path,
                ids=np.array(list(embeddings.keys())),
                embeddings=np.stack(list(embeddings.values())),
            )
        else:  # JSON
            data = {k: v.tolist() for k, v in embeddings.items()}
            with open(output_path, "w") as f:
                json.dump(data, f)


def process_directory(
    model_path: str,
    input_dir: str,
    output_path: str,
    dim: int = 64,
    batch_size: int = 512,
) -> ProcessingStats:
    """Processes all HTML files in a directory."""
    processor = BatchProcessor(model_path, dim=dim, batch_size=batch_size)
    iterator = html_file_iterator(input_dir)
    _, stats = processor.process_iterator(iterator, output_path)
    return stats


def process_jsonl(
    model_path: str,
    input_file: str,
    output_path: str,
    html_field: str = "html",
    dim: int = 64,
    batch_size: int = 512,
) -> ProcessingStats:
    """Processes HTML content from a JSONL file."""
    processor = BatchProcessor(model_path, dim=dim, batch_size=batch_size)
    iterator = jsonl_iterator(input_file, html_field=html_field)
    _, stats = processor.process_iterator(iterator, output_path)
    return stats


if __name__ == "__main__":
    print("=== Batch Processing Examples ===")

    # Example 1: Process directory
    print("\n1. Process HTML directory:")
    print("""
    stats = process_directory(
        model_path="models/best.ckpt",
        input_dir="data/html_files/",
        output_path="embeddings/output.npz",
        dim=64,
        batch_size=512,
    )
    print(f"Processed {stats.processed_docs} docs at {stats.throughput:.1f} docs/sec")
    """)

    # Example 2: Process JSONL
    print("\n2. Process JSONL file:")
    print("""
    stats = process_jsonl(
        model_path="models/best.ckpt",
        input_file="data/pages.jsonl",
        output_path="embeddings/output.npz",
        html_field="content",
        dim=64,
    )
    """)

    # Example 3: Custom iterator
    print("\n3. Custom processing with iterator:")
    print("""
    processor = BatchProcessor("models/best.ckpt", dim=64, batch_size=512)

    # Custom iterator from database, API, etc.
    def my_iterator():
        for record in database.query("SELECT id, html FROM pages"):
            yield record.id, record.html

    embeddings, stats = processor.process_iterator(my_iterator())
    """)

    # Example 4: Loading saved embeddings
    print("\n4. Load and use saved embeddings:")
    print("""
    # Load NPZ
    data = np.load("embeddings/output.npz")
    ids = data["ids"]
    embeddings = data["embeddings"]

    # Build index for similarity search
    from sklearn.neighbors import NearestNeighbors
    index = NearestNeighbors(n_neighbors=10, metric="cosine")
    index.fit(embeddings)

    # Query
    query_embedding = embeddings[0].reshape(1, -1)
    distances, indices = index.kneighbors(query_embedding)
    """)
