#!/usr/bin/env python3
"""Production Deployment: Optimize model for production use.

Use case: Deploy Wubba in production with quantization and ONNX export
for faster inference and cross-platform compatibility.
"""

from pathlib import Path

from wubba import Config, WubbaInference
from wubba.inference import validate_onnx


def optimize_for_cpu_deployment(
    checkpoint_path: str,
    output_dir: str = "models/optimized/",
) -> dict:
    """Optimizes model for CPU deployment with INT8 quantization."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model (quantize() automatically loads if needed)
    inference = WubbaInference(checkpoint_path, Config(), use_compile=False)

    # Quantize
    print("Applying INT8 quantization...")
    inference.quantize(backend="x86")

    # Benchmark
    print("Benchmarking...")
    metrics = inference.benchmark(num_runs=50)

    print(f"  Latency: {metrics['avg_latency_ms']:.2f} ms")
    print(f"  Throughput: {metrics['throughput_samples_per_sec']:.1f} samples/sec")

    return metrics


def export_for_cross_platform(
    checkpoint_path: str,
    output_path: str = "models/wubba.onnx",
) -> Path:
    """Exports model to ONNX for cross-platform deployment."""
    inference = WubbaInference(checkpoint_path, Config(), use_compile=False)

    print("Exporting to ONNX...")
    onnx_path = inference.export_onnx(output_path)

    print(f"  Exported to: {onnx_path}")
    print(f"  Size: {onnx_path.stat().st_size / 1024 / 1024:.2f} MB")

    return onnx_path


def validate_onnx_export(
    checkpoint_path: str,
    onnx_path: str,
) -> bool:
    """Validates ONNX export matches PyTorch output."""
    import torch

    from wubba.model import WubbaLightningModule

    print("Validating ONNX export...")

    model = WubbaLightningModule.load_from_checkpoint(checkpoint_path)
    sample_input = torch.randint(0, 100, (4, 256, 10), dtype=torch.long)

    results = validate_onnx(onnx_path, model, sample_input)

    print(f"  Max diff: {results['max_diff']:.6f}")
    print(f"  Mean diff: {results['mean_diff']:.6f}")
    print(f"  Match: {'✓' if results['is_close'] else '✗'}")

    return results["is_close"]


class ProductionInference:
    """Production-ready inference with batching and caching."""

    def __init__(
        self,
        model_path: str,
        use_quantization: bool = True,
        dim: int = 64,
    ):
        self.inference = WubbaInference(
            model_path,
            Config(),
            use_compile=True,
        )

        if use_quantization:
            self.inference.quantize()

        self.dim = dim
        self._cache: dict[int, list[float]] = {}

    def get_embedding(self, html: str, use_cache: bool = True) -> list[float]:
        """Gets embedding with optional caching."""
        cache_key = hash(html)

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        embedding = self.inference.predict([html], dim=self.dim)[0].tolist()

        if use_cache:
            self._cache[cache_key] = embedding

        return embedding

    def get_embeddings_batch(
        self,
        html_docs: list[str],
        batch_size: int = 256,
    ) -> list[list[float]]:
        """Gets embeddings for a batch of documents."""
        embeddings = self.inference.predict(
            html_docs,
            batch_size=batch_size,
            dim=self.dim,
        )
        return embeddings.tolist()

    def clear_cache(self):
        """Clears the embedding cache."""
        self._cache.clear()


if __name__ == "__main__":
    checkpoint = "models/best.ckpt"

    # CPU optimization
    print("=== CPU Optimization (INT8 Quantization) ===")
    try:
        metrics = optimize_for_cpu_deployment(checkpoint)
    except FileNotFoundError:
        print("  Checkpoint not found, skipping...")

    # ONNX export
    print("\n=== ONNX Export ===")
    try:
        onnx_path = export_for_cross_platform(checkpoint)

        # Validate
        print("\n=== ONNX Validation ===")
        validate_onnx_export(checkpoint, str(onnx_path))
    except FileNotFoundError:
        print("  Checkpoint not found, skipping...")
    except ImportError as e:
        print(f"  Optional dependency not installed: {e}")

    # Production usage example
    print("\n=== Production Usage Example ===")
    print("""
    # Initialize once
    inference = ProductionInference("models/best.ckpt", use_quantization=True, dim=64)

    # Single document
    embedding = inference.get_embedding(html_doc)

    # Batch processing
    embeddings = inference.get_embeddings_batch(html_docs, batch_size=256)
    """)
