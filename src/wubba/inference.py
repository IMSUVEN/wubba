"""Inference wrapper with quantization and ONNX export."""

from dataclasses import asdict
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from lightning import Trainer
from torch.utils.data import DataLoader

from wubba.config import Config
from wubba.data import HTMLDataProcessor, HTMLLayoutDataset
from wubba.model import WubbaLightningModule


def quantize_model(
    model: WubbaLightningModule,
    backend: Literal["x86", "qnnpack", "onednn"] = "x86",
    calibration_data: list[torch.Tensor] | None = None,
) -> torch.nn.Module:
    """Applies INT8 dynamic quantization to Linear layers."""
    torch.backends.quantized.engine = backend
    model = model.cpu()
    model.eval()

    quantized_model = torch.quantization.quantize_dynamic(
        model.model,  # Quantize the inner Wubba model
        {torch.nn.Linear},  # Quantize Linear layers
        dtype=torch.qint8,
    )

    return quantized_model


def benchmark_quantized(
    original_model: torch.nn.Module,
    quantized_model: torch.nn.Module,
    sample_input: torch.Tensor,
    num_runs: int = 100,
) -> dict[str, float]:
    """Compares latency and size between FP32 and INT8 models."""
    import time

    original_model.eval()
    quantized_model.eval()

    sample_input = sample_input.cpu()

    with torch.no_grad():
        for _ in range(10):
            original_model(sample_input)
            quantized_model(sample_input)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            original_model(sample_input)
    original_time = (time.perf_counter() - start) / num_runs

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            quantized_model(sample_input)
    quantized_time = (time.perf_counter() - start) / num_runs

    def get_model_size(model: torch.nn.Module) -> float:
        param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / (1024 * 1024)  # MB

    return {
        "original_time_ms": original_time * 1000,
        "quantized_time_ms": quantized_time * 1000,
        "speedup": original_time / quantized_time,
        "original_size_mb": get_model_size(original_model),
        "quantized_size_mb": get_model_size(quantized_model),
        "size_reduction": get_model_size(original_model)
        / max(get_model_size(quantized_model), 0.01),
    }


def export_to_onnx(
    model: WubbaLightningModule,
    output_path: str | Path,
    sample_input: torch.Tensor | None = None,
    opset_version: int = 17,
    dynamic_batch: bool = True,
) -> Path:
    """Exports model to ONNX format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.cpu()
    model.eval()

    if sample_input is None:
        batch_size = 2
        seq_len = 256
        feature_dim = model.hparams.get("feature_dim", 10)
        sample_input = torch.randint(0, 100, (batch_size, seq_len, feature_dim), dtype=torch.long)

    dynamic_axes = {}
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch_size", 1: "seq_len"},
            "output": {0: "batch_size"},
        }

    torch.onnx.export(
        model.model,
        sample_input,
        str(output_path),
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )

    return output_path


def validate_onnx(
    onnx_path: str | Path,
    pytorch_model: WubbaLightningModule,
    sample_input: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> dict[str, float]:
    """Validates ONNX output matches PyTorch."""
    import numpy as np

    try:
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError("onnxruntime is required for ONNX validation") from e

    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model.model(sample_input.cpu(), return_projection=False)
        pytorch_output = pytorch_output.numpy()

    ort_session = ort.InferenceSession(str(onnx_path))
    ort_inputs = {"input": sample_input.numpy()}
    onnx_output = ort_session.run(None, ort_inputs)[0]

    max_diff = np.abs(pytorch_output - onnx_output).max()
    mean_diff = np.abs(pytorch_output - onnx_output).mean()
    is_close = np.allclose(pytorch_output, onnx_output, rtol=rtol, atol=atol)

    return {
        "max_diff": float(max_diff),
        "mean_diff": float(mean_diff),
        "is_close": is_close,
    }


class WubbaInference:
    """Inference wrapper with Matryoshka truncation, quantization, and ONNX export."""

    def __init__(
        self,
        model_path: str,
        config: Config | None = None,
        use_compile: bool = True,
        preprocess_mode: Literal["clean", "normalize"] = "normalize",
    ):
        self.config = config or Config()
        self.model_path = model_path
        self.use_compile = use_compile
        self.preprocess_mode = preprocess_mode
        self.model: WubbaLightningModule | None = None
        self.trainer: Trainer | None = None
        self._is_quantized = False

        self.data_processor = HTMLDataProcessor(
            max_depth=self.config.max_depth,
            max_position=self.config.max_position,
            max_sequence_length=self.config.max_sequence_length,
            max_children=self.config.max_children,
            max_siblings=self.config.max_siblings,
        )

    def _load_model(self):
        if self.model is not None:
            return

        self.model = WubbaLightningModule.load_from_checkpoint(
            self.model_path,
            map_location=self.config.device,
            **asdict(self.config),
        )
        self.model.eval()

        if self.use_compile:
            self.model = torch.compile(self.model)  # type: ignore[assignment]

        self.trainer = Trainer(
            accelerator="auto",
            devices=1,  # Inference on a single device
            precision=self.config.mixed_precision,
            logger=False,
            enable_checkpointing=False,
        )

    def _get_transform(self):
        if self.preprocess_mode == "normalize":
            return self.data_processor.html_normalize_to_tensor
        else:
            return self.data_processor.html_clean_to_tensor

    @torch.inference_mode()
    def predict(
        self,
        html_documents: list[str],
        batch_size: int = 1024,
        dim: int | None = None,
    ) -> torch.Tensor:
        """Returns L2-normalized embeddings for HTML documents."""
        self._load_model()

        dataset = HTMLLayoutDataset(
            html_documents,
            transform=self._get_transform(),
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

        assert self.trainer is not None
        predictions = self.trainer.predict(self.model, dataloader)
        if not predictions:
            output_dim = dim if dim is not None else self.config.transformer_dim
            return torch.empty(0, output_dim)

        embeddings = torch.cat(predictions, dim=0)  # type: ignore[arg-type]

        if dim is not None:
            if dim not in self.config.matryoshka_dims:
                available = ", ".join(map(str, self.config.matryoshka_dims))
                raise ValueError(f"dim={dim} not in matryoshka_dims. Available: {available}")
            embeddings = embeddings[:, :dim]

        return F.normalize(embeddings, p=2, dim=-1)

    @torch.inference_mode()
    def predict_single(
        self,
        html_content: str,
        dim: int | None = None,
    ) -> torch.Tensor:
        """Returns embedding for a single HTML document."""
        embeddings = self.predict([html_content], batch_size=1, dim=dim)
        return embeddings[0]

    @torch.inference_mode()
    def compute_similarity(
        self,
        html1: str,
        html2: str,
        dim: int | None = None,
    ) -> float:
        """Returns cosine similarity between two HTML documents."""
        embeddings = self.predict([html1, html2], batch_size=2, dim=dim)
        similarity = F.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0))
        return similarity.item()

    @property
    def available_dims(self) -> list[int]:
        """Available Matryoshka dimensions."""
        return self.config.matryoshka_dims

    def quantize(
        self,
        backend: Literal["x86", "qnnpack", "onednn"] = "x86",
    ) -> "WubbaInference":
        """Applies INT8 quantization (CPU only)."""
        self._load_model()
        assert self.model is not None

        self.model.model = quantize_model(self.model, backend=backend)  # type: ignore[assignment]
        self._is_quantized = True

        return self

    def export_onnx(
        self,
        output_path: str | Path,
        opset_version: int = 17,
    ) -> Path:
        """Exports to ONNX format."""
        self._load_model()
        assert self.model is not None

        return export_to_onnx(
            self.model,
            output_path,
            opset_version=opset_version,
        )

    def benchmark(
        self,
        html_documents: list[str] | None = None,
        num_runs: int = 100,
    ) -> dict[str, float]:
        """Returns latency and throughput metrics."""
        import time

        self._load_model()
        assert self.model is not None

        if html_documents is None:
            sample_input = torch.randint(
                0,
                100,
                (32, self.config.max_sequence_length, self.config.feature_dim),
                dtype=torch.long,
            )
        else:
            transform = self._get_transform()
            tensors = [transform(doc) for doc in html_documents[:32]]
            sample_input = torch.stack(tensors)

        sample_input = sample_input.to(self.config.device)
        self.model.eval()

        with torch.no_grad():
            for _ in range(10):
                self.model(sample_input, return_projection=False)

        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_runs):
                self.model(sample_input, return_projection=False)

        if self.config.device == "cuda":
            torch.cuda.synchronize()

        total_time = time.perf_counter() - start
        avg_time = total_time / num_runs
        throughput = sample_input.size(0) / avg_time

        return {
            "avg_latency_ms": avg_time * 1000,
            "throughput_samples_per_sec": throughput,
            "batch_size": sample_input.size(0),
            "num_runs": num_runs,
        }
