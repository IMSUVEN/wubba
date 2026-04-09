import warnings
from pathlib import Path

import pytest
import torch

from wubba.config import Config
from wubba.inference import export_to_onnx, validate_onnx
from wubba.model import WubbaLightningModule


def make_minimal_model(tmp_path: Path) -> tuple[Config, WubbaLightningModule]:
    config = Config(
        data_dir=tmp_path / "data",
        model_dir=tmp_path / "models",
        transformer_dim=32,
        transformer_heads=4,
        transformer_layers=1,
        embedding_dim=16,
        projection_dim=32,
        max_sequence_length=8,
        max_position=8,
        max_depth=8,
        max_subtree_depth=8,
        matryoshka_dims=[16, 32],
        matryoshka_unlock_epochs=[0, 1],
        batch_size=2,
        num_workers=0,
        pin_memory=False,
    )
    return config, WubbaLightningModule(**config.__dict__)


def test_export_to_onnx_matches_embedding_output(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    config, model = make_minimal_model(tmp_path)
    sample_input = torch.randint(
        0,
        10,
        (2, config.max_sequence_length, config.feature_dim),
        dtype=torch.long,
    )
    output_path = tmp_path / "wubba-smoke.onnx"

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The feature will be removed. Please remove usage of this function",
            category=DeprecationWarning,
        )
        exported_path = export_to_onnx(model, output_path, sample_input=sample_input)
    results = validate_onnx(exported_path, model, sample_input)

    assert exported_path.exists()
    assert results["is_close"]
    assert results["max_diff"] < 1e-4
