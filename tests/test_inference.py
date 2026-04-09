from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wubba.config import Config
from wubba.inference import WubbaInference


def make_inference(tmp_path: Path, **config_overrides: object) -> WubbaInference:
    if "matryoshka_dims" in config_overrides and "matryoshka_unlock_epochs" not in config_overrides:
        dims = config_overrides["matryoshka_dims"]
        assert isinstance(dims, list)
        config_overrides["matryoshka_unlock_epochs"] = list(range(0, len(dims) * 10, 10))

    config = Config(
        data_dir=tmp_path / "data",
        model_dir=tmp_path / "models",
        num_workers=0,
        pin_memory=False,
        **config_overrides,
    )
    return WubbaInference("dummy.ckpt", config=config, use_compile=False)


def test_available_dims_reflect_config(tmp_path: Path) -> None:
    inference = make_inference(tmp_path, matryoshka_dims=[16, 32, 64])

    assert inference.available_dims == [16, 32, 64]


def test_predict_returns_empty_tensor_with_requested_dim_for_empty_input(tmp_path: Path) -> None:
    inference = make_inference(tmp_path)
    inference._load_model = lambda: setattr(inference, "trainer", SimpleNamespace(predict=lambda model, dataloader: []))  # type: ignore[method-assign]

    embeddings = inference.predict([], dim=64)

    assert embeddings.shape == (0, 64)


def test_predict_rejects_unknown_matryoshka_dim(tmp_path: Path) -> None:
    inference = make_inference(tmp_path, matryoshka_dims=[32, 64])
    inference.model = object()  # type: ignore[assignment]
    inference.trainer = SimpleNamespace(
        predict=lambda model, dataloader: [torch.ones(1, 64)],
    )
    inference._load_model = lambda: None  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="Available: 32, 64"):
        inference.predict(["<body><div></div></body>"], dim=48)


def test_compute_similarity_uses_normalized_embeddings(tmp_path: Path) -> None:
    inference = make_inference(tmp_path, matryoshka_dims=[2, 4])
    inference.predict = lambda html_documents, batch_size=1024, dim=None: torch.tensor(  # type: ignore[method-assign]
        [[1.0, 0.0], [1.0, 0.0]]
    )

    similarity = inference.compute_similarity("<body></body>", "<body></body>", dim=2)

    assert similarity == pytest.approx(1.0)
