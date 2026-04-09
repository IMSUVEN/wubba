import importlib
from pathlib import Path

import pandas as pd

from wubba.config import Config
from wubba.data import WubbaDataModule


def make_config(tmp_path: Path) -> Config:
    return Config(
        data_dir=tmp_path / "data",
        model_dir=tmp_path / "models",
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        prefetch_factor=2,
    )


def test_data_module_loads_train_and_validation_html(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    train_df = pd.DataFrame({"html": ["<body><div></div></body>", "<body><p></p></body>"]})
    val_df = pd.DataFrame({"html": ["<body><section></section></body>"]})
    train_df.to_parquet(config.data_dir / "train_data.parquet")
    val_df.to_parquet(config.data_dir / "val_data.parquet")

    data_module = WubbaDataModule(config)
    data_module.setup("fit")

    assert data_module.train_dataset is not None
    assert data_module.val_dataset is not None
    assert len(data_module.train_dataset) == 2
    assert len(data_module.val_dataset) == 1


def test_data_module_dataloaders_use_configured_batch_size(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    train_df = pd.DataFrame({"html": ["<body><div></div></body>", "<body><p></p></body>"]})
    val_df = pd.DataFrame({"html": ["<body><section></section></body>"]})
    train_df.to_parquet(config.data_dir / "train_data.parquet")
    val_df.to_parquet(config.data_dir / "val_data.parquet")

    data_module = WubbaDataModule(config)
    data_module.setup("fit")
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    assert train_loader.batch_size == 2
    assert val_loader.batch_size == 2


def test_train_quick_disables_optional_training_features(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    train_module = importlib.import_module("wubba.train")

    def fake_train(**kwargs):
        captured.update(kwargs)
        return "model", "trainer"

    base_config = make_config(tmp_path)

    monkeypatch.setattr(train_module, "train", fake_train)

    model, trainer = train_module.train_quick(base_config, num_epochs=3, batch_size=4)

    config = captured["config"]
    assert model == "model"
    assert trainer == "trainer"
    assert isinstance(config, Config)
    assert config is not base_config
    assert config.num_epochs == 3
    assert config.batch_size == 4
    assert config.enable_multitask is False
    assert config.use_ema is False
    assert config.use_self_paced is False
    assert config.progressive_matryoshka is False
    assert captured["use_curriculum"] is False
    assert captured["use_progressive_matryoshka"] is False
    assert captured["use_ema_checkpoints"] is False
    assert captured["use_collapse_monitor"] is False
    assert captured["use_compile"] is False
