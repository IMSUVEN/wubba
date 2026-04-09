from pathlib import Path

import pytest

from wubba.config import Config


def test_config_updates_feature_dim_when_extended_features_enabled(tmp_path: Path) -> None:
    config = Config(
        data_dir=tmp_path / "data",
        model_dir=tmp_path / "models",
        use_extended_features=True,
    )

    assert config.feature_dim == 15
    assert config.data_dir.exists()
    assert config.model_dir.exists()


def test_config_rejects_incompatible_transformer_dimensions(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="transformer_dim"):
        Config(
            data_dir=tmp_path / "data",
            model_dir=tmp_path / "models",
            transformer_dim=250,
            transformer_heads=8,
        )


def test_config_requires_matching_matryoshka_schedule_lengths(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="matryoshka_unlock_epochs"):
        Config(
            data_dir=tmp_path / "data",
            model_dir=tmp_path / "models",
            progressive_matryoshka=True,
            matryoshka_dims=[32, 64, 128],
            matryoshka_unlock_epochs=[0, 20],
        )

