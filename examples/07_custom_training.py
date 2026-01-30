#!/usr/bin/env python3
"""Custom Training: Advanced training configurations and callbacks.

Use case: Researchers customizing training for specific experiments,
ablation studies, or hyperparameter tuning.
"""

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from wubba import (
    CollapseMonitorCallback,
    Config,
    CurriculumLearningCallback,
    EMACheckpointCallback,
    ProgressiveMatryoshkaCallback,
    WubbaDataModule,
    WubbaLightningModule,
)


def train_with_custom_callbacks(
    config: Config,
    experiment_name: str = "custom_training",
) -> tuple[WubbaLightningModule, L.Trainer]:
    """Trains with custom callback configuration."""
    # Data module (uses config.data_dir)
    data_module = WubbaDataModule(config)

    # Model
    model = WubbaLightningModule(**config.__dict__)

    # Callbacks
    callbacks = [
        # Curriculum learning: easy -> hard samples
        CurriculumLearningCallback(
            num_epochs=config.num_epochs,
            phase1_ratio=0.2,
            phase2_ratio=0.3,
            phase1_aug_strong_prob=0.3,
            phase2_aug_strong_prob=0.8,
            phase3_aug_strong_prob=0.5,
        ),
        # Progressive Matryoshka: unlock dimensions gradually
        ProgressiveMatryoshkaCallback(
            matryoshka_dims=config.matryoshka_dims,
            unlock_epochs=config.matryoshka_unlock_epochs,
        ),
        # EMA checkpointing
        EMACheckpointCallback(
            save_dir=str(config.model_dir / "ema"),
        ),
        # Collapse detection
        CollapseMonitorCallback(
            check_interval=100,
            rank_threshold=0.3,
            auto_adjust=True,
        ),
        # Learning rate monitoring
        LearningRateMonitor(logging_interval="step"),
        # Early stopping on validation loss
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min",
        ),
        # Checkpointing
        ModelCheckpoint(
            dirpath=str(config.model_dir),
            filename="best-{epoch}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
        ),
    ]

    # Logger
    logger = TensorBoardLogger(
        save_dir=str(config.log_dir),
        name=experiment_name,
    )

    # Trainer
    trainer = L.Trainer(
        max_epochs=config.num_epochs,
        accelerator="auto",
        devices=1,
        precision=config.mixed_precision,
        accumulate_grad_batches=config.gradient_accumulation_steps,
        gradient_clip_val=config.max_grad_norm,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
    )

    # Train
    trainer.fit(model, data_module)

    return model, trainer


def ablation_study(
    base_config: Config,
    variations: dict[str, dict],
) -> dict[str, dict]:
    """Runs ablation study with different configurations."""
    results = {}

    for name, overrides in variations.items():
        print(f"\n=== Running: {name} ===")

        # Create modified config
        config_dict = base_config.__dict__.copy()
        config_dict.update(overrides)
        config = Config(**{k: v for k, v in config_dict.items() if not k.startswith("_")})

        # Quick training (uses config.data_dir)
        data_module = WubbaDataModule(config)
        model = WubbaLightningModule(**config.__dict__)

        trainer = L.Trainer(
            max_epochs=10,  # Short for ablation
            accelerator="auto",
            devices=1,
            precision=config.mixed_precision,
            enable_progress_bar=False,
            enable_checkpointing=False,
            logger=False,
        )

        trainer.fit(model, data_module)

        # Collect metrics
        results[name] = {
            "final_loss": trainer.callback_metrics.get("train_loss", float("nan")),
            "config": overrides,
        }

    return results


if __name__ == "__main__":
    # Base configuration
    config = Config(
        # Model
        transformer_dim=256,
        transformer_layers=6,
        use_matryoshka=True,
        matryoshka_dims=[32, 64, 128, 256],
        # Training
        loss_type="enhanced_hybrid",
        num_epochs=100,
        batch_size=512,
        learning_rate=1e-3,
        # Advanced features
        use_ema=True,
        use_self_paced=True,
        progressive_matryoshka=True,
        enable_multitask=True,
        # Data
        use_extended_features=True,
        use_contextual_aug=True,
        use_tree_mixup=True,
    )

    print("=== Custom Training with All Callbacks ===")
    print(
        f"Config: {config.loss_type}, {config.transformer_dim}d, {config.transformer_layers} layers"
    )

    # Example: Run training (set config.data_dir before calling)
    # from pathlib import Path
    # config.data_dir = Path("data/html_samples/")
    # model, trainer = train_with_custom_callbacks(config)

    # Example: Ablation study
    print("\n=== Ablation Study Design ===")
    ablations = {
        "baseline": {},
        "no_ema": {"use_ema": False},
        "no_multitask": {"enable_multitask": False},
        "no_self_paced": {"use_self_paced": False},
        "vicreg_only": {"loss_type": "vicreg"},
        "infonce_only": {"loss_type": "infonce"},
        "small_model": {"transformer_dim": 128, "transformer_layers": 4},
        "large_model": {"transformer_dim": 512, "transformer_layers": 8},
    }

    print("Variations to test:")
    for name, overrides in ablations.items():
        print(f"  - {name}: {overrides if overrides else '(baseline)'}")

    # config.data_dir = Path("data/html_samples/")
    # results = ablation_study(config, ablations)
