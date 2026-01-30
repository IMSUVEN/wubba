from dataclasses import asdict
from typing import Tuple

import lightning as L
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)

from wubba.config import Config
from wubba.data import WubbaDataModule
from wubba.model import WubbaLightningModule


def train(
    config: Config = Config(), resume: bool = False
) -> Tuple[WubbaLightningModule, L.Trainer]:
    """Sets up and runs the training process.

    This function orchestrates the entire training pipeline, including:
    1. Setting up the data module.
    2. Initializing the self-contained Lightning module.
    3. Setting up Lightning callbacks.
    4. Creating and running the Lightning Trainer.

    Args:
        config: A Config object containing all hyperparameters.
        resume: If True, attempts to resume training from the last checkpoint.

    Returns:
        A tuple containing the trained Lightning module and the Trainer instance.
    """
    data_module = WubbaDataModule(config)

    # The model is initialized with all hyperparameters from the config.
    # The LightningModule is self-contained and configures its own components.
    model = WubbaLightningModule(**asdict(config))

    # Compile the model for performance
    model = torch.compile(model)

    # Configure callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=config.model_dir,
            filename="wubba-{epoch:02d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        EarlyStopping(monitor="val/loss", mode="min", patience=10, min_delta=1e-5),
        LearningRateMonitor(logging_interval="step"),
        RichModelSummary(max_depth=2),
        RichProgressBar(
            leave=True,
        ),
    ]

    trainer = L.Trainer(
        max_epochs=config.num_epochs,
        accelerator="auto",
        devices="auto",
        precision=config.mixed_precision,
        accumulate_grad_batches=config.gradient_accumulation_steps,
        gradient_clip_val=config.max_grad_norm or None,
        callbacks=callbacks,
    )

    ckpt_path = config.model_dir / "last.ckpt"
    if resume and ckpt_path.exists():
        trainer.fit(model, data_module, ckpt_path=str(ckpt_path))
    else:
        trainer.fit(model, data_module)

    return model, trainer
