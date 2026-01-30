"""Training pipeline with curriculum learning, progressive Matryoshka, and EMA."""

from dataclasses import asdict
from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)

from wubba.config import Config
from wubba.data import WubbaDataModule
from wubba.model import WubbaLightningModule


class CurriculumLearningCallback(Callback):
    """Adjusts augmentation and loss weights across training phases."""

    def __init__(
        self,
        num_epochs: int,
        phase1_ratio: float = 0.3,
        phase2_ratio: float = 0.4,
        phase1_aug_strong_prob: float = 0.3,
        phase1_vicreg_variance_weight: float = 35.0,
        phase1_vicreg_invariance_weight: float = 15.0,
        phase1_infonce_weight: float = 0.2,
        phase2_aug_strong_prob: float = 0.8,
        phase2_vicreg_variance_weight: float = 25.0,
        phase2_vicreg_invariance_weight: float = 25.0,
        phase2_infonce_weight: float = 0.5,
        phase3_aug_strong_prob: float = 0.5,
        phase3_vicreg_variance_weight: float = 15.0,
        phase3_vicreg_invariance_weight: float = 35.0,
        phase3_infonce_weight: float = 0.8,
    ):
        super().__init__()
        self.num_epochs = num_epochs
        self.phase1_end = int(num_epochs * phase1_ratio)
        self.phase2_end = int(num_epochs * (phase1_ratio + phase2_ratio))

        # Phase configurations
        self.phase_configs = {
            1: {
                "aug_strong_prob": phase1_aug_strong_prob,
                "vicreg_variance_weight": phase1_vicreg_variance_weight,
                "vicreg_invariance_weight": phase1_vicreg_invariance_weight,
                "infonce_weight": phase1_infonce_weight,
            },
            2: {
                "aug_strong_prob": phase2_aug_strong_prob,
                "vicreg_variance_weight": phase2_vicreg_variance_weight,
                "vicreg_invariance_weight": phase2_vicreg_invariance_weight,
                "infonce_weight": phase2_infonce_weight,
            },
            3: {
                "aug_strong_prob": phase3_aug_strong_prob,
                "vicreg_variance_weight": phase3_vicreg_variance_weight,
                "vicreg_invariance_weight": phase3_vicreg_invariance_weight,
                "infonce_weight": phase3_infonce_weight,
            },
        }

    def _get_phase(self, epoch: int) -> int:
        if epoch < self.phase1_end:
            return 1
        elif epoch < self.phase2_end:
            return 2
        else:
            return 3

    def on_train_epoch_start(
        self,
        trainer: L.Trainer,
        pl_module: WubbaLightningModule,
    ):
        current_epoch = trainer.current_epoch
        phase = self._get_phase(current_epoch)
        config = self.phase_configs[phase]

        # Update datamodule's processor augmentation strength
        if hasattr(trainer, "datamodule") and trainer.datamodule is not None:
            processor = trainer.datamodule.data_processor
            processor.aug_strong_prob = config["aug_strong_prob"]

        criterion = pl_module.criterion

        if hasattr(criterion, "variance_weight"):
            criterion.variance_weight = config["vicreg_variance_weight"]
            criterion.invariance_weight = config["vicreg_invariance_weight"]

        if hasattr(criterion, "vicreg"):
            criterion.vicreg.variance_weight = config["vicreg_variance_weight"]
            criterion.vicreg.invariance_weight = config["vicreg_invariance_weight"]

        if hasattr(criterion, "hybrid_loss") and hasattr(criterion.hybrid_loss, "vicreg"):
            criterion.hybrid_loss.vicreg.variance_weight = config["vicreg_variance_weight"]
            criterion.hybrid_loss.vicreg.invariance_weight = config["vicreg_invariance_weight"]

        if hasattr(criterion, "infonce_weight"):
            criterion.infonce_weight = config["infonce_weight"]

        pl_module.log("curriculum/phase", float(phase), on_epoch=True, prog_bar=False)
        pl_module.log(
            "curriculum/aug_strong_prob",
            config["aug_strong_prob"],
            on_epoch=True,
            prog_bar=False,
        )


class ProgressiveMatryoshkaCallback(Callback):
    """Gradually unlocks higher embedding dimensions during training."""

    def __init__(
        self,
        matryoshka_dims: list[int],
        unlock_epochs: list[int],
    ):
        super().__init__()
        self.matryoshka_dims = sorted(matryoshka_dims)
        self.unlock_epochs = unlock_epochs

    def on_train_epoch_start(
        self,
        trainer: L.Trainer,
        pl_module: WubbaLightningModule,
    ):
        current_epoch = trainer.current_epoch
        active_dims = []
        for i, unlock_epoch in enumerate(self.unlock_epochs):
            if current_epoch >= unlock_epoch and i < len(self.matryoshka_dims):
                active_dims.append(self.matryoshka_dims[i])

        if not active_dims:
            active_dims = [self.matryoshka_dims[0]]

        if hasattr(pl_module, "active_matryoshka_dims"):
            old_dims = pl_module.active_matryoshka_dims.copy()
            pl_module.active_matryoshka_dims = sorted(active_dims)

            if len(active_dims) > len(old_dims):
                new_dim = max(active_dims)
                pl_module.log("matryoshka/unlocked_dim", float(new_dim), on_epoch=True)

        if hasattr(pl_module.criterion, "active_dims"):
            pl_module.criterion.active_dims = sorted(active_dims)

        pl_module.log("matryoshka/n_active_dims", float(len(active_dims)), on_epoch=True)


class EMACheckpointCallback(Callback):
    """Saves separate checkpoints for the EMA model."""

    def __init__(
        self,
        save_dir: str,
        save_interval: int = 10,
    ):
        super().__init__()
        self.save_dir = save_dir
        self.save_interval = save_interval

    def on_train_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: WubbaLightningModule,
    ):
        if pl_module.ema_model is None:
            return

        epoch = trainer.current_epoch
        if (epoch + 1) % self.save_interval == 0:
            pl_module.ema_model.apply_shadow()
            try:
                ckpt_path = f"{self.save_dir}/wubba-ema-epoch{epoch:03d}.ckpt"
                trainer.save_checkpoint(ckpt_path)
            finally:
                pl_module.ema_model.restore()


class CollapseMonitorCallback(Callback):
    """Monitors and auto-adjusts on representation collapse."""

    def __init__(
        self,
        check_interval: int = 50,
        rank_threshold: float = 0.3,
        auto_adjust: bool = True,
    ):
        super().__init__()
        self.check_interval = check_interval
        self.rank_threshold = rank_threshold
        self.auto_adjust = auto_adjust
        self._step_count = 0
        self._collapse_count = 0

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: WubbaLightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ):
        self._step_count += 1

        if self._step_count % self.check_interval != 0:
            return

        if pl_module.collapse_detector is None:
            return

        detector = pl_module.collapse_detector

        if detector.history:
            latest = detector.history[-1]
            if latest["rank_ratio"] < self.rank_threshold:
                self._collapse_count += 1
                pl_module.log("collapse/warning_count", float(self._collapse_count))

                if self.auto_adjust and self._collapse_count >= 3:
                    criterion = pl_module.criterion
                    if hasattr(criterion, "vicreg"):
                        old_weight = float(criterion.vicreg.variance_weight)
                        criterion.vicreg.variance_weight = min(50.0, old_weight * 1.2)
                        pl_module.log(
                            "collapse/adjusted_variance_weight", criterion.vicreg.variance_weight
                        )
                    self._collapse_count = 0


def train(
    config: Config | None = None,
    resume: bool = False,
    use_curriculum: bool = True,
    use_progressive_matryoshka: bool | None = None,
    use_ema_checkpoints: bool = True,
    use_collapse_monitor: bool = True,
    use_compile: bool = True,
) -> tuple[WubbaLightningModule, L.Trainer]:
    """Runs the full training pipeline with all enhancements."""
    if config is None:
        config = Config()

    if use_progressive_matryoshka is None:
        use_progressive_matryoshka = config.progressive_matryoshka

    data_module = WubbaDataModule(config)
    model = WubbaLightningModule(**asdict(config))

    if use_compile:
        model = torch.compile(model)  # type: ignore[assignment]

    callbacks: list[Callback] = [
        ModelCheckpoint(
            dirpath=config.model_dir,
            filename="wubba-{epoch:02d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val/loss",
            mode="min",
            patience=15,
            min_delta=1e-5,
        ),
        LearningRateMonitor(logging_interval="step"),
        RichModelSummary(max_depth=2),
        RichProgressBar(leave=True),
    ]

    if use_curriculum:
        callbacks.append(
            CurriculumLearningCallback(
                num_epochs=config.num_epochs,
                phase1_ratio=0.3,
                phase2_ratio=0.4,
            )
        )

    if use_progressive_matryoshka:
        callbacks.append(
            ProgressiveMatryoshkaCallback(
                matryoshka_dims=config.matryoshka_dims,
                unlock_epochs=config.matryoshka_unlock_epochs,
            )
        )

    if use_ema_checkpoints and config.use_ema:
        callbacks.append(
            EMACheckpointCallback(
                save_dir=str(config.model_dir),
                save_interval=10,
            )
        )

    if use_collapse_monitor and config.collapse_detection:
        callbacks.append(
            CollapseMonitorCallback(
                check_interval=100,
                rank_threshold=config.collapse_rank_threshold,
                auto_adjust=True,
            )
        )

    trainer = L.Trainer(
        max_epochs=config.num_epochs,
        accelerator="auto",
        devices="auto",
        precision=config.mixed_precision,
        accumulate_grad_batches=config.gradient_accumulation_steps,
        gradient_clip_val=config.max_grad_norm or None,
        callbacks=callbacks,
        log_every_n_steps=10,
    )

    ckpt_path = config.model_dir / "last.ckpt"
    if resume and ckpt_path.exists():
        trainer.fit(model, data_module, ckpt_path=str(ckpt_path))
    else:
        trainer.fit(model, data_module)

    return model, trainer


def train_quick(
    config: Config | None = None,
    num_epochs: int = 10,
    batch_size: int = 256,
) -> tuple[WubbaLightningModule, L.Trainer]:
    """Simplified training for development and testing."""
    from copy import deepcopy

    config = deepcopy(config) if config is not None else Config()

    config.num_epochs = num_epochs
    config.batch_size = batch_size
    config.enable_multitask = False
    config.use_ema = False
    config.use_self_paced = False
    config.progressive_matryoshka = False

    return train(
        config=config,
        use_curriculum=False,
        use_progressive_matryoshka=False,
        use_ema_checkpoints=False,
        use_collapse_monitor=False,
        use_compile=False,
    )
