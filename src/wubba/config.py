from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch

from wubba.const import NUM_SEMANTIC_GROUPS, NUM_TAG_ROLES, VOCAB


@dataclass
class Config:
    """Wubba model and training configuration."""

    # Model Architecture
    vocab_size: int = field(default_factory=lambda: len(VOCAB))
    embedding_dim: int = 128
    transformer_dim: int = 256
    transformer_heads: int = 8
    transformer_layers: int = 6
    num_semantic_groups: int | None = field(default_factory=lambda: NUM_SEMANTIC_GROUPS)
    num_tag_roles: int | None = field(default_factory=lambda: NUM_TAG_ROLES)
    projection_dim: int = 256
    use_cls_token: bool = True
    dropout: float = 0.1

    # RoPE
    rope_position_base: float = 10000.0
    rope_depth_base: float = 1000.0
    rope_subtree_base: float = 500.0

    # Matryoshka
    use_matryoshka: bool = True
    matryoshka_dims: list[int] = field(default_factory=lambda: [32, 64, 128, 256])
    matryoshka_weights: list[float] | None = None

    # Features
    feature_dim: int = 10
    use_extended_features: bool = False
    max_subtree_depth: int = 32

    # Training
    batch_size: int = 1024
    learning_rate: float = 1e-3
    num_epochs: int = 100
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16-mixed"
    gradient_accumulation_steps: int = 1

    # LR Scheduler
    lr_scheduler: Literal["onecycle", "cosine_restarts"] = "cosine_restarts"
    restart_epochs: list[int] = field(default_factory=lambda: [30, 60, 80])
    min_lr_ratio: float = 0.01

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.999
    ema_update_after_step: int = 100

    # Self-Paced Learning
    use_self_paced: bool = True
    self_paced_lambda_init: float = 2.0
    self_paced_lambda_growth: float = 1.05
    self_paced_mode: Literal["linear", "log", "mixture"] = "mixture"

    # Loss Function
    loss_type: Literal["vicreg", "infonce", "hybrid", "matryoshka_hybrid", "enhanced_hybrid"] = (
        "enhanced_hybrid"
    )

    # VICReg
    vicreg_invariance_weight: float = 25.0
    vicreg_variance_weight: float = 25.0
    vicreg_covariance_weight: float = 1.0
    vicreg_variance_gamma: float = 1.0

    # InfoNCE
    infonce_temperature: float = 0.07

    # Hybrid Loss
    hybrid_vicreg_weight: float = 1.0
    hybrid_infonce_weight: float = 0.5

    # Enhanced Loss
    use_spectral_loss: bool = True
    spectral_weight: float = 0.1
    use_hard_negative: bool = True
    hard_negative_ratio: float = 0.2
    hard_negative_mix_alpha: float = 0.5
    use_alignment_uniformity: bool = True
    alignment_weight: float = 1.0
    uniformity_weight: float = 1.0
    uniformity_t: float = 2.0

    # Multi-Task Learning
    enable_multitask: bool = True
    mnp_enabled: bool = True
    mnp_weight: float = 0.3
    mnp_mask_prob: float = 0.15
    structure_pred_enabled: bool = True
    structure_pred_weight: float = 0.2
    progressive_matryoshka: bool = True
    matryoshka_unlock_epochs: list[int] = field(default_factory=lambda: [0, 20, 40, 60])

    # Data Processing
    max_depth: int = 256
    max_position: int = 256
    max_sequence_length: int = 256
    max_children: int = 64
    max_siblings: int = 64

    # Augmentation
    aug_strong_prob: float = 0.8
    aug_depth_truncate_max: int = 8
    aug_sibling_drop_prob: float = 0.3
    aug_subtree_sample_ratio: float = 0.7
    aug_semantic_replace_prob: float = 0.2
    aug_skeleton_keep_groups: list[str] = field(
        default_factory=lambda: ["container", "navigation", "list", "table", "layout"]
    )
    aug_tier_weights: tuple[float, float, float] = (0.2, 0.5, 0.3)
    use_contextual_aug: bool = True
    contextual_structure_prob: float = 0.3
    contextual_semantic_prob: float = 0.2
    use_tree_mixup: bool = True
    mixup_alpha: float = 0.2
    mixup_prob: float = 0.3

    # Dataloader
    num_workers: int = 8
    pin_memory: bool = True
    prefetch_factor: int = 4

    # Monitoring
    log_embedding_metrics: bool = True
    embedding_log_interval: int = 100
    collapse_detection: bool = True
    collapse_rank_threshold: float = 0.3

    # Paths
    data_dir: Path = field(default_factory=lambda: Path("data"))
    model_dir: Path = field(default_factory=lambda: Path("models"))

    # System
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        """Validates configuration and creates directories."""
        # Basic parameter validation
        assert self.transformer_dim % self.transformer_heads == 0, (
            f"transformer_dim ({self.transformer_dim}) must be divisible by "
            f"transformer_heads ({self.transformer_heads})"
        )
        assert self.learning_rate > 0, f"learning_rate must be positive, got {self.learning_rate}"
        assert self.batch_size > 0, f"batch_size must be positive, got {self.batch_size}"

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        if self.use_extended_features:
            self.feature_dim = 15

        if self.progressive_matryoshka:
            assert len(self.matryoshka_unlock_epochs) == len(self.matryoshka_dims), (
                f"matryoshka_unlock_epochs ({len(self.matryoshka_unlock_epochs)}) "
                f"must match matryoshka_dims ({len(self.matryoshka_dims)})"
            )
