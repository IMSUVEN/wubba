from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch


@dataclass
class Config:
    """Configuration class for the Wubba model and training pipeline.

    This class centralizes all hyperparameters and settings.
    """

    # --- Model Architecture ---
    vocab_size: int = 16
    """Size of the HTML tag vocabulary."""
    embedding_dim: int = 128
    """Dimension of the tag and positional embeddings."""
    transformer_dim: int = 256
    """Dimension of the transformer layers and the final output embedding."""
    transformer_heads: int = 8
    """Number of attention heads in the transformer."""
    transformer_layers: int = 6
    """Number of layers in the transformer encoder."""

    # --- Training ---
    batch_size: int = 1024
    """Number of samples per batch."""
    learning_rate: float = 1e-3
    """Peak learning rate for the OneCycleLR scheduler."""
    num_epochs: int = 100
    """Total number of training epochs."""
    weight_decay: float = 1e-2
    """Weight decay for the AdamW optimizer."""
    max_grad_norm: float = 1.0
    """Maximum norm for gradient clipping."""
    mixed_precision: Literal["bf16-mixed", "16-mixed", "32-true"] = "bf16-mixed"
    """Mixed precision setting for training."""
    gradient_accumulation_steps: int = 1
    """Number of steps to accumulate gradients before updating weights."""

    # --- VICReg Loss ---
    vicreg_invariance_weight: float = 25.0
    """Weight for the invariance term in VICReg loss."""
    vicreg_variance_weight: float = 25.0
    """Weight for the variance term in VICReg loss."""
    vicreg_covariance_weight: float = 1.0
    """Weight for the covariance term in VICReg loss."""
    vicreg_variance_gamma: float = 1.0
    """Target value for the standard deviation in the variance loss."""

    # --- Data Processing ---
    max_depth: int = 256
    """Maximum depth of a node in the DOM tree to consider."""
    max_position: int = 256
    """Maximum position of a node among its siblings to consider."""
    max_sequence_length: int = 256
    """Fixed sequence length for model input."""

    # --- Dataloader ---
    num_workers: int = 8
    """Number of worker processes for data loading."""
    pin_memory: bool = True
    """If True, copies tensors to pinned memory before returning them."""
    prefetch_factor: int = 4
    """Number of batches to prefetch for each worker."""

    # --- Paths ---
    data_dir: Path = field(default_factory=lambda: Path("data"))
    """Directory containing the training and validation data."""
    model_dir: Path = field(default_factory=lambda: Path("models"))
    """Directory to save model checkpoints."""
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    """Directory to save training logs."""

    # --- System ---
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    """The device to use for training and inference."""

    def __post_init__(self):
        """Ensures that path directories exist after initialization."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
