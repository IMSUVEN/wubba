import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from wubba.utils import create_mask


class PositionalEncoding(nn.Module):
    """Injects positional information into the embeddings.

    Uses sinusoidal functions to create separate positional encodings for
    node depth and sibling position, then concatenates them.

    Attributes:
        depth_pe: Positional encodings for node depth.
        position_pe: Positional encodings for sibling position.
    """

    def __init__(self, embedding_dim: int, max_depth: int, max_position: int):
        """Initializes the PositionalEncoding layer.

        Args:
            embedding_dim: The total dimension of the embedding. Must be even.
            max_depth: The maximum depth to encode.
            max_position: The maximum sibling position to encode.
        """
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError(f"embedding_dim must be even, but got {embedding_dim}")

        half_dim = embedding_dim // 2
        self.register_buffer("depth_pe", self._create_pe_matrix(max_depth, half_dim))
        self.register_buffer(
            "position_pe", self._create_pe_matrix(max_position, half_dim)
        )

    def _create_pe_matrix(self, max_len: int, dim: int) -> torch.Tensor:
        """Creates a positional encoding matrix of size (max_len, dim)."""
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(
        self, depth: torch.Tensor, position: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Looks up and combines positional encodings.

        Args:
            depth: A tensor of node depths, shape (batch_size, seq_len).
            position: A tensor of node positions, shape (batch_size, seq_len).
            mask: An optional float tensor to zero out padded positions.

        Returns:
            A combined positional encoding tensor of shape
            (batch_size, seq_len, embedding_dim).
        """
        depth_pe = self.depth_pe[depth.clamp(0, self.depth_pe.size(0) - 1)]
        pos_pe = self.position_pe[position.clamp(0, self.position_pe.size(0) - 1)]
        pe = torch.cat([depth_pe, pos_pe], dim=-1)

        if mask is not None:
            pe = pe * mask.unsqueeze(-1)

        return pe


class Wubba(nn.Module):
    """The main encoder model for HTML documents.

    This model uses a Transformer encoder to process a sequence of HTML node
    features (tag, depth, position) and produce a fixed-size document embedding.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        transformer_dim: int,
        transformer_heads: int,
        transformer_layers: int,
        max_depth: int,
        max_position: int,
    ):
        super().__init__()
        self.tag_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.positional_encoding = PositionalEncoding(
            embedding_dim, max_depth, max_position
        )
        self.input_projection = nn.Linear(embedding_dim, transformer_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Performs the forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, seq_len, 3), where the
               last dimension contains [tag_id, depth, position].
            mask: Optional float tensor indicating valid (1.0) vs.
                  padded (0.0) tokens.

        Returns:
            A document embedding of shape (batch_size, transformer_dim).
        """
        tags, depth, position = x[..., 0], x[..., 1], x[..., 2]

        tag_embeds = self.tag_embedding(tags)
        pos_embeds = self.positional_encoding(depth, position, mask)
        embeddings = self.input_projection(tag_embeds + pos_embeds)

        # PyTorch's TransformerEncoder uses `True` for masked positions.
        attention_mask = None if mask is None else mask == 0

        encoded = self.transformer(embeddings, src_key_padding_mask=attention_mask)

        # Masked average pooling
        if mask is not None:
            encoded = encoded * mask.unsqueeze(-1)
            # Add epsilon to avoid division by zero for empty sequences
            pooled = encoded.sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)
        else:
            pooled = encoded.mean(dim=1)

        return pooled


class VICRegLoss(nn.Module):
    """Implements the VICReg (Variance-Invariance-Covariance Regularization) loss.

    This loss function is designed for self-supervised learning. It encourages
    the embeddings of two augmented views of the same sample to be similar
    (invariance), while preventing the model from collapsing to a trivial
    solution by maintaining variance in the embeddings and decorrelating
    the different feature dimensions (covariance).

    Reference: https://arxiv.org/abs/2105.04906
    """

    def __init__(
        self,
        invariance_weight: float = 25.0,
        variance_weight: float = 25.0,
        covariance_weight: float = 1.0,
        gamma: float = 1.0,
    ):
        super().__init__()
        self.invariance_weight = invariance_weight
        self.variance_weight = variance_weight
        self.covariance_weight = covariance_weight
        self.gamma = gamma

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Computes the VICReg loss.

        Args:
            z1: Embeddings of the first view, shape (batch_size, dim).
            z2: Embeddings of the second view, shape (batch_size, dim).

        Returns:
            The total VICReg loss.
        """
        # Invariance loss (MSE)
        sim_loss = F.mse_loss(z1, z2)

        # Variance loss
        std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
        std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)
        var_loss = torch.mean(F.relu(self.gamma - std_z1)) + torch.mean(
            F.relu(self.gamma - std_z2)
        )

        # Covariance loss
        z1_centered = z1 - z1.mean(dim=0)
        z2_centered = z2 - z2.mean(dim=0)
        cov_z1 = (z1_centered.T @ z1_centered) / (z1.size(0) - 1)
        cov_z2 = (z2_centered.T @ z2_centered) / (z2.size(0) - 1)
        cov_z1.fill_diagonal_(0)
        cov_z2.fill_diagonal_(0)
        cov_loss = cov_z1.pow(2).sum() / z1.size(1) + cov_z2.pow(2).sum() / z2.size(1)

        return (
            self.invariance_weight * sim_loss
            + self.variance_weight * var_loss
            + self.covariance_weight * cov_loss
        )


class WubbaLightningModule(L.LightningModule):
    """LightningModule that wraps the Wubba model for training.

    This module integrates the Wubba model with PyTorch Lightning, handling
    the training, validation, and prediction steps, as well as optimizer
    and loss function configuration.
    """

    def __init__(
        self,
        # Model architecture params
        vocab_size: int,
        embedding_dim: int,
        transformer_dim: int,
        transformer_heads: int,
        transformer_layers: int,
        max_depth: int,
        max_position: int,
        # Training params for optimizer/scheduler
        learning_rate: float,
        weight_decay: float,
        num_epochs: int,
        gradient_accumulation_steps: int,
        # VICReg Loss params
        vicreg_invariance_weight: float,
        vicreg_variance_weight: float,
        vicreg_covariance_weight: float,
        vicreg_variance_gamma: float,
        **other_hparams,
    ):
        super().__init__()
        # Save all passed arguments to self.hparams
        self.save_hyperparameters()

        # Loss function is created and stored internally
        self.criterion = VICRegLoss(
            invariance_weight=self.hparams.vicreg_invariance_weight,
            variance_weight=self.hparams.vicreg_variance_weight,
            covariance_weight=self.hparams.vicreg_covariance_weight,
            gamma=self.hparams.vicreg_variance_gamma,
        )

        # Core model is created
        self.model = Wubba(
            vocab_size=self.hparams.vocab_size,
            embedding_dim=self.hparams.embedding_dim,
            transformer_dim=self.hparams.transformer_dim,
            transformer_heads=self.hparams.transformer_heads,
            transformer_layers=self.hparams.transformer_layers,
            max_depth=self.hparams.max_depth,
            max_position=self.hparams.max_position,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        return self.model(x, mask)

    def _shared_step(self, batch, batch_idx):
        x1, x2 = batch
        mask1 = create_mask(x1)
        mask2 = create_mask(x2)
        z1 = self.model(x1, mask1)
        z2 = self.model(x2, mask2)
        loss = self.criterion(z1, z2)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log("test/loss", loss, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        mask = create_mask(batch)
        return self.model(batch, mask)

    def configure_optimizers(self):
        """Sets up the AdamW optimizer and OneCycleLR scheduler."""
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        # The trainer datamodule is available at this stage
        train_loader = self.trainer.datamodule.train_dataloader()
        total_steps = (
            len(train_loader)
            * self.hparams.num_epochs
            // self.trainer.accumulate_grad_batches
        )

        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.hparams.learning_rate,
            total_steps=total_steps,
            anneal_strategy="cos",
            pct_start=0.1,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
