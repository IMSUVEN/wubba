"""Embedding quality metrics and collapse detection.

Reference: Wang & Isola, ICML 2020.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingMetrics:
    """Embedding quality metrics for geometric properties of representations."""

    @staticmethod
    def alignment(z1: torch.Tensor, z2: torch.Tensor) -> float:
        """Returns mean squared distance between normalized positive pairs (lower=better)."""
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        return (z1 - z2).norm(dim=-1).pow(2).mean().item()

    @staticmethod
    def uniformity(z: torch.Tensor, t: float = 2.0) -> float:
        """Returns log of avg pairwise Gaussian potential (lower=more uniform)."""
        if z.size(0) < 2:
            return float("inf")
        z = F.normalize(z, dim=-1)
        sq_pdist = torch.pdist(z, p=2).pow(2)
        return sq_pdist.mul(-t).exp().mean().log().item()

    @staticmethod
    def effective_rank(z: torch.Tensor, eps: float = 1e-8) -> float:
        """Returns effective rank (1 to dim). Low=collapse, high=good utilization."""
        if z.size(0) < 2:
            return 1.0
        z_centered = z - z.mean(dim=0)
        cov = (z_centered.T @ z_centered) / (z.size(0) - 1)
        eigenvalues = torch.linalg.eigvalsh(cov).clamp(min=eps)
        p = eigenvalues / eigenvalues.sum()
        entropy = -(p * torch.log(p + eps)).sum()
        return torch.exp(entropy).item()

    @staticmethod
    def rank_ratio(z: torch.Tensor) -> float:
        """Returns effective_rank / dim."""
        eff_rank = EmbeddingMetrics.effective_rank(z)
        return eff_rank / z.size(-1)

    @staticmethod
    def std_per_dim(z: torch.Tensor) -> tuple[float, float, float]:
        """Returns (min_std, mean_std, max_std) across dimensions."""
        std = z.std(dim=0)
        return std.min().item(), std.mean().item(), std.max().item()

    @staticmethod
    def cosine_similarity_stats(z: torch.Tensor) -> dict[str, float]:
        """Returns {sim_min, sim_mean, sim_max, sim_std} of pairwise similarities."""
        if z.size(0) < 2:
            return {"sim_min": 0.0, "sim_mean": 0.0, "sim_max": 0.0, "sim_std": 0.0}
        z = F.normalize(z, dim=-1)
        sim_matrix = z @ z.T
        mask = torch.triu(torch.ones_like(sim_matrix), diagonal=1).bool()
        similarities = sim_matrix[mask]

        return {
            "sim_min": similarities.min().item(),
            "sim_mean": similarities.mean().item(),
            "sim_max": similarities.max().item(),
            "sim_std": similarities.std().item(),
        }

    @staticmethod
    @torch.no_grad()
    def tolerance(
        model: nn.Module,
        x: torch.Tensor,
        augment_fn: Callable[[torch.Tensor], torch.Tensor],
        n_augs: int = 5,
    ) -> float:
        """Returns mean distance between original and augmented embeddings (lower=more invariant)."""
        model.eval()
        base_emb = model(x)

        distances = []
        for _ in range(n_augs):
            aug_x = augment_fn(x)
            aug_emb = model(aug_x)

            dist = (
                (F.normalize(base_emb, dim=-1) - F.normalize(aug_emb, dim=-1)).norm(dim=-1).mean()
            )
            distances.append(dist.item())

        return sum(distances) / len(distances)

    @staticmethod
    def compute_all(
        z1: torch.Tensor,
        z2: torch.Tensor | None = None,
        uniformity_t: float = 2.0,
    ) -> dict[str, float]:
        """Computes all metrics at once. Includes alignment if z2 provided."""
        metrics = {}
        z = z1

        metrics["effective_rank"] = EmbeddingMetrics.effective_rank(z)
        metrics["rank_ratio"] = EmbeddingMetrics.rank_ratio(z)

        min_std, mean_std, max_std = EmbeddingMetrics.std_per_dim(z)
        metrics["std_min"] = min_std
        metrics["std_mean"] = mean_std
        metrics["std_max"] = max_std

        metrics["uniformity"] = EmbeddingMetrics.uniformity(z, t=uniformity_t)

        sim_stats = EmbeddingMetrics.cosine_similarity_stats(z)
        metrics.update(sim_stats)

        if z2 is not None:
            metrics["alignment"] = EmbeddingMetrics.alignment(z1, z2)

        return metrics


@dataclass
class CollapseStatus:
    """Collapse detection result."""

    is_collapsing: bool = False
    warning: str | None = None
    effective_rank: float = 0.0
    rank_ratio: float = 0.0
    min_std: float = 0.0
    trend: str = "stable"  # "stable", "declining", "improving"


@dataclass
class CollapseDetector:
    """Monitors effective rank and variance to detect collapse early."""

    rank_threshold: float = 0.3
    std_threshold: float = 0.1
    history_size: int = 10
    history: list[dict[str, float]] = field(default_factory=list)

    def check(self, embeddings: torch.Tensor) -> CollapseStatus:
        """Returns CollapseStatus with detection results."""
        status = CollapseStatus()

        eff_rank = EmbeddingMetrics.effective_rank(embeddings)
        max_rank = embeddings.size(-1)
        rank_ratio = eff_rank / max_rank
        min_std, _, _ = EmbeddingMetrics.std_per_dim(embeddings)

        status.effective_rank = eff_rank
        status.rank_ratio = rank_ratio
        status.min_std = min_std

        self.history.append({"rank_ratio": rank_ratio, "min_std": min_std})
        if len(self.history) > self.history_size:
            self.history.pop(0)

        warnings = []
        if rank_ratio < self.rank_threshold:
            status.is_collapsing = True
            warnings.append(f"Low effective rank: {eff_rank:.1f}/{max_rank}")
        if min_std < self.std_threshold:
            status.is_collapsing = True
            warnings.append(f"Low dimension std: {min_std:.4f}")

        if len(self.history) >= 3:
            recent_ranks = [h["rank_ratio"] for h in self.history[-3:]]
            if all(recent_ranks[i] > recent_ranks[i + 1] for i in range(2)):
                status.trend = "declining"
                if not status.is_collapsing:
                    warnings.append("Rank declining trend")
            elif all(recent_ranks[i] < recent_ranks[i + 1] for i in range(2)):
                status.trend = "improving"

        if warnings:
            status.warning = "; ".join(warnings)
        return status

    def reset(self):
        """Clears history."""
        self.history = []


class AlignmentUniformityLoss(nn.Module):
    """Alignment + Uniformity loss (Wang & Isola, ICML 2020)."""

    def __init__(
        self,
        alignment_weight: float = 1.0,
        uniformity_weight: float = 1.0,
        uniformity_t: float = 2.0,
    ):
        super().__init__()
        self.alignment_weight = alignment_weight
        self.uniformity_weight = uniformity_weight
        self.uniformity_t = uniformity_t

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Returns {loss, alignment, uniformity}."""
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        alignment = (z1 - z2).norm(dim=-1).pow(2).mean()

        z = torch.cat([z1, z2], dim=0)
        sq_pdist = torch.pdist(z, p=2).pow(2)
        uniformity = sq_pdist.mul(-self.uniformity_t).exp().mean().log()

        loss = self.alignment_weight * alignment + self.uniformity_weight * uniformity

        return {"loss": loss, "alignment": alignment, "uniformity": uniformity}


class SelfPacedWeighter:
    """Sample weights for self-paced learning with growing threshold."""

    def __init__(
        self,
        lambda_init: float = 2.0,
        lambda_growth: float = 1.05,
        mode: str = "mixture",
    ):
        self.lambda_val = lambda_init
        self.lambda_growth = lambda_growth
        self.mode = mode
        self._current_epoch = 0

    def compute_weights(self, losses: torch.Tensor) -> torch.Tensor:
        """Returns sample weights based on losses."""
        if self.mode == "linear":
            weights = (losses < self.lambda_val).float()
        elif self.mode == "log":
            weights = torch.clamp(1 - losses / self.lambda_val, min=0, max=1)
        else:  # mixture
            soft_weights = torch.clamp(1 - losses / self.lambda_val, min=0, max=1)
            hard_sampling = torch.rand_like(losses) < 0.1
            weights = torch.where(hard_sampling, torch.ones_like(losses) * 0.3, soft_weights)
        return weights.clamp(min=0.01)

    def step_epoch(self):
        """Updates lambda for next epoch."""
        self._current_epoch += 1
        self.lambda_val *= self.lambda_growth

    @property
    def current_lambda(self) -> float:
        """Returns current threshold."""
        return self.lambda_val


class EMAModel:
    """Exponential Moving Average of model parameters."""

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        update_after_step: int = 100,
    ):
        self.model = model
        self.decay = decay
        self.update_after_step = update_after_step
        self.step_count = 0

        # Initialize shadow parameters
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self):
        """Updates shadow parameters with EMA."""
        self.step_count += 1

        if self.step_count < self.update_after_step:
            return

        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self):
        """Applies shadow parameters for evaluation."""
        if self.backup:
            raise RuntimeError("Shadow already applied. Call restore() first.")
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restores original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self) -> dict[str, Any]:
        """Returns checkpoint state."""
        return {
            "shadow": self.shadow,
            "step_count": self.step_count,
            "decay": self.decay,
        }

    def load_state_dict(self, state_dict: dict[str, Any]):
        """Loads checkpoint state."""
        self.shadow = state_dict["shadow"]
        self.step_count = state_dict["step_count"]
        self.decay = state_dict.get("decay", self.decay)
