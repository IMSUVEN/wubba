import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

from wubba.const import NUM_SEMANTIC_GROUPS, NUM_TAG_ROLES
from wubba.metrics import (
    CollapseDetector,
    EMAModel,
    EmbeddingMetrics,
    SelfPacedWeighter,
)
from wubba.utils import create_mask


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (arXiv:1910.07467)."""

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (arXiv:2104.09864)."""

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
    ):
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even for RoPE"

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequency matrix
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self,
        seq_len: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (cos, sin) tensors for the given sequence length."""
        if seq_len <= self.cos_cached.size(0):
            return (
                self.cos_cached[:seq_len].to(device),
                self.sin_cached[:seq_len].to(device),
            )

        # Compute on-the-fly for longer sequences
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


class HierarchicalRoPE(nn.Module):
    """RoPE extended for DOM tree structure: position, depth, and subtree depth."""

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 512,
        max_depth: int = 256,
        max_subtree_depth: int = 32,
        position_base: float = 10000.0,
        depth_base: float = 1000.0,
        subtree_base: float = 500.0,
    ):
        super().__init__()

        # Allocate dimensions: 2/3 position, 1/6 depth, 1/6 subtree
        self.head_dim = head_dim
        self.position_dim = (head_dim // 3) * 2  # 2/3 of dim
        self.depth_dim = head_dim // 6  # 1/6 of dim
        self.subtree_dim = head_dim - self.position_dim - self.depth_dim  # remainder

        # Ensure dimensions are even (required for RoPE rotation)
        self.position_dim = (self.position_dim // 2) * 2
        self.depth_dim = (self.depth_dim // 2) * 2
        self.subtree_dim = (self.subtree_dim // 2) * 2

        # Add remaining dimensions (ensure even by rounding down)
        remaining = head_dim - self.position_dim - self.depth_dim - self.subtree_dim
        self.position_dim += (remaining // 2) * 2

        if self.position_dim > 0:
            inv_freq_pos = 1.0 / (
                position_base ** (torch.arange(0, self.position_dim, 2).float() / self.position_dim)
            )
            self.register_buffer("inv_freq_pos", inv_freq_pos, persistent=False)

        if self.depth_dim > 0:
            inv_freq_depth = 1.0 / (
                depth_base ** (torch.arange(0, self.depth_dim, 2).float() / self.depth_dim)
            )
            self.register_buffer("inv_freq_depth", inv_freq_depth, persistent=False)

        if self.subtree_dim > 0:
            inv_freq_subtree = 1.0 / (
                subtree_base ** (torch.arange(0, self.subtree_dim, 2).float() / self.subtree_dim)
            )
            self.register_buffer("inv_freq_subtree", inv_freq_subtree, persistent=False)

        self._build_cache(max_seq_len, max_depth, max_subtree_depth)

    def _build_cache(self, max_seq_len: int, max_depth: int, max_subtree_depth: int) -> None:
        if self.position_dim > 0:
            t_pos = torch.arange(max_seq_len, dtype=self.inv_freq_pos.dtype)
            freqs_pos = torch.outer(t_pos, self.inv_freq_pos)
            emb_pos = torch.cat([freqs_pos, freqs_pos], dim=-1)
            self.register_buffer("cos_pos", emb_pos.cos(), persistent=False)
            self.register_buffer("sin_pos", emb_pos.sin(), persistent=False)

        if self.depth_dim > 0:
            t_depth = torch.arange(max_depth, dtype=self.inv_freq_depth.dtype)
            freqs_depth = torch.outer(t_depth, self.inv_freq_depth)
            emb_depth = torch.cat([freqs_depth, freqs_depth], dim=-1)
            self.register_buffer("cos_depth", emb_depth.cos(), persistent=False)
            self.register_buffer("sin_depth", emb_depth.sin(), persistent=False)

        if self.subtree_dim > 0:
            t_subtree = torch.arange(max_subtree_depth, dtype=self.inv_freq_subtree.dtype)
            freqs_subtree = torch.outer(t_subtree, self.inv_freq_subtree)
            emb_subtree = torch.cat([freqs_subtree, freqs_subtree], dim=-1)
            self.register_buffer("cos_subtree", emb_subtree.cos(), persistent=False)
            self.register_buffer("sin_subtree", emb_subtree.sin(), persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        depths: torch.Tensor,
        subtree_depths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (cos, sin) tensors combining position, depth, and subtree embeddings."""
        cos_parts = []
        sin_parts = []

        if self.position_dim > 0:
            pos_idx = positions.clamp(0, self.cos_pos.size(0) - 1)
            cos_parts.append(self.cos_pos[pos_idx])
            sin_parts.append(self.sin_pos[pos_idx])

        if self.depth_dim > 0:
            depth_idx = depths.clamp(0, self.cos_depth.size(0) - 1)
            cos_parts.append(self.cos_depth[depth_idx])
            sin_parts.append(self.sin_depth[depth_idx])

        if self.subtree_dim > 0:
            subtree_idx = subtree_depths.clamp(0, self.cos_subtree.size(0) - 1)
            cos_parts.append(self.cos_subtree[subtree_idx])
            sin_parts.append(self.sin_subtree[subtree_idx])

        cos = torch.cat(cos_parts, dim=-1)
        sin = torch.cat(sin_parts, dim=-1)

        return cos, sin


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies RoPE rotation to query and key tensors."""

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class FlashAttentionEncoderLayer(nn.Module):
    """Transformer layer with SDPA, pre-norm, RMSNorm, HierarchicalRoPE, and SwiGLU."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        max_depth: int = 256,
        max_subtree_depth: int = 32,
        rope_position_base: float = 10000.0,
        rope_depth_base: float = 1000.0,
        rope_subtree_base: float = 500.0,
    ):
        super().__init__()

        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout_p = dropout

        # Multi-head attention projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.rope = HierarchicalRoPE(
            head_dim=self.head_dim,
            max_seq_len=max_seq_len,
            max_depth=max_depth,
            max_subtree_depth=max_subtree_depth,
            position_base=rope_position_base,
            depth_base=rope_depth_base,
            subtree_base=rope_subtree_base,
        )

        # SwiGLU FFN
        self.gate_proj = nn.Linear(d_model, dim_feedforward, bias=False)
        self.up_proj = nn.Linear(d_model, dim_feedforward, bias=False)
        self.down_proj = nn.Linear(dim_feedforward, d_model, bias=False)

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        depths: torch.Tensor,
        subtree_depths: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self._attention_block(x, positions, depths, subtree_depths, key_padding_mask)
        x = residual + self.dropout(x)

        residual = x
        x = self.norm2(x)
        x = self._feedforward_block(x)
        x = residual + self.dropout(x)

        return x

    def _attention_block(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        depths: torch.Tensor,
        subtree_depths: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.nhead, self.head_dim)
        k = k.view(batch_size, seq_len, self.nhead, self.head_dim)
        v = v.view(batch_size, seq_len, self.nhead, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = self.rope(positions, depths, subtree_depths)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)

        dropout_p = self.dropout_p if self.training else 0.0
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        return self.out_proj(attn_output)

    def _feedforward_block(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class FlashAttentionEncoder(nn.Module):
    """Stack of FlashAttentionEncoderLayers with hierarchical RoPE."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        max_depth: int = 256,
        max_subtree_depth: int = 32,
        rope_position_base: float = 10000.0,
        rope_depth_base: float = 1000.0,
        rope_subtree_base: float = 500.0,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                FlashAttentionEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    max_seq_len=max_seq_len,
                    max_depth=max_depth,
                    max_subtree_depth=max_subtree_depth,
                    rope_position_base=rope_position_base,
                    rope_depth_base=rope_depth_base,
                    rope_subtree_base=rope_subtree_base,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        depths: torch.Tensor,
        subtree_depths: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, positions, depths, subtree_depths, key_padding_mask)

        return self.norm(x)


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 3,
    ):
        super().__init__()
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(in_dim, hidden_dim))
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, out_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))

            # Add BatchNorm and ReLU for all but the last layer
            if i < num_layers - 1:
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU(inplace=True))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MatryoshkaProjectionHead(nn.Module):
    """Projection head with truncatable embeddings (arXiv:2205.13147)."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 256,
        hidden_dim: int | None = None,
        num_layers: int = 3,
        matryoshka_dims: list[int] | None = None,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = in_dim

        if matryoshka_dims is None:
            matryoshka_dims = [32, 64, 128, 256]

        self.matryoshka_dims = sorted(matryoshka_dims)
        self.out_dim = out_dim

        # Ensure out_dim is at least as large as the largest matryoshka dim
        assert out_dim >= max(self.matryoshka_dims), (
            f"out_dim ({out_dim}) must be >= max matryoshka_dim ({max(self.matryoshka_dims)})"
        )

        # Build MLP layers
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(in_dim, hidden_dim))
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, out_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))

            # Add BatchNorm and GELU for all but the last layer
            if i < num_layers - 1:
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        dim: int | None = None,
    ) -> torch.Tensor:
        out = self.net(x)

        if dim is not None and dim < self.out_dim:
            out = out[:, :dim]

        return out

    def get_matryoshka_outputs(
        self,
        x: torch.Tensor,
    ) -> dict[int, torch.Tensor]:
        """Returns embeddings at all matryoshka dimensions."""
        full_out = self.net(x)
        return {dim: full_out[:, :dim] for dim in self.matryoshka_dims}


class Wubba(nn.Module):
    """HTML document encoder with Flash Attention, Hierarchical RoPE, and Matryoshka embeddings."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        transformer_dim: int,
        transformer_heads: int,
        transformer_layers: int,
        max_depth: int,
        max_position: int,
        max_children: int = 64,
        max_siblings: int = 64,
        max_subtree_depth: int = 32,
        num_semantic_groups: int | None = None,
        num_tag_roles: int | None = None,
        projection_dim: int = 256,
        use_cls_token: bool = True,
        dropout: float = 0.1,
        rope_position_base: float = 10000.0,
        rope_depth_base: float = 1000.0,
        rope_subtree_base: float = 500.0,
    ):
        super().__init__()

        if num_semantic_groups is None:
            num_semantic_groups = NUM_SEMANTIC_GROUPS
        if num_tag_roles is None:
            num_tag_roles = NUM_TAG_ROLES

        self.use_cls_token = use_cls_token
        self.transformer_dim = transformer_dim
        self.max_depth = max_depth
        self.max_position = max_position
        self.max_subtree_depth = max_subtree_depth

        self.tag_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.parent_tag_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.semantic_embedding = nn.Embedding(num_semantic_groups + 1, embedding_dim // 4)
        self.children_embedding = nn.Embedding(max_children + 1, embedding_dim // 8)
        self.siblings_embedding = nn.Embedding(max_siblings + 1, embedding_dim // 8)
        self.leaf_embedding = nn.Embedding(2, embedding_dim // 8)
        self.role_embedding = nn.Embedding(num_tag_roles + 1, embedding_dim // 8)

        input_dim = (
            embedding_dim  # tag
            + embedding_dim  # parent_tag
            + embedding_dim // 4  # semantic
            + embedding_dim // 8  # children
            + embedding_dim // 8  # siblings
            + embedding_dim // 8  # leaf
            + embedding_dim // 8  # role
        )
        self.input_projection = nn.Linear(input_dim, transformer_dim)

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, transformer_dim))
            self.cls_position = nn.Parameter(torch.zeros(1, 1, transformer_dim))

        self.encoder = FlashAttentionEncoder(
            d_model=transformer_dim,
            nhead=transformer_heads,
            num_layers=transformer_layers,
            dim_feedforward=transformer_dim * 4,
            dropout=dropout,
            max_seq_len=max_position + 1,  # +1 for CLS token
            max_depth=max_depth,
            max_subtree_depth=max_subtree_depth,
            rope_position_base=rope_position_base,
            rope_depth_base=rope_depth_base,
            rope_subtree_base=rope_subtree_base,
        )

        self.projection_head = ProjectionHead(
            in_dim=transformer_dim,
            hidden_dim=transformer_dim,
            out_dim=projection_dim,
            num_layers=3,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_projection: bool = True,
    ) -> torch.Tensor:
        batch_size, _, _ = x.shape

        tags = x[..., 0].long()
        semantic_group = x[..., 1].long()
        depths = x[..., 2].long()
        positions = x[..., 3].long()
        num_children = x[..., 4].long()
        sibling_count = x[..., 5].long()
        is_leaf = x[..., 6].long()
        parent_tag = x[..., 7].long()
        tag_role = x[..., 8].long()
        subtree_depths = x[..., 9].long()

        tag_embeds = self.tag_embedding(tags)
        parent_embeds = self.parent_tag_embedding(parent_tag)
        semantic_embeds = self.semantic_embedding(
            semantic_group.clamp(0, self.semantic_embedding.num_embeddings - 1)
        )
        children_embeds = self.children_embedding(
            num_children.clamp(0, self.children_embedding.num_embeddings - 1)
        )
        siblings_embeds = self.siblings_embedding(
            sibling_count.clamp(0, self.siblings_embedding.num_embeddings - 1)
        )
        leaf_embeds = self.leaf_embedding(is_leaf.clamp(0, 1))
        role_embeds = self.role_embedding(tag_role.clamp(0, self.role_embedding.num_embeddings - 1))

        combined = torch.cat(
            [
                tag_embeds,
                parent_embeds,
                semantic_embeds,
                children_embeds,
                siblings_embeds,
                leaf_embeds,
                role_embeds,
            ],
            dim=-1,
        )

        embeddings = self.input_projection(combined)

        positions = positions.clamp(0, self.max_position - 1)
        depths = depths.clamp(0, self.max_depth - 1)
        subtree_depths = subtree_depths.clamp(0, self.max_subtree_depth - 1)

        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            embeddings = torch.cat([cls_tokens, embeddings], dim=1)

            cls_pos = torch.zeros(batch_size, 1, device=positions.device, dtype=positions.dtype)
            cls_depth = torch.zeros(batch_size, 1, device=depths.device, dtype=depths.dtype)
            cls_subtree = torch.zeros(
                batch_size, 1, device=subtree_depths.device, dtype=subtree_depths.dtype
            )

            positions = torch.cat([cls_pos, positions], dim=1)
            depths = torch.cat([cls_depth, depths], dim=1)
            subtree_depths = torch.cat([cls_subtree, subtree_depths], dim=1)

            if mask is not None:
                cls_mask = torch.ones(batch_size, 1, device=mask.device, dtype=mask.dtype)
                mask = torch.cat([cls_mask, mask], dim=1)

        key_padding_mask = None
        if mask is not None:
            key_padding_mask = mask == 0

        encoded = self.encoder(
            embeddings,
            positions,
            depths,
            subtree_depths,
            key_padding_mask,
        )

        if self.use_cls_token:
            pooled = encoded[:, 0]
        else:
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1)
                encoded = encoded * mask_expanded
                pooled = encoded.sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)
            else:
                pooled = encoded.mean(dim=1)

        if return_projection:
            return self.projection_head(pooled)
        else:
            return pooled


class VICRegLoss(nn.Module):
    """VICReg loss: variance-invariance-covariance regularization (arXiv:2105.04906)."""

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
        sim_loss = F.mse_loss(z1, z2)

        std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
        std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)
        var_loss = torch.mean(F.relu(self.gamma - std_z1)) + torch.mean(F.relu(self.gamma - std_z2))

        # Covariance loss requires batch_size >= 2
        if z1.size(0) < 2:
            cov_loss = torch.tensor(0.0, device=z1.device, dtype=z1.dtype)
        else:
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


class InfoNCELoss(nn.Module):
    """InfoNCE/NT-Xent contrastive loss (arXiv:2002.05709)."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        batch_size = z1.size(0)
        device = z1.device

        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)

        embeddings = torch.cat([z1, z2], dim=0)
        sim_matrix = torch.mm(embeddings, embeddings.T) / self.temperature

        mask = torch.eye(2 * batch_size, device=device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(mask, float("-inf"))

        pos_indices = torch.cat(
            [
                torch.arange(batch_size, 2 * batch_size, device=device),
                torch.arange(0, batch_size, device=device),
            ]
        )

        return F.cross_entropy(sim_matrix, pos_indices)


class HybridContrastiveLoss(nn.Module):
    """Combined VICReg + InfoNCE loss."""

    def __init__(
        self,
        vicreg_weight: float = 1.0,
        infonce_weight: float = 0.5,
        vicreg_invariance_weight: float = 25.0,
        vicreg_variance_weight: float = 25.0,
        vicreg_covariance_weight: float = 1.0,
        vicreg_variance_gamma: float = 1.0,
        infonce_temperature: float = 0.07,
    ):
        super().__init__()
        self.vicreg_weight = vicreg_weight
        self.infonce_weight = infonce_weight

        self.vicreg = VICRegLoss(
            invariance_weight=vicreg_invariance_weight,
            variance_weight=vicreg_variance_weight,
            covariance_weight=vicreg_covariance_weight,
            gamma=vicreg_variance_gamma,
        )
        self.infonce = InfoNCELoss(temperature=infonce_temperature)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        vicreg_loss = self.vicreg(z1, z2)
        infonce_loss = self.infonce(z1, z2)

        return self.vicreg_weight * vicreg_loss + self.infonce_weight * infonce_loss


class MatryoshkaLoss(nn.Module):
    """Applies base loss at multiple truncated dimensions (arXiv:2205.13147)."""

    def __init__(
        self,
        base_loss: nn.Module,
        matryoshka_dims: list[int] | None = None,
        dim_weights: list[float] | None = None,
    ):
        super().__init__()

        if matryoshka_dims is None:
            matryoshka_dims = [32, 64, 128, 256]

        self.matryoshka_dims = sorted(matryoshka_dims)
        self.base_loss = base_loss

        # Set dimension weights
        if dim_weights is None:
            # Equal weights, normalized
            dim_weights = [1.0 / len(self.matryoshka_dims)] * len(self.matryoshka_dims)
        else:
            assert len(dim_weights) == len(self.matryoshka_dims), (
                f"dim_weights length ({len(dim_weights)}) must match "
                f"matryoshka_dims length ({len(self.matryoshka_dims)})"
            )
            # Normalize weights
            total = sum(dim_weights)
            dim_weights = [w / total for w in dim_weights]

        self.register_buffer(
            "dim_weights",
            torch.tensor(dim_weights, dtype=torch.float32),
        )

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=z1.device, dtype=z1.dtype)

        for i, dim in enumerate(self.matryoshka_dims):
            z1_truncated = z1[:, :dim]
            z2_truncated = z2[:, :dim]
            loss_at_dim = self.base_loss(z1_truncated, z2_truncated)
            total_loss = total_loss + self.dim_weights[i] * loss_at_dim

        return total_loss


class MatryoshkaHybridLoss(nn.Module):
    """Matryoshka + VICReg + InfoNCE combined loss."""

    def __init__(
        self,
        matryoshka_dims: list[int] | None = None,
        dim_weights: list[float] | None = None,
        vicreg_weight: float = 1.0,
        infonce_weight: float = 0.5,
        vicreg_invariance_weight: float = 25.0,
        vicreg_variance_weight: float = 25.0,
        vicreg_covariance_weight: float = 1.0,
        vicreg_variance_gamma: float = 1.0,
        infonce_temperature: float = 0.07,
    ):
        super().__init__()

        if matryoshka_dims is None:
            matryoshka_dims = [32, 64, 128, 256]

        self.matryoshka_dims = sorted(matryoshka_dims)
        self.vicreg_weight = vicreg_weight
        self.infonce_weight = infonce_weight

        self.hybrid_loss = HybridContrastiveLoss(
            vicreg_weight=vicreg_weight,
            infonce_weight=infonce_weight,
            vicreg_invariance_weight=vicreg_invariance_weight,
            vicreg_variance_weight=vicreg_variance_weight,
            vicreg_covariance_weight=vicreg_covariance_weight,
            vicreg_variance_gamma=vicreg_variance_gamma,
            infonce_temperature=infonce_temperature,
        )

        self.matryoshka_loss = MatryoshkaLoss(
            base_loss=self.hybrid_loss,
            matryoshka_dims=matryoshka_dims,
            dim_weights=dim_weights,
        )

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        return self.matryoshka_loss(z1, z2)


class SpectralContrastiveLoss(nn.Module):
    """Spectral regularization for covariance condition number (NeurIPS 2021)."""

    def __init__(self, weight: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z = torch.cat([z1, z2], dim=0)
        z_centered = z - z.mean(dim=0)
        cov = (z_centered.T @ z_centered) / (z.size(0) - 1)

        try:
            _, s, _ = torch.linalg.svd(cov)
        except RuntimeError:
            return torch.tensor(0.0, device=z.device, dtype=z.dtype)

        s_max = s.max()
        s_min = s.min().clamp(min=self.eps)
        condition_number = s_max / s_min
        spectral_loss = torch.log(condition_number + 1)
        return self.weight * spectral_loss


class HardNegativeMiner(nn.Module):
    """Synthesizes hard negatives via embedding interpolation."""

    def __init__(
        self,
        ratio: float = 0.2,
        mix_alpha: float = 0.5,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.ratio = ratio
        self.mix_alpha = mix_alpha
        self.temperature = temperature

    def forward(
        self,
        z: torch.Tensor,
        exclude_diagonal: bool = True,
    ) -> torch.Tensor:
        batch_size = z.size(0)
        n_hard = max(1, int(batch_size * self.ratio))

        # Compute similarity matrix
        z_norm = F.normalize(z, dim=-1)
        sim_matrix = z_norm @ z_norm.T

        if exclude_diagonal:
            mask = torch.eye(batch_size, device=z.device, dtype=torch.bool)
            sim_matrix = sim_matrix.masked_fill(mask, float("-inf"))

        sim_probs = F.softmax(sim_matrix / self.temperature, dim=-1)
        hard_negatives = []

        for _ in range(n_hard):
            src_idx = torch.randint(0, batch_size, (1,), device=z.device)
            target_idx = torch.multinomial(sim_probs[src_idx], 1).squeeze()
            alpha = torch.rand(1, device=z.device) * self.mix_alpha
            hard_neg = alpha * z[src_idx] + (1 - alpha) * z[target_idx]
            hard_negatives.append(hard_neg.squeeze(0))

        return torch.stack(hard_negatives)


class InfoNCEWithHardNegatives(nn.Module):
    """InfoNCE with synthesized hard negatives in the denominator."""

    def __init__(
        self,
        temperature: float = 0.07,
        hard_negative_ratio: float = 0.2,
        hard_negative_alpha: float = 0.5,
    ):
        super().__init__()
        self.temperature = temperature
        self.hard_miner = HardNegativeMiner(
            ratio=hard_negative_ratio,
            mix_alpha=hard_negative_alpha,
        )

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        batch_size = z1.size(0)
        device = z1.device

        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        z_all = torch.cat([z1, z2], dim=0)
        hard_negs = self.hard_miner(z_all)
        hard_negs = F.normalize(hard_negs, dim=-1)

        embeddings = torch.cat([z1, z2, hard_negs], dim=0)
        query = torch.cat([z1, z2], dim=0)
        sim_matrix = query @ embeddings.T / self.temperature

        n_query = 2 * batch_size
        mask = torch.eye(n_query, embeddings.size(0), device=device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(mask, float("-inf"))

        pos_indices = torch.cat(
            [
                torch.arange(batch_size, 2 * batch_size, device=device),
                torch.arange(0, batch_size, device=device),
            ]
        )

        return F.cross_entropy(sim_matrix, pos_indices)


class EnhancedHybridLoss(nn.Module):
    """VICReg + InfoNCE + Spectral + Matryoshka combined loss."""

    def __init__(
        self,
        matryoshka_dims: list[int] | None = None,
        dim_weights: list[float] | None = None,
        vicreg_weight: float = 1.0,
        vicreg_invariance_weight: float = 25.0,
        vicreg_variance_weight: float = 25.0,
        vicreg_covariance_weight: float = 1.0,
        vicreg_variance_gamma: float = 1.0,
        infonce_weight: float = 0.5,
        infonce_temperature: float = 0.07,
        use_hard_negative: bool = True,
        hard_negative_ratio: float = 0.2,
        hard_negative_alpha: float = 0.5,
        use_spectral: bool = True,
        spectral_weight: float = 0.1,
        active_dims: list[int] | None = None,
    ):
        super().__init__()

        if matryoshka_dims is None:
            matryoshka_dims = [32, 64, 128, 256]

        self.matryoshka_dims = sorted(matryoshka_dims)
        self.active_dims = active_dims or self.matryoshka_dims.copy()
        self.vicreg_weight = vicreg_weight
        self.infonce_weight = infonce_weight
        self.use_hard_negative = use_hard_negative
        self.use_spectral = use_spectral

        self.vicreg = VICRegLoss(
            invariance_weight=vicreg_invariance_weight,
            variance_weight=vicreg_variance_weight,
            covariance_weight=vicreg_covariance_weight,
            gamma=vicreg_variance_gamma,
        )

        if use_hard_negative:
            self.infonce = InfoNCEWithHardNegatives(
                temperature=infonce_temperature,
                hard_negative_ratio=hard_negative_ratio,
                hard_negative_alpha=hard_negative_alpha,
            )
        else:
            self.infonce = InfoNCELoss(temperature=infonce_temperature)

        if use_spectral:
            self.spectral = SpectralContrastiveLoss(weight=spectral_weight)

        if dim_weights is None:
            dim_weights = [1.0 / len(self.matryoshka_dims)] * len(self.matryoshka_dims)
        else:
            total = sum(dim_weights)
            dim_weights = [w / total for w in dim_weights]

        self.register_buffer(
            "dim_weights",
            torch.tensor(dim_weights, dtype=torch.float32),
        )

        # Pre-build dim->index mapping for O(1) lookup
        self._dim_to_idx = {dim: i for i, dim in enumerate(self.matryoshka_dims)}

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        total_loss = torch.tensor(0.0, device=z1.device, dtype=z1.dtype)
        components = {}

        for _i, dim in enumerate(self.matryoshka_dims):
            if dim not in self.active_dims:
                continue

            z1_d = z1[:, :dim]
            z2_d = z2[:, :dim]

            vicreg_loss = self.vicreg(z1_d, z2_d)
            infonce_loss = self.infonce(z1_d, z2_d)
            dim_loss = self.vicreg_weight * vicreg_loss + self.infonce_weight * infonce_loss

            weight_idx = self._dim_to_idx[dim]
            total_loss = total_loss + self.dim_weights[weight_idx] * dim_loss

            if return_components:
                components[f"vicreg_{dim}"] = vicreg_loss
                components[f"infonce_{dim}"] = infonce_loss

        if self.use_spectral:
            spectral_loss = self.spectral(z1, z2)
            total_loss = total_loss + spectral_loss
            if return_components:
                components["spectral"] = spectral_loss

        if return_components:
            components["total"] = total_loss
            return components

        return total_loss


class MaskedNodePrediction(nn.Module):
    """BERT-style masked node prediction for DOM nodes."""

    def __init__(
        self,
        transformer_dim: int,
        vocab_size: int,
        mask_prob: float = 0.15,
        mask_token_id: int = 1,
    ):
        super().__init__()
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size

        self.predictor = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim),
            nn.GELU(),
            nn.LayerNorm(transformer_dim),
            nn.Linear(transformer_dim, vocab_size),
        )

    def create_masked_input(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (masked_input, labels) where labels=-100 for non-masked."""
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Random mask (excluding padding)
        rand = torch.rand(batch_size, seq_len, device=device)
        mask = rand < self.mask_prob

        if padding_mask is not None:
            mask = mask & (padding_mask > 0)

        labels = x[..., 0].clone()
        labels[~mask] = -100

        x_masked = x.clone()

        # 80% [MASK], 10% random, 10% keep
        mask_token = mask & (torch.rand_like(rand) < 0.8)
        x_masked[mask_token, 0] = self.mask_token_id

        random_token = mask & ~mask_token & (torch.rand_like(rand) < 0.5)
        x_masked[random_token, 0] = torch.randint(
            2, self.vocab_size, (random_token.sum(),), device=device
        )

        return x_masked, labels.long()

    def forward(
        self,
        encoded: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.predictor(encoded)
        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            labels.view(-1),
            ignore_index=-100,
        )
        return loss


class StructurePrediction(nn.Module):
    """Predicts tree depth and node count bucket."""

    def __init__(
        self,
        transformer_dim: int,
        max_depth: int = 32,
        num_size_buckets: int = 8,
    ):
        super().__init__()
        self.max_depth = max_depth
        self.num_size_buckets = num_size_buckets

        self.depth_predictor = nn.Linear(transformer_dim, max_depth)
        self.size_predictor = nn.Linear(transformer_dim, num_size_buckets)

    def forward(
        self,
        cls_embedding: torch.Tensor,
        depth_labels: torch.Tensor,
        size_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        depth_logits = self.depth_predictor(cls_embedding)
        size_logits = self.size_predictor(cls_embedding)

        depth_labels = depth_labels.clamp(0, self.max_depth - 1)
        size_labels = size_labels.clamp(0, self.num_size_buckets - 1)

        depth_loss = F.cross_entropy(depth_logits, depth_labels)
        size_loss = F.cross_entropy(size_logits, size_labels)

        return {
            "depth_loss": depth_loss,
            "size_loss": size_loss,
            "total": depth_loss + size_loss,
        }


class WubbaLightningModule(L.LightningModule):
    """Lightning training module with all loss variants and training enhancements."""

    def __init__(
        self,
        # Model architecture params
        vocab_size: int,
        embedding_dim: int = 128,
        transformer_dim: int = 256,
        transformer_heads: int = 8,
        transformer_layers: int = 6,
        max_depth: int = 256,
        max_position: int = 256,
        max_children: int = 64,
        max_siblings: int = 64,
        max_subtree_depth: int = 32,
        num_semantic_groups: int | None = None,
        num_tag_roles: int | None = None,
        projection_dim: int = 256,
        use_cls_token: bool = True,
        dropout: float = 0.1,
        # RoPE configuration
        rope_position_base: float = 10000.0,
        rope_depth_base: float = 1000.0,
        rope_subtree_base: float = 500.0,
        # Matryoshka configuration
        use_matryoshka: bool = True,
        matryoshka_dims: list[int] | None = None,
        matryoshka_weights: list[float] | None = None,
        progressive_matryoshka: bool = True,
        matryoshka_unlock_epochs: list[int] | None = None,
        # Training params
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
        num_epochs: int = 100,
        gradient_accumulation_steps: int = 1,
        lr_scheduler: str = "cosine_restarts",
        restart_epochs: list[int] | None = None,
        min_lr_ratio: float = 0.01,
        # EMA
        use_ema: bool = True,
        ema_decay: float = 0.999,
        ema_update_after_step: int = 100,
        # Self-paced learning
        use_self_paced: bool = True,
        self_paced_lambda_init: float = 2.0,
        self_paced_lambda_growth: float = 1.05,
        self_paced_mode: str = "mixture",
        # Loss function params
        loss_type: str = "enhanced_hybrid",
        vicreg_invariance_weight: float = 25.0,
        vicreg_variance_weight: float = 25.0,
        vicreg_covariance_weight: float = 1.0,
        vicreg_variance_gamma: float = 1.0,
        infonce_temperature: float = 0.07,
        hybrid_vicreg_weight: float = 1.0,
        hybrid_infonce_weight: float = 0.5,
        # Enhanced loss params
        use_spectral_loss: bool = True,
        spectral_weight: float = 0.1,
        use_hard_negative: bool = True,
        hard_negative_ratio: float = 0.2,
        hard_negative_mix_alpha: float = 0.5,
        # Multi-task learning
        enable_multitask: bool = True,
        mnp_enabled: bool = True,
        mnp_weight: float = 0.3,
        mnp_mask_prob: float = 0.15,
        structure_pred_enabled: bool = True,
        structure_pred_weight: float = 0.2,
        # Monitoring
        log_embedding_metrics: bool = True,
        embedding_log_interval: int = 100,
        collapse_detection: bool = True,
        collapse_rank_threshold: float = 0.3,
        # Feature config
        feature_dim: int = 10,
        use_extended_features: bool = False,
        **other_hparams,
    ):
        super().__init__()
        self.save_hyperparameters()

        if matryoshka_dims is None:
            matryoshka_dims = [32, 64, 128, 256]

        if matryoshka_unlock_epochs is None:
            matryoshka_unlock_epochs = [0, 20, 40, 60]

        if restart_epochs is None:
            restart_epochs = [30, 60, 80]

        self.active_matryoshka_dims = (
            [matryoshka_dims[0]] if progressive_matryoshka else matryoshka_dims.copy()
        )

        self._setup_loss_function(matryoshka_dims)

        self.model = Wubba(
            vocab_size=self.hparams.vocab_size,
            embedding_dim=self.hparams.embedding_dim,
            transformer_dim=self.hparams.transformer_dim,
            transformer_heads=self.hparams.transformer_heads,
            transformer_layers=self.hparams.transformer_layers,
            max_depth=self.hparams.max_depth,
            max_position=self.hparams.max_position,
            max_children=self.hparams.max_children,
            max_siblings=self.hparams.max_siblings,
            max_subtree_depth=self.hparams.max_subtree_depth,
            num_semantic_groups=self.hparams.num_semantic_groups,
            num_tag_roles=self.hparams.num_tag_roles,
            projection_dim=self.hparams.projection_dim,
            use_cls_token=self.hparams.use_cls_token,
            dropout=self.hparams.dropout,
            rope_position_base=self.hparams.rope_position_base,
            rope_depth_base=self.hparams.rope_depth_base,
            rope_subtree_base=self.hparams.rope_subtree_base,
        )

        if self.hparams.use_matryoshka:
            self.model.projection_head = MatryoshkaProjectionHead(
                in_dim=self.hparams.transformer_dim,
                out_dim=self.hparams.projection_dim,
                hidden_dim=self.hparams.transformer_dim,
                num_layers=3,
                matryoshka_dims=matryoshka_dims,
            )

        if self.hparams.enable_multitask:
            if self.hparams.mnp_enabled:
                self.mnp_head = MaskedNodePrediction(
                    transformer_dim=self.hparams.transformer_dim,
                    vocab_size=self.hparams.vocab_size,
                    mask_prob=self.hparams.mnp_mask_prob,
                )

            if self.hparams.structure_pred_enabled:
                self.structure_head = StructurePrediction(
                    transformer_dim=self.hparams.transformer_dim,
                    max_depth=min(32, self.hparams.max_depth),
                )

        self.self_paced_weighter: SelfPacedWeighter | None = None
        if self.hparams.use_self_paced:
            self.self_paced_weighter = SelfPacedWeighter(
                lambda_init=self.hparams.self_paced_lambda_init,
                lambda_growth=self.hparams.self_paced_lambda_growth,
                mode=self.hparams.self_paced_mode,
            )

        self.ema_model: EMAModel | None = None
        self.collapse_detector: CollapseDetector | None = None
        if self.hparams.collapse_detection:
            self.collapse_detector = CollapseDetector(
                rank_threshold=self.hparams.collapse_rank_threshold,
            )

        self._train_step_count = 0

    def _setup_loss_function(self, matryoshka_dims: list[int]):
        if self.hparams.loss_type == "vicreg":
            self.criterion = VICRegLoss(
                invariance_weight=self.hparams.vicreg_invariance_weight,
                variance_weight=self.hparams.vicreg_variance_weight,
                covariance_weight=self.hparams.vicreg_covariance_weight,
                gamma=self.hparams.vicreg_variance_gamma,
            )
        elif self.hparams.loss_type == "infonce":
            self.criterion = InfoNCELoss(
                temperature=self.hparams.infonce_temperature,
            )
        elif self.hparams.loss_type == "hybrid":
            self.criterion = HybridContrastiveLoss(
                vicreg_weight=self.hparams.hybrid_vicreg_weight,
                infonce_weight=self.hparams.hybrid_infonce_weight,
                vicreg_invariance_weight=self.hparams.vicreg_invariance_weight,
                vicreg_variance_weight=self.hparams.vicreg_variance_weight,
                vicreg_covariance_weight=self.hparams.vicreg_covariance_weight,
                vicreg_variance_gamma=self.hparams.vicreg_variance_gamma,
                infonce_temperature=self.hparams.infonce_temperature,
            )
        elif self.hparams.loss_type == "matryoshka_hybrid":
            self.criterion = MatryoshkaHybridLoss(
                matryoshka_dims=matryoshka_dims,
                dim_weights=self.hparams.matryoshka_weights,
                vicreg_weight=self.hparams.hybrid_vicreg_weight,
                infonce_weight=self.hparams.hybrid_infonce_weight,
                vicreg_invariance_weight=self.hparams.vicreg_invariance_weight,
                vicreg_variance_weight=self.hparams.vicreg_variance_weight,
                vicreg_covariance_weight=self.hparams.vicreg_covariance_weight,
                vicreg_variance_gamma=self.hparams.vicreg_variance_gamma,
                infonce_temperature=self.hparams.infonce_temperature,
            )
        elif self.hparams.loss_type == "enhanced_hybrid":
            self.criterion = EnhancedHybridLoss(
                matryoshka_dims=matryoshka_dims,
                dim_weights=self.hparams.matryoshka_weights,
                vicreg_weight=self.hparams.hybrid_vicreg_weight,
                vicreg_invariance_weight=self.hparams.vicreg_invariance_weight,
                vicreg_variance_weight=self.hparams.vicreg_variance_weight,
                vicreg_covariance_weight=self.hparams.vicreg_covariance_weight,
                vicreg_variance_gamma=self.hparams.vicreg_variance_gamma,
                infonce_weight=self.hparams.hybrid_infonce_weight,
                infonce_temperature=self.hparams.infonce_temperature,
                use_hard_negative=self.hparams.use_hard_negative,
                hard_negative_ratio=self.hparams.hard_negative_ratio,
                hard_negative_alpha=self.hparams.hard_negative_mix_alpha,
                use_spectral=self.hparams.use_spectral_loss,
                spectral_weight=self.hparams.spectral_weight,
                active_dims=self.active_matryoshka_dims,
            )
        else:
            raise ValueError(f"Unknown loss type: {self.hparams.loss_type}")

    def on_fit_start(self):
        if self.hparams.use_ema:
            self.ema_model = EMAModel(
                self.model,
                decay=self.hparams.ema_decay,
                update_after_step=self.hparams.ema_update_after_step,
            )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_projection: bool = False,
        dim: int | None = None,
    ) -> torch.Tensor:
        out = self.model(x, mask, return_projection=return_projection)
        if dim is not None and return_projection:
            out = out[:, :dim]
        return out

    def _compute_contrastive_loss(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        mask1: torch.Tensor,
        mask2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z1 = self.model(x1, mask1, return_projection=True)
        z2 = self.model(x2, mask2, return_projection=True)
        loss = self.criterion(z1, z2)
        return loss, z1, z2

    def _compute_multitask_loss(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        losses = {}

        if not self.hparams.enable_multitask:
            return losses

        if self.hparams.mnp_enabled and hasattr(self, "mnp_head"):
            x_masked, mnp_labels = self.mnp_head.create_masked_input(x, mask)
            encoded = self.model(x_masked, mask, return_projection=False)

            # Skip CLS token position for MNP prediction
            if self.hparams.use_cls_token:
                encoded = encoded[:, 1:]
                mnp_labels = mnp_labels[:, 1:]

            mnp_loss = self.mnp_head(encoded, mnp_labels)
            losses["mnp"] = mnp_loss * self.hparams.mnp_weight

        if self.hparams.structure_pred_enabled and hasattr(self, "structure_head"):
            encoded = self.model(x, mask, return_projection=False)
            cls_emb = encoded[:, 0] if self.hparams.use_cls_token else encoded.mean(dim=1)

            depths = x[..., 2].max(dim=1).values.long()
            sizes = mask.sum(dim=1)
            size_buckets = (sizes / 32).clamp(0, 7).long()

            struct_losses = self.structure_head(cls_emb, depths, size_buckets)
            losses["structure"] = struct_losses["total"] * self.hparams.structure_pred_weight

        return losses

    def training_step(self, batch, batch_idx):
        x1, x2 = batch
        mask1 = create_mask(x1)
        mask2 = create_mask(x2)

        contrastive_loss, z1, z2 = self._compute_contrastive_loss(x1, x2, mask1, mask2)

        # Apply self-paced learning weights
        if self.self_paced_weighter is not None:
            with torch.no_grad():
                weights = self.self_paced_weighter.compute_weights(contrastive_loss.detach())
            contrastive_loss = (contrastive_loss * weights).mean()

        multitask_losses = self._compute_multitask_loss(x1, mask1)
        total_loss = contrastive_loss + sum(multitask_losses.values())

        self.log("train/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/contrastive_loss", contrastive_loss, on_step=True, on_epoch=False)

        for name, loss in multitask_losses.items():
            self.log(f"train/{name}_loss", loss, on_step=True, on_epoch=False)

        if self.ema_model is not None:
            self.ema_model.update()

        self._train_step_count += 1
        if (
            self.hparams.log_embedding_metrics
            and self._train_step_count % self.hparams.embedding_log_interval == 0
        ):
            with torch.no_grad():
                metrics = EmbeddingMetrics.compute_all(z1, z2)
                for name, value in metrics.items():
                    self.log(f"train/embed_{name}", value, on_step=True, on_epoch=False)

                if self.collapse_detector is not None:
                    status = self.collapse_detector.check(z1)
                    self.log("train/effective_rank", status.effective_rank, on_step=True)
                    if status.is_collapsing:
                        self.log("train/collapse_warning", 1.0, on_step=True)

        return total_loss

    def on_train_epoch_end(self):
        if self.self_paced_weighter is not None:
            self.self_paced_weighter.step_epoch()
            self.log("train/self_paced_lambda", self.self_paced_weighter.current_lambda)

        if self.hparams.progressive_matryoshka:
            current_epoch = self.current_epoch
            unlock_epochs = self.hparams.matryoshka_unlock_epochs
            matryoshka_dims = self.hparams.matryoshka_dims

            for i, unlock_epoch in enumerate(unlock_epochs):
                if current_epoch >= unlock_epoch and i < len(matryoshka_dims):
                    dim = matryoshka_dims[i]
                    if dim not in self.active_matryoshka_dims:
                        self.active_matryoshka_dims.append(dim)
                        self.active_matryoshka_dims.sort()

            if hasattr(self.criterion, "active_dims"):
                self.criterion.active_dims = self.active_matryoshka_dims

    def validation_step(self, batch, batch_idx):
        x1, x2 = batch
        mask1 = create_mask(x1)
        mask2 = create_mask(x2)

        if self.ema_model is not None:
            self.ema_model.apply_shadow()

        try:
            loss, z1, z2 = self._compute_contrastive_loss(x1, x2, mask1, mask2)
        finally:
            if self.ema_model is not None:
                self.ema_model.restore()

        self.log("val/loss", loss, on_epoch=True, prog_bar=True)

        if batch_idx == 0:
            with torch.no_grad():
                metrics = EmbeddingMetrics.compute_all(z1, z2)
                for name, value in metrics.items():
                    self.log(f"val/embed_{name}", value, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        x1, x2 = batch
        mask1 = create_mask(x1)
        mask2 = create_mask(x2)
        loss, _, _ = self._compute_contrastive_loss(x1, x2, mask1, mask2)
        self.log("test/loss", loss, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        mask = create_mask(batch)

        if self.ema_model is not None:
            self.ema_model.apply_shadow()

        try:
            result = self.model(batch, mask, return_projection=False)
        finally:
            if self.ema_model is not None:
                self.ema_model.restore()

        return result

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        # Get steps per epoch from datamodule or estimate
        if self.trainer.datamodule is not None:
            train_loader = self.trainer.datamodule.train_dataloader()
            steps_per_epoch = len(train_loader) // self.trainer.accumulate_grad_batches
        else:
            steps_per_epoch = self.hparams.get("estimated_steps_per_epoch", 100)
        total_steps = steps_per_epoch * self.hparams.num_epochs

        if self.hparams.lr_scheduler == "onecycle":
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
        else:  # cosine_restarts
            # T_0 is the number of epochs until first restart
            t_0 = self.hparams.restart_epochs[0] if self.hparams.restart_epochs else 30
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=t_0 * steps_per_epoch,
                T_mult=2,
                eta_min=self.hparams.learning_rate * self.hparams.min_lr_ratio,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
