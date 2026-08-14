from __future__ import annotations
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
import numpy as np

from reconstruction_model.models.muon import Muon

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_PACKAGE_ROOT = Path(__file__).parent
_CACHE_DIR = _PACKAGE_ROOT / "cache"

@dataclass
class TransformerConfig:
    max_seq_len: int = 65536
    n_head: int = 4
    d_model_channel: int = 64
    d_ff_channel: int = 256
    d_model_temporal: int = 56
    d_ff_temporal: int = 224
    norm_eps: float = 1e-6
    n_time_blocks: int = 2
    n_channel_blocks: int = 5
    rope_base: float = 10000.0
    n_channels: int = 56
    dropout: float = 0.3
    pairwise_feats_path: str | Path = _CACHE_DIR / "pos_diff.npy"


def precompute_rope_angles(
    max_seq_len: int,
    d_head: int,
    base: float = 10000.0,
) -> Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, d_head, 2)[: (d_head // 2)]) / d_head)
    t = torch.arange(max_seq_len, device=freqs.device)  # (N,)
    rope_angles = torch.outer(t, freqs)  # (N, d_head // 2)

    return torch.stack(
        [rope_angles.cos(), rope_angles.sin()], dim=-1
    )  # (N, d_h // 2, 2)


def apply_rotary_embeddings(x: Tensor, rope_cache: Tensor):
    ndim = x.ndim
    assert ndim == 4

    # (N, d_h // 2, 2) -> (1, N, 1, d_h // 2, 2)
    rope_cache = rope_cache[:x.size(1)].unsqueeze(0).unsqueeze(2)
    # (B * C, N, n_h, d_h) -> (B * C, N, n_h, d_h // 2, 2)
    x_reshaped = x.reshape(*x.shape[:-1], -1, 2)

    # (B * C, N, n_h, d_h // 2, 2)
    x_out = torch.stack(
        [
            rope_cache[..., 0] * x_reshaped[..., 0]
            - rope_cache[..., 1] * x_reshaped[..., 1],
            rope_cache[..., 1] * x_reshaped[..., 0]
            + rope_cache[..., 0] * x_reshaped[..., 1],
        ],
        dim=-1,
    )
    # (B * C, N, n_h, d_h // 2, 2, 2) -> (B * C, N, n_h, d_h // 2, 4)
    return x_out.flatten(-2).type_as(x)

def load_pairwise_features_as_tensor(filepath):
    pairwise_features = np.load(filepath)
    return torch.from_numpy(pairwise_features).float().flatten(0, 1)

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        config,
        d_model,
        use_rope: bool,
    ):
        super().__init__()
        self.use_rope = use_rope
        self.d_model = d_model
        self.n_head = config.n_head
        self.d_head = self.d_model // self.n_head
        self.qkv_proj = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        self.o_proj = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, rope_cache, attn_mask: Tensor | None = None):
        # (B * C, N, 3 * d_model) -> (B * C, N, 3 * d_model)
        qkv = self.qkv_proj(x)
        # (B * C, N, 3 * d_model) -> ..., (B * C, N, d_model)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # (B * C, N, d_model) -> (B * C, N, n_head, d_head)
        q = q.view(-1, q.size(1), self.n_head, self.d_head)
        k = k.view(-1, k.size(1), self.n_head, self.d_head)
        # (B * C, N, d_model) -> (B * C, n_head, N, d_head)
        v = v.view(-1, v.size(1), self.n_head, self.d_head).transpose(1, 2)
        if self.use_rope:
            q = apply_rotary_embeddings(q, rope_cache)
            k = apply_rotary_embeddings(k, rope_cache)
        # (B * C, N, n_head, d_head) -? (B * C, n_head, N, d_head)
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        # (B * C, N, n_head, d_head)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask).transpose(1, 2)
        # (B * C, N, d_model)
        attn_combined = attn.reshape(-1, x.size(1), x.size(-1))
        return self.dropout(self.o_proj(attn_combined))

class AttentionPooling(nn.Module):
    def __init__(self, config, d_model):
        super().__init__()
        # Inbuilt torch MHA for cross attention, if it works will be replaced
        # by own implementation
        self.attn = nn.MultiheadAttention(
            d_model,
            config.n_head,
            batch_first=True,
        )
        self.query = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.normal_(self.query, mean=0.0, std=0.02)

    def forward(self, x):
        # (1, C, d_model) -> (B, C, d_model)
        query = self.query.expand(x.size(0), -1, -1)
        # (B, C, d_model) -> (B, 1, d_model)
        pooled, _ = self.attn(query, x, x)
        # (B, d_model)
        return pooled.squeeze(-2)


class TransformerBlock(nn.Module):
    def __init__(self, config, d_model, d_ff, use_rope: bool):
        super().__init__()
        self.attn_norm = nn.RMSNorm(d_model, eps=config.norm_eps)
        self.attn = MultiHeadAttention(config, d_model, use_rope)
        self.ffn_norm = nn.RMSNorm(d_model, eps=config.norm_eps)
        self.ffn = FFN(d_model, d_ff)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, rope_cache, attn_mask: Tensor | None = None):
        x = x + self.attn(self.attn_norm(x), rope_cache, attn_mask=attn_mask)
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.up_proj(x)) * self.gate_proj(x))

class AttentionBiasFFN(nn.Module):
    def __init__(self, n_channels: int, hidden_dim: int):
        super().__init__()
        self.pairwise_feat_ffn = nn.Sequential(
            # (C * C, 3) @ (3, h) -> (C * C, h)
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            # (C * C, h) @ (h, 1) -> (C * C, 1)
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.pairwise_feat_ffn(x)

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, n_channels: int, d_model: int):
        super().__init__()
        # (1, C, d_model)
        self.pos_embed = nn.Parameter(torch.empty(1, n_channels, d_model))
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def forward(self, x):
        return x + self.pos_embed[:, : x.size(1), :]


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.register_buffer(
            "_pairwise_features",
            load_pairwise_features_as_tensor(
                config.pairwise_feats_path,
            ),
            persistent=False,
        )
        # Lout = floor((Lin - k) / s) + 1
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(
                config.n_channels,
                config.n_channels * 32,
                kernel_size=64,
                stride=16,
                groups=config.n_channels,
            ),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Conv1d(
                config.n_channels * 32,
                config.n_channels,
                kernel_size=16,
                stride=8,
                groups=config.n_channels,
            ),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.AdaptiveAvgPool1d(config.d_model_channel),
        )
        self.rel_pos_bias_ffn = AttentionBiasFFN(config.n_channels, config.d_model_channel)
        self.spatial_blocks = nn.ModuleList(
            [TransformerBlock(config, config.d_model_channel, config.d_ff_channel, False) for _ in range(config.n_channel_blocks)]
        )
        self.spatial_head = nn.Sequential(
            nn.Linear(config.d_model_channel * config.n_channels, 256),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 2)
        )
        # Fully Connected Layer for Energy Regression (Uses ReLU)
        self.energy_head = nn.Sequential(
            nn.Linear(config.d_model_channel * config.n_channels, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 1) # Predicts normalised energy
        )
        self.classification_head = nn.Sequential(
            nn.Linear(config.d_model_channel * config.n_channels, 256),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 1)
        )

    def init_weights(self):
        # Recursively apply muP initialisation to all nn.Linear modules
        self.apply(self._init_weights)
        # Selectively zero out certain weights
        for block in self.spatial_blocks:
            nn.init.zeros_(block.attn.o_proj.weight)
            nn.init.zeros_(block.ffn.down_proj.weight)
        for module in self.spatial_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
        for module in self.energy_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Spectral condition muP initialisation
            # sigma = min(1, root(d_out / d_in)) / root(d_in)
            # For more information: https://arxiv.org/pdf/2310.17813
            d_out, d_in = module.weight.size(0), module.weight.size(1)
            sigma = min(1.0, math.sqrt(d_out / d_in)) / math.sqrt(d_in)
            nn.init.normal_(module.weight, mean=0.0, std=sigma)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def configure_optimisers(
        self,
        adamw_lr: float = 0.001,
        adamw_betas: tuple[float] = (0.9, 0.999),
        adamw_weight_decay: float = 0.0,
        adamw_fused: bool = True,
        use_muon: bool = True,
        muon_lr: float = 0.001,
        muon_weight_decay: float = 0.0,
        muon_momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ):
        adamw_kwargs = dict(
            lr=adamw_lr,
            betas=adamw_betas,
            weight_decay=adamw_weight_decay,
            fused=adamw_fused,
        )
        aux_params = []
        block_weight_params = []
        for p in self.spatial_blocks.parameters():
            if p.ndim >= 2:
                block_weight_params.append(p)
            else:
                aux_params.append(p)

        if use_muon:
            adamw_param_groups = [
                dict(params=self.feature_extractor.parameters()),
                dict(params=self.rel_pos_bias_ffn.parameters()),
                dict(params=self.spatial_head.parameters()),
                dict(params=self.energy_head.parameters()),
                dict(params=self.classification_head.parameters()),
                dict(params=aux_params),
            ]
            muon_kwargs = dict(
                lr=muon_lr,
                momentum=muon_momentum,
                nesterov=nesterov,
                ns_steps=ns_steps,
            )
            muon_optimiser = Muon(block_weight_params, **muon_kwargs)
        else:
            adamw_param_groups = [
                dict(params=self.feature_extractor.parameters()),
                dict(params=self.rel_pos_bias_ffn.parameters()),
                dict(params=self.spatial_head.parameters()),
                dict(params=self.energy_head.parameters()),
                dict(params=self.classification_head.parameters()),
                dict(params=aux_params),
                dict(params=block_weight_params),
            ]
            muon_optimiser = None
        adamw_optimiser = AdamW(adamw_param_groups, **adamw_kwargs)

        return adamw_optimiser, muon_optimiser

    def forward(self, x):
        # (B, C, T) -> (B, C, 64)
        x = self.feature_extractor(x)
        rel_pos_bias = self.rel_pos_bias_ffn(self._pairwise_features).view(
            self.config.n_channels,
            self.config.n_channels,
        )
        for block in self.spatial_blocks:
            x = block(x, None, attn_mask=rel_pos_bias)

        x = x.flatten(start_dim=1)
        # (B, C, d_model) -> (B, d_model)
        spatial_pred = self.spatial_head(x)
        energy_pred = self.energy_head(x)
        class_logits = self.classification_head(x)
        return spatial_pred, energy_pred, class_logits
