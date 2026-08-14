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
    patch_len: int = 128
    patch_stride: int = 128
    d_model: int  = 128
    d_ff: int = 512
    n_channel_blocks: int = 4
    n_temporal_blocks: int = 4

    n_head: int = 4
    dropout_transformer: float = 0.15

    d_hidden_bias_ffn: int = 64
    dropout_ffn: float = 0.15

    rope_base: float = 10000.0
    n_channels: int = 56
    norm_eps: float = 1e-6
    n_edge_feats: int = 4

    pairwise_feats_path: str | Path = _CACHE_DIR / "edge_feats.npy"

def patch_input_sequence(x: Tensor, patch_len: int, patch_stride: int) -> Tensor:
    # (B * C, N * P) -> (B * C, N, P)
    return x.unfold(-1, patch_len, patch_stride)

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
        use_cross_attn: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.use_rope = use_rope
        self.use_cross_attn = use_cross_attn
        self.d_model = d_model
        self.n_head = config.n_head
        self.d_head = self.d_model // self.n_head
        if use_cross_attn:
            self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
            self.kv_proj = nn.Linear(self.d_model, 2 * self.d_model, bias=False)
        else:
            self.qkv_proj = nn.Linear(self.d_model, 3 * self.d_model, bias=False)

        self.o_proj = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()
    
    def _reset_parameters(self):
        if self.use_cross_attn:
            nn.init.xavier_uniform_(self.q_proj.weight)
            nn.init.xavier_uniform_(self.kv_proj.weight)
        else:
            nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)


    def forward(
        self,
        x,
        rope_cache,
        kv_input: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ):
        if self.use_cross_attn:
            if kv_input is None:
                raise ValueError("kv_input required for cross attention")
            # (B * C, N, d_model) -> (B * C, N, d_model)
            q = self.q_proj(x)
            # (B * C, N, 2 * d_model) -> (B * C, N, 2 * d_model)
            kv = self.kv_proj(kv_input)
            k, v = torch.chunk(kv, 2, dim=-1)
        else:
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
        # (B * C, N, n_head, d_head) -> (B * C, n_head, N, d_head)
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        # (B * C, N, n_head, d_head)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask).transpose(1, 2)
        # (B * C, N, d_model)
        attn_combined = attn.reshape(-1, x.size(1), self.d_model)
        return self.dropout(self.o_proj(attn_combined))


class AttentionPooling(nn.Module):
    def __init__(self, config, d_model):
        super().__init__()
        self.attn = MultiHeadAttention(
            config,
            d_model,
            use_rope=False,
            use_cross_attn=True,
            dropout=0.0,
        )
        self.query = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.normal_(self.query, mean=0.0, std=0.02)

    def forward(self, x):
        # (1, N, d_model) -> (B * C, N, d_model)
        query = self.query.expand(x.size(0), -1, -1)
        # (B, * C, N, d_model) -> (B * C, 1, d_model)
        pooled = self.attn(query, rope_cache=None, kv_input=x, attn_mask=None)
        # (B * C, 1, d_model) -> (B * C, d_model)
        return pooled.squeeze(-2)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        config,
        d_model: int,
        d_ff: int,
        use_rope: bool,
    ):
        super().__init__()
        self.attn_norm = nn.RMSNorm(d_model, eps=config.norm_eps)
        self.attn = MultiHeadAttention(
            config,
            d_model,
            use_rope,
            use_cross_attn=False,
            dropout=config.dropout_transformer,
        )
        self.ffn_norm = nn.RMSNorm(d_model, eps=config.norm_eps)
        self.ffn = FFN(
            d_model,
            d_ff,
            dropout=config.dropout_transformer,
        )
    def forward(
        self,
        x,
        rope_cache,
        attn_mask: Tensor | None = None,
    ):
        x = x + self.attn(self.attn_norm(x), rope_cache, kv_input=None, attn_mask=attn_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class FFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.down_proj(F.silu(self.up_proj(x)) * self.gate_proj(x)))

class AttentionBiasFFN(nn.Module):
    def __init__(self, n_channels: int, n_edge_feats: int, hidden_dim: int):
        super().__init__()
        self.pairwise_feat_ffn = nn.Sequential(
            # (C * C, 3) @ (3, h) -> (C * C, h)
            nn.Linear(n_edge_feats, hidden_dim),
            nn.GELU(),
            # (C * C, h) @ (h, 1) -> (C * C, 1)
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.pairwise_feat_ffn(x)


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.max_patches = config.max_seq_len // config.patch_len
        d_head = config.d_model // config.n_head
        self.register_buffer(
            "_rope_cache",
            precompute_rope_angles(
                self.max_patches,
                d_head,
                config.rope_base,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_pairwise_features",
            load_pairwise_features_as_tensor(
                config.pairwise_feats_path,
            ),
            persistent=False,
        )
        self.patch_embedding = nn.Linear(config.patch_len, config.d_model, bias=True)
        self.temporal_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config,
                    config.d_model,
                    config.d_ff,
                    use_rope=True,
                )
                for _ in range(config.n_temporal_blocks)
            ]
        )
        self.attn_pool = AttentionPooling(config, config.d_model)
        self.rel_pos_bias_ffn = AttentionBiasFFN(
            config.n_channels,
            config.n_edge_feats,
            config.d_hidden_bias_ffn,
        )
        self.spatial_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config,
                    config.d_model,
                    config.d_ff,
                    use_rope=False,
                )
                for _ in range(config.n_channel_blocks)
            ]
        )
        self.final_norm = nn.RMSNorm(config.d_model * config.n_channels, eps=config.norm_eps)
        self.spatial_head = nn.Sequential(
            nn.Linear(config.n_channels * config.d_model, 128),
            nn.GELU(),
            nn.Dropout(config.dropout_ffn),
            nn.Linear(128, 2)
        )
        self.energy_head = nn.Sequential(
            nn.Linear(config.d_model * config.n_channels, 128),
            nn.GELU(),
            nn.Dropout(config.dropout_ffn),
            nn.Linear(128, 1)
        )
        self.classification_head = nn.Sequential(
            nn.Linear(config.n_channels * config.d_model, 128),
            nn.GELU(),
            nn.Dropout(config.dropout_ffn),
            nn.Linear(128, 1)
        )

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

        for p in self.temporal_blocks.parameters():
            if p.ndim >= 2:
                block_weight_params.append(p)
            else:
                aux_params.append(p)

        for p in self.spatial_blocks.parameters():
            if p.ndim >= 2:
                block_weight_params.append(p)
            else:
                aux_params.append(p)

        if use_muon:
            adamw_param_groups = [
                dict(params=self.patch_embedding.parameters()),
                dict(params=self.attn_pool.parameters()),
                dict(params=self.rel_pos_bias_ffn.parameters()),
                dict(params=self.final_norm.parameters()),
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
                dict(params=self.patch_embedding.parameters()),
                dict(params=self.attn_pool.parameters()),
                dict(params=self.rel_pos_bias_ffn.parameters()),
                dict(params=self.final_norm.parameters()),
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
        # (B, C, N * P) -> (B * C, N * P)
        batch_size, _, seq_len = x.size()
        x = x.reshape(batch_size * self.config.n_channels, seq_len)
        # (B * C, N * P) -> (B * C, N, P)
        x = patch_input_sequence(x, self.config.patch_len, self.config.patch_stride)
        # (B * C, N, P) -> (B * C, N, d_model)
        x = self.patch_embedding(x)
        for block in self.temporal_blocks:
            x = block(x, self._rope_cache, attn_mask=None,)
        
        # (B * C, N, d_model) -> (B * C, d_model)
        x = self.attn_pool(x)
        # (B * C, d_model) -> (B, C, d_model)
        x = x.reshape(batch_size, self.config.n_channels, self.config.d_model)

        rel_pos_bias = self.rel_pos_bias_ffn(self._pairwise_features).view(
            self.config.n_channels,
            self.config.n_channels,
        )
        for block in self.spatial_blocks:
            x = block(x, None, attn_mask=rel_pos_bias)
        # (B, C, d_model) -> (B, C * d_model)
        x = self.final_norm(x.flatten(start_dim=1))
        spatial_pred = self.spatial_head(x)
        energy_pred = self.energy_head(x)
        class_logits = self.classification_head(x)
        return spatial_pred, energy_pred, class_logits
