from __future__ import annotations
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from jaxtyping import Float

from reconstruction_model.models.muon import Muon


@dataclass
class TransformerConfig:
    d_model: int = 256
    d_ff: int = 1024
    max_seq_len: int = 65536
    patch_len: int = 64
    patch_stride: int = 64
    n_head: int = 4
    n_time_layers: int = 1
    n_channel_layers: int = 1
    rope_base: float = 10000.0
    norm_eps: float = 1e-6


def normalise_input_sequence(x: Tensor, eps: float = 1e-6) -> Tensor:
    x_mean = torch.nanmean(x, dim=-1, keepdim=True)
    x_std = torch.std(x, dim=-1, keepdim=True)
    return (x - x_mean) / (x_std + eps)


def patch_input_sequence(x: Tensor, patch_len: int, patch_stride: int) -> Tensor:
    # (B * C, N * P) -> (B * C, N, P)
    return x.unfold(dimension=-1, size=patch_len, step=patch_stride)


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


def apply_rotary_embeddings(x: Float[Tensor, ""], rope_cache: Float[Tensor, ""]):
    ndim = x.ndim
    assert ndim == 4
    # (B * C, N, n_h, d_h) -> (B * C, N, n_h, d_h // 2, 2)
    x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
    # (N, d_h // 2, 2) -> (1, N, 1, d_h // 2, 2)
    rope_cache = rope_cache[: x.size(1)].unsqueeze(0).unsqueeze(2)
    # (B * C, N, n_h, d_h // 2, 2, 2)
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


class MultiHeadAttention(nn.Module):
    def __init__(self, config, use_rope: bool):
        super().__init__()
        self.use_rope = use_rope
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.d_head = self.d_model // self.n_head
        self.qkv_proj = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        self.o_proj = nn.Linear(self.d_model, self.d_model)

    def forward(self, x, rope_cache):
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
        attn = F.scaled_dot_product_attention(q, k, v).transpose(1, 2)
        # (B * C, N, d_model)
        attn_combined = attn.reshape(-1, x.size(1), x.size(-1))
        return self.o_proj(attn_combined)


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, max_patches: int, d_model: int):
        super().__init__()
        # (1, N, d_model)
        self.pos_embed = nn.Parameter(torch.empty(1, max_patches, d_model))

    def forward(self, x):
        return x + self.pos_embed[:, : x.size(1), :]


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.up_proj(x)) * self.gate_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config, use_rope: bool):
        super().__init__()
        self.attn_norm = nn.RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = MultiHeadAttention(config, use_rope)
        self.ffn_norm = nn.RMSNorm(config.d_model, eps=config.norm_eps)
        self.ffn = FFN(config.d_model, config.d_ff)

    def forward(self, x, rope_cache):
        x = x + self.attn(self.attn_norm(x), rope_cache)
        x = x + self.ffn(self.ffn_norm(x))
        return x


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

        self.patch_embedding = nn.Linear(config.patch_len, config.d_model, bias=True)
        self.temporal_layers = nn.ModuleList(
            [TransformerBlock(config, True) for _ in range(config.n_time_layers)]
        )
        self.abs_pos_embd = AbsolutePositionalEmbedding(
            self.max_patches, config.d_model
        )
        self.spatial_layers = nn.ModuleList(
            [TransformerBlock(config, False) for _ in range(config.n_channel_layers)]
        )
        self.final_proj_norm = nn.RMSNorm(config.d_model, config.norm_eps)
        self.spatial_head = nn.Linear(config.d_model, 3, bias=True)
        self.energy_head = nn.Linear(config.d_model, 1, bias=True)

    def init_weights(self):
        self.apply(self._init_weights)
        for temporal_block in self.temporal_layers:
            nn.init.zeros_(temporal_block.attn.o_proj.weight)
            nn.init.zeros_(temporal_block.ffn.down_proj.weight)
        for spatial_block in self.spatial_layers:
            nn.init.zeros_(spatial_block.attn.o_proj.weight)
            nn.init.zeros_(spatial_block.ffn.down_proj.weight)
        nn.init.zeros_(self.spatial_head.weight)
        nn.init.zeros_(self.energy_head.weight)

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
        muon_lr: float = 0.001,
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
        block_weight_params = []
        aux_params = []
        block_weight_params = []
        for p in self.temporal_layers.parameters():
            if p.ndim >= 2:
                block_weight_params.append(p)
            else:
                aux_params.append(p)
        for p in self.spatial_layers.parameters():
            if p.ndim >= 2:
                block_weight_params.append(p)
            else:
                aux_params.append(p)
        aux_params.extend(self.abs_pos_embd.parameters())
        aux_params.extend(self.final_proj_norm.parameters())
        adamw_param_groups = [
            dict(params=self.patch_embedding.parameters()),
            dict(params=self.spatial_head.parameters()),
            dict(params=self.energy_head.parameters()),
            dict(params=aux_params),
        ]
        adamw_optimiser = AdamW(adamw_param_groups, **adamw_kwargs)

        muon_kwargs = dict(
            lr=muon_lr,
            momentum=muon_momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
        )
        muon_optimiser = Muon(block_weight_params, **muon_kwargs)
        optimisers = [adamw_optimiser, muon_optimiser]
        return optimisers

    def forward(self, x):
        batch_size, channel_dim, seq_len = x.size()
        # (B, C, N * P) -> (B * C, N * P)
        x = x.reshape(batch_size * channel_dim, seq_len)
        x = normalise_input_sequence(x)
        # (B * C, N * P) -> (B * C, N, P)
        x = patch_input_sequence(x, self.config.patch_len, self.config.patch_stride)
        # (B * C, N, P) -> (B * C, N, d_model)
        x = self.patch_embedding(x)
        for layer in self.temporal_layers:
            x = layer(x, self._rope_cache)
        # (B * C, N, d_model) -> (B, C, N, d_model)
        x = x.view(batch_size, channel_dim, -1, self.config.d_model)
        # (B, C, N, d_model) -> (B, N, C, d_model) -> (B * N, C, d_model)
        x = x.transpose(1, 2).reshape(-1, channel_dim, self.config.d_model)
        x = self.abs_pos_embd(x)
        for layer in self.spatial_layers:
            x = layer(x, None)
        x = self.final_proj_norm(x)
        # (B * N, C, d_model) -> (B, N, C, d_model)
        x = x.view(batch_size, -1, channel_dim, self.config.d_model)
        # (B, N, C, d_model) -> (B, d_model)
        x = x.mean(dim=(1, 2))
        # (B, d_model) -> (B, 3)
        spatial_pred = self.spatial_head(x)
        # (B, d_model) -> (B, 1)
        energy_pred = self.energy_head(x)
        return spatial_pred, energy_pred
