"""TIDMAD reconstruction model: the ``current_compact`` backbone plus a
per-patch reconstruction head (PLAN_04 §3.3).

The backbone is reused unchanged through the temporal stage; the cross-band
stage replaces the backbone's patch-scaled absolute embedding with a band
embedding dimensioned to the number of bands ``C``. This is deliberate (round-2
correction T2): the backbone sizes its absolute positional embedding over the
*channel* axis from ``max_seq_len // patch_len``, which would otherwise force the
frame count ``M`` to scale with the band count. Decoupling the band embedding
lets the window be chosen on scientific grounds. Positional embeddings are
otherwise exactly as the backbone ships them: RoPE on the temporal axis, learned
absolute embeddings across bands.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from reconstruction_model.model import (
    AbsolutePositionalEmbedding,
    Transformer,
    TransformerConfig,
    normalise_input_sequence,
)


class TidmadTransformer(Transformer):
    """``current_compact`` backbone with a reconstruction head.

    Input and output use the external ``(B, 2F, M)`` real-valued measurement
    representation (all real rows followed by all imaginary rows). Internally,
    real and imaginary values are paired as two features of each physical
    frequency band, so cross-band attention has ``F`` tokens rather than
    ``2F`` unrelated tokens.
    """

    def __init__(self, config: TransformerConfig, n_bands: int | None = None):
        super().__init__(config)
        # Remove the task heads of the DELight model and the backbone's
        # patch-scaled absolute embedding (replaced by a band embedding below).
        del self.spatial_head
        del self.energy_head
        del self.abs_pos_embd
        self.patch_len = int(config.patch_len)
        if self.max_patches * self.patch_len != config.max_seq_len:
            raise ValueError(
                f"max_seq_len ({config.max_seq_len}) must be an exact multiple of "
                f"patch_len ({self.patch_len}); got {self.max_patches} patches"
            )
        self.n_bands = int(n_bands) if n_bands is not None else self.max_patches
        del self.patch_embedding
        self.patch_embedding = nn.Linear(2 * self.patch_len, config.d_model, bias=True)
        self.band_pos_embd = AbsolutePositionalEmbedding(self.n_bands, config.d_model)
        self.reconstruction_head = nn.Linear(config.d_model, 2 * self.patch_len, bias=True)

    def init_weights(self):
        # Replicate the backbone init (muP spectral init + zeroed residual
        # projections) without the base class's deleted head zeroing.
        self.apply(self._init_weights)
        for temporal_block in self.temporal_layers:
            nn.init.zeros_(temporal_block.attn.o_proj.weight)
            nn.init.zeros_(temporal_block.ffn.down_proj.weight)
        for spatial_block in self.spatial_layers:
            nn.init.zeros_(spatial_block.attn.o_proj.weight)
            nn.init.zeros_(spatial_block.ffn.down_proj.weight)
        nn.init.normal_(self.band_pos_embd.pos_embed, std=0.02)
        nn.init.zeros_(self.reconstruction_head.weight)
        nn.init.zeros_(self.reconstruction_head.bias)

    def configure_optimisers(self, **kwargs):
        """Reuse the backbone parameter split but treat the head as aux params."""
        adamw_kwargs = dict(
            lr=kwargs.get("adamw_lr", 1e-3),
            betas=kwargs.get("adamw_betas", (0.9, 0.999)),
            weight_decay=kwargs.get("adamw_weight_decay", 0.0),
            fused=kwargs.get("adamw_fused", True),
        )
        block_weight_params: list[torch.nn.Parameter] = []
        aux_params: list[torch.nn.Parameter] = []
        for block in (*self.temporal_layers, *self.spatial_layers):
            for p in block.parameters():
                (block_weight_params if p.ndim >= 2 else aux_params).append(p)
        aux_params.extend(self.band_pos_embd.parameters())
        aux_params.extend(self.final_proj_norm.parameters())
        aux_params.extend(self.reconstruction_head.parameters())
        from torch.optim import AdamW

        from reconstruction_model.muon import Muon

        adamw = AdamW(
            [
                dict(params=self.patch_embedding.parameters()),
                dict(params=aux_params),
            ],
            **adamw_kwargs,
        )
        muon = Muon(
            block_weight_params,
            lr=kwargs.get("muon_lr", 1e-3),
            momentum=kwargs.get("muon_momentum", 0.95),
            nesterov=kwargs.get("muon_nesterov", True),
            ns_steps=kwargs.get("muon_ns_steps", 5),
        )
        return [adamw, muon]

    def _temporal_stage(self, x: torch.Tensor) -> torch.Tensor:
        """Input ``(B, 2F, M)`` -> temporal activations ``(B*F, N, d)``."""
        batch_size, stacked_dim, seq_len = x.size()
        if stacked_dim != 2 * self.n_bands:
            raise ValueError(
                f"expected 2F={2 * self.n_bands} stacked real/imag rows, got {stacked_dim}"
            )
        # Match the original per-row standardisation, then pair real/imaginary
        # components inside each non-overlapping temporal patch.
        x = normalise_input_sequence(x.reshape(batch_size * stacked_dim, seq_len))
        x = x.view(batch_size, 2, self.n_bands, seq_len).permute(0, 2, 3, 1)
        x = x.unfold(2, self.patch_len, self.config.patch_stride)
        # unfold: (B,F,N,2,P) -> (B*F,N,2P), one physical-frequency token.
        x = x.permute(0, 1, 2, 4, 3).reshape(
            batch_size * self.n_bands, self.max_patches, 2 * self.patch_len
        )
        x = self.patch_embedding(x)
        for layer in self.temporal_layers:
            x = layer(x, self._rope_cache)
        return x

    def _band_stage(self, temporal: torch.Tensor, batch_size: int, channel_dim: int) -> torch.Tensor:
        """Temporal activations -> ``(B, N, C, d)`` after band embedding + spatial layers + norm."""
        x = temporal.view(batch_size, channel_dim, -1, self.config.d_model)
        x = x.transpose(1, 2).reshape(-1, channel_dim, self.config.d_model)
        x = self.band_pos_embd(x)
        for layer in self.spatial_layers:
            x = layer(x, None)
        x = self.final_proj_norm(x)
        return x.view(batch_size, -1, channel_dim, self.config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, stacked_dim, _ = x.size()
        if stacked_dim != 2 * self.n_bands:
            raise ValueError(f"expected {2 * self.n_bands} input rows, got {stacked_dim}")
        x = self._temporal_stage(x)
        x = self._band_stage(x, batch_size, self.n_bands)
        x = self.reconstruction_head(x)  # (B,N,F,2*patch_len)
        x = x.view(batch_size, self.max_patches, self.n_bands, 2, self.patch_len)
        x = x.permute(0, 3, 2, 1, 4).reshape(batch_size, 2 * self.n_bands, -1)
        return x  # (B,2F,M), real rows followed by imaginary rows
