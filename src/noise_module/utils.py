# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the modular noise simulator written for the ORACLE study.
# If you use this module in published work, please cite it: see CITATION.cff
# at the repository root.
"""Shared helpers for modular noise generation."""

from __future__ import annotations

from typing import Any

import numpy as np


def resolve_rng(rng: Any = None, seed: int | None = None) -> np.random.Generator:
    """Return a NumPy Generator from a seed, Generator, or integer-like input."""
    if rng is not None and seed is not None:
        raise ValueError("Pass either `rng` or `seed`, not both.")
    if isinstance(rng, np.random.Generator):
        return rng
    if rng is None:
        return np.random.default_rng(seed)
    return np.random.default_rng(rng)


def spawn_rng(rng: np.random.Generator) -> np.random.Generator:
    """Create a child generator without sharing stateful global randomness."""
    return np.random.default_rng(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))


def sample_range(
    rng: np.random.Generator,
    bounds: list[float] | tuple[float, float] | np.ndarray | float,
    size: int | tuple[int, ...] | None = None,
) -> np.ndarray | float:
    """Sample uniformly from a scalar or [low, high] interval."""
    if np.isscalar(bounds):
        value = float(bounds)
        if size is None:
            return value
        return np.full(size, value, dtype=float)
    low, high = float(bounds[0]), float(bounds[1])
    if size is None:
        return float(rng.uniform(low, high))
    return rng.uniform(low, high, size=size)


def concatenate_with_crossfade(
    segments: list[np.ndarray], crossfade_len: int = 0
) -> np.ndarray:
    """Concatenate arrays along the last axis with a cosine crossfade."""
    if not segments:
        raise ValueError("At least one segment is required.")

    output = np.array(segments[0], copy=True)
    for segment in segments[1:]:
        current = np.array(segment, copy=False)
        overlap = min(int(crossfade_len), output.shape[-1], current.shape[-1])
        if overlap > 0:
            alpha = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, overlap))
            reshape = (1,) * (output.ndim - 1) + (overlap,)
            alpha = alpha.reshape(reshape)
            output[..., -overlap:] = (
                (1.0 - alpha) * output[..., -overlap:] + alpha * current[..., :overlap]
            )
            output = np.concatenate([output, current[..., overlap:]], axis=-1)
        else:
            output = np.concatenate([output, current], axis=-1)
    return output


def match_target_std(
    x: np.ndarray,
    target_std: float,
    axis: int = -1,
    eps: float = 1e-12,
) -> np.ndarray:
    """Rescale an array to a target standard deviation along an axis."""
    current_std = np.std(x, axis=axis, keepdims=True)
    scale = target_std / np.maximum(current_std, eps)
    return x * scale


def mean_offdiag_corrcoef(X: np.ndarray) -> float:
    """Return the mean off-diagonal correlation for channels x samples data."""
    if X.ndim != 2 or X.shape[0] < 2:
        return 0.0
    corr = np.corrcoef(X)
    offdiag = corr[~np.eye(corr.shape[0], dtype=bool)]
    return float(np.mean(offdiag))


def to_jsonable(value: Any) -> Any:
    """Recursively convert simulator metadata to JSON-compatible values."""
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {"real": value.real.tolist(), "imag": value.imag.tolist()}
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value
