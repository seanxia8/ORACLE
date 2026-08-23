"""Minimal one-sided inverse-PSD convention required by the vendored helpers."""

from __future__ import annotations

import numpy as np


def regularize_psd(psd: np.ndarray, floor_fraction: float = 1e-8) -> np.ndarray:
    values = np.asarray(psd, dtype=np.float64)
    if np.any(~np.isfinite(values)):
        raise ValueError("PSD contains non-finite values")
    positive = values[values > 0]
    if positive.size == 0:
        raise ValueError("PSD has no positive bins")
    floor = max(float(np.median(positive)) * floor_fraction, np.finfo(float).tiny)
    return np.clip(values, floor, None)


def inverse_psd_weights(psd: np.ndarray, trace_len: int, zero_dc: bool = True) -> np.ndarray:
    values = regularize_psd(psd)
    expected = int(trace_len) // 2 + 1
    if values.shape[0] != expected:
        raise ValueError(f"PSD length {values.shape[0]} does not match rfft bins {expected}")
    weights = np.zeros_like(values)
    if trace_len % 2 == 0:
        weights[1:-1] = 2.0 / values[1:-1]
        weights[-1] = 1.0 / (2.0 * values[-1])
    else:
        weights[1:] = 2.0 / values[1:]
    if not zero_dc:
        weights[0] = 1.0 / values[0]
    return weights
