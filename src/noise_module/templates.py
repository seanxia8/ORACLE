# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the modular noise simulator written for the ORACLE study.
# If you use this module in published work, please cite it: see CITATION.cff
# at the repository root.
"""Synthetic pulse and artifact templates."""

from __future__ import annotations

import numpy as np


def pulse_template_2(
    t: np.ndarray,
    t0: float,
    An: float,
    At: float,
    tau_n: float,
    tau_in: float,
    tau_t: float,
) -> np.ndarray:
    """Return the causal two-component athermal detector pulse.

    The convention follows Cait's ``pulse_template_2``/the Probst model.
    Values before ``t0`` are zero. Time constants are in the same units as
    ``t`` and must be positive.
    """
    t = np.asarray(t, dtype=float)
    parameters = np.asarray([t0, An, At, tau_n, tau_in, tau_t], dtype=float)
    if np.any(~np.isfinite(parameters)):
        raise ValueError("Pulse-template parameters must be finite.")
    if min(tau_n, tau_in, tau_t) <= 0.0:
        raise ValueError("Pulse-template time constants must be positive.")

    elapsed = t - t0
    pulse = np.zeros_like(elapsed)
    active = elapsed >= 0.0
    u = elapsed[active]
    pulse[active] = (
        An * (np.exp(-u / tau_n) - np.exp(-u / tau_in))
        + At * (np.exp(-u / tau_t) - np.exp(-u / tau_n))
    )
    return pulse


def generate_glitch_template(
    kind: str,
    duration: int,
    sampling_frequency: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build a normalized transient glitch template."""
    duration = max(int(duration), 4)
    t = np.arange(duration, dtype=float) / max(float(sampling_frequency), 1.0)
    unit = np.zeros(duration, dtype=float)
    center = duration // 2

    if kind == "impulse":
        unit[center] = 1.0
    elif kind == "exp_decay":
        tau = max(duration / max(sampling_frequency, 1.0) / 6.0, 1e-6)
        unit[center:] = np.exp(-(t[center:] - t[center]) / tau)
    elif kind == "damped_sine":
        freq = rng.uniform(0.05, 0.25) * sampling_frequency
        tau = max(duration / max(sampling_frequency, 1.0) / 4.0, 1e-6)
        envelope = np.exp(-(t - t[0]) / tau)
        unit = envelope * np.sin(2.0 * np.pi * freq * t)
    elif kind == "ringing":
        freq = rng.uniform(0.02, 0.15) * sampling_frequency
        sigma = max(duration / max(sampling_frequency, 1.0) / 8.0, 1e-6)
        center_t = t[center]
        envelope = np.exp(-0.5 * ((t - center_t) / sigma) ** 2)
        unit = envelope * np.cos(2.0 * np.pi * freq * (t - center_t))
    else:
        raise ValueError(f"Unsupported glitch template kind: {kind}")

    max_abs = np.max(np.abs(unit))
    if max_abs == 0:
        return unit
    return unit / max_abs


def generate_burst_template(
    duration: int,
    sampling_frequency: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build a short Gaussian-windowed burst."""
    duration = max(int(duration), 8)
    t = np.arange(duration, dtype=float) / max(float(sampling_frequency), 1.0)
    center_t = t[(duration - 1) // 2]
    sigma = max(duration / max(sampling_frequency, 1.0) / 6.0, 1e-6)
    envelope = np.exp(-0.5 * ((t - center_t) / sigma) ** 2)
    freq = rng.uniform(0.03, 0.2) * sampling_frequency
    burst = envelope * np.sin(2.0 * np.pi * freq * (t - center_t))
    max_abs = np.max(np.abs(burst))
    if max_abs == 0:
        return burst
    return burst / max_abs
