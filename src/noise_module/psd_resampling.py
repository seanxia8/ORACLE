# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the modular noise simulator written for the ORACLE study.
# If you use this module in published work, please cite it: see CITATION.cff
# at the repository root.
"""Utilities for adapting empirical one-sided PSDs to new sampling rates.

The functions in this module operate on PSD *densities* with units such as
ADC^2 / Hz, stored as a two-row ``.npy`` array: ``[frequency_hz, psd]``.
That is the input convention expected by ``NoiseGenerator`` for custom PSDs.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import resample_poly, welch

from .utils import resolve_rng


def validate_psd_density(
    frequencies: np.ndarray, psd: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return validated one-dimensional frequency and PSD-density arrays."""
    f = np.asarray(frequencies, dtype=float)
    p = np.asarray(psd, dtype=float)
    if f.ndim != 1 or p.ndim != 1:
        raise ValueError("frequencies and psd must be one-dimensional arrays.")
    if f.shape != p.shape:
        raise ValueError("frequencies and psd must have the same shape.")
    if f.size < 2:
        raise ValueError("PSD must contain at least two frequency bins.")
    if np.any(~np.isfinite(f)) or np.any(~np.isfinite(p)):
        raise ValueError("PSD frequency and value arrays must be finite.")
    if np.any(f < 0.0):
        raise ValueError("one-sided PSD frequencies must be non-negative.")
    if np.any(np.diff(f) <= 0.0):
        raise ValueError("PSD frequencies must be strictly increasing.")
    if np.any(p < 0.0):
        raise ValueError("PSD values must be non-negative.")
    return f, p


def load_psd_density(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a two-row ``[frequency_hz, psd_density]`` array from ``path``."""
    data = np.load(Path(path))
    if data.ndim != 2 or data.shape[0] != 2:
        raise ValueError("PSD file must have shape (2, n_bins).")
    return validate_psd_density(data[0], data[1])


def save_psd_density(
    path: str | Path, frequencies: np.ndarray, psd: np.ndarray
) -> None:
    """Save a validated two-row PSD-density file."""
    f, p = validate_psd_density(frequencies, psd)
    np.save(Path(path), np.vstack([f, p]))


def target_rfft_grid(
    sampling_frequency: float, trace_samples: int
) -> np.ndarray:
    """Return the target one-sided rFFT frequency grid."""
    fs = float(sampling_frequency)
    n = int(trace_samples)
    if fs <= 0.0:
        raise ValueError("sampling_frequency must be positive.")
    if n <= 1:
        raise ValueError("trace_samples must be greater than one.")
    return np.fft.rfftfreq(n, d=1.0 / fs)


def cosine_lowpass_power(
    frequencies: np.ndarray,
    cutoff_hz: float,
    transition_width_hz: float,
) -> np.ndarray:
    """Return a smooth low-pass power response on ``frequencies``.

    The returned response is one below ``cutoff_hz``, zero above
    ``cutoff_hz + transition_width_hz``, and cosine-tapered in between.
    It is a power response, so it can be multiplied directly into a PSD.
    """
    f = np.asarray(frequencies, dtype=float)
    cutoff = float(cutoff_hz)
    width = float(transition_width_hz)
    if cutoff < 0.0:
        raise ValueError("cutoff_hz must be non-negative.")
    if width < 0.0:
        raise ValueError("transition_width_hz must be non-negative.")

    response = np.ones_like(f, dtype=float)
    if width == 0.0:
        response[f > cutoff] = 0.0
        return response

    stop = cutoff + width
    response[f >= stop] = 0.0
    transition = (f > cutoff) & (f < stop)
    phase = (f[transition] - cutoff) / width
    response[transition] = 0.5 * (1.0 + np.cos(np.pi * phase))
    return response


def inband_resample_psd_density(
    source_frequencies: np.ndarray,
    source_psd: np.ndarray,
    target_sampling_frequency: float,
    target_samples: int,
    *,
    anti_alias_cutoff_hz: float | None = None,
    anti_alias_transition_hz: float = 0.0,
    allow_extrapolation: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Interpolate an empirical PSD density onto a new rFFT grid.

    This is the appropriate deterministic approximation when high-frequency
    noise is removed by the new acquisition chain before sampling.
    """
    source_f, source_p = validate_psd_density(source_frequencies, source_psd)
    target_f = target_rfft_grid(target_sampling_frequency, target_samples)
    if target_f[-1] > source_f[-1] and not allow_extrapolation:
        raise ValueError(
            "target Nyquist exceeds source PSD support; pass "
            "allow_extrapolation=True only if edge extrapolation is intended."
        )

    psd_for_interp = source_p
    if anti_alias_cutoff_hz is not None:
        psd_for_interp = source_p * cosine_lowpass_power(
            source_f,
            anti_alias_cutoff_hz,
            anti_alias_transition_hz,
        )

    target_p = np.interp(
        target_f,
        source_f,
        psd_for_interp,
        left=psd_for_interp[0],
        right=psd_for_interp[-1] if allow_extrapolation else 0.0,
    )
    metadata = {
        "method": "inband_interpolation",
        "target_sampling_frequency": float(target_sampling_frequency),
        "target_samples": int(target_samples),
        "target_nyquist_hz": float(target_f[-1]),
        "source_max_frequency_hz": float(source_f[-1]),
        "anti_alias_cutoff_hz": (
            None if anti_alias_cutoff_hz is None else float(anti_alias_cutoff_hz)
        ),
        "anti_alias_transition_hz": float(anti_alias_transition_hz),
        "allow_extrapolation": bool(allow_extrapolation),
    }
    return target_f, target_p, metadata


def alias_fold_psd_density(
    source_frequencies: np.ndarray,
    source_psd: np.ndarray,
    target_sampling_frequency: float,
    target_samples: int,
    *,
    anti_alias_cutoff_hz: float | None = None,
    anti_alias_transition_hz: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fold source PSD density into the target Nyquist band.

    Use this as an explicit "no ideal anti-alias filter" approximation. It is
    only as complete as the source PSD bandwidth; any noise above the source
    Nyquist remains unknown.
    """
    source_f, source_p = validate_psd_density(source_frequencies, source_psd)
    target_f = target_rfft_grid(target_sampling_frequency, target_samples)
    fs = float(target_sampling_frequency)
    max_order = int(np.ceil(source_f[-1] / fs)) + 1
    interp_source = lambda x: np.interp(x, source_f, source_p, left=0.0, right=0.0)

    folded = np.zeros_like(target_f, dtype=float)
    orders = range(-max_order, max_order + 1)
    for order in orders:
        aliases = np.abs(target_f + order * fs)
        contribution = interp_source(aliases)
        if anti_alias_cutoff_hz is not None:
            contribution *= cosine_lowpass_power(
                aliases,
                anti_alias_cutoff_hz,
                anti_alias_transition_hz,
            )
        folded += contribution

    # DC and Nyquist are one-sided boundary bins. Avoid double-counting symmetric
    # aliases there by recomputing from unique absolute source frequencies.
    boundary_indices = [0]
    if target_samples % 2 == 0:
        boundary_indices.append(target_f.size - 1)
    for idx in boundary_indices:
        aliases = np.array(
            [abs(float(target_f[idx] + order * fs)) for order in orders],
            dtype=float,
        )
        aliases = np.unique(np.round(aliases, decimals=6))
        contribution = interp_source(aliases)
        if anti_alias_cutoff_hz is not None:
            contribution *= cosine_lowpass_power(
                aliases,
                anti_alias_cutoff_hz,
                anti_alias_transition_hz,
            )
        folded[idx] = float(np.sum(contribution))

    metadata = {
        "method": "alias_fold",
        "target_sampling_frequency": fs,
        "target_samples": int(target_samples),
        "target_nyquist_hz": float(target_f[-1]),
        "source_max_frequency_hz": float(source_f[-1]),
        "alias_orders": [-max_order, max_order],
        "anti_alias_cutoff_hz": (
            None if anti_alias_cutoff_hz is None else float(anti_alias_cutoff_hz)
        ),
        "anti_alias_transition_hz": float(anti_alias_transition_hz),
    }
    return target_f, folded, metadata


def _density_to_rfft_power(
    psd_density: np.ndarray, sampling_frequency: float, samples: int
) -> np.ndarray:
    """Convert one-sided PSD density to expected rFFT coefficient power."""
    power = np.asarray(psd_density, dtype=float) * float(sampling_frequency) * int(samples)
    if samples > 2:
        upper = samples // 2 + 1 - (samples + 1) % 2
        power[1:upper] *= 0.5
    return np.clip(power, a_min=0.0, a_max=None)


def _sample_gaussian_from_density(
    source_frequencies: np.ndarray,
    source_psd: np.ndarray,
    sampling_frequency: float,
    samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample stationary Gaussian noise from a PSD-density array."""
    grid = target_rfft_grid(sampling_frequency, samples)
    density = np.interp(grid, source_frequencies, source_psd, left=source_psd[0], right=0.0)
    power = _density_to_rfft_power(density, sampling_frequency, samples)
    amplitude = np.sqrt(power)

    spectrum = np.zeros_like(amplitude, dtype=complex)
    if spectrum.size:
        spectrum[0] = amplitude[0] * rng.standard_normal()
    if spectrum.size > 2:
        re = rng.standard_normal(spectrum.size - 2)
        im = rng.standard_normal(spectrum.size - 2)
        spectrum[1:-1] = amplitude[1:-1] * (re + 1j * im) / np.sqrt(2.0)
    if spectrum.size > 1:
        if samples % 2 == 0:
            spectrum[-1] = amplitude[-1] * rng.standard_normal()
        else:
            re, im = rng.standard_normal(2)
            spectrum[-1] = amplitude[-1] * (re + 1j * im) / np.sqrt(2.0)
    return np.fft.irfft(spectrum, n=samples)


def synthetic_resample_psd_density(
    source_frequencies: np.ndarray,
    source_psd: np.ndarray,
    source_sampling_frequency: float,
    target_sampling_frequency: float,
    target_samples: int,
    *,
    n_traces: int = 64,
    seed: int | None = None,
    rng: Any = None,
    source_edge_samples: int | None = None,
    window: str = "hann",
    average: str = "mean",
    detrend: str | bool = "constant",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Generate old-rate noise, polyphase-resample it, and estimate a new PSD.

    This is a Monte Carlo substitute for measured 1 MHz baselines. It is more
    operationally faithful than direct interpolation because the same resampling
    filter is applied to synthetic time-domain noise, but it still inherits the
    old PSD and the assumed anti-alias behavior of ``scipy.signal.resample_poly``.
    """
    source_f, source_p = validate_psd_density(source_frequencies, source_psd)
    if source_sampling_frequency <= 0.0 or target_sampling_frequency <= 0.0:
        raise ValueError("sampling frequencies must be positive.")
    if int(target_samples) <= 1:
        raise ValueError("target_samples must be greater than one.")
    if int(n_traces) <= 0:
        raise ValueError("n_traces must be positive.")
    if average not in {"mean", "median"}:
        raise ValueError("average must be 'mean' or 'median'.")

    generator = resolve_rng(rng=rng, seed=seed)
    ratio = Fraction(
        float(target_sampling_frequency) / float(source_sampling_frequency)
    ).limit_denominator(100000)
    up, down = int(ratio.numerator), int(ratio.denominator)
    needed_source = int(np.ceil(int(target_samples) * down / up))
    edge = int(source_edge_samples) if source_edge_samples is not None else max(256, 8 * down)
    source_samples = needed_source + 2 * edge

    traces = np.empty((int(n_traces), int(target_samples)), dtype=float)
    for idx in range(int(n_traces)):
        source_trace = _sample_gaussian_from_density(
            source_f,
            source_p,
            source_sampling_frequency,
            source_samples,
            generator,
        )
        target_trace = resample_poly(source_trace, up, down)
        if target_trace.size < target_samples:
            raise RuntimeError("resampled trace is shorter than target_samples.")
        start = (target_trace.size - int(target_samples)) // 2
        traces[idx] = target_trace[start : start + int(target_samples)]

    frequencies, psd = welch(
        traces.reshape(-1),
        fs=float(target_sampling_frequency),
        window=window,
        nperseg=int(target_samples),
        noverlap=0,
        nfft=int(target_samples),
        detrend=detrend,
        return_onesided=True,
        scaling="density",
        average=average,
    )
    metadata = {
        "method": "synthetic_resample",
        "source_sampling_frequency": float(source_sampling_frequency),
        "target_sampling_frequency": float(target_sampling_frequency),
        "target_samples": int(target_samples),
        "source_samples": int(source_samples),
        "source_edge_samples": int(edge),
        "n_traces": int(n_traces),
        "resample_up": up,
        "resample_down": down,
        "welch_window": window,
        "welch_average": average,
        "welch_detrend": detrend,
        "seed": seed,
    }
    return np.asarray(frequencies, dtype=float), np.asarray(psd, dtype=float), metadata


def make_target_psd_density(
    source_frequencies: np.ndarray,
    source_psd: np.ndarray,
    target_sampling_frequency: float,
    target_samples: int,
    *,
    method: str = "inband",
    source_sampling_frequency: float | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Dispatch helper for target-rate PSD construction."""
    if method == "inband":
        return inband_resample_psd_density(
            source_frequencies,
            source_psd,
            target_sampling_frequency,
            target_samples,
            **kwargs,
        )
    if method == "alias_fold":
        return alias_fold_psd_density(
            source_frequencies,
            source_psd,
            target_sampling_frequency,
            target_samples,
            **kwargs,
        )
    if method == "synthetic_resample":
        if source_sampling_frequency is None:
            raise ValueError("source_sampling_frequency is required for synthetic_resample.")
        return synthetic_resample_psd_density(
            source_frequencies,
            source_psd,
            source_sampling_frequency,
            target_sampling_frequency,
            target_samples,
            **kwargs,
        )
    raise ValueError("method must be one of: inband, alias_fold, synthetic_resample.")
