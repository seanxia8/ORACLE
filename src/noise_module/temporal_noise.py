# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the modular noise simulator written for the ORACLE study.
# If you use this module in published work, please cite it: see CITATION.cff
# at the repository root.
"""Temporal wrappers for non-stationary extensions of the base noise model."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.signal import butter, sosfiltfilt

from .config import CONFIG_SCHEMA_VERSION, TemporalNoiseConfig
from .NoiseGenerator import NoiseGenerator
from .utils import concatenate_with_crossfade, resolve_rng, sample_range, spawn_rng


class TemporalNoiseWrapper:
    """Apply piecewise stationarity, drift, and slow variance changes."""

    DEFAULT_CONFIG = {
        "mode": "none",
        "n_segments": 4,
        "segment_length": None,
        "crossfade_len": 128,
        "vary_noise_power": True,
        "noise_power_scale_range": [0.8, 1.2],
        "vary_psd_slope": False,
        "psd_slope_range": [-0.1, 0.1],
        "add_drift": False,
        "drift_type": "spline",
        "drift_sigma": 0.05,
        "drift_n_knots": 6,
        "variance_modulation": False,
        "variance_scale_range": [0.95, 1.05],
        "variance_n_knots": 6,
        "multichannel_shared_drift": True,
    }

    COLOR_EXPONENTS = {
        "brownian": -2.0,
        "pink": -1.0,
        "white": 0.0,
        "blue": 1.0,
        "violet": 2.0,
    }

    def __init__(
        self,
        config: dict[str, Any] | TemporalNoiseConfig | None = None,
        rng: Any = None,
        seed: int | None = None,
        *,
        strict_config: bool = True,
    ):
        resolved = deepcopy(self.DEFAULT_CONFIG)
        if isinstance(config, TemporalNoiseConfig):
            resolved.update(config.to_dict())
        elif config:
            resolved.update(config)
        self.config_model = TemporalNoiseConfig.from_mapping(
            resolved, strict=strict_config
        )
        self.config = self.config_model.to_dict()
        self.seed = seed
        self.rng = resolve_rng(rng=rng, seed=seed)

    def apply(
        self,
        x: np.ndarray,
        base_generator: NoiseGenerator | None = None,
        return_metadata: bool = False,
        return_components: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Apply temporal effects to a single-channel trace."""
        x = np.asarray(x, dtype=float)
        if x.ndim != 1:
            raise ValueError("Single-channel input must be one-dimensional.")

        metadata: dict[str, Any] = {
            "metadata_schema_version": CONFIG_SCHEMA_VERSION,
            "mode": self.config["mode"],
        }
        y = np.array(x, copy=True)

        if base_generator is not None:
            import warnings
            warnings.warn(
                "apply(..., base_generator=...) no longer regenerates and discards x; "
                "use generate_piecewise() explicitly.",
                DeprecationWarning,
                stacklevel=2,
            )
        envelope = np.ones(len(y))
        drift = np.zeros(len(y))
        if self.config["mode"] == "piecewise":
            y, piecewise_envelope, piecewise_meta = self._apply_piecewise_scaling(
                y, return_metadata=True, return_envelope=True
            )
            envelope *= piecewise_envelope
            metadata["piecewise"] = piecewise_meta

        if self.config.get("variance_modulation", False):
            y, variance_envelope = self.modulate_variance(y, return_envelope=True)
            envelope *= variance_envelope
            metadata["variance_envelope_range"] = [
                float(np.min(variance_envelope)),
                float(np.max(variance_envelope)),
            ]

        if self.config.get("add_drift", False):
            y, drift = self.add_drift(y, return_drift=True)
            metadata["drift_std"] = float(np.std(drift))

        metadata["operations"] = {
            "regenerated": False,
            "multiplied": bool(self.config.get("variance_modulation") or self.config["mode"] == "piecewise"),
            "added": bool(self.config.get("add_drift")),
        }
        if return_components:
            metadata["components"] = {"multiplicative_envelope": envelope, "additive_drift": drift}
        if return_metadata or return_components:
            metadata["output_std"] = float(np.std(y))
            return y, metadata
        return y

    def modulate_variance(
        self, x: np.ndarray, *, return_envelope: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Multiply an existing trace by a positive amplitude envelope."""
        x = np.asarray(x, dtype=float)
        envelope = self._build_variance_envelope(x.shape[-1])
        output = x * envelope
        return (output, envelope) if return_envelope else output

    def add_drift(
        self, x: np.ndarray, *, return_drift: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Add a generated drift to an existing trace."""
        x = np.asarray(x, dtype=float)
        drift = self.generate_drift(x.shape[-1])
        output = x + drift
        return (output, drift) if return_drift else output

    def apply_multichannel(
        self,
        X: np.ndarray,
        base_generator: Any = None,
        return_metadata: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Apply temporal effects to a multichannel array of shape (C, N)."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("Multichannel input must have shape (C, N).")

        C, N = X.shape
        Y = np.array(X, copy=True)
        metadata: dict[str, Any] = {
            "metadata_schema_version": CONFIG_SCHEMA_VERSION,
            "mode": self.config["mode"],
            "n_channels": C,
        }

        if base_generator is not None:
            import warnings
            warnings.warn(
                "apply_multichannel(..., base_generator=...) no longer regenerates "
                "and discards X; call generation explicitly.",
                DeprecationWarning,
                stacklevel=2,
            )
        if self.config["mode"] == "piecewise":
            scaled_channels = []
            channel_meta = []
            for channel in Y:
                channel_rng = spawn_rng(self.rng)
                channel_wrapper = TemporalNoiseWrapper(self.config, rng=channel_rng)
                scaled, channel_piecewise = channel_wrapper._apply_piecewise_scaling(
                    channel,
                    return_metadata=True,
                )
                scaled_channels.append(scaled)
                channel_meta.append(channel_piecewise)
            Y = np.vstack(scaled_channels)
            piecewise_meta = {"channels": channel_meta}
            metadata["piecewise"] = piecewise_meta

        if self.config.get("variance_modulation", False):
            frac = float(self.config.get("multichannel_shared_fraction", 1.0))
            shared_envelope = self._build_variance_envelope(N)
            private = np.vstack([self._build_variance_envelope(N) for _ in range(C)])
            envelopes = np.exp(
                np.sqrt(frac) * np.log(shared_envelope)[None, :]
                + np.sqrt(1.0 - frac) * np.log(private)
            )
            Y = Y * envelopes
            metadata["variance_envelope_range"] = [
                float(np.min(envelopes)),
                float(np.max(envelopes)),
            ]

        if self.config.get("add_drift", False):
            frac = (
                float(self.config.get("multichannel_shared_fraction", 1.0))
                if self.config.get("multichannel_shared_drift", True)
                else 0.0
            )
            shared = self.generate_drift(N)
            private = np.vstack([self.generate_drift(N) for _ in range(C)])
            drift = np.sqrt(frac) * shared[None, :] + np.sqrt(1-frac) * private
            Y = Y + drift
            metadata["drift_std"] = float(np.std(drift))

        if return_metadata:
            metadata["output_std"] = float(np.std(Y))
            return Y, metadata
        return Y

    def generate_evolutionary(
        self,
        N: int,
        base_generator: NoiseGenerator,
        *,
        window_samples: int = 512,
        hop_samples: int | None = None,
        parameter_schedule: list[dict[str, Any]] | None = None,
        return_metadata: bool = False,
    ):
        """Generate smoothly evolving noise by square-root Hann overlap-add."""
        hop = int(hop_samples or window_samples // 2)
        if window_samples < 4 or hop <= 0 or hop >= window_samples:
            raise ValueError("Invalid evolutionary window/hop.")
        if N <= 0:
            raise ValueError("N must be positive.")
        # Include pre-roll windows so the leading edge is covered by the
        # interior of a window. Starting at zero would multiply sample zero by
        # the zero-valued edge of the Hann window.
        starts = list(range(-window_samples + hop, N, hop))
        output = np.zeros(N)
        weight = np.zeros_like(output)
        window = np.sqrt(np.hanning(window_samples + 1)[:-1])
        local_meta = []
        boundary_meta = []
        for index, start in enumerate(starts):
            cfg = self._sample_local_config(self._extract_base_config(base_generator))
            if parameter_schedule:
                schedule_index = min(
                    max(start // hop, 0), len(parameter_schedule) - 1
                )
                cfg.update(parameter_schedule[schedule_index])
            segment = NoiseGenerator(cfg, rng=spawn_rng(self.rng)).generate_noise(window_samples)
            destination_start = max(start, 0)
            destination_end = min(start + window_samples, N)
            source_start = destination_start - start
            source_end = source_start + destination_end - destination_start
            if destination_end > destination_start:
                local_window = window[source_start:source_end]
                output[destination_start:destination_end] += (
                    segment[source_start:source_end] * local_window
                )
                weight[destination_start:destination_end] += local_window**2
            window_meta = {
                "start": start,
                "end": start + window_samples,
                "config": cfg,
                "is_boundary_window": start < 0 or start + window_samples > N,
            }
            if start < 0:
                boundary_meta.append(window_meta)
            else:
                local_meta.append(window_meta)
        if np.any(weight <= 1e-12):
            raise RuntimeError("Evolutionary overlap-add left uncovered output samples.")
        result = output / np.sqrt(weight)
        meta = {
            "metadata_schema_version": CONFIG_SCHEMA_VERSION,
            "method": "evolutionary_overlap_add",
            "window": "sqrt_hann",
            "window_samples": window_samples,
            "hop_samples": hop,
            "edge_policy": "pre_roll_and_post_roll",
            "minimum_overlap_weight": float(np.min(weight)),
            "maximum_overlap_weight": float(np.max(weight)),
            "constant_overlap_add_error": (
                float(np.max(np.abs(weight - 1.0)))
                if hop == window_samples // 2
                else None
            ),
            "windows": local_meta,
            "pre_roll_windows": boundary_meta,
        }
        return (result, meta) if return_metadata else result

    def generate_piecewise(
        self,
        N: int,
        base_generator: NoiseGenerator,
        return_metadata: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Generate a piecewise-stationary single-channel trace from scratch."""
        if self.config.get("boundary_policy") == "continuous":
            segments = self._build_segments(N)
            schedule = [
                self._sample_local_config(self._extract_base_config(base_generator))
                for _ in segments
            ]
            window = max(4, 2 * max(end-start for start, end in segments))
            signal, evo = self.generate_evolutionary(
                N,
                base_generator,
                window_samples=window,
                hop_samples=max(1, window//2),
                parameter_schedule=schedule,
                return_metadata=True,
            )
            metadata = {
                "segments": [
                    {"start": start, "end": end, **schedule[i]}
                    for i, (start, end) in enumerate(segments)
                ],
                "crossfade_len": window//2,
                "boundary_policy": "continuous",
                "evolutionary": evo,
            }
            return (signal, metadata) if return_metadata else signal
        signal, metadata = self._generate_piecewise_array(
            N,
            base_generator=base_generator,
            return_metadata=True,
        )
        if signal.ndim != 1:
            raise RuntimeError("Expected single-channel output from piecewise generation.")
        if return_metadata:
            return signal, metadata
        return signal

    def generate_drift(self, N: int) -> np.ndarray:
        """Generate a smooth additive low-frequency drift component."""
        if N <= 0:
            raise ValueError("N must be positive.")
        drift_type = self.config.get("drift_type", "spline")
        sigma = float(self.config.get("drift_rms") if self.config.get("drift_rms") is not None else self.config.get("drift_sigma", 0.05))
        if sigma == 0:
            return np.zeros(N, dtype=float)

        if drift_type == "random_walk":
            steps = self.rng.normal(0.0, sigma / max(np.sqrt(N), 1.0), size=N)
            drift = np.cumsum(steps)
            drift = drift - np.mean(drift)
            return self._match_rms(drift, sigma)
        if drift_type == "deterministic":
            values = np.asarray(
                self.config.get("deterministic_drift_values") or [-sigma, sigma],
                dtype=float,
            )
            drift = np.interp(
                np.arange(N), np.linspace(0, N-1, len(values)), values
            )
            return drift - np.mean(drift)
        if drift_type == "lowpass":
            fs = float(self.config.get("sampling_frequency", 1.0))
            cutoff = self.config.get("drift_cutoff_hz")
            if cutoff is None:
                timescale = float(self.config.get("drift_timescale_seconds") or max(N/fs/10, 1/fs))
                cutoff = min(0.5 / timescale, 0.45 * fs)
            sos = butter(4, float(cutoff), btype="lowpass", fs=fs, output="sos")
            drift = sosfiltfilt(sos, self.rng.standard_normal(N))
            return self._match_rms(drift - np.mean(drift), sigma)

        n_knots = max(int(self.config.get("drift_n_knots", 6)), 2)
        knot_x = np.linspace(0.0, N - 1, n_knots)
        knot_y = self.rng.normal(0.0, sigma, size=n_knots)
        if n_knots >= 4:
            interpolator = CubicSpline(knot_x, knot_y, bc_type="natural")
            drift = interpolator(np.arange(N, dtype=float))
        else:
            drift = np.interp(np.arange(N, dtype=float), knot_x, knot_y)
        drift = drift - np.mean(drift)
        return self._match_rms(drift, sigma)

    @staticmethod
    def _match_rms(x: np.ndarray, rms: float) -> np.ndarray:
        std = float(np.std(x))
        return np.zeros_like(x) if std == 0 or rms == 0 else x * (rms / std)

    def _generate_piecewise_array(
        self,
        N: int,
        base_generator: Any,
        return_metadata: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        segments = self._build_segments(N)
        policy = self.config.get("boundary_policy", "overlap_add")
        crossfade_len = 0 if policy == "hard" else max(int(self.config.get("crossfade_len", 0)), 0)
        outputs = []
        metadata_segments = []

        for idx, (start, end) in enumerate(segments):
            seg_len = end - start
            effective_len = seg_len + (crossfade_len if idx > 0 else 0)
            local_cfg = self._sample_local_config(self._extract_base_config(base_generator))
            local_output = self._generate_local_segment(
                base_generator=base_generator,
                local_config=local_cfg,
                seg_len=effective_len,
            )
            outputs.append(local_output)
            metadata_segments.append(
                {
                    "start": start,
                    "end": end,
                    "noise_power": float(local_cfg["noise_power"]),
                    "noise_type": local_cfg["noise_type"],
                    "psd_exponent": local_cfg.get("psd_exponent"),
                }
            )

        combined = concatenate_with_crossfade(outputs, crossfade_len)
        if combined.shape[-1] != N:
            combined = combined[..., :N]

        if return_metadata:
            boundary_mask = np.zeros(N, dtype=bool)
            for _, boundary in segments[:-1]:
                boundary_mask[max(0, boundary-crossfade_len):min(N, boundary+crossfade_len)] = True
            interior_mask = ~boundary_mask
            boundary_power = float(np.mean(np.asarray(combined)[..., boundary_mask]**2)) if np.any(boundary_mask) else None
            interior_power = float(np.mean(np.asarray(combined)[..., interior_mask]**2)) if np.any(interior_mask) else None
            spectral_distance = None
            if np.sum(boundary_mask) >= 8 and np.sum(interior_mask) >= np.sum(boundary_mask):
                boundary_values = np.asarray(combined)[..., boundary_mask].reshape(-1)
                interior_values = np.asarray(combined)[..., interior_mask].reshape(-1)[:len(boundary_values)]
                bp = np.abs(np.fft.rfft(boundary_values))**2
                ip = np.abs(np.fft.rfft(interior_values))**2
                bp /= max(float(np.sum(bp)), 1e-15)
                ip /= max(float(np.sum(ip)), 1e-15)
                spectral_distance = float(0.5*np.sum(np.abs(bp-ip)))
            metadata = {
                "segments": metadata_segments,
                "crossfade_len": crossfade_len,
                "boundary_policy": policy,
                "boundary_affected_samples": crossfade_len * max(len(segments) - 1, 0),
                "boundary_fraction": crossfade_len * max(len(segments)-1, 0) / N,
                "boundary_power": boundary_power,
                "interior_power": interior_power,
                "boundary_to_interior_power_ratio": (
                    boundary_power / interior_power
                    if boundary_power is not None and interior_power not in {None, 0.0}
                    else None
                ),
                "boundary_spectral_total_variation": spectral_distance,
            }
            return combined, metadata
        return combined

    def _apply_piecewise_scaling(
        self,
        x: np.ndarray,
        return_metadata: bool = False,
        return_envelope: bool = False,
    ) -> (
        np.ndarray
        | tuple[np.ndarray, dict[str, Any]]
        | tuple[np.ndarray, np.ndarray, dict[str, Any]]
    ):
        segments = self._build_segments(len(x))
        policy = self.config.get("boundary_policy", "overlap_add")
        crossfade_len = 0 if policy == "hard" else max(int(self.config.get("crossfade_len", 0)), 0)
        outputs = []
        envelope_segments = []
        metadata_segments = []
        base_power = float(np.var(x)) if np.var(x) > 0 else 1.0

        for idx, (start, end) in enumerate(segments):
            scale = 1.0
            if self.config.get("vary_noise_power", True):
                power_scale = sample_range(
                    self.rng,
                    self.config.get("noise_power_scale_range", [0.8, 1.2]),
                )
                scale = float(np.sqrt(max(power_scale, 0.0)))
            segment_start = max(0, start - crossfade_len) if idx > 0 else start
            segment = np.array(x[segment_start:end], copy=True) * scale
            outputs.append(segment)
            envelope_segments.append(np.full(end - segment_start, scale, dtype=float))
            metadata_segments.append(
                {
                    "start": start,
                    "end": end,
                    "amplitude_scale": scale,
                    "approx_noise_power": base_power * scale**2,
                }
            )

        combined = concatenate_with_crossfade(outputs, crossfade_len)
        combined = combined[: len(x)]
        envelope = concatenate_with_crossfade(envelope_segments, crossfade_len)[: len(x)]
        if return_metadata:
            metadata = {"segments": metadata_segments, "boundary_policy": policy}
            if return_envelope:
                return combined, envelope, metadata
            return combined, metadata
        if return_envelope:
            return combined, envelope, {}
        return combined

    def _build_segments(self, N: int) -> list[tuple[int, int]]:
        if N <= 0:
            raise ValueError("N must be positive.")
        segment_length = self.config.get("segment_length")
        if segment_length is None:
            n_segments = max(int(self.config.get("n_segments", 1)), 1)
            segment_edges = np.linspace(0, N, n_segments + 1, dtype=int)
            return [
                (int(segment_edges[i]), int(segment_edges[i + 1]))
                for i in range(n_segments)
                if int(segment_edges[i + 1]) > int(segment_edges[i])
            ]

        segment_length = max(int(segment_length), 1)
        starts = np.arange(0, N, segment_length, dtype=int)
        ends = np.minimum(starts + segment_length, N)
        return [(int(start), int(end)) for start, end in zip(starts, ends) if end > start]

    def _sample_local_config(self, base_config: dict[str, Any]) -> dict[str, Any]:
        local_config = deepcopy(base_config)
        if self.config.get("vary_noise_power", True):
            scale = sample_range(
                self.rng,
                self.config.get("noise_power_scale_range", [0.8, 1.2]),
            )
            local_config["noise_power"] = float(base_config["noise_power"]) * float(scale)
        if self.config.get("vary_psd_slope", False):
            noise_type = str(base_config.get("noise_type", "")).lower()
            if noise_type not in self.COLOR_EXPONENTS and base_config.get("psd_exponent") is None:
                raise ValueError(
                    "vary_psd_slope requires an analytic noise_type or a base "
                    "psd_exponent; arbitrary custom PSD slopes are not well-defined."
                )
            configured_exponent = base_config.get("psd_exponent")
            base_exponent = float(
                self.COLOR_EXPONENTS[noise_type]
                if configured_exponent is None
                else configured_exponent
            )
            delta = sample_range(
                self.rng,
                self.config.get("psd_slope_range", [-0.1, 0.1]),
            )
            local_config["psd_exponent"] = base_exponent + float(delta)
        for name, bounds in self.config.get("local_parameter_ranges", {}).items():
            local_config[name] = sample_range(self.rng, bounds)
        return local_config

    @staticmethod
    def _extract_base_config(base_generator: Any) -> dict[str, Any]:
        if hasattr(base_generator, "base_config"):
            return deepcopy(base_generator.base_config)
        if hasattr(base_generator, "config"):
            return deepcopy(base_generator.config)
        raise ValueError("base_generator does not expose a usable configuration.")

    def _generate_local_segment(
        self,
        base_generator: Any,
        local_config: dict[str, Any],
        seg_len: int,
    ) -> np.ndarray:
        local_rng = spawn_rng(self.rng)
        if hasattr(base_generator, "generate_noise"):
            local_generator = base_generator.__class__(local_config, rng=local_rng)
            return np.asarray(local_generator.generate_noise(seg_len), dtype=float)

        if hasattr(base_generator, "generate"):
            local_generator = base_generator.__class__(
                local_config,
                config=getattr(base_generator, "config", None),
                rng=local_rng,
            )
            return np.asarray(local_generator.generate(seg_len), dtype=float)

        raise ValueError("Unsupported base_generator type for piecewise generation.")

    def _build_variance_envelope(self, N: int) -> np.ndarray:
        n_knots = max(int(self.config.get("variance_n_knots", 6)), 2)
        knot_x = np.linspace(0.0, N - 1, n_knots)
        variance_ratio = sample_range(
            self.rng,
            self.config.get("variance_scale_range", [0.95, 1.05]),
            size=n_knots,
        )
        knot_y = np.sqrt(variance_ratio)
        log_y = np.log(np.asarray(knot_y))
        method = self.config.get("envelope_method", "log_pchip")
        if method == "log_pchip":
            envelope = np.exp(PchipInterpolator(knot_x, log_y)(np.arange(N)))
        elif method == "log_linear":
            envelope = np.exp(np.interp(np.arange(N), knot_x, log_y))
        else:
            fs = float(self.config.get("sampling_frequency", 1.0))
            cutoff = float(self.config.get("envelope_lowpass_hz") or 0.05 * fs)
            sos = butter(3, cutoff, btype="lowpass", fs=fs, output="sos")
            raw = sosfiltfilt(sos, self.rng.standard_normal(N))
            raw /= max(np.std(raw), 1e-12)
            lo, hi = self.config.get("variance_scale_range", [0.95, 1.05])
            envelope = np.exp(raw * 0.25 * np.log(hi/lo) + 0.5*np.log(lo*hi))
        lo, hi = self.config.get("variance_scale_range", [0.95, 1.05])
        return np.clip(envelope, np.sqrt(lo), np.sqrt(hi))
