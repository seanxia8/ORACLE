# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the modular noise simulator written for the ORACLE study.
# If you use this module in published work, please cite it: see CITATION.cff
# at the repository root.
"""Artifact injection on top of baseline noise traces."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from .config import CONFIG_SCHEMA_VERSION, ArtifactConfig
from .templates import generate_burst_template, generate_glitch_template
from .utils import resolve_rng, sample_range, spawn_rng


class ArtifactInjector:
    """Inject deterministic lines and transient artifacts into traces."""

    DEFAULT_CONFIG = {
        "sampling_frequency": 1.0,
        "enable_lines": False,
        "lines": [],
        "enable_glitches": False,
        "glitch_rate": 1.0,  # events / second
        "glitch_amp_range": [0.05, 0.2],
        "glitch_templates": ["impulse", "exp_decay", "damped_sine"],
        "glitch_duration_samples": [32, 256],
        "enable_bursts": False,
        "burst_rate": 0.2,  # events / second
        "burst_amp_range": [0.03, 0.1],
        "burst_duration_samples": [128, 512],
        "enable_sparse_impulses": False,
        "impulse_probability": 1e-4,
        "impulse_sigma": 0.1,
        "channel_amplitude_jitter": 0.05,
    }

    def __init__(
        self,
        config: dict[str, Any] | ArtifactConfig | None = None,
        rng: Any = None,
        seed: int | None = None,
        *,
        strict_config: bool = True,
    ):
        resolved = deepcopy(self.DEFAULT_CONFIG)
        if isinstance(config, ArtifactConfig):
            resolved.update(config.to_dict())
        elif config:
            resolved.update(config)
        self.config_model = ArtifactConfig.from_mapping(
            resolved, strict=strict_config
        )
        self.config = self.config_model.to_dict()
        self.seed = seed
        self.rng = resolve_rng(rng=rng, seed=seed)

    def apply(
        self,
        x: np.ndarray,
        return_metadata: bool = False,
        return_components: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Apply configured artifacts to a single-channel trace."""
        y = np.array(x, dtype=float, copy=True)
        metadata: dict[str, Any] = {
            "metadata_schema_version": CONFIG_SCHEMA_VERSION,
            "amplitude_definition": (
                "rms_energy_ratio is ||artifact||_2 / baseline_RMS; "
                "it is not colored-noise matched-filter SNR"
            ),
        }
        components: dict[str, np.ndarray] = {}
        masks: dict[str, np.ndarray] = {}

        if self.config.get("enable_lines", False):
            before = y.copy()
            y, line_meta = self.add_lines(y, return_metadata=True)
            metadata["lines"] = line_meta
            components["lines"] = y - before
            masks["lines"] = np.abs(y - before) > 0
        if self.config.get("enable_glitches", False):
            before = y.copy()
            y, glitch_meta = self.add_glitches(y, return_metadata=True)
            metadata["glitches"] = glitch_meta
            components["glitches"] = y - before
            masks["glitches"] = np.abs(y - before) > 0
        if self.config.get("enable_bursts", False):
            before = y.copy()
            y, burst_meta = self.add_bursts(y, return_metadata=True)
            metadata["bursts"] = burst_meta
            components["bursts"] = y - before
            masks["bursts"] = np.abs(y - before) > 0
        if self.config.get("enable_sparse_impulses", False):
            before = y.copy()
            y, impulse_meta = self.add_sparse_impulses(y, return_metadata=True)
            metadata["sparse_impulses"] = impulse_meta
            components["sparse_impulses"] = y - before
            masks["sparse_impulses"] = np.abs(y - before) > 0

        if return_components:
            metadata["artifact_components"] = components
            metadata["artifact_masks"] = masks
            metadata["artifact_only"] = y - x
            metadata["combined_mask"] = np.abs(y - x) > 0
        if return_metadata or return_components:
            metadata["output_std"] = float(np.std(y))
            return y, metadata
        return y

    def apply_multichannel(
        self,
        X: np.ndarray,
        return_metadata: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Apply artifacts to a multichannel array of shape (C, N)."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("Multichannel input must have shape (C, N).")

        C, _ = X.shape
        output_channels = []
        metadata_channels = []
        for idx in range(C):
            channel_rng = spawn_rng(self.rng)
            channel_injector = ArtifactInjector(self.config, rng=channel_rng)
            jitter_sigma = float(self.config.get("channel_amplitude_jitter", 0.05))
            # A log-normal gain is positive and has unit expectation. Generating
            # against the actual channel is essential for baseline/local-RMS and
            # SNR-relative amplitudes; generating against zeros collapses all
            # three units to the numerical RMS floor.
            channel_gain = float(
                np.exp(self.rng.normal(-0.5 * jitter_sigma**2, jitter_sigma))
            )
            processed, channel_meta = channel_injector.apply(
                X[idx], return_components=True
            )
            artifact = processed - X[idx]
            applied_artifact = channel_gain * artifact
            output_channels.append(X[idx] + applied_artifact)
            channel_meta["channel_amplitude_gain"] = channel_gain
            channel_meta["artifact_only"] = applied_artifact
            channel_meta["combined_mask"] = applied_artifact != 0
            channel_meta["output_std"] = float(np.std(X[idx] + applied_artifact))
            for name, component in channel_meta.get("artifact_components", {}).items():
                channel_meta["artifact_components"][name] = channel_gain * component
            for section_name, event_name in (
                ("lines", "lines"),
                ("glitches", "glitches"),
                ("bursts", "bursts"),
            ):
                for event in channel_meta.get(section_name, {}).get(event_name, []):
                    event["pre_channel_gain_applied_amp"] = event["applied_amp"]
                    event["applied_amp"] *= channel_gain
            sparse = channel_meta.get("sparse_impulses")
            if sparse is not None:
                sparse["pre_channel_gain_sigma"] = sparse["sigma"]
                sparse["sigma"] *= channel_gain
            metadata_channels.append(channel_meta)

        Y = np.vstack(output_channels)
        if return_metadata:
            return Y, {
                "metadata_schema_version": CONFIG_SCHEMA_VERSION,
                "channels": metadata_channels,
                "artifact_only": Y - X,
                "combined_mask": Y != X,
            }
        return Y

    def add_lines(
        self,
        x: np.ndarray,
        return_metadata: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Add deterministic sinusoidal spectral lines."""
        y = np.array(x, dtype=float, copy=True)
        fs = float(self.config.get("sampling_frequency", 1.0))
        t = np.arange(len(y), dtype=float) / max(fs, 1.0)
        added_lines = []

        for line_cfg in self.config.get("lines", []):
            base_freq = float(line_cfg["freq"])
            base_amp = line_cfg["amp"]
            phase_cfg = line_cfg.get("phase", "random")
            harmonics = line_cfg.get("harmonics", [1])
            for harmonic in harmonics:
                freq = base_freq * float(harmonic)
                if not np.isfinite(freq) or not 0.0 <= freq <= fs / 2.0:
                    raise ValueError(
                        f"Line frequency {freq} Hz must lie in [0, fs / 2]."
                    )
                amp = sample_range(self.rng, base_amp)
                phase = (
                    self.rng.uniform(0.0, 2.0 * np.pi)
                    if phase_cfg == "random"
                    else float(phase_cfg)
                )
                template = np.sin(2.0 * np.pi * freq * t + phase)
                applied_amp = self._resolve_amplitude(y, template, float(amp), 0, len(y))
                y += applied_amp * template
                added_lines.append({
                    "freq": freq, "requested_amp": float(amp),
                    "applied_amp": applied_amp, "phase": float(phase),
                    "harmonic": float(harmonic), "amplitude_unit": self.config["amplitude_unit"],
                })

        if return_metadata:
            return y, {"count": len(added_lines), "lines": added_lines}
        return y

    def add_glitches(
        self,
        x: np.ndarray,
        return_metadata: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Add localized transient glitch templates."""
        y = np.array(x, dtype=float, copy=True)
        fs = float(self.config.get("sampling_frequency", 1.0))
        glitch_rate = float(self.config.get("glitch_rate", 0.0))
        if glitch_rate < 0.0:
            raise ValueError("glitch_rate must be non-negative events per second.")
        count_contract = self._event_count_contract(len(y), glitch_rate)
        expected_count = count_contract["expected_total_count"]
        starts = self._sample_event_starts(len(y), glitch_rate)
        injected = []

        occupied = np.zeros(len(y), dtype=bool)
        for proposed_start in starts:
            duration = self._sample_duration("glitch")
            duration = max(duration, 4)
            if duration >= len(y):
                duration = max(len(y) - 1, 4)
            if duration <= 0 or duration >= len(y):
                continue
            start = self._place_event(int(proposed_start), duration, len(y))
            if start is not None and self.config["boundary_policy"] == "truncate":
                duration = min(duration, len(y) - start)
            if duration < 4:
                continue
            if start is not None and self.config["overlap_policy"] == "resample" and np.any(occupied[start:start+duration]):
                candidates = np.flatnonzero(
                    np.convolve((~occupied).astype(int), np.ones(duration, dtype=int), mode="valid")
                    == duration
                )
                if candidates.size:
                    start = int(self.rng.choice(candidates))
            if start is None or (
                self.config["overlap_policy"] in {"reject", "resample"}
                and np.any(occupied[start:start+duration])
            ):
                continue
            amp = float(sample_range(self.rng, self.config["glitch_amp_range"]))
            kind = str(self.rng.choice(self.config["glitch_templates"]))
            template = generate_glitch_template(kind, duration, fs, self.rng)
            applied_amp = self._resolve_amplitude(y, template, amp, start, duration)
            y[start : start + duration] += applied_amp * template
            occupied[start:start+duration] = True
            injected.append({"start": start, "time_seconds": start/fs, "duration": duration,
                "duration_seconds": duration/fs, "requested_amp": amp, "applied_amp": applied_amp,
                "kind": kind, "amplitude_unit": self.config["amplitude_unit"]})

        if return_metadata:
            return y, {
                "count": len(injected),
                "rate_hz": glitch_rate,
                "expected_count": expected_count,
                "event_process": self.config["event_process"],
                **count_contract,
                "overlap_policy": self.config["overlap_policy"],
                "boundary_policy": self.config["boundary_policy"],
                "glitches": injected,
            }
        return y

    def add_bursts(
        self,
        x: np.ndarray,
        return_metadata: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Add short bursty packets."""
        y = np.array(x, dtype=float, copy=True)
        fs = float(self.config.get("sampling_frequency", 1.0))
        burst_rate = float(self.config.get("burst_rate", 0.0))
        if burst_rate < 0.0:
            raise ValueError("burst_rate must be non-negative events per second.")
        count_contract = self._event_count_contract(len(y), burst_rate)
        expected_count = count_contract["expected_total_count"]
        starts = self._sample_event_starts(len(y), burst_rate)
        injected = []

        occupied = np.zeros(len(y), dtype=bool)
        for proposed_start in starts:
            duration = self._sample_duration("burst")
            duration = max(duration, 8)
            if duration >= len(y):
                duration = max(len(y) - 1, 8)
            if duration <= 0 or duration >= len(y):
                continue
            start = self._place_event(int(proposed_start), duration, len(y))
            if start is not None and self.config["boundary_policy"] == "truncate":
                duration = min(duration, len(y) - start)
            if duration < 8:
                continue
            if start is not None and self.config["overlap_policy"] == "resample" and np.any(occupied[start:start+duration]):
                candidates = np.flatnonzero(
                    np.convolve((~occupied).astype(int), np.ones(duration, dtype=int), mode="valid")
                    == duration
                )
                if candidates.size:
                    start = int(self.rng.choice(candidates))
            if start is None or (
                self.config["overlap_policy"] in {"reject", "resample"}
                and np.any(occupied[start:start+duration])
            ):
                continue
            amp = float(sample_range(self.rng, self.config["burst_amp_range"]))
            burst = generate_burst_template(duration, fs, self.rng)
            applied_amp = self._resolve_amplitude(y, burst, amp, start, duration)
            y[start : start + duration] += applied_amp * burst
            occupied[start:start+duration] = True
            injected.append({"start": start, "time_seconds": start/fs, "duration": duration,
                "duration_seconds": duration/fs, "requested_amp": amp,
                "applied_amp": applied_amp, "amplitude_unit": self.config["amplitude_unit"]})

        if return_metadata:
            return y, {
                "count": len(injected),
                "rate_hz": burst_rate,
                "expected_count": expected_count,
                "event_process": self.config["event_process"],
                **count_contract,
                "overlap_policy": self.config["overlap_policy"],
                "boundary_policy": self.config["boundary_policy"],
                "bursts": injected,
            }
        return y

    def add_sparse_impulses(
        self,
        x: np.ndarray,
        return_metadata: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Add sparse heavy-tail outliers."""
        y = np.array(x, dtype=float, copy=True)
        probability = float(self.config.get("impulse_probability", 0.0))
        sigma = float(self.config.get("impulse_sigma", 0.0))
        mask = self.rng.uniform(0.0, 1.0, size=len(y)) < probability
        impulses = self.rng.normal(0.0, sigma, size=int(np.sum(mask)))
        y[mask] += impulses

        if return_metadata:
            return y, {"count": int(np.sum(mask)), "sigma": sigma}
        return y

    def _sample_duration(self, kind: str) -> int:
        fs = float(self.config["sampling_frequency"])
        seconds = self.config.get(f"{kind}_duration_seconds")
        if seconds is not None:
            return max(1, int(round(sample_range(self.rng, seconds) * fs)))
        return max(1, int(round(sample_range(self.rng, self.config[f"{kind}_duration_samples"]))))

    def _sample_event_starts(self, N: int, rate_hz: float) -> np.ndarray:
        fs = float(self.config["sampling_frequency"])
        process = self.config.get("event_process", "homogeneous")
        if process == "homogeneous":
            return np.sort(self.rng.integers(0, N, size=self.rng.poisson(rate_hz*N/fs)))
        if process == "nonhomogeneous":
            profile = np.asarray(self.config.get("rate_profile") or [1.0], dtype=float)
            profile = np.interp(np.arange(N), np.linspace(0, N-1, len(profile)), profile)
            profile = np.clip(profile, 0, None)
            if np.mean(profile) > 0:
                profile /= np.mean(profile)
            # Piecewise-constant inhomogeneous Poisson process. Per-bin Poisson
            # counts preserve the requested rate and allow multiple arrivals
            # in a sample interval at high intensity.
            counts = self.rng.poisson(rate_hz * profile / fs)
            return np.repeat(np.arange(N, dtype=int), counts)
        # Optional stationary Hawkes cluster approximation: immigrants plus
        # geometric offspring with exponential delays.
        starts = list(self.rng.integers(0, N, size=self.rng.poisson(rate_hz*N/fs)))
        branching = float(self.config["hawkes_branching_ratio"])
        decay = float(self.config["hawkes_decay_seconds"])
        queue = list(starts)
        while queue:
            parent = queue.pop()
            for _ in range(self.rng.poisson(branching)):
                child = parent + int(round(self.rng.exponential(decay)*fs))
                if child < N:
                    starts.append(child)
                    queue.append(child)
        return np.sort(np.asarray(starts, dtype=int))

    def _event_count_contract(self, N: int, rate_hz: float) -> dict[str, float | str]:
        """Return the configured point-process count expectation."""
        fs = float(self.config["sampling_frequency"])
        immigrant_expected = rate_hz * N / fs
        process = self.config.get("event_process", "homogeneous")
        if process == "nonhomogeneous":
            profile = np.asarray(self.config.get("rate_profile") or [1.0], dtype=float)
            profile = np.clip(profile, 0.0, None)
            expected = immigrant_expected if np.mean(profile) > 0.0 else 0.0
            return {
                "expected_total_count": float(expected),
                "count_expectation": "integrated_piecewise_constant_intensity",
            }
        if process == "hawkes":
            branching = float(self.config["hawkes_branching_ratio"])
            return {
                "immigrant_expected_count": float(immigrant_expected),
                "expected_total_count": float(immigrant_expected / (1.0 - branching)),
                "count_expectation": (
                    "asymptotic branching-process expectation; finite-record "
                    "right-censoring reduces realized count"
                ),
            }
        return {
            "expected_total_count": float(immigrant_expected),
            "count_expectation": "homogeneous_poisson",
        }

    def _place_event(self, proposed: int, duration: int, N: int) -> int | None:
        policy = self.config.get("boundary_policy", "truncate")
        if policy == "center":
            proposed -= duration // 2
        if 0 <= proposed and proposed + duration <= N:
            return proposed
        if policy == "reject":
            return None
        if policy == "truncate":
            return min(max(proposed, 0), N - 1)
        return min(max(proposed, 0), max(N-duration, 0))

    def _resolve_amplitude(
        self, baseline: np.ndarray, template: np.ndarray, requested: float,
        start: int, duration: int,
    ) -> float:
        unit = self.config.get("amplitude_unit", "raw")
        if unit == "raw":
            return requested
        global_rms = max(float(np.std(baseline)), 1e-12)
        if unit == "baseline_rms":
            return requested * global_rms
        if unit == "local_rms":
            half = int(self.config.get("local_rms_window_samples", 256)) // 2
            local = baseline[max(0, start-half):min(len(baseline), start+duration+half)]
            return requested * max(float(np.std(local)), global_rms)
        return requested * global_rms / max(float(np.linalg.norm(template)), 1e-12)
