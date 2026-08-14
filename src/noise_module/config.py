"""Typed, versioned configuration models for the noise simulation modules."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Mapping, TypeVar

import numpy as np


CONFIG_SCHEMA_VERSION = 3
T = TypeVar("T", bound="ConfigModel")


def _finite(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def _ordered_range(name: str, value: Any) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly [low, high].")
    low, high = _finite(f"{name}[0]", value[0]), _finite(f"{name}[1]", value[1])
    if low > high:
        raise ValueError(f"{name} must be ordered low <= high.")
    return low, high


def _integer(name: str, value: Any, *, minimum: int | None = None) -> int:
    """Validate an integer without silently truncating fractional values."""
    numeric = _finite(name, value)
    if not numeric.is_integer():
        raise ValueError(f"{name} must be an integer.")
    result = int(numeric)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return result


@dataclass
class ConfigModel:
    """Base class providing strict dictionary conversion and serialization."""

    schema_version: int = CONFIG_SCHEMA_VERSION

    @classmethod
    def from_mapping(cls: type[T], value: Mapping[str, Any] | T, *, strict: bool = True) -> T:
        if isinstance(value, cls):
            return deepcopy(value)
        data = migrate_config(dict(value), kind=cls.__name__)
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(data) - allowed)
        if strict and unknown:
            raise ValueError(f"Unknown {cls.__name__} field(s): {', '.join(unknown)}.")
        return cls(**{key: val for key, val in data.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def migrate_config(config: Mapping[str, Any], kind: str = "NoiseConfig") -> dict[str, Any]:
    """Migrate a legacy dictionary to the current configuration schema.

    The function is pure: the caller's dictionary is never modified. Unknown
    fields are retained so strict model conversion can report them.
    """
    data = deepcopy(dict(config))
    version = int(data.get("schema_version", 1))
    if version > CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"Configuration schema {version} is newer than supported schema "
            f"{CONFIG_SCHEMA_VERSION}."
        )
    if kind == "NoiseConfig" and version < 2:
        if data.get("custom_psd_scaling") == "multiply":
            data["custom_psd_scaling"] = "scale"
            data.setdefault("psd_scale", data.get("noise_power", 1.0))
    if kind == "ArtifactConfig" and version < 3:
        if data.get("overlap_policy") == "allow":
            data["overlap_policy"] = "superpose"
        if data.get("amplitude_unit") == "snr":
            data["amplitude_unit"] = "rms_energy_ratio"
    data["schema_version"] = CONFIG_SCHEMA_VERSION
    return data


@dataclass
class NoiseConfig(ConfigModel):
    """Configuration for stationary single-channel spectral synthesis."""

    noise_type: str = "white"
    noise_power: float = 1.0
    sampling_frequency: float = 1.0
    power_definition: str = "variance"
    deterministic_mean: float = 0.0
    psd_exponent: float | None = None
    low_frequency_cutoff: float | None = None
    high_frequency_cutoff: float | None = None
    custom_psd_scaling: str = "absolute"
    psd_scale: float = 1.0
    custom_out_of_band: str = "edge"
    custom_interpolation: str = "linear"
    components: list[dict[str, Any]] = field(default_factory=list)
    composite_psd_scaling: str = "normalize"

    def __post_init__(self) -> None:
        self.sampling_frequency = _finite("sampling_frequency", self.sampling_frequency)
        self.noise_power = _finite("noise_power", self.noise_power)
        self.deterministic_mean = _finite("deterministic_mean", self.deterministic_mean)
        self.psd_scale = _finite("psd_scale", self.psd_scale)
        if self.sampling_frequency <= 0.0:
            raise ValueError("sampling_frequency must be positive.")
        if self.noise_power < 0.0:
            raise ValueError("noise_power must be non-negative.")
        if self.psd_scale < 0.0:
            raise ValueError("psd_scale must be non-negative.")
        self.power_definition = str(self.power_definition).lower()
        if self.power_definition not in {"variance", "mean_square"}:
            raise ValueError("power_definition must be 'variance' or 'mean_square'.")
        self.custom_psd_scaling = str(self.custom_psd_scaling).lower()
        if self.custom_psd_scaling == "multiply":
            self.custom_psd_scaling = "scale"
            self.psd_scale = self.noise_power
        if self.custom_psd_scaling not in {"absolute", "normalize", "scale"}:
            raise ValueError(
                "custom_psd_scaling must be 'absolute', 'normalize', or 'scale'."
            )
        if self.custom_out_of_band not in {"error", "zero", "edge", "power_law"}:
            raise ValueError(
                "custom_out_of_band must be 'error', 'zero', 'edge', or 'power_law'."
            )
        if self.custom_interpolation not in {"linear", "loglog"}:
            raise ValueError("custom_interpolation must be 'linear' or 'loglog'.")
        if self.composite_psd_scaling not in {"absolute", "normalize"}:
            raise ValueError("composite_psd_scaling must be 'absolute' or 'normalize'.")
        if self.psd_exponent is not None:
            self.psd_exponent = _finite("psd_exponent", self.psd_exponent)
        nyquist = self.sampling_frequency / 2.0
        for name in ("low_frequency_cutoff", "high_frequency_cutoff"):
            value = getattr(self, name)
            if value is not None:
                value = _finite(name, value)
                if value < 0.0 or value > nyquist:
                    raise ValueError(f"{name} must lie in [0, sampling_frequency / 2].")
                setattr(self, name, value)
        if (
            self.low_frequency_cutoff is not None
            and self.high_frequency_cutoff is not None
            and self.low_frequency_cutoff > self.high_frequency_cutoff
        ):
            raise ValueError("low_frequency_cutoff cannot exceed high_frequency_cutoff.")


@dataclass
class TemporalNoiseConfig(ConfigModel):
    mode: str = "none"
    n_segments: int = 4
    segment_length: int | None = None
    crossfade_len: int = 128
    vary_noise_power: bool = True
    noise_power_scale_range: tuple[float, float] = (0.8, 1.2)
    vary_psd_slope: bool = False
    psd_slope_range: tuple[float, float] = (-0.1, 0.1)
    add_drift: bool = False
    drift_type: str = "spline"
    drift_sigma: float = 0.05
    drift_n_knots: int = 6
    variance_modulation: bool = False
    variance_scale_range: tuple[float, float] = (0.95, 1.05)
    variance_n_knots: int = 6
    multichannel_shared_drift: bool = True
    boundary_policy: str = "overlap_add"
    local_parameter_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    envelope_method: str = "log_pchip"
    envelope_lowpass_hz: float | None = None
    drift_rms: float | None = None
    drift_timescale_seconds: float | None = None
    drift_cutoff_hz: float | None = None
    sampling_frequency: float = 1.0
    multichannel_shared_fraction: float = 1.0
    deterministic_drift_values: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.mode not in {"none", "piecewise"}:
            raise ValueError("mode must be 'none' or 'piecewise'.")
        self.n_segments = _integer("n_segments", self.n_segments, minimum=1)
        self.crossfade_len = _integer("crossfade_len", self.crossfade_len, minimum=0)
        if self.segment_length is not None:
            self.segment_length = _integer(
                "segment_length", self.segment_length, minimum=1
            )
        self.noise_power_scale_range = _ordered_range(
            "noise_power_scale_range", self.noise_power_scale_range
        )
        self.psd_slope_range = _ordered_range("psd_slope_range", self.psd_slope_range)
        self.variance_scale_range = _ordered_range(
            "variance_scale_range", self.variance_scale_range
        )
        if self.noise_power_scale_range[0] < 0.0 or self.variance_scale_range[0] <= 0.0:
            raise ValueError("power scales must be non-negative and variance scales positive.")
        self.drift_sigma = _finite("drift_sigma", self.drift_sigma)
        self.drift_n_knots = _integer("drift_n_knots", self.drift_n_knots, minimum=2)
        self.variance_n_knots = _integer(
            "variance_n_knots", self.variance_n_knots, minimum=2
        )
        if self.drift_sigma < 0.0:
            raise ValueError("drift settings and knot counts are invalid.")
        if self.boundary_policy not in {"hard", "overlap_add", "continuous"}:
            raise ValueError("boundary_policy must be hard, overlap_add, or continuous.")
        if self.drift_type not in {"spline", "random_walk", "lowpass", "deterministic"}:
            raise ValueError("Unsupported drift_type.")
        if any(not np.isfinite(float(value)) for value in self.deterministic_drift_values):
            raise ValueError("deterministic_drift_values must be finite.")
        if self.envelope_method not in {"log_pchip", "log_linear", "lowpass"}:
            raise ValueError("Unsupported envelope_method.")
        self.sampling_frequency = _finite("sampling_frequency", self.sampling_frequency)
        if self.sampling_frequency <= 0:
            raise ValueError("sampling_frequency must be positive.")
        if self.drift_rms is not None and _finite("drift_rms", self.drift_rms) < 0:
            raise ValueError("drift_rms must be non-negative.")
        for name in ("drift_timescale_seconds", "drift_cutoff_hz", "envelope_lowpass_hz"):
            value = getattr(self, name)
            if value is not None and _finite(name, value) <= 0:
                raise ValueError(f"{name} must be positive.")
        self.multichannel_shared_fraction = _finite(
            "multichannel_shared_fraction", self.multichannel_shared_fraction
        )
        if not 0 <= self.multichannel_shared_fraction <= 1:
            raise ValueError("multichannel_shared_fraction must lie in [0, 1].")
        for name, bounds in self.local_parameter_ranges.items():
            self.local_parameter_ranges[name] = _ordered_range(name, bounds)


@dataclass
class ArtifactConfig(ConfigModel):
    sampling_frequency: float = 1.0
    enable_lines: bool = False
    lines: list[dict[str, Any]] = field(default_factory=list)
    enable_glitches: bool = False
    glitch_rate: float = 1.0
    glitch_amp_range: tuple[float, float] = (0.05, 0.2)
    glitch_templates: list[str] = field(
        default_factory=lambda: ["impulse", "exp_decay", "damped_sine"]
    )
    glitch_duration_samples: tuple[int, int] = (32, 256)
    enable_bursts: bool = False
    burst_rate: float = 0.2
    burst_amp_range: tuple[float, float] = (0.03, 0.1)
    burst_duration_samples: tuple[int, int] = (128, 512)
    enable_sparse_impulses: bool = False
    impulse_probability: float = 1e-4
    impulse_sigma: float = 0.1
    channel_amplitude_jitter: float = 0.05
    amplitude_unit: str = "raw"
    local_rms_window_samples: int = 256
    glitch_duration_seconds: tuple[float, float] | None = None
    burst_duration_seconds: tuple[float, float] | None = None
    event_process: str = "homogeneous"
    rate_profile: list[float] = field(default_factory=list)
    hawkes_branching_ratio: float = 0.0
    hawkes_decay_seconds: float = 1.0
    overlap_policy: str = "superpose"
    boundary_policy: str = "truncate"

    def __post_init__(self) -> None:
        self.sampling_frequency = _finite("sampling_frequency", self.sampling_frequency)
        if self.sampling_frequency <= 0.0:
            raise ValueError("sampling_frequency must be positive.")
        for name in ("glitch_rate", "burst_rate", "impulse_sigma", "channel_amplitude_jitter"):
            value = _finite(name, getattr(self, name))
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative.")
            setattr(self, name, value)
        self.impulse_probability = _finite("impulse_probability", self.impulse_probability)
        if not 0.0 <= self.impulse_probability <= 1.0:
            raise ValueError("impulse_probability must lie in [0, 1].")
        self.glitch_amp_range = _ordered_range("glitch_amp_range", self.glitch_amp_range)
        self.burst_amp_range = _ordered_range("burst_amp_range", self.burst_amp_range)
        for name in ("glitch_duration_samples", "burst_duration_samples"):
            low, high = _ordered_range(name, getattr(self, name))
            if low <= 0 or not low.is_integer() or not high.is_integer():
                raise ValueError(f"{name} must contain positive integer sample counts.")
            setattr(self, name, (int(low), int(high)))
        supported_templates = {"impulse", "exp_decay", "damped_sine", "ringing"}
        if not self.glitch_templates or not set(self.glitch_templates) <= supported_templates:
            raise ValueError(
                f"glitch_templates must be drawn from {sorted(supported_templates)}."
            )
        for index, line in enumerate(self.lines):
            if not {"freq", "amp"} <= set(line):
                raise ValueError(f"lines[{index}] requires freq and amp.")
            frequency = _finite(f"lines[{index}].freq", line["freq"])
            harmonics = line.get("harmonics", [1])
            if not harmonics:
                raise ValueError(f"lines[{index}].harmonics cannot be empty.")
            for harmonic in harmonics:
                harmonic = _finite(f"lines[{index}].harmonic", harmonic)
                if harmonic <= 0.0 or not 0.0 <= frequency * harmonic <= self.sampling_frequency / 2.0:
                    raise ValueError(
                        f"lines[{index}] frequencies must lie in [0, fs / 2]."
                    )
            amplitude = line["amp"]
            if np.isscalar(amplitude):
                _finite(f"lines[{index}].amp", amplitude)
            else:
                _ordered_range(f"lines[{index}].amp", amplitude)
        if self.amplitude_unit == "snr":
            # Schema-v3 uses a scientifically explicit name. This historical
            # quantity is an unweighted template-energy/RMS ratio, not
            # matched-filter SNR in colored noise.
            self.amplitude_unit = "rms_energy_ratio"
        if self.amplitude_unit not in {
            "raw", "baseline_rms", "local_rms", "rms_energy_ratio"
        }:
            raise ValueError("Unsupported amplitude_unit.")
        if self.event_process not in {"homogeneous", "nonhomogeneous", "hawkes"}:
            raise ValueError("Unsupported event_process.")
        if self.overlap_policy == "allow":
            self.overlap_policy = "superpose"
        if self.overlap_policy == "merge":
            raise ValueError(
                "overlap_policy='merge' had no defined waveform semantics and was removed; "
                "use superpose, reject, or resample."
            )
        if self.overlap_policy not in {"superpose", "reject", "resample"}:
            raise ValueError("Unsupported overlap_policy.")
        if self.boundary_policy not in {"truncate", "reject", "center"}:
            raise ValueError("Unsupported boundary_policy.")
        self.local_rms_window_samples = _integer(
            "local_rms_window_samples", self.local_rms_window_samples, minimum=1
        )
        for name in ("glitch_duration_seconds", "burst_duration_seconds"):
            value = getattr(self, name)
            if value is not None:
                bounds = _ordered_range(name, value)
                if bounds[0] <= 0:
                    raise ValueError(f"{name} must be positive.")
                setattr(self, name, bounds)
        self.hawkes_branching_ratio = _finite(
            "hawkes_branching_ratio", self.hawkes_branching_ratio
        )
        self.hawkes_decay_seconds = _finite(
            "hawkes_decay_seconds", self.hawkes_decay_seconds
        )
        if not 0 <= self.hawkes_branching_ratio < 1 or self.hawkes_decay_seconds <= 0:
            raise ValueError("Invalid Hawkes parameters.")


@dataclass
class MultiChannelConfig(ConfigModel):
    mode: str = "shared_private"
    n_channels: int = 56
    corr_strength: float = 0.3
    channel_gain_jitter: float = 0.05
    n_latent: int = 2
    latent_strength_range: tuple[float, float] = (0.1, 0.4)
    private_strength_range: tuple[float, float] = (0.8, 1.2)
    normalize_channel_variance: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"independent", "shared_private", "lowrank"}:
            raise ValueError("Unsupported multichannel mode.")
        self.n_channels = _integer("n_channels", self.n_channels, minimum=1)
        self.n_latent = _integer("n_latent", self.n_latent, minimum=1)
        self.corr_strength = _finite("corr_strength", self.corr_strength)
        if not 0.0 <= self.corr_strength < 1.0:
            raise ValueError("corr_strength must lie in [0, 1).")
        self.channel_gain_jitter = _finite(
            "channel_gain_jitter", self.channel_gain_jitter
        )
        if self.channel_gain_jitter < 0.0:
            raise ValueError("channel_gain_jitter must be non-negative.")
        self.latent_strength_range = _ordered_range(
            "latent_strength_range", self.latent_strength_range
        )
        self.private_strength_range = _ordered_range(
            "private_strength_range", self.private_strength_range
        )
        if self.latent_strength_range[0] < 0.0 or self.private_strength_range[0] < 0.0:
            raise ValueError("Channel strength ranges must be non-negative.")
