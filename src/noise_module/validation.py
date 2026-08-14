"""Confidence-aware statistical validation for generated noise."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np
from scipy.signal import welch
from scipy.stats import kstest, norm, poisson, skew, kurtosis


@dataclass
class ValidationConfig:
    ensemble_size: int = 256
    confidence_level: float = 0.95
    seeds: tuple[int, ...] = (11, 23, 37, 51)
    bootstrap_samples: int = 500
    relative_psd_tolerance: float = 0.15
    max_whitened_autocorrelation: float | None = None
    minimum_distribution_pvalue: float = 0.01
    maximum_fourier_skewness: float = 0.10
    maximum_fourier_excess_kurtosis: float = 0.15
    relative_power_tolerance: float = 0.10
    relative_csd_magnitude_tolerance: float = 0.20
    absolute_phase_tolerance_radians: float = 0.20
    absolute_coherence_tolerance: float = 0.10
    relative_covariance_tolerance: float = 0.15

    def __post_init__(self):
        if self.ensemble_size < 2 or self.bootstrap_samples < 20:
            raise ValueError("Validation ensemble/bootstrap sizes are too small.")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must lie in (0, 1).")
        for name in (
            "relative_psd_tolerance",
            "minimum_distribution_pvalue",
            "maximum_fourier_skewness",
            "maximum_fourier_excess_kurtosis",
            "relative_power_tolerance",
            "relative_csd_magnitude_tolerance",
            "absolute_phase_tolerance_radians",
            "absolute_coherence_tolerance",
            "relative_covariance_tolerance",
        ):
            value = float(getattr(self, name))
            if value < 0.0 or not np.isfinite(value):
                raise ValueError(f"{name} must be finite and non-negative.")


@dataclass
class ValidationResult:
    name: str
    passed: bool | None
    statistics: dict[str, Any]
    thresholds: dict[str, Any] = field(default_factory=dict)
    confidence_level: float = 0.95

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


def _jsonable(value: Any) -> Any:
    """Convert statistical results, including complex arrays, to JSON data."""
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {"real": value.real.tolist(), "imag": value.imag.tolist()}
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _validation_rng(config: ValidationConfig) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(config.seeds))


def _normality_test(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """Stable KS normality diagnostic on at most 5000 standardized samples."""
    samples = np.asarray(values, dtype=float).ravel()
    samples = samples[np.isfinite(samples)]
    if len(samples) < 8 or np.std(samples) == 0.0:
        return float("nan"), float("nan")
    samples = (samples - np.mean(samples)) / np.std(samples)
    if len(samples) > 5000:
        samples = rng.choice(samples, 5000, replace=False)
    result = kstest(samples, "norm")
    return float(result.statistic), float(result.pvalue)


def bootstrap_interval(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    confidence_level: float = 0.95,
    n_resamples: int = 500,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Return a percentile bootstrap confidence interval."""
    values = np.asarray(values)
    rng = rng or np.random.default_rng(0)
    estimates = np.array(
        [statistic(rng.choice(values, len(values), replace=True)) for _ in range(n_resamples)]
    )
    alpha = 1 - confidence_level
    return tuple(np.quantile(estimates, [alpha / 2, 1 - alpha / 2]))


def validate_stationary_gaussian(generator, N: int, config: ValidationConfig | None = None):
    """Validate power, PSD, Gaussian marginals, periodograms, and whitening.

    Every acceptance decision is exposed in ``thresholds["gates"]``. Practical
    tolerances are expanded, when necessary, by an ensemble-size confidence
    allowance so validation does not become stricter merely because a small
    ensemble was requested.
    """
    cfg = config or ValidationConfig()
    rng = _validation_rng(cfg)
    traces = np.vstack([generator.generate_noise(N) for _ in range(cfg.ensemble_size)])
    _, target = generator.build_psd_density(N)
    coeff = np.fft.rfft(traces, axis=1)
    estimates = np.abs(coeff) ** 2 / (generator.sampling_frequency * N)
    upper = N // 2 + 1 - (N + 1) % 2
    estimates[:, 1:upper] *= 2
    mean_psd = estimates.mean(axis=0)
    supported = target > 0
    relative_psd_error = float(
        np.mean(np.abs(mean_psd[supported] / target[supported] - 1))
    )
    variances = np.var(traces, axis=1)
    variance_ci = bootstrap_interval(
        variances,
        confidence_level=cfg.confidence_level,
        n_resamples=cfg.bootstrap_samples,
        rng=rng,
    )
    interior = coeff[:, 1:upper]
    expected_power = generator.psd_density_to_rfft_power(
        target, generator.sampling_frequency, N
    )[1:upper]
    normalized_periodogram = (
        np.abs(interior) ** 2 / np.maximum(expected_power[None, :], 1e-30)
    ).ravel()
    periodogram_test_values = normalized_periodogram
    if len(periodogram_test_values) > 5000:
        periodogram_test_values = rng.choice(
            periodogram_test_values, 5000, replace=False
        )
    exponential_ks = kstest(periodogram_test_values, "expon")
    sigma = np.sqrt(np.maximum(expected_power, 1e-30) / 2.0)
    standardized = np.concatenate(
        [(interior.real / sigma[None, :]).ravel(),
         (interior.imag / sigma[None, :]).ravel()]
    )
    whitened = coeff / np.sqrt(
        np.maximum(
            generator.psd_density_to_rfft_power(target, generator.sampling_frequency, N),
            1e-30,
        )
    )
    white_time = np.fft.irfft(whitened, n=N, axis=1)
    lag1 = float(np.mean(white_time[:, :-1] * white_time[:, 1:]) / np.var(white_time))
    autocorr_limit = cfg.max_whitened_autocorrelation or (
        norm.ppf(0.5 + cfg.confidence_level / 2) / np.sqrt(cfg.ensemble_size * N)
    )
    target_variance = float(np.sum(target[1:]) * generator.sampling_frequency / N)
    power_relative_error = float(
        abs(np.mean(variances) - target_variance) / max(target_variance, 1e-30)
    )
    variance_standard_error = float(
        np.std(variances, ddof=1) / np.sqrt(len(variances))
    )
    z_score = float(norm.ppf(0.5 + cfg.confidence_level / 2))
    power_limit = max(
        cfg.relative_power_tolerance,
        z_score * variance_standard_error / max(target_variance, 1e-30),
    )
    real_ks_stat, real_ks_pvalue = _normality_test(
        interior.real / sigma[None, :], rng
    )
    imag_ks_stat, imag_ks_pvalue = _normality_test(
        interior.imag / sigma[None, :], rng
    )
    # One time sample per realization avoids treating strongly correlated
    # within-record samples as independent observations.
    marginal_ks_stat, marginal_ks_pvalue = _normality_test(
        traces[:, N // 2], rng
    )
    endpoint_pvalues: dict[str, float | None] = {"dc": None, "nyquist": None}
    endpoint_gates: dict[str, bool | None] = {"dc": None, "nyquist": None}
    endpoint_power = generator.psd_density_to_rfft_power(
        target, generator.sampling_frequency, N
    )
    for name, index in (("dc", 0), ("nyquist", -1)):
        applicable = name == "dc" or N % 2 == 0
        if applicable and endpoint_power[index] > 0.0:
            _, pvalue = _normality_test(coeff[:, index].real, rng)
            endpoint_pvalues[name] = pvalue
            endpoint_gates[name] = bool(pvalue >= cfg.minimum_distribution_pvalue)
    gates = {
        "power": power_relative_error <= power_limit,
        "psd": relative_psd_error <= cfg.relative_psd_tolerance,
        "fourier_real_normality": real_ks_pvalue >= cfg.minimum_distribution_pvalue,
        "fourier_imag_normality": imag_ks_pvalue >= cfg.minimum_distribution_pvalue,
        "time_marginal_normality": marginal_ks_pvalue >= cfg.minimum_distribution_pvalue,
        "periodogram_exponential": exponential_ks.pvalue >= cfg.minimum_distribution_pvalue,
        "fourier_skewness": abs(float(skew(standardized))) <= cfg.maximum_fourier_skewness,
        "fourier_excess_kurtosis": (
            abs(float(kurtosis(standardized))) <= cfg.maximum_fourier_excess_kurtosis
        ),
        "whitened_lag1": abs(lag1) <= max(autocorr_limit, 0.02),
        "dc_real_constraint": bool(np.all(np.isreal(coeff[:, 0]))),
        "nyquist_real_constraint": bool(N % 2 or np.all(np.isreal(coeff[:, -1]))),
        "dc_distribution": endpoint_gates["dc"],
        "nyquist_distribution": endpoint_gates["nyquist"],
    }
    stats = {
        "variance_mean": float(np.mean(variances)),
        "variance_confidence_interval": variance_ci,
        "target_variance": target_variance,
        "relative_power_error": power_relative_error,
        "mean_relative_psd_error": relative_psd_error,
        "fourier_skewness": float(skew(standardized)),
        "fourier_excess_kurtosis": float(kurtosis(standardized)),
        "fourier_real_normality_ks_statistic": real_ks_stat,
        "fourier_real_normality_ks_pvalue": real_ks_pvalue,
        "fourier_imag_normality_ks_statistic": imag_ks_stat,
        "fourier_imag_normality_ks_pvalue": imag_ks_pvalue,
        "time_marginal_normality_ks_statistic": marginal_ks_stat,
        "time_marginal_normality_ks_pvalue": marginal_ks_pvalue,
        "normalized_periodogram_ks_statistic": float(exponential_ks.statistic),
        "normalized_periodogram_ks_pvalue": float(exponential_ks.pvalue),
        "whitened_lag1_autocorrelation": lag1,
        "dc_is_real": bool(np.all(np.isreal(coeff[:, 0]))),
        "nyquist_is_real": bool(N % 2 or np.all(np.isreal(coeff[:, -1]))),
        "endpoint_normality_pvalues": endpoint_pvalues,
    }
    passed = all(value for value in gates.values() if value is not None)
    return ValidationResult(
        "stationary_gaussian",
        passed,
        stats,
        {
            "gates": gates,
            "relative_power_tolerance": power_limit,
            "relative_psd_tolerance": cfg.relative_psd_tolerance,
            "minimum_distribution_pvalue": cfg.minimum_distribution_pvalue,
            "maximum_fourier_skewness": cfg.maximum_fourier_skewness,
            "maximum_fourier_excess_kurtosis": cfg.maximum_fourier_excess_kurtosis,
            "whitened_autocorrelation_limit": max(autocorr_limit, 0.02),
        },
        cfg.confidence_level,
    )


def validate_local_nonstationarity(
    x: np.ndarray,
    sampling_frequency: float,
    window_samples: int,
    *,
    expected_mean_range: tuple[float, float] | None = None,
    expected_variance_range: tuple[float, float] | None = None,
    expected_slope_range: tuple[float, float] | None = None,
    minimum_variance_ratio: float | None = None,
) -> ValidationResult:
    """Measure local statistics and evaluate explicitly supplied contracts.

    Without at least one expected range or variation threshold this function is
    diagnostic-only and returns ``passed=None`` rather than claiming success.
    """
    x = np.asarray(x)
    if x.ndim != 1 or window_samples < 4 or len(x) < window_samples:
        raise ValueError("x must be one-dimensional and contain at least one valid window.")
    windows = [
        x[start : start + window_samples]
        for start in range(0, len(x) - window_samples + 1, window_samples)
    ]
    rows = []
    for window in windows:
        f, p = welch(window, fs=sampling_frequency, nperseg=min(256, len(window)))
        positive = (f > 0) & (p > 0)
        slope = np.polyfit(np.log(f[positive]), np.log(p[positive]), 1)[0]
        midpoint = len(p) // 2
        rows.append(
            [np.mean(window), np.var(window), slope, np.sum(p[:midpoint]), np.sum(p[midpoint:])]
        )
    rows = np.asarray(rows)
    gates: dict[str, bool] = {}
    if expected_mean_range is not None:
        gates["local_mean"] = bool(
            np.all((rows[:, 0] >= expected_mean_range[0]) & (rows[:, 0] <= expected_mean_range[1]))
        )
    if expected_variance_range is not None:
        gates["local_variance"] = bool(
            np.all(
                (rows[:, 1] >= expected_variance_range[0])
                & (rows[:, 1] <= expected_variance_range[1])
            )
        )
    if expected_slope_range is not None:
        gates["local_psd_slope"] = bool(
            np.all(
                (rows[:, 2] >= expected_slope_range[0])
                & (rows[:, 2] <= expected_slope_range[1])
            )
        )
    if minimum_variance_ratio is not None:
        ratio = float(np.max(rows[:, 1]) / max(np.min(rows[:, 1]), 1e-30))
        gates["variance_modulation"] = ratio >= minimum_variance_ratio
    else:
        ratio = float(np.max(rows[:, 1]) / max(np.min(rows[:, 1]), 1e-30))
    passed = all(gates.values()) if gates else None
    return ValidationResult(
        "local_nonstationarity",
        passed,
        {
            "local_mean": rows[:, 0],
            "local_variance": rows[:, 1],
            "local_psd_slope": rows[:, 2],
            "local_low_band_power": rows[:, 3],
            "local_high_band_power": rows[:, 4],
            "local_variance_ratio": ratio,
        },
        {
            "gates": gates,
            "diagnostic_only": not bool(gates),
            "expected_mean_range": expected_mean_range,
            "expected_variance_range": expected_variance_range,
            "expected_slope_range": expected_slope_range,
            "minimum_variance_ratio": minimum_variance_ratio,
        },
    )


def validate_artifacts(
    baseline: np.ndarray,
    output: np.ndarray,
    metadata: dict[str, Any],
    *,
    confidence_level: float = 0.99,
) -> ValidationResult:
    """Cross-check artifact identity, count, duration, amplitude, and tails."""
    artifact = np.asarray(output) - np.asarray(baseline)
    reported = np.asarray(metadata.get("artifact_only", artifact))
    events = []
    for key in ("glitches", "bursts"):
        section = metadata.get(key, {})
        events.extend(section.get(key, []))
    starts = np.array([event["start"] for event in events], dtype=float)
    intervals = np.diff(np.sort(starts)) if len(starts) > 1 else np.array([])
    standardized = (output - np.mean(output)) / max(np.std(output), 1e-12)
    count_gates = []
    count_intervals = {}
    for key in ("glitches", "bursts"):
        section = metadata.get(key, {})
        expected = section.get("expected_total_count", section.get("expected_count"))
        policy = section.get("overlap_policy")
        process = section.get("event_process")
        if expected is None or policy not in {"superpose", "allow"} or process == "hawkes":
            continue
        lower, upper = poisson.interval(confidence_level, float(expected))
        observed_count = int(section.get("count", 0))
        accepted = int(lower) <= observed_count <= int(upper)
        count_gates.append(accepted)
        count_intervals[key] = {
            "expected": float(expected),
            "observed": observed_count,
            "confidence_interval": [int(lower), int(upper)],
            "passed": accepted,
        }
    durations = np.array([event["duration"] for event in events], dtype=int)
    amplitudes = np.array([event["applied_amp"] for event in events], dtype=float)
    gates = {
        "component_reconstruction": float(np.max(np.abs(artifact - reported))) < 1e-12,
        "mask_identity": bool(
            np.array_equal(metadata.get("combined_mask", artifact != 0), artifact != 0)
        ),
        "event_counts": all(count_gates) if count_gates else None,
        "positive_durations": bool(np.all(durations > 0)) if durations.size else None,
        "finite_amplitudes": bool(np.all(np.isfinite(amplitudes))) if amplitudes.size else None,
    }
    stats = {
        "reconstruction_max_error": float(np.max(np.abs(artifact - reported))),
        "mask_exact": bool(
            np.array_equal(metadata.get("combined_mask", artifact != 0), artifact != 0)
        ),
        "event_count": len(events),
        "mean_interarrival_samples": float(np.mean(intervals)) if intervals.size else None,
        "duration_samples": durations,
        "applied_amplitudes": amplitudes,
        "tail_exceedance_4sigma": float(np.mean(np.abs(standardized) > 4)),
        "count_confidence_intervals": count_intervals,
    }
    return ValidationResult(
        "artifacts",
        all(value for value in gates.values() if value is not None),
        stats,
        {"gates": gates, "count_confidence_level": confidence_level},
        confidence_level,
    )


def validate_csd_ensemble(
    X: np.ndarray,
    target_csd: np.ndarray,
    sampling_frequency: float,
    config: ValidationConfig | None = None,
) -> ValidationResult:
    """Separately validate PSD, CSD magnitude, phase, coherence, and power."""
    from .multichannel_noise import MultiChannelNoiseGenerator

    cfg = config or ValidationConfig()
    X = np.asarray(X, dtype=float)
    if X.ndim != 3 or X.shape[0] < 2:
        raise ValueError("CSD ensemble validation requires shape (R, C, N) with R >= 2.")
    diagnostics = MultiChannelNoiseGenerator.csd_diagnostics(X, sampling_frequency)
    observed = diagnostics["csd"]
    target = np.asarray(target_csd, dtype=complex)
    if target.shape != observed.shape:
        raise ValueError("target_csd must match the ensemble rFFT grid and channel count.")
    F, C, _ = target.shape
    offdiagonal_mask = np.broadcast_to(~np.eye(C, dtype=bool), (F, C, C))
    target_magnitude = np.abs(target)
    support_floor = max(float(np.max(target_magnitude)) * 1e-12, 1e-30)
    psd_mask = np.broadcast_to(np.eye(C, dtype=bool), (F, C, C)) & (
        target_magnitude > support_floor
    )
    cross_mask = offdiagonal_mask & (target_magnitude > support_floor)
    psd_relative_error = float(
        np.mean(np.abs(observed[psd_mask].real - target[psd_mask].real)
                / np.maximum(np.abs(target[psd_mask].real), support_floor))
    )
    magnitude_relative_error = (
        float(
            np.mean(
                np.abs(np.abs(observed[cross_mask]) - target_magnitude[cross_mask])
                / target_magnitude[cross_mask]
            )
        )
        if np.any(cross_mask)
        else 0.0
    )
    phase_difference = np.angle(observed * target.conj())
    phase_error = (
        float(np.mean(np.abs(phase_difference[cross_mask])))
        if np.any(cross_mask)
        else 0.0
    )
    target_diagonal = np.real(np.diagonal(target, axis1=1, axis2=2))
    target_coherence = target_magnitude**2 / np.maximum(
        target_diagonal[:, :, None] * target_diagonal[:, None, :], support_floor
    )
    coherence_error = (
        float(
            np.mean(
                np.abs(
                    diagnostics["coherence"][cross_mask]
                    - target_coherence[cross_mask]
                )
            )
        )
        if np.any(cross_mask)
        else 0.0
    )
    df = sampling_frequency / X.shape[-1]
    target_cov = np.sum(target, axis=0) * df
    samples = np.transpose(X, (1, 0, 2)).reshape(X.shape[1], -1)
    observed_cov = np.cov(samples)
    covariance_error = float(np.max(np.abs(observed_cov - target_cov.real)))
    covariance_scale = max(float(np.max(np.abs(target_cov.real))), 1e-30)
    covariance_relative_error = covariance_error / covariance_scale
    z_score = float(norm.ppf(0.5 + cfg.confidence_level / 2))
    ensemble_allowance = z_score / np.sqrt(X.shape[0])
    thresholds = {
        "relative_psd": max(cfg.relative_psd_tolerance, 2.0 * ensemble_allowance),
        "relative_csd_magnitude": max(
            cfg.relative_csd_magnitude_tolerance, 2.0 * ensemble_allowance
        ),
        "absolute_phase_radians": max(
            cfg.absolute_phase_tolerance_radians, ensemble_allowance
        ),
        "absolute_coherence": max(
            cfg.absolute_coherence_tolerance, ensemble_allowance
        ),
        "relative_covariance": max(
            cfg.relative_covariance_tolerance, 2.0 * ensemble_allowance
        ),
    }
    gates = {
        "psd": psd_relative_error <= thresholds["relative_psd"],
        "csd_magnitude": magnitude_relative_error <= thresholds["relative_csd_magnitude"],
        "phase": phase_error <= thresholds["absolute_phase_radians"],
        "coherence": coherence_error <= thresholds["absolute_coherence"],
        "integrated_covariance": covariance_relative_error <= thresholds["relative_covariance"],
    }
    return ValidationResult(
        "csd_ensemble",
        all(gates.values()),
        {
            "mean_relative_csd_error": magnitude_relative_error,
            "mean_relative_psd_error": psd_relative_error,
            "mean_relative_csd_magnitude_error": magnitude_relative_error,
            "mean_absolute_phase_error_radians": phase_error,
            "mean_absolute_coherence_error": coherence_error,
            "maximum_covariance_error": covariance_error,
            "relative_covariance_error": covariance_relative_error,
            "observed_coherence": diagnostics["coherence"],
            "observed_phase": diagnostics["phase"],
            "target_coherence": target_coherence,
            "target_phase": np.angle(target),
            "target_covariance": target_cov,
            "observed_covariance": observed_cov,
        },
        {"gates": gates, **thresholds},
        cfg.confidence_level,
    )
