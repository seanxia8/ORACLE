from __future__ import annotations

import numpy as np
from scipy.signal import periodogram
from scipy.stats import kurtosis, skew

from noise_module import (
    ArtifactInjector,
    MultiChannelNoiseGenerator,
    NoiseGenerator,
    TemporalNoiseWrapper,
)


def _base_config(**updates):
    config = {
        "noise_type": "white",
        "noise_power": 1.0,
        "sampling_frequency": 1024.0,
    }
    config.update(updates)
    return config


def test_analytic_noise_power_is_expected_variance() -> None:
    N = 2048
    variances = [
        np.var(NoiseGenerator(_base_config(), seed=seed).generate_noise(N))
        for seed in range(128)
    ]

    assert abs(np.mean(variances) - 1.0) < 0.02


def test_ensemble_periodogram_matches_white_target_psd() -> None:
    N = 2048
    generator = NoiseGenerator(_base_config(), seed=123)
    _, target = generator.build_psd_density(N)
    estimates = []
    for _ in range(128):
        x = generator.generate_noise(N)
        _, estimate = periodogram(
            x,
            fs=generator.sampling_frequency,
            window="boxcar",
            detrend=False,
            scaling="density",
            return_onesided=True,
        )
        estimates.append(estimate)
    mean_estimate = np.mean(estimates, axis=0)

    # Exclude real-valued endpoint bins, whose estimator distribution differs.
    relative_error = np.mean(np.abs(mean_estimate[1:-1] / target[1:-1] - 1.0))
    assert relative_error < 0.09


def test_white_noise_ensemble_has_gaussian_marginals() -> None:
    generator = NoiseGenerator(_base_config(), seed=321)
    samples = np.concatenate([generator.generate_noise(2048) for _ in range(64)])

    assert abs(skew(samples)) < 0.03
    assert abs(kurtosis(samples, fisher=True)) < 0.06


def test_custom_psd_scaling_modes_control_integrated_power(tmp_path) -> None:
    path = tmp_path / "custom_psd.npy"
    frequencies = np.linspace(0.0, 512.0, 513)
    density = np.full_like(frequencies, 0.25)
    np.save(path, np.vstack([frequencies, density]))

    absolute = NoiseGenerator(
        _base_config(noise_type=str(path), noise_power=2.0), seed=1
    )
    normalized = NoiseGenerator(
        _base_config(
            noise_type=str(path),
            noise_power=2.0,
            custom_psd_scaling="normalize",
        ),
        seed=1,
    )
    multiplied = NoiseGenerator(
        _base_config(
            noise_type=str(path),
            noise_power=2.0,
            custom_psd_scaling="multiply",
        ),
        seed=1,
    )

    df = 1024.0 / 1024
    _, absolute_psd = absolute.build_psd_density(1024)
    _, normalized_psd = normalized.build_psd_density(1024)
    _, multiplied_psd = multiplied.build_psd_density(1024)
    assert np.isclose(np.sum(normalized_psd) * df, 2.0)
    assert np.allclose(multiplied_psd, 2.0 * absolute_psd)


def test_artifact_multichannel_is_identity_when_disabled() -> None:
    X = np.arange(200, dtype=float).reshape(2, 100)
    injector = ArtifactInjector(
        {"sampling_frequency": 100.0, "channel_amplitude_jitter": 0.5},
        seed=4,
    )

    assert np.array_equal(injector.apply_multichannel(X), X)


def test_event_rates_are_normalized_by_trace_duration() -> None:
    counts = []
    for seed in range(1000):
        injector = ArtifactInjector(
            {
                "sampling_frequency": 100.0,
                "enable_glitches": True,
                "glitch_rate": 3.0,
                "glitch_duration_samples": [4, 4],
            },
            seed=seed,
        )
        _, metadata = injector.apply(np.zeros(200), return_metadata=True)
        counts.append(metadata["glitches"]["count"])

    # Two seconds at 3 Hz gives a Poisson mean of six events.
    assert abs(np.mean(counts) - 6.0) < 0.25


def test_piecewise_psd_slope_variation_is_implemented() -> None:
    wrapper = TemporalNoiseWrapper(
        {
            "mode": "piecewise",
            "n_segments": 2,
            "crossfade_len": 0,
            "vary_noise_power": False,
            "vary_psd_slope": True,
            "psd_slope_range": [0.5, 0.5],
        },
        seed=2,
    )
    _, metadata = wrapper.generate_piecewise(
        2048,
        NoiseGenerator(_base_config(noise_type="pink"), seed=1),
        return_metadata=True,
    )

    assert all(
        np.isclose(segment["psd_exponent"], -0.5)
        for segment in metadata["segments"]
    )


def test_target_csd_synthesis_recovers_covariance() -> None:
    N = 1024
    fs = 1024.0
    covariance = np.array([[1.0, 0.4], [0.4, 2.0]])
    n_positive_bins = N // 2
    density = covariance / (n_positive_bins * fs / N)
    target_csd = np.repeat(density[None, :, :], N // 2 + 1, axis=0)
    target_csd[0] = 0.0

    generator = MultiChannelNoiseGenerator(_base_config(), seed=50)
    realizations = [
        generator.generate_from_csd(target_csd, N) for _ in range(128)
    ]
    samples = np.concatenate(realizations, axis=1)

    assert np.allclose(np.cov(samples), covariance, rtol=0.05, atol=0.04)


def test_target_csd_rejects_non_positive_semidefinite_matrix() -> None:
    N = 32
    target_csd = np.zeros((N // 2 + 1, 2, 2), dtype=float)
    target_csd[:] = [[1.0, 2.0], [2.0, 1.0]]

    generator = MultiChannelNoiseGenerator(_base_config(), seed=1)
    with np.testing.assert_raises_regex(ValueError, "positive semidefinite"):
        generator.generate_from_csd(target_csd, N)
