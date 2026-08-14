"""EXP-01: paper-convention statistical gates for NoiseGenerator.

Closes two coverage gaps found while auditing existing evidence
(``results/audits/noise_generator_provenance.md``):

1. ``test_ensemble_periodogram_matches_white_target_psd`` in
   ``test_noise_statistical_validation.py`` explicitly excludes the DC and
   Nyquist bins ("whose estimator distribution differs"), so no test
   empirically confirms those endpoint bins carry the full ``P_k * f_s * N``
   power (as opposed to the halved interior-bin power) for the actual
   ``NoiseGenerator`` class used by the modular noise stack.
2. ``test_circulant_embedding_recovers_requested_covariance`` empirically
   checks ensemble recovery for the circulant-embedding path of
   ``sample_from_covariance``, but the "dense"/exact-finite-Toeplitz path is
   only checked via its deterministic ``maximum_covariance_error`` metadata,
   never via an empirical ensemble statistic.

These are additive regression gates; they do not replace or alter any
existing test.
"""

from __future__ import annotations

import numpy as np

from noise_module import NoiseGenerator


def _base_config(**updates):
    config = {
        "noise_type": "white",
        "noise_power": 1.0,
        "sampling_frequency": 1024.0,
    }
    config.update(updates)
    return config


def test_dc_and_nyquist_bins_carry_full_endpoint_power() -> None:
    """DC and Nyquist rFFT bins must have expected power P_k*f_s*N (the full,
    undivided endpoint formula), not the halved interior-bin formula.
    """
    N = 2048
    # power_definition="variance" (the default) deliberately zeroes the DC
    # bin, so it cannot be used to test DC endpoint scaling; "mean_square"
    # keeps DC stochastic and lets this test exercise the actual formula.
    generator = NoiseGenerator(_base_config(power_definition="mean_square"), seed=7)
    _, density = generator.build_psd_density(N)
    power = generator.psd_density_to_rfft_power(density, generator.sampling_frequency, N)

    n_draws = 4000
    dc_samples = np.empty(n_draws)
    nyq_samples = np.empty(n_draws)
    for i in range(n_draws):
        x = generator.generate_noise(N)
        X = np.fft.rfft(x)
        dc_samples[i] = X[0].real
        nyq_samples[i] = X[-1].real

    dc_empirical_power = float(np.mean(dc_samples**2))
    nyq_empirical_power = float(np.mean(nyq_samples**2))
    assert abs(dc_empirical_power / power[0] - 1.0) < 0.15
    assert abs(nyq_empirical_power / power[-1] - 1.0) < 0.15
    # Endpoint power must be double the neighbouring interior bin's power
    # (same density, but interior bins are halved relative to endpoints).
    assert power[0] > 1.5 * power[1]
    assert power[-1] > 1.5 * power[-2]


def test_dc_and_nyquist_bins_are_real_valued() -> None:
    N = 512
    generator = NoiseGenerator(_base_config(), seed=11)
    for _ in range(20):
        X = np.fft.rfft(generator.generate_noise(N))
        assert abs(X[0].imag) < 1e-8
        assert abs(X[-1].imag) < 1e-8


def test_normalized_interior_periodogram_is_exponential() -> None:
    """Interior-bin |X_k|^2 / E|X_k|^2 pooled across bins and draws should
    have Exponential(mean=1) first and second moments -- the Rayleigh-modulus
    signature of a genuinely complex-Gaussian (not fixed-modulus) bin.
    """
    N = 512
    generator = NoiseGenerator(_base_config(), seed=5)
    _, density = generator.build_psd_density(N)
    power = generator.psd_density_to_rfft_power(density, generator.sampling_frequency, N)

    draws = []
    for _ in range(3000):
        X = np.fft.rfft(generator.generate_noise(N))
        draws.append(np.abs(X[2:-1]) ** 2)
    normalized = np.array(draws) / power[2:-1][None, :]
    pooled = normalized.ravel()
    assert abs(float(np.mean(pooled)) - 1.0) < 0.05
    assert abs(float(np.var(pooled)) - 1.0) < 0.15


def test_dense_toeplitz_ensemble_recovers_requested_covariance() -> None:
    """Empirical companion to test_dense_covariance_fallback_and_indefinite_
    rejection (which only checks the deterministic eigendecomposition
    metadata): actually draw an ensemble through the dense/exact-Toeplitz
    path and confirm the sample covariance matches the requested covariance.
    """
    first_row = 0.6 ** np.arange(10)
    generator = NoiseGenerator(_base_config(), seed=3)
    samples = np.vstack(
        [generator.sample_from_covariance(first_row, method="dense") for _ in range(6000)]
    )
    empirical_var = float(np.var(samples[:, 0]))
    empirical_lag1 = float(np.mean(samples[:, 0] * samples[:, 1]))
    assert abs(empirical_var - first_row[0]) < 0.05
    assert abs(empirical_lag1 - first_row[1]) < 0.05
