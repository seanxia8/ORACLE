from __future__ import annotations

import numpy as np
import pytest

from noise_module import (
    AL2O3_DEFAULT_SAMPLES,
    AL2O3_DEFAULT_SAMPLING_FREQUENCY,
    al2o3_athermal_noise_generator,
    build_optimal_filter,
    fit_reference_pulse,
    pulse_template_2,
    recommend_record_length,
)


def test_phase3_composite_generator_uses_absolute_reference_psd() -> None:
    generator = al2o3_athermal_noise_generator(seed=4)
    frequency, psd = generator.build_psd_density(4096)
    expected_high_frequency_floor = 0.0030307426626081345 + 0.09
    assert generator.sampling_frequency == 1_000_000.0
    assert np.all(psd[1:] > 0.0)
    assert psd[-1] == pytest.approx(expected_high_frequency_floor, rel=3e-4)
    assert generator.generate_noise(4096).shape == (4096,)


def test_phase5_pulse_fit_reproduces_signal_magnitude() -> None:
    fit = fit_reference_pulse()
    assert fit.tau_in == pytest.approx(5e-6, rel=1e-8)
    assert fit.tau_t == pytest.approx(1e-3, rel=1e-8)
    assert fit.rms_log_error < 1e-10
    assert fit.unidentifiable_parameters == ("t0", "tau_n")

    t = np.arange(20_000) / 1_000_000.0
    pulse = pulse_template_2(
        t, 0.001, fit.An, fit.At, fit.tau_n, fit.tau_in, fit.tau_t
    )
    assert np.all(pulse[t < 0.001] == 0.0)
    assert np.max(np.abs(pulse)) > 0.0


def test_record_length_balances_pulse_and_frequency_resolution() -> None:
    recommendation = recommend_record_length(1e-3)
    assert recommendation["n_samples"] == AL2O3_DEFAULT_SAMPLES
    assert recommendation["frequency_resolution_hz"] < 10.0
    assert recommendation["pulse_decay_constants"] > 100.0
    one_hz = recommend_record_length(1e-3, minimum_frequency_hz=1.0)
    assert one_hz["n_samples"] == 1_048_576


def test_phase6_optimal_filter_recovers_injected_amplitude() -> None:
    optimal_filter = build_optimal_filter(n_samples=16_384)
    assert optimal_filter.sampling_frequency == AL2O3_DEFAULT_SAMPLING_FREQUENCY
    assert optimal_filter.estimate_amplitude(2.75 * optimal_filter.template) == pytest.approx(
        2.75, rel=1e-12
    )

    generator = al2o3_athermal_noise_generator(seed=12)
    estimates = []
    for noise in generator.generate_ensemble(16, 16_384):
        trace = noise + 5.0 * optimal_filter.template
        estimates.append(optimal_filter.estimate_amplitude(trace))
    assert abs(np.mean(estimates) - 5.0) < 0.5
