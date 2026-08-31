# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the modular noise simulator written for the ORACLE study.
# If you use this module in published work, please cite it: see CITATION.cff
# at the repository root.
from __future__ import annotations

import numpy as np

from noise_module import (
    ArtifactInjector,
    MultiChannelNoiseGenerator,
    NoiseGenerator,
    NonGaussianNoiseGenerator,
    TemporalNoiseWrapper,
)


BASE = {"noise_type": "white", "noise_power": 1.0, "sampling_frequency": 256.0}


def test_temporal_apply_never_discards_input_and_returns_components() -> None:
    x = np.linspace(-1, 1, 512)
    wrapper = TemporalNoiseWrapper(
        {
            "mode": "none",
            "variance_modulation": True,
            "variance_scale_range": [0.8, 1.2],
            "add_drift": True,
            "drift_rms": 0.1,
        },
        seed=2,
    )
    y, meta = wrapper.apply(x, return_components=True)
    components = meta["components"]
    reconstructed = x * components["multiplicative_envelope"] + components["additive_drift"]
    assert np.allclose(y, reconstructed)
    assert meta["operations"] == {"regenerated": False, "multiplied": True, "added": True}
    assert np.all((components["multiplicative_envelope"] >= 0.8) &
                  (components["multiplicative_envelope"] <= 1.2))


def test_temporal_disabled_apply_is_exact_identity() -> None:
    x = np.arange(100.0)
    y, meta = TemporalNoiseWrapper({}, seed=1).apply(x, return_metadata=True)
    assert np.array_equal(x, y)
    assert not any(meta["operations"].values())


def test_piecewise_boundary_policies_and_arbitrary_local_parameters() -> None:
    base = NoiseGenerator(BASE, seed=1)
    for policy in ("hard", "overlap_add", "continuous"):
        wrapper = TemporalNoiseWrapper(
            {
                "mode": "piecewise",
                "n_segments": 3,
                "crossfade_len": 16,
                "boundary_policy": policy,
                "local_parameter_ranges": {"deterministic_mean": [0.2, 0.2]},
            },
            seed=4,
        )
        x, meta = wrapper.generate_piecewise(301, base, return_metadata=True)
        assert x.shape == (301,)
        assert meta["boundary_policy"] == policy
        assert all(seg["start"] < seg["end"] for seg in meta["segments"])


def test_evolutionary_overlap_add_has_exact_length_and_schedule() -> None:
    wrapper = TemporalNoiseWrapper({"vary_noise_power": False}, seed=7)
    x, meta = wrapper.generate_evolutionary(
        1000,
        NoiseGenerator(BASE, seed=2),
        window_samples=256,
        hop_samples=128,
        parameter_schedule=[{"noise_power": 0.5}, {"noise_power": 2.0}],
        return_metadata=True,
    )
    assert x.shape == (1000,)
    assert meta["method"] == "evolutionary_overlap_add"
    assert meta["window"] == "sqrt_hann"
    assert meta["windows"][0]["config"]["noise_power"] == 0.5
    assert meta["windows"][1]["config"]["noise_power"] == 2.0
    assert np.var(x[650:900]) > 2 * np.var(x[64:192])


def test_lowpass_drift_has_requested_rms_and_low_frequency_power() -> None:
    wrapper = TemporalNoiseWrapper(
        {
            "drift_type": "lowpass",
            "drift_rms": 0.3,
            "drift_cutoff_hz": 2.0,
            "sampling_frequency": 256.0,
        },
        seed=8,
    )
    drift = wrapper.generate_drift(8192)
    spectrum = np.abs(np.fft.rfft(drift)) ** 2
    frequencies = np.fft.rfftfreq(len(drift), 1 / 256.0)
    assert np.isclose(np.std(drift), 0.3)
    assert spectrum[frequencies <= 4].sum() > 20 * spectrum[frequencies >= 20].sum()


def test_artifact_components_reconstruct_output_and_preserve_parameters() -> None:
    x = np.random.default_rng(1).normal(size=2048)
    injector = ArtifactInjector(
        {
            "sampling_frequency": 256.0,
            "enable_lines": True,
            "lines": [{"freq": 32, "amp": 3.0, "phase": 0.25}],
            "amplitude_unit": "snr",
        },
        seed=3,
    )
    y, meta = injector.apply(x, return_components=True)
    assert np.allclose(y, x + meta["artifact_only"])
    assert np.array_equal(meta["combined_mask"], np.abs(meta["artifact_only"]) > 0)
    line = meta["lines"]["lines"][0]
    achieved = np.linalg.norm(meta["artifact_components"]["lines"]) / np.std(x)
    assert np.isclose(achieved, 3.0)
    assert line["phase"] == 0.25 and line["harmonic"] == 1.0


def test_seconds_duration_and_nonhomogeneous_events_are_deterministic() -> None:
    config = {
        "sampling_frequency": 100.0,
        "enable_glitches": True,
        "glitch_rate": 20.0,
        "glitch_duration_seconds": [0.1, 0.1],
        "event_process": "nonhomogeneous",
        "rate_profile": [0.0, 2.0],
        "boundary_policy": "reject",
        "overlap_policy": "reject",
    }
    a, ma = ArtifactInjector(config, seed=9).apply(np.zeros(1000), return_metadata=True)
    b, mb = ArtifactInjector(config, seed=9).apply(np.zeros(1000), return_metadata=True)
    assert np.array_equal(a, b)
    assert ma["glitches"] == mb["glitches"]
    assert all(event["duration"] == 10 for event in ma["glitches"]["glitches"])


def test_non_gaussian_families_and_diagnostics() -> None:
    student = NonGaussianNoiseGenerator(
        {"family": "student_t", "degrees_of_freedom": 6, "noise_power": 2.0}, seed=1
    ).generate(200_000)
    laplace = NonGaussianNoiseGenerator(
        {"family": "laplace", "noise_power": 1.0}, seed=2
    ).generate(200_000)
    diagnostics = NonGaussianNoiseGenerator.diagnostics(laplace)
    assert abs(np.var(student) - 2.0) < 0.08
    assert 2.5 < diagnostics["excess_kurtosis"] < 3.5
    assert diagnostics["tail_exceedance_probability"] > 6e-5


def test_non_gaussian_coloring_reports_marginal_contract() -> None:
    _, meta = NonGaussianNoiseGenerator(
        {
            "family": "gaussian_scale_mixture",
            "noise_power": 1.0,
            "psd_config": {**BASE, "noise_type": "pink"},
        },
        seed=3,
    ).generate(4096, return_metadata=True)
    assert meta["colored"] is True
    assert meta["marginal_preserved_after_coloring"] is False


def test_alpha_stable_metadata_states_variance_is_undefined() -> None:
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        _, meta = NonGaussianNoiseGenerator(
            {"family": "alpha_stable", "alpha": 1.7, "noise_power": 2.0}, seed=5
        ).generate(1000, return_metadata=True)
    assert caught
    assert "variance undefined" in meta["power_parameter_meaning"]


def test_shared_private_metadata_matches_closed_form_covariance() -> None:
    generator = MultiChannelNoiseGenerator(
        BASE,
        {"channel_gain_jitter": 0.0, "private_strength_range": [1.0, 1.0]},
        seed=10,
    )
    X, meta = generator.generate_shared_private(3, 100_000, 0.4, True)
    assert np.allclose(meta["implied_covariance"], meta["realized_covariance"], atol=0.02)
    assert np.allclose(np.diag(meta["implied_correlation"]), 1)
    assert meta["requested_mixing_strength"] == 0.4
    assert meta["per_realization_normalization"] is False


def _complex_csd(frequencies: np.ndarray) -> np.ndarray:
    phase = 0.6
    matrix = np.array([[1.0, 0.5 * np.exp(1j * phase)],
                       [0.5 * np.exp(-1j * phase), 1.5]], dtype=complex)
    csd = np.repeat(matrix[None], len(frequencies), axis=0)
    csd[0] = csd[0].real
    csd[-1] = csd[-1].real
    return csd


def test_csd_interpolation_batch_phase_and_coherence() -> None:
    N = 256
    source_f = np.linspace(0, 128, 17)
    generator = MultiChannelNoiseGenerator(BASE, seed=11)
    X, meta = generator.generate_from_csd(
        _complex_csd(source_f),
        N,
        target_frequencies=source_f,
        n_realizations=600,
        return_metadata=True,
    )
    coefficients = np.fft.rfft(X, axis=2)
    k = 80
    cross = np.mean(coefficients[:, 0, k] * coefficients[:, 1, k].conj())
    auto0 = np.mean(np.abs(coefficients[:, 0, k]) ** 2)
    auto1 = np.mean(np.abs(coefficients[:, 1, k]) ** 2)
    coherence = abs(cross) ** 2 / (auto0 * auto1)
    assert X.shape == (600, 2, N)
    assert abs(np.angle(cross) - 0.6) < 0.12
    assert abs(coherence - (0.5**2 / 1.5)) < 0.04
    assert meta["interpolation"] == "convex_linear_psd_preserving"


def test_csd_repair_and_low_rank_factor_input() -> None:
    N = 64
    generator = MultiChannelNoiseGenerator(BASE, seed=4)
    bad = np.repeat(np.array([[[1.0, 1.01], [1.01, 1.0]]]), N // 2 + 1, axis=0)
    _, meta = generator.generate_from_csd(
        bad, N, repair_policy="diagonal_loading", return_metadata=True
    )
    assert meta["maximum_regularization"] > 0
    factor = np.ones((N // 2 + 1, 2, 1))
    X, factor_meta = generator.generate_from_csd_factor(
        factor, N, n_realizations=2, return_metadata=True
    )
    assert X.shape == (2, 2, N)
    assert factor_meta["factor_rank"] == 1
