# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the modular noise simulator written for the ORACLE study.
# If you use this module in published work, please cite it: see CITATION.cff
# at the repository root.
from __future__ import annotations

import json

import numpy as np
import pytest

from noise_module import (
    ArtifactInjector,
    MultiChannelNoiseGenerator,
    NoiseGenerator,
    ReferenceDataset,
    StreamingNoiseGenerator,
    TemporalNoiseWrapper,
    ValidationConfig,
    benchmark_generation,
    bootstrap_interval,
    calibrate_dataset,
    validate_artifacts,
    validate_csd_ensemble,
    validate_local_nonstationarity,
    validate_stationary_gaussian,
)


BASE = {"noise_type": "white", "noise_power": 1.0, "sampling_frequency": 256.0}


@pytest.mark.statistical
@pytest.mark.parametrize("color", ["white", "pink", "brownian", "blue", "violet"])
@pytest.mark.parametrize("N", [255, 256])
def test_stationary_validation_matrix_all_colors_odd_even(color, N) -> None:
    result = validate_stationary_gaussian(
        NoiseGenerator({**BASE, "noise_type": color}, seed=10),
        N,
        ValidationConfig(
            ensemble_size=96,
            bootstrap_samples=100,
            relative_psd_tolerance=0.35,
        ),
    )
    assert result.statistics["dc_is_real"]
    assert result.statistics["nyquist_is_real"]
    assert result.statistics["mean_relative_psd_error"] < 0.35
    assert abs(result.statistics["fourier_skewness"]) < 0.08
    assert abs(result.statistics["fourier_excess_kurtosis"]) < 0.12


@pytest.mark.statistical
def test_confidence_utilities_record_thresholds_and_statistics() -> None:
    values = np.random.default_rng(1).normal(size=500)
    low, high = bootstrap_interval(values, n_resamples=200)
    result = validate_stationary_gaussian(
        NoiseGenerator(BASE, seed=2),
        512,
        ValidationConfig(ensemble_size=128, bootstrap_samples=100),
    )
    assert low < np.mean(values) < high
    assert result.thresholds
    json.dumps(result.to_dict())


@pytest.mark.integration
def test_nonstationary_artifact_and_component_validation() -> None:
    base = NoiseGenerator(BASE, seed=1).generate_noise(4096)
    temporal = TemporalNoiseWrapper(
        {
            "variance_modulation": True,
            "variance_scale_range": [0.5, 2.0],
            "variance_n_knots": 4,
        },
        seed=2,
    ).apply(base)
    injector = ArtifactInjector(
        {
            "sampling_frequency": 256,
            "enable_glitches": True,
            "glitch_rate": 3,
            "glitch_duration_seconds": [0.02, 0.05],
        },
        seed=3,
    )
    output, metadata = injector.apply(temporal, return_components=True)
    local = validate_local_nonstationarity(temporal, 256, 512)
    artifacts = validate_artifacts(temporal, output, metadata)
    assert len(local.statistics["local_variance"]) == 8
    assert np.ptp(local.statistics["local_variance"]) > 0
    assert artifacts.passed


@pytest.mark.statistical
def test_csd_validation_dense_complex_singular_and_repaired() -> None:
    N = 128
    F = N // 2 + 1
    phase = 0.4
    matrix = np.array(
        [[1.0, 0.4 * np.exp(1j * phase)], [0.4 * np.exp(-1j * phase), 1.0]]
    )
    target = np.repeat(matrix[None], F, axis=0)
    target[0] = target[0].real
    target[-1] = target[-1].real
    generator = MultiChannelNoiseGenerator(BASE, seed=4)
    X = generator.generate_from_csd(target, N, n_realizations=1000)
    result = validate_csd_ensemble(X, target, 256)
    assert result.statistics["maximum_covariance_error"] < 2.0
    singular_factor = np.ones((F, 2, 1)) / np.sqrt(F * 2)
    singular = generator.generate_from_csd_factor(singular_factor, N)
    assert np.allclose(singular[0], singular[1])
    bad = target.copy()
    bad[10] = [[1, 1.01], [1.01, 1]]
    _, metadata = generator.generate_from_csd(
        bad, N, repair_policy="nearest_psd", return_metadata=True
    )
    assert metadata["regularization"]


@pytest.mark.calibration
def test_reference_dataset_and_preset_roundtrip(tmp_path) -> None:
    generator = NoiseGenerator({**BASE, "noise_type": "pink"}, seed=5)
    records = generator.generate_ensemble(24, 2048)
    dataset = ReferenceDataset(
        records,
        256,
        "synthetic-pink-v1",
        two_dimensional_layout="records_samples",
        units="ADC",
        acquisition={"instrument": "synthetic"},
        preprocessing=["mean removal"],
    )
    dataset_path = tmp_path / "reference.npz"
    dataset.save(dataset_path)
    loaded = ReferenceDataset.load(dataset_path)
    assert np.array_equal(dataset.data, loaded.data)

    preset = calibrate_dataset(
        loaded, nperseg=256, bootstrap_samples=60, seed=7
    )
    preset_path = tmp_path / "preset.json"
    preset.save(preset_path)
    restored = type(preset).load(preset_path)
    regenerated = restored.generate(512, seed=8)
    sampled = restored.sample_parameters(seed=9)
    assert regenerated.shape == (512,)
    assert sampled["psd"].shape == restored.psd.shape
    assert restored.provenance["dataset_id"] == "synthetic-pink-v1"
    assert restored.not_modeled
    assert restored.heldout_validation["passed"]


def test_vectorized_ensemble_streaming_and_benchmark_contracts() -> None:
    ensemble, metadata = NoiseGenerator(BASE, seed=1).generate_ensemble(
        10, 257, return_metadata=True
    )
    assert ensemble.shape == (10, 257)
    assert metadata["vectorized"]

    a = StreamingNoiseGenerator(BASE, chunk_samples=128, seed=2)
    b = StreamingNoiseGenerator(BASE, chunk_samples=128, seed=2)
    assert np.array_equal(a.generate(300), b.generate(300))
    assert a.contract["one_shot_equality"] is False
    metrics = benchmark_generation(
        BASE, realizations=4, channels=2, samples=512,
    )
    assert metrics["passed"]
    assert metrics["peak_memory_bytes"] > 0


def test_seeded_child_streams_are_reproducible_and_independent() -> None:
    config = {"mode": "independent", "n_channels": 3}
    a = MultiChannelNoiseGenerator(BASE, config, seed=44).generate(1024)
    b = MultiChannelNoiseGenerator(BASE, config, seed=44).generate(1024)
    assert np.array_equal(a, b)
    assert not np.array_equal(a[0], a[1])
    assert abs(np.corrcoef(a)[0, 1]) < 0.15
