# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the modular noise simulator written for the ORACLE study.
# If you use this module in published work, please cite it: see CITATION.cff
# at the repository root.
from __future__ import annotations

import json
import tomllib
from pathlib import Path

import numpy as np
import pytest

from noise_module import (
    ArtifactConfig,
    ArtifactInjector,
    MultiChannelConfig,
    MultiChannelNoiseGenerator,
    NoiseGenerator,
    ReferenceDataset,
    StreamingNoiseGenerator,
    TemporalNoiseConfig,
    TemporalNoiseWrapper,
    ValidationConfig,
    calibrate_dataset,
    migrate_config,
    to_jsonable,
    validate_artifacts,
    validate_csd_ensemble,
    validate_local_nonstationarity,
    validate_stationary_gaussian,
)


BASE = {"noise_type": "white", "noise_power": 1.0, "sampling_frequency": 128.0}


def test_multichannel_relative_artifacts_use_each_real_baseline() -> None:
    X = np.random.default_rng(1).normal(size=(2, 4096))
    injector = ArtifactInjector(
        {
            "sampling_frequency": 128.0,
            "enable_lines": True,
            "lines": [{"freq": 8.0, "amp": 1.0, "phase": 0.0}],
            "amplitude_unit": "baseline_rms",
            "channel_amplitude_jitter": 0.0,
        },
        seed=3,
    )
    Y, metadata = injector.apply_multichannel(X, return_metadata=True)
    artifact_rms = np.std(Y - X, axis=1)
    assert np.all(artifact_rms > 0.5 * np.std(X, axis=1))
    assert np.allclose(metadata["artifact_only"], Y - X)


def test_piecewise_temporal_components_exactly_reconstruct_output() -> None:
    x = np.linspace(-2.0, 2.0, 1024)
    wrapper = TemporalNoiseWrapper(
        {
            "mode": "piecewise",
            "n_segments": 4,
            "crossfade_len": 32,
            "noise_power_scale_range": [0.25, 4.0],
            "variance_modulation": True,
            "variance_scale_range": [0.5, 2.0],
            "add_drift": True,
            "drift_rms": 0.1,
        },
        seed=4,
    )
    y, metadata = wrapper.apply(x, return_components=True)
    components = metadata["components"]
    reconstructed = (
        x * components["multiplicative_envelope"]
        + components["additive_drift"]
    )
    assert np.allclose(y, reconstructed, atol=1e-12)


def test_csd_repair_does_not_mutate_input() -> None:
    N = 64
    target = np.repeat(
        np.array([[[1.0, 1.1], [1.1, 1.0]]], dtype=complex),
        N // 2 + 1,
        axis=0,
    )
    original = target.copy()
    MultiChannelNoiseGenerator(BASE, seed=2).generate_from_csd(
        target, N, repair_policy="nearest_psd"
    )
    assert np.array_equal(target, original)


def test_factor_synthesis_is_direct_and_reports_storage_contract(monkeypatch) -> None:
    N = 64
    generator = MultiChannelNoiseGenerator(BASE, seed=2)

    def dense_path_must_not_run(*args, **kwargs):
        raise AssertionError("dense CSD path was called")

    monkeypatch.setattr(generator, "generate_from_csd", dense_path_must_not_run)
    factor = np.ones((N // 2 + 1, 3, 1), dtype=float) / np.sqrt(10.0)
    X, metadata = generator.generate_from_csd_factor(
        factor, N, n_realizations=4, return_metadata=True
    )
    assert X.shape == (4, 3, N)
    assert metadata["dense_csd_materialized"] is False
    assert np.allclose(X[:, 0], X[:, 1])


def test_evolutionary_overlap_add_covers_record_edges() -> None:
    wrapper = TemporalNoiseWrapper({"vary_noise_power": False}, seed=8)
    x, metadata = wrapper.generate_evolutionary(
        257,
        NoiseGenerator(BASE, seed=9),
        window_samples=64,
        hop_samples=32,
        return_metadata=True,
    )
    assert x[0] != 0.0
    assert metadata["minimum_overlap_weight"] > 0.0
    assert metadata["edge_policy"] == "pre_roll_and_post_roll"


def test_nonhomogeneous_process_allows_multiple_arrivals_per_sample() -> None:
    injector = ArtifactInjector(
        {
            "sampling_frequency": 100.0,
            "event_process": "nonhomogeneous",
            "rate_profile": [1.0],
        },
        seed=10,
    )
    starts = injector._sample_event_starts(100, 1000.0)
    assert len(starts) > 100
    assert np.max(np.bincount(starts, minlength=100)) > 1


def test_hawkes_metadata_reports_total_branching_expectation() -> None:
    injector = ArtifactInjector(
        {
            "sampling_frequency": 100.0,
            "enable_glitches": True,
            "glitch_rate": 2.0,
            "glitch_duration_samples": [4, 4],
            "event_process": "hawkes",
            "hawkes_branching_ratio": 0.5,
        },
        seed=2,
    )
    _, metadata = injector.apply(np.zeros(1000), return_metadata=True)
    section = metadata["glitches"]
    assert section["expected_total_count"] == 2 * section["immigrant_expected_count"]


def test_calibration_counts_exceedance_run_once() -> None:
    records = np.zeros((4, 100))
    records[:, 50] = 100.0
    preset = calibrate_dataset(
        ReferenceDataset(
            records,
            sampling_frequency=100.0,
            dataset_id="one-spike",
            two_dimensional_layout="records_samples",
        ),
        nperseg=50,
        bootstrap_samples=20,
        seed=1,
    )
    assert np.isclose(preset.statistics["artifact_rate_per_second"], 1.0)


def test_legacy_snr_name_migrates_to_explicit_energy_ratio() -> None:
    migrated = migrate_config(
        {"sampling_frequency": 10.0, "amplitude_unit": "snr"},
        kind="ArtifactConfig",
    )
    assert migrated["amplitude_unit"] == "rms_energy_ratio"
    injector = ArtifactInjector(migrated)
    assert injector.config["amplitude_unit"] == "rms_energy_ratio"


def test_single_record_csd_uses_segment_averaging() -> None:
    X = np.random.default_rng(3).normal(size=(2, 4096))
    diagnostics = MultiChannelNoiseGenerator.csd_diagnostics(X, 128.0)
    assert diagnostics["estimator"] == "welch_segment_average"
    assert np.mean(diagnostics["coherence"][:, 0, 1]) < 0.3


def test_implied_covariance_uses_actual_absolute_custom_psd(tmp_path) -> None:
    path = tmp_path / "absolute.npy"
    frequencies = np.linspace(0.0, 64.0, 65)
    np.save(path, np.vstack([frequencies, np.full_like(frequencies, 2.0)]))
    base = {
        "noise_type": str(path),
        "noise_power": 999.0,
        "sampling_frequency": 128.0,
        "custom_psd_scaling": "absolute",
    }
    generator = MultiChannelNoiseGenerator(base, seed=4)
    _, metadata = generator.generate_independent(2, 128, return_metadata=True)
    _, density = NoiseGenerator(base).build_psd_density(128)
    expected = np.sum(density[1:])
    assert np.allclose(np.diag(metadata["implied_covariance"]), expected)


def test_validation_reports_separate_gates_and_serializes() -> None:
    stationary = validate_stationary_gaussian(
        NoiseGenerator(BASE, seed=4),
        256,
        ValidationConfig(ensemble_size=128, bootstrap_samples=50),
    )
    expected = {
        "power",
        "psd",
        "fourier_real_normality",
        "fourier_imag_normality",
        "time_marginal_normality",
        "periodogram_exponential",
    }
    assert expected <= stationary.thresholds["gates"].keys()
    json.dumps(stationary.to_dict())

    local = validate_local_nonstationarity(
        np.random.default_rng(4).normal(size=1024), 128.0, 128
    )
    assert local.passed is None

    baseline = np.zeros(256)
    output, metadata = ArtifactInjector({}).apply(
        baseline, return_components=True
    )
    artifacts = validate_artifacts(baseline, output, metadata)
    assert {"component_reconstruction", "mask_identity"} <= artifacts.thresholds[
        "gates"
    ].keys()
    json.dumps(artifacts.to_dict())


def test_csd_validation_has_independent_scientific_gates() -> None:
    N = 64
    F = N // 2 + 1
    target = np.repeat(
        np.array([[[1.0, 0.25], [0.25, 1.0]]], dtype=complex),
        F,
        axis=0,
    )
    generator = MultiChannelNoiseGenerator(BASE, seed=5)
    X = generator.generate_from_csd(target, N, n_realizations=600)
    result = validate_csd_ensemble(X, target, 128.0)
    assert {
        "psd",
        "csd_magnitude",
        "phase",
        "coherence",
        "integrated_covariance",
    } == result.thresholds["gates"].keys()
    json.dumps(result.to_dict())


def test_calibration_requires_holdout_and_reports_multimetric_gates() -> None:
    with pytest.raises(ValueError, match="at least two records"):
        calibrate_dataset(
            ReferenceDataset(np.ones(32), 32.0, "single"),
            bootstrap_samples=20,
        )
    records = np.random.default_rng(6).normal(size=(12, 512))
    preset = calibrate_dataset(
        ReferenceDataset(
            records,
            128.0,
            "heldout",
            two_dimensional_layout="records_samples",
        ),
        nperseg=128,
        bootstrap_samples=20,
        seed=7,
    )
    assert {
        "power_coverage",
        "psd_median_log_ratio",
        "psd_interval_coverage",
        "tail_exceedance",
        "csd_relative_frobenius",
    } == preset.heldout_validation["gates"].keys()


def test_configuration_is_strict_and_integer_fields_are_exact() -> None:
    with pytest.raises(ValueError, match="typo"):
        NoiseGenerator({**BASE, "typo": 1})
    with pytest.raises(ValueError, match="integer"):
        TemporalNoiseConfig(n_segments=1.5)
    with pytest.raises(ValueError, match="integer"):
        MultiChannelConfig(n_channels=2.5)
    with pytest.raises(ValueError, match="integer"):
        ArtifactConfig(local_rms_window_samples=4.5)


def test_negative_or_nonfinite_rfft_power_is_rejected() -> None:
    generator = NoiseGenerator(BASE)
    with pytest.raises(ValueError, match="non-negative"):
        generator.sample_stationary_gaussian_from_rfft_power(
            np.array([0.0, -1.0, 1.0]), N=4
        )
    with pytest.raises(ValueError, match="finite"):
        generator.sample_stationary_gaussian_from_rfft_power(
            np.array([0.0, np.nan, 1.0]), N=4
        )


def test_streaming_partial_requests_use_one_fixed_grid() -> None:
    one_call = StreamingNoiseGenerator(
        {**BASE, "noise_type": "pink"}, chunk_samples=128, seed=12
    )
    split_calls = StreamingNoiseGenerator(
        {**BASE, "noise_type": "pink"}, chunk_samples=128, seed=12
    )
    expected = one_call.generate(300)
    observed = np.concatenate([split_calls.next_chunk(100) for _ in range(3)])
    assert np.array_equal(expected, observed)
    assert split_calls.contract["underlying_block_samples"] == 128
    assert split_calls.contract["blocks_generated"] == 3


def test_overlap_policy_has_only_implemented_semantics() -> None:
    assert ArtifactConfig(overlap_policy="allow").overlap_policy == "superpose"
    with pytest.raises(ValueError, match="no defined waveform semantics"):
        ArtifactConfig(overlap_policy="merge")


def test_reference_dataset_requires_explicit_2d_layout() -> None:
    with pytest.raises(ValueError, match="ambiguous"):
        ReferenceDataset(np.ones((2, 32)), 32.0, "ambiguous")
    channels = ReferenceDataset(
        np.ones((2, 32)),
        32.0,
        "channels",
        two_dimensional_layout="channels_samples",
    )
    assert channels.data.shape == (1, 2, 32)


def test_metadata_converter_and_packaging_have_single_version_source() -> None:
    output, metadata = TemporalNoiseWrapper(
        {"variance_modulation": True}, seed=3
    ).apply(np.ones(64), return_components=True)
    assert output.shape == (64,)
    json.dumps(to_jsonable(metadata))

    distribution_root = Path(__file__).resolve().parents[1]
    project = tomllib.loads(
        (distribution_root / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert "version" not in project["project"]
    assert "version" in project["project"]["dynamic"]
    assert project["tool"]["setuptools"]["dynamic"]["version"] == {
        "attr": "noise_module.__version__"
    }
