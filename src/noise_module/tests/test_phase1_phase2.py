# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the modular noise simulator written for the ORACLE study.
# If you use this module in published work, please cite it: see CITATION.cff
# at the repository root.
from __future__ import annotations

import json

import numpy as np

from noise_module import (
    ArtifactConfig,
    ArtifactInjector,
    BandLimited,
    Line,
    Lorentzian,
    MultiChannelConfig,
    MultiChannelNoiseGenerator,
    NoiseConfig,
    NoiseGenerator,
    PowerLaw,
    RollOff,
    TemporalNoiseConfig,
    TemporalNoiseWrapper,
    White,
    migrate_config,
)


def _config(**updates):
    result = {
        "noise_type": "white",
        "noise_power": 1.0,
        "sampling_frequency": 1024.0,
    }
    result.update(updates)
    return result


def test_typed_configs_work_as_public_inputs() -> None:
    base = NoiseConfig(**_config())
    temporal = TemporalNoiseConfig(mode="piecewise", n_segments=2)
    artifact = ArtifactConfig(sampling_frequency=1024.0)
    multichannel = MultiChannelConfig(n_channels=2)

    assert NoiseGenerator(base, seed=1).generate_noise(32).shape == (32,)
    assert TemporalNoiseWrapper(temporal, seed=1).apply(np.zeros(32)).shape == (32,)
    assert ArtifactInjector(artifact, seed=1).apply(np.zeros(32)).shape == (32,)
    assert MultiChannelNoiseGenerator(base, multichannel, seed=1).generate(32).shape == (
        2,
        32,
    )


def test_strict_mode_rejects_unknown_fields() -> None:
    with np.testing.assert_raises_regex(ValueError, "Unknown NoiseConfig"):
        NoiseGenerator({**_config(), "typo_power": 3}, strict_config=True)


def test_configuration_migration_is_pure_and_json_round_trips() -> None:
    legacy = {
        **_config(noise_power=2.5),
        "custom_psd_scaling": "multiply",
    }
    migrated = migrate_config(legacy)
    assert legacy["custom_psd_scaling"] == "multiply"
    assert migrated["custom_psd_scaling"] == "scale"
    assert migrated["psd_scale"] == 2.5

    resolved = NoiseConfig.from_mapping(migrated)
    reloaded = NoiseConfig.from_mapping(json.loads(json.dumps(resolved.to_dict())))
    a = NoiseGenerator(resolved, seed=44).generate_noise(256)
    b = NoiseGenerator(reloaded, seed=44).generate_noise(256)
    assert np.array_equal(a, b)


def test_power_metadata_and_deterministic_mean_are_explicit() -> None:
    generator = NoiseGenerator(
        _config(noise_power=2.0, deterministic_mean=3.0), seed=3
    )
    _, density, metadata = generator.build_psd_density(1024, return_metadata=True)
    trace, trace_metadata = generator.generate_noise(1024, return_metadata=True)

    assert density[0] == 0.0
    assert np.isclose(metadata["expected_variance"], 2.0)
    assert np.isclose(metadata["expected_mean_square"], 11.0)
    assert metadata["stochastic_dc_power"] == 0.0
    assert metadata["integration_rule"]
    assert abs(np.mean(trace) - 3.0) < 1e-12
    assert np.isclose(trace_metadata["deterministic_mean"], 3.0)


def test_power_acceptance_uses_monte_carlo_standard_error() -> None:
    realizations = 1000
    N = 256
    generator = NoiseGenerator(_config(), seed=902)
    observed = np.mean([np.var(generator.generate_noise(N)) for _ in range(realizations)])
    # For zero-mean Gaussian white noise with one fitted/removed mean, the
    # relative standard error of the ensemble variance mean is sqrt(2/(R(N-1))).
    standard_error = np.sqrt(2.0 / (realizations * (N - 1)))
    assert abs(observed - 1.0) < 5.0 * standard_error


def test_parseval_and_odd_even_endpoint_contracts() -> None:
    for N in (255, 256):
        generator = NoiseGenerator(_config(), seed=N)
        trace = generator.generate_noise(N)
        coefficients = np.fft.rfft(trace)
        endpoint_power = abs(coefficients[0]) ** 2
        interior = coefficients[1:]
        if N % 2 == 0:
            endpoint_power += abs(coefficients[-1]) ** 2
            interior = coefficients[1:-1]
        frequency_mean_square = (
            endpoint_power + 2.0 * np.sum(np.abs(interior) ** 2)
        ) / N**2
        assert np.isclose(np.mean(trace**2), frequency_mean_square, atol=1e-13)
        assert np.isrealobj(trace)


def test_n1_and_zero_power_contracts() -> None:
    with np.testing.assert_raises_regex(ValueError, "N=1"):
        NoiseGenerator(_config()).generate_noise(1)
    output = NoiseGenerator(
        _config(noise_power=0.0, deterministic_mean=4.0), seed=2
    ).generate_noise(1)
    assert np.array_equal(output, [4.0])


def test_custom_psd_out_of_band_policies_and_loglog_interpolation(tmp_path) -> None:
    path = tmp_path / "limited.npy"
    source_f = np.array([1.0, 2.0, 4.0, 8.0])
    source_p = source_f**-2
    np.save(path, np.vstack([source_f, source_p]))

    with np.testing.assert_raises_regex(ValueError, "exceeds"):
        NoiseGenerator(
            _config(
                noise_type=str(path),
                sampling_frequency=16.0,
                custom_out_of_band="error",
                custom_interpolation="loglog",
            )
        ).build_psd_density(16)

    zero = NoiseGenerator(
        _config(
            noise_type=str(path),
            sampling_frequency=16.0,
            custom_out_of_band="zero",
            custom_interpolation="loglog",
        )
    )
    f, p = zero.build_psd_density(16)
    assert np.all(p[f > 8.0] == 0.0)

    extrapolated = NoiseGenerator(
        _config(
            noise_type=str(path),
            sampling_frequency=16.0,
            custom_out_of_band="power_law",
            custom_interpolation="loglog",
            power_definition="mean_square",
        )
    )
    f, p, metadata = extrapolated.build_psd_density(16, return_metadata=True)
    assert np.isclose(p[np.where(f == 4.0)[0][0]], 4.0**-2)
    assert metadata["config"]["custom_out_of_band"] == "power_law"
    assert metadata["custom_psd_source"]["frequency_range_hz"] == [1.0, 8.0]
    assert metadata["custom_psd_source"]["interpolation"] == "loglog"


def test_custom_psd_scale_uses_dedicated_psd_scale(tmp_path) -> None:
    path = tmp_path / "psd.npy"
    np.save(path, np.vstack([np.arange(9.0), np.ones(9)]))
    absolute = NoiseGenerator(
        _config(noise_type=str(path), sampling_frequency=16.0)
    )
    scaled = NoiseGenerator(
        _config(
            noise_type=str(path),
            sampling_frequency=16.0,
            custom_psd_scaling="scale",
            psd_scale=3.0,
        )
    )
    _, a = absolute.build_psd_density(16)
    _, b = scaled.build_psd_density(16)
    assert np.allclose(b, 3.0 * a)


def test_power_law_cutoffs_are_applied_before_normalization() -> None:
    generator = NoiseGenerator(
        _config(
            psd_exponent=-1.3,
            low_frequency_cutoff=100.0,
            high_frequency_cutoff=300.0,
        )
    )
    f, p = generator.build_psd_density(1024)
    assert np.all(p[f < 100.0] == 0.0)
    assert np.all(p[f > 300.0] == 0.0)
    assert np.isclose(np.sum(p) * (1024.0 / 1024), 1.0)


def test_composable_spectrum_reports_component_power() -> None:
    spectrum = (
        White(scale=0.2, normalization="power", name="floor")
        + PowerLaw(scale=0.3, normalization="power", exponent=-1.0, name="pink")
        + Lorentzian(
            scale=0.1,
            normalization="power",
            center_hz=150.0,
            half_width_hz=10.0,
            name="resonance",
        )
        + BandLimited(
            scale=0.1,
            normalization="power",
            low_hz=200.0,
            high_hz=250.0,
            name="band",
        )
        + RollOff(
            scale=0.1,
            normalization="power",
            corner_hz=300.0,
            name="rolloff",
        )
        + Line(
            scale=0.2,
            normalization="power",
            frequency_hz=60.0,
            name="line",
        )
    )
    generator = NoiseGenerator(
        _config(noise_type=spectrum, composite_psd_scaling="absolute"),
        seed=2,
    )
    f, p, metadata = generator.build_psd_density(1024, return_metadata=True)
    contributions = metadata["component_contributions"]

    assert np.isclose(np.sum(p) * (f[1] - f[0]), 1.0)
    assert len(contributions) == 6
    assert np.isclose(sum(item["integrated_power"] for item in contributions), 1.0)
    assert {item["name"] for item in contributions} == {
        "floor",
        "pink",
        "resonance",
        "band",
        "rolloff",
        "line",
    }


def test_serializable_composite_config_normalizes_total_power() -> None:
    generator = NoiseGenerator(
        _config(
            noise_type="composite",
            noise_power=4.0,
            components=[
                {"type": "white", "scale": 1.0},
                {
                    "type": "lorentzian",
                    "scale": 2.0,
                    "center_hz": 100.0,
                    "half_width_hz": 5.0,
                },
            ],
        )
    )
    _, p, metadata = generator.build_psd_density(1024, return_metadata=True)
    assert np.isclose(np.sum(p), 4.0)
    assert all(
        "integrated_power_after_global_scaling" in item
        for item in metadata["component_contributions"]
    )
    json.dumps(metadata["config"])


def test_ensemble_periodogram_recovers_composite_resonance() -> None:
    N = 1024
    generator = NoiseGenerator(
        _config(
            noise_type="composite",
            components=[
                {"type": "white", "scale": 0.2, "normalization": "power"},
                {
                    "type": "lorentzian",
                    "scale": 0.8,
                    "normalization": "power",
                    "center_hz": 160.0,
                    "half_width_hz": 5.0,
                },
            ],
            composite_psd_scaling="absolute",
        ),
        seed=87,
    )
    frequencies, target = generator.build_psd_density(N)
    estimates = []
    for _ in range(160):
        coefficients = np.fft.rfft(generator.generate_noise(N))
        density = np.abs(coefficients) ** 2 / (generator.sampling_frequency * N)
        density[1:-1] *= 2.0
        estimates.append(density)
    observed = np.mean(estimates, axis=0)
    resonance = (frequencies >= 150.0) & (frequencies <= 170.0)
    background = (frequencies >= 350.0) & (frequencies <= 450.0)

    assert np.mean(observed[resonance]) > 10.0 * np.mean(observed[background])
    assert np.mean(np.abs(observed[1:-1] / target[1:-1] - 1.0)) < 0.08


def test_circulant_embedding_recovers_requested_covariance() -> None:
    first_row = 0.8 ** np.arange(24)
    generator = NoiseGenerator(_config(), seed=12)
    samples = np.vstack(
        [generator.sample_from_covariance(first_row, method="circulant") for _ in range(3000)]
    )
    empirical = np.mean(samples[:, :-1] * samples[:, 1:], axis=1).mean()
    _, metadata = generator.sample_from_covariance(
        first_row, method="circulant", return_metadata=True
    )
    assert abs(np.var(samples) - 1.0) < 0.04
    assert abs(empirical - 0.8) < 0.04
    assert metadata["method"] == "circulant"
    assert metadata["exact_finite_covariance"] is True
    assert metadata["maximum_covariance_error"] < 1e-12


def test_dense_covariance_fallback_and_indefinite_rejection() -> None:
    generator = NoiseGenerator(_config(), seed=2)
    _, metadata = generator.sample_from_covariance(
        np.array([1.0, 0.8, 0.3]), method="auto", return_metadata=True
    )
    assert metadata["method"] == "dense"
    assert metadata["maximum_covariance_error"] < 1e-12
    with np.testing.assert_raises_regex(ValueError, "positive semidefinite"):
        generator.sample_from_covariance(np.array([1.0, 2.0]), method="dense")
