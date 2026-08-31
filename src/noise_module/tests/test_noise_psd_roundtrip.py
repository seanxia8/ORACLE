# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the modular noise simulator written for the ORACLE study.
# If you use this module in published work, please cite it: see CITATION.cff
# at the repository root.
"""Noise-module integration checks (formerly at the bottom of test_qp_simulator.py)."""

from __future__ import annotations

import numpy as np

from noise_module import MultiChannelNoiseGenerator, NoiseGenerator


def test_noise_generator_psd_density_integrates_to_noise_power() -> None:
    noise_power = 2.0
    sampling_frequency = 2_000.0
    n_samples = 2_048

    for noise_type in ["white", "pink", "blue", "violet", "brownian"]:
        generator = NoiseGenerator(
            {
                "noise_type": noise_type,
                "noise_power": noise_power,
                "sampling_frequency": sampling_frequency,
            },
            seed=1,
        )
        _, density = generator.build_psd_density(n_samples)
        recovered = float(np.sum(density) * sampling_frequency / n_samples)
        assert np.isclose(recovered, noise_power)


def test_build_psd_alias_returns_rfft_power_not_density() -> None:
    config = {
        "noise_type": "white",
        "noise_power": 1.0,
        "sampling_frequency": 1_000.0,
    }
    generator = NoiseGenerator(config, seed=2)

    freqs_density, density = generator.build_psd_density(128)
    freqs_power, rfft_power = generator.build_psd(128)
    expected_power = NoiseGenerator.psd_density_to_rfft_power(density, 1_000.0, 128)

    assert np.allclose(freqs_density, freqs_power)
    assert np.allclose(rfft_power, expected_power)
    assert not np.allclose(rfft_power, density)


def test_multichannel_variance_normalization_is_opt_in() -> None:
    generator = MultiChannelNoiseGenerator(
        {
            "noise_type": "white",
            "noise_power": 1.0,
            "sampling_frequency": 2_000.0,
        },
        seed=3,
    )

    assert generator.config["normalize_channel_variance"] is False
