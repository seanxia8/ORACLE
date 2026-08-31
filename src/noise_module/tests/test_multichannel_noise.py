# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the modular noise simulator written for the ORACLE study.
# If you use this module in published work, please cite it: see CITATION.cff
# at the repository root.
from __future__ import annotations

import numpy as np

from noise_module import MultiChannelNoiseGenerator


def test_multichannel_generator_independent_shape_and_low_correlation() -> None:
    base_config = {
        "noise_type": "pink",
        "noise_power": 1.0,
        "sampling_frequency": 2000.0,
    }
    generator = MultiChannelNoiseGenerator(base_config, seed=30)
    X, meta = generator.generate_independent(6, 4096, return_metadata=True)

    assert X.shape == (6, 4096)
    assert abs(meta["mean_offdiag_corr"]) < 0.1


def test_shared_private_mode_is_more_correlated_than_independent() -> None:
    base_config = {
        "noise_type": "pink",
        "noise_power": 1.0,
        "sampling_frequency": 2000.0,
    }
    generator = MultiChannelNoiseGenerator(
        base_config,
        config={"corr_strength": 0.45, "normalize_channel_variance": True},
        seed=31,
    )

    independent, independent_meta = generator.generate_independent(8, 4096, return_metadata=True)
    shared, shared_meta = generator.generate_shared_private(
        8,
        4096,
        corr_strength=0.45,
        return_metadata=True,
    )

    assert independent.shape == shared.shape == (8, 4096)
    assert shared_meta["mean_offdiag_corr"] > independent_meta["mean_offdiag_corr"] + 0.15
