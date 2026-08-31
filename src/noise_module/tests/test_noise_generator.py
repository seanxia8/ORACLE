# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the modular noise simulator written for the ORACLE study.
# If you use this module in published work, please cite it: see CITATION.cff
# at the repository root.
from __future__ import annotations

import numpy as np

from noise_module import NoiseGenerator


def test_noise_generator_is_reproducible_with_seed() -> None:
    config = {
        "noise_type": "pink",
        "noise_power": 1.0,
        "sampling_frequency": 2000.0,
    }

    a = NoiseGenerator(config, seed=123).generate_noise(1024)
    b = NoiseGenerator(config, seed=123).generate_noise(1024)
    c = NoiseGenerator(config, seed=124).generate_noise(1024)

    assert np.allclose(a, b)
    assert not np.allclose(a, c)


def test_noise_generator_build_psd_and_metadata_are_consistent() -> None:
    config = {
        "noise_type": "white",
        "noise_power": 0.5,
        "sampling_frequency": 1000.0,
    }
    generator = NoiseGenerator(config, seed=42)

    freqs, psd, psd_meta = generator.build_psd(2048, return_metadata=True)
    trace, trace_meta = generator.generate_noise(2048, return_metadata=True)

    assert freqs.shape == psd.shape
    assert freqs.ndim == 1
    assert len(freqs) == 1025
    assert np.all(psd >= 0.0)
    assert trace.shape == (2048,)
    assert psd_meta["noise_type"] == "white"
    assert trace_meta["n_samples"] == 2048
    assert "frequencies" in trace_meta
