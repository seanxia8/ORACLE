# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the modular noise simulator written for the ORACLE study.
# If you use this module in published work, please cite it: see CITATION.cff
# at the repository root.
from __future__ import annotations

import numpy as np

from noise_module import ArtifactInjector


def test_artifact_injector_adds_line_at_configured_frequency() -> None:
    fs = 512.0
    x = np.zeros(2048, dtype=float)
    injector = ArtifactInjector(
        {
            "sampling_frequency": fs,
            "enable_lines": True,
            "lines": [{"freq": 64.0, "amp": 1.0, "phase": 0.0, "harmonics": [1]}],
        },
        seed=3,
    )

    y, meta = injector.apply(x, return_metadata=True)
    freqs = np.fft.rfftfreq(len(y), d=1.0 / fs)
    spectrum = np.abs(np.fft.rfft(y))
    peak_freq = freqs[int(np.argmax(spectrum[1:]) + 1)]

    assert meta["lines"]["count"] == 1
    assert np.isclose(peak_freq, 64.0, atol=0.5)


def test_artifact_injector_sparse_impulses_report_count() -> None:
    x = np.zeros(4096, dtype=float)
    injector = ArtifactInjector(
        {
            "enable_sparse_impulses": True,
            "impulse_probability": 0.01,
            "impulse_sigma": 0.5,
        },
        seed=9,
    )

    y, meta = injector.apply(x, return_metadata=True)

    assert meta["sparse_impulses"]["count"] > 0
    assert np.count_nonzero(y) == meta["sparse_impulses"]["count"]
