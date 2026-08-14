from __future__ import annotations

import numpy as np

from noise_module import NoiseGenerator, TemporalNoiseWrapper


def test_temporal_wrapper_generates_piecewise_trace_and_metadata() -> None:
    base_config = {
        "noise_type": "pink",
        "noise_power": 1.0,
        "sampling_frequency": 2000.0,
    }
    temporal_config = {
        "mode": "piecewise",
        "n_segments": 4,
        "crossfade_len": 32,
        "vary_noise_power": True,
        "noise_power_scale_range": [0.5, 1.5],
        "add_drift": True,
        "drift_sigma": 0.05,
        "drift_n_knots": 5,
    }

    base = NoiseGenerator(base_config, seed=10)
    wrapper = TemporalNoiseWrapper(temporal_config, seed=11)
    trace, meta = wrapper.generate_piecewise(2048, base_generator=base, return_metadata=True)

    assert trace.shape == (2048,)
    assert len(meta["segments"]) == 4
    assert meta["crossfade_len"] == 32
    powers = [segment["noise_power"] for segment in meta["segments"]]
    assert max(powers) > min(powers)


def test_temporal_wrapper_adds_smooth_drift() -> None:
    wrapper = TemporalNoiseWrapper({"add_drift": True, "drift_sigma": 0.1, "drift_n_knots": 6}, seed=21)
    drift = wrapper.generate_drift(1024)

    assert drift.shape == (1024,)
    assert abs(np.mean(drift)) < 1e-10
    assert np.std(np.diff(drift)) < np.std(drift)
