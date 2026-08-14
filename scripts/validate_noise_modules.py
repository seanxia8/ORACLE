"""Quick validation script for the modular noise generation stack."""

from __future__ import annotations

import argparse

import numpy as np

from noise_module import (
    ArtifactInjector,
    MultiChannelNoiseGenerator,
    NoiseGenerator,
    TemporalNoiseWrapper,
    validate_reference_noise,
)
from noise_module.utils import mean_offdiag_corrcoef


def build_example_configs(fs: float) -> tuple[dict, dict, dict, dict]:
    base_config = {
        "noise_type": "pink",
        "noise_power": 1.0,
        "sampling_frequency": fs,
    }
    temporal_config = {
        "mode": "piecewise",
        "n_segments": 4,
        "crossfade_len": 128,
        "vary_noise_power": True,
        "noise_power_scale_range": [0.7, 1.3],
        "add_drift": True,
        "drift_type": "spline",
        "drift_sigma": 0.08,
        "drift_n_knots": 6,
    }
    artifact_config = {
        "sampling_frequency": fs,
        "enable_lines": True,
        "lines": [
            {"freq": 50.0, "amp": 0.03, "phase": "random", "harmonics": [1, 2]},
            {"freq": 120.0, "amp": [0.01, 0.02], "phase": "random", "harmonics": [1]},
        ],
        "enable_glitches": True,
        "glitch_rate": 2.0,
        "glitch_amp_range": [0.1, 0.25],
        "glitch_templates": ["impulse", "exp_decay", "damped_sine", "ringing"],
        "glitch_duration_samples": [32, 160],
        "enable_sparse_impulses": True,
        "impulse_probability": 2e-3,
        "impulse_sigma": 0.2,
    }
    multichannel_config = {
        "mode": "shared_private",
        "n_channels": 8,
        "corr_strength": 0.35,
        "channel_gain_jitter": 0.05,
        "normalize_channel_variance": True,
    }
    return base_config, temporal_config, artifact_config, multichannel_config


def dominant_frequency(x: np.ndarray, fs: float) -> float:
    spectrum = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    if len(freqs) <= 1:
        return 0.0
    idx = int(np.argmax(spectrum[1:]) + 1)
    return float(freqs[idx])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=4096, help="Trace length for validation.")
    parser.add_argument("--seed", type=int, default=1234, help="Master seed for reproducibility.")
    parser.add_argument("--sampling-frequency", type=float, default=2000.0, help="Sampling rate in Hz.")
    parser.add_argument(
        "--al2o3-athermal",
        action="store_true",
        help="Also run the Al2O3/Al reference ensemble PSD regression.",
    )
    args = parser.parse_args()

    base_config, temporal_config, artifact_config, multichannel_config = build_example_configs(
        args.sampling_frequency
    )

    rng = np.random.default_rng(args.seed)
    base = NoiseGenerator(base_config, rng=np.random.default_rng(rng.integers(0, 2**32)))
    temporal = TemporalNoiseWrapper(temporal_config, rng=np.random.default_rng(rng.integers(0, 2**32)))
    artifact = ArtifactInjector(artifact_config, rng=np.random.default_rng(rng.integers(0, 2**32)))
    multichannel = MultiChannelNoiseGenerator(
        base_config,
        config=multichannel_config,
        rng=np.random.default_rng(rng.integers(0, 2**32)),
    )

    baseline, base_meta = base.generate_noise(args.samples, return_metadata=True)
    temporal_trace, temporal_meta = temporal.apply(
        baseline,
        return_metadata=True,
    )
    artifact_trace, artifact_meta = artifact.apply(temporal_trace, return_metadata=True)

    independent, _ = multichannel.generate_independent(
        multichannel_config["n_channels"],
        args.samples,
        return_metadata=True,
    )
    shared, shared_meta = multichannel.generate_shared_private(
        multichannel_config["n_channels"],
        args.samples,
        corr_strength=multichannel_config["corr_strength"],
        return_metadata=True,
    )

    segment_stds = []
    for segment in temporal_meta.get("piecewise", {}).get("segments", []):
        start = segment["start"]
        end = segment["end"]
        segment_stds.append(float(np.std(temporal_trace[start:end])))

    print("Base generator")
    print(f"  noise_type: {base_meta['noise_type']}")
    print(f"  variance:   {np.var(baseline):.4f}")
    print(f"  mean:       {np.mean(baseline):.4e}")

    print("Temporal wrapper")
    print(f"  segment stds: {np.array2string(np.array(segment_stds), precision=4)}")
    print(f"  drift std:    {temporal_meta.get('drift_std', 0.0):.4f}")

    print("Artifact injector")
    print(f"  dominant frequency after artifacts: {dominant_frequency(artifact_trace, args.sampling_frequency):.2f} Hz")
    print(f"  glitch count: {artifact_meta.get('glitches', {}).get('count', 0)}")
    print(f"  impulse count: {artifact_meta.get('sparse_impulses', {}).get('count', 0)}")

    print("Multi-channel generator")
    print(f"  independent mean offdiag corr: {mean_offdiag_corrcoef(independent):.4f}")
    print(f"  shared/private mean offdiag corr: {shared_meta['mean_offdiag_corr']:.4f}")

    if args.al2o3_athermal:
        diagnostics = validate_reference_noise(
            n_samples=args.samples,
            sampling_frequency=1_000_000.0,
            seed=args.seed,
        )
        print("Al2O3/Al athermal reference")
        print(f"  median ASD ratio: {diagnostics['median_asd_ratio']:.4f}")
        print(
            "  RMS fractional ASD error: "
            f"{diagnostics['rms_fractional_asd_error']:.4f}"
        )


if __name__ == "__main__":
    main()
