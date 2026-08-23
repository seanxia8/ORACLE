"""End-to-end inference contract tests (audit B1/B2/C10 regressions).

Builds a tiny synthetic release-layout HDF5 pair, runs ``infer.infer`` with a
small untrained model, and asserts the written product's coordinate frame,
dtype, length (tail included) and two-channel layout. These are exactly the
failure modes that previously produced a silently garbage benchmark score:
int8 saturation from the un-removed ``+128`` shift, and an uninitialised tail.
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
import torch

from tidmad.config import (
    FROZEN,
    TidmadDataConfig,
    TidmadModelConfig,
    TidmadRunConfig,
    TidmadSTFTConfig,
    TidmadTrainConfig,
)


def _tiny_run(contract_path: str) -> TidmadRunConfig:
    stft = TidmadSTFTConfig(n_fft=64, hop_length=32, win_length=64, n_bands_used=16)
    # window_samples chosen so M = (L-64)//32+1 = 8 frames, patch_len=4 -> 2 patches
    data = TidmadDataConfig(contract_path=contract_path, window_samples=288)
    model = TidmadModelConfig(d_model=16, d_ff=32, n_head=2, patch_len=4)
    train = TidmadTrainConfig(num_steps=1, device_batch_size=1)
    return TidmadRunConfig(stft, model, train, data)


@pytest.fixture()
def synthetic_release(tmp_path):
    """One validation file whose length is NOT a multiple of the window."""
    n_windows = 3
    run = _tiny_run("unused")
    n = run.data.window_samples
    tail = 100  # deliberately non-zero remainder
    n_samples = n_windows * n + tail
    rng = np.random.default_rng(7)
    squid = rng.integers(-40, 40, size=n_samples).astype(np.int8)
    ref = rng.integers(-40, 40, size=n_samples).astype(np.int8)
    path = tmp_path / "abra_validation_0001.h5"
    with h5py.File(path, "w") as f:
        for channel, values in (("channel0001", squid), ("channel0002", ref)):
            ds = f.create_dataset(f"timeseries/{channel}/timeseries", data=values)
            group = f[f"timeseries/{channel}"]
            group.attrs["voltage_range_mV"] = 1000.0
            group.attrs["sampling_frequency"] = 10_000_000.0
    contract = {
        "verified": True,
        "squid_dataset": "timeseries/channel0001/timeseries",
        "reference_dataset": "timeseries/channel0002/timeseries",
        "dtype": "int8",
        "sampling_frequency_hz": 10_000_000.0,
        "max_samples": int(n_samples),
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract))
    return tmp_path, contract_path, n_samples, squid


def test_infer_writes_valid_unsaturated_two_channel_file(synthetic_release, tmp_path, monkeypatch):
    data_dir, contract_path, n_samples, squid = synthetic_release
    run = _tiny_run(str(contract_path))

    from tidmad import infer as infer_mod
    from tidmad.train import build_model

    # Bypass checkpoint loading: an untrained model is enough to test the
    # writer's coordinate frame, dtype, tail and layout contracts.
    monkeypatch.setattr(infer_mod, "load_run_config", lambda _path: run)
    monkeypatch.setattr(infer_mod, "load_checkpoint", lambda model, path, device: None)

    out_dir = tmp_path / "out"
    torch.manual_seed(0)
    infer_mod.infer(
        config="ignored",
        checkpoint="ignored",
        data_dir=data_dir,
        out_dir=out_dir,
        model_tag="testarm",
        device="cpu",
    )

    out_path = out_dir / "abra_validation_denoised_testarm_0001.h5"
    assert out_path.exists()
    with h5py.File(out_path, "r") as f:
        den = np.asarray(f["timeseries/channel0001/timeseries"])
        ref = np.asarray(f["timeseries/channel0002/timeseries"])
    # Layout and tail: both channels full-length, including the remainder.
    assert den.shape[0] == n_samples
    assert ref.shape[0] == n_samples
    # Frame: values must live in the release's signed frame, not saturate at
    # +127. An untrained (zero-init head) model reproduces ~the per-row mean of
    # the input, so the output must correlate with a plausible signal range.
    assert den.dtype == np.int8
    assert np.unique(den).size > 2, "denoised channel is constant: saturation bug"
    assert den.max() < 127, "int8 saturation: the +shift frame reached the writer"
    # Tail region specifically must be written (was np.empty garbage): dtype
    # int8 already bounds the range; require structure, and require the bulk of
    # the tail to be moderate (uninitialised memory reinterpreted as int8 is
    # uniform over [-128,127], median |.| ~ 64; a real reconstruction of the
    # near-mean signal is small except for the known one-hop STFT edge artifact).
    tail_region = den[-100:]
    assert np.unique(tail_region).size > 1, "tail region left uninitialised"
    assert np.median(np.abs(tail_region.astype(np.int64))) < 32, "tail looks like garbage memory"


def test_verify_denoised_file_rejects_saturation(tmp_path):
    from tidmad.infer import _verify_denoised_file

    path = tmp_path / "abra_validation_denoised_bad_0001.h5"
    n = 1 << 12
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "timeseries/channel0001/timeseries", data=np.full(n, 127, dtype=np.int8)
        )
        f.create_dataset(
            "timeseries/channel0002/timeseries", data=np.zeros(n, dtype=np.int8)
        )
    with pytest.raises(RuntimeError, match="saturation"):
        _verify_denoised_file(path, n)
