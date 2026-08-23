"""Config-equality gate and frozen-config sanity tests (PLAN_04 §3.4)."""

from __future__ import annotations

import numpy as np

from tidmad.config import FROZEN, TidmadRunConfig, TidmadTrainConfig
from tidmad.train import assert_configs_differ_only_in_loss, load_run_config


def test_frozen_config_consistency():
    stft = FROZEN.stft
    m = FROZEN.frames_per_window
    assert m % FROZEN.model.patch_len == 0, "M must be a multiple of patch_len"
    # Round-2 T2: the band embedding is dimensioned to C independently, so M no
    # longer has to satisfy M // patch_len >= C.
    assert stft.n_bands_stacked == 2 * stft.n_bands_used
    # Frozen paired-contrast band reaches the pre-P1 file-0015 endpoint. P1
    # showed later files are out of band, so external baseline comparison is dropped.
    assert stft.band_centre_frequencies_hz[-1] > 760_000
    # Window is a scientific choice (~20 ms), not an artefact of the band count.
    assert stft.frames_for_window(FROZEN.data.window_samples) == 96
    assert FROZEN.data.window_samples < 1_000_000, "window must be ~20 ms, not inflated by C"
    # The data contract is an external artifact (documented in docs/tidmad.md);
    # skip rather than fail where it is absent, so a fresh clone's suite is green.
    import pytest

    if not __import__("pathlib").Path(FROZEN.data.contract_path).is_file():
        pytest.skip(f"external data contract {FROZEN.data.contract_path} not present")


def test_arm_configs_differ_only_in_loss():
    left = load_run_config(__import__("pathlib").Path("src/tidmad/configs/t_mse.yaml"))
    right = load_run_config(__import__("pathlib").Path("src/tidmad/configs/t_chi2.yaml"))
    assert left.train.loss == "mse"
    assert right.train.loss == "chi2"
    # This is the gate: fails loudly if anything else drifted.
    assert_configs_differ_only_in_loss(left, right)


def test_chi2_of_config_is_gone():
    """t_chi2_of.yaml was a silent alias of the chi2 arm and was removed (audit B3).

    Reintroduce it only together with an actual optimal-filter forward pass.
    """
    assert not __import__("pathlib").Path("src/tidmad/configs/t_chi2_of.yaml").exists()


def test_config_equality_rejects_nonloss_drift():
    a = FROZEN
    b = TidmadRunConfig(a.stft, a.model, TidmadTrainConfig(**{**a.train.__dict__, "seed": 1}), a.data)
    try:
        assert_configs_differ_only_in_loss(a, b)
        raise AssertionError("seed drift must be rejected")
    except AssertionError as exc:
        assert "beyond the loss flag" in str(exc)


def test_training_loop_smoke_two_steps():
    """The full model + optimiser + scheduler + loss path runs on a synthetic batch."""
    import torch

    from tidmad.config import TidmadDataConfig, TidmadModelConfig, TidmadRunConfig, TidmadSTFTConfig, TidmadTrainConfig
    from tidmad.loss import reconstruction_losses
    from tidmad.train import build_model, train_step

    stft = TidmadSTFTConfig(n_fft=64, hop_length=32, win_length=64, n_bands_used=4)
    model_cfg = TidmadModelConfig(d_model=16, d_ff=32, n_head=2, patch_len=4)
    frames = stft.n_bands_stacked * model_cfg.patch_len
    data = TidmadDataConfig(window_samples=(frames - 1) * stft.hop_length + stft.n_fft)
    run = TidmadRunConfig(
        stft=stft, model=model_cfg, train=TidmadTrainConfig(num_steps=2, device_batch_size=1), data=data
    )
    j_stack = torch.ones(run.stft.n_bands_stacked)
    model = build_model(run)
    adamw, muon = model.configure_optimisers(adamw_fused=False)
    from reconstruction_model.schedulers import cosine_scheduler_with_linear_warmup

    scheds = [
        cosine_scheduler_with_linear_warmup(adamw, 0, run.train.num_steps),
        cosine_scheduler_with_linear_warmup(muon, 0, run.train.num_steps),
    ]
    # One streamed window is (2F, M); train_step adds the batch dimension.
    # The real NumPy STFT path produces float64; train_step must cast it to the
    # model's float32 dtype at the device boundary.
    spec_in = torch.randn(run.stft.n_bands_stacked, run.frames_per_window, dtype=torch.float64)
    spec_tgt = torch.randn_like(spec_in)
    for _ in range(2):
        metrics = train_step(model, [adamw, muon], scheds, [(spec_in, spec_tgt)], run, j_stack, "cpu")
        assert np.isfinite(metrics["mse"]) and np.isfinite(metrics["chi2"])


def test_registry_resolves_tidmad_stft():
    from tidmad.registry import get_model_entry, load_model_objects

    entry = get_model_entry("tidmad_stft")
    assert entry.requires_pairwise_cache is False
    cls, cfg_cls = load_model_objects("tidmad_stft")
    assert cls.__name__ == "TidmadTransformer"
