"""Gates for the P4 training-loop fixes (checklist P4).

The checklist's gate is: a smoke run writes a checkpoint, is killed mid-run,
resumes from that checkpoint, and reports throughput. These tests cover the
mechanical half of that — that a checkpoint round-trips exactly and that the
chronological split leaks nothing. The wall-clock half is the smoke profile of
``scripts/train_tidmad_local.sh``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tidmad_transformer.config import (  # noqa: E402
    TidmadDataConfig,
    TidmadModelConfig,
    TidmadRunConfig,
    TidmadSTFTConfig,
    TidmadTrainConfig,
)
from tidmad_transformer.train import (  # noqa: E402
    _checkpoint,
    _restore,
    build_model,
    chronological_split,
    objective_key,
)
from reconstruction_model.schedulers import cosine_scheduler_with_linear_warmup  # noqa: E402


def _tiny_run(**train_overrides) -> TidmadRunConfig:
    """A run small enough to instantiate on CPU in a test."""
    stft = TidmadSTFTConfig(n_fft=64, hop_length=32, win_length=64, n_bands_used=4)
    model = TidmadModelConfig(d_model=16, d_ff=32, n_head=2, patch_len=2)
    train = TidmadTrainConfig(**{"num_steps": 10, "warmup_steps": 1, **train_overrides})
    data = TidmadDataConfig(window_samples=64 + 32 * 7)
    return TidmadRunConfig(stft=stft, model=model, train=train, data=data)


def test_checkpoint_round_trips_exactly(tmp_path: Path):
    """Model, both optimisers, both schedulers and the step counter must restore.

    A resume that silently drops optimiser or scheduler state would continue
    from a different trajectory than the one that was interrupted, which is
    exactly the failure a resumable run exists to avoid.
    """
    run = _tiny_run()
    model = build_model(run)
    adamw, muon = model.configure_optimisers(adamw_fused=False)
    scheds = [
        cosine_scheduler_with_linear_warmup(adamw, run.train.warmup_steps, run.train.num_steps),
        cosine_scheduler_with_linear_warmup(muon, run.train.warmup_steps, run.train.num_steps),
    ]

    # Advance the schedulers so their state is non-trivial.
    for _ in range(3):
        for s in scheds:
            s.step()

    path = _checkpoint(tmp_path, model, adamw, muon, scheds, step=7, run=run)
    assert path.exists()
    assert (tmp_path / "latest.pt").exists(), "latest.pt must exist for --resume auto"

    before = {k: v.clone() for k, v in model.state_dict().items()}
    lrs_before = [s.get_last_lr() for s in scheds]

    # Perturb everything, then restore.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    for _ in range(4):
        for s in scheds:
            s.step()

    fresh = build_model(run)
    f_adamw, f_muon = fresh.configure_optimisers(adamw_fused=False)
    f_scheds = [
        cosine_scheduler_with_linear_warmup(f_adamw, run.train.warmup_steps, run.train.num_steps),
        cosine_scheduler_with_linear_warmup(f_muon, run.train.warmup_steps, run.train.num_steps),
    ]
    step = _restore(path, fresh, f_adamw, f_muon, f_scheds, device="cpu")

    assert step == 7
    for key, value in fresh.state_dict().items():
        assert torch.allclose(value, before[key]), f"{key} did not restore"
    assert [s.get_last_lr() for s in f_scheds] == lrs_before, "scheduler state did not restore"


def test_chronological_split_leaves_a_guard_gap():
    """No held-out window may share samples with a fit window."""
    run = _tiny_run()
    positions = [(Path("f.h5"), i * 1000) for i in range(100)]
    fit, val = chronological_split(positions, run)

    assert fit and val
    assert len(fit) == int(100 * run.data.fit_fraction)
    # The guard windows between the two splits belong to neither.
    assert len(fit) + len(val) == 100 - run.data.guard_windows
    assert set(fit).isdisjoint(val)
    # Validation must be strictly later in time than every fit window.
    assert min(p for _, p in val) > max(p for _, p in fit)


def test_unknown_loss_fails_loudly():
    """Only implemented objectives are accepted; no silent aliases (audit B3).

    ``chi2_of`` used to alias to ``chi2`` silently, so its "arm" was a
    bit-identical duplicate of the chi2 arm. It must now raise until the
    optimal-filter forward pass actually exists.
    """
    assert objective_key(_tiny_run(loss="mse")) == "mse"
    assert objective_key(_tiny_run(loss="chi2")) == "chi2"
    import pytest

    with pytest.raises(ValueError, match="unknown loss"):
        objective_key(_tiny_run(loss="chi2_of"))


def test_grad_clip_is_configurable_and_on_by_default():
    assert _tiny_run().train.grad_clip == 1.0
    assert _tiny_run(grad_clip=0.0).train.grad_clip == 0.0
