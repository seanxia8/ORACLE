"""Minimal training loop for the two TIDMAD arms (PLAN_04 §3.4).

Only the residual metric in the objective differs between arms (``train.loss``
in the run config). Everything else — architecture, data, budget, seeds — is
identical and asserted by :func:`assert_configs_differ_only_in_loss`.

Reuses the backbone optimisers (Muon + AdamW) and the cosine-with-warmup
schedulers from the vendored backbone in ``tidmad.backbone``. Checkpointing is
local to this module because it must round-trip both optimisers, both
schedulers and the step counter for ``--resume`` to be exact.
"""

from __future__ import annotations

import argparse
import json
import random
import signal
import time
import sys
from pathlib import Path

import numpy as np
import torch

from reconstruction_model.schedulers import cosine_scheduler_with_linear_warmup
from .seeding import set_seed
from reconstruction_model.model import TransformerConfig

from .config import FROZEN, TidmadRunConfig
from .loss import reconstruction_losses
from .model import TidmadTransformer

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional
    yaml = None


def load_run_config(path: Path) -> TidmadRunConfig:
    """Build a run config from a YAML arm file on top of the frozen defaults."""
    payload = {}
    if yaml is not None and Path(path).exists():
        with open(path) as fh:
            payload = yaml.safe_load(fh) or {}
    return _apply_overrides(FROZEN, payload)


def _apply_overrides(frozen: TidmadRunConfig, payload: dict) -> TidmadRunConfig:
    data = {**frozen.data.__dict__, **payload.get("data", {})}
    train = {**frozen.train.__dict__, **payload.get("train", {})}
    model = {**frozen.model.__dict__, **payload.get("model", {})}
    stft = {**frozen.stft.__dict__, **payload.get("stft", {})}
    from .config import TidmadDataConfig, TidmadModelConfig, TidmadSTFTConfig, TidmadTrainConfig

    return TidmadRunConfig(
        stft=TidmadSTFTConfig(**stft),
        model=TidmadModelConfig(**model),
        train=TidmadTrainConfig(**train),
        data=TidmadDataConfig(**data),
    )


def assert_configs_differ_only_in_loss(left: TidmadRunConfig, right: TidmadRunConfig) -> None:
    """Gate: the two arms' configs must differ in ``train.loss`` and nothing else."""
    a, b = left.as_dict(), right.as_dict()
    diffs: list[str] = []
    for key in sorted(set(a) | set(b)):
        if a[key] != b[key]:
            if key == "train" and _only_loss_differs(a["train"], b["train"]):
                continue
            diffs.append(key)
    if diffs:
        raise AssertionError(
            f"arm configs differ beyond the loss flag: {diffs}. The contrast is invalid."
        )
    if a["train"]["loss"] == b["train"]["loss"]:
        raise AssertionError("arm configs have the same loss flag; the contrast is empty.")


def _only_loss_differs(a: dict, b: dict) -> bool:
    return all(k == "loss" or a[k] == b[k] for k in set(a) | set(b))


def write_run_config(run: TidmadRunConfig, out: Path, num_params: int) -> None:
    """Write ``run_config.json`` in the ``save_resolved_config`` shape, plus loss."""
    Path(out).mkdir(parents=True, exist_ok=True)
    payload = {
        "model_variant": "tidmad_stft",
        "run_config": run.as_dict(),
        "num_trainable_params": num_params,
    }
    (out / "run_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_model(run: TidmadRunConfig) -> TidmadTransformer:
    cfg = run.model
    torch_cfg = TransformerConfig(
        d_model=cfg.d_model,
        d_ff=cfg.d_ff,
        max_seq_len=run.frames_per_window,
        patch_len=cfg.patch_len,
        patch_stride=cfg.patch_len,
        n_head=cfg.n_head,
        n_time_layers=cfg.n_time_layers,
        n_channel_layers=cfg.n_channel_layers,
        rope_base=cfg.rope_base,
        norm_eps=cfg.norm_eps,
    )
    model = TidmadTransformer(torch_cfg, n_bands=run.stft.n_bands_used)
    model.init_weights()
    return model


def objective_key(run: TidmadRunConfig) -> str:
    """Which loss the arm optimises. ``chi2_of`` trains on the chi2 objective."""
    return "chi2" if run.train.loss == "chi2_of" else run.train.loss


def train_step(model, optimisers, schedulers, batch, run, j_stack, device):
    """One optimiser step over ``device_batch_size`` windows.

    ``batch`` is a list of ``(input_spec, target_spec)`` pairs, each a
    measurement-coordinate ``(C, M)`` spectrogram. Gradients are accumulated
    across the list so the effective batch matches the frozen config; before
    round 2 this consumed a single window and ``device_batch_size`` was ignored.
    """
    key = objective_key(run)
    n = max(len(batch), 1)
    for opt in optimisers:
        opt.zero_grad()

    totals = {"mse": 0.0, "chi2": 0.0}
    for input_spec, target_spec in batch:
        input_spec = torch.as_tensor(input_spec).unsqueeze(0).to(device=device, dtype=torch.float32)
        target_spec = torch.as_tensor(target_spec).unsqueeze(0).to(device=device, dtype=torch.float32)
        out_std = model(input_spec)
        losses = reconstruction_losses(out_std, target_spec, input_spec, j_stack)
        (losses[key] / n).backward()
        for k, v in losses.items():
            totals[k] += float(v.item()) / n

    if run.train.grad_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), run.train.grad_clip)
    for opt in optimisers:
        opt.step()
    for sched in schedulers:
        sched.step()
    return totals


@torch.no_grad()
def evaluate(model, windows, run, j_stack, device, max_batches: int) -> dict:
    """Held-out evaluation on the chronological validation split.

    Reports *both* metrics for both arms, always: the paired contrast is
    between objectives, so each arm must be scored under both.
    """
    model.eval()
    totals = {"mse": 0.0, "chi2": 0.0}
    seen = 0
    for input_spec, target_spec in windows:
        if seen >= max_batches:
            break
        input_spec = torch.as_tensor(input_spec).unsqueeze(0).to(device=device, dtype=torch.float32)
        target_spec = torch.as_tensor(target_spec).unsqueeze(0).to(device=device, dtype=torch.float32)
        losses = reconstruction_losses(model(input_spec), target_spec, input_spec, j_stack)
        for k, v in losses.items():
            totals[k] += float(v.item())
        seen += 1
    model.train()
    if seen == 0:
        return {"val_mse": float("nan"), "val_chi2": float("nan"), "val_windows": 0}
    return {
        "val_mse": totals["mse"] / seen,
        "val_chi2": totals["chi2"] / seen,
        "val_windows": seen,
    }


def _checkpoint(out: Path, model, adamw, muon, schedulers, step: int, run: TidmadRunConfig) -> Path:
    """Write a resumable checkpoint. Returns the path written."""
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"checkpoint_{step:08d}.pt"
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "adamw": adamw.state_dict(),
            "muon": muon.state_dict(),
            "adamw_scheduler": schedulers[0].state_dict(),
            "muon_scheduler": schedulers[1].state_dict(),
            "run_config": run.as_dict(),
            "loss": run.train.loss,
        },
        path,
    )
    latest = out / "latest.pt"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    torch.save(torch.load(path, map_location="cpu", weights_only=False), latest)
    return path


def _restore(resume: Path, model, adamw, muon, schedulers, device) -> int:
    """Restore model, both optimisers, both schedulers and the step counter."""
    state = torch.load(resume, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    adamw.load_state_dict(state["adamw"])
    muon.load_state_dict(state["muon"])
    schedulers[0].load_state_dict(state["adamw_scheduler"])
    schedulers[1].load_state_dict(state["muon_scheduler"])
    step = int(state["step"])
    print(f"[tidmad.train] resumed from {resume} at step {step}")
    return step


def _noise_only_windows(data_dir: Path, run: TidmadRunConfig):
    """Stream ``n_calibration_windows`` windows of noise-only science data."""
    from ._vendor.loader import TidmadFile, load_contract

    contract = load_contract(run.data.contract_path)
    contract.require_verified()
    science = sorted(data_dir.glob(run.data.psd_source_glob))
    if not science:
        raise FileNotFoundError(f"no science files matched {run.data.psd_source_glob!r} in {data_dir}")
    n = run.data.calibration_window_samples
    produced = 0
    for path in science:
        with TidmadFile(path, contract) as fh:
            for win in fh.iter_squid_windows(n, stride=n, limit=run.data.n_calibration_windows - produced):
                yield win
                produced += 1
                if produced >= run.data.n_calibration_windows:
                    return


def _window_positions(data_dir: Path, run: TidmadRunConfig) -> list[tuple[Path, int]]:
    """All ``(file, sample_offset)`` window starts, in chronological order."""
    from ._vendor.loader import TidmadFile, load_contract

    contract = load_contract(run.data.contract_path)
    contract.require_verified()
    files = [Path(f) for f in run.data.train_files] or sorted(data_dir.glob("abra_training_*.h5"))
    if not files:
        raise FileNotFoundError(f"no training files found under {data_dir}")
    n = run.data.window_samples
    positions: list[tuple[Path, int]] = []
    for path in files:
        with TidmadFile(path, contract) as fh:
            pos = 0
            while pos + n <= fh.n_samples:
                positions.append((path, pos))
                pos += n
    return positions


def chronological_split(
    positions: list[tuple[Path, int]], run: TidmadRunConfig
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    """Split chronologically with a guard gap, mirroring Paper 1's TIDMAD protocol.

    The gap between fit and evaluation is dropped entirely so no evaluation
    window shares samples with a training window.
    """
    n_fit = int(len(positions) * run.data.fit_fraction)
    guard = run.data.guard_windows
    fit = positions[:n_fit]
    val = positions[n_fit + guard :]
    return fit, val


def _stream(positions, run: TidmadRunConfig, *, loop: bool):
    """Yield ``(input_spec, target_spec)`` for the given window positions."""
    from ._vendor.loader import TidmadFile, load_contract  # noqa: F401

    from .data import build_window_stft

    from ._vendor.loader import load_contract as _load

    contract = _load(run.data.contract_path)
    while True:
        for path, pos in positions:
            yield build_window_stft(path, contract, pos, run.stft, run.data.window_samples)
        if not loop:
            return


def _batches(stream, size: int):
    """Group a window stream into lists of ``size``."""
    batch = []
    for item in stream:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="checkpoint to resume from; 'auto' picks <out>/latest.pt if present",
    )
    parser.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    parser.add_argument("--wandb-project", default="rps-tidmad-rung4")
    args = parser.parse_args()

    run = load_run_config(args.config)
    if args.steps is not None:
        run = TidmadRunConfig(
            run.stft, run.model, run.train.__class__(**{**run.train.__dict__, "num_steps": args.steps}), run.data
        )
    set_seed(run.train.seed)
    device = torch.device(args.device)

    from .psd import estimate_band_psd, tile_for_stacked_bands

    j_band, _ = estimate_band_psd(
        _noise_only_windows(args.data_dir, run),
        run.stft,
        sampling_frequency=run.stft.sampling_frequency_hz,
        window=run.data.psd_window,
        average=run.data.psd_average,
    )
    j_stack = torch.as_tensor(tile_for_stacked_bands(j_band, run.stft), dtype=torch.float32)

    model = build_model(run).to(device)
    adamw, muon = model.configure_optimisers(
        adamw_lr=run.train.adamw_lr,
        adamw_betas=run.train.adamw_betas,
        adamw_weight_decay=run.train.adamw_weight_decay,
        adamw_fused=False,
        muon_lr=run.train.muon_lr,
        muon_momentum=run.train.muon_momentum,
        muon_nesterov=run.train.muon_nesterov,
        muon_ns_steps=run.train.muon_ns_steps,
    )
    schedulers = [
        cosine_scheduler_with_linear_warmup(adamw, run.train.warmup_steps, run.train.num_steps),
        cosine_scheduler_with_linear_warmup(muon, run.train.warmup_steps, run.train.num_steps),
    ]

    n_params = sum(p.numel() for p in model.parameters())
    write_run_config(run, args.out, n_params)
    print(f"[tidmad.train] arm loss={run.train.loss} model_params={n_params}")

    j_stack = j_stack.to(device)

    # ---- resume -----------------------------------------------------------
    start_step = 0
    resume = args.resume
    if resume is not None and str(resume) == "auto":
        candidate = args.out / "latest.pt"
        resume = candidate if candidate.exists() else None
    if resume is not None:
        start_step = _restore(resume, model, adamw, muon, schedulers, device)

    # ---- data splits ------------------------------------------------------
    positions = _window_positions(args.data_dir, run)
    fit_positions, val_positions = chronological_split(positions, run)
    print(
        f"[tidmad.train] windows: {len(positions)} total, "
        f"{len(fit_positions)} fit, {len(val_positions)} held out "
        f"(guard {run.data.guard_windows})"
    )
    if not fit_positions:
        raise RuntimeError("no training windows after the chronological split")

    # ---- clean stop on SIGTERM / SIGINT ------------------------------------
    # `timeout` sends SIGTERM. Without this a capped run dies with no
    # checkpoint and the whole segment is lost.
    stop = {"requested": False}

    def _request_stop(signum, _frame):
        stop["requested"] = True
        print(f"\n[tidmad.train] signal {signum} received; checkpointing then exiting.", flush=True)

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    # ---- optional experiment tracking -------------------------------------
    wb = None
    if args.wandb:
        try:
            import wandb as wb  # type: ignore

            wb.init(
                project=args.wandb_project,
                name=f"{run.train.loss}_{args.out.name}",
                config=run.as_dict(),
                resume="allow",
            )
        except ImportError:
            print("[tidmad.train] wandb requested but not installed; continuing without it")
            wb = None

    # ---- training ---------------------------------------------------------
    stream = _stream(fit_positions, run, loop=True)
    started = time.time()
    last_report = started
    step = start_step
    final_path = None

    for batch in _batches(stream, run.train.device_batch_size):
        if step >= run.train.num_steps or stop["requested"]:
            break
        metrics = train_step(model, [adamw, muon], schedulers, batch, run, j_stack, device)
        step += 1

        if step % run.train.eval_period == 0:
            now = time.time()
            steps_per_s = run.train.eval_period / max(now - last_report, 1e-9)
            last_report = now
            val = evaluate(
                model,
                _stream(val_positions, run, loop=False),
                run,
                j_stack,
                device,
                max_batches=run.train.eval_num_windows,
            )
            elapsed = now - started
            remaining = (run.train.num_steps - step) / max(steps_per_s, 1e-9)
            record = {
                "step": step,
                "train_mse": metrics["mse"],
                "train_chi2": metrics["chi2"],
                **val,
                "steps_per_s": steps_per_s,
                "elapsed_s": elapsed,
                "eta_s": remaining,
            }
            print(
                f"step {step:>8}  train_mse={metrics['mse']:.6g} train_chi2={metrics['chi2']:.6g}  "
                f"val_mse={val['val_mse']:.6g} val_chi2={val['val_chi2']:.6g}  "
                f"{steps_per_s:.2f} steps/s  eta {remaining/3600:.2f} h",
                flush=True,
            )
            if wb is not None:
                wb.log(record, step=step)

        if step % run.train.checkpoint_period == 0:
            final_path = _checkpoint(args.out, model, adamw, muon, schedulers, step, run)

    final_path = _checkpoint(args.out, model, adamw, muon, schedulers, step, run)
    total = time.time() - started
    print(
        f"[tidmad.train] stopped at step {step}/{run.train.num_steps} "
        f"after {total/3600:.2f} h ({(step - start_step)/max(total, 1e-9):.2f} steps/s). "
        f"Final checkpoint: {final_path}"
    )
    if stop["requested"]:
        print("[tidmad.train] stop was requested; resume with --resume auto")
    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
