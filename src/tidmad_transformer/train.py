"""Minimal training loop for the two TIDMAD arms (PLAN_04 §3.4).

Only the residual metric in the objective differs between arms (``train.loss``
in the run config). Everything else — architecture, data, budget, seeds — is
identical and asserted by :func:`assert_configs_differ_only_in_loss`.

Reuses the backbone optimisers (Muon + AdamW) and the cosine-with-warmup
schedulers from the vendored backbone in ``tidmad_transformer.backbone``. Checkpointing is
local to this module because it must round-trip both optimisers, both
schedulers and the step counter for ``--resume`` to be exact.
"""

from __future__ import annotations

import argparse
import json
import shutil
import signal
import time
from pathlib import Path

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
    """Build a run config from a YAML arm file on top of the frozen defaults.

    Fails loudly on a missing file or missing PyYAML: the previous silent
    fallback to ``FROZEN`` meant a typo'd ``--config`` path trained the default
    (mse) arm while claiming to be whatever the filename said.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"arm config {path} does not exist")
    if yaml is None:
        raise ImportError("PyYAML is required to load arm configs")
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


def write_run_config(
    run: TidmadRunConfig, out: Path, num_params: int, provenance: dict | None = None
) -> None:
    """Write ``run_config.json`` in the ``save_resolved_config`` shape, plus loss.

    ``provenance`` records execution facts the frozen config cannot see (the
    CLI data directory, any ``--steps`` override, window counts, the whitening
    control). ``status`` starts as ``"started"`` and is finalised by
    :func:`finalise_run_config` — a ``run_config.json`` still reading
    ``"started"`` marks an interrupted or crashed run (audit C6).
    """
    Path(out).mkdir(parents=True, exist_ok=True)
    payload = {
        "model_variant": "tidmad_stft",
        "run_config": run.as_dict(),
        "num_trainable_params": num_params,
        "provenance": provenance or {},
        "status": "started",
        "completed_steps": None,
    }
    (out / "run_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def finalise_run_config(out: Path, completed_steps: int, interrupted: bool) -> None:
    """Record the realised step count so the archive matches execution (C6)."""
    path = Path(out) / "run_config.json"
    payload = json.loads(path.read_text())
    payload["completed_steps"] = int(completed_steps)
    payload["status"] = "interrupted" if interrupted else "completed"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


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
    """Which loss the arm optimises. Only implemented objectives are accepted.

    The former ``chi2_of`` alias silently trained the plain chi2 arm (audit
    B3/C3); it is removed until the optimal-filter forward pass exists. An
    unknown loss now fails loudly instead of impersonating another arm.
    """
    if run.train.loss not in ("mse", "chi2"):
        raise ValueError(
            f"unknown loss {run.train.loss!r}; implemented objectives: 'mse', 'chi2'. "
            "('chi2_of' was removed as a silent alias — implement the OF forward "
            "pass before reintroducing the config.)"
        )
    return run.train.loss


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
    shutil.copyfile(path, latest)
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


def _noise_only_windows(data_dir: Path, run: TidmadRunConfig, count: int | None = None):
    """Stream windows of noise-only science data, in the training frame.

    The ``+ sample_shift`` offset is applied so the calibration windows live in
    the SAME coordinate frame as the training/eval spectrograms built by
    ``data.read_channel_pair`` (audit M6/C9): J(f) must be estimated in the
    frame it later divides.
    """
    from ._vendor.loader import TidmadFile, load_contract

    contract = load_contract(run.data.contract_path)
    contract.require_verified()
    science = sorted(data_dir.glob(run.data.psd_source_glob))
    if not science:
        raise FileNotFoundError(f"no science files matched {run.data.psd_source_glob!r} in {data_dir}")
    n = run.data.calibration_window_samples
    total = int(count) if count is not None else run.data.n_calibration_windows
    shift = float(run.stft.sample_shift)
    produced = 0
    for path in science:
        with TidmadFile(path, contract) as fh:
            for win in fh.iter_squid_windows(n, stride=n, limit=total - produced):
                yield win + shift
                produced += 1
                if produced >= total:
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
    parser.add_argument(
        "--paired-config",
        type=Path,
        default=None,
        help="the OTHER arm's YAML; when given, assert_configs_differ_only_in_loss "
        "runs at startup so the single-difference gate executes in production, "
        "not only in the test suite (audit M3/C4)",
    )
    args = parser.parse_args()

    run = load_run_config(args.config)
    if args.paired_config is not None:
        assert_configs_differ_only_in_loss(run, load_run_config(args.paired_config))
        print(f"[tidmad.train] single-difference gate passed against {args.paired_config}")
    objective_key(run)  # fail loudly on an unimplemented loss before any work
    if args.steps is not None:
        run = TidmadRunConfig(
            run.stft, run.model, run.train.__class__(**{**run.train.__dict__, "num_steps": args.steps}), run.data
        )
    set_seed(run.train.seed)
    device = torch.device(args.device)

    from .psd import estimate_band_psd, tile_for_stacked_bands, whitening_identity_error

    # J(f) from the first n_calibration_windows; the NEXT n_calibration_windows
    # are held out for the T1.6 whitening positive control, so the check is not
    # scored on the windows that produced the estimate (audit N4/C22).
    noise_windows = list(_noise_only_windows(args.data_dir, run, count=2 * run.data.n_calibration_windows))
    cal_windows = noise_windows[: run.data.n_calibration_windows]
    check_windows = noise_windows[run.data.n_calibration_windows :]
    j_band, _ = estimate_band_psd(
        cal_windows,
        run.stft,
        sampling_frequency=run.stft.sampling_frequency_hz,
        window=run.data.psd_window,
        average=run.data.psd_average,
    )
    whitening_error = None
    if check_windows:
        whitening_error = whitening_identity_error(check_windows, j_band, run.stft)
        print(f"[tidmad.train] whitening identity error (held-out): {whitening_error:.4f}")
        if run.data.whitening_error_max is not None and whitening_error > run.data.whitening_error_max:
            raise RuntimeError(
                f"whitening identity error {whitening_error:.4f} exceeds the declared "
                f"gate {run.data.whitening_error_max}; J(f) does not whiten the "
                "calibration noise — the chi2 arm's weighting is not trustworthy."
            )
    del noise_windows, check_windows
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
    # Budget accounting (audit M2/C6): the fit stream cycles, so the declared
    # budget is always *reachable*; what must be recorded is how many passes
    # over the data it implies. Fail loudly only on an empty split.
    windows_needed = run.train.num_steps * run.train.device_batch_size
    effective_epochs = windows_needed / max(len(fit_positions), 1)
    print(
        f"[tidmad.train] windows: {len(positions)} total, "
        f"{len(fit_positions)} fit, {len(val_positions)} held out "
        f"(guard {run.data.guard_windows}); declared budget = {windows_needed} "
        f"window visits = {effective_epochs:.2f} passes over the fit split"
    )
    if not fit_positions:
        raise RuntimeError("no training windows after the chronological split")

    write_run_config(
        run,
        args.out,
        n_params,
        provenance={
            "data_dir": str(args.data_dir),
            "config_file": str(args.config),
            "paired_config": str(args.paired_config) if args.paired_config else None,
            "cli_steps_override": args.steps,
            "resume_from": str(resume) if resume else None,
            "n_windows_total": len(positions),
            "n_windows_fit": len(fit_positions),
            "n_windows_val": len(val_positions),
            "effective_epochs": effective_epochs,
            "whitening_identity_error": whitening_error,
            "whitening_error_max": run.data.whitening_error_max,
        },
    )

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
    interrupted = stop["requested"] or step < run.train.num_steps
    finalise_run_config(args.out, completed_steps=step, interrupted=interrupted)
    total = time.time() - started
    print(
        f"[tidmad.train] stopped at step {step}/{run.train.num_steps} "
        f"after {total/3600:.2f} h ({(step - start_step)/max(total, 1e-9):.2f} steps/s). "
        f"Final checkpoint: {final_path}"
    )
    if interrupted:
        print(
            "[tidmad.train] run did NOT reach the declared budget; run_config.json "
            "records status='interrupted'. Resume with --resume auto"
        )
    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
