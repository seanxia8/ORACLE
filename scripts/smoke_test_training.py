#!/usr/bin/env python3
"""Run a tiny end-to-end training smoke test on a local CUDA GPU.

The script creates a miniature dataset in the same on-disk format expected by
the real dataloader, then runs the normal training loop with a small model,
AdamW, Muon, cosine schedulers, checkpointing, and optional W&B logging.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
import zstandard as zstd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reconstruction_model.dataset import DataConfig, create_dataloaders
from reconstruction_model.model import Transformer, TransformerConfig
from reconstruction_model.schedulers import cosine_scheduler_with_linear_warmup
from reconstruction_model.train import TrainingConfig, resolve_amp, set_seed, train
from reconstruction_model.utils import count_model_params


DEFAULT_WORK_DIR = PROJECT_ROOT / "artifacts" / "smoke_test"


def _byte_shuffle_traces(traces: np.ndarray) -> bytes:
    """Match the byte shuffle reversed by dataset._unshuffle_batch."""
    traces = np.ascontiguousarray(traces)
    batch_size = traces.shape[0]
    flat = traces.reshape(batch_size, -1)
    dtype = flat.dtype
    u8 = flat.view(np.uint8).reshape(batch_size, flat.shape[1], dtype.itemsize)
    return u8.swapaxes(1, 2).reshape(batch_size, -1).tobytes()


def _write_energy_file(
    recoil_dir: Path,
    recoil_type: str,
    energy: int,
    *,
    n_events: int,
    n_channels: int,
    trace_samples: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    recoil_dir.mkdir(parents=True, exist_ok=True)

    traces = rng.normal(0.0, 0.02, size=(n_events, n_channels, trace_samples)).astype(
        np.float16
    )
    positions = rng.uniform([-30.0, -30.0, 5.0], [30.0, 30.0, 90.0], size=(n_events, 3))

    pulse_len = min(32, trace_samples)
    pulse_t = np.linspace(0.0, 1.0, pulse_len, dtype=np.float32)
    pulse = np.exp(-((pulse_t - 0.35) ** 2) / 0.02)
    for event_idx in range(n_events):
        channel = event_idx % n_channels
        start = int(rng.integers(0, max(1, trace_samples - pulse_len)))
        amplitude = energy / 500.0 + (0.01 * positions[event_idx, 2])
        traces[event_idx, channel, start : start + pulse_len] += (
            amplitude * pulse
        ).astype(np.float16)

    compressed = zstd.ZstdCompressor(level=1).compress(_byte_shuffle_traces(traces))
    (recoil_dir / f"traces_energy_{energy}.zst").write_bytes(compressed)

    events_dtype = np.dtype(
        [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("energy", "f4"),
            ("type_recoil", "S2"),
            ("no_noise", "?"),
            ("quantize", "?"),
        ]
    )
    events = np.zeros(n_events, dtype=events_dtype)
    events["x"] = positions[:, 0]
    events["y"] = positions[:, 1]
    events["z"] = positions[:, 2]
    events["energy"] = energy
    events["type_recoil"] = recoil_type.encode("utf-8")
    events["no_noise"] = True
    events["quantize"] = False

    with h5py.File(recoil_dir / f"meta_energy_{energy}.h5", "w") as handle:
        handle.attrs["n_channels"] = n_channels
        handle.attrs["trace_samples"] = trace_samples
        handle.attrs["trace_dtype"] = "float16"
        handle.create_dataset("events", data=events)


def create_smoke_dataset(
    data_root: Path,
    *,
    n_events_per_recoil: int,
    n_channels: int,
    trace_samples: int,
    energy: int,
    seed: int,
) -> None:
    if data_root.exists():
        shutil.rmtree(data_root)
    for offset, recoil_type in enumerate(("ER", "NR")):
        _write_energy_file(
            data_root / "train" / recoil_type,
            recoil_type,
            energy,
            n_events=n_events_per_recoil,
            n_channels=n_channels,
            trace_samples=trace_samples,
            seed=seed + offset,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--num-steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--n-events-per-recoil", type=int, default=12)
    parser.add_argument("--n-channels", type=int, default=4)
    parser.add_argument("--trace-samples", type=int, default=256)
    parser.add_argument("--energy", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true", help="Use torch.compile.")
    parser.add_argument(
        "--wandb-mode",
        choices=("disabled", "offline", "online"),
        default="disabled",
        help="W&B logging mode for the smoke run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke test exercises the CUDA training path.")

    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    data_root = args.work_dir / "training_data"
    cache_root = args.work_dir / "cache"
    checkpoint_dir = args.work_dir / "checkpoints" / datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    create_smoke_dataset(
        data_root,
        n_events_per_recoil=args.n_events_per_recoil,
        n_channels=args.n_channels,
        trace_samples=args.trace_samples,
        energy=args.energy,
        seed=args.seed,
    )

    data_config = DataConfig(
        access_mode="local",
        local_data_path=data_root,
        local_cache_path=cache_root,
        max_seq_len=args.trace_samples,
        recoil_types=["ER", "NR"],
        train_split=0.75,
        val_split=0.25,
    )
    training_config = TrainingConfig(
        num_steps=args.num_steps,
        eval_step_period=1,
        eval_num_batches=1,
        save_checkpoint_period=1,
        total_batch_size=args.batch_size,
        device_batch_size=args.batch_size,
        num_workers=0,
        checkpoint_dir=checkpoint_dir,
        adamw_fused=True,
        adamw_warmup_steps=1,
        muon_warmup_steps=1,
        wandb_run=args.wandb_mode != "disabled",
        wandb_run_name=f"smoke_local_gpu_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        wandb_project_name="DELight_Reconstruction_Smoke",
    )
    os.environ["WANDB_MODE"] = args.wandb_mode

    model_config = TransformerConfig(
        d_model=32,
        d_ff=64,
        max_seq_len=args.trace_samples,
        patch_len=32,
        patch_stride=32,
        n_head=4,
        n_time_layers=1,
        n_channel_layers=1,
    )

    device = torch.device("cuda")
    model = Transformer(model_config)
    model.to(device=device)
    model.init_weights()
    base_model = model
    if args.compile:
        model = torch.compile(model)

    adamw_optimiser, muon_optimiser = model.configure_optimisers(
        adamw_lr=training_config.adamw_lr,
        adamw_betas=training_config.adamw_betas,
        adamw_weight_decay=training_config.adamw_weight_decay,
        adamw_fused=training_config.adamw_fused,
        muon_lr=training_config.muon_lr,
        muon_momentum=training_config.muon_momentum,
        nesterov=training_config.nesterov,
        ns_steps=training_config.ns_steps,
    )
    adamw_scheduler = cosine_scheduler_with_linear_warmup(
        adamw_optimiser,
        num_warmup_steps=training_config.adamw_warmup_steps,
        total_steps=training_config.num_steps,
    )
    muon_scheduler = cosine_scheduler_with_linear_warmup(
        muon_optimiser,
        num_warmup_steps=training_config.muon_warmup_steps,
        total_steps=training_config.num_steps,
    )

    dataloaders = create_dataloaders(
        data_config,
        batch_size=training_config.device_batch_size,
        num_workers=training_config.num_workers,
    )
    print("Smoke data:", data_root)
    print("Checkpoints:", checkpoint_dir)
    print("Model params:", f"{count_model_params(model, trainable_only=True):,}")
    print("Training config:", asdict(training_config))
    amp_dtype, scaler = resolve_amp(device, "auto")

    train(
        model,
        adamw_optimiser,
        adamw_scheduler,
        muon_optimiser,
        muon_scheduler,
        dataloaders["train"],
        dataloaders["val"],
        device,
        training_config,
        amp_dtype=amp_dtype,
        scaler=scaler,
        model_config=model_config,
        data_config=data_config,
        checkpoint_model=base_model,
    )


if __name__ == "__main__":
    main()
