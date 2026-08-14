"""
Training loop / logging script. Note that the configurations are messy
right now to be later fixed with the hydra package or raw JSON files.
"""
from __future__ import annotations 

import time
import json
import logging
import os
import math
import random
from dataclasses import dataclass, field, fields, asdict
from datetime import datetime
import pathlib
from pathlib import Path
from typing import Any

import wandb
import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np

from reconstruction_model.dataset import (
    DataConfig,
    create_dataloaders,
)
from reconstruction_model.models import (
    create_model,
    get_model_entry,
    load_model_objects,
)
from reconstruction_model.utils import count_model_params
from reconstruction_model.schedulers import cosine_scheduler_with_linear_warmup
from reconstruction_model.checkpoints import load_resume_checkpoint, save_checkpoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
_BASE_CHECKPOINT_PATH = Path(__file__).parent / "training_checkpoints"
_BASE_CHECKPOINT_PATH.mkdir(exist_ok=True, parents=True)


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    return value


def save_resolved_config(
    checkpoint_dir: str | Path,
    model_variant: str,
    model_config: Any,
    training_config: TrainingConfig,
    data_config: DataConfig,
    num_trainable_params: int,
):
    model_entry = get_model_entry(model_variant)
    config_path = Path(checkpoint_dir) / "run_config.json"
    payload = {
        "model_variant": model_variant,
        "model_entry": _json_ready(asdict(model_entry)),
        "model_config": _json_ready(asdict(model_config)),
        "training_config": _json_ready(asdict(training_config)),
        "data_config": _json_ready(asdict(data_config)),
        "num_trainable_params": num_trainable_params,
    }
    config_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    logger.info("Resolved run config saved at: %s", config_path)


@dataclass
class TrainingConfig:
    # General training config
    num_steps: int = 100000
    num_epochs: int | None = None
    eval_step_period: int = 10
    eval_num_batches: int = 32
    save_checkpoint_period: int = 250
    total_batch_size: int = 64
    device_batch_size: int = 16
    num_workers: int = 1
    scalar_loss_weights: tuple[float, ...] = (1.0, 1.0, 1.0)
    recoil_classification: bool = False
    spatial_target_indices: tuple[int, ...] | None = None
    model_variant: str = "current_compact"
    amp_dtype: str = "auto"
    seed: int = 42
    resume_checkpoint: str | None = None
    remote_checkpoint_dir: str | None = None
    data_staging_seconds: float = 0.0
    grad_clip: float = 1.0
    checkpoint_dir: str | Path = _BASE_CHECKPOINT_PATH
    # Optimiser specific config
    adamw_lr: float = 0.001
    adamw_betas: tuple[float] = (0.9, 0.999)
    adamw_weight_decay: float = 0.0
    adamw_fused: bool = True
    muon_lr: float = 0.001
    muon_momentum: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5
    # Scheduler specific config
    adamw_warmup_steps: int = 0
    muon_warmup_steps: int = 0
    # Weights and biases config
    wandb_run: bool = True
    wandb_checkpoint_artifacts: bool = False
    wandb_run_name: str = field(
        default_factory=lambda: f"reconstruction_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    wandb_project_name: str = "DELight_Reconstruction"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def _get_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None or value == "" else int(value)


def _get_env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None or value == "" else float(value)


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _get_env_int_list(name: str):
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _get_env_float_tuple(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def _get_env_int_tuple(name: str):
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _get_env_optional_int(name: str):
    value = os.environ.get(name)
    return None if value is None or value == "" else int(value)


def ddp_enabled() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def ddp_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))


def ddp_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def init_ddp() -> tuple[bool, int]:
    world_size = ddp_world_size()
    if world_size <= 1:
        return False, 0
    local_rank = ddp_local_rank()
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"Local rank {local_rank} is out of bounds for visible CUDA devices "
            f"({torch.cuda.device_count()})"
        )
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return True, local_rank


def build_training_config_from_env() -> TrainingConfig:
    return TrainingConfig(
        num_steps=_get_env_int("RECONSTRUCTION_NUM_STEPS", TrainingConfig.num_steps),
        num_epochs=_get_env_optional_int("RECONSTRUCTION_NUM_EPOCHS"),
        eval_step_period=_get_env_int(
            "RECONSTRUCTION_EVAL_STEP_PERIOD",
            TrainingConfig.eval_step_period,
        ),
        eval_num_batches=_get_env_int(
            "RECONSTRUCTION_EVAL_NUM_BATCHES",
            TrainingConfig.eval_num_batches,
        ),
        save_checkpoint_period=_get_env_int(
            "RECONSTRUCTION_SAVE_CHECKPOINT_PERIOD",
            TrainingConfig.save_checkpoint_period,
        ),
        total_batch_size=_get_env_int(
            "RECONSTRUCTION_TOTAL_BATCH_SIZE",
            TrainingConfig.total_batch_size,
        ),
        device_batch_size=_get_env_int(
            "RECONSTRUCTION_DEVICE_BATCH_SIZE",
            TrainingConfig.device_batch_size,
        ),
        num_workers=_get_env_int("RECONSTRUCTION_NUM_WORKERS", TrainingConfig.num_workers),
        scalar_loss_weights=_get_env_float_tuple(
            "RECONSTRUCTION_SCALAR_LOSS_WEIGHTS",
            TrainingConfig.scalar_loss_weights,
        ),
        recoil_classification=_get_env_bool(
            "RECONSTRUCTION_RECOIL_CLASSIFICATION",
            TrainingConfig.recoil_classification,
        ),
        spatial_target_indices=_get_env_int_tuple("RECONSTRUCTION_SPATIAL_TARGET_INDICES"),
        model_variant=os.environ.get(
            "RECONSTRUCTION_MODEL_VARIANT",
            TrainingConfig.model_variant,
        ),
        amp_dtype=os.environ.get("RECONSTRUCTION_AMP_DTYPE", TrainingConfig.amp_dtype),
        seed=_get_env_int("RECONSTRUCTION_SEED", TrainingConfig.seed),
        resume_checkpoint=os.environ.get("RECONSTRUCTION_RESUME_CHECKPOINT") or None,
        remote_checkpoint_dir=os.environ.get("RECONSTRUCTION_REMOTE_CHECKPOINT_DIR") or None,
        data_staging_seconds=_get_env_float(
            "RECONSTRUCTION_DATA_STAGING_SECONDS",
            TrainingConfig.data_staging_seconds,
        ),
        checkpoint_dir=os.environ.get(
            "RECONSTRUCTION_CHECKPOINT_DIR",
            str(_BASE_CHECKPOINT_PATH),
        ),
        adamw_fused=_get_env_bool("RECONSTRUCTION_ADAMW_FUSED", TrainingConfig.adamw_fused),
        adamw_warmup_steps=_get_env_int(
            "RECONSTRUCTION_ADAMW_WARMUP_STEPS",
            TrainingConfig.adamw_warmup_steps,
        ),
        muon_warmup_steps=_get_env_int(
            "RECONSTRUCTION_MUON_WARMUP_STEPS",
            TrainingConfig.muon_warmup_steps,
        ),
        wandb_run=_get_env_bool("RECONSTRUCTION_WANDB_RUN", TrainingConfig.wandb_run),
        wandb_checkpoint_artifacts=_get_env_bool(
            "RECONSTRUCTION_WANDB_CHECKPOINT_ARTIFACTS",
            TrainingConfig.wandb_checkpoint_artifacts,
        ),
        wandb_run_name=os.environ.get(
            "RECONSTRUCTION_WANDB_RUN_NAME",
            f"reconstruction_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        ),
        wandb_project_name=os.environ.get(
            "RECONSTRUCTION_WANDB_PROJECT",
            TrainingConfig.wandb_project_name,
        ),
    )


def _coerce_env_config_value(raw_value: str, default_value):
    if isinstance(default_value, bool):
        return raw_value.lower() in {"1", "true", "yes", "on"}
    if isinstance(default_value, int) and not isinstance(default_value, bool):
        return int(raw_value)
    if isinstance(default_value, float):
        return float(raw_value)
    if isinstance(default_value, Path):
        return Path(raw_value)
    if isinstance(default_value, tuple):
        return tuple(raw_value.split(","))
    if isinstance(default_value, list):
        item_type = type(default_value[0]) if default_value else str
        return [item_type(part.strip()) for part in raw_value.split(",") if part.strip()]
    return raw_value


def build_model_from_env(model_variant: str):
    model_cls, config_cls = load_model_objects(model_variant)
    default_config = config_cls()
    overrides = {}
    for config_field in fields(default_config):
        env_name = f"RECONSTRUCTION_MODEL_{config_field.name.upper()}"
        raw_value = os.environ.get(env_name)
        if raw_value is None or raw_value == "":
            continue
        overrides[config_field.name] = _coerce_env_config_value(
            raw_value,
            getattr(default_config, config_field.name),
        )

    model, model_config = create_model(model_variant, **overrides)
    return model, model_config


def build_data_config_from_env() -> DataConfig:
    return DataConfig(
        data_format=os.environ.get("RECONSTRUCTION_DATA_FORMAT", DataConfig.data_format),
        local_data_path=os.environ.get(
            "RECONSTRUCTION_LOCAL_DATA_PATH",
            str(DataConfig.local_data_path),
        ),
        local_cache_path=os.environ.get(
            "RECONSTRUCTION_LOCAL_CACHE_PATH",
            DataConfig.local_cache_path,
        ),
        max_seq_len=_get_env_int("RECONSTRUCTION_MAX_SEQ_LEN", DataConfig.max_seq_len),
        energies=_get_env_int_list("RECONSTRUCTION_ENERGIES"),
        max_h5_files_per_energy_recoil=_get_env_optional_int(
            "RECONSTRUCTION_MAX_H5_FILES_PER_ENERGY_RECOIL"
        ),
        expected_h5_events_per_file=_get_env_optional_int(
            "RECONSTRUCTION_EXPECTED_H5_EVENTS_PER_FILE"
        ),
        max_open_h5_files=_get_env_int(
            "RECONSTRUCTION_MAX_OPEN_H5_FILES",
            DataConfig.max_open_h5_files,
        ),
        train_split=_get_env_float("RECONSTRUCTION_TRAIN_SPLIT", DataConfig.train_split),
        val_split=_get_env_float("RECONSTRUCTION_VAL_SPLIT", DataConfig.val_split),
    )


def resolve_amp(device: torch.device, requested: str) -> tuple[torch.dtype, torch.amp.GradScaler]:
    capability = torch.cuda.get_device_capability(device)
    if capability < (7, 0):
        raise RuntimeError(
            f"GPU compute capability {capability} is unsupported; require Volta or newer"
        )

    requested = requested.lower()
    if requested == "auto":
        requested = "bf16" if capability >= (8, 0) else "fp16"
    if requested not in {"bf16", "fp16"}:
        raise ValueError("RECONSTRUCTION_AMP_DTYPE must be auto, bf16, or fp16")
    if requested == "bf16" and capability < (8, 0):
        raise RuntimeError("BF16 requires an Ampere-or-newer GPU; use AMP_DTYPE=fp16")

    dtype = torch.bfloat16 if requested == "bf16" else torch.float16
    scaler = torch.amp.GradScaler(
        "cuda",
        init_scale=1024.0,
        enabled=dtype == torch.float16,
    )
    return dtype, scaler


def get_next_batch(dataloader, device):
    # Note the dataloader must have pinned memory for safe async memory transfers
    inputs_cpu, spatial_targets_cpu, energy_targets_cpu, recoil_types_cpu = next(dataloader)
    inputs = inputs_cpu.to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )
    spatial_targets = spatial_targets_cpu.to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )
    energy_targets = energy_targets_cpu.to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )
    class_targets_cpu = torch.tensor(
        [1.0 if str(recoil_type).upper() == "NR" else 0.0 for recoil_type in recoil_types_cpu],
        dtype=torch.float32,
    )
    class_targets = class_targets_cpu.to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )
    return inputs, spatial_targets, energy_targets, class_targets


def unpack_model_outputs(outputs):
    if not isinstance(outputs, (tuple, list)) or len(outputs) < 2:
        raise TypeError(
            "Model forward must return at least (spatial_pred, energy_pred). "
            f"Got {type(outputs)!r}."
        )
    class_logits = outputs[2] if len(outputs) >= 3 else None
    return outputs[0], outputs[1], class_logits


def select_spatial_targets(
    spatial_targets: torch.Tensor,
    spatial_logits: torch.Tensor,
    spatial_target_indices: tuple[int, ...] | None,
) -> torch.Tensor:
    if spatial_target_indices is not None:
        index = torch.tensor(
            spatial_target_indices,
            device=spatial_targets.device,
            dtype=torch.long,
        )
        return spatial_targets.index_select(dim=-1, index=index)

    if spatial_targets.size(-1) == spatial_logits.size(-1):
        return spatial_targets
    if spatial_targets.size(-1) > spatial_logits.size(-1):
        return spatial_targets[..., : spatial_logits.size(-1)]

    raise ValueError(
        "Spatial target dimension is smaller than model output dimension: "
        f"targets={spatial_targets.size(-1)}, outputs={spatial_logits.size(-1)}"
    )


def loss_weight(config: TrainingConfig, index: int, default: float = 1.0) -> float:
    if index >= len(config.scalar_loss_weights):
        return default
    return config.scalar_loss_weights[index]


def publish_wandb_checkpoint_artifact(
    model_path: Path,
    resume_path: Path,
    *,
    checkpoint_dir: str | Path,
    run_name: str,
    step: int,
    epoch: float,
) -> None:
    artifact = wandb.Artifact(
        name=f"{run_name}-checkpoint",
        type="model-checkpoint",
        metadata={"step": step, "epoch": epoch},
    )
    artifact.add_file(str(model_path), name=model_path.name)
    artifact.add_file(str(resume_path), name=resume_path.name)
    run_config_path = Path(checkpoint_dir) / "run_config.json"
    if run_config_path.exists():
        artifact.add_file(str(run_config_path), name="run_config.json")
    wandb.log_artifact(
        artifact,
        aliases=["latest", f"step-{step}", f"epoch-{epoch:.3f}"],
    )
    logger.info(
        "Published W&B checkpoint artifact %s at step %d",
        artifact.name,
        step,
    )


class CyclingDataloader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            return next(self.iterator)


# Note this function will contain other metrics later i.e. F1, accuracy etc
# when ER/NR classification is included.
@torch.no_grad()
def eval_model(model, dataloader, device, config, amp_dtype):
    model_was_training = model.training
    model.eval()
    totals = {
        "loss": 0.0,
        "spatial_mse": 0.0,
        "energy_mse": 0.0,
        "class_loss": 0.0,
        "class_correct": 0.0,
        "class_count": 0,
    }
    for _ in range(config.eval_num_batches):
        val_inputs, val_spatial_targets, val_energy_targets, val_class_targets = get_next_batch(
            dataloader,
            device,
        )
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            val_outputs = model(val_inputs)
        val_spatial_logits, val_energy_logits, val_class_logits = unpack_model_outputs(
            val_outputs
        )
        val_spatial_logits = val_spatial_logits.float()
        val_energy_logits = val_energy_logits.float()
        val_spatial_targets = select_spatial_targets(
            val_spatial_targets,
            val_spatial_logits,
            config.spatial_target_indices,
        )

        val_spatial_loss = F.mse_loss(val_spatial_logits, val_spatial_targets)
        val_energy_loss = F.mse_loss(
            val_energy_logits,
            val_energy_targets.view_as(val_energy_logits),
        )
        val_total_loss = (
            loss_weight(config, 0) * val_spatial_loss
            + loss_weight(config, 1) * val_energy_loss
        )
        totals["spatial_mse"] += val_spatial_loss.item()
        totals["energy_mse"] += val_energy_loss.item()

        if config.recoil_classification and val_class_logits is not None:
            val_class_logits = val_class_logits.float()
            val_class_loss = F.binary_cross_entropy_with_logits(
                val_class_logits,
                val_class_targets.view_as(val_class_logits),
            )
            val_total_loss = val_total_loss + loss_weight(config, 2) * val_class_loss
            val_class_pred = (torch.sigmoid(val_class_logits) >= 0.5).float()
            totals["class_loss"] += val_class_loss.item()
            totals["class_correct"] += (
                val_class_pred.eq(val_class_targets.view_as(val_class_pred)).sum().item()
            )
            totals["class_count"] += val_class_pred.numel()
        totals["loss"] += val_total_loss.item()

    divisor = float(config.eval_num_batches)
    metrics = {
        "val_loss": totals["loss"] / divisor,
        "val_spatial_rmse": math.sqrt(totals["spatial_mse"] / divisor),
        "val_energy_rmse": math.sqrt(totals["energy_mse"] / divisor),
    }
    if totals["class_count"]:
        metrics["val_class_loss"] = totals["class_loss"] / divisor
        metrics["val_class_accuracy"] = (
            totals["class_correct"] / totals["class_count"]
        )
    model.train(model_was_training)
    return metrics


def train(
    model,
    adamw_optimiser,
    adamw_scheduler,
    muon_optimiser,
    muon_scheduler,
    train_dataloader,
    val_dataloader,
    device,
    config,
    *,
    amp_dtype,
    scaler,
    model_config,
    data_config,
    start_step=0,
    checkpoint_model=None,
):
    if checkpoint_model is None:
        checkpoint_model = model
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if config.total_batch_size % (config.device_batch_size * world_size):
        raise ValueError(
            "RECONSTRUCTION_TOTAL_BATCH_SIZE must be divisible by "
            "RECONSTRUCTION_DEVICE_BATCH_SIZE * world_size"
        )
    grad_accum_steps = config.total_batch_size // (
        config.device_batch_size * world_size
    )
    steps_per_epoch = math.ceil(len(train_dataloader) / grad_accum_steps)

    is_rank_zero = not dist.is_initialized() or dist.get_rank() == 0
    if config.wandb_run and is_rank_zero:
        wandb.init(
            project=config.wandb_project_name,
            name=config.wandb_run_name,
            config={
                **_json_ready(asdict(config)),
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_compute_capability": ".".join(
                    str(part) for part in torch.cuda.get_device_capability(device)
                ),
                "amp_dtype_resolved": str(amp_dtype).removeprefix("torch."),
                "steps_per_epoch": steps_per_epoch,
            },
        )
        wandb.log({"data_staging_seconds": config.data_staging_seconds}, step=start_step)
    train_batches = CyclingDataloader(train_dataloader)
    val_batches = CyclingDataloader(val_dataloader)
    total_training_time = 0.0
    model.train()
    model.zero_grad(set_to_none=True)
    completed_step = start_step
    overflow_attempts = 0
    while completed_step < config.num_steps:
        if dist.is_initialized() and hasattr(train_dataloader, "sampler"):
            try:
                train_dataloader.sampler.set_epoch(
                    completed_step // steps_per_epoch
                )
            except Exception:
                pass
        step_start_time = time.perf_counter()
        step_start_time = time.perf_counter()
        train_sums = {
            "loss": 0.0,
            "spatial_mse": 0.0,
            "energy_mse": 0.0,
            "class_loss": 0.0,
            "class_accuracy": 0.0,
            "class_batches": 0,
        }
        for _ in range(grad_accum_steps):
            train_inputs, train_spatial_targets, train_energy_targets, train_class_targets = (
                get_next_batch(train_batches, device)
            )
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                train_outputs = model(train_inputs)
            train_spatial_logits, train_energy_logits, train_class_logits = unpack_model_outputs(
                train_outputs
            )
            train_spatial_logits = train_spatial_logits.float()
            train_energy_logits = train_energy_logits.float()
            train_spatial_targets_for_loss = select_spatial_targets(
                train_spatial_targets,
                train_spatial_logits,
                config.spatial_target_indices,
            )
            spatial_loss = F.mse_loss(
                train_spatial_logits,
                train_spatial_targets_for_loss,
                reduction="mean",
            )
            energy_loss = F.mse_loss(
                train_energy_logits,
                train_energy_targets.view_as(train_energy_logits),
                reduction="mean",
            )
            unscaled_loss = (
                loss_weight(config, 0) * spatial_loss
                + loss_weight(config, 1) * energy_loss
            )
            train_class_loss = None
            train_class_accuracy = None
            if config.recoil_classification and train_class_logits is not None:
                train_class_logits = train_class_logits.float()
                train_class_loss = F.binary_cross_entropy_with_logits(
                    train_class_logits,
                    train_class_targets.view_as(train_class_logits),
                    reduction="mean",
                )
                unscaled_loss = (
                    unscaled_loss + loss_weight(config, 2) * train_class_loss
                )
                train_class_pred = (torch.sigmoid(train_class_logits) >= 0.5).float()
                train_class_accuracy = (
                    train_class_pred.eq(train_class_targets.view_as(train_class_pred))
                    .float()
                    .mean()
                    .item()
                )
            train_sums["loss"] += unscaled_loss.detach().item()
            train_sums["spatial_mse"] += spatial_loss.detach().item()
            train_sums["energy_mse"] += energy_loss.detach().item()
            if train_class_loss is not None:
                train_sums["class_loss"] += train_class_loss.detach().item()
                train_sums["class_accuracy"] += train_class_accuracy
                train_sums["class_batches"] += 1

            scaler.scale(unscaled_loss / grad_accum_steps).backward()

        scaler.unscale_(adamw_optimiser)
        if muon_optimiser is not None:
            scaler.unscale_(muon_optimiser)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        grad_norm_cpu = grad_norm.item()

        scale_before_step = scaler.get_scale()
        scaler.step(adamw_optimiser)
        if muon_optimiser is not None:
            scaler.step(muon_optimiser)
        scaler.update()
        scale_after_step = scaler.get_scale()
        optimizer_step_succeeded = (
            not scaler.is_enabled() or scale_after_step >= scale_before_step
        )
        if optimizer_step_succeeded:
            adamw_scheduler.step()
            if muon_scheduler is not None:
                muon_scheduler.step()
        model.zero_grad(set_to_none=True)

        train_total_loss = train_sums["loss"] / grad_accum_steps
        train_spatial_rmse = math.sqrt(train_sums["spatial_mse"] / grad_accum_steps)
        train_energy_rmse = math.sqrt(train_sums["energy_mse"] / grad_accum_steps)
        train_class_loss_value = (
            train_sums["class_loss"] / train_sums["class_batches"]
            if train_sums["class_batches"]
            else None
        )
        train_class_accuracy_value = (
            train_sums["class_accuracy"] / train_sums["class_batches"]
            if train_sums["class_batches"]
            else None
        )

        step_end_time = time.perf_counter()
        step_duration = step_end_time - step_start_time
        total_training_time += step_duration
        if not optimizer_step_succeeded:
            overflow_attempts += 1
            logger.warning(
                "AMP overflow before step %d; reduced scale from %.1f to %.1f",
                completed_step + 1,
                scale_before_step,
                scale_after_step,
            )
            if config.wandb_run:
                wandb.log(
                    {
                        "amp_scale": scale_after_step,
                        "amp_step_skipped": 1,
                        "amp_overflow_attempts": overflow_attempts,
                    },
                    step=completed_step,
                )
            continue

        completed_step += 1
        epoch = completed_step / steps_per_epoch
        final_step = completed_step == config.num_steps
        if final_step or completed_step % config.eval_step_period == 0:
            val_metrics = {}
            if is_rank_zero:
                val_metrics = eval_model(
                    model,
                    val_batches,
                    device,
                    config,
                    amp_dtype,
                )
            val_class_text = (
                f" Val class accuracy: {val_metrics['val_class_accuracy']:.4f} |"
                if "val_class_accuracy" in val_metrics
                else ""
            )
            if is_rank_zero:
                logger.info(
                    f"Step: {completed_step:d} |"
                    f" Epoch: {epoch:.3f} |"
                    f" Training loss: {train_total_loss:.4f} |"
                    f" Train spatial RMSE: {train_spatial_rmse:.4f} |"
                    f" Train energy RMSE: {train_energy_rmse:.4f} |"
                    f" Validation loss: {val_metrics['val_loss']:.4f} |"
                    f" Val spatial RMSE: {val_metrics['val_spatial_rmse']:.4f} |"
                    f" Val energy RMSE: {val_metrics['val_energy_rmse']:.4f} |"
                    f"{val_class_text}"
                    f" Grad norm: {grad_norm_cpu:.4f} |"
                    f" Step duration: {step_duration:.2f} s"
                )
                if config.wandb_run:
                    log_payload = {
                        "step": completed_step,
                        "epoch": epoch,
                        "train_loss": train_total_loss,
                        "train_spatial_rmse": train_spatial_rmse,
                        "train_energy_rmse": train_energy_rmse,
                        "grad_norm": grad_norm_cpu,
                        "step_duration": step_duration,
                        "total_training_time": total_training_time,
                        "amp_scale": scale_after_step,
                        "amp_step_skipped": int(not optimizer_step_succeeded),
                        **val_metrics,
                    }
                    if train_class_loss_value is not None:
                        log_payload["train_class_loss"] = train_class_loss_value
                    if train_class_accuracy_value is not None:
                        log_payload["train_class_accuracy"] = train_class_accuracy_value
                    wandb.log(log_payload)
        else:
            if is_rank_zero:
                logger.info(
                    f"Step: {completed_step:d} |"
                    f" Epoch: {epoch:.3f} |"
                    f" Training loss: {train_total_loss:.4f} |"
                    f" Train spatial RMSE: {train_spatial_rmse:.4f} |"
                    f" Train energy RMSE: {train_energy_rmse:.4f} |"
                    f" Grad norm: {grad_norm_cpu:.4f} |"
                    f" Step duration: {step_duration:.2f} s"
                )
        if final_step or completed_step % config.save_checkpoint_period == 0:
            if is_rank_zero:
                model_path, resume_path = save_checkpoint(
                    config.checkpoint_dir,
                    model_state_dict=checkpoint_model.state_dict(),
                    adamw_state_dict=adamw_optimiser.state_dict(),
                    muon_state_dict=(
                        muon_optimiser.state_dict() if muon_optimiser is not None else None
                    ),
                    adamw_scheduler_state_dict=adamw_scheduler.state_dict(),
                    muon_scheduler_state_dict=(
                        muon_scheduler.state_dict() if muon_scheduler is not None else None
                    ),
                    scaler_state_dict=scaler.state_dict(),
                    step=completed_step,
                    epoch=epoch,
                    model_variant=config.model_variant,
                    model_config=_json_ready(asdict(model_config)),
                    training_config=_json_ready(asdict(config)),
                    data_config=_json_ready(asdict(data_config)),
                    remote_directory=config.remote_checkpoint_dir,
                )
                if config.wandb_run and config.wandb_checkpoint_artifacts:
                    publish_wandb_checkpoint_artifact(
                        model_path,
                        resume_path,
                        checkpoint_dir=config.checkpoint_dir,
                        run_name=config.wandb_run_name,
                        step=completed_step,
                        epoch=epoch,
                    )
            if dist.is_initialized():
                dist.barrier()
    if config.wandb_run:
        wandb.finish()
    logger.info("Training finished!")


def main():
    use_ddp, local_rank = init_ddp()
    assert torch.cuda.is_available(), "No CUDA device detected!"
    device = torch.device("cuda", local_rank) if use_ddp else torch.device("cuda")
    training_config = build_training_config_from_env()
    if use_ddp and dist.get_rank() != 0:
        training_config.wandb_run = False
    set_seed(training_config.seed)
    torch.set_float32_matmul_precision("high")
    amp_dtype, scaler = resolve_amp(device, training_config.amp_dtype)

    model_variant = training_config.model_variant
    model_entry = get_model_entry(model_variant)
    logger.info("Using model variant: %s (%s)", model_variant, model_entry.description)
    logger.info(
        "GPU: %s, capability: %s, AMP dtype: %s",
        torch.cuda.get_device_name(device),
        torch.cuda.get_device_capability(device),
        amp_dtype,
    )
    model, model_config = build_model_from_env(model_variant)
    model.to(device)
    if hasattr(model, "init_weights"):
        model.init_weights()
    base_model = model
    num_trainable_params = count_model_params(base_model, trainable_only=True)
    logger.info(f"Number of trainable parameters: {num_trainable_params:,}")

    checkpoint_dir = Path(training_config.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    data_config = build_data_config_from_env()
    dataloaders = create_dataloaders(
        data_config,
        batch_size=training_config.device_batch_size,
        num_workers=training_config.num_workers,
        distributed=dist.is_initialized(),
    )
    grad_accum_steps = (
        training_config.total_batch_size // training_config.device_batch_size
    )
    if grad_accum_steps < 1 or (
        training_config.total_batch_size % training_config.device_batch_size
    ):
        raise ValueError(
            "RECONSTRUCTION_TOTAL_BATCH_SIZE must be divisible by "
            "RECONSTRUCTION_DEVICE_BATCH_SIZE"
        )
    steps_per_epoch = math.ceil(len(dataloaders["train"]) / grad_accum_steps)
    if training_config.num_epochs is not None:
        training_config.num_steps = training_config.num_epochs * steps_per_epoch
    logger.info(
        "Resolved schedule: %d train samples, %d batches, %d optimizer steps/epoch, "
        "%d total steps",
        len(dataloaders["train"].dataset),
        len(dataloaders["train"]),
        steps_per_epoch,
        training_config.num_steps,
    )

    optimisers = base_model.configure_optimisers(
        adamw_lr=training_config.adamw_lr,
        adamw_betas=training_config.adamw_betas,
        adamw_weight_decay=training_config.adamw_weight_decay,
        adamw_fused=training_config.adamw_fused,
        muon_lr=training_config.muon_lr,
        muon_momentum=training_config.muon_momentum,
        nesterov=training_config.nesterov,
        ns_steps=training_config.ns_steps,
    )
    adamw_optimiser, muon_optimiser = optimisers
    adamw_scheduler = cosine_scheduler_with_linear_warmup(
        adamw_optimiser,
        num_warmup_steps=training_config.adamw_warmup_steps,
        total_steps=training_config.num_steps,
    )
    muon_scheduler = None
    if muon_optimiser is not None:
        muon_scheduler = cosine_scheduler_with_linear_warmup(
            muon_optimiser,
            num_warmup_steps=training_config.muon_warmup_steps,
            total_steps=training_config.num_steps,
        )

    logger.info("Training config: %s", asdict(training_config))
    logger.info("Model config: %s", asdict(model_config))
    logger.info("Data config: %s", asdict(data_config))
    if not dist.is_initialized() or dist.get_rank() == 0:
        save_resolved_config(
            training_config.checkpoint_dir,
            model_variant,
            model_config,
            training_config,
            data_config,
            num_trainable_params,
        )
    if dist.is_initialized():
        dist.barrier()

    start_step = 0
    if training_config.resume_checkpoint:
        resume_payload = load_resume_checkpoint(
            training_config.resume_checkpoint,
            model=base_model,
            adamw=adamw_optimiser,
            muon=muon_optimiser,
            adamw_scheduler=adamw_scheduler,
            muon_scheduler=muon_scheduler,
            scaler=scaler,
            device=device,
        )
        if resume_payload["model_variant"] != model_variant:
            raise ValueError(
                "Resume checkpoint model variant does not match requested variant: "
                f"{resume_payload['model_variant']} != {model_variant}"
            )
        start_step = int(resume_payload["step"])
        if start_step >= training_config.num_steps:
            raise ValueError(
                f"Resume step {start_step} is not below total steps "
                f"{training_config.num_steps}"
            )

    # Compile only after optimizers and resume state are attached to the base model.
    model = base_model
    if dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index],
            output_device=device.index,
        )
    elif os.environ.get("RECONSTRUCTION_DISABLE_TORCH_COMPILE") != "1":
        model = torch.compile(base_model)

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
        start_step=start_step,
        checkpoint_model=base_model,
    )


if __name__ == "__main__":
    main()
