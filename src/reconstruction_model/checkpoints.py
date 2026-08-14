"""Checkpoint save, resume, and optional XRootD publishing helpers."""

from __future__ import annotations

import json
import logging
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _run_remote_command(command: list[str], *, attempts: int = 3) -> None:
    for attempt in range(1, attempts + 1):
        try:
            subprocess.run(command, check=True, timeout=300)
            return
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            if attempt == attempts:
                raise
            logger.warning(
                "Remote command failed on attempt %d/%d; retrying: %s",
                attempt,
                attempts,
                " ".join(command),
            )
            time.sleep(10 * attempt)


def capture_rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _atomic_torch_save(payload: Any, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def _remote_parts(remote_directory: str) -> tuple[str, str]:
    marker = "://"
    if marker not in remote_directory:
        raise ValueError(
            "Remote checkpoint directory must be an XRootD URL, for example "
            "root://ceph-node-j.etp.kit.edu//dwong/training_runs/run-name"
        )
    scheme, remainder = remote_directory.split(marker, 1)
    host, separator, path = remainder.partition("/")
    if not separator or not path:
        raise ValueError(f"Invalid XRootD directory: {remote_directory}")
    return f"{scheme}://{host}", "/" + path.lstrip("/")


def publish_checkpoint(
    files: list[Path],
    remote_directory: str,
    *,
    step: int,
    epoch: float,
) -> None:
    server, remote_path = _remote_parts(remote_directory.rstrip("/"))
    _run_remote_command(
        ["xrdfs", server, "mkdir", "-p", remote_path],
    )

    published = {}
    for local_path in files:
        if not local_path.exists():
            continue
        remote_url = f"{server}//{remote_path.lstrip('/')}/{local_path.name}"
        _run_remote_command(
            [
                "xrdcp",
                "--nopbar",
                "--force",
                "--retry",
                "3",
                "--retry-policy",
                "continue",
                str(local_path),
                remote_url,
            ],
        )
        published[local_path.name] = remote_url

    latest_path = files[0].parent / "latest.json"
    latest_path.write_text(
        json.dumps(
            {
                "step": step,
                "epoch": epoch,
                "files": published,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    latest_url = f"{server}//{remote_path.lstrip('/')}/latest.json"
    _run_remote_command(
        ["xrdcp", "--nopbar", "--force", str(latest_path), latest_url],
    )
    logger.info("Published checkpoint step %d to %s", step, remote_directory)


def save_checkpoint(
    checkpoint_dir: str | Path,
    *,
    model_state_dict,
    adamw_state_dict,
    muon_state_dict,
    adamw_scheduler_state_dict,
    muon_scheduler_state_dict,
    scaler_state_dict,
    step: int,
    epoch: float,
    model_variant: str,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    data_config: dict[str, Any],
    remote_directory: str | None = None,
) -> tuple[Path, Path]:
    directory = Path(checkpoint_dir)
    directory.mkdir(exist_ok=True, parents=True)

    model_path = directory / f"reconstruction_model_{step}.pt"
    resume_path = directory / f"reconstruction_resume_{step}.pt"
    _atomic_torch_save(model_state_dict, model_path)
    _atomic_torch_save(
        {
            "format_version": 1,
            "step": step,
            "epoch": epoch,
            "model_variant": model_variant,
            "model_config": model_config,
            "training_config": training_config,
            "data_config": data_config,
            "model_state_dict": model_state_dict,
            "adamw_state_dict": adamw_state_dict,
            "muon_state_dict": muon_state_dict,
            "adamw_scheduler_state_dict": adamw_scheduler_state_dict,
            "muon_scheduler_state_dict": muon_scheduler_state_dict,
            "scaler_state_dict": scaler_state_dict,
            "rng_state": capture_rng_state(),
        },
        resume_path,
    )
    logger.info("Model checkpoint saved at: %s", model_path)
    logger.info("Resume checkpoint saved at: %s", resume_path)

    if remote_directory:
        run_config_path = directory / "run_config.json"
        publish_checkpoint(
            [model_path, resume_path, run_config_path],
            remote_directory,
            step=step,
            epoch=epoch,
        )
    return model_path, resume_path


def load_resume_checkpoint(
    checkpoint_path: str | Path,
    *,
    model,
    adamw,
    muon,
    adamw_scheduler,
    muon_scheduler,
    scaler,
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError(f"Unsupported resume checkpoint: {checkpoint_path}")

    model.load_state_dict(payload["model_state_dict"], strict=True)
    adamw.load_state_dict(payload["adamw_state_dict"])
    if muon is not None and payload["muon_state_dict"] is not None:
        muon.load_state_dict(payload["muon_state_dict"])
    adamw_scheduler.load_state_dict(payload["adamw_scheduler_state_dict"])
    if muon_scheduler is not None and payload["muon_scheduler_state_dict"] is not None:
        muon_scheduler.load_state_dict(payload["muon_scheduler_state_dict"])
    if scaler is not None and payload["scaler_state_dict"] is not None:
        scaler.load_state_dict(payload["scaler_state_dict"])
    restore_rng_state(payload.get("rng_state"))
    logger.info(
        "Resumed checkpoint %s at step %d (epoch %.3f)",
        checkpoint_path,
        payload["step"],
        payload["epoch"],
    )
    return payload


def load_checkpoint(model, model_path: str | Path, device: str):
    """Load a model-only inference checkpoint."""

    model_state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(model_state_dict, strict=True)
    return model
