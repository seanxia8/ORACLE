"""Functions for saving and loading checkpoints."""

from __future__ import annotations 
import logging
from pathlib import Path
from typing import Any

import torch
from torch.optim import Optimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_checkpoint(
    checkpoint_dir: str | Path,
    step: int,
    model_state_dict: dict[str, Any],
    adamw_state_dict: dict[str, Any],
    muon_state_dict: dict[str, Any] | None = None,
    timestamp: str | None = None,
):
    
    identifier = f"_{timestamp}" if timestamp else ""
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)

    model_checkpoint_path = directory / f"reconstruction_model{identifier}_{step}.pt"
    adamw_checkpoint_path = directory / f"reconstruction_adamw{identifier}_{step}.pt"

    torch.save(model_state_dict, model_checkpoint_path)
    logger.info(f"Model checkpoint saved at: {model_checkpoint_path}")
    torch.save(adamw_state_dict, adamw_checkpoint_path)
    logger.info(f"AdamW checkpoint saved at: {adamw_checkpoint_path}")

    if muon_state_dict is not None:
        muon_checkpoint_path = directory / f"reconstruction_muon{identifier}_{step}.pt"
        torch.save(muon_state_dict, muon_checkpoint_path)
        logger.info(f"Muon checkpoint saved at: {muon_checkpoint_path}")


def load_checkpoint(
    model,
    model_path: str | Path,
    device: str,
    adamw: Optimizer | None = None,
    adamw_path: str | Path | None = None,
    muon: Optimizer | None = None,
    muon_path: str | Path | None = None,
    is_compiled: bool = False,
):  
    model_state_dict = torch.load(model_path, map_location=device)
    # Compiled models have their parameters saved with a "_orig_mod." prefix
    # Strip this prefix if loading a non-compiled model
    if is_compiled is False:
        if any(k.startswith("_orig_mod.") for k in model_state_dict.keys()):
            model_state_dict = {k.replace("_orig_mod.", ""): v for k, v in model_state_dict.items()}
    model.load_state_dict(model_state_dict, strict=True, assign=True)
    if isinstance(adamw, Optimizer) and adamw_path is not None:
        adamw_state_dict = torch.load(adamw_path, map_location=device)
        adamw.load_state_dict(adamw_state_dict, strict=True, assign=True)
    if isinstance(muon, Optimizer) and muon_path is not None:
        muon_state_dict = torch.load(muon_path, map_location=device)
        muon.load_state_dict(muon_state_dict, strict=True, assign=True)

    return model, adamw, muon
