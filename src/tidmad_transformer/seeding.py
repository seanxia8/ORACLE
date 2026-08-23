"""Seed control for the TIDMAD arms.

Inlined rather than imported from the former ``reconstruction_model.train``: that module
pulls in the detector dataset stack, which this package does not use and which
has been removed. The behaviour is identical to the original.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and Torch (CPU and CUDA) from one value.

    Also pins the cuDNN/cuBLAS execution paths so two arms launched with the
    same seed follow bitwise-paired trajectories on GPU. Without these flags
    ``F.scaled_dot_product_attention``'s backward (non-deterministic atomics)
    and cuDNN autotuning break the pairing even under identical seeds.
    ``warn_only=True`` keeps ops without a deterministic implementation from
    aborting the run; any such op is reported once via a warning so the
    provenance record shows whether the pairing was exact.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
