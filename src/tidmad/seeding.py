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
    """Seed Python, NumPy and Torch (CPU and CUDA) from one value."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
