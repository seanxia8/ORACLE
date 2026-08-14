import os
import sys
from pathlib import Path

# Local CPU testing: Muon's torch.compile (Newton-Schulz) needs a working
# C++/OpenMP toolchain and fails on macOS. The training box keeps the default
# (compiled) path; correctness is identical either way.
os.environ.setdefault("RECONSTRUCTION_DISABLE_TORCH_COMPILE", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
