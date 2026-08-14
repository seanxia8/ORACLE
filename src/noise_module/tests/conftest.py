"""Shared pytest setup for the modular-noise package."""

from __future__ import annotations

import sys
from pathlib import Path


# Ensure the package is importable when tests are run from a checkout without
# an editable install (the src/ layout is one level above the package).
PACKAGE_ROOT = Path(__file__).resolve().parents[2]  # .../src
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
