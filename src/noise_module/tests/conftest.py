# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the modular noise simulator written for the ORACLE study.
# If you use this module in published work, please cite it: see CITATION.cff
# at the repository root.
"""Shared pytest setup for the modular-noise package."""

from __future__ import annotations

import sys
from pathlib import Path


# Ensure the package is importable when tests are run from a checkout without
# an editable install (the src/ layout is one level above the package).
PACKAGE_ROOT = Path(__file__).resolve().parents[2]  # .../src
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


def pytest_sessionstart(session):  # noqa: ARG001
    """Recreate the analytic reference tables if they are absent.

    ``data/Al2O3_Al_athermal/*.dat`` are regenerable build artifacts derived
    from ``noise_module.reference_budget`` rather than source data, and are
    git-ignored. Tests that read them regenerate them on demand.
    """
    from noise_module.reference_budget import AL2O3_AL_ATHERMAL, COMPONENT_FILES

    data_dir = PACKAGE_ROOT / "noise_module" / "data" / AL2O3_AL_ATHERMAL.name
    expected = [*COMPONENT_FILES.values(), "signal.dat"]
    if all((data_dir / name).exists() for name in expected):
        return
    AL2O3_AL_ATHERMAL.write_reference_asd(data_dir)
