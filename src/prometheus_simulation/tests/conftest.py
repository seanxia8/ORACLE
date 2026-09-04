# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""Shared fixtures: demo geometry and a toy clean population."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[2]  # .../src
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from prometheus_simulation.detector import DetectorGeometry  # noqa: E402
from prometheus_simulation.toy import toy_population  # noqa: E402


@pytest.fixture(scope="session")
def geometry() -> DetectorGeometry:
    return DetectorGeometry.demo_grid(n_strings_side=4, oms_per_string=12)


@pytest.fixture(scope="session")
def clean_events(geometry):
    return toy_population(60, geometry, seed=42)
