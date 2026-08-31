#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""Regenerate the analytic Al2O3/Al athermal reference ASD tables.

The tables under ``src/noise_module/data/Al2O3_Al_athermal/`` are build
artifacts derived from ``noise_module.reference_budget``, not source data.
See that directory's README.md for provenance.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from noise_module.reference_budget import _main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(_main())
