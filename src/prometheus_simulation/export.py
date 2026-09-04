# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""Provenance-carrying export.

Two tables per production, written as Parquet (GraphNeT's Dataset classes
read Parquet, so the released architectures run on the files unchanged):

pulse table  — one row per pulse: event_no, om position, string, time,
               charge, signal fraction.
truth table  — one row per event: injection_id (THE pairing key across
               geometries — the column no public set has), geometry,
               stratum, the exact intervention parameters as JSON, seeds,
               and the physics truth.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .events import EventPulses
from .detector import DetectorGeometry

PULSE_COLUMNS = (
    "event_no", "om_id", "dom_x", "dom_y", "dom_z", "string_id",
    "dom_time", "charge", "signal_fraction",
)
TRUTH_COLUMNS = (
    "event_no", "injection_id", "geometry", "stratum", "intervention_json",
    "response_seed", "passed_cuts", "n_pulses", "total_charge",
    "time_spread", "window_ns",
)


def provenance_record(
    event: EventPulses,
    injection_id: int,
    geometry_name: str,
    stratum: str,
    interventions: list[dict[str, Any]] | None = None,
    response_seed: int | None = None,
) -> dict[str, Any]:
    """The per-event truth row. Physics truth fields ride along verbatim."""
    rec = {
        "event_no": event.event_id,
        "injection_id": injection_id,
        "geometry": geometry_name,
        "stratum": stratum,
        "intervention_json": json.dumps(interventions or [], sort_keys=True),
        "response_seed": response_seed,
        "passed_cuts": bool(event.passed_cuts),
        "n_pulses": int(event.n_pulses),
        "total_charge": float(event.total_charge_pe),
        "time_spread": float(event.time_spread_ns),
        "window_ns": float(event.window_ns),
    }
    for key, value in event.truth.items():
        rec.setdefault(f"truth_{key}", value)
    return rec


def _pulse_rows(events: list[EventPulses], geometry: DetectorGeometry):
    rows = []
    for e in events:
        pos = geometry.positions_m[e.om_id]
        for k in range(e.n_pulses):
            rows.append((
                e.event_id, int(e.om_id[k]),
                float(pos[k, 0]), float(pos[k, 1]), float(pos[k, 2]),
                int(geometry.string_id[e.om_id[k]]),
                float(e.t_ns[k]), float(e.charge_pe[k]),
                float(e.signal_fraction[k]),
            ))
    return rows


def export_parquet(
    events: list[EventPulses],
    geometry: DetectorGeometry,
    truth_records: list[dict[str, Any]],
    out_dir: str | Path,
    prefix: str = "prometheus_simulation",
) -> tuple[Path, Path]:
    """Write <prefix>_pulses.parquet and <prefix>_truth.parquet."""
    import pandas as pd  # optional dependency, deferred

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pulses = pd.DataFrame(_pulse_rows(events, geometry), columns=PULSE_COLUMNS)
    truth = pd.DataFrame(truth_records)
    missing = [c for c in TRUTH_COLUMNS if c not in truth.columns]
    if missing:
        raise ValueError(f"Truth records missing required columns: {missing}")
    p_path = out_dir / f"{prefix}_pulses.parquet"
    t_path = out_dir / f"{prefix}_truth.parquet"
    pulses.to_parquet(p_path, index=False)
    truth.to_parquet(t_path, index=False)
    return p_path, t_path
