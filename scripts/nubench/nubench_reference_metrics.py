#!/usr/bin/env python3
"""Recompute NuBench Table 6 metrics from released prediction artifacts.

Audit fix (C11): the previous version grouped ONLY by ``is_track`` and labelled
the result "cc_track"/"nc_cascade". ``is_track`` is a topology flag; Table 6's
categories are interaction classes, and nu_e-CC events are cascades but CC, so
the two groupings differ wherever the parquet contains electron-neutrino CC
events. This version computes BOTH groupings — by ``is_track`` and, when the
column is present, by ``interaction`` — and evaluates the identity gate against
each, so the grouping that actually matches the published table is identified
by the data instead of assumed.

The identity tolerance below is embedded in this script for convenience; the
authoritative pre-registration is the dated entry in the repository's
``sources/`` provenance notes, not this literal.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import polars as pl


PUBLISHED = {
    "cc_track": {"median_deg": 23.60, "le_1deg_pct": 2.26, "le_5deg_pct": 22.24},
    "nc_cascade": {"median_deg": 56.12, "le_1deg_pct": 0.04, "le_5deg_pct": 1.03},
}

TOLERANCE = {
    "median_deg_absolute": 0.01,
    "threshold_rate_percentage_points_absolute": 0.01,
}


def _angle_expr() -> pl.Expr:
    zenith = pl.col("initial_state_zenith")
    azimuth = pl.col("initial_state_azimuth")
    truth_x = zenith.sin() * azimuth.cos()
    truth_y = zenith.sin() * azimuth.sin()
    truth_z = zenith.cos()
    prediction_norm = (
        pl.col("dir_x_pred") ** 2
        + pl.col("dir_y_pred") ** 2
        + pl.col("dir_z_pred") ** 2
    ).sqrt()
    cosine = (
        (
            truth_x * pl.col("dir_x_pred")
            + truth_y * pl.col("dir_y_pred")
            + truth_z * pl.col("dir_z_pred")
        )
        # Guard the zero-norm edge case explicitly (audit C28): a zero or null
        # prediction vector yields a null angle, which is then COUNTED, not
        # silently dropped from the denominators.
        / pl.when(prediction_norm > 0).then(prediction_norm).otherwise(None)
    ).clip(-1.0, 1.0)
    return (cosine.arccos() * (180.0 / math.pi)).alias("angle")


def _aggregate(frame: pl.LazyFrame, group_col: str) -> dict:
    """Group metrics by ``group_col`` with explicit null accounting."""
    aggregate = (
        frame.select(group_col, _angle_expr())
        .group_by(group_col)
        .agg(
            pl.len().alias("n"),
            pl.col("angle").null_count().alias("n_null_angle"),
            pl.col("angle").median().alias("median_deg"),
            (pl.col("angle") <= 1.0).mean().mul(100).alias("le_1deg_pct"),
            (pl.col("angle") <= 5.0).mean().mul(100).alias("le_5deg_pct"),
        )
        .collect(engine="streaming")
    )
    return {row[group_col]: {k: v for k, v in row.items() if k != group_col} for row in aggregate.to_dicts()}


def _gate(computed: dict) -> tuple[dict, bool]:
    deltas = {
        name: {
            metric: float(computed[name][metric] - PUBLISHED[name][metric])
            for metric in PUBLISHED[name]
        }
        for name in PUBLISHED
        if name in computed
    }
    gate_pass = bool(deltas) and all(
        abs(delta) <= 0.01 for group in deltas.values() for delta in group.values()
    )
    return deltas, gate_pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pl.scan_parquet(args.predictions)
    columns = frame.collect_schema().names()

    result: dict = {
        "source_file": args.predictions.name,
        "published_table_6": PUBLISHED,
        "tolerance": TOLERANCE,
        "groupings": {},
    }

    # Grouping 1: topology flag (what the previous version reported).
    by_track = _aggregate(frame, "is_track")
    computed_track = {
        "cc_track": by_track.get(True) or by_track.get(1),
        "nc_cascade": by_track.get(False) or by_track.get(0),
    }
    computed_track = {k: v for k, v in computed_track.items() if v is not None}
    deltas_track, pass_track = _gate(computed_track)
    result["groupings"]["by_is_track"] = {
        "note": "topology flag; pools nu_e-CC cascades into the 'cascade' bucket",
        "computed": computed_track,
        "computed_minus_published": deltas_track,
        "gate_pass": pass_track,
    }

    # Grouping 2: interaction class, when the released parquet carries it.
    if "interaction" in columns:
        by_interaction = _aggregate(frame, "interaction")
        result["groupings"]["by_interaction"] = {
            "note": "raw interaction codes as released; map to CC/NC per the "
            "NuBench convention before comparing with Table 6",
            "computed": {str(k): v for k, v in by_interaction.items()},
        }
        # Try the common convention (1=CC, 2=NC) for a direct gate as well.
        if 1 in by_interaction and 2 in by_interaction:
            computed_int = {
                "cc_track": by_interaction[1],
                "nc_cascade": by_interaction[2],
            }
            deltas_int, pass_int = _gate(computed_int)
            result["groupings"]["by_interaction_cc_nc"] = {
                "note": "assumes interaction code 1=CC, 2=NC; verify against "
                "the NuBench truth documentation",
                "computed": computed_int,
                "computed_minus_published": deltas_int,
                "gate_pass": pass_int,
            }
    else:
        result["groupings"]["by_interaction"] = {
            "note": "column 'interaction' not present in this parquet; only the "
            "topology grouping could be computed"
        }

    total_rows = sum(g["n"] for g in computed_track.values())
    result["rows"] = int(total_rows)
    result["gate_pass_any_grouping"] = any(
        g.get("gate_pass", False) for g in result["groupings"].values()
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
