"""The analysis gates must fail on missing evidence, not pass.

This pins the fix for a defect the remote agent found: the parallel run path
omitted `vertex_residual_max_m` from plan.json, and the gate read
`(vres is None) or (vres < 1e-6)` -- so running in parallel made
`vertices_on_points` vacuously true. The gate was strongest on the path least
likely to be used and absent on the one most likely.
"""

from __future__ import annotations

import pandas as pd

from prometheus_simulation.analyze import evaluate_gates


def _pairing(all_paired: bool = True) -> pd.DataFrame:
    return pd.DataFrame({"arm": ["a", "b"], "paired": [True, all_paired]})


def test_missing_residual_fails_rather_than_passes():
    gates, msgs = evaluate_gates(_pairing(), {})
    assert gates["vertices_on_points"] is False
    assert any("no vertex_residual_max_m" in m for m in msgs)


def test_recorded_small_residual_passes():
    gates, msgs = evaluate_gates(_pairing(), {"vertex_residual_max_m": 1e-12})
    assert gates["vertices_on_points"] is True
    assert msgs == []


def test_recorded_large_residual_fails():
    gates, msgs = evaluate_gates(_pairing(), {"vertex_residual_max_m": 0.5})
    assert gates["vertices_on_points"] is False
    assert any("residual 0.5" in m for m in msgs)


def test_unpaired_arm_fails_and_is_named():
    gates, msgs = evaluate_gates(_pairing(all_paired=False),
                                 {"vertex_residual_max_m": 0.0})
    assert gates["all_arms_paired"] is False
    assert any("['b']" in m for m in msgs)
