"""--n-events and --arms must work, because every documented run uses them.

n_events became a derived, read-only property when the design moved to fixed
injection points, but the CLI kept assigning to it. Every command in
AGENT_PROMPT.md that passes --n-events (including the null run, which is the
gate everything else depends on) died with
`AttributeError: can't set attribute 'n_events'` before simulating anything.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "prometheus_simulation.simulate", *args],
        cwd=REPO, capture_output=True, text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(REPO / "src")})


def test_n_events_sets_events_per_point(tmp_path):
    r = _run("--out", str(tmp_path), "--n-events", "30")
    assert r.returncode == 0, r.stderr
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert plan["n_events"] == 30
    assert plan["events_per_point"] == 10
    assert sorted(set(plan["event_point_assignment"])) == ["centre", "radial", "vertical"]


def test_indivisible_n_events_rounds_down_and_says_so(tmp_path):
    r = _run("--out", str(tmp_path), "--n-events", "20")
    assert r.returncode == 0, r.stderr
    assert "not divisible" in r.stdout
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert plan["n_events"] == 18


def test_unknown_arm_is_rejected_with_the_valid_list(tmp_path):
    r = _run("--out", str(tmp_path), "--arms", "not_an_arm")
    assert r.returncode != 0
    assert "unknown arm" in (r.stderr + r.stdout)
    assert "hexagon" in (r.stderr + r.stdout)


def test_arm_subset_keeps_the_full_plan(tmp_path):
    """A subset run must not shrink the plan: the injection stays the full one,
    so a later full run is still paired with what the subset produced."""
    r = _run("--out", str(tmp_path), "--arms", "flower_s,flower_l")
    assert r.returncode == 0, r.stderr
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert len(plan["arms"]) == 8
    assert plan["n_events"] == 600
