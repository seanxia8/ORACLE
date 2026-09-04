"""The ice arm must be re-runnable, and two ice arms must not share scratch.

Prometheus guards its PPC tmpdir with

    if tmpdir.exists() and not force: raise PpcTmpdirExistsError(...)
    tmpdir.mkdir(parents=True, exist_ok=False)

so `force=True` skips the typed guard and then hits a raw FileExistsError from
the unconditional exist_ok=False. `force` therefore does NOT make a re-run
safe, which is what run_arm assumed. It also defaults to `./.ppc_tmp`, relative
to the working directory, so concurrent ice arms would share one directory.

These tests exercise run_arm's setup up to the Prometheus call, which is
stubbed: the point is the tmpdir contract, not the simulation.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from prometheus_simulation import simulate
from prometheus_simulation.physics import PhysicsParameters


class _Cfg(dict):
    """Attribute-access stand-in for Prometheus' nested config object."""
    def __getattr__(self, k):
        return self.setdefault(k, _Cfg())
    def __setattr__(self, k, v):
        self[k] = v


@pytest.fixture
def stub_prometheus(monkeypatch):
    """Install a fake `prometheus` module and capture the config it is given."""
    seen = {}
    cfg = _Cfg()

    class _Prom:
        def __init__(self):
            pass
        def sim(self):
            paths = cfg.photon_propagator.ppc.paths
            seen["tmpdir"] = paths.get("ppc_tmpdir")
            seen["name"] = cfg.photon_propagator.get("name")
            # mirror Prometheus: mkdir with exist_ok=False, force notwithstanding
            Path(seen["tmpdir"]).mkdir(parents=True, exist_ok=False)
            # run_arm now gates on a non-empty parquet landing in the arm dir,
            # so the stub has to produce one or the gate (correctly) fires.
            out = Path(cfg.run["storage_prefix"])
            out.mkdir(parents=True, exist_ok=True)
            (out / "out.parquet").write_bytes(b"PAR1" * 16)

    mod = types.ModuleType("prometheus")
    mod.Prometheus = _Prom
    mod.config = cfg
    monkeypatch.setitem(sys.modules, "prometheus", mod)
    return seen


def _ice_arm(name: str = "hexagon_ice_le") -> dict:
    return {"arm": name, "geofile": "icecube.geo", "medium": "ice",
            "recentre_delta_m": [0.0, 0.0, 0.0], "photon_seed": 1,
            "role": "medium_control"}


def _prepare(tmp_path, monkeypatch):
    """Neutralise the file work run_arm does before touching the propagator."""
    monkeypatch.setattr(simulate, "recentre_injection", lambda *a, **k: None)
    monkeypatch.setattr(simulate, "override_medium",
                        lambda geo, medium, dst: Path(dst))
    return tmp_path / "inj.h5", tmp_path / "inj.lic"


def test_tmpdir_is_per_arm_not_shared(tmp_path, monkeypatch, stub_prometheus):
    inj, lic = _prepare(tmp_path, monkeypatch)
    params = PhysicsParameters()
    simulate.run_arm(_ice_arm(), params, inj, lic, tmp_path, tmp_path / "out")
    tmpdir = Path(stub_prometheus["tmpdir"])
    assert tmpdir.parent.name == "hexagon_ice_le", \
        "tmpdir must live under the arm's own directory, not ./.ppc_tmp"
    assert stub_prometheus["name"] == "PPC"


def test_rerun_after_a_leftover_tmpdir_succeeds(tmp_path, monkeypatch,
                                                stub_prometheus):
    """The regression guard: this is what a --arm retry after a crash does."""
    inj, lic = _prepare(tmp_path, monkeypatch)
    params = PhysicsParameters()
    out = tmp_path / "out"

    simulate.run_arm(_ice_arm(), params, inj, lic, tmp_path, out)
    leftover = Path(stub_prometheus["tmpdir"])
    assert leftover.exists(), "stub should have created it"
    (leftover / "stale.tmp").write_text("junk from the crashed run")

    # Without the rmtree in run_arm this raises FileExistsError despite force.
    simulate.run_arm(_ice_arm(), params, inj, lic, tmp_path, out)
    assert not (Path(stub_prometheus["tmpdir"]) / "stale.tmp").exists()


def test_two_ice_arms_get_different_tmpdirs(tmp_path, monkeypatch,
                                            stub_prometheus):
    inj, lic = _prepare(tmp_path, monkeypatch)
    params = PhysicsParameters()
    out = tmp_path / "out"
    simulate.run_arm(_ice_arm("ice_a"), params, inj, lic, tmp_path, out)
    first = stub_prometheus["tmpdir"]
    simulate.run_arm(_ice_arm("ice_b"), params, inj, lic, tmp_path, out)
    assert stub_prometheus["tmpdir"] != first
