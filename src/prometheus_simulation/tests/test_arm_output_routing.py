"""Every arm must write its own output, and be caught when it doesn't.

Prometheus' `config` is a process-wide singleton and derives `run.outfile` from
`storage_prefix` only while outfile is None. Whoever sets it first fixes it for
the whole process, so in a serial run every arm after the injection wrote to
the injection's path -- and Prometheus reported "Simulation completed
successfully" each time, because from its side nothing failed.

The parallel path hid this: each arm is a separate process with a fresh config.
So the bug was invisible exactly where the run was most likely to be launched
from, and fatal where it was easiest to debug.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from prometheus_simulation import simulate
from prometheus_simulation.physics import PhysicsParameters


class _Cfg(dict):
    def __getattr__(self, k):
        return self.setdefault(k, _Cfg())
    def __setattr__(self, k, v):
        self[k] = v


@pytest.fixture
def fake_prometheus(monkeypatch):
    """A Prometheus stand-in that reproduces the singleton outfile behaviour."""
    cfg = _Cfg()
    cfg.run.outfile = None
    written: list[Path] = []

    class _Prom:
        def sim(self):
            # upstream: derive outfile from storage_prefix ONLY when unset
            if cfg.run.get("outfile") is None:
                cfg.run.outfile = str(Path(cfg.run["storage_prefix"]) / "out.parquet")
            target = Path(cfg.run["outfile"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"PAR1" * 16)
            written.append(target)

    mod = types.ModuleType("prometheus")
    mod.Prometheus = _Prom
    mod.config = cfg
    monkeypatch.setitem(sys.modules, "prometheus", mod)
    monkeypatch.setattr(simulate, "recentre_injection", lambda *a, **k: None)
    monkeypatch.setattr(simulate, "override_medium",
                        lambda geo, medium, dst: Path(dst))
    return cfg, written


def _arm(name: str) -> dict:
    return {"arm": name, "geofile": "orca.geo", "medium": "water",
            "recentre_delta_m": [0.0, 0.0, 0.0], "photon_seed": 1,
            "role": "geometry"}


def test_each_arm_writes_into_its_own_directory(tmp_path, fake_prometheus):
    cfg, written = fake_prometheus
    params = PhysicsParameters()
    out = tmp_path / "run"
    # simulate a prior injection having fixed outfile for the process
    cfg.run.outfile = str(out / "_injection" / "1337_photons.parquet")

    for name in ("flower_s", "flower_l", "flower_xl"):
        simulate.run_arm(_arm(name), params, tmp_path / "i.h5",
                         tmp_path / "i.lic", tmp_path, out)

    assert len(written) == 3
    assert len({w.parent for w in written}) == 3, \
        f"arms collapsed onto shared paths: {written}"
    for name in ("flower_s", "flower_l", "flower_xl"):
        assert (out / name).glob("*.parquet"), f"{name} wrote nothing of its own"
    assert not (out / "_injection" / "1337_photons.parquet").exists(), \
        "an arm overwrote the injection's output"


def test_misrouted_output_raises_instead_of_reporting_success(tmp_path,
                                                              monkeypatch):
    """The gate itself: a run that writes elsewhere must fail loudly."""
    cfg = _Cfg()
    elsewhere = tmp_path / "elsewhere"

    class _Prom:
        def sim(self):
            elsewhere.mkdir(parents=True, exist_ok=True)
            (elsewhere / "stray.parquet").write_bytes(b"PAR1")

    mod = types.ModuleType("prometheus")
    mod.Prometheus = _Prom
    mod.config = cfg
    monkeypatch.setitem(sys.modules, "prometheus", mod)
    monkeypatch.setattr(simulate, "recentre_injection", lambda *a, **k: None)
    monkeypatch.setattr(simulate, "override_medium",
                        lambda geo, medium, dst: Path(dst))

    with pytest.raises(RuntimeError, match="wrote no non-empty parquet"):
        simulate.run_arm(_arm("flower_s"), PhysicsParameters(),
                         tmp_path / "i.h5", tmp_path / "i.lic",
                         tmp_path, tmp_path / "run")


def test_empty_parquet_is_not_accepted(tmp_path, monkeypatch):
    cfg = _Cfg()

    class _Prom:
        def sim(self):
            d = Path(cfg.run["storage_prefix"])
            d.mkdir(parents=True, exist_ok=True)
            (d / "out.parquet").write_bytes(b"")      # zero bytes

    mod = types.ModuleType("prometheus")
    mod.Prometheus = _Prom
    mod.config = cfg
    monkeypatch.setitem(sys.modules, "prometheus", mod)
    monkeypatch.setattr(simulate, "recentre_injection", lambda *a, **k: None)
    monkeypatch.setattr(simulate, "override_medium",
                        lambda geo, medium, dst: Path(dst))

    with pytest.raises(RuntimeError):
        simulate.run_arm(_arm("flower_s"), PhysicsParameters(),
                         tmp_path / "i.h5", tmp_path / "i.lic",
                         tmp_path, tmp_path / "run")
