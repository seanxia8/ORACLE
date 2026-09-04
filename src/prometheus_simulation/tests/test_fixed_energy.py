"""The injection must reach LeptonInjector's monoenergetic branch.

Two defects are pinned here, both found only by executing `Prometheus().sim()`
for the first time -- nothing else in the suite calls it, which is exactly why
they survived to first contact.

1. Prometheus' `check_consistency` rejects `min_e >= max_e`, so it is stricter
   than the library it wraps: LeptonInjector's `SampleEnergy()` opens with
   `if(energyMinimum==energyMaximum) return energyMinimum`. `inject_once`
   therefore passes a placeholder band through `Prometheus.__init__` (where the
   check runs) and sets the exact single energy before `sim()` (where `inject()`
   re-reads it).

   The failure mode this guards against is silent, not loud: the obvious "fix"
   is to widen the energy into a narrow band, but `SampleEnergy` samples
   LOG-UNIFORMLY between min and max whenever `powerlawIndex == 1.0` -- which
   ours is -- so any band turns the exactly fixed energy into a distribution
   and nothing raises.

2. `--n-events` assigned to `n_events`, which is a derived read-only property,
   so the documented null command raised AttributeError before simulating
   anything.
"""

from __future__ import annotations

import types

import pytest

from prometheus_simulation import simulate
from prometheus_simulation.physics import PhysicsParameters


class _Sim:
    """Stand-in for config.injection.lepton_injector.simulation."""


class _FakeConfig:
    """Enough of the Prometheus config tree for inject_once to walk it."""

    def __init__(self):
        self.run = types.SimpleNamespace(
            nevents=0, random_state_seed=0, storage_prefix="")
        self.detector = types.SimpleNamespace(geo_file="")
        li = types.SimpleNamespace(inject=False, simulation=_Sim(),
                                   paths=types.SimpleNamespace())
        self.injection = types.SimpleNamespace(name="", lepton_injector=li)


def _params() -> PhysicsParameters:
    return PhysicsParameters()


# ------------------------------------------------------------- 1. energy ---
def test_energies_seen_by_the_constructor_and_by_inject_differ():
    """The constructor must see a valid band; inject() must see min == max."""
    params = _params()
    E = float(params.energy_gev)
    sim = _Sim()
    seen_at_construction = {}

    def fake_prometheus():
        # check_consistency runs here, so record what it would have validated
        seen_at_construction["min"] = sim.minimal_energy
        seen_at_construction["max"] = sim.maximal_energy
        return types.SimpleNamespace(sim=lambda: None)

    # Mirror inject_once's two-step exactly.
    sim.minimal_energy = E
    sim.maximal_energy = E * 2.0
    fake_prometheus()
    sim.minimal_energy = sim.maximal_energy = E

    # What Prometheus validated: a legal band (min < max), so no ValueError.
    assert seen_at_construction["min"] < seen_at_construction["max"], (
        "check_consistency rejects min >= max; the constructor must see a band")
    # What LeptonInjector actually samples from: exactly one energy.
    assert sim.minimal_energy == sim.maximal_energy == E, (
        "inject() must see min == max so SampleEnergy takes its monoenergetic "
        "branch; a band would sample log-uniformly at power_law == 1.0")


def test_inject_once_leaves_the_config_monoenergetic(monkeypatch, tmp_path):
    """End-to-end over inject_once itself, with Prometheus stubbed out."""
    params = _params()
    E = float(params.energy_gev)
    cfg = _FakeConfig()
    at_construction = {}

    class _FakeProm:
        def __init__(self):
            s = cfg.injection.lepton_injector.simulation
            at_construction["min"] = s.minimal_energy
            at_construction["max"] = s.maximal_energy
            if s.minimal_energy >= s.maximal_energy:
                raise ValueError(
                    f"injection minimal energy ({s.minimal_energy}) must be < "
                    f"maximal energy ({s.maximal_energy})")

        def sim(self):
            pass

    fake_mod = types.ModuleType("prometheus")
    fake_mod.Prometheus = _FakeProm
    fake_mod.config = cfg
    monkeypatch.setitem(__import__("sys").modules, "prometheus", fake_mod)

    plan_d = {"reference_geometry": "flower_xl",
              "common_region": {"max_radius_m": 57.7, "max_abs_z_m": 95.4}}

    # inject_once looks for the produced .h5 afterwards; we only care about the
    # config it leaves behind, so let the FileNotFoundError through.
    with pytest.raises(FileNotFoundError):
        simulate.inject_once(params, plan_d, simulate.default_geodir(),
                             tmp_path / "_injection")

    assert at_construction["min"] < at_construction["max"], (
        "the constructor must be handed a legal band or check_consistency raises")
    s = cfg.injection.lepton_injector.simulation
    assert s.minimal_energy == s.maximal_energy == E, (
        "after inject_once the config LeptonInjector reads must be monoenergetic")


# ----------------------------------------------------------- 2. --n-events ---
def test_n_events_is_derived_and_has_no_setter():
    """The property that made the CLI raise. If this ever gains a setter the
    CLI guard below is no longer load-bearing and should be revisited."""
    params = _params()
    assert params.n_events == params.events_per_point * len(params.injection_points)
    with pytest.raises(AttributeError):
        params.n_events = 60


@pytest.mark.parametrize("n_events, expected_per_point", [(60, 20), (3, 1), (600, 200)])
def test_divisible_n_events_sets_events_per_point(n_events, expected_per_point):
    params = _params()
    npts = len(params.injection_points)
    assert n_events % npts == 0
    params.events_per_point = n_events // npts
    assert params.events_per_point == expected_per_point
    assert params.n_events == n_events


def test_indivisible_n_events_still_yields_a_balanced_set():
    """An indivisible count must never produce unequal events per point.

    simulate.main() floors to the largest equal split and says so on stdout
    rather than raising. What must hold either way is the design invariant: the
    assignment is contiguous blocks of events_per_point, so every point gets the
    same number of events and the realised total may be lower than asked.
    """
    params = _params()
    npts = len(params.injection_points)
    n_events = npts * 20 + 1                     # 61 over 3 points
    assert n_events % npts != 0

    per = max(1, n_events // npts)               # the rule main() applies
    params.events_per_point = per

    assert params.n_events == per * npts, "realised total must be an exact multiple"
    assert params.n_events < n_events, "an indivisible request must round DOWN"
    # The invariant that actually matters downstream: equal blocks per point.
    assignment = [n for n in params.injection_points for _ in range(params.events_per_point)]
    counts = {n: assignment.count(n) for n in params.injection_points}
    assert len(set(counts.values())) == 1, f"unbalanced set: {counts}"
