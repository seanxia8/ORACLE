# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""Prometheus run-plan emission: the pairing mechanism, verbatim."""

from __future__ import annotations

import json

from prometheus_simulation.plans import (
    InjectionPlan, ProductionPlan, prometheus_config_pairs,
    tau_plan, throughgoing_muon_plan,
)


def test_pair_configs_share_one_injection_but_recentre_it():
    """The pairing mechanism, and the correction it needs.

    Every arm replays the SAME injection -- that is the pairing. But a stored
    injection sits in the first detector's absolute frame, because Prometheus
    applies the detector offset in place and only when injecting, and ignores
    it on load. So each replaying arm must read its own RECENTRED copy of that
    one injection, not the original. This test pins both halves: one source,
    per-arm destinations, and a translation that is actually non-zero.
    """
    plan = ProductionPlan()
    cfgs = prometheus_config_pairs(plan)
    assert len(cfgs) == 2
    first, second = cfgs
    assert first["injection"]["lepton_injector"]["inject"] is True
    assert second["injection"]["lepton_injector"]["inject"] is False

    f = first["injection"]["lepton_injector"]["paths"]["injection_file"]
    s = second["injection"]["lepton_injector"]["paths"]["injection_file"]
    assert f != s, "each replaying arm needs its own recentred injection file"
    assert second["_pairing"]["source_injection_file"] == f, \
        "but it must derive from the one shared injection"

    assert first["detector"]["geo_file"] != second["detector"]["geo_file"]
    assert first["run"]["random_state_seed"] == second["run"]["random_state_seed"]
    assert first["_pairing"]["recentre_delta_m"] == [0.0, 0.0, 0.0]

    # nothing else differs
    a = json.dumps({**first, "detector": {}, "injection": {}, "_pairing": {}},
                   sort_keys=True)
    b = json.dumps({**second, "detector": {}, "injection": {}, "_pairing": {}},
                   sort_keys=True)
    assert a == b


def test_replay_recentre_delta_is_non_zero_for_offset_geometries():
    """The regression guard for the defect this replaces.

    ORCA is centred at z = +95 m and ARCA at z = -3194 m. If the delta comes
    back as zeros for that pair, the recentring has silently stopped working
    and every replayed arm will produce no hits.
    """
    cfgs = prometheus_config_pairs(ProductionPlan())
    delta = cfgs[1]["_pairing"]["recentre_delta_m"]
    assert len(delta) == 3
    assert abs(delta[2]) > 1000.0, (
        f"expected a ~3 km z-translation between ORCA and ARCA, got {delta}")


def test_u_family_plans_change_only_what_they_declare():
    base = InjectionPlan()
    tau = tau_plan(base)
    assert tau.final_state_1 == "TauMinus"
    assert tau.minimal_energy_gev == base.minimal_energy_gev
    mu = throughgoing_muon_plan(base)
    assert mu.cylinder_radius_m > base.cylinder_radius_m
    assert mu.final_state_1 == base.final_state_1


def test_plan_serialises(tmp_path):
    path = ProductionPlan().to_json(tmp_path / "plan.json")
    payload = json.loads(path.read_text())
    assert payload["geometries"][0]["geo_file"] == "orca.geo"
    assert payload["response"]["quantum_efficiency"] == 0.20
    assert payload["overproduction_factor"] == 4.0
