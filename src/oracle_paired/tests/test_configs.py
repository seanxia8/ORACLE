# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""Prometheus run-plan emission: the pairing mechanism, verbatim."""

from __future__ import annotations

import json

from oracle_paired.config import (
    InjectionPlan, ProductionPlan, prometheus_config_pairs,
    tau_plan, throughgoing_muon_plan,
)


def test_pair_configs_differ_only_in_geometry_and_inject_flag():
    plan = ProductionPlan()
    cfgs = prometheus_config_pairs(plan)
    assert len(cfgs) == 2
    first, second = cfgs
    assert first["injection"]["lepton_injector"]["inject"] is True
    assert second["injection"]["lepton_injector"]["inject"] is False
    f = first["injection"]["lepton_injector"]["paths"]["injection_file"]
    s = second["injection"]["lepton_injector"]["paths"]["injection_file"]
    assert f == s  # same injection file: the pairing mechanism
    assert first["detector"]["geo_file"] != second["detector"]["geo_file"]
    assert first["run"]["random_state_seed"] == second["run"]["random_state_seed"]
    # nothing else differs
    a = json.dumps({**first, "detector": {}, "injection": {}}, sort_keys=True)
    b = json.dumps({**second, "detector": {}, "injection": {}}, sort_keys=True)
    assert a == b


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
