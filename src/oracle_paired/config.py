# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""Production plans and Prometheus run configurations.

This module emits the *configuration* for a paired Prometheus production; it
never imports Prometheus. The cluster runs Prometheus from the emitted JSON;
this package then consumes the photon output through the response stage.

Pairing mechanism (verified against the Prometheus source and its paper,
sec. 5.4.2): run LeptonInjector once, then for every further geometry set
injection.lepton_injector.inject = False and point
injection.lepton_injector.paths.injection_file at the first run's file,
changing only detector.geo_file. run.random_state_seed seeds
injection and propagation. Water geometries (olympus) replay exactly;
ice (PPC) is seedable only to Poisson-level photon variation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .response import ResponseConfig


@dataclass(frozen=True)
class InjectionPlan:
    """One seeded LeptonInjector run, shared by every geometry."""

    nevents: int = 120_000
    seed: int = 20260831
    final_state_1: str = "MuMinus"
    final_state_2: str = "Hadrons"
    minimal_energy_gev: float = 1e1
    maximal_energy_gev: float = 1e4
    power_law: float = 1.0
    min_zenith_deg: float = 0.0
    max_zenith_deg: float = 180.0
    #: size the cylinder against the LARGEST geometry so every detector is
    #: contained (D1 caveat b); metres.
    cylinder_radius_m: float = 700.0
    cylinder_height_m: float = 1000.0


@dataclass(frozen=True)
class GeometryPlan:
    """One detector in the pair. geo_file names a Prometheus geofile."""

    name: str
    geo_file: str
    medium: str = "water"


#: The production pair: ORCA vs ARCA, both water, layout not confounded with
#: medium, olympus photon stage exactly reproducible.
DEFAULT_GEOMETRIES = (
    GeometryPlan("orca", "orca.geo"),
    GeometryPlan("arca", "arca.geo"),
)


@dataclass(frozen=True)
class ProductionPlan:
    injection: InjectionPlan = field(default_factory=InjectionPlan)
    geometries: tuple[GeometryPlan, ...] = DEFAULT_GEOMETRIES
    response: ResponseConfig = field(default_factory=ResponseConfig)
    #: overproduction of clean events for the matching pool
    overproduction_factor: float = 4.0

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        payload = {
            "injection": asdict(self.injection),
            "geometries": [asdict(g) for g in self.geometries],
            "response": asdict(self.response),
            "overproduction_factor": self.overproduction_factor,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n")
        return path


def _base_config(plan: ProductionPlan, geometry: GeometryPlan) -> dict[str, Any]:
    inj = plan.injection
    return {
        "run": {
            "random_state_seed": inj.seed,
            "nevents": inj.nevents,
        },
        "detector": {"geo_file": geometry.geo_file},
        "injection": {
            "name": "LeptonInjector",
            "lepton_injector": {
                "inject": True,
                "simulation": {
                    "final_state_1": inj.final_state_1,
                    "final_state_2": inj.final_state_2,
                    "minimal_energy": inj.minimal_energy_gev,
                    "maximal_energy": inj.maximal_energy_gev,
                    "power_law": inj.power_law,
                    "min_zenith": inj.min_zenith_deg,
                    "max_zenith": inj.max_zenith_deg,
                    "cylinder_radius": inj.cylinder_radius_m,
                    "cylinder_height": inj.cylinder_height_m,
                },
                "paths": {},
            },
        },
    }


def prometheus_config_pairs(
    plan: ProductionPlan, injection_file: str = "injection/paired_injection.h5"
) -> list[dict[str, Any]]:
    """One config dict per geometry.

    The first geometry injects and writes injection_file; every later
    geometry re-reads it with inject = False. Only detector.geo_file
    differs otherwise — that is the entire experimental manipulation.
    """
    configs = []
    for i, geom in enumerate(plan.geometries):
        cfg = _base_config(plan, geom)
        li = cfg["injection"]["lepton_injector"]
        if i == 0:
            li["paths"]["injection_file"] = injection_file
        else:
            li["inject"] = False
            li["paths"]["injection_file"] = injection_file
        configs.append(cfg)
    return configs


def throughgoing_muon_plan(base: InjectionPlan, offset_m: float = 200.0) -> InjectionPlan:
    """U2 — a separate small injection whose vertices sit OUTSIDE the
    instrumented volume, so only the outgoing muon crosses the detector:
    a bare track with no contained vertex. Implemented by enlarging the
    cylinder beyond the detector by offset_m and cutting contained
    vertices downstream."""
    from dataclasses import replace
    return replace(
        base,
        nevents=max(1, base.nevents // 10),
        seed=base.seed + 7,
        cylinder_radius_m=base.cylinder_radius_m + offset_m,
        cylinder_height_m=base.cylinder_height_m + 2 * offset_m,
    )


def tau_plan(base: InjectionPlan) -> InjectionPlan:
    """U1 — nu_tau CC double-bang: same volume, tau final state."""
    from dataclasses import replace
    return replace(
        base, nevents=max(1, base.nevents // 10), seed=base.seed + 11,
        final_state_1="TauMinus", final_state_2="Hadrons",
    )
