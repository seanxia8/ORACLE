# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""ORACLE-Paired dataset production.

One seeded Prometheus injection replayed through multiple detector
geometries, an in-project implementation of the NuBench-style detector
response (the stage where every acquisition knob lives), declared N/S/U
intervention strata, content matching, and provenance-carrying export.

Design: docs/EXPERIMENT_DESIGN.md and docs/archive/DATASET_PRODUCTION_PLAN.md.
"""

__version__ = "0.1.0"

from .config import (
    GeometryPlan,
    InjectionPlan,
    ProductionPlan,
    ResponseConfig,
    prometheus_config_pairs,
)
from .geometry import DetectorGeometry
from .events import EventPhotons, EventPulses
from .response import emulate_response
from .interventions import (
    GainDrift,
    HitThinning,
    Intervention,
    ModuleLoss,
    NoiseRateScale,
    TimingJitter,
)
from .strata import (
    edge_vertex_mask,
    horizontal_mask,
    low_energy_mask,
    overlay_pileup,
    inject_correlated_noise,
)
from .matching import content_features, match_clean_controls
from .export import export_parquet, provenance_record
from .toy import toy_cascade, toy_population, toy_track

__all__ = [
    "DetectorGeometry",
    "EventPhotons",
    "EventPulses",
    "GainDrift",
    "GeometryPlan",
    "HitThinning",
    "InjectionPlan",
    "Intervention",
    "ModuleLoss",
    "NoiseRateScale",
    "ProductionPlan",
    "ResponseConfig",
    "content_features",
    "edge_vertex_mask",
    "emulate_response",
    "export_parquet",
    "horizontal_mask",
    "inject_correlated_noise",
    "low_energy_mask",
    "match_clean_controls",
    "overlay_pileup",
    "prometheus_config_pairs",
    "provenance_record",
    "toy_cascade",
    "toy_population",
    "toy_track",
]
