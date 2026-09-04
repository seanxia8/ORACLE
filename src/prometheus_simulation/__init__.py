"""Prometheus-based cross-geometry event simulation.

One physics event set, replayed through every detector geometry, so that
"same event, different geometry" is a within-event comparison rather than a
between-population one.

Public entry points:
    geometry.load_geometries   parse the Prometheus geofiles
    geometry.injection_cylinder choose the shared injection volume
    physics.PhysicsParameters   the recorded, provenance-tagged config
    simulate.build_event_set    inject once, replay everywhere
    readout.load_event_set      read the parquet output into tidy frames
    recon.reconstruct           simple geometry-free baseline reconstructions
"""

from .geometry import (Geometry, load_geo, load_geometries, injection_cylinder,
                       recentre_delta, common_region, default_geodir)
from .physics import PhysicsParameters

# --- absorbed from oracle_paired (2026-09-04) --------------------------------
# Detector response, N/S/U strata, matching and export: everything downstream
# of Prometheus. `detector.DetectorGeometry` is the OM-table abstraction the
# response stage works on, distinct from `geometry.Geometry`, which parses the
# .geo files and owns offsets and containment.
from .detector import DetectorGeometry
from .events import EventPhotons, EventPulses
from .response import ResponseConfig, emulate_response
from .plans import (GeometryPlan, InjectionPlan, ProductionPlan,
                    prometheus_config_pairs)
from .interventions import (GainDrift, HitThinning, Intervention, ModuleLoss,
                            NoiseRateScale, TimingJitter)
from .strata import (edge_vertex_mask, horizontal_mask, low_energy_mask,
                     overlay_pileup, inject_correlated_noise)
from .matching import content_features, match_clean_controls
from .export import export_parquet, provenance_record
from .toy import toy_cascade, toy_population, toy_track

__all__ = [
    "Geometry",
    "load_geo",
    "load_geometries",
    "injection_cylinder",
    "recentre_delta",
    "common_region",
    "default_geodir",
    "PhysicsParameters",
    # absorbed from oracle_paired
    "DetectorGeometry",
    "EventPhotons",
    "EventPulses",
    "ResponseConfig",
    "emulate_response",
    "GeometryPlan",
    "InjectionPlan",
    "ProductionPlan",
    "prometheus_config_pairs",
    "GainDrift",
    "HitThinning",
    "Intervention",
    "ModuleLoss",
    "NoiseRateScale",
    "TimingJitter",
    "edge_vertex_mask",
    "horizontal_mask",
    "low_energy_mask",
    "overlay_pileup",
    "inject_correlated_noise",
    "content_features",
    "match_clean_controls",
    "export_parquet",
    "provenance_record",
    "toy_cascade",
    "toy_population",
    "toy_track",
]

__version__ = "0.1.0"

PROMETHEUS_UPSTREAM = "https://github.com/Harvard-Neutrino/prometheus"  # LGPL-2.1
