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

__all__ = [
    "Geometry",
    "load_geo",
    "load_geometries",
    "injection_cylinder",
    "recentre_delta",
    "common_region",
    "default_geodir",
    "PhysicsParameters",
]

__version__ = "0.1.0"

PROMETHEUS_UPSTREAM = "https://github.com/Harvard-Neutrino/prometheus"  # LGPL-2.1
