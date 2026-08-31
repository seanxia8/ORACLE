# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""Synthetic photon events for tests and the tutorial notebook.

NOT physics: a stand-in for Prometheus output with the right *shape* —
photons at OMs with times and truth records — so the response, intervention,
matching and export stages can be developed, tested and demonstrated without
a Prometheus installation. The real production replaces this module and
nothing else.
"""

from __future__ import annotations

import numpy as np

from .events import EventPhotons
from .geometry import DetectorGeometry

C_ICE_M_PER_NS = 0.2998 / 1.33  # group velocity of light in water/ice, m/ns


def toy_track(
    event_id: int,
    geometry: DetectorGeometry,
    energy_gev: float = 1e3,
    zenith_rad: float = 1.2,
    azimuth_rad: float = 0.7,
    vertex_m: np.ndarray | None = None,
    photons_per_gev: float = 0.35,
    attenuation_m: float = 60.0,
    rng: np.random.Generator | int | None = None,
) -> EventPhotons:
    """A muon-track-like event: photons on OMs near a line, times ordered
    along it. Yield scales with energy; OMs light up with probability
    falling exponentially in distance from the track."""
    rng = np.random.default_rng(rng)
    if vertex_m is None:
        vertex_m = geometry.positions_m.mean(axis=0)
    d = np.array([
        np.sin(zenith_rad) * np.cos(azimuth_rad),
        np.sin(zenith_rad) * np.sin(azimuth_rad),
        np.cos(zenith_rad),
    ])
    rel = geometry.positions_m - vertex_m
    s = rel @ d                          # distance along the track
    perp = np.linalg.norm(rel - np.outer(s, d), axis=1)
    n_target = max(1, int(rng.poisson(photons_per_gev * energy_gev)))
    weights = np.exp(-perp / attenuation_m)
    weights = weights / weights.sum()
    om_choice = rng.choice(geometry.n_om, size=n_target, p=weights)
    t = s[om_choice] / C_ICE_M_PER_NS
    t = t - t.min() + rng.normal(0.0, 2.0, n_target)
    return EventPhotons(
        event_id, om_choice, t, np.ones(n_target, dtype=bool),
        truth={
            "energy_gev": float(energy_gev),
            "zenith_rad": float(zenith_rad),
            "azimuth_rad": float(azimuth_rad),
            "vertex_x": float(vertex_m[0]), "vertex_y": float(vertex_m[1]),
            "vertex_z": float(vertex_m[2]), "interaction": "numu_cc",
            "topology": "track",
        },
    )


def toy_cascade(
    event_id: int,
    geometry: DetectorGeometry,
    energy_gev: float = 1e3,
    vertex_m: np.ndarray | None = None,
    photons_per_gev: float = 0.35,
    attenuation_m: float = 40.0,
    rng: np.random.Generator | int | None = None,
) -> EventPhotons:
    """A cascade-like event: photons on OMs around a point, times set by
    radial light travel from the vertex."""
    rng = np.random.default_rng(rng)
    if vertex_m is None:
        vertex_m = geometry.positions_m.mean(axis=0)
    r = np.linalg.norm(geometry.positions_m - vertex_m, axis=1)
    n_target = max(1, int(rng.poisson(photons_per_gev * energy_gev)))
    weights = np.exp(-r / attenuation_m)
    weights = weights / weights.sum()
    om_choice = rng.choice(geometry.n_om, size=n_target, p=weights)
    t = r[om_choice] / C_ICE_M_PER_NS + rng.normal(0.0, 2.0, n_target)
    t = t - t.min()
    return EventPhotons(
        event_id, om_choice, t, np.ones(n_target, dtype=bool),
        truth={
            "energy_gev": float(energy_gev),
            "vertex_x": float(vertex_m[0]), "vertex_y": float(vertex_m[1]),
            "vertex_z": float(vertex_m[2]), "interaction": "nue_cc",
            "topology": "cascade",
        },
    )


def toy_population(
    n: int,
    geometry: DetectorGeometry,
    track_fraction: float = 0.7,
    e_min_gev: float = 50.0,
    e_max_gev: float = 5e3,
    seed: int = 0,
) -> list[EventPhotons]:
    """A mixed clean population on a power-law-ish energy spectrum with
    uniform vertices inside the instrumented volume."""
    rng = np.random.default_rng(seed)
    lo, hi = geometry.positions_m.min(axis=0), geometry.positions_m.max(axis=0)
    events = []
    for i in range(n):
        e = float(np.exp(rng.uniform(np.log(e_min_gev), np.log(e_max_gev))))
        v = rng.uniform(lo, hi)
        if rng.random() < track_fraction:
            ev = toy_track(
                i, geometry, energy_gev=e, vertex_m=v,
                zenith_rad=float(np.arccos(rng.uniform(-1, 1))),
                azimuth_rad=float(rng.uniform(0, 2 * np.pi)),
                rng=rng,
            )
        else:
            ev = toy_cascade(i, geometry, energy_gev=e, vertex_m=v, rng=rng)
        events.append(ev)
    return events
