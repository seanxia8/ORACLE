# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""S (supported-but-rare) selectors and U (undeclared) builders.

S families are *selections* over clean events by truth or observed content —
physically valid events from sparse regions of the training support. U
families are held entirely outside the declared contract and exist for the
abstention endpoint (C3); two of them are built here as event-level
constructions (pile-up overlay, correlated module noise), the other two
(tau double-bang, through-going muons) are injection-level and are produced
by their own Prometheus run plans in config.py.
"""

from __future__ import annotations

import numpy as np

from .events import EventPhotons
from .geometry import DetectorGeometry


# ----------------------------------------------------------------- S masks

def low_energy_mask(energies_gev: np.ndarray, quantile: float = 0.1) -> np.ndarray:
    """S1 — the natural low-energy tail (bottom quantile of true E)."""
    e = np.asarray(energies_gev, dtype=float)
    return e <= np.quantile(e, quantile)


def edge_vertex_mask(
    vertices_m: np.ndarray, geometry: DetectorGeometry, margin_m: float = 50.0
) -> np.ndarray:
    """S2 — vertices within margin_m of the instrumented boundary (or
    outside it): partial containment, hence naturally low multiplicity."""
    return geometry.boundary_distance_m(vertices_m) <= margin_m


def horizontal_mask(
    zenith_rad: np.ndarray, band_deg: float = 10.0
) -> np.ndarray:
    """S3 — near-horizon zenith band, |zenith - 90 deg| <= band."""
    z = np.degrees(np.asarray(zenith_rad, dtype=float))
    return np.abs(z - 90.0) <= band_deg


# ----------------------------------------------------------- U constructions

def overlay_pileup(
    a: EventPhotons, b: EventPhotons, offset_ns: float = 500.0,
    event_id: int | None = None,
) -> EventPhotons:
    """U3 — two events in one trigger window. b is shifted by
    offset_ns relative to a; truth keeps both parents."""
    om = np.concatenate([a.om_id, b.om_id])
    t = np.concatenate([a.t_ns - a.t_ns.min() if a.t_ns.size else a.t_ns,
                        (b.t_ns - b.t_ns.min() if b.t_ns.size else b.t_ns) + offset_ns])
    sig = np.concatenate([a.is_signal, b.is_signal])
    return EventPhotons(
        event_id if event_id is not None else a.event_id,
        om, t, sig,
        {"pileup_parents": [a.event_id, b.event_id],
         "pileup_offset_ns": offset_ns, **{f"a_{k}": v for k, v in a.truth.items()}},
    )


def inject_correlated_noise(
    photons: EventPhotons,
    geometry: DetectorGeometry,
    n_bursts: int = 3,
    oms_per_burst: int = 8,
    burst_sigma_ns: float = 50.0,
    photons_per_om: int = 3,
    window_hint_ns: float = 5000.0,
    rng: np.random.Generator | int | None = None,
) -> EventPhotons:
    """U4 — spatially correlated noise: bursts hitting a *neighbourhood* of
    OMs nearly simultaneously. No declared N family has this structure
    (N5 noise is uniform and independent), so it must route to abstention."""
    rng = np.random.default_rng(rng)
    om_list, t_list = [photons.om_id], [photons.t_ns]
    sig_list = [photons.is_signal]
    for _ in range(n_bursts):
        center = int(rng.integers(0, geometry.n_om))
        d = np.linalg.norm(
            geometry.positions_m - geometry.positions_m[center], axis=1
        )
        members = np.argsort(d)[:oms_per_burst]
        t0 = rng.uniform(0.0, window_hint_ns)
        for om_idx in members:
            k = photons_per_om
            om_list.append(np.full(k, om_idx))
            t_list.append(t0 + rng.normal(0.0, burst_sigma_ns, k))
            sig_list.append(np.zeros(k, dtype=bool))
    return EventPhotons(
        photons.event_id,
        np.concatenate(om_list),
        np.concatenate(t_list),
        np.concatenate(sig_list),
        {**photons.truth, "correlated_noise_bursts": n_bursts},
    )
