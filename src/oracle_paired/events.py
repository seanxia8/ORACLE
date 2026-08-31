# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""In-memory event representations: photons in, pulses out."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class EventPhotons:
    """Photon arrivals at OMs for one event, before detector response.

    om_id     : (n_photon,) int — index into the geometry's OM table.
    t_ns      : (n_photon,) float — arrival times.
    is_signal : (n_photon,) bool — physics photon (True) or noise (False).
    truth     : free-form truth record (energy, direction, vertex, ...).
    """

    event_id: int
    om_id: np.ndarray
    t_ns: np.ndarray
    is_signal: np.ndarray | None = None
    truth: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.om_id = np.asarray(self.om_id, dtype=int)
        self.t_ns = np.asarray(self.t_ns, dtype=float)
        if self.om_id.shape != self.t_ns.shape:
            raise ValueError("om_id and t_ns must have equal length.")
        if self.is_signal is None:
            self.is_signal = np.ones_like(self.om_id, dtype=bool)
        else:
            self.is_signal = np.asarray(self.is_signal, dtype=bool)

    @property
    def n_photons(self) -> int:
        return self.om_id.size


@dataclass
class EventPulses:
    """Merged, smeared pulses for one event, after detector response.

    signal_fraction is the fraction of physics photons among those merged
    into each pulse (the NuBench per-pulse feature).
    """

    event_id: int
    om_id: np.ndarray
    t_ns: np.ndarray
    charge_pe: np.ndarray
    signal_fraction: np.ndarray
    passed_cuts: bool
    window_ns: float
    truth: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_pulses(self) -> int:
        return self.om_id.size

    @property
    def total_charge_pe(self) -> float:
        return float(np.sum(self.charge_pe))

    @property
    def time_spread_ns(self) -> float:
        if self.t_ns.size < 2:
            return 0.0
        return float(np.percentile(self.t_ns, 90) - np.percentile(self.t_ns, 10))
