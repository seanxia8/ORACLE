# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""Detector geometry: OM positions and string membership."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DetectorGeometry:
    """Optical-module positions and string ids.

    positions_m : (n_om, 3) float — x, y, z in metres.
    string_id   : (n_om,) int — which string each OM belongs to.
    """

    positions_m: np.ndarray
    string_id: np.ndarray

    def __post_init__(self):
        p = np.asarray(self.positions_m, dtype=float)
        s = np.asarray(self.string_id, dtype=int)
        if p.ndim != 2 or p.shape[1] != 3:
            raise ValueError("positions_m must be (n_om, 3).")
        if s.shape != (p.shape[0],):
            raise ValueError("string_id must have one entry per OM.")
        object.__setattr__(self, "positions_m", p)
        object.__setattr__(self, "string_id", s)

    @property
    def n_om(self) -> int:
        return self.positions_m.shape[0]

    @property
    def strings(self) -> np.ndarray:
        return np.unique(self.string_id)

    def boundary_distance_m(self, points_m: np.ndarray) -> np.ndarray:
        """Distance from each point to the instrumented volume's bounding
        cylinder wall (positive inside, negative outside).

        A cheap containment proxy for the edge-vertex stratum: the cylinder is
        the (x, y) circumradius and z range of the OM positions.
        """
        pts = np.atleast_2d(np.asarray(points_m, dtype=float))
        center_xy = self.positions_m[:, :2].mean(axis=0)
        radius = np.max(np.linalg.norm(self.positions_m[:, :2] - center_xy, axis=1))
        z_lo, z_hi = self.positions_m[:, 2].min(), self.positions_m[:, 2].max()
        radial = radius - np.linalg.norm(pts[:, :2] - center_xy, axis=1)
        vertical = np.minimum(pts[:, 2] - z_lo, z_hi - pts[:, 2])
        return np.minimum(radial, vertical)

    @classmethod
    def from_geo_rows(cls, rows: np.ndarray) -> "DetectorGeometry":
        """Build from (n_om, 4) rows of x, y, z, string_id — the layout of a
        Prometheus .geo table once comments are stripped."""
        rows = np.asarray(rows, dtype=float)
        return cls(rows[:, :3], rows[:, 3].astype(int))

    @classmethod
    def demo_grid(
        cls, n_strings_side: int = 3, oms_per_string: int = 10,
        spacing_m: float = 100.0, vertical_spacing_m: float = 20.0,
    ) -> "DetectorGeometry":
        """A small synthetic square-grid detector for tests and tutorials."""
        xs = np.arange(n_strings_side) * spacing_m
        positions, strings = [], []
        sid = 0
        for x in xs:
            for y in xs:
                for k in range(oms_per_string):
                    positions.append((x, y, -k * vertical_spacing_m))
                    strings.append(sid)
                sid += 1
        return cls(np.asarray(positions, dtype=float), np.asarray(strings))
