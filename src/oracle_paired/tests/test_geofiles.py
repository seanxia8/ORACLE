# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""The six vendored NuBench geometries load and match NuBench Table 1."""

from __future__ import annotations

import numpy as np
import pytest

from oracle_paired.geometry import DetectorGeometry

# NuBench Table 1 (arXiv:2511.13111): module counts per geometry.
EXPECTED = {
    "orca": 3300, "arca": 2070, "trident": 24220,
    "pone_triangle": 60, "gvd": 288, "icecube": 5160,
}


@pytest.mark.parametrize("name,n_om", sorted(EXPECTED.items()))
def test_bundled_geometry_module_counts(name, n_om):
    g = DetectorGeometry.bundled(name)
    assert g.n_om == n_om
    assert g.positions_m.shape == (n_om, 3)
    assert g.strings.size >= 3


def test_centered_has_zero_centroid():
    g = DetectorGeometry.bundled("orca").centered()
    assert np.allclose(g.positions_m.mean(axis=0), 0.0, atol=1e-9)


def test_orca_is_denser_than_arca():
    orca = DetectorGeometry.bundled("orca")
    arca = DetectorGeometry.bundled("arca")
    assert orca.median_string_spacing_m < arca.median_string_spacing_m
    assert orca.footprint_extent_m < arca.footprint_extent_m
