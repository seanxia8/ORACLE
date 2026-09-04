# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""Content matching and provenance export."""

from __future__ import annotations

import json

import numpy as np
import pytest

from prometheus_simulation.export import (
    PULSE_COLUMNS, TRUTH_COLUMNS, export_parquet, provenance_record,
)
from prometheus_simulation.interventions import ModuleLoss, apply_interventions
from prometheus_simulation.matching import content_features, match_clean_controls
from prometheus_simulation.response import ResponseConfig, emulate_response
from prometheus_simulation.strata import (
    edge_vertex_mask, horizontal_mask, inject_correlated_noise,
    low_energy_mask, overlay_pileup,
)
from prometheus_simulation.toy import toy_population


def _pulses(events, geometry, interventions=(), seed=21):
    cfg = ResponseConfig(noise_rate_per_om_us=0.001)
    out = []
    for ev in events:
        ph, cfg2, _ = apply_interventions(ev, geometry, cfg, list(interventions))
        out.append(emulate_response(ph, geometry, cfg2, rng=seed + ev.event_id))
    return out


def test_matching_recovers_identical_twins(geometry, clean_events):
    pool = _pulses(clean_events, geometry)
    treated = pool[:10]  # identical members of the pool must match themselves
    res = match_clean_controls(treated, pool)
    assert res.n_matched == 10
    assert np.allclose(res.distance, 0.0)
    assert set(res.control_index.tolist()) == set(range(10))


def test_matching_respects_caliper(geometry, clean_events):
    pool = _pulses(clean_events[:20], geometry)
    X = content_features(pool)
    far = X.copy()[:3]
    far[:, 0] += 1e6  # absurd multiplicity: nothing within any sane caliper
    res = match_clean_controls(far, X, caliper=1.0)
    assert res.n_matched == 0 and res.unmatched.size == 3


def test_matched_moduleloss_events_share_observed_content(geometry, clean_events):
    clean = _pulses(clean_events, geometry)
    treated = _pulses(clean_events[:20], geometry, [ModuleLoss(fraction=0.3, seed=5)])
    res = match_clean_controls(treated, clean, caliper=1.5)
    assert res.n_matched >= 10
    Xt = content_features(treated)[res.treated_index]
    Xc = content_features(clean)[res.control_index]
    sd = content_features(clean).std(axis=0)
    assert np.all(np.abs(Xt - Xc) <= 1.5 * sd + 1e-9)


def test_strata_masks(geometry, clean_events):
    e = np.array([ev.truth["energy_gev"] for ev in clean_events])
    m1 = low_energy_mask(e, quantile=0.2)
    assert 0 < m1.sum() <= max(1, int(0.25 * e.size))
    v = np.array([[ev.truth["vertex_x"], ev.truth["vertex_y"],
                   ev.truth["vertex_z"]] for ev in clean_events])
    m2 = edge_vertex_mask(v, geometry, margin_m=40.0)
    assert m2.dtype == bool and m2.shape == (len(clean_events),)
    z = np.array([ev.truth.get("zenith_rad", 0.0) for ev in clean_events])
    m3 = horizontal_mask(z, band_deg=15.0)
    assert m3.shape == (len(clean_events),)


def test_pileup_overlay_combines_photons(geometry, clean_events):
    a, b = clean_events[0], clean_events[1]
    u3 = overlay_pileup(a, b, offset_ns=800.0, event_id=99)
    assert u3.n_photons == a.n_photons + b.n_photons
    assert u3.truth["pileup_parents"] == [a.event_id, b.event_id]


def test_correlated_noise_is_clustered(geometry, clean_events):
    u4 = inject_correlated_noise(clean_events[0], geometry, rng=3)
    added = ~u4.is_signal
    assert added.sum() > 0
    # burst photons concentrate on few OMs: unique/total well below uniform
    ratio = np.unique(u4.om_id[added]).size / added.sum()
    assert ratio < 0.6


def test_provenance_and_parquet_roundtrip(tmp_path, geometry, clean_events):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    pulses = _pulses(clean_events[:8], geometry)
    records = [
        provenance_record(p, injection_id=p.event_id, geometry_name="demo",
                          stratum="clean", interventions=[], response_seed=21)
        for p in pulses
    ]
    p_path, t_path = export_parquet(pulses, geometry, records, tmp_path)
    dfp, dft = pd.read_parquet(p_path), pd.read_parquet(t_path)
    assert list(dfp.columns) == list(PULSE_COLUMNS)
    for c in TRUTH_COLUMNS:
        assert c in dft.columns
    assert dft.shape[0] == 8
    assert dfp["event_no"].nunique() <= 8
    assert json.loads(dft["intervention_json"].iloc[0]) == []
    assert "truth_energy_gev" in dft.columns  # physics truth rode along
