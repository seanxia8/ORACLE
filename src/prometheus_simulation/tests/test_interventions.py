# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""Each N family produces its predicted observable change — the
'what changed to check what' contract, executable."""

from __future__ import annotations

import numpy as np
import pytest

from prometheus_simulation.interventions import (
    GainDrift, HitThinning, ModuleLoss, NoiseRateScale, TimingJitter,
    apply_interventions,
)
from prometheus_simulation.response import ResponseConfig, emulate_response
from prometheus_simulation.toy import toy_track

CFG = ResponseConfig(noise_rate_per_om_us=0.001)


def _respond(ev, geometry, interventions, cfg=CFG, seed=11):
    ph, cfg2, log = apply_interventions(ev, geometry, cfg, interventions)
    out = emulate_response(ph, geometry, cfg2, rng=seed)
    for iv in interventions:
        out = iv.apply_pulses(out, geometry)
    return out, log


def test_module_loss_removes_dropped_oms(geometry):
    ev = toy_track(0, geometry, energy_gev=3e3, rng=4)
    iv = ModuleLoss(fraction=0.5, seed=9)
    out, log = _respond(ev, geometry, [iv])
    dead = set(iv.dropped_oms(geometry).tolist())
    assert not dead.intersection(set(out.om_id.tolist()) )
    assert log[0]["name"] == "ModuleLoss" and log[0]["fraction"] == 0.5


def test_module_loss_by_string_kills_whole_strings(geometry):
    iv = ModuleLoss(fraction=0.25, by_string=True, seed=3)
    dead = iv.dropped_oms(geometry)
    dead_strings = np.unique(geometry.string_id[dead])
    for s in dead_strings:
        assert np.all(np.isin(np.flatnonzero(geometry.string_id == s), dead))


def test_hit_thinning_matches_module_loss_in_expectation(geometry):
    ev = toy_track(0, geometry, energy_gev=5e3, rng=6)
    clean, _ = _respond(ev, geometry, [])
    thinned, _ = _respond(ev, geometry, [HitThinning(p=0.5, seed=8)])
    ratio = thinned.total_charge_pe / clean.total_charge_pe
    assert 0.35 <= ratio <= 0.65  # ~0.5 with Poisson slack


def test_timing_jitter_inflates_time_residuals(geometry):
    ev = toy_track(0, geometry, energy_gev=5e3, rng=10)
    base_cfg = ResponseConfig(noise_rate_per_om_us=0.0, tts_merge_ns=0.0,
                              charge_smear_pe=0.0)
    clean, _ = _respond(ev, geometry, [], cfg=base_cfg)
    jit, _ = _respond(ev, geometry, [TimingJitter(sigma_ns=30.0)], cfg=base_cfg)
    # same photons, same merge -> compare per-pulse residual scatter
    assert jit.n_pulses == clean.n_pulses
    resid = np.std(np.sort(jit.t_ns) - np.sort(clean.t_ns))
    assert resid > 10.0


def test_gain_drift_changes_charge_by_string_only(geometry):
    ev = toy_track(0, geometry, energy_gev=5e3, rng=12)
    clean, _ = _respond(ev, geometry, [])
    drift, _ = _respond(ev, geometry, [GainDrift(sigma=0.5, seed=2)])
    assert clean.n_pulses == drift.n_pulses
    factors = drift.charge_pe / np.where(clean.charge_pe == 0, 1, clean.charge_pe)
    by_string = {}
    for k in range(clean.n_pulses):
        if clean.charge_pe[k] <= 0:
            continue
        s = int(geometry.string_id[clean.om_id[k]])
        by_string.setdefault(s, []).append(factors[k])
    for s, fs in by_string.items():
        assert np.std(fs) < 1e-9  # one factor per string
    assert len({round(float(np.mean(fs)), 6) for fs in by_string.values()}) > 1


def test_noise_rate_scale_raises_noise_fraction(geometry):
    ev = toy_track(0, geometry, energy_gev=2e3, rng=13)
    clean, _ = _respond(ev, geometry, [])
    noisy, _ = _respond(ev, geometry, [NoiseRateScale(factor=30.0)])
    def noise_frac(e):
        return 1.0 - float(np.mean(e.signal_fraction)) if e.n_pulses else 0.0
    assert noise_frac(noisy) > noise_frac(clean)
