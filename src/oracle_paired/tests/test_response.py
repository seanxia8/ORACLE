# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""The response stage: determinism, QE, merging, window, cuts."""

from __future__ import annotations

import numpy as np
import pytest

from oracle_paired.events import EventPhotons
from oracle_paired.response import ResponseConfig, emulate_response
from oracle_paired.toy import toy_track


def test_deterministic_under_seed(geometry):
    ev = toy_track(0, geometry, energy_gev=2e3, rng=1)
    a = emulate_response(ev, geometry, rng=123)
    b = emulate_response(ev, geometry, rng=123)
    assert np.array_equal(a.om_id, b.om_id)
    assert np.array_equal(a.t_ns, b.t_ns)
    assert np.array_equal(a.charge_pe, b.charge_pe)


def test_qe_thins_to_expected_fraction(geometry):
    n = 200_000
    ev = EventPhotons(0, np.zeros(n, dtype=int), np.linspace(0, 1e6, n))
    cfg = ResponseConfig(noise_rate_per_om_us=0.0, tts_merge_ns=0.0,
                         time_smear_ns=0.0, charge_smear_pe=0.0)
    out = emulate_response(ev, geometry, cfg, rng=5)
    # zero merge window -> one pulse per surviving photon (distinct times)
    assert out.n_pulses == pytest.approx(cfg.quantum_efficiency * n, rel=0.02)


def test_merge_window_reduces_pulses_and_sums_charge(geometry):
    # 30 photons on one OM inside 1 ns -> one pulse of charge ~30 under a
    # 2 ns merge window with no smearing or noise.
    ev = EventPhotons(0, np.zeros(30, dtype=int), np.linspace(0.0, 0.9, 30))
    cfg = ResponseConfig(quantum_efficiency=1.0, noise_rate_per_om_us=0.0,
                         time_smear_ns=0.0, charge_smear_pe=0.0)
    out = emulate_response(ev, geometry, cfg, rng=0)
    assert out.n_pulses == 1
    assert out.charge_pe[0] == pytest.approx(30.0)
    assert out.signal_fraction[0] == 1.0


def test_trigger_window_is_at_least_5us(geometry):
    ev = toy_track(0, geometry, energy_gev=500, rng=2)
    out = emulate_response(ev, geometry, rng=3)
    assert out.window_ns >= 5_000.0


def test_noise_pulses_marked_and_rate_scales(geometry):
    ev = EventPhotons(0, np.zeros(0, dtype=int), np.zeros(0))
    lo = emulate_response(
        ev, geometry, ResponseConfig(noise_rate_per_om_us=0.005), rng=7)
    hi = emulate_response(
        ev, geometry, ResponseConfig(noise_rate_per_om_us=0.05), rng=7)
    assert hi.n_pulses > lo.n_pulses
    if lo.n_pulses:
        assert float(np.max(lo.signal_fraction)) == 0.0


def test_min_pulse_cut(geometry):
    ev = EventPhotons(0, np.zeros(0, dtype=int), np.zeros(0))
    out = emulate_response(
        ev, geometry, ResponseConfig(noise_rate_per_om_us=0.0), rng=0)
    assert out.n_pulses == 0 and not out.passed_cuts
