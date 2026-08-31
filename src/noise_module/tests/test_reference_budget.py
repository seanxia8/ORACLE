# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the modular noise simulator written for the ORACLE study.
# If you use this module in published work, please cite it: see CITATION.cff
# at the repository root.
"""Evidence that the supplied reference tables are analytic, not measured.

These tests are the permanent record of the provenance argument documented in
``noise_module/data/Al2O3_Al_athermal/README.md``. They do not depend on the
``.dat`` files being present: the handful of values the supplied tables
contained at the sampled indices are pinned here as golden constants, so the
tables themselves can be treated as regenerable build artifacts.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from noise_module.reference_budget import AL2O3_AL_ATHERMAL, COMPONENT_FILES
from noise_module.al2o3_athermal import FIT_FILE

BUDGET = AL2O3_AL_ATHERMAL

# Values read from the originally supplied tables at these grid indices.
GOLDEN_INDICES = [0, 4096, 8192, 12288, 16383]

GOLDEN_FREQ = [
    1.0,
    74.02772661815034,
    5480.104308251605,
    405679.66357019736,
    30000000.00000001,
]

GOLDEN_ASD = {
    "Johnson": [
        0.05505218127021067,
        0.05505218127021067,
        0.05505218127021067,
        0.05505218127021067,
        0.05505218127021067,
    ],
    "SQUID": [
        3.014962686336267,
        0.4599739873191176,
        0.30272479995239987,
        0.30003697270799146,
        0.30000049999958334,
    ],
    "TD": [
        0.2444569821346819,
        0.22165752530999655,
        0.007096745779794027,
        9.59064640316874e-05,
        1.296910168539796e-06,
    ],
    "Er": [
        0.7377207740517152,
        0.10633213345036192,
        0.015326290111105064,
        0.002209070399959679,
        0.0003185572170392047,
    ],
    "total": [
        3.1140041585143177,
        0.5244477026391795,
        0.30815305635034906,
        0.30505379336186916,
        0.30501007219924825,
    ],
}

GOLDEN_SIGNAL = [
    0.0019999605217643256,
    0.0018134278574538766,
    5.721837688129202e-05,
    6.137633875370207e-08,
    1.1257902956413918e-11,
]

#: One unit in the last place of a float64 mantissa, with a little headroom.
ULP = 4e-16

#: The signal response squares, sums and square-roots two large ratios, so its
#: deepest tail value (~1e-11 at 30 MHz) accumulates a few ulp rather than one.
SIGNAL_TOL = 1e-15


@pytest.fixture(scope="module")
def freqs() -> np.ndarray:
    return BUDGET.grid.frequencies()


# -- the grid is generated, not measured --------------------------------


def test_grid_is_geometric_to_machine_precision(freqs):
    ratios = freqs[1:] / freqs[:-1]
    assert np.allclose(ratios, ratios[0], rtol=0, atol=1e-14)
    assert freqs.size == 16384  # 2**14
    assert freqs[0] == pytest.approx(1.0, rel=ULP)
    assert freqs[-1] == pytest.approx(3.0e7, rel=ULP)


def test_grid_matches_the_supplied_frequencies(freqs):
    got = freqs[GOLDEN_INDICES]
    assert np.max(np.abs(got / np.array(GOLDEN_FREQ) - 1.0)) < ULP


# -- the components are closed-form ------------------------------------


def test_johnson_is_exactly_flat(freqs):
    johnson = BUDGET.component_asds(freqs)["Johnson"]
    assert np.all(johnson == johnson[0]), "Johnson term is white by construction"


def test_total_is_the_quadrature_sum_to_machine_epsilon(freqs):
    psds = BUDGET.component_psds(freqs)
    parts = sum(psds[k] for k in ("Johnson", "SQUID", "TD", "Er"))
    assert np.max(np.abs(parts / psds["total"] - 1.0)) < ULP


@pytest.mark.parametrize(
    "attribute,expected",
    [
        ("thermal_corner_hz", 1.0 / (2.0 * np.pi * 1.0e-3)),
        ("signal_decay_corner_hz", 1.0 / (2.0 * np.pi * 1.0e-3)),
        ("signal_rise_corner_hz", 1.0 / (2.0 * np.pi * 5.0e-6)),
    ],
)
def test_corner_frequencies_are_round_time_constants(attribute, expected):
    assert getattr(BUDGET, attribute) == pytest.approx(expected, rel=ULP)


def test_squid_knee_and_floor_are_round_design_values():
    assert BUDGET.squid_white_psd == 0.09
    assert BUDGET.squid_knee_hz == 100.0
    assert BUDGET.squid_one_over_f_psd == pytest.approx(9.0, rel=ULP)
    assert BUDGET.signal_amplitude == 2.0e-3


# -- the model reproduces the supplied tables ---------------------------


@pytest.mark.parametrize("channel", sorted(GOLDEN_ASD))
def test_reproduces_supplied_asd_to_one_ulp(freqs, channel):
    got = BUDGET.component_asds(freqs)[channel][GOLDEN_INDICES]
    assert np.max(np.abs(got / np.array(GOLDEN_ASD[channel]) - 1.0)) < ULP


def test_reproduces_supplied_signal_response(freqs):
    got = BUDGET.signal_magnitude(freqs)[GOLDEN_INDICES]
    assert np.max(np.abs(got / np.array(GOLDEN_SIGNAL) - 1.0)) < SIGNAL_TOL


def test_every_component_file_has_a_golden_reference():
    assert set(COMPONENT_FILES) == set(GOLDEN_ASD)


# -- the budget is interchangeable with the stored fit ------------------


def test_to_composite_matches_the_stored_fit_json(freqs):
    stored = json.loads(FIT_FILE.read_text())["components"]
    got = {c.name: c for c in BUDGET.to_composite().components}
    assert set(got) == {c["name"] for c in stored}
    for entry in stored:
        component = got[entry["name"]]
        assert component.scale == pytest.approx(entry["scale"], rel=1e-12)


def test_composite_total_matches_component_sum(freqs):
    # normalization="density" ignores df, so any positive bin width will do.
    composite, _ = BUDGET.to_composite().evaluate(freqs, 1.0)
    assert np.max(np.abs(composite / BUDGET.component_psds(freqs)["total"] - 1.0)) < 1e-12


# -- regeneration round-trip -------------------------------------------


def test_write_reference_asd_round_trips(tmp_path, freqs):
    written = BUDGET.write_reference_asd(tmp_path)
    assert len(written) == len(COMPONENT_FILES) + 1
    for channel, filename in COMPONENT_FILES.items():
        table = np.loadtxt(tmp_path / filename)
        assert table.shape == (freqs.size, 2)
        expected = BUDGET.component_asds(freqs)[channel]
        assert np.max(np.abs(table[:, 1] / expected - 1.0)) < ULP
