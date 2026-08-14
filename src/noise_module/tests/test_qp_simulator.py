from __future__ import annotations

import numpy as np

from qp_simulator import QPSimulator


class ZeroExponentialRng:
    def exponential(self, scale, size=None):
        return np.zeros(size, dtype=float)


def test_generate_reports_total_amplitude_and_peak_separately() -> None:
    sim = QPSimulator(
        sampling_frequency=1_000.0,
        trace_samples=128,
        tau_rise=1.0e6,
        tau_decay=1.0e6,
        trigger_time=20.0e6,
        gain_QE=1_000.0,
        E_to_ADC=1.0,
        meV_per_QP=1.0,
        arrival_mode="histogram",
    )

    trace, total_amplitude, metadata = sim.generate(
        [0.0, 40.0e6],
        return_amplitude=True,
        return_metadata=True,
    )

    assert np.isclose(total_amplitude, 2.0)
    assert np.isclose(metadata["total_qp_amplitude_ADC"], 2.0)
    assert metadata["n_QP_inside"] == 2
    assert np.isclose(metadata["peak_ADC"], np.max(trace))
    assert metadata["peak_ADC"] < total_amplitude


def test_digitize_trace_rounds_and_clips_to_adc_range() -> None:
    sim = QPSimulator(
        adc_lsb=0.5,
        adc_bits=3,
        adc_signed=False,
    )

    codes = sim.digitize_trace(np.array([-1.0, 0.24, 0.26, 10.0]))

    assert np.array_equal(codes, np.array([0, 0, 1, 7]))
    assert codes.dtype == np.int64


def test_generate_family_applies_t0_shift_once() -> None:
    sim = QPSimulator(
        sampling_frequency=1_000.0,
        trace_samples=128,
        tau_rise=1.0e6,
        tau_decay=2.0e6,
        trigger_time=10.0e6,
        gain_QE=1_000.0,
        E_to_ADC=1.0,
        meV_per_QP=1.0,
        arrival_mode="histogram",
    )

    traces, params = sim.generate_family(
        1,
        tau_decay_range=(2.0e6, 2.0e6),
        t0_jitter_range=(5.0e6, 5.0e6),
        n_QP_range=(1, 1),
        rng=ZeroExponentialRng(),
        arrival_mode="histogram",
    )

    peak_time = int(np.argmax(traces[0])) * sim.dt
    assert np.isclose(params["t0_shift"][0], 5.0e6)
    assert np.isclose(peak_time, sim.trigger_time + params["t0_shift"][0])
    assert np.isclose(params["peak_ADC"][0], np.max(traces[0]))
    assert np.isclose(params["amplitude_ADC"][0], params["total_qp_amplitude_ADC"][0])


def test_linear_arrival_mode_preserves_subsample_information() -> None:
    sim = QPSimulator(
        sampling_frequency=1_000.0,
        trace_samples=16,
        trigger_time=4.0e6,
    )

    linear_counts, n_linear = sim._arrival_counts(np.array([0.5 * sim.dt]), "linear")
    hist_counts, n_hist = sim._arrival_counts(np.array([0.5 * sim.dt]), "histogram")

    assert n_linear == n_hist == 1
    assert np.isclose(linear_counts[0], 0.5)
    assert np.isclose(linear_counts[1], 0.5)
    assert hist_counts[0] == 1.0
    assert hist_counts[1] == 0.0
