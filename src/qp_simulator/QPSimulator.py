"""
Minimal QP (Quasi-Particle) trace simulator — standalone.

Generates a clean (noise-free) QP signal trace from an explicit array of
QP arrival times. Each QP arrival contributes one single-QP response template
to the trace; the result is the sum.

Conventions follow the DELight phonon path:
    * Single-QP template = exp(+(t - trigger_time)/tau_rise) for t <= trigger_time,
                           exp(-(t - trigger_time)/tau_decay) for t >  trigger_time
      So a QP arriving at t=0 produces a peak at t=trigger_time, with a leading
      rise of timescale tau_rise and a trailing decay of timescale tau_decay.
    * Per-QP ADC amplitude = gain_QE * E_to_ADC * 1e-3 * meV_per_QP
      (the 1e-3 converts the assumed ~1 meV per phonon to eV).

This module is standalone and has no external package dependencies.
Add your own noise on top of the returned trace.

Example
-------
>>> sim = QPSimulator()
>>> rng = np.random.default_rng(0)
>>> arrival_times_ns = rng.exponential(scale=2e6, size=5000) + 6e6  # ns
>>> trace = sim.generate(arrival_times_ns)        # shape (16384,)
>>> trace_with_noise = trace + my_noise_module.generate(len(trace))
"""

from __future__ import annotations

import numpy as np


class QPSimulator:
    """Minimal clean-trace QP simulator.

    Parameters
    ----------
    sampling_frequency : float
        Digitizer sampling frequency in Hz. Default 2.5e5.
    trace_samples : int
        Number of samples in the output trace. Default 16384.
    tau_rise : float
        Single-QP response rise time in ns. Default 50e3.
    tau_decay : float
        Single-QP response decay time in ns. Default 3e6.
    trigger_time : float or None
        Time within the trace (ns) at which a QP arriving at t=0 will peak.
        If None, defaults to 10% of the trace duration.
    gain_QE : float
        Quasi-particle energy-to-charge gain.
    E_to_ADC : float
        eV -> ADC conversion factor.
    meV_per_QP : float
        Mean energy per QP in meV. Default 1.0.
    adc_lsb : float
        ADC least significant bit in analog ADC-count units. Used only by
        ``digitize_trace`` or ``generate(..., digitize=True)``.
    adc_offset : float
        Analog pedestal corresponding to ADC code zero. Used only by explicit
        digitization.
    adc_bits : int or None
        Optional ADC bit depth. When set, digitized codes are clipped to the
        configured signed or unsigned range.
    adc_signed : bool
        If ``adc_bits`` is set, choose signed two's-complement style limits
        when true and unsigned limits when false.
    arrival_mode : {"linear", "histogram"}
        Default arrival-time deposition mode. ``"linear"`` preserves sub-sample
        timing to first order by splitting each arrival between neighbouring
        samples. ``"histogram"`` reproduces the legacy whole-sample binning.
    """

    def __init__(
        self,
        sampling_frequency: float = 2.5e5,
        trace_samples: int = 16_384,
        tau_rise: float = 50e3,
        tau_decay: float = 3e6,
        trigger_time: float | None = None,
        gain_QE: float = 15.0,
        E_to_ADC: float = 2.0,
        meV_per_QP: float = 1.0,
        adc_lsb: float = 1.0,
        adc_offset: float = 0.0,
        adc_bits: int | None = None,
        adc_signed: bool = True,
        arrival_mode: str = "linear",
    ):
        # Digitizer
        self.frequency = float(sampling_frequency)
        self.dt = 1.0 / self.frequency * 1e9              # ns per sample
        self.trace_samples = int(trace_samples)
        self.trace_duration = self.dt * self.trace_samples  # ns

        # Pulse shape
        self.tau_rise = float(tau_rise)
        self.tau_decay = float(tau_decay)
        self.trigger_time = (
            0.1 * self.trace_duration if trigger_time is None else float(trigger_time)
        )

        # Amplitude (ADC counts per single QP)
        self.gain_QE = float(gain_QE)
        self.E_to_ADC = float(E_to_ADC)
        self.meV_per_QP = float(meV_per_QP)
        self.qp_amplitude = self.gain_QE * self.E_to_ADC * 1e-3 * self.meV_per_QP

        # Explicit ADC model. The simulator still returns analog float traces
        # by default; digitization is opt-in so existing analyses are unchanged.
        self.adc_lsb = float(adc_lsb)
        self.adc_offset = float(adc_offset)
        self.adc_bits = None if adc_bits is None else int(adc_bits)
        self.adc_signed = bool(adc_signed)
        self.arrival_mode = self._validate_arrival_mode(arrival_mode)

        # Histogram bin edges for arrival times: length trace_samples + 1
        self.t_edges = np.arange(0.0, self.trace_duration + self.dt / 2.0, self.dt)

        self._build_template()

    # ------------------------------------------------------------------ #
    # Template
    # ------------------------------------------------------------------ #
    def _build_template(self) -> None:
        """Build single-QP response template (length 2.5 * trace_samples)."""
        xs = np.arange(0.0, self.trace_duration * 2.5, self.dt)
        # Cap the exponent at 0 in each branch so the unused half never
        # overflows; the `np.where` selects the correct half regardless.
        rising = np.exp(np.minimum((xs - self.trigger_time) / self.tau_rise, 0.0))
        falling = np.exp(np.minimum(-(xs - self.trigger_time) / self.tau_decay, 0.0))
        self.template = np.where(xs <= self.trigger_time, rising, falling)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def generate(
        self,
        arrival_times,
        return_amplitude: bool = False,
        return_metadata: bool = False,
        digitize: bool = False,
        arrival_mode: str | None = None,
    ) -> "np.ndarray | tuple":
        """Return a clean QP trace (in ADC counts) for the given arrivals.

        Parameters
        ----------
        arrival_times : array_like of float
            QP arrival times in ns. Each entry is one QP. Times outside
            [0, trace_duration) are silently dropped.
        return_amplitude : bool, optional
            When True, also return the true signal amplitude in ADC counts:
            ``amplitude_true = n_inside * self.qp_amplitude`` where
            ``n_inside`` is the number of arrival times that fall within
            ``[0, trace_duration)``. Default False (original behaviour).
        return_metadata : bool, optional
            When True, return a metadata dictionary containing ``n_QP_inside``,
            ``total_qp_amplitude_ADC``, and ``peak_ADC``.
        digitize : bool, optional
            When True, quantize the analog trace with ``digitize_trace``.
        arrival_mode : {"linear", "histogram"} or None
            Override the instance default arrival-time deposition mode.

        Returns
        -------
        trace : np.ndarray, shape (trace_samples,)
            Clean QP trace. Float analog ADC counts unless ``digitize=True``.
        amplitude_true : float
            Only returned when ``return_amplitude=True``. True signal
            total per-QP amplitude scale in ADC counts.
        metadata : dict
            Only returned when ``return_metadata=True``.
        """
        t = np.asarray(arrival_times, dtype=float).ravel()
        counts, n_inside = self._arrival_counts(t, arrival_mode=arrival_mode)
        trace = self._sum_template(counts) * self.qp_amplitude
        analog_peak = float(np.max(trace)) if trace.size else 0.0
        if digitize:
            trace = self.digitize_trace(trace)

        total_amplitude = float(n_inside) * self.qp_amplitude
        metadata = {
            "n_QP_inside": int(n_inside),
            "total_qp_amplitude_ADC": total_amplitude,
            "peak_ADC": analog_peak,
            "arrival_mode": self._validate_arrival_mode(arrival_mode)
            if arrival_mode is not None
            else self.arrival_mode,
            "digitized": bool(digitize),
        }
        outputs: list = [trace]
        if return_amplitude:
            outputs.append(total_amplitude)
        if return_metadata:
            outputs.append(metadata)
        if len(outputs) == 1:
            return trace
        return tuple(outputs)

    def generate_family(
        self,
        n_events: int,
        tau_decay_range: "tuple[float, float]" = (3e6, 3e6),
        t0_jitter_range: "tuple[float, float]" = (0.0, 0.0),
        n_QP_range: "tuple[int, int]" = (5000, 5000),
        rng: "np.random.Generator | None" = None,
        digitize: bool = False,
        arrival_mode: str | None = None,
    ) -> "tuple[np.ndarray, dict]":
        """Generate ``n_events`` clean traces with independently drawn per-event
        parameters.

        For each event:

        * ``tau_decay`` is drawn uniformly from ``tau_decay_range``.
        * ``trigger_time`` = ``self.trigger_time`` + Uniform(``t0_jitter_range``).
        * ``n_QP`` is drawn uniformly (integer) from ``n_QP_range``.
        * Arrival times are drawn as Exponential(scale=2e6 ns). The per-event
          trigger-time jitter is applied once through the temporary simulator's
          ``trigger_time``; it is not also added to the arrivals.
        * A temporary :class:`QPSimulator` is built with the per-event
          ``tau_decay`` and ``trigger_time`` to generate each trace; all
          other parameters are inherited from ``self``.

        No noise is added. Apply noise externally using the ``noise_module``
        classes before passing traces to an estimator.

        Parameters
        ----------
        n_events : int
            Number of events (traces) to generate.
        tau_decay_range : (float, float)
            Inclusive range for uniform draw of ``tau_decay`` in ns.
            Pass ``(x, x)`` for a fixed value.
        t0_jitter_range : (float, float)
            Inclusive range for uniform jitter (ns) added to
            ``self.trigger_time``. Pass ``(0.0, 0.0)`` for no jitter.
        n_QP_range : (int, int)
            Inclusive range for uniform integer draw of the number of QPs
            per event.
        rng : np.random.Generator or None
            Random generator for reproducibility. If None, uses
            ``np.random.default_rng()``.
        digitize : bool
            When true, return quantized integer ADC codes.
        arrival_mode : {"linear", "histogram"} or None
            Override the instance default arrival-time deposition mode.

        Returns
        -------
        traces : np.ndarray, shape (n_events, trace_samples)
            Clean traces (ADC counts). Add noise externally.
        params : dict
            Per-event ground-truth with keys:

            * ``'tau_decay'``    — np.ndarray, shape (n_events,)
            * ``'t0_shift'``    — np.ndarray, shape (n_events,), jitter in ns
            * ``'n_QP'``        — np.ndarray, shape (n_events,), dtype int
            * ``'total_qp_amplitude_ADC'`` — np.ndarray, shape (n_events,)
            * ``'peak_ADC'``      — np.ndarray, shape (n_events,), true trace peak
            * ``'amplitude_ADC'`` — backward-compatible alias for total QP
              amplitude scale, not the waveform peak
        """
        if rng is None:
            rng = np.random.default_rng()

        tau_lo, tau_hi = float(tau_decay_range[0]), float(tau_decay_range[1])
        t0_lo, t0_hi = float(t0_jitter_range[0]), float(t0_jitter_range[1])
        nqp_lo, nqp_hi = int(n_QP_range[0]), int(n_QP_range[1])

        tau_decays = (
            np.full(n_events, tau_lo)
            if tau_lo == tau_hi
            else rng.uniform(tau_lo, tau_hi, size=n_events)
        )
        t0_shifts = (
            np.full(n_events, t0_lo)
            if t0_lo == t0_hi
            else rng.uniform(t0_lo, t0_hi, size=n_events)
        )
        n_qps = (
            np.full(n_events, nqp_lo, dtype=int)
            if nqp_lo == nqp_hi
            else rng.integers(nqp_lo, nqp_hi + 1, size=n_events)
        )

        trace_dtype = np.int64 if digitize else float
        traces = np.empty((n_events, self.trace_samples), dtype=trace_dtype)
        total_amplitudes = np.empty(n_events, dtype=float)
        peaks = np.empty(n_events, dtype=float)
        n_inside = np.empty(n_events, dtype=int)

        for i in range(n_events):
            ev_tau = tau_decays[i]
            ev_t0 = self.trigger_time + t0_shifts[i]
            ev_nqp = int(n_qps[i])

            ev_sim = QPSimulator(
                sampling_frequency=self.frequency,
                trace_samples=self.trace_samples,
                tau_rise=self.tau_rise,
                tau_decay=ev_tau,
                trigger_time=ev_t0,
                gain_QE=self.gain_QE,
                E_to_ADC=self.E_to_ADC,
                meV_per_QP=self.meV_per_QP,
                adc_lsb=self.adc_lsb,
                adc_offset=self.adc_offset,
                adc_bits=self.adc_bits,
                adc_signed=self.adc_signed,
                arrival_mode=self.arrival_mode,
            )

            arrivals = rng.exponential(scale=2e6, size=ev_nqp)
            generated = ev_sim.generate(
                arrivals,
                return_amplitude=True,
                return_metadata=True,
                digitize=digitize,
                arrival_mode=arrival_mode,
            )
            trace, amp, meta = generated
            traces[i] = trace
            total_amplitudes[i] = amp
            peaks[i] = float(meta["peak_ADC"])
            n_inside[i] = int(meta["n_QP_inside"])

        params = {
            "tau_decay": tau_decays,
            "t0_shift": t0_shifts,
            "n_QP": n_qps,
            "n_QP_inside": n_inside,
            "total_qp_amplitude_ADC": total_amplitudes,
            "peak_ADC": peaks,
            "amplitude_ADC": total_amplitudes,
        }
        return traces, params

    def get_template_at_shift(self, t0_shift_ns: float) -> np.ndarray:
        """Return the single-QP response template shifted by ``t0_shift_ns``
        nanoseconds relative to ``self.trigger_time``.

        Builds a temporary :class:`QPSimulator` with
        ``trigger_time = self.trigger_time + t0_shift_ns`` and returns its
        template (the ``self.template`` attribute) truncated to
        ``self.trace_samples``.

        Parameters
        ----------
        t0_shift_ns : float
            Shift in nanoseconds. Positive = later peak.

        Returns
        -------
        np.ndarray, shape (trace_samples,)
            Shifted single-QP template (normalised, not scaled by
            ``qp_amplitude``).
        """
        shifted_sim = QPSimulator(
            sampling_frequency=self.frequency,
            trace_samples=self.trace_samples,
            tau_rise=self.tau_rise,
            tau_decay=self.tau_decay,
            trigger_time=self.trigger_time + float(t0_shift_ns),
            gain_QE=self.gain_QE,
            E_to_ADC=self.E_to_ADC,
            meV_per_QP=self.meV_per_QP,
            adc_lsb=self.adc_lsb,
            adc_offset=self.adc_offset,
            adc_bits=self.adc_bits,
            adc_signed=self.adc_signed,
            arrival_mode=self.arrival_mode,
        )
        return shifted_sim.template[: self.trace_samples].copy()

    def digitize_trace(self, trace: np.ndarray) -> np.ndarray:
        """Quantize an analog ADC-count trace to integer ADC codes.

        Quantization is round-to-nearest after subtracting ``adc_offset`` and
        dividing by ``adc_lsb``. If ``adc_bits`` is configured, codes are clipped
        to the corresponding signed or unsigned range.
        """
        if self.adc_lsb <= 0.0:
            raise ValueError("adc_lsb must be positive.")
        codes = np.rint((np.asarray(trace, dtype=float) - self.adc_offset) / self.adc_lsb)
        if self.adc_bits is not None:
            if self.adc_bits <= 0:
                raise ValueError("adc_bits must be positive when set.")
            if self.adc_signed:
                low = -(2 ** (self.adc_bits - 1))
                high = 2 ** (self.adc_bits - 1) - 1
            else:
                low = 0
                high = 2**self.adc_bits - 1
            codes = np.clip(codes, low, high)
        return codes.astype(np.int64)

    @staticmethod
    def estimate_psd(
        noise_traces: np.ndarray,
        sampling_frequency: float,
    ) -> "tuple[np.ndarray, np.ndarray]":
        """Estimate the one-sided noise PSD from an array of noise-only traces.

        For each trace, computes the real FFT, takes the squared magnitude,
        and averages across traces. The result is normalised so that
        ``sum(J_k) * (fs / N)`` equals the mean noise power, matching the
        convention used by ``NoiseGenerator``.

        Normalisation: ``J_k = mean(|FFT(x)|^2) * 2 / (N * fs)``

        where ``N = trace_samples`` and ``fs = sampling_frequency``.
        The factor of 2 accounts for folding negative frequencies into the
        one-sided spectrum (the DC and Nyquist bins are **not** doubled, in
        keeping with the standard ``numpy.fft.rfft`` convention, but for
        traces long enough for those bins to be negligible the difference
        is immaterial).

        Parameters
        ----------
        noise_traces : np.ndarray, shape (N_traces, trace_samples)
            Noise-only traces. Must be 2-D.
        sampling_frequency : float
            Digitizer sampling rate in Hz.

        Returns
        -------
        frequencies : np.ndarray, shape (trace_samples // 2 + 1,)
            One-sided frequency axis in Hz.
        J_k : np.ndarray, shape (trace_samples // 2 + 1,)
            One-sided PSD estimate (units: ADC² / Hz).
        """
        noise_traces = np.asarray(noise_traces, dtype=float)
        if noise_traces.ndim != 2:
            raise ValueError(
                "noise_traces must be 2-D with shape (N_traces, trace_samples)."
            )
        N = noise_traces.shape[1]
        fs = float(sampling_frequency)

        spectra = np.abs(np.fft.rfft(noise_traces, axis=1)) ** 2  # (N_traces, N//2+1)
        mean_spectrum = spectra.mean(axis=0)

        # One-sided PSD: multiply by 2 to fold negative frequencies, then
        # normalise by (N * fs) so the integral over positive freqs = power.
        # The DC and Nyquist bins are NOT doubled (each appears once in the
        # two-sided spectrum), matching the docstring, PSDCalculator, and the
        # OptimumFilter unfolding convention.
        J_k = mean_spectrum * 2.0 / (N * fs)
        J_k[0] /= 2.0
        if N % 2 == 0:
            J_k[-1] /= 2.0

        frequencies = np.fft.rfftfreq(N, d=1.0 / fs)
        return frequencies, J_k

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _sum_template(self, counts: np.ndarray) -> np.ndarray:
        """Sum-of-templates: place the template at every non-empty bin."""
        n = len(counts)
        out = np.zeros(n, dtype=float)
        for i in np.nonzero(counts)[0]:
            out[i:] += counts[i] * self.template[: n - i]
        return out

    def _arrival_counts(
        self,
        arrival_times: np.ndarray,
        arrival_mode: str | None = None,
    ) -> tuple[np.ndarray, int]:
        """Deposit arrival times onto the sampled trace grid."""
        mode = self._validate_arrival_mode(arrival_mode) if arrival_mode else self.arrival_mode
        t = np.asarray(arrival_times, dtype=float).ravel()
        counts = np.zeros(self.trace_samples, dtype=float)
        if t.size == 0:
            return counts, 0

        inside = (t >= 0.0) & (t < self.trace_duration)
        t_inside = t[inside]
        n_inside = int(t_inside.size)
        if n_inside == 0:
            return counts, 0

        if mode == "histogram":
            hist, _ = np.histogram(t_inside, bins=self.t_edges)
            return hist.astype(float), n_inside

        positions = t_inside / self.dt
        left = np.floor(positions).astype(int)
        frac = positions - left
        valid_left = (left >= 0) & (left < self.trace_samples)
        np.add.at(counts, left[valid_left], 1.0 - frac[valid_left])

        right = left + 1
        valid_right = (frac > 0.0) & (right < self.trace_samples)
        np.add.at(counts, right[valid_right], frac[valid_right])
        return counts, n_inside

    @staticmethod
    def _validate_arrival_mode(arrival_mode: str) -> str:
        mode = str(arrival_mode).lower()
        if mode not in {"linear", "histogram"}:
            raise ValueError("arrival_mode must be 'linear' or 'histogram'.")
        return mode


# ---------------------------------------------------------------------- #
# Self-test / quick demo
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sim = QPSimulator()
    print(
        f"dt = {sim.dt:g} ns | trace = {sim.trace_duration*1e-6:.2f} ms "
        f"| trigger_time = {sim.trigger_time*1e-6:.2f} ms "
        f"| QP amplitude = {sim.qp_amplitude:g} ADC"
    )

    rng = np.random.default_rng(0)
    n_qp = 5000

    # Same number of QPs, three different arrival-time distributions
    bursts = {
        "early (offset 0 ms, tau 2 ms)":   rng.exponential(2e6, n_qp) + 0.0,
        "mid   (offset 6 ms, tau 2 ms)":   rng.exponential(2e6, n_qp) + 6e6,
        "late  (offset 15 ms, tau 5 ms)":  rng.exponential(5e6, n_qp) + 15e6,
    }

    t_axis_ms = np.arange(sim.trace_samples) * sim.dt * 1e-6
    plt.figure(figsize=(9, 4))
    for label, t_arr in bursts.items():
        plt.plot(t_axis_ms, sim.generate(t_arr), lw=1.0, label=label)
    plt.xlabel("time [ms]")
    plt.ylabel("ADC counts")
    plt.title(f"QPSimulator — clean traces, {n_qp} QPs each")
    plt.legend(fontsize=8)
    plt.tight_layout()
    out_png = "qp_simulator_demo.png"
    plt.savefig(out_png, dpi=120)
    print(f"saved {out_png}")
