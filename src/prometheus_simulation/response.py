
"""NuBench-style detector response, implemented in-project.

The emulation follows NuBench sec. 3.2 (arXiv:2511.13111), which the release
never shipped as code: quantum-efficiency thinning, a trigger window of at
least min_window_us, stochastic noise photons sampled uniformly in the
window, photon merging within a transit-time-spread (TTS) window per OM,
Gaussian smearing of pulse time and charge, and event cuts on pulse count.
Owning this stage is deliberate: it is where every acquisition-side (N) knob
lives — the noise rate, the smearing widths, the merging window, the QE.

Values NuBench states: OM radius 30 cm and QE 20% (geometry-level effects are
upstream, in the photon simulation), trigger window >= 5 us, time smearing
1 ns, charge smearing 0.25 pe, cuts at < 4 and > 1e6 pulses. The TTS merging
window length is NOT stated in the paper; the default here is an assumption,
declared in the config and recorded in every provenance record.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .events import EventPhotons, EventPulses
from .detector import DetectorGeometry


@dataclass(frozen=True)
class ResponseConfig:
    """All sec.-3.2 knobs. Every field is an acquisition-contract term."""

    quantum_efficiency: float = 0.20
    min_window_us: float = 5.0
    window_pad_ns: float = 100.0
    #: expected noise photons per OM per microsecond of trigger window
    noise_rate_per_om_us: float = 0.01
    #: TTS merging window; NuBench does not state its value — assumption
    tts_merge_ns: float = 2.0
    time_smear_ns: float = 1.0
    charge_smear_pe: float = 0.25
    min_pulses: int = 4
    max_pulses: int = 1_000_000

    def with_(self, **kwargs) -> "ResponseConfig":
        return replace(self, **kwargs)


def _merge_om_photons(
    times: np.ndarray, signal: np.ndarray, tts_ns: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy left-to-right merge of sorted photon times within tts_ns.

    Returns (pulse_time, pulse_charge, pulse_signal_fraction). The pulse time
    is the first photon's time in the merge group (leading-edge convention);
    the charge is the photon count.
    """
    order = np.argsort(times, kind="stable")
    times, signal = times[order], signal[order]
    starts = [0]
    for i in range(1, times.size):
        if times[i] - times[starts[-1]] > tts_ns:
            starts.append(i)
    bounds = starts + [times.size]
    t_out = np.empty(len(starts))
    q_out = np.empty(len(starts))
    f_out = np.empty(len(starts))
    for j, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        t_out[j] = times[a]
        q_out[j] = b - a
        f_out[j] = float(np.mean(signal[a:b]))
    return t_out, q_out, f_out


def emulate_response(
    photons: EventPhotons,
    geometry: DetectorGeometry,
    config: ResponseConfig = ResponseConfig(),
    rng: np.random.Generator | int | None = None,
) -> EventPulses:
    """Apply the full sec.-3.2 chain to one event. Deterministic under seed."""
    rng = np.random.default_rng(rng)

    # 1. quantum-efficiency thinning of physics photons (noise is added later
    #    at its own rate, so QE applies to the incoming photons only).
    keep = rng.random(photons.n_photons) < config.quantum_efficiency
    om = photons.om_id[keep]
    t = photons.t_ns[keep]
    sig = photons.is_signal[keep]

    # 2. shift into a trigger window of at least min_window_us.
    if t.size:
        t = t - t.min() + config.window_pad_ns
        span = float(t.max()) + config.window_pad_ns
    else:
        span = 0.0
    window_ns = max(config.min_window_us * 1e3, span)

    # 3. stochastic noise photons, uniform in the window across all OMs.
    lam = config.noise_rate_per_om_us * geometry.n_om * window_ns / 1e3
    n_noise = int(rng.poisson(lam))
    if n_noise:
        om = np.concatenate([om, rng.integers(0, geometry.n_om, n_noise)])
        t = np.concatenate([t, rng.uniform(0.0, window_ns, n_noise)])
        sig = np.concatenate([sig, np.zeros(n_noise, dtype=bool)])

    # 4. per-OM TTS merge into pulses.
    om_out, t_out, q_out, f_out = [], [], [], []
    for om_idx in np.unique(om):
        mask = om == om_idx
        tm, qm, fm = _merge_om_photons(t[mask], sig[mask], config.tts_merge_ns)
        om_out.append(np.full(tm.size, om_idx))
        t_out.append(tm)
        q_out.append(qm)
        f_out.append(fm)
    if om_out:
        om_p = np.concatenate(om_out)
        t_p = np.concatenate(t_out)
        q_p = np.concatenate(q_out)
        f_p = np.concatenate(f_out)
    else:
        om_p = np.empty(0, dtype=int)
        t_p = q_p = f_p = np.empty(0)

    # 5. Gaussian smearing of pulse time and charge.
    t_p = t_p + rng.normal(0.0, config.time_smear_ns, t_p.size)
    q_p = np.clip(q_p + rng.normal(0.0, config.charge_smear_pe, q_p.size), 0.0, None)

    # 6. event cuts on pulse count.
    passed = config.min_pulses <= om_p.size <= config.max_pulses

    return EventPulses(
        event_id=photons.event_id,
        om_id=om_p,
        t_ns=t_p,
        charge_pe=q_p,
        signal_fraction=f_p,
        passed_cuts=passed,
        window_ns=window_ns,
        truth=dict(photons.truth),
        meta={"response_config": config.__dict__.copy()},
    )
