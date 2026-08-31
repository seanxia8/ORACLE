# Phase 1–5 consistency audit

The Phase 1–5 APIs use the following common contracts:

- PSD density is one-sided signal-unit²/Hz; CSD density is one-sided
  signal-unit_i × signal-unit_j/Hz.
- `noise_power` is expected variance by default, with stochastic DC zero.
- Single-channel shape is `(N,)`, multichannel shape is `(C, N)`, and batched
  shape is `(R, C, N)` or `(R, N)`.
- Event rates are events/second and preferred event durations are seconds.
- `seed` and `rng` are mutually exclusive. Child generators receive spawned
  streams and do not use NumPy global randomness.
- Transforming `apply` methods preserve their input identity when disabled and
  never silently regenerate it.
- Optional component arrays reconstruct combined output exactly.
- Configurations and metadata carry explicit schema versions.

Corrections made during this audit:

1. `variance_scale_range` now denotes variance ratios; the applied amplitude
   envelope is their square root.
2. The relocated module now has an independently installable package definition.
3. Validation, calibration, and streaming APIs use the same frequency, units,
   shape, and RNG conventions.

Known intentional distinctions:

- FFT synthesis is a periodic finite-grid process; exact Toeplitz covariance
  simulation is a separate API.
- Coloring non-Gaussian innovations does not promise preservation of their
  original marginal distribution.
- Chunked colored generation is reproducible for an identical chunk schedule
  but is not sample-identical to one-shot FFT generation and does not preserve
  correlations across chunk boundaries.
