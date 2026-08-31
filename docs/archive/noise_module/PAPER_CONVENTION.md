# Paper-ready discrete Gaussian noise convention (EXP-01)

This document is the equation-ready contract for the stationary Gaussian
generator implemented in `NoiseGenerator.py`. It supersedes Paper 1 Equation
81 (the fixed-modulus, random-phase-only sampler) and should be copied
directly into the manuscript's Appendix G rather than paraphrased. The
machine-readable companion is `paper_convention.json`.

## Symbols and units

| Symbol | Meaning | Units |
| --- | --- | --- |
| `N` | trace length (samples) | samples |
| `f_s` | sampling frequency | Hz |
| `P_k` | one-sided PSD density at rFFT bin `k` | (signal units)^2 / Hz |
| `X_k` | rFFT coefficient at bin `k` (NumPy/`scipy.fft` unnormalised convention) | signal units |
| `n_bins` | number of rFFT bins, `N // 2 + 1` | -- |

`P_k` is a *density*: `build_psd_density()` returns exactly this quantity.
`build_rfft_power()` / `psd_density_to_rfft_power()` convert it to expected
rFFT-bin power under NumPy's unnormalised `rfft`/`irfft` convention, which is
the quantity actually sampled.

## Expected rFFT-bin power

```text
E|X_k|^2 = f_s * N * P_k / 2      for interior bins (0 < k < N/2)
E|X_k|^2 = f_s * N * P_k          for the DC bin (k = 0)
E|X_k|^2 = f_s * N * P_k          for the Nyquist bin (k = N/2, even N only)
```

Interior bins are **independent circular complex Gaussian**:
`X_k = (Re + i*Im) / sqrt(2) * sqrt(E|X_k|^2)` with `Re, Im ~ N(0,1)` i.i.d.
This is the corrected replacement for Paper 1 Eq. 81's fixed-modulus,
random-phase-only construction (`X_k = sqrt(P_k) * e^{i*phi}`, `phi ~
Uniform[0, 2*pi)`): the old sampler reproduces the correct *mean* PSD but
gives every realization identical per-bin power, so any distributional
statistic built from residual energies (chi-square goodness-of-fit,
Kolmogorov-Smirnov whiteness tests, periodogram-variance checks) is
degenerate and scientifically invalid under it.

DC and (for even `N`) Nyquist bins are **real Gaussian**, `X_k = Re *
sqrt(E|X_k|^2)`, `Re ~ N(0,1)`, using the *full* endpoint power above (not
halved). A `power_definition="variance"` configuration additionally forces
the DC bin's contribution to zero so that the reported `noise_power` is the
process's stochastic variance rather than a random per-draw mean-square
value; a nonzero deterministic mean is represented separately via
`deterministic_mean` and is not part of the stochastic PSD.

`irfft` normalization: `x = irfft(X, n=N)` with NumPy's unnormalised
convention (forward transform unscaled, inverse scaled by `1/N`) is used
throughout; the endpoint/interior halving above is exactly what is required
for `Var[x_t] = integral of P(f) df` (one-sided) to hold under that
convention.

## Covariance class

The FFT-grid process defined above has **periodic / circulant** covariance
on the generated grid of length `N` -- every realization is implicitly
periodic with period `N`, and the induced time-domain covariance matrix is
circulant, not a general (exact finite) Toeplitz matrix. This distinction
matters at trace boundaries and for any claim about a truly stationary
infinite process truncated to a finite window.

`NoiseGenerator.sample_from_covariance()` is the separate, exact operation
for when a genuine finite Toeplitz covariance (not a periodic surrogate) is
the primary contract: it samples via circulant embedding when the embedding
is positive semidefinite (`method="circulant"`, exact for the retained
length-`N` prefix), and falls back to a dense eigendecomposition of the
literal Toeplitz matrix otherwise (`method="dense"`, exact by construction,
verified against the requested first row via `maximum_covariance_error` in
the returned metadata). Do not conflate this with the FFT-grid generator: it
is a different sampling operation, only sharing the "stationary Gaussian"
property.

## Custom PSD scaling modes

`build_psd_density()`'s `custom_psd_scaling` accepts `"absolute"` (file units
preserved), `"normalize"` (file treated as a shape, its rFFT-grid integral
normalized to `noise_power`), and `"scale"` (file density multiplied by
`psd_scale`, which defaults to `noise_power` when not set explicitly).
`"multiply"` is accepted as a backward-compatible alias for `"scale"`
(`config.py` normalizes it on load). A prior version of this docstring said
`"multiply"` where the code actually checked `"scale"`; both the docstring
and this document now name the canonical value.

## Known defect and fix (EXP-01)

The smaller, separate core-repo generator
`src/noise_geometry/noise/synthetic.py:generate_colored_noise` (used by
synthetic gates S0, S1, S2, S5 -- see
`results/audits/noise_generator_provenance.json`) previously applied the
*interior*-bin scale to the DC bin instead of the endpoint scale above,
under-scaling DC by `sqrt(2)` in amplitude (a factor of 2 in variance) while
correctly scaling the Nyquist bin. This has been fixed to match the
convention in this document; see `tests/test_generate_colored_noise_convention.py`
for the regression gates. `NoiseGenerator.py`'s own
`psd_density_to_rfft_power()` already implemented DC/Nyquist correctly
(confirmed by direct code reading and by the new statistical gates in
`tests/test_paper_convention_gates.py`); it required no fix.

## Statistical acceptance gates

Implemented in `tests/test_paper_convention_gates.py` (this package) and
`tests/test_generate_colored_noise_convention.py` (core repo), each
parametrized over multiple fixed seeds with moment/effect tolerances rather
than a single p-value:

- expected time-domain variance recovers the one-sided PSD integral;
- interior-bin normalized periodogram (`|X_k|^2 / E|X_k|^2`, pooled across
  bins and draws) has Exponential(mean=1) first and second moments;
- DC and Nyquist bins are real-valued and carry the full endpoint power, not
  the halved interior power;
- ensemble PSD recovery matches the target density away from the endpoints
  (pre-existing `test_ensemble_periodogram_matches_white_target_psd`);
- circulant covariance recovery on the FFT grid (pre-existing
  `test_circulant_embedding_recovers_requested_covariance`);
- exact Toeplitz recovery through `sample_from_covariance(method="dense")`,
  both via the deterministic eigendecomposition metadata (pre-existing) and
  via empirical ensemble variance/lag-1 covariance (new, this package).
