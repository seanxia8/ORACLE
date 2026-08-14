# Noise module configuration contract

All configuration models use schema version 3. Dictionary inputs remain
supported and are migrated before validation. Unknown keys are rejected by
default. Pass `strict_config=False` only in an explicit legacy adapter when
silently ignoring extra keys is intentional.

## `NoiseConfig`

| Field | Meaning and unit |
|---|---|
| `noise_type` | Analytic color, custom `.npy` path, `"composite"`, or a `CompositeSpectrum` |
| `noise_power` | Expected variance or mean-square power in signal-unit² |
| `sampling_frequency` | Sample frequency in Hz |
| `power_definition` | `"variance"` (zero stochastic DC) or `"mean_square"` |
| `deterministic_mean` | Deterministic signal offset in signal units |
| `psd_exponent` | Continuous exponent `alpha` in `S(f) proportional to f^alpha` |
| `low_frequency_cutoff` | Inclusive lower support boundary in Hz |
| `high_frequency_cutoff` | Inclusive upper support boundary in Hz |
| `custom_psd_scaling` | `"absolute"`, `"normalize"`, or `"scale"` |
| `psd_scale` | Non-negative multiplier used by custom `"scale"` mode |
| `custom_out_of_band` | `"error"`, `"zero"`, `"edge"`, or `"power_law"` |
| `custom_interpolation` | `"linear"` or `"loglog"` |
| `components` | Serializable composite-component dictionaries |
| `composite_psd_scaling` | `"absolute"` or globally `"normalize"` to `noise_power` |

Custom PSD files contain a `2 x F` array of frequency in Hz and one-sided PSD
density in signal-unit²/Hz.

## Spectral components

Every component has `scale` and `normalization`. With
`normalization="density"`, `scale` multiplies the component shape and carries
PSD-density units. With `normalization="power"`, the component is normalized
on the requested discrete grid and `scale` is its integrated power in
signal-unit².

- `White`: flat density.
- `PowerLaw`: exponent, reference frequency in Hz, and optional cutoffs in Hz.
- `Lorentzian` / `Resonance`: center and half-width in Hz.
- `BandLimited`: inclusive low and high frequencies in Hz.
- `RollOff`: corner frequency in Hz, positive order, and low/high-pass kind.
- `Line`: center and optional Gaussian width in Hz; zero width selects the
  closest discrete bin.

## `TemporalNoiseConfig`

Segment lengths, crossfades, and knot counts are samples/counts. Noise-power
and variance ranges are dimensionless multipliers. `drift_sigma` is in signal
units. PSD-slope changes are dimensionless changes to the power-law exponent.
`boundary_policy` selects hard, overlap-add, or continuously interpolated
evolutionary synthesis. Envelopes are generated in log amplitude by PCHIP,
linear interpolation, or low-pass stochastic modulation. `drift_rms` is in
signal units; drift timescale is seconds and cutoff is Hz. Shared fractions
between zero and one mix private and common multichannel envelopes and drifts.

## `ArtifactConfig`

`sampling_frequency` is Hz. Glitch and burst rates are events/second.
Preferred duration fields are seconds; sample fields are compatibility
aliases. Amplitudes and impulse sigma use signal units. Line frequencies are
Hz and must remain at or below Nyquist after harmonic multiplication.
`impulse_probability` is probability per sample.

`amplitude_unit` is one of:

- `raw`: template coefficient in signal units;
- `baseline_rms`: coefficient as a multiple of full-record baseline RMS;
- `local_rms`: coefficient as a multiple of local baseline RMS;
- `rms_energy_ratio`: chooses the coefficient so
  `||artifact||_2 / baseline_RMS` equals the requested value.

`rms_energy_ratio` is not a colored-noise matched-filter SNR. Schema migration
maps the old ambiguous name `snr` to `rms_energy_ratio`.

Homogeneous events use a Poisson count over physical duration.
Nonhomogeneous events use piecewise-constant Poisson interval counts from the
normalized `rate_profile`, including multiple arrivals in one sample
interval. The Hawkes option records immigrant and branching-process total
count expectations separately. Supported overlap policies are `superpose`,
`reject`, and `resample`; legacy `allow` is canonicalized to `superpose`.

## `MultiChannelConfig`

Channel and latent counts are positive integers. Correlation and all strength
or gain fields are dimensionless. A target CSD passed to
`generate_from_csd()` has shape `(N // 2 + 1, C, C)` and units
signal-unit_i × signal-unit_j / Hz.
Target CSDs may include an explicit source frequency grid; convex linear
interpolation preserves positive semidefiniteness. Repair policies are
`error`, `clip`, `nearest_psd`, and `diagonal_loading`. Batched output has shape
`(realizations, channels, samples)`, while a single realization retains
`(channels, samples)`. Low-rank spectral factors use shape `(F, C, R)` and
are synthesized directly without materializing `(F, C, C)`. Factors at DC
and even-length Nyquist must be real.

Single-record CSD diagnostics use Welch segment averaging. Ensemble inputs
use averaged periodograms on the exact rFFT grid; raw single-record coherence
is intentionally not reported because it is identically one.

## Non-Gaussian noise

`NonGaussianNoiseGenerator` supports finite-variance Student-t, Laplace,
Gaussian scale-mixture, compound-Poisson/shot-noise, and alpha-stable
innovations. For alpha-stable models, `noise_power` is scale squared and
variance may be undefined. Optional spectral coloring is reported separately
because linear filtering generally changes the named innovation marginal.
Diagnostics include skewness, excess kurtosis, tail exceedance, and QQ
quantiles.

## Reproducibility and metadata

Pass either `seed` or `rng`, never both. Metadata records the resolved,
versioned configuration, discrete PSD integration rule, expected variance,
stochastic DC contribution, deterministic mean, component power
contributions, and custom PSD provenance. A JSON round-trip of a resolved
dictionary configuration preserves seeded output. Use `to_jsonable(metadata)`
to serialize metadata containing NumPy or complex arrays without discarding
their real and imaginary parts.

## Validation and calibration

Validation results expose each acceptance decision under
`thresholds["gates"]`. Stationary validation gates power, PSD, Fourier
normality, time marginal normality, exponential periodogram ordinates,
skewness, kurtosis, whitening, and real-FFT endpoint constraints separately.
CSD validation separately gates diagonal PSD, cross-spectrum magnitude,
phase, coherence, and integrated covariance.

Local nonstationarity validation is diagnostic-only (`passed=None`) until the
caller supplies an expected range or minimum-variation contract. Calibration
requires at least two records and validates held-out power, PSD, CSD, and
tail behavior. Two-dimensional reference arrays require
`two_dimensional_layout="records_samples"` or `"channels_samples"`.

## Streaming

`StreamingNoiseGenerator` always synthesizes fixed
`chunk_samples`-length FFT blocks. Partial requests consume a buffer from
those blocks, so a short final request does not change the frequency grid or
low-frequency cutoff. Independent FFT blocks do not model cross-block
correlation and are not sample-identical to one-shot generation.
