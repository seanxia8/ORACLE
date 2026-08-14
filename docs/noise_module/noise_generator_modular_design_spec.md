# Noise Generator Modular Design Spec

## Purpose
This document defines a **clean, feasible extension plan** for the current `NoiseGenerator` so an agent can implement a paper-ready noise simulation stack without rewriting everything from scratch.

The design goal is:
- keep the current `NoiseGenerator` as the **stationary Gaussian single-channel spectral core**
- add a small number of modules around it
- make ablation studies easy
- support future paper experiments: robustness, mismatch, drift, artifacts, and synthetic multi-channel correlation

---

# 1. Core Design Decision

## Keep the current `NoiseGenerator`
The current `NoiseGenerator` should remain the base module for:
- stationary single-channel Gaussian noise
- PSD-based frequency-domain synthesis
- analytic or custom PSD support

It should **not** be overloaded with every advanced feature.

## Add three new modules

### New modules
1. `TemporalNoiseWrapper`
   - non-stationarity
   - piecewise stationarity
   - drift
   - local variance changes

2. `ArtifactInjector`
   - spectral lines
   - glitches
   - bursts
   - sparse non-Gaussian artifacts

3. `MultiChannelNoiseGenerator`
   - independent channels
   - synthetic correlated channels
   - shared/private and low-rank multi-channel generation

---

# 2. Final Architecture

```text
NoiseGenerator                # existing base module
├── stationary Gaussian single-channel spectral synthesis
│
├── TemporalNoiseWrapper      # new module
│   ├── stationary passthrough
│   ├── piecewise stationary
│   ├── drift
│   └── variance modulation
│
├── ArtifactInjector          # new module
│   ├── spectral lines
│   ├── glitches
│   ├── bursts
│   └── sparse impulses / heavy-tail artifacts
│
└── MultiChannelNoiseGenerator # new module
    ├── independent channels
    ├── shared + private correlated channels
    ├── low-rank latent correlated channels
    └── future full CSD model
```

---

# 3. Integration Strategy

## Single-channel generation

```python
base = NoiseGenerator(config, rng=rng)
x = base.generate_noise(N)

x = TemporalNoiseWrapper(...).apply(x, base_generator=base)
x = ArtifactInjector(...).apply(x)
```

## Multi-channel generation

```python
mc = MultiChannelNoiseGenerator(base_config, rng=rng)
X = mc.generate_shared_private(C, N, corr_strength=0.3)

X = TemporalNoiseWrapper(...).apply_multichannel(X, base_generator=mc)
X = ArtifactInjector(...).apply_multichannel(X)
```

## Important design rule
Do **not** implement a single huge function like:

```python
generate_noise(N, correlated=True, nonstationary=True, drift=True, artifact=True, ...)
```

This is hard to validate and hard to ablate.

Use a composable pipeline instead:

```python
x = base.generate_noise(N)
x = temporal.apply(x)
x = artifact.apply(x)
```

or

```python
X = multichannel.generate_shared_private(...)
X = temporal.apply_multichannel(X)
X = artifact.apply_multichannel(X)
```

This is much better for experiments and paper figures.

---

# 4. Required Small Refactor to Existing `NoiseGenerator`

The current `NoiseGenerator` should be minimally refactored before adding the new modules.

## Required changes

### 4.1 Add RNG support
Replace global random calls with a stored random generator.

### Required pattern
```python
self.rng = np.random.default_rng(seed)
```

All random draws should use `self.rng`.

### Why
- reproducibility
- experiment sweeps
- deterministic debugging

---

### 4.2 Split PSD construction from sampling
Separate:
- build PSD
- sample stationary Gaussian noise from PSD

### Recommended methods
```python
def build_psd(self, N):
    """Return freqs and one-sided PSD."""


def sample_stationary_gaussian_from_psd(self, psd, N=None):
    """Sample single-channel stationary Gaussian time series from PSD."""


def generate_noise(self, N):
    freqs, psd = self.build_psd(N)
    return self.sample_stationary_gaussian_from_psd(psd, N=N)
```

### Why
The wrappers need access to PSD generation without duplicating logic.

---

### 4.3 Add optional metadata output
Recommended optional return:

```python
x, meta = generate_noise(N, return_metadata=True)
```

Metadata may contain:
- noise type
- PSD parameters
- total power
- seed

This is useful for experiment logging.

---

# 5. Module 1 — `TemporalNoiseWrapper`

## Purpose
This module adds **time-dependent behavior** on top of the stationary Gaussian base.

It handles:
- piecewise stationarity
- local variance changes
- low-frequency drift
- slow condition changes across a trace

It should **not** handle artifacts like glitches or bursts.

---

## 5.1 Responsibilities

### Supported effects
- stationary passthrough
- piecewise stationary segments
- drift
- variance modulation

### Not supported here
- transient glitch templates
- line noise injection
- sparse impulsive artifacts

Those belong in `ArtifactInjector`.

---

## 5.2 Main API

### Constructor
```python
TemporalNoiseWrapper(config: dict, rng=None)
```

### Core methods
```python
def apply(self, x: np.ndarray, base_generator=None, return_metadata=False) -> np.ndarray:
    """Apply temporal effects to a single-channel trace."""


def apply_multichannel(self, X: np.ndarray, base_generator=None, return_metadata=False) -> np.ndarray:
    """Apply temporal effects to multi-channel trace array of shape (C, N)."""


def generate_piecewise(self, N: int, base_generator, return_metadata=False) -> np.ndarray:
    """Generate a new piecewise-stationary trace from scratch using the base generator."""


def generate_drift(self, N: int) -> np.ndarray:
    """Generate additive low-frequency drift."""
```

---

## 5.3 Detailed implementation

## A. Piecewise stationarity
Split the trace into segments.

For each segment:
- sample local noise parameters
- create a local `NoiseGenerator` config
- generate one segment
- concatenate segments
- optionally smooth at boundaries

### Parameters to vary per segment
- `noise_power`
- PSD slope / color parameter
- custom PSD scale
- line-amplitude multiplier if needed later
- variance scale

### Minimal implementation
```python
segments = [(start0, end0), (start1, end1), ...]
for each segment:
    local_cfg = sample_local_config(base_cfg)
    local_ng = NoiseGenerator(local_cfg, rng=sub_rng)
    x_seg = local_ng.generate_noise(seg_len)
concatenate x_seg
```

### Boundary smoothing
Recommended simple solution:
- overlap neighboring segments by `fade_len`
- linear or cosine cross-fade

### Why
Without smoothing, segment boundaries may create unrealistic jumps.

---

## B. Drift
Drift is a **slow additive component**. Treat it separately from piecewise stationarity.

### Recommended first implementation: spline drift
1. choose `n_knots`
2. draw knot amplitudes from Gaussian distribution
3. interpolate across the trace
4. optionally low-pass smooth

### Example conceptual model
```python
knot_x = np.linspace(0, N - 1, n_knots)
knot_y = rng.normal(0, sigma_drift, size=n_knots)
drift = cubic_spline(knot_x, knot_y)(np.arange(N))
```

### Alternative options
- random walk
- very-low-frequency colored process

### Recommendation
Start with **spline drift** because:
- easy to tune
- smooth
- numerically stable
- easy to visualize in paper figures

---

## C. Variance modulation
This introduces slow amplitude changes across time.

### Implementation
Sample a smooth positive envelope `a(t)` and rescale the trace:

```python
x_mod = a(t) * x
```

Where `a(t)` may be generated from:
- spline interpolation around 1.0
- piecewise constant values with smoothing

### Recommendation
Keep this optional. Minimal paper requirement is already satisfied by:
- piecewise stationarity
- additive drift

---

## 5.4 Minimal configuration

```python
temporal_config = {
    "mode": "piecewise",              # "none", "piecewise"
    "n_segments": 4,
    "segment_length": None,            # if None, derive from N and n_segments
    "crossfade_len": 128,
    "vary_noise_power": True,
    "noise_power_scale_range": [0.8, 1.2],
    "vary_psd_slope": False,
    "psd_slope_range": [-0.1, 0.1],

    "add_drift": True,
    "drift_type": "spline",          # "spline", "random_walk"
    "drift_sigma": 0.05,
    "drift_n_knots": 6,

    "variance_modulation": False,
    "variance_scale_range": [0.95, 1.05],
}
```

---

## 5.5 Minimal requirement for paper
For the first paper-ready version, implement only:
- piecewise stationarity with local `noise_power` variation
- spline drift
- optional cross-fade smoothing

That is enough to support:
- non-stationarity study
- drift robustness study
- mismatch experiments

---

# 6. Module 2 — `ArtifactInjector`

## Purpose
This module injects **discrete or structured artifacts** on top of the baseline noise.

It handles:
- spectral lines
- glitches
- bursts
- sparse impulsive non-Gaussian events

It should not be responsible for the stationary Gaussian baseline.

---

## 6.1 Responsibilities

### Supported effects
- narrowband sinusoidal lines
- harmonic lines
- transient glitch templates
- short ringing bursts
- sparse impulses / heavy-tail artifacts

### Not supported here
- piecewise stationarity
- drift
- channel covariance generation

---

## 6.2 Main API

```python
ArtifactInjector(config: dict, rng=None)


def apply(self, x: np.ndarray, return_metadata=False) -> np.ndarray:
    """Apply artifacts to a single-channel trace."""


def apply_multichannel(self, X: np.ndarray, return_metadata=False) -> np.ndarray:
    """Apply artifacts to shape (C, N)."""


def add_lines(self, x: np.ndarray) -> np.ndarray:
    ...


def add_glitches(self, x: np.ndarray) -> np.ndarray:
    ...


def add_bursts(self, x: np.ndarray) -> np.ndarray:
    ...


def add_sparse_impulses(self, x: np.ndarray) -> np.ndarray:
    ...
```

---

## 6.3 Detailed implementation

## A. Spectral lines
These are deterministic or quasi-deterministic sinusoidal components.

### Minimal implementation
For each configured line:

```python
line = A * np.sin(2 * np.pi * f0 * t + phi)
x += line
```

### Options
- fixed amplitude
- amplitude sampled from range
- multiple harmonics
- per-channel amplitude scaling

### Recommended config structure
```python
"lines": [
    {"freq": 50.0, "amp": 0.02, "phase": "random", "harmonics": [1, 2, 3]},
    {"freq": 1200.0, "amp": 0.01, "phase": "random", "harmonics": [1]}
]
```

---

## B. Glitches
Glitches are short localized transients.

### Minimal implementation
Use a template library and inject at random times.

### Template types
Start with synthetic shapes if no real glitch library is available:
- impulse spike
- exponential decay
- damped sinusoid
- short ringing packet

### Injection rule
```python
x[t0:t0+L] += a * template[:valid_len]
```

### If real glitch snippets become available later
Replace synthetic templates with real extracted templates.

---

## C. Bursts
Bursts are short intervals of elevated activity.

### Minimal implementation
Generate a short local packet:
- Gaussian-windowed sinusoid
- short colored-noise packet
- ringing waveform

Then add it over a local interval.

---

## D. Sparse non-Gaussian impulses
This is the easiest heavy-tail model.

### Minimal implementation
Randomly choose a small set of time indices and add large outliers.

```python
mask = rng.uniform(size=N) < p_impulse
x[mask] += rng.normal(0, sigma_impulse, size=mask.sum())
```

### Why this is useful
- creates heavy tails
- simple
- tunable
- no need for a full non-Gaussian baseline generator

---

## 6.4 Minimal configuration

```python
artifact_config = {
    "enable_lines": True,
    "lines": [
        {"freq": 50.0, "amp": 0.02, "phase": "random", "harmonics": [1, 2]},
    ],

    "enable_glitches": True,
    "glitch_rate": 1.0,                # expected glitches per second
    "glitch_amp_range": [0.05, 0.2],
    "glitch_templates": ["impulse", "exp_decay", "damped_sine"],
    "glitch_duration_samples": [32, 256],

    "enable_bursts": False,
    "burst_rate": 0.2,                 # expected bursts per second
    "burst_amp_range": [0.03, 0.1],
    "burst_duration_samples": [128, 512],

    "enable_sparse_impulses": True,
    "impulse_probability": 1e-4,
    "impulse_sigma": 0.1,
}
```

---

## 6.5 Minimal requirement for paper
For the first paper-ready version, implement only:
- spectral lines
- simple synthetic glitch templates
- sparse impulse artifacts

This is enough to support:
- non-Gaussian robustness study
- artifact contamination study
- false-trigger and residual-tail experiments

---

# 7. Module 3 — `MultiChannelNoiseGenerator`

## Purpose
This module generates **multi-channel noise arrays** of shape `(C, N)`.

It should be a separate module because multi-channel correlated generation is not a small patch to a single-channel PSD sampler.

You currently only have one-channel real detector noise, so the minimal implementation should use **synthetic correlation models**, not a full real cross-spectral model.

---

## 7.1 Responsibilities

### Supported effects
- independent channels
- shared + private correlated channels
- low-rank latent correlation model
- future extension to full cross-spectral density model

### Not required yet
- real 56-channel detector covariance estimation
- full measured CSD from real multichannel baselines

Those can be added later when data becomes available.

---

## 7.2 Main API

```python
MultiChannelNoiseGenerator(base_config: dict, rng=None)


def generate_independent(self, C: int, N: int, return_metadata=False) -> np.ndarray:
    """Generate C independent channels using base NoiseGenerator."""


def generate_shared_private(self, C: int, N: int, corr_strength=0.3, return_metadata=False) -> np.ndarray:
    """Generate channels with one shared latent process plus private noise."""


def generate_lowrank_correlated(self, C: int, N: int, n_latent=2, return_metadata=False) -> np.ndarray:
    """Generate channels using a low-rank latent shared model."""
```

---

## 7.3 Detailed implementation

## A. Independent channels
This is the baseline.

For each channel:
- instantiate or reuse base `NoiseGenerator`
- generate one channel independently

### Purpose
This is the null model for comparison.

---

## B. Shared + private correlated channels
This should be the **first correlated implementation**.

### Model
For each channel `c`:

```text
x_c(t) = a_c * g_shared(t) + b_c * g_c_private(t)
```

Where:
- `g_shared(t)` is one common colored Gaussian process
- `g_c_private(t)` is an independent colored Gaussian process for channel `c`
- `a_c` controls shared correlation strength
- `b_c` controls private variance

### Minimal implementation
1. generate one shared process using base `NoiseGenerator`
2. generate one private process per channel
3. combine using mixing coefficients
4. optionally normalize channel variance after mixing

### Why this is good
- very easy to implement
- gives tunable channel correlation
- already enough for joint OF / EMPCA ablation studies

---

## C. Low-rank latent correlated model
This is a stronger but still feasible synthetic model.

### Model
```text
x_c(t) = Σ_m W[c, m] * z_m(t) + ε_c(t)
```

Where:
- `z_m(t)` are shared latent colored processes
- `W` is channel mixing matrix
- `ε_c(t)` is private channel noise

### Minimal implementation
1. generate `n_latent` shared latent traces
2. sample mixing matrix `W`
3. combine latent traces into each channel
4. add private channel noise

### Why this is useful
- more realistic than single shared mode
- introduces channel groups and structured correlation
- closer to what EMPCA might actually exploit

---

## D. Future full cross-spectral density model
Do not implement for minimal requirement unless multichannel real noise becomes available.

### Future model
At each frequency bin, sample multivariate complex Gaussian coefficients using cross-spectral covariance matrix `S(f)`.

This is the correct end-state, but not required now.

---

## 7.4 Minimal configuration

```python
multichannel_config = {
    "mode": "shared_private",         # "independent", "shared_private", "lowrank"
    "n_channels": 56,

    "corr_strength": 0.3,
    "channel_gain_jitter": 0.05,

    "n_latent": 2,
    "latent_strength_range": [0.1, 0.4],
    "private_strength_range": [0.8, 1.2],

    "normalize_channel_variance": True,
}
```

---

## 7.5 Minimal requirement for paper
For the first paper-ready version, implement only:
- independent channels
- shared + private correlated channels

Optional but useful:
- low-rank latent model

This is enough for:
- synthetic correlation heatmaps
- joint vs independent OF comparison
- EMPCA sensitivity to channel correlation strength

---

# 8. Minimal Outside Inputs Required

## Required for minimal implementation

### 8.1 Sampling frequency
Needed to define:
- PSD frequency axis
- spectral line locations
- drift timescale interpretation

### 8.2 Baseline PSD or existing PSD configs
Needed for the base `NoiseGenerator`.

Minimal acceptable inputs:
- analytic white/pink PSD parameters, or
- one empirical single-channel PSD from real noise

### 8.3 Trace length conventions
Needed for:
- segment sizes
- drift knot count
- artifact duration scales

---

## Not required for minimal implementation
These are **not required now**:
- real 56-channel noise recordings
- measured cross-spectral density matrix
- real glitch library

These can be replaced temporarily by:
- synthetic shared/private correlation model
- synthetic glitch templates
- simple line-noise configuration

---

## Strongly recommended external inputs
If available, provide these to improve realism:

1. **one representative empirical single-channel PSD**
2. **typical detector line frequencies**
   - e.g. 50 Hz, harmonics, digitizer lines, microphonic peaks
3. **rough drift amplitude / timescale** from real baselines
4. **typical glitch amplitude / duration range** if known

These are enough to make the minimal model much more realistic.

---

# 9. Recommended File Layout

```text
noise/
├── noise_generator.py              # existing base NoiseGenerator
├── temporal_noise.py              # TemporalNoiseWrapper
├── artifact_injector.py           # ArtifactInjector
├── multichannel_noise.py          # MultiChannelNoiseGenerator
├── templates.py                   # synthetic glitch / burst templates
├── utils.py                       # RNG helpers, normalization, fades
└── configs/
    ├── base_noise.yaml
    ├── temporal.yaml
    ├── artifact.yaml
    └── multichannel.yaml
```

---

# 10. Recommended Development Order

## Step 1 — Refactor `NoiseGenerator`
Implement:
- RNG support
- `build_psd()`
- `sample_stationary_gaussian_from_psd()`
- keep `generate_noise()` as convenience wrapper

## Step 2 — Implement `TemporalNoiseWrapper`
Minimal first version:
- piecewise stationarity
- spline drift
- cross-fade between segments

## Step 3 — Implement `ArtifactInjector`
Minimal first version:
- spectral lines
- synthetic glitches
- sparse impulse artifacts

## Step 4 — Implement `MultiChannelNoiseGenerator`
Minimal first version:
- independent channels
- shared + private correlated channels

## Step 5 — Add validation notebooks/scripts
Needed for:
- PSD sanity plots
- histogram / Gaussianity checks
- drift visualization
- correlation heatmaps
- artifact examples

---

# 11. Validation Requirements for Coding Agent

The implementation should include quick validation scripts for each module.

## Base `NoiseGenerator`
Validate:
- generated PSD matches target PSD on average
- variance scaling behaves correctly

## `TemporalNoiseWrapper`
Validate:
- visible segment-dependent statistics
- drift is smooth and low-frequency
- no sharp discontinuities after cross-fading

## `ArtifactInjector`
Validate:
- spectral lines appear at expected frequencies
- glitch count roughly matches configured rate
- sparse impulses create heavy tails in histogram

## `MultiChannelNoiseGenerator`
Validate:
- independent mode gives near-zero off-diagonal correlation
- shared/private mode gives controllable positive correlation
- correlation heatmap changes with `corr_strength`

---

# 12. Practical Minimal Deliverable

For a first coding pass, the agent should produce:

## Must implement
- refactored `NoiseGenerator`
- `TemporalNoiseWrapper`
- `ArtifactInjector`
- `MultiChannelNoiseGenerator`

## Must support
- single-channel stationary Gaussian baseline
- piecewise stationarity
- spline drift
- spectral lines
- simple synthetic glitches
- sparse impulse artifacts
- independent multichannel generation
- shared/private correlated multichannel generation

## Must include
- docstrings
- configuration examples
- a small validation notebook or script

---

# 13. Explicit Instruction to Coding Agent

## Implementation target
Build a modular noise simulation framework around the existing `NoiseGenerator` without rewriting the base PSD synthesis from scratch.

## Constraints
- keep the code clean and composable
- avoid a giant monolithic `generate_noise()` API
- make ablation easy
- use reproducible RNG handling
- keep the minimal version physically plausible and easy to validate

## Minimal external information requested
Please ask for the following if available:
1. sampling rate
2. one representative empirical PSD or PSD file
3. trace length(s) used in experiments
4. any known detector line frequencies
5. rough drift amplitude / timescale
6. rough glitch duration / amplitude ranges

If these are unavailable, implement with sensible defaults and document the assumptions clearly.

---

# 14. Bottom Line

The correct design is:
- **keep** `NoiseGenerator` as the stationary Gaussian spectral core
- **add** `TemporalNoiseWrapper`
- **add** `ArtifactInjector`
- **add** `MultiChannelNoiseGenerator`

This is the cleanest, most feasible path for the paper and for agent-assisted coding.
