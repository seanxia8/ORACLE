# Modular noise simulator

Composable noise generation for detector-physics waveform studies.

Copyright (c) 2026 Dowling Wong. Released under the MIT licence (see
[`LICENSE`](LICENSE)). **If you use this package in published work, please cite
it** — see [`CITATION.cff`](../../CITATION.cff) at the repository root.

## What it provides

| Module | Role |
|---|---|
| `NoiseGenerator` | stationary Gaussian single-channel synthesis from an arbitrary one-sided PSD |
| `multichannel_noise` | correlated channels in independent, shared-private and low-rank latent modes, returning both the implied and the realized covariance |
| `temporal_noise` | non-stationarity, piecewise stationarity, drift, local variance change |
| `artifact_injector` | spectral lines, glitches, bursts, sparse non-Gaussian artifacts |
| `psd_resampling` | alias-folding and in-band resampling of PSD densities between sampling rates |
| `validation` | stationarity, Gaussianity, CSD-ensemble and artifact checks with bootstrap intervals |
| `spectral_models` | composable analytic PSD components (white, power law, Lorentzian, roll-off, line, band-limited) |
| `reference_budget` | closed-form athermal-calorimeter noise budget; regenerates the reference tables |
| `calibration`, `streaming`, `non_gaussian`, `templates` | presets, chunked generation, non-Gaussian innovations, pulse templates |

Design rationale is in
[`docs/noise_module/noise_generator_modular_design_spec.md`](../../docs/noise_module/noise_generator_modular_design_spec.md).

## Reference data

The `data/Al2O3_Al_athermal/*.dat` tables are **generated build artifacts**, not
source data, and are git-ignored. Recreate them with:

```bash
python scripts/regenerate_reference_asd.py
```

See [`data/Al2O3_Al_athermal/README.md`](data/Al2O3_Al_athermal/README.md) for
the provenance of the reference budget and the evidence that it is analytic.

## Tests

```bash
pytest src/noise_module/tests
```
