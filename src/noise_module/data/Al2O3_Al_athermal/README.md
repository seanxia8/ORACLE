# Al2O3/Al athermal reference budget — provenance

## Summary

The two-column tables in this directory (`Johnson_noise.dat`, `SQUID_noise.dat`,
`TD_noise.dat`, `Er_noise.dat`, `total_noise.dat`, `signal.dat`) were **supplied
to this project as an external reference**, alongside a notebook
(`noise_sample/read_dat_1.ipynb`) that is not part of this repository. They were
used to check that the noise generator could reproduce a realistic detector
noise budget.

**They contain no measured data.** They are an analytic noise budget sampled on
a synthetic frequency grid. This was established numerically, and every file is
now reproduced from the closed-form model in
[`noise_module/reference_budget.py`](../../reference_budget.py) to within a few
units in the last place (<= 4e-16 relative for the five noise channels,
<= 1e-15 for the signal response in its deepest tail). The tables are therefore treated as **regenerable build
artifacts, not as source data**: they are git-ignored and are recreated on
demand by

```bash
python scripts/regenerate_reference_asd.py
```

Nothing of external origin is redistributed with this package. The only files
kept under version control here are this README and
`al2o3_athermal_fit.json` (the five fitted spectral parameters).

## Evidence that the tables are analytic

Reproduce with `pytest src/noise_module/tests/test_reference_budget.py`.

| Observation | Value | Why it matters |
|---|---|---|
| Grid | 16384 points, geometric, 1 Hz to 30 MHz | 2^14 points on an exactly constant ratio (1.001051440977 to 15 digits) is a generated grid, not an instrument's output |
| `Johnson_noise.dat` | constant at 5.505218127021067e-2 for all 16384 rows | a measurement is never flat to 18 significant digits over 7.5 decades |
| `total_noise.dat` | equals the quadrature sum of the four components to 3.3e-16 | machine epsilon: the total was *computed* from the components, not measured alongside them |
| Thermal corner | 159.15494309189538 Hz = 1/(2*pi*1 ms) | a round chosen time constant, tau = 1.000000000 ms |
| SQUID knee | exactly 100 Hz, white floor 0.09, 1/f amplitude 9.0 | round design values, hand-set |
| `signal.dat` | two-pole, tau_rise = 5 us, tau_decay = 1 ms, plateau 2.0e-3 | both time constants round; reproduced to <= 1e-15 |

## The model

One-sided PSDs in (readout unit)^2/Hz. The `.dat` files store amplitude
spectral density, i.e. `sqrt(PSD)`.

| Channel | Form | Constants |
|---|---|---|
| Johnson | white: `S_J` | `S_J = 3.0307426626081345e-3` |
| SQUID | `S_sq * (1 + f_knee / f)` | `S_sq = 0.09`, `f_knee = 100 Hz` |
| TD (thermodynamic) | `S_TD / (1 + (f / f_TD)^2)` | `S_TD = 5.976157531368565e-2`, `tau = 1 ms` |
| Er (paramagnetic spin) | `S_Er * (f / 1 Hz)^alpha` | `S_Er = 0.5442319404674618`, `alpha = -0.9` |
| total | quadrature sum of the above | — |
| signal | `A / sqrt((1+(f/f_dec)^2)(1+(f/f_rise)^2))` | `A = 2.0e-3`, `tau_rise = 5 us`, `tau_decay = 1 ms` |

Each functional form is standard and independently citable — a white Johnson
term, the conventional two-parameter (white + 1/f) SQUID readout model, a
single thermal pole, and a sub-1/f paramagnetic spin term. None of them is
specific to any one experiment's design. Of the constants, `S_sq`, `f_knee`,
`A`, `tau_rise` and `tau_decay` are round design values; the remaining three
amplitudes are those that reproduce the supplied curves, and are retained only
so the regenerated tables match the reference bit for bit.

Because the budget is now parameterised
(`noise_module.reference_budget.AthermalNoiseBudget`), it can be retuned for
any comparable athermal-calorimeter channel. `AL2O3_AL_ATHERMAL` is one preset,
not a hard-wired dataset.

## Figures

`spectrum_detector_signal_noise.svg` in this directory is an **externally
supplied matplotlib figure** (embedded creation date 2024-12-04, i.e. predating
this project). A PNG of it is currently included in
`latex/paper3_proposal.tex` as figure panel (b), whose caption calls panels
(b)-(c) "project outputs" — that is inaccurate for (b) and should be corrected,
or the panel regenerated from `reference_budget.py` so that it genuinely is a
project output. See the repository's provenance notes.
