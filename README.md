# Consequence-Aware Failure Diagnostics for Particle Reconstruction

This repository is a research concept-note and feasibility package for studying
failure diagnostics in learned particle-reconstruction systems. It asks two
questions that ordinary distribution-shift detection does not answer:

1. Can a frozen model distinguish corrupted acquisition from clean physics that
   is under-represented in its training data?
2. Does the magnitude of a monitoring alarm track the downstream scientific
   damage caused by that shift?

The project is currently a **proposal with a validated development pilot**, not
a completed study or a production monitoring library. The main deliverable is
the three-page concept note in [`latex/paper3_proposal.pdf`](latex/paper3_proposal.pdf).

## Study design

The proposal defines three failure contracts:

- **N — acquisition/noise shift:** corrupted measurements such as module loss,
  hit thinning, jitter, drift, glitches, or covariance changes.
- **S — training-support shift:** clean, physically valid events from sparsely
  represented parts of the training distribution.
- **E — evaluation-contract fault:** deterministic pipeline errors such as
  incorrect units, coordinates, output semantics, or metrics.

N versus S is the main attribution problem; E is handled as a deterministic
validation gate. The proposed experiments evaluate detection, attribution with
abstention, monitoring cost, and the relationship between alarm magnitude and
scientific consequence. They compare layerwise representation monitoring with
input, output, uncertainty, MMD, classifier two-sample, and embedding baselines
under the same false-alert budget.

Two complementary testbeds are planned:

- **NuBench neutrino-telescope point clouds** provide a realistic frozen graph
  neural network and a 128-dimensional pre-head event representation.
- **Controlled waveforms** combine the public TIDMAD benchmark with the local
  noise simulator. The TIDMAD arm compares MSE and inverse-PSD-weighted training
  in a compact two-stage transformer; the simulator keeps assumed and realized
  covariance known.

## What has been run

The checked-in NuBench work is a feasibility test on the Hexagon Ice LE DynEdge
direction model:

- Rescoring 2,989,339 released predictions comes close to the published Table 6
  values, but the largest discrepancy (0.055 degrees or 0.051 percentage point)
  exceeds the predeclared strict 0.01-unit identity tolerance.
- Restoring the released 1,358,099-parameter checkpoint exposes a
  128-dimensional backbone representation, but CPU re-inference on 256 events
  differs from the released directions by 0.944 degrees at the median (7.708
  degrees at the 95th percentile).
- An exploratory module-dropout pilot shows the expected monotone response from
  0% to 50% dropout: median angular error rises from 15.28 to 25.12 degrees,
  standardized embedding displacement rises from 0.000 to 0.149, and 10-nearest
  neighbour retention falls from 1.000 to 0.393.

These results establish that the perturbation and embedding hook are useful for
protocol development. They **do not validate the proposal's scientific claims**:
the 256-event sample is deliberately enriched for high-multiplicity events, and
exact checkpoint/released-prediction parity remains unresolved. The full
interpretation and blocker are kept with the NuBench feasibility work, outside
this repository.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/noise_module/` | Validated stationary/nonstationary/multichannel noise and PSD simulation (numpy/scipy only) |
| `src/qp_simulator/` | Minimal standalone quasi-particle (QP) trace simulator (numpy only) |
| `src/reconstruction_model/` | DELight transformer reconstruction model + architecture catalog |
| `src/tidmad/` | TIDMAD band-frame STFT denoising arm (backbone from `reconstruction_model`, vendored Paper-1 benchmark helpers) |
| `notebooks/` | Smoke/inference notebooks and the noise-module tutorials |
| `scripts/` | Local/Condor training helpers and smoke tests |
| `containers/` | Runtime container image definition |
| `docs/` | Package documentation (`noise_module/`, `reconstruction_model.md`, `tidmad.md`) |
| `reference/` | Selected external training notes and legacy helper scripts |
| `scripts/nubench/` | NuBench feasibility scripts (migrated 2026-08-17, post-audit) |
| `results/` | Checked-in feasibility results with audit caveats |
| `docs/PAPER3_AUDIT.md` | Adversarial audit of the proposal, repo and plan (2026-08-17) |
| `docs/OPEN_DECISIONS.md` | Researched resolutions of the open technical decisions |
| `docs/REVISION_PLAN.md` | Canonical shared execution plan (migrated 2026-08-23; see its banner for the audit conflicts still to reconcile) |
| `docs/PERSONAL_RESEARCH_GUIDE.md` | Private working record and review log — **not for the shared view** |
| `reference/papers/` | Prior-art and testbed PDFs, with the novelty analysis (`papers.tsv` is the manifest; the README has the fetch loop) |
| `latex/paper3_proposal.tex` | Source for the three-page collaboration concept note |
| `latex/paper3_proposal.pdf` | Compiled proposal |
| `latex/figures/` | Proposal figures |
| `references.bib` | Working bibliography; the current proposal uses a self-contained bibliography in the TeX source |

## Python packages

This is a [`uv`](https://docs.astral.sh/uv/) workspace with four members sharing
one lockfile:

- `src/noise_module/` — `modular-noise-simulator` (numpy/scipy only)
- `src/qp_simulator/` — `qp-simulator` (numpy only)
- `src/reconstruction_model/` — `delight-reconstruction` (PyTorch CUDA build)
- `src/tidmad/` — `tidmad` (PyTorch CUDA build, plus h5py/scipy/PyYAML)

Install everything with:

```bash
uv sync
```

`reconstruction_model` and `tidmad` pin `torch==2.5.1+cu124` (Linux GPU nodes).
On macOS/CPU, override the index before syncing, or install just the noise
package, which has no PyTorch dependency:

```bash
uv sync --package modular-noise-simulator
```

Run the test suites (the noise package passes without any external data):

```bash
uv run pytest src/noise_module/tests
uv run pytest src/tidmad/tests      # 1 test needs external docs/tidmad_data_contract.json
```

The noise-module tutorials live in
[`notebooks/noise_module_tutorial.ipynb`](notebooks/noise_module_tutorial.ipynb) and
[`notebooks/noise_psd_1mhz_resampling_tutorial.ipynb`](notebooks/noise_psd_1mhz_resampling_tutorial.ipynb);
they import the installed package directly (no repository-path probing).

## Build the proposal

A LaTeX installation providing `extarticle`, `tcolorbox`, `booktabs`,
`tabularx`, `microtype`, and the other packages imported by the source is
required. Compile from `latex/` so the figure paths resolve:

```bash
cd latex
pdflatex -interaction=nonstopmode -halt-on-error paper3_proposal.tex
pdflatex -interaction=nonstopmode -halt-on-error paper3_proposal.tex
```

The second pass resolves internal references and the self-contained
bibliography.

## Reproduce the NuBench checks

The NuBench pilot scripts now live in [`scripts/nubench/`](scripts/nubench/)
and the feasibility results (with post-audit caveats) in
[`results/nubench_hexagon_ice_le_dynedge/`](results/nubench_hexagon_ice_le_dynedge/RESULT.md);
both were migrated from the external research folder on 17 August 2026 with
the audit fixes applied (`docs/PAPER3_AUDIT.md`, C11–C17).

The large NuBench database, released prediction Parquet file, and model
checkpoint are external artifacts and are not included in this repository.
The pilot script is designed for the official GraphNeT 1.8.0 CPU environment
and additionally imports PyTorch, PyTorch Geometric, NumPy, pandas, PyArrow,
scikit-learn, and Matplotlib. The metric-only script requires Polars.

Recompute the released-prediction metrics (post-audit: computes BOTH the
`is_track` and `interaction` groupings against Table 6):

```bash
python scripts/nubench/nubench_reference_metrics.py \
  --predictions /path/to/DynEdge_predictions.parquet \
  --output /path/to/reference_metrics.json
```

Run the paired module-dropout pilot (post-audit: uniform-random sampling,
nested severities, alignment assert, paired bootstrap CIs):

```bash
python scripts/nubench/nubench_smoke_pilot.py \
  --database /path/to/hexagon_ice_le.db \
  --checkpoint /path/to/DynEdge_checkpoint.pth \
  --released-predictions /path/to/DynEdge_predictions.parquet \
  --output-dir /path/to/pilot-output
```

The SQLite database must contain the NuBench `pulses_no_noise` and `mc_truth`
tables expected by the script. The pilot defaults to 256 balanced track/cascade
events and dropout fractions of 0, 0.1, 0.25, and 0.5. It writes
`pilot_metrics.csv`, `smoke_test.json`, and PNG/PDF plots. Because GraphNeT's
loader deserializes the released model, only use a checkpoint from a trusted
source.

## Current blocker and next step

Confirm the exact NuBench commit, Python/PyTorch/PyG versions, detector class,
graph-construction backend, and inference command used to produce the released
Hexagon Ice LE DynEdge predictions. Confirmatory experiments should remain
gated until clean re-inference either matches those predictions within an
agreed backend tolerance or the deterministic variation is explained and the
downstream metric passes a predeclared tolerance.

The parallel TIDMAD code path is locally validated on synthetic HDF5 fixtures.
Its remaining blocking gate is the public-data smoke run because the large
TIDMAD files are intentionally external; follow
[`docs/tidmad.md`](docs/tidmad.md).
