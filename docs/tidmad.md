# TIDMAD representation-diagnosis arm

This package adapts the compact DELight two-stage transformer to the public
[TIDMAD](https://github.com/jessicafry/TIDMAD) waveform-denoising benchmark.
It is the parallel waveform arm of Paper 3: the scientific contrast holds the
architecture, data, seed, and training budget fixed and changes only the
measurement-coordinate objective (`mse` versus inverse-PSD-weighted `chi2`).

The frozen configuration has roughly 0.6 million parameters. It is a compact
domain transformer, not a foundation model. Foundation-model comparison and
LoRA perturbations are later, gated experiments; reinforcement learning is out
of scope.

## What is implemented

- self-contained, pinned TIDMAD data, split, PSD, and benchmark helpers under
  `_vendor/`;
- batched training and held-out validation through a PyTorch `DataLoader`;
- finite step budgets across repeated epochs;
- model-only and resumable checkpoints;
- streaming inference using the same standardization helpers as training,
  without materializing multi-gigabyte channels as float arrays;
- a fail-fast check of the two-channel HDF5 output contract;
- the unmodified upstream benchmark wrapper.

The vendored helpers came from
`noise-weighted-subspace-reconstruction@89a4a063db36ca271e2ffcf2a9438dee56d6def7`.

## Local verification

From `transformer/`:

```bash
.venv/bin/python -m pytest tidmad/tests -q
```

The tests use small synthetic HDF5 files and cover batching, train/validation
splits, both losses, model and resume checkpoints, STFT reconstruction, and the
two-channel denoised-file layout. They do not substitute for the public-data
benchmark.

## External data

Large HDF5 data and trained checkpoints stay outside Git. TIDMAD is public and
CC BY 4.0. The upstream repository supplies `download_data.py`; a minimal smoke
set is one training, one validation, and one noise-only science file:

```bash
git clone https://github.com/jessicafry/TIDMAD.git /path/to/TIDMAD
python /path/to/TIDMAD/download_data.py \
  --output_dir /external/tidmad \
  --train_files 1 \
  --validation_files 1 \
  --science_files 1 \
  --force
```

The data directory must also contain `tidmad_data_contract.json`. Training
refuses an unverified contract. At minimum, verify and record the release
convention that `timeseries/channel0001/timeseries` is the SQUID readout,
`timeseries/channel0002/timeseries` is the injected reference, the dtype is
`int8`, and the sampling rate is 10 MHz.

## Train, infer, and score

Run from `transformer/`. The 20-step command is a plumbing smoke only:

```bash
.venv/bin/python -m tidmad.train \
  --config tidmad/configs/t_mse.yaml \
  --data-dir /external/tidmad \
  --out /external/paper3-runs/tidmad-mse-smoke \
  --steps 20

.venv/bin/python -m tidmad.infer \
  --config tidmad/configs/t_mse.yaml \
  --checkpoint /external/paper3-runs/tidmad-mse-smoke/checkpoints/reconstruction_model_20.pt \
  --data-dir /external/tidmad \
  --out /external/tidmad \
  --model-tag paper3_mse_smoke

.venv/bin/python -m tidmad.score \
  --upstream /path/to/TIDMAD \
  --data-dir /external/tidmad \
  --model-tag paper3_mse_smoke \
  --coarse
```

For a resumed run, pass the matching
`reconstruction_resume_<step>.pt` file to `tidmad.train --resume`. Do not read
the confirmatory score until the scientific tolerance and whitening positive
control have been frozen.
