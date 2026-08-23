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
- single-window training with gradient accumulation to the declared
  `device_batch_size` (there is deliberately no DataLoader/worker pool; the
  fit stream is a chronological generator that cycles the fit split — every
  pass visits the identical window sequence, which keeps the two arms exactly
  paired), plus a held-out chronological validation pass at `eval_period`;
- a declared step budget whose realised value is archived: `run_config.json`
  records the effective number of passes over the fit split and is finalised
  with `completed_steps` and a `status` of `completed`/`interrupted`, so a
  SIGTERM'd run can never masquerade as a full one;
- resumable checkpoints (`checkpoint_<step>.pt` plus a `latest.pt` copy)
  round-tripping model, both optimisers, both schedulers and the step counter;
- a startup whitening positive control: the whitening identity error of J(f)
  is measured on held-out calibration windows at every training start,
  archived, and optionally enforced via `whitening_error_max` (T1.6);
- the single-difference gate runs in production: pass `--paired-config` with
  the other arm's YAML and `assert_configs_differ_only_in_loss` executes at
  startup;
- streaming inference using the shared `loss.per_row_stats`/`unstandardise`
  helpers, with the `+128` release shift removed before the int8 write, the
  file tail denoised via a final overlapped window, and a fail-fast check of
  the written two-channel HDF5 product (layout, length, saturation);
- the unmodified upstream benchmark wrapper.

The vendored helpers came from
`noise-weighted-subspace-reconstruction@89a4a063db36ca271e2ffcf2a9438dee56d6def7`.

## Local verification

From the repository root:

```bash
uv run pytest src/tidmad/tests -q
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

Run from the repository root with `src/` on `PYTHONPATH` (or inside the uv
environment). The 20-step command is a plumbing smoke only. Training writes
`checkpoint_<step>.pt` files plus a `latest.pt` copy into `--out`:

```bash
uv run python -m tidmad.train \
  --config src/tidmad/configs/t_mse.yaml \
  --paired-config src/tidmad/configs/t_chi2.yaml \
  --data-dir /external/tidmad \
  --out /external/paper3-runs/tidmad-mse-smoke \
  --steps 20

uv run python -m tidmad.infer \
  --config src/tidmad/configs/t_mse.yaml \
  --checkpoint /external/paper3-runs/tidmad-mse-smoke/checkpoint_00000020.pt \
  --data-dir /external/tidmad \
  --out /external/tidmad \
  --model-tag paper3_mse_smoke

uv run python -m tidmad.score \
  --upstream /path/to/TIDMAD \
  --data-dir /external/tidmad \
  --model-tag paper3_mse_smoke \
  --coarse
```

For a resumed run, pass `latest.pt` (or a specific `checkpoint_<step>.pt`) to
`tidmad.train --resume`, or `--resume auto`. Do not read the confirmatory score
until the scientific tolerance (kappa_m) and the whitening positive-control
threshold (`whitening_error_max`) have been frozen; note that the upstream
denoising score is a development consequence metric only — the TIDMAD authors
state it "lacks direct relevance to fundamental science" (see
`docs/OPEN_DECISIONS.md`).
