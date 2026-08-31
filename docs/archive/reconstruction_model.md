# DELight Transformer Reconstruction

This document describes the `src/reconstruction_model/` package in THIS
repository (ORACLE): the DELight transformer reconstruction model and its
architecture catalog. The large training data is intentionally not included.

> **Provenance note (audit C20).** An earlier revision of this file was carried
> over verbatim from the upstream standalone repository and described a
> different layout — including the claim that `src/reconstruction_model/` "has
> been removed", which is false here: it is the live package that
> `src/tidmad_transformer/` imports its backbone, optimisers, schedulers, and checkpoint
> helpers from. Path statements below have been corrected to this repository.

## Repository Status

The active Python package in this repository is:

```text
src/reconstruction_model/
```

Use that package for training, inference, checkpoint loading, and model
selection. The model-catalog architectures live in:

```text
src/reconstruction_model/models/
```

- `src/reconstruction_model/` is the active package. `src/tidmad_transformer/` imports
  `reconstruction_model.model`, `reconstruction_model.muon`,
  `reconstruction_model.schedulers`, and `reconstruction_model.checkpoints`
  from it.
- `src/reconstruction_model/models/` is the active architecture catalog.
  Known duplication (audit C23): `models/current_compact.py` is a 6-line fork
  of `../model.py` (import-path difference only) and both `muon.py` copies must
  stay in sync; several registry entries reference model modules not present in
  this repository and will `ImportError` if selected. Consolidation is tracked
  in `docs/OPEN_DECISIONS.md`.
- `src/reconstruction_model/legacy/` preserves old evaluation, visualisation,
  normalization, and XGBoost scripts that still need adaptation before they are
  treated as production code.
- `reference/finetuning/` keeps a small selected set of Transformer
  training/fine-tuning notebooks.
- Generated outputs such as `artifacts/`, `cache/`, `results/`, logs, and local
  checkpoints should stay out of Git unless a small file is intentionally kept
  as documentation.

### Cleanup And Merge Map

Recommended source of truth:

```text
reconstruction_model/          active package
reconstruction_model/models/   active model variants
reconstruction_model/legacy/   old utilities kept for migration/reference
scripts/                       active local/Condor helpers
notebooks/                     small smoke/inference notebooks
containers/                    runtime image definition
submit_ref/                    HTCondor reference templates
src/finetuning_references/     selected external Transformer-training notes
```

Content already migrated from the old source copies:

- Old model variants and geometry caches now live in
  `reconstruction_model/models/`.
- Old evaluation, visualization, normalization, and XGBoost utilities now live
  in `reconstruction_model/legacy/`.
- Selected Transformer fine-tuning/training reference notebooks now live in
  `src/finetuning_references/`.

Content still worth reviewing:

- `reference/legacy_scripts/`: old trace-generation helpers kept for reference;
  the previous phonon-simulator scripts were removed for provenance/rights
  reasons and the clean QP path is now provided by the `qp_simulator` package.
- `src/results/`: keep only summarized metrics in documentation or a small
  curated `reports/` file; do not keep raw generated result dumps by default.
- `submit_ref/`: keep if these templates still document working Condor
  patterns; otherwise merge the useful comments into `scripts/`.
- `src/cache/`, `src/results/`, `src/log.txt`, `src/out.txt`, `src/err.txt`,
  and `src/docker_stderror`, because they are generated/local outputs.
- Generated Condor logs under `scripts/train_*/*.log`, `*.out`, and `*.err`.

### Naming Note

Use one public package name:

```text
reconstruction_model
```

Keep model variants under:

```text
reconstruction_model/models/
```

The old plural package name `reconstruction_models` has been retired.

## What Is Included

- `reconstruction_model/train.py`: main training entry point.
- `reconstruction_model/model.py`: transformer model definition.
- `reconstruction_model/dataset.py`: data discovery, loading, train/validation split, and dataloaders.
- `reconstruction_model/models/`: selectable architecture catalog, including
  the pairwise-channel-masking model used by current full-run scripts.
- `reconstruction_model/muon.py`: custom Muon optimiser.
- `reconstruction_model/schedulers.py`: cosine learning-rate schedule with optional warmup.
- `reconstruction_model/checkpoints.py`: checkpoint save/load helpers.
- `tidmad/`: Paper 3's public TIDMAD representation-diagnosis arm; see
  [`tidmad/README.md`](tidmad/README.md).
- `reconstruction_model/utils.py`: utility functions.
- `pyproject.toml` and `uv.lock`: Python dependency definition and lock file.
- `run.sh`: batch-job runtime script used by Condor.
- `scripts/`: smoke tests, data preparation, local training, and HTCondor
  submission helpers.
- `containers/Dockerfile`: container image definition for remote/runtime use.
- `notebooks/`: small smoke/inference notebooks.

The placeholder `main.py`, old logs, temporary `uv-*` lock files, and training
data are not required for training. Old logs, temporary lock files, and
training data are not included here.

## Training Entry Point

The actual training command is:

```bash
python -m reconstruction_model.train
```

Do not use `main.py` for training. In the original repository it only printed a
hello message.

## Environment Setup

This project uses `uv`.

```bash
uv sync
source .venv/bin/activate
```

The key dependencies are PyTorch `2.5.1+cu124`, NumPy, h5py, pandas,
zstandard, matplotlib, jaxtyping, and wandb.

Training expects a CUDA GPU. Production runs support compute capability 7.0 or
higher: V100/V100S uses scaled FP16, while A100/L40S uses BF16.

## Expected Data Layout

The training data is not stored in this directory. By default,
`reconstruction_model/dataset.py` expects local data under:

```text
training_data/
  train/
    ER/
      traces_energy_10.zst
      meta_energy_10.h5
      traces_energy_20.zst
      meta_energy_20.h5
      ...
    NR/
      traces_energy_10.zst
      meta_energy_10.h5
      traces_energy_20.zst
      meta_energy_20.h5
      ...
```

The important defaults are:

```python
local_data_path = PROJECT_ROOT / "training_data"
train_path = "train"
recoil_types = ["ER", "NR"]
max_seq_len = 65536
train_split = 0.8
val_split = 0.2
```

Each energy/recoil pair should have both files:

```text
traces_energy_<ENERGY>.zst
meta_energy_<ENERGY>.h5
```

The metadata HDF5 file must contain an `events` dataset with fields used by the
loader:

- `x`
- `y`
- `z`
- `energy`
- `type_recoil`
- `no_noise`
- `quantize`

The metadata file attributes should include:

- `n_channels`
- `trace_samples`
- `trace_dtype`

## How Data Is Loaded

Training and validation are both created from the `training_data/train`
directory. For every discovered `(energy, recoil_type)` group, the dataset uses
a deterministic NumPy permutation with seed `0`, then takes:

- 80 percent for training
- 20 percent for validation

The model input returned by the dataloader has shape:

```text
batch, channels, samples
```

The targets are:

- spatial target: `(x, y, z)`
- energy target: scalar energy
- recoil type: string label, optionally used for ER/NR classification

The default supervised training loss is:

```text
spatial_weight * MSE(spatial_prediction, spatial_target)
+ energy_weight * MSE(energy_prediction, energy_target)
+ class_weight * BCEWithLogits(class_logits, recoil_class)
```

Classification is enabled with `RECONSTRUCTION_RECOIL_CLASSIFICATION=1`. The
pairwise variants usually predict `(x, y)`, so production pairwise runs set
`RECONSTRUCTION_SPATIAL_TARGET_INDICES=0,1`.

## Model Summary

The model is defined in `reconstruction_model/model.py`.

The default transformer config is:

```python
d_model = 256
d_ff = 1024
max_seq_len = 65536
patch_len = 64
patch_stride = 64
n_head = 4
n_time_layers = 1
n_channel_layers = 1
```

The forward pass does the following:

1. Normalises each channel sequence.
2. Splits the waveform into length-64 patches.
3. Applies temporal transformer layers with rotary embeddings.
4. Reshapes across channels.
5. Applies channel/spatial transformer layers.
6. Mean-pools over patches and channels.
7. Predicts `(x, y, z)` and energy with separate heads.

## Optimisation

The model uses two optimisers:

- AdamW for patch embedding, output heads, positional embeddings, and auxiliary parameters.
- Muon for matrix-valued transformer block weights.

Both optimisers use cosine learning-rate schedules with optional linear warmup.

Default training values are currently hard-coded in
`reconstruction_model/train.py`:

```python
num_steps = 100000
eval_step_period = 10
save_checkpoint_period = 250
total_batch_size = 64
device_batch_size = 16
num_workers = 1
adamw_lr = 0.001
muon_lr = 0.001
grad_clip = 1.0
wandb_run = True
```

Gradient accumulation is:

```python
grad_accum_steps = total_batch_size // device_batch_size
```

With the defaults, this is `64 // 16 = 4` micro-batches per optimiser step.

## Running Locally

From this directory:

```bash
uv sync
source .venv/bin/activate
python -m reconstruction_model.train
```

Before running, make sure `training_data/train/ER` and
`training_data/train/NR` exist and contain the `.zst` trace files plus `.h5`
metadata files.

## Smoke Test

To check the full local CUDA training path with a tiny generated dataset:

```bash
uv run python scripts/smoke_test_training.py
```

This creates small synthetic files under `artifacts/smoke_test/training_data`,
then runs the normal training loop with the real dataloader, Transformer,
AdamW, Muon, cosine schedulers, checkpointing, and optional W&B logging. To
include W&B without requiring network sync:

```bash
uv run python scripts/smoke_test_training.py --wandb-mode offline
```

The companion notebook is:

```text
notebooks/local_gpu_smoke_training.ipynb
```

If you do not want Weights & Biases logging, edit `TrainingConfig` in
`reconstruction_model/train.py` and set:

```python
wandb_run = False
```

## Running With Condor

From this directory, make sure the runtime script is executable and submit the
job:

```bash
cd /ceph/srv/dowling/transformer
chmod +x run.sh
condor_submit submit.jdl
```

`submit.jdl` runs:

```text
Executable = ./run.sh
```

`run.sh` does the following:

1. Checks that the GPU compute capability is at least 7.0.
2. Selects BF16 on Ampere-or-newer GPUs or scaled FP16 on V100/V100S.
3. Uses direct TopAS Ceph access or stages the sharded H5 dataset on NEMO2.
4. Installs `uv` if needed.
5. Runs `uv sync`.
6. Activates the virtual environment.
7. Runs `python -m reconstruction_model.train`.
8. Publishes model and resume checkpoints through XRootD.

Before submitting, export the Weights & Biases API key:

```bash
export WANDB_API_KEY="..."
```

For shared or reproducible runs, it is better to provide this through the job
environment instead of storing a personal key in the script.

After submission, monitor jobs and their site-specific logs with:

```bash
./scripts/train_full_l40s/submit.sh status
./scripts/train_full_a100/submit.sh status
```

## Checkpoints

By default checkpoints are written to:

```text
reconstruction_model/training_checkpoints/
```

At each checkpoint step, two files are saved:

```text
reconstruction_model_<STEP>.pt
reconstruction_resume_<STEP>.pt
```

The model checkpoint is a plain state dict for inference. The resume checkpoint
also contains model/config metadata, optimizer and scheduler state, AMP scaler
state, step/epoch, and RNG state. Production runs publish these files plus
`run_config.json` and `latest.json` under a run-specific XRootD directory.

Resume a run with:

```bash
RECONSTRUCTION_RESUME_CHECKPOINT=/path/to/reconstruction_resume_<STEP>.pt \
python -m reconstruction_model.train
```

## Model Architecture Catalog

Legacy Transformer variants from `src/DELight_reconstruction-dev1` are now
available from the active training package under:

```text
reconstruction_model/models/
```

This package includes the original Transformer, pairwise Transformer,
pairwise-channel-masking Transformer, triangular-pairwise Transformer,
CNN+Transformer variant, and the current compact model. It also carries the
small pairwise feature cache files needed to instantiate those architectures.

Use the lazy registry to inspect or instantiate variants:

```bash
python - <<'PY'
from reconstruction_model.models import available_models, create_model

print(available_models())
model, config = create_model("pairwise_channel_masking")
print(config)
PY
```

Set `RECONSTRUCTION_MODEL_VARIANT=pairwise_channel_masking` when launching
`python -m reconstruction_model.train` to train the DELight pairwise model used
by the L40S/A100 submission scripts.

See `reconstruction_model/models/README.md` for the model table, expected output
signatures, cache notes, and optional XGBoost baseline dependencies.

## Current Best Direction

The strongest current line of work is `pairwise_channel_masking`. It combines
temporal waveform modeling, detector channel geometry, pairwise channel updates,
and stochastic top/bottom channel masking. The completed full local L40S run:

```text
artifacts/pairwise_full_local_l40s/checkpoints/20260623_134638
```

trained `pairwise_channel_masking` for 20 epochs over ER/NR events at energies
`10,20,50,100,200,500`. It used about 6.8M trainable parameters and produced
late-run validation metrics around:

```text
final step 150000:
  val spatial RMSE: 5.03
  val energy RMSE:  9.95
  val class acc:    0.78

best logged validation loss:
  step 74750
  val spatial RMSE: 5.49
  val energy RMSE:  6.55
  val class acc:    0.78

best logged spatial RMSE:
  step 102500
  val spatial RMSE: 3.67
  val energy RMSE:  15.08
  val class acc:    0.91
```

Validation currently samples a small number of batches, so these checkpoint
rankings are noisy. Before changing the architecture heavily, add a fixed,
larger evaluator over full shards or stratified batches and re-score several
checkpoints, especially steps `74750`, `92500`, `102500`, and `150000`.

Recommended model priority:

1. Keep `pairwise_channel_masking` as the main architecture.
2. Use `current_compact`, `original`, and `cnn_transformer` as baselines.
3. Treat `triangular_pairwise` as a heavier ablation after the evaluation
   protocol is stable.

## Research Roadmap

Short-term improvements should focus on evaluation, loss scaling, and
pretraining rather than immediate architecture churn.

### Evaluation

- Build a deterministic test/evaluation script that loads a checkpoint and
  computes metrics over fixed shards or a fixed stratified sample.
- Report metrics by energy, recoil type, channel-mask mode, and full aggregate.
- Select best checkpoints by this evaluator, not by the final training step.

### Loss Functions

Current MSE training can be dominated by energy spikes. Study these losses in a
controlled sweep:

1. Normalize spatial and energy targets, then keep MSE as the baseline.
2. Tune `RECONSTRUCTION_SCALAR_LOSS_WEIGHTS` for spatial, energy, and class
   terms.
3. Compare SmoothL1/Huber loss for robustness to outlier events.
4. Add uncertainty-aware position prediction with a Gaussian negative
   log-likelihood:

```text
L_pos = (target - mu)^T Sigma^-1 (target - mu) + log det Sigma
```

This is the useful Mahalanobis-style direction: the model predicts both the
position and event-dependent uncertainty. Start with diagonal covariance, then
try full 2D covariance for `(x, y)`.

### Optimisation

The current AdamW plus Muon setup is a good baseline:

- AdamW handles embeddings, heads, norms, and auxiliary parameters.
- Muon handles matrix-valued Transformer block weights.

Useful ablations:

- AdamW+Muon versus AdamW-only.
- Warmup of `1000` to `5000` steps versus no warmup.
- AdamW weight decay `0.0` versus `0.01`.
- Resume from a strong checkpoint with lower learning rate.
- Optional EMA weights for evaluation.

### Foundation-Model Pretraining

The Transformer can be studied with LLM-like training ideas. In this domain:

```text
waveform patches -> tokens
channels         -> detector token groups
geometry cache   -> spatial/edge positional bias
reconstruction   -> downstream fine-tuning task
```

Promising pretraining objectives:

1. Masked waveform modeling: mask time patches and reconstruct them.
2. Masked channel modeling: hide top, bottom, or random channels and predict
   missing channel representations or traces.
3. Contrastive event learning: two augmentations of the same event should have
   nearby embeddings; different events should separate.
4. Multi-task supervised fine-tuning: fine-tune the pretrained encoder for
   `(x, y)`, energy, ER/NR classification, and uncertainty.

The foundation-model study question is:

```text
Does self-supervised detector-waveform pretraining improve sample efficiency,
calibration, and cross-energy generalization?
```

A clean experimental sequence is:

1. Train the current supervised `pairwise_channel_masking` baseline.
2. Pretrain the same encoder with masked patch/channel objectives.
3. Fine-tune on all labels.
4. Fine-tune on low-data subsets to measure sample efficiency.
5. Test transfer to held-out energies, recoil types, noise regimes, or threshold
   data.

## Full Remote Runs

The production entry points are:

```text
scripts/train_full_l40s/
scripts/train_full_a100/
```

Both production routes require a recent successful site probe, a CMS proxy
valid for at least 170 hours, and `WANDB_API_KEY`. The L40S route uses the proxy
for authenticated XRootD dataset staging and publishes checkpoints to both
XRootD and W&B. See the README in each directory for exact commands.

## Recommended Next Cleanup

For reproducible research, add:

```text
configs/
  train_default.yaml
  data_local.yaml
  condor_a100.yaml

runs/
checkpoints/
cache/
```

Then update `train.py` so experiments can be launched as:

```bash
python -m reconstruction_model.train \
  --train-config configs/train_default.yaml \
  --data-config configs/data_local.yaml \
  --run-dir runs/baseline_001
```

Each run should save:

- resolved config
- git commit or source snapshot
- logs
- metrics
- checkpoints
- W&B run ID, if used
