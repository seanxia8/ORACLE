# Reconstruction Model Catalog

This package collects the Transformer-family architectures from the old
`src/DELight_reconstruction-dev1` snapshot in the active training package. The
training script can select these variants with `RECONSTRUCTION_MODEL_VARIANT`.

## Layout

```text
reconstruction_model/models/
  current_compact.py              current compact model used by the root package
  original.py                     legacy baseline Transformer
  pairwise.py                     pairwise channel-relation Transformer
  pairwise_channel_masking.py     pairwise model with channel masking helpers
  triangular_pairwise.py          pairwise model with triangular pair updates
  cnn_transformer.py              grouped Conv1D encoder plus spatial Transformer
  integration_classifier.py       optional XGBoost baseline, not a Transformer
  registry.py                     lazy architecture registry
  cache/
    edge_feats.npy                56x56 pairwise edge features from old repo
    pos_diff.npy                  pairwise position differences for CNN/current variants
  position_MMC_V2.dat             detector channel positions used to regenerate edge features
```

The copied model variants were sourced from:

```text
src/DELight_reconstruction-dev1/reconstruction/models/
```

They have been lightly adapted so they import `reconstruction_model.models.muon` and
load package-local cache files instead of reaching back into the old
`reconstruction` package.

## Available Variants

| Registry name | File | Outputs | Notes |
| --- | --- | --- | --- |
| `current_compact` | `current_compact.py` | `spatial_pred, energy_pred` | Same architecture as the current root `reconstruction_model/model.py`. |
| `original` | `original.py` | `spatial_pred, energy_pred, class_logits` | Legacy temporal/channel Transformer baseline. |
| `pairwise` | `pairwise.py` | `spatial_pred, energy_pred, class_logits` | Updates channel-pair features and uses them as attention bias. |
| `pairwise_channel_masking` | `pairwise_channel_masking.py` | `spatial_pred, energy_pred, class_logits` | Pairwise model with train/inference channel masking support. |
| `triangular_pairwise` | `triangular_pairwise.py` | `spatial_pred, energy_pred, class_logits` | Pairwise model with triangular multiplication and attention updates. |
| `cnn_transformer` | `cnn_transformer.py` | `spatial_pred, energy_pred, class_logits` | Grouped Conv1D feature extractor followed by channel Transformer blocks. |
| `integration_classifier` | `integration_classifier.py` | XGBoost classifier outputs | Optional non-Transformer baseline requiring extra packages. |

## Usage

Import this package directly from the repo root:

```bash
python - <<'PY'
from reconstruction_model.models import available_models, create_model

print(available_models())
model, config = create_model("pairwise_channel_masking")
print(config)
print(type(model).__name__)
PY
```

For a direct import:

```python
from reconstruction_model.models.pairwise_channel_masking import Transformer, TransformerConfig

config = TransformerConfig()
model = Transformer(config)
```

All Transformer variants expect waveform tensors shaped like:

```text
(batch, channels, samples)
```

Most legacy variants assume:

```text
channels = 56
samples = 65536
```

The current compact model can be configured differently, but checkpoints only
load into the exact architecture that produced them.

## Cache Files

The pairwise variants load precomputed detector geometry features from:

```text
reconstruction_model/models/cache/edge_feats.npy
reconstruction_model/models/cache/pos_diff.npy
```

Regenerate `edge_feats.npy` from the package-local `position_MMC_V2.dat` with:

```bash
python -m reconstruction_model.models.precompute_pairwise_features
```

## Training Integration Notes

The old repo trained the channel-masking pairwise model by default:

```python
from reconstruction.models.model_pairwise_channel_masking import TransformerConfig, Transformer
```

The equivalent new import is:

```python
from reconstruction_model.models.pairwise_channel_masking import TransformerConfig, Transformer
```

To train one of these variants with the current root training script:

```bash
RECONSTRUCTION_MODEL_VARIANT=pairwise_channel_masking \
RECONSTRUCTION_RECOIL_CLASSIFICATION=1 \
RECONSTRUCTION_SPATIAL_TARGET_INDICES=0,1 \
python -m reconstruction_model.train
```

Pairwise variants output `(x, y)` spatial predictions by default, while the
current dataset target is `(x, y, z)`, so the submission scripts train them with
`RECONSTRUCTION_SPATIAL_TARGET_INDICES=0,1`.

## Optional Dependencies

`integration_classifier.py` is kept for reference and baseline experiments, but
it requires optional packages not needed by the Transformer models:

```text
xgboost
scikit-learn
seaborn
tqdm
```

The package `__init__` and registry are lazy, so these optional dependencies are
not imported unless you explicitly load `integration_classifier`.
