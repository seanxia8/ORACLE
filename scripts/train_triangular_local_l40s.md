# Local L40S Triangular Pairwise Training

This script trains the full locally mounted H5 dataset with the
`triangular_pairwise` architecture. It wraps `scripts/train_local_l40s.sh` and
sets triangular-specific model names, W&B names, cache paths, checkpoint paths,
and conservative L40S batch defaults. It defaults to `GPU_INDEX=1`.

Check the selected L40S:

```bash
./scripts/train_triangular_local_l40s.sh check
```

Run a one-epoch full-dataset pilot:

```bash
./scripts/train_triangular_local_l40s.sh full-pilot
```

Run the default 20-epoch full-dataset training:

```bash
./scripts/train_triangular_local_l40s.sh full
```

Defaults:

```text
RECONSTRUCTION_MODEL_VARIANT=triangular_pairwise
RECONSTRUCTION_LOCAL_DATA_PATH=/ceph/srv/dwong/training_samples_h5
RECONSTRUCTION_SPATIAL_TARGET_INDICES=0,1
RECONSTRUCTION_TOTAL_BATCH_SIZE=16
RECONSTRUCTION_DEVICE_BATCH_SIZE=2
GPU_INDEX=1
```

Override values from the shell when needed:

```bash
GPU_INDEX=1 \
RECONSTRUCTION_DEVICE_BATCH_SIZE=4 \
RECONSTRUCTION_TOTAL_BATCH_SIZE=32 \
./scripts/train_triangular_local_l40s.sh full-pilot
```
