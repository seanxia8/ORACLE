# Local L40S training

Run this directly on a machine with a working NVIDIA L40S driver. It does not
use HTCondor, a proxy, or XRootD.

GPU and ECC health check:

```bash
./scripts/train_local_l40s.sh check
```

Train the prepared compact dataset:

```bash
./scripts/train_compact_l40s/submit.sh prepare
./scripts/train_local_l40s.sh compact
```

Train the full locally mounted dataset:

```bash
./scripts/train_local_l40s.sh full-pilot
./scripts/train_local_l40s.sh full
```

Train the full locally mounted dataset with the heavier triangular-pairwise
variant:

```bash
./scripts/train_triangular_local_l40s.sh full-pilot
./scripts/train_triangular_local_l40s.sh full
```

`full-pilot` runs one conservative epoch first. Use it to verify GPU memory,
dataset throughput, representative validation metrics, and checkpoint output
before starting the default 20-epoch `full` profile. The full profile starts
fresh with the same tested device/global batch sizes; it does not resume the
pilot's one-epoch cosine schedule, which has already decayed its learning rate
to zero.

Choose another physical GPU:

```bash
GPU_INDEX=1 ./scripts/train_local_l40s.sh compact
```

Set `WANDB_API_KEY` before launching to enable W&B checkpoint artifacts.
Otherwise all checkpoints stay under `artifacts/`.
