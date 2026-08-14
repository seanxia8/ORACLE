# Compact Pairwise Training: NEMO2 L40S

This profile needs neither a VOMS proxy nor XRootD. It extracts 10 real events
for every ER/NR and energy combination from the local full dataset, producing
12 Condor-transfer files below 100 MB each.

The resulting training set has 120 real events covering:

```text
ER, NR x 10, 20, 50, 100, 200, 500
```

It trains `pairwise_channel_masking` for 20 epochs. The model-only inference
checkpoint, resume checkpoint, and `run_config.json` are returned in
`artifacts/`. When `WANDB_API_KEY` is set, the same checkpoints are also stored
as W&B artifacts.

```bash
./scripts/train_compact_l40s/submit.sh prepare
./scripts/train_compact_l40s/submit.sh dry-run
./scripts/train_compact_l40s/submit.sh submit
./scripts/train_compact_l40s/submit.sh status
```

The prepared input data lives under `.condor_data/compact_l40s/` and can be
regenerated from the local full H5 dataset.
