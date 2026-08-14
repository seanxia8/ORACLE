# Full Pairwise Training: NEMO2 L40S

This run trains `pairwise_channel_masking` for 20 epochs over all 3,000 H5
shards covering ER/NR energies `10,20,50,100,200,500`.

NEMO2 does not mount `/ceph/srv`, so the job requests 2.6 TB scratch and stages
the 2.4 TB dataset once from:

```text
root://ceph-node-j.etp.kit.edu//dwong/training_samples_h5
```

It uses device batch size 8, global batch size 64, BF16, 32-batch validation,
W&B tracking, versioned W&B checkpoint artifacts, and authenticated XRootD
checkpoint publishing.

Create a CMS VOMS proxy with enough lifetime for queueing and the four-day job,
then export the W&B key:

```bash
mkdir -p "$HOME/.globus"
voms-proxy-init -rfc -bits 4096 \
  --voms cms:/cms/country/de \
  --valid 192:00 \
  --out "$HOME/.globus/x509_proxy"
export X509_USER_PROXY="$HOME/.globus/x509_proxy"
export WANDB_API_KEY="..."
```

Run the mandatory data/GPU probe and wait for its marker:

```bash
./scripts/train_full_l40s/submit.sh probe
./scripts/train_full_l40s/submit.sh status
```

Then submit:

```bash
./scripts/train_full_l40s/submit.sh submit
```

Static validation:

```bash
./scripts/train_full_l40s/submit.sh dry-run
```

Each checkpoint is stored in the W&B project
`DELight_Reconstruction_Pairwise_Full` as versions of:

```text
pairwise_channel_masking_l40s_<Cluster>-checkpoint
```

Every artifact version contains:

```text
reconstruction_model_<STEP>.pt
reconstruction_resume_<STEP>.pt
run_config.json
```

Aliases include `latest` and `step-<STEP>`.

The same checkpoint versions are also published under:

```text
root://ceph-node-j.etp.kit.edu//dwong/training_runs/pairwise_l40s_<Cluster>
```

`reconstruction_model_<STEP>.pt` is the model state dict for inference.
`run_config.json` records the exact model variant and architecture settings,
and `latest.json` points to the newest persistent checkpoint.
