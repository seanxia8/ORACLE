# Full Pairwise Training: TopAS A100/V100S

This run trains `pairwise_channel_masking` for 20 epochs over all 3,000 H5
shards covering ER/NR energies `10,20,50,100,200,500`.

TopAS reads the dataset directly from:

```text
/ceph/srv/dwong/training_samples_h5
```

The job accepts A100 40 GB or V100S 32 GB and ranks A100 higher. AMP is selected
at runtime: BF16 on A100 and FP16 with gradient scaling on V100S. Both use
device batch size 8, global batch size 64, 32-batch validation, W&B tracking,
and XRootD checkpoint publishing.

Create and export a CMS proxy. The submit helper requires at least 170 hours
remaining:

```bash
voms-proxy-init -rfc -bits 4096 \
  --voms cms:/cms/country/de \
  --valid 192:00 \
  --out "$HOME/.globus/x509_proxy"
export X509_USER_PROXY="$HOME/.globus/x509_proxy"
export WANDB_API_KEY="..."
```

Run the mandatory direct-Ceph/GPU probe and wait for its marker:

```bash
./scripts/train_full_a100/submit.sh probe
./scripts/train_full_a100/submit.sh status
```

Then submit:

```bash
./scripts/train_full_a100/submit.sh submit
```

Static validation without a proxy:

```bash
./scripts/train_full_a100/submit.sh dry-run
```

Checkpoints are published under:

```text
root://ceph-node-j.etp.kit.edu//dwong/training_runs/pairwise_topas_<Cluster>
```
