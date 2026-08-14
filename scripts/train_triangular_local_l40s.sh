#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PROFILE="${1:-full-pilot}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
export GPU_INDEX="${GPU_INDEX:-1}"

case "$PROFILE" in
    check)
        exec "$REPO_ROOT/scripts/train_local_l40s.sh" check
        ;;
    full-pilot|full)
        ;;
    *)
        echo "Usage: $0 [check|full-pilot|full]" >&2
        exit 2
        ;;
esac

run_prefix="triangular_pairwise_${PROFILE}_local_l40s"
run_prefix="${run_prefix//-/_}"

export RUN_STAMP
export RECONSTRUCTION_MODEL_VARIANT="${RECONSTRUCTION_MODEL_VARIANT:-triangular_pairwise}"
export RECONSTRUCTION_RECOIL_CLASSIFICATION="${RECONSTRUCTION_RECOIL_CLASSIFICATION:-1}"
export RECONSTRUCTION_SPATIAL_TARGET_INDICES="${RECONSTRUCTION_SPATIAL_TARGET_INDICES:-0,1}"
export RECONSTRUCTION_SCALAR_LOSS_WEIGHTS="${RECONSTRUCTION_SCALAR_LOSS_WEIGHTS:-1.0,1.0,1.0}"

export RECONSTRUCTION_LOCAL_DATA_PATH="${RECONSTRUCTION_LOCAL_DATA_PATH:-/ceph/srv/dwong/training_samples_h5}"
export RECONSTRUCTION_LOCAL_CACHE_PATH="${RECONSTRUCTION_LOCAL_CACHE_PATH:-$REPO_ROOT/cache/${run_prefix}}"
export RECONSTRUCTION_CHECKPOINT_DIR="${RECONSTRUCTION_CHECKPOINT_DIR:-$REPO_ROOT/artifacts/${run_prefix}/checkpoints/$RUN_STAMP}"

# Triangular pair updates keep a C x C pair representation, so start below the
# pairwise-channel-masking local defaults. Increase these after a successful
# pilot if memory headroom is comfortable.
export RECONSTRUCTION_TOTAL_BATCH_SIZE="${RECONSTRUCTION_TOTAL_BATCH_SIZE:-16}"
export RECONSTRUCTION_DEVICE_BATCH_SIZE="${RECONSTRUCTION_DEVICE_BATCH_SIZE:-2}"
export RECONSTRUCTION_NUM_WORKERS="${RECONSTRUCTION_NUM_WORKERS:-8}"
export RECONSTRUCTION_MAX_OPEN_H5_FILES="${RECONSTRUCTION_MAX_OPEN_H5_FILES:-32}"
export RECONSTRUCTION_EVAL_STEP_PERIOD="${RECONSTRUCTION_EVAL_STEP_PERIOD:-250}"
export RECONSTRUCTION_EVAL_NUM_BATCHES="${RECONSTRUCTION_EVAL_NUM_BATCHES:-32}"
export RECONSTRUCTION_WANDB_PROJECT="${RECONSTRUCTION_WANDB_PROJECT:-DELight_Reconstruction_Triangular_Full}"
export RECONSTRUCTION_WANDB_RUN_NAME="${RECONSTRUCTION_WANDB_RUN_NAME:-${run_prefix}_${RUN_STAMP}}"

if [ "$PROFILE" = "full-pilot" ]; then
    export RECONSTRUCTION_NUM_EPOCHS="${RECONSTRUCTION_NUM_EPOCHS:-1}"
    export RECONSTRUCTION_SAVE_CHECKPOINT_PERIOD="${RECONSTRUCTION_SAVE_CHECKPOINT_PERIOD:-500}"
else
    export RECONSTRUCTION_NUM_EPOCHS="${RECONSTRUCTION_NUM_EPOCHS:-20}"
    export RECONSTRUCTION_SAVE_CHECKPOINT_PERIOD="${RECONSTRUCTION_SAVE_CHECKPOINT_PERIOD:-5000}"
fi

exec "$REPO_ROOT/scripts/train_local_l40s.sh" "$PROFILE"
