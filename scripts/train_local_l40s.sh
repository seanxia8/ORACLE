#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PROFILE="${1:-compact}"
GPU_INDEX="${GPU_INDEX:-0}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python environment not found at $PYTHON" >&2
    echo "Create it with: uv sync --no-dev" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export RECONSTRUCTION_DISABLE_TORCH_COMPILE="${RECONSTRUCTION_DISABLE_TORCH_COMPILE:-1}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export RECONSTRUCTION_MODEL_VARIANT="${RECONSTRUCTION_MODEL_VARIANT:-pairwise_channel_masking}"
export RECONSTRUCTION_RECOIL_CLASSIFICATION="${RECONSTRUCTION_RECOIL_CLASSIFICATION:-1}"
export RECONSTRUCTION_SPATIAL_TARGET_INDICES="${RECONSTRUCTION_SPATIAL_TARGET_INDICES:-0,1}"
export RECONSTRUCTION_SCALAR_LOSS_WEIGHTS="${RECONSTRUCTION_SCALAR_LOSS_WEIGHTS:-1.0,1.0,1.0}"
export RECONSTRUCTION_DATA_FORMAT=h5_batch
export RECONSTRUCTION_ENERGIES="${RECONSTRUCTION_ENERGIES:-10,20,50,100,200,500}"
export RECONSTRUCTION_AMP_DTYPE="${RECONSTRUCTION_AMP_DTYPE:-auto}"
export RECONSTRUCTION_SEED="${RECONSTRUCTION_SEED:-42}"

if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE="${WANDB_MODE:-online}"
    export RECONSTRUCTION_WANDB_RUN="${RECONSTRUCTION_WANDB_RUN:-1}"
    export RECONSTRUCTION_WANDB_CHECKPOINT_ARTIFACTS="${RECONSTRUCTION_WANDB_CHECKPOINT_ARTIFACTS:-1}"
else
    export WANDB_MODE=disabled
    export RECONSTRUCTION_WANDB_RUN=0
    export RECONSTRUCTION_WANDB_CHECKPOINT_ARTIFACTS=0
fi

case "$PROFILE" in
    check)
        ;;
    compact)
        export RECONSTRUCTION_LOCAL_DATA_PATH="${RECONSTRUCTION_LOCAL_DATA_PATH:-$REPO_ROOT/.condor_data/compact_l40s}"
        export RECONSTRUCTION_LOCAL_CACHE_PATH="${RECONSTRUCTION_LOCAL_CACHE_PATH:-$REPO_ROOT/cache/local_compact_l40s}"
        export RECONSTRUCTION_MAX_H5_FILES_PER_ENERGY_RECOIL="${RECONSTRUCTION_MAX_H5_FILES_PER_ENERGY_RECOIL:-1}"
        export RECONSTRUCTION_EXPECTED_H5_EVENTS_PER_FILE="${RECONSTRUCTION_EXPECTED_H5_EVENTS_PER_FILE:-10}"
        export RECONSTRUCTION_NUM_EPOCHS="${RECONSTRUCTION_NUM_EPOCHS:-20}"
        export RECONSTRUCTION_TOTAL_BATCH_SIZE="${RECONSTRUCTION_TOTAL_BATCH_SIZE:-16}"
        export RECONSTRUCTION_DEVICE_BATCH_SIZE="${RECONSTRUCTION_DEVICE_BATCH_SIZE:-4}"
        export RECONSTRUCTION_NUM_WORKERS="${RECONSTRUCTION_NUM_WORKERS:-4}"
        export RECONSTRUCTION_MAX_OPEN_H5_FILES="${RECONSTRUCTION_MAX_OPEN_H5_FILES:-12}"
        export RECONSTRUCTION_EVAL_STEP_PERIOD="${RECONSTRUCTION_EVAL_STEP_PERIOD:-10}"
        export RECONSTRUCTION_EVAL_NUM_BATCHES="${RECONSTRUCTION_EVAL_NUM_BATCHES:-6}"
        export RECONSTRUCTION_SAVE_CHECKPOINT_PERIOD="${RECONSTRUCTION_SAVE_CHECKPOINT_PERIOD:-60}"
        export RECONSTRUCTION_WANDB_PROJECT="${RECONSTRUCTION_WANDB_PROJECT:-DELight_Reconstruction_Pairwise_Local}"
        export RECONSTRUCTION_WANDB_RUN_NAME="${RECONSTRUCTION_WANDB_RUN_NAME:-pairwise_compact_local_l40s_${RUN_STAMP}}"
        export RECONSTRUCTION_CHECKPOINT_DIR="${RECONSTRUCTION_CHECKPOINT_DIR:-$REPO_ROOT/artifacts/pairwise_compact_local_l40s/checkpoints/$RUN_STAMP}"
        ;;
    full-pilot|full)
        export RECONSTRUCTION_LOCAL_DATA_PATH="${RECONSTRUCTION_LOCAL_DATA_PATH:-/ceph/srv/dwong/training_samples_h5}"
        export RECONSTRUCTION_LOCAL_CACHE_PATH="${RECONSTRUCTION_LOCAL_CACHE_PATH:-$REPO_ROOT/cache/local_full_l40s}"
        export RECONSTRUCTION_MAX_H5_FILES_PER_ENERGY_RECOIL="${RECONSTRUCTION_MAX_H5_FILES_PER_ENERGY_RECOIL:-250}"
        export RECONSTRUCTION_EXPECTED_H5_EVENTS_PER_FILE="${RECONSTRUCTION_EXPECTED_H5_EVENTS_PER_FILE:-100}"
        if [ "$PROFILE" = "full-pilot" ]; then
            export RECONSTRUCTION_NUM_EPOCHS="${RECONSTRUCTION_NUM_EPOCHS:-1}"
            export RECONSTRUCTION_TOTAL_BATCH_SIZE="${RECONSTRUCTION_TOTAL_BATCH_SIZE:-32}"
            export RECONSTRUCTION_DEVICE_BATCH_SIZE="${RECONSTRUCTION_DEVICE_BATCH_SIZE:-4}"
            export RECONSTRUCTION_SAVE_CHECKPOINT_PERIOD="${RECONSTRUCTION_SAVE_CHECKPOINT_PERIOD:-500}"
            run_prefix="pairwise_full_pilot_local_l40s"
        else
            export RECONSTRUCTION_NUM_EPOCHS="${RECONSTRUCTION_NUM_EPOCHS:-20}"
            # Keep the exact device/global batch configuration proven by the
            # full-dataset pilot. This avoids an untested memory jump.
            export RECONSTRUCTION_TOTAL_BATCH_SIZE="${RECONSTRUCTION_TOTAL_BATCH_SIZE:-32}"
            export RECONSTRUCTION_DEVICE_BATCH_SIZE="${RECONSTRUCTION_DEVICE_BATCH_SIZE:-4}"
            export RECONSTRUCTION_SAVE_CHECKPOINT_PERIOD="${RECONSTRUCTION_SAVE_CHECKPOINT_PERIOD:-5000}"
            run_prefix="pairwise_full_local_l40s"
        fi
        export RECONSTRUCTION_NUM_WORKERS="${RECONSTRUCTION_NUM_WORKERS:-8}"
        export RECONSTRUCTION_MAX_OPEN_H5_FILES="${RECONSTRUCTION_MAX_OPEN_H5_FILES:-32}"
        export RECONSTRUCTION_EVAL_STEP_PERIOD="${RECONSTRUCTION_EVAL_STEP_PERIOD:-250}"
        export RECONSTRUCTION_EVAL_NUM_BATCHES="${RECONSTRUCTION_EVAL_NUM_BATCHES:-32}"
        export RECONSTRUCTION_WANDB_PROJECT="${RECONSTRUCTION_WANDB_PROJECT:-DELight_Reconstruction_Pairwise_Full}"
        export RECONSTRUCTION_WANDB_RUN_NAME="${RECONSTRUCTION_WANDB_RUN_NAME:-${run_prefix}_${RUN_STAMP}}"
        export RECONSTRUCTION_CHECKPOINT_DIR="${RECONSTRUCTION_CHECKPOINT_DIR:-$REPO_ROOT/artifacts/${run_prefix}/checkpoints/$RUN_STAMP}"
        ;;
    *)
        echo "Usage: $0 [check|compact|full-pilot|full]" >&2
        exit 2
        ;;
esac

echo "Checking local L40S GPU index $GPU_INDEX..."
"$PYTHON" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is unavailable. Run this script on the machine that physically owns "
        "the L40S and has a working NVIDIA driver."
    )
if torch.cuda.device_count() != 1:
    raise RuntimeError(
        "Expected exactly one GPU after CUDA_VISIBLE_DEVICES filtering, "
        f"found {torch.cuda.device_count()}."
    )

name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
properties = torch.cuda.get_device_properties(0)
if "L40S" not in name.upper():
    raise RuntimeError(f"Selected GPU is {name!r}, not an NVIDIA L40S.")
if capability < (8, 9):
    raise RuntimeError(f"Unexpected L40S compute capability: {capability}")

# Force allocation, arithmetic, and synchronization. This catches common
# driver/ECC failures before dataset indexing or checkpoint creation begins.
x = torch.randn((2048, 2048), device="cuda", dtype=torch.float16)
y = x @ x
checksum = y.float().mean().item()
torch.cuda.synchronize()
del x, y
torch.cuda.empty_cache()

print(
    "LOCAL_L40S_OK "
    f"name={name!r} "
    f"capability={capability[0]}.{capability[1]} "
    f"memory_mib={properties.total_memory // (1024 * 1024)} "
    f"checksum={checksum:.6f}"
)
PY

if [ "$PROFILE" = "check" ]; then
    exit 0
fi

if [ ! -d "$RECONSTRUCTION_LOCAL_DATA_PATH/ER" ] \
    || [ ! -d "$RECONSTRUCTION_LOCAL_DATA_PATH/NR" ]; then
    echo "ERROR: Dataset is missing ER/NR directories:" >&2
    echo "  $RECONSTRUCTION_LOCAL_DATA_PATH" >&2
    if [ "$PROFILE" = "compact" ]; then
        echo "Prepare it with: ./scripts/train_compact_l40s/submit.sh prepare" >&2
    fi
    exit 1
fi

mkdir -p "$RECONSTRUCTION_CHECKPOINT_DIR" "$RECONSTRUCTION_LOCAL_CACHE_PATH"

echo "Starting local L40S training"
echo "  Profile: $PROFILE"
echo "  Dataset: $RECONSTRUCTION_LOCAL_DATA_PATH"
echo "  Model: $RECONSTRUCTION_MODEL_VARIANT"
echo "  Epochs: $RECONSTRUCTION_NUM_EPOCHS"
echo "  Device/global batch: $RECONSTRUCTION_DEVICE_BATCH_SIZE/$RECONSTRUCTION_TOTAL_BATCH_SIZE"
echo "  Checkpoints: $RECONSTRUCTION_CHECKPOINT_DIR"
echo "  W&B mode: $WANDB_MODE"

"$PYTHON" -m reconstruction_model.train \
    2>&1 | tee "$RECONSTRUCTION_CHECKPOINT_DIR/training.log"

echo "Training completed. Saved files:"
find "$RECONSTRUCTION_CHECKPOINT_DIR" -maxdepth 1 -type f -printf '%f\n' | sort
