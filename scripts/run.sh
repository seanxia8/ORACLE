#!/bin/sh
set -eu

check_cuda_compatibility() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi not found; will verify CUDA from Python after installing the environment."
        return 0
    fi

    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1)
    if [ -z "$COMPUTE_CAP" ]; then
        echo "nvidia-smi could not detect cuda compute capability; will verify CUDA from Python after installing the environment."
        return 0
    fi

    COMPUTE_CAP_INT=$(echo "$COMPUTE_CAP" | awk '{printf "%d", $1*10}')

    if [ "$COMPUTE_CAP_INT" -lt 70 ]; then
        echo "ERROR: GPU compatibility < 7.0 (Volta)"
        echo "Node: $(hostname)"
        echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
        exit 1
    fi

    echo "GPU Check passed: Compute capability $COMPUTE_CAP (>= 7.0)"
}
# Run CUDA compatibility check
check_cuda_compatibility

# Set HOME to current job dir
HOME=$_CONDOR_JOB_IWD
export HOME
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$HOME/.uv-cache"

echo "Job host: $(hostname)"
echo "Job pwd: $(pwd)"
echo "Job user: $(id)"
JOB_MODE="${JOB_MODE:-smoke}"
SMOKE_NUM_STEPS="${SMOKE_NUM_STEPS:-5}"
SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE:-2}"
H5_SOURCE_ROOT="${H5_SOURCE_ROOT:-root://ceph-node-j.etp.kit.edu://ssjostrom/training_small_complete}"
H5_ENERGIES="${H5_ENERGIES:-100}"
H5_FILES_PER_ENERGY_RECOIL="${H5_FILES_PER_ENERGY_RECOIL:-1}"
WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_MODE
printf '{"status":"starting","job_mode":"%s","host":"%s"}\n' \
    "$JOB_MODE" "$(hostname)" > training_job_summary.json
write_job_summary() {
    status="$?"
    trap - 0
    printf '{"status":"%s","exit_code":%s,"job_mode":"%s","host":"%s","finished_at":"%s"}\n' \
        "$(if [ "$status" -eq 0 ]; then echo success; else echo failed; fi)" \
        "$status" "$JOB_MODE" "$(hostname)" "$(date -Is)" \
        > training_job_summary.json
    exit "$status"
}
trap write_job_summary 0
echo "Job mode: $JOB_MODE"
if [ -n "${WANDB_API_KEY:-}" ]; then
    echo "WANDB_API_KEY is set for this job."
else
    echo "WANDB_API_KEY is NOT set for this job."
fi
echo "Visible Ceph paths:"
ls -ld /ceph /ceph/srv /ceph/srv/ssjostrom /ceph/srv/ssjostrom/data_temp /ceph/srv/ssjostrom/data_temp/train 2>&1 || true
echo "Visible /srv paths:"
ls -ld /srv /srv/ceph /srv/ceph/srv /srv/ceph/srv/ssjostrom /srv/ceph/srv/ssjostrom/data_temp /srv/ceph/srv/ssjostrom/data_temp/train 2>&1 || true

if [ "${RECONSTRUCTION_USE_PREBUILT_ENV:-0}" = "1" ]; then
    echo "Using prebuilt Python environment: $(command -v python)"
else
    # Install pip if not already installed
    python3.9 -m ensurepip --upgrade || dnf install -y python3.9-pip
    # Install uv if not already installed
    command -v uv >/dev/null 2>&1 || python3.9 -m pip install uv
    # Create venv. Keep this explicit because the repository .python-version is 3.13,
    # while torch.compile in torch 2.5.1 requires Python < 3.13.
    [ -d ".venv" ] || uv venv --python python3.9
    # Sync runtime dependencies only; notebooks/dev tools are not needed in Condor.
    uv sync --no-dev
    # Activate venv
    . .venv/bin/activate
fi
python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot see a CUDA GPU inside the job environment.")

capability = torch.cuda.get_device_capability(0)
if capability < (7, 0):
    raise RuntimeError(f"GPU compatibility {capability} < (7, 0)")

print(f"PyTorch CUDA check passed: {torch.cuda.get_device_name(0)} capability {capability}")
PY
# W&B is enabled in TrainingConfig. Pass WANDB_API_KEY through Condor instead
# of storing a personal token in this script.
if [ "${RECONSTRUCTION_WANDB_RUN:-1}" != "0" ] && [ -z "${WANDB_API_KEY:-}" ]; then
    echo "ERROR: WANDB_API_KEY is not set. Export it before condor_submit or disable wandb_run."
    exit 1
fi
mkdir -p artifacts reconstruction_model/training_checkpoints

prepare_resume_checkpoint() {
    case "${RECONSTRUCTION_RESUME_CHECKPOINT:-}" in
        root://*)
            mkdir -p resume_checkpoint
            local_resume="resume_checkpoint/$(basename "$RECONSTRUCTION_RESUME_CHECKPOINT")"
            xrdcp --nopbar --force --retry 3 --retry-policy continue \
                "$RECONSTRUCTION_RESUME_CHECKPOINT" "$local_resume"
            RECONSTRUCTION_RESUME_CHECKPOINT="$PWD/$local_resume"
            export RECONSTRUCTION_RESUME_CHECKPOINT
            ;;
    esac
}

run_full_h5_training() {
    data_root="$1"
    export RECONSTRUCTION_DATA_FORMAT=h5_batch
    export RECONSTRUCTION_LOCAL_DATA_PATH="$data_root"
    export RECONSTRUCTION_LOCAL_CACHE_PATH="${RECONSTRUCTION_LOCAL_CACHE_PATH:-$PWD/cache}"
    export RECONSTRUCTION_ENERGIES="${RECONSTRUCTION_ENERGIES:-10,20,50,100,200,500}"
    export RECONSTRUCTION_EXPECTED_H5_EVENTS_PER_FILE="${RECONSTRUCTION_EXPECTED_H5_EVENTS_PER_FILE:-100}"
    export RECONSTRUCTION_MAX_OPEN_H5_FILES="${RECONSTRUCTION_MAX_OPEN_H5_FILES:-32}"
    export RECONSTRUCTION_NUM_EPOCHS="${RECONSTRUCTION_NUM_EPOCHS:-20}"
    prepare_resume_checkpoint
    python -m reconstruction_model.train
}

if [ "$JOB_MODE" = "smoke" ]; then
    echo "Running Condor smoke training with synthetic data."
    python scripts/smoke_test_training.py \
        --num-steps "$SMOKE_NUM_STEPS" \
        --batch-size "$SMOKE_BATCH_SIZE" \
        --wandb-mode "$WANDB_MODE"
elif [ "$JOB_MODE" = "full_probe" ]; then
    scripts/full_training_probe.sh
elif [ "$JOB_MODE" = "full_h5_stage" ]; then
    echo "Staging the complete H5 dataset from XRootD."
    stage_start="$(date +%s)"
    export FULL_H5_LOCAL_ROOT="${FULL_H5_LOCAL_ROOT:-$PWD/training_data_h5}"
    scripts/stage_full_h5_dataset.sh
    stage_end="$(date +%s)"
    export RECONSTRUCTION_DATA_STAGING_SECONDS="$((stage_end - stage_start))"
    run_full_h5_training "$FULL_H5_LOCAL_ROOT"
elif [ "$JOB_MODE" = "full_h5_direct" ]; then
    DATA_ROOT="${FULL_H5_DIRECT_ROOT:-/ceph/srv/dwong/training_samples_h5}"
    echo "Using direct full H5 dataset: $DATA_ROOT"
    python scripts/validate_full_h5_dataset.py "$DATA_ROOT" --sample-only
    export RECONSTRUCTION_DATA_STAGING_SECONDS=0
    run_full_h5_training "$DATA_ROOT"
elif [ "$JOB_MODE" = "full" ]; then
    # Use the shared Ceph dataset directly. The dataloader expects
    # training_data/train/{ER,NR}/..., so the container must expose /ceph.
    DATA_ROOT=""
    for candidate in \
        /ceph/srv/ssjostrom/data_temp \
        /srv/ssjostrom/data_temp \
        /srv/ceph/srv/ssjostrom/data_temp \
        /ceph/ssjostrom/data_temp \
        /srv/ceph/ssjostrom/data_temp
    do
        if [ -d "$candidate/train/ER" ] && [ -d "$candidate/train/NR" ]; then
            DATA_ROOT="$candidate"
            break
        fi
    done

    if [ -z "$DATA_ROOT" ]; then
        echo "ERROR: Full training needs Sebastian data, but no candidate data root is visible."
        echo "Checked:"
        echo "  /ceph/srv/ssjostrom/data_temp/train/{ER,NR}"
        echo "  /srv/ssjostrom/data_temp/train/{ER,NR}"
        echo "  /srv/ceph/srv/ssjostrom/data_temp/train/{ER,NR}"
        echo "  /ceph/ssjostrom/data_temp/train/{ER,NR}"
        echo "  /srv/ceph/ssjostrom/data_temp/train/{ER,NR}"
        echo "Nearby /ceph entries:"
        find /ceph -maxdepth 3 -type d \( -name ssjostrom -o -name data_temp -o -name train \) 2>&1 | head -50 || true
        echo "Nearby /srv entries:"
        find /srv -maxdepth 5 -type d \( -name ceph -o -name ssjostrom -o -name data_temp -o -name train \) 2>&1 | head -80 || true
        exit 1
    fi
    rm -rf training_data
    ln -s "$DATA_ROOT" training_data
    echo "Using training data via symlink: training_data -> $DATA_ROOT"
    echo "Files available: $(find training_data/train -type f | wc -l)"
    du -sh training_data/train

    python -m reconstruction_model.train
elif [ "$JOB_MODE" = "h5_subset" ]; then
    echo "Staging H5 subset from $H5_SOURCE_ROOT"
    echo "H5 energies: $H5_ENERGIES"
    echo "H5 files per energy/recoil: $H5_FILES_PER_ENERGY_RECOIL"
    case "$H5_SOURCE_ROOT" in
        root://ceph-node-j.etp.kit.edu://*)
            H5_REMOTE_DIR="/${H5_SOURCE_ROOT#root://ceph-node-j.etp.kit.edu://}"
            ;;
        root://ceph-node-j.etp.kit.edu/*)
            H5_REMOTE_DIR="${H5_SOURCE_ROOT#root://ceph-node-j.etp.kit.edu}"
            ;;
        *)
            echo "ERROR: H5_SOURCE_ROOT must be a root://ceph-node-j.etp.kit.edu path."
            exit 1
            ;;
    esac
    echo "H5 remote directory: $H5_REMOTE_DIR"
    rm -rf training_data_h5
    mkdir -p training_data_h5/ER training_data_h5/NR

    for recoil in ER NR
    do
        for energy in $(echo "$H5_ENERGIES" | tr ',' ' ')
        do
            manifest="h5_${recoil}_${energy}.txt"
            xrdfs root://ceph-node-j.etp.kit.edu ls "${H5_REMOTE_DIR}/${recoil}" \
                | grep "${recoil}_traces_energy_${energy}_batch_.*\\.h5$" \
                | sort \
                | head -n "$H5_FILES_PER_ENERGY_RECOIL" > "$manifest"

            if [ ! -s "$manifest" ]; then
                echo "ERROR: No H5 files found for ${recoil} energy ${energy}."
                exit 1
            fi

            copied=0
            while IFS= read -r remote_file
            do
                filename=$(basename "$remote_file")
                source_path="root://ceph-node-j.etp.kit.edu/${remote_file}"
                dest_path="training_data_h5/${recoil}/${filename}"
                echo "Copying $source_path"
                xrdcp --nopbar --retry 3 --retry-policy continue --continue "$source_path" "$dest_path"
                copied=$((copied + 1))
            done < "$manifest"

            if [ "$copied" -ne "$H5_FILES_PER_ENERGY_RECOIL" ]; then
                echo "ERROR: Copied $copied files for ${recoil} energy ${energy}, expected $H5_FILES_PER_ENERGY_RECOIL."
                exit 1
            fi
        done
    done

    echo "H5 files copied: $(find training_data_h5 -type f -name '*.h5' | wc -l)"
    du -sh training_data_h5

    export RECONSTRUCTION_DATA_FORMAT=h5_batch
    export RECONSTRUCTION_LOCAL_DATA_PATH="$PWD/training_data_h5"
    export RECONSTRUCTION_LOCAL_CACHE_PATH="$PWD/cache"
    export RECONSTRUCTION_ENERGIES="$H5_ENERGIES"
    export RECONSTRUCTION_MAX_H5_FILES_PER_ENERGY_RECOIL="$H5_FILES_PER_ENERGY_RECOIL"
    export RECONSTRUCTION_NUM_STEPS="${RECONSTRUCTION_NUM_STEPS:-20}"
    export RECONSTRUCTION_EVAL_STEP_PERIOD="${RECONSTRUCTION_EVAL_STEP_PERIOD:-5}"
    export RECONSTRUCTION_SAVE_CHECKPOINT_PERIOD="${RECONSTRUCTION_SAVE_CHECKPOINT_PERIOD:-10}"
    export RECONSTRUCTION_TOTAL_BATCH_SIZE="${RECONSTRUCTION_TOTAL_BATCH_SIZE:-16}"
    export RECONSTRUCTION_DEVICE_BATCH_SIZE="${RECONSTRUCTION_DEVICE_BATCH_SIZE:-4}"
    export RECONSTRUCTION_NUM_WORKERS="${RECONSTRUCTION_NUM_WORKERS:-0}"
    export RECONSTRUCTION_CHECKPOINT_DIR="${RECONSTRUCTION_CHECKPOINT_DIR:-$PWD/artifacts/h5_subset/checkpoints/$(date +%Y%m%d_%H%M%S)}"
    export RECONSTRUCTION_WANDB_PROJECT="${RECONSTRUCTION_WANDB_PROJECT:-DELight_Reconstruction_H5_Subset}"
    export RECONSTRUCTION_WANDB_RUN_NAME="${RECONSTRUCTION_WANDB_RUN_NAME:-h5_subset_$(date +%Y%m%d_%H%M%S)}"
    mkdir -p "$RECONSTRUCTION_CHECKPOINT_DIR"

    python -m reconstruction_model.train
elif [ "$JOB_MODE" = "h5_transferred" ]; then
    echo "Using H5 files transferred by Condor."
    echo "H5 energies: $H5_ENERGIES"
    echo "H5 files per energy/recoil: $H5_FILES_PER_ENERGY_RECOIL"
    rm -rf training_data_h5
    mkdir -p training_data_h5/ER training_data_h5/NR

    h5_count=0
    for h5_file in *_traces_energy_*_batch_*.h5
    do
        if [ ! -f "$h5_file" ]; then
            continue
        fi
        case "$h5_file" in
            ER_*) recoil=ER ;;
            NR_*) recoil=NR ;;
            *)
                echo "Skipping unrecognized H5 file: $h5_file"
                continue
                ;;
        esac
        ln -s "$PWD/$h5_file" "training_data_h5/${recoil}/${h5_file}"
        h5_count=$((h5_count + 1))
    done

    if [ "$h5_count" -eq 0 ]; then
        echo "ERROR: No transferred H5 batch files found in $PWD."
        ls -la
        exit 1
    fi

    echo "H5 transferred files linked: $h5_count"
    find training_data_h5 -type l -print
    du -sh . training_data_h5 || true

    export RECONSTRUCTION_DATA_FORMAT=h5_batch
    export RECONSTRUCTION_LOCAL_DATA_PATH="$PWD/training_data_h5"
    export RECONSTRUCTION_LOCAL_CACHE_PATH="$PWD/cache"
    export RECONSTRUCTION_ENERGIES="$H5_ENERGIES"
    export RECONSTRUCTION_MAX_H5_FILES_PER_ENERGY_RECOIL="$H5_FILES_PER_ENERGY_RECOIL"
    export RECONSTRUCTION_NUM_STEPS="${RECONSTRUCTION_NUM_STEPS:-20}"
    export RECONSTRUCTION_EVAL_STEP_PERIOD="${RECONSTRUCTION_EVAL_STEP_PERIOD:-5}"
    export RECONSTRUCTION_SAVE_CHECKPOINT_PERIOD="${RECONSTRUCTION_SAVE_CHECKPOINT_PERIOD:-10}"
    export RECONSTRUCTION_TOTAL_BATCH_SIZE="${RECONSTRUCTION_TOTAL_BATCH_SIZE:-16}"
    export RECONSTRUCTION_DEVICE_BATCH_SIZE="${RECONSTRUCTION_DEVICE_BATCH_SIZE:-4}"
    export RECONSTRUCTION_NUM_WORKERS="${RECONSTRUCTION_NUM_WORKERS:-0}"
    export RECONSTRUCTION_CHECKPOINT_DIR="${RECONSTRUCTION_CHECKPOINT_DIR:-$PWD/artifacts/h5_subset/checkpoints/$(date +%Y%m%d_%H%M%S)}"
    export RECONSTRUCTION_WANDB_PROJECT="${RECONSTRUCTION_WANDB_PROJECT:-DELight_Reconstruction_H5_Subset}"
    export RECONSTRUCTION_WANDB_RUN_NAME="${RECONSTRUCTION_WANDB_RUN_NAME:-h5_transferred_$(date +%Y%m%d_%H%M%S)}"
    mkdir -p "$RECONSTRUCTION_CHECKPOINT_DIR"

    python -m reconstruction_model.train
else
    echo "ERROR: Unknown JOB_MODE=$JOB_MODE. Use smoke, full_h5_stage, full_h5_direct, h5_subset, h5_transferred, or full."
    exit 1
fi

# Condor transfers artifacts/ back via transfer_output_files. Only production
# runs need the extra XRootD checkpoint copy.
if find reconstruction_model/training_checkpoints artifacts -type f -name 'reconstruction_model_*.pt' 2>/dev/null | grep -q .; then
    find artifacts reconstruction_model/training_checkpoints -type f -name '*.pt' -print
    if [ "$JOB_MODE" = "full" ]; then
        CHECKPOINT_COPY_SOURCE="${RECONSTRUCTION_CHECKPOINT_DIR:-reconstruction_model/training_checkpoints}"
        xrdfs root://ceph-node-j.etp.kit.edu mkdir -p /${USER}/training_checkpoints || true
        if [ -d "$CHECKPOINT_COPY_SOURCE" ]; then
            xrdcp -r "$CHECKPOINT_COPY_SOURCE"/* root://ceph-node-j.etp.kit.edu://${USER}/training_checkpoints/ 2>/dev/null || true
        fi
    fi
else
    echo "No checkpoints found to copy."
fi
