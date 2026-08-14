#!/bin/bash

check_cuda_compatibility() {
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1)
    if [ -z "$COMPUTE_CAP" ]; then
        echo "ERROR: Could not detect cuda compute capability."
        exit 1
    fi

    COMPUTE_CAP_INT=$(echo "$COMPUTE_CAP" | awk '{printf "%d", $1*10}')

    if [ "$COMPUTE_CAP_INT" -lt 80 ]; then
        echo "ERROR: GPU compatibility < 8.0 (Ampere)"
        echo "Node: $(hostname)"
        echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
        exit 1
    fi

    echo "GPU Check passed: Compute capability $COMPUTE_CAP (>= 8.0)"
}
# Run CUDA compatibility check
check_cuda_compatibility
# Set HOME to current job dir
HOME=$_CONDOR_JOB_IWD
# Create a directory for the data
mkdir -p data/ER data/NR 
# mkdir -p ${HOME}/reconstruction_model/training_data
# Upload the data
xrdcp -r root://ceph-node-j.etp.kit.edu://ssjostrom/training_small_complete/ER data/
xrdcp -r root://ceph-node-j.etp.kit.edu://ssjostrom/training_small_complete/NR data/
echo "Files copied: $(find data/ -type f | wc -l)"
du -sh data/
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh

pip install --user wandb
pip install --user triton==3.3.0

export PATH="$HOME/.local/bin:$PATH"
# Install pip if not already installed
# python3.9 -m ensurepip --upgrade || dnf install -y python3.9-pip
# Install uv if not already installed
# command -v uv &> /dev/null || python3.9 -m pip install uv
# Create venv
# [ -d ".venv" ] || uv venv 
# Setup wandb api key, automatically detected when wandb.init() is called
# in the training script. Provide it via the environment instead of hardcoding
# a personal key in the repository.
export WANDB_API_KEY="${WANDB_API_KEY:-}"
# Run the training script
python -m reconstruction.training.train
# Save the model checkpoints




