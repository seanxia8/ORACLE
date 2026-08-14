#!/bin/sh
set -eu

echo "=== Condor Method Probe ==="
echo "Probe method: ${PROBE_METHOD:-unknown}"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "PWD: $(pwd)"
echo "User: $(id)"
echo "Condor IWD: ${_CONDOR_JOB_IWD:-unset}"
echo "WANDB_API_KEY: $(if [ -n "${WANDB_API_KEY:-}" ]; then echo set; else echo unset; fi)"
echo

echo "=== GPU Visibility ==="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
    nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader || true
else
    echo "nvidia-smi not found"
fi
echo

echo "=== Filesystem Visibility ==="
for path in \
    /ceph \
    /ceph/srv \
    /ceph/srv/ssjostrom \
    /ceph/srv/ssjostrom/training_small_complete \
    /srv \
    /srv/ceph \
    /srv/ssjostrom
do
    ls -ld "$path" 2>&1 || true
done
echo

echo "=== Python Before CVMFS ==="
for py in python python3 python3.9 python3.10
do
    if command -v "$py" >/dev/null 2>&1; then
        printf "%s: " "$py"
        "$py" --version || true
    fi
done
echo

echo "=== CVMFS LCG CUDA View ==="
LCG_SETUP=/cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
if [ -f "$LCG_SETUP" ]; then
    echo "Found $LCG_SETUP"
    if (
        set +u
        # shellcheck disable=SC1090
        . "$LCG_SETUP"
        echo "PATH after LCG setup: $PATH"
        python --version || true
        python - <<'PY' || true
import importlib.util
print("torch importable:", importlib.util.find_spec("torch") is not None)
print("wandb importable:", importlib.util.find_spec("wandb") is not None)
print("h5py importable:", importlib.util.find_spec("h5py") is not None)
PY
    ); then
        echo "LCG setup probe completed"
    else
        echo "LCG setup probe failed; continuing to XRootD checks"
    fi
else
    echo "LCG setup not found: $LCG_SETUP"
fi
echo

echo "=== XRootD Visibility ==="
if command -v xrdfs >/dev/null 2>&1; then
    echo "xrdfs: $(command -v xrdfs)"
    echo "Listing /ssjostrom/training_small_complete/ER:"
    if xrdfs root://ceph-node-j.etp.kit.edu ls /ssjostrom/training_small_complete/ER | head -n 5; then
        echo "xrdfs list succeeded"
    else
        echo "xrdfs list failed"
    fi
else
    echo "xrdfs not found"
fi

if command -v xrdcp >/dev/null 2>&1; then
    echo "xrdcp: $(command -v xrdcp)"
    rm -f xrdcp_probe_meta.h5
    if xrdcp --nopbar --force root://ceph-node-j.etp.kit.edu://ssjostrom/data_temp/train/ER/meta_energy_100.h5 xrdcp_probe_meta.h5; then
        echo "xrdcp small metadata copy succeeded"
        ls -lh xrdcp_probe_meta.h5
    else
        echo "xrdcp small metadata copy failed"
    fi
else
    echo "xrdcp not found"
fi
echo

echo "PROBE_DONE"
