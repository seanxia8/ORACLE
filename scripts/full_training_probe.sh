#!/usr/bin/env bash
set -euo pipefail

SITE_MODE="${SITE_MODE:?Set SITE_MODE=topas or nemo2}"
PROBE_MARKER="full_training_probe_${SITE_MODE}.ok"

write_probe_result() {
    exit_code="$?"
    trap - EXIT
    if [ "$exit_code" -eq 0 ]; then
        status="success"
    else
        status="failed"
    fi
    printf 'site=%s status=%s exit_code=%s host=%s date=%s\n' \
        "$SITE_MODE" "$status" "$exit_code" "$(hostname)" "$(date -Is)" \
        > "$PROBE_MARKER"
    exit "$exit_code"
}
trap write_probe_result EXIT

retry_command() {
    attempts="$1"
    shift
    attempt=1
    while ! "$@"; do
        if [ "$attempt" -ge "$attempts" ]; then
            echo "ERROR: Command failed after $attempt attempts: $*" >&2
            return 1
        fi
        echo "Command failed on attempt $attempt/$attempts; retrying in 10 seconds: $*" >&2
        attempt=$((attempt + 1))
        sleep 10
    done
}

echo "Probe site mode: $SITE_MODE"
echo "Host: $(hostname)"

if [ -z "${X509_USER_PROXY:-}" ] || [ ! -r "$X509_USER_PROXY" ]; then
    echo "ERROR: A readable X509_USER_PROXY is required for remote XRootD access." >&2
    exit 1
fi
echo "X509 proxy: $X509_USER_PROXY"
if command -v voms-proxy-info >/dev/null 2>&1; then
    voms-proxy-info -file "$X509_USER_PROXY" -timeleft
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version \
        --format=csv,noheader
else
    echo "nvidia-smi is unavailable in the container; reporting GPU through PyTorch."
fi

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot see a CUDA GPU in the probe job.")

device = torch.device("cuda", 0)
properties = torch.cuda.get_device_properties(device)
capability = torch.cuda.get_device_capability(device)
if capability < (7, 0):
    raise RuntimeError(
        f"Unsupported GPU compute capability {capability}; require Volta or newer."
    )

print(
    "PYTORCH_GPU_OK "
    f"name={properties.name!r} "
    f"capability={capability[0]}.{capability[1]} "
    f"memory_mib={properties.total_memory // (1024 * 1024)}"
)
PY

if [ "$SITE_MODE" = "topas" ]; then
    DATA_ROOT="${FULL_H5_DIRECT_ROOT:-/ceph/srv/dwong/training_samples_h5}"
    test -d "$DATA_ROOT/ER"
    test -d "$DATA_ROOT/NR"
    python scripts/validate_full_h5_dataset.py "$DATA_ROOT" --sample-only
elif [ "$SITE_MODE" = "nemo2" ]; then
    retry_command 3 timeout 120 \
        xrdfs root://ceph-node-j.etp.kit.edu stat \
        /dwong/training_samples_h5/ER/ER_traces_energy_100_batch_0000.h5
    rm -f probe_shard.h5
    retry_command 3 timeout 300 \
        xrdcp --nopbar --force --retry 3 --retry-policy continue \
        root://ceph-node-j.etp.kit.edu//dwong/training_samples_h5/ER/ER_traces_energy_100_batch_0000.h5 \
        probe_shard.h5
    mkdir -p probe_dataset/ER probe_dataset/NR
    mv probe_shard.h5 probe_dataset/ER/ER_traces_energy_100_batch_0000.h5
    python - <<'PY'
import h5py
path = "probe_dataset/ER/ER_traces_energy_100_batch_0000.h5"
with h5py.File(path, "r") as handle:
    assert handle["traces"].shape == (100, 56, 65536)
    assert len(handle["events"]) == 100
print("NEMO2_XROOTD_H5_OK")
PY
else
    echo "ERROR: Unknown SITE_MODE=$SITE_MODE" >&2
    exit 2
fi

echo "PROBE_OK"
