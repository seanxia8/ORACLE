#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ACTION="${1:-submit}"
RUN_DIR="probe_runs/$(date +%Y%m%d_%H%M%S)"
LATEST_FILE="probe_runs/latest_jobs.txt"

archive_logs() {
    mkdir -p "$RUN_DIR"
    local moved=0
    for file in \
        probe_l40s_singularity.out probe_l40s_singularity.err probe_l40s_singularity.log \
        probe_a100_docker.out probe_a100_docker.err probe_a100_docker.log \
        out.txt err.txt log.txt
    do
        if [ -e "$file" ]; then
            mv "$file" "$RUN_DIR/$file.previous"
            moved=1
        fi
    done
    if [ "$moved" -eq 1 ]; then
        echo "Archived previous logs under $RUN_DIR"
    fi
}

submit_one() {
    local name="$1"
    local jdl="$2"
    local output
    echo
    echo "Submitting $name: $jdl"
    output="$(condor_submit "$jdl")"
    echo "$output"
    local cluster
    cluster="$(printf '%s\n' "$output" | sed -n 's/.*submitted to cluster \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
    if [ -n "$cluster" ]; then
        printf "%s %s\n" "$name" "$cluster" >> "$LATEST_FILE"
    else
        printf "%s unknown\n" "$name" >> "$LATEST_FILE"
    fi
}

show_status() {
    echo "=== Queue ==="
    condor_q "$USER" || true
    echo

    if [ -f "$LATEST_FILE" ]; then
        echo "=== Latest Submitted Test Jobs ==="
        cat "$LATEST_FILE"
        echo
        while read -r name cluster; do
            [ -n "${cluster:-}" ] || continue
            [ "$cluster" = "unknown" ] && continue
            echo "=== History/queue for $name ($cluster) ==="
            condor_q "$cluster" -af ClusterId ProcId JobStatus HoldReason 2>/dev/null || \
                condor_history "$cluster" -limit 1 -af ClusterId ProcId JobStatus ExitCode ExitStatus CompletionDate 2>/dev/null || true
            echo
        done < "$LATEST_FILE"
    else
        echo "No $LATEST_FILE yet. Run: $0 submit"
    fi

    echo "=== Recent Output Tails ==="
    for file in \
        probe_l40s_singularity.out probe_l40s_singularity.err probe_l40s_singularity.log \
        probe_a100_docker.out probe_a100_docker.err probe_a100_docker.log \
        out.txt err.txt log.txt
    do
        if [ -f "$file" ]; then
            echo
            echo "--- $file ---"
            tail -n 60 "$file"
        fi
    done
}

case "$ACTION" in
    submit)
        mkdir -p probe_runs
        archive_logs
        : > "$LATEST_FILE"

        echo "Submitting transfer/environment tests from $REPO_ROOT"
        echo
        echo "Methods:"
        echo "  l40s_singularity_probe : L40S + vanilla/Singularity, tests /ceph, CVMFS, xrdfs, xrdcp small metadata copy"
        echo "  a100_docker_probe      : submit_ref-style Docker universe on topas A100, tests the same probes"
        echo "  h5_condor_transfer     : proven L40S + Condor input-transfer of two H5 files, runs 20-step H5 training smoke"
        echo

        if [ -z "${WANDB_API_KEY:-}" ]; then
            echo "WARNING: WANDB_API_KEY is not exported. The H5 training smoke needs it and may fail."
            echo "         The two environment/data-transfer probes do not require W&B."
            echo
        fi

        submit_one l40s_singularity_probe scripts/submissions/submit_probe_l40s_singularity.jdl
        submit_one a100_docker_probe scripts/submissions/submit_probe_a100_docker.jdl
        submit_one h5_condor_transfer scripts/submissions/submit_h5_subset.jdl

        echo
        echo "Saved submitted clusters to $LATEST_FILE"
        echo "Watch with:"
        echo "  $0 status"
        echo "  tail -f probe_l40s_singularity.out probe_l40s_singularity.err probe_l40s_singularity.log probe_a100_docker.out probe_a100_docker.err probe_a100_docker.log out.txt err.txt log.txt"
        ;;
    status)
        show_status
        ;;
    *)
        echo "Usage: $0 [submit|status]" >&2
        exit 2
        ;;
esac
