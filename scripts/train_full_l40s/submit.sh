#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
JDL="$SCRIPT_DIR/submit.jdl"
PROBE_JDL="$SCRIPT_DIR/probe.jdl"
PROBE_MARKER="$SCRIPT_DIR/full_training_probe_nemo2.ok"
ACTION="${1:-submit}"

cd "$REPO_ROOT"
# shellcheck source=../remote_submit_common.sh
source scripts/remote_submit_common.sh

case "$ACTION" in
    probe)
        ensure_proxy
        ensure_dataset_visible
        rm -f "$PROBE_MARKER"
        submit_dry_run "$PROBE_JDL" /tmp/train_full_l40s_probe.ad
        condor_submit "$PROBE_JDL"
        ;;
    submit)
        ensure_proxy
        ensure_wandb
        ensure_dataset_visible
        require_recent_probe "$PROBE_MARKER" nemo2
        submit_dry_run "$JDL" /tmp/train_full_l40s.ad
        condor_submit "$JDL"
        ;;
    dry-run)
        submit_dry_run "$JDL" /tmp/train_full_l40s.ad
        submit_dry_run "$PROBE_JDL" /tmp/train_full_l40s_probe.ad
        ;;
    status)
        condor_q "$USER" || true
        [ ! -f "$PROBE_MARKER" ] || cat "$PROBE_MARKER"
        ;;
    *)
        echo "Usage: $0 [probe|submit|dry-run|status]" >&2
        exit 2
        ;;
esac
