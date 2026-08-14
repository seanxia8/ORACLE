#!/usr/bin/env bash

ensure_proxy() {
    X509_USER_PROXY="${X509_USER_PROXY:-$HOME/.globus/x509_proxy}"
    export X509_USER_PROXY
    if [ ! -f "$X509_USER_PROXY" ]; then
        echo "ERROR: Missing CMS proxy: $X509_USER_PROXY" >&2
        echo "Create it with:" >&2
        echo "  voms-proxy-init -rfc -bits 4096 --voms cms:/cms/country/de --valid 192:00 --out '$HOME/.globus/x509_proxy'" >&2
        exit 1
    fi
    if ! voms-proxy-info -file "$X509_USER_PROXY" -exists -valid 170:00; then
        echo "ERROR: CMS proxy must remain valid for at least 170 hours." >&2
        exit 1
    fi
    if ! voms-proxy-info -file "$X509_USER_PROXY" -vo | grep -qi cms; then
        echo "ERROR: Proxy does not contain a CMS VOMS attribute." >&2
        exit 1
    fi
}

ensure_wandb() {
    if [ -z "${WANDB_API_KEY:-}" ]; then
        echo "ERROR: WANDB_API_KEY is not exported." >&2
        exit 1
    fi
}

ensure_dataset_visible() {
    for recoil in ER NR; do
        count="$(
            xrdfs root://ceph-node-j.etp.kit.edu ls "/dwong/training_samples_h5/$recoil" \
                | grep -Ec "${recoil}_traces_energy_(10|20|50|100|200|500)_batch_[0-9]+\\.h5$"
        )"
        if [ "$count" -ne 1500 ]; then
            echo "ERROR: XRootD reports $count $recoil shards, expected 1500." >&2
            exit 1
        fi
    done
}

require_recent_probe() {
    marker="$1"
    site="$2"
    if [ ! -f "$marker" ] || ! find "$marker" -mmin -1440 -print -quit | grep -q .; then
        echo "ERROR: A successful $site probe from the last 24 hours is required." >&2
        echo "Run the probe action and wait for it to complete before submitting." >&2
        exit 1
    fi
    grep -q "site=$site" "$marker" || {
        echo "ERROR: Probe marker does not match site $site: $marker" >&2
        exit 1
    }
    grep -q "status=success" "$marker" || {
        echo "ERROR: Probe did not succeed: $marker" >&2
        cat "$marker" >&2
        exit 1
    }
}

submit_dry_run() {
    jdl="$1"
    output="$2"
    if [ -n "${X509_USER_PROXY:-}" ] && [ -f "$X509_USER_PROXY" ]; then
        condor_submit -dry-run "$output" "$jdl"
    else
        condor_submit -append "use_x509userproxy = False" -dry-run "$output" "$jdl"
    fi
}
