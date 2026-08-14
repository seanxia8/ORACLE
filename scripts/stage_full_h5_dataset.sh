#!/usr/bin/env bash
set -euo pipefail

SERVER="${FULL_H5_XROOTD_SERVER:-root://ceph-node-j.etp.kit.edu}"
REMOTE_ROOT="${FULL_H5_REMOTE_ROOT:-/dwong/training_samples_h5}"
DESTINATION="${FULL_H5_LOCAL_ROOT:-$PWD/training_data_h5}"
PARALLELISM="${FULL_H5_STAGE_PARALLELISM:-4}"
ENERGIES="${RECONSTRUCTION_ENERGIES:-10,20,50,100,200,500}"
FILES_PER_GROUP="${FULL_H5_FILES_PER_ENERGY_RECOIL:-250}"
EXPECTED_FILES=$((2 * 6 * FILES_PER_GROUP))
MANIFEST_DIR="${FULL_H5_MANIFEST_DIR:-$PWD/full_h5_manifests}"

list_remote_directory() {
    remote_directory="$1"
    output_file="$2"
    attempt=1
    while [ "$attempt" -le 3 ]; do
        temporary_file="${output_file}.attempt-${attempt}"
        if timeout 300 xrdfs "$SERVER" ls -l "$remote_directory" > "$temporary_file"; then
            mv "$temporary_file" "$output_file"
            return 0
        fi
        rm -f "$temporary_file"
        if [ "$attempt" -lt 3 ]; then
            echo "XRootD listing failed on attempt $attempt/3; retrying in 15 seconds." >&2
            sleep 15
        fi
        attempt=$((attempt + 1))
    done
    echo "ERROR: Unable to list $remote_directory after 3 attempts." >&2
    return 1
}

mkdir -p "$DESTINATION/ER" "$DESTINATION/NR" "$MANIFEST_DIR"
combined_manifest="$MANIFEST_DIR/all_shards.txt"
: > "$combined_manifest"

for recoil in ER NR; do
    recoil_manifest="$MANIFEST_DIR/${recoil}.txt"
    raw_listing="$MANIFEST_DIR/${recoil}.listing.txt"
    list_remote_directory "$REMOTE_ROOT/$recoil" "$raw_listing"
    awk '$5 ~ /\.h5$/ {print $4, $5}' "$raw_listing" \
        | grep -E "/${recoil}_traces_energy_($(echo "$ENERGIES" | tr ',' '|'))_batch_[0-9]+\.h5$" \
        | sort -k2 > "$recoil_manifest"
    count="$(wc -l < "$recoil_manifest")"
    if [ "$count" -ne $((6 * FILES_PER_GROUP)) ]; then
        echo "ERROR: $recoil manifest contains $count files, expected $((6 * FILES_PER_GROUP))." >&2
        exit 1
    fi
    cat "$recoil_manifest" >> "$combined_manifest"
done

manifest_count="$(wc -l < "$combined_manifest")"
if [ "$manifest_count" -ne "$EXPECTED_FILES" ]; then
    echo "ERROR: Combined manifest contains $manifest_count files, expected $EXPECTED_FILES." >&2
    exit 1
fi

expected_bytes="$(awk '{sum += $1} END {printf "%.0f", sum}' "$combined_manifest")"
echo "Staging $manifest_count shards ($expected_bytes bytes) with parallelism $PARALLELISM"
if [ "${FULL_H5_STAGE_MANIFEST_ONLY:-0}" = "1" ]; then
    echo "MANIFEST_OK files=$manifest_count bytes=$expected_bytes"
    exit 0
fi

export SERVER DESTINATION
cut -d' ' -f2- "$combined_manifest" \
    | xargs -r -n1 -P "$PARALLELISM" bash -c '
        remote_path="$1"
        filename="$(basename "$remote_path")"
        case "$filename" in
            ER_*) recoil=ER ;;
            NR_*) recoil=NR ;;
            *) echo "Unexpected shard: $remote_path" >&2; exit 1 ;;
        esac
        destination="$DESTINATION/$recoil/$filename"
        xrdcp --nopbar --retry 5 --retry-policy continue --continue \
            "$SERVER/$remote_path" "$destination"
    ' _

actual_count="$(find "$DESTINATION" -type f -name '*.h5' | wc -l)"
actual_bytes="$(find "$DESTINATION" -type f -name '*.h5' -printf '%s\n' \
    | awk '{sum += $1} END {printf "%.0f", sum}')"
if [ "$actual_count" -ne "$EXPECTED_FILES" ] || [ "$actual_bytes" -ne "$expected_bytes" ]; then
    echo "ERROR: Staged dataset verification failed." >&2
    echo "Expected: $EXPECTED_FILES files, $expected_bytes bytes" >&2
    echo "Actual:   $actual_count files, $actual_bytes bytes" >&2
    exit 1
fi

python scripts/validate_full_h5_dataset.py "$DESTINATION"
echo "STAGING_OK files=$actual_count bytes=$actual_bytes"
