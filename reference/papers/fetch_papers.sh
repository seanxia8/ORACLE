#!/usr/bin/env bash
# Download the prior-art and testbed reference PDFs listed in papers.tsv from arXiv.
#
# Run this from a machine with ordinary internet access:
#     bash reference/papers/fetch_papers.sh
#
# Idempotent: existing files are skipped. arXiv asks for a delay between
# requests, so this sleeps 3 s between downloads.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
manifest="$here/papers.tsv"
ua="oracle-reference-fetcher/1.0 (research use; contact repository owner)"

[ -f "$manifest" ] || { echo "missing $manifest" >&2; exit 1; }

ok=0; skip=0; fail=0
while IFS=$'\t' read -r id dir name title; do
  [ -n "${id:-}" ] || continue
  case "$id" in \#*) continue ;; esac
  mkdir -p "$here/$dir"
  out="$here/$dir/${id}_${name}.pdf"
  if [ -s "$out" ]; then
    printf 'skip  %s  (%s)\n' "$id" "$title"; skip=$((skip+1)); continue
  fi
  printf 'fetch %s  %s\n' "$id" "$title"
  if curl -fsSL -A "$ua" --retry 3 --retry-delay 5 --max-time 180 \
        -o "$out.part" "https://arxiv.org/pdf/$id"; then
    if head -c 4 "$out.part" | grep -q '%PDF'; then
      mv "$out.part" "$out"; ok=$((ok+1))
    else
      echo "  !! not a PDF, discarding" >&2; rm -f "$out.part"; fail=$((fail+1))
    fi
  else
    echo "  !! download failed" >&2; rm -f "$out.part"; fail=$((fail+1))
  fi
  sleep 3
done < "$manifest"

printf '\ndownloaded %d, skipped %d, failed %d\n' "$ok" "$skip" "$fail"
[ "$fail" -eq 0 ]
