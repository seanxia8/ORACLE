#!/usr/bin/env bash
# Clone and pin Prometheus into external/ (gitignored).
# Upstream is LGPL-2.1: we use it as a library and never patch the clone.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REF="${PROM_REF:-8c199384062012009094862bc244fa55f7694ee0}"   # pinned
mkdir -p "$HERE/external"
if [ ! -d "$HERE/external/prometheus/.git" ]; then
  git clone https://github.com/Harvard-Neutrino/prometheus.git "$HERE/external/prometheus"
fi
cd "$HERE/external/prometheus"
git fetch --all --tags --quiet || true
git checkout --quiet "$REF" 2>/dev/null || echo "WARNING: could not check out $REF (shallow clone?)"
git rev-parse HEAD > "$HERE/external/prometheus_commit.txt"
echo "prometheus at $(cat "$HERE/external/prometheus_commit.txt")"
