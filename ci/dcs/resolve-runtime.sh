#!/bin/bash
# Locate a published store entry for one runtime key. A pure lookup:
# canonical path on stdout and exit 0, or empty output and exit 1 -
# absence is a status, never a sentinel string. Readers never write to
# the store, so this script performs no effect beyond the probe.
#
# Probes this job's own scratch base first (ASTRA_STORE_ROOT, exported by
# runner-job.sbatch), then every network scratch base: a job that queued
# for days may start after a newer expires-* directory appeared and must
# still find the entry the seed published on the older one.
#
# Usage: resolve-runtime.sh <key>
set -euo pipefail

key="${1:?usage: resolve-runtime.sh <key>}"

candidates=("${ASTRA_STORE_ROOT:-}")
for base in "${DCS_SCRATCH_BASE:-/scratch/scratch-space}"/expires-*; do
    candidates+=("$base/asteria/store")
done

for root in "${candidates[@]}"; do
    [[ -n "$root" ]] || continue
    entry="$root/$key"
    if [[ -d "$entry" ]]; then
        # Canonical physical path: /scratch is a symlink farm, and the
        # loader records what it resolves - consumers must see one
        # spelling, not two.
        (cd "$entry" && pwd -P)
        exit 0
    fi
done
exit 1
