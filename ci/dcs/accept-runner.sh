#!/bin/bash
# Forced SSH command: the ONLY operation the CI deploy key can perform.
# Reads a just-in-time runner config on stdin, stores it in a private file,
# and submits exactly one runner job to SLURM. Whatever command the client
# asked for is ignored, so a leaked key cannot get a shell.
set -euo pipefail

ROOT="${DCS_CI_ROOT:-$HOME/asteria-ci}"
umask 077
cd "$ROOT/jobs"

jit_file=$(mktemp "$ROOT/jobs/jitconfig.XXXXXX")
# A JIT config blob is a few KB; cap stdin so a hostile client cannot fill
# the filesystem through this channel.
head -c 65536 > "$jit_file"
if [[ ! -s "$jit_file" ]]; then
    rm -f "$jit_file"
    echo "empty jitconfig on stdin" >&2
    exit 1
fi

# The client's command word carries a display name for the SLURM job so
# squeue shows which experiment a runner serves. It is never executed;
# sanitize it to a safe token before using it as a name.
word=$(printf '%s' "${SSH_ORIGINAL_COMMAND:-runner}" | tr -cd 'A-Za-z0-9._-')
name=${word:0:64}

# A word ending in -c<N> asks for N cores and 4 GB per core in place of the
# sbatch script's 2 cores and 16 GB; the job then splits them into job
# steps itself. The bound caps what a leaked key can hold. Read before the
# name's cut, so a long name cannot clip the suffix.
resources=()
if [[ "$word" =~ -c([0-9]+)$ ]]; then
    cores=${BASH_REMATCH[1]}
    if (( ${#cores} > 3 || 10#$cores < 3 || 10#$cores > 128 )); then
        rm -f "$jit_file"
        echo "cores ${cores} outside 3 to 128" >&2
        exit 1
    fi
    cores=$((10#$cores))
    resources=(--cpus-per-task="$cores" --mem="$((cores * 4))G")
fi

# Submit the repository's copy of the sbatch script, freshened by a quiet
# fast-forward pull, so a pushed fix is live on the next provision without
# re-running setup.sh. A push already gates what this script does; offline
# or diverged, the pull is skipped and the last checkout still works. Only
# this accept script itself still deploys through setup.sh.
REPO="${DCS_CI_REPO:-$HOME/asteria}"
git -C "$REPO" pull --ff-only --quiet 2>/dev/null || true
sbatch --parsable --job-name="${name:-runner}" ${resources[@]+"${resources[@]}"} \
    "$REPO/ci/dcs/runner-job.sbatch" "$jit_file"
