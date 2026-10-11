#!/bin/bash
# One-time DCS setup for the control plane only. Jobs are self-contained:
# each downloads its own pinned runner and conda toolchain onto job-local
# scratch (see runner-job.sbatch and build.sh), so the home directory
# holds nothing but this repository checkout, the forced-command script,
# and per-provision jitconfig files and logs - well under 1 GB, and no
# state shared between jobs.
#
# Run from a repository checkout on the cluster:
#   bash ci/dcs/setup.sh
set -euo pipefail

ROOT="${DCS_CI_ROOT:-$HOME/asteria-ci}"
script_dir=$(cd "$(dirname "$0")" && pwd)

mkdir -p "$ROOT/bin" "$ROOT/jobs"

# The forced SSH command runs outside any checkout, so it is installed to
# a stable path; everything else deploys via the fast-forward pull in
# accept-runner.sh on each provision.
install -m 700 "$script_dir/accept-runner.sh" "$ROOT/bin/"

cat <<EOF

Setup OK under $ROOT.

Next: add the CI deploy key to ~/.ssh/authorized_keys as a forced command,
so the key can submit runner jobs and do nothing else:

  command="$ROOT/bin/accept-runner.sh",restrict ssh-ed25519 AAAA... asteria-ci

Then create the GitHub secrets/vars listed in ci/dcs/README.md.
EOF
