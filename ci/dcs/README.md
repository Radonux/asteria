# DCS cluster runners for heavy evaluations

One just-in-time (JIT) GitHub Actions runner per experiment, scheduled by
SLURM, alive only while the experiment runs. No root, no docker, no PAT on
the cluster, nothing from the system but glibc.

## How it works

Every record in the evaluation matrix runs on the cluster, whatever its
`kind`; the reusable workflow runs three jobs per record:

1. `provision` (GitHub-hosted, ~1 min) validates the record's `kind`, then
   mints a single-job JIT runner config with a one-hour GitHub App
   installation token and pipes it over SSH to the cluster. The SSH key is
   bound to a forced command (`slice.sh`, below) that applies to SLURM for
   a slice and runs `runner-job.sbatch` on it. The runner is labeled with
   the run id, so concurrent experiments can never swap runners.
2. `Experiment on the cluster` waits for that runner, downloads the run's
   prebuilt runtime bundle, and runs `ci/dcs/evaluate.sh <kind>` as one job
   (self-hosted jobs may run 5 days). The fleet never compiles: the
   `cluster-build` job in the main workflow builds ns-3 once per run with
   the rootless conda toolchain - one bundle per instruction-set level
   (`x86-64-v3` fleet floor, `x86-64-v4` for the AVX-512 nodes) - and each
   node selects its level from `/proc/cpuinfo`. Bundles travel as run
   artifacts and are archived permanently on the run's release.
3. `courier`, on a second freshly minted runner, collects the results the
   arm left in the scratch outbox and uploads them.

Job scratch lives in the cluster's sanctioned network scratch: the newest
`/scratch/scratch-space/expires-<date>` directory, whose expiry always
exceeds the job walltime, never in quota'd NFS home, the small `/var/tmp`
partition, or RAM-billed tmpfs `/tmp` (each of which has caused a distinct
fleet failure; the sbatch script probes them only as fallbacks).

The JIT runner deregisters itself and exits when its one job ends
(completed, failed, or cancelled), which ends the SLURM job and returns the
allocation. A 30-minute idle guard in `runner-job.sbatch` covers the
cancelled-while-queued case; `#SBATCH --time` is the absolute backstop.
Write-scoped jobs (`archive`, ledger) stay on GitHub-hosted runners: the
cluster only ever holds a `contents: read` token.

## Slices

`slice.sh` is installed once and never edited on the cluster. The SSH
command word names a slice as `key=value` pairs (plus at most one bare
word, the job name), and stdin carries a payload of at most 64 KiB:

| Key | Meaning |
|-----|---------|
| `name` | SLURM job name, sanitized, cut to 64 characters |
| `branch` | Branch of `caddyreverse/asteria` the script is taken from (default `main`) |
| `script` | Path of the job script in that branch (default `ci/dcs/runner-job.sbatch`) |
| `cpus`, `mem`, `time` | `--cpus-per-task` (1 to 256), `--mem` (`<n>[M\|G]`, at most 2048G), `--time` |
| `nodes`, `ntasks` | `--nodes` (1 to 8), `--ntasks` (1 to 1024) |
| `constraint`, `partition`, `qos`, `gres` | Passed on after a charset check |
| `test=1` | `sbatch --test-only`: validate and estimate a start, submit nothing |

It resolves the branch's head with `git ls-remote`, takes the script at
exactly that commit from GitHub, and submits it with the payload's path as
its first argument and `DCS_SLICE_REPO`, `DCS_SLICE_BRANCH` and
`DCS_SLICE_SHA` in its environment, so a script can fetch the rest of the
repository at the same commit. An option left out comes from the
script's own `#SBATCH` lines, so the defaults change with a push too.
Anything else is refused before SLURM sees it.

What a slice runs is therefore whatever is on any branch of the pinned
repository: push access to it, together with the deploy key, is the
trust boundary. The key alone can only run code from those branches, and
only within the bounds above. (A runner is no narrower: whoever holds
the key can hand it a JIT config for any repository.)

Jobs are self-contained: each downloads its pinned runner onto job-local
scratch, and the home directory stays control-plane only (forced-command
script, payloads, logs - under 1 GB).

The multi-gigabyte runtime extraction is shared, not duplicated: before
the wave provisions, a single seed job (`cluster-seed` in the main
workflow) publishes both ISA bundles as sealed read-only entries under
`<scratch base>/asteria/store/<artifact name>/` (`ci/dcs/seed-store.sh`),
and every evaluation job links against the entry its CPU supports
(`resolve-runtime.sh` + `install-runtime.sh`), cutting a wave's
environment footprint from ~33 private copies to one per ISA level. The
store is lockless by construction: the DAG runs the only writer before
any reader exists, an entry is published by one atomic rename (its
existence is the proof of completeness), entries are immutable
afterwards, and reaping is a pure age sweep (> walltime + slack) in the
next run's seed. Sharing is an optimization layer only - on any miss
(no network scratch, expired base, failed seed) a job downloads its own
bundle and extracts privately, exactly the pre-store behavior. No toolchain ever lands on scratch: `buildenv-packages.txt`
and `runtime-packages.txt` are consumed by the hosted `cluster-build`
job (`ci/dcs/build.sh`), whose bundle ships the declared runtime env the
binary resolves against (`ci/dcs/install-runtime.sh` verifies this on
the node before the sims start). Only `slice.sh` itself deploys through
`setup.sh`.

## Cluster setup (once)

```sh
git clone --no-recurse-submodules <repo-url> ~/asteria && cd ~/asteria
bash ci/dcs/setup.sh          # control-plane scripts into ~/asteria-ci
```

Generate a dedicated key pair (`ssh-keygen -t ed25519 -f asteria-ci -N ''`)
and add the public half to `~/.ssh/authorized_keys` exactly as `setup.sh`
prints, with the forced command and `restrict`.

## Sharing an account with another repository's runners

These scripts can run on a cluster account that already serves another
repository with the same layout (astra-sim's `~/astra-ci`, `~/astra-sim`
and `<scratch base>/astra-sim/`). Every path here is this repository's
own, so neither side reads, reaps or sweeps the other's:

| What | This repository | Defined in |
|------|-----------------|------------|
| Control plane (forced command, payloads, job logs) | `~/asteria-ci` | `slice.sh`, `setup.sh` (`DCS_CI_ROOT` overrides) |
| Repository a slice's script comes from | `caddyreverse/asteria` on GitHub | `slice.sh` (`DCS_SLICE_REPO` overrides) |
| Job scratch, one tree per SLURM job | `<scratch base>/asteria/<slurm job id>` | `runner-job.sbatch` |
| Shared runtime store | `<network scratch>/asteria/store` | `runner-job.sbatch`, `resolve-runtime.sh` |
| Courier outbox | `<network scratch>/asteria/outbox` | `runner-job.sbatch`, `outbox.sh` |
| SLURM job name | `asteria-gh-runner`, or the provision's runner name | `runner-job.sbatch`, `slice.sh` |

The orphan reaper only visits numeric job directories under
`<scratch base>/asteria/`, and the store's age sweep only its own store,
so the other repository's trees are never candidates. The deploy key is a
second `authorized_keys` line with its own forced command; the scratch
base selection (newest `expires-*` generation, then the job-local
fallbacks) is unchanged.

## GitHub configuration (once)

First create an org-owned GitHub App (no webhook, Repository permissions →
Administration: Read and write) and install it on this repository only.

| Kind   | Name                  | Value |
|--------|-----------------------|-------|
| secret | `DCS_APP_PRIVATE_KEY` | The app's .pem private key, file contents verbatim. |
| secret | `DCS_SSH_KEY`         | Private half of the dedicated key pair. |
| var    | `DCS_APP_ID`          | The app's Client ID (or numeric App ID). |
| var    | `DCS_SSH_DEST`        | e.g. `jfang@comps0.cs.toronto.edu` |
| var    | `DCS_SSH_HOST_KEY`    | One line from `ssh-keyscan -t ed25519 <host>` (pins the host, defeats MITM on the jitconfig hand-off). |

## Routing

Nothing to configure per entry. Every record in
`.github/workflows/evaluation-matrix.json` runs here; its `kind` field
(`comparison`, `single`, or `smoke`) selects which command
`ci/dcs/evaluate.sh` runs on the node. Whichever it is, the report lands at
`<run_directory>/report.md` and the courier uploads the same
`<artifact_name>-<run id>` bundle, so the aggregations and the ledger read
one shape.

## Verifying the lifecycle contract

The provision job logs the SLURM job id and `squeue` shows the runner job.
Cancelling an Actions job mid-run must end its SLURM job within a minute;
that is the lifecycle contract working; a runner whose job never arrives
tears itself down after the 30-minute idle guard.

## Scratch tenancy

A completed experiment leaves nothing on the shared scratch volume. All
jobs live under one `asteria/<slurm job id>` parent per scratch base, so
a full wave presents as a single directory. Three mechanisms compose: the
mid-run segment hasher (`.github/scripts/segment_hasher.py`) folds each
sealed raw-log segment's uncompressed content into the arm's stream
digest and deletes it as the sim runs (nothing raw is ever uploaded); the
attestation carries the hashes, and results reach GitHub through the
scratch outbox (`ci/dcs/outbox.sh`) collected by a freshly tokened
courier job, because runner tokens stop refreshing after 24 hours; the
job's EXIT trap removes its own tree
(and the shared parent, when it is the last job out) whenever the job ends
for any reason that lets bash run; and every new job reaps orphaned
sibling trees whose SLURM job id no longer exists (the backstop for node
crashes and kills that outrun the trap). The non-numeric `store` and
`outbox` siblings are exempt from the reaper by construction; orphaned
outboxes die with their scratch generation. Only the control-plane files under
`~/asteria-ci` (job logs, jitconfig staging) live outside scratch, and
teardown below removes those.

## Teardown

Remove the `authorized_keys` line, delete the two GitHub secrets, and
`rm -rf ~/asteria-ci ~/asteria`. There are no services, cron entries, or standing
registrations to clean up; JIT runners deregister themselves.
