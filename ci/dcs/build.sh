#!/bin/bash
# Build the ns-3 backend for the DCS cluster, once per workflow run, on a
# GitHub-hosted runner. The fleet no longer compiles: every SLURM job
# downloads the bundle this script produces, so a 16-job wave pays for one
# build instead of sixteen cold toolchains (each of which used to download
# ~1 GB of conda packages and extract ~100k files onto its scratch).
#
# The toolchain is the same rootless conda stack the cluster builds used:
# compiler, protoc, boost, MPI, zlib, and zstd all come from a fresh env
# under this job's own temp dir, nothing from the system but glibc - so
# the binary's platform floor (conda's sysroot) is unchanged from the
# days the cluster compiled for itself.
#
# One build per instruction-set level in MARCH_LEVELS: the compile node is
# no longer the run node, so -march=native is gone. x86-64-v3 is the
# fleet floor (AMD_Epyc7453 is Zen 3); x86-64-v4 covers the AVX-512 nodes
# (EPYC_9634, AMD_Epyc9754, XeonGold-6348). Each bundle carries the ns-3
# build tree plus the declared runtime env (ci/dcs/runtime-packages.txt),
# and is published only after its library closure resolves against the
# bundle alone - the build env is deleted before that check, so a
# dependency that silently leaked from the toolchain fails here, minutes
# in, not on the cluster, days in.
set -euo pipefail

MAMBA_VERSION=2.3.2
# Levels may be passed as arguments - the CI matrix builds one level per
# runner, in parallel - and no arguments builds every fleet level, so the
# local invocation stays total. Validate before the 10-minute env solve,
# not at the first -march error after it.
if (( $# )); then
    MARCH_LEVELS=("$@")
else
    MARCH_LEVELS=(x86-64-v3 x86-64-v4)
fi
for march in "${MARCH_LEVELS[@]}"; do
    if [[ ! "$march" =~ ^x86-64-v[0-9]$ ]]; then
        echo "::error::unknown instruction-set level '${march}'" >&2
        exit 1
    fi
done
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
ns3_dir="$repo_root/extern/network_backend/ns-3"
work="${RUNNER_TEMP:?}"
BUILDENV="$work/buildenv"
RUNTIME="$work/runtimeenv"
stage="$work/stage"

echo "toolchain: fetching micromamba ${MAMBA_VERSION}"
curl -fsSL "https://micro.mamba.pm/api/micromamba/linux-64/${MAMBA_VERSION}" \
    | tar -xj -C "$work" bin/micromamba
export MAMBA_ROOT_PREFIX="$work/mamba-root"

create_env() {
    local prefix=$1 spec=$2
    grep -vE '^[[:space:]]*(#|$)' "$spec" \
        | xargs "$work/bin/micromamba" create -y -p "$prefix" -c conda-forge
}
# Two envs from two declared lists: the toolchain env dies with this job,
# the runtime env ships inside every bundle. Both are solved in the same
# session against the same channel state, so the shipped libraries are
# the ones the build links against; the closure check below is the
# enforcer if that ever drifts (pin the offender in both lists to fix).
echo "toolchain: creating the build and runtime envs"
create_env "$BUILDENV" "$repo_root/ci/dcs/buildenv-packages.txt"
create_env "$RUNTIME" "$repo_root/ci/dcs/runtime-packages.txt"
# The package cache (downloaded tarballs plus extracted copies) is dead
# weight once the envs are linked.
"$work/bin/micromamba" clean --all --yes

export PATH="$BUILDENV/bin:$PATH"
export CC="$BUILDENV/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$BUILDENV/bin/x86_64-conda-linux-gnu-g++"
export CMAKE_PREFIX_PATH="$BUILDENV"
export LD_LIBRARY_PATH="$BUILDENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Memoize compilation when the caller mounted a cache directory.
# compiler_check=content and empty sloppiness keep every
# correctness-relevant input in the hash, which is what makes restoring an
# entry published by a different conda solve safe: a stale entry is a miss,
# never a wrong object file.
#
# The config file is written whole rather than with ccache --set-config:
# setting the empty sloppiness value rewrites each blank line after it
# twice, and the restored cache carries ccache.conf from build to build, so
# the file would double per build, and every ccache call parses all of it.
if [[ -n "${CCACHE_DIR:-}" ]]; then
    mkdir -p "$CCACHE_DIR"
    cat > "$CCACHE_DIR/ccache.conf" <<EOF
cache_dir = $CCACHE_DIR
compiler_check = content
sloppiness =
compression = true
max_size = ${CCACHE_MAX_SIZE:-4G}
EOF
    ccache --zero-stats
    # Five lines, whatever ccache.conf the restored cache carried.
    conf_lines="$(wc -l < "$CCACHE_DIR/ccache.conf")"
    if (( conf_lines != 5 )); then
        echo "::error::ccache.conf has ${conf_lines} lines, not the 5 written" >&2
        exit 1
    fi
    export CMAKE_C_COMPILER_LAUNCHER=ccache
    export CMAKE_CXX_COMPILER_LAUNCHER=ccache
fi

for march in "${MARCH_LEVELS[@]}"; do
    echo "build: ns-3 backend for ${march}"
    NS3_MARCH="$march" bash "$repo_root/build/astra_ns3/build.sh" -c
    mkdir -p "$stage/$march/extern/network_backend/ns-3"
    mv "$ns3_dir/build" "$stage/$march/extern/network_backend/ns-3/build"
    # The object tree is rebuilt for the next level anyway (the flags
    # changed), and two trees at once would crowd a hosted runner's disk.
    rm -rf "$ns3_dir/cmake-cache"
done

[[ -z "${CCACHE_DIR:-}" ]] || ccache --show-stats

# Delete the toolchain before checking closures: the binaries bake
# absolute RUNPATHs into $BUILDENV, and only its absence proves that a
# bundle resolves every library from its own runtime env.
rm -rf "$BUILDENV" "$MAMBA_ROOT_PREFIX" "$work/bin/micromamba"

for march in "${MARCH_LEVELS[@]}"; do
    build_tree="$stage/$march/extern/network_backend/ns-3/build"
    mapfile -t binaries < <(
        find "$build_tree/scratch" -maxdepth 1 -type f -executable \
             -name '*AstraSimNetwork*' | sort)
    if (( ${#binaries[@]} == 0 )); then
        echo "::error::${march} build produced no AstraSimNetwork binary"
        exit 1
    fi
    unresolved="$(env -i PATH=/usr/bin:/bin \
        LD_LIBRARY_PATH="$build_tree/lib:$RUNTIME/lib" \
        ldd "${binaries[0]}" | grep 'not found' || true)"
    if [[ -n "$unresolved" ]]; then
        echo "::error::${march} bundle does not resolve from the shipped runtime env:"
        echo "$unresolved"
        exit 1
    fi
    # tar preserves the executable bit and the env's symlinks; the
    # artifact uploader's own zip does not. Two -C roots place the build
    # tree at its checkout-relative path and the runtime env beside it.
    tar -czf "$work/cluster-runtime-${march}.tar.gz" \
        -C "$stage/$march" extern \
        -C "$work" runtimeenv
    ls -l "$work/cluster-runtime-${march}.tar.gz"
done

# Remnants: only the two bundle tarballs survive for the upload steps.
rm -rf "$stage" "$RUNTIME"
