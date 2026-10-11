#!/bin/bash
# Lockless outbox between a multi-day evaluation arm and its minutes-old
# courier job. No GitHub-bound upload can be trusted from a job older than
# 24 hours (the runner's tokens stop refreshing), so the arm performs zero
# GitHub effects: it renames its results onto network scratch here, and a
# freshly provisioned courier job - sequenced after the arm by the
# workflow's `needs:` DAG, so single writer then single reader, no locks -
# collects and uploads them with a token that is minutes old.
#
#   place   <group> <key> <source-dir>  stage-then-rename the results in;
#                                       raw *.zst segments are excluded (the
#                                       hasher owns them; anything left is
#                                       forensic residue, not results)
#   locate  <group> <key>               print the outbox path, exit 1 if none
#   discard <group> <key>               remove a collected outbox
#
# `place` resolves the newest expires-* scratch generation AT PLACEMENT
# TIME, not at job start: the cluster guarantees only that the newest
# generation expires more than five days out *now*, and an arm places its
# outbox up to 4d20h after its job started - trusting the generation the
# sbatch resolved at launch would leave the courier as little as four
# hours before the admins wipe it. Re-resolving restores a >=5-day
# collection window, which is what makes "re-run the failed courier" a
# real recovery rather than a race. ASTRA_OUTBOX_ROOT (exported by
# runner-job.sbatch) is only the fallback when the glob turns up nothing;
# with no network scratch at all, placing fails loudly, because the
# courier could never see a job-local path anyway. `locate`/`discard`
# glob every generation, so a rollover between the arm and its courier
# cannot hide the outbox. Orphans (arm died before courier, courier
# never ran) are wiped with the scratch generation itself when it
# expires.
set -euo pipefail

usage() {
    echo "usage: outbox.sh place <group> <key> <source-dir>" >&2
    echo "       outbox.sh locate <group> <key>" >&2
    echo "       outbox.sh discard <group> <key>" >&2
    exit 2
}

verb="${1:-}"
group="${2:-}"
key="${3:-}"
[[ -n "$verb" && -n "$group" && -n "$key" ]] || usage
# Both name segments become path components; admit only the ledger's own
# key alphabet so a hostile value cannot traverse.
for component in "$group" "$key"; do
    if [[ ! "$component" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "outbox: invalid name component: $component" >&2
        exit 2
    fi
done

candidates() {
    # Newest scratch generation first: after a rollover both may briefly
    # exist, and the newer one is the one a newer arm wrote into.
    ls -1dt "${DCS_SCRATCH_BASE:-/scratch/scratch-space}"/expires-*/asteria/outbox/"$group/$key" 2>/dev/null || true
}

# The freshest generation's outbox root, ranked by the parsed expiry date
# exactly as runner-job.sbatch ranks scratch bases - month names do not
# sort lexically. Empty output and status 1 when no generation exists.
newest_outbox_root() {
    local dir stamp y m d epoch best="" best_epoch=0
    for dir in "${DCS_SCRATCH_BASE:-/scratch/scratch-space}"/expires-*; do
        [[ -d "$dir" ]] || continue
        stamp="${dir##*expires-}"
        read -r y m d <<< "${stamp//-/ }"
        epoch=$(date -d "$d $m $y" +%s 2>/dev/null) || continue
        if (( epoch > best_epoch )); then
            best="$dir"
            best_epoch=$epoch
        fi
    done
    [[ -n "$best" ]] && printf '%s\n' "$best/asteria/outbox"
}

case "$verb" in
place)
    source_dir="${4:?usage: outbox.sh place <group> <key> <source-dir>}"
    root="$(newest_outbox_root || true)"
    root="${root:-${ASTRA_OUTBOX_ROOT:-}}"
    if [[ -z "$root" ]]; then
        echo "outbox: no network scratch on this node; results cannot reach the courier" >&2
        exit 1
    fi
    mkdir -p "$root/$group"
    temp="$(mktemp -d "$root/$group/.tmp-${key}.XXXXXX")"
    # tar preserves permissions and symlink-free layout across NFS; the
    # exclusion keeps multi-gigabyte raw segments (hasher residue) out of
    # the delivery path by construction.
    tar -C "$source_dir" -cf - --exclude='*.zst.[0-9]*' . | tar -C "$temp" -xf -
    final="$root/$group/$key"
    # A previous attempt's outbox is superseded wholesale: the rename
    # publishes a complete result set or nothing.
    rm -rf "$final"
    mv -T "$temp" "$final"
    echo "outbox: placed $(du -sh "$final" | cut -f1) at $final"
    ;;
locate)
    found="$(candidates | head -n 1)"
    if [[ -z "$found" ]]; then
        echo "outbox: nothing placed for $group/$key" >&2
        exit 1
    fi
    printf '%s\n' "$found"
    ;;
discard)
    while IFS= read -r found; do
        [[ -n "$found" ]] || continue
        rm -rf "$found"
        # Fold empty group directories away; siblings keep theirs alive.
        rmdir --ignore-fail-on-non-empty "$(dirname "$found")" 2>/dev/null || true
        echo "outbox: discarded $found"
    done < <(candidates)
    ;;
*)
    usage
    ;;
esac
