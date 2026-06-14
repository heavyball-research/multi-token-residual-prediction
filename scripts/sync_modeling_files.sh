#!/bin/bash
# Sync canonical model python files into every checkpoint dir that hosts a copy.
#
# Two families are managed:
#   1. Dense SDAR-MRP: source = src/models/SDAR-1.7B-MRP/,
#      sentinel file = modeling_sdar_mrp.py
#   2. MoE SDAR-MRP : source = src/models/SDAR-30B-A3B-MRP/,
#      sentinel file = modeling_sdar_mrp_moe.py
#
# Usage:
#   ./scripts/sync_modeling_files.sh                 # sync both families
#   ./scripts/sync_modeling_files.sh --check         # dry run; non-zero exit if drift
#   ./scripts/sync_modeling_files.sh --verbose       # print every copied file
#   ./scripts/sync_modeling_files.sh --only dense    # only the dense family
#   ./scripts/sync_modeling_files.sh --only moe      # only the MoE family

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODE="sync"
VERBOSE=0
ONLY=""
for arg in "$@"; do
    case "$arg" in
        --check) MODE="check" ;;
        --verbose|-v) VERBOSE=1 ;;
        --only) shift; ONLY="${1:-}" ;;
        --only=*) ONLY="${arg#--only=}" ;;
        dense|moe) ONLY="$arg" ;;
        -h|--help)
            sed -n '2,15p' "$0"
            exit 0
            ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

# ----------------------------------------------------------------------------
# list_sync_files <src_dir>
#   Print (one per line) the .py and .json file names directly under <src_dir>,
#   EXCLUDING per-checkpoint files that must not be cross-copied across model
#   sizes: config.json (architecture dims) and generation_config.json.
list_sync_files() {
    local src_dir="$1"
    (cd "$src_dir" && ls *.py *.json 2>/dev/null \
        | grep -vFx -e config.json -e generation_config.json \
        | sort -u)
}

# ----------------------------------------------------------------------------
# find_target_dirs <sentinel_file> <src_dir>
#   Print (one per line) every dir under REPO_ROOT containing <sentinel_file>,
#   excluding <src_dir> itself and .git/node_modules.
find_target_dirs() {
    local sentinel="$1"
    local src_dir="$2"
    find -L "$REPO_ROOT" \
        -type d \( -name .git -o -name node_modules \) -prune -o \
        -type f -name "$sentinel" -print 2>/dev/null \
        | xargs -n1 dirname \
        | sort -u \
        | grep -vFx "$src_dir" || true
}

# ----------------------------------------------------------------------------
# sync_one_dir <src_dir> <tgt_dir>  (uses globals: MODE, VERBOSE)
#   Sync every .py and .json from <src_dir> into <tgt_dir>. Echoes status
#   lines and increments the globals `copied` and `drift` as appropriate.
sync_one_dir() {
    local src_dir="$1"
    local tgt_dir="$2"
    local src_files=()
    while IFS= read -r f; do src_files+=("$f"); done < <(list_sync_files "$src_dir")
    local f src_path tgt_path
    for f in "${src_files[@]}"; do
        src_path="$src_dir/$f"
        tgt_path="$tgt_dir/$f"
        if [[ ! -f "$tgt_path" ]]; then
            if [[ "$MODE" == "check" ]]; then
                echo "MISSING: $tgt_path"
                drift=1
            else
                cp "$src_path" "$tgt_path"
                copied=$((copied + 1))
                if [[ $VERBOSE -eq 1 ]]; then
                    echo "  + $tgt_path (new)"
                fi
            fi
            continue
        fi
        if ! cmp -s "$src_path" "$tgt_path"; then
            if [[ "$MODE" == "check" ]]; then
                echo "DRIFT  : $tgt_path"
                drift=1
            else
                cp "$src_path" "$tgt_path"
                copied=$((copied + 1))
                if [[ $VERBOSE -eq 1 ]]; then
                    echo "  ~ $tgt_path"
                fi
            fi
        else
            if [[ $VERBOSE -eq 1 ]]; then
                echo "  = $tgt_path"
            fi
        fi
    done
}

# ----------------------------------------------------------------------------
# sync_family <label> <src_dir> <sentinel>
#   High-level driver: discover targets for <sentinel> and sync each one
#   from <src_dir>.
sync_family() {
    local label="$1"
    local src_dir="$2"
    local sentinel="$3"
    echo "=========================================================="
    echo "family   : $label"
    echo "source   : $src_dir"
    echo "sentinel : $sentinel"
    if [[ ! -d "$src_dir" ]]; then
        echo "  (source dir missing — skipping family)"
        return 0
    fi
    local src_files=()
    while IFS= read -r f; do src_files+=("$f"); done < <(list_sync_files "$src_dir")
    if [[ ${#src_files[@]} -eq 0 ]]; then
        echo "  (no .py/.json files in source — skipping family)"
        return 0
    fi
    local targets=()
    while IFS= read -r d; do targets+=("$d"); done < <(find_target_dirs "$sentinel" "$src_dir")
    echo "files    : ${src_files[*]}"
    if [[ ${#targets[@]} -eq 0 ]]; then
        echo "  (no target dirs found — nothing to sync)"
        return 0
    fi
    echo "targets  :"
    for d in "${targets[@]}"; do echo "  - $d"; done
    echo
    local tgt
    for tgt in "${targets[@]}"; do
        sync_one_dir "$src_dir" "$tgt"
    done
}

# ----------------------------------------------------------------------------
# Family registry. Add new (label, src_dir, sentinel_file) tuples here when a
# new prototype lands under src/models/.
DENSE_SRC="${REPO_ROOT}/src/models/SDAR-1.7B-MRP"
DENSE_SENTINEL="modeling_sdar_mrp.py"

MOE_SRC="${REPO_ROOT}/src/models/SDAR-30B-A3B-MRP"
MOE_SENTINEL="modeling_sdar_mrp_moe.py"

echo "mode : $MODE"
echo

drift=0
copied=0

if [[ -z "$ONLY" || "$ONLY" == "dense" ]]; then
    sync_family "dense SDAR-MRP" "$DENSE_SRC" "$DENSE_SENTINEL"
fi
if [[ -z "$ONLY" || "$ONLY" == "moe" ]]; then
    sync_family "MoE SDAR-MRP"   "$MOE_SRC"   "$MOE_SENTINEL"
fi

echo
if [[ "$MODE" == "check" ]]; then
    if [[ $drift -ne 0 ]]; then
        echo "drift detected. run without --check to sync."
        exit 1
    fi
    echo "all target dirs in sync with their prototypes"
else
    echo "synced $copied file(s)"
fi
