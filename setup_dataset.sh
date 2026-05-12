#!/usr/bin/env bash
#
# setup_dataset.sh — scaffold the nextFrag active-learning directory layout
# for a single dataset, place its initial train/pool/val/test files, and
# record the project root to ${XDG_CONFIG_HOME:-$HOME/.config}/nextfrag/root
# so downstream commands can find it.
#
# Layout produced (relative to --root):
#   <root>/<dataset>/
#     ├── round_0/data/
#     │   ├── train.txt
#     │   └── pool.txt
#     ├── val.txt
#     ├── test.txt
#     └── results.tsv        (header row; appended to by train_al_model)
#
# Round-0 model checkpoints are written lazily to
#   round_0/{arch}_{seed}/model/  by the training scripts.
# Per-round/strategy/arch/seed subdirectories are created lazily by the
# AL loop itself.
#
# Usage:
#   ./setup_dataset.sh \
#       [--root /path/to/project_root] \  (required on first run; defaults to
#                                          the recorded root on subsequent runs)
#       --dataset yeast \
#       --train /path/to/train.txt \
#       --pool  /path/to/pool.txt  \
#       --val   /path/to/val.txt   \
#       --test  /path/to/test.txt  \
#       [--test-ood /path/to/test_ood.txt]  (optional; placed alongside test.txt)
#       [--test-vep /path/to/test_vep.txt]  (optional; placed alongside test.txt)
#       [--n-selected N]             (record N as the default n_selected; runs with
#                                    this count use the plain strategy name in paths)
#       [--mode copy|move|symlink]   (default: copy)
#       [--force]                    (overwrite existing destination files)
#       [--no-write-config]          (skip recording root to config file)
#
# Re-running is safe: directories use `mkdir -p`, and existing destination
# files are left alone unless --force is given. If --root differs from a
# previously recorded root, a warning is printed and the recorded root is
# overwritten.

set -euo pipefail

# ---------- defaults ----------
ROOT=""
DATASET=""
TRAIN=""
POOL=""
VAL=""
TEST=""
TEST_OOD=""
TEST_VEP=""
N_SELECTED=""
MODE="copy"
FORCE=0
WRITE_CONFIG=1

CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/nextfrag"
CONFIG_FILE="$CONFIG_DIR/root"
N_SELECTED_FILE="$CONFIG_DIR/n_selected"

# ---------- helpers ----------
die()  { printf 'Error: %s\n' "$*" >&2; exit 1; }
warn() { printf 'Warning: %s\n' "$*" >&2; }

usage() {
    sed -n '2,45p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

# Place a single source file at a destination according to $MODE.
# Honors $FORCE for overwrite behavior.
place_file() {
    local src="$1" dst="$2"
    [[ -f "$src" ]] || die "source file not found: $src"

    if [[ -e "$dst" || -L "$dst" ]]; then
        if (( FORCE )); then
            rm -f "$dst"
        else
            printf '  skip (exists): %s\n' "$dst"
            return 0
        fi
    fi

    case "$MODE" in
        copy)    cp    -- "$src" "$dst" ;;
        move)    mv    -- "$src" "$dst" ;;
        symlink) ln -s -- "$(realpath "$src")" "$dst" ;;
        *)       die "invalid --mode: $MODE (expected copy|move|symlink)" ;;
    esac
    printf '  %s -> %s\n' "$MODE" "$dst"
}

# ---------- arg parsing ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)            ROOT="$2";        shift 2 ;;
        --dataset)         DATASET="$2";     shift 2 ;;
        --train)           TRAIN="$2";       shift 2 ;;
        --pool)            POOL="$2";        shift 2 ;;
        --val)             VAL="$2";         shift 2 ;;
        --test)            TEST="$2";        shift 2 ;;
        --test-ood)        TEST_OOD="$2";    shift 2 ;;
        --test-vep)        TEST_VEP="$2";    shift 2 ;;
        --n-selected)      N_SELECTED="$2";  shift 2 ;;
        --mode)            MODE="$2";        shift 2 ;;
        --force)           FORCE=1;          shift ;;
        --no-write-config) WRITE_CONFIG=0;   shift ;;
        -h|--help)         usage 0 ;;
        *)                 die "unknown argument: $1 (try --help)" ;;
    esac
done

# ---------- resolve ROOT ----------
if [[ -z "$ROOT" ]]; then
    if [[ -f "$CONFIG_FILE" ]]; then
        ROOT="$(head -n1 "$CONFIG_FILE" | tr -d '[:space:]')"
        [[ -n "$ROOT" ]] || die "recorded project root at $CONFIG_FILE is empty; pass --root explicitly"
        printf 'Using project root from %s: %s\n' "$CONFIG_FILE" "$ROOT"
    else
        die "--root is required (no recorded root at $CONFIG_FILE)"
    fi
fi

# ---------- validation ----------
[[ -n "$DATASET" ]] || die "--dataset is required"
[[ -n "$TRAIN"   ]] || die "--train is required"
[[ -n "$POOL"    ]] || die "--pool is required"
[[ -n "$VAL"     ]] || die "--val is required"
[[ -n "$TEST"    ]] || die "--test is required"

case "$MODE" in
    copy|move|symlink) ;;
    *) die "invalid --mode: $MODE (expected copy|move|symlink)" ;;
esac

if [[ "$DATASET" == */* || "$DATASET" == "." || "$DATASET" == ".." ]]; then
    die "invalid --dataset name: $DATASET"
fi

if [[ -n "$N_SELECTED" ]] && ! [[ "$N_SELECTED" =~ ^[1-9][0-9]*$ ]]; then
    die "--n-selected must be a positive integer"
fi

# ---------- scaffold ----------
mkdir -p -- "$ROOT"
ROOT="$(cd "$ROOT" && pwd)"   # canonicalize after creation

DATASET_DIR="$ROOT/$DATASET"
DATA_DIR="$DATASET_DIR/round_0/data"

printf 'Setting up dataset "%s" under %s\n' "$DATASET" "$ROOT"
mkdir -p -- "$DATA_DIR"

# ---------- place files ----------
printf 'Placing files (mode=%s, force=%s):\n' "$MODE" "$FORCE"
place_file "$TRAIN" "$DATA_DIR/train.txt"
place_file "$POOL"  "$DATA_DIR/pool.txt"
place_file "$VAL"   "$DATASET_DIR/val.txt"
place_file "$TEST"  "$DATASET_DIR/test.txt"
[[ -n "$TEST_OOD" ]] && place_file "$TEST_OOD" "$DATASET_DIR/test_ood.txt"
[[ -n "$TEST_VEP" ]] && place_file "$TEST_VEP" "$DATASET_DIR/test_vep.txt"

# ---------- results.tsv ----------
RESULTS_TSV="$DATASET_DIR/results.tsv"
if [[ ! -f "$RESULTS_TSV" ]] || (( FORCE )); then
    case "$DATASET" in
        human)
            printf 'arch\tacquisition\tseed\tn_selected\tround\tid_r\tvep_r\tood_r\n' \
                > "$RESULTS_TSV"
            ;;
        yeast)
            printf 'arch\tacquisition\tseed\tn_selected\tround\tpearson_score\tall_r\thigh_r\tlow_r\tyeast_r\trandom_r\tchallenging_r\tsnvs_r\tmotif_perturbation_r\tmotif_tiling_r\n' \
                > "$RESULTS_TSV"
            ;;
        *)
            printf 'arch\tacquisition\tseed\tn_selected\tround\n' \
                > "$RESULTS_TSV"
            ;;
    esac
    printf '  created: %s\n' "$RESULTS_TSV"
else
    printf '  skip (exists): %s\n' "$RESULTS_TSV"
fi

# ---------- record config ----------
if (( WRITE_CONFIG )); then
    mkdir -p -- "$CONFIG_DIR"

    if [[ -f "$CONFIG_FILE" ]]; then
        existing="$(head -n1 "$CONFIG_FILE" | tr -d '[:space:]')"
        if [[ -n "$existing" && "$existing" != "$ROOT" ]]; then
            warn "overwriting recorded root: $existing -> $ROOT"
        fi
    fi
    printf '%s\n' "$ROOT" > "$CONFIG_FILE"
    printf '\nProject root recorded at %s\n' "$CONFIG_FILE"

    if [[ -n "$N_SELECTED" ]]; then
        printf '%s\n' "$N_SELECTED" > "$N_SELECTED_FILE"
        printf 'Default n_selected recorded at %s: %s\n' "$N_SELECTED_FILE" "$N_SELECTED"
    fi
else
    printf '\nDone. (config file not updated; --no-write-config was passed)\n'
fi
