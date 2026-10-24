# Shared helpers for the benchmark scripts.
# Usage:  DATA_ROOT=/path/to/datasets scripts/train_mipnerf360.sh [extra args...]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the directory holding the datasets}"
OUT_ROOT="${OUT_ROOT:-$REPO/output}"

# Hierarchy settings used for the results in the paper.
NUM_LEVELS="${NUM_LEVELS:-8}"
SAMPLE_LEVEL_RATIO="${SAMPLE_LEVEL_RATIO:-1.1}"
FT_ITERATIONS="${FT_ITERATIONS:-30000}"
PRETRAIN_ITERATIONS="${PRETRAIN_ITERATIONS:-30000}"

# Lower bound of the hierarchy (n_min, Sec. S2.3 of the paper): 300k primitives on Mip-NeRF 360,
# 100k on Tanks & Temples and Deep Blending. Setting HIER_MIN overrides it for every dataset.
hier_min() {
    case "$1" in
        nerf_real_360) echo 300000 ;;
        tandt|db)      echo 100000 ;;
        *)             echo 150000 ;;       # Blender is not part of the paper's evaluation
    esac
}

# train <scene> <dataset-dir> <config> [extra args...]
train() {
    local scene=$1 data=$2 config=$3; shift 3
    python3 "$REPO/train.py" \
        --config "$REPO/config/$config" --scene "$scene" --eval \
        -s "$DATA_ROOT/$data/$scene" -m "$OUT_ROOT/$scene" "$@"
}

# scalable <scene> <dataset-dir> <config> [extra args...]
scalable() {
    local scene=$1 data=$2 config=$3; shift 3
    python3 "$REPO/scalable_train.py" \
        --config "$REPO/config/$config" --scene "$scene" --eval \
        -s "$DATA_ROOT/$data/$scene" -m "$OUT_ROOT/gode/$scene" \
        --checkpoint "$OUT_ROOT/$scene/point_cloud/iteration_$PRETRAIN_ITERATIONS" \
        --num_levels "$NUM_LEVELS" --hier_min "${HIER_MIN:-$(hier_min "$data")}" \
        --sample_level_ratio "$SAMPLE_LEVEL_RATIO" \
        --ft_iterations "$FT_ITERATIONS" --save_images "$@"
}
