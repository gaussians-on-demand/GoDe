# Shared helpers for the benchmark scripts.
# Usage:  DATA_ROOT=/path/to/datasets scripts/train_mipnerf360.sh [extra args...]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the directory holding the datasets}"
OUT_ROOT="${OUT_ROOT:-$REPO/output}"

# Hierarchy settings used for the results in the paper.
NUM_LEVELS="${NUM_LEVELS:-8}"
HIER_MIN="${HIER_MIN:-150000}"
SAMPLE_LEVEL_RATIO="${SAMPLE_LEVEL_RATIO:-1.1}"
FT_ITERATIONS="${FT_ITERATIONS:-30000}"

# train <scene> <dataset-dir> [extra args...]
train() {
    local scene=$1 data=$2; shift 2
    python3 "$REPO/train.py" \
        --config "$REPO/configs/$scene.json" --eval \
        -s "$DATA_ROOT/$data/$scene" -m "$OUT_ROOT/$scene" "$@"
}

# scalable <scene> <dataset-dir> [extra args...]
scalable() {
    local scene=$1 data=$2; shift 2
    python3 "$REPO/scalable_train.py" \
        --config "$REPO/configs/$scene.json" \
        -s "$DATA_ROOT/$data/$scene" -m "$OUT_ROOT/gode/$scene" \
        --checkpoint "$OUT_ROOT/$scene/point_cloud/iteration_30000/point_cloud.ply" \
        --num_levels "$NUM_LEVELS" --hier_min "$HIER_MIN" \
        --sample_level_ratio "$SAMPLE_LEVEL_RATIO" \
        --ft_iterations "$FT_ITERATIONS" --save_images "$@"
}
