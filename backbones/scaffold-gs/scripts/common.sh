# Shared helpers for the benchmark scripts.
# Usage:  DATA_ROOT=/path/to/datasets scripts/train_mipnerf360.sh [extra args...]
#
# Anchor hyper-parameters (voxel size, update factor, appearance dim) keep the
# repository defaults; override them through the environment when a dataset
# needs the values reported in the Scaffold-GS paper.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the directory holding the datasets}"
OUT_ROOT="${OUT_ROOT:-$REPO/output}"
SCAFFOLD_ARGS="${SCAFFOLD_ARGS:-}"

# Hierarchy settings used for the results in the paper.
NUM_LEVELS="${NUM_LEVELS:-8}"
SAMPLE_LEVEL_RATIO="${SAMPLE_LEVEL_RATIO:-1.1}"
FT_ITERATIONS="${FT_ITERATIONS:-30000}"
LOAD_ITER="${LOAD_ITER:-30000}"

# Lower bound of the hierarchy (n_min, Sec. S2.3 of the paper): 100k anchors on Mip-NeRF 360,
# Tanks & Temples and Deep Blending. Setting HIER_MIN overrides it.
hier_min() {
    echo 100000
}

# train <scene> <dataset-dir> [extra args...]
train() {
    local scene=$1 data=$2; shift 2
    python3 "$REPO/train.py" --eval \
        -s "$DATA_ROOT/$data/$scene" -m "$OUT_ROOT/$scene" \
        $SCAFFOLD_ARGS "$@"
}

# scalable <scene> <dataset-dir> [extra args...]
scalable() {
    local scene=$1 data=$2; shift 2
    python3 "$REPO/scalable_train.py" \
        -s "$DATA_ROOT/$data/$scene" -m "$OUT_ROOT/gode/$scene" \
        --checkpoint "$OUT_ROOT/$scene/point_cloud/iteration_$LOAD_ITER" \
        --num_levels "$NUM_LEVELS" --hier_min "${HIER_MIN:-$(hier_min "$data")}" \
        --sample_level_ratio "$SAMPLE_LEVEL_RATIO" \
        --ft_iterations "$FT_ITERATIONS" --save_images \
        $SCAFFOLD_ARGS "$@"
}
