#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

for scene in bonsai kitchen room counter; do
    scalable "$scene" nerf_real_360 --images images_2 "$@"
done
for scene in bicycle garden flowers stump treehill; do
    scalable "$scene" nerf_real_360 --images images_4 "$@"
done
