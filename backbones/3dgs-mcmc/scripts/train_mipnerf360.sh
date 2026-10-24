#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

for scene in bonsai kitchen room counter; do
    train "$scene" nerf_real_360 --images images_2 "$@"
done
for scene in bicycle garden stump treehill; do
    train "$scene" nerf_real_360 --images images_4 "$@"
done
