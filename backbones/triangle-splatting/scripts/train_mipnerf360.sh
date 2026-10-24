#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

for scene in bonsai counter kitchen room; do
    train "$scene" nerf_real_360 indoor.json "$@"
done
for scene in bicycle flowers garden stump treehill; do
    train "$scene" nerf_real_360 outdoor.json "$@"
done
