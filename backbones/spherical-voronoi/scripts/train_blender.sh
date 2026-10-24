#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

for scene in lego mic ship chair ficus hotdog materials drums; do
    train "$scene" nerf_synthetic blender.json --white_background "$@"
done
