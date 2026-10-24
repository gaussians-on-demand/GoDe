#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

for scene in lego mic chair ficus hotdog materials drums; do
    scalable "$scene" nerf_synthetic "$@"
done
