#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

for scene in train truck; do
    scalable "$scene" tandt "$@"
done
