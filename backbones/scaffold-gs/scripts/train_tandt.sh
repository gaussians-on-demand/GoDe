#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

for scene in train truck; do
    train "$scene" tandt "$@"
done
