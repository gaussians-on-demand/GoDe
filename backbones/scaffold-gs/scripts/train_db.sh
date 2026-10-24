#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

for scene in drjohnson playroom; do
    train "$scene" db "$@"
done
