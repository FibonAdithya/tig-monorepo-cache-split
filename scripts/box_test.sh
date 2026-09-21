#!/usr/bin/env bash
# Runs ON tig-gpu, inside a gpuq job. Usage: box_test.sh <log-name> <cargo test filter and flags...>
set -euo pipefail
name=$1
shift
export PATH="/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH"
# Absolute and outside the runner's checkout: the runner may clean its tree
# between jobs. Only the #[ignore]d dump test in Task 10 writes here.
export TIG_DUMP_DIR="${TIG_DUMP_DIR:-/workspace/tig-dumps}"
mkdir -p runs/box "$TIG_DUMP_DIR"
# TIG_TEST_EXTRA carries extra libtest flags, e.g. --ignored. Unquoted on
# purpose so that an empty value adds no argument.
# --test-threads=1: the GPU tests share one card and one PTX build.
cargo +nightly-2025-02-10 test -p tig-challenges --features vector_search "$@" \
    -- --test-threads=1 --nocapture ${TIG_TEST_EXTRA:-} 2>&1 | tee "runs/box/$name.log"
