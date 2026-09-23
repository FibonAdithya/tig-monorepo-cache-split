#!/usr/bin/env bash
# Runs ON tig-gpu, inside a gpuq job, from the checkout root.
# Usage: box_build_fuel.sh <log-name>
#
# Measures what an index build costs in BUILD FUEL on every c004 track, through
# the real path: the fuel-instrumented .so and .ptx from build_so/build_ptx,
# and `tig-runtime build-index`, whose stderr reports the fuel each meter
# charged. Each build is followed by one nonce solved against the blob and
# verified, so the row also says what the index is worth (recall as the
# verifier's `quality`), not only what it cost.
#
# Needs, on the box: /opt/llvm (the tig-foundation LLVM release that carries
# LLVMFuelRTSig.so; Dockerfile.dev has the URL), nvcc, and the
# nightly-2025-02-10 toolchain with rust-src.
#
# Env knobs: ALGO (ivf_kmeans), SEED (buildfuel1), TRACKS, CONFIGS (space
# separated "n_lists,n_iters,train_fraction,n_probe"), BUILD_FUEL (1e14, i.e.
# unbounded for this purpose -- the point is to read the meter, not to trip it),
# SOLVE_FUEL (mainnet 5e12).
set -uo pipefail
name=$1
export PATH="/opt/llvm/bin:/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH"
export CHALLENGE=vector_search
ALGO=${ALGO:-ivf_kmeans}
SEED=${SEED:-buildfuel1}
TRACKS=${TRACKS:-"sift_128 glove_100 nytimes_256"}
CONFIGS=${CONFIGS:-"256,0,0.5,8 256,20,0.5,8 1024,0,0.5,32 1024,10,0.5,32 1024,20,0.5,32 1024,20,1.0,32 2048,20,0.5,64 4096,0,0.5,128 4096,20,0.5,128"}
BUILD_FUEL=${BUILD_FUEL:-100000000000000}
SOLVE_FUEL=${SOLVE_FUEL:-5000000000000}

OUT="runs/box/$name"
mkdir -p "$OUT"
exec > >(tee "$OUT/driver.log") 2>&1

echo "== $(date -u +%FT%TZ) HEAD $(git rev-parse HEAD) on $(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

set -e
echo "== building tig-runtime and tig-verifier (--features c004)"
cargo +nightly-2025-02-10 build -r -p tig-runtime --features c004
cargo +nightly-2025-02-10 build -r -p tig-verifier --features c004
RUNTIME=target/release/tig-runtime
VERIFIER=target/release/tig-verifier

echo "== building $ALGO .so (INDEX_BUILD=1) and instrumented .ptx"
INDEX_BUILD=1 bash tig-binary/scripts/build_so "$ALGO" > "$OUT/build_so.log" 2>&1
python3 tig-binary/scripts/build_ptx "$ALGO" > "$OUT/build_ptx.log" 2>&1
SO="tig-algorithms/lib/$CHALLENGE/amd64/$ALGO.so"
PTX="tig-algorithms/lib/$CHALLENGE/ptx/$ALGO.ptx"
ls -l "$SO" "$PTX"
echo "index ABI symbols:"; nm -D "$SO" | grep -E ' T (build_index|load_index)$'
sha256sum "$SO" "$PTX"
set +e

TSV="$OUT/fuel.tsv"
printf 'track\tn_lists\tn_iters\ttrain_fraction\tn_probe\tbuild_rc\tbuild_ms\tblob_bytes\tcpu_fuel\tgpu_fuel\tsolve_rc\tsolve_ms\tsolve_fuel\tverify_rc\tquality\n' > "$TSV"

now_ms() { date +%s%3N; }

for track in $TRACKS; do
  SETTINGS="{\"algorithm_id\":\"\",\"challenge_id\":\"c004\",\"track_id\":\"s=$track\",\"block_id\":\"\",\"player_id\":\"\"}"
  for cfg in $CONFIGS; do
    IFS=, read -r n_lists n_iters train_fraction n_probe <<< "$cfg"
    HP="{\"n_lists\":$n_lists,\"n_iters\":$n_iters,\"train_fraction\":$train_fraction,\"n_probe\":$n_probe}"
    tag="${track}_L${n_lists}_I${n_iters}_T${train_fraction}_P${n_probe}"
    dir="$OUT/$tag"; mkdir -p "$dir"
    echo "== $tag"

    t0=$(now_ms)
    "$RUNTIME" build-index "$SETTINGS" "$SEED" "$SO" \
        --build-fuel "$BUILD_FUEL" --index-out "$dir/index.blob" \
        --ptx "$PTX" --gpu 0 --hyperparameters "$HP" > "$dir/build.out" 2> "$dir/build.err"
    build_rc=$?
    build_ms=$(( $(now_ms) - t0 ))
    line=$(grep -m1 'fuel used' "$dir/build.err")
    cpu_fuel=$(sed -n 's/.*cpu fuel used \([0-9]*\).*/\1/p' <<< "$line")
    gpu_fuel=$(sed -n 's/.*gpu fuel used \([0-9]*\).*/\1/p' <<< "$line")
    blob_bytes=$(stat -c %s "$dir/index.blob" 2>/dev/null || echo 0)
    echo "   build rc=$build_rc ${build_ms}ms blob=$blob_bytes cpu_fuel=${cpu_fuel:-?} gpu_fuel=${gpu_fuel:-?}"
    [ "$build_rc" -ne 0 ] && cat "$dir/build.err"

    solve_rc=; solve_ms=; solve_fuel=; verify_rc=; quality=
    if [ "$build_rc" -eq 0 ]; then
      t0=$(now_ms)
      "$RUNTIME" "$SETTINGS" "$SEED" 0 "$SO" --fuel "$SOLVE_FUEL" --output "$dir" \
          --ptx "$PTX" --gpu 0 --index "$dir/index.blob" --hyperparameters "$HP" \
          > "$dir/solve.out" 2> "$dir/solve.err"
      solve_rc=$?
      solve_ms=$(( $(now_ms) - t0 ))
      solve_fuel=$(python3 -c "import json,sys; print(json.load(open('$dir/0.json'))['fuel_consumed'])" 2>/dev/null)
      if [ "$solve_rc" -eq 0 ]; then
        "$VERIFIER" "$SETTINGS" "$SEED" 0 "$dir/0.json" --ptx "$PTX" --gpu 0 > "$dir/verify.out" 2> "$dir/verify.err"
        verify_rc=$?
        quality=$(sed -n 's/^quality: \([0-9]*\).*/\1/p' "$dir/verify.out" | tail -1)
      else
        cat "$dir/solve.err"
      fi
      echo "   solve rc=$solve_rc ${solve_ms}ms fuel=${solve_fuel:-?} verify rc=${verify_rc:-?} quality=${quality:-?}"
      rm -f "$dir/index.blob"   # regenerable; the ids alone are 4 MB per row set
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$track" "$n_lists" "$n_iters" "$train_fraction" "$n_probe" \
      "$build_rc" "$build_ms" "$blob_bytes" "${cpu_fuel:-}" "${gpu_fuel:-}" \
      "${solve_rc:-}" "${solve_ms:-}" "${solve_fuel:-}" "${verify_rc:-}" "${quality:-}" >> "$TSV"
  done
done

echo "== done $(date -u +%FT%TZ)"
cat "$TSV"
