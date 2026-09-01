# tig-runtime

A Rust crate that execute an algorithm (compiled from [`tig-binary`](../tig-binary/README.md)) for a single nonce, generating runtime signature and fuel consumed for verification.

# Getting Started

Users who don't intend to customise `tig-runtime` are recommended to download pre-compiled version available in [TIG's runtime docker images](../README.md#docker-images).

Note there is a different `tig-runtime` for each challenge.

**Example:**
```
CHALLENGE=knapsack
VERSION=0.0.1
docker run -it ghcr.io/tig-foundation/tig-monorepo/$CHALLENGE/runtime:$VERSION

# inside docker
tig-runtime --help
```

## Compiling

The required rust environment for development are available via [TIG's development docker images](../README.md#docker-images).

You will need to add `--features <CHALLENGE>` to compile for a specific challenge.

**Example:**
```
# clone this repo
cd tig-monorepo
CHALLENGE=knapsack
VERSION=0.0.1
docker run -it -v $(pwd):/app ghcr.io/tig-foundation/tig-monorepo/$CHALLENGE/dev:$VERSION

# inside docker
cargo build -p tig-runtime --release --features knapsack
```

# Usage

`tig-runtime` has three forms. The legacy single-nonce form is unchanged; the two
subcommands were added for challenges that build a search structure once per
precommit (currently only c004 `vector_search`).

```
Usage: tig-runtime [OPTIONS] <SETTINGS> <RAND_HASH> <NONCE> <BINARY>
       tig-runtime build-index [OPTIONS] --build-fuel <FUEL> --index-out <PATH> <SETTINGS> <RAND_HASH> <BINARY>
       tig-runtime batch [OPTIONS] --start-nonce <N> --num-nonces <N> <SETTINGS> <RAND_HASH> <BINARY>

Arguments:
  <SETTINGS>   Settings json string or path to json file
  <RAND_HASH>  A string used in seed generation
  <NONCE>      Nonce value                            (legacy form only)
  <BINARY>     Path to a shared object (*.so) file

Options:
      --hyperparameters [<HYPERPARAMETERS>]  Hyperparameters json string or path to json file
      --ptx [<PTX>]             Path to a CUDA ptx file
      --fuel [<FUEL>]           Optional maximum fuel parameter [default: 2000000000]
      --output [<OUTPUT_FOLDER>]  If set, output data is saved to this folder (default current directory)
      --index [<PATH>]          Index blob to load before solving
      --gpu [<GPU>]             Which GPU device to use [default: 0]
  -h, --help                    Print help
```

> **Changed:** `--output` takes a **folder**, not a file path — earlier versions
> of this document said file. `--compress` has been **removed**.

## `build-index`

Builds an index over the precommit's database and exits. **Takes no nonce**, by
design: the build process must not be able to derive the query set it will later
be asked about.

```
      --build-fuel <FUEL>     Fuel budget for the build phase              [required]
      --index-out <PATH>      Where to write the index blob               [required]
      --memory-cap [<BYTES>]  Device memory the algorithm may allocate on top of the
                              context and database  [default: 2147483648 (2 GiB)]
      --build-timeout [<SECS>]  Wall-clock watchdog for the build          [default: 600]
```

`--ptx` is effectively required here: omitting it is a runtime error, not a
default. The 2 GiB memory cap is roughly 2x the measured 1,010 MiB build-phase
peak; the balloon refuses to run at all when free memory is below the cap plus
64 MiB, which is why the cap is not larger.

Requires a build with `--features c004`. Any other build exits 84 rather than
silently succeeding without building anything.

## `batch`

Solves a contiguous run of nonces in one process, loading the index once instead
of once per nonce.

```
      --start-nonce <N>   First nonce of a batch                    [required]
      --num-nonces <N>    How many nonces to solve (minimum 1)      [required]
      --fuel [<FUEL>]     Optional maximum fuel parameter [default: 2000000000]
      --index [<PATH>]    Index blob to load before solving
      --output [<OUTPUT_FOLDER>]  Folder for the per-nonce output files
```

`batch` is not a second execution path — the legacy single-nonce form is exactly
`start_nonce = NONCE, num_nonces = 1`, reaching the same call. Two paths is
precisely where batched and single-nonce results would silently diverge.

> **`batch` has no callers today.** The reference slave in
> `tig-benchmarker/slave/main.py` still runs one process per nonce. The
> subcommand works, but nothing in the shipped stack invokes it.

## Exit codes

* 0 - solution found
* 82 - cuda out of memory
* 83 - host out of memory
* 84 - runtime error
* 85 - no solution found
* 86 - invalid solution
* 87 - out of fuel

**Example (legacy form):**
```
CHALLENGE=satisfiability
VERSION=0.0.1
docker run -it -v $(pwd):/app ghcr.io/tig-foundation/tig-monorepo/$CHALLENGE/runtime:$VERSION

# inside docker
download_algorithm sat_global_opt --testnet

ARCH=$(if [ "$(uname -i)" = "aarch64" ] || [ "$(uname -i)" = "arm64" ] || [ "$(arch 2>/dev/null || echo "")" = "aarch64" ] || [ "$(arch 2>/dev/null || echo "")" = "arm64" ]; then
    echo "arm64"
else
    echo "amd64"
fi)
SETTINGS='{"challenge_id":"c001","difficulty":[50,300],"algorithm_id":"","player_id":"","block_id":""}'
RANDHASH='rand_hash'
NONCE=1337
FUEL=987654321123456789
SO_PATH=./tig-algorithms/lib/satisfiability/$ARCH/sat_global_opt.so

tig-runtime $SETTINGS $RANDHASH $NONCE $SO_PATH --fuel $FUEL
```

**Example Output:**
```
{"cpu_arch":"arm64","fuel_consumed":97188,"nonce":1337,"runtime_signature":13607024390209669967,"solution":{"variables":[1,0,0,0,0,1,1,1,0,1,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0]}}
```

# c004: two-phase index build + solve

c004 (`vector_search`) runs in two phases. The build happens once per precommit;
the solve happens once per nonce and loads what the build produced.

## The mount that makes it work

The index blob is written by one process and read by another. It crosses between
them **only** through a shared bind mount. In `tig-benchmarker/slave.yml` both
the slave and each challenge container get:

```yaml
volumes:
  - ${ALGORITHMS_DIR}:/app/algorithms   # the .so and .ptx
  - ${RESULTS_DIR}:/app/results         # where the index blob lives
```

`--index-out` **must** name a path under `/app/results`. Point it anywhere else
and the build exits 0 having written a blob the solve phase cannot open. GPU
challenges additionally need the nvidia runtime (`runtime: nvidia` in compose,
or `--gpus all` on a bare `docker run`).

## The two invocations

Taken from `run_build_index` and `run_tig_runtime` in
`tig-benchmarker/slave/main.py`:

```bash
BATCH=<batch-id>
SETTINGS='{"challenge_id":"c004","difficulty":"s=sift_128", ...}'

# Phase 1 — once per precommit. Note: no nonce argument exists on this form.
docker exec vector_search tig-runtime build-index \
    "$SETTINGS" "$RAND_HASH" "$SO_PATH" \
    --build-fuel "$BUILD_FUEL" \
    --index-out "/app/results/$BATCH/index.blob" \
    --ptx "$PTX_PATH"

# Phase 2 — once per nonce, consuming the blob from phase 1.
docker exec vector_search tig-runtime \
    "$SETTINGS" "$RAND_HASH" "$NONCE" "$SO_PATH" \
    --fuel "$FUEL" \
    --output "/app/results/$BATCH" \
    --index "/app/results/$BATCH/index.blob" \
    --ptx "$PTX_PATH"
```

## When the build phase actually fires

The slave decides by **membership**, not truthiness
(`tig-benchmarker/common/batch.py`):

```python
return "build_fuel_budget" in batch
```

`build_fuel_budget == 0` is a real value — "no build fuel granted" — and is not
the same as "this challenge has no build phase".

> **No batch the master produces today carries `build_fuel_budget`,** so
> `needs_index_build` returns `False` for every live batch and the build phase
> never fires. That is the expected state, not a regression: wiring it needs a
> postgres schema migration and a Python reimplementation of the Rust fuel
> arithmetic. Until then, do not set `build_fuel_alpha` in protocol config — with
> the master unwired, every c004 batch would fail.

Design and rationale: `docs/superpowers/specs/2026-08-31-c004-index-build-split-design.md`.
Challenge semantics: [`vector_search/README.md`](../tig-challenges/src/vector_search/README.md).

# License

[End User License Agreement](../docs/licenses/TIG_Game_Source_Code_End_User_License_Agreement_v1.0.pdf)