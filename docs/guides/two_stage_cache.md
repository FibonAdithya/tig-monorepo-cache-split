# Two-stage solving: the cache split

> **Not for the final implementation.** This file and the pointer to it in
> `tig-runtime/README.md` exist only to explain what changed on this branch
> and why, for anyone reviewing or continuing it. Drop both before the work is
> proposed upstream; the code comments and the existing crate READMEs are the
> lasting documentation.

This guide explains the changes on this branch to `tig-runtime`, `tig-verifier`,
`tig-binary`, `tig-challenges/vector_search`, `tig-structs` and the benchmarker
slave, relative to `main` of the monorepo. Read it before touching any of them.

## The problem it solves

On `main`, each nonce is one `tig-runtime` process. For c004 (`vector_search`)
that process regenerates the 700,000-vector database and its 7,000 queries,
hands both to the algorithm, solves, writes its output and exits. Nothing
survives to the next nonce. Two consequences:

* Generation dominates the wall-clock of a nonce. A do-nothing solver costs
  about the same as a real one.
* An index built inside a nonce is thrown away at the end of it, so no search
  structure can pay for itself. The rational algorithm is a brute-force scan.

Real ANN benchmarks separate index construction from query search. This branch
does the same: the database is fixed per precommit, only the queries vary per
nonce, and both the database and whatever the algorithm builds over it are
cached on disk and reused by every later nonce.

## The shape in one picture

```
per precommit                          per nonce (one process each, as before)
--------------------------------       ---------------------------------------
db_seed = H(settings, rand_hash)       instance_seed = H(settings, rand_hash, nonce)
Database::generate(db_seed)            algo_seed     = H(settings, rand_hash, nonce, "algo")
  -> written to the challenge cache
algorithm.build_cache(&db, build_seed)   runtime:
  -> written to the algorithm cache        1. Database from the challenge cache
                                           2. algorithm.load_cache(&db, bytes)
                                           3. Challenge::for_nonce(&db, seeds)
                                           4. initialize_kernel   <- fuel meter starts here
                                           5. entry_point(&challenge, ...)
                                           6. finalize_kernel     <- fuel read here
```

The first process of a precommit on a slave finds both caches missing, writes
them, and continues straight into its own solve. Every later process reads
them. The slave makes sure only one process runs until the caches exist.

## Seeds (`tig-structs`, `tig-challenges`)

`BenchmarkSettings` derives four seeds instead of one. All go through the same
hash with a domain tag so none can collide.

| Method | Nonce? | Who receives it | Purpose |
|---|---|---|---|
| `calc_db_seed(rand_hash)` | no | challenge crate only | generates the database |
| `calc_build_seed(rand_hash)` | no | the algorithm's `build_cache` | algorithm RNG during the build |
| `calc_seed(rand_hash, nonce)` | yes | challenge crate only | generates the per-nonce instance (c004: the queries) |
| `calc_algo_seed(rand_hash, nonce)` | yes | the algorithm, as `Challenge::seed` | algorithm RNG during the solve |

`calc_seed` is unchanged, so c005, c006 and c004's queries are bit-identical
to `main`. The Python mirrors in `tig-benchmarker/common/structs.py` are pinned
to the Rust ones by identical byte vectors in both test suites.

`tig_challenges::Seeds { db, build, instance, algo }` carries all four. The GPU
challenges' `generate_instance` takes `&Seeds`; c005 and c006 read only
`instance`. The algorithm is never handed a seed that generates its own input.
(It runs in the runtime's process and could read the command line, so "hidden"
means "not handed over", not a hard barrier.)

## The c004 instance split (`tig-challenges/src/vector_search/mod.rs`)

* `generate_vectors` is one pass of the GAN into one buffer.
* `Database { scenario, vector_dims, database_size, d_database_vectors }` and
  `Database::generate(db_seed, track, ...)`: the rows, from the db seed alone.
* `Challenge::for_nonce(&db, &seeds, track, ...)`: the queries from the
  instance seed at the same latent offset as before (so they match `main`),
  a device copy of the database into the `Challenge` struct (whose layout
  compiled algorithms depend on), and `seed` filled from `seeds.algo`.
* `Challenge::generate_instance` is now the two calls in sequence, so every
  existing caller, including the verifier, is unchanged.
* `encode_database` / `decode_database` and `Database::to_bytes` /
  `from_bytes`: the on-disk form. Header: magic `TIGVSDB1`, the db seed, dims,
  row count; then little-endian floats. Decoding refuses a wrong seed, a wrong
  scenario, or a length that disagrees with the header.

## The algorithm binary (`tig-binary`)

Two optional exports next to `entry_point`, in a CPU and a CUDA flavour:

```rust
// CUDA (c004 today)
fn build_cache(&Database, seed: &[u8; 32], Option<String>, Arc<CudaModule>, Arc<CudaStream>, &cudaDeviceProp) -> Result<Vec<u8>>
fn load_cache (&Database, blob: &[u8],                     Arc<CudaModule>, Arc<CudaStream>, &cudaDeviceProp) -> Result<()>
// CPU (no challenge uses it yet)
fn build_cache(&Database, seed: &[u8; 32], Option<String>) -> Result<Vec<u8>>
fn load_cache (&Database, blob: &[u8]) -> Result<()>
```

They compile only behind the `cache` cargo feature, which `build_so` turns on
when `BUILD_CACHE=1` is set, and the linker export map lists both names. An
algorithm that defines neither builds exactly as before and the runtime solves
without a cache.

**Contract.** `build_cache` returns the bytes to persist *and* leaves the
algorithm ready to solve, exactly as `load_cache` on those bytes would. The
process that builds continues straight into `solve_challenge` without a
reload. The usual pattern is a `static OnceLock` that either function fills and
`solve_challenge` reads. The bytes are opaque to everything but the algorithm.

See the commented signatures at the bottom of
`tig-algorithms/src/vector_search/template.rs`.

## The runtime (`tig-runtime`)

The legacy single-nonce form is unchanged. Two optional flags were added:

```
--challenge-cache PATH   the challenge's shared input (the database):
                         written if missing, read if present
--algorithm-cache PATH   the algorithm's build_cache output:
                         written if missing (build), read if present (load)
```

Either flag works alone. Every challenge without a shared input refuses both.

Inside, the two-stage logic is one function, `run_cache_stage`, that takes the
device- and challenge-specific steps as closures: generate, encode, decode,
build, load. The `gpu_cached` arm (c004) and the `cpu_cached` arm (no shipped
challenge yet; a test-only fake keeps it compiling) differ only in what they
pass. The old GPU and CPU bodies became `gpu_with` and `cpu_with`, taking their
challenge from a closure.

After a build the runtime checks the device error flag, since a fuel trap on
the device is asynchronous, and refuses to write a cache from a build that
trapped. Files are written to `<path>.tmp.<pid>` and renamed, so a reader never
sees a partial file and two racing builders cannot corrupt each other.

**Fuel.** Nothing in either stage is charged to the nonce. The device meter
starts at `initialize_kernel`, which runs after both stages, and the CPU
counter is re-armed after them. The build's kernels do run under the nonce's
patched device limit, so a build that needs more than one nonce's budget traps.

## The verifier (`tig-verifier`)

One flag, `--challenge-cache PATH`, read only. If the file is present and its
header matches the seed the verifier decodes it instead of regenerating; if
not it regenerates. There is deliberately no algorithm-cache flag: verification
depends only on what the challenge crate derives from the seed. Consensus
verification runs without the flag and always regenerates.

## The slave (`tig-benchmarker`)

`common/cache.py` owns everything the slave knows about caches:

* `CACHED_CHALLENGES = {"c004"}` and `has_cache(batch)`: which challenges get
  the flags.
* `cache_paths(results_dir, batch)`: the file names, keyed by precommit:

  ```
  <results_dir>/<benchmark_id>_<challenge_id>_challenge_cache.bin
  <results_dir>/<benchmark_id>_<algorithm_id>_algorithm_cache.bin
  ```

  `benchmark_id` is what pins the database seed, so every batch of a precommit
  on one slave shares one build. `block_id` would not do: two precommits on
  one block with different tracks or algorithms have different databases.
* `gate_for(benchmark_id)` and `run_gated`: only one worker runs while the
  caches are missing. The others wake every quarter second and proceed as soon
  as both files exist, which the builder writes before it starts its own solve.
  A failed builder hands the lock to the next waiter. After the first success
  nothing is gated.
* `purge_cache_files`: the files are deleted once no batch of the precommit is
  pending, processing, ready or awaiting purge.

`slave/main.py` passes both paths to `tig-runtime` and only the challenge path
to `tig-verifier` (its local quality check), and wraps each nonce in
`run_gated`.

Lifetime of a cache: one precommit on one slave. Nothing persists across
precommits, because `rand_hash` changes the database.

## Running it by hand

Inside the c004 runtime container, with an algorithm built with `BUILD_CACHE=1`:

```bash
SETTINGS='{"challenge_id":"c004","difficulty":"s=sift_128","algorithm_id":"...","player_id":"...","block_id":"..."}'
tig-runtime "$SETTINGS" "$RAND_HASH" 0 algo.so --ptx algo.ptx \
    --challenge-cache /app/results/x_c004_challenge_cache.bin \
    --algorithm-cache /app/results/x_algo_algorithm_cache.bin      # builds both
tig-runtime "$SETTINGS" "$RAND_HASH" 1 algo.so --ptx algo.ptx \
    --challenge-cache ... --algorithm-cache ...                     # reads both
tig-verifier "$SETTINGS" "$RAND_HASH" 1 1.json --ptx algo.ptx \
    --challenge-cache /app/results/x_c004_challenge_cache.bin      # reads the database only
```

## Adding a challenge to the two-stage path

1. In the challenge crate: a `Database` with `generate`, `to_bytes` and
   `from_bytes`, and `Challenge::for_nonce(&db, &seeds, track, ...)`. The
   shared half must not depend on the nonce; the per-nonce half must be
   regenerable by the verifier from seeds alone.
2. In `tig-runtime` and `tig-verifier`: switch the challenge's arm in the
   `match` from `cpu` / `gpu` to `cpu_cached` / `gpu_cached`.
3. In `tig-benchmarker/common/cache.py`: add the challenge id to
   `CACHED_CHALLENGES`.

The `fake_cpu` module in `tig-runtime/src/main.rs` is the minimal example of
what step 1 must provide.

## What is verified and what is not

Everything that compiles without CUDA is tested here: the seed derivations in
both languages, the database codec, `run_cache_stage` with fakes, the CPU arm
via the fake challenge, the CLI, and the slave's gate, naming and purge logic.

Everything behind the CUDA feature is not: the c004, c005 and c006 arms in the
runtime and verifier, the challenge crate's split, and the shim. Those need
`cargo check --features c004,c005,c006` on a machine with `nvcc`, then one real
batch through a slave with an algorithm that exports the pair.

## Limits to keep in mind

* Nothing in consensus re-runs a build or reads a cache. The caches are a
  benchmarker-local optimisation; a tampered one only makes that benchmarker's
  own solutions fail verification.
* Only the final per-nonce solution is scored. A blob is a claim nobody
  checks, so an intermediate result can never carry score.
* A build runs in a process that also holds a nonce. An earlier design kept
  the build in a nonce-free process as defence in depth for a future where fuel
  is a score term; that is not the case today.
