# TIG monorepo: two-stage cache branch

This is a private fork of [tig-foundation/tig-monorepo](https://github.com/tig-foundation/tig-monorepo)
carrying one change on top of `main`: c004 (`vector_search`) generates its
database once per precommit and lets an algorithm build a search structure over
it once, instead of regenerating and rebuilding on every nonce. The runtime,
verifier, binary shim and benchmarker slave gained a generic two-stage cache
harness to carry that, usable by future challenges on CPU or GPU.

> **This README is a branch-only explainer, not part of the final
> implementation.** Before the work is proposed upstream, restore the original
> README (kept verbatim at the bottom of this file) and rely on the code
> comments and crate READMEs. Everything above that line exists to make the
> branch reviewable.

## How to navigate the change

Read in this order; each layer only depends on the ones above it.

| Layer | File | What to look for |
|---|---|---|
| Seeds | `tig-structs/src/core.rs` | `calc_db_seed`, `calc_build_seed`, `calc_algo_seed` next to the original `calc_seed`; Python mirror in `tig-benchmarker/common/structs.py` |
| Seeds | `tig-challenges/src/lib.rs` | `Seeds { db, build, instance, algo }` with a comment per field saying who receives it |
| Challenge | `tig-challenges/src/vector_search/mod.rs` | `generate_vectors`, `Database::{generate, to_bytes, from_bytes}`, `Challenge::for_nonce`, the `generate_instance` wrapper, `encode_database` / `decode_database` |
| Binary | `tig-binary/src/entry_point_template.rs` | `build_cache` and `load_cache` shims, CPU and CUDA flavours, behind the `cache` feature; `build_so` enables it with `BUILD_CACHE=1` and lists both in the export map |
| Binary | `tig-algorithms/src/vector_search/template.rs` | the author-facing contract, at the bottom |
| Runtime | `tig-runtime/src/main.rs` | `--challenge-cache` / `--algorithm-cache`; `CacheStage` + `run_cache_stage`; arms `cpu`, `cpu_cached`, `cpu_with`, `gpu`, `gpu_cached`, `gpu_with`; the test-only `fake_cpu` challenge and its `"c000"` arm |
| Verifier | `tig-verifier/src/main.rs` | `--challenge-cache` only, read-only; matching arms |
| Slave | `tig-benchmarker/common/cache.py` | `CACHED_CHALLENGES`, `cache_paths`, `gate_for` / `run_gated`, `purge_cache_files` |
| Slave | `tig-benchmarker/slave/main.py` | where the flags are passed and the gate is applied |
| Tests | `tig-structs/tests/core.rs`, `tig-runtime/src/main.rs` (bottom), `tig-benchmarker/tests/{data,cache}.py` | seed vectors, stage branches with fakes, CLI, gate, naming, purge |

The commits on the branch follow the same order: seeds, challenge split and
verifier, binary exports, runtime and slave, docs.

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

---

# Original monorepo README

Everything below is the upstream `README.md`, unchanged.

<h1 align="center">
  <a href="https://tig.foundation/"><img src="docs/images/logo_black.png" width="75" alt="TIG logo" /></a><br>
  <b>The Innovation Game</b><br>
  <sub>The Network for Algorithmic Breakthroughs </sub>
</h1>
<p align="center">
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/rust-%E2%89%A51.70-orange?logo=rust&logoColor=white" alt="Rust" /></a>
  <a href="https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses"><img src="https://img.shields.io/badge/license-TIG-0ea5e9.svg" alt="License" /></a>
  <a href="https://docs.tig.foundation/"><img src="https://img.shields.io/badge/docs-tig.foundation-6366f1.svg" alt="Docs" /></a>
  <a href="docs/whitepaper.pdf"><img src="https://img.shields.io/badge/whitepaper-PDF-B31B1B.svg" alt="Whitepaper" /></a>
  <a href="https://play.tig.foundation/dashboard"><img src="https://img.shields.io/badge/play-dashboard-10b981.svg" alt="Play" /></a>
  <a href="https://discord.gg/tigfoundation"><img src="https://img.shields.io/badge/Discord-join-5865F2.svg?logo=discord&logoColor=white" alt="Discord" /></a>
  <a href="https://x.com/tigfoundation"><img src="https://img.shields.io/badge/follow-%40tigfoundation-000000.svg?logo=x&logoColor=white" alt="X" /></a>
</p>

The Innovation Game (TIG) creates a new economic framework for algorithmic development - one that aligns incentives, rewards contribution, and keeps innovation open.

At its core, TIG uses a novel proof-of-work variant built around computational challenges grounded in scientifically important problems. Innovators submit algorithms that solve these challenges, and benchmarkers are incentivized to adopt the most efficient ones for proof-of-work, creating a manipulation-resistant signal for rewarding the top-performing algorithms.

In this way, TIG democratizes algorithmic innovation, turning contribution into a sustainable economic opportunity, coordinating global intelligence to compete with centralized incumbents.



## Challenges

TIG currently has 8 active computational challenges:

| ID | Challenge | Description | CPU/GPU |
|----|-----------|-------------|------|
| c001 | [satisfiability](tig-challenges/src/satisfiability/README.md) | Boolean Satisfiability (SAT) | CPU |
| c002 | [vehicle_routing](tig-challenges/src/vehicle_routing/README.md) | Capacitated Vehicle Routing with Time Windows | CPU |
| c003 | [knapsack](tig-challenges/src/knapsack/README.md) | Quadratic Knapsack Problem | CPU |
| c004 | [vector_search](tig-challenges/src/vector_search/README.md) | Vector Range Search | GPU |
| c005 | [hypergraph](tig-challenges/src/hypergraph/README.md) | Hypergraph Partitioning | GPU |
| c006 | [neuralnet_optimizer](tig-challenges/src/neuralnet_optimizer/README.md) | Neural Network Optimizer | GPU |
| c007 | [job_scheduling](tig-challenges/src/job_scheduling/README.md) | Flexible Job Shop Scheduling | CPU |
| c008 | [energy_arbitrage](tig-challenges/src/energy_arbitrage/README.md) | Energy Market Arbitrage | CPU |

## Glossary

| Term | Definition |
|------|-----------|
| **Innovator** | Participant who submits algorithms (code or advances) to solve challenges |
| **Benchmarker** | Participant who runs algorithm benchmarks and submits proofs |
| **Challenge** | A computational problem adapted for optimisable proof-of-work |
| **Code** | An algorithm source code submission by an Innovator |
| **Advance** | An algorithm improvement submission (documentation/paper) by an Innovator |
| **Fuel** | Computational cost metric — algorithms must solve within a fuel budget |
| **OPoW** | Optimisable Proof of Work — TIG's core consensus mechanism |
| **Nonce** | Input seed for a single benchmark run |
| **Runtime Signature** | Hash produced during algorithm execution, used for verification |
| **Binary** | A compiled shared object (`.so`) built from an algorithm submission |

## Important Links

* [Getting Started for Innovators](https://docs.tig.foundation/innovating)
* [Getting Started for Benchmarkers](https://docs.tig.foundation/benchmarking)
* [TIG Documentation](https://docs.tig.foundation/)
* [TIG Whitepaper](docs/whitepaper.pdf)
* [TIG Licensing Explainer](docs/guides/anatomy.md)
* [Code vs Advances](docs/guides/advances.md)
* [Voting Guidelines for Token Holders](docs/guides/voting.md)

## Repo Contents

| Crate | Description |
|-------|-------------|
| [tig-algorithms](./tig-algorithms/README.md) | Hosts algorithm submissions (code and advances) made by Innovators |
| [tig-benchmarker](./tig-benchmarker/README.md) | Python scripts for running TIG's benchmarker in master/slave configuration |
| [tig-binary](./tig-binary/README.md) | Wraps an algorithm submission for compilation into a shared object |
| [tig-challenges](./tig-challenges/README.md) | Implementations of TIG's 8 computational challenges |
| [tig-protocol](./tig-protocol/README.md) | Core protocol logic (block processing, submissions, verification) |
| [tig-runtime](./tig-runtime/README.md) | Executes a compiled algorithm for a single nonce, generating runtime signature and fuel consumed |
| [tig-structs](./tig-structs/README.md) | Shared struct definitions used throughout TIG |
| [tig-token](./tig-token/README.md) | Solidity ERC20 token contract deployed on Ethereum L2 Base chain |
| [tig-utils](./tig-utils/README.md) | Utility functions (hashing, Merkle trees, serialization, etc.) |
| [tig-verifier](./tig-verifier/README.md) | Verifies a single solution or Merkle proof |

## Docker Images

TIG Docker images are hosted on [GitHub Packages](https://github.com/orgs/tig-foundation/packages), supporting `linux/arm64` and `linux/amd64` platforms.

> **Note:** Check `tig-benchmarker/.env` for the current `VERSION` (currently `0.0.5`).

### Dev Images (for Innovators)

Development environment for writing and compiling algorithms:

| Challenge | Image |
|-----------|-------|
| satisfiability | [satisfiability/dev](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fsatisfiability%2Fdev) |
| vehicle_routing | [vehicle_routing/dev](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fvehicle_routing%2Fdev) |
| knapsack | [knapsack/dev](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fknapsack%2Fdev) |
| vector_search | [vector_search/dev](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fvector_search%2Fdev) |
| hypergraph | [hypergraph/dev](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fhypergraph%2Fdev) |
| neuralnet_optimizer | [neuralnet_optimizer/dev](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fneuralnet_optimizer%2Fdev) |
| job_scheduling | [job_scheduling/dev](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fjob_scheduling%2Fdev) |
| energy_arbitrage | [energy_arbitrage/dev](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fenergy_arbitrage%2Fdev) |

### Slave Images (for Benchmarkers)

Runtime images spun up as part of [`slave.yml`](tig-benchmarker/slave.yml) (see [benchmarker README](tig-benchmarker/README.md)):

| Component | Image |
|-----------|-------|
| Slave orchestrator | [benchmarker/slave](https://github.com/orgs/tig-foundation/packages/container/package/tig-monorepo%2Fbenchmarker%2Fslave) |
| satisfiability | [satisfiability/runtime](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fsatisfiability%2Fruntime) |
| vehicle_routing | [vehicle_routing/runtime](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fvehicle_routing%2Fruntime) |
| knapsack | [knapsack/runtime](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fknapsack%2Fruntime) |
| vector_search | [vector_search/runtime](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fvector_search%2Fruntime) |
| hypergraph | [hypergraph/runtime](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fhypergraph%2Fruntime) |
| neuralnet_optimizer | [neuralnet_optimizer/runtime](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fneuralnet_optimizer%2Fruntime) |
| job_scheduling | [job_scheduling/runtime](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fjob_scheduling%2Fruntime) |
| energy_arbitrage | [energy_arbitrage/runtime](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fenergy_arbitrage%2Fruntime) |

### Master Images (for Benchmarkers)

Spun up as part of [`master.yml`](tig-benchmarker/master.yml) (see [benchmarker README](tig-benchmarker/README.md)):

| Component | Image |
|-----------|-------|
| Master orchestrator | [benchmarker/master](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fbenchmarker%2Fmaster) |
| Dashboard UI | [benchmarker/ui](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fbenchmarker%2Fui) |
| PostgreSQL | [benchmarker/postgres](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fbenchmarker%2Fpostgres) |
| Nginx reverse proxy | [benchmarker/nginx](https://github.com/tig-foundation/tig-monorepo/pkgs/container/tig-monorepo%2Fbenchmarker%2Fnginx) |

### Useful Scripts

The `runtime` and `dev` images include these scripts on `PATH`:

```bash
list_algorithms                              # List available algorithms for the challenge
download_algorithm <algorithm_name_or_id>    # Download an algorithm's source
test_algorithm <algorithm_name> <difficulty>  # Test an algorithm locally
```

> The container automatically sets the `CHALLENGE` environment variable (e.g. `knapsack/runtime` sets `CHALLENGE=knapsack`). Use `--testnet` to target testnet.

## License

See README for individual folders.
