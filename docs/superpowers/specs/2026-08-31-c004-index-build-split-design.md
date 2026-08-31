# c004: Splitting Index Building from Query Search

**Date:** 2026-08-31
**Status:** Design approved in chat; not implemented.
**Branch:** `vector_search/gan_instance_gen`

## Problem

c004 today generates a fresh 700,000-vector database *and* its 7,000 query
vectors from a single per-nonce seed, hands both to the algorithm at once, and
meters the whole solve against one fuel budget. Two consequences:

1. **Generation dominates.** Earlier measurement on `tig-gpu` (recorded in the
   `## Validation` section of `2026-08-13-gan-instance-generation-design.md`): a
   real algorithm takes 1165 ms per nonce and a do-nothing solver takes 1171 ms.
   Essentially the entire nonce is instance generation. Whatever an algorithm
   does to search faster is invisible against that floor.
2. **No index can pay for itself.** An index built inside a nonce is thrown away
   at the end of that nonce. The rational strategy is therefore a brute-force or
   near-brute-force scan, which is what the field does.

The goal is the framing real ANN benchmarks use: **index construction is free
within a bounded budget, and only query search is charged.**

## Established facts

Everything below was read out of this tree or the live protocol config, not
assumed.

| Fact | Source |
|---|---|
| `seed = u8s_from_str("{jsonify(settings)}_{rand_hash}_{nonce}")` | `tig-structs/src/core.rs:171` |
| `BenchmarkSettings = {player_id, block_id, challenge_id, algorithm_id, track_id}` | `tig-structs/src/core.rs:161` |
| `fuel_budget` lives in `PrecommitDetails`, **not** in settings | `tig-structs/src/core.rs:447` |
| The verifier does **not** re-run the algorithm; it regenerates the instance and evaluates the submitted solution | `tig-verifier/src/main.rs:139` |
| The fuel *limit* is baked into the PTX at load (`0xdeadbeefdeadbeef` -> `max_fuel * 20`) | `tig-runtime/src/main.rs:199` |
| `initialize_kernel` zeroes `gbl_FUELUSAGE`, and the runtime calls it *after* `generate_instance` | `tig-binary/src/framework.cu:15`, `tig-runtime/src/main.rs:213-233` |
| `fuel_budget <= max_fuel_budget` is validated in the protocol | `tig-protocol/src/contracts/benchmarks.rs:89` |
| The slave already batches: `start_nonce`, `num_nonces`, `batch_size`, one `rand_hash` per batch, per-batch Merkle root | `tig-benchmarker/slave/main.py:60` |
| Generation is a two-pass loop `[(false, database_size), (true, n_queries)]` with an `index_base` offset; each latent is a pure function of `(seed, global_index)` | `tig-challenges/src/vector_search/mod.rs:125-150` |
| `SIFT_128`: 7,000 queries, 700,000 database rows, 128 dims, `min_recall` 0.95, 1,000 audit samples | `tig-challenges/src/vector_search/scenarios.rs` |
| Live c004 config: `max_fuel_budget` 5e12, `min_num_bundles` 4, `num_nonces_per_bundle` 20 on the 7000 track, `max_qualifiers_per_track` 100 | `https://mainnet-api.tig.foundation/get-block` |
| Brute-force exact 1-NN costs ~3.84e11 fuel; `there_v10` costs ~4.67e9 | `2026-08-13-gan-instance-generation-design.md` |

Two of these are load-bearing in a way that is easy to miss:

- **Fuel is not the binding constraint.** Brute force at 3.84e11 fits inside
  `max_fuel_budget` of 5e12 with 13x to spare. What actually limits a
  benchmarker is wall-clock per nonce. Exempting the build from *fuel* is
  bookkeeping; exempting it from *per-nonce wall-clock* is the substance, and
  that only happens through amortisation across many nonces.
- **A free phase before the meter already exists.** `generate_instance` runs
  before `initialize_kernel` zeroes the fuel counter, so instance generation is
  already fuel-exempt. This design extends an existing pattern rather than
  introducing one.

## Decisions

**D1. The database is fixed per precommit; only queries vary per nonce.**
Derived from `hash(settings, rand_hash)` with no nonce. `rand_hash` is fresh per
precommit, so the database is not precomputable offline, but it is shared by
every nonce of that precommit and an index over it amortises.

**D2. Scoring is unchanged: quality stays audited recall@1.** Fuel remains a
spend limit, not a score term. The build is simply not charged against it.
Competitive pressure stays where it is today - a benchmarker prefers whichever
algorithm clears recall 0.95 with the least wall-clock, because that yields the
most nonces per hour.

*Rejected:* making quality a function of query-phase fuel. `fuel_consumed` is
self-reported and reproduced by nobody (see the verifier row above), so a
fuel-scored quality is unenforceable without verified re-execution. Deferred,
not dismissed; D6's phase boundary is where it would attach.

**D3. The build runs in a separate process that is never given a nonce.**
This is the anti-gaming property the whole design rests on. If the build phase
could see the query vectors, the winning strategy would be to brute-force all
7,000 queries during the free phase and replay stored answers for near-zero
fuel: brute force costs ~1.2 s against a 10-minute window, roughly 500x more
room than it needs. Withholding queries via API discipline is weak, because the
algorithm's `.so` shares an address space with whatever the runtime holds.
Withholding them by never passing the nonce into the build process is
structural - the material to derive any query set is not present.

**D4. The bundle's nonces run in one batched query process.** It loads the index
blob once and loops the batch's nonces, so the index upload is paid once per
precommit rather than once per nonce. This is also forced by the PTX fuel patch:
the fuel limit is baked in at module load, so a build budget and a solve budget
cannot coexist in one loaded module.

**D5. `Challenge` keeps its exact current layout; each nonce gets a
device-to-device copy of the database.** A borrowed `&CudaSlice` would change
the struct layout, and every existing `.so` reads those fields by offset - which
means all 105 c004 algorithms rebuilt and resubmitted. A device-to-device copy
of 358 MB costs ~1.4 ms at T4 bandwidth (~3% of an estimated 50 ms nonce) and
358 MB of extra device memory, and buys a completely unbroken ABI.

**D6. The two new algorithm symbols are optional, and the index handle never
crosses the FFI boundary.** The algorithm stashes it in its own `static` during
`load_index`. `entry_point` keeps today's signature, so the `gpu` arm of
`dispatch_challenge!` - shared by c004, c005 and c006 - is untouched.

**D7. The build budget is proportional to the work it is amortised over, not a
flat wall-clock.** See "Why not a flat 10 minutes" below.

**D8. The memory cap is enforced by a balloon allocation, not by polling.**

## Why not a flat 10 minutes

A precommit's floor is `min_num_bundles * num_nonces_per_bundle` = 4 * 20 = **80
nonces**. Against today's measured 1.2 s/nonce and an *estimated* 0.05 s/nonce
after the split (~12 ms query generation plus indexed search):

| nonces/precommit | today | with a 600 s build | verdict |
|---|---|---|---|
| 80 (protocol floor) | 96 s | 604 s | 6.3x **worse** |
| 500 | 600 s | 625 s | break-even |
| 2,000 | 2,400 s | 700 s | 3.4x better |
| 10,000 | 12,000 s | 1,100 s | 10.9x better |

Break-even sits near 520 nonces. A flat 10-minute cap is therefore not a cap at
small precommit sizes; it is a footgun that makes the benchmarker strictly
slower with nothing in the protocol to stop them. Instead:

```
build_fuel_budget = min(alpha * num_nonces * fuel_budget, max_build_fuel_budget)
```

`alpha` is a protocol constant (**provisional 0.25**, to be set after
measurement - see Open Questions). The rule reads: the build may consume at most
this fraction of the total scored fuel the precommit will spend. It self-scales,
it is deterministic and hardware-independent, and it makes a heavy index
something a benchmarker must commit to a large precommit to earn.

`max_build_fuel_budget` is not cosmetic. The runtime patches the PTX with
`build_fuel_budget * gpu_fuel_scale` where `gpu_fuel_scale = 20`; at
`fuel_budget` 5e12, `num_nonces` 100,000 and `alpha` 0.25 the product is 1.25e17,
and 2.5e18 after scaling - within `u64` but with less than one order of
magnitude of headroom. The product must be computed in `u128` (or with
`checked_mul`) and `max_build_fuel_budget * gpu_fuel_scale < u64::MAX` must be
asserted at config load.

The 10-minute wall-clock survives as a **watchdog ceiling**: a flat safety net
that kills a runaway build, not the budget itself.

## Architecture

```
                 settings + rand_hash              (no nonce anywhere)
                          |
      tig-runtime --build-index                    PTX patched with build_fuel_budget
                          |
        db_seed = H(settings, rand_hash)
                          |
              Database::generate()  -->  700k x 128 vectors  (~358 MB)
                          |
              [balloon allocated: free - memory_cap]
                          |
                 algorithm::build_index()          watchdog: 10 min wall-clock
                          |
                    index.blob                     opaque, algorithm-defined bytes
                          |
      tig-runtime --start-nonce A --num-nonces M   PTX patched with fuel_budget
                          |
        Database::generate()   (same db_seed, regenerated once)
        algorithm::load_index(&blob)               one upload, held in the .so's static
                          |
         for nonce in A..A+M:
             d2d copy database into Challenge       ~1.4 ms
             generate_queries(H(settings, rand_hash, nonce))
             initialize_kernel(sig)                <- fuel meter zeroed HERE
             entry_point(&challenge, save_solution, ..)
             finalize_kernel()                     <- fuel_consumed for THIS nonce
```

Verification is unchanged: `tig-verifier` derives both seeds, regenerates both
halves, and audits 1,000 salt-sampled queries. It never loads an index and never
runs the build.

## Interfaces

### Seed derivation

`tig-structs/src/core.rs`, mirrored in `tig-benchmarker/common/structs.py` (the single Python copy;
`slave/common` and `master/common` symlink to it):

```rust
pub fn calc_seed(&self, rand_hash: &String, nonce: u64) -> [u8; 32]   // unchanged
pub fn calc_db_seed(&self, rand_hash: &String) -> [u8; 32] {
    u8s_from_str(&format!("{}_{}_db", jsonify(&self), rand_hash))
}
```

The `_db` tag is domain separation. A nonce renders as a `u64`, so it can never
render as `db`; collision with the nonce form is impossible by construction
rather than by argument.

Because `BenchmarkSettings` carries `player_id` and `algorithm_id`, the database
is already per-player and per-algorithm: no player can free-ride on another's
index, and switching algorithms correctly invalidates it. Because `fuel_budget`
is *not* in settings, changing the budget does not churn the database.

### Types

```rust
pub struct Database {
    pub scenario: Scenario,
    pub vector_dims: u32,
    pub database_size: u32,
    pub d_database_vectors: CudaSlice<f32>,
}

// UNCHANGED from today, field for field, by D5.
pub struct Challenge {
    pub seed: [u8; 32],
    pub scenario: Scenario,
    pub num_queries: u32,
    pub vector_dims: u32,
    pub database_size: u32,
    pub d_database_vectors: CudaSlice<f32>,   // d2d copy of Database's
    pub d_query_vectors: CudaSlice<f32>,
}

impl Database { pub fn generate(db_seed, track, module, stream, prop) -> Result<Self> }
impl Challenge { pub fn for_nonce(db: &Database, seed, track, module, stream, prop) -> Result<Self> }
```

`Database::generate` is the `(false, database_size)` pass of today's loop with
`index_base = 0`; `Challenge::for_nonce` is the `(true, n_queries)` pass. The
`index_base = database_size` offset is retained even though the differing seeds
already separate the streams - keeping it means **`kernels.cu` is not touched**,
which `scenarios.rs` documents as the thing that avoids forcing a network-wide
rebuild of the generation path.

### Algorithm ABI

```rust
// OPTIONAL symbol. Absent -> the runtime skips the build entirely.
fn build_index(&Database, Option<String>, Arc<CudaModule>, Arc<CudaStream>, &cudaDeviceProp)
    -> Result<Vec<u8>>;

// OPTIONAL symbol. Required if build_index is present.
fn load_index(&Database, &[u8], Arc<CudaModule>, Arc<CudaStream>, &cudaDeviceProp)
    -> Result<()>;

fn entry_point(..)   // UNCHANGED
```

`Vec<u8>` and `&[u8]` across the dylib boundary are no worse than what already
crosses it (`&dyn Fn`, `Arc<CudaModule>`, `anyhow::Result`), which already
assumes an identical-toolchain build.

`build_index` present but `load_index` absent is an error raised at **build**
time, not at query time - the failure must not wait until after a 10-minute
build has been paid for.

An empty blob (`Vec::new()`) is legal and means "no index"; `load_index` receives
an empty slice.

### The shared GPU dispatch arm

`generate_instance` is called from the `gpu` arm of `dispatch_challenge!` in both
the runtime and the verifier, and that arm is shared by c004, c005 and c006. The
verifier needs *both* seeds for c004, so the arm cannot keep passing a single
`[u8; 32]`.

Resolution: the shared signature takes a struct, and the two challenges that do
not split ignore the extra field.

```rust
pub struct Seeds { pub nonce: [u8; 32], pub db: [u8; 32] }

// all three GPU challenges
fn generate_instance(seeds: &Seeds, track: &Track, module, stream, prop) -> Result<Self>
```

c005 and c006 read `seeds.nonce` and ignore `seeds.db`, exactly as they already
take `_audit_salt` and ignore it (`hypergraph/mod.rs`). That precedent is why
this is a safe shape rather than a novel one.

c004 keeps `generate_instance` as the single-nonce path (it derives both halves
itself) and additionally exposes `Database::generate` / `Challenge::for_nonce`
for the batched path. c005 and c006 are never run in batched mode, so they need
no equivalent.

### Runtime CLI

Additive. Single-nonce mode is preserved because `tig-verifier` and the slave's
re-verify path depend on it.

```
tig-runtime --build-index <SETTINGS> <RAND_HASH> <BINARY> --ptx P \
            --build-fuel N --memory-cap BYTES --build-timeout SECS --index-out F
tig-runtime <SETTINGS> <RAND_HASH> <NONCE> <BINARY> [--index F]              # unchanged
tig-runtime <SETTINGS> <RAND_HASH> --start-nonce A --num-nonces M <BINARY> \
            [--index F] --output D
```

- `--build-index` with `--start-nonce`, `--num-nonces` or a positional `NONCE`
  is rejected at argument parsing. The build process must not be able to learn a
  nonce even by operator error.
- `--num-nonces 0` is rejected rather than silently producing nothing.
- The index blob is written to a temporary path and atomically renamed, so a
  watchdog kill never leaves a partial blob for the query process to consume.

## Enforcement

| Cap | Mechanism | On violation |
|---|---|---|
| Build fuel | PTX patched with `build_fuel_budget * 20`; `trap` sets `gbl_ERRORSTAT` | non-zero exit, no index written |
| Wall-clock (watchdog, 600 s) | watchdog thread kills the process | non-zero exit, no index written |
| Device memory (8 GB provisional) | balloon: after generating the database, allocate `free - cap` and hold it | `cudaMalloc` fails inside `build_index`, which returns `Err` |

The balloon is chosen over polling `cudaMemGetInfo` because polling samples, and
a transient spike between samples goes unseen. With the balloon held, the
effective cap is identical on every GPU whose total memory is at least
`cap + headroom`. **If `free < cap` at balloon time the run must error out, not
allocate a zero-sized balloon** - a silently unenforced cap is worse than a
failed build.

Sizing on the weakest listed `ComputeType` (`AWS_G4dn`, T4, 16 GB): fixed costs
are ~0.94 GB, measured by parsing `weights/v1_sift.bin` (layers
128->512->1024->1024->128, so `widest` is 1024): 358.4 MB database, 536.9 MB for
the two `FORWARD_CHUNK * widest` scratch buffers, 33.6 MB of latents, ~7 MB of
weights, plus context overhead. An 8 GB cap therefore leaves comfortable
headroom, and the query process holds database +
its per-nonce copy + index + queries inside the same 16 GB.

Every failure path lands in the same place: no index, no nonces computed, a
wasted precommit. That is entirely the benchmarker's loss - nothing is
submitted, so there is nothing to verify and no consensus impact.

### These caps are not security controls

Nothing in consensus re-runs the build. A benchmarker can ignore the wall-clock
and build for an hour; the gain is a better index and more nonces per hour, the
same class of advantage as owning a faster GPU, which TIG already tolerates. The
caps exist to keep submitted algorithms runnable on standard benchmarker
hardware and to keep algorithm ranking comparable across operators. Anyone
reading this spec looking for a guarantee that the build was bounded will not
find one; the honest fix for that is verified re-execution, which D2 defers.

## Blast radius

Because the split is host-side and `kernels.cu` is untouched, existing PTX
already exports `gan_sample_latents` and `gan_linear` with the signatures the new
generation path calls. Combined with D5's unchanged `Challenge` layout, **no
existing algorithm needs rebuilding**: old algorithms keep running index-free,
new ones opt in by exporting the two symbols. Trap #1 from the migration notes
(`build_ptx` baking the challenge's `.cu` into every algorithm's PTX) bites only
on kernel changes. *This is an inference from the code and must be verified by
running an unmodified algorithm against the split runtime before it is relied
on.*

What does change:

- `calc_db_seed` in two files that must not drift: `tig-structs/src/core.rs` and
  `tig-benchmarker/common/structs.py`. There is exactly one Python copy -
  `slave/common` and `master/common` are both symlinks to `../common` - so the
  drift risk is Rust-vs-Python only, not three-way.
- `tig-runtime`: `--build-index` mode, batched mode, watchdog, balloon.
- `tig-benchmarker/slave/main.py`: one build invocation before the nonce loop.
- `tig-protocol`: `build_fuel_budget` validation beside the existing
  `fuel_budget` check; `alpha` and `max_build_fuel_budget` in `ChallengeConfig`.
- **Cross-repo:** `pentest-harness` (tig-pentesting) has a path dependency on
  `tig-challenges` and calls `generate_instance` and `evaluate_solution`
  directly. The `Seeds` change breaks it the same way the `evaluate_solution`
  audit-salt change did, and needs the same treatment: paired plans on both
  sides, an explicit interface-contract table, and a unification step that is
  the first build of the two trees together.
- **Not** `golden_vectors.json`: it pins `forward_cpu` against PyTorch
  (`generator.rs:114`, test `cpu_forward_matches_pytorch_golden_vectors`) and has
  nothing to do with instance vectors. Generated database and query values do
  change, but no test in this tree pins them - the GPU tests build instances from
  an arbitrary `[seed_byte; 32]` via `gpu_instance`, so they are agnostic to the
  seed split.

## Testing

Each test names the mutation it catches. A test that catches nothing does not
belong here.

| Test | Mutation it catches |
|---|---|
| `db_seed` identical across two nonces while `calc_seed` differs | forgetting to drop the nonce - the design silently reverts to per-nonce databases and every claim above evaporates |
| Rust and Python `calc_db_seed` agree on a committed golden vector | runtime and benchmarker drifting apart; a database mismatch would surface only as mass invalid solutions |
| split generation reproduces the monolithic pass for the same `(seed, index_base)` | an off-by-one in `index_base`, which shifts every query vector by one latent |
| nonce *k*'s `fuel_consumed` is independent of nonce *k-1* in batched mode | a missing `initialize_kernel` between nonces - `gbl_FUELUSAGE` is a device global, so fuel would accumulate across a bundle and exhaust the budget mid-batch with no error naming the cause |
| a `build_index` allocating `cap + 1` bytes returns `Err` | a balloon sized from the wrong baseline, i.e. a memory cap that does not cap |
| `free < cap` at balloon time errors rather than proceeding | the zero-sized-balloon path, which silently disables the cap |
| build-fuel exhaustion leaves no index file | a partial or corrupt blob being consumed by the query process |
| `--build-index` with a nonce argument is rejected at parse time | the anti-gaming property of D3 being defeated by an operator flag |
| `--num-nonces 0` is rejected | a batch that silently produces no output and reports success |
| `build_index` present, `load_index` absent fails at build time | discovering the mismatch only after paying for a 10-minute build |
| c005 and c006 produce byte-identical output before and after the `Seeds` change | the shared dispatch arm silently feeding `seeds.db` to a challenge that should only ever see `seeds.nonce` |
| no `--index` reproduces today's single-nonce output byte-for-byte | regression in the fallback path that every existing algorithm uses |
| `build_fuel_budget` computation overflows only in `u128`, and config load rejects `max_build_fuel_budget * 20 >= u64::MAX` | integer overflow producing a tiny patched fuel limit, i.e. a build that traps immediately for reasons no message explains |

## Open questions

1. **The 0.05 s/nonce estimate is unmeasured and everything rests on it.** The
   break-even table and the value of `alpha` both follow from it. Measure
   post-split per-nonce time on `tig-gpu` before fixing `alpha`.
2. **Does `lifespan_period` tolerate a build inserted between precommit
   confirmation and the first nonce?** The build cannot start until `rand_hash`
   is known, so it is serial latency ahead of every precommit.
3. **Verify the no-rebuild claim** by running an unmodified c004 algorithm
   against the split runtime.
4. **Throughput knock-on.** If per-nonce time falls ~50-100x, a benchmarker's
   nonce rate rises correspondingly. `per_nonce_fee`, `max_qualifiers_per_track`
   (100) and the qualifier dynamics were calibrated against today's rate and may
   need revisiting. Out of scope here, but it is a consequence of this change
   and should not surprise anyone later.
