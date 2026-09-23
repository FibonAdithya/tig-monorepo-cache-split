# c004: Splitting Index Building from Query Search

**Date:** 2026-08-31
**Status:** Implemented on `vector_search/gan_instance_gen`, with one gap.
**Branch:** `vector_search/gan_instance_gen`

> **Before configuring this: `tig-runtime batch` has zero callers.** The
> reference slave still runs one process per nonce, and every constant in the
> measurement note assumes it does not. See
> "The slave does not use `batch`, and the calibration assumes it does" below.
> `build_fuel_alpha` must not be set in protocol config until that is resolved.

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
of 358 MB costs **<= 3.894 ms measured** (alloc + `clone_dtod`, on an RTX 3060,
against a 22.7 ms marginal nonce - so <= ~17 % of a nonce, itself an upper
bound) and 358 MB of extra device memory, and buys a completely unbroken ABI.

> **Corrected 2026-08-31.** This decision originally priced the copy at "~1.4 ms
> at T4 bandwidth (~3% of an estimated 50 ms nonce)". Both halves were wrong in
> the same, optimistic direction: 1.4 ms counts one direction only at full
> advertised bandwidth, and the 50 ms nonce was an estimate that measurement
> replaced with 22.7 ms. See
> `docs/measurements/2026-08-31-c004-post-split-nonce-time.md` 3.4.

> **The justification above is VOIDED by measurement (2026-08-31, Validation
> below). The layout claim survives; the reason for making it does not.**
>
> "All 105 c004 algorithms rebuilt and resubmitted" was the cost this copy was
> bought to avoid. Part 1 of the Validation *measured* that they must be rebuilt
> anyway - directly for all 88 of the 105 that have a runnable mainnet binary -
> because the GAN `kernels.cu` rewrite already on this branch changed the kernel
> names the generation path calls. So the unbroken ABI is
> unbroken only for algorithms **already rebuilt against this branch**, and this
> spec's own Blast radius section records that there are currently **zero** of
> those. The copy is being paid to preserve compatibility with a population that
> is empty.
>
> **This decision should be revisited.** It is a design call with real tradeoffs
> and it belongs to the user, not to this document. The alternative - borrowing
> the database rather than copying it - would need a lifetime parameter on
> `Challenge` (`Challenge<'a>` holding `&'a CudaSlice<f32>`), which is itself a
> further ABI change on top of the one the GAN work already forces. Set against
> that: ~358 MB of device memory per nonce and <= 3.894 ms of copy (measured;
> the "~1.4 ms" this sentence used to quote was a bandwidth calculation, not a
> measurement).
>
> **Sized against the deployment target, not the test rig.** The target is the
> weakest listed `ComputeType` - `AWS_G4dn`, a T4 with **16 GB** - as the
> Enforcement section sizes it, where the query process must hold the database,
> its per-nonce copy, the index and the queries together. 358 MB is ~2.2% of
> that 16 GB, against ~0.94 GB of fixed costs. So **memory pressure is a weak
> argument for revisiting this decision, and it should not be leaned on**: the
> real cost of the copy is the ABI commitment it locks in, not the bytes. (An
> earlier draft of this note priced the copy against "a card where the 12 GB
> budget is already the binding constraint". That was the RTX 3060 the Validation
> below was measured on, not the deployment target, and 12 GB appears nowhere
> else in this document. Corrected.)
>
> Since every algorithm has to be rebuilt regardless, there is an argument that
> making both ABI changes in one rebuild is cheaper than making them in two - but
> only if the decision is taken *before* the rebuild happens. That is offered as
> a reason to look at the question now, not as a costed claim; nothing here
> measures the cost of a second resubmission round.
>
> Nothing downstream is blocked by this and no code was changed on account of
> it: the implementation is correct and reviewed, and D5's layout claim is
> measured to hold. What is recorded here is that the *rationale* no longer
> follows from the evidence.

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

> **`max_build_fuel_budget` set to 1.1e12 on 2026-09-23**, superseding the
> 7.0e12 of `docs/measurements/2026-08-31-c004-post-split-nonce-time.md` §5.5,
> which was derived from the watchdog (600 s of fuel) rather than from any
> index. The new value comes from metering real builds on every track
> (`docs/measurements/2026-09-23-c004-index-build-fuel.md`): a cuVS-default
> IVF-Flat build costs 0.40x-0.68x of it and the graph builds measured at most
> 0.46x. It is the same for every track because the key is per challenge. The
> `alpha` term of the formula above is unchanged and still unset; at 1.1e12 the
> cap binds from `alpha * N * fuel_budget` >= 1.1e12, i.e. N >= 74 nonces at
> `alpha` = 0.003 and `fuel_budget` = 5e12, so under that `alpha` the cap is
> what nearly every precommit gets.

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
             d2d copy database into Challenge       <= 3.894 ms (measured)
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
already separate the streams - keeping it means **`kernels.cu` is not touched by
this design**, which `scenarios.rs` documents as the thing that avoids forcing a
network-wide rebuild of the generation path.

**Stale as a benefit claim, corrected 2026-08-31 (Validation below):** *this
design* touches no kernel, and that is measured - `kernels.cu` is byte-identical
across `dbb30e99..be650a7c`. But the network-wide rebuild it is described as
avoiding is **already forced** by the GAN `kernels.cu` rewrite earlier on this
branch. Retaining `index_base` therefore avoids *adding* a second rebuild to one
already owed; it does not avoid a rebuild. The same correction applies wherever
this document treats "no kernel change" as implying "no resubmission".

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
tig-runtime batch <SETTINGS> <RAND_HASH> <BINARY> --ptx P \
            --start-nonce A --num-nonces M [--index F] --output D
```

(As implemented, the batched form is a `batch` **subcommand**, not bare flags on
the root command; an earlier draft of this synopsis showed the latter. The
legacy four-positional form is unchanged, which is what
`subcommand_negates_reqs` preserves.)

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
| Device memory (**2 GiB**, was "8 GB provisional") | balloon: after generating the database, allocate `free - (cap + 64 MiB)` and hold it | `cudaMalloc` fails inside `build_index`, which returns `Err` |
| CPU fuel during `load_index` | none - see the note under this table | n/a |

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
weights, plus context overhead. The query process holds database + its per-nonce
copy + index + queries inside the same 16 GB.

**The cap is 2 GiB, not the 8 GB this section provisionally named.** Three
things forced the change, and only the first is about headroom:

1. **8 GiB makes the build refuse itself.** `balloon_size` errors out when
   `free < cap + 64 MiB` - deliberately, because a cap that cannot be enforced
   is worse than no build. The reference slave never passes `--memory-cap`, and
   `process_batch` runs concurrently with `num_workers` solve threads that are
   already holding device memory. On a 12 GB card under load, `free` after
   `Database::generate` is routinely below 8.06 GiB, so the default itself is
   the failure.
2. **The measured build-phase peak is 1,010 MiB**, whole-card, including the
   CUDA context and the 352 MiB database
   (`docs/measurements/2026-08-31-c004-post-split-nonce-time.md` 6.2). 2 GiB is
   ~2x that, and the cap is what the *algorithm* may allocate **on top of** the
   context and database, because the balloon is sized from `free` measured
   after `Database::generate`.
3. 8 GiB was never derived from anything. It was a provisional number in this
   document; 2 GiB is derived from a measurement, which is the only change in
   kind.

What 2 GiB does **not** settle: `nullstub`'s `build_index` allocates nothing, so
what a real 700,000-vector ANN index costs to build is still unmeasured, and no
such algorithm exists on this branch to measure. If one turns out to need more,
the number to raise is this default, and the flag exists precisely so an
operator can. There is also a tension the balloon design does not resolve: a
*lower* cap means a *larger* balloon, so the build phase holds more of the card
away from any concurrent solve thread. Lowering the cap converts "the build
refuses to start" into "the build starves its neighbours", which is a better
failure but not a good one.

### `load_index` and the fuel meters

`load_index` runs above the nonce loop, before the loop's first
`initialize_kernel`. So:

- **Device fuel: free.** `initialize_kernel` is what zeroes `gbl_FUELUSAGE`, and
  it has not run yet. This is the same free phase `generate_instance` already
  occupies - see "A free phase before the meter already exists" above.
- **CPU fuel: metered, but charged to nothing scored.** `compute_solution`
  primes `__fuel_remaining` to `max_fuel` before `$make_db` runs, and then
  *resets* it to `max_fuel` at the top of every nonce. So instrumented CPU code
  inside `load_index` decrements a real meter and will exit 87 if it exhausts
  `max_fuel`, but whatever it spends is discarded before nonce 0 is solved.
  Priming it is not optional: without it `load_index` decrements whatever the
  `.so`'s static initialiser left, and a zero exits 87 on the first
  instrumented instruction with nothing naming the cause.

Neither is exploitable - `load_index` receives only the `Database` and the blob,
never a query - but both are stated because the table above previously said
nothing about either.

### Per-nonce fuel independence is a property of the runtime's counters only

The Testing table's row "nonce *k*'s `fuel_consumed` is independent of nonce
*k-1* in batched mode" is true, and the code enforces it: `initialize_kernel`
zeroes the device counter and `__fuel_remaining` is reset, both at the top of
every nonce. But it is a statement about **the runtime's counters**, not about
**the algorithm's state**.

In `batch` mode the algorithm's `static`s persist across the whole bundle - by
design; that is where D6 puts the index handle. Nothing stops an algorithm from
spending nonce 0's entire `max_fuel` building state that nonces 1..N then read
for free. At the recommended constants that back door is **~40x more generous
than the build budget it routes around**: nonce 0's `max_fuel` is
`fuel_budget` (up to 5e12), while `alpha * N * fuel_budget` capped at
`max_build_fuel_budget` = 7.0e12 is what a legitimate build gets for the whole
precommit. (The cap was reset to **1.1e12** on 2026-09-23, see the note at the
end of "Why not a flat 10 minutes"; a smaller cap widens the gap between the
back door and the budget, which strengthens rather than changes this paragraph.)

This is **inside** the stated model, not a hole in it: D2 makes fuel a spend
limit rather than a score term, and "these caps are not security controls"
already says nothing in consensus re-runs the build. It is recorded because the
per-nonce-independence row reads, on its own, like a stronger guarantee than it
is. A benchmarker using this back door pays for it in nonce 0's wall-clock and
gets no protocol credit for the build; the only thing it defeats is the *shape*
of D7's proportional rule, and it defeats it by ignoring the build phase
entirely rather than by exceeding it.

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

## The slave does not use `batch`, and the calibration assumes it does

**Read this before setting `build_fuel_alpha` in any config.**

`tig-runtime batch` is implemented, tested, and has **zero callers**.
`tig-benchmarker/slave/main.py` wires the build (`run_build_index`, once per
batch) and passes `--index` to each nonce, but `process_nonces` still pulls one
nonce off a queue and calls `run_tig_runtime`, which spawns one
`docker exec ... tig-runtime <settings> <rand_hash> <nonce> <so>` process **per
nonce** - the legacy form. `NUM_WORKERS` of those run concurrently. Nothing in
this repository, outside `tig-runtime`'s own tests, passes `--start-nonce`.

That is deliberate and it is a gap in the *plan*, not a defect in the
implementation: Task 10's brief asked only for the build call plus "pass
`--index` to each `run_tig_runtime` call". Rewiring the slave's solve path is a
behaviour change to **every** challenge, not just c004, and it needs integration
testing that the session which found this could not do. It is left undone on
purpose.

**What it costs.** Every constant in §5 of
`docs/measurements/2026-08-31-c004-post-split-nonce-time.md` is derived from
`t_new` = 0.378 s/nonce, which amortises per-process startup (2.078 s, of which
2.005 s is `CudaContext::new`) plus `Database::generate` (0.766 s) over
`batch_size` = 8 nonces **in one process**. Under the slave as shipped, the
applicable row of that same table is `batch_size` = 1:

| | `batch_size` 8 (what the constants assume) | `batch_size` 1 (what the slave does) |
|---|---|---|
| `t_new` | 0.378 s/nonce | **2.866 s/nonce** |
| `delta` = `t_old` - `t_new` | 2.861 s/nonce | **0.373 s/nonce** - **7.7x smaller** |
| break-even for a flat 600 s build | 210 nonces | **~1,610 nonces** |
| net-win bound `alpha <= delta * R / F` (`R_low`) | 8.79e-3 | **1.145e-3** |
| recommended `alpha` (bound / 2 / 1.25 T4) | 3.52e-3 -> **0.003** | 4.58e-4 -> **0.0004** |

So **`alpha` = 0.003 is ~2.6x above the bare net-win bound at `batch_size` = 1**
(2.0x at `R_measured`). Concretely, at the 80-nonce protocol floor it authorises
1.2e12 build fuel = 78 s of build against 30 s of saving - a net **loss** of
48 s. The proportional rule only turns positive once
`max_build_fuel_budget` = 7.0e12 has clamped it, at roughly 1,200 nonces
(`R_low`) or 900 (`R_measured`).

**The rule this section exists to state: `build_fuel_alpha` must not be set in
protocol config until either the slave is wired to `batch`, or the value is
re-derived for `batch_size` = 1.** Today the key is inert - `ChallengeConfig`
parses and validates it, but `calc_build_fuel_budget` is explicitly dead code
with no caller, and the slave receives `build_fuel_budget` from the master
rather than computing it. The moment the master is wired, a `build_fuel_alpha`
of 0.003 against a per-nonce slave makes the split **net-negative** at every
precommit size below ~1,000 nonces, and nothing anywhere reports that. Wiring
the slave to `batch` is the better of the two fixes, because it is also what
makes the rest of this document's cost model true.

## Blast radius

**Corrected 2026-08-31 by measurement — see Validation below. The original wording
("no existing algorithm needs rebuilding") was false as written.** The accurate
statement is two-part:

- **The split adds no *algorithm-rebuild* blast radius of its own.** It is
  host-side, `kernels.cu` is byte-identical across `dbb30e99..be650a7c`, and
  D5's `Challenge` layout is unchanged. Both halves are now measured: an
  algorithm binary built before the split runs unmodified against the split
  runtime and reads every `Challenge` field correctly. **That scoping is
  deliberate.** This design plainly has blast radius of other kinds - every item
  in the "What does change" list below is real work, including a `calc_db_seed`
  that must not drift between Rust and Python, new `tig-protocol` config, a new
  build step in the benchmarker slave, a cross-repo `pentest-harness` break, and
  a changed database instance. What was measured is narrower than "no blast
  radius": it is that **no algorithm has to be rebuilt on account of this
  design.**
- **But the GAN `kernels.cu` rewrite already on this branch requires every
  mainnet c004 algorithm (105 listed) to be rebuilt and resubmitted, and that
  cost was already owed before this design existed.** Mainnet PTX exports
  `generate_clusters` / `generate_vectors`; this branch's generation path calls
  `gan_sample_latents`, `gan_linear` and `recall_audit`. Measured across every
  mainnet c004 binary that exists (88 of the 105 listed algorithms have a
  downloadable, non-empty artifact): **88/88 export the pre-GAN pair, 0/88 export
  any GAN kernel.** A mainnet algorithm dies `CUDA_ERROR_NOT_FOUND "named symbol
  not found"` against this branch — and dies identically against the commit
  immediately *before* this design's first commit.

So "old algorithms keep running index-free, new ones opt in by exporting
`build_index`/`load_index`" holds only among algorithms already rebuilt against
the GAN branch. Trap #1 from the migration notes (`build_ptx` baking the
challenge's `.cu` into every algorithm's PTX) bites only on kernel changes — the
split makes none, the GAN work made many.

What does change:

- `calc_db_seed` in two files that must not drift: `tig-structs/src/core.rs` and
  `tig-benchmarker/common/structs.py`. There is exactly one Python copy -
  `slave/common` and `master/common` are both symlinks to `../common` - so the
  drift risk is Rust-vs-Python only, not three-way.
- `tig-runtime`: `--build-index` mode, batched mode, watchdog, balloon.
- `tig-benchmarker/slave/main.py`: one build invocation before the nonce loop.
  **Shipped state: the build invocation and `--index` are wired; the batched
  query process is NOT.** See "The slave does not use `batch`" below - this is
  the one place where the implementation and this document's cost model do not
  meet.
- `tig-protocol`: `build_fuel_budget` validation beside the existing
  `fuel_budget` check; `alpha` and `max_build_fuel_budget` in `ChallengeConfig`.
- **Operational, and it is atomic: `tig-runtime` and `tig-verifier` must be
  upgraded together.** They are two halves of one instance derivation. An old
  verifier paired with a new runtime regenerates the *pre-split* database - one
  derived from the per-nonce seed rather than from `calc_db_seed` - and
  therefore rejects **every honest c004 solution** the new runtime produces. The
  reverse pairing fails the same way for the same reason. Neither failure names
  its cause: the verifier reports an invalid solution, which is
  indistinguishable from a genuinely wrong answer, and the operator sees a
  benchmarker that has silently stopped earning. Deploy both binaries in one
  step, or take c004 out of the selection while they are mismatched. This is not
  a c004-only concern for a mixed fleet: a slave running the old pair and a
  slave running the new pair are computing different instances for the same
  precommit.
- **Cross-repo:** `pentest-harness` (tig-pentesting) has a path dependency on
  `tig-challenges` and calls `generate_instance` and `evaluate_solution`
  directly. The `Seeds` change breaks it the same way the `evaluate_solution`
  audit-salt change did, and needs the same treatment: paired plans on both
  sides, an explicit interface-contract table, and a unification step that is
  the first build of the two trees together.
- **Not** `golden_vectors.json`: it pins `forward_cpu` against PyTorch
  (`generator.rs:114`, test `cpu_forward_matches_pytorch_golden_vectors`) and has
  nothing to do with instance vectors. Generated **database** values do change;
  **query values do not** - corrected 2026-08-31, and measured, not reasoned:
  `for_nonce` generates queries from `seeds.nonce`, which is `calc_seed`
  unchanged by this design, at the same `index_base = database_size` as before,
  so query vectors are **bit-identical across the split**. The Validation section
  below shows the same query bytes out of both the pre-split and post-split
  runtimes, and the database bytes differing, on the same nonce. Either way no
  test in this tree pins these values - the GPU tests build instances from an
  arbitrary `[seed_byte; 32]` via `gpu_instance`, so they are agnostic to the
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
3. ~~**Verify the no-rebuild claim** by running an unmodified c004 algorithm
   against the split runtime.~~ **ANSWERED 2026-08-31 — see Blast radius and
   Validation above, both corrected in place; read them rather than this
   summary.** Three things, and the scoping on the first is load-bearing:
   (a) the split adds no ***algorithm-rebuild*** blast radius of its own — it
   does have blast radius of other kinds, every item in Blast radius' "What does
   change" list, so this is not "no blast radius";
   (b) D5's `Challenge` layout claim is measured to hold, but its *rationale* is
   voided — see the note under D5, since what it protects is a population of
   zero;
   (c) the GAN `kernels.cu` rewrite already on this branch does require the
   mainnet c004 algorithms to be rebuilt and resubmitted (measured: 88 of the
   105 listed algorithms have a runnable binary; 88/88 export the pre-GAN
   kernels, 0/88 export any GAN kernel), and that cost was already owed before
   this design existed.
   **Not closed by this:** the branch-level blocker in Validation — no
   competitive algorithm has been shown to clear the 0.95 recall bar.
4. **Throughput knock-on.** If per-nonce time falls ~50-100x, a benchmarker's
   nonce rate rises correspondingly. `per_nonce_fee`, `max_qualifiers_per_track`
   (100) and the qualifier dynamics were calibrated against today's rate and may
   need revisiting. Out of scope here, but it is a consequence of this change
   and should not surprise anyone later.

## Validation

Measured 2026-08-31 on `tig-gpu` (RTX 3060, sm_86, CUDA 12.6.3) against branch HEAD
`be650a7cc528655e19e576eeb089d0850ca6d328`.

**This section is self-contained on purpose.** An earlier draft pointed at
`.superpowers/sdd/2026-08-31-c004-index-build-split-monorepo/task-7-report.md` for the raw logs.
That pointer dangles for anyone reading the committed repository: `.gitignore:34` ignores
`.superpowers/`, and `git ls-files .superpowers/` returns nothing, so the working notes are not
part of the git record and never will be. Every load-bearing command and its raw output is
therefore inlined below. The `.superpowers` report still exists in the working tree and carries the
longer narrative, the environment provenance and the list of defects found in the task brief, but
nothing here depends on it.

**Which binaries were run.** Everything ran inside the `tig-dev-vector_search` container under a
`gpu-claim`. The image's baked `/usr/local/bin/tig-runtime` (sha256 `79d6d510…`, dated
2026-08-27) is **stale** and was never invoked; every runtime was built in-container with
`cargo build -r -p tig-runtime -p tig-verifier --features vector_search` in an isolated `/tmp`
clone and invoked by absolute path, and every run script re-printed the sha256 of the binary it was
about to execute, so each raw block below certifies its own provenance.

### The claim under test

The Blast radius section above asserted that "existing PTX already exports
`gan_sample_latents` and `gan_linear`" and therefore **no existing algorithm needs
rebuilding**, flagged there as an inference to be verified. It has two halves, and both have now
been run:

- **Part 1 — the PTX half.** Does an unmodified mainnet algorithm's PTX carry the kernels the
  generation path calls? **Measured, and refuted for mainnet — but the split is not what refutes
  it.**
- **Part 2 — the D5 half.** Does an algorithm binary built before the split read `Challenge`'s
  fields at the right offsets under the split runtime? **Measured, and it holds.**

The two interact, and not comfortably: Part 2 confirms the layout is preserved, while Part 1
measures that the population the preservation was for is empty. **D5's layout claim survives its
own rationale.** See the voided-rationale note under D5.

### Part 1 result: the sentence is false for mainnet, and the split is not what makes it false

`there_v10` (`c004_a100`, merged @ round 124) was downloaded prebuilt via
`scripts/download_algorithm` and never rebuilt. Its PTX exports the *pre-GAN* kernel set:

```
$ grep -oE '\.visible \.entry [A-Za-z0-9_]+' there_v10.ptx | sed 's/.* //' | sort -u
evaluate_total_distance   finalize_kernel   generate_clusters   generate_vectors
initialize_kernel   there_v10_reduce_gemm_tile_f32_full_first   ... (algorithm-private kernels)
$ grep -c "gan_sample_latents\|gan_linear" there_v10.ptx
0
```

Three runs, each inside the `tig-dev-vector_search` container under a `gpu-claim`, each against a
runtime built *in that container* (never the image's stale baked `/usr/local/bin/tig-runtime`,
sha256 `79d6d510…`, which was captured for contrast and never invoked). Every run script printed the
sha256 of the binary it was about to execute.

| # | Runtime tree | Commit | `tig-runtime` sha256 | `track_id` | `--fuel` | Result | Exit |
|---|---|---|---|---|---|---|---|
| A | post-split (this design) | `be650a7c` | `2180fe7f…` | `s=sift_128` | `2000000000` | `DriverError(CUDA_ERROR_NOT_FOUND, "named symbol not found")`, no output file | 84 |
| B | **pre-split**, post-GAN | `dbb30e99` | `216af0f9…` | `s=sift_128` | `2000000000` | `DriverError(CUDA_ERROR_NOT_FOUND, "named symbol not found")`, no output file | 84 |
| C1 | **pre-GAN** | `3efbdec9` | `7dabf8f7…` | `n_queries=7000` | `2000000000` | `DriverError(TIG_ERROR_OUT_OF_FUEL, "ran out of fuel")`, no output file | 84 |
| C2 | **pre-GAN** | `3efbdec9` | `7dabf8f7…` | `n_queries=7000` | `5000000000000` | solution written in 3.8 s; `quality: 72174` | 0 |

**C1 is not a failed attempt to be skipped over - it is what removes the fuel confound.** A and B
ran at `--fuel 2000000000`, and if C had only ever been run at 5e12 a reader could reasonably ask
whether A and B died of a budget that was simply too small. C1 answers that: **at the same 2e9
budget as A and B**, on the pre-GAN runtime, the run got far enough to exhaust its fuel, which means
PTX loading and every `module.load_function` had already succeeded. `OUT_OF_FUEL` and
`CUDA_ERROR_NOT_FOUND` are different failures at different stages. C2 then re-ran at the live c004
`max_fuel_budget` of 5e12 (recorded at line 42 above) and completed.

**Where the isolation actually comes from.** It comes from **A vs B**, which is a genuinely
single-variable comparison: same `.so`, same `.ptx`, same settings file, same `rand_hash`, same
nonce, same `--fuel`, same container, same card - only the runtime commit differs. C is *not*
single-variable against A: it differs in runtime commit, in `track_id` (`n_queries=7000` vs
`s=sift_128`, because `Track` itself changed shape), therefore in the seed and the whole generated
instance, and in C2 also in fuel. C's job is narrower and it should not be read as more: it shows
the algorithm binary, the harness, the container, the claim and the invocation form are all sound,
so A and B's failure is not an artifact of how they were run.

Exact commands, per run - the settings are passed as a **file**, because the inline-JSON form in
the task brief loses its inner quotes to `bash -c` and dies `Failed to parse settings`:

```bash
# Built in-container, in an isolated /tmp clone, not /workspace/tig-bench.
# Run once per tree: /tmp/task7/{head,presplit,pregan}
cargo build -r -p tig-runtime -p tig-verifier --features vector_search

# settings.json         {"player_id":"audit","block_id":"audit","challenge_id":"c004",
#                        "algorithm_id":"c004_a100","track_id":"s=sift_128"}
# settings-pregan.json  ... "track_id":"n_queries=7000"

# --- A: post-split runtime ---
docker run --rm --gpus all -v /tmp/task7/head:/app -v /tmp/task7/algo:/algo \
  -v /tmp/task7/out-head:/out -w /app -e CHALLENGE=vector_search \
  -e RUSTUP_TOOLCHAIN=nightly-2025-02-10 -e CUDA_VISIBLE_DEVICES=0 \
  tig-dev-vector_search bash -c '
    sha256sum /app/target/release/tig-runtime
    /app/target/release/tig-runtime /algo/settings.json norebuildcheck 0 \
      /algo/there_v10.so --ptx /algo/there_v10.ptx --fuel 2000000000 --output /out'
2180fe7fe01e430c74c6e00c0cb692ad590818fd6ab5fd04e61c0cc62e43bbfe  /app/target/release/tig-runtime
Runtime Error: DriverError(CUDA_ERROR_NOT_FOUND, "named symbol not found")
INNER_EXIT=84

# --- B: pre-split runtime. Identical invocation, /tmp/task7/presplit mounted at /app ---
216af0f968eaf4deae5b734e35d6fc480c71ed37700e81524cf252ebc06d1e26  /app/target/release/tig-runtime
Runtime Error: DriverError(CUDA_ERROR_NOT_FOUND, "named symbol not found")
INNER_EXIT=84

# --- C1: pre-GAN runtime, SAME 2e9 fuel as A and B, settings-pregan.json ---
7dabf8f76745f1a3cfe6280aadcf33dd1ce71adf38add0b8506dee75b5fd9fe3  /app/target/release/tig-runtime
/app/target/release/tig-runtime /algo/settings-pregan.json norebuildcheck 0 \
  /algo/there_v10.so --ptx /algo/there_v10.ptx --fuel 2000000000 --output /out
Runtime Error: DriverError(TIG_ERROR_OUT_OF_FUEL, "ran out of fuel")
INNER_EXIT=84

# --- C2: same, at the live max_fuel_budget ---
/app/target/release/tig-runtime /algo/settings-pregan.json norebuildcheck 0 \
  /algo/there_v10.so --ptx /algo/there_v10.ptx --fuel 5000000000000 --output /out
real    0m3.813s
INNER_EXIT=0
-rw-r--r-- 1 root root 31472 Aug 31 15:00 /out/0.json

# --- C2 verified ---
/app/target/release/tig-verifier /algo/settings-pregan.json norebuildcheck 0 /out/0.json \
  --ptx /algo/there_v10.ptx
quality: 72174
VERIFY_EXIT=0
```

Reading the exit codes: **84 in A and B is the PTX trap, not a reap and not the teardown crash.**
Both failed in about a second (nowhere near the ~60 s reaper window, `gpu-claim --status` clean),
and both left `/out` **empty** — whereas the teardown crash exits 137/139 *after* writing the
output. No 137 or 139 occurred anywhere in this validation; C exited 0.

`quality: 72174` in C is on the **retired pre-GAN mean-distance quality map** and is not comparable
to the 950000 recall bar, which exists only on this branch. There is no quality figure for the
split runtime, because no run against it produced a solution. `min_recall` was confirmed at the
shipped 0.95 in every tree used.

### The population claim, measured rather than asserted

The `there_v10` grep above is n=1. Since this whole task exists to separate inference from
measurement, the claim about the *population* was measured too: every c004 algorithm on mainnet at
round 132 (block `03dca1bf547aa92e1bab9ea3d7d702e7`) was enumerated, its binary downloaded, and its
PTX scanned for `.visible .entry` symbols.

```
TOTAL c004 algorithms listed: 105
PTX successfully scanned:      88

  exporting BOTH gan_sample_latents and gan_linear  : 0
  exporting EITHER gan kernel                       : 0
  exporting recall_audit                            : 0
  exporting BOTH generate_clusters/generate_vectors : 88
  exporting evaluate_total_distance                 : 88

NO_BINARY (15): c004_a003 greedyvector, a004 hnsw_opt, a005 prodquant, a006 gpuvector,
                a007 hnsw_alt, a008 hnsw_comp, a009 search, a010 lsearch, a011 notgreedy,
                a012 nearest, a013 prodq, a031 tabus, a032 simulate, a039 vector_church,
                a080 stat_filter_sigma
ERROR    (2):  c004_a041 vs_test_wasm, c004_a067 is_floatfour  (both: empty tarball)
```

**88 of 88 scannable binaries export the pre-GAN kernel set and zero export any GAN kernel.** The
17 not scanned are not a gap in the argument: 15 have no binary record at all and 2 serve an empty
tarball, so none of them has a runnable artifact on mainnet in the first place. The accurate
statement is therefore: **every mainnet c004 algorithm that has a runnable binary exports
`generate_clusters`/`generate_vectors` and none exports `gan_sample_latents`, `gan_linear` or
`recall_audit`.**

### What Part 1 establishes

1. An unmodified mainnet c004 algorithm **cannot** run against the split runtime — measured
   directly for `there_v10`, and by PTX scan for all 88 mainnet c004 binaries that exist.
2. It cannot run against the **pre-split** runtime either — run B is bit-for-bit the same failure at
   the commit immediately before this design's first commit (`43ed7cd0`). **The index-build split
   adds no *algorithm-rebuild* blast radius over the branch it sits on.** That scoping is
   deliberate and is the only thing A-vs-B measured: the split plainly *does* have blast radius of
   other kinds, all of it listed in the section above — `calc_db_seed` must stay in sync across
   Rust and Python, `tig-protocol` gains config, the benchmarker slave gains a build step, the
   cross-repo `pentest-harness` breaks, and the generated database instance itself changes. None of
   those require an algorithm rebuild; all of them are real work.
3. C is a **soundness** control, not a matched control. It shows the `.so`, the `.ptx`, the
   container, the claim and the invocation form all work, so A and B's failure is not an artifact
   of the harness. It is *not* a matched comparison against A and B: C runs a different runtime
   commit, a different `track_id` (hence a different seed and a different generated instance), and
   in C2 a different fuel budget. The single-variable comparison is A vs B.

The Blast radius section's original "no existing algorithm needs rebuilding" must therefore be read
as scoped to *an algorithm already rebuilt against the GAN branch* — of which there are currently
zero anywhere. Reworded honestly: **the split imposes no algorithm rebuild beyond the one the GAN
work already imposes.** That section has been corrected in place.

Part 1 did **not** touch D5. Execution died inside `Database::generate` → `generate_vectors` →
`module.load_function("gan_sample_latents")`, which the runtime reaches **before** it ever calls the
algorithm's `entry_point` with a `&Challenge` (see the `gpu_db` dispatch arm,
`tig-runtime/src/main.rs`). So Part 2 was run as a separate experiment.

### Part 2 result: D5 holds — measured, not inferred

**Why this needed measuring at all.** D5 is what makes `Challenge::for_nonce` take an owned
device-to-device copy of ~358 MB rather than a cheap borrow, and the stated justification for that
cost is "the layout must not change or every existing algorithm reads its fields at the wrong
offsets." If D5 were wrong the plan would have bought an expensive copy for nothing *and* broken
the network. (Part 1 has since shown the justification does not hold for a different reason — the
population it protects is empty — but that is an argument for revisiting the decision, not for
leaving its technical claim untested.)

`kernels.cu` is **byte-identical** across `dbb30e99..be650a7c`
(`git diff dbb30e99 be650a7c -- tig-challenges/src/vector_search/kernels.cu` outputs nothing), so
PTX compatibility is not a variable here and **the `Challenge` layout is the only thing under
test.**

**The probe.** `d5probe`, a purpose-built algorithm compiled **at `dbb30e99`** — post-GAN,
pre-split, the commit immediately before this design's first commit — with `build_ptx` +
`build_so` in the container. It exports `entry_point` and `help` only; it deliberately has no
`build_index`/`load_index`, so it is exactly the shape of a pre-split algorithm. Its
`solve_challenge` reads **every** field of `Challenge`, prints each one, does a real
device-to-host copy of both trailing `CudaSlice` fields, and encodes the scalars into the saved
`Solution`. A stub that merely exits 0 would prove nothing; this one cannot pass without actually
dereferencing the struct.

Build, in the container, at `dbb30e99`:

```
$ python3 tig-binary/scripts/build_ptx d5probe
kernel: initialize_kernel,        #blocks: 1,   status: SKIPPED
kernel: finalize_kernel,          #blocks: 1,   status: SKIPPED
kernel: gan_sample_latents,       #blocks: 31,  status: SKIPPED
kernel: evaluate_total_distance,  #blocks: 15,  status: SKIPPED
kernel: gan_linear,               #blocks: 147, status: SKIPPED
kernel: recall_audit,             #blocks: 353, status: SKIPPED
Wrote ptx to tig-algorithms/lib/vector_search/ptx/d5probe.ptx

$ bash tig-binary/scripts/build_so d5probe
Linking into shared library 'tig-algorithms/lib/vector_search/amd64/d5probe.so'
Done

$ nm -D --defined-only d5probe.so | grep -E 'entry_point|help|__fuel_remaining|__runtime_signature'
0000000000908030 D __fuel_remaining
0000000000908038 D __runtime_signature
0000000000045bf0 T entry_point
0000000000047200 T help

$ sha256sum d5probe.so d5probe.ptx
68b643ced745cd64ee59568e58653928008d76da508b93f5b5385814fbb67d06  d5probe.so
98936af1564a6cb734fddfbe87f3cb341bd49cbe2493d1ec287565d6b6165fab  d5probe.ptx
```

That unmodified `.so`/`.ptx` was then run against **two** runtimes with identical settings,
`rand_hash`, nonce and `--fuel 5000000000000` — the `dbb30e99` runtime, where the probe's
`tig-challenges` and the runtime's are the same copy and the offsets are correct by construction,
and the `be650a7c` split runtime, which is the test. Raw stdout of both, verbatim:

```
##### D5-A: runtime /tmp/task7/presplit  (dbb30e99) #####
216af0f968eaf4deae5b734e35d6fc480c71ed37700e81524cf252ebc06d1e26  /app/target/release/tig-runtime
68b643ced745cd64ee59568e58653928008d76da508b93f5b5385814fbb67d06  /algo/d5probe.so
98936af1564a6cb734fddfbe87f3cb341bd49cbe2493d1ec287565d6b6165fab  /algo/d5probe.ptx
D5PROBE scenario=sift_128
D5PROBE num_queries=7000
D5PROBE vector_dims=128
D5PROBE database_size=700000
D5PROBE seed=a22085613ccf2c029517e8b68f3d008f77acd3cc0f4df4995e11366da94f7e27
D5PROBE qslice_len=896000
D5PROBE dbslice_len=89600000
D5PROBE qv_len=896000
D5PROBE qv_head8=3f112c3e,3d5c1f3c,3db7c65e,3dc85ebe,bb6a20c8,3c02a7aa,3b683b16,3efeb50c
D5PROBE qv_tail4=3e5db0cc,3cd5715d,3cb0a28d,3d46c139
D5PROBE db_len=89600000
D5PROBE db_head8=3d5a85a9,3cb7af31,3cb01224,bbb21b4b,3e5bd249,3e47e741,3c0e42c8,3c8cdad9
real    0m3.923s
INNER_EXIT=0
{"cpu_arch":"amd64","fuel_consumed":3315,"nonce":0,"runtime_signature":13667326539545663817,
 "solution":"\"H4sIAAAAAAAA/y3HMQ0AQAhD0VtYLkEBJnADI1IQRNBISOny+ulhLjCvoz9OMVSjZQBwH3cvMAAAAA==\""}

##### D5-B: runtime /tmp/task7/head  (be650a7c, THE SPLIT RUNTIME) #####
2180fe7fe01e430c74c6e00c0cb692ad590818fd6ab5fd04e61c0cc62e43bbfe  /app/target/release/tig-runtime
68b643ced745cd64ee59568e58653928008d76da508b93f5b5385814fbb67d06  /algo/d5probe.so
98936af1564a6cb734fddfbe87f3cb341bd49cbe2493d1ec287565d6b6165fab  /algo/d5probe.ptx
D5PROBE scenario=sift_128
D5PROBE num_queries=7000
D5PROBE vector_dims=128
D5PROBE database_size=700000
D5PROBE seed=a22085613ccf2c029517e8b68f3d008f77acd3cc0f4df4995e11366da94f7e27
D5PROBE qslice_len=896000
D5PROBE dbslice_len=89600000
D5PROBE qv_len=896000
D5PROBE qv_head8=3f112c3e,3d5c1f3c,3db7c65e,3dc85ebe,bb6a20c8,3c02a7aa,3b683b16,3efeb50c
D5PROBE qv_tail4=3e5db0cc,3cd5715d,3cb0a28d,3d46c139
D5PROBE db_len=89600000
D5PROBE db_head8=3c1ee431,3def408d,3d726fb3,3b1d4856,3d45e27b,3d5b170e,3d91f2bd,3c2e9bce
real    0m3.992s
INNER_EXIT=0
{"cpu_arch":"amd64","fuel_consumed":3315,"nonce":0,"runtime_signature":13667326539545663817,
 "solution":"\"H4sIAAAAAAAA/y3HMQ0AQAhD0VtYLkEBJnADI1IQRNBISOny+ulhLjCvoz9OMVSjZQBwH3cvMAAAAA==\""}
```

Side by side:

| Field read across the FFI boundary | `dbb30e99` runtime (`216af0f9…`) | `be650a7c` split runtime (`2180fe7f…`) | |
|---|---|---|---|
| `scenario` | `sift_128` | `sift_128` | = |
| `num_queries` | `7000` | `7000` | = |
| `vector_dims` | `128` | `128` | = |
| `database_size` | `700000` | `700000` | = |
| `seed` | `a22085613ccf2c029517e8b68f3d008f77acd3cc0f4df4995e11366da94f7e27` | identical | = |
| `d_query_vectors.len()` | `896000` | `896000` | = |
| `d_database_vectors.len()` | `89600000` | `89600000` | = |
| query vectors `[0..8]`, raw f32 bits | `3f112c3e,3d5c1f3c,3db7c65e,3dc85ebe,bb6a20c8,3c02a7aa,3b683b16,3efeb50c` | identical | = |
| query vectors, last 4, raw bits | `3e5db0cc,3cd5715d,3cb0a28d,3d46c139` | identical | = |
| database vectors `[0..8]`, raw bits | `3d5a85a9,3cb7af31,3cb01224,bbb21b4b,…` | `3c1ee431,3def408d,3d726fb3,3b1d4856,…` | **differs — by design** |

Both runs exited **0** in ~3.9 s, both wrote a solution, and both produced byte-identical output
files (`fuel_consumed: 3315`, the same `runtime_signature`, the same solution blob). The solution
decodes to `indexes = [7000, 128, 700000, 896000, 89600000]`, which matches
`ScenarioConfig::from(Scenario::SIFT_128)` at `be650a7c` (`n_queries: 7_000`, `vector_dims: 128`,
`database_size: 700_000`) exactly.

**Why a layout change was a priori very unlikely, which bears on how much this proves.**
`Challenge` is declared with **no `#[repr]` attribute**, so it is `repr(Rust)`: its field order and
padding are rustc's choice, not the declaration order, and reasoning about "field order shifting"
is not the right model. What rustc guarantees is that the layout is a deterministic function of the
type list and the compiler version. Both are pinned here:

- the type list is field-for-field identical between `dbb30e99` and `be650a7c` (shown above);
- **`Cargo.lock`'s `cudarc` entry is byte-identical** across the two commits - version `0.16.4`,
  git rev `b3fccf5003c6e356bdd36e6808bf6e66f08f98d2` - so `CudaSlice<f32>`, the only non-primitive
  field type, is *literally the same type*, not merely a same-named one. The `tig-challenges`
  `Cargo.lock` entry is likewise unchanged. (For precision: `Cargo.lock` as a whole is **not**
  unchanged across the split - exactly one line differs, `libc` added to **`tig-runtime`**'s
  dependency list for the watchdog/balloon work. Nothing on the algorithm side of the boundary
  moved.)
- both `.so` and both runtimes were built with the same pinned `nightly-2025-02-10`.

With an identical type list *and* an identical dependency pin *and* the same compiler, a layout
difference would have required a rustc bug. **So this experiment is confirmatory, not
discriminating:** it establishes that the thing did not happen, which is worth having written down
against a decision this expensive, but it was never likely to fail.

**What the database difference does and does not prove.** The database vectors differ, and that is
precisely what D1/D3 intend: post-split the database is generated from the nonce-free `db_seed`
while queries still come from `calc_seed`, which is unchanged. That difference is genuinely
valuable - it proves the probe is reading **live device data through the struct**, not printing
constants, not returning a cached buffer, and not silently no-op'ing. But it is a *data* mutation,
not a *layout* mutation, so it does not make the experiment discriminating about layout. The
control that would have done that is one I did not run: rebuild the probe against a deliberately
mutated `Challenge` - a reordered or inserted field - and show that it then faults or reads
garbage. Without that, "the probe would have caught a layout change" is itself an inference.

**Verdict: D5 holds.** An algorithm binary built before the split reads `Challenge`'s fields
correctly under the split runtime, including dereferencing both `CudaSlice` fields for 3.6 MB and
358 MB. Open question 3 above is **closed** - though see the voided-rationale note under D5, since
what this result now protects is a population of zero.

**Coverage this run picked up for free, worth recording because none of it was separately tested.**
The probe crossing the boundary successfully also exercised, on the split runtime, against a
binary built before the split:

- `entry_point`'s **signature and calling convention** - six arguments including a `&dyn Fn`
  trait object and two `Arc<...>` - resolved and called correctly;
- the **`save_solution` callback** invoked from inside the `.so` back into the runtime, and
  `Solution`'s base64/gzip/bincode round-trip;
- the **`__fuel_remaining` and `__runtime_signature` globals**, which the runtime reads out of the
  `.so` by symbol: `fuel_consumed: 3315` and `runtime_signature: 13667326539545663817` came out
  **identical** from both runtimes, so the fuel and signature accounting path is unbroken across
  the split as well.

None of that was the target of the experiment; all of it would have shown up as a failure or a
mismatch had the split disturbed it.

### BRANCH-LEVEL BLOCKER — not a Task 7 residual, do not retire this with this task

**No competitive c004 algorithm has ever been shown to clear the 0.95 recall bar on this branch,
and none can be run until it is rebuilt against the GAN `kernels.cu`.** This is recorded here as a
**branch-level** open item rather than a Task 7 note precisely so that closing Task 7 does not
close it. It is out of scope for this design - it belongs to the GAN instance-generation work
underneath - but it is the difference between *"the split works"* and *"the challenge is
solvable"*, and only the first has been demonstrated.

The two are independent: everything Part 1 and Part 2 establish would remain true on a branch whose
recall bar no algorithm could reach. `scenarios.rs` concedes the gap in its own comment on
`min_recall` - "the upper end is not measured - no ANN method was run - so '0.95 is not trivially
cleared by an approximate method' is judgement, not measurement". The only end-to-end quality figure
anywhere in this validation is `quality: 72174`, and that is from a **pre-GAN** runtime on the
retired mean-distance map; it says nothing about recall. Whoever ships this branch owes a
measurement of a real ANN algorithm against the 950000 bar before the rebuild-and-resubmit is asked
of the c004 algorithm authors.

### What else remains unestablished

- **`Database`'s layout is a brand-new ABI commitment with no baseline to appeal to.** D5 was
  verified on one algorithm shape (`entry_point` only, no index). An algorithm exporting
  `build_index`/`load_index` also crosses the boundary with `&Database`, and `Database` is
  `repr(Rust)` and **new in this design** - there is no "unchanged from today" argument available
  for it, because there is no previous version of it. Its layout is therefore **frozen from the
  moment the first algorithm ships against it**, and any later field added, removed or retyped is
  a silent ABI break with exactly the failure mode D5 exists to prevent. No pre-split binary can
  exercise it; Task 4's stub work is the only coverage. This deserves the same explicit
  "layout is fixed" discipline D5 gives `Challenge`.
- The measurement was taken on one card (RTX 3060, sm_86). Struct layout is ABI-determined and
  architecture-independent for this target, so this is noted for completeness rather than as a real
  doubt.
- The discriminating control described above - a probe rebuilt against a deliberately mutated
  `Challenge`, shown to fault - was not run.
