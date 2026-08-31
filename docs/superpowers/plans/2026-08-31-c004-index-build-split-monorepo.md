# c004 Index Build / Query Search Split — Monorepo Side

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make c004 index construction free within a bounded budget so that only query search is charged, by fixing the database per precommit and building the index in a process that is never given a nonce.

**Architecture:** `BenchmarkSettings` gains `calc_db_seed`, which drops the nonce. The three GPU challenges take a `Seeds { nonce, db }` struct instead of a bare `[u8; 32]`. c004's generation splits into `Database::generate` (the `(false, database_size)` pass) and `Challenge::for_nonce` (the `(true, n_queries)` pass plus a device-to-device copy of the database, which keeps `Challenge`'s layout byte-identical and therefore the algorithm ABI unbroken). `tig-runtime` gains a `--build-index` mode that never accepts a nonce, and a batched mode that loads the index once and loops a bundle's nonces, resetting the fuel meter between each.

**Tech Stack:** Rust (edition per workspace, toolchain `nightly-2025-02-10`), CUDA 12.6 via the tig-foundation `cudarc` fork, CUDA C (`kernels.cu` — **not modified**), Python 3 (`tig-benchmarker`), `cargo test`, `unittest`.

**Spec:** `docs/superpowers/specs/2026-08-31-c004-index-build-split-design.md`

## Global Constraints

- **Never push.** TIG is a live competition. Local commits on `vector_search/gan_instance_gen` only; no `git push`, no PR, no outward-facing git operation without asking.
- **Never `git add -A` / `git add .` / `git commit -a`.** Stage explicit paths. Run `git status --short` before every commit.
- **Do not modify `tig-challenges/src/vector_search/kernels.cu`.** `scenarios.rs` documents why: `build_ptx` compiles the challenge's `.cu` into *every algorithm's* PTX, so a kernel change forces a network-wide rebuild-and-resubmit of all 105 c004 algorithms. Every change in this plan is host-side.
- **`vector_search` cannot compile locally.** There is no CUDA toolkit on this machine. Any task marked **[GPU]** must run on the `tig-gpu` host via the `gpu-jobs` skill, holding a `gpu-claim` or the reaper will SIGKILL it. Build with `nightly-2025-02-10` and CUDA 12.6 — cudarc rejects CUDA 13, and a mismatched Rust toolchain makes working algorithms fail spuriously.
- **Python tests cannot run locally.** `blake3` is not installed and there is no `pip` and no `ensurepip` (verified: `python3 -m pip` and `python3 -m venv` both fail). Run them in the master image:
  ```bash
  docker build -t tig-bench-py -f tig-benchmarker/master/Dockerfile tig-benchmarker
  docker run --rm -v "$(pwd)/tig-benchmarker:/src" -w /src tig-bench-py python -m unittest tests.data -v
  ```
- **Tasks marked [local] are verified locally.** `cargo test -p tig-structs` and `cargo build -p tig-runtime --features c001` both work here (verified).
- **Provisional constants.** `alpha = 0.25`, `memory_cap = 8 GiB`, `build watchdog = 600 s`. Task 8 measures and fixes them; Task 9 must not be started until Task 8 reports.
- **Exit codes.** 84 = runtime error (existing), 87 = out of fuel (existing), **85 = build watchdog timeout** (new, introduced in Task 5).

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `tig-structs/src/core.rs` | `calc_db_seed` beside `calc_seed` | 1 |
| `tig-structs/tests/core.rs` | Rust golden vector, must match Python | 1 |
| `tig-benchmarker/common/structs.py` | Python `calc_db_seed` (single copy; `slave/common` and `master/common` are symlinks to `../common`) | 1 |
| `tig-benchmarker/tests/data.py` | Python golden vector, must match Rust | 1 |
| `tig-challenges/src/lib.rs` | `Seeds` struct — ungated, no cudarc needed | 2 |
| `tig-challenges/src/{hypergraph,neuralnet_optimizer}/mod.rs` | take `&Seeds`, read `.nonce`, ignore `.db` | 2 |
| `tig-challenges/src/vector_search/mod.rs` | `Database`, `Challenge::for_nonce`, shared `generate_vectors` helper | 2, 3 |
| `tig-runtime/src/main.rs` | `seeds_for`, `--build-index` mode, watchdog, balloon, batched mode | 2, 4, 5, 6 |
| `tig-verifier/src/main.rs` | build `Seeds` from both derivations | 2 |
| `tig-structs/src/config.rs` | `build_fuel_alpha`, `max_build_fuel_budget` in `ChallengeConfig` | 9 |
| `tig-protocol/src/contracts/benchmarks.rs` | validate `build_fuel_budget` | 9 |
| `tig-benchmarker/slave/main.py` | one build invocation before the nonce loop | 10 |

---

## Task 1: `calc_db_seed` in Rust and Python **[local]**

**Files:**
- Modify: `tig-structs/src/core.rs:170-173`
- Modify: `tig-structs/tests/core.rs` (append two tests; extend the `use` line)
- Modify: `tig-benchmarker/common/structs.py:48-49`
- Modify: `tig-benchmarker/tests/data.py` (append one test inside `class TestData`)

**Interfaces:**
- Consumes: nothing.
- Produces: `BenchmarkSettings::calc_db_seed(&self, rand_hash: &String) -> [u8; 32]` (Rust) and `BenchmarkSettings.calc_db_seed(self, rand_hash: str) -> bytes` (Python). Every later task depends on the Rust one.

- [ ] **Step 1: Write the failing Rust tests**

Change the first line of `tig-structs/tests/core.rs` imports from
`use tig_utils::MerkleHash;` to `use tig_utils::{jsonify, u8s_from_str, MerkleHash};`,
then append:

```rust
#[test]
fn test_calc_db_seed() {
    let settings = BenchmarkSettings {
        player_id: "some_player".to_string(),
        block_id: "some_block".to_string(),
        challenge_id: "some_challenge".to_string(),
        algorithm_id: "some_algorithm".to_string(),
        track_id: "a=1,b=2".to_string(),
    };

    let rand_hash = "random_hash".to_string();

    // Assert same as Python version: tig-benchmarker/tests/data.py
    assert_eq!(
        settings.calc_db_seed(&rand_hash),
        [
            209, 209, 150, 41, 179, 131, 168, 223, 27, 59, 221, 124, 237, 86, 161, 52, 118, 79,
            8, 0, 171, 205, 118, 2, 64, 244, 59, 240, 176, 44, 51, 185
        ]
    );
}

#[test]
fn test_db_seed_carries_its_domain_tag() {
    // The mutation this catches is dropping the `_db` suffix, which would make
    // the database seed the plain hash of "{settings}_{rand_hash}". That string
    // is one a future format change could collide with; the tag makes the two
    // derivations unrelated by construction.
    let settings = BenchmarkSettings {
        player_id: "some_player".to_string(),
        block_id: "some_block".to_string(),
        challenge_id: "some_challenge".to_string(),
        algorithm_id: "some_algorithm".to_string(),
        track_id: "a=1,b=2".to_string(),
    };
    let rand_hash = "random_hash".to_string();

    let untagged = u8s_from_str(&format!("{}_{}", jsonify(&settings), rand_hash));
    assert_ne!(
        settings.calc_db_seed(&rand_hash),
        untagged,
        "calc_db_seed lost its `_db` domain tag"
    );
}
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cargo test -p tig-structs --test core
```
Expected: FAIL, `no method named 'calc_db_seed' found for struct 'BenchmarkSettings'`.

- [ ] **Step 3: Implement `calc_db_seed`**

In `tig-structs/src/core.rs`, inside `impl BenchmarkSettings`, directly below `calc_seed`:

```rust
    /// The seed for the part of an instance that is constant across a
    /// precommit. Takes no nonce, deliberately: `tig-runtime --build-index`
    /// derives its instance from this and is never given a nonce, so a build
    /// process cannot compute any query set. That is the property the whole
    /// index-build design rests on.
    pub fn calc_db_seed(&self, rand_hash: &String) -> [u8; 32] {
        u8s_from_str(&format!("{}_{}_db", jsonify(&self), rand_hash))
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cargo test -p tig-structs --test core
```
Expected: PASS, 5 tests.

- [ ] **Step 5: Write the failing Python test**

Append inside `class TestData` in `tig-benchmarker/tests/data.py`:

```python
    def test_calc_db_seed(self):
        settings = BenchmarkSettings(
            player_id="some_player",
            block_id="some_block",
            challenge_id="some_challenge",
            algorithm_id="some_algorithm",
            track_id="a=1,b=2"
        )

        rand_hash = "random_hash"

        # Assert same as Rust version: tig-structs/tests/core.rs
        expected = bytes([
            209, 209, 150, 41, 179, 131, 168, 223, 27, 59, 221, 124, 237, 86, 161, 52, 118, 79,
            8, 0, 171, 205, 118, 2, 64, 244, 59, 240, 176, 44, 51, 185
        ])
        self.assertEqual(settings.calc_db_seed(rand_hash), expected)
```

- [ ] **Step 6: Run the Python test to verify it fails**

```bash
docker build -t tig-bench-py -f tig-benchmarker/master/Dockerfile tig-benchmarker
docker run --rm -v "$(pwd)/tig-benchmarker:/src" -w /src tig-bench-py python -m unittest tests.data -v
```
Expected: FAIL, `AttributeError: 'BenchmarkSettings' object has no attribute 'calc_db_seed'`.

- [ ] **Step 7: Implement the Python side**

In `tig-benchmarker/common/structs.py`, inside `class BenchmarkSettings`, below `calc_seed`:

```python
    def calc_db_seed(self, rand_hash: str) -> bytes:
        return u8s_from_str(f"{jsonify(self)}_{rand_hash}_db")
```

- [ ] **Step 8: Run the Python test to verify it passes**

```bash
docker run --rm -v "$(pwd)/tig-benchmarker:/src" -w /src tig-bench-py python -m unittest tests.data -v
```
Expected: PASS, 4 tests.

- [ ] **Step 9: Commit**

```bash
git status --short
git add tig-structs/src/core.rs tig-structs/tests/core.rs \
        tig-benchmarker/common/structs.py tig-benchmarker/tests/data.py
git commit -m "feat(c004): derive a database seed that carries no nonce"
```

---

## Task 2: `Seeds` through the shared GPU dispatch arm **[GPU]**

The `gpu` arm of `dispatch_challenge!` is shared by c004, c005 and c006 in both the runtime and the verifier. c004 needs two seeds, so the shared signature takes a struct and the other two ignore the extra field — the same shape as the existing `_audit_salt` parameter they already ignore.

**Files:**
- Modify: `tig-challenges/src/lib.rs` (add `Seeds`)
- Modify: `tig-challenges/src/vector_search/mod.rs:79-84` (signature only; the split is Task 3)
- Modify: `tig-challenges/src/hypergraph/mod.rs:53-59`
- Modify: `tig-challenges/src/neuralnet_optimizer/mod.rs:124-130`
- Modify: `tig-runtime/src/main.rs:88-90, 213-219` (add `seeds_for`, use it)
- Modify: `tig-verifier/src/main.rs:151, 244-250`
- Modify: `tig-challenges/src/vector_search/mod.rs:792` (the `gpu_instance` test helper)
- Test: `tig-runtime/src/main.rs` (new `#[cfg(test)] mod tests`)

**Interfaces:**
- Consumes: `BenchmarkSettings::calc_db_seed` (Task 1).
- Produces:
  - `tig_challenges::Seeds { pub nonce: [u8; 32], pub db: [u8; 32] }`
  - `Challenge::generate_instance(seeds: &Seeds, track: &Track, module: Arc<CudaModule>, stream: Arc<CudaStream>, prop: &cudaDeviceProp) -> Result<Self>` for c004, c005 and c006
  - `fn seeds_for(settings: &BenchmarkSettings, rand_hash: &String, nonce: u64) -> Seeds` in `tig-runtime/src/main.rs`

- [ ] **Step 1: Write the failing test for `seeds_for`**

The highest-value mutation here is the two seeds being swapped when the dispatch macro constructs `Seeds` — that silently restores a per-nonce database and quietly destroys the entire design, with no error anywhere. Extracting the construction into a function makes it testable without a GPU.

Append to `tig-runtime/src/main.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn test_settings() -> BenchmarkSettings {
        BenchmarkSettings {
            player_id: "some_player".to_string(),
            block_id: "some_block".to_string(),
            challenge_id: "some_challenge".to_string(),
            algorithm_id: "some_algorithm".to_string(),
            track_id: "a=1,b=2".to_string(),
        }
    }

    #[test]
    fn seeds_for_puts_each_derivation_in_the_right_field() {
        // Catches the swap. If `nonce` and `db` are exchanged, the database
        // becomes per-nonce again and every claim the index-build design makes
        // is false -- with nothing failing and no message naming the cause.
        let settings = test_settings();
        let rand_hash = "random_hash".to_string();

        let seeds = seeds_for(&settings, &rand_hash, 1337);
        assert_eq!(seeds.nonce, settings.calc_seed(&rand_hash, 1337));
        assert_eq!(seeds.db, settings.calc_db_seed(&rand_hash));
    }

    #[test]
    fn seeds_for_holds_the_database_constant_across_nonces() {
        let settings = test_settings();
        let rand_hash = "random_hash".to_string();

        let a = seeds_for(&settings, &rand_hash, 0);
        let b = seeds_for(&settings, &rand_hash, u64::MAX);
        assert_eq!(a.db, b.db, "database seed must not vary with the nonce");
        assert_ne!(a.nonce, b.nonce, "query seed must vary with the nonce");
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cargo test -p tig-runtime --features c001
```
Expected: FAIL, `cannot find function 'seeds_for' in this scope`.

- [ ] **Step 3: Add `Seeds` and `seeds_for`**

In `tig-challenges/src/lib.rs`, near the top (ungated — it must compile without `cudarc`):

```rust
/// The two seeds a GPU challenge instance is derived from.
///
/// `nonce` is the per-nonce seed every challenge has always used. `db` is
/// derived without the nonce and is therefore constant across a precommit.
/// Only c004 reads `db`, to generate a database an index can be built over once
/// and reused by every nonce; c005 and c006 ignore it, exactly as they already
/// take `_audit_salt` and ignore it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Seeds {
    pub nonce: [u8; 32],
    pub db: [u8; 32],
}
```

In `tig-runtime/src/main.rs`, above `compute_solution`:

```rust
fn seeds_for(settings: &BenchmarkSettings, rand_hash: &String, nonce: u64) -> Seeds {
    Seeds {
        nonce: settings.calc_seed(rand_hash, nonce),
        db: settings.calc_db_seed(rand_hash),
    }
}
```

`tig_challenges::*` is already glob-imported by `tig-runtime`, so `Seeds` resolves without a new `use`.

- [ ] **Step 4: Run the test to verify it passes**

```bash
cargo test -p tig-runtime --features c001
```
Expected: PASS, 2 tests.

- [ ] **Step 5: Change the three `generate_instance` signatures**

In each of `hypergraph/mod.rs` and `neuralnet_optimizer/mod.rs`, change
`seed: &[u8; 32],` to `seeds: &Seeds,` and insert as the first line of the body:

```rust
        // c005/c006 are not split across a precommit; `seeds.db` is not theirs
        // to read.
        let seed = &seeds.nonce;
```

In `vector_search/mod.rs` do the same. Task 3 replaces this line with the real split; here the change is signature-only so the two concerns stay reviewable apart.

- [ ] **Step 6: Update the call sites**

`tig-runtime/src/main.rs` — replace `let seed = settings.calc_seed(&rand_hash, nonce);` with:

```rust
    let seeds = seeds_for(&settings, &rand_hash, nonce);
    let seed = seeds.nonce;
```

(`seed` stays in scope because the CPU arm and the `__runtime_signature` initialisation both still use it.) In the `gpu` arm, change `&seed,` in the `generate_instance` call to `&seeds,`.

`tig-verifier/src/main.rs` — replace `let seed = settings.calc_seed(&rand_hash, nonce);` with:

```rust
    let seeds = Seeds {
        nonce: settings.calc_seed(&rand_hash, nonce),
        db: settings.calc_db_seed(&rand_hash),
    };
    let seed = seeds.nonce;
```

and change `&seed,` to `&seeds,` in the `gpu` arm's `generate_instance` call. Add `use tig_challenges::Seeds;` if the glob import does not already cover it.

`tig-challenges/src/vector_search/mod.rs:792` — in `gpu_instance`, replace `&[seed_byte; 32],` with:

```rust
            &Seeds {
                nonce: [seed_byte; 32],
                db: [seed_byte; 32],
            },
```

- [ ] **Step 7: Build all three GPU challenges on tig-gpu**

Submit through the `gpu-jobs` skill, holding a `gpu-claim`:

```bash
cargo build -p tig-runtime --features c004
cargo build -p tig-runtime --features c005
cargo build -p tig-runtime --features c006
cargo build -p tig-verifier --features c004
```
Expected: all four succeed.

- [ ] **Step 8: Prove c005 and c006 output is unchanged**

The spec requires this and it cannot be a unit test: neither `hypergraph/mod.rs`
nor `neuralnet_optimizer/mod.rs` has a test module, and adding GPU test
infrastructure to two challenges this change does not otherwise touch is not
worth it. Diff their runtime output across the change instead. The mutation it
catches is the shared dispatch arm feeding `seeds.db` to a challenge that should
only ever see `seeds.nonce`.

Do this **without `git stash`** — the stash stack is shared with other worktrees
and other sessions. Capture the "before" output first, from a build made at the
commit this task started from:

```bash
# BEFORE: at the parent commit, in a scratch worktree
git worktree add /tmp/c004-before HEAD
cargo build --manifest-path /tmp/c004-before/Cargo.toml -p tig-runtime --features c005
/tmp/c004-before/target/debug/tig-runtime "$C005_SETTINGS" "$RAND_HASH" 0 \
    c005_algo.so --ptx c005.ptx --fuel 2000000000 --output /tmp/c005_before
```

Then, after this task's changes:

```bash
cargo build -p tig-runtime --features c005
./target/debug/tig-runtime "$C005_SETTINGS" "$RAND_HASH" 0 \
    c005_algo.so --ptx c005.ptx --fuel 2000000000 --output /tmp/c005_after
diff /tmp/c005_before/0.json /tmp/c005_after/0.json && echo "c005 IDENTICAL"
```

Repeat for c006 with `--features c006`. Expected: both print `IDENTICAL`.

Then clean up: `git worktree remove /tmp/c004-before`.

- [ ] **Step 8b: Run the existing c004 GPU test suite as a regression check**

```bash
cargo test -p tig-challenges --features vector_search 2>&1 | tail -30
```
Expected: PASS, the same test count as before this task. These tests build instances from an arbitrary `[seed_byte; 32]`, so they are agnostic to the seed split and any failure here is a real regression in the signature change.

- [ ] **Step 9: Commit**

```bash
git status --short
git add tig-challenges/src/lib.rs tig-challenges/src/vector_search/mod.rs \
        tig-challenges/src/hypergraph/mod.rs \
        tig-challenges/src/neuralnet_optimizer/mod.rs \
        tig-runtime/src/main.rs tig-verifier/src/main.rs
git commit -m "feat(c004): give GPU challenges both seeds via a Seeds struct"
```

---

## Task 3: Split c004 generation into database and queries **[GPU]**

**Files:**
- Modify: `tig-challenges/src/vector_search/mod.rs:79-256` (extract the shared pass, add `Database`, add `Challenge::for_nonce`)
- Test: `tig-challenges/src/vector_search/mod.rs` (append to the existing `mod tests`)

**Interfaces:**
- Consumes: `Seeds` (Task 2).
- Produces:
  - `pub struct Database { pub scenario: Scenario, pub vector_dims: u32, pub database_size: u32, pub d_database_vectors: CudaSlice<f32> }`
  - `Database::generate(db_seed: &[u8; 32], track: &Track, module: Arc<CudaModule>, stream: Arc<CudaStream>, prop: &cudaDeviceProp) -> Result<Database>`
  - `Challenge::for_nonce(db: &Database, seeds: &Seeds, track: &Track, module: Arc<CudaModule>, stream: Arc<CudaStream>, prop: &cudaDeviceProp) -> Result<Challenge>`
  - `Challenge::generate_instance` retained as a thin wrapper over those two, with its Task 2 signature.
- **`Challenge`'s fields do not change.** Task 4 and 6 depend on that; so does every existing algorithm `.so`.

- [ ] **Step 1: Write the failing tests**

Append to `mod tests` in `vector_search/mod.rs`:

```rust
    #[test]
    fn the_database_is_identical_across_nonces_and_the_queries_are_not() {
        // This is the design in one assertion. If it fails, an index built once
        // per precommit is worthless because every nonce sees different rows.
        let ptx = Ptx::from_file(test_ptx_path().clone());
        let ctx = CudaContext::new(0).unwrap();
        ctx.set_blocking_synchronize().unwrap();
        let module = ctx.load_module(ptx).unwrap();
        let stream = ctx.default_stream();
        let prop = get_device_prop(0).unwrap();
        let track = Track { s: Scenario::SIFT_128 };

        let db_seed = [7u8; 32];
        let a = Challenge::generate_instance(
            &Seeds { nonce: [1u8; 32], db: db_seed },
            &track, module.clone(), stream.clone(), &prop,
        ).unwrap();
        let b = Challenge::generate_instance(
            &Seeds { nonce: [2u8; 32], db: db_seed },
            &track, module.clone(), stream.clone(), &prop,
        ).unwrap();

        // Compare a prefix rather than 358 MB twice: a seed change perturbs
        // every row, so the first 4,096 floats are as decisive as all of them
        // and the test stays fast enough to keep.
        let head = |c: &Challenge, which: u8| -> Vec<f32> {
            let src = if which == 0 { &c.d_database_vectors } else { &c.d_query_vectors };
            stream.memcpy_dtov(&src.slice(0..4096)).unwrap()
        };

        assert_eq!(head(&a, 0), head(&b, 0), "database must not depend on the nonce seed");
        assert_ne!(head(&a, 1), head(&b, 1), "queries must depend on the nonce seed");
    }

    #[test]
    fn for_nonce_reproduces_what_generate_instance_produces() {
        // Catches the two halves drifting apart -- e.g. `for_nonce` forgetting
        // the `index_base` offset, which would shift every query vector by one
        // latent and silently change the instance.
        let ptx = Ptx::from_file(test_ptx_path().clone());
        let ctx = CudaContext::new(0).unwrap();
        ctx.set_blocking_synchronize().unwrap();
        let module = ctx.load_module(ptx).unwrap();
        let stream = ctx.default_stream();
        let prop = get_device_prop(0).unwrap();
        let track = Track { s: Scenario::SIFT_128 };
        let seeds = Seeds { nonce: [3u8; 32], db: [9u8; 32] };

        let whole = Challenge::generate_instance(
            &seeds, &track, module.clone(), stream.clone(), &prop,
        ).unwrap();

        let db = Database::generate(
            &seeds.db, &track, module.clone(), stream.clone(), &prop,
        ).unwrap();
        let split = Challenge::for_nonce(
            &db, &seeds, &track, module.clone(), stream.clone(), &prop,
        ).unwrap();

        assert_eq!(
            stream.memcpy_dtov(&whole.d_query_vectors.slice(0..4096)).unwrap(),
            stream.memcpy_dtov(&split.d_query_vectors.slice(0..4096)).unwrap(),
        );
        assert_eq!(
            stream.memcpy_dtov(&whole.d_database_vectors.slice(0..4096)).unwrap(),
            stream.memcpy_dtov(&split.d_database_vectors.slice(0..4096)).unwrap(),
        );
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

On tig-gpu:
```bash
cargo test -p tig-challenges --features vector_search the_database_is_identical
```
Expected: FAIL, `cannot find struct 'Database'`.

- [ ] **Step 3: Extract the shared generation pass**

In `vector_search/mod.rs`, lift the body of the `for (dest_is_query, count)` loop
(currently `mod.rs:125-243`) into a free function. It keeps the existing kernel
launches verbatim — no `kernels.cu` change, no launch-geometry change:

```rust
/// One forward pass of the generator into `dest`.
///
/// `index_base` is preserved from the pre-split generator even though the two
/// halves now use different seeds and no longer need it to separate their
/// latent streams. Keeping it means `gan_sample_latents` is called exactly as
/// before, so `kernels.cu` and every algorithm's PTX are untouched.
fn generate_vectors(
    seed: &[u8; 32],
    count: usize,
    index_base: usize,
    dest: &mut CudaSlice<f32>,
    layers: &[generator::Layer],
    widest: usize,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
) -> Result<()> {
    let sample_latents_kernel = module.load_function("gan_sample_latents")?;
    let linear_kernel = module.load_function("gan_linear")?;

    let d_seed = stream.memcpy_stod(seed)?;
    let mut d_weights = Vec::with_capacity(layers.len());
    for layer in layers {
        d_weights.push((
            stream.memcpy_stod(&layer.weights)?,
            stream.memcpy_stod(&layer.bias)?,
        ));
    }
    let mut d_scratch_a = stream.alloc_zeros::<f32>(FORWARD_CHUNK * widest)?;
    let mut d_scratch_b = stream.alloc_zeros::<f32>(FORWARD_CHUNK * widest)?;
    let mut d_latents = stream.alloc_zeros::<f32>(FORWARD_CHUNK * LATENT_DIM)?;

    for chunk_start in (0..count).step_by(FORWARD_CHUNK) {
        // ... the existing chunk body, verbatim, writing into `dest` ...
    }
    Ok(())
}
```

Move the existing chunk body across unchanged, replacing the `destination`
selection (`if dest_is_query { &mut d_query_vectors } else { &mut d_database_vectors }`)
with `dest`.

- [ ] **Step 4: Add `Database` and `Challenge::for_nonce`**

```rust
pub struct Database {
    pub scenario: Scenario,
    pub vector_dims: u32,
    pub database_size: u32,
    pub d_database_vectors: CudaSlice<f32>,
}

impl Database {
    pub fn generate(
        db_seed: &[u8; 32],
        track: &Track,
        module: Arc<CudaModule>,
        stream: Arc<CudaStream>,
        _prop: &cudaDeviceProp,
    ) -> Result<Self> {
        let config = ScenarioConfig::from(track.s);
        let weights = weights_from(config.weights)?;
        let layers = &weights.layers;
        let vector_dims = layers
            .last()
            .ok_or_else(|| anyhow!("generator has no layers"))?
            .out_dim;
        if vector_dims != config.vector_dims {
            return Err(anyhow!(
                "scenario {} declares {} dims but its blob produces {}",
                track.s, config.vector_dims, vector_dims
            ));
        }
        let widest = layers.iter().map(|l| l.out_dim).max().unwrap();
        let database_size = config.database_size;

        let mut d_database_vectors =
            stream.alloc_zeros::<f32>(database_size as usize * vector_dims)?;
        generate_vectors(
            db_seed, database_size as usize, 0, &mut d_database_vectors,
            layers, widest, module, stream.clone(),
        )?;
        stream.synchronize()?;

        Ok(Self {
            scenario: track.s,
            vector_dims: vector_dims as u32,
            database_size,
            d_database_vectors,
        })
    }
}

impl Challenge {
    pub fn for_nonce(
        db: &Database,
        seeds: &Seeds,
        track: &Track,
        module: Arc<CudaModule>,
        stream: Arc<CudaStream>,
        _prop: &cudaDeviceProp,
    ) -> Result<Self> {
        let config = ScenarioConfig::from(track.s);
        let weights = weights_from(config.weights)?;
        let layers = &weights.layers;
        let widest = layers.iter().map(|l| l.out_dim).max().unwrap();
        let vector_dims = db.vector_dims as usize;
        let n_queries = config.n_queries;

        let mut d_query_vectors =
            stream.alloc_zeros::<f32>(n_queries as usize * vector_dims)?;
        generate_vectors(
            &seeds.nonce, n_queries as usize, db.database_size as usize,
            &mut d_query_vectors, layers, widest, module, stream.clone(),
        )?;

        // Owned copy, not a borrow: `Challenge`'s layout must stay
        // byte-identical or every existing algorithm .so reads these fields at
        // the wrong offsets. ~358 MB at T4 bandwidth is ~1.4 ms.
        let d_database_vectors = stream.clone_dtod(&db.d_database_vectors)?;
        stream.synchronize()?;

        Ok(Self {
            seed: seeds.nonce,
            scenario: db.scenario,
            num_queries: n_queries,
            vector_dims: db.vector_dims,
            database_size: db.database_size,
            d_database_vectors,
            d_query_vectors,
        })
    }
}
```

Then replace the body of `Challenge::generate_instance` with:

```rust
        let db = Database::generate(&seeds.db, track, module.clone(), stream.clone(), _prop)?;
        Self::for_nonce(&db, seeds, track, module, stream, _prop)
```

If `stream.clone_dtod` is unavailable in the pinned cudarc, use the two-line
equivalent: `let mut dst = stream.alloc_zeros::<f32>(len)?; stream.memcpy_dtod(&db.d_database_vectors, &mut dst)?;`

- [ ] **Step 5: Run the tests to verify they pass**

```bash
cargo test -p tig-challenges --features vector_search 2>&1 | tail -30
```
Expected: PASS, previous count + 2.

- [ ] **Step 6: Mutation-check the new tests**

Break the code, confirm the test fails, restore. Both mutations must be run:

1. In `Challenge::for_nonce`, change `&seeds.nonce` to `&seeds.db`.
   Expected: `the_database_is_identical_across_nonces_and_the_queries_are_not` FAILS on the `assert_ne`.
2. In `Challenge::for_nonce`, change `db.database_size as usize` (the `index_base` argument) to `0`.
   Expected: `for_nonce_reproduces_what_generate_instance_produces` FAILS.

Restore both before committing.

- [ ] **Step 7: Commit**

```bash
git status --short
git add tig-challenges/src/vector_search/mod.rs
git commit -m "feat(c004): split instance generation into database and queries"
```

---

## Task 4: `tig-runtime --build-index` **[GPU]**

**Files:**
- Modify: `tig-runtime/src/main.rs` (CLI, new `build_index` function, dispatch)
- Test: `tig-runtime/src/main.rs` (`mod tests`, CLI-level assertions run locally)

**Interfaces:**
- Consumes: `Database::generate` (Task 3), `calc_db_seed` (Task 1).
- Produces: the algorithm ABI other algorithms will be written against —
  - `build_index(&Database, Option<String>, Arc<CudaModule>, Arc<CudaStream>, &cudaDeviceProp) -> Result<Vec<u8>>`, symbol `b"build_index"`, OPTIONAL
  - `load_index(&Database, &[u8], Arc<CudaModule>, Arc<CudaStream>, &cudaDeviceProp) -> Result<()>`, symbol `b"load_index"`, OPTIONAL but required whenever `build_index` is present
  - CLI: `tig-runtime --build-index <SETTINGS> <RAND_HASH> <BINARY> --ptx P --build-fuel N --index-out F`

- [ ] **Step 1: Write the failing CLI tests**

Append to `mod tests` in `tig-runtime/src/main.rs`:

```rust
    #[test]
    fn build_index_mode_refuses_a_nonce() {
        // The anti-gaming property of the design is that the build process has
        // no nonce in scope. An operator flag that smuggles one in defeats it,
        // so the CLI must reject the combination outright rather than ignore it.
        let err = cli()
            .try_get_matches_from(vec![
                "tig-runtime", "--build-index", "{}", "hash", "lib.so",
                "--start-nonce", "0",
            ])
            .unwrap_err();
        assert!(
            err.to_string().contains("--build-index"),
            "error must name the offending flag, got: {}",
            err
        );
    }

    #[test]
    fn batched_mode_refuses_zero_nonces() {
        // A batch that produces nothing and exits 0 is indistinguishable from a
        // batch that worked.
        let err = cli()
            .try_get_matches_from(vec![
                "tig-runtime", "{}", "hash", "lib.so",
                "--start-nonce", "0", "--num-nonces", "0",
            ])
            .unwrap_err();
        assert!(err.to_string().contains("num-nonces"), "got: {}", err);
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cargo test -p tig-runtime --features c001
```
Expected: FAIL, unknown arguments `--build-index` / `--start-nonce`.

- [ ] **Step 3: Extend the CLI**

In `cli()`, change the positional nonce from `arg!(<NONCE> ..)` to
`arg!([NONCE] ..)` so batched and build modes can omit it, and add:

```rust
        .arg(arg!(--"build-index" "Build an index over the precommit's database and exit").action(clap::ArgAction::SetTrue))
        .arg(arg!(--"build-fuel" [FUEL] "Fuel budget for the build phase").value_parser(clap::value_parser!(u64)))
        .arg(arg!(--"index-out" [PATH] "Where to write the index blob").value_parser(clap::value_parser!(PathBuf)))
        .arg(arg!(--index [PATH] "Index blob to load before solving").value_parser(clap::value_parser!(PathBuf)))
        .arg(arg!(--"start-nonce" [N] "First nonce of a batch").value_parser(clap::value_parser!(u64)))
        .arg(arg!(--"num-nonces" [N] "How many nonces to solve").value_parser(clap::value_parser!(u64)))
        .arg(arg!(--"memory-cap" [BYTES] "Device memory the build may use").value_parser(clap::value_parser!(u64)).default_value("8589934592"))
        .arg(arg!(--"build-timeout" [SECS] "Wall-clock watchdog for the build").value_parser(clap::value_parser!(u64)).default_value("600"))
        .group(clap::ArgGroup::new("build_excludes").args(["build-index"]).conflicts_with_all(["start-nonce", "num-nonces", "NONCE", "index"]))
```

and validate `num-nonces` with a range parser that rejects 0:

```rust
        .value_parser(clap::value_parser!(u64).range(1..))
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cargo test -p tig-runtime --features c001
```
Expected: PASS, 4 tests.

- [ ] **Step 5: Implement the build mode**

Add to `tig-runtime/src/main.rs`, gated `#[cfg(feature = "c004")]` — it names
`c004::Database` directly, so `cuda` alone is not enough:

```rust
pub fn build_index(
    settings: String,
    rand_hash: String,
    library_path: PathBuf,
    hyperparameters: Option<String>,
    ptx_path: PathBuf,
    build_fuel: u64,
    memory_cap_bytes: u64,
    timeout_secs: u64,
    index_out: PathBuf,
    gpu_device: Option<usize>,
) -> Result<()> {
    let settings = load_settings(&settings);
    // No nonce is derived here and none can be passed. This is the property the
    // design rests on: a build process cannot compute any query set because the
    // material to derive one is not present.
    let db_seed = settings.calc_db_seed(&rand_hash);
    let hyperparameters = hyperparameters.map(|x| load_hyperparameters(&x));

    let library = load_module(&library_path)?;
    let fuel_remaining_ptr = unsafe { *library.get::<*mut u64>(b"__fuel_remaining")? };
    unsafe { *fuel_remaining_ptr = build_fuel };

    let gpu_fuel_scale = 20u64;
    let ptx_content = std::fs::read_to_string(&ptx_path)?;
    let scaled = build_fuel
        .checked_mul(gpu_fuel_scale)
        .ok_or_else(|| anyhow!("build fuel {} overflows when scaled by {}", build_fuel, gpu_fuel_scale))?;
    let modified_ptx = ptx_content.replace("0xdeadbeefdeadbeef", &format!("0x{:016x}", scaled));

    let gpu_device = gpu_device.unwrap_or(0);
    let ctx = CudaContext::new(gpu_device)?;
    ctx.set_blocking_synchronize()?;
    let module = ctx.load_module(Ptx::from_src(modified_ptx))?;
    let stream = ctx.fuel_check_stream();
    let prop = get_device_prop(gpu_device as i32)?;

    let build_index_fn = unsafe {
        library.get::<fn(&c004::Database, Option<String>, Arc<CudaModule>, Arc<CudaStream>, &cudaDeviceProp) -> Result<Vec<u8>>>(b"build_index")
    }.map_err(|_| anyhow!("algorithm does not export `build_index`; it does not support index building"))?;

    // Checked before the build, not after: discovering the mismatch after
    // paying for a ten-minute build is a waste with no upside.
    unsafe {
        library.get::<fn(&c004::Database, &[u8], Arc<CudaModule>, Arc<CudaStream>, &cudaDeviceProp) -> Result<()>>(b"load_index")
    }.map_err(|_| anyhow!("algorithm exports `build_index` but not `load_index`"))?;

    let track = parse_track(&settings)?;
    let database = c004::Database::generate(&db_seed, &track, module.clone(), stream.clone(), &prop)?;

    let initialize_kernel = module.load_function("initialize_kernel")?;
    unsafe {
        stream.launch_builder(&initialize_kernel)
            .arg(&u64::from_be_bytes(db_seed[8..16].try_into().unwrap()))
            .launch(LaunchConfig { grid_dim: (1,1,1), block_dim: (1,1,1), shared_mem_bytes: 0 })?;
    }

    let blob = build_index_fn(&database, hyperparameters, module.clone(), stream.clone(), &prop)?;

    // Atomic: a watchdog kill must never leave a partial blob for the query
    // process to load.
    let tmp = index_out.with_extension("tmp");
    fs::write(&tmp, &blob)?;
    fs::rename(&tmp, &index_out)?;
    eprintln!("index written: {} bytes", blob.len());
    Ok(())
}
```

Factor the `track_id` parsing already duplicated in `dispatch_challenge!` into
`fn parse_track(settings: &BenchmarkSettings) -> Result<c004::Track>` and call it
from both places.

- [ ] **Step 6: Wire the mode into `main`**

```rust
    if matches.get_flag("build-index") {
        #[cfg(not(feature = "cuda"))]
        panic!("--build-index requires a GPU challenge build");
        #[cfg(feature = "c004")]
        if let Err(e) = build_index(
            matches.get_one::<String>("SETTINGS").unwrap().clone(),
            matches.get_one::<String>("RAND_HASH").unwrap().clone(),
            matches.get_one::<PathBuf>("BINARY").unwrap().clone(),
            matches.get_one("hyperparameters").cloned(),
            matches
                .get_one::<PathBuf>("ptx")
                .cloned()
                .expect("--ptx is required for --build-index"),
            *matches.get_one::<u64>("build-fuel").unwrap(),
            *matches.get_one::<u64>("memory-cap").unwrap(),
            *matches.get_one::<u64>("build-timeout").unwrap(),
            matches.get_one::<PathBuf>("index-out").unwrap().clone(),
            matches.get_one::<usize>("gpu").cloned(),
        ) {
            eprintln!("Runtime Error: {}", e);
            std::process::exit(84);
        }
        return;
    }
```

- [ ] **Step 7: Verify end to end on tig-gpu with a stub algorithm**

Write a throwaway algorithm exporting `build_index` (returns `vec![1,2,3]`) and
`load_index` (returns `Ok(())`), build it, then:

```bash
tig-runtime --build-index "$SETTINGS" "$RAND_HASH" stub.so --ptx stub.ptx \
            --build-fuel 100000000000 --index-out /tmp/idx.blob
test -f /tmp/idx.blob && wc -c /tmp/idx.blob
```
Expected: exit 0, `3 /tmp/idx.blob`, and no `.tmp` file left behind.

Then remove `load_index` from the stub, rebuild, and re-run.
Expected: non-zero exit, stderr contains `exports \`build_index\` but not \`load_index\``, and **no index file written**.

- [ ] **Step 8: Commit**

```bash
git status --short
git add tig-runtime/src/main.rs
git commit -m "feat(c004): add a build-index runtime mode that never sees a nonce"
```

---

## Task 5: Watchdog and balloon memory cap **[GPU]**

**Files:**
- Modify: `tig-runtime/src/main.rs` (inside `build_index`)

**Interfaces:**
- Consumes: `build_index` (Task 4).
- Produces: exit code **85** for a watchdog timeout; a hard device-memory cap in force for the duration of `build_index_fn`.

- [ ] **Step 1: Add the watchdog**

Immediately after `load_settings` in `build_index`:

```rust
    // A flat ceiling that kills a runaway build. It is not the budget -- the
    // budget is `--build-fuel`, which is deterministic and hardware
    // independent. This only catches a build that will never finish.
    let timeout = Duration::from_secs(timeout_secs);
    std::thread::spawn(move || {
        std::thread::sleep(timeout);
        eprintln!("build exceeded the {}s wall-clock watchdog", timeout.as_secs());
        std::process::exit(85);
    });
```

Add `use std::time::Duration;`. `timeout_secs` and `memory_cap_bytes` are
already parameters of `build_index` from Task 4 — do not change the signature
here.

- [ ] **Step 2: Add the balloon**

Directly after `Database::generate` returns and before `build_index_fn` is
called — the order matters, so the database and the generator scratch are
outside the algorithm's cap rather than inside it:

```rust
    // A hard cap, not a sampled one. Polling `mem_get_info` from the watchdog
    // would miss a spike between samples; holding the surplus makes any
    // allocation past the cap fail as an ordinary cudaMalloc error inside the
    // algorithm.
    let (free, total) = cudarc::driver::result::mem_get_info()?;
    if (free as u64) < memory_cap_bytes {
        return Err(anyhow!(
            "device has {} bytes free of {} but the memory cap is {}; refusing \
             to run with a cap that would not be enforced",
            free, total, memory_cap_bytes
        ));
    }
    let balloon_bytes = free as u64 - memory_cap_bytes;
    let _balloon = stream.alloc_zeros::<u8>(balloon_bytes as usize)?;
```

`_balloon` must stay in scope until after `build_index_fn` returns.

- [ ] **Step 3: Verify the cap holds, on tig-gpu**

Change the stub algorithm's `build_index` to allocate `memory_cap + 256 MB`:

```bash
tig-runtime --build-index "$SETTINGS" "$RAND_HASH" stub_greedy.so --ptx stub.ptx \
            --build-fuel 100000000000 --memory-cap 2147483648 --index-out /tmp/idx.blob
```
Expected: non-zero exit, an allocation failure from inside `build_index`, and no
`/tmp/idx.blob`.

Then set `--memory-cap` above the device's free memory.
Expected: non-zero exit with `refusing to run with a cap that would not be enforced` — **not** a silent zero-sized balloon.

- [ ] **Step 4: Verify the watchdog fires**

Change the stub's `build_index` to sleep 10 s, then:

```bash
tig-runtime --build-index "$SETTINGS" "$RAND_HASH" stub_slow.so --ptx stub.ptx \
            --build-fuel 100000000000 --build-timeout 2 --index-out /tmp/idx.blob
echo "exit: $?"
```
Expected: `exit: 85`, stderr names the watchdog, and no index file.

- [ ] **Step 5: Verify build-fuel exhaustion**

Change the stub to launch a kernel that burns more than `--build-fuel`, then run
with a small `--build-fuel`.
Expected: non-zero exit and **no index file** — a partially built index must
never reach the query process.

- [ ] **Step 6: Commit**

```bash
git status --short
git add tig-runtime/src/main.rs
git commit -m "feat(c004): cap the build phase with a watchdog and a memory balloon"
```

---

## Task 6: Batched query mode **[GPU]**

**Files:**
- Modify: `tig-runtime/src/main.rs` (`compute_solution` becomes a loop; index loaded once)

**Interfaces:**
- Consumes: `Database::generate`, `Challenge::for_nonce` (Task 3); `load_index` (Task 4).
- Produces: `tig-runtime <SETTINGS> <RAND_HASH> --start-nonce A --num-nonces M <BINARY> [--index F] --output D`, writing one `{nonce}.json` per nonce exactly as single-nonce mode does.

- [ ] **Step 1: Restructure the gpu arm into a loop**

`start_nonce`, `num_nonces`, `index_path` and `output_dir` come from the new
CLI arguments; single-nonce mode sets `start_nonce = NONCE` and `num_nonces = 1`
before dispatch. `db_seed` is `settings.calc_db_seed(&rand_hash)`, computed once
above the loop.

In the `gpu` arm of `dispatch_challenge!`, hoist everything that does not depend
on the nonce above the loop — context, module, `Database::generate`, and the
`load_index` call — and put the per-nonce work inside:

```rust
            let database = $c::Database::generate(
                &db_seed, &track, module.clone(), stream.clone(), &prop,
            )?;

            if let Some(index_path) = index_path.as_ref() {
                let blob = fs::read(index_path)?;
                let load_index_fn = unsafe {
                    library.get::<fn(&$c::Database, &[u8], Arc<CudaModule>, Arc<CudaStream>, &cudaDeviceProp) -> Result<()>>(b"load_index")
                }.map_err(|_| anyhow!("--index was given but the algorithm does not export `load_index`"))?;
                load_index_fn(&database, &blob, module.clone(), stream.clone(), &prop)?;
            }

            for nonce in start_nonce..start_nonce + num_nonces {
                let seeds = seeds_for(&settings, &rand_hash, nonce);

                // Reset BOTH CPU counters per nonce. They were set once before
                // the loop when this was one process per nonce; leaving them
                // there would carry nonce k-1's spend into nonce k.
                unsafe { *fuel_remaining_ptr = max_fuel };
                unsafe {
                    *runtime_signature_ptr =
                        u64::from_be_bytes(seeds.nonce[0..8].try_into().unwrap())
                };

                let challenge = $c::Challenge::for_nonce(
                    &database, &seeds, &track, module.clone(), stream.clone(), &prop,
                )?;

                // Zeroes gbl_FUELUSAGE and gbl_SIGNATURE on the device.
                unsafe {
                    stream.launch_builder(&initialize_kernel)
                        .arg(&u64::from_be_bytes(seeds.nonce[8..16].try_into().unwrap()))
                        .launch(cfg)?;
                }

                let output_file = output_dir.join(format!("{}.json", nonce));
                // ... existing save_solution_fn, closed over `output_file` ...
                let result = solve_challenge_fn(
                    &challenge, &save_solution_fn, hyperparameters.clone(),
                    module.clone(), stream.clone(), &prop,
                );
                if !output_file.exists() {
                    save_solution_fn(&$c::Solution::new())?;
                }
                result?;
            }
```

Single-nonce mode is `start_nonce = NONCE, num_nonces = 1`, so there is one code
path, not two.

- [ ] **Step 2: Write the fuel-isolation test**

This is the highest-value test in the plan. `gbl_FUELUSAGE` is a device global
and the CPU counter is a process global; if either is not reset per nonce, fuel
accumulates across a bundle and the batch dies partway through with an
out-of-fuel exit that names nothing.

On tig-gpu, with a stub algorithm that burns a fixed, known amount of fuel:

```bash
tig-runtime "$SETTINGS" "$RAND_HASH" --start-nonce 0 --num-nonces 5 stub_fixed.so \
            --ptx stub.ptx --fuel 2000000000 --output /tmp/batch
python3 - <<'EOF'
import json, glob
vals = [json.load(open(f))["fuel_consumed"] for f in sorted(glob.glob("/tmp/batch/*.json"))]
print(vals)
spread = max(vals) - min(vals)
assert spread * 100 < min(vals), f"fuel is not isolated per nonce: {vals}"
EOF
```
Expected: five near-identical values. A missing reset makes them strictly
increasing, which the assertion catches.

- [ ] **Step 3: Verify single-nonce output is unchanged**

```bash
tig-runtime "$SETTINGS" "$RAND_HASH" 7 real_algo.so --ptx real.ptx --fuel 2000000000 --output /tmp/a
tig-runtime "$SETTINGS" "$RAND_HASH" --start-nonce 7 --num-nonces 1 real_algo.so --ptx real.ptx --fuel 2000000000 --output /tmp/b
diff /tmp/a/7.json /tmp/b/7.json && echo IDENTICAL
```
Expected: `IDENTICAL`.

- [ ] **Step 4: Verify the batch verifies**

```bash
for n in 0 1 2 3 4; do
  tig-verifier "$SETTINGS" "$RAND_HASH" $n /tmp/batch/$n.json --ptx real.ptx --audit-salt "$(openssl rand -hex 32)"
done
```
Expected: every nonce prints a `quality:` line and exits 0.

- [ ] **Step 5: Commit**

```bash
git status --short
git add tig-runtime/src/main.rs
git commit -m "feat(c004): solve a bundle in one process, loading the index once"
```

---

## Task 7: Verify the no-rebuild claim **[GPU]**

The spec claims no existing algorithm needs rebuilding, because the split is
host-side and `kernels.cu` is untouched. It is an inference from the code, not a
measurement, and the migration story depends on it. This task turns it into
evidence or kills it.

**Files:** none modified. Produces a measurement note.

- [ ] **Step 1: Pick an unmodified algorithm**

Use `scripts/download_algorithm` to fetch a c004 algorithm binary built before
this branch — `there_v10` if available, otherwise any mainnet c004 algorithm.
Do **not** rebuild it.

- [ ] **Step 2: Run it against the split runtime**

```bash
tig-runtime "$SETTINGS" "$RAND_HASH" 0 there_v10.so --ptx there_v10.ptx \
            --fuel 2000000000 --output /tmp/norebuild
echo "exit: $?"
```
Expected if the claim holds: exit 0 and a solution file. Expected if it does
not: `CUDA_ERROR_NOT_FOUND "named symbol not found"`, which is the signature of
the PTX trap recorded in the migration notes.

- [ ] **Step 3: Verify the solution**

```bash
tig-verifier "$SETTINGS" "$RAND_HASH" 0 /tmp/norebuild/0.json --ptx there_v10.ptx \
             --audit-salt "$(openssl rand -hex 32)"
```
Expected: a `quality:` line at or above 950000 (recall 0.95).

- [ ] **Step 4: Record the result**

Append a `## Validation` entry to
`docs/superpowers/specs/2026-08-31-c004-index-build-split-design.md` giving the
algorithm name, the exact commands, and the raw output. **If the run fails, stop
and report** — the blast-radius section of the spec is wrong and the migration
needs the network-wide rebuild treatment.

- [ ] **Step 5: Commit**

```bash
git status --short
git add docs/superpowers/specs/2026-08-31-c004-index-build-split-design.md
git commit -m "docs(c004): record whether the split needs an algorithm rebuild"
```

---

## Task 8: Measure per-nonce time and fix the constants **[GPU]**

`alpha`, the memory cap and the break-even table all rest on an estimate of
0.05 s/nonce. Task 9 hard-codes constants derived from it, so this must report
first.

**Files:** creates `docs/measurements/2026-08-31-c004-post-split-nonce-time.md`.

- [ ] **Step 1: Establish the pre-split baseline**

Do not check out an older commit and do not use `git stash`. The pre-split
figures are already recorded in the `## Validation` section of
`2026-08-13-gan-instance-generation-design.md`: 1165 ms per nonce for a real
algorithm, 1171 ms for a do-nothing solver, at track 7000. Quote those. If a
fresh baseline is wanted, build it in a scratch worktree
(`git worktree add /tmp/c004-before <sha>`) and remove the worktree afterwards.

- [ ] **Step 2: Measure the post-split per-nonce time**

```bash
/usr/bin/time -v tig-runtime "$SETTINGS" "$RAND_HASH" --start-nonce 0 --num-nonces 20 \
    real_algo.so --ptx real.ptx --fuel 2000000000 --output /tmp/timing 2>&1 | tee /tmp/timing.log
```
Record: total wall-clock, per-nonce mean, and separately the one-off
`Database::generate` cost (time a `--num-nonces 1` run and subtract).

- [ ] **Step 3: Recompute break-even**

With `t_new` measured and `t_old = 1.2 s`, recompute the break-even nonce count
`B / (t_old - t_new)` for a 600 s build, and the table in the spec's "Why not a
flat 10 minutes" section.

- [ ] **Step 4: Choose `alpha`**

`alpha` must satisfy: at the precommit sizes benchmarkers actually use, the
build is a minority of total work AND the split is a net win over the pre-split
baseline. State the chosen value with the arithmetic that produced it. If no
`alpha` satisfies both, say so — that is a finding about the design, not a
failure of the task.

- [ ] **Step 5: Measure peak device memory during a build**

```bash
nvidia-smi --query-gpu=memory.used --format=csv -l 1 > /tmp/mem.log &
MONPID=$!
tig-runtime --build-index ... --index-out /tmp/idx.blob
kill -TERM "$MONPID"
sort -k1 -n -r /tmp/mem.log | head -3
```
Use the peak to confirm or revise the 8 GiB cap against the 16 GB T4 floor.

- [ ] **Step 6: Check the build fits inside a precommit's lifespan**

Spec open question 2. The build cannot start until `rand_hash` is known, so it is
serial latency ahead of every precommit's first nonce.

```bash
curl -s https://mainnet-api.tig.foundation/get-block \
  | python3 -c "import json,sys; c=json.load(sys.stdin)['block']['config']; \
      print('lifespan_period', c['challenges']['c004']['lifespan_period']); \
      print('seconds_between_blocks', c['rounds']['seconds_between_blocks'])"
```

Multiply `lifespan_period` by `seconds_between_blocks` to get the wall-clock a
precommit stays active, and compare it against the measured build time plus the
measured time to compute `num_nonces` nonces. Record the margin. **If the build
consumes a large fraction of the lifespan, say so** — that constrains `alpha`
harder than the amortisation argument does, and it is a finding about the design.

- [ ] **Step 7: Write and commit the measurement note**

```bash
git status --short
git add docs/measurements/2026-08-31-c004-post-split-nonce-time.md
git commit -m "docs(c004): measure post-split per-nonce time and set alpha"
```

---

## Task 9: Protocol validation of `build_fuel_budget` **[local]**

**Files:**
- Modify: `tig-structs/src/config.rs:83-96` (`ChallengeConfig`)
- Modify: `tig-protocol/src/contracts/benchmarks.rs:77-95`
- Test: `tig-protocol/src/contracts/benchmarks.rs` (new `#[cfg(test)] mod tests`)

**Interfaces:**
- Consumes: `alpha` and the memory cap from Task 8.
- Produces: `ChallengeConfig { build_fuel_alpha: f64, max_build_fuel_budget: u64, .. }` and a validation branch beside the existing `fuel_budget` check.

- [ ] **Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_fuel_budget_scales_with_the_nonces_it_is_amortised_over() {
        // 0.25 * 1000 nonces * 2e9 fuel = 5e11
        assert_eq!(calc_build_fuel_budget(0.25, 1_000, 2_000_000_000, u64::MAX), 500_000_000_000);
    }

    #[test]
    fn build_fuel_budget_is_capped() {
        assert_eq!(calc_build_fuel_budget(0.25, 1_000, 2_000_000_000, 1_000), 1_000);
    }

    #[test]
    fn build_fuel_budget_does_not_overflow_at_the_protocol_maximum() {
        // The runtime later multiplies this by gpu_fuel_scale = 20. The
        // mutation this catches is doing the product in u64: 0.25 * 1e6 nonces
        // * 5e12 fuel is 1.25e18, and 1.25e18 * 20 exceeds u64, which would
        // wrap to a tiny patched fuel limit and make every build trap
        // immediately for reasons no message explains.
        let cap = 100_000_000_000_000u64;
        let got = calc_build_fuel_budget(0.25, 1_000_000, 5_000_000_000_000, cap);
        assert_eq!(got, cap);
        assert!(got.checked_mul(20).is_some(), "scaled build fuel must fit in u64");
    }
}
```

- [ ] **Step 2: Run to verify failure**

```bash
cargo test -p tig-protocol
```
Expected: FAIL, `cannot find function 'calc_build_fuel_budget'`.

- [ ] **Step 3: Implement**

In `tig-protocol/src/contracts/benchmarks.rs`:

```rust
/// The build phase's fuel budget: a fixed fraction of the total scored fuel the
/// precommit will spend, so a heavy index is something a benchmarker must
/// commit to a large precommit to earn.
///
/// The product is computed in `u128`. In `u64` it wraps at the protocol
/// maximum, and a wrapped budget becomes a tiny patched PTX fuel limit rather
/// than an error.
pub fn calc_build_fuel_budget(
    alpha: f64,
    num_nonces: u64,
    fuel_budget: u64,
    max_build_fuel_budget: u64,
) -> u64 {
    let scaled = (num_nonces as u128)
        .saturating_mul(fuel_budget as u128)
        .saturating_mul((alpha * 1_000_000.0) as u128)
        / 1_000_000u128;
    scaled.min(max_build_fuel_budget as u128) as u64
}
```

Add to `ChallengeConfig` in `tig-structs/src/config.rs`:

```rust
        build_fuel_alpha: f64,
        max_build_fuel_budget: u64,
```

and beside the existing check at `benchmarks.rs:89`:

```rust
    if challenge_config.max_build_fuel_budget.checked_mul(20).is_none() {
        return Err(anyhow!(
            "max_build_fuel_budget {} overflows when scaled by the GPU fuel scale",
            challenge_config.max_build_fuel_budget
        ));
    }
```

- [ ] **Step 4: Run to verify pass**

```bash
cargo test -p tig-protocol
```
Expected: PASS, 3 tests.

- [ ] **Step 5: Mutation-check**

Change `u128` to `u64` in `calc_build_fuel_budget`.
Expected: `build_fuel_budget_does_not_overflow_at_the_protocol_maximum` FAILS.
Restore.

- [ ] **Step 6: Commit**

```bash
git status --short
git add tig-structs/src/config.rs tig-protocol/src/contracts/benchmarks.rs
git commit -m "feat(c004): scale the build fuel budget to the precommit size"
```

---

## Task 10: Benchmarker slave builds the index once per batch **[local]**

**Files:**
- Modify: `tig-benchmarker/slave/main.py:60-95` (`run_tig_runtime`), plus the batch loop that calls it

**Interfaces:**
- Consumes: the `--build-index` and `--start-nonce/--num-nonces` CLI (Tasks 4, 6); `calc_db_seed` (Task 1).
- Produces: one build invocation per batch before any nonce is computed.

- [ ] **Step 1: Add the build invocation**

```python
def run_build_index(batch, so_path, ptx_path, results_dir):
    """Build the index once per batch, before any nonce is computed.

    Deliberately passes no nonce: the build process must not be able to derive
    a query set. See docs/superpowers/specs/2026-08-31-c004-index-build-split-design.md.
    """
    index_path = f"{results_dir}/{batch['id']}/index.blob"
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    settings = json.dumps(batch["settings"], separators=(',', ':'))
    cmd = [
        "docker", "exec", batch["challenge"], "tig-runtime",
        "--build-index", settings, batch["rand_hash"], so_path,
        "--build-fuel", str(batch["build_fuel_budget"]),
        "--index-out", index_path,
    ]
    if ptx_path is not None:
        cmd += ["--ptx", ptx_path]
    if batch["hyperparameters"] is not None:
        cmd += ["--hyperparameters", json.dumps(batch["hyperparameters"], separators=(',', ':'))]
    logger.debug(f"building index: {' '.join(cmd)}")
    ret = subprocess.run(cmd, capture_output=True, text=True)
    if ret.returncode != 0:
        raise Exception(f"index build failed (exit {ret.returncode}): {ret.stderr.strip()}")
    return index_path
```

Call it once per batch before the nonce loop, and pass `--index <index_path>` to
each `run_tig_runtime` call. When `batch` carries no `build_fuel_budget` — every
challenge but c004 — skip the build and pass no `--index`.

- [ ] **Step 2: Verify the skip path**

The build must be skipped for challenges that do not support it, or every CPU
challenge breaks. Assert this in `tig-benchmarker/tests/data.py`:

```python
    def test_build_is_skipped_without_a_build_fuel_budget(self):
        from slave.main import needs_index_build
        self.assertFalse(needs_index_build({"challenge": "c001"}))
        self.assertTrue(needs_index_build({"challenge": "c004", "build_fuel_budget": 1}))
        # A budget of 0 is a real value, not an absent one: it means "no build
        # fuel", which is different from "this challenge has no build phase".
        self.assertTrue(needs_index_build({"challenge": "c004", "build_fuel_budget": 0}))
```

with:

```python
def needs_index_build(batch) -> bool:
    return "build_fuel_budget" in batch
```

- [ ] **Step 3: Run the Python tests**

```bash
docker run --rm -v "$(pwd)/tig-benchmarker:/src" -w /src tig-bench-py python -m unittest tests.data -v
```
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git status --short
git add tig-benchmarker/slave/main.py tig-benchmarker/tests/data.py
git commit -m "feat(c004): build the index once per batch in the slave"
```

---

## Out of scope for this plan

The cross-repo break is deliberately excluded. `pentest-harness` (tig-pentesting)
has a path dependency on `tig-challenges` and calls `generate_instance` and
`evaluate_solution` directly; the `Seeds` change breaks it exactly as the
`evaluate_solution` audit-salt change did. That needs its own paired plan with an
explicit interface-contract table and a unification step, on the model of
`2026-08-27-recall-gated-c004-{monorepo,harness,unification}.md`. **The two trees
are inconsistent from Task 2 until that plan lands.** That is expected, and it is
why Task 2 does not attempt a harness build.
