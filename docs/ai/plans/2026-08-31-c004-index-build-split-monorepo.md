# c004 Index Build / Query Search Split — Monorepo Side

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make c004 index construction free within a bounded budget so that only query search is charged, by fixing the database per precommit and building the index in a process that is never given a nonce.

**Architecture:** `BenchmarkSettings` gains `calc_db_seed`, which drops the nonce. The three GPU challenges take a `Seeds { nonce, db }` struct instead of a bare `[u8; 32]`. c004's generation splits into `Database::generate` (the `(false, database_size)` pass) and `Challenge::for_nonce` (the `(true, n_queries)` pass plus a device-to-device copy of the database, which keeps `Challenge`'s layout byte-identical and therefore the algorithm ABI unbroken). `tig-runtime` gains a `build-index` subcommand that has no nonce argument at all, and a `batch` subcommand that loads the index once and loops a bundle's nonces, resetting the fuel meter between each. (The spec sketches both as flags; clap cannot express that shape — see Task 4's ruling.)

**Tech Stack:** Rust (edition per workspace, toolchain `nightly-2025-02-10`), CUDA 12.6 via the tig-foundation `cudarc` fork, CUDA C (`kernels.cu` — **not modified**), Python 3 (`tig-benchmarker`), `cargo test`, `unittest`.

**Spec:** `docs/ai/specs/2026-08-31-c004-index-build-split-design.md`

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
- **Everything that touches an algorithm `.so` runs inside Docker on `tig-gpu`.** Verified
  2026-08-31, in this order:
  - `build_so` **cannot** run on the host: `/opt/llvm/bin/{opt,llc}` die with
    ``version `GLIBC_2.36' not found`` (host is Ubuntu 22.04, glibc 2.35).
  - `build_so` **does** work inside the `tig-dev-vector_search` image (Ubuntu 24.04, glibc 2.39,
    LLVM 19.1.7 with `LLVMFuelRTSig.so`, `nightly-2025-02-10` + `rust-src`, CUDA 12.6). It runs
    to completion and the `.so` exports `entry_point`, `__fuel_remaining`, `__runtime_signature`.
    So stubs and real algorithms are built normally, with `build_so`, in the container — and the
    LLVM fuel instrumentation is real, not stubbed out.
  - A container-built `.so` **cannot be `dlopen`'d on the host** (``GLIBC_2.39' not found``;
    the offending symbols are *weak*, which makes it look skippable — it is not, the loader
    rejects the version reference before symbol binding). Every prebuilt mainnet `.so` from
    `scripts/download_algorithm` has the same requirement. **Therefore `tig-runtime` itself must
    also run inside the container** for Tasks 4, 5, 6 and 7.
  - **`/usr/local/bin/tig-runtime` inside the image is STALE** — baked by a `COPY .` at
    image-build time. Task 2 changed `tig-runtime` and `tig-challenges`, so every task must
    `cargo build -r -p tig-runtime --features vector_search` inside the container and invoke the
    binary it just built, by path. Running the baked binary gives a **silently wrong answer, not
    a crash.**
  - Only 4 of 58 `.ll` files get fuel instrumentation (`cudarc` triggers `build_so`'s CUDA
    whitelist: `std-`, `tig_challenges`, `tig_binary`, `tig_algorithms`). An algorithm's own code
    is in `tig_algorithms` and *is* instrumented — but do not over-claim fuel coverage.
  - `build_so` silently clobbers `tig-binary/src/entry_point.rs` on every run.
  - `build_ptx` is nvcc-only and works on the host or in the container.
  - The exact verified `docker run` recipes live in the SDD workspace's `gpu-context.md`.
  - Tasks 3 and 8 need no `.so` and run directly on the host under a `gpu-claim`.
- **`tig-gpu` is an RTX 3060, 12 GB** (verified 2026-08-31), *not* the 16 GB T4 the spec sizes
  the memory cap against. Any memory measurement taken here is a lower bound on a T4 and must
  say so; a passing 8 GiB cap cannot be demonstrated on this box.
- **c006 needs cuDNN, which is not installed system-wide on `tig-gpu`.** Running
  `--features c006` fails at startup with cudarc's "Unable to dynamically load the cudnn shared
  library" unless the documented symlink workaround is applied (symlink each
  `libcudnn*.so.9` under
  `/opt/venvs/wgan-synthetic/lib/python3.12/site-packages/nvidia/cudnn/lib` to a bare
  `libcudnn*.so` in a directory you own and put it first on `LD_LIBRARY_PATH`, exactly as
  `Dockerfile.runtime:28` does).
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
| `tig-runtime/src/main.rs` | `seeds_for`, `build-index` subcommand, watchdog, balloon, `batch` subcommand | 2, 4, 5, 6 |
| `tig-verifier/src/main.rs` | build `Seeds` from both derivations | 2 |
| `tig-structs/src/config.rs` | `build_fuel_alpha`, `max_build_fuel_budget` in `ChallengeConfig` | 9 |
| `tig-protocol/src/contracts/benchmarks.rs` | validate `build_fuel_budget` | 9 |
| `tig-binary/scripts/build_so` | export `build_index` / `load_index` from the version script | 4b |
| `tig-binary/src/entry_point_template.rs` | optional forwarding shims for the two new symbols | 4b |
| `tig-benchmarker/common/batch.py` | `needs_index_build` (importable without the slave's dep graph) | 10 |
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

Change the *second* import line of `tig-structs/tests/core.rs` (line 1 is the `tig_structs::core` import) from
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
    /// precommit. Takes no nonce, deliberately: `tig-runtime build-index`
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

**Get the inputs first.** Neither `$C005_SETTINGS` nor a c005 `.so`/`.ptx` exists yet, and
`build_so` cannot make one (see Global Constraints), so both come off the network:

```bash
export RAND_HASH=auditbaseline
export C005_SETTINGS='{"player_id":"audit","block_id":"audit","challenge_id":"c005","algorithm_id":"<algo>","track_id":"<track>"}'
export CHALLENGE=c005                        # both scripts read the challenge from the ENV,
scripts/list_algorithms                      # not as a positional -- verified in the scripts
scripts/download_algorithm <algo>            # fetches a prebuilt .so; no build_so needed
```
`algorithm_id` and `track_id` in `$C005_SETTINGS` **must** match what you downloaded and a track
c005 actually declares — `calc_seed` hashes the jsonified settings, so a wrong `track_id` changes
the instance, and an unparseable one fails at dispatch.

**c006 is compile-only.** Do not attempt the runtime diff for c006: cuDNN is not installed
system-wide on `tig-gpu`, so `--features c006` fails at startup before it reaches
`generate_instance`. For c006 the evidence is Step 7's successful build plus the fact that its
`generate_instance` body is the two-line `let seed = &seeds.nonce;` change — record that
explicitly in the task report rather than claiming a runtime diff you did not run. If the cuDNN
symlink workaround from the Global Constraints is applied and works, run the diff and say so.

Do this **without `git stash`** — the stash stack is shared with other worktrees
and other sessions. Run it **before committing this task**, so `HEAD` is still Task 1's commit.
Capture the "before" output first, from a build made at the
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

Expected: `c005 IDENTICAL`.

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
    fn for_nonce_keeps_the_index_base_offset() {
        // Pins `for_nonce` against `generate_vectors` called directly at both
        // candidate offsets. Comparing it against `generate_instance` instead
        // would assert nothing: after Step 4 `generate_instance` IS
        // `Database::generate` + `for_nonce`, so an `index_base` mutation moves
        // both sides of that equality together and the test still passes.
        // Against a direct call the mutation is visible.
        let ptx = Ptx::from_file(test_ptx_path().clone());
        let ctx = CudaContext::new(0).unwrap();
        ctx.set_blocking_synchronize().unwrap();
        let module = ctx.load_module(ptx).unwrap();
        let stream = ctx.default_stream();
        let prop = get_device_prop(0).unwrap();
        let track = Track { s: Scenario::SIFT_128 };
        let seeds = Seeds { nonce: [3u8; 32], db: [9u8; 32] };

        let config = ScenarioConfig::from(track.s);
        let weights = weights_from(config.weights).unwrap();
        let layers = &weights.layers;
        let widest = layers.iter().map(|l| l.out_dim).max().unwrap();
        let dims = layers.last().unwrap().out_dim;
        let n = config.n_queries as usize;

        let db = Database::generate(
            &seeds.db, &track, module.clone(), stream.clone(), &prop,
        ).unwrap();
        let split = Challenge::for_nonce(
            &db, &seeds, &track, module.clone(), stream.clone(), &prop,
        ).unwrap();

        let direct = |index_base: usize| -> Vec<f32> {
            let mut dest = stream.alloc_zeros::<f32>(n * dims).unwrap();
            generate_vectors(
                &seeds.nonce, n, index_base, &mut dest,
                layers, widest, module.clone(), stream.clone(),
            ).unwrap();
            stream.synchronize().unwrap();
            stream.memcpy_dtov(&dest.slice(0..4096)).unwrap()
        };

        let got = stream.memcpy_dtov(&split.d_query_vectors.slice(0..4096)).unwrap();
        let at_db_size = direct(db.database_size as usize);
        let at_zero = direct(0);

        // The two offsets must actually differ, or the assertion below is
        // vacuous and would pass against any implementation.
        assert_ne!(
            at_db_size, at_zero,
            "index_base has no effect on the generator; this test cannot discriminate"
        );
        assert_eq!(got, at_db_size, "for_nonce must offset latents by database_size");
    }
```

`generate_vectors` is a private free function in this module, so `mod tests`
(a child module) can call it with no visibility change.

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
        // A `Database` built for a different scenario has different dims and a
        // different row count; silently mixing them would produce a Challenge
        // whose fields disagree with its buffers.
        if db.scenario != track.s {
            return Err(anyhow!(
                "database was generated for scenario {} but this nonce is on {}",
                db.scenario, track.s
            ));
        }
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
cargo test -p tig-challenges --features vector_search -- --test-threads=1 2>&1 | tail -30
```
Expected: PASS, previous count + 2. `--test-threads=1` is not optional on this box: each of
these tests holds a 358 MB database (two of them hold two or three at once) and `tig-gpu` has
12 GB, so the default parallel harness can exhaust the card. An out-of-memory failure here
shows up in `dmesg` as `NVRM: ... Out of memory`, not as a Rust panic.

- [ ] **Step 6: Mutation-check the new tests**

Break the code, confirm the test fails, restore. Both mutations must be run:

1. In `Challenge::for_nonce`, change `&seeds.nonce` to `&seeds.db`.
   Expected: `the_database_is_identical_across_nonces_and_the_queries_are_not` FAILS on the `assert_ne`.
2. In `Challenge::for_nonce`, change `db.database_size as usize` (the `index_base` argument) to `0`.
   Expected: `for_nonce_keeps_the_index_base_offset` FAILS on the final `assert_eq`.
   (It must NOT fail on the `assert_ne` guard — if it does, the guard itself is broken.)

Restore both before committing.

- [ ] **Step 7: Commit**

```bash
git status --short
git add tig-challenges/src/vector_search/mod.rs
git commit -m "feat(c004): split instance generation into database and queries"
```

---

## Task 4: `tig-runtime build-index` **[GPU]**

**Files:**
- Modify: `tig-runtime/src/main.rs` (CLI, new `build_index` function, dispatch)
- Test: `tig-runtime/src/main.rs` (`mod tests`, CLI-level assertions run locally)

**Interfaces:**
- Consumes: `Database::generate` (Task 3), `calc_db_seed` (Task 1).
- Produces: the algorithm ABI other algorithms will be written against —
  - `build_index(&Database, Option<String>, Arc<CudaModule>, Arc<CudaStream>, &cudaDeviceProp) -> Result<Vec<u8>>`, symbol `b"build_index"`, OPTIONAL
  - `load_index(&Database, &[u8], Arc<CudaModule>, Arc<CudaStream>, &cudaDeviceProp) -> Result<()>`, symbol `b"load_index"`, OPTIONAL but required whenever `build_index` is present
  - CLI: `tig-runtime build-index <SETTINGS> <RAND_HASH> <BINARY> --ptx P --build-fuel N --index-out F`
- Task 4b makes these two symbols actually reachable in a `build_so`-built algorithm. Nothing in
  this task depends on 4b (the stubs here are hand-built cdylibs), but a real algorithm does.

### Ruling: subcommands, not a `--build-index` flag

The spec's CLI sketch writes the new modes as flags on the existing command
(`tig-runtime --build-index <SETTINGS> <RAND_HASH> <BINARY> ...`). **That shape cannot be built
in clap 4.** It requires the positional `NONCE` to become optional while the positional `BINARY`
after it stays required, and clap panics on that at `Command` construction — verified:

```
thread 'main' panicked at clap_builder/src/builder/debug_asserts.rs:657:
Found non-required positional argument with a lower index than a required
positional argument: "NONCE" index Some(3)
```

Making `BINARY` optional too does not help: `tig-runtime --build-index "{}" "hash" "lib.so"`
then binds `lib.so` to the `NONCE` slot, and the run fails on a `u64` parse error that never
mentions `--build-index`.

Resolution: the two new modes become **subcommands**, `build-index` and `batch`, with the legacy
four-positional form left exactly as it is via `.subcommand_negates_reqs(true)` and
`.args_conflicts_with_subcommands(true)`. Verified working, including all four rejections the
spec asks for. This is *stronger* than the spec's shape for the property D3 rests on: the
`build-index` subcommand has no nonce argument at all, so a nonce cannot be smuggled in even by
an operator who wants to — the parser rejects it as an unexpected argument rather than as a
configured conflict.

**Cost if this ruling is wrong:** the spec's CLI documentation and Task 10's slave invocation
both have to change wording (both are updated in this plan). No protocol or ABI surface moves.

- [ ] **Step 1: Write the failing CLI tests**

Append to `mod tests` in `tig-runtime/src/main.rs`:

```rust
    #[test]
    fn build_index_mode_has_no_nonce_argument() {
        // The anti-gaming property of D3 is that the build process has no nonce
        // in scope. `build-index` must reject a nonce in every form it could
        // arrive in: as a positional, or as a batch flag.
        for extra in [
            vec!["7"],                       // a positional nonce
            vec!["--start-nonce", "0"],      // the batch flag
            vec!["--num-nonces", "5"],
        ] {
            let mut args = vec![
                "tig-runtime", "build-index", "{}", "hash", "lib.so",
                "--build-fuel", "1000", "--index-out", "/tmp/i",
            ];
            args.extend(extra.iter().copied());
            let err = cli().try_get_matches_from(&args).unwrap_err();
            assert!(
                err.to_string().contains("unexpected argument"),
                "build-index must reject {:?}, got: {}", extra, err
            );
        }
    }

    #[test]
    fn build_index_requires_its_outputs() {
        // Without `.required(true)` these parse fine and the mode then panics
        // on `.unwrap()` deep inside build_index, after the process has already
        // opened a CUDA context.
        let base = ["tig-runtime", "build-index", "{}", "hash", "lib.so",
                    "--build-fuel", "1000", "--index-out", "/tmp/i"];
        for missing in ["--build-fuel", "--index-out"] {
            let mut kept = Vec::new();
            let mut skip = false;
            for a in base {
                if skip { skip = false; continue; }   // drop the flag's value too
                if a == missing { skip = true; continue; }
                kept.push(a);
            }
            let err = cli().try_get_matches_from(&kept).unwrap_err();
            assert!(
                err.to_string().contains(missing),
                "missing {} must be an error naming it, got: {}", missing, err
            );
        }
    }

    #[test]
    fn batched_mode_refuses_zero_nonces() {
        // A batch that produces nothing and exits 0 is indistinguishable from a
        // batch that worked.
        let err = cli()
            .try_get_matches_from(vec![
                "tig-runtime", "batch", "{}", "hash", "lib.so",
                "--start-nonce", "0", "--num-nonces", "0",
            ])
            .unwrap_err();
        assert!(err.to_string().contains("num-nonces"), "got: {}", err);
    }

    #[test]
    fn the_legacy_single_nonce_form_still_parses() {
        // Every existing caller -- tig-verifier's sibling CLI, the slave, and
        // scripts/test_algorithm -- uses this form. Adding subcommands must not
        // move it.
        let m = cli()
            .try_get_matches_from(vec![
                "tig-runtime", "{}", "hash", "7", "lib.so", "--ptx", "p.ptx",
            ])
            .unwrap();
        assert_eq!(m.subcommand_name(), None);
        assert_eq!(*m.get_one::<u64>("NONCE").unwrap(), 7);
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cargo test -p tig-runtime --features c001
```
Expected: FAIL — `build-index` / `batch` are unrecognised subcommands.

- [ ] **Step 3: Extend the CLI**

Leave every existing argument on the root command untouched, and add the two subcommands. The
root gains only `--index`, which the legacy form may also pass.

```rust
fn shared_args() -> Vec<clap::Arg> {
    vec![
        arg!(<SETTINGS> "Settings json string or path to json file")
            .value_parser(clap::value_parser!(String)),
        arg!(<RAND_HASH> "A string used in seed generation")
            .value_parser(clap::value_parser!(String)),
        arg!(<BINARY> "Path to a shared object (*.so) file")
            .value_parser(clap::value_parser!(PathBuf)),
        arg!(--hyperparameters [HYPERPARAMETERS] "Hyperparameters json string or path to json file")
            .value_parser(clap::value_parser!(String)),
        arg!(--ptx [PTX] "Path to a CUDA ptx file")
            .value_parser(clap::value_parser!(PathBuf)),
        arg!(--gpu [GPU] "Which GPU device to use")
            .value_parser(clap::value_parser!(usize)),
    ]
}
```

On the root `Command`, add:

```rust
        .subcommand_negates_reqs(true)
        .args_conflicts_with_subcommands(true)
        .arg(arg!(--index [PATH] "Index blob to load before solving")
            .value_parser(clap::value_parser!(PathBuf)))
        .subcommand(
            Command::new("build-index")
                .about("Build an index over the precommit's database and exit. Takes no nonce.")
                .args(shared_args())
                .arg(arg!(--"build-fuel" <FUEL> "Fuel budget for the build phase")
                    .required(true)
                    .value_parser(clap::value_parser!(u64)))
                .arg(arg!(--"index-out" <PATH> "Where to write the index blob")
                    .required(true)
                    .value_parser(clap::value_parser!(PathBuf)))
                .arg(arg!(--"memory-cap" [BYTES] "Device memory the build may use")
                    .value_parser(clap::value_parser!(u64))
                    .default_value("8589934592"))
                .arg(arg!(--"build-timeout" [SECS] "Wall-clock watchdog for the build")
                    .value_parser(clap::value_parser!(u64))
                    .default_value("600")),
        )
        .subcommand(
            Command::new("batch")
                .about("Solve a contiguous run of nonces in one process")
                .args(shared_args())
                .arg(arg!(--"start-nonce" <N> "First nonce of a batch")
                    .required(true)
                    .value_parser(clap::value_parser!(u64)))
                .arg(arg!(--"num-nonces" <N> "How many nonces to solve")
                    .required(true)
                    .value_parser(clap::value_parser!(u64).range(1..)))
                .arg(arg!(--fuel [FUEL] "Optional maximum fuel parameter")
                    .default_value("2000000000")
                    .value_parser(clap::value_parser!(u64)))
                .arg(arg!(--index [PATH] "Index blob to load before solving")
                    .value_parser(clap::value_parser!(PathBuf)))
                .arg(arg!(--output [OUTPUT_FOLDER] "Folder for the per-nonce output files")
                    .value_parser(clap::value_parser!(PathBuf))),
        )
```

`.required(true)` on `--build-fuel`, `--index-out`, `--start-nonce` and `--num-nonces` is
load-bearing, not decorative: `arg!(--"build-fuel" <FUEL>)` alone makes only the *value*
mandatory, so the flag stays optional and `get_one(...).unwrap()` panics when it is omitted.
Verified — without `.required(true)`, `build-index "{}" hash lib.so --index-out /tmp/i` parses
clean.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cargo test -p tig-runtime --features c001
```
Expected: PASS, 6 tests (2 from Task 2, 4 from this task).

- [ ] **Step 5: Implement the build mode**

Add to `tig-runtime/src/main.rs`, gated `#[cfg(feature = "c004")]` — it names
`c004::Database` directly, so `cuda` alone is not enough:

```rust
#[cfg(feature = "c004")]
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
    // No nonce is derived here and none can be passed -- the `build-index`
    // subcommand has no nonce argument at all. This is the property the design
    // rests on: a build process cannot compute any query set because the
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

    let cfg = LaunchConfig { grid_dim: (1,1,1), block_dim: (1,1,1), shared_mem_bytes: 0 };
    let initialize_kernel = module.load_function("initialize_kernel")?;
    unsafe {
        stream.launch_builder(&initialize_kernel)
            .arg(&u64::from_be_bytes(db_seed[8..16].try_into().unwrap()))
            .launch(cfg)?;
    }

    // (Task 5 inserts the balloon here, between generation and the build.)

    let blob = build_index_fn(&database, hyperparameters, module.clone(), stream.clone(), &prop)?;

    // A GPU fuel trap is asynchronous: `build_index_fn` can return Ok while the
    // device has already trapped and set gbl_ERRORSTAT. Without this, a build
    // that blew its fuel budget writes an index and exits 0, and the spec's
    // "build-fuel exhaustion leaves no index file" is simply false.
    stream.synchronize()?;
    ctx.synchronize()?;
    let mut fuel_usage = stream.alloc_zeros::<u64>(1)?;
    let mut signature = stream.alloc_zeros::<u64>(1)?;
    let mut error_stat = stream.alloc_zeros::<u64>(1)?;
    let finalize_kernel = module.load_function("finalize_kernel")?;
    unsafe {
        stream.launch_builder(&finalize_kernel)
            .arg(&mut fuel_usage).arg(&mut signature).arg(&mut error_stat)
            .launch(cfg)?;
    }
    let error_stat = stream.memcpy_dtov(&error_stat)?[0];
    let gpu_fuel_used = stream.memcpy_dtov(&fuel_usage)?[0] / gpu_fuel_scale;
    if error_stat != 0 {
        return Err(anyhow!(
            "build failed on the device (error_stat {}, gpu fuel used {} of {}); no index written",
            error_stat, gpu_fuel_used, build_fuel
        ));
    }

    // Atomic: a watchdog kill must never leave a partial blob for the query
    // process to load.
    let tmp = index_out.with_extension("tmp");
    fs::write(&tmp, &blob)?;
    fs::rename(&tmp, &index_out)?;
    eprintln!("index written: {} bytes, gpu fuel used {} of {}", blob.len(), gpu_fuel_used, build_fuel);
    Ok(())
}
```

Factor the `track_id` parsing already duplicated in `dispatch_challenge!` into
`fn parse_track(settings: &BenchmarkSettings) -> Result<c004::Track>` and call it
from both places.

- [ ] **Step 6: Wire the mode into `main`**

```rust
    if let Some(sub) = matches.subcommand_matches("build-index") {
        #[cfg(not(feature = "c004"))]
        {
            let _ = sub;
            eprintln!("Runtime Error: build-index requires a build with '--features c004'");
            std::process::exit(84);
        }
        #[cfg(feature = "c004")]
        {
            if let Err(e) = build_index(
                sub.get_one::<String>("SETTINGS").unwrap().clone(),
                sub.get_one::<String>("RAND_HASH").unwrap().clone(),
                sub.get_one::<PathBuf>("BINARY").unwrap().clone(),
                sub.get_one("hyperparameters").cloned(),
                sub.get_one::<PathBuf>("ptx")
                    .cloned()
                    .ok_or(())
                    .unwrap_or_else(|_| {
                        eprintln!("Runtime Error: --ptx is required for build-index");
                        std::process::exit(84);
                    }),
                *sub.get_one::<u64>("build-fuel").unwrap(),
                *sub.get_one::<u64>("memory-cap").unwrap(),
                *sub.get_one::<u64>("build-timeout").unwrap(),
                sub.get_one::<PathBuf>("index-out").unwrap().clone(),
                sub.get_one::<usize>("gpu").cloned(),
            ) {
                eprintln!("Runtime Error: {}", e);
                std::process::exit(84);
            }
        }
        return;
    }
```

The `#[cfg(not(feature = "c004"))]` arm must **exit non-zero**, not fall through. Written as the
spec's original sketch (a bare `panic!` under `not(feature = "cuda")` plus a body under
`feature = "c004"`), a `--features c005` build reaches `if flag { return; }` and **exits 0 having
built nothing** — a silent success on an unsupported binary. `not(feature = "c004")` is the right
gate, not `not(feature = "cuda")`: c005 and c006 are cuda builds without c004.

The `.unwrap()`s on `build-fuel`, `index-out`, `start-nonce` and `num-nonces` are safe only
because Step 3 marks all four `.required(true)`. If that changes, these become panics.

- [ ] **Step 7: Verify end to end on tig-gpu with a stub algorithm**

The stub is a **real algorithm built with `build_so`, inside the container** (see Global
Constraints — `build_so` works there and only there, and a container-built `.so` cannot be loaded
on the host, so `tig-runtime` runs in the container too).

Write it as a normal c004 algorithm under `tig-algorithms/src/vector_search/<stub_name>/`, add
its `pub mod` line to `tig-algorithms/src/vector_search/mod.rs`, and give it a `solve_challenge`
plus the two new symbols. It needs **Task 4b's `index_build` feature** to export `build_index`
and `load_index` — `build_so`'s version script hides everything it does not name, so without 4b
the runtime cannot find them no matter what the source exports. If 4b is not done yet, do it
first or temporarily add the two names to the version script and say so in your report.

The stub's `build_index` returns `Ok(vec![1, 2, 3])` and its `load_index` returns `Ok(())`.
Because `build_so` applies the real LLVM fuel pass, this stub's CPU fuel **is** metered, unlike
the plain-cargo route this plan originally specified.

Remove the stub's `pub mod` line and its directory when the task is done, and say so.

```bash
docker run --rm --gpus all -v /workspace/tig-bench:/app -w /app \
  -e CHALLENGE=vector_search -e RUSTUP_TOOLCHAIN=nightly-2025-02-10 \
  tig-dev-vector_search bash -c '
    set -e
    bash tig-binary/scripts/build_so <STUB>
    cargo build -r -p tig-runtime --features vector_search   # MANDATORY: baked binary is stale
    ./target/release/tig-runtime build-index "$SETTINGS" "$RAND_HASH" \
        tig-algorithms/lib/vector_search/amd64/<STUB>.so \
        --ptx tig-algorithms/lib/vector_search/ptx/<STUB>.ptx \
        --build-fuel 100000000000 --index-out /tmp/idx.blob
    test -f /tmp/idx.blob && wc -c /tmp/idx.blob
    ls /tmp/idx.tmp 2>/dev/null && echo "BUG: temp file left behind"
  '
```
Expected: exit 0, `3 /tmp/idx.blob`, and no `.tmp` file left behind.

Then remove `load_index` from the stub, rebuild, and re-run.
Expected: non-zero exit, stderr contains ``exports `build_index` but not `load_index` ``, and
**no index file written**.

Then remove `build_index` too, rebuild, and re-run.
Expected: non-zero exit naming `does not export`, and no index file.
- [ ] **Step 8: Commit**

```bash
git status --short
git add tig-runtime/src/main.rs
git commit -m "feat(c004): add a build-index runtime mode that never sees a nonce"
```


---

## Task 4b: Make the two new symbols survive `build_so` **[local]**

Task 4 defines an ABI that no real algorithm can satisfy yet. `tig-binary/scripts/build_so`
links with an explicit version script:

```
cat > export_symbols.map << 'EOF'
{
  global:
    help;
    entry_point;
    __fuel_remaining;
    ...
  local: *;
};
EOF
```

`local: *` means **every symbol not on that list is hidden**. `build_index` and `load_index` are
not on it, so `library.get(b"build_index")` on any `build_so`-built algorithm fails with "symbol
not found" no matter what the algorithm's source exports. Nothing in this plan would have caught
that: every runtime step in Tasks 4-6 uses a hand-built stub cdylib with no version script, and
Task 7 runs a *pre-split* algorithm that exports neither symbol.

**Files:**
- Modify: `tig-binary/scripts/build_so` (version script)
- Modify: `tig-binary/src/entry_point_template.rs` (optional forwarding shims)

**Interfaces:**
- Consumes: the symbol names fixed by Task 4.
- Produces: nothing this plan's other tasks call. It is what makes the ABI reachable for the
  first real index-building algorithm, which is out of scope here.

- [ ] **Step 1: Add the two symbols to the version script**

In `tig-binary/scripts/build_so`, inside the `export_symbols.map` heredoc, beside `entry_point;`:

```
    build_index;
    load_index;
```

- [ ] **Step 2: Add the forwarding shims to the template**

`entry_point_template.rs` is the only file `build_so` compiles that can carry a
`#[unsafe(no_mangle)]` symbol. The shims must be **conditional on the algorithm actually
providing the functions**, or every one of the 105 existing algorithms stops compiling. Rust has
no "call this if it exists", so gate them on a per-algorithm cargo feature that defaults off:

```rust
#[cfg(all(feature = "cuda", feature = "index_build"))]
#[unsafe(no_mangle)]
pub fn build_index(
    database: &Database,
    hyperparameters: Option<String>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> Result<Vec<u8>> {
    catch_unwind(AssertUnwindSafe(|| {
        let hyperparameters = hyperparameters.map(|x| serde_json::from_str::<Map<String, Value>>(&x).unwrap());
        {ALGORITHM}::build_index(database, &hyperparameters, module, stream, prop)
    })).unwrap_or_else(|_| Err(anyhow!("Panic occurred calling build_index")))
}

#[cfg(all(feature = "cuda", feature = "index_build"))]
#[unsafe(no_mangle)]
pub fn load_index(
    database: &Database,
    blob: &[u8],
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> Result<()> {
    catch_unwind(AssertUnwindSafe(|| {
        {ALGORITHM}::load_index(database, blob, module, stream, prop)
    })).unwrap_or_else(|_| Err(anyhow!("Panic occurred calling load_index")))
}
```

Add `index_build = []` to `tig-binary/Cargo.toml`'s `[features]`, and in `build_so` append it to
`FEATURES` only when the caller asks:

```bash
FEATURES="entry_point $CHALLENGE"
if [ -n "$INDEX_BUILD" ]; then FEATURES="$FEATURES index_build"; fi
```

A dead config gate is exactly what this could become, so the check is `-n` on an unset-or-empty
variable, never a string comparison against `"0"`.

- [ ] **Step 3: Prove existing algorithms still build**

`build_so` works inside the `tig-dev-vector_search` container (Global Constraints), so this is a
**real build test**, not a grep. That matters here more than anywhere else in the plan: the whole
point of this task is that a symbol can be present in the source and still be absent from the
`.so`, which is exactly what a grep cannot see.

Using any c004 algorithm (the Task 4 stub, or a throwaway one):

```bash
docker run --rm -v /workspace/tig-bench:/app -w /app \
  -e CHALLENGE=vector_search -e RUSTUP_TOOLCHAIN=nightly-2025-02-10 \
  tig-dev-vector_search bash -c '
    set -e
    SO=tig-algorithms/lib/vector_search/amd64/<ALGO>.so
    # 1. WITHOUT the feature: the two symbols must be ABSENT, and every
    #    existing algorithm must still build. This is the regression guard.
    bash tig-binary/scripts/build_so <ALGO>
    nm -D --defined-only "$SO" | grep -Ec "build_index|load_index"   # expect 0
    nm -D --defined-only "$SO" | grep -c entry_point                 # expect 1
    # 2. WITH the feature: both must now be exported.
    INDEX_BUILD=1 bash tig-binary/scripts/build_so <ALGO>
    nm -D --defined-only "$SO" | grep -Ec "build_index|load_index"   # expect 2
  '
```

Expected: `0`, `1`, then `2`. The first block is what proves the change is backward compatible;
the second is what proves it does anything. A run that only does the second half has not tested
the version script at all.

`build_so` clobbers `tig-binary/src/entry_point.rs` on every invocation — back it up first if it
exists as untracked scratch on the box, and restore it afterwards.

- [ ] **Step 4: Commit**

```bash
git status --short
git add tig-binary/scripts/build_so tig-binary/src/entry_point_template.rs tig-binary/Cargo.toml
git commit -m "feat(c004): export build_index and load_index from algorithm binaries"
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
    /// No allocator hands out 100% of reported free memory -- fragmentation and
    /// per-allocation bookkeeping always leave a sliver unreachable -- so a
    /// balloon of exactly `free - cap` fails on a healthy device and every build
    /// errors out. Reserve a fixed, documented sliver instead, and treat the
    /// effective cap as `memory_cap_bytes + BALLOON_SLACK`.
    const BALLOON_SLACK: u64 = 64 * 1024 * 1024;

    // A hard cap, not a sampled one. Polling `mem_get_info` from the watchdog
    // would miss a spike between samples; holding the surplus makes any
    // allocation past the cap fail as an ordinary cudaMalloc error inside the
    // algorithm.
    let (free, total) = cudarc::driver::result::mem_get_info()?;
    let free = free as u64;
    if free < memory_cap_bytes + BALLOON_SLACK {
        return Err(anyhow!(
            "device has {} bytes free of {} but the memory cap is {} (+{} slack); \
             refusing to run with a cap that would not be enforced",
            free, total, memory_cap_bytes, BALLOON_SLACK
        ));
    }
    let balloon_bytes = free - memory_cap_bytes - BALLOON_SLACK;
    let _balloon = stream.alloc_zeros::<u8>(balloon_bytes as usize).map_err(|e| {
        anyhow!(
            "could not inflate the {}-byte memory balloon ({}); the cap would be \
             unenforced, so the build is refused rather than run uncapped",
            balloon_bytes, e
        )
    })?;
```

`_balloon` must stay in scope until after `build_index_fn` returns. Do **not** retry the
allocation at a smaller size on failure: a smaller balloon is a *looser* cap, so a shrink loop
silently converts "cannot enforce the cap" into "ran without one", which is the exact failure
the spec calls out as worse than not building at all.

If `cudarc::driver::result::mem_get_info` is not present in the pinned cudarc fork, the
equivalent is `cudarc::driver::sys::cuMemGetInfo_v2` behind the same `Result` wrapper. Check
before writing the rest of the step; this is the one API in the plan not already used elsewhere
in this tree.

- [ ] **Step 3: Verify the cap holds, on tig-gpu**

Change the stub algorithm's `build_index` to allocate `memory_cap + 256 MB` (same `build_so`
stub as Task 4 Step 7, rebuilt and run **inside the container** — see Global Constraints). Use a
cap small enough to leave room on a 12 GB RTX 3060:

```bash
./target/release/tig-runtime build-index "$SETTINGS" "$RAND_HASH" \
    tig-algorithms/lib/vector_search/amd64/<STUB>.so \
    --ptx tig-algorithms/lib/vector_search/ptx/<STUB>.ptx \
    --build-fuel 100000000000 --memory-cap 2147483648 --index-out /tmp/idx.blob
echo "exit: $?"; ls /tmp/idx.blob 2>/dev/null && echo "BUG: index written"
```
Expected: non-zero exit, an allocation failure from inside `build_index`, and no
`/tmp/idx.blob`.

Then set `--memory-cap` above the device's free memory (e.g. `--memory-cap 34359738368` on a
12 GB card).
Expected: non-zero exit with `refusing to run with a cap that would not be enforced` — **not** a
silent zero-sized balloon.

Note in the report that the **8 GiB production cap is not exercised here**: this box is a 12 GB
RTX 3060, and 8 GiB plus the database, scratch and context does not leave enough headroom to
demonstrate the cap the spec sizes for a 16 GB T4. What this step proves is that the mechanism
works, at a cap the box can hold.

- [ ] **Step 4: Verify the watchdog fires**

Change the stub's `build_index` to sleep 10 s, then:

```bash
./target/release/tig-runtime build-index "$SETTINGS" "$RAND_HASH" \
    tig-algorithms/lib/vector_search/amd64/<STUB>.so \
    --ptx tig-algorithms/lib/vector_search/ptx/<STUB>.ptx \
    --build-fuel 100000000000 --build-timeout 2 --index-out /tmp/idx.blob
echo "exit: $?"; ls /tmp/idx.blob /tmp/idx.tmp 2>/dev/null
```
Expected: `exit: 85`, stderr names the watchdog, and neither `idx.blob` nor `idx.tmp` exists.

- [ ] **Step 5: Verify build-fuel exhaustion**

Change the stub to launch a kernel that burns more than `--build-fuel`, then run
with a small `--build-fuel`:

```bash
./target/release/tig-runtime build-index "$SETTINGS" "$RAND_HASH" \
    tig-algorithms/lib/vector_search/amd64/<STUB>.so \
    --ptx tig-algorithms/lib/vector_search/ptx/<STUB>.ptx \
    --build-fuel 1000 --index-out /tmp/idx.blob
echo "exit: $?"; ls /tmp/idx.blob 2>/dev/null && echo "BUG: index written after a fuel trap"
```
Expected: non-zero exit naming `error_stat`, and **no index file** — a partially built index must
never reach the query process. This step is what the `finalize_kernel` / `error_stat` check added
in Task 4 Step 5 exists for; without it `build_index_fn` returns `Ok` and the blob is written
even though the device trapped. If this step passes *without* that check present, the step is not
testing what it claims — check that the stub's kernel really exceeds the budget.

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
- Produces: `tig-runtime batch <SETTINGS> <RAND_HASH> <BINARY> --start-nonce A --num-nonces M [--index F] --output D`, writing one `{nonce}.json` per nonce exactly as single-nonce mode does.

- [ ] **Step 0: Change `compute_solution`'s signature**

The plan's sketch below silently assumes four things the current code does not do. State them as
edits, because each is a defect if it is merely assumed:

1. **`compute_solution` takes a range, not a nonce.** Replace the `nonce: u64` parameter with
   `start_nonce: u64, num_nonces: u64, index_path: Option<PathBuf>`. Both `main` arms fill it in:
   the legacy form passes `(NONCE, 1, matches.get_one("index").cloned())`, `batch` passes its own
   three.
2. **`output_file` becomes per-nonce.** Today it is computed *once*, above the macro, from the
   outer `nonce`. Inside the loop it must be recomputed as
   `output_dir.join(format!("{}.json", nonce))` — otherwise every nonce in a bundle overwrites
   one file and the batch reports a single result.
3. **`OutputData.nonce` becomes the loop's nonce.** The `save_solution_fn` closure writes
   `nonce,` from the enclosing scope. If it keeps closing over the old outer binding, every
   output file in the batch is stamped with the batch's first nonce, the Merkle root the slave
   computes is wrong, and *nothing fails* — the files are well-formed and the verifier is never
   asked about the mismatch. Make the closure take the nonce, or rebuild it inside the loop.
4. **GPU selection stops depending on the nonce.** `gpu_device.unwrap_or((nonce % num_gpus) as usize)`
   must become `start_nonce % num_gpus`: the context is created once, above the loop, so there is
   no per-nonce device to pick.

`start_nonce`, `num_nonces`, `index_path` and `output_dir` come from the new
CLI arguments; the legacy single-nonce form sets `start_nonce = NONCE` and `num_nonces = 1`
before dispatch, so there is one code path and not two. `db_seed` is
`settings.calc_db_seed(&rand_hash)`, computed once above the loop.

- [ ] **Step 1: Restructure the gpu arm into a loop**

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

                // Rebuilt per nonce -- see Step 0 items 2 and 3. Both the path
                // and the `nonce` field inside OutputData must be this nonce.
                let output_file = output_dir.join(format!("{}.json", nonce));
                let save_solution_fn = |solution: &$c::Solution| -> Result<()> {
                    // ... body unchanged from today, except that `nonce` and
                    // `output_file` now resolve to the loop's bindings ...
                };
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

`load_index` runs **before the first `initialize_kernel`**, so like `generate_instance` it is
outside the fuel meter. That is deliberate and not exploitable — `load_index` receives only the
`Database` and the blob, never a query — but it is a second free phase that the spec's
enforcement table does not mention. Note it in the task report so the spec can pick it up.

- [ ] **Step 2: Write the fuel-isolation test**

This is the highest-value test in the plan. `gbl_FUELUSAGE` is a device global
and the CPU counter is a process global; if either is not reset per nonce, fuel
accumulates across a bundle and the batch dies partway through with an
out-of-fuel exit that names nothing.

Inside the container, with a stub algorithm whose `entry_point` launches a kernel burning a
fixed, known amount of GPU fuel and then calls `save_solution` (the same `build_so` stub as
Task 4 Step 7). Because `build_so` applies the real fuel pass, **both** the CPU counter and the
GPU counter are genuinely under test here — which is what makes this the plan's highest-value
test.

```bash
rm -rf /tmp/batch
./target/release/tig-runtime batch "$SETTINGS" "$RAND_HASH" \
    tig-algorithms/lib/vector_search/amd64/<STUB>.so \
    --start-nonce 0 --num-nonces 5 \
    --ptx tig-algorithms/lib/vector_search/ptx/<STUB>.ptx \
    --fuel 2000000000 --output /tmp/batch
python3 - <<'EOF'
import json, glob
files = sorted(glob.glob("/tmp/batch/*.json"))
assert len(files) == 5, f"expected 5 output files, got {len(files)}: {files}"
recs = [json.load(open(f)) for f in files]
# Step 0 item 3: each file must be stamped with its own nonce, not the batch's first.
assert sorted(r["nonce"] for r in recs) == [0, 1, 2, 3, 4], [r["nonce"] for r in recs]
vals = [r["fuel_consumed"] for r in recs]
print(vals)
assert min(vals) > 0, f"stub burned no measurable fuel; this test cannot discriminate: {vals}"
spread = max(vals) - min(vals)
assert spread * 100 < min(vals), f"fuel is not isolated per nonce: {vals}"
EOF
```
Expected: five near-identical, non-zero values, one per nonce. A missing reset makes them
strictly increasing, which the spread assertion catches; a missing per-nonce `output_file` leaves
one file, which the count assertion catches; a stale `OutputData.nonce` gives five copies of `0`,
which the nonce assertion catches. The `min(vals) > 0` guard exists because with a stub that
burns nothing every value is `0`, `spread` is `0`, and the spread assertion passes against any
implementation at all.

- [ ] **Step 3: Verify single-nonce output is unchanged**

Use a real solver — either the prebuilt algorithm from Task 7 or one you build with `build_so`.
Both sides must run **inside the container** with the **freshly built** `tig-runtime`, not the
stale baked one.

```bash
./target/release/tig-runtime "$SETTINGS" "$RAND_HASH" 7 "$ALGO_SO" --ptx "$ALGO_PTX" --fuel 2000000000 --output /tmp/a
./target/release/tig-runtime batch "$SETTINGS" "$RAND_HASH" "$ALGO_SO" --start-nonce 7 --num-nonces 1 --ptx "$ALGO_PTX" --fuel 2000000000 --output /tmp/b
diff /tmp/a/7.json /tmp/b/7.json && echo IDENTICAL
```
Expected: `IDENTICAL`. Note that both sides run the *post-split* runtime, so this checks the
batch path against the single-nonce path, not against pre-split output. Pre-split equivalence is
not expected and not claimed: D1 deliberately changes the database seed, so every generated
vector differs from before this branch.

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

```bash
export CHALLENGE=c004                        # the script reads CHALLENGE from the environment
scripts/list_algorithms
scripts/download_algorithm there_v10         # or any mainnet c004 algorithm
```

Fetch a c004 algorithm binary built **before** this branch — `there_v10` if available, otherwise
any mainnet c004 algorithm. Do **not** rebuild it; a rebuild would defeat the whole point of the
task, and in any case `build_so` cannot run on this box.

The downloaded `.so` statically links its own pre-split copy of `tig-challenges`, so it reads
`Challenge`'s fields at the offsets that copy compiled. That is exactly the claim under test: if
D5's "layout unchanged" holds, it reads the right fields; if it does not, it reads garbage or
faults.

- [ ] **Step 2: Run it against the split runtime**

```bash
export RAND_HASH=norebuildcheck
export SETTINGS='{"player_id":"audit","block_id":"audit","challenge_id":"c004","algorithm_id":"there_v10","track_id":"<track>"}'
# Inside the container: the prebuilt .so needs GLIBC_2.39 and the host has 2.35,
# and the image's baked tig-runtime is stale, so build the runtime first.
docker run --rm --gpus all -v /workspace/tig-bench:/app -w /app \
  -e CHALLENGE=vector_search -e RUSTUP_TOOLCHAIN=nightly-2025-02-10 \
  tig-dev-vector_search bash -c '
    set -e
    cargo build -r -p tig-runtime --features vector_search
    ./target/release/tig-runtime "'"$SETTINGS"'" "'"$RAND_HASH"'" 0 \
        there_v10.so --ptx there_v10.ptx --fuel 2000000000 --output /tmp/norebuild
    echo "exit: $?"
  '
```

Note the exit code but judge on the output file: `tig-runtime` has a **pre-existing** teardown
crash that exits 137/139 *after* the solution is written, reproducible on unmodified binaries
(found in Task 2). Do not chase it, and do not read it as a failure of the no-rebuild claim.
`algorithm_id` and `track_id` must be the real ones for the algorithm you downloaded —
`calc_seed` and `calc_db_seed` both hash the jsonified settings, and an unparseable `track_id`
fails at dispatch before any of this is exercised. Use the legacy single-nonce form here
deliberately: it is the path every existing algorithm uses, and Task 6's `batch` subcommand is
not what this task is checking.
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
`docs/ai/specs/2026-08-31-c004-index-build-split-design.md` giving the
algorithm name, the exact commands, and the raw output. **If the run fails, stop
and report** — the blast-radius section of the spec is wrong and the migration
needs the network-wide rebuild treatment.

- [ ] **Step 5: Commit**

```bash
git status --short
git add docs/ai/specs/2026-08-31-c004-index-build-split-design.md
git commit -m "docs(c004): record whether the split needs an algorithm rebuild"
```

---

## Task 8: Measure per-nonce time and fix the constants **[GPU]**

`alpha`, the memory cap and the break-even table all rest on an estimate of
0.05 s/nonce. Task 9 hard-codes constants derived from it, so this must report
first.

**Files:** creates `docs/measurements/2026-08-31-c004-post-split-nonce-time.md`. The directory
does not exist — `mkdir -p docs/measurements` first, or Step 7's `git add` fails.

**Hardware caveat that must appear in the note.** `tig-gpu` is a 12 GB RTX 3060, not the 16 GB
T4 (`AWS_G4dn`) the spec sizes the memory cap against, and not the card the pre-split 1165 ms/
1171 ms baseline was measured on (an RTX 4060, per the design's `## Validation`). So:
- Step 3's break-even arithmetic mixes a 4060 baseline with a 3060 measurement. Either re-measure
  the baseline on this card in a scratch worktree (Step 1 already describes how) or state the
  cross-card comparison explicitly as a caveat on `alpha`. Do not quietly mix them.
- Step 5's peak-memory figure is a measurement on a 3060; it can rule the 8 GiB cap *out* but it
  cannot confirm it fits a T4's budget alongside the query process. Say which.

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
tig-runtime build-index ... --index-out /tmp/idx.blob
kill -TERM "$MONPID"
sort -k1 -n -r /tmp/mem.log | head -3
```
Use the peak to confirm or revise the 8 GiB cap against the 16 GB T4 floor — subject to the
hardware caveat above. `MONPID` is captured at launch and killed by exact PID, never by name
pattern.

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
        build_fuel_alpha: Option<f64>,
        max_build_fuel_budget: Option<u64>,
```

**`Option<...>`, not bare fields.** `serializable_struct_with_getters!` emits `#[serde(default)]`
*only* on the `Option<$type>` arm (`tig-structs/src/lib.rs:15-28`); a bare `f64` field is
required by serde. Adding two required fields makes **every existing config JSON fail to
deserialize**, including the live `https://mainnet-api.tig.foundation/get-block` payload that
`tig-protocol` and the benchmarker both read — a change that breaks the running system to add a
constant nothing reads yet. The macro's `Option` arm also generates the panicking accessor
`challenge_config.max_build_fuel_budget()`, which is the codebase's existing idiom for
config that is present for some challenges and absent for others. Read them through those
accessors only on the c004 path.

and beside the existing check at `benchmarks.rs:89`:

```rust
    // `Some(0)` is a legitimate value -- "this challenge grants no build fuel" --
    // and is not the same as `None`, "this challenge has no build phase". Match
    // on the Option; never `if let Some(x) = .. if x > 0`, and never a falsy
    // check on the u64.
    if let Some(max_build_fuel_budget) = challenge_config.max_build_fuel_budget {
        if max_build_fuel_budget.checked_mul(20).is_none() {
            return Err(anyhow!(
                "max_build_fuel_budget {} overflows when scaled by the GPU fuel scale",
                max_build_fuel_budget
            ));
        }
    }
```

- [ ] **Step 3b: Decide, and record, where `build_fuel_budget` reaches the batch**

`calc_build_fuel_budget` as specified has **no caller**, and Task 10's slave reads
`batch["build_fuel_budget"]` — a key nothing in this plan ever writes. Left as-is,
`needs_index_build` is always `False`, no index is ever built, and the whole feature is dead code
that every test in Tasks 9 and 10 still passes. That is precisely the shape of defect this plan
is meant to avoid, so it must be settled here rather than discovered later.

Two acceptable outcomes; pick one and write it into the task report:

- **Wire it.** Call `calc_build_fuel_budget(alpha, num_nonces, fuel_budget, max)` where the
  precommit's per-batch payload is assembled in `tig-benchmarker/master`, and add
  `build_fuel_budget` to that payload. Grep for where `fuel_budget` reaches a batch dict and put
  it beside that; if the master turns out to need protocol changes beyond this plan's scope,
  take the second option rather than half-wiring it.
- **Scope it out explicitly.** Keep `calc_build_fuel_budget` and its tests as the protocol-side
  arithmetic, and state in the plan's "Out of scope" section that the master is not wired in this
  branch and the feature stays dark end to end. Then Task 10 must say the same, so that a
  `needs_index_build` returning `False` for every real batch is the *expected* state and not a
  silent regression someone chases later.

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
- Consumes: the `build-index` and `batch` subcommands (Tasks 4, 6); `calc_db_seed` (Task 1).
- Produces: one build invocation per batch before any nonce is computed.

- [ ] **Step 1: Add the build invocation**

```python
def run_build_index(batch, so_path, ptx_path, results_dir):
    """Build the index once per batch, before any nonce is computed.

    Deliberately passes no nonce: the build process must not be able to derive
    a query set. See docs/ai/specs/2026-08-31-c004-index-build-split-design.md.
    """
    index_path = f"{results_dir}/{batch['id']}/index.blob"
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    settings = json.dumps(batch["settings"], separators=(',', ':'))
    cmd = [
        "docker", "exec", batch["challenge"], "tig-runtime",
        "build-index", settings, batch["rand_hash"], so_path,
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

`build-index` is a **subcommand**, not a flag, and it comes before the positionals — see Task 4's
ruling. Note also that `so_path` is the third positional here; under the flag shape Task 4
rejected, clap would have bound it to the `NONCE` slot.

Call it once per batch before the nonce loop, and pass `--index <index_path>` to
each `run_tig_runtime` call. When `batch` carries no `build_fuel_budget` — every
challenge but c004 — skip the build and pass no `--index`.

Whether any real batch ever carries `build_fuel_budget` is settled by Task 9 Step 3b. If that
step scoped the master out, say so in this task's report: `needs_index_build` returning `False`
for every live batch is then the expected state, and the code below is the landing pad for when
the master is wired.

- [ ] **Step 2: Verify the skip path**

The build must be skipped for challenges that do not support it, or every CPU
challenge breaks. Assert this in `tig-benchmarker/tests/data.py`:

Put `needs_index_build` in **`tig-benchmarker/common/batch.py`**, not in `slave/main.py`.
`common/` is a real package (it has `__init__.py`) that both the master and the slave already
import, and `tests/data.py` puts `tig-benchmarker/` on `sys.path` and imports from it. Importing
`slave.main` instead would drag the slave's entire module graph — `randomname`, `requests`,
`tarfile`, `zlib` — into a unit test of a two-line predicate, for no benefit. `slave/main.py`
imports it from `common.batch`.

```python
    def test_build_is_skipped_without_a_build_fuel_budget(self):
        from common.batch import needs_index_build
        self.assertFalse(needs_index_build({"challenge": "c001"}))
        self.assertTrue(needs_index_build({"challenge": "c004", "build_fuel_budget": 1}))
        # A budget of 0 is a real value, not an absent one: it means "no build
        # fuel", which is different from "this challenge has no build phase".
        self.assertTrue(needs_index_build({"challenge": "c004", "build_fuel_budget": 0}))
```

with, in `tig-benchmarker/common/batch.py`:

```python
def needs_index_build(batch) -> bool:
    """Whether this batch has a build phase.

    Membership, not truthiness. `build_fuel_budget == 0` is a real value -- "no
    build fuel granted" -- and `if batch.get("build_fuel_budget"):` would treat
    it as "this challenge has no build phase", silently skipping the build for a
    c004 batch and producing nonces that never load an index.
    """
    return "build_fuel_budget" in batch
```

- [ ] **Step 3: Run the Python tests**

```bash
docker run --rm -v "$(pwd)/tig-benchmarker:/src" -w /src tig-bench-py python -m unittest tests.data -v
```
Expected: PASS, 5 tests (3 pre-existing, 1 from Task 1, 1 from this task).

- [ ] **Step 4: Commit**

```bash
git status --short
git add tig-benchmarker/common/batch.py tig-benchmarker/slave/main.py tig-benchmarker/tests/data.py
git commit -m "feat(c004): build the index once per batch in the slave"
```

---

## Deviations from the spec, and why

- **The new runtime modes are subcommands, not flags.** The spec writes
  `tig-runtime --build-index <SETTINGS> <RAND_HASH> <BINARY>`; that shape makes clap panic at
  `Command` construction (Task 4 carries the verified error). `build-index` and `batch` are
  subcommands instead. This strengthens D3 rather than weakening it — the build subcommand has no
  nonce argument to smuggle one into — but the spec's "Runtime CLI" section and the enforcement
  table's parse-time row should be updated to match once this lands.
- **Two new `ChallengeConfig` fields are `Option`, not required.** Required fields would break
  deserialization of the live block config. See Task 9.
- **`tig-binary` is in scope after all** (Task 4b). The spec's blast-radius section does not list
  it, but `build_so`'s version script hides every symbol it does not name, so without that change
  the ABI the spec defines is unreachable from any real algorithm.

## Out of scope for this plan

The cross-repo break is deliberately excluded. `pentest-harness` (tig-pentesting)
has a path dependency on `tig-challenges` and calls `generate_instance` and
`evaluate_solution` directly; the `Seeds` change breaks it exactly as the
`evaluate_solution` audit-salt change did. That needs its own paired plan with an
explicit interface-contract table and a unification step, on the model of
`2026-08-27-recall-gated-c004-{monorepo,harness,unification}.md`. **The two trees
are inconsistent from Task 2 until that plan lands.** That is expected, and it is
why Task 2 does not attempt a harness build.
