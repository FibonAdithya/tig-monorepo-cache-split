# Recall-Gated c004 — Monorepo Side (tig-challenges, tig-verifier)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace c004's mean-distance quality with a declared-recall quality audited by a salted subsample, and rank pentest rounds on solve time instead of the `runtime/verifier` ratio.

**Architecture:** `evaluate_solution` measures recall on a salt-selected subsample of queries and returns it scaled to `QUALITY_PRECISION`; the protocol compares that to the player's declaration. The bar `r` lives per-scenario in `ScenarioConfig` and reaches the pentest plugin as `min_recall(track)`, from which `min_quality(track)` is derived. Pentest ranking switches to `solve_ms`, measured by subtracting a null-algorithm baseline from round elapsed time.

**Tech Stack:** Rust (`tig-challenges`, `tig-verifier`, `pentest-harness`), CUDA C (`kernels.cu`, compiled `-arch compute_70` with `--use_fast_math`), Python 3 (`challenges/`, `bundle_scoring.py`, `pentest_core.py`, `scorer_api.py`), pytest.

**Spec:** `~/TIG/tig-pentesting/docs/superpowers/specs/2026-08-27-recall-gated-c004-design.md` (lives in the tig-pentesting repo)

## Where this plan sits

Three plans, split so the monorepo and harness sides can run in parallel:

| Plan | Location | Tasks |
|---|---|---|
| **Monorepo** | `~/TIG/tig-worktrees/vector_search-gan_instance_gen/docs/superpowers/plans/2026-08-27-recall-gated-c004-monorepo.md` | 1–6 |
| **Harness** | `docs/superpowers/plans/2026-08-27-recall-gated-c004-harness.md` (tig-pentesting) | 7–12 |
| **Unification** | `docs/superpowers/plans/2026-08-27-recall-gated-c004-unification.md` (tig-pentesting) | U1–U3 |

**Monorepo Task 1 blocks both plans** — it sets `r`, and it can invalidate the
design outright. Nothing else starts until it reports. After it, monorepo Tasks
2–6 and harness Tasks 7–11 run concurrently; harness Task 12 is written but not
built; unification runs last.

## Interface contract — verbatim in all three plans

These are the seams between the two trees. Each side builds against them
independently; the unification plan verifies they actually meet. **If you change
any row, change it in all three plans and re-run `audit-plan` on each** — two
briefs that disagree about an interface is the defect this section exists to
prevent.

| # | Contract | Owner | Consumer |
|---|---|---|---|
| C1 | `tig_challenges::audit_sampling::sample_query_ids(salt: &[u8; 32], num_queries: u32, num_samples: u32) -> Vec<u32>` — sorted ascending, no duplicates, length `min(num_samples, num_queries)` | monorepo | both |
| C2 | `ScenarioConfig { min_recall: f32, recall_tolerance: f32, audit_samples: u32 }` = `0.95` / `1e-6` / `1_000`; `quality_offset` and `quality_scale` removed | monorepo | harness via C5 |
| C3 | `Challenge::measure_recall(&self, solution, salt: &[u8; 32], module, stream, prop) -> Result<f32>`, in `0.0..=1.0` | monorepo | harness |
| C4 | `Challenge::evaluate_solution(&self, solution, audit_salt: &[u8; 32], module, stream, prop) -> Result<i32>` = `round(recall * 1_000_000)` — **BREAKING, adds a parameter** | monorepo | harness (`pentest-harness/src/main.rs:978`) |
| C5 | `vs-generate` payload gains `"min_recall": f32`, read from `ScenarioConfig` | harness | plugin |
| C6 | `vs-evaluate` payload gains `"recall": f32` — exact over all queries, not sampled | harness | plugin |
| C7 | `tig-verifier --audit-salt <64 hex>` and `test_algorithm --audit-salt <64 hex>`, both defaulting to all-zeros when absent | monorepo | scorer |
| C8 | `recall_audit` kernel signature, as written in monorepo plan Task 4 | monorepo | harness PTX (copied verbatim) |
| C9 | `r = 0.95`, **PROVISIONAL** until monorepo plan Task 1 reports | monorepo | both |

### C4 is breaking, and the two trees are briefly inconsistent by design

`pentest-harness/src/main.rs:978` calls the five-argument `evaluate_solution`
today, and the harness has a path dependency on
`/root/tig-monorepo/tig-challenges`. The moment monorepo Task 5 lands, **the
harness does not compile** until harness Task 12 lands.

That is expected. The harness plan writes `vs-evaluate` against the NEW signature
from the start and does not attempt a Rust build; the first build of the two
trees together is unification Task U1. Every harness task before Task 12 is pure
Python and fully testable on its own, which is what makes the two plans genuinely
parallel.

## Global Constraints

- **`r` is provisional until Task 1 reports.** Every task that writes a bar uses `0.95` as a placeholder value that Task 1 replaces. It is a real value in the code, not a `TODO`.
- **`QUALITY_PRECISION = 1_000_000.`** `quality = round(recall * QUALITY_PRECISION)`; `min_active_quality = round(r * QUALITY_PRECISION)`.
- **The hit test compares distances, never indexes:** `d(q, returned) <= d_min(q) * (1 + tau)`, with `tau = recall_tolerance` from `ScenarioConfig`. Kernels work in squared distance, so the comparison factor is `(1 + tau)^2`.
- **No `sqrt` in any new kernel.** `build_ptx` compiles with `--use_fast_math`, which makes `sqrt` approximate and potentially architecture-dependent.
- **Determinism:** fixed-order reductions only. No atomics into a shared accumulator, no scheduling-dependent ordering.
- **The dev box has no NVIDIA card.** `cargo test -p tig-challenges --features vector_search` cannot even compile locally. Every task below is tagged **[GPU]** or **[local]**; `[GPU]` tasks run on `tig-gpu` via the `gpu-jobs` skill.
- **There is no system pytest.** Run the suite as `make check PYTHON=/home/fibonadithya/TIG/tig-pentesting/.venv/bin/python`.
- **Never `git add -A` / `git add .` / `git commit -a`.** Stage explicit paths; run `git status --short` before every commit.
- **This plan touches ONE tree:** the worktree `~/TIG/tig-worktrees/vector_search-gan_instance_gen` on branch `vector_search/gan_instance_gen`. Do not edit anything under `~/TIG/tig-pentesting` — the harness agent owns that tree and is working in it concurrently.
- **The one exception** is monorepo Task 1's measurement note, which lands in `~/TIG/tig-pentesting/docs/measurements/`. Coordinate that single file, or hand the numbers to the harness agent to commit.
- **Mutation-check every new test** after it passes: break the code under test, confirm the test fails, restore.

---


---

## Task 1: Measure d₂/d₁ before anything else **[GPU] [challenge]**

**This task blocks every other task.** It sets `r`, and it can invalidate the
whole design: if `r` cannot be placed where exact brute force does not trivially
clear it and no approximate method can reach it, the redesign does not work and
the remaining tasks should not be written.

**Files:**
- Create: `~/TIG/tig-worktrees/vector_search-gan_instance_gen/tig-challenges/src/vector_search/two_nn_probe.cu` (throwaway, not committed to the challenge)
- Create: `docs/measurements/2026-08-27-c004-d2-over-d1.md` (in this repo)

- [ ] **Step 1: Write the two-NN kernel**

Adapted from `fixtures/vector_search_1nn/kernels.cu` — one block per query, but
tracking the two smallest squared distances instead of one.

```c
#define PROBE_BLOCK 256
#define PROBE_MAX_DIMS 128

extern "C" __global__ void two_nn(
    const float *__restrict__ queries,
    const float *__restrict__ database,
    const int num_queries,
    const int database_size,
    const int dims,
    float *__restrict__ out_d1_sq,
    float *__restrict__ out_d2_sq)
{
    const int q = blockIdx.x;
    if (q >= num_queries) return;

    __shared__ float s_query[PROBE_MAX_DIMS];
    for (int i = threadIdx.x; i < dims; i += PROBE_BLOCK)
        s_query[i] = queries[(long long)q * dims + i];
    __syncthreads();

    float b1 = 3.0e38f, b2 = 3.0e38f;
    for (int j = threadIdx.x; j < database_size; j += PROBE_BLOCK) {
        const float *cand = database + (long long)j * dims;
        float d = 0.0f;
        for (int k = 0; k < dims; ++k) {
            const float diff = s_query[k] - cand[k];
            d = fmaf(diff, diff, d);
        }
        if (d < b1)      { b2 = b1; b1 = d; }
        else if (d < b2) { b2 = d; }
    }

    __shared__ float s1[PROBE_BLOCK], s2[PROBE_BLOCK];
    s1[threadIdx.x] = b1; s2[threadIdx.x] = b2;
    __syncthreads();
    for (int stride = PROBE_BLOCK / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            // Merge two sorted pairs, keeping the two smallest overall.
            float o1 = s1[threadIdx.x + stride], o2 = s2[threadIdx.x + stride];
            float m1 = s1[threadIdx.x],          m2 = s2[threadIdx.x];
            float n1 = m1 < o1 ? m1 : o1;
            float hi = m1 < o1 ? o1 : m1;
            float lo = m1 < o1 ? m2 : o2;
            float n2 = lo < hi ? lo : hi;
            s1[threadIdx.x] = n1; s2[threadIdx.x] = n2;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { out_d1_sq[q] = s1[0]; out_d2_sq[q] = s2[0]; }
}
```

- [ ] **Step 2: Run it against a real `s=sift_128` instance**

Reuse the `fixtures/vector_search_1nn/` submission path — `/compile` with this
`.cu` and a driver `.rs` that writes `out_d1_sq`/`out_d2_sq` to the host, then
prints them. Run over at least 3 seeds so the spread is instance-to-instance,
not one draw.

- [ ] **Step 3: Write the measurement note**

`docs/measurements/2026-08-27-c004-d2-over-d1.md` must record, per seed:
mean and sd of `sqrt(d1_sq)`; mean, median, p1 and p99 of `sqrt(d2_sq/d1_sq)`;
and the fraction of queries where `d2/d1 - 1 < 0.01`.

- [ ] **Step 4: Choose `r` and record the reasoning**

State in the note which `r` is chosen and what it selects for. If the
distribution shows that no `r` separates a tiled exact kernel from a plausible
approximate method, say so — that is a valid and important outcome, and it stops
this plan.

- [ ] **Step 5: Commit the note**

```bash
git status --short
git add docs/measurements/2026-08-27-c004-d2-over-d1.md
git commit -m "docs(measurements): d2/d1 distribution for c004 s=sift_128, and the r it implies"
```

---

## Task 2: Salt-derived query sampling, host side **[local] [challenge]**

Pure Rust, no CUDA — this is the one challenge-side task testable on the dev box.

**Files:**
- Create: `tig-challenges/src/audit_sampling.rs`
- Modify: `tig-challenges/src/lib.rs`

**Interfaces:**
- Produces: `tig_challenges::audit_sampling::sample_query_ids(salt: &[u8; 32], num_queries: u32, num_samples: u32) -> Vec<u32>` — sorted ascending, no duplicates, length `min(num_samples, num_queries)`.

**Why a new ungated module rather than `vector_search/mod.rs`.** `lib.rs:194`
gates `pub mod vector_search` on `#[cfg(feature = "c004")]`, and
`c004 = ["cudarc"]` pulls a CUDA toolkit through cudarc's
`cuda-version-from-build-system`. There is no `nvcc` and no `/usr/local/cuda` on
the dev box, so anything inside that module is GPU-only. An ungated module
compiles with default features and makes this the one challenge-side task
testable locally. Declare it in `lib.rs` with no `#[cfg]`:

```rust
pub mod audit_sampling;
```

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn sampling_is_deterministic_in_the_salt() {
    let salt = [7u8; 32];
    assert_eq!(sample_query_ids(&salt, 7000, 1000), sample_query_ids(&salt, 7000, 1000));
}

#[test]
fn a_different_salt_gives_a_different_sample() {
    let a = sample_query_ids(&[1u8; 32], 7000, 1000);
    let b = sample_query_ids(&[2u8; 32], 7000, 1000);
    assert_ne!(a, b, "sample must depend on the salt, or the audit is fixed");
}

#[test]
fn sample_has_no_duplicates_and_is_in_range() {
    let s = sample_query_ids(&[3u8; 32], 7000, 1000);
    assert_eq!(s.len(), 1000);
    let mut seen = std::collections::HashSet::new();
    for &i in &s {
        assert!(i < 7000, "index {} out of range", i);
        assert!(seen.insert(i), "duplicate index {} — a repeated query is one \
            audited query counted twice, which biases recall toward it", i);
    }
}

#[test]
fn asking_for_more_samples_than_queries_yields_every_query_once() {
    let s = sample_query_ids(&[4u8; 32], 10, 1000);
    assert_eq!(s.len(), 10);
}

#[test]
fn sample_is_sorted_ascending() {
    // The kernel reads sample_query_ids in order; sorted order makes the
    // query-vector reads sequential rather than scattered.
    let s = sample_query_ids(&[5u8; 32], 7000, 1000);
    let mut sorted = s.clone();
    sorted.sort_unstable();
    assert_eq!(s, sorted);
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test -p tig-challenges --lib audit_sampling -- --nocapture`
(no `--features`, so `vector_search` and cudarc are not compiled)
Expected: FAIL, `cannot find function 'sample_query_ids'`

- [ ] **Step 3: Implement**

```rust
use rand::{rngs::StdRng, Rng, SeedableRng};

/// Pick which queries the recall audit checks.
///
/// Seeded from the audit salt, NOT from the instance seed. The instance seed
/// reaches the algorithm — `Challenge.seed` is public, and even if it were not,
/// `tig-runtime` takes RAND_HASH and NONCE on argv and the algorithm runs
/// in-process, so it can recompute the seed from /proc/self/cmdline. An
/// algorithm that knows the audit set brute-forces only those queries for
/// `num_samples / num_queries` of the honest work.
///
/// `StdRng::seed_from_u64` matches how tig-protocol draws `sampled_nonces`.
pub fn sample_query_ids(
    salt: &[u8; 32],
    num_queries: u32,
    num_samples: u32,
) -> Vec<u32> {
    let n = num_samples.min(num_queries) as usize;
    let mut rng = StdRng::seed_from_u64(u64::from_le_bytes(
        salt[0..8].try_into().expect("salt is 32 bytes"),
    ));
    let mut all: Vec<u32> = (0..num_queries).collect();
    // Partial Fisher-Yates, written out rather than SliceRandom::shuffle:
    // tig-challenges pins rand with `default-features = false` and only the
    // `std_rng` / `small_rng` features, so the `seq` traits are not guaranteed
    // present. `gen_range` needs only `Rng`.
    //
    // `all.len() - i` is at least 1 for every i < n <= all.len(), so the range
    // is never empty and gen_range cannot panic.
    for i in 0..n {
        let j = i + rng.gen_range(0..(all.len() - i));
        all.swap(i, j);
    }
    all.truncate(n);
    all.sort_unstable();
    all
}
```

- [ ] **Step 4: Run to verify they pass**

Run: `cargo test -p tig-challenges --lib audit_sampling -- --nocapture`
Expected: PASS (5 tests)

- [ ] **Step 5: Mutation-check**

Replace `StdRng::seed_from_u64(...)` with `StdRng::seed_from_u64(0)` and confirm
`a_different_salt_gives_a_different_sample` fails. Restore. Then drop the
`.sort_unstable()` and confirm `sample_is_sorted_ascending` fails. Restore.

- [ ] **Step 6: Commit**

```bash
git status --short
git add tig-challenges/src/audit_sampling.rs tig-challenges/src/lib.rs
git commit -m "feat(c004): derive the recall audit sample from an audit salt"
```

---

## Task 3: `ScenarioConfig` carries the recall parameters **[local] [challenge]**

**Files:**
- Modify: `tig-challenges/src/vector_search/scenarios.rs`

**Interfaces:**
- Produces: `ScenarioConfig { min_recall: f32, recall_tolerance: f32, audit_samples: u32, .. }`; `quality_offset` and `quality_scale` removed.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn sift_128_declares_its_recall_bar() {
    let c = ScenarioConfig::from(Scenario::SIFT_128);
    // Asserted against the value, not `is_finite()` — a bar that silently
    // defaulted to 0.0 would qualify every solution including all-zeros.
    assert_eq!(c.min_recall, 0.95);
    assert_eq!(c.audit_samples, 1_000);
    assert_eq!(c.recall_tolerance, 1e-6);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p tig-challenges --lib vector_search::scenarios -- --nocapture`
Expected: FAIL, `no field 'min_recall' on type 'ScenarioConfig'`

- [ ] **Step 3: Implement**

**Add the new fields; do NOT delete the old ones yet.** `mod.rs:297` still reads
`config.quality_offset` and `config.quality_scale` until Task 5 replaces
`evaluate_solution`. Deleting them here makes this task's own Step 4 fail to
compile. Task 5 removes them.

In the `ScenarioConfig` struct, add:

```rust
    /// Recall@1 a solution must declare to qualify. Per scenario because the
    /// achievable recall/speed frontier depends on the corpus.
    /// PROVISIONAL: 0.95 pending docs/measurements/2026-08-27-c004-d2-over-d1.md.
    pub min_recall: f32,
    /// A returned vector counts as a hit when its distance is within this
    /// relative tolerance of the true minimum. Absorbs cross-architecture float
    /// noise and makes an equidistant alternative a hit by construction.
    pub recall_tolerance: f32,
    /// Queries the audit checks. Verification cost is linear in this.
    pub audit_samples: u32,
```

In the `SIFT_128` arm, keep the two quality constants for now and add:

```rust
                min_recall: 0.95,
                recall_tolerance: 1e-6,
                audit_samples: 1_000,
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p tig-challenges --lib vector_search::scenarios -- --nocapture`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git status --short
git add tig-challenges/src/vector_search/scenarios.rs
git commit -m "feat(c004): put the recall bar, tolerance and audit size in ScenarioConfig"
```

---

## Task 4: The `recall_audit` kernel, correctness first **[GPU] [challenge]**

Correct before fast. Task 6 optimises it against a measured target; this task
establishes what "correct" means so that optimisation has a gate to pass.

**Files:**
- Modify: `tig-challenges/src/vector_search/kernels.cu`
- Modify: `tig-challenges/src/vector_search/mod.rs`

**Interfaces:**
- Consumes: `sample_query_ids` (Task 2), `ScenarioConfig::{min_recall, recall_tolerance, audit_samples}` (Task 3).
- Produces: `Challenge::measure_recall(&self, solution: &Solution, salt: &[u8; 32], module: Arc<CudaModule>, stream: Arc<CudaStream>, prop: &cudaDeviceProp) -> Result<f32>` returning recall in `0.0..=1.0`.

- [ ] **Step 1: Write the kernel**

Append to `kernels.cu`. Note there is no index tie-breaking anywhere: the audit
needs only the minimum *distance*, never which index achieved it, so the tie
problem that shapes `fixtures/vector_search_1nn/kernels.cu` does not arise here.

```c
#define AUDIT_BLOCK 256
#define AUDIT_MAX_DIMS 128

// One block per audited query. Writes hits[s] = 1 when the submitted answer for
// query sample_query_ids[s] is within tolerance of the true nearest neighbour.
//
// tolerance_sq is (1 + tau)^2: the comparison runs in squared distance so that
// no sqrt appears, and --use_fast_math makes sqrt approximate.
extern "C" __global__ void recall_audit(
    const uint32_t vector_dims,
    const uint32_t database_size,
    const uint32_t num_samples,
    const float *__restrict__ query_vectors,
    const float *__restrict__ database_vectors,
    const size_t *__restrict__ solution_indexes,
    const uint32_t *__restrict__ sample_query_ids,
    const float tolerance_sq,
    uint32_t *__restrict__ hits,
    int *__restrict__ error_flag)
{
    const int s = blockIdx.x;
    if (s >= num_samples) return;
    const uint32_t q = sample_query_ids[s];

    __shared__ float s_query[AUDIT_MAX_DIMS];
    for (int i = threadIdx.x; i < vector_dims; i += AUDIT_BLOCK) {
        s_query[i] = query_vectors[(long long)q * vector_dims + i];
    }
    __syncthreads();

    __shared__ float s_returned;
    if (threadIdx.x == 0) {
        const size_t idx = solution_indexes[q];
        if (idx >= database_size) {
            *error_flag = 1;
            s_returned = 3.0e38f;
        } else {
            const float *cand = database_vectors + idx * vector_dims;
            float d = 0.0f;
            for (int k = 0; k < vector_dims; ++k) {
                const float diff = s_query[k] - cand[k];
                d = fmaf(diff, diff, d);
            }
            s_returned = d;
        }
    }

    float best = 3.0e38f;
    for (int j = threadIdx.x; j < database_size; j += AUDIT_BLOCK) {
        const float *cand = database_vectors + (long long)j * vector_dims;
        float d = 0.0f;
        for (int k = 0; k < vector_dims; ++k) {
            const float diff = s_query[k] - cand[k];
            d = fmaf(diff, diff, d);
        }
        if (d < best) best = d;
    }

    __shared__ float s_best[AUDIT_BLOCK];
    s_best[threadIdx.x] = best;
    __syncthreads();
    for (int stride = AUDIT_BLOCK / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float other = s_best[threadIdx.x + stride];
            if (other < s_best[threadIdx.x]) s_best[threadIdx.x] = other;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        hits[s] = (s_returned <= s_best[0] * tolerance_sq) ? 1u : 0u;
    }
}
```

- [ ] **Step 2: Write the host wrapper**

In `mod.rs`, alongside `evaluate_average_distance`:

```rust
    /// Recall@1 measured on the salt-selected subsample.
    ///
    /// This function MEASURES; it never compares against `min_recall` and never
    /// sees a declaration. The protocol does the comparing. Three callers share
    /// it: a benchmarker estimating its own recall before declaring, a verifier
    /// auditing with the protocol's salt, and the pentest reading the result
    /// straight through as quality.
    pub fn measure_recall(
        &self,
        solution: &Solution,
        salt: &[u8; 32],
        module: Arc<CudaModule>,
        stream: Arc<CudaStream>,
        _prop: &cudaDeviceProp,
    ) -> Result<f32> {
        if solution.indexes.len() != self.num_queries as usize {
            return Err(anyhow!(
                "Invalid number of indexes. Expected: {}, Actual: {}",
                self.num_queries,
                solution.indexes.len()
            ));
        }
        let config = ScenarioConfig::from(self.scenario);
        let ids = sample_query_ids(salt, self.num_queries, config.audit_samples);
        let num_samples = ids.len() as u32;

        let kernel = module.load_function("recall_audit")?;
        let d_indexes = stream.memcpy_stod(&solution.indexes)?;
        let d_ids = stream.memcpy_stod(&ids)?;
        let mut d_hits = stream.alloc_zeros::<u32>(ids.len())?;
        let mut d_err = stream.alloc_zeros::<u32>(1)?;

        let tol = 1.0f32 + config.recall_tolerance;
        let tolerance_sq = tol * tol;

        unsafe {
            stream
                .launch_builder(&kernel)
                .arg(&self.vector_dims)
                .arg(&self.database_size)
                .arg(&num_samples)
                .arg(&self.d_query_vectors)
                .arg(&self.d_database_vectors)
                .arg(&d_indexes)
                .arg(&d_ids)
                .arg(&tolerance_sq)
                .arg(&mut d_hits)
                .arg(&mut d_err)
                .launch(LaunchConfig {
                    grid_dim: (num_samples, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        stream.synchronize()?;

        if stream.memcpy_dtov(&d_err)?[0] != 0 {
            return Err(anyhow!("Invalid index in solution"));
        }
        let hits = stream.memcpy_dtov(&d_hits)?;
        // u32 accumulator: 1,000 samples cannot overflow, but summing into u32
        // rather than the element type keeps it correct if audit_samples grows.
        let total: u32 = hits.iter().sum();
        Ok(total as f32 / num_samples as f32)
    }
```

- [ ] **Step 3: Write the GPU correctness tests**

```rust
#[test]
fn exact_1nn_measures_recall_1() {
    // The fixture's answer is known: recall against itself must be exactly 1.0.
    let (challenge, module, stream, prop) = gpu_instance(1);
    let exact = brute_force_1nn(&challenge, module.clone(), stream.clone());
    let r = challenge.measure_recall(&exact, &[9u8; 32], module, stream, &prop).unwrap();
    assert_eq!(r, 1.0);
}

#[test]
fn all_zeros_measures_recall_near_0() {
    let (challenge, module, stream, prop) = gpu_instance(1);
    let sol = Solution { indexes: vec![0; challenge.num_queries as usize] };
    let r = challenge.measure_recall(&sol, &[9u8; 32], module, stream, &prop).unwrap();
    assert!(r < 0.01, "all-zeros scored recall {}", r);
}

#[test]
fn corrupting_a_sampled_query_drops_recall_by_exactly_one_sample() {
    // Pick a query that IS in the sample, so the expected drop is exact. An
    // earlier "changed by one step OR not at all" form of this assertion passed
    // when recall never moved -- i.e. when the audit was entirely broken.
    let (challenge, module, stream, prop) = gpu_instance(1);
    let salt = [9u8; 32];
    let config = ScenarioConfig::from(challenge.scenario);
    let ids = tig_challenges::audit_sampling::sample_query_ids(
        &salt, challenge.num_queries, config.audit_samples);
    let victim = ids[0] as usize;

    let mut sol = brute_force_1nn(&challenge, module.clone(), stream.clone());
    let before = challenge.measure_recall(&sol, &salt, module.clone(), stream.clone(), &prop).unwrap();
    assert_eq!(before, 1.0);

    // A different index in 128 dims over 700k vectors is not within 1e-6 of the
    // true minimum except by astronomical coincidence; if this ever flakes,
    // that tie is the reason.
    sol.indexes[victim] = (sol.indexes[victim] + 1) % challenge.database_size as usize;
    let after = challenge.measure_recall(&sol, &salt, module, stream, &prop).unwrap();
    let step = 1.0f32 / config.audit_samples as f32;
    assert!((before - after - step).abs() < 1e-6,
        "recall moved by {}, expected exactly {}", before - after, step);
}

#[test]
fn corrupting_an_unsampled_query_does_not_move_recall() {
    // The other half, and the one that proves the audit is a SUBSAMPLE: if it
    // silently looked at every query, this fails.
    let (challenge, module, stream, prop) = gpu_instance(1);
    let salt = [9u8; 32];
    let config = ScenarioConfig::from(challenge.scenario);
    let ids: std::collections::HashSet<u32> = tig_challenges::audit_sampling::sample_query_ids(
        &salt, challenge.num_queries, config.audit_samples).into_iter().collect();
    let victim = (0..challenge.num_queries).find(|q| !ids.contains(q))
        .expect("audit_samples < num_queries, so some query is unsampled") as usize;

    let mut sol = brute_force_1nn(&challenge, module.clone(), stream.clone());
    let before = challenge.measure_recall(&sol, &salt, module.clone(), stream.clone(), &prop).unwrap();
    sol.indexes[victim] = (sol.indexes[victim] + 1) % challenge.database_size as usize;
    let after = challenge.measure_recall(&sol, &salt, module, stream, &prop).unwrap();
    assert_eq!(before, after);
}

#[test]
fn an_out_of_range_index_is_an_error_not_a_miss() {
    // A miss scores badly; an invalid index must be rejected. Conflating them
    // lets a malformed solution look like a merely bad one.
    let (challenge, module, stream, prop) = gpu_instance(1);
    let mut sol = Solution { indexes: vec![0; challenge.num_queries as usize] };
    sol.indexes[0] = challenge.database_size as usize;
    assert!(challenge.measure_recall(&sol, &[9u8; 32], module, stream, &prop).is_err());
}
```

`gpu_instance(seed)` and `brute_force_1nn` are test helpers: the former calls
`Challenge::generate_instance` with a CUDA context, the latter runs the
`fixtures/vector_search_1nn` kernel. Write them in the same `#[cfg(test)]` module.

- [ ] **Step 4: Run on tig-gpu**

Run: `cargo test -p tig-challenges --features c004 --lib vector_search -- --test-threads=1 --nocapture`
Expected: PASS (5 tests). `--test-threads=1` because each test allocates the
358 MB database on a 12 GB card.

- [ ] **Step 5: Mutation-check**

Change `hits[s] = (s_returned <= s_best[0] * tolerance_sq)` to `>=` and confirm
`exact_1nn_measures_recall_1` and `all_zeros_measures_recall_near_0` both fail.
Restore. Then change `tolerance_sq` to `1.0e6` and confirm
`all_zeros_measures_recall_near_0` fails. Restore.

- [ ] **Step 6: Commit**

```bash
git status --short
git add tig-challenges/src/vector_search/kernels.cu tig-challenges/src/vector_search/mod.rs
git commit -m "feat(c004): measure recall@1 on a salt-selected subsample"
```

---

## Task 5: `evaluate_solution` returns recall; salt reaches the verifier **[GPU] [challenge]**

**Files:**
- Modify: `tig-challenges/src/vector_search/mod.rs`
- Modify: `tig-challenges/src/hypergraph/mod.rs`
- Modify: `tig-challenges/src/neuralnet_optimizer/mod.rs`
- Modify: `tig-verifier/src/main.rs`
- Modify: `tig-verifier/Cargo.toml`
- Modify: `scripts/test_algorithm`

**Interfaces:**
- Consumes: `Challenge::measure_recall` (Task 4).
- Produces: `evaluate_solution(&self, solution, audit_salt: &[u8; 32], module, stream, prop) -> Result<i32>` on all three CUDA challenges; `tig-verifier --audit-salt <64-hex>`; `test_algorithm --audit-salt <64-hex>`.

The cpu arm of `dispatch_challenge!` calls `evaluate_solution(&solution)` and the
cuda arm calls `evaluate_solution(&solution, module, stream, &prop)` — the
signatures already differ, so the salt goes on the cuda arm only. The five CPU
challenges are untouched.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn quality_is_recall_scaled_to_quality_precision() {
    let (challenge, module, stream, prop) = gpu_instance(1);
    let exact = brute_force_1nn(&challenge, module.clone(), stream.clone());
    let q = challenge.evaluate_solution(&exact, &[9u8; 32], module.clone(), stream.clone(), &prop).unwrap();
    assert_eq!(q, QUALITY_PRECISION, "exact 1-NN must score full quality");

    let zeros = Solution { indexes: vec![0; challenge.num_queries as usize] };
    let qz = challenge.evaluate_solution(&zeros, &[9u8; 32], module, stream, &prop).unwrap();
    let config = ScenarioConfig::from(challenge.scenario);
    let bar = (config.min_recall * QUALITY_PRECISION as f32).round() as i32;
    assert!(qz < bar, "all-zeros scored {} against a bar of {}", qz, bar);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p tig-challenges --features c004 --lib quality_is_recall -- --test-threads=1`
Expected: FAIL — `evaluate_solution` takes 4 arguments, 5 supplied.

- [ ] **Step 3: Replace `evaluate_solution` in `vector_search/mod.rs`**

```rust
    conditional_pub!(
        fn evaluate_solution(
            &self,
            solution: &Solution,
            audit_salt: &[u8; 32],
            module: Arc<CudaModule>,
            stream: Arc<CudaStream>,
            prop: &cudaDeviceProp,
        ) -> Result<i32> {
            let recall = self.measure_recall(solution, audit_salt, module, stream, prop)?;
            Ok((recall * QUALITY_PRECISION as f32).round() as i32)
        }
    );
```

Delete `evaluate_average_distance`'s use in the quality path but keep the
function: `vs-evaluate` still reports `avg_distance` as a diagnostic.

**Now** delete `quality_offset` and `quality_scale` from `ScenarioConfig` and
from the `SIFT_128` arm in `scenarios.rs` — Task 3 deliberately left them so the
tree kept compiling. Nothing reads them after this edit; `cargo build` is the
check.

- [ ] **Step 4: Add the ignored parameter to the two sibling CUDA challenges**

In `hypergraph/mod.rs` and `neuralnet_optimizer/mod.rs`, add `_audit_salt: &[u8; 32],`
as the second parameter of `evaluate_solution`. Bodies are unchanged.

- [ ] **Step 5: Plumb the salt through `tig-verifier`**

`hex` is **not** currently a `tig-verifier` dependency — `grep hex
tig-verifier/Cargo.toml` returns nothing — so add it first, matching the version
`tig-utils` already pins:

```toml
hex = "0.4.3"
```

Add to `cli()`:

```rust
        .arg(arg!(--"audit-salt" [AUDIT_SALT] "32-byte audit salt as 64 hex chars")
            .value_parser(clap::value_parser!(String)))
```

Thread it into `verify_solution` and decode once, defaulting to all-zeros when
absent so the five CPU lanes and local debugging keep working:

```rust
    let audit_salt: [u8; 32] = match audit_salt_hex {
        Some(h) => hex::decode(&h)
            .map_err(|e| anyhow::anyhow!("--audit-salt is not hex: {}", e))?
            .try_into()
            .map_err(|_| anyhow::anyhow!("--audit-salt must decode to exactly 32 bytes"))?,
        None => [0u8; 32],
    };
```

In the cuda arm of `dispatch_challenge!`, pass `&audit_salt` as the second
argument to `evaluate_solution`.

- [ ] **Step 6: Plumb the salt through `test_algorithm`**

Add `parser.add_argument("--audit-salt", type=str, default=None, ...)`, and in
`cmd2`:

```python
            if args.audit_salt is not None:
                cmd2 += ["--audit-salt", args.audit_salt]
```

- [ ] **Step 7: Run to verify it passes**

Run: `cargo test -p tig-challenges --features c004 --lib -- --test-threads=1`
Expected: PASS. Then `cargo build -p tig-verifier --features c004` — must compile.

- [ ] **Step 8: Confirm the siblings still build**

Run: `cargo build -p tig-challenges --features c005 && cargo build -p tig-challenges --features c006`
Expected: both succeed. This is the check that the cuda-arm change did not break a sibling GPU lane.

- [ ] **Step 9: Mutation-check**

Change `(recall * QUALITY_PRECISION as f32).round()` to `(recall).round()` and
confirm `quality_is_recall_scaled_to_quality_precision` fails. Restore.

- [ ] **Step 10: Commit**

```bash
git status --short
git add tig-challenges/src/vector_search/mod.rs tig-challenges/src/vector_search/scenarios.rs \
        tig-challenges/src/hypergraph/mod.rs tig-challenges/src/neuralnet_optimizer/mod.rs \
        tig-verifier/src/main.rs tig-verifier/Cargo.toml scripts/test_algorithm
git commit -m "feat(c004): quality is audited recall; salt reaches the verifier"
```

---

## Task 6: Make the audit kernel fast enough **[GPU] [challenge]**

Task 4's kernel is one block per audited query, so each of 1,000 blocks streams
the whole 358 MB database: ~358 GB of reads, an estimated ~1 s. That is slower
than the 3.1 s generation is tolerable next to, and far past the "much cheaper
than solve" property the whole design rests on. This task closes that gap.

**Files:**
- Modify: `tig-challenges/src/vector_search/kernels.cu`

- [ ] **Step 1: Write the failing performance test**

```rust
#[test]
fn audit_is_much_cheaper_than_a_naive_solve() {
    // The design property: verification must be far cheaper than solving.
    // The naive full-database scan for 7,000 queries measured 27,000 ms
    // (docs/measurements/2026-08-26-c004-lane-probe.md); a 1,000-query audit
    // that is not tiled costs ~1/7 of that. 150 ms is a deliberately loose
    // ceiling that a tiled kernel clears comfortably and an untiled one cannot.
    let (challenge, module, stream, prop) = gpu_instance(1);
    let sol = Solution { indexes: vec![0; challenge.num_queries as usize] };
    // Warm the context so JIT and allocation are not in the measurement.
    let _ = challenge.measure_recall(&sol, &[1u8; 32], module.clone(), stream.clone(), &prop).unwrap();

    let mut ms = u128::MAX;
    for i in 0..3u8 {
        let t = std::time::Instant::now();
        let _ = challenge.measure_recall(&sol, &[i; 32], module.clone(), stream.clone(), &prop).unwrap();
        ms = ms.min(t.elapsed().as_millis());
    }
    assert!(ms < 150, "audit took {} ms; the untiled kernel is still in place", ms);
}
```

Hold a GPU claim for the whole task (`gpu-claim`; see the `gpu-jobs` skill). The
box runs a gpuq reaper with `kill_orphan_cuda = true` on a 60 s cycle, and
`docs/measurements/2026-08-26-c004-lane-probe.md` records one unexplained SIGKILL
under exactly these conditions. An unclaimed run can be slowed or killed
mid-measurement, which would read as a failed performance gate.

The test takes the **best of three** timings for the same reason: the gate is
separating ~1,000 ms from ~30 ms, so a single scheduling hiccup must not fail it,
and best-of-three cannot turn a genuinely slow kernel into a passing one.

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p tig-challenges --features c004 --lib audit_is_much_cheaper -- --test-threads=1 --nocapture`
Expected: FAIL, reporting roughly 1,000 ms.

- [ ] **Step 3: Tile the kernel over queries**

Restructure so each block stages `TQ` audited queries in shared memory and
streams the database once for all of them, cutting database traffic by a factor
of `TQ`. Threads read database rows **cooperatively and coalesced** into shared
memory rather than each thread walking its own 512-byte row, which is the defect
that makes `fixtures/vector_search_1nn/kernels.cu` run at under 1% of achievable
rate.

Start at `TQ = 16` (16 running minima per thread in registers, 8 KB of staged
queries per block). If occupancy is the limit, try `TQ = 8` and `TQ = 32` and
keep the fastest that passes Step 4. Record the chosen value and the measured
time in a comment above the kernel.

Constraints that must survive the rewrite: fixed-order reduction, no `sqrt`,
squared-distance comparison, `hits[s]` written by exactly one thread.

- [ ] **Step 4: Run to verify correctness AND performance**

Run: `cargo test -p tig-challenges --features c004 --lib vector_search -- --test-threads=1 --nocapture`
Expected: PASS — all four Task 4 correctness tests plus the performance test.
The correctness tests are the guard here: a tiling bug that makes the kernel fast
and wrong fails them.

- [ ] **Step 5: Commit**

```bash
git status --short
git add tig-challenges/src/vector_search/kernels.cu
git commit -m "perf(c004): tile the recall audit over queries"
```

---

## Self-review notes

Covers spec Decisions 1, 2, 3, 3a (challenge half), 4 and the spec's blocking
Task 1. Does **not** cover Decision 5 (`solve_ms` ranking) or Decision 7
(`min_recall`/`min_quality` in the plugin) — those are the harness plan.

**Deliberately out of scope:** the tig-protocol change (`BenchmarkDetails`
gaining `audit_salt`, drawn where `sampled_nonces` is drawn at
`tig-protocol/src/contracts/benchmarks.rs:238`). This plan takes the salt only as
far as `tig-verifier`'s CLI, which is what the pentest lane needs. The spec's
open question — where `submit_benchmark`'s `seed` originates and whether a
benchmarker can predict it — must be settled before that wiring is written.
