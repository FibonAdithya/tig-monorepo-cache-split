# Per-scenario GAN tracks (Milestone 1: SIFT-only) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `vector_search`'s size-based track (`Track { n_queries }`) with a corpus-based one (`Track { s: Scenario }`), shipping a single `SIFT_128` scenario that reuses the already-trained v1 generator.

**Architecture:** A new `scenarios.rs` holds a `Scenario` enum and a `ScenarioConfig` struct carrying `n_queries`, `database_size`, `vector_dims`, the weight blob, and the two calibration constants. `mod.rs` derives everything from `ScenarioConfig::from(track.s)` instead of `track.n_queries`. `Challenge` gains a `scenario` field so `evaluate_solution` can reach per-scenario calibration constants. `kernels.cu` is not touched — that is load-bearing, see Global Constraints.

**Tech Stack:** Rust (`nightly-2025-02-10`), CUDA 12.6, `cudarc`, `anyhow`, `paste`. Tests via `cargo test -p tig-challenges --features vector_search`.

**Spec:** `docs/superpowers/specs/2026-08-25-per-scenario-gan-tracks-design.md`

## Global Constraints

- **`tig-challenges/src/vector_search/kernels.cu` MUST NOT be modified.** Generation kernels are sourced from each *algorithm's* PTX (`tig-runtime/src/main.rs:196-217`), so changing them forces a network-wide resubmit. Every later scenario depends on this staying frozen.
- **Never `git push`.** TIG is a live competition. Local commits only.
- Work in `/home/fibonadithya/TIG/tig-worktrees/vector_search-gan_instance_gen`, branch `vector_search/gan_instance_gen`. Do not `cd` out of it.
- **`vector_search` cannot compile on the local machine** — no CUDA toolkit. All builds and tests run on `tig-gpu`. See Build & Test Recipe.
- The wire form of a track is lowercase snake_case: `s=sift_128`.
- Existing 8 generator tests must stay green, unmodified, throughout.
- Calibration constants for `SIFT_128`: `QUALITY_OFFSET = 1.616563`, `QUALITY_SCALE = 6.399004`. `n_queries = 7000`, `database_size = 700000`, `vector_dims = 128`.

## Build & Test Recipe

Every "Run tests" step means this, from the local worktree:

```bash
# 1. sync the worktree to tig-gpu (bundle transfer; the ~200 KB cap in older
#    notes does NOT apply — 23 MB moves in ~13s)
cd /home/fibonadithya/TIG/tig-worktrees/vector_search-gan_instance_gen
git bundle create /tmp/wip.bundle HEAD
scp -o BatchMode=yes /tmp/wip.bundle tig-gpu:/workspace/wip.bundle
ssh -o BatchMode=yes tig-gpu 'cd /workspace && rm -rf tig-bench && git clone -q /workspace/wip.bundle tig-bench'

# 2. build and test
ssh -o BatchMode=yes tig-gpu '
export PATH=/usr/local/cuda-12.6/bin:$HOME/.cargo/bin:$PATH
export CUDA_PATH=/usr/local/cuda-12.6 RUSTUP_TOOLCHAIN=nightly-2025-02-10
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:/usr/local/cuda-12.6/lib64/stubs:$(rustc +nightly-2025-02-10 --print target-libdir):$LD_LIBRARY_PATH
cd /workspace/tig-bench && cargo +nightly-2025-02-10 test -p tig-challenges --features vector_search 2>&1 | tail -30'
```

**Tasks 1-3 need no working GPU** — they are CPU-side tests, and compilation needs only the toolkit. Task 4 needs a live device; as of 2026-08-25 the card is hardware-faulted (`nvidia-smi`: *No devices were found*), so Task 4 is **blocked, not skipped**.

---

### Task 1: `Scenario` enum and `ScenarioConfig`

**Files:**
- Create: `tig-challenges/src/vector_search/scenarios.rs`
- Modify: `tig-challenges/src/vector_search/mod.rs:9-10` (add `mod scenarios;`)

**Interfaces:**
- Consumes: nothing.
- Produces: `pub enum Scenario { SIFT_128 }` implementing `Clone, Copy, Debug, Eq, PartialEq, Display, FromStr<Err = anyhow::Error>`; `pub struct ScenarioConfig { n_queries: u32, database_size: u32, vector_dims: usize, weights: &'static [u8], quality_offset: f64, quality_scale: f64 }`; `impl From<Scenario> for ScenarioConfig`.

Note: `Display` and `FromStr` are **not optional decoration**. `impl_kv_string_serde` (`tig-challenges/src/lib.rs:15-88`) is hand-written, not derive-based: it serializes with `format!("{}={}", field, self.field)` and deserializes with `.parse::<$ty>()`. Without both traits the `Track` in Task 3 will not compile.

- [ ] **Step 1: Write the failing tests**

Create `tig-challenges/src/vector_search/scenarios.rs` with only the test module at first:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn scenario_display_is_lowercase_snake_case() {
        // The literal string matters: it is what the protocol puts in
        // settings.track_id. A round-trip test alone would pass for any
        // self-consistent encoding, including one the protocol cannot emit.
        assert_eq!(Scenario::SIFT_128.to_string(), "sift_128");
    }

    #[test]
    fn scenario_from_str_is_case_insensitive() {
        assert_eq!(Scenario::from_str("sift_128").unwrap(), Scenario::SIFT_128);
        assert_eq!(Scenario::from_str("SIFT_128").unwrap(), Scenario::SIFT_128);
    }

    #[test]
    fn scenario_from_str_rejects_unknown() {
        let err = Scenario::from_str("glove_300").unwrap_err();
        assert!(
            err.to_string().contains("glove_300"),
            "error should name the offending input, got: {}",
            err
        );
    }

    #[test]
    fn sift_config_matches_calibrated_constants() {
        let c = ScenarioConfig::from(Scenario::SIFT_128);
        assert_eq!(c.n_queries, 7_000);
        assert_eq!(c.database_size, 700_000);
        assert_eq!(c.vector_dims, 128);
        assert_eq!(c.quality_offset, 1.616563);
        assert_eq!(c.quality_scale, 6.399004);
        assert!(!c.weights.is_empty());
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run the Build & Test Recipe.
Expected: FAIL to compile — `cannot find type Scenario in this scope`.

- [ ] **Step 3: Write the implementation**

Prepend to `tig-challenges/src/vector_search/scenarios.rs`:

```rust
use anyhow::{anyhow, Result};

/// One scenario per real embedding corpus. Tracks are one-to-one with
/// scenarios, so adding a variant adds a track.
///
/// Adding a variant is a runtime-side change only: generation kernels come
/// from each algorithm's PTX, while blobs and config come from the
/// runtime's own tig-challenges. Do not change kernels.cu to add one.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(non_camel_case_types)]
pub enum Scenario {
    SIFT_128,
}

pub struct ScenarioConfig {
    pub n_queries: u32,
    pub database_size: u32,
    /// Expected dims. Asserted against the blob's final layer so a
    /// mismatched blob fails in tests rather than network-wide.
    pub vector_dims: usize,
    pub weights: &'static [u8],
    /// quality = (offset - avg_dist) / scale. Per-scenario because mean
    /// distance scales with dims and clustering; one shared pair cannot
    /// span two corpora.
    pub quality_offset: f64,
    pub quality_scale: f64,
}

const SIFT_BLOB: &[u8] = include_bytes!("weights/v1_sift.bin");

impl From<Scenario> for ScenarioConfig {
    fn from(scenario: Scenario) -> Self {
        match scenario {
            Scenario::SIFT_128 => ScenarioConfig {
                n_queries: 7_000,
                database_size: 700_000,
                vector_dims: 128,
                weights: SIFT_BLOB,
                quality_offset: 1.616563,
                quality_scale: 6.399004,
            },
        }
    }
}

impl std::fmt::Display for Scenario {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Scenario::SIFT_128 => write!(f, "sift_128"),
        }
    }
}

impl std::str::FromStr for Scenario {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "sift_128" => Ok(Scenario::SIFT_128),
            _ => Err(anyhow!("Invalid scenario type: {}", s)),
        }
    }
}
```

Add to `tig-challenges/src/vector_search/mod.rs` after line 9 (`mod generator;`):

```rust
mod scenarios;
pub use scenarios::{Scenario, ScenarioConfig};
```

- [ ] **Step 4: Run tests to verify they pass**

Run the Build & Test Recipe.
Expected: 12 passed (8 existing + 4 new), 0 failed.

- [ ] **Step 5: Mutation-check the wire-format test**

Change `write!(f, "sift_128")` to `write!(f, "sift128")`, re-run.
Expected: `scenario_display_is_lowercase_snake_case` FAILS. Restore the underscore and confirm it passes again. If it did not fail, the test is not guarding the wire format and must be fixed before proceeding.

- [ ] **Step 6: Commit**

```bash
git add tig-challenges/src/vector_search/scenarios.rs tig-challenges/src/vector_search/mod.rs
git commit -m "Add Scenario enum and per-scenario config for vector_search"
```

---

### Task 2: `weights_from` and blob/scenario dimension agreement

**Files:**
- Modify: `tig-challenges/src/vector_search/generator.rs:103-105`
- Modify: `tig-challenges/src/vector_search/generator.rs` (test module, appended)

**Interfaces:**
- Consumes: `Scenario`, `ScenarioConfig` from Task 1.
- Produces: `pub fn weights_from(blob: &[u8]) -> Result<GeneratorWeights>`. `v1_weights()` keeps its signature and becomes a wrapper, so all existing callers and tests are untouched.

- [ ] **Step 1: Write the failing test**

Append inside the existing `mod tests` in `generator.rs`:

```rust
    #[test]
    fn every_scenario_blob_matches_its_declared_dims() {
        // Iterate over ALL variants, not just the one being added, so a
        // future scenario cannot silently skip this check. A mismatch here
        // would otherwise surface as network-wide verification failure.
        for scenario in [crate::vector_search::Scenario::SIFT_128] {
            let config = crate::vector_search::ScenarioConfig::from(scenario);
            let weights = weights_from(config.weights).unwrap_or_else(|e| {
                panic!("scenario {} has an unparseable blob: {}", scenario, e)
            });
            let derived = weights.layers.last().unwrap().out_dim;
            assert_eq!(
                derived, config.vector_dims,
                "scenario {}: blob produces {} dims but config declares {}",
                scenario, derived, config.vector_dims
            );
            assert_eq!(
                weights.layers[0].in_dim, LATENT_DIM,
                "scenario {}: first layer must consume LATENT_DIM", scenario
            );
        }
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run the Build & Test Recipe.
Expected: FAIL to compile — `cannot find function weights_from in this scope`.

- [ ] **Step 3: Write the implementation**

Replace `generator.rs:103-105` with:

```rust
/// Parse a generator from an arbitrary weight blob. `ScenarioConfig::weights`
/// selects which one.
pub fn weights_from(blob: &[u8]) -> Result<GeneratorWeights> {
    parse_weights(blob)
}

pub fn v1_weights() -> Result<GeneratorWeights> {
    weights_from(V1_BLOB)
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run the Build & Test Recipe.
Expected: 13 passed, 0 failed.

- [ ] **Step 5: Mutation-check the agreement test**

In `scenarios.rs`, change `vector_dims: 128` to `vector_dims: 96`, re-run.
Expected: `every_scenario_blob_matches_its_declared_dims` FAILS with a message naming both numbers. Restore `128`. If it passed, the test is not comparing what it claims to.

- [ ] **Step 6: Commit**

```bash
git add tig-challenges/src/vector_search/generator.rs
git commit -m "Parse generator weights from any scenario blob"
```

---

### Task 3: `Track { s: Scenario }` and wiring generation through `ScenarioConfig`

**Files:**
- Modify: `tig-challenges/src/vector_search/mod.rs:13-17` (Track), `:29-36` (Challenge struct), `:44-60` (delete module consts), `:65-100` (generate_instance), `:219-227` (Challenge construction), `:294-302` (evaluate_solution)

**Interfaces:**
- Consumes: `Scenario`, `ScenarioConfig` (Task 1); `weights_from` (Task 2).
- Produces: `Track { s: Scenario }`; `Challenge` with a new `pub scenario: Scenario` field.

`Challenge` must carry the scenario because `evaluate_solution` needs the calibration constants and currently reads module-level `const`s. Exposing it is deliberate: an algorithm may legitimately specialise on which corpus it is solving, which is what "realism" implies.

- [ ] **Step 1: Write the failing test**

Append inside the existing `mod tests` in `generator.rs`, or create `mod tests` at the end of `mod.rs` — put it in `mod.rs` since it tests `Track`:

```rust
#[cfg(test)]
mod track_tests {
    use super::*;

    #[test]
    fn track_serialises_to_protocol_wire_form() {
        let track = Track { s: Scenario::SIFT_128 };
        let encoded = serde_json::to_string(&track).unwrap();
        // serde_json wraps the kv-string in quotes, exactly as
        // tig-runtime/src/main.rs:110-122 expects to receive it.
        assert_eq!(encoded, r#""s=sift_128""#);
    }

    #[test]
    fn track_deserialises_from_protocol_wire_form() {
        let track: Track = serde_json::from_str(r#""s=sift_128""#).unwrap();
        assert_eq!(track.s, Scenario::SIFT_128);
    }

    #[test]
    fn track_rejects_unknown_scenario() {
        let err = serde_json::from_str::<Track>(r#""s=glove_300""#).unwrap_err();
        assert!(
            err.to_string().contains("glove_300"),
            "error should name the offending scenario, got: {}",
            err
        );
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run the Build & Test Recipe.
Expected: FAIL — `Track` has no field `s` (it still has `n_queries`).

- [ ] **Step 3: Change `Track` and the `Challenge` struct**

In `mod.rs`, replace the `impl_kv_string_serde!` block:

```rust
impl_kv_string_serde! {
    Track {
        s: Scenario,
    }
}
```

Add `scenario` to `Challenge`:

```rust
pub struct Challenge {
    pub seed: [u8; 32],
    pub scenario: Scenario,
    pub num_queries: u32,
    pub vector_dims: u32,
    pub database_size: u32,
    pub d_database_vectors: CudaSlice<f32>,
    pub d_query_vectors: CudaSlice<f32>,
}
```

Delete the two module-level constants and their doc comment (`mod.rs:44-60`) — the calibration rationale now lives in `scenarios.rs` and in the spec. Change the import on line 10 to:

```rust
use generator::{weights_from, LATENT_DIM};
```

- [ ] **Step 4: Rewire `generate_instance`**

Replace the opening of `generate_instance` (the lines from `let weights = v1_weights()?;` through `let database_size = 100 * track.n_queries;`) with:

```rust
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
                track.s,
                config.vector_dims,
                vector_dims
            ));
        }
        let widest = layers.iter().map(|layer| layer.out_dim).max().unwrap();
        let database_size = config.database_size;
        let n_queries = config.n_queries;
```

Then replace every remaining `track.n_queries` in the function body with `n_queries`. There are three: the `d_query_vectors` allocation, the `(true, track.n_queries as usize)` tuple in the generation loop, and the `num_queries` field in the returned struct.

The returned struct becomes:

```rust
        Ok(Self {
            seed: seed.clone(),
            scenario: track.s,
            num_queries: n_queries,
            vector_dims: vector_dims as u32,
            database_size,
            d_database_vectors,
            d_query_vectors,
        })
```

- [ ] **Step 5: Rewire `evaluate_solution`**

Replace the two constant references in `evaluate_solution`:

```rust
            let avg_dist = self.evaluate_average_distance(solution, module, stream, prop)?;
            let config = ScenarioConfig::from(self.scenario);
            let quality = (config.quality_offset - avg_dist as f64) / config.quality_scale;
            let quality = quality.clamp(-10.0, 10.0) * QUALITY_PRECISION as f64;
            let quality = quality.round() as i32;
            Ok(quality)
```

- [ ] **Step 6: Run tests to verify they pass**

Run the Build & Test Recipe.
Expected: 16 passed, 0 failed. The 8 original tests must still be among them, unmodified.

- [ ] **Step 7: Mutation-check the wire-format tests**

Change the `Track` field name from `s` to `scenario` in the `impl_kv_string_serde!` block, re-run.
Expected: `track_serialises_to_protocol_wire_form` FAILS, because the encoded string becomes `scenario=sift_128`. Restore `s`.

Be aware of what is **not** covered here: the `vector_dims != config.vector_dims` guard added to `generate_instance` in Step 4 cannot be reached from a CPU test, because `generate_instance` requires a live `CudaModule`. Task 2's `every_scenario_blob_matches_its_declared_dims` catches the same mismatch at the config level, which is the case that actually occurs (a blob updated without its config). The runtime guard is defence-in-depth for a blob swapped at build time, and is first exercised for real in Task 4. Do not claim it is tested before then.

- [ ] **Step 8: Commit**

```bash
git add tig-challenges/src/vector_search/mod.rs
git commit -m "Make vector_search tracks select a corpus scenario"
```

---

### Task 4: On-GPU validation — BLOCKED on hardware

**Files:**
- Modify: `docs/superpowers/specs/2026-08-25-per-scenario-gan-tracks-design.md` (append measured results)

**Interfaces:**
- Consumes: everything from Tasks 1-3.
- Produces: measured evidence only; no new code.

**This task cannot start until `tig-gpu` has a working GPU.** As of 2026-08-25 the card is hardware-faulted. Do not fake, skip, or mark this complete without the measurements.

- [ ] **Step 1: Confirm the GPU is actually alive**

```bash
ssh -o BatchMode=yes tig-gpu 'nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader'
```
Expected: a device line. If it says *No devices were found*, stop — the task is still blocked.

- [ ] **Step 2: Run the calibration band check**

Generate and solve with the reference exact 1-NN solver (`refsearch`) over a **fixed list of 24 nonces: 0 through 23**. Never a random or time-derived selection.

Assert: every nonce clears `min_active_quality` (68,500), and the mean of the 24 is within **±400** of 71,862.

The ±400 band is derived: per-nonce sigma is ~500, so a 24-nonce mean has standard error ~102 and ±400 is close to 4 SE. Tighter than ~±300 fails on sampling noise alone; looser than ~±800 stops catching a mis-fitted constant.

- [ ] **Step 3: Re-run launch-invariance digests**

FNV-1a over the raw device buffers across four forward-chunk / latent-block combinations (65536/256, 32768/256, 65536/128, 131072/512), plus a negative control on a different scenario or size that must produce a *different* digest.

Expected: identical digests within a configuration, different for the control. A control that matches means the digest is not input-sensitive and proves nothing.

- [ ] **Step 4: Record results in the spec and commit**

Append a `## Validation` section with the measured numbers. Then:

```bash
git add docs/superpowers/specs/2026-08-25-per-scenario-gan-tracks-design.md
git commit -m "Record on-GPU validation of per-scenario tracks"
```

- [ ] **Step 5: Delete scratch verifier bins**

Any `tig-verifier/src/bin/*.rs` created for digests must be deleted after use — they need the `vector_search`/`cuda` features, so leaving them breaks `cargo test --workspace`, which builds without features.

---

## Notes for the executor

- **Do not modify `kernels.cu`.** If a task seems to need it, stop and re-read Global Constraints — the requirement is almost certainly satisfiable another way.
- **Do not modify the 8 existing generator tests.** They are the regression bar. If one breaks, the change is wrong, not the test.
- The migration of ~105 algorithms is **out of scope** for this plan. This plan changes only `tig-challenges`.
- `n_queries` disappearing from `Track` means the protocol config for c004 must be updated to list `s=sift_128` as its track. That is a config/governance change outside this repo and is not a task here.
