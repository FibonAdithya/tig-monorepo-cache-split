use crate::audit_sampling::sample_query_ids;
use crate::QUALITY_PRECISION;
use crate::Seeds;
use anyhow::{anyhow, Result};
use cudarc::{
    driver::{safe::LaunchConfig, CudaModule, CudaSlice, CudaStream, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use std::sync::Arc;

mod generator;
mod scenarios;
use crate::gan_generator::Generator;
pub use scenarios::{Scenario, ScenarioConfig};

impl_kv_string_serde! {
    Track {
        s: Scenario,
    }
}

impl_base64_serde! {
    Solution {
        indexes: Vec<usize>,
    }
}

impl Solution {
    pub fn new() -> Self {
        Self {
            indexes: Vec::new(),
        }
    }
}

pub struct Challenge {
    pub seed: [u8; 32],
    pub scenario: Scenario,
    pub num_queries: u32,
    pub vector_dims: u32,
    pub database_size: u32,
    pub d_database_vectors: CudaSlice<f32>,
    pub d_query_vectors: CudaSlice<f32>,
}

pub const MAX_THREADS_PER_BLOCK: u32 = 1024;
const FORWARD_CHUNK: usize = 65_536;

/// Must equal `AUDIT_BLOCK` in kernels.cu. In the tiled kernel one thread owns
/// one row of an `AUDIT_BLOCK`-row database tile, the cooperative staging maps
/// `AUDIT_BLOCK` threads onto the tile, and the minimum is reduced over that
/// many shared-memory slots. So the launch's block_dim is not a free parameter:
/// the two are one constant that happens to live in two languages. A smaller
/// launch would leave tile rows unexamined and report recall *higher* than
/// reality; the kernel refuses to run rather than rely on this alone.
const AUDIT_BLOCK: u32 = 256;

/// Must equal `AUDIT_TQ` in kernels.cu. One block audits this many consecutive
/// samples, staging their query vectors in shared memory and sweeping the
/// database once for all of them, so `grid_dim` is `ceil(num_samples /
/// AUDIT_TQ)` rather than `num_samples`.
///
/// The kernel derives each block's first sample from *its* `AUDIT_TQ`, so a
/// mismatch is a coverage bug, not a crash. Too large a value here launches too
/// few blocks and leaves the tail of the sample list unaudited -- those samples
/// keep the zero `hits` they were allocated with and count as misses, so recall
/// reads *lower* than reality. Too small a value launches surplus blocks whose
/// `sample_base` is past the end, and they return immediately. Both directions
/// fail conservatively, but only equality is correct. The value itself is a
/// tuning result; kernels.cu carries the measurements behind it.
const AUDIT_TQ: u32 = 18;

/// Must equal `AUDIT_MAX_DIMS` in kernels.cu, which sizes the kernel's
/// shared-memory query staging buffer. A scenario declaring more dims than this
/// would overrun that buffer with no error, so it is checked on the host.
const AUDIT_MAX_DIMS: u32 = 256;


/// Generate `count` vectors into `dest`.
///
/// `index_base` offsets the curand sequence so the database (base 0) and the
/// queries (base `database_size`) never share a row index.
fn generate_vectors(
    seed: &[u8; 32],
    count: usize,
    index_base: usize,
    dest: &mut CudaSlice<f32>,
    generator: &Generator,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
) -> Result<()> {
    generate_vectors_with(
        seed,
        count,
        index_base,
        dest,
        generator,
        module,
        stream,
        FORWARD_CHUNK,
        generator::ROW_BLOCK,
    )
}

/// `generate_vectors` with the launch geometry exposed, so a test can show the
/// output does not depend on it.
///
/// The local `generator` (a `&Generator`) and the module `generator` (this
/// file's `mod generator;`) share a name but live in different namespaces, so
/// `generator::ROW_BLOCK` below resolves to the module, not the value.
fn generate_vectors_with(
    seed: &[u8; 32],
    count: usize,
    index_base: usize,
    dest: &mut CudaSlice<f32>,
    generator: &Generator,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    chunk: usize,
    row_block: u32,
) -> Result<()> {
    // Row indices reach the kernels as i32, and the spherical driver shifts
    // its second latent stream by 1 << 30. Both need this bound.
    if index_base + count >= (1usize << 30) {
        return Err(anyhow!(
            "index_base {} + count {} must stay below 2^30",
            index_base,
            count
        ));
    }
    let d_seed = stream.memcpy_stod(seed)?;
    let mut device = generator::DeviceGenerator::new(
        generator,
        chunk.min(count),
        row_block,
        &module,
        stream.clone(),
    )?;
    for chunk_start in (0..count).step_by(chunk) {
        let rows = chunk.min(count - chunk_start);
        device.sample_inputs(&d_seed, rows, index_base + chunk_start)?;
        device.forward(rows, dest, chunk_start)?;
    }
    Ok(())
}

/// The database half of a c004 instance: the rows every nonce of a precommit
/// searches. It is derived from `Seeds::db`, which carries no nonce, so it is
/// constant across a precommit and an index over it can be built once and
/// amortised over every nonce.
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
        let generator = Generator::from_blob(config.weights)?;
        // No "generator has no layers" arm any more: every architecture's
        // parser rejects an empty layer list, so `output_dim` cannot be asked
        // of a generator that has none.
        let vector_dims = generator.output_dim();
        if vector_dims != config.vector_dims {
            return Err(anyhow!(
                "scenario {} declares {} dims but its blob produces {}",
                track.s,
                config.vector_dims,
                vector_dims
            ));
        }
        let database_size = config.database_size;

        let mut d_database_vectors =
            stream.alloc_zeros::<f32>(database_size as usize * vector_dims)?;
        generate_vectors(
            db_seed,
            database_size as usize,
            0,
            &mut d_database_vectors,
            &generator,
            module,
            stream.clone(),
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
                db.scenario,
                track.s
            ));
        }
        let config = ScenarioConfig::from(track.s);
        let generator = Generator::from_blob(config.weights)?;
        let vector_dims = db.vector_dims as usize;
        let n_queries = config.n_queries;

        let mut d_query_vectors = stream.alloc_zeros::<f32>(n_queries as usize * vector_dims)?;
        generate_vectors(
            &seeds.nonce,
            n_queries as usize,
            db.database_size as usize,
            &mut d_query_vectors,
            &generator,
            module,
            stream.clone(),
        )?;

        // Owned copy, not a borrow: `Challenge`'s layout must stay
        // byte-identical or every existing algorithm .so reads these fields at
        // the wrong offsets.
        //
        // Cost, measured rather than computed: this alloc + `clone_dtod` is
        // **<= 3.894 ms** on an RTX 3060
        // (docs/measurements/2026-08-31-c004-post-split-nonce-time.md 3.4),
        // against a 22.7 ms marginal nonce. An earlier version of this comment
        // said "~358 MB at T4 bandwidth is ~1.4 ms"; that figure counted one
        // direction only, at full advertised bandwidth, and is wrong by
        // roughly 3x in the optimistic direction. The measured number is an
        // upper bound (the instrumented build adds a sync, and the phase
        // includes the allocation), so the true cost lies between.
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

impl Challenge {
    /// The whole instance in one call: the database pass followed by the query
    /// pass. Kept as a thin wrapper so callers that do not amortise an index
    /// across nonces (tests, `vs-evaluate`) need not know about the split.
    pub fn generate_instance(
        seeds: &Seeds,
        track: &Track,
        module: Arc<CudaModule>,
        stream: Arc<CudaStream>,
        _prop: &cudaDeviceProp,
    ) -> Result<Self> {
        let db = Database::generate(&seeds.db, track, module.clone(), stream.clone(), _prop)?;
        Self::for_nonce(&db, seeds, track, module, stream, _prop)
    }

    /// Diagnostic only -- NOT the quality path. Since quality became audited
    /// recall this has zero callers in this repo, and rustc will never warn
    /// about that (pub fn, pub type, pub mod), so: its sole consumer is
    /// `vs-evaluate`, which reports `avg_distance`. Do not delete as dead code.
    pub fn evaluate_average_distance(
        &self,
        solution: &Solution,
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

        let evaluate_total_distance_kernel = module.load_function("evaluate_total_distance")?;

        let d_solution_indexes = stream.memcpy_stod(&solution.indexes)?;
        let mut d_total_distance = stream.alloc_zeros::<f32>(1)?;
        let mut errorflag = stream.alloc_zeros::<u32>(1)?;

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream
                .launch_builder(&evaluate_total_distance_kernel)
                .arg(&self.vector_dims)
                .arg(&self.database_size)
                .arg(&self.num_queries)
                .arg(&self.d_query_vectors)
                .arg(&self.d_database_vectors)
                .arg(&d_solution_indexes)
                .arg(&mut d_total_distance)
                .arg(&mut errorflag)
                .launch(cfg)?;
        }

        stream.synchronize()?;

        let total_distance = stream.memcpy_dtov(&d_total_distance)?[0];
        let error_flag = stream.memcpy_dtov(&errorflag)?[0];

        match error_flag {
            0 => {}
            1 => {
                return Err(anyhow!("Invalid index in solution"));
            }
            _ => {
                return Err(anyhow!("Unknown error code: {}", error_flag));
            }
        }

        let avg_dist = total_distance / self.num_queries as f32;
        Ok(avg_dist)
    }

    // Recall@1 measured on the salt-selected subsample of `audit_samples`
    // queries -- the audit as the protocol runs it.
    //
    // This function MEASURES; it never compares against `min_recall` and never
    // sees a declaration. The protocol does the comparing. Three callers share
    // it: a benchmarker estimating its own recall before declaring, a verifier
    // auditing with the protocol's salt, and the pentest reading the result
    // straight through as quality. Contract C3 is this signature, and the
    // harness build does not set `hide_verification`, so wrapping it here does
    // not narrow what the harness can call -- it only stops an algorithm, which
    // DOES build with `hide_verification`, from reaching its own scorer.
    //
    // (Plain comments, not doc comments: conditional_pub! matches on a leading
    // `fn`, so an attribute -- which is what /// desugars to -- would not match
    // the macro arm.)
    conditional_pub!(
        fn measure_recall(
            &self,
            solution: &Solution,
            salt: &[u8; 32],
            module: Arc<CudaModule>,
            stream: Arc<CudaStream>,
            prop: &cudaDeviceProp,
        ) -> Result<f32> {
            let config = ScenarioConfig::from(self.scenario);
            self.measure_recall_with_samples(
                solution,
                salt,
                config.audit_samples,
                module,
                stream,
                prop,
            )
        }
    );

    // The same measurement with the sample size chosen by the caller, which is
    // what spec Decision 7 / contract C6 needs: `vs-evaluate` reports recall
    // EXACT over all queries, and passing `num_queries` here is how it says so.
    //
    // Split out rather than given a defaulted parameter because the difference
    // is not cosmetic. Asking `measure_recall` for an exact figure used to hand
    // back a 1,000-of-7,000 estimate with no error and no type change -- the
    // failure mode being a number that is merely wrong, which nothing downstream
    // could detect. `num_samples` above `num_queries` is not an error: the
    // sampler clamps with `min`, so `u32::MAX` and `num_queries` mean the same
    // thing, "every query".
    //
    // Wrapped in conditional_pub! for the same reason `measure_recall` is: an
    // algorithm that could call this could score itself, and it would not even
    // need the salt to matter, since over all queries the salt selects nothing.
    conditional_pub!(
        fn measure_recall_with_samples(
            &self,
            solution: &Solution,
            salt: &[u8; 32],
            num_samples: u32,
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
            // Unconditional, over every index rather than only the sampled ones.
            // The kernel cannot do this: it reads solution_indexes[q] only for the
            // queries it audits, so a bad index elsewhere would be invisible. Spec
            // Decision 3a gives the benchmarker, the verifier and the pentest three
            // different salts, so a salt-dependent check would let the same
            // solution be Ok for one role and Err for another. Well-formedness is
            // not a property of a random draw.
            if let Some(&bad) = solution
                .indexes
                .iter()
                .find(|&&i| i >= self.database_size as usize)
            {
                return Err(anyhow!(
                    "Invalid index in solution: {} >= {}",
                    bad,
                    self.database_size
                ));
            }
            if self.vector_dims > AUDIT_MAX_DIMS {
                return Err(anyhow!(
                    "recall_audit stages the query in {} floats of shared memory, but \
                     this instance has {} dims",
                    AUDIT_MAX_DIMS,
                    self.vector_dims
                ));
            }
            let config = ScenarioConfig::from(self.scenario);
            let ids = sample_query_ids(salt, self.num_queries, num_samples);
            // `ids.len()`, not `num_samples`: the sampler clamps with
            // `min(num_samples, num_queries)`, so a caller asking for every
            // query -- or for more than there are, which C6's exact path may
            // well do -- still divides by what was actually audited.
            let num_audited = ids.len() as u32;
            // Makes the divide-by-zero below unreachable by construction rather
            // than by argument, and avoids a zero-block launch. Reachable when
            // num_queries is 0, which no scenario declares, or when a caller
            // asks for 0 samples, which is a caller bug worth naming.
            if num_audited == 0 {
                return Err(anyhow!("No queries to audit"));
            }

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
                    .arg(&num_audited)
                    .arg(&self.d_query_vectors)
                    .arg(&self.d_database_vectors)
                    .arg(&d_indexes)
                    .arg(&d_ids)
                    .arg(&tolerance_sq)
                    .arg(&mut d_hits)
                    .arg(&mut d_err)
                    .launch(LaunchConfig {
                        // One block per AUDIT_TQ samples, not one per sample:
                        // ceil, so the final partial group still gets a block.
                        grid_dim: (num_audited.div_ceil(AUDIT_TQ), 1, 1),
                        block_dim: (AUDIT_BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }
            stream.synchronize()?;

            match stream.memcpy_dtov(&d_err)?[0] {
                0 => {}
                // Unreachable in tree -- AUDIT_BLOCK is the launch's block_dim --
                // but if it ever fires it must not masquerade as a bad solution.
                2 => {
                    return Err(anyhow!(
                        "recall_audit was launched with a block size other than {}; \
                         its tiling is only correct at that width",
                        AUDIT_BLOCK
                    ))
                }
                _ => return Err(anyhow!("Invalid index in solution")),
            }
            let hits = stream.memcpy_dtov(&d_hits)?;
            // u32 accumulator: neither 1,000 samples nor all 7,000 queries can
            // overflow it, but summing into u32 rather than the element type
            // keeps it correct if either grows.
            let total: u32 = hits.iter().sum();
            Ok(total as f32 / num_audited as f32)
        }
    );

    // Quality is the audited recall@1, scaled to QUALITY_PRECISION.
    //
    // No clamp: measure_recall returns hits/samples, which is already in
    // [0, 1], so the result lands in [0, QUALITY_PRECISION] by construction.
    // (A plain comment, not a doc comment: conditional_pub! matches on a
    // leading `fn`, so an attribute -- which is what /// desugars to -- would
    // not match the macro arm.)
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
}

#[cfg(test)]
mod track_tests {
    use super::*;

    #[test]
    fn track_serialises_to_protocol_wire_form() {
        let track = Track {
            s: Scenario::SIFT_128,
        };
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
    fn glove_track_uses_the_protocol_wire_form() {
        let encoded = serde_json::to_string(&Track {
            s: Scenario::GLOVE_100,
        })
        .unwrap();
        assert_eq!(encoded, r#""s=glove_100""#);
        let track: Track = serde_json::from_str(r#""s=glove_100""#).unwrap();
        assert_eq!(track.s, Scenario::GLOVE_100);
    }

    #[test]
    fn nytimes_track_uses_the_protocol_wire_form() {
        let encoded = serde_json::to_string(&Track {
            s: Scenario::NYTIMES_256,
        })
        .unwrap();
        assert_eq!(encoded, r#""s=nytimes_256""#);
        let track: Track = serde_json::from_str(r#""s=nytimes_256""#).unwrap();
        assert_eq!(track.s, Scenario::NYTIMES_256);
    }

    #[test]
    fn track_rejects_unknown_scenario() {
        // `deep_96`, not `glove_300`: glove_100 is a real scenario now, and a
        // rejection test whose input is one character away from a valid name
        // is one typo away from asserting nothing. `deep_96` is a corpus this
        // crate does not ship.
        let err = serde_json::from_str::<Track>(r#""s=deep_96""#).unwrap_err();
        assert!(
            err.to_string().contains("deep_96"),
            "error should name the offending scenario, got: {}",
            err
        );
    }
}

#[cfg(test)]
mod recall_audit_tests {
    use super::*;
    use crate::audit_sampling::sample_query_ids;
    // For the hand-built generator in
    // `gate_apply_handles_the_softplus_overflow_range_and_the_tied_fallback`;
    // every other test here gets its generator from a scenario blob.
    use crate::gan_generator::{Layer, StructuredGate};
    use cudarc::{driver::CudaContext, nvrtc::Ptx, runtime::result::device::get_device_prop};
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::sync::OnceLock;

    /// The two .cu files `tig-binary/scripts/build_ptx` concatenates, in the
    /// order it concatenates them. The tests compile the same source with the
    /// same flags, so what they exercise is the PTX production actually runs.
    const FRAMEWORK_CU: &str = include_str!("../../../tig-binary/src/framework.cu");
    const KERNELS_CU: &str = include_str!("kernels.cu");

    /// Exact 1-NN by brute force: the reference answer the audit is measured
    /// against.
    ///
    /// Embedded as a string rather than read from
    /// `fixtures/vector_search_1nn/kernels.cu`, which lives in the separate
    /// tig-pentesting checkout. Path-depending on another repository would make
    /// this crate's tests unrunnable on a machine that has only the monorepo,
    /// and would couple them to a tree this repo does not control.
    ///
    /// Ties break to the lowest database index at every comparison and the
    /// reduction runs in a fixed order, so the answer cannot depend on
    /// scheduler interleaving. No sqrt: argmin of squared distance is argmin of
    /// distance, and `--use_fast_math` makes sqrt approximate.
    ///
    /// It emits the second-nearest index alongside the nearest.
    /// `second_nearest_answers_score_a_miss` needs it: a solution built from
    /// the runner-up is the only probe here that lands in the narrow band where
    /// an audit that over-estimates the minimum flips a miss into a hit, and
    /// that band is where a tiling bug lives.
    const REFERENCE_1NN_KERNEL: &str = r#"
#define REF_BLOCK 256
#define REF_MAX_DIMS 256

// Total order over candidates: nearer first, and among exact ties the lower
// database index. Every comparison in this kernel goes through it, so the
// answer cannot depend on scheduler interleaving.
__device__ __forceinline__ bool ref_before(
    const float da, const unsigned long long ia,
    const float db, const unsigned long long ib)
{
    return da < db || (da == db && ia < ib);
}

extern "C" __global__ void reference_nn_search(
    const float *__restrict__ queries,
    const float *__restrict__ database,
    const int num_queries,
    const int database_size,
    const int dims,
    unsigned long long *__restrict__ out_indexes,
    unsigned long long *__restrict__ out_second_indexes)
{
    const int q = blockIdx.x;
    if (q >= num_queries) {
        return;
    }

    __shared__ float s_query[REF_MAX_DIMS];
    for (int i = threadIdx.x; i < dims; i += REF_BLOCK) {
        s_query[i] = queries[(long long)q * dims + i];
    }
    __syncthreads();

    // 3.0e38 rather than INFINITY: --use_fast_math permits relaxations around
    // specials, and a finite sentinel needs none of them.
    float best = 3.0e38f;
    unsigned long long best_idx = 0ULL;
    float second = 3.0e38f;
    unsigned long long second_idx = 0ULL;

    for (int j = threadIdx.x; j < database_size; j += REF_BLOCK) {
        const float *cand = database + (long long)j * dims;
        float d = 0.0f;
        for (int k = 0; k < dims; ++k) {
            const float diff = s_query[k] - cand[k];
            d = fmaf(diff, diff, d);
        }
        const unsigned long long jj = (unsigned long long)j;
        // The `best` branch is character-for-character the decision the
        // one-output version made, so the 1-NN answer is unchanged; `second`
        // only catches what `best` displaces or what falls just short of it.
        if (ref_before(d, jj, best, best_idx)) {
            second = best;
            second_idx = best_idx;
            best = d;
            best_idx = jj;
        } else if (ref_before(d, jj, second, second_idx)) {
            second = d;
            second_idx = jj;
        }
    }

    __shared__ float s_best[REF_BLOCK];
    __shared__ unsigned long long s_idx[REF_BLOCK];
    __shared__ float s_second[REF_BLOCK];
    __shared__ unsigned long long s_sidx[REF_BLOCK];
    s_best[threadIdx.x] = best;
    s_idx[threadIdx.x] = best_idx;
    s_second[threadIdx.x] = second;
    s_sidx[threadIdx.x] = second_idx;
    __syncthreads();

    // Merging two ordered pairs, rather than picking one winner. Each database
    // index lives in exactly one thread's stride subset, so the four candidates
    // entering a merge are four distinct rows: the combined runner-up is the
    // loser of the final between the two leaders, whichever side it came from.
    for (int stride = REF_BLOCK / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const int o = threadIdx.x + stride;
            const float mb = s_best[threadIdx.x];
            const unsigned long long mbi = s_idx[threadIdx.x];
            const float ms = s_second[threadIdx.x];
            const unsigned long long msi = s_sidx[threadIdx.x];
            const float ob = s_best[o];
            const unsigned long long obi = s_idx[o];
            const float os = s_second[o];
            const unsigned long long osi = s_sidx[o];

            if (ref_before(mb, mbi, ob, obi)) {
                // Mine leads; the runner-up is my own runner-up or their leader.
                if (!ref_before(ms, msi, ob, obi)) {
                    s_second[threadIdx.x] = ob;
                    s_sidx[threadIdx.x] = obi;
                }
            } else {
                // Theirs leads; the runner-up is their runner-up or my leader.
                s_best[threadIdx.x] = ob;
                s_idx[threadIdx.x] = obi;
                if (ref_before(os, osi, mb, mbi)) {
                    s_second[threadIdx.x] = os;
                    s_sidx[threadIdx.x] = osi;
                } else {
                    s_second[threadIdx.x] = mb;
                    s_sidx[threadIdx.x] = mbi;
                }
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out_indexes[q] = s_idx[0];
        out_second_indexes[q] = s_sidx[0];
    }
}

// TEST ONLY. The ONLY difference from `gan_gate_noise` is the curand sequence
// argument: this one passes `global_i` where the shipped kernel passes
// `GATE_NOISE_SEQUENCE_BASE + global_i`. Both call the same
// `gate_noise_from_uniform`, so neither the clamps nor the formula are
// duplicated here and the two cannot drift apart.
//
// It exists so that sift_gate_noise_does_not_reuse_the_latent_sequence can
// compare the production noise against what the UNSHIFTED sequence produces: if
// the base were dropped from kernels.cu the two would agree bit for bit.
extern "C" __global__ void test_gate_noise_unshifted(
    const uint8_t *seed, const int n, const int dim, const float eps,
    float *noise, const int index_offset)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
        const int global_i = index_offset + i;
        curandState state;
        curand_init(((const uint64_t *)(seed))[global_i % 4], global_i, 0, &state);
        float *row = noise + (long long)i * dim;
        for (int j = 0; j < dim; ++j) {
            row[j] = gate_noise_from_uniform(curand_uniform(&state), eps);
        }
    }
}

// TEST ONLY. `gate_noise_from_uniform` on caller-supplied uniforms, so that
// gate_noise_from_uniform_is_finite_at_both_ends_of_the_unit_interval can feed
// it the two endpoints curand can actually produce. It calls the SHIPPED
// function, so removing either clamp from kernels.cu changes what this writes.
extern "C" __global__ void test_gate_noise_from_uniform(
    const float *u, const int n, const float eps, float *out)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
        out[i] = gate_noise_from_uniform(u[i], eps);
    }
}
"#;

    /// Locate nvcc. Panics with an actionable message rather than skipping:
    /// a test that silently passes because it never ran verifies nothing.
    fn nvcc_path() -> PathBuf {
        if let Ok(p) = std::env::var("NVCC") {
            let p = PathBuf::from(p);
            assert!(p.exists(), "NVCC={} does not exist", p.display());
            return p;
        }
        if Command::new("nvcc").arg("--version").output().is_ok() {
            return PathBuf::from("nvcc");
        }
        for candidate in [
            "/usr/local/cuda/bin/nvcc",
            "/usr/local/cuda-12.6/bin/nvcc",
            "/usr/local/cuda-12/bin/nvcc",
        ] {
            if Path::new(candidate).exists() {
                return PathBuf::from(candidate);
            }
        }
        panic!(
            "nvcc not found. These tests compile the challenge's own PTX; there \
             is deliberately no skip path, because a green run that compiled \
             nothing would verify nothing. Put nvcc on PATH (e.g. \
             PATH=/usr/local/cuda/bin:$PATH) or set NVCC=/path/to/nvcc."
        );
    }

    /// Compile framework.cu + kernels.cu + the reference 1-NN kernel to PTX,
    /// once per test process.
    ///
    /// tig-challenges has no build script and the production PTX is built
    /// externally (by `build_ptx`, from the same two .cu files), so the test
    /// harness has to build its own. The flags mirror the ones the design's
    /// Task 1 measurement used, `--use_fast_math` included -- testing against a
    /// PTX built with different flags would test a kernel nobody runs.
    fn test_ptx_path() -> &'static PathBuf {
        static PTX: OnceLock<PathBuf> = OnceLock::new();
        PTX.get_or_init(|| {
            // include_str!, not a runtime read off BUILD_TIME_PATH: the sources
            // are baked into the test binary, so it does not depend on the
            // source tree still being where it was at compile time, and cargo
            // rebuilds the tests when either .cu file changes.
            let mut combined = String::new();
            combined.push_str(FRAMEWORK_CU);
            combined.push('\n');
            combined.push_str(KERNELS_CU);
            combined.push('\n');
            combined.push_str(REFERENCE_1NN_KERNEL);

            let dir = std::env::temp_dir()
                .join(format!("tig_c004_recall_audit_{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let cu = dir.join("combined.cu");
            let ptx = dir.join("combined.ptx");
            std::fs::write(&cu, combined).unwrap();

            let out = Command::new(nvcc_path())
                .arg("-ptx")
                .arg(&cu)
                .arg("-o")
                .arg(&ptx)
                .args([
                    "-arch",
                    "compute_70",
                    "-code",
                    "sm_70",
                    "--use_fast_math",
                    "-dopt=on",
                ])
                .output()
                .unwrap_or_else(|e| panic!("failed to run nvcc: {}", e));
            assert!(
                out.status.success(),
                "nvcc failed compiling {}:\n{}",
                cu.display(),
                String::from_utf8_lossy(&out.stderr)
            );
            ptx
        })
    }

    /// A module and stream on a real GPU, with no instance attached. Panics
    /// with an actionable message if there is no CUDA device -- no skip path,
    /// for the same reason `nvcc_path` has none.
    fn gpu_context() -> (Arc<CudaModule>, Arc<CudaStream>) {
        let ptx = Ptx::from_file(test_ptx_path().clone());
        let ctx = CudaContext::new(0).unwrap_or_else(|e| {
            panic!(
                "cannot open CUDA device 0: {}. These tests need a GPU and do \
                 not skip without one.",
                e
            )
        });
        ctx.set_blocking_synchronize().unwrap();
        (ctx.load_module(ptx).unwrap(), ctx.default_stream())
    }

    /// A real instance on a real GPU, on the caller's scenario.
    fn gpu_instance_for(
        scenario: Scenario,
        seed_byte: u8,
    ) -> (Challenge, Arc<CudaModule>, Arc<CudaStream>, cudaDeviceProp) {
        let (module, stream) = gpu_context();
        let prop = get_device_prop(0).unwrap();
        let track = Track { s: scenario };
        let challenge = Challenge::generate_instance(
            // The two fields must never hold the same bytes. `db` seeds the
            // database and `nonce` seeds the queries, so identical bytes would
            // make a swap between the two halves of the split invisible to
            // every test in this module. `x ^ 0xff` flips every bit, so it
            // differs from `x` in all eight and cannot collide for any
            // `seed_byte`.
            &Seeds {
                nonce: [seed_byte; 32],
                db: [seed_byte ^ 0xff; 32],
            },
            &track,
            module.clone(),
            stream.clone(),
            &prop,
        )
        .unwrap();
        (challenge, module, stream, prop)
    }

    /// The SIFT_128 instance every test here used before there was a second
    /// scenario. Kept so those tests still read as being about the audit
    /// rather than about which corpus they run on.
    fn gpu_instance(
        seed_byte: u8,
    ) -> (Challenge, Arc<CudaModule>, Arc<CudaStream>, cudaDeviceProp) {
        gpu_instance_for(Scenario::SIFT_128, seed_byte)
    }

    // The three floats that make the hit tolerance's two edges reachable, and
    // the hand-built instance that carries them. (Plain comments: a `///` block
    // here would attach to `TOL_DB_NEAR` alone, and this describes the group.)
    //
    // A generated SIFT instance cannot exercise `tolerance_sq`. Every probe
    // available on one is either the exact argmin (a hit at any tolerance,
    // including zero) or a uniformly random row (a miss at any tolerance short
    // of absurd), so both directions of the band are invisible: setting
    // `tolerance_sq` to 1.0 or to 1e9 leaves every other test in this module
    // green. The band is only observable against a database whose second row
    // is placed *inside* it on purpose.
    //
    // Every query is the origin, so a row's squared distance is its own
    // squared norm and the arithmetic is checkable by hand:
    //
    // | row | first coord | d^2                    | role                    |
    // |-----|-------------|------------------------|-------------------------|
    // | 0   | 1.0         | 1.0                    | the true argmin         |
    // | 1   | 1.0 + 5e-7  | 1 + 8 ulp ~= 1.0000010 | inside the band: a HIT  |
    // | 2   | 2.0         | 4.0                    | outside it: a MISS      |
    //
    // `1.0 + 5e-7` is not a no-op: an f32 ulp at 1.0 is 1.19e-7, so it rounds
    // to exactly 4 ulps above 1.0 and its square to exactly 8 ulps above 1.0.
    // The tolerance is `(1 + 1e-6)^2`, which is 16 ulps above 1.0, so row 1
    // clears the bar with 8 ulps to spare while a `tolerance_sq` of 1.0 rejects
    // it. This is the deliberate difference from the "make rows 0 and 1
    // identical" construction: identical rows give `s_returned == s_red[0]`
    // exactly, and the kernel's test is `<=`, so they would be a hit even at
    // `tolerance_sq = 1.0` and would discriminate nothing.
    //
    // `scenario` is SIFT_128 because `measure_recall` reads only
    // `audit_samples` and `recall_tolerance` off the config -- the shape comes
    // from the `Challenge` fields, which are this instance's own. With 4
    // queries and `audit_samples` 1,000, `min(1000, 4) = 4`: every query is
    // audited, so recall here is exact and not a subsample estimate.
    const TOL_DB_NEAR: f32 = 1.0f32 + 5e-7;
    const TOL_DIMS: usize = 8;
    const TOL_NUM_QUERIES: usize = 4;
    const TOL_DB_ROWS: usize = 3;

    /// The synthetic instance described above.
    fn tolerance_probe_instance() -> (Challenge, Arc<CudaModule>, Arc<CudaStream>, cudaDeviceProp) {
        let ptx = Ptx::from_file(test_ptx_path().clone());
        let ctx = CudaContext::new(0).unwrap_or_else(|e| {
            panic!(
                "cannot open CUDA device 0: {}. These tests need a GPU and do \
                 not skip without one.",
                e
            )
        });
        ctx.set_blocking_synchronize().unwrap();
        let module = ctx.load_module(ptx).unwrap();
        let stream = ctx.default_stream();
        let prop = get_device_prop(0).unwrap();

        let queries = vec![0.0f32; TOL_NUM_QUERIES * TOL_DIMS];
        let mut database = vec![0.0f32; TOL_DB_ROWS * TOL_DIMS];
        database[0 * TOL_DIMS] = 1.0;
        database[1 * TOL_DIMS] = TOL_DB_NEAR;
        database[2 * TOL_DIMS] = 2.0;

        let d_query_vectors = stream.memcpy_stod(&queries).unwrap();
        let d_database_vectors = stream.memcpy_stod(&database).unwrap();
        stream.synchronize().unwrap();

        let challenge = Challenge {
            seed: [0u8; 32],
            scenario: Scenario::SIFT_128,
            num_queries: TOL_NUM_QUERIES as u32,
            vector_dims: TOL_DIMS as u32,
            database_size: TOL_DB_ROWS as u32,
            d_database_vectors,
            d_query_vectors,
        };
        (challenge, module, stream, prop)
    }

    /// The exact answers, computed independently of `recall_audit`: the true
    /// nearest neighbour of every query, and the true *second* nearest.
    ///
    /// One launch produces both, because the second-nearest is only meaningful
    /// against the same scan that produced the first.
    fn brute_force_1nn_and_2nn(
        challenge: &Challenge,
        module: Arc<CudaModule>,
        stream: Arc<CudaStream>,
    ) -> (Solution, Solution) {
        let kernel = module.load_function("reference_nn_search").unwrap();
        let mut d_out = stream
            .alloc_zeros::<u64>(challenge.num_queries as usize)
            .unwrap();
        let mut d_out_second = stream
            .alloc_zeros::<u64>(challenge.num_queries as usize)
            .unwrap();
        unsafe {
            stream
                .launch_builder(&kernel)
                .arg(&challenge.d_query_vectors)
                .arg(&challenge.d_database_vectors)
                .arg(&(challenge.num_queries as i32))
                .arg(&(challenge.database_size as i32))
                .arg(&(challenge.vector_dims as i32))
                .arg(&mut d_out)
                .arg(&mut d_out_second)
                .launch(LaunchConfig {
                    grid_dim: (challenge.num_queries, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .unwrap();
        }
        stream.synchronize().unwrap();
        let to_solution = |v: Vec<u64>| Solution {
            indexes: v.into_iter().map(|i| i as usize).collect(),
        };
        (
            to_solution(stream.memcpy_dtov(&d_out).unwrap()),
            to_solution(stream.memcpy_dtov(&d_out_second).unwrap()),
        )
    }

    /// The exact answer, computed independently of `recall_audit`.
    fn brute_force_1nn(
        challenge: &Challenge,
        module: Arc<CudaModule>,
        stream: Arc<CudaStream>,
    ) -> Solution {
        brute_force_1nn_and_2nn(challenge, module, stream).0
    }

    #[test]
    fn exact_1nn_measures_recall_1() {
        // The reference answer is the true 1-NN: recall against it must be
        // exactly 1.0.
        let (challenge, module, stream, prop) = gpu_instance(1);
        let exact = brute_force_1nn(&challenge, module.clone(), stream.clone());
        let r = challenge
            .measure_recall(&exact, &[9u8; 32], module, stream, &prop)
            .unwrap();
        assert_eq!(r, 1.0);
    }

    #[test]
    fn all_zeros_measures_recall_near_0() {
        let (challenge, module, stream, prop) = gpu_instance(1);
        let sol = Solution {
            indexes: vec![0; challenge.num_queries as usize],
        };
        let r = challenge
            .measure_recall(&sol, &[9u8; 32], module, stream, &prop)
            .unwrap();
        assert!(r < 0.01, "all-zeros scored recall {}", r);
    }

    #[test]
    fn corrupting_a_sampled_query_drops_recall_by_exactly_one_sample() {
        // Pick a query that IS in the sample, so the expected drop is exact. An
        // earlier "changed by one step OR not at all" form of this assertion
        // passed when recall never moved -- i.e. when the audit was entirely
        // broken.
        let (challenge, module, stream, prop) = gpu_instance(1);
        let salt = [9u8; 32];
        let config = ScenarioConfig::from(challenge.scenario);
        let ids = sample_query_ids(&salt, challenge.num_queries, config.audit_samples);
        let victim = ids[0] as usize;

        let mut sol = brute_force_1nn(&challenge, module.clone(), stream.clone());
        let before = challenge
            .measure_recall(&sol, &salt, module.clone(), stream.clone(), &prop)
            .unwrap();
        assert_eq!(before, 1.0);

        // A different index in 128 dims over 700k vectors is not within 1e-6 of
        // the true minimum except by astronomical coincidence; if this ever
        // flakes, that tie is the reason.
        sol.indexes[victim] = (sol.indexes[victim] + 1) % challenge.database_size as usize;
        let after = challenge
            .measure_recall(&sol, &salt, module, stream, &prop)
            .unwrap();
        let step = 1.0f32 / config.audit_samples as f32;
        assert!(
            (before - after - step).abs() < 1e-6,
            "recall moved by {}, expected exactly {}",
            before - after,
            step
        );
    }

    #[test]
    fn corrupting_an_unsampled_query_does_not_move_recall() {
        // The other half, and the one that proves the audit is a SUBSAMPLE: if
        // it silently looked at every query, this fails.
        let (challenge, module, stream, prop) = gpu_instance(1);
        let salt = [9u8; 32];
        let config = ScenarioConfig::from(challenge.scenario);
        let ids: std::collections::HashSet<u32> =
            sample_query_ids(&salt, challenge.num_queries, config.audit_samples)
                .into_iter()
                .collect();
        let victim = (0..challenge.num_queries)
            .find(|q| !ids.contains(q))
            .expect("audit_samples < num_queries, so some query is unsampled")
            as usize;

        let mut sol = brute_force_1nn(&challenge, module.clone(), stream.clone());
        let before = challenge
            .measure_recall(&sol, &salt, module.clone(), stream.clone(), &prop)
            .unwrap();
        // Pin the value, not only the relation. `assert_eq!(before, after)`
        // alone is satisfied by any measure_recall that returns a constant --
        // it passed under both of round 1's mutations. With this line the test
        // stands on its own.
        assert_eq!(before, 1.0);
        sol.indexes[victim] = (sol.indexes[victim] + 1) % challenge.database_size as usize;
        let after = challenge
            .measure_recall(&sol, &salt, module, stream, &prop)
            .unwrap();
        assert_eq!(before, after);
    }

    #[test]
    fn an_out_of_range_index_is_an_error_not_a_miss() {
        // A miss scores badly; an invalid index must be rejected. Conflating
        // them lets a malformed solution look like a merely bad one.
        //
        // The bad index goes on a SAMPLED query. The audit reads only the
        // queries it samples, so an out-of-range index on an unsampled query is
        // invisible to it by construction -- with salt [9; 32] query 0 is not
        // sampled (the first sampled id is 3), so asserting on index 0 would
        // assert nothing at all.
        let (challenge, module, stream, prop) = gpu_instance(1);
        let salt = [9u8; 32];
        let config = ScenarioConfig::from(challenge.scenario);
        let ids = sample_query_ids(&salt, challenge.num_queries, config.audit_samples);
        let mut sol = Solution {
            indexes: vec![0; challenge.num_queries as usize],
        };
        sol.indexes[ids[0] as usize] = challenge.database_size as usize;
        let err = challenge
            .measure_recall(&sol, &salt, module, stream, &prop)
            .unwrap_err();
        // The message is asserted, not merely `is_err()`: a bare `is_err()`
        // passes when the kernel is missing entirely, which is exactly how this
        // test went green before `recall_audit` existed.
        assert!(
            err.to_string().contains("Invalid index in solution"),
            "expected the invalid-index error, got: {}",
            err
        );
    }

    #[test]
    fn an_out_of_range_index_on_an_unsampled_query_is_also_an_error() {
        // Well-formedness must not be a function of a random draw. Spec
        // Decision 3a hands three callers three different salts -- the
        // benchmarker's self-estimate, the verifier's protocol salt, the
        // pentest's seed-derived salt -- so a kernel-only range check makes the
        // same solution Ok for one role and Err for another. The audit reads
        // only the queries it samples, so this is the case the host check
        // exists for, and the only one that fails without it.
        let (challenge, module, stream, prop) = gpu_instance(1);
        let salt = [9u8; 32];
        let config = ScenarioConfig::from(challenge.scenario);
        let ids: std::collections::HashSet<u32> =
            sample_query_ids(&salt, challenge.num_queries, config.audit_samples)
                .into_iter()
                .collect();
        let victim = (0..challenge.num_queries)
            .find(|q| !ids.contains(q))
            .expect("audit_samples < num_queries, so some query is unsampled")
            as usize;

        let mut sol = brute_force_1nn(&challenge, module.clone(), stream.clone());
        sol.indexes[victim] = challenge.database_size as usize;
        let err = challenge
            .measure_recall(&sol, &salt, module, stream, &prop)
            .unwrap_err();
        assert!(
            err.to_string().contains("Invalid index in solution"),
            "expected the invalid-index error, got: {}",
            err
        );
    }

    #[test]
    fn quality_is_recall_scaled_to_quality_precision() {
        // Quality IS the audited recall, scaled. Asserting the exact
        // QUALITY_PRECISION value (not "> 0") is what pins the scale: an
        // implementation returning the raw 0..1 recall, or the old
        // mean-distance quality, misses it.
        let (challenge, module, stream, prop) = gpu_instance(1);
        let exact = brute_force_1nn(&challenge, module.clone(), stream.clone());
        let q = challenge
            .evaluate_solution(&exact, &[9u8; 32], module.clone(), stream.clone(), &prop)
            .unwrap();
        assert_eq!(q, QUALITY_PRECISION, "exact 1-NN must score full quality");

        // The lower end: a solution that answers nothing must land below the
        // bar the protocol will compare against, or the bar gates nothing.
        let zeros = Solution {
            indexes: vec![0; challenge.num_queries as usize],
        };
        let qz = challenge
            .evaluate_solution(&zeros, &[9u8; 32], module, stream, &prop)
            .unwrap();
        let config = ScenarioConfig::from(challenge.scenario);
        let bar = (config.min_recall * QUALITY_PRECISION as f32).round() as i32;
        assert!(qz < bar, "all-zeros scored {} against a bar of {}", qz, bar);
    }

    #[test]
    fn evaluate_solution_audits_the_salt_it_is_given() {
        // The one behavioural link this task introduces: evaluate_solution must
        // forward ITS caller's salt to measure_recall. Neither assertion in
        // quality_is_recall_scaled_to_quality_precision can see that -- an exact
        // 1-NN scores 1.0 and an all-zeros solution scores ~0 under *every*
        // salt -- and Task 4's salt-sensitivity tests all call measure_recall
        // directly, so they cannot see it either. Without this test, hardcoding
        // `&[0u8; 32]` in the body passes all 31 tests, which is the same
        // fully-predictable-audit-set failure --audit-salt's required value
        // exists to prevent, reached from the other end.
        let (challenge, module, stream, prop) = gpu_instance(1);
        let config = ScenarioConfig::from(challenge.scenario);
        let salt_a = [9u8; 32];
        let salt_b = [7u8; 32];

        // A query audited under A but not under B, taken straight from the set
        // difference so the choice is deterministic rather than hopeful.
        let ids_b: std::collections::HashSet<u32> =
            sample_query_ids(&salt_b, challenge.num_queries, config.audit_samples)
                .into_iter()
                .collect();
        let victim = sample_query_ids(&salt_a, challenge.num_queries, config.audit_samples)
            .into_iter()
            .find(|q| !ids_b.contains(q))
            .expect("two different salts over 1,000-of-7,000 must differ somewhere")
            as usize;

        let mut sol = brute_force_1nn(&challenge, module.clone(), stream.clone());
        // Same corruption Task 4's tests use: a different index in 128 dims over
        // 700k vectors is not within 1e-6 of the true minimum except by
        // astronomical coincidence.
        sol.indexes[victim] = (sol.indexes[victim] + 1) % challenge.database_size as usize;

        let q_a = challenge
            .evaluate_solution(&sol, &salt_a, module.clone(), stream.clone(), &prop)
            .unwrap();
        let q_b = challenge
            .evaluate_solution(&sol, &salt_b, module, stream, &prop)
            .unwrap();

        // The guard: one solution, two salts, two different qualities. A body
        // that ignores its salt argument returns the same number twice.
        assert_ne!(
            q_a, q_b,
            "the same solution scored {} under both salts -- evaluate_solution \
             is not forwarding the salt it was given",
            q_a
        );
        // Pin both ends too, so this cannot be satisfied by a body that merely
        // varies: B does not audit the corrupted query, so it is still perfect;
        // A does, so it must be strictly worse.
        assert_eq!(
            q_b, QUALITY_PRECISION,
            "salt B does not audit the corrupted query"
        );
        assert!(
            q_a < q_b,
            "salt A audits the corrupted query, so {} must be < {}",
            q_a,
            q_b
        );
    }

    #[test]
    fn second_nearest_answers_score_a_miss() {
        // The guard the rest of this module does not provide: that the audit
        // actually finds the MINIMUM, not merely some small distance.
        //
        // Every other test here submits an answer that is either the exact
        // argmin or a uniformly random row. The first is a hit under any kernel
        // that over-estimates the minimum -- `exact_1nn_measures_recall_1` is
        // structurally incapable of failing that way, since the true 1-NN's
        // distance is by definition <= any minimum computed over a subset. The
        // second is a miss under any kernel at all. So a `recall_audit` that
        // scanned only part of the database passed all nine of the tests that
        // existed before this one; that was measured, not supposed.
        //
        // The second-nearest neighbour is the probe that lands in the band
        // between those two. Its distance is above the true minimum, so a
        // correct audit calls it a miss -- but it is the smallest distance
        // above it, so the moment the audit fails to visit the true 1-NN of a
        // query, the runner-up BECOMES that query's minimum and scores a hit.
        // That makes recall on a 2nd-NN solution a direct read-out of the
        // fraction of the database the audit skipped: recall ~= f_skipped.
        //
        // SENSITIVITY FLOOR -- do not over-trust this test. The threshold below
        // is 0.01 against a 1,000-sample audit whose resolution is 1/1000, so
        // it catches skipped fractions of roughly 1% and up. Structural bugs
        // clear that comfortably: a half-scan is 50%, and truncating
        // `rows_in_tile` by one row of a 16-row staging pass is 6.25%. A
        // single-row off-by-one over 700,000 rows is 0.39% and would NOT be
        // caught here. This test is a guard against the tiling being wrong in
        // shape, not a proof that it is right in every index.
        let (challenge, module, stream, prop) = gpu_instance(1);
        // Fixed seed and fixed salt: the value below is then deterministic, and
        // a regression moves it for a reason rather than by luck of the draw.
        let salt = [9u8; 32];
        let (first, second) =
            brute_force_1nn_and_2nn(&challenge, module.clone(), stream.clone());

        // The runner-up must actually be a different row, or this asserts
        // nothing at all -- it would just be exact_1nn_measures_recall_1 again.
        assert_eq!(first.indexes.len(), second.indexes.len());
        assert!(
            first
                .indexes
                .iter()
                .zip(second.indexes.iter())
                .all(|(a, b)| a != b),
            "the reference kernel returned the same row as both nearest and \
             second nearest for some query"
        );

        let r = challenge
            .measure_recall(&second, &salt, module, stream, &prop)
            .unwrap();
        // Printed because the value is informative when it moves: under a
        // broken audit it is approximately the fraction of database skipped.
        println!("second-nearest solution scored recall {}", r);

        // `< 0.01`, deliberately not `== 0.0`. The hit test admits anything
        // within (1 + tau) of the minimum with tau = 1e-6, and Task 1 measured
        // min sqrt(d2/d1) = 1.0000004598 over 42,000 queries -- BELOW that
        // threshold. So the tightest near-tie query in the instance is a
        // legitimate hit even when answered with its second-nearest neighbour,
        // and this reads 0.000 usually and 0.001 when such a query lands in the
        // sampled subset. An equality assertion would be intermittently flaky,
        // and a flaky guard is a guard someone deletes at 2am.
        assert!(
            r < 0.01,
            "second-nearest answers scored recall {}, so the audit is not \
             finding the true minimum -- recall on a 2nd-NN solution is roughly \
             the fraction of the database the audit skipped",
            r
        );
    }

    #[test]
    fn recall_over_all_queries_is_exact_not_a_sample_estimate() {
        // Spec Decision 7 / contract C6: `vs-evaluate` reports recall EXACT over
        // all 7,000 queries, not the 1,000-sample estimate the audit runs on.
        // `measure_recall` reads `config.audit_samples` with no override, so a
        // caller asking for the exact figure silently received an estimate --
        // no error, no type change, just a different number. C6 was
        // unreachable, and failed quietly.
        let (challenge, module, stream, prop) = gpu_instance(1);
        let config = ScenarioConfig::from(challenge.scenario);
        let salt = [9u8; 32];
        let n = challenge.num_queries;
        assert!(
            config.audit_samples < n,
            "this test needs unaudited queries to exist; audit_samples is {} of {}",
            config.audit_samples,
            n
        );

        // An exact 1-NN is right on every query, so the exact path must return
        // exactly 1.0.
        let exact_answer = brute_force_1nn(&challenge, module.clone(), stream.clone());
        let r = challenge
            .measure_recall_with_samples(
                &exact_answer,
                &salt,
                n,
                module.clone(),
                stream.clone(),
                &prop,
            )
            .unwrap();
        assert_eq!(r, 1.0, "an exact 1-NN must score 1.0 over all queries");

        // The discriminating half: corrupt exactly one query that this salt does
        // NOT audit. `corrupting_an_unsampled_query_does_not_move_recall`
        // already pins that the sampled path cannot see such a corruption, so if
        // the two paths agree here, the "exact" path is still sampling.
        let sampled: std::collections::HashSet<u32> =
            sample_query_ids(&salt, n, config.audit_samples)
                .into_iter()
                .collect();
        let victim = (0..n)
            .find(|q| !sampled.contains(q))
            .expect("audit_samples < num_queries, so some query is unsampled")
            as usize;
        let mut corrupted = exact_answer.clone();
        corrupted.indexes[victim] =
            (corrupted.indexes[victim] + 1) % challenge.database_size as usize;

        let sampled_recall = challenge
            .measure_recall(&corrupted, &salt, module.clone(), stream.clone(), &prop)
            .unwrap();
        assert_eq!(
            sampled_recall, 1.0,
            "the corrupted query is unsampled, so the sampled path must still \
             read 1.0 -- if it does not, the victim was chosen wrongly and the \
             comparison below proves nothing"
        );

        let exact_recall = challenge
            .measure_recall_with_samples(&corrupted, &salt, n, module.clone(), stream.clone(), &prop)
            .unwrap();
        // The exact VALUE, not merely "different": this pins the denominator to
        // all 7,000 queries. A body that ignored `num_samples` returns 1.0; one
        // that used it for the numerator but kept `audit_samples` as the
        // denominator lands somewhere else again.
        assert_eq!(
            exact_recall,
            (n - 1) as f32 / n as f32,
            "one wrong answer out of {} queries must read as exactly {} over the \
             exact path; got {}, while the sampled path read {}",
            n,
            (n - 1) as f32 / n as f32,
            exact_recall,
            sampled_recall
        );

        // And the delegation: `measure_recall` must be exactly
        // `measure_recall_with_samples` at `config.audit_samples`, not at some
        // other constant.
        //
        // The probe has to be a solution whose recall MOVES with the sample
        // size, and `corrupted` is not one: `sample_query_ids` builds its
        // n-sample draw as the first n swaps of the same Fisher-Yates walk, so
        // a k-sample set is a subset of the 1,000-sample set for any k < 1,000,
        // and a query unsampled at 1,000 is unsampled at every smaller k too.
        // Comparing on `corrupted` therefore compares 1.0 against 1.0 and
        // passes for any constant at all -- measured, not supposed: with
        // `measure_recall` mutated to pass 500 it stayed green.
        //
        // Every even-indexed query answered wrongly gives roughly half recall,
        // and the even fraction of a 500-sample prefix is not the even fraction
        // of the 1,000-sample draw.
        let mut half_wrong = exact_answer.clone();
        for q in (0..half_wrong.indexes.len()).step_by(2) {
            half_wrong.indexes[q] = (half_wrong.indexes[q] + 1) % challenge.database_size as usize;
        }
        let via_measure_recall = challenge
            .measure_recall(&half_wrong, &salt, module.clone(), stream.clone(), &prop)
            .unwrap();
        let via_explicit_samples = challenge
            .measure_recall_with_samples(
                &half_wrong,
                &salt,
                config.audit_samples,
                module,
                stream,
                &prop,
            )
            .unwrap();
        assert!(
            via_measure_recall > 0.2 && via_measure_recall < 0.8,
            "half the queries are answered wrongly, so recall should be near \
             0.5; {} means the probe is not discriminating and the equality \
             below proves nothing",
            via_measure_recall
        );
        assert_eq!(
            via_measure_recall, via_explicit_samples,
            "measure_recall must be measure_recall_with_samples at \
             config.audit_samples ({}), and nothing else",
            config.audit_samples
        );
    }

    #[test]
    fn a_near_tie_inside_the_tolerance_counts_as_a_hit() {
        // The spec row that says a solution equal to exact-1NN with a near-tie
        // swapped still counts as a hit. Nothing in this module tested it, and
        // nothing tested `tolerance_sq` at all: with `tolerance_sq` hardcoded to
        // 1.0 -- no tolerance whatsoever -- every other test here stays green.
        //
        // Row 1 is NOT the argmin (row 0 is, at exactly 1.0), so a kernel that
        // simply compared the returned index against the true one would fail
        // this; the hit comes from the distance being inside the band.

        // The CPU reference for the same arithmetic, asserted first so that a
        // future f32 surprise shows up here as a clear failure rather than as a
        // mysterious recall of 0. This is the "audit kernel vs a CPU reference
        // on a small synthetic instance" check.
        let config = ScenarioConfig::from(Scenario::SIFT_128);
        let d0_sq = 1.0f32;
        let d1_sq = TOL_DB_NEAR * TOL_DB_NEAR;
        let tol = 1.0f32 + config.recall_tolerance;
        let tolerance_sq = tol * tol;
        assert!(
            TOL_DB_NEAR > 1.0f32,
            "1.0 + 5e-7 rounded back to 1.0 in f32, so rows 0 and 1 are \
             identical and this test degenerates: identical rows are a hit even \
             at tolerance_sq = 1.0, because the kernel's test is `<=`"
        );
        assert!(
            d1_sq > d0_sq,
            "row 1 must be strictly farther than row 0 ({} vs {}), or the \
             tolerance is not what is being measured",
            d1_sq,
            d0_sq
        );
        assert!(
            d1_sq <= d0_sq * tolerance_sq,
            "row 1 at {} is outside the tolerance band {} and could never be a \
             hit; the construction is wrong, not the kernel",
            d1_sq,
            d0_sq * tolerance_sq
        );

        let (challenge, module, stream, prop) = tolerance_probe_instance();
        let near_tie = Solution {
            indexes: vec![1; TOL_NUM_QUERIES],
        };
        let r = challenge
            .measure_recall(&near_tie, &[3u8; 32], module, stream, &prop)
            .unwrap();
        assert_eq!(
            r, 1.0,
            "an answer {} times the true minimum, inside the (1 + {})^2 \
             tolerance, must count as a hit",
            d1_sq / d0_sq,
            config.recall_tolerance
        );
    }

    #[test]
    fn an_answer_outside_the_tolerance_is_a_miss() {
        // The other edge. Without this, `tolerance_sq` could be raised to any
        // value at all -- 4.0, 1e9 -- and every test in this module would stay
        // green, because a tolerance that admits everything turns the audit into
        // "did you return a valid index", which every solution passes.
        //
        // Row 2 is at d^2 = 4.0 against a minimum of 1.0: four times the
        // minimum, so this fails the moment the tolerance grows past a factor
        // of 4 in squared distance (a factor of 2 in distance).
        let (challenge, module, stream, prop) = tolerance_probe_instance();
        let far = Solution {
            indexes: vec![2; TOL_NUM_QUERIES],
        };
        let r = challenge
            .measure_recall(&far, &[3u8; 32], module, stream, &prop)
            .unwrap();
        assert_eq!(
            r, 0.0,
            "an answer at 4x the true minimum squared distance must be a miss; \
             the tolerance is meant to absorb float noise, not to excuse a \
             wrong neighbour"
        );
    }

    #[test]
    fn audit_is_much_cheaper_than_a_naive_solve() {
        // The design property: verification must be far cheaper than solving.
        // The naive full-database scan for 7,000 queries measured 27,000 ms
        // (docs/measurements/2026-08-26-c004-lane-probe.md). 150 ms is a
        // deliberately loose ceiling that a tiled kernel clears comfortably and
        // an untiled one cannot: measured on an RTX 3060, the untiled
        // one-block-per-query kernel this replaced took 2,623 ms and the tiled
        // one takes 85-87 ms. So the gate has room on both sides -- it is not
        // near enough to either number to be decided by a bad run.
        let (challenge, module, stream, prop) = gpu_instance(1);
        let sol = Solution {
            indexes: vec![0; challenge.num_queries as usize],
        };
        // Warm the context so JIT and allocation are not in the measurement.
        let _ = challenge
            .measure_recall(&sol, &[1u8; 32], module.clone(), stream.clone(), &prop)
            .unwrap();

        // Best of three: the gate separates 2,623 ms from 86 ms, so one
        // scheduling hiccup must not fail it, and taking the minimum cannot
        // turn a genuinely slow kernel into a passing one.
        let mut ms = u128::MAX;
        for i in 0..3u8 {
            let t = std::time::Instant::now();
            let _ = challenge
                .measure_recall(&sol, &[i; 32], module.clone(), stream.clone(), &prop)
                .unwrap();
            ms = ms.min(t.elapsed().as_millis());
        }
        println!("recall_audit best-of-three: {} ms", ms);
        assert!(
            ms < 150,
            "audit took {} ms. The ceiling implies a floor of roughly 134 GB/s \
             of database reads (56 sweeps of the 358 MB database is about 20 GB, \
             and 20 GB in 150 ms is 134 GB/s), which an RTX 3060 clears at 85-87 \
             ms and the untiled one-block-per-query kernel misses at 2,623 ms. \
             So this is either the untiled kernel back in place, or a card whose \
             achievable bandwidth is below that floor -- check which before \
             assuming a regression",
            ms
        );
    }

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
        let track = Track {
            s: Scenario::SIFT_128,
        };

        let db_seed = [7u8; 32];
        let a = Challenge::generate_instance(
            &Seeds {
                nonce: [1u8; 32],
                db: db_seed,
            },
            &track,
            module.clone(),
            stream.clone(),
            &prop,
        )
        .unwrap();
        let b = Challenge::generate_instance(
            &Seeds {
                nonce: [2u8; 32],
                db: db_seed,
            },
            &track,
            module.clone(),
            stream.clone(),
            &prop,
        )
        .unwrap();

        // Compare a prefix rather than 358 MB twice: a seed change perturbs
        // every row, so the first 4,096 floats are as decisive as all of them
        // and the test stays fast enough to keep.
        let head = |c: &Challenge, which: u8| -> Vec<f32> {
            let src = if which == 0 {
                &c.d_database_vectors
            } else {
                &c.d_query_vectors
            };
            stream.memcpy_dtov(&src.slice(0..4096)).unwrap()
        };

        assert_eq!(
            head(&a, 0),
            head(&b, 0),
            "database must not depend on the nonce seed"
        );
        assert_ne!(
            head(&a, 1),
            head(&b, 1),
            "queries must depend on the nonce seed"
        );
    }

    #[test]
    fn for_nonce_keeps_the_index_base_offset() {
        // Pins `for_nonce` against `generate_vectors` called directly at both
        // candidate offsets. Comparing it against `generate_instance` instead
        // would assert nothing: after the split `generate_instance` IS
        // `Database::generate` + `for_nonce`, so an `index_base` mutation moves
        // both sides of that equality together and the test still passes.
        // Against a direct call the mutation is visible.
        let ptx = Ptx::from_file(test_ptx_path().clone());
        let ctx = CudaContext::new(0).unwrap();
        ctx.set_blocking_synchronize().unwrap();
        let module = ctx.load_module(ptx).unwrap();
        let stream = ctx.default_stream();
        let prop = get_device_prop(0).unwrap();
        let track = Track {
            s: Scenario::SIFT_128,
        };
        let seeds = Seeds {
            nonce: [3u8; 32],
            db: [9u8; 32],
        };

        let config = ScenarioConfig::from(track.s);
        let generator = Generator::from_blob(config.weights).unwrap();
        let dims = generator.output_dim();
        let n = config.n_queries as usize;

        let db = Database::generate(&seeds.db, &track, module.clone(), stream.clone(), &prop)
            .unwrap();
        let split =
            Challenge::for_nonce(&db, &seeds, &track, module.clone(), stream.clone(), &prop)
                .unwrap();

        let direct = |index_base: usize| -> Vec<f32> {
            let mut dest = stream.alloc_zeros::<f32>(n * dims).unwrap();
            generate_vectors(
                &seeds.nonce,
                n,
                index_base,
                &mut dest,
                &generator,
                module.clone(),
                stream.clone(),
            )
            .unwrap();
            stream.synchronize().unwrap();
            stream.memcpy_dtov(&dest.slice(0..4096)).unwrap()
        };

        let got = stream
            .memcpy_dtov(&split.d_query_vectors.slice(0..4096))
            .unwrap();
        let at_db_size = direct(db.database_size as usize);
        let at_zero = direct(0);

        // The two offsets must actually differ, or the assertion below is
        // vacuous and would pass against any implementation.
        assert_ne!(
            at_db_size, at_zero,
            "index_base has no effect on the generator; this test cannot discriminate"
        );
        assert_eq!(
            got, at_db_size,
            "for_nonce must offset latents by database_size"
        );
    }

    #[test]
    fn the_database_depends_on_the_db_seed() {
        // Catches a `Database::generate` that ignores `db_seed` entirely -- a
        // hardcoded constant, say. Its sibling
        // `the_database_is_identical_across_nonces_and_the_queries_are_not`
        // hands both instances the *same* `db_seed`, so a seed-ignoring
        // generator still produces two matching databases and that test stays
        // green. `BenchmarkSettings::calc_db_seed` mixes in `player_id` and
        // `algorithm_id`: if the seed did not reach the rows, every player and
        // every precommit would search one shared database, D1's "not
        // precomputable offline" property would be gone, and without this test
        // nothing anywhere would fail.
        let ptx = Ptx::from_file(test_ptx_path().clone());
        let ctx = CudaContext::new(0).unwrap();
        ctx.set_blocking_synchronize().unwrap();
        let module = ctx.load_module(ptx).unwrap();
        let stream = ctx.default_stream();
        let prop = get_device_prop(0).unwrap();
        let track = Track {
            s: Scenario::SIFT_128,
        };

        let a = Database::generate(&[11u8; 32], &track, module.clone(), stream.clone(), &prop)
            .unwrap();
        let b = Database::generate(&[12u8; 32], &track, module.clone(), stream.clone(), &prop)
            .unwrap();

        // Same 4,096-float prefix as the tests above rather than 358 MB twice:
        // a seed change perturbs every row.
        let head = |d: &Database| -> Vec<f32> {
            stream.memcpy_dtov(&d.d_database_vectors.slice(0..4096)).unwrap()
        };

        assert_ne!(
            head(&a),
            head(&b),
            "the database must depend on its own seed"
        );
    }

    #[test]
    fn database_generate_keeps_the_index_base_at_zero() {
        // The mirror of `for_nonce_keeps_the_index_base_offset`, on the
        // database half. It catches `index_base` in `Database::generate`
        // changed from 0 to any other constant -- a mutation every other test
        // here misses: `the_database_is_identical_across_nonces_and_the_queries_are_not`
        // compares two databases that both carry the mutated value,
        // `for_nonce_keeps_the_index_base_offset` never inspects the database,
        // and the recall tests only ever measure the database against itself.
        //
        // Pinned against `generate_vectors` called directly, for the same
        // reason its sibling is: comparing against another `Database::generate`
        // would move both sides of the equality together.
        let ptx = Ptx::from_file(test_ptx_path().clone());
        let ctx = CudaContext::new(0).unwrap();
        ctx.set_blocking_synchronize().unwrap();
        let module = ctx.load_module(ptx).unwrap();
        let stream = ctx.default_stream();
        let prop = get_device_prop(0).unwrap();
        let track = Track {
            s: Scenario::SIFT_128,
        };
        let seeds = Seeds {
            nonce: [4u8; 32],
            db: [5u8; 32],
        };

        let config = ScenarioConfig::from(track.s);
        let generator = Generator::from_blob(config.weights).unwrap();
        let dims = generator.output_dim();
        let n = config.database_size as usize;

        let db = Database::generate(&seeds.db, &track, module.clone(), stream.clone(), &prop)
            .unwrap();

        let direct = |index_base: usize| -> Vec<f32> {
            let mut dest = stream.alloc_zeros::<f32>(n * dims).unwrap();
            generate_vectors(
                &seeds.db,
                n,
                index_base,
                &mut dest,
                &generator,
                module.clone(),
                stream.clone(),
            )
            .unwrap();
            stream.synchronize().unwrap();
            stream.memcpy_dtov(&dest.slice(0..4096)).unwrap()
        };

        let got = stream
            .memcpy_dtov(&db.d_database_vectors.slice(0..4096))
            .unwrap();
        let at_zero = direct(0);
        let at_db_size = direct(n);

        // The two offsets must actually differ, or the assertion below is
        // vacuous and would pass against any implementation.
        assert_ne!(
            at_zero, at_db_size,
            "index_base has no effect on the generator; this test cannot discriminate"
        );
        assert_eq!(
            got, at_zero,
            "Database::generate must sample latents from index_base 0"
        );
    }

    // ---- Cross-scenario probes -------------------------------------------
    //
    // Everything below is written against a `Scenario` parameter rather than
    // against SIFT_128, because what they check is the GPU driver, not the
    // corpus: a second architecture or a second blob must satisfy the same
    // four properties. Each helper is called by a `#[test]` per scenario, so a
    // failure names the scenario in the test name as well as in the message.

    /// GPU forward == CPU reference, on the inputs the GPU actually drew.
    ///
    /// Reading the latents back off the device rather than redrawing them on
    /// the host is the point: it compares the two forward passes on identical
    /// inputs, so a curand difference cannot be mistaken for a forward-pass
    /// difference (and could not be reproduced on the host anyway).
    ///
    /// Returns how many rows were compared (rows with a gate margin too close
    /// to zero are skipped; see the structured_gate caller).
    fn assert_gpu_matches_cpu(
        scenario: Scenario,
        skip_row: impl Fn(&Generator, &[f32], Option<&[f32]>) -> bool,
    ) -> usize {
        let generator = Generator::from_blob(ScenarioConfig::from(scenario).weights).unwrap();
        assert_generator_gpu_matches_cpu(&scenario.to_string(), &generator, skip_row).0
    }

    /// The body of `assert_gpu_matches_cpu`, taking a `Generator` rather than a
    /// `Scenario` so that a test can drive a hand-built generator through the
    /// real `DeviceGenerator`. `label` is what a failure names in place of the
    /// scenario. Returns the compared-row count and the whole GPU output, so a
    /// caller can assert on the values as well as on the match.
    fn assert_generator_gpu_matches_cpu(
        label: &str,
        generator: &Generator,
        skip_row: impl Fn(&Generator, &[f32], Option<&[f32]>) -> bool,
    ) -> (usize, Vec<f32>) {
        const ROWS: usize = 1024;
        let (module, stream) = gpu_context();
        let (latent_dim, out_dim) = (generator.latent_dim(), generator.output_dim());
        let d_seed = stream.memcpy_stod(&[7u8; 32]).unwrap();
        let mut device = generator::DeviceGenerator::new(
            generator,
            ROWS,
            generator::ROW_BLOCK,
            &module,
            stream.clone(),
        )
        .unwrap();
        let mut dest = stream.alloc_zeros::<f32>(ROWS * out_dim).unwrap();
        device.sample_inputs(&d_seed, ROWS, 0).unwrap();
        device.forward(ROWS, &mut dest, 0).unwrap();
        stream.synchronize().unwrap();
        let got = stream.memcpy_dtov(&dest).unwrap();
        let (latents, noise) = device.read_inputs(ROWS).unwrap();

        let mut compared = 0;
        for row in 0..ROWS {
            let latent = &latents[row * latent_dim..(row + 1) * latent_dim];
            let row_noise = noise.as_ref().map(|n| &n[row * out_dim..(row + 1) * out_dim]);
            if skip_row(generator, latent, row_noise) {
                continue;
            }
            let expected = generator.forward_cpu(latent, row_noise).unwrap();
            for j in 0..out_dim {
                let g = got[row * out_dim + j];
                assert!(
                    (g - expected[j]).abs() < 1e-5,
                    "{} row {} coord {}: gpu {} cpu {}",
                    label,
                    row,
                    j,
                    g,
                    expected[j]
                );
            }
            compared += 1;
        }
        (compared, got)
    }

    #[test]
    fn glove_gpu_forward_matches_the_cpu_reference() {
        // The returned count is asserted, not discarded: a `skip_row` that
        // skipped everything would make the loop body unreachable and the test
        // vacuous. This caller skips nothing, so all 1024 rows must compare.
        assert_eq!(assert_gpu_matches_cpu(Scenario::GLOVE_100, |_, _, _| false), 1024);
    }

    #[test]
    fn nytimes_gpu_forward_matches_the_cpu_reference() {
        // Skips nothing, so all 1024 rows must compare -- see the GloVe caller.
        assert_eq!(assert_gpu_matches_cpu(Scenario::NYTIMES_256, |_, _, _| false), 1024);
    }

    /// Rows where some gate's margin is within 1e-4 of zero are skipped: the
    /// GPU's tanh and the CPU's differ in the last bits, a gate that close can
    /// legitimately fall either way, and a flipped gate changes the whole row.
    /// ESTIMATE (unverified): about 4 of 1024 rows. The assertion below fails
    /// if far more are skipped, so the skip cannot hide a broken kernel.
    #[test]
    fn sift_gpu_forward_matches_the_cpu_reference() {
        let compared = assert_gpu_matches_cpu(Scenario::SIFT_128, |generator, latent, noise| {
            let Generator::StructuredGate(s) = generator else { panic!("SIFT_128 should be structured_gate") };
            s.gate_margin_cpu(latent, noise.unwrap()).iter().any(|m| m.abs() < 1e-4)
        });
        assert!(compared >= 1000, "only {compared} of 1024 rows were comparable");
    }

    /// The two branches of `gan_gate_apply` that real SIFT weights never reach,
    /// driven by a hand-built generator through the real `DeviceGenerator`.
    ///
    /// (i) The softplus threshold. In f32, `log1p(exp(x)) == x` for every x from
    /// 20 up to about 88.7, and is `inf` above that (MEASURED: equal at 20.5,
    /// 25, 50 and 88; `inf` at 89 and 100). So a kernel missing the
    /// `m[j] > 20.0f` branch is indistinguishable from a correct one unless a
    /// magnitude pre-activation exceeds about 88.7. Here every pre-activation is
    /// exactly 128, so a missing branch computes `log1p(exp(128))` =
    /// `log1p(inf)` = `inf` and the row normalises to NaN.
    ///
    /// (ii) The all-gates-closed fallback and its FIRST-maximum tie-break. All
    /// weights are zero, so every coordinate's logit is computed from
    /// bit-identical inputs: `4 * tanh(-50 / 4)` = `4 * tanh(-12.5)`. tanh(12.5)
    /// is 1 - 2.8e-11, which rounds to 1.0 in f32, so each logit is -4.0 and the
    /// four-way maximum is an exact tie -- only the tie-break decides which
    /// coordinate the fallback opens. A gate opens only when its own logistic
    /// noise exceeds +4, which has probability 1/(1 + e^4) = 0.01799.
    ///
    /// The magnitude bias is 128 rather than 100. Both are far above the f32
    /// overflow point, so (i) holds either way, but `--use_fast_math` makes
    /// division and sqrt approximate: a fallback row's norm is sqrt(128^2) =
    /// sqrt(2^14) = 2^7 and 128/128 = 1 exactly for any approximation that is
    /// exact at powers of two, while an approximate reciprocal of 100 need not
    /// give exactly 1.0 -- which would fail the exact-one-hot assertion below
    /// for a reason that is not a defect.
    #[test]
    fn gate_apply_handles_the_softplus_overflow_range_and_the_tied_fallback() {
        // Row-major [out_dim][in_dim], as `gan_linear` and `dense_cpu` index it.
        let zero = |out_dim: usize, in_dim: usize, bias: Vec<f32>| Layer {
            in_dim,
            out_dim,
            weights: vec![0.0; out_dim * in_dim],
            bias,
        };
        let identity: Vec<f32> = (0..4)
            .flat_map(|r| (0..4).map(move |c| if r == c { 1.0 } else { 0.0 }))
            .collect();
        let generator = Generator::StructuredGate(StructuredGate {
            trunk: vec![zero(2, 2, vec![0.0; 2])],
            magnitude_head: zero(4, 2, vec![128.0; 4]),
            gate_head: zero(4, 2, vec![0.0; 4]),
            sparsity_head: zero(1, 2, vec![-50.0]),
            coupling: zero(4, 4, vec![0.0; 4]),
            // Identity, so the smoothed noise IS the drawn logistic noise and
            // the margin is logit + noise with nothing in between.
            smoothing: Layer { in_dim: 4, out_dim: 4, weights: identity, bias: vec![0.0; 4] },
            logit_clamp: 4.0,
            magnitude_floor: 1e-6,
            eps: 1e-8,
        });

        let (compared, got) = assert_generator_gpu_matches_cpu(
            "tied_gate",
            &generator,
            |generator, latent, noise| {
                let Generator::StructuredGate(s) = generator else {
                    panic!("the literal above is a structured_gate")
                };
                s.gate_margin_cpu(latent, noise.unwrap()).iter().any(|m| m.abs() < 1e-4)
            },
        );
        // The logistic density at 4 is e^-4 / (1 + e^-4)^2 = 0.01766, so a margin
        // lands within 1e-4 of zero with probability 2e-4 * 0.01766 = 3.5e-6 per
        // coordinate and 0.0145 per 4,096-coordinate run: about one skipped row
        // every 70 runs. If many are skipped the tie construction has broken and
        // the assertions below would be measuring nothing.
        assert!(compared >= 1000, "only {compared} of 1024 rows were comparable");
        assert!(got.iter().all(|v| v.is_finite()), "the gate output contains inf or NaN");

        // Fallback rows: all four gates closed, so the kernel opens the first
        // argmax and the row is exactly [1, 0, 0, 0]. Expected fraction
        // (1 - P(logistic > 4))^4 = 0.98201^4 = 0.9300, so 952 of 1024 rows with
        // a binomial standard deviation of sqrt(1024 * 0.93 * 0.07) = 8.2. The
        // 800 below is 18.6 sigma under that mean and cannot fail by chance,
        // while a last-maximum tie-break would write [0, 0, 0, 1] on every
        // fallback row, leaving only the 17 rows where gate 0 alone opens, and a
        // kernel without the softplus branch would write NaN on all of them.
        let one_hot_first = got
            .chunks_exact(4)
            .filter(|row| row[0] == 1.0 && row[1] == 0.0 && row[2] == 0.0 && row[3] == 0.0)
            .count();
        assert!(one_hot_first >= 800, "only {one_hot_first} of 1024 rows are exactly [1, 0, 0, 0]");

        // And the open-gate path ran, so the 1e-5 match above is not a statement
        // about the fallback alone. A nonzero coordinate past index 0 can only be
        // an open gate, because the fallback always writes coordinate 0. At least
        // one of gates 1, 2, 3 opens with probability 1 - 0.98201^3 = 0.0529, so
        // 54 rows are expected and none at all has probability 0.94709^1024,
        // which is 6e-25.
        let with_an_open_gate = got
            .chunks_exact(4)
            .filter(|row| row[1] != 0.0 || row[2] != 0.0 || row[3] != 0.0)
            .count();
        assert!(with_an_open_gate >= 1, "no row opened a gate; only the fallback path ran");
    }

    fn generated_rows(
        scenario: Scenario,
        seed: [u8; 32],
        count: usize,
        chunk: usize,
        row_block: u32,
    ) -> Vec<f32> {
        let (module, stream) = gpu_context();
        let generator = Generator::from_blob(ScenarioConfig::from(scenario).weights).unwrap();
        let mut dest = stream
            .alloc_zeros::<f32>(count * generator.output_dim())
            .unwrap();
        generate_vectors_with(
            &seed,
            count,
            0,
            &mut dest,
            &generator,
            module,
            stream.clone(),
            chunk,
            row_block,
        )
        .unwrap();
        stream.synchronize().unwrap();
        stream.memcpy_dtov(&dest).unwrap()
    }

    /// Bit-exact equality across the four geometries the 2026-08-25 spec used,
    /// and a different seed as the control that the comparison can fail.
    /// Over 200,000 rows, not the full 700,000: enough for four chunks at the
    /// smallest chunk size, and a third of the generation time.
    fn assert_invariant_to_launch_geometry(scenario: Scenario) {
        const COUNT: usize = 200_000;
        let reference = generated_rows(scenario, [3u8; 32], COUNT, 65_536, 256);
        for (chunk, block) in [(32_768, 256), (65_536, 128), (131_072, 512)] {
            let other = generated_rows(scenario, [3u8; 32], COUNT, chunk, block);
            let differing = reference
                .iter()
                .zip(&other)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                differing, 0,
                "{}: chunk {} block {} changed {} values",
                scenario, chunk, block, differing
            );
        }
        // Without this the equalities above are satisfied by any function that
        // returns a constant -- including one that never ran the generator.
        let control = generated_rows(scenario, [4u8; 32], COUNT, 65_536, 256);
        assert!(
            reference
                .iter()
                .zip(&control)
                .any(|(a, b)| a.to_bits() != b.to_bits()),
            "a different seed must change the output"
        );
    }

    #[test]
    fn glove_output_is_invariant_to_launch_geometry() {
        assert_invariant_to_launch_geometry(Scenario::GLOVE_100);
    }

    #[test]
    fn nytimes_output_is_invariant_to_launch_geometry() {
        // Kept alongside the GPU-vs-CPU test rather than folded into it:
        // `assert_gpu_matches_cpu` runs one 1,024-row chunk at global index 0
        // and out_row_offset 0, so only this test and the unit-norm one below
        // can see a spherical driver that mishandles a non-zero index or offset
        // -- including the second latent stream taking a chunk-local index.
        assert_invariant_to_launch_geometry(Scenario::NYTIMES_256);
    }

    #[test]
    fn sift_output_is_invariant_to_launch_geometry() {
        // The gate's noise stream has its own curand index, so this is also where
        // a `gan_gate_noise` launch that passed a chunk-local index rather than
        // the global one would show up: the same row would draw different noise
        // at a different chunk size.
        assert_invariant_to_launch_geometry(Scenario::SIFT_128);
    }

    /// Every database row is unit-norm. Returns the rows for further checks.
    fn assert_database_rows_are_unit_norm(scenario: Scenario) -> (Vec<f32>, usize) {
        let (challenge, _module, stream, _prop) = gpu_instance_for(scenario, 11);
        let dims = challenge.vector_dims as usize;
        let rows = stream.memcpy_dtov(&challenge.d_database_vectors).unwrap();
        // Against the challenge's own declared size, not a literal: a literal
        // would silently tie this helper to one scenario's row count.
        assert_eq!(rows.len(), challenge.database_size as usize * dims);
        for (i, row) in rows.chunks_exact(dims).enumerate() {
            // Accumulated in f64 so the tolerance measures the GPU's
            // normalisation and not the host's summation of 100 f32 squares.
            let norm = row.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-4, "{} row {} has norm {}", scenario, i, norm);
        }
        (rows, dims)
    }

    #[test]
    fn glove_database_rows_are_unit_norm() {
        assert_database_rows_are_unit_norm(Scenario::GLOVE_100);
    }

    #[test]
    fn nytimes_database_rows_are_unit_norm() {
        // Unit norm is structural for the spherical generator, not a separate
        // normalise step: out = cos_r * u + sin_r * t with u and t orthonormal
        // has norm sqrt(cos_r^2 + sin_r^2) = 1. So this fails if
        // `gan_sphere_combine` skips either normalise or the projection.
        assert_database_rows_are_unit_norm(Scenario::NYTIMES_256);
    }

    #[test]
    fn sift_database_rows_are_unit_norm_non_negative_and_sparse_like_sift() {
        let (rows, _dims) = assert_database_rows_are_unit_norm(Scenario::SIFT_128);
        assert!(rows.iter().all(|v| *v >= 0.0), "a SIFT coordinate is negative");
        let zero_fraction = rows.iter().filter(|v| **v == 0.0).count() as f64 / rows.len() as f64;
        // docs/datasets/sift.md (WGAN repo): v4's exact-zero fraction is 0.239,
        // real SIFT's 0.230. An inverted gate would give about 0.76.
        assert!((0.20..=0.28).contains(&zero_fraction), "exact-zero fraction is {zero_fraction}");
    }

    /// The same two probes `exact_1nn_measures_recall_1` and
    /// `all_zeros_measures_recall_near_0` run on SIFT, on another scenario.
    /// GloVe's 100 dims are not a multiple of the audit kernel's AUDIT_KC = 16,
    /// so this is where a staging loop that assumes a whole number of column
    /// groups shows up: over-counting breaks the all-zeros bound, under-counting
    /// breaks the exact-1-NN equality.
    fn assert_recall_probes(scenario: Scenario) {
        let (challenge, module, stream, prop) = gpu_instance_for(scenario, 1);
        let exact = brute_force_1nn(&challenge, module.clone(), stream.clone());
        let r = challenge
            .measure_recall(&exact, &[9u8; 32], module.clone(), stream.clone(), &prop)
            .unwrap();
        assert_eq!(r, 1.0, "{}: the exact 1-NN must score recall 1.0", scenario);

        let zeros = Solution {
            indexes: vec![0; challenge.num_queries as usize],
        };
        let r = challenge
            .measure_recall(&zeros, &[9u8; 32], module, stream, &prop)
            .unwrap();
        assert!(r < 0.01, "{}: all-zeros scored recall {}", scenario, r);
    }

    #[test]
    fn recall_probes_hold_on_glove() {
        assert_recall_probes(Scenario::GLOVE_100);
    }

    #[test]
    fn recall_probes_hold_on_nytimes() {
        // 256 dims is exactly AUDIT_MAX_DIMS, so this is where an audit that
        // stages one element past the end of its buffer shows up.
        assert_recall_probes(Scenario::NYTIMES_256);
    }

    /// Pearson correlation of two equal-length samples, in f64.
    fn correlation(x: &[f32], y: &[f32]) -> f64 {
        let n = x.len() as f64;
        let (mx, my) = (
            x.iter().map(|v| *v as f64).sum::<f64>() / n,
            y.iter().map(|v| *v as f64).sum::<f64>() / n,
        );
        let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
        for (a, b) in x.iter().zip(y) {
            let (da, db) = (*a as f64 - mx, *b as f64 - my);
            sxy += da * db;
            sxx += da * da;
            syy += db * db;
        }
        sxy / (sxx * syy).sqrt()
    }

    fn column(data: &[f32], width: usize, offset: usize) -> Vec<f32> {
        data.chunks_exact(width).map(|row| row[offset]).collect()
    }

    /// The trunk and skip latents must be independent draws. If the 1 << 30
    /// offset were dropped, both calls would seed curand identically and
    /// column j of z_t would EQUAL column j of z_s: correlation 1.0.
    /// For independent columns over 65,536 rows the correlation has standard
    /// deviation 1/sqrt(65536) = 0.0039, so 0.05 is about 13 sigma.
    ///
    /// The column indexing is the contract on `read_inputs` for a spherical
    /// generator: each row comes back as `[z_t (256) | z_s (256)]`, width 512,
    /// which is the order `Spherical::forward_cpu` splits the latent in. The
    /// `latent_dim` assertion below pins those literals against the blob.
    #[test]
    fn nytimes_trunk_and_skip_latents_are_independent() {
        const ROWS: usize = 65_536;
        let (module, stream) = gpu_context();
        let generator =
            Generator::from_blob(ScenarioConfig::from(Scenario::NYTIMES_256).weights).unwrap();
        assert_eq!(generator.latent_dim(), 512, "the 512/256 offsets below assume this shape");
        let d_seed = stream.memcpy_stod(&[7u8; 32]).unwrap();
        let mut device = generator::DeviceGenerator::new(
            &generator,
            ROWS,
            generator::ROW_BLOCK,
            &module,
            stream.clone(),
        )
        .unwrap();
        device.sample_inputs(&d_seed, ROWS, 0).unwrap();
        stream.synchronize().unwrap();
        let (latents, _) = device.read_inputs(ROWS).unwrap();
        assert_eq!(latents.len(), ROWS * 512);
        for j in 0..256 {
            let r = correlation(&column(&latents, 512, j), &column(&latents, 512, 256 + j));
            assert!(r.abs() < 0.05, "z_t column {j} correlates with z_s column {j}: r = {r}");
        }
    }

    #[test]
    fn sift_gate_noise_is_finite_standard_logistic_and_independent_of_the_latents() {
        const ROWS: usize = 65_536;
        let (module, stream) = gpu_context();
        let generator = Generator::from_blob(ScenarioConfig::from(Scenario::SIFT_128).weights).unwrap();
        // The 128 offsets below are the output width, not the latent width: there
        // is one gate-noise value per output coordinate.
        assert_eq!(generator.output_dim(), 128, "the 128 offsets below assume this shape");
        let d_seed = stream.memcpy_stod(&[7u8; 32]).unwrap();
        let mut device = generator::DeviceGenerator::new(&generator, ROWS, generator::ROW_BLOCK, &module, stream.clone()).unwrap();
        device.sample_inputs(&d_seed, ROWS, 0).unwrap();
        stream.synchronize().unwrap();
        let (latents, noise) = device.read_inputs(ROWS).unwrap();
        let noise = noise.expect("structured_gate has gate noise");

        assert!(noise.iter().all(|v| v.is_finite()), "gate noise contains inf or NaN");
        let n = noise.len() as f64;
        let mean = noise.iter().map(|v| *v as f64).sum::<f64>() / n;
        let var = noise.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / n;
        // Standard logistic: mean 0, variance pi^2/3 = 3.2899. With 8.4 million
        // draws the standard error of the mean is 0.0006 and of the variance
        // about 0.002, so these bounds are many sigma wide and still reject
        // log(u) alone (mean -1) or a uniform left untransformed (variance 0.083).
        assert!(mean.abs() < 0.02, "gate noise mean is {mean}");
        assert!((var - 3.2899).abs() < 0.1, "gate noise variance is {var}");

        for j in 0..128 {
            let r = correlation(&column(&latents, 128, j), &column(&noise, 128, j));
            assert!(r.abs() < 0.05, "latent column {j} correlates with gate noise column {j}: r = {r}");
        }
    }

    /// The gate noise must not come from the curand state the latents use.
    /// Statistics cannot show this: Box-Muller maps its uniforms through a
    /// cosine, so a normal and the logit of the uniform behind it are
    /// uncorrelated even when they share a state. So compare directly against
    /// what the UNSHIFTED sequence produces, using a test-only kernel.
    #[test]
    fn sift_gate_noise_does_not_reuse_the_latent_sequence() {
        const ROWS: usize = 4096;
        let (module, stream) = gpu_context();
        let generator = Generator::from_blob(ScenarioConfig::from(Scenario::SIFT_128).weights).unwrap();
        assert_eq!(generator.output_dim(), 128, "the 128 widths below assume this shape");
        let d_seed = stream.memcpy_stod(&[7u8; 32]).unwrap();
        let mut device = generator::DeviceGenerator::new(&generator, ROWS, generator::ROW_BLOCK, &module, stream.clone()).unwrap();
        device.sample_inputs(&d_seed, ROWS, 0).unwrap();
        let (_, noise) = device.read_inputs(ROWS).unwrap();
        let production = noise.unwrap();

        let kernel = module.load_function("test_gate_noise_unshifted").unwrap();
        let mut d_unshifted = stream.alloc_zeros::<f32>(ROWS * 128).unwrap();
        unsafe {
            stream.launch_builder(&kernel)
                .arg(&d_seed).arg(&(ROWS as i32)).arg(&128i32).arg(&1.0e-8f32)
                .arg(&mut d_unshifted).arg(&0i32)
                .launch(LaunchConfig { grid_dim: ((ROWS as u32 + 255) / 256, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })
                .unwrap();
        }
        stream.synchronize().unwrap();
        let unshifted = stream.memcpy_dtov(&d_unshifted).unwrap();
        let equal = production.iter().zip(&unshifted).filter(|(a, b)| a.to_bits() == b.to_bits()).count();
        assert!(equal < production.len() / 100,
            "{} of {} gate-noise values equal the unshifted sequence's", equal, production.len());
    }

    /// The two clamps in `gate_noise_from_uniform`, driven directly rather than
    /// through curand: curand's endpoints are far too rare to reach from a test,
    /// since `u == 1.0` comes up about once in 2^32 draws.
    ///
    /// Mutations caught. Remove the UPPER clamp and `out[0]` becomes
    /// `logf(1.0) - log1pf(-1.0)` = `0 - (-inf)` = `+inf`, so the finiteness
    /// assertion, `out[0] == out[1]` and `out[0] > 16.0` all fail. Remove the
    /// LOWER clamp and `out[4]` becomes `logf(0.0) - log1pf(-0.0)` = `-inf`,
    /// while `out[5]` becomes -69.08 instead of -18.42, so the finiteness
    /// assertion and two of the bit-equalities fail. (Both effects MEASURED on
    /// the host, running the same expression in f32.)
    ///
    /// This is the only test that reaches those clamps. The 700,000-row
    /// unit-norm test cannot, because the damage is silent: an infinity in the
    /// raw noise becomes a NaN across the WHOLE smoothed row, since the
    /// smoothing layer sums every tap and a zero weight times an infinity is a
    /// NaN. `logit + NaN > 0` is then false for every coordinate, so `any_open`
    /// stays 0 and `gan_gate_apply`'s fallback replaces the row with a one-hot
    /// vector -- finite, unit-norm, and wrong.
    #[test]
    fn gate_noise_from_uniform_is_finite_at_both_ends_of_the_unit_interval() {
        // The kernel's upper clamp is the literal `0.99999994f`. These two
        // assertions prove the f32 this test uploads is bit for bit that same
        // float -- the largest one below 1.0 -- rather than a near neighbour,
        // which would make `out[0] == out[1]` pass without testing the clamp.
        const LARGEST_BELOW_ONE: f32 = 0.99999994;
        assert!(LARGEST_BELOW_ONE < 1.0, "the clamp constant is not below 1.0");
        assert_eq!(
            f32::from_bits(LARGEST_BELOW_ONE.to_bits() + 1),
            1.0,
            "the clamp constant is not the LARGEST float below 1.0"
        );

        const EPS: f32 = 1.0e-8;
        // 1.0 and 0.0 are the two values the clamps exist for; EPS itself is the
        // lower clamp's boundary and passes through, because the test is
        // `u < eps`; 1e-30 is far below it; 0.5 is the midpoint.
        let u = [1.0f32, LARGEST_BELOW_ONE, 0.5, EPS, 0.0, 1.0e-30];

        let (module, stream) = gpu_context();
        let d_u = stream.memcpy_stod(&u).unwrap();
        let mut d_out = stream.alloc_zeros::<f32>(u.len()).unwrap();
        let kernel = module.load_function("test_gate_noise_from_uniform").unwrap();
        unsafe {
            stream
                .launch_builder(&kernel)
                .arg(&d_u)
                .arg(&(u.len() as i32))
                .arg(&EPS)
                .arg(&mut d_out)
                .launch(LaunchConfig {
                    grid_dim: ((u.len() as u32 + 255) / 256, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .unwrap();
        }
        stream.synchronize().unwrap();
        let out = stream.memcpy_dtov(&d_out).unwrap();

        assert!(out.iter().all(|v| v.is_finite()), "gate noise is not finite: {out:?}");

        // u = 1.0 is clamped to the largest float below 1.0, so it must produce
        // bit for bit what that float itself produces.
        assert_eq!(
            out[0].to_bits(), out[1].to_bits(),
            "u = 1.0 was not clamped: {} vs {}", out[0], out[1]
        );
        // 0 and a value far under eps are both clamped to eps, so both must
        // produce bit for bit what eps itself produces.
        assert_eq!(
            out[4].to_bits(), out[3].to_bits(),
            "u = 0 was not clamped: {} vs {}", out[4], out[3]
        );
        assert_eq!(
            out[5].to_bits(), out[3].to_bits(),
            "u = 1e-30 was not clamped: {} vs {}", out[5], out[3]
        );

        // The map is the logit, odd about u = 0.5, so the midpoint is
        // log(0.5) - log1p(-0.5) = 0. Not asserted exactly: `--use_fast_math`
        // replaces `logf` with an approximate intrinsic but leaves `log1pf`
        // accurate, so the two terms can differ by a few ulp of ln 2 = 0.6931,
        // one ulp being 6.0e-8. 1e-6 is about 17 ulp: far above what fast-math
        // can move, and far below what a wrong formula would give. In f32 on
        // the host the same expression is exactly 0.0 (MEASURED).
        assert!(out[2].abs() < 1e-6, "the midpoint is {}, not 0", out[2]);

        // The endpoint magnitudes, derived rather than copied from the brief.
        // 1 - u[1] is exactly 2^-24, so log1p(-u[1]) = log(2^-24) = -24 ln 2 =
        // -16.6355, while log(u[1]) is about -2^-24, i.e. 0: the noise is
        // +16.6355. Bounding at 16.0 leaves 0.6 of margin -- millions of ulp
        // more than fast-math can move it -- and still rejects a dropped log1p
        // term, which would give -6e-8.
        assert!(out[0] > 16.0, "u at the top of the interval gave {}", out[0]);
        // And log(1e-8) = -8 ln 10 = -18.4207 with log1p(-1e-8) about -1e-8, so
        // the noise is -18.4207. Bounding at -18.0 leaves 0.42 of margin.
        assert!(out[3] < -18.0, "u at the bottom of the interval gave {}", out[3]);
    }
}
