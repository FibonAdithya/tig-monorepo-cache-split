use crate::audit_sampling::sample_query_ids;
use crate::QUALITY_PRECISION;
use anyhow::{anyhow, Result};
use cudarc::{
    driver::{safe::LaunchConfig, CudaModule, CudaSlice, CudaStream, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use std::sync::Arc;

mod generator;
mod scenarios;
use generator::{weights_from, LATENT_DIM};
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
const AUDIT_MAX_DIMS: u32 = 128;


impl Challenge {
    pub fn generate_instance(
        seed: &[u8; 32],
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
                track.s,
                config.vector_dims,
                vector_dims
            ));
        }
        let widest = layers.iter().map(|layer| layer.out_dim).max().unwrap();
        let database_size = config.database_size;
        let n_queries = config.n_queries;

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
        let mut d_database_vectors =
            stream.alloc_zeros::<f32>(database_size as usize * vector_dims)?;
        let mut d_query_vectors =
            stream.alloc_zeros::<f32>(n_queries as usize * vector_dims)?;

        for (dest_is_query, count) in [
            (false, database_size as usize),
            (true, n_queries as usize),
        ] {
            let index_base = if dest_is_query {
                database_size as usize
            } else {
                0
            };

            for chunk_start in (0..count).step_by(FORWARD_CHUNK) {
                let rows = FORWARD_CHUNK.min(count - chunk_start);

                unsafe {
                    stream
                        .launch_builder(&sample_latents_kernel)
                        .arg(&d_seed)
                        .arg(&(rows as i32))
                        .arg(&(LATENT_DIM as i32))
                        .arg(&mut d_latents)
                        .arg(&((index_base + chunk_start) as i32))
                        .launch(LaunchConfig {
                            grid_dim: ((rows as u32 + 255) / 256, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        })?;
                }

                for (i, layer) in layers.iter().enumerate() {
                    let is_last = i + 1 == layers.len();
                    let (d_weight, d_bias) = &d_weights[i];
                    let cfg = LaunchConfig {
                        grid_dim: (
                            (rows as u32 + 127) / 128,
                            (layer.out_dim as u32 + 63) / 64,
                            1,
                        ),
                        block_dim: (16, 16, 1),
                        shared_mem_bytes: 0,
                    };
                    let apply_activation = (!is_last) as i32;
                    let out_row_offset = if is_last { chunk_start as i32 } else { 0 };

                    if is_last {
                        let input: &CudaSlice<f32> = if i == 0 {
                            &d_latents
                        } else if i % 2 == 1 {
                            &d_scratch_a
                        } else {
                            &d_scratch_b
                        };
                        let destination = if dest_is_query {
                            &mut d_query_vectors
                        } else {
                            &mut d_database_vectors
                        };
                        unsafe {
                            stream
                                .launch_builder(&linear_kernel)
                                .arg(input)
                                .arg(d_weight)
                                .arg(d_bias)
                                .arg(destination)
                                .arg(&(rows as i32))
                                .arg(&(layer.in_dim as i32))
                                .arg(&(layer.out_dim as i32))
                                .arg(&apply_activation)
                                .arg(&out_row_offset)
                                .launch(cfg)?;
                        }
                    } else if i == 0 {
                        unsafe {
                            stream
                                .launch_builder(&linear_kernel)
                                .arg(&d_latents)
                                .arg(d_weight)
                                .arg(d_bias)
                                .arg(&mut d_scratch_a)
                                .arg(&(rows as i32))
                                .arg(&(layer.in_dim as i32))
                                .arg(&(layer.out_dim as i32))
                                .arg(&apply_activation)
                                .arg(&out_row_offset)
                                .launch(cfg)?;
                        }
                    } else if i % 2 == 1 {
                        unsafe {
                            stream
                                .launch_builder(&linear_kernel)
                                .arg(&d_scratch_a)
                                .arg(d_weight)
                                .arg(d_bias)
                                .arg(&mut d_scratch_b)
                                .arg(&(rows as i32))
                                .arg(&(layer.in_dim as i32))
                                .arg(&(layer.out_dim as i32))
                                .arg(&apply_activation)
                                .arg(&out_row_offset)
                                .launch(cfg)?;
                        }
                    } else {
                        unsafe {
                            stream
                                .launch_builder(&linear_kernel)
                                .arg(&d_scratch_b)
                                .arg(d_weight)
                                .arg(d_bias)
                                .arg(&mut d_scratch_a)
                                .arg(&(rows as i32))
                                .arg(&(layer.in_dim as i32))
                                .arg(&(layer.out_dim as i32))
                                .arg(&apply_activation)
                                .arg(&out_row_offset)
                                .launch(cfg)?;
                        }
                    }
                }
            }
        }
        stream.synchronize()?;

        Ok(Self {
            seed: seed.clone(),
            scenario: track.s,
            num_queries: n_queries,
            vector_dims: vector_dims as u32,
            database_size,
            d_database_vectors,
            d_query_vectors,
        })
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
        let ids = sample_query_ids(salt, self.num_queries, config.audit_samples);
        let num_samples = ids.len() as u32;
        // Makes the divide-by-zero below unreachable by construction rather
        // than by argument, and avoids a zero-block launch. Only reachable when
        // num_queries is 0, which no scenario declares.
        if num_samples == 0 {
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
                .arg(&num_samples)
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
                    grid_dim: (num_samples.div_ceil(AUDIT_TQ), 1, 1),
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
        // u32 accumulator: 1,000 samples cannot overflow, but summing into u32
        // rather than the element type keeps it correct if audit_samples grows.
        let total: u32 = hits.iter().sum();
        Ok(total as f32 / num_samples as f32)
    }

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
    fn track_rejects_unknown_scenario() {
        let err = serde_json::from_str::<Track>(r#""s=glove_300""#).unwrap_err();
        assert!(
            err.to_string().contains("glove_300"),
            "error should name the offending scenario, got: {}",
            err
        );
    }
}

#[cfg(test)]
mod recall_audit_tests {
    use super::*;
    use crate::audit_sampling::sample_query_ids;
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
#define REF_MAX_DIMS 128

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

    /// A real instance on a real GPU. Panics with an actionable message if
    /// there is no CUDA device -- again, no skip path.
    fn gpu_instance(
        seed_byte: u8,
    ) -> (Challenge, Arc<CudaModule>, Arc<CudaStream>, cudaDeviceProp) {
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
        let track = Track {
            s: Scenario::SIFT_128,
        };
        let challenge = Challenge::generate_instance(
            &[seed_byte; 32],
            &track,
            module.clone(),
            stream.clone(),
            &prop,
        )
        .unwrap();
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
            "audit took {} ms; the untiled kernel is still in place",
            ms
        );
    }
}
