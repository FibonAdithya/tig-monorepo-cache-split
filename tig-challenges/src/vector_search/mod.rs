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

/// Must equal `AUDIT_BLOCK` in kernels.cu. `recall_audit` strides its candidate
/// scan by that literal and reduces over that many shared-memory slots, so the
/// launch's block_dim is not a free parameter: the two are one constant that
/// happens to live in two languages. Task 6 tiles this kernel -- change it here
/// and there together, or the scan silently skips candidates.
const AUDIT_BLOCK: u32 = 256;

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
                    grid_dim: (num_samples, 1, 1),
                    block_dim: (AUDIT_BLOCK, 1, 1),
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

    conditional_pub!(
        fn evaluate_solution(
            &self,
            solution: &Solution,
            module: Arc<CudaModule>,
            stream: Arc<CudaStream>,
            prop: &cudaDeviceProp,
        ) -> Result<i32> {
            let avg_dist = self.evaluate_average_distance(solution, module, stream, prop)?;
            let config = ScenarioConfig::from(self.scenario);
            let quality = (config.quality_offset - avg_dist as f64) / config.quality_scale;
            let quality = quality.clamp(-10.0, 10.0) * QUALITY_PRECISION as f64;
            let quality = quality.round() as i32;
            Ok(quality)
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
    const REFERENCE_1NN_KERNEL: &str = r#"
#define REF_BLOCK 256
#define REF_MAX_DIMS 128

extern "C" __global__ void reference_nn_search(
    const float *__restrict__ queries,
    const float *__restrict__ database,
    const int num_queries,
    const int database_size,
    const int dims,
    unsigned long long *__restrict__ out_indexes)
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

    for (int j = threadIdx.x; j < database_size; j += REF_BLOCK) {
        const float *cand = database + (long long)j * dims;
        float d = 0.0f;
        for (int k = 0; k < dims; ++k) {
            const float diff = s_query[k] - cand[k];
            d = fmaf(diff, diff, d);
        }
        const unsigned long long jj = (unsigned long long)j;
        if (d < best || (d == best && jj < best_idx)) {
            best = d;
            best_idx = jj;
        }
    }

    __shared__ float s_best[REF_BLOCK];
    __shared__ unsigned long long s_idx[REF_BLOCK];
    s_best[threadIdx.x] = best;
    s_idx[threadIdx.x] = best_idx;
    __syncthreads();

    for (int stride = REF_BLOCK / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float other = s_best[threadIdx.x + stride];
            const unsigned long long other_idx = s_idx[threadIdx.x + stride];
            const float mine = s_best[threadIdx.x];
            const unsigned long long mine_idx = s_idx[threadIdx.x];
            if (other < mine || (other == mine && other_idx < mine_idx)) {
                s_best[threadIdx.x] = other;
                s_idx[threadIdx.x] = other_idx;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out_indexes[q] = s_idx[0];
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

    /// The exact answer, computed independently of `recall_audit`.
    fn brute_force_1nn(
        challenge: &Challenge,
        module: Arc<CudaModule>,
        stream: Arc<CudaStream>,
    ) -> Solution {
        let kernel = module.load_function("reference_nn_search").unwrap();
        let mut d_out = stream
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
                .launch(LaunchConfig {
                    grid_dim: (challenge.num_queries, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .unwrap();
        }
        stream.synchronize().unwrap();
        let indexes = stream.memcpy_dtov(&d_out).unwrap();
        Solution {
            indexes: indexes.into_iter().map(|i| i as usize).collect(),
        }
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
}
