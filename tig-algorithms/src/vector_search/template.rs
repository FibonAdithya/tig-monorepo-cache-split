// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use crate::{seeded_hasher, HashMap, HashSet};
use anyhow::{anyhow, Result};
use cudarc::{
    driver::{safe::LaunchConfig, CudaModule, CudaStream, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::vector_search::*;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    // Optionally define hyperparameters here. Example:
    // pub param1: usize,
    // pub param2: f64,
}

pub fn help() {
    // Print help information about your algorithm here. It will be invoked with `help_algorithm` script
    println!("No help information provided.");
}

// when launching kernels, you should not exceed this const or else it may not be deterministic
const MAX_THREADS_PER_BLOCK: u32 = 1024;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> anyhow::Result<Option<Solution>> {
    // If you need random numbers, recommend using SmallRng with challenge.seed:
    //      use rand::{rngs::SmallRng, Rng, SeedableRng};
    //      let mut rng = SmallRng::from_seed(challenge.seed);

    // If you need HashMap or HashSet, make sure to use a deterministic hasher for consistent runtime_signature:
    // use crate::{seeded_hasher, HashMap, HashSet};
    // let hasher = seeded_hasher(&challenge.seed);
    // let map = HashMap::with_hasher(hasher);

    // Support hyperparameters if needed:
    // let hyperparameters = match hyperparameters {
    //     Some(hyperparameters) => {
    //         serde_json::from_value::<Hyperparameters>(Value::Object(hyperparameters.clone()))
    //             .map_err(|e| anyhow!("Failed to parse hyperparameters: {}", e))?
    //     }
    //     None => Hyperparameters { /* set default values here */ },
    // };

    // when launching kernels, you should hardcode the LaunchConfig for determinism:
    //      Example:
    //      LaunchConfig {
    //          grid_dim: (1024, 1, 1), // do not exceed 1024 for compatibility with compute 3.6
    //          block_dim: ((arr_len + 1023) / 1024, 1, 1),
    //          shared_mem_bytes: 400,
    //      }

    // use save_solution(&Solution) to save your solution. Overwrites any previous solution

    // return Err(<msg>) if your algorithm encounters an error
    // return Ok(()) if your algorithm is finished
    Err(anyhow!("Not implemented"))
}

// ---------------------------------------------------------------------------
// Optional: two stages. Build a cache once per batch, solve every nonce with it.
// ---------------------------------------------------------------------------
//
// The database is derived from the precommit alone, so it is identical for
// every nonce of a precommit. Define BOTH functions below and build the .so
// with `BUILD_CACHE=1` to have tig-runtime call them when it is given
// `--algorithm-cache <path>` (the database itself is cached separately via
// `--challenge-cache`, which the verifier also reads; your blob never reaches
// the verifier):
//
//   * cache miss: `build_cache` runs, its bytes are written to disk, and the
//     SAME process continues straight into `solve_challenge`. So `build_cache`
//     must leave the algorithm ready to solve (stash the structure in a static,
//     e.g. a `OnceLock`), exactly as `load_cache` would.
//   * cache hit: `load_cache` runs with the bytes from disk, then
//     `solve_challenge`.
//
// The slave lets only one worker run until the first build has succeeded.
// Without both functions, or without BUILD_CACHE, the .so exports neither and
// the runtime solves without a cache.
//
// Seeds: `build_cache` gets its own RNG seed (per precommit, no nonce), and
// `challenge.seed` in `solve_challenge` is the per-nonce RNG seed. Neither is
// the seed that generated the database or the queries.
//
// The blob is opaque to the runtime: whatever bytes you return come back to
// `load_cache` unchanged. Nothing in consensus re-runs the build.
//
// pub fn build_cache(
//     database: &Database,
//     seed: &[u8; 32],
//     hyperparameters: &Option<Map<String, Value>>,
//     module: Arc<CudaModule>,
//     stream: Arc<CudaStream>,
//     prop: &cudaDeviceProp,
// ) -> anyhow::Result<Vec<u8>> {
//     Err(anyhow!("Not implemented"))
// }
//
// pub fn load_cache(
//     database: &Database,
//     blob: &[u8],
//     module: Arc<CudaModule>,
//     stream: Arc<CudaStream>,
//     prop: &cudaDeviceProp,
// ) -> anyhow::Result<()> {
//     Err(anyhow!("Not implemented"))
// }

// Important! Do not include any tests in this file, it will result in your submission being rejected
