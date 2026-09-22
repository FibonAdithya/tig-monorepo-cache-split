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
) -> anyhow::Result<()> {
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
// Optional: the index-building ABI (vector_search / c004 only)
// ---------------------------------------------------------------------------
//
// The database a nonce searches is derived from the precommit's `rand_hash`
// alone -- no nonce is involved -- so it is identical for every nonce of a
// precommit. An algorithm may therefore build an index over it ONCE and reuse
// that index for every nonce, instead of rebuilding per nonce.
//
// Both functions below are OPTIONAL. An algorithm that defines neither still
// works exactly as before: `tig-runtime` resolves them by symbol name and, if
// they are absent, simply never runs a build phase. Define them as a PAIR --
// `build_index` without `load_index` is rejected before the build even starts,
// because discovering the mismatch after paying for a ten-minute build is a
// waste with no upside.
//
// To ship them you must build the .so with the index-building shims enabled:
//
//     INDEX_BUILD=1 ./tig-binary/scripts/build_so ...
//
// which turns on tig-binary's `index_build` cargo feature. Without that
// environment variable the two `#[no_mangle]` shims are not compiled, the .so
// exports neither symbol, and the runtime silently falls back to the
// index-free path -- so a build that "works" is not evidence the ABI is wired.
//
// How the runtime drives them:
//
//   * `tig-runtime build-index <SETTINGS> <RAND_HASH> <BINARY> --ptx P
//        --build-fuel N --index-out F` runs in its OWN process, which is never
//     given a nonce and therefore cannot derive any query. It calls
//     `build_index` and writes the returned bytes to `F`.
//   * `tig-runtime batch <SETTINGS> <RAND_HASH> <BINARY> --ptx P
//        --start-nonce A --num-nonces M --index F` calls `load_index` ONCE,
//     above the nonce loop, then solves nonces A..A+M.
//
// The blob is opaque to the runtime: it is whatever bytes you return, and it
// is handed back to `load_index` unchanged. Nothing in consensus re-runs the
// build, and `tig-verifier` never loads an index.
//
// The index handle never crosses the FFI boundary. `load_index` stashes it in
// your own `static` (e.g. a `OnceLock`), and `solve_challenge` reads it from
// there. Note that in batch mode that `static` persists across every nonce of
// the bundle.
//
// Fuel: the build phase draws on `--build-fuel`, a separate budget from the
// per-nonce `--fuel`. `load_index` runs before the nonce loop's first
// `initialize_kernel`, so the work it does is not charged to any nonce.
//
// Uncomment and implement:
//
// pub fn build_index(
//     database: &Database,
//     hyperparameters: &Option<Map<String, Value>>,
//     module: Arc<CudaModule>,
//     stream: Arc<CudaStream>,
//     prop: &cudaDeviceProp,
// ) -> anyhow::Result<Vec<u8>> {
//     // Build whatever structure your search wants over `database`, and
//     // serialise it. Return Err(<msg>) if the build fails; no index file is
//     // written and the precommit is abandoned.
//     Err(anyhow!("Not implemented"))
// }
//
// pub fn load_index(
//     database: &Database,
//     blob: &[u8],
//     module: Arc<CudaModule>,
//     stream: Arc<CudaStream>,
//     prop: &cudaDeviceProp,
// ) -> anyhow::Result<()> {
//     // Deserialise `blob` (and upload whatever of it belongs on the device),
//     // then stash the handle in a `static` for `solve_challenge` to read.
//     Err(anyhow!("Not implemented"))
// }

// Important! Do not include any tests in this file, it will result in your submission being rejected
