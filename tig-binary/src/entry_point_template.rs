use anyhow::{anyhow, Result};
use serde_json::{Map, Value};
use std::panic::{catch_unwind, AssertUnwindSafe};
use tig_algorithms::{CHALLENGE}::{ALGORITHM};
use tig_challenges::{CHALLENGE}::*;

#[cfg(feature = "cuda")]
use cudarc::{
    driver::{CudaModule, CudaStream},
    runtime::sys::cudaDeviceProp,
};
#[cfg(feature = "cuda")]
use std::sync::Arc;


#[cfg(not(feature = "cuda"))]
#[unsafe(no_mangle)]
pub fn entry_point(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: Option<String>,
) -> Result<()>
{
    catch_unwind(AssertUnwindSafe(|| {
        let hyperparameters = hyperparameters.map(|x| serde_json::from_str::<Map<String, Value>>(&x).unwrap());
        {ALGORITHM}::solve_challenge(challenge, save_solution, &hyperparameters)
    })).unwrap_or_else(|_| {
        Err(anyhow!("Panic occurred calling solve_challenge"))
    })
}


#[cfg(feature = "cuda")]
#[unsafe(no_mangle)]
pub fn entry_point(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: Option<String>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> Result<()>
{
    catch_unwind(AssertUnwindSafe(|| {
        let hyperparameters = hyperparameters.map(|x| serde_json::from_str::<Map<String, Value>>(&x).unwrap());
        {ALGORITHM}::solve_challenge(challenge, save_solution, &hyperparameters, module, stream, prop)
    })).unwrap_or_else(|_| {
        Err(anyhow!("Panic occurred calling solve_challenge"))
    })
}

#[no_mangle]
pub extern "C" fn help() {
    {ALGORITHM}::help();
}

// Optional two-stage ABI, in a CPU and a CUDA flavour. Stage 1 builds a cache
// over the challenge's shared input (its `Database`; for c004, an index over
// it) and stage 2 solves nonces with it. Only algorithms that define both `build_cache` and `load_cache` can
// compile these shims, so they sit behind the `cache` cargo feature, off by
// default; `build_so` adds it when BUILD_CACHE is set in the environment.
//
// Contract: `build_cache` returns the bytes to persist AND leaves the
// algorithm ready to solve, exactly as `load_cache` on those bytes would. The
// runtime that builds continues straight into the solve without reloading.
#[cfg(all(feature = "cuda", feature = "cache"))]
#[unsafe(no_mangle)]
pub fn build_cache(
    database: &Database,
    seed: &[u8; 32],
    hyperparameters: Option<String>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> Result<Vec<u8>>
{
    catch_unwind(AssertUnwindSafe(|| {
        let hyperparameters = hyperparameters.map(|x| serde_json::from_str::<Map<String, Value>>(&x).unwrap());
        {ALGORITHM}::build_cache(database, seed, &hyperparameters, module, stream, prop)
    })).unwrap_or_else(|_| {
        Err(anyhow!("Panic occurred calling build_cache"))
    })
}

#[cfg(all(not(feature = "cuda"), feature = "cache"))]
#[unsafe(no_mangle)]
pub fn build_cache(
    database: &Database,
    seed: &[u8; 32],
    hyperparameters: Option<String>,
) -> Result<Vec<u8>>
{
    catch_unwind(AssertUnwindSafe(|| {
        let hyperparameters = hyperparameters.map(|x| serde_json::from_str::<Map<String, Value>>(&x).unwrap());
        {ALGORITHM}::build_cache(database, seed, &hyperparameters)
    })).unwrap_or_else(|_| {
        Err(anyhow!("Panic occurred calling build_cache"))
    })
}

#[cfg(all(not(feature = "cuda"), feature = "cache"))]
#[unsafe(no_mangle)]
pub fn load_cache(
    database: &Database,
    blob: &[u8],
) -> Result<()>
{
    catch_unwind(AssertUnwindSafe(|| {
        {ALGORITHM}::load_cache(database, blob)
    })).unwrap_or_else(|_| {
        Err(anyhow!("Panic occurred calling load_cache"))
    })
}

#[cfg(all(feature = "cuda", feature = "cache"))]
#[unsafe(no_mangle)]
pub fn load_cache(
    database: &Database,
    blob: &[u8],
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> Result<()>
{
    catch_unwind(AssertUnwindSafe(|| {
        {ALGORITHM}::load_cache(database, blob, module, stream, prop)
    })).unwrap_or_else(|_| {
        Err(anyhow!("Panic occurred calling load_cache"))
    })
}
