// NN-descent k-NN graph on the index-build ABI: a FUEL-METERING probe of the
// CAGRA build path, not a tuned entry. Build parameters are hyperparameters so
// one instrumented .so covers a sweep.
//
// Build (charged to --build-fuel): random initial lists, then `rounds` local
// joins with NN-descent's new/old rule (build.cu). Search (charged to --fuel):
// the shared greedy beam search (search.cu) from hash-chosen entry points with
// `ef_search` candidates, over the directed k-NN graph as built (no reverse
// edges, no reordering -- CAGRA adds both to its final graph; this probe
// measures the k-NN graph construction that dominates its build).
//
// Blob: u32 degree, u32 dims, u32 n_db, then n_db * degree u32 neighbour ids.
use anyhow::{anyhow, Result};
use cudarc::driver::{CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::runtime::sys::cudaDeviceProp;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::sync::{Arc, OnceLock};
use tig_challenges::vector_search::*;

const THREADS: u32 = 256;
/// Must equal `ND_MAX_K` in build.cu and `GS_MAX_DEGREE` in search.cu.
const MAX_DEGREE: u32 = 64;
/// Must equal `GS_MAX_EF` in search.cu.
const MAX_EF: u32 = 512;
/// Must equal `ND_MAX_DIMS` / `GS_MAX_DIMS`; rows must also be float4-wide.
const MAX_DIMS: u32 = 256;
const BUILD_SEED: u64 = 0x5EED_0000_0000_0001;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    #[serde(default = "default_degree")]
    pub degree: u32,
    #[serde(default = "default_rounds")]
    pub rounds: u32,
    #[serde(default = "default_ef_search")]
    pub ef_search: u32,
}

fn default_degree() -> u32 {
    32
}
fn default_rounds() -> u32 {
    10
}
fn default_ef_search() -> u32 {
    64
}

impl Default for Hyperparameters {
    fn default() -> Self {
        Self {
            degree: default_degree(),
            rounds: default_rounds(),
            ef_search: default_ef_search(),
        }
    }
}

fn parse_hyperparameters(hyperparameters: &Option<Map<String, Value>>) -> Result<Hyperparameters> {
    let hp = match hyperparameters {
        Some(h) => serde_json::from_value::<Hyperparameters>(Value::Object(h.clone()))
            .map_err(|e| anyhow!("Failed to parse hyperparameters: {}", e))?,
        None => Hyperparameters::default(),
    };
    if hp.degree == 0 || hp.degree > MAX_DEGREE {
        return Err(anyhow!("degree must be in 1..={}, got {}", MAX_DEGREE, hp.degree));
    }
    if hp.ef_search == 0 || hp.ef_search > MAX_EF {
        return Err(anyhow!("ef_search must be in 1..={}, got {}", MAX_EF, hp.ef_search));
    }
    Ok(hp)
}

pub fn help() {
    println!("NN-descent k-NN graph on the index-build ABI (fuel probe).");
    println!("Hyperparameters (JSON): degree (32), rounds (10), ef_search (64).");
}

struct LoadedIndex {
    adj: CudaSlice<u32>,
    degree: i32,
    dims: i32,
    n_db: i32,
}

static INDEX: OnceLock<LoadedIndex> = OnceLock::new();

fn cfg(grid: u32, block: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn check_dims(dims: u32) -> Result<()> {
    if dims == 0 || dims > MAX_DIMS || dims % 4 != 0 {
        return Err(anyhow!("dims must be a multiple of 4 in 4..={}, got {}", MAX_DIMS, dims));
    }
    Ok(())
}

pub fn build_index(
    database: &Database,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    _prop: &cudaDeviceProp,
) -> anyhow::Result<Vec<u8>> {
    let hp = parse_hyperparameters(hyperparameters)?;
    check_dims(database.vector_dims)?;
    let n_db = database.database_size as i32;
    let dims = database.vector_dims as i32;
    let k = hp.degree as i32;
    if n_db <= k {
        return Err(anyhow!("degree {} needs more than {} rows", k, n_db));
    }
    let total = (n_db as usize) * (k as usize);
    if total > i32::MAX as usize {
        return Err(anyhow!("n_db * degree overflows the kernels' i32 indexing"));
    }

    let k_init = module.load_function("nnd_init")?;
    let k_round = module.load_function("nnd_round")?;
    let k_flags = module.load_function("nnd_flags")?;
    let k_strip = module.load_function("nnd_strip")?;

    let mut d_adj = stream.alloc_zeros::<u32>(total)?;
    let mut d_dst = stream.alloc_zeros::<f32>(total)?;
    let total_i = total as i32;

    unsafe {
        stream
            .launch_builder(&k_init)
            .arg(&database.d_database_vectors)
            .arg(&n_db)
            .arg(&dims)
            .arg(&k)
            .arg(&BUILD_SEED)
            .arg(&mut d_adj)
            .arg(&mut d_dst)
            .launch(cfg(n_db as u32, THREADS))?;
        for _ in 0..hp.rounds {
            stream
                .launch_builder(&k_round)
                .arg(&database.d_database_vectors)
                .arg(&n_db)
                .arg(&dims)
                .arg(&k)
                .arg(&mut d_adj)
                .arg(&mut d_dst)
                .launch(cfg(n_db as u32, THREADS))?;
            stream
                .launch_builder(&k_flags)
                .arg(&mut d_adj)
                .arg(&total_i)
                .launch(cfg((total as u32).div_ceil(THREADS), THREADS))?;
        }
        stream
            .launch_builder(&k_strip)
            .arg(&mut d_adj)
            .arg(&total_i)
            .launch(cfg((total as u32).div_ceil(THREADS), THREADS))?;
    }
    stream.synchronize()?;

    let adj = stream.memcpy_dtov(&d_adj)?;
    let mut blob = Vec::with_capacity(12 + adj.len() * 4);
    blob.extend_from_slice(&(k as u32).to_le_bytes());
    blob.extend_from_slice(&(dims as u32).to_le_bytes());
    blob.extend_from_slice(&(n_db as u32).to_le_bytes());
    for v in &adj {
        blob.extend_from_slice(&v.to_le_bytes());
    }
    Ok(blob)
}

pub fn load_index(
    _database: &Database,
    blob: &[u8],
    _module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    _prop: &cudaDeviceProp,
) -> anyhow::Result<()> {
    if blob.len() < 12 {
        return Err(anyhow!("index blob too short: {} bytes", blob.len()));
    }
    let rd_u32 = |b: &[u8]| u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    let degree = rd_u32(&blob[0..4]);
    let dims = rd_u32(&blob[4..8]);
    let n_db = rd_u32(&blob[8..12]);
    if degree == 0 || degree > MAX_DEGREE || n_db == 0 || n_db > i32::MAX as u32 {
        return Err(anyhow!("index blob header out of range: degree={} dims={} n_db={}", degree, dims, n_db));
    }
    check_dims(dims)?;
    let total = (n_db as usize)
        .checked_mul(degree as usize)
        .ok_or_else(|| anyhow!("index blob header overflows"))?;
    let want = total
        .checked_mul(4)
        .and_then(|b| b.checked_add(12))
        .ok_or_else(|| anyhow!("index blob length overflows a usize"))?;
    if blob.len() != want {
        return Err(anyhow!("index blob is {} bytes, expected {}", blob.len(), want));
    }
    let adj: Vec<u32> = blob[12..].chunks_exact(4).map(rd_u32).collect();
    let loaded = LoadedIndex {
        adj: stream.memcpy_stod(&adj)?,
        degree: degree as i32,
        dims: dims as i32,
        n_db: n_db as i32,
    };
    stream.synchronize()?;
    INDEX.set(loaded).map_err(|_| anyhow!("load_index called twice"))?;
    Ok(())
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    _prop: &cudaDeviceProp,
) -> anyhow::Result<()> {
    let idx = INDEX.get().ok_or_else(|| {
        anyhow!("load_index was never called -- run with --index, or the measurement is of nothing")
    })?;
    let hp = parse_hyperparameters(hyperparameters)?;
    let n_q = challenge.num_queries as i32;
    let dims = challenge.vector_dims as i32;
    if dims != idx.dims {
        return Err(anyhow!("index built for {} dims, challenge has {}", idx.dims, dims));
    }
    let n_db = challenge.database_size as i32;
    if n_db != idx.n_db {
        return Err(anyhow!(
            "index built over {} vectors, challenge has {} -- build-index and the solve must use the same seed and track",
            idx.n_db, n_db
        ));
    }
    let ef = hp.ef_search as i32;
    let out_k = 1i32;
    let seed = u64::from_le_bytes(challenge.seed[0..8].try_into().unwrap());

    let k_search = module.load_function("graph_search")?;
    let mut d_out = stream.alloc_zeros::<u32>(n_q as usize)?;
    unsafe {
        stream
            .launch_builder(&k_search)
            .arg(&challenge.d_query_vectors)
            .arg(&n_q)
            .arg(&challenge.d_database_vectors)
            .arg(&dims)
            .arg(&idx.adj)
            .arg(&idx.degree)
            .arg(&n_db)
            .arg(&ef)
            .arg(&out_k)
            .arg(&seed)
            .arg(&mut d_out)
            .launch(cfg(n_q as u32, THREADS))?;
    }
    stream.synchronize()?;
    let out = stream.memcpy_dtov(&d_out)?;
    save_solution(&Solution {
        indexes: out.into_iter().map(|x| (x.min(n_db as u32 - 1)) as usize).collect(),
    })?;
    Ok(())
}
