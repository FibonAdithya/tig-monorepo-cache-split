// Batched NSW (single-layer HNSW) on the index-build ABI: a FUEL-METERING
// probe of incremental graph insertion, not a tuned entry. Build parameters
// are hyperparameters so one instrumented .so covers a sweep.
//
// Build (charged to --build-fuel): nodes are inserted in batches whose size
// doubles from 1 up to `batch`; each batch node beam-searches the inserted
// prefix with `ef_construction` candidates (search.cu), keeps its `m` nearest
// as out-edges, and nsw_link adds the reverse edges (build.cu). Search
// (charged to --fuel): the same beam search with `ef_search`.
//
// Blob: u32 m, u32 dims, u32 n_db, then n_db * m u32 neighbour ids, with
// 0xFFFFFFFF in unused slots.
use anyhow::{anyhow, Result};
use cudarc::driver::{CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::runtime::sys::cudaDeviceProp;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::sync::{Arc, OnceLock};
use tig_challenges::vector_search::*;

const THREADS: u32 = 256;
/// Must equal `GS_MAX_DEGREE` in search.cu.
const MAX_DEGREE: u32 = 64;
/// Must equal `GS_MAX_EF` in search.cu.
const MAX_EF: u32 = 512;
/// Must equal `GS_MAX_DIMS`; rows must also be float4-wide.
const MAX_DIMS: u32 = 256;
const EMPTY: u32 = 0xFFFF_FFFF;
const BUILD_SEED: u64 = 0x5EED_0000_0000_0002;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    #[serde(default = "default_m")]
    pub m: u32,
    #[serde(default = "default_ef_construction")]
    pub ef_construction: u32,
    #[serde(default = "default_batch")]
    pub batch: u32,
    #[serde(default = "default_ef_search")]
    pub ef_search: u32,
}

fn default_m() -> u32 {
    16
}
fn default_ef_construction() -> u32 {
    100
}
fn default_batch() -> u32 {
    4096
}
fn default_ef_search() -> u32 {
    64
}

impl Default for Hyperparameters {
    fn default() -> Self {
        Self {
            m: default_m(),
            ef_construction: default_ef_construction(),
            batch: default_batch(),
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
    if hp.m == 0 || hp.m > MAX_DEGREE {
        return Err(anyhow!("m must be in 1..={}, got {}", MAX_DEGREE, hp.m));
    }
    if hp.ef_construction < hp.m || hp.ef_construction > MAX_EF {
        return Err(anyhow!("ef_construction must be in m..={}, got {}", MAX_EF, hp.ef_construction));
    }
    if hp.ef_search == 0 || hp.ef_search > MAX_EF {
        return Err(anyhow!("ef_search must be in 1..={}, got {}", MAX_EF, hp.ef_search));
    }
    if hp.batch == 0 {
        return Err(anyhow!("batch must be positive"));
    }
    Ok(hp)
}

pub fn help() {
    println!("Batched NSW (single-layer HNSW) on the index-build ABI (fuel probe).");
    println!("Hyperparameters (JSON): m (16), ef_construction (100), batch (4096), ef_search (64).");
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
    let m = hp.m as i32;
    let total = (n_db as usize) * (m as usize);
    if total > i32::MAX as usize {
        return Err(anyhow!("n_db * m overflows the kernels' i32 indexing"));
    }

    let k_init = module.load_function("nsw_init")?;
    let k_search = module.load_function("graph_search")?;
    let k_link = module.load_function("nsw_link")?;

    let mut d_adj = stream.alloc_zeros::<u32>(total)?;
    let mut d_adjd = stream.alloc_zeros::<f32>(total)?;
    let mut d_count = stream.alloc_zeros::<u32>(n_db as usize)?;
    let mut d_lock = stream.alloc_zeros::<u32>(n_db as usize)?;
    let mut d_found = stream.alloc_zeros::<u32>((hp.batch as usize) * (m as usize))?;
    let total_i = total as i32;
    let ef_c = hp.ef_construction as i32;

    unsafe {
        stream
            .launch_builder(&k_init)
            .arg(&mut d_adj)
            .arg(&mut d_adjd)
            .arg(&total_i)
            .arg(&mut d_count)
            .arg(&mut d_lock)
            .arg(&n_db)
            .launch(cfg((total as u32).max(n_db as u32).div_ceil(THREADS), THREADS))?;

        let mut start: i32 = 0;
        let mut size: i32 = 1;
        while start < n_db {
            let nb = size.min(n_db - start);
            let q = database
                .d_database_vectors
                .slice((start as usize * dims as usize)..((start + nb) as usize * dims as usize));
            stream
                .launch_builder(&k_search)
                .arg(&q)
                .arg(&nb)
                .arg(&database.d_database_vectors)
                .arg(&dims)
                .arg(&d_adj)
                .arg(&m)
                .arg(&start) // n_nodes: only the inserted prefix is searchable
                .arg(&ef_c)
                .arg(&m) // out_k
                .arg(&BUILD_SEED)
                .arg(&mut d_found)
                .launch(cfg(nb as u32, THREADS))?;
            stream
                .launch_builder(&k_link)
                .arg(&database.d_database_vectors)
                .arg(&dims)
                .arg(&m)
                .arg(&start)
                .arg(&nb)
                .arg(&d_found)
                .arg(&mut d_adj)
                .arg(&mut d_adjd)
                .arg(&mut d_count)
                .arg(&mut d_lock)
                .launch(cfg((nb as u32).div_ceil(THREADS), THREADS))?;
            start += nb;
            size = (size * 2).min(hp.batch as i32);
        }
    }
    stream.synchronize()?;

    let adj = stream.memcpy_dtov(&d_adj)?;
    let mut blob = Vec::with_capacity(12 + adj.len() * 4);
    blob.extend_from_slice(&(m as u32).to_le_bytes());
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
        return Err(anyhow!("index blob header out of range: m={} dims={} n_db={}", degree, dims, n_db));
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
    if adj.iter().any(|&v| v != EMPTY && v >= n_db) {
        return Err(anyhow!("index blob has a neighbour id outside the database"));
    }
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
