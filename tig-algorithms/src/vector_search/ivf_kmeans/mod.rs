// IVF-Flat with a k-means build, on the index-build ABI. A FUEL-METERING
// probe, not a tuned entry: it exists so that `tig-runtime build-index` can be
// run over every scenario's generated database and asked what a real index
// build costs in build fuel. The build parameters are hyperparameters rather
// than consts so one compiled .so covers the whole sweep.
//
// Build (charged to --build-fuel): strided initial centroids, `n_iters` Lloyd
// iterations over the first `train_fraction` of the database, then one
// assignment pass over the whole database and CSR inverted lists. This is the
// build half of an IVF-Flat index as cuVS constructs one (cuVS defaults:
// n_lists 1024, kmeans_n_iters 20, kmeans_trainset_fraction 0.5), minus the
// copy of the vectors into list order, which is n_db * dims of work against the
// assignment pass's n_db * n_lists * dims and is not stored here because the
// search reads rows through the id lists instead.
//
// Search (charged to --fuel): the `n_probe` nearest centroids per query, then
// an exact scan of those lists. Ported from tig-pentesting's
// fixtures/vector_search_ann with the nprobe made a hyperparameter.
//
// The database is a pure function of seed and track, and the ids in the blob
// are positions in the database build_index saw: build-index and the solve
// must be given the SAME seed and track. `solve_challenge` turns a size
// mismatch into a named error; matching seed and track is the operator's job.
use anyhow::{anyhow, Result};
use cudarc::driver::{CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::runtime::sys::cudaDeviceProp;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::sync::{Arc, OnceLock};
use tig_challenges::vector_search::*;

const THREADS: u32 = 256;
const A_DB_TILE: u32 = 8;
const UPD_THREADS: u32 = 128;
/// Must equal `A_PAD - 1` in kernels.cu: the assignment kernel stages one row
/// per `A_PAD` floats of shared memory, so a wider row would overrun it.
const MAX_DIMS: u32 = 256;
/// `ivfk_probe` stages `dims + n_cent` floats of dynamic shared memory; at
/// 8192 lists and 256 dims that is 33 KiB, under the 48 KiB default. 16384
/// would need an opt-in attribute the launch does not set.
const MAX_LISTS: u32 = 8192;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    #[serde(default = "default_n_lists")]
    pub n_lists: u32,
    #[serde(default = "default_n_iters")]
    pub n_iters: u32,
    #[serde(default = "default_train_fraction")]
    pub train_fraction: f64,
    #[serde(default = "default_n_probe")]
    pub n_probe: u32,
}

fn default_n_lists() -> u32 {
    1024
}
fn default_n_iters() -> u32 {
    20
}
fn default_train_fraction() -> f64 {
    0.5
}
fn default_n_probe() -> u32 {
    32
}

impl Default for Hyperparameters {
    fn default() -> Self {
        Self {
            n_lists: default_n_lists(),
            n_iters: default_n_iters(),
            train_fraction: default_train_fraction(),
            n_probe: default_n_probe(),
        }
    }
}

fn parse_hyperparameters(hyperparameters: &Option<Map<String, Value>>) -> Result<Hyperparameters> {
    let hp = match hyperparameters {
        Some(h) => serde_json::from_value::<Hyperparameters>(Value::Object(h.clone()))
            .map_err(|e| anyhow!("Failed to parse hyperparameters: {}", e))?,
        None => Hyperparameters::default(),
    };
    if hp.n_lists == 0 || hp.n_lists > MAX_LISTS {
        return Err(anyhow!("n_lists must be in 1..={}, got {}", MAX_LISTS, hp.n_lists));
    }
    if !(hp.train_fraction > 0.0 && hp.train_fraction <= 1.0) {
        return Err(anyhow!("train_fraction must be in (0, 1], got {}", hp.train_fraction));
    }
    if hp.n_probe == 0 || hp.n_probe > hp.n_lists {
        return Err(anyhow!("n_probe must be in 1..=n_lists ({}), got {}", hp.n_lists, hp.n_probe));
    }
    Ok(hp)
}

pub fn help() {
    println!("IVF-Flat with a k-means build on the index-build ABI (fuel probe).");
    println!("Hyperparameters (JSON): n_lists (1024), n_iters (20), train_fraction (0.5), n_probe (32).");
}

struct LoadedIndex {
    cent: CudaSlice<f32>,
    offsets: CudaSlice<u32>,
    ids: CudaSlice<u32>,
    n_cent: i32,
    dims: i32,
    n_db: i32,
}

static INDEX: OnceLock<LoadedIndex> = OnceLock::new();

fn cfg(grid: u32, block: u32, shared: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: shared,
    }
}

/// count -> scan -> scatter over the first `n_rows` assignments, into CSR
/// lists. `d_counts` and `d_cursor` are zeroed here so the same buffers serve
/// every k-means iteration.
#[allow(clippy::too_many_arguments)]
unsafe fn build_lists(
    module: &Arc<CudaModule>,
    stream: &Arc<CudaStream>,
    d_assign: &CudaSlice<u32>,
    n_rows: i32,
    n_cent: i32,
    d_counts: &mut CudaSlice<u32>,
    d_offsets: &mut CudaSlice<u32>,
    d_cursor: &mut CudaSlice<u32>,
    d_ids: &mut CudaSlice<u32>,
) -> Result<()> {
    let k_count = module.load_function("ivfk_count")?;
    let k_scan = module.load_function("ivfk_scan")?;
    let k_scatter = module.load_function("ivfk_scatter")?;
    stream.memset_zeros(&mut *d_counts)?;
    stream.memset_zeros(&mut *d_cursor)?;
    stream
        .launch_builder(&k_count)
        .arg(&*d_assign)
        .arg(&n_rows)
        .arg(&mut *d_counts)
        .launch(cfg((n_rows as u32).div_ceil(THREADS), THREADS, 0))?;
    stream
        .launch_builder(&k_scan)
        .arg(&*d_counts)
        .arg(&n_cent)
        .arg(&mut *d_offsets)
        .launch(cfg(1, 1, 0))?;
    stream
        .launch_builder(&k_scatter)
        .arg(&*d_assign)
        .arg(&n_rows)
        .arg(&*d_offsets)
        .arg(&mut *d_cursor)
        .arg(&mut *d_ids)
        .launch(cfg((n_rows as u32).div_ceil(THREADS), THREADS, 0))?;
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
    let n_db = database.database_size as i32;
    let dims = database.vector_dims as i32;
    if database.vector_dims > MAX_DIMS {
        return Err(anyhow!("{} dims exceeds the kernel's {}", dims, MAX_DIMS));
    }
    let n_cent = hp.n_lists as i32;
    if n_db < n_cent {
        return Err(anyhow!("n_lists {} exceeds the database's {} rows", n_cent, n_db));
    }
    // At least n_cent rows, so the strided initial pick never repeats a row.
    let n_train = ((n_db as f64 * hp.train_fraction).floor() as i32).clamp(n_cent, n_db);

    let k_pick = module.load_function("ivfk_pick_centroids")?;
    let k_assign = module.load_function("ivfk_assign")?;
    let k_update = module.load_function("ivfk_update")?;

    let mut d_cent = stream.alloc_zeros::<f32>((n_cent * dims) as usize)?;
    let mut d_assign = stream.alloc_zeros::<u32>(n_db as usize)?;
    let mut d_counts = stream.alloc_zeros::<u32>(n_cent as usize)?;
    let mut d_offsets = stream.alloc_zeros::<u32>((n_cent + 1) as usize)?;
    let mut d_cursor = stream.alloc_zeros::<u32>(n_cent as usize)?;
    let mut d_ids = stream.alloc_zeros::<u32>(n_db as usize)?;

    unsafe {
        stream
            .launch_builder(&k_pick)
            .arg(&database.d_database_vectors)
            .arg(&n_train)
            .arg(&dims)
            .arg(&n_cent)
            .arg(&mut d_cent)
            .launch(cfg(((n_cent * dims) as u32).div_ceil(THREADS), THREADS, 0))?;

        // Lloyd iterations over the training subset: assign, list, update.
        for _ in 0..hp.n_iters {
            stream
                .launch_builder(&k_assign)
                .arg(&database.d_database_vectors)
                .arg(&n_train)
                .arg(&d_cent)
                .arg(&n_cent)
                .arg(&dims)
                .arg(&mut d_assign)
                .launch(cfg((n_train as u32).div_ceil(A_DB_TILE), THREADS, 0))?;
            build_lists(
                &module, &stream, &d_assign, n_train, n_cent,
                &mut d_counts, &mut d_offsets, &mut d_cursor, &mut d_ids,
            )?;
            stream
                .launch_builder(&k_update)
                .arg(&database.d_database_vectors)
                .arg(&dims)
                .arg(&d_offsets)
                .arg(&d_ids)
                .arg(&n_cent)
                .arg(&mut d_cent)
                .launch(cfg(n_cent as u32, UPD_THREADS, 0))?;
        }

        // Final pass: every database row into its list.
        stream
            .launch_builder(&k_assign)
            .arg(&database.d_database_vectors)
            .arg(&n_db)
            .arg(&d_cent)
            .arg(&n_cent)
            .arg(&dims)
            .arg(&mut d_assign)
            .launch(cfg((n_db as u32).div_ceil(A_DB_TILE), THREADS, 0))?;
        build_lists(
            &module, &stream, &d_assign, n_db, n_cent,
            &mut d_counts, &mut d_offsets, &mut d_cursor, &mut d_ids,
        )?;
    }
    stream.synchronize()?;

    let cent = stream.memcpy_dtov(&d_cent)?;
    let offsets = stream.memcpy_dtov(&d_offsets)?;
    let ids = stream.memcpy_dtov(&d_ids)?;

    let mut blob = Vec::with_capacity(12 + (cent.len() + offsets.len() + ids.len()) * 4);
    blob.extend_from_slice(&(n_cent as u32).to_le_bytes());
    blob.extend_from_slice(&(dims as u32).to_le_bytes());
    blob.extend_from_slice(&(n_db as u32).to_le_bytes());
    for v in &cent {
        blob.extend_from_slice(&v.to_le_bytes());
    }
    for v in &offsets {
        blob.extend_from_slice(&v.to_le_bytes());
    }
    for v in &ids {
        blob.extend_from_slice(&v.to_le_bytes());
    }
    Ok(blob)
}

// Called exactly once per process: tig-runtime builds the nonce-free database
// and loads the index once, above the nonce loop.
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
    let n_cent = rd_u32(&blob[0..4]);
    let dims = rd_u32(&blob[4..8]);
    let n_db = rd_u32(&blob[8..12]);
    if n_cent == 0 || dims == 0 || n_db == 0 || n_cent > i32::MAX as u32 || dims > i32::MAX as u32 || n_db > i32::MAX as u32 {
        return Err(anyhow!(
            "index blob header out of range: n_cent={} dims={} n_db={}",
            n_cent, dims, n_db
        ));
    }
    let n_cent_f = (n_cent as usize)
        .checked_mul(dims as usize)
        .ok_or_else(|| anyhow!("index blob header overflows: n_cent={} dims={}", n_cent, dims))?;
    let n_off = n_cent as usize + 1;
    let n_ids = n_db as usize;
    let mut want = 12usize;
    for n in [n_cent_f, n_off, n_ids] {
        let bytes = n.checked_mul(4).ok_or_else(|| anyhow!("index blob length overflows a usize"))?;
        want = want.checked_add(bytes).ok_or_else(|| anyhow!("index blob length overflows a usize"))?;
    }
    if blob.len() != want {
        return Err(anyhow!("index blob is {} bytes, expected {}", blob.len(), want));
    }

    let body = &blob[12..];
    let (cent_b, rest) = body.split_at(n_cent_f * 4);
    let (off_b, ids_b) = rest.split_at(n_off * 4);
    let cent: Vec<f32> = cent_b
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let offsets: Vec<u32> = off_b.chunks_exact(4).map(rd_u32).collect();
    let ids: Vec<u32> = ids_b.chunks_exact(4).map(rd_u32).collect();

    let loaded = LoadedIndex {
        cent: stream.memcpy_stod(&cent)?,
        offsets: stream.memcpy_stod(&offsets)?,
        ids: stream.memcpy_stod(&ids)?,
        n_cent: n_cent as i32,
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
    // No silent fallback to a scan: a run that quietly searched without an
    // index would report a number for something else entirely.
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
    let n_cent = idx.n_cent;
    let nprobe = (hp.n_probe as i32).min(n_cent);

    let k_probe = module.load_function("ivfk_probe")?;
    let k_search = module.load_function("ivfk_search")?;

    let mut d_probes = stream.alloc_zeros::<u32>((n_q * nprobe) as usize)?;
    let mut d_out = stream.alloc_zeros::<u64>(n_q as usize)?;

    unsafe {
        stream
            .launch_builder(&k_probe)
            .arg(&challenge.d_query_vectors)
            .arg(&n_q)
            .arg(&idx.cent)
            .arg(&n_cent)
            .arg(&dims)
            .arg(&nprobe)
            .arg(&mut d_probes)
            .launch(cfg(n_q as u32, THREADS, ((dims + n_cent) as u32) * 4))?;
        stream
            .launch_builder(&k_search)
            .arg(&challenge.d_query_vectors)
            .arg(&n_q)
            .arg(&challenge.d_database_vectors)
            .arg(&dims)
            .arg(&idx.offsets)
            .arg(&idx.ids)
            .arg(&d_probes)
            .arg(&nprobe)
            .arg(&mut d_out)
            .launch(cfg(n_q as u32, THREADS, (dims as u32) * 4))?;
    }
    stream.synchronize()?;
    let out = stream.memcpy_dtov(&d_out)?;
    save_solution(&Solution {
        indexes: out.into_iter().map(|x| x as usize).collect(),
    })?;
    Ok(())
}
