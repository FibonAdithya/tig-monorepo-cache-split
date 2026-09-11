use crate::Seeds;
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

/// One forward pass of the generator into `dest`.
///
/// `index_base` is preserved from the pre-split generator even though the two
/// halves now use different seeds and no longer need it to separate their
/// latent streams. Keeping it means `gan_sample_latents` is called exactly as
/// before, so `kernels.cu` and every algorithm's PTX are untouched.
fn generate_vectors(
    seed: &[u8; 32],
    count: usize,
    index_base: usize,
    dest: &mut CudaSlice<f32>,
    layers: &[generator::Layer],
    widest: usize,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
) -> Result<()> {
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
                unsafe {
                    stream
                        .launch_builder(&linear_kernel)
                        .arg(input)
                        .arg(d_weight)
                        .arg(d_bias)
                        .arg(&mut *dest)
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

/// On-disk form of a `Database`, so a slave can generate it once per batch
/// and let every later runtime and local-verifier process read it back
/// instead of regenerating. The header carries the seed it was generated
/// from, so a copy from another precommit is refused rather than searched.
///
/// This is a local optimisation only: consensus verification regenerates the
/// database from the seed and never reads this file.
const DB_MAGIC: &[u8; 8] = b"TIGVSDB1";
const DB_HEADER_LEN: usize = 8 + 32 + 4 + 4;

pub fn encode_database(
    db_seed: &[u8; 32],
    vector_dims: u32,
    database_size: u32,
    host: &[f32],
) -> Vec<u8> {
    let mut out = Vec::with_capacity(DB_HEADER_LEN + host.len() * 4);
    out.extend_from_slice(DB_MAGIC);
    out.extend_from_slice(db_seed);
    out.extend_from_slice(&vector_dims.to_le_bytes());
    out.extend_from_slice(&database_size.to_le_bytes());
    for x in host {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// Returns `(vector_dims, database_size, rows)` or an error naming what was
/// wrong with the file.
pub fn decode_database(bytes: &[u8], db_seed: &[u8; 32]) -> Result<(u32, u32, Vec<f32>)> {
    if bytes.len() < DB_HEADER_LEN || &bytes[0..8] != DB_MAGIC {
        return Err(anyhow!("database cache is not a TIGVSDB1 file"));
    }
    if &bytes[8..40] != db_seed {
        return Err(anyhow!(
            "database cache was generated from a different seed (another precommit?)"
        ));
    }
    let vector_dims = u32::from_le_bytes(bytes[40..44].try_into().unwrap());
    let database_size = u32::from_le_bytes(bytes[44..48].try_into().unwrap());
    let expected = DB_HEADER_LEN + (vector_dims as usize) * (database_size as usize) * 4;
    if bytes.len() != expected {
        return Err(anyhow!(
            "database cache is {} bytes but its header implies {}",
            bytes.len(),
            expected
        ));
    }
    let rows = bytes[DB_HEADER_LEN..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    Ok((vector_dims, database_size, rows))
}

impl Database {
    /// Serialise for the on-disk cache. Downloads the rows from the device.
    pub fn to_bytes(&self, db_seed: &[u8; 32], stream: Arc<CudaStream>) -> Result<Vec<u8>> {
        let host: Vec<f32> = stream.memcpy_dtov(&self.d_database_vectors)?;
        Ok(encode_database(
            db_seed,
            self.vector_dims,
            self.database_size,
            &host,
        ))
    }

    /// Rebuild from the on-disk cache. Refuses a file from another seed or a
    /// different scenario, so a stale cache is an error, never a wrong answer.
    pub fn from_bytes(
        bytes: &[u8],
        db_seed: &[u8; 32],
        track: &Track,
        stream: Arc<CudaStream>,
    ) -> Result<Self> {
        let (vector_dims, database_size, rows) = decode_database(bytes, db_seed)?;
        let config = ScenarioConfig::from(track.s);
        if vector_dims as usize != config.vector_dims || database_size != config.database_size {
            return Err(anyhow!(
                "database cache is {}x{} but scenario {} is {}x{}",
                database_size,
                vector_dims,
                track.s,
                config.database_size,
                config.vector_dims
            ));
        }
        let d_database_vectors = stream.memcpy_stod(&rows)?;
        stream.synchronize()?;
        Ok(Self {
            scenario: track.s,
            vector_dims,
            database_size,
            d_database_vectors,
        })
    }

    pub fn generate(
        db_seed: &[u8; 32],
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

        let mut d_database_vectors =
            stream.alloc_zeros::<f32>(database_size as usize * vector_dims)?;
        generate_vectors(
            db_seed,
            database_size as usize,
            0,
            &mut d_database_vectors,
            layers,
            widest,
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
        let weights = weights_from(config.weights)?;
        let layers = &weights.layers;
        let widest = layers.iter().map(|layer| layer.out_dim).max().unwrap();
        let vector_dims = db.vector_dims as usize;
        let n_queries = config.n_queries;

        let mut d_query_vectors = stream.alloc_zeros::<f32>(n_queries as usize * vector_dims)?;
        generate_vectors(
            &seeds.instance,
            n_queries as usize,
            db.database_size as usize,
            &mut d_query_vectors,
            layers,
            widest,
            module,
            stream.clone(),
        )?;

        // Owned copy, not a borrow: `Challenge`'s layout must stay
        // byte-identical or every existing algorithm .so reads these fields at
        // the wrong offsets. The copy is a few ms per nonce, measured on the
        // reference branch.
        let d_database_vectors = stream.clone_dtod(&db.d_database_vectors)?;
        stream.synchronize()?;

        Ok(Self {
            // The algorithm's RNG seed, not the one that made the queries.
            seed: seeds.algo,
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
    fn database_codec_round_trips() {
        let seed = [7u8; 32];
        let rows = vec![0.5f32, -1.25, 3.0, 1e-7, 2.0, 4.0];
        let bytes = encode_database(&seed, 3, 2, &rows);
        let (dims, size, got) = decode_database(&bytes, &seed).unwrap();
        assert_eq!((dims, size), (3, 2));
        assert_eq!(got, rows);
    }

    #[test]
    fn database_codec_refuses_another_seed_and_a_truncated_file() {
        let seed = [7u8; 32];
        let bytes = encode_database(&seed, 3, 2, &[0.0; 6]);
        let err = decode_database(&bytes, &[8u8; 32]).unwrap_err();
        assert!(err.to_string().contains("different seed"), "{}", err);
        let err = decode_database(&bytes[..bytes.len() - 4], &seed).unwrap_err();
        assert!(err.to_string().contains("header implies"), "{}", err);
        let err = decode_database(b"nope", &seed).unwrap_err();
        assert!(err.to_string().contains("TIGVSDB1"), "{}", err);
    }

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
