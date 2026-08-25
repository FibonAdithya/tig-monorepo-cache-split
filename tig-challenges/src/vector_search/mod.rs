use crate::QUALITY_PRECISION;
use anyhow::{anyhow, Result};
use cudarc::{
    driver::{safe::LaunchConfig, CudaModule, CudaSlice, CudaStream, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use std::sync::Arc;

mod generator;
mod scenarios;
use generator::{v1_weights, LATENT_DIM};
pub use scenarios::{Scenario, ScenarioConfig};

impl_kv_string_serde! {
    Track {
        n_queries: u32,
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
    pub num_queries: u32,
    pub vector_dims: u32,
    pub database_size: u32,
    pub d_database_vectors: CudaSlice<f32>,
    pub d_query_vectors: CudaSlice<f32>,
}

pub const MAX_THREADS_PER_BLOCK: u32 = 1024;
const FORWARD_CHUNK: usize = 65_536;

/// Calibrated so GAN instances reproduce the quality band the Gaussian
/// generator produced on mainnet: ~71,862 at n_queries=7000 rising to ~77,696
/// at 15000, against a min_active_quality of 68,500 on every track.
///
/// The previous form, `(11.0 - avg_dist) / 11.0`, assumed 250-dim hypercube
/// distances. GAN output has an optimal avg_dist of ~1.12-1.16 rather than
/// ~10.18, so under it an exact solver scored 894,982 against a target of
/// 71,862 -- twelve times too high, with every solution including a
/// deliberately terrible one landing far above min_active_quality.
///
/// Two constants rather than one because matching the spread alone leaves the
/// absolute level wrong, and the level is what the qualifier machinery keys on.
/// Fitted by scripts/calibrate_vector_search.py across all five active tracks
/// from 24 nonces each, measured on real generated instances. Worst residual
/// 282 quality units, at n_queries=11000; every other track within 138.
const QUALITY_OFFSET: f64 = 1.616563;
const QUALITY_SCALE: f64 = 6.399004;

impl Challenge {
    pub fn generate_instance(
        seed: &[u8; 32],
        track: &Track,
        module: Arc<CudaModule>,
        stream: Arc<CudaStream>,
        _prop: &cudaDeviceProp,
    ) -> Result<Self> {
        let weights = v1_weights()?;
        let layers = &weights.layers;
        let vector_dims = layers
            .last()
            .ok_or_else(|| anyhow!("generator has no layers"))?
            .out_dim;
        let widest = layers.iter().map(|layer| layer.out_dim).max().unwrap();
        let database_size = 100 * track.n_queries;

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
            stream.alloc_zeros::<f32>(track.n_queries as usize * vector_dims)?;

        for (dest_is_query, count) in [
            (false, database_size as usize),
            (true, track.n_queries as usize),
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
            num_queries: track.n_queries.clone(),
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

    conditional_pub!(
        fn evaluate_solution(
            &self,
            solution: &Solution,
            module: Arc<CudaModule>,
            stream: Arc<CudaStream>,
            prop: &cudaDeviceProp,
        ) -> Result<i32> {
            let avg_dist = self.evaluate_average_distance(solution, module, stream, prop)?;
            let quality = (QUALITY_OFFSET - avg_dist as f64) / QUALITY_SCALE;
            let quality = quality.clamp(-10.0, 10.0) * QUALITY_PRECISION as f64;
            let quality = quality.round() as i32;
            Ok(quality)
        }
    );
}
