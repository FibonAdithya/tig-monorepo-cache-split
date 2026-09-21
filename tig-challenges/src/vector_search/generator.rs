//! GPU drivers: one forward pass per generator architecture, composed from
//! `gan_linear` and the row-wise kernels. Parsing and the CPU references live
//! in `crate::gan_generator`, outside the c004 gate.

use crate::gan_generator::{Generator, Layer};
use anyhow::{anyhow, Result};
use cudarc::driver::{
    safe::LaunchConfig, CudaFunction, CudaModule, CudaSlice, CudaStream, PushKernelArg,
};
use std::sync::Arc;

/// Threads per block for every row-wise kernel, `gan_sample_latents` included.
/// Output must not depend on it; `output_is_invariant_to_launch_geometry` checks.
pub(super) const ROW_BLOCK: u32 = 256;

struct DeviceLayer {
    weight: CudaSlice<f32>,
    bias: CudaSlice<f32>,
    in_dim: usize,
    out_dim: usize,
}

impl DeviceLayer {
    fn upload(stream: &Arc<CudaStream>, layer: &Layer) -> Result<Self> {
        Ok(Self {
            weight: stream.memcpy_stod(&layer.weights)?,
            bias: stream.memcpy_stod(&layer.bias)?,
            in_dim: layer.in_dim,
            out_dim: layer.out_dim,
        })
    }
}

fn row_launch(rows: usize, row_block: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((rows as u32 + row_block - 1) / row_block, 1, 1),
        block_dim: (row_block, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// One `gan_linear` launch. `block_dim` is fixed at (16,16) by the kernel's
/// tile constants and is not a tuning knob.
fn launch_linear(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    input: &CudaSlice<f32>,
    layer: &DeviceLayer,
    output: &mut CudaSlice<f32>,
    rows: usize,
    activate: bool,
    out_row_offset: usize,
) -> Result<()> {
    let cfg = LaunchConfig {
        grid_dim: (
            (rows as u32 + 127) / 128,
            (layer.out_dim as u32 + 63) / 64,
            1,
        ),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(kernel)
            .arg(input)
            .arg(&layer.weight)
            .arg(&layer.bias)
            .arg(output)
            .arg(&(rows as i32))
            .arg(&(layer.in_dim as i32))
            .arg(&(layer.out_dim as i32))
            .arg(&(activate as i32))
            .arg(&(out_row_offset as i32))
            .launch(cfg)?;
    }
    Ok(())
}

fn sample_latents(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    d_seed: &CudaSlice<u8>,
    latents: &mut CudaSlice<f32>,
    rows: usize,
    latent_dim: usize,
    index_offset: usize,
    row_block: u32,
) -> Result<()> {
    unsafe {
        stream
            .launch_builder(kernel)
            .arg(d_seed)
            .arg(&(rows as i32))
            .arg(&(latent_dim as i32))
            .arg(latents)
            .arg(&(index_offset as i32))
            .launch(row_launch(rows, row_block))?;
    }
    Ok(())
}

struct MlpDevice {
    layers: Vec<DeviceLayer>,
    latents: CudaSlice<f32>,
    /// Hidden activations alternate between these; `forward` swaps them so
    /// the current input is always `a`.
    a: CudaSlice<f32>,
    b: CudaSlice<f32>,
    normalize: Option<(CudaFunction, f32)>,
}

enum DeviceArch {
    Mlp(MlpDevice),
}

pub(super) struct DeviceGenerator {
    stream: Arc<CudaStream>,
    sample_latents_kernel: CudaFunction,
    linear_kernel: CudaFunction,
    row_block: u32,
    max_rows: usize,
    arch: DeviceArch,
}

impl DeviceGenerator {
    pub fn new(
        generator: &Generator,
        max_rows: usize,
        row_block: u32,
        module: &Arc<CudaModule>,
        stream: Arc<CudaStream>,
    ) -> Result<Self> {
        let arch = match generator {
            Generator::Mlp(m) => {
                // The widest output over all layers, so one pair of scratch
                // buffers serves every hidden step. The last layer writes to
                // the caller's `dest`, not to these, but including it costs
                // nothing and keeps the bound obviously safe.
                let widest = m.layers.iter().map(|l| l.out_dim).max().unwrap();
                let mut layers = Vec::with_capacity(m.layers.len());
                for layer in &m.layers {
                    layers.push(DeviceLayer::upload(&stream, layer)?);
                }
                DeviceArch::Mlp(MlpDevice {
                    layers,
                    latents: stream.alloc_zeros::<f32>(max_rows * generator.latent_dim())?,
                    a: stream.alloc_zeros::<f32>(max_rows * widest)?,
                    b: stream.alloc_zeros::<f32>(max_rows * widest)?,
                    normalize: match m.normalize_eps {
                        Some(eps) => Some((module.load_function("gan_row_normalize")?, eps)),
                        None => None,
                    },
                })
            }
            // Reachable: `Generator` also has `Spherical` and `StructuredGate`,
            // whose CPU references exist but whose drivers do not yet. A
            // scenario declaring one of those fails here rather than silently
            // generating something else.
            _ => return Err(anyhow!("no GPU driver for this generator architecture yet")),
        };
        Ok(Self {
            sample_latents_kernel: module.load_function("gan_sample_latents")?,
            linear_kernel: module.load_function("gan_linear")?,
            stream,
            row_block,
            max_rows,
            arch,
        })
    }

    /// Fill this chunk's random inputs. `global_index` is the instance-wide
    /// index of the chunk's first row; it enters the curand sequence, so a
    /// row's randomness does not depend on where chunk boundaries fall.
    pub fn sample_inputs(
        &mut self,
        d_seed: &CudaSlice<u8>,
        rows: usize,
        global_index: usize,
    ) -> Result<()> {
        if rows > self.max_rows {
            return Err(anyhow!(
                "chunk of {} rows exceeds the {} this generator was sized for",
                rows,
                self.max_rows
            ));
        }
        match &mut self.arch {
            DeviceArch::Mlp(m) => sample_latents(
                &self.stream,
                &self.sample_latents_kernel,
                d_seed,
                &mut m.latents,
                rows,
                m.layers[0].in_dim,
                global_index,
                self.row_block,
            ),
        }
    }

    /// Forward the sampled inputs into `dest` rows
    /// `[out_row_offset, out_row_offset + rows)`.
    pub fn forward(
        &mut self,
        rows: usize,
        dest: &mut CudaSlice<f32>,
        out_row_offset: usize,
    ) -> Result<()> {
        match &mut self.arch {
            DeviceArch::Mlp(m) => {
                // Destructured rather than used through `m`, so `a` and `b` are
                // distinct bindings the borrow checker can see are disjoint:
                // `input` reborrows `a` while `b` is borrowed mutably.
                let MlpDevice {
                    layers,
                    latents,
                    a,
                    b,
                    normalize,
                } = m;
                let count = layers.len();
                for (i, layer) in layers.iter().enumerate() {
                    let last = i + 1 == count;
                    let input: &CudaSlice<f32> = if i == 0 { latents } else { a };
                    if last {
                        // Only the last layer knows where in `dest` this chunk
                        // belongs; the hidden steps always start at row 0 of
                        // their own scratch buffer.
                        launch_linear(
                            &self.stream,
                            &self.linear_kernel,
                            input,
                            layer,
                            dest,
                            rows,
                            false,
                            out_row_offset,
                        )?;
                    } else {
                        launch_linear(
                            &self.stream,
                            &self.linear_kernel,
                            input,
                            layer,
                            b,
                            rows,
                            true,
                            0,
                        )?;
                        // `input` is dead by here -- its last use is the call
                        // above -- so the swap does not overlap its borrow.
                        std::mem::swap(a, b);
                    }
                }
                if let Some((kernel, eps)) = normalize {
                    let dim = layers[count - 1].out_dim;
                    // `out_row_offset` again, not 0: the kernel must normalise
                    // the rows this chunk just wrote, wherever in `dest` they
                    // landed.
                    unsafe {
                        self.stream
                            .launch_builder(kernel)
                            .arg(dest)
                            .arg(&(rows as i32))
                            .arg(&(dim as i32))
                            .arg(&*eps)
                            .arg(&(out_row_offset as i32))
                            .launch(row_launch(rows, self.row_block))?;
                    }
                }
                Ok(())
            }
        }
    }

    #[cfg(test)]
    pub fn read_inputs(&self, rows: usize) -> Result<(Vec<f32>, Option<Vec<f32>>)> {
        match &self.arch {
            DeviceArch::Mlp(m) => {
                let dim = m.layers[0].in_dim;
                Ok((
                    self.stream.memcpy_dtov(&m.latents.slice(0..rows * dim))?,
                    None,
                ))
            }
        }
    }
}
