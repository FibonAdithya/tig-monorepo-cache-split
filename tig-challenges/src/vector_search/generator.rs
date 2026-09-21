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

/// Shift applied to the second latent stream's curand index. The spherical
/// generator draws two latents per row, and `gan_sample_latents` derives a row's
/// whole state from its global index -- `curand_init(seed[global_i % 4],
/// global_i, 0, ..)`, so the index picks both the seed word and the sequence.
/// Two calls at the SAME index would therefore draw the identical numbers, `z_s`
/// would equal `z_t` coordinate for coordinate, and the skip path would carry no
/// randomness of its own. `nytimes_trunk_and_skip_latents_are_independent` exists
/// to catch exactly that.
///
/// The shift changes curand's sequence argument, which is what makes the two
/// streams independent; it leaves `global_i % 4` alone, so both streams read the
/// same seed word, as two sequences of one seed are meant to.
///
/// `generate_vectors_with` rejects `index_base + count >= 1 << 30`, so every row
/// index is below this shift and the two streams cannot meet. A shifted index is
/// then below `1 << 31` and still fits the `int` the kernel takes.
const SKIP_LATENT_INDEX_SHIFT: usize = 1 << 30;

struct SphericalDevice {
    trunk: Vec<DeviceLayer>,
    direction: DeviceLayer,
    tangent_in: DeviceLayer,
    gamma: DeviceLayer,
    beta: DeviceLayer,
    tangent_out: DeviceLayer,
    film_kernel: CudaFunction,
    combine_kernel: CudaFunction,
    cos_r: f32,
    sin_r: f32,
    eps: f32,
    z_trunk: CudaSlice<f32>,
    z_skip: CudaSlice<f32>,
    /// Trunk activations alternate between these; `forward` swaps them so the
    /// trunk's result is always in `h_a`.
    h_a: CudaSlice<f32>,
    h_b: CudaSlice<f32>,
    /// `direction(h)`, then the unit direction `u` in place.
    d: CudaSlice<f32>,
    /// `tangent_in(z_skip)`.
    a: CudaSlice<f32>,
    /// `gamma(h)`.
    g: CudaSlice<f32>,
    /// `beta(h)`.
    b: CudaSlice<f32>,
    /// The modulated value, `leaky(a * (1 + g) + b)`. A buffer of its own rather
    /// than `a` overwritten, because cudarc will not lend one slice as both an
    /// input and an output argument of the same launch.
    m: CudaSlice<f32>,
    /// `tangent_out(m)`, then the tangent `t` in place.
    v: CudaSlice<f32>,
}

enum DeviceArch {
    Mlp(MlpDevice),
    Spherical(SphericalDevice),
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
            Generator::Spherical(s) => {
                // The widest trunk output, so one pair of scratch buffers serves
                // every trunk step -- the same argument as the mlp arm's.
                let widest = s.trunk.iter().map(|l| l.out_dim).max().unwrap();
                // `tangent_in.out_dim`: the blob parser has already checked that
                // gamma, beta and tangent_out all agree with it, so one width
                // sizes `a`, `g`, `b` and `m`.
                let width = s.tangent_in.out_dim;
                let out_dim = s.direction.out_dim;
                let mut trunk = Vec::with_capacity(s.trunk.len());
                for layer in &s.trunk {
                    trunk.push(DeviceLayer::upload(&stream, layer)?);
                }
                DeviceArch::Spherical(SphericalDevice {
                    trunk,
                    // `direction` is bias-free, but the parser gave its `Layer` a
                    // zero bias, so the upload and the launch need nothing
                    // special.
                    direction: DeviceLayer::upload(&stream, &s.direction)?,
                    tangent_in: DeviceLayer::upload(&stream, &s.tangent_in)?,
                    gamma: DeviceLayer::upload(&stream, &s.gamma)?,
                    beta: DeviceLayer::upload(&stream, &s.beta)?,
                    tangent_out: DeviceLayer::upload(&stream, &s.tangent_out)?,
                    film_kernel: module.load_function("gan_film_leaky")?,
                    combine_kernel: module.load_function("gan_sphere_combine")?,
                    cos_r: s.cos_r,
                    sin_r: s.sin_r,
                    eps: s.eps,
                    // Every buffer is sized by `max_rows`, never by a constant:
                    // `forward` allocates nothing, so a chunk larger than this
                    // would run off the end rather than reallocate.
                    z_trunk: stream.alloc_zeros::<f32>(max_rows * s.trunk[0].in_dim)?,
                    z_skip: stream.alloc_zeros::<f32>(max_rows * s.tangent_in.in_dim)?,
                    h_a: stream.alloc_zeros::<f32>(max_rows * widest)?,
                    h_b: stream.alloc_zeros::<f32>(max_rows * widest)?,
                    d: stream.alloc_zeros::<f32>(max_rows * out_dim)?,
                    a: stream.alloc_zeros::<f32>(max_rows * width)?,
                    g: stream.alloc_zeros::<f32>(max_rows * width)?,
                    b: stream.alloc_zeros::<f32>(max_rows * width)?,
                    m: stream.alloc_zeros::<f32>(max_rows * width)?,
                    v: stream.alloc_zeros::<f32>(max_rows * out_dim)?,
                })
            }
            // Reachable: `Generator` also has `StructuredGate`, whose CPU
            // reference exists but whose driver does not yet. A scenario
            // declaring it fails here rather than silently generating something
            // else.
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
            DeviceArch::Spherical(s) => {
                sample_latents(
                    &self.stream,
                    &self.sample_latents_kernel,
                    d_seed,
                    &mut s.z_trunk,
                    rows,
                    s.trunk[0].in_dim,
                    global_index,
                    self.row_block,
                )?;
                sample_latents(
                    &self.stream,
                    &self.sample_latents_kernel,
                    d_seed,
                    &mut s.z_skip,
                    rows,
                    s.tangent_in.in_dim,
                    global_index + SKIP_LATENT_INDEX_SHIFT,
                    self.row_block,
                )
            }
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
            DeviceArch::Spherical(s) => {
                // Destructured for the same reason the mlp arm is: `h_a` and
                // `h_b` must be bindings the borrow checker can see are
                // disjoint, so one can be swapped while the other was the input.
                let SphericalDevice {
                    trunk,
                    direction,
                    tangent_in,
                    gamma,
                    beta,
                    tangent_out,
                    film_kernel,
                    combine_kernel,
                    cos_r,
                    sin_r,
                    eps,
                    z_trunk,
                    z_skip,
                    h_a,
                    h_b,
                    d,
                    a,
                    g,
                    b,
                    m,
                    v,
                } = s;
                for (i, layer) in trunk.iter().enumerate() {
                    // Activation on EVERY trunk layer, the last included: the
                    // PyTorch trunk is [Linear, LeakyReLU] x T. The mlp's last
                    // layer has none, so this is not a copy of that arm.
                    if i == 0 {
                        launch_linear(
                            &self.stream,
                            &self.linear_kernel,
                            z_trunk,
                            layer,
                            h_a,
                            rows,
                            true,
                            0,
                        )?;
                    } else {
                        launch_linear(
                            &self.stream,
                            &self.linear_kernel,
                            h_a,
                            layer,
                            h_b,
                            rows,
                            true,
                            0,
                        )?;
                        // `h_a`'s last use is the call above, so the swap does
                        // not overlap its borrow. The trunk's result is in `h_a`.
                        std::mem::swap(h_a, h_b);
                    }
                }
                // No activation on any of these four: the only nonlinearity left
                // is the leaky ReLU inside `gan_film_leaky`, and `direction` and
                // `tangent_out` feed the geometry directly.
                launch_linear(
                    &self.stream,
                    &self.linear_kernel,
                    h_a,
                    direction,
                    d,
                    rows,
                    false,
                    0,
                )?;
                launch_linear(
                    &self.stream,
                    &self.linear_kernel,
                    z_skip,
                    tangent_in,
                    a,
                    rows,
                    false,
                    0,
                )?;
                launch_linear(
                    &self.stream,
                    &self.linear_kernel,
                    h_a,
                    gamma,
                    g,
                    rows,
                    false,
                    0,
                )?;
                launch_linear(
                    &self.stream,
                    &self.linear_kernel,
                    h_a,
                    beta,
                    b,
                    rows,
                    false,
                    0,
                )?;
                // Element-wise, so the launch covers rows * width values rather
                // than rows. At the largest chunk this crate launches that is
                // 131,072 x 512 = 67,108,864, inside i32.
                let count = rows * tangent_in.out_dim;
                unsafe {
                    self.stream
                        .launch_builder(film_kernel)
                        .arg(&*a)
                        .arg(&*g)
                        .arg(&*b)
                        .arg(&mut *m)
                        .arg(&(count as i32))
                        .launch(row_launch(count, self.row_block))?;
                }
                launch_linear(
                    &self.stream,
                    &self.linear_kernel,
                    m,
                    tangent_out,
                    v,
                    rows,
                    false,
                    0,
                )?;
                let dim = direction.out_dim;
                unsafe {
                    // `d` and `v` are read AND written: the kernel normalises
                    // them in place into `u` and `t`. `out_row_offset`, not 0 --
                    // this is the step that lands the chunk in `dest`.
                    self.stream
                        .launch_builder(combine_kernel)
                        .arg(&mut *d)
                        .arg(&mut *v)
                        .arg(dest)
                        .arg(&(rows as i32))
                        .arg(&(dim as i32))
                        .arg(&*cos_r)
                        .arg(&*sin_r)
                        .arg(&*eps)
                        .arg(&(out_row_offset as i32))
                        .launch(row_launch(rows, self.row_block))?;
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
            DeviceArch::Spherical(s) => {
                // Each row as `[z_t | z_s]`, the order `Spherical::forward_cpu`
                // splits the latent in: `latent[..trunk[0].in_dim]` is the trunk
                // part and the rest is the skip part. The two streams live in
                // separate buffers on the device, so this interleaves them.
                let (t, k) = (s.trunk[0].in_dim, s.tangent_in.in_dim);
                let zt = self.stream.memcpy_dtov(&s.z_trunk.slice(0..rows * t))?;
                let zs = self.stream.memcpy_dtov(&s.z_skip.slice(0..rows * k))?;
                let mut out = Vec::with_capacity(rows * (t + k));
                for row in 0..rows {
                    out.extend_from_slice(&zt[row * t..(row + 1) * t]);
                    out.extend_from_slice(&zs[row * k..(row + 1) * k]);
                }
                Ok((out, None))
            }
        }
    }
}
