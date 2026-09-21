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
/// `generate_vectors_with` rejects `index_base + count >= SKIP_LATENT_INDEX_SHIFT`,
/// so every row index is below this shift and the two streams cannot meet. A
/// shifted index is then below `1 << 31` and still fits the `int` the kernel
/// takes. `pub(super)` so that guard reads this constant rather than repeating
/// the literal: two literals for one quantity could be changed apart, and
/// raising only this one would let the two streams overlap again.
pub(super) const SKIP_LATENT_INDEX_SHIFT: usize = 1 << 30;

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

struct GateDevice {
    trunk: Vec<DeviceLayer>,
    magnitude_head: DeviceLayer,
    gate_head: DeviceLayer,
    sparsity_head: DeviceLayer,
    coupling: DeviceLayer,
    smoothing: DeviceLayer,
    noise_kernel: CudaFunction,
    apply_kernel: CudaFunction,
    logit_clamp: f32,
    magnitude_floor: f32,
    eps: f32,
    latents: CudaSlice<f32>,
    /// Trunk activations alternate between these; `forward` swaps them so the
    /// trunk's result is always in `h_a`.
    h_a: CudaSlice<f32>,
    h_b: CudaSlice<f32>,
    /// `magnitude_head(h)`, before the softplus and the floor -- both of which
    /// `gan_gate_apply` applies, so nothing here is pre-activated.
    mag_pre: CudaSlice<f32>,
    /// `gate_head(h)`.
    gate: CudaSlice<f32>,
    /// `coupling(gate)`. A buffer of its own rather than `gate` overwritten,
    /// because cudarc will not lend one slice as both an input and an output
    /// argument of the same launch.
    coupled: CudaSlice<f32>,
    /// `sparsity_head(h)`: ONE value per row, not per coordinate, so this is
    /// `max_rows` floats and `gan_gate_apply` indexes it by row.
    sparsity: CudaSlice<f32>,
    /// The logistic noise `gan_gate_noise` draws, before smoothing. Filled by
    /// `sample_inputs`, not by `forward`, because it is a random input like the
    /// latents; `read_inputs` hands this one back.
    raw_noise: CudaSlice<f32>,
    /// `smoothing(raw_noise)`, which is what `gan_gate_apply` adds to the logit.
    smooth_noise: CudaSlice<f32>,
}

enum DeviceArch {
    Mlp(MlpDevice),
    Spherical(SphericalDevice),
    StructuredGate(GateDevice),
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
            Generator::StructuredGate(g) => {
                // The widest trunk output, so one pair of scratch buffers serves
                // every trunk step -- the same argument as the mlp arm's.
                let widest = g.trunk.iter().map(|l| l.out_dim).max().unwrap();
                // `magnitude_head.out_dim`: the blob parser has already checked
                // that gate_head, coupling and smoothing all agree with it, so
                // one width sizes every per-coordinate buffer. Read from the
                // blob, never a literal: the synthetic gate test in mod.rs
                // drives this driver with output_dim 4 and a latent of 2, so
                // nothing here may assume SIFT's 128.
                let out_dim = g.magnitude_head.out_dim;
                let mut trunk = Vec::with_capacity(g.trunk.len());
                for layer in &g.trunk {
                    trunk.push(DeviceLayer::upload(&stream, layer)?);
                }
                DeviceArch::StructuredGate(GateDevice {
                    trunk,
                    magnitude_head: DeviceLayer::upload(&stream, &g.magnitude_head)?,
                    gate_head: DeviceLayer::upload(&stream, &g.gate_head)?,
                    sparsity_head: DeviceLayer::upload(&stream, &g.sparsity_head)?,
                    // `coupling` and `smoothing` are bias-free, but the parser
                    // gave each `Layer` a zero bias, so the upload and the
                    // launch need nothing special -- as with `direction` above.
                    coupling: DeviceLayer::upload(&stream, &g.coupling)?,
                    smoothing: DeviceLayer::upload(&stream, &g.smoothing)?,
                    noise_kernel: module.load_function("gan_gate_noise")?,
                    apply_kernel: module.load_function("gan_gate_apply")?,
                    logit_clamp: g.logit_clamp,
                    magnitude_floor: g.magnitude_floor,
                    eps: g.eps,
                    // Every buffer is sized by `max_rows`, never by a constant:
                    // `forward` allocates nothing, so a chunk larger than this
                    // would run off the end rather than reallocate.
                    latents: stream.alloc_zeros::<f32>(max_rows * g.trunk[0].in_dim)?,
                    h_a: stream.alloc_zeros::<f32>(max_rows * widest)?,
                    h_b: stream.alloc_zeros::<f32>(max_rows * widest)?,
                    mag_pre: stream.alloc_zeros::<f32>(max_rows * out_dim)?,
                    gate: stream.alloc_zeros::<f32>(max_rows * out_dim)?,
                    coupled: stream.alloc_zeros::<f32>(max_rows * out_dim)?,
                    // One value per row, so no `out_dim` factor. The parser
                    // checks `sparsity_head.out_dim == 1`.
                    sparsity: stream.alloc_zeros::<f32>(max_rows)?,
                    raw_noise: stream.alloc_zeros::<f32>(max_rows * out_dim)?,
                    smooth_noise: stream.alloc_zeros::<f32>(max_rows * out_dim)?,
                })
            }
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
            DeviceArch::StructuredGate(g) => {
                sample_latents(
                    &self.stream,
                    &self.sample_latents_kernel,
                    d_seed,
                    &mut g.latents,
                    rows,
                    g.trunk[0].in_dim,
                    global_index,
                    self.row_block,
                )?;
                // The gate's noise is a random INPUT, like the latents, so it is
                // drawn here rather than in `forward`: `read_inputs` must be
                // able to return it after `sample_inputs` alone, which is how
                // both gate-noise tests call it.
                //
                // `global_index`, not a chunk-local index, for exactly the
                // reason `gan_sample_latents` takes it: the kernel derives a
                // row's whole curand state from the index it is given, so a
                // chunk-local index would make a row's noise depend on where
                // chunk boundaries happen to fall, and
                // `sift_output_is_invariant_to_launch_geometry` would fail.
                // `gan_gate_noise` adds GATE_NOISE_SEQUENCE_BASE to it inside
                // the kernel, which is what keeps this stream disjoint from the
                // latents' -- so nothing is shifted here, unlike the spherical
                // arm's second `sample_latents` call above.
                let dim = g.magnitude_head.out_dim as i32;
                let eps = g.eps;
                unsafe {
                    self.stream
                        .launch_builder(&g.noise_kernel)
                        .arg(d_seed)
                        .arg(&(rows as i32))
                        .arg(&dim)
                        .arg(&eps)
                        .arg(&mut g.raw_noise)
                        .arg(&(global_index as i32))
                        .launch(row_launch(rows, self.row_block))?;
                }
                Ok(())
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
            DeviceArch::StructuredGate(g) => {
                // Destructured for the same reason the other two arms are:
                // `h_a` and `h_b` must be bindings the borrow checker can see
                // are disjoint, so one can be swapped while the other was the
                // input to the launch before it. `noise_kernel` is not used
                // here -- the noise is drawn in `sample_inputs` -- but it is
                // named rather than covered by `..` so that a field added to
                // `GateDevice` later cannot be silently ignored.
                let GateDevice {
                    trunk,
                    magnitude_head,
                    gate_head,
                    sparsity_head,
                    coupling,
                    smoothing,
                    noise_kernel: _,
                    apply_kernel,
                    logit_clamp,
                    magnitude_floor,
                    eps,
                    latents,
                    h_a,
                    h_b,
                    mag_pre,
                    gate,
                    coupled,
                    sparsity,
                    raw_noise,
                    smooth_noise,
                } = g;
                for (i, layer) in trunk.iter().enumerate() {
                    // Activation on EVERY trunk layer, the last included: the
                    // PyTorch trunk is [Linear, LeakyReLU] x T, and
                    // `StructuredGate::trunk_cpu` passes `true` for every
                    // layer. The mlp arm's last layer has none, so this is not
                    // a copy of that arm -- it matches the spherical one.
                    if i == 0 {
                        launch_linear(
                            &self.stream,
                            &self.linear_kernel,
                            latents,
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
                // No activation on any of these five. Every remaining
                // nonlinearity is inside `gan_gate_apply` -- the tanh on the
                // logit, the softplus on the magnitude, and the gate itself --
                // and `coupling` and `smoothing` are linear maps the exporter
                // baked from a Conv3d and a fixed smoothing kernel.
                //
                // The order matters: `coupling` consumes `gate`, so that launch
                // follows `gate_head`'s. Launches on one stream run in issue
                // order, so nothing else is needed to sequence them.
                launch_linear(
                    &self.stream,
                    &self.linear_kernel,
                    h_a,
                    magnitude_head,
                    mag_pre,
                    rows,
                    false,
                    0,
                )?;
                launch_linear(
                    &self.stream,
                    &self.linear_kernel,
                    h_a,
                    gate_head,
                    gate,
                    rows,
                    false,
                    0,
                )?;
                launch_linear(
                    &self.stream,
                    &self.linear_kernel,
                    gate,
                    coupling,
                    coupled,
                    rows,
                    false,
                    0,
                )?;
                // out_dim 1: one sparsity value per row. `gan_linear`'s grid is
                // ceil(out_dim / 64) = 1 block in y, and its `col >= out_dim`
                // guards stop every thread but one from writing, so the narrow
                // layer needs no special case.
                launch_linear(
                    &self.stream,
                    &self.linear_kernel,
                    h_a,
                    sparsity_head,
                    sparsity,
                    rows,
                    false,
                    0,
                )?;
                // The noise the gate compares against is the SMOOTHED noise.
                // `raw_noise` is what `sample_inputs` drew and what
                // `read_inputs` returns; the CPU reference smooths it itself.
                launch_linear(
                    &self.stream,
                    &self.linear_kernel,
                    raw_noise,
                    smoothing,
                    smooth_noise,
                    rows,
                    false,
                    0,
                )?;
                let dim = magnitude_head.out_dim;
                unsafe {
                    // `out_row_offset`, not 0 -- this is the step that lands the
                    // chunk in `dest`.
                    self.stream
                        .launch_builder(apply_kernel)
                        .arg(&*mag_pre)
                        .arg(&*coupled)
                        .arg(&*sparsity)
                        .arg(&*smooth_noise)
                        .arg(dest)
                        .arg(&(rows as i32))
                        .arg(&(dim as i32))
                        .arg(&*logit_clamp)
                        .arg(&*magnitude_floor)
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
            DeviceArch::StructuredGate(g) => {
                // The RAW, pre-smoothing noise, which is exactly the
                // `gate_noise` argument `StructuredGate::forward_cpu` takes --
                // it applies the smoothing map itself. Returning `smooth_noise`
                // would make the CPU reference smooth an already-smoothed
                // vector, and `sift_gpu_forward_matches_the_cpu_reference` would
                // fail for a reason that is not a kernel defect.
                let latent_dim = g.trunk[0].in_dim;
                let dim = g.magnitude_head.out_dim;
                Ok((
                    self.stream
                        .memcpy_dtov(&g.latents.slice(0..rows * latent_dim))?,
                    Some(
                        self.stream
                            .memcpy_dtov(&g.raw_noise.slice(0..rows * dim))?,
                    ),
                ))
            }
        }
    }
}
