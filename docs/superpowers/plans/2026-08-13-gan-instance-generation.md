# GAN Instance Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the vector_search Gaussian-mixture instance generator with the trained v1 WGAN-GP generator, producing instances that resemble real SIFT embedding data.

**Architecture:** A 7.09 MB f32 weight blob is embedded in `tig-challenges` via `include_bytes!` and uploaded to the device each `generate_instance`. Latent vectors are drawn per-index with `curand` (elementwise, therefore deterministic), then pushed through four hand-written GEMM+LeakyReLU kernels whose K-loop accumulation order is fixed. cuBLAS is never used, because its kernel selection is heuristic and varies by architecture.

The forward pass is **chunked**, and this is a hard requirement rather than an optimisation. At the largest track (`n_queries=15000`, 1,515,000 vectors) a single 1024-wide activation buffer is 6.21 GB and two are needed at once, against 8.59 GB of device memory — a whole-batch pass cannot fit. Processing `FORWARD_CHUNK = 65536` rows at a time caps activations at 537 MB, and because each chunk writes its final layer straight into the destination buffer at a row offset, no device-to-device slicing is needed either.

**Tech Stack:** Rust (edition 2021, `nightly-2025-02-10`), CUDA 12.6, `cudarc` (tig-foundation fork), PyTorch (for weight export and golden-vector generation only).

## Global Constraints

- Build `tig-runtime`/`tig-verifier` with `cargo +nightly-2025-02-10`; a different toolchain makes working algorithms fail spuriously.
- Build against CUDA 12.6 (`CUDA_PATH=/usr/local/cuda-12.6`); cudarc rejects CUDA 13.0.
- The algorithm `.so` needs `libstd-399a850a7013db0c.so` on `LD_LIBRARY_PATH` — that is `$(rustc +nightly-2025-02-10 --print target-libdir)`.
- All GPU work runs on the `tig-gpu` host (vast.ai, single RTX 4060, sm_89). Repo copy at `/workspace/tig-bench`.
- **Never `git push`.** Local commits only until the user confirms the work is verified.
- The forward pass must use only `fmaf`, `mul`, `add` and comparison/select. No division, no `sqrt`, no transcendentals — `build_ptx` compiles with `--use_fast_math`, which makes those approximate and architecture-dependent.
- Never call cuBLAS or any library that selects kernels by runtime heuristic.
- `QUALITY_PRECISION` is `1_000_000` (`tig-challenges/src/lib.rs:3`).
- Generator: latent 128 → 512 → 1024 → 1024 → output 128, LeakyReLU(0.2) after the first three layers only.
- Source checkpoint: `/workspace/annbench-root/runs/x100k_ema_only/best_generator.pt`, key `generator_state_dict` (v1 has no `ema_params`).

---

## File Structure

| File | Responsibility |
|---|---|
| `tig-challenges/src/vector_search/weights/v1_sift.bin` | **Create.** The f32 weight blob, ~7.09 MB, committed. |
| `scripts/export_generator_weights.py` | **Create.** PyTorch checkpoint → blob. Records provenance. |
| `tig-challenges/src/vector_search/generator.rs` | **Create.** Blob parsing, layer metadata, the `Generator` trait boundary that lets v4 drop in later. |
| `tig-challenges/src/vector_search/kernels.cu` | **Modify.** Replace `generate_clusters`/`generate_vectors` with `gan_sample_latents` and `gan_linear`. Keep `evaluate_total_distance`. |
| `tig-challenges/src/vector_search/mod.rs` | **Modify.** `generate_instance` drives the forward pass; `vector_dims` 250→128; recalibrated quality constants. |
| `scripts/calibrate_vector_search.py` | **Create.** Fits the quality constants across all five active tracks. |

---

### Task 1: Weight blob format and exporter

Produces the committed artifact every later task depends on. The blob is self-describing so a second generator can be added without changing the parser.

**Files:**
- Create: `scripts/export_generator_weights.py`
- Create: `tig-challenges/src/vector_search/weights/v1_sift.bin`

**Interfaces:**
- Consumes: nothing.
- Produces: a blob with layout — magic `b"TIGGAN01"` (8 bytes), `u32` layer count, then per layer `u32 in_dim`, `u32 out_dim`, `f32[out_dim * in_dim]` weights row-major (`w[out][in]`, matching PyTorch `nn.Linear.weight`), `f32[out_dim]` bias. All little-endian.

- [ ] **Step 1: Write the exporter**

Create `scripts/export_generator_weights.py`:

```python
#!/usr/bin/env python3
"""Export a trained generator's weights to the flat blob tig-challenges embeds.

Every verifier must load byte-identical weights, so the blob is committed to git
rather than downloaded. The format is self-describing (per-layer dims in the
header) so a second generator architecture can be added without changing the
Rust parser.

Row-major `w[out_dim][in_dim]` matches PyTorch's `nn.Linear.weight` exactly, so
the CUDA kernel can index `weight + col * in_dim` with no transpose.
"""

import argparse
import hashlib
import struct
import sys
from pathlib import Path

import torch

MAGIC = b"TIGGAN01"


def export(checkpoint: Path, state_key: str, out: Path) -> None:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state_key not in ckpt:
        raise SystemExit(
            f"{checkpoint} has no key {state_key!r}; found {sorted(ckpt)}"
        )
    state = ckpt[state_key]

    # nn.Sequential names them net.0, net.2, net.4, net.6 (odd indices are the
    # activations, which have no parameters). Sorting numerically keeps layer
    # order correct beyond net.9.
    indices = sorted(
        {int(k.split(".")[1]) for k in state if k.startswith("net.")}
    )
    if not indices:
        raise SystemExit(f"no 'net.N.*' parameters in {state_key}")

    blob = bytearray(MAGIC)
    blob += struct.pack("<I", len(indices))
    for i in indices:
        w = state[f"net.{i}.weight"]
        b = state[f"net.{i}.bias"]
        out_dim, in_dim = w.shape
        if b.shape[0] != out_dim:
            raise SystemExit(f"layer {i}: bias {b.shape} does not match {w.shape}")
        blob += struct.pack("<II", in_dim, out_dim)
        blob += w.to(torch.float32).contiguous().numpy().tobytes()
        blob += b.to(torch.float32).contiguous().numpy().tobytes()

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(blob)

    print(f"checkpoint : {checkpoint}")
    print(f"state key  : {state_key}  (step {ckpt.get('step')})")
    print(f"layers     : {[(state[f'net.{i}.weight'].shape[1], state[f'net.{i}.weight'].shape[0]) for i in indices]}")
    print(f"bytes      : {len(blob)}")
    print(f"sha256     : {hashlib.sha256(blob).hexdigest()}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--state-key", default="generator_state_dict")
    args = p.parse_args()
    export(args.checkpoint, args.state_key, args.output)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the exporter on tig-gpu**

```bash
scp scripts/export_generator_weights.py tig-gpu:/workspace/tig-bench/scripts/
ssh tig-gpu 'source /venv/main/bin/activate && cd /workspace/tig-bench && \
  python scripts/export_generator_weights.py \
    /workspace/annbench-root/runs/x100k_ema_only/best_generator.pt \
    tig-challenges/src/vector_search/weights/v1_sift.bin'
```

Expected: `layers : [(128, 512), (512, 1024), (1024, 1024), (1024, 128)]` and `bytes : 7088652`.

That byte count is `8 + 4 + 4*8 + 4*1772160` — header, layer count, four `(in,out)` pairs, and 1,772,160 f32 parameters.

- [ ] **Step 3: Copy the blob back and record its hash**

```bash
scp tig-gpu:/workspace/tig-bench/tig-challenges/src/vector_search/weights/v1_sift.bin \
    tig-challenges/src/vector_search/weights/v1_sift.bin
sha256sum tig-challenges/src/vector_search/weights/v1_sift.bin
```

Paste that hash into the commit message — it is the provenance record tying the blob to the checkpoint.

- [ ] **Step 4: Commit the blob**

`.gitignore` excludes `*.pt`/`*.npy`-style dumps but deliberately not `*.bin`, so this needs no `-f`. If a future change starts ignoring `*.bin`, add a negation for this path rather than force-adding it.

```bash
git add tig-challenges/src/vector_search/weights/v1_sift.bin
git add scripts/export_generator_weights.py
git commit -m "Add v1 generator weight blob and its exporter

Committed rather than downloaded because every verifier must load
byte-identical weights. sha256: <paste from step 3>"
```

---

### Task 2: Deterministic GEMM kernel and the performance decision

The spec leaves open whether a fixed-order kernel is fast enough. This task answers that before anything is built on top, so a "distill the generator" decision is still cheap.

**Files:**
- Modify: `tig-challenges/src/vector_search/kernels.cu`

**Interfaces:**
- Consumes: nothing.
- Produces: `gan_linear(const float* input, const float* weight, const float* bias, float* output, int n, int in_dim, int out_dim, int apply_activation, int out_row_offset)` — one CUDA block per input row, one output column per thread, `in_dim` floats of dynamic shared memory required. `out_row_offset` lets a chunk's final layer write directly into the destination buffer.

- [ ] **Step 1: Add the kernel**

Append to `tig-challenges/src/vector_search/kernels.cu`:

```cuda
// Fixed-order dense layer with optional LeakyReLU(0.2).
//
// Determinism is the whole point. Each thread owns one complete output element
// and accumulates over k sequentially, so the summation order does not depend
// on grid shape, block size, or how the scheduler interleaves work. That is the
// property cuBLAS cannot offer: it picks kernels by heuristic, and a different
// architecture picks a different reduction order and a different last bit.
//
// Only fmaf/mul/add/select appear here. build_ptx compiles with --use_fast_math,
// which makes division, sqrt and transcendentals approximate and potentially
// architecture-dependent; these operations are all IEEE-754 exactly rounded, so
// identical PTX yields identical results on any conforming GPU.
//
// The input row is staged in shared memory because every thread in the block
// reads all of it. Weights are left in global memory: they total under 5 MB per
// layer and stay resident in L2 across the whole launch.
//
// `out_row_offset` shifts writes within the destination buffer, so the caller
// can process a chunk of rows and land the last layer's output directly in its
// final position. `n` is the chunk's row count, not the instance's.
extern "C" __global__ void gan_linear(
    const float *__restrict__ input,
    const float *__restrict__ weight,
    const float *__restrict__ bias,
    float *__restrict__ output,
    const int n,
    const int in_dim,
    const int out_dim,
    const int apply_activation,
    const int out_row_offset
)
{
    extern __shared__ float s_input[];

    for (int row = blockIdx.x; row < n; row += gridDim.x)
    {
        const float *x = input + (long long)row * in_dim;
        for (int i = threadIdx.x; i < in_dim; i += blockDim.x) {
            s_input[i] = x[i];
        }
        __syncthreads();

        const long long out_row = (long long)(out_row_offset + row);
        for (int col = threadIdx.x; col < out_dim; col += blockDim.x)
        {
            const float *w = weight + (long long)col * in_dim;
            float acc = bias[col];
            for (int k = 0; k < in_dim; ++k) {
                acc = fmaf(s_input[k], w[k], acc);
            }
            if (apply_activation) {
                acc = (acc >= 0.0f) ? acc : (acc * 0.2f);
            }
            output[out_row * out_dim + col] = acc;
        }
        // Guard the next iteration's overwrite of s_input against threads still
        // reading it in the loop above.
        __syncthreads();
    }
}
```

- [ ] **Step 2: Write a standalone timing harness**

Create `/workspace/tig-bench/bench_gan_linear.cu` on tig-gpu (scratch only, not committed). It must chunk, for the same reason the real implementation does — a whole-batch pass needs 12.41 GB of activations against 8.59 GB of device memory:

```cuda
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Paste the gan_linear kernel above this line when compiling standalone.
extern "C" __global__ void gan_linear(const float*, const float*, const float*,
                                      float*, int, int, int, int, int);

static void *alloc(size_t bytes) { void *p; cudaMalloc(&p, bytes); cudaMemset(p, 0, bytes); return p; }

int main(int argc, char **argv)
{
    const int n = (argc > 1) ? atoi(argv[1]) : 1515000;
    const int chunk = (argc > 2) ? atoi(argv[2]) : 65536;
    const int dims[5] = {128, 512, 1024, 1024, 128};

    // Two chunk-sized scratch buffers at the widest layer, plus the real output.
    float *buf[2];
    buf[0] = (float *)alloc((size_t)chunk * 1024 * sizeof(float));
    buf[1] = (float *)alloc((size_t)chunk * 1024 * sizeof(float));
    float *out = (float *)alloc((size_t)n * dims[4] * sizeof(float));

    float *w[4], *b[4];
    for (int l = 0; l < 4; ++l) {
        w[l] = (float *)alloc((size_t)dims[l] * dims[l + 1] * sizeof(float));
        b[l] = (float *)alloc((size_t)dims[l + 1] * sizeof(float));
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    for (int warm = 0; warm < 2; ++warm) {
        if (warm) cudaEventRecord(start);
        for (int base = 0; base < n; base += chunk) {
            const int rows = (n - base < chunk) ? (n - base) : chunk;
            for (int l = 0; l < 4; ++l) {
                const int block = 256;
                const int grid = (rows < 65535) ? rows : 65535;
                const int last = (l == 3);
                gan_linear<<<grid, block, dims[l] * sizeof(float)>>>(
                    buf[l % 2], w[l], b[l],
                    last ? out : buf[(l + 1) % 2],
                    rows, dims[l], dims[l + 1],
                    last ? 0 : 1,
                    last ? base : 0);
            }
        }
        if (warm) { cudaEventRecord(stop); cudaEventSynchronize(stop); }
        else cudaDeviceSynchronize();
    }

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("n=%d chunk=%d  forward=%.1f ms\n", n, chunk, ms);
    if (cudaGetLastError() != cudaSuccess) { printf("CUDA ERROR\n"); return 1; }
    return 0;
}
```

- [ ] **Step 3: Run it at every active track size**

```bash
ssh tig-gpu 'cd /workspace/tig-bench && \
  /usr/local/cuda-12.6/bin/nvcc -O3 -arch=sm_89 --use_fast_math \
    bench_gan_linear.cu -o bench_gan_linear && \
  for n in 707000 909000 1111000 1313000 1515000; do ./bench_gan_linear $n; done'
```

Reference points to compare against:
- Current Gaussian generator: ~120 ms at 707k, ~250 ms at 1515k.
- cuBLAS floor for this network: 462 ms at 707k, 1045 ms at 1515k.

- [ ] **Step 4: Sweep the chunk size**

Chunk size trades throughput against memory. Peak activation cost is `2 * chunk * 1024 * 4` bytes, so 65536 costs 537 MB and 262144 costs 2.1 GB — and the device also holds the output buffer (776 MB at the largest track) and must leave room for the solving algorithm afterwards.

```bash
ssh tig-gpu 'cd /workspace/tig-bench && \
  for c in 16384 65536 262144; do ./bench_gan_linear 1515000 $c; done'
```

Report the timings alongside each option's memory cost. Do not pick a chunk size that leaves under ~2 GB free at the largest track.

- [ ] **Step 5: Decision gate — report and stop**

Report the measured numbers against those two reference points and **ask the user before continuing**. This is the point at which distilling the generator is still cheap; after Task 3 it is not.

- Within ~1.5× of the cuBLAS floor: proceed as planned.
- Materially worse: propose either a tiled variant that also stages weights in shared memory (still fixed-order, so still deterministic), or distilling v1 to fewer than 1,769,472 MACs per vector.

Note that the cuBLAS floor was itself measured on 65536-row chunks, so it is a fair comparison.

- [ ] **Step 6: Commit the kernel**

```bash
git add tig-challenges/src/vector_search/kernels.cu
git commit -m "Add fixed-order dense layer kernel for GAN instance generation

Each thread owns one complete output element and accumulates over k
sequentially, so summation order is independent of launch geometry.
Uses only exactly-rounded operations, since build_ptx compiles with
--use_fast_math.

Measured forward pass: <paste step 3 numbers>"
```

---

### Task 3: Weight blob parser

Pure Rust, no GPU, so this is unit-testable with `cargo test` on any machine.

**Deliberate divergence from the spec.** The spec calls for "a small trait — weights blob, layer dims, forward pass" so v4 can drop in later. This plan delivers that outcome without the trait: the blob format carries per-layer dimensions, and `generate_instance` derives `vector_dims` and the layer loop from the parsed weights, so any MLP-shaped generator works with no code change. A trait with a single implementor would be abstraction ahead of a second case, and v4 is not MLP-shaped anyway — it needs Conv3d, gate sampling and normalization, so the right boundary for it cannot be designed from v1 alone. If a reviewer wants the trait, add it when v4 arrives and its real requirements are visible.

**Files:**
- Create: `tig-challenges/src/vector_search/generator.rs`
- Modify: `tig-challenges/src/vector_search/mod.rs` (add `mod generator;`)

**Interfaces:**
- Consumes: the blob format from Task 1.
- Produces:
  - `pub struct Layer { pub in_dim: usize, pub out_dim: usize, pub weights: Vec<f32>, pub bias: Vec<f32> }`
  - `pub struct GeneratorWeights { pub layers: Vec<Layer> }`
  - `pub fn parse_weights(blob: &[u8]) -> Result<GeneratorWeights>`
  - `pub fn v1_weights() -> Result<GeneratorWeights>` — parses the embedded blob
  - `pub const LATENT_DIM: usize = 128;`

- [ ] **Step 1: Write the failing tests**

Create `tig-challenges/src/vector_search/generator.rs` with tests first:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn blob_with(layers: &[(u32, u32)]) -> Vec<u8> {
        let mut v = Vec::from(*b"TIGGAN01");
        v.extend_from_slice(&(layers.len() as u32).to_le_bytes());
        for &(in_dim, out_dim) in layers {
            v.extend_from_slice(&in_dim.to_le_bytes());
            v.extend_from_slice(&out_dim.to_le_bytes());
            for _ in 0..(in_dim * out_dim + out_dim) {
                v.extend_from_slice(&1.5f32.to_le_bytes());
            }
        }
        v
    }

    #[test]
    fn parses_layer_dims_and_values() {
        let g = parse_weights(&blob_with(&[(2, 3)])).unwrap();
        assert_eq!(g.layers.len(), 1);
        assert_eq!(g.layers[0].in_dim, 2);
        assert_eq!(g.layers[0].out_dim, 3);
        assert_eq!(g.layers[0].weights.len(), 6);
        assert_eq!(g.layers[0].bias.len(), 3);
        assert_eq!(g.layers[0].weights[0], 1.5);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut b = blob_with(&[(2, 3)]);
        b[0] = b'X';
        assert!(parse_weights(&b).is_err());
    }

    #[test]
    fn rejects_truncated_blob() {
        let b = blob_with(&[(2, 3)]);
        assert!(parse_weights(&b[..b.len() - 4]).is_err());
    }

    #[test]
    fn rejects_trailing_bytes() {
        let mut b = blob_with(&[(2, 3)]);
        b.push(0);
        assert!(parse_weights(&b).is_err());
    }

    #[test]
    fn embedded_v1_blob_has_expected_shape() {
        let g = v1_weights().unwrap();
        let dims: Vec<(usize, usize)> =
            g.layers.iter().map(|l| (l.in_dim, l.out_dim)).collect();
        assert_eq!(dims, vec![(128, 512), (512, 1024), (1024, 1024), (1024, 128)]);
        assert_eq!(g.layers[0].in_dim, LATENT_DIM);
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p tig-challenges --features vector_search generator`
Expected: FAIL — `parse_weights` and `v1_weights` are not defined.

- [ ] **Step 3: Write the implementation**

Prepend to `tig-challenges/src/vector_search/generator.rs`:

```rust
use anyhow::{anyhow, Result};

/// Latent dimensionality the generator samples from.
pub const LATENT_DIM: usize = 128;

const MAGIC: &[u8; 8] = b"TIGGAN01";

/// Weights are committed rather than fetched: every verifier regenerates the
/// instance independently and compares a fixed-point quality integer exactly,
/// so a single differing byte would fail verification network-wide.
const V1_BLOB: &[u8] = include_bytes!("weights/v1_sift.bin");

pub struct Layer {
    pub in_dim: usize,
    pub out_dim: usize,
    /// Row-major `[out_dim][in_dim]`, matching PyTorch `nn.Linear.weight`, so
    /// the kernel indexes `weight + col * in_dim` without a transpose.
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
}

pub struct GeneratorWeights {
    pub layers: Vec<Layer>,
}

fn take<'a>(blob: &'a [u8], at: &mut usize, len: usize) -> Result<&'a [u8]> {
    let end = at
        .checked_add(len)
        .ok_or_else(|| anyhow!("weight blob length overflow"))?;
    if end > blob.len() {
        return Err(anyhow!(
            "weight blob truncated: needed {} bytes at offset {}, have {}",
            len,
            at,
            blob.len()
        ));
    }
    let out = &blob[*at..end];
    *at = end;
    Ok(out)
}

fn read_u32(blob: &[u8], at: &mut usize) -> Result<u32> {
    let b = take(blob, at, 4)?;
    Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn read_f32s(blob: &[u8], at: &mut usize, count: usize) -> Result<Vec<f32>> {
    let b = take(blob, at, count * 4)?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

pub fn parse_weights(blob: &[u8]) -> Result<GeneratorWeights> {
    let mut at = 0usize;
    if take(blob, &mut at, 8)? != MAGIC {
        return Err(anyhow!("weight blob has wrong magic; expected TIGGAN01"));
    }
    let num_layers = read_u32(blob, &mut at)? as usize;
    if num_layers == 0 {
        return Err(anyhow!("weight blob declares zero layers"));
    }

    let mut layers = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        let in_dim = read_u32(blob, &mut at)? as usize;
        let out_dim = read_u32(blob, &mut at)? as usize;
        if in_dim == 0 || out_dim == 0 {
            return Err(anyhow!("layer {} has a zero dimension", i));
        }
        let weights = read_f32s(blob, &mut at, in_dim * out_dim)?;
        let bias = read_f32s(blob, &mut at, out_dim)?;
        layers.push(Layer { in_dim, out_dim, weights, bias });
    }

    // Trailing bytes mean the blob and this parser disagree about the format,
    // which is worth failing loudly rather than silently ignoring.
    if at != blob.len() {
        return Err(anyhow!(
            "weight blob has {} trailing bytes",
            blob.len() - at
        ));
    }
    Ok(GeneratorWeights { layers })
}

pub fn v1_weights() -> Result<GeneratorWeights> {
    parse_weights(V1_BLOB)
}
```

Add to `tig-challenges/src/vector_search/mod.rs`, immediately after the existing `use` statements:

```rust
mod generator;
use generator::{v1_weights, LATENT_DIM};
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p tig-challenges --features vector_search generator`
Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add tig-challenges/src/vector_search/generator.rs tig-challenges/src/vector_search/mod.rs
git commit -m "Add weight blob parser for GAN instance generation

Self-describing format so a second generator architecture can be added
without changing the parser. Rejects bad magic, truncation and trailing
bytes rather than silently mis-parsing."
```

---

### Task 4: Latent sampling kernel and the forward pass

**Files:**
- Modify: `tig-challenges/src/vector_search/kernels.cu`
- Modify: `tig-challenges/src/vector_search/mod.rs:42-139` (`generate_instance`)

**Interfaces:**
- Consumes: `gan_linear` (Task 2), `v1_weights()`/`LATENT_DIM` (Task 3).
- Produces: `Challenge { seed, num_queries, vector_dims: 128, database_size, d_database_vectors, d_query_vectors }` — the struct fields are unchanged, only how they are filled.

- [ ] **Step 1: Add the latent kernel**

Append to `tig-challenges/src/vector_search/kernels.cu`:

```cuda
// Draw one latent vector per output vector.
//
// Seeded per index exactly as the previous generator was, so each vector's
// latent depends only on its own global index. No inter-thread coordination
// means no ordering to get wrong, and the draw is reproducible on any launch
// geometry.
//
// `index_offset` is the chunk's first global index. It enters curand_init, so a
// vector's latent is a function of where it sits in the instance, never of how
// the work was chunked. Database vectors occupy global indices
// [0, database_size) and queries [database_size, total), mirroring the previous
// generator's convention.
extern "C" __global__ void gan_sample_latents(
    const uint8_t *seed,
    const int n,
    const int latent_dim,
    float *latents,
    const int index_offset
)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n;
         i += blockDim.x * gridDim.x)
    {
        const int global_i = index_offset + i;
        curandState state;
        curand_init(((uint64_t *)(seed))[global_i % 4], global_i, 0, &state);
        float *row = latents + (long long)i * latent_dim;
        for (int j = 0; j < latent_dim; ++j) {
            row[j] = curand_normal(&state);
        }
    }
}
```

- [ ] **Step 2: Delete the obsolete kernels**

Remove `generate_clusters`, `binary_search` and `truncated_normal` from `kernels.cu`. Keep `evaluate_total_distance` unchanged. `generate_vectors` goes too — nothing references it after this task.

- [ ] **Step 3: Rewrite `generate_instance`**

Replace the body of `generate_instance` in `tig-challenges/src/vector_search/mod.rs` (currently lines 42-139) with:

```rust
    /// Rows processed per forward-pass chunk.
    ///
    /// Not a tuning knob — a requirement. At the largest active track
    /// (n_queries=15000, 1,515,000 vectors) one 1024-wide activation buffer is
    /// 6.21 GB and the pass needs two live at once, against 8.59 GB of device
    /// memory on the reference GPU. Chunking caps activations at 537 MB.
    const FORWARD_CHUNK: usize = 65536;

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
        let widest = layers.iter().map(|l| l.out_dim).max().unwrap();
        let database_size = 100 * track.n_queries;

        let sample_latents_kernel = module.load_function("gan_sample_latents")?;
        let linear_kernel = module.load_function("gan_linear")?;

        let d_seed = stream.memcpy_stod(seed)?;

        // Weights are uploaded once and reused by every chunk.
        let mut d_weights = Vec::with_capacity(layers.len());
        for layer in layers {
            d_weights.push((
                stream.memcpy_stod(&layer.weights)?,
                stream.memcpy_stod(&layer.bias)?,
            ));
        }

        // Two scratch buffers sized for the widest layer, ping-ponged between
        // layers, plus one latent buffer. Reused across chunks.
        let mut d_scratch_a = stream.alloc_zeros::<f32>(FORWARD_CHUNK * widest)?;
        let mut d_scratch_b = stream.alloc_zeros::<f32>(FORWARD_CHUNK * widest)?;
        let mut d_latents = stream.alloc_zeros::<f32>(FORWARD_CHUNK * LATENT_DIM)?;

        let mut d_database_vectors =
            stream.alloc_zeros::<f32>(database_size as usize * vector_dims)?;
        let mut d_query_vectors =
            stream.alloc_zeros::<f32>(track.n_queries as usize * vector_dims)?;

        let block_size = 256u32;

        // Database vectors take global indices [0, database_size), queries take
        // [database_size, total). The global index seeds the latent draw, so a
        // vector's value depends on its position in the instance and not on the
        // chunk boundaries.
        for (dest_is_query, count) in
            [(false, database_size as usize), (true, track.n_queries as usize)]
        {
            let index_base = if dest_is_query { database_size as usize } else { 0 };

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
                            grid_dim: (
                                (rows as u32 + block_size - 1) / block_size,
                                1,
                                1,
                            ),
                            block_dim: (block_size, 1, 1),
                            shared_mem_bytes: 0,
                        })?;
                }

                // One block per row, capped at the launch limit; the kernel
                // strides over rows when there are more rows than blocks.
                let grid = (rows as u32).min(65535);

                for (i, layer) in layers.iter().enumerate() {
                    let is_last = i + 1 == layers.len();
                    let (d_w, d_b) = &d_weights[i];

                    // Layer 0 reads the latents; later layers read whichever
                    // scratch buffer the previous layer wrote.
                    let input: &CudaSlice<f32> = if i == 0 {
                        &d_latents
                    } else if i % 2 == 1 {
                        &d_scratch_a
                    } else {
                        &d_scratch_b
                    };

                    let mut builder = stream.launch_builder(&linear_kernel);
                    builder.arg(input).arg(d_w).arg(d_b);

                    // The last layer writes straight into its final position,
                    // which is why no device-to-device copy is needed.
                    if is_last {
                        let dest = if dest_is_query {
                            &mut d_query_vectors
                        } else {
                            &mut d_database_vectors
                        };
                        builder.arg(dest);
                    } else if i % 2 == 0 {
                        builder.arg(&mut d_scratch_a);
                    } else {
                        builder.arg(&mut d_scratch_b);
                    }

                    let apply_activation = (!is_last) as i32;
                    let out_row_offset =
                        if is_last { chunk_start as i32 } else { 0 };

                    unsafe {
                        builder
                            .arg(&(rows as i32))
                            .arg(&(layer.in_dim as i32))
                            .arg(&(layer.out_dim as i32))
                            .arg(&apply_activation)
                            .arg(&out_row_offset)
                            .launch(LaunchConfig {
                                grid_dim: (grid, 1, 1),
                                block_dim: (block_size, 1, 1),
                                shared_mem_bytes: (layer.in_dim * 4) as u32,
                            })?;
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
```

Two notes for the implementer:

`vector_dims` is derived from the final layer rather than hardcoded, so a differently-shaped generator needs no change here.

The borrow checker may object to how `builder` is built up across the `if is_last` branches, since `d_scratch_a`/`d_scratch_b` are borrowed mutably while `input` holds an immutable borrow of one of them. If it does, restructure so each layer's launch happens inside its own branch with the full argument chain rather than sharing a `builder` binding — the ping-pong means input and output are never the same buffer, but that fact is not visible to the compiler. Add `use cudarc::driver::safe::CudaSlice;` to the imports for the `input` type annotation.

- [ ] **Step 4: Build and run one nonce end to end**

The two production algorithms cannot run at 128 dims, so validate against the challenge alone — a solution of all-zero indexes is structurally valid and exercises generation plus evaluation:

```bash
ssh tig-gpu 'export PATH=/usr/local/cuda-12.6/bin:$HOME/.cargo/bin:$PATH; \
  export CUDA_PATH=/usr/local/cuda-12.6 CUDA_ROOT=/usr/local/cuda-12.6 \
         CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.6; \
  export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH; \
  cd /workspace/tig-bench && \
  cargo +nightly-2025-02-10 build --release -p tig-runtime -p tig-verifier \
    --features vector_search 2>&1 | tail -3'
```

Expected: builds clean.

- [ ] **Step 5: Commit**

```bash
git add tig-challenges/src/vector_search/kernels.cu tig-challenges/src/vector_search/mod.rs
git commit -m "Generate vector_search instances from the v1 GAN

Replaces the Gaussian mixture with per-index latent sampling and four
fixed-order dense layers. vector_dims is now derived from the final
layer (128) rather than hardcoded, so a differently-shaped generator
needs no change here."
```

---

### Task 5: Golden-vector and determinism tests

The two properties verification depends on: the port matches PyTorch, and output does not vary with launch geometry.

**Files:**
- Create: `scripts/dump_golden_vectors.py`
- Modify: `tig-challenges/src/vector_search/generator.rs` (add a CPU reference forward pass and tests)

**Interfaces:**
- Consumes: `parse_weights`, `Layer` (Task 3).
- Produces: `pub fn forward_cpu(weights: &GeneratorWeights, latent: &[f32]) -> Vec<f32>` — reference implementation, same accumulation order as the kernel.

- [ ] **Step 1: Write the golden-vector dumper**

Create `scripts/dump_golden_vectors.py`:

```python
#!/usr/bin/env python3
"""Dump PyTorch generator outputs for fixed latents, as a port-correctness oracle.

Latents are a deterministic ramp rather than random draws: the point is to pin
the arithmetic of the forward pass, and a fixed pattern is reproducible without
agreeing on an RNG between Python and Rust.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/workspace/checkouts/wgan-synthetic")
from src.models.generator import Generator


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--count", type=int, default=4)
    args = p.parse_args()

    model = Generator(latent_dim=128, output_dim=128,
                      hidden_dims=[512, 1024, 1024], negative_slope=0.2)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["generator_state_dict"])
    model.eval()

    # Values in [-1, 1); distinct per row and per column.
    latents = torch.stack([
        torch.arange(128, dtype=torch.float32) / 64.0 - 1.0 + i * 0.01
        for i in range(args.count)
    ])

    with torch.no_grad():
        out = model(latents)

    args.output.write_text(json.dumps({
        "latents": latents.tolist(),
        "outputs": out.tolist(),
    }, indent=2))
    print(f"wrote {args.count} golden vectors to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate the golden file**

```bash
scp scripts/dump_golden_vectors.py tig-gpu:/workspace/tig-bench/scripts/
ssh tig-gpu 'source /venv/main/bin/activate && cd /workspace/tig-bench && \
  python scripts/dump_golden_vectors.py \
    /workspace/annbench-root/runs/x100k_ema_only/best_generator.pt \
    tig-challenges/src/vector_search/weights/golden_vectors.json'
scp tig-gpu:/workspace/tig-bench/tig-challenges/src/vector_search/weights/golden_vectors.json \
    tig-challenges/src/vector_search/weights/golden_vectors.json
```

- [ ] **Step 3: Write the failing test**

Add to the `tests` module in `tig-challenges/src/vector_search/generator.rs`:

```rust
    #[test]
    fn cpu_forward_matches_pytorch_golden_vectors() {
        #[derive(serde::Deserialize)]
        struct Golden {
            latents: Vec<Vec<f32>>,
            outputs: Vec<Vec<f32>>,
        }
        let golden: Golden = serde_json::from_str(include_str!(
            "weights/golden_vectors.json"
        ))
        .unwrap();

        let weights = v1_weights().unwrap();
        for (latent, expected) in golden.latents.iter().zip(&golden.outputs) {
            let got = forward_cpu(&weights, latent);
            assert_eq!(got.len(), expected.len());
            for (i, (g, e)) in got.iter().zip(expected).enumerate() {
                // PyTorch sums a dot product in a different order than our
                // fixed sequential loop, so exact equality is not expected;
                // this bound catches a wrong transpose, a missing bias or a
                // misapplied activation, which is what the test is for.
                assert!(
                    (g - e).abs() < 1e-4,
                    "coordinate {i}: got {g}, expected {e}"
                );
            }
        }
    }
```

- [ ] **Step 4: Run it to verify it fails**

Run: `cargo test -p tig-challenges --features vector_search cpu_forward`
Expected: FAIL — `forward_cpu` is not defined.

- [ ] **Step 5: Implement the reference forward pass**

Add to `tig-challenges/src/vector_search/generator.rs`:

```rust
/// Reference forward pass, matching `gan_linear`'s accumulation order exactly.
///
/// Exists to pin the port against PyTorch in tests. `generate_instance` never
/// calls it — instances are always produced on the GPU.
pub fn forward_cpu(weights: &GeneratorWeights, latent: &[f32]) -> Vec<f32> {
    let mut current = latent.to_vec();
    for (i, layer) in weights.layers.iter().enumerate() {
        let apply_activation = i + 1 < weights.layers.len();
        let mut next = Vec::with_capacity(layer.out_dim);
        for col in 0..layer.out_dim {
            let w = &layer.weights[col * layer.in_dim..(col + 1) * layer.in_dim];
            let mut acc = layer.bias[col];
            for k in 0..layer.in_dim {
                acc = current[k].mul_add(w[k], acc);
            }
            next.push(if apply_activation && acc < 0.0 { acc * 0.2 } else { acc });
        }
        current = next;
    }
    current
}
```

- [ ] **Step 6: Run it to verify it passes**

Run: `cargo test -p tig-challenges --features vector_search cpu_forward`
Expected: PASS.

- [ ] **Step 7: Verify GPU determinism across launch geometries**

The CPU test proves the port is right; this proves the kernel does not vary with grid shape. On tig-gpu, temporarily change `block_size` in `generate_instance` from `256` to `128`, regenerate the same nonce, and confirm the instance is byte-identical:

```bash
ssh tig-gpu 'cd /workspace/tig-bench && \
  ./target/release/tig-verifier "{\"algorithm_id\":\"\",\"challenge_id\":\"c004\",\"track_id\":\"n_queries=7000\",\"block_id\":\"\",\"player_id\":\"\"}" \
    rand_hash 0 /tmp/sol.json --ptx <ptx> --gpu 0 --verbose'
```

Run once at `block_size = 256` and once at `128`; the reported quality must be identical. Restore `block_size = 256` afterwards.

- [ ] **Step 8: Commit**

```bash
git add tig-challenges/src/vector_search/weights/golden_vectors.json
git add scripts/dump_golden_vectors.py tig-challenges/src/vector_search/generator.rs
git commit -m "Pin the GAN port against PyTorch and check launch-geometry invariance

Golden vectors catch a wrong transpose, missing bias or misapplied
activation. The block-size comparison checks the property verification
actually depends on: identical output regardless of launch geometry."
```

---

### Task 6: Quality recalibration

The hardcoded baseline of 11.0 is wrong for the GAN's distance scale — measured optimal `avg_dist` is ~1.14 against ~10.18 for the Gaussian generator. Left alone, every solution scores ~0.89 and the competition collapses.

**Files:**
- Create: `scripts/calibrate_vector_search.py`
- Modify: `tig-challenges/src/vector_search/mod.rs:201-215` (`evaluate_solution`)

**Interfaces:**
- Consumes: a working `generate_instance` (Task 4).
- Produces: `QUALITY_OFFSET: f64` and `QUALITY_SCALE: f64` constants.

- [ ] **Step 1: Write the calibration script**

Create `scripts/calibrate_vector_search.py`:

```python
#!/usr/bin/env python3
"""Fit the quality constants so GAN instances land in today's quality band.

Naively matching the optimal-vs-random spread is the wrong target. Live mainnet
data shows competing algorithms sit within ~80 quality units of each other out
of ~72,000, all within ~1% of exact 1-NN, and that quality drifts ~6,000 units
across the active track range. What must be preserved is therefore the
achievable band at each track, which is what this fits.

quality = (offset - avg_dist) / scale

Two unknowns against five tracks, so it is a least-squares fit; the residuals
say whether fixed constants suffice or whether they must vary with n_queries.
"""

import argparse
import json
from pathlib import Path

import numpy as np

QUALITY_PRECISION = 1_000_000

# Median qualifier quality per track on mainnet, block 1298164 (44-53 qualifiers
# each). These are what GAN instances must reproduce. Medians rather than maxima:
# the target is where the field sits, not the single best nonce.
#
#   track   n   min     max     median
#    7000  44   71840   71920   71862
#    9000  42   73718   73801   73739
#   11000  50   75210   75291   75234
#   13000  45   76501   76600   76523
#   15000  53   77664   77778   77696
#
# Do not interpolate these. The observed spread within a track is only ~80-110
# units, so a guessed value is wrong by more than the entire competitive range.
MAINNET_OPTIMAL = {
    7000: 71_862,
    9000: 73_739,
    11000: 75_234,
    13000: 76_523,
    15000: 77_696,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("measurements", type=Path,
                   help='JSON: {"7000": {"optimal": 1.142, "random": 3.437}, ...}')
    args = p.parse_args()
    measured = json.loads(args.measurements.read_text())

    # quality * PRECISION = (offset - avg) / scale * PRECISION
    # Linear in (1/scale, offset/scale):  q = A*offset - A*avg  where A = 1/scale
    rows, targets = [], []
    for track, target in MAINNET_OPTIMAL.items():
        avg = measured[str(track)]["optimal"]
        rows.append([1.0, -avg])
        targets.append(target / QUALITY_PRECISION)

    # Solve for [offset/scale, 1/scale]
    coeffs, *_ = np.linalg.lstsq(np.array(rows), np.array(targets), rcond=None)
    inv_scale = coeffs[1]
    scale = 1.0 / inv_scale
    offset = coeffs[0] * scale

    print(f"QUALITY_OFFSET = {offset:.6f}")
    print(f"QUALITY_SCALE  = {scale:.6f}")
    print()
    print(f"{'track':>7} {'avg_opt':>9} {'target':>9} {'fitted':>9} {'resid':>8}")
    for track, target in MAINNET_OPTIMAL.items():
        avg = measured[str(track)]["optimal"]
        fitted = round((offset - avg) / scale * QUALITY_PRECISION)
        print(f"{track:>7} {avg:>9.4f} {target:>9} {fitted:>9} {fitted - target:>8}")

    print()
    for track in MAINNET_OPTIMAL:
        rnd = measured[str(track)]["random"]
        print(f"  n_queries={track}: random scores "
              f"{round((offset - rnd) / scale * QUALITY_PRECISION)}")
    print("\nAll random scores should sit below min_active_quality (68,500).")
    print("Residuals materially above ~100 mean fixed constants do not suffice")
    print("and the constants must vary with n_queries.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Measure optimal and random distances per track**

This reads the **committed blob** rather than the PyTorch checkpoint, so the fitted constants describe what actually ships. It computes exact 1-NN itself, which is what breaks the otherwise circular dependency on Task 7's solver.

Create `/workspace/tig-bench/measure_distances.py` on tig-gpu (scratch, not committed):

```python
#!/usr/bin/env python3
"""Measure optimal and random nearest-neighbour distances on GAN instances.

Reads the committed weight blob, not the PyTorch checkpoint: the calibration
constants must describe the bytes that ship. Computes exact 1-NN by brute force,
so it does not depend on the Rust reference solver and can run before it exists.

Latent draws use torch RNG rather than curand, so instances are not identical to
the real generator's. That is fine here — the fit needs the distance
*statistics*, which converge over a million vectors.
"""

import json
import struct
import sys

import torch

BLOB = "tig-challenges/src/vector_search/weights/v1_sift.bin"
LATENT_DIM = 128
TRACKS = [7000, 9000, 11000, 13000, 15000]
CHUNK = 65536
DEV = torch.device("cuda:0")


def load_blob(path):
    data = open(path, "rb").read()
    assert data[:8] == b"TIGGAN01", "bad magic"
    at = 8
    (num_layers,) = struct.unpack_from("<I", data, at); at += 4
    layers = []
    for _ in range(num_layers):
        in_dim, out_dim = struct.unpack_from("<II", data, at); at += 8
        w = torch.frombuffer(data, dtype=torch.float32, count=in_dim * out_dim,
                             offset=at).reshape(out_dim, in_dim).clone()
        at += in_dim * out_dim * 4
        b = torch.frombuffer(data, dtype=torch.float32, count=out_dim,
                             offset=at).clone()
        at += out_dim * 4
        layers.append((w.to(DEV), b.to(DEV)))
    assert at == len(data), f"{len(data) - at} trailing bytes"
    return layers


@torch.no_grad()
def generate(layers, n, gen):
    out = torch.empty(n, layers[-1][0].shape[0], device=DEV)
    for start in range(0, n, CHUNK):
        rows = min(CHUNK, n - start)
        x = torch.randn(rows, LATENT_DIM, device=DEV, generator=gen)
        for i, (w, b) in enumerate(layers):
            x = torch.nn.functional.linear(x, w, b)
            if i + 1 < len(layers):
                x = torch.nn.functional.leaky_relu(x, 0.2)
        out[start : start + rows] = x
    return out


@torch.no_grad()
def avg_nn_distance(db, queries, chunk=128):
    db_sq = (db * db).sum(1)
    total = 0.0
    for i in range(0, queries.shape[0], chunk):
        q = queries[i : i + chunk]
        d = db_sq.unsqueeze(0) - 2.0 * (q @ db.T) + (q * q).sum(1, keepdim=True)
        total += d.clamp(min=0).min(1).values.sqrt().sum().item()
    return total / queries.shape[0]


@torch.no_grad()
def avg_random_distance(db, queries, gen):
    idx = torch.randint(0, db.shape[0], (queries.shape[0],), device=DEV, generator=gen)
    diff = queries - db[idx]
    return diff.pow(2).sum(1).sqrt().mean().item()


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    layers = load_blob(BLOB)
    results = {}
    for nq in TRACKS:
        gen = torch.Generator(device=DEV).manual_seed(20260813)
        db = generate(layers, 100 * nq, gen)
        queries = generate(layers, nq, gen)
        results[str(nq)] = {
            "optimal": avg_nn_distance(db, queries),
            "random": avg_random_distance(db, queries, gen),
        }
        print(f"n_queries={nq}: {results[str(nq)]}", flush=True)
        del db, queries
        torch.cuda.empty_cache()
    with open("measurements.json", "w") as f:
        json.dump(results, f, indent=2)
    print("wrote measurements.json")


if __name__ == "__main__":
    sys.exit(main())
```

Run it:

```bash
ssh tig-gpu 'source /venv/main/bin/activate && cd /workspace/tig-bench && \
  python measure_distances.py'
```

- [ ] **Step 3: Run the fit**

```bash
ssh tig-gpu 'source /venv/main/bin/activate && cd /workspace/tig-bench && \
  python scripts/calibrate_vector_search.py measurements.json'
```

- [ ] **Step 4: Apply the constants**

Replace `evaluate_solution` in `tig-challenges/src/vector_search/mod.rs`:

```rust
    /// Calibrated so GAN instances reproduce the quality band the Gaussian
    /// generator produced on mainnet (~68,500-78,000 across active tracks).
    /// A single-constant form cannot do this: matching the spread alone leaves
    /// the absolute level wrong, and the level is what the quality-target
    /// machinery keys on. Derived by scripts/calibrate_vector_search.py.
    const QUALITY_OFFSET: f64 = 0.0; // <- paste from step 3
    const QUALITY_SCALE: f64 = 1.0;  // <- paste from step 3

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
```

Place the two constants at module scope, next to `MAX_THREADS_PER_BLOCK`.

- [ ] **Step 5: Verify the band**

Re-run the measurement at all five tracks and confirm optimal quality lands within ~100 units of the mainnet targets and random scores fall below 68,500.

- [ ] **Step 6: Commit**

```bash
git add scripts/calibrate_vector_search.py tig-challenges/src/vector_search/mod.rs
git commit -m "Recalibrate vector_search quality for the GAN distance scale

The hardcoded 11.0 baseline assumed 250-dim hypercube distances; GAN
output has optimal avg_dist ~1.14 against ~10.18, so every solution
would score ~0.89. Two constants fitted across all five active tracks,
because matching the spread alone leaves the absolute level wrong.

Fit residuals: <paste from step 3>"
```

---

### Task 7: Reference algorithm patch and migration validation

Existing algorithms cannot run at 128 dims. This produces a dimension-parameterised reference build to prove the new challenge is solvable and lands in the right quality band — it is validation, not a shipped artifact. Redeploying player-submitted code is a governance decision, not this plan's.

**Files:**
- Create: `/workspace/tig-bench/tig-algorithms/src/vector_search/refsearch/mod.rs` (tig-gpu scratch, not committed)
- Create: `/workspace/tig-bench/tig-algorithms/src/vector_search/refsearch/kernels.cu` (tig-gpu scratch, not committed)

**Interfaces:**
- Consumes: the `Challenge` struct from Task 4.
- Produces: nothing the codebase depends on — a measurement only.

- [ ] **Step 1: Write a brute-force reference solver**

Start from `tig-algorithms/src/vector_search/template.rs` and `template.cu`. It needs only exact 1-NN, reading `challenge.vector_dims` rather than assuming any value — the point is to establish the achievable quality, not to be fast:

```cuda
extern "C" __global__ void ref_nearest(
    const float *__restrict__ database,
    const float *__restrict__ queries,
    int *__restrict__ best_index,
    const int database_size,
    const int num_queries,
    const int vector_dims
)
{
    for (int q = blockIdx.x; q < num_queries; q += gridDim.x)
    {
        const float *query = queries + (long long)q * vector_dims;
        float local_best = 3.402823466e+38f;
        int local_arg = 0;
        for (int d = threadIdx.x; d < database_size; d += blockDim.x) {
            const float *cand = database + (long long)d * vector_dims;
            float dist = 0.0f;
            for (int i = 0; i < vector_dims; ++i) {
                float diff = query[i] - cand[i];
                dist = fmaf(diff, diff, dist);
            }
            if (dist < local_best) { local_best = dist; local_arg = d; }
        }
        // Reduce across the block. Ties break toward the lower index so the
        // result does not depend on which thread got there first.
        __shared__ float s_best[256];
        __shared__ int s_arg[256];
        s_best[threadIdx.x] = local_best;
        s_arg[threadIdx.x] = local_arg;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                int other = threadIdx.x + stride;
                if (s_best[other] < s_best[threadIdx.x] ||
                    (s_best[other] == s_best[threadIdx.x] &&
                     s_arg[other] < s_arg[threadIdx.x])) {
                    s_best[threadIdx.x] = s_best[other];
                    s_arg[threadIdx.x] = s_arg[other];
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) best_index[q] = s_arg[0];
        __syncthreads();
    }
}
```

- [ ] **Step 2: Build it**

```bash
ssh tig-gpu 'export CHALLENGE=vector_search; cd /workspace/tig-bench && \
  build_algorithm refsearch'
```

If `build_algorithm` is not on PATH (it ships in the dev image), invoke `tig-binary/scripts/build_ptx` and `build_so` directly.

- [ ] **Step 3: Run across all five tracks**

```bash
ssh tig-gpu 'export PATH=/usr/local/cuda-12.6/bin:$HOME/.cargo/bin:$PATH; \
  export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$(rustc +nightly-2025-02-10 --print target-libdir):$LD_LIBRARY_PATH; \
  export CHALLENGE=vector_search; cd /workspace/tig-bench && \
  for nq in 7000 9000 11000 13000 15000; do \
    python3 scripts/test_algorithm refsearch n_queries=$nq null \
      --tig-runtime-path ./target/release/tig-runtime \
      --tig-verifier-path ./target/release/tig-verifier \
      --lib-dir ./tig-algorithms/lib --nonces 3; done'
```

Expected: 3/3 valid at every track, with `avg_quality` above `min_active_quality` (68,500) and near the mainnet targets from Task 6.

- [ ] **Step 4: Record the results in the spec**

Append a "Validation" section to `docs/superpowers/specs/2026-08-13-gan-instance-generation-design.md` with the measured qualities and per-track timings, then:

```bash
git add docs/superpowers/specs/2026-08-13-gan-instance-generation-design.md
git commit -m "Record end-to-end validation of GAN instance generation

A dimension-parameterised reference solver scores above
min_active_quality on all five active tracks, confirming the new
instances are solvable and land in the intended quality band."
```

---

## Notes for the executor

- **Do not push.** Every task commits locally. Publishing waits on the user.
- Task 2 ends in a decision gate. Stop there and report; do not proceed on your own judgement.
- The two production algorithms (`autovector_g`, `there_v10`) will not run at 128 dims at any point in this plan. That is expected and is the subject of the migration window, not a regression to debug.
- Only 2 of ~92 vector_search algorithms were ever tested against the dimension change. Do not generalise from them.
- Probes from the design phase are in the session scratchpad: `determinism_probe.py`, `scale_probe.py`, `calibration_probe.py`, `timing_probe.py`. Task 6 Step 2 extends `scale_probe.py`.
