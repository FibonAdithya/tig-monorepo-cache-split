# GAN-based instance generation for vector_search

Status: implemented and validated on a single GPU architecture; see "Validation".
Migration and cross-architecture confirmation remain open.
Branch: `vector_search/gan_instance_gen` (forked from `blank_slate`)

## Goal

Replace the vector_search instance generator with a trained GAN so that challenge
instances resemble real embedding data rather than a synthetic Gaussian mixture.
The motivation is transfer: algorithms that win on TIG should be good at vector
search on real data.

## Background

`tig-challenges/src/vector_search/mod.rs` currently generates instances from a
log-normal-weighted mixture of anisotropic truncated Gaussians on `[-1,1]^250`,
via two CUDA kernels (`generate_clusters`, `generate_vectors`). `database_size =
100 * n_queries` and `vector_dims = 250`.

A WGAN-GP generator has already been trained on SIFT-1M in the `wgan-synthetic`
project. Two candidates were considered:

| | v1 (`x100k_ema_only`) | v4 (`x100k_structured`) |
|---|---|---|
| architecture | MLP 128→512→1024→1024→128, LeakyReLU(0.2) | trunk + magnitude/gate/sparsity heads, Conv3d gate coupling |
| params | 1,772,160 (7.09 MB) | 1,904,412 (7.62 MB) |
| step | 88,000 | 28,000 |
| sampling weights | `generator_state_dict` | `ema_params` |

**Decision: build on v1**, with the generator behind an interface so v4 can be
added later.

## Evidence

All measurements on tig-gpu (RTX 4060, sm_89). Probes are preserved in the
session scratchpad (`determinism_probe.py`, `scale_probe.py`,
`calibration_probe.py`).

### Determinism

`tig-verifier/src/main.rs:136` regenerates the instance on the verifier's own GPU
and re-scores the submitted indexes. Quality is a fixed-point integer compared
exactly, so the generator must reproduce bit-identically on heterogeneous
hardware.

Both variants were bitwise reproducible run-to-run. Under a rechunked GEMM
(cuBLAS selecting a different kernel — the same class of difference a different
architecture produces):

| | max Δ @ 200k | max Δ @ 1M | gate flips @ 1M | quality Δ |
|---|---|---|---|---|
| v1 | 5.6e-6 | 5.6e-6 | — | 0 |
| v4 | 4.2e-3 | 7.9e-2 | 18 | 0 |

v1's error is scale-invariant. v4's grows with database size because its hard
`soft > 0.5` gate threshold turns a 1-ULP difference into a flipped coordinate
and, after renormalisation, a materially different vector. With TF32 enabled both
variants shifted the quality integer by 1 and would fail verification.

The flip risk is not intrinsic to v4 — it is an artifact of calling cuBLAS, whose
kernel selection is heuristic. A hand-written fixed-order GEMM removes it for
both. v1 was chosen on the remaining grounds: simpler port, and better solver
discrimination (below).

### Calibration

Measured at `n_queries = 10,000` (db = 1,000,000):

| | avg_dist optimal | avg_dist random | quality optimal | quality random |
|---|---|---|---|---|
| current Gaussian | 10.180 | 12.344 | 74,518 | −122,203 |
| v1 GAN | 1.142 | 3.437 | — | — |

The GAN's optimal-vs-random spread (208,568) is within 6% of the current
generator's (196,721), so it produces comparably discriminating instances.

**However**, live mainnet data (block 1298164) shows this is the wrong target.
Active tracks are `n_queries` ∈ {7000, 9000, 11000, 13000, 15000}, all with
`min_active_quality: 68,500`, and real qualifier qualities are:

| track | observed range | spread |
|---|---|---|
| n_queries=7000 | 71,840 – 71,920 | ~80 |
| n_queries=9000 | 73,718 – 73,801 | ~83 |
| n_queries=15000 | 77,665 – 77,752 | ~87 |

Competing algorithms sit within ~80 quality units of each other, all within ~1%
of exact 1-NN. Quality also drifts ~6,000 units across the track range, so it is
not invariant to instance size. Calibration must preserve the achievable band
(~68,500–78,000) across all five active tracks, not merely the optimal-vs-random
spread.

## Design

### Challenge parameters

`vector_dims` becomes 128, matching the trained weights. `database_size = 100 *
n_queries`, `Solution`, and the evaluation path are unchanged.

This breaks every existing algorithm and therefore ships with a migration window
(see "Migration" below). The alternative — retraining at 250 dims on a
substituted corpus — was rejected: SIFT is natively 128, so 250 would require
moving to nytimes-256, gist-960 or openai-1536 with a projection down to 250.
That costs a training run, degrades the very statistics the feature exists to
improve (truncation lowers LID and raises relative contrast, making search
easier), and retires v4 permanently, since its `layout: [4,4,8]` *is* the SIFT
descriptor grid and is meaningless on text embeddings.

### Generator

v1's weights are exported to a flat f32 blob (7.09 MB) and embedded with
`include_bytes!`. A binary in git guarantees byte-identical weights on every
verifier; 7 MB does not warrant LFS. The blob is uploaded to the device once per
`generate_instance`.

### Deterministic forward pass

Four GEMMs and a LeakyReLU, hand-written in `kernels.cu` with a compile-time tile
size and explicit `fmaf`. **Never cuBLAS** — its kernel selection is the sole
source of the cross-architecture divergence measured above. TIG ships PTX and
loads it via `Ptx::from_file`, so every verifier executes the same instruction
sequence; a fixed accumulation order then makes the result bit-identical. A
shared-memory tiled kernel with a compile-time tile size is both deterministic
and fast. Cost is ~1.8 TFLOP at 1M vectors.

### Latent sampling

Unchanged in form from the existing kernel: `curand_init(seed[i % 4], i, 0,
&state)` per vector, then 128 normal draws. Elementwise and per-index, therefore
deterministic by construction with no inter-thread coordination.

### Quality recalibration

The hardcoded baseline of 11.0 is wrong for the GAN's distance scale. A
two-constant form is needed, because matching only the spread leaves the absolute
level wrong (optimal would score 0.902 against today's 0.075), which would
disturb the quality-target machinery:

```
quality = (offset - avg_dist) / scale
```

Constants must be derived from a committed calibration run that covers all five
active tracks, chosen so the achievable quality band matches today's
(~68,500–78,000) at each `n_queries`. Because quality drifts with instance size,
the calibration must confirm whether fixed constants suffice or whether they need
to vary with `n_queries`.

### Pluggability

The generator sits behind a small trait — weights blob, layer dims, forward pass —
so adding v4 later means adding its kernels and blob, not reworking the challenge.

**As built, the trait was deliberately not written**, while the outcome it was
for was delivered. The blob format carries per-layer dimensions, and
`generate_instance` derives `vector_dims`, the layer count and the loop bounds
from the parsed weights, so any MLP-shaped generator drops in with no code
change. A trait with one implementor would be abstraction ahead of a second
case, and v4 is not MLP-shaped anyway: it needs Conv3d, gate sampling and
renormalisation, so the right boundary for it cannot be designed from v1 alone.
Add the trait when v4 arrives and its real requirements are visible.

## Baseline measurements

Measured on tig-gpu (RTX 4060) with `tig-runtime`/`tig-verifier` built from this
tree at the pinned `nightly-2025-02-10` toolchain, against the two production
algorithms `autovector_g` (c004_a103, 38.6% adoption) and `there_v10`
(c004_a100, 54.1%) — 92.7% combined.

The reproduction is faithful: `autovector_g` averages quality 71,849 at
`n_queries=7000` against mainnet's observed 71,840–71,920. Both algorithms return
identical qualities per nonce, i.e. both reach the same near-optimal solutions and
compete only on speed.

| n_queries | tig-runtime | tig-verifier |
|---|---|---|
| 7000 | 520–643 ms | 455–521 ms |
| 9000 | 627–683 ms | 474–556 ms |
| 11000 | 724–816 ms | 505–637 ms |
| 13000 | 851–934 ms | 543–609 ms |
| 15000 | 1003–1082 ms | 587–648 ms |

Regressing verifier time against database size gives ~0.17 ms per 1k vectors plus
~350 ms fixed CUDA context/module overhead, so **current instance generation costs
~120 ms at 700k vectors and ~250 ms at 1.5M**.

The GAN forward pass at the same sizes costs **462 ms (7000) to 1045 ms (15000)** —
about **4x the current generator**, at ~5.1 effective TFLOPS. That is a cuBLAS
floor; the deterministic hand-written kernel required by this design will not beat
it. Generation runs in both the runtime and the verifier, so this adds ~340–800 ms
to each and roughly doubles verification cost.

**Toolchain note.** `tig-runtime` must be built with `nightly-2025-02-10`, matching
`Dockerfile.dev`, and the algorithm `.so` needs that toolchain's
`libstd-399a850a7013db0c.so` on `LD_LIBRARY_PATH`. Building the runtime with a
different toolchain makes correctly-working algorithms fail, which cost a
misdiagnosis during this investigation.

## Migration

**Decision: ship 128 dims with a migration window.**

Existing algorithms produce 0/3 valid solutions at `vector_dims = 128` while
scoring 3/3 at 250 (controlled by rebuilding both configurations with the pinned
toolchain). They do not, however, blindly assume 250 — both guard explicitly:

```rust
const DIMS: usize = 250;
const DIMS_PAD: usize = 256;
...
if challenge.vector_dims as usize != DIMS {
    return Err(anyhow!("expects {} dimensions, got {}", DIMS, challenge.vector_dims));
}
```

They refuse rather than compute wrong answers. The observed SIGSEGV is most likely
that error crossing the dylib boundary between a locally-built runtime and the
officially-compiled `.so`, not the guard itself.

The required change is therefore small: two constants in `mod.rs`, and four `250`
literals in 672 lines of `kernels.cu` (a stride `row * 250ll`, a loop bound, and
two kernel names) which must become a `dims` parameter. Roughly 20–40 lines per
algorithm. At 128 the fiddliest part gets simpler — `pack_norm_fp16_250_to_256`
exists because 250 is not a tensor-core-friendly width, whereas 128 is already
aligned.

Two further findings:

- Both algorithms also hardcode the track sizes (`match num_queries { 7_000 |
  9_000 | 11_000 | 13_000 | 15_000 => ... }` with per-track tuned WMMA chunk
  constants), so the ecosystem broadly assumes challenge parameters are frozen.
- Only 2 of ~92 vector_search algorithms were tested. The others may fail
  differently.

Because these are player submissions earning adoption-based rewards, TIG patching
and redeploying them is a governance question, not an engineering one — see
`docs/licenses/inbound_license.pdf`, `innovator_outbound_license.pdf` and
`docs/agreements/ip_policy.pdf`. The assumed path is that authors resubmit against
a published 128-dim spec.

## Testing

- Determinism: bit-identical output across different launch geometries.
- Golden vectors: a few hundred outputs pinned against the PyTorch reference.
- Calibration: optimal and random quality stay within the expected band on every
  active track.
- Migration: dimension-parameterised builds of `autovector_g` and `there_v10`
  must score above `min_active_quality` (68,500) on the new instances. Their
  current builds cannot run at 128 dims at all, so patched versions are a
  precondition for this test rather than an outcome of it.

## Validation

End-to-end on tig-gpu (RTX 4060, sm_89, CUDA 12.6, `nightly-2025-02-10`), with
every GPU run holding a `gpu-claim`.

### Instances are solvable and land in the intended band

`refsearch`, a brute-force exact 1-NN reference reading `challenge.vector_dims`
rather than assuming a value, run through `tig-runtime` and `tig-verifier` under
the recalibrated constants. 15/15 solutions valid, 30/30 zero exit codes, no
invalid solutions or warnings:

| track | nonce 0 | nonce 1 | nonce 2 | 3-nonce mean | mainnet median |
|---|---|---|---|---|---|
| 7000 | 72,099 | 71,214 | 70,431 | 71,248 | 71,862 |
| 9000 | 73,751 | 73,254 | 73,255 | 73,420 | 73,739 |
| 11000 | 75,361 | 74,746 | 75,383 | 75,163 | 75,234 |
| 13000 | 76,728 | 76,698 | 76,764 | 76,730 | 76,523 |
| 15000 | 77,056 | 77,083 | 77,142 | 77,094 | 77,696 |

Every value clears `min_active_quality` (68,500), the whole set sits inside
mainnet's observed 71,840–77,778 range, and quality rises with track size as it
does on mainnet. The 3-nonce means deviate from the medians by -614 to +207,
which is what a per-nonce sigma of ~400–500 predicts at n=3 (standard error
~230–290) — nonce noise, not calibration error. The 24-nonce fit is the
authoritative figure.

Exact 1-NN is admissible: `fuel_consumed` is 3.84e11 at n_queries=7000 and
scales to ~1.76e12 at 15000, against a `max_fuel_budget` of 5e12.

### The port matches PyTorch

8/8 `tig-challenges` generator tests pass, including golden vectors dumped from
the source checkpoint. Largest observed deviation 4.47e-7 against a 1e-5 bound.

### Output is invariant to launch geometry

This is the property verification depends on, and the quality integer cannot
show it — one quality unit is ~0.077 of total distance while a last-bit change
in one coordinate moves it ~1e-7. So the raw device buffers were hashed
(FNV-1a over the little-endian f32 bytes) across the two knobs that genuinely
change launch geometry for the tiled kernel.

n_queries=7000, 89,600,000 database floats and 896,000 query floats:

| forward chunk | latent block | database digest | queries digest |
|---|---|---|---|
| 65,536 | 256 | `4353c1756d375efd` | `ca3bc89bed2c9253` |
| 32,768 | 256 | `4353c1756d375efd` | `ca3bc89bed2c9253` |
| 65,536 | 128 | `4353c1756d375efd` | `ca3bc89bed2c9253` |
| 131,072 | 512 | `4353c1756d375efd` | `ca3bc89bed2c9253` |

n_queries=15000, 192,000,000 database floats: `58e13837b2fcfdbd` /
`b274ea82caa8c1ac` at both 65,536/256 and 131,072/512.

Negative control, n_queries=9000: `4edf76f71ce42dda` / `7933addf12ee2963` —
different, so the digest is input-sensitive and the matches above are real.

Separately, the first four database floats are bit-identical across all three
tracks (`3e727ac5 3cac0367 3c160a62 3c47de90`), independently confirming that a
vector's value depends only on its global index and not on instance size.

### Memory and cost

The chunked pass holds at the largest track with no OOM. Combined
runtime+verifier time for generation plus evaluation is ~2.3 s per nonce at
n_queries=7000 and ~3.8 s at 15000.

Cross-architecture reproducibility remains argued rather than measured: only one
GPU model was available.

## Risks and open items

- **Instance generation cost — measured, resolved, accepted.** The first
  deterministic kernel used one block per input row and re-read all 7.08 MB of
  weights per vector, giving 5.00 TB of weight traffic and 22,515 ms at
  n_queries=7000 — ~49x the cuBLAS floor, and bandwidth-bound at ~81% of the
  4060's 272 GB/s. The shipped kernel tiles a 128x64 output block with an 8x4
  per-thread register tile, which reuses each weight across 128 rows and lands
  at 628 ms (707k vectors) and 1,335 ms (1,515k) — inside 1.5x of the cuBLAS
  floor. Determinism survives because tiling changes which block owns an output
  element, not the accumulation order within it; split-K would break that and is
  deliberately not used.

  The residual cost is inherent: ~5x the previous generator's ~120–250 ms, in
  both the runtime and the verifier, from 1,769,472 MACs per vector. No kernel
  removes it — only a smaller generator would. Accepted.

- **Per-nonce quality variance is ~7–9x wider than today — accepted, and the
  most consequential property of this change.** `quality = (offset - avg_dist) /
  scale` is linear, so the single constant `scale` fixes both how far quality
  drifts across the five tracks and how much per-nonce noise reaches quality.
  Their ratio is a property of the data, not the constants:

  | | cross-track drift | per-nonce noise | ratio |
  |---|---|---|---|
  | mainnet (Gaussian, 250-dim) | ~5,834 units | ~80 units | ~73 |
  | GAN (SIFT-like, 128-dim) | ~5,834 units | ~500 units | 12.3 |

  The cause is concentration of measure. At 250 uniform dimensions per-query 1-NN
  distance has CV ~0.7%, so its average over thousands of queries barely moves
  between nonces. Real clustered embedding data at 128 dims has per-query CV
  ~75%, roughly 100x more relative dispersion.

  Because bundle quality is the *median* of `num_nonces_per_bundle` nonces
  (`tig-protocol/src/contracts/benchmarks.rs:218`), the fair comparison is of
  bundle medians. Predicted spread across ~50 qualifiers, against mainnet's
  observed spread of the same quantity:

  | track | nonces/bundle | predicted | mainnet | ratio |
  |---|---|---|---|---|
  | 7000 | 20 | 647 | 80 | 8.1x |
  | 9000 | 17 | 663 | 83 | 8.0x |
  | 11000 | 15 | 728 | 81 | 9.0x |
  | 13000 | 10 | 708 | 99 | 7.1x |
  | 15000 | 5 | 1,073 | 114 | 9.4x |

  Three mitigations were tested and rejected. Normalising by a cheap
  per-instance reference distance the verifier could compute without solving
  1-NN correlates only 0.32–0.76 with the 1-NN scale and cuts the noise by
  5–35%, because the variance is mostly query-sampling noise rather than
  instance-scale noise. Raising `num_nonces_per_bundle` helps only as
  1/sqrt(N), so track 15000 would need ~443 nonces instead of 5. Per-track
  constants remove the 282-unit fit residual but not the ~700-unit noise, and
  hardcode the five track sizes into `evaluate_solution`.

  Accepted deliberately: matching the per-track band is what preserves the
  level the qualifier machinery keys on, and the variance is inherent to using
  realistic data, which is the point of the change.

- **The signal was not measured, only the noise — open.** Calibration targets
  exact 1-NN. Mainnet's tight ~80-unit spread exists because every qualifier
  sits within ~1% of exact 1-NN on the *old* instances, and exact 1-NN is
  affordable: brute force consumes 3.84e11 fuel at n_queries=7000 and ~1.76e12
  at 15000, against a `max_fuel_budget` of 5e12. If that remains true on
  clustered data, serious algorithms will again cluster at the optimum and the
  ~700-unit nonce noise will dominate the competitive gap. If instead clustered
  data spreads approximate ANN algorithms over thousands of quality units, the
  noise is proportionately fine. Deciding this needs a real approximate ANN
  algorithm at 128 dims, which is beyond this plan's scope.
- **Migration is a coordination problem, not a technical one.** The per-algorithm
  fix is ~30 lines, but only 2 of ~92 vector_search algorithms were tested, and
  redeploying player-submitted code is a governance question. Rollout needs
  notice and sign-off, not just a merge.
- **Fidelity scope.** 128 dims is fixed by the trained weights, and the training
  corpus is SIFT specifically. "Realism" here means SIFT-like, which is narrower
  than embeddings in general.
- **Single-architecture evidence.** Only one GPU model was available, so
  cross-architecture reproducibility is argued from fixed-order arithmetic and
  shipped PTX rather than measured on differing hardware.
