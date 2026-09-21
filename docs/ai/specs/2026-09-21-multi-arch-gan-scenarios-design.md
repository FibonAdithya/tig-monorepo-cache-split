# Multi-architecture GAN scenarios for vector_search: SIFT v4, NYTimes v3, GloVe v1

Design, 2026-09-21. Extends `2026-08-25-per-scenario-gan-tracks-design.md`.
That document's `Track { s: Scenario }` model, wire format and recall-gated
verification all stand. This design **reverses one of its decisions**: the
kernel ABI freeze to a single MLP architecture family. The reason is below.

Numbers in this document are one of three kinds, and say which: MEASURED
(command run on 2026-09-21, shown or named), computed (arithmetic from stated
inputs, shown), or ESTIMATE (unverified). Figures quoted from an earlier spec
are attributed to it and were not re-measured.

## Why the freeze is lifted

The 2026-08-25 spec froze `kernels.cu` so that adding a scenario would never
force algorithms to be rebuilt: every generator had to be an MLP with
LeakyReLU(0.2) and a 128-wide latent. Since then the generators in
`~/TIG/wgan-synthetic` were trained against per-corpus gates, and the ones that
pass are not all MLPs:

| corpus | accepted variant | `generator_type` | latent | fits the frozen ABI |
|---|---|---|---|---|
| GloVe-100 | `v1`, five seeds pass `gates/glove.yaml` | `mlp` | 128 | yes |
| NYTimes-256 | `v3` (`v3_best`), passes `gates/nytimes.yaml` | `spherical` | 512 | no |
| SIFT-128 | `v4` at 100k steps, passes `gates/sift.yaml` | `structured_gated` | 128 | no |

Source: `configs/{glove/v1,nytimes/v3,sift/v4}.yaml` and the `## Gate` sections
of `docs/datasets/{glove,nytimes,sift}.md` in the WGAN repo at commit `5b96976`.

The MLP rungs for NYTimes and SIFT exist but fail their gates:
`docs/datasets/nytimes.md` states that `v0` "does not reproduce NYTimes' search
difficulty at any checkpoint". Shipping them would make the tracks realistic in
name only, which defeats the purpose the 2026-08-25 spec gave for having
per-corpus tracks.

The freeze has not yet taken effect. The one network-wide resubmit that the
250→128 dimension change requires has not happened, so changing `kernels.cu`
now adds no second migration event. After that resubmit the freeze applies
again, to the larger kernel set defined here.

## Scenarios

| track id | generator | architecture | dims | latent | checkpoint on tig-gpu |
|---|---|---|---|---|---|
| `s=sift_128` | SIFT v4, 100k retrain, selected checkpoint | `structured_gate` | 128 | 128 | `/workspace/sift-v4/v4_sift1m_x100k/best_generator.pt` |
| `s=nytimes_256` | NYTimes v3, seed 42, `v3_best` | `spherical` | 256 | 512 | `/workspace/nytimes-v3/v3_seed42/best_generator.pt` |
| `s=glove_100` | GloVe v1, seed 42 | `mlp` | 100 | 128 | `/workspace/glove-probes/probe_spectrum_seed42/best_generator.pt` |

`probe_spectrum_seed42` is GloVe v1 seed 42: `configs/glove/v1_seed4x.yaml`
record that provenance in their headers. Seed 42 was chosen because the other
two generators are seed 42, not because it is the best of the five.

Checkpoint sizes and hashes, MEASURED with `ls -l` and `sha256sum` over SSH:

| checkpoint | bytes | sha256 |
|---|---|---|
| SIFT v4 | 39,962,577 | `09eaac8fd84d7cd4c96fd2ddbcbeb967d55d422fecfa7d26f0a34590879a1e6c` |
| NYTimes v3 | 53,002,329 | `b1acfdda41795d310ea384d75e00c975073ce47ffbe58e551c3e5b4727ee363a` |
| GloVe v1 s42 | 30,059,597 | `38d8013a748c83b02f4e4aa99bb64fcda221cbc14e4ac8bd9c8f20404b66d7a3` |

~~One thing to confirm before exporting NYTimes: that `v3_best` in
`gates/nytimes.yaml` and `tests/test_check_gate.py` refers to
`v3_seed42/best_generator.pt` and not to a step checkpoint or the
`v3_seed42_100k` continuation. The doc says the selected step "is a transient",
so the file identity matters. Check the committed `docs/results/nytimes-*`
summary that carries the `v3_best` label.~~

**Correction 2026-09-21 (measured):** confirmed. Three copies of
`best_generator.pt` on the box are byte-identical (sha256 `b1acfdda…`). See
`docs/measurements/2026-09-21-c004-multi-arch-generators.md`.

### Scenario decisions

1. **SIFT v4 replaces v1 under the same `sift_128` id.** Every algorithm
   rebuilds for the kernel change anyway; a second SIFT track would only split
   qualifiers. `weights/v1_sift.bin` and `weights/golden_vectors.json` stay in
   the repo as the regression fixture for the `TIGGAN01` parser and the MLP
   reference forward pass. They are no longer wired to any scenario.
2. **All three tracks use `n_queries = 7,000`, `database_size = 700,000`.** The
   2026-08-25 argument holds: 20 nonces per bundle at this size. Database VRAM,
   computed as `700,000 × dims × 4`: NYTimes 716.8 MB, SIFT 358.4 MB, GloVe
   280.0 MB. Not checked: the WGAN gates were measured at N = 20,000, and the
   challenge samples 700,000 rows, more than NYTimes' ~290k real rows. SIFT
   already works this way, but no gate statistic has been measured at 700,000.
3. **`min_recall = 0.9`, `recall_tolerance = 1e-6`, `audit_samples = 1,000`**
   for the two new tracks, copied from SIFT. The 0.9 bar was derived from
   SIFT's d2/d1 distribution. For NYTimes and GloVe it is a placeholder, not a
   measurement. See Follow-ups.
4. **`LATENT_DIM` stops being a global constant** and becomes a per-generator
   value from the blob header. `gan_sample_latents` already takes `latent_dim`
   as an argument, so that kernel is untouched.

## Blob format `TIGGAN02`

A second format beside `TIGGAN01`, not a replacement. All integers
little-endian, all floats f32 little-endian, tensors row-major `[rows][cols]`
as in `TIGGAN01`.

```
magic      "TIGGAN02"                     8 B
arch       u32    0 = mlp, 1 = structured_gate, 2 = spherical
latent_dim u32
output_dim u32
n_scalars  u32,   then n_scalars × f32
n_tensors  u32,   then per tensor: rows u32, cols u32, rows×cols × f32
```

Tensors are positional. Each architecture fixes its tensor order, its scalar
order, and the shape relations between tensors; the parser checks all three.
There are no names, strings or offsets, so the parser stays a close relative of
the hardened `TIGGAN01` one. A bias is a `rows × 1` tensor.

### `arch = 0`, mlp

Scalars: `eps`. Tensors: `(W_i, b_i)` for each layer in order. Relations:
`W_0.cols == latent_dim`; `W_i.cols == W_{i-1}.rows`; `b_i.rows == W_i.rows`;
last `W.rows == output_dim`. Forward: LeakyReLU(0.2) after every layer but the
last, then `x / max(‖x‖, eps)`.

`eps = 1e-8`, matching `src/sample/generate.py:69-71` in the WGAN repo, which
normalises every sampled row unconditionally. The gates measured normalised
rows, so the challenge must normalise too. This is also what makes Euclidean
1-NN equal to angular 1-NN for the angular corpora: for unit vectors
`d² = 2 − 2·cos`.

### `arch = 1`, structured_gate

Scalars, in order: `logit_clamp`, `magnitude_floor` (= `eps × 100`), `eps`.
Tensors, in order: trunk `(W_i, b_i)` × 3, `magnitude_head (W, b)`,
`gate_head (W, b)`, `sparsity_head (W, b)` with `W.rows == 1`,
`coupling` (`output_dim × output_dim`), `smoothing` (`output_dim × output_dim`).

Forward, for trunk output `h`. The trunk is `[Linear, LeakyReLU(0.2)] × 3`, so
the activation follows its last layer too, unlike the mlp's final layer:

```
m      = max(softplus(magnitude_head(h)), magnitude_floor)
logit  = logit_clamp · tanh((coupling · gate_head(h) + sparsity_head(h)) / logit_clamp)
noise  = smoothing · (log(u) − log1p(−u)),   u = clamp(uniform, eps, 1 − eps)
open_j = (logit_j + noise_j > 0)
if no gate is open: open = one_hot(argmax_j logit_j)
x      = open ⊙ m;   out = x / max(‖x‖, eps)
```

Three things the exporter or the formula removes from the GPU path:

- `coupling` is the `Conv3d(1,1,3)` over the (4,4,8) grid with circular padding
  on orientation and replicate padding on the spatial axes. It is a linear map
  on 128 values, so it is baked into a dense matrix.
- `smoothing` is the fixed Gaussian noise kernel composed with the per-position
  `noise_scale`. Also linear, also baked.
- PyTorch computes `hard = sigmoid((logit + noise) / T) > 0.5`. That is
  `logit + noise > 0` for any `T > 0`, so neither the sigmoid nor
  `gate_temperature` appears at inference. The straight-through term
  `soft − soft.detach()` is zero in the forward pass.

### `arch = 2`, spherical

Scalars, in order: `cos_r`, `sin_r`, `eps`. Tensors, in order: trunk
`(W_i, b_i)` × 3 with `W_0.cols == latent_dim − skip_dim`, `direction (W)`
with no bias, `tangent_in (W, b)` with `W.cols == skip_dim`, `gamma (W, b)`,
`beta (W, b)`, `tangent_out (W, b)`. `skip_dim` is not stored: it is
`tangent_in.W.cols`, and the parser checks
`trunk.W_0.cols + tangent_in.W.cols == latent_dim`.

Forward, for latent `z = [z_t | z_s]`:

```
h = trunk(z_t)                                  LeakyReLU(0.2) after every trunk layer
u = unit(direction(h))
a = tangent_in(z_s) · (1 + gamma(h)) + beta(h)
v = tangent_out(leaky(a))
v = v − (v·u) u;   t = unit(v)
out = cos_r · u + sin_r · t
```

`r = radius_min + (radius_max − radius_min) · sigmoid(radius_raw)` is one
learned scalar. The exporter computes `cos(r)` and `sin(r)` in float64 and
rounds to f32, so no transcendental runs on the GPU for this architecture.
Note the trunk here applies the activation after its last layer too, unlike
the mlp's final layer: `SphericalGenerator.trunk` is
`[Linear, LeakyReLU] × 3`.

### Blob sizes

Computed exactly from the layer shapes in the three configs (float count × 4,
plus headers of under 200 B):

| blob | floats | bytes (computed) | bytes (measured, `ls -l`) | header |
|---|---|---|---|---|
| `glove_100_v1.bin` | 1,743,460 | 6,973,840 + header | 6,973,936 | 96 B |
| `sift_128_v4.bin` | 1,937,153 | 7,748,612 + header | 7,748,764 | 152 B |
| `nytimes_256_v3.bin` | 3,281,152 | 13,124,608 + header | 13,124,768 | 160 B |

**Correction 2026-09-21 (measured):** the "bytes (measured)" and "header"
columns are new. `ls -l` on the box. See
`docs/measurements/2026-09-21-c004-multi-arch-generators.md`, "Blob sizes
(weights)".

These are arithmetic from config shapes, not file measurements; the exporter
prints the real byte count. All three exceed the 1 MB commit guard, as
`v1_sift.bin` (7,088,684 B, MEASURED `ls -l`) already does, so committing them
needs the deliberate `# allow-large-commit` and `ALLOW_LARGE_COMMIT=1` bypass.
Every algorithm `.so` links `tig-challenges`, so all three blobs plus the v1
fixture are embedded in each. If that matters, the v1 fixture can move behind
`#[cfg(test)]`; decide in the plan after measuring one `.so`.

## Exporter

`scripts/export_generator_weights.py` gains `--arch`, `--run-config` and
`--wgan-repo PATH`. It imports `build_generator` from the WGAN repo through the
path argument instead of copying the model code, rebuilds the module from the
run config, loads `generator_state_dict`, and writes `TIGGAN02`.

`generator_state_dict` is the key the WGAN sampler loads
(`src/sample/generate.py:57`). Exporting the same key means the shipped weights
are the ones the gates measured. This matters for SIFT v4, which trains with
`ema_decay: 0.999`: the plan must confirm which weights the trainer writes to
that key, and must not switch to an EMA key on the assumption that EMA is
better.

Baked matrices are derived by pushing the `output_dim` one-hot basis rows
through the module's own `_couple` and `_smooth_noise`, not by re-deriving the
padding arithmetic. `_position_noise_scale` in the WGAN repo already uses this
method.

Provenance goes in `weights/PROVENANCE.md`: per blob, the blob sha256, the
checkpoint sha256, the checkpoint path, the WGAN commit, and the exporter
command line.

`scripts/dump_golden_vectors.py` is extended to write, per blob, 8 fixed
latents and the PyTorch outputs. For SIFT v4 it also writes the fixed uniform
noise `u`, and the module's `_sample_gate` is driven with that `u` instead of
`torch.rand_like`, because a stochastic gate cannot be compared without pinning
its noise.

Checkpoints come off the box by the standing transfer rule: remote
`sha256sum` (done, table above), stream with
`ssh tig-gpu 'tar czf - <files>' | tar xzf - -C <dest>`, verify local hashes
against the table. Checkpoints are never committed; only blobs are.

## Kernels

Unchanged: `gan_sample_latents`, `gan_linear`, `evaluate_total_distance`. Every
matrix product in all three architectures goes through `gan_linear`, including
the two baked maps and bias-free layers, which pass an uploaded zero bias.
`gan_linear` has not been exercised with `out_dim = 1` (`sparsity_head`) or
`out_dim = 100` (GloVe); its bounds checks suggest both work, and test 6 below
covers both.

New kernels. Each thread owns whole rows and accumulates over dims
sequentially in index order, so results do not depend on launch geometry.

| kernel | computes | used by |
|---|---|---|
| `gan_row_normalize` | `x / max(‖x‖, eps)` in place, sum of squares by sequential `fmaf` | mlp |
| `gan_film_leaky` | `leaky(a · (1 + γ) + β)`, element-wise | spherical |
| `gan_sphere_combine` | `u = unit(d)`; `v −= (v·u)u`; `t = unit(v)`; `out = cos_r·u + sin_r·t`, written at `out_row_offset` | spherical |
| `gan_gate_noise` | per row and coordinate, `u = clamp(curand_uniform, eps, 1−eps)`; `log(u) − log1p(−u)` | structured_gate |
| `gan_gate_apply` | the `logit`, `open`, fallback, `m` and normalise steps above, written at `out_row_offset` | structured_gate |

### Two random-stream rules

1. **Spherical latent split.** `gan_linear` reads rows at stride `in_dim`, so it
   cannot consume the first 256 columns of a 512-wide buffer. The driver draws
   `z_t` and `z_s` as two 256-wide calls to `gan_sample_latents`. The second
   call's `index_offset` is shifted by `1 << 30`. Without the shift both calls
   would seed curand identically and return the same numbers, making
   `z_t == z_s`. The host asserts `index_base + count < 1 << 30` so the `i32`
   offset cannot overflow. Matching PyTorch's column split is not needed: the
   latent is i.i.d. normal.
2. **Gate noise.** `gan_gate_noise` calls
   `curand_init(seed_word, (1ULL << 40) + global_i, 0, &state)`. No
   `gan_sample_latents` call can reach that sequence range with an `i32` index.
   The cost of `curand_init` at that sequence offset is unmeasured; the plan
   measures it before committing to this scheme, and the fallback is a distinct
   seed word with the plain `global_i` sequence.

### Audit kernel width

`recall_audit` stages queries in `s_query[AUDIT_TQ][AUDIT_MAX_DIMS]` with
`AUDIT_MAX_DIMS = 128` (`kernels.cu:190`, mirrored at `mod.rs:76`). A 256-dim
scenario would overrun it with no error.

Shared memory today, computed from the four arrays: `s_query` 18×128×4 = 9,216,
`s_db` 256×17×4 = 17,408, `s_red` 1,024, `s_returned` 72; total 27,720 B, which
equals the figure in the kernel's own comment. At `AUDIT_MAX_DIMS = 256` it is
36,936 B. The kernel comment records that losing a resident block cost
107 → 182 ms against a 150 ms gate, so the raise may slow SIFT's audit.

Decided by measurement, in this order:

1. Raise `AUDIT_MAX_DIMS` to 256 in both files; re-time SIFT's audit against
   the existing gate in `audit_is_much_cheaper_than_a_naive_solve`.
2. If SIFT regresses past the gate: leave `recall_audit` byte-identical and add
   `recall_audit_wide` for `vector_dims > 128` with its own `AUDIT_TQ`, tuned
   the way the existing comment documents.

## Rust structure

```
vector_search/
  generator/
    mod.rs              Generator enum, TIGGAN01 + TIGGAN02 parsers, launch_linear
    mlp.rs              parse-checks, forward_cpu, forward_chunk
    structured_gate.rs  same three
    spherical.rs        same three
  scenarios.rs          three variants, Scenario::ALL
  mod.rs                generate_vectors delegates to Generator::forward_chunk
```

- `enum Generator { Mlp(..), StructuredGate(..), Spherical(..) }`, built by
  `Generator::from_blob(&[u8])`, which dispatches on the magic.
- `generate_vectors` keeps its chunk loop and its `index_base` contract and
  calls `generator.forward_chunk(rows, chunk_start, index_base, dest, ..)`.
  The four copy-pasted `gan_linear` launches (`mod.rs:144-211`) collapse into
  one `launch_linear` helper; the new drivers would otherwise add about ten
  more copies.
- Scratch buffers become a per-architecture named set sized from the blob,
  replacing the two alternating buffers.
- `Scenario::ALL: [Scenario; 3]`, tied to the enum by a wildcard-free `match`
  in a const fn, so a new variant that is not added to `ALL` fails to compile.

Rollout order, by risk: GloVe (new blob, one new kernel), then NYTimes, then
SIFT v4 (most new ops, second random stream). Each lands with its tests before
the next starts. All three land before the resubmit.

## Determinism

`build_ptx` compiles with `--use_fast_math`, which permits approximate `sqrt`,
division and transcendentals. Until now the post-latent generation path used
only `fmaf`, `mul`, `add` and `select`. This design adds, per row: GloVe one
`sqrt` and one division; NYTimes two of each; SIFT one of each plus `tanh`,
`exp`, `log1p` and `log` per coordinate.

- **Verification tolerates small differences.** Quality is recall@1 within a
  relative 1e-6, on 1,000 salted queries. A last-bit change to a row moves its
  distances by far less than a typical first-to-second-neighbour gap.
- **SIFT has a discontinuity.** A gate with `logit + noise` within rounding
  distance of 0 can open on one architecture and close on another, which
  changes the whole row after normalisation. ~~ESTIMATE (unverified): of order
  1–10 coordinates per instance of 89.6 million gate draws (700,000 × 128).~~
  **Correction 2026-09-21 (measured):** on 4,096 rows (524,288 gates), 0 gate
  margins fell within 1e-5 of zero, 5 within 1e-4, 55 within 1e-3; density
  ~0.05 per unit margin per gate. Extrapolated to a full 700,000-row database
  (89.6 million gates), ESTIMATE (extrapolated from the MEASURED density):
  ~9 within 1e-6, ~94 within 1e-5, ~940 within 1e-4 — the 1–10 figure only
  holds for the tightest of these bands, and the cross-GPU disagreement
  window itself is still unmeasured (no second architecture was available).
  See `docs/measurements/2026-09-21-c004-multi-arch-generators.md`, "SIFT
  near-threshold gates". A changed row matters only to an audited query whose
  true nearest neighbour is that row. The plan replaces this estimate with a
  measurement: count draws with `|logit + noise| < 1e-5` in one generated
  instance and state the implied upper bound on recall disagreement between
  two verifiers.
- **NYTimes projected-tangent norm sensitivity.** **Correction 2026-09-21
  (measured), addition — a risk this design did not anticipate:** the
  spherical forward divides by `max(‖x‖, eps)`; if the pre-normalisation norm
  were near the `eps = 1e-8` clamp, rounding error would be amplified. Measured
  over 8,192 generated rows, the minimum projected-tangent norm is 1.499242
  (p0.1 1.714236, p1 1.9771563, p50 3.765099); 0 rows fall below 1e-3 or
  1e-5. With norm ≥ 1.499, the division scales rounding error by at most
  1/1.499 ≈ 0.67x, so the eps clamp does not come into play on generated
  data. See `docs/measurements/2026-09-21-c004-multi-arch-generators.md`,
  "NYTimes projected-tangent norm".
- **Cross-architecture bit-exactness stays open**, as the 2026-08-25 spec left
  it. Validation on the sm_86 box can show launch-geometry invariance and
  in-band results. It cannot show agreement with a second architecture. If an
  sm_89 card is available, one digest comparison per scenario closes it; that
  is an optional task, and this design does not claim it.

## Tests

Fixed seed and nonce lists throughout; nothing wall-clock-derived. After
implementation every new test is mutation-checked: break the code under test,
confirm the test fails, restore.

| # | test | runs on | mutation it catches |
|---|---|---|---|
| 1 | Literal wire strings `s=nytimes_256` and `s=glove_100` serialise and parse. The unknown-scenario tests switch from `glove_300` to `deep_96`. | CPU | renamed variant; changed case convention |
| 2 | `every_scenario_blob_matches_its_declared_dims` iterates `Scenario::ALL`; asserts output dims, asserts latent dim against the blob header, asserts `vector_dims <= AUDIT_MAX_DIMS` (or the wide-kernel bound). | CPU | wrong blob wired to a scenario; variant missing from `ALL` (compile error); 256-dim scenario against a 128-wide audit buffer |
| 3 | `TIGGAN02` rejects: bad magic, truncation, trailing bytes, absurd tensor count, element-count overflow, unknown `arch`, wrong tensor count for the arch, shape-relation mismatch. One test each. | CPU | removal of each check |
| 4 | Per-architecture `forward_cpu` against PyTorch goldens at 1e-5; SIFT support pattern compared exactly. | CPU | transposed matrix; missing bias; `a·γ` in place of `a·(1+γ)`; swapped `cos`/`sin`; projection using un-normalised `u`; missing argmax fallback; trunk-final activation dropped for spherical; baked matrix with wrong padding |
| 5 | Exporter pytest: baked `coupling` and `smoothing` equal the module's own conv output on 64 inputs from `torch.manual_seed(0)`, at 1e-6. | CPU | transposed or interior-only baked map |
| 6 | GPU forward equals `forward_cpu` per architecture on 1,024 rows at 1e-5, latents and noise read back from the device. Includes `out_dim = 1` and `out_dim = 100` through `gan_linear`. | GPU | a new kernel computing the wrong formula; `out_row_offset` mishandled |
| 7 | Launch-invariance FNV-1a digests per scenario across the four existing chunk/block combinations, with the nonce-1 negative control. | GPU | chunk-dependent output, including a reused RNG index |
| 8 | Stream separation on device readback: `z_t != z_s` with max absolute correlation over columns below 0.05 on 65,536 rows; same bound between gate noise and latents. | GPU | the `1 << 30` or `1ULL << 40` offset dropped |
| 9 | Row norms within 1e-4 of 1.0 over a full 700,000-row database, all three scenarios; SIFT rows have no negative coordinate. | GPU | normalisation skipped; misfiring `eps` clamp; gate applied to a signed magnitude |
| 10 | WGAN gate: 20,000 GPU-generated rows per scenario pass `python -m src.eval.check_gate` against `gates/{sift,nytimes,glove}.yaml`. | GPU + WGAN repo | a port that is self-consistent but samples a different distribution from the accepted one |
| 11 | Existing `recall_audit_tests` unchanged on SIFT with the 150 ms gate re-measured; `exact_1nn_measures_recall_1` and `all_zeros_measures_recall_near_0` also run on NYTimes-256 and GloVe-100. | GPU | audit overrun at 256 dims; a dims-dependent audit bug |

The 0.05 correlation bound in test 8: for independent columns the sample
correlation over 65,536 rows has standard deviation `1/√65536 ≈ 0.0039`, so
0.05 is about 13 sigma, and a dropped offset gives correlation 1.0.

Test 10 is the acceptance test for the port. Tests 4 and 6 pin arithmetic; only
test 10 shows that the challenge samples the distribution the gates accepted.
Its known gap is decision 2 above: gates are measured at N = 20,000 and say
nothing about 700,000.

**Correction 2026-09-21 (measured):** test 10's outcome is MET for sift and
glove (pass on all four statistics) and NOT MET for nytimes (`ivf_gini`
0.7246, below the band [0.7602, 0.8403]). The investigation shows this is not
a port bug: the gate's own recorded sample and five fresh PyTorch draws of
the same checkpoint also fail `ivf_gini` at rates consistent with the port's
result (mean 0.7440, sd 0.0180 over six draws including the gate's own; only
1 of 6 clears the band). The port is accepted on the stated alternative
criterion — its statistics lie inside the range of PyTorch's own samples of
the same checkpoint — not on the plan's original all-pass criterion. See
`docs/measurements/2026-09-21-c004-multi-arch-generators.md`, "WGAN gate
acceptance" and "The NYTimes investigation".

GPU tests run on tig-gpu under a `gpu-claim`, through the job queue.

## Migration

One network-wide resubmit, the same one the 250→128 change already requires.
It now carries three tracks. The inherited obligation on algorithm authors is
to read `challenge.vector_dims` at runtime. What is new: dims now differ within
one deployment (100, 128, 256), so a hardcoded `128` passes the SIFT track and
fails the other two.

## Follow-ups, not in this change

- `_MIN_RECALL_BY_TRACK` in `tig-pentesting`'s `challenges/vector_search.py`
  needs entries for `nytimes_256` and `glove_100` equal to the values here. The
  existing comment in `scenarios.rs` explains what goes wrong when the two
  repos disagree.
- The protocol config must list the three track ids.
- A per-corpus d2/d1 measurement to replace the placeholder `min_recall = 0.9`
  for NYTimes and GloVe.
- `calc_build_fuel_budget` constants were measured at 128 dims. Behaviour at
  100 and 256 dims is unchecked.
- Qualifier-slot economics, still deferred from the 2026-08-25 spec; three
  tracks changes the arithmetic there but this design does not address it.
- The DEEP-96 milestone from the 2026-08-25 spec is not part of this change.
  DEEP's accepted rung would need the same check made here: which
  `generator_type` it uses.

## Risks

- **The kernel set is larger, so the re-frozen ABI is harder to keep frozen.**
  A future architecture that needs an op outside these five kernels forces
  another resubmit. Drivers are host-side, so an architecture composed of the
  existing kernels is still a runtime-only change.
- **SIFT gate discontinuity across architectures**, as under Determinism.
  Unmeasured until the plan's count is run.
- **`v3_best` rests on one training seed**, and `docs/datasets/nytimes.md`
  calls its selected step a transient. That is a property of the generator,
  accepted upstream by the gate owner; this design ships it as accepted and
  does not re-judge it.
- **Audit timing for SIFT may regress** if `AUDIT_MAX_DIMS` is raised; handled
  by the measured fallback above.
- ~~**Generation time is unmeasured for the new architectures.**~~
  **Correction 2026-09-21 (measured):** `Database::generate`, 700,000 rows,
  three runs each: sift_128 885/812/811 ms, glove_100 680/669/666 ms,
  nytimes_256 1585/1584/1581 ms. SIFT/GloVe = 1.22x (median/median), under
  the plan's 2x threshold for the gate-noise cost decision, so the curand
  sequence base `2^40` stays and the fallback scheme was not needed. See
  `docs/measurements/2026-09-21-c004-multi-arch-generators.md`, "Generation
  time". SIFT v1 took 628 ms at this size (2026-08-25 spec, not re-measured
  here, and not comparable: different card). NYTimes has
  1.85× the floats of the v1 MLP (3,281,152 against 1,772,171) and SIFT v4 adds two 128×128 products and
  a transcendental pass per coordinate. The plan measures all three.

## Validation

Measured 2026-09-21 on tig-gpu, RTX 3060 Ti (sm_86). Full detail:
`docs/measurements/2026-09-21-c004-multi-arch-generators.md`.

| claim | value | MEASURED / ESTIMATED |
|---|---|---|
| WGAN gate, sift | pass on all four statistics | MEASURED |
| WGAN gate, glove | pass on all four statistics | MEASURED |
| WGAN gate, nytimes | FAIL on `ivf_gini` (0.7246, band [0.7602, 0.8403]) | MEASURED |
| nytimes port vs. five fresh PyTorch draws of the same checkpoint | port's four statistics fall inside the range of the five draws on every statistic | MEASURED |
| generation time, sift/glove/nytimes (median of 3, 700,000 rows) | 812 / 669 / 1584 ms | MEASURED |
| SIFT gate-margin density near the threshold | ~0.05 per unit margin per gate, over 524,288 gates | MEASURED |
| NYTimes projected-tangent norm, minimum over 8,192 rows | 1.499242 | MEASURED |

### What this establishes

Launch-geometry invariance and in-band results on sm_86, for all three
architectures (test counts, mutation checks, output-digest sanity — see the
measurement note). Generation time for all three architectures. That the
SIFT gate-margin discontinuity is rare and its per-database rate is
estimated from a measured density. That the NYTimes projected-tangent
sensitivity the spec did not originally anticipate does not occur on
generated data. That SIFT and GloVe pass their WGAN gates on all four
statistics. That the NYTimes port samples the same distribution as the
PyTorch generator as far as the four gate statistics can tell, even though
neither the port nor five fresh PyTorch draws of the same checkpoint
robustly clears the gate's `ivf_gini` band.

### What this does NOT establish

Cross-architecture bit-exactness (one card only; no second architecture was
available). Gate statistics at N = 700,000 (the gates are defined and
measured at N = 20,000). `min_recall` for the two new corpora (copied from
SIFT as a placeholder) and for SIFT on v4 instances (the 0.9 bar was derived
on v1 instances). `AUDIT_TQ` optimality at 256. Timings on non-Ampere cards.
Fuel budgets at 100 and 256 dims.
