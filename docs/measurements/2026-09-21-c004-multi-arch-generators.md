# c004 multi-architecture GAN generators: acceptance against the WGAN gates

**Date:** 2026-09-21
**Measures:** `docs/ai/specs/2026-09-21-multi-arch-gan-scenarios-design.md`
**Code under test:** branch `vector_search/gan_instance_gen`, commits `9f9aa9d`
through `16753ee` (see the test-count table below for the commits in between).

GPU work ran on `tig-gpu`, NVIDIA GeForce RTX 3060 Ti, compute capability 8.6,
8192 MiB, CUDA 12.8, rustc `nightly-2025-02-10` (1.86.0-nightly 124cc9219
2025-02-09), PTX built `-ptx -arch compute_70 -code sm_70 --use_fast_math
-dopt=on`, through the `gpuq` job queue, whole card each job. Gate checks ran
locally in `~/TIG/wgan-synthetic` at commit `5b96976`, in its `.venv` (torch
2.13.0 CPU). SIFT and GloVe pass their WGAN gates on all four statistics.
NYTimes does not, on one statistic (`ivf_gini`). The investigation below shows
this is a property of the NYTimes v3 generator and its gate band, not of the
Rust/CUDA port: under the same measurement, the port's four statistics sit
inside the range of five fresh PyTorch draws of the same checkpoint.

## Claim table

| claim | value | MEASURED / ESTIMATED | command or file that produced it | when |
|---|---|---|---|---|
| test count, baseline before any change (`9f9aa9d`) | 44 / 0 passed/failed, wall 128.70 s, audit 87 ms | MEASURED | `scripts/box_submit.sh <name> gpu`, whole suite, queue `.out` log | 2026-09-21 |
| test count, after the driver refactor (`aa2f0dc`, Task 6 A) | 84 / 0, wall 132.05 s, audit 87 ms | MEASURED | same | 2026-09-21 |
| test count, + GloVe scenario (`3df8cb2`) | 92 / 0, wall 194.94 s, audit 87 ms | MEASURED | same | 2026-09-21 |
| test count, + NYTimes scenario (`7f659f0`) | 99 / 0, wall 366.23 s, audit 87 ms | MEASURED | same | 2026-09-21 |
| test count, + structured-gate driver, SIFT on v4 (`e5ff819`) | 105 / 0, wall 450.60 s, audit 88 ms | MEASURED | same | 2026-09-21 |
| test count, + clamp test (`4fcd5af`) | 106 / 0, wall 446.98 s, audit 87 ms | MEASURED | same | 2026-09-21 |
| compiler warnings, `3df8cb2` through `4fcd5af` | 0 in every run | MEASURED | queue `.out` logs | 2026-09-21 |
| local ungated suite at `16753ee`, `tig-challenges` | 60 passed | MEASURED | `cargo test -p tig-challenges` | 2026-09-21 |
| local ungated suite at `16753ee`, `gan_generator` | 50 passed | MEASURED | `cargo test` (module filter `gan_generator`) | 2026-09-21 |
| audit ms, `AUDIT_MAX_DIMS=128`, best-of-three, SIFT_128 workload | 87, 87, 87 ms | MEASURED | `audit_is_much_cheaper_than_a_naive_solve`, three whole-suite jobs `6c03f4`, `6041bf`, `4f95a5` | 2026-09-21 |
| audit ms, `AUDIT_MAX_DIMS=256`, best-of-three | 88, 88, 87 ms | MEASURED | same test, three jobs filtered to `recall_audit_tests` at `fc45511` | 2026-09-21 |
| audit gate | 150 ms | MEASURED (existing gate, unchanged) | `audit_is_much_cheaper_than_a_naive_solve` | 2026-09-21 |
| shared memory at `AUDIT_MAX_DIMS=128` | 27,720 B (`s_query` 9,216 + `s_db` 17,408 + `s_red` 1,024 + `s_returned` 72) | MEASURED (equals the kernel comment's own figure) | kernel comment, `kernels.cu` | 2026-09-21 |
| shared memory at `AUDIT_MAX_DIMS=256` | 36,936 B | computed (`s_query` 18×256×4 = 18,432, other three arrays unchanged at 18,504; 18,432 + 18,504 = 36,936) | four `__shared__` declarations in `kernels.cu` | 2026-09-21 |
| generation time, `sift_128` (v4 structured_gate), 700,000 rows, three runs | 885, 812, 811 ms | MEASURED | `print_generation_times`, `Database::generate`, at `16753ee` | 2026-09-21 |
| generation time, `glove_100` (mlp), three runs | 680, 669, 666 ms | MEASURED | same | 2026-09-21 |
| generation time, `nytimes_256` (spherical), three runs | 1585, 1584, 1581 ms | MEASURED | same | 2026-09-21 |
| SIFT/GloVe generation-time ratio (median 812 / median 669) | 1.22x | computed | arithmetic on the two rows above | 2026-09-21 |
| SIFT gate margins near threshold, rows=4096, gates=524,288, CPU reference on GPU-drawn latents/noise | \|logit + smoothed noise\| < 1e-5: 0; < 1e-4: 5; < 1e-3: 55 | MEASURED | `count_sift_gates_near_the_threshold` at `16753ee` | 2026-09-21 |
| gate-margin density, < 1e-3 band | 0.052 per unit margin per gate (55 / (2e-3 × 524,288) = 0.05245) | computed | arithmetic on the row above | 2026-09-21 |
| gate-margin density, < 1e-4 band | 0.048 per unit margin per gate (5 / (2e-4 × 524,288) = 0.04768) | computed | arithmetic on the row above | 2026-09-21 |
| gates in a full database | 89.6 million (700,000 × 128) | computed | arithmetic | 2026-09-21 |
| near-threshold gates per database, extrapolated | ~9 within 1e-6, ~94 within 1e-5, ~940 within 1e-4 | ESTIMATE (extrapolated from the MEASURED density above) | facts file section 4; cross-GPU disagreement window itself is NOT measured | 2026-09-21 |
| NYTimes projected-tangent norm, rows=8192 | min 1.499242, p0.1 1.714236, p1 1.9771563, p50 3.765099; below 1e-3: 0; below 1e-5: 0 | MEASURED | `measure_nytimes_projected_tangent_norms` at `16753ee` | 2026-09-21 |
| rounding-error amplification bound from the min norm | at most 0.67x (1 / 1.499242 ≈ 0.667) | computed | arithmetic: the spherical forward divides by `max(‖x‖, eps)`; with norm ≥ 1.499 the division scales rounding error by at most 1/1.499 | 2026-09-21 |
| output digest, `sift_128`, 50,000 rows | `51df86ed2098fc4f` (first 1,000 rows: `4cfb568534ce9c28`) | MEASURED | `dump_rows_for_the_wgan_gates` at `16753ee`, seed `[42u8;32]`, chunk 65,536, block 256, FNV-1a over little-endian f32 bytes | 2026-09-21 |
| output digest, `glove_100`, 50,000 rows | `1173c3a0f4b57438` (first 1,000: `ff383e83f2915ab3`) | MEASURED | same | 2026-09-21 |
| output digest, `nytimes_256`, 50,000 rows | `e2ba765706561abe` (first 1,000: `ebf2344ec89adef1`) | MEASURED | same | 2026-09-21 |
| dump sanity: finite rows, all three | 50,000 / 50,000 finite | MEASURED | numpy on the box | 2026-09-21 |
| dump sanity: row norms, all three | within 1e-6 of 1 | MEASURED | numpy on the box | 2026-09-21 |
| SIFT exact-zero fraction in the dump | 0.2390 (WGAN doc v4: 0.239; real SIFT: 0.230) | MEASURED | numpy on the box | 2026-09-21 |
| SIFT dump min value | 0 | MEASURED | numpy on the box | 2026-09-21 |
| NYTimes/GloVe dump zero fraction | 0.0000 (both have negative coordinates) | MEASURED | numpy on the box | 2026-09-21 |
| sha256 of the three `.npy` dumps, box vs. local after tar stream | glove `d00c0593…`, nytimes `4f528d1e…`, sift `59341553…` (matched) | MEASURED | `sha256sum` on box and locally | 2026-09-21 |
| mutation checks, Task 6 (GloVe/mlp) | 3 / 3 caught | MEASURED | throwaway-branch mutation runs, deleted afterwards | 2026-09-21 |
| mutation checks, Task 8 (NYTimes/spherical) | 4 / 4 caught | MEASURED | same | 2026-09-21 |
| mutation checks, Task 9 (SIFT/structured_gate) | 8 / 8 caught after the fix (upper-clamp mutation not caught at `e5ff819`, caught at `4fcd5af`) | MEASURED | same | 2026-09-21 |
| gate statistics, `sift`, verdict pass | lid_median 16.4328, relative_contrast_median 2.3217, hubness_skew 1.7991, ivf_gini 0.2853 | MEASURED | `eda_report` + `check_gate --dataset sift`, `--stats-name tig` | 2026-09-21 |
| gate statistics, `glove`, verdict pass | lid_median 36.1402, relative_contrast_median 1.3758, hubness_skew 4.5502, ivf_gini 0.5851 | MEASURED | `eda_report` + `check_gate --dataset glove`, `--stats-name tig` | 2026-09-21 |
| gate statistics, `nytimes`, verdict FAIL on ivf_gini | lid_median 55.6434, relative_contrast_median 1.2175, hubness_skew 3.4091, ivf_gini 0.7246 | MEASURED | `eda_report` + `check_gate --dataset nytimes`, `--stats-name tig` | 2026-09-21 |
| NYTimes gate bands | lid [53.2123, 58.8137], contrast [1.2060, 1.3330], hubness [2.0057, 3.5101], ivf_gini [0.7602, 0.8403] | MEASURED (from `gates/nytimes.yaml`) | `gates/nytimes.yaml` | 2026-09-21 |
| NYTimes gate's own recorded `v3_best` statistics | lid 55.6884, contrast 1.2183, hubness 3.3920, gini 0.7704 | MEASURED (recorded in the gate file) | `gates/nytimes.yaml` | 2026-09-21 |
| six-series `eda_report` run: torch seed 42 | lid 55.8164, contrast 1.2165, hubness 3.2490, gini 0.7511, fail | MEASURED | `eda_report` over PyTorch CPU samples of checkpoint sha256 `b1acfdda…`, 50,000 rows via `src.sample.generate`, seed 42 | 2026-09-21 |
| six-series run: torch seed 1 | lid 55.6271, contrast 1.2174, hubness 3.5301 (out of band), gini 0.7409, fail | MEASURED | same, seed 1 | 2026-09-21 |
| six-series run: torch seed 2 | lid 55.3722, contrast 1.2176, hubness 3.3807, gini 0.7250, fail | MEASURED | same, seed 2 | 2026-09-21 |
| six-series run: torch seed 3 | lid 56.2370, contrast 1.2155, hubness 3.1361, gini 0.7234, fail | MEASURED | same, seed 3 | 2026-09-21 |
| six-series run: torch seed 4 | lid 55.6307, contrast 1.2174, hubness 3.3608, gini 0.7530, fail | MEASURED | same, seed 4 | 2026-09-21 |
| six-series run: tig (GPU port) | lid 55.6434, contrast 1.2175, hubness 3.4091, gini 0.7246, fail | MEASURED | same run as the main gate table above | 2026-09-21 |
| reproduction of the gate's own sample file, measured here | lid 55.6884, contrast 1.2183, hubness 3.3920, gini 0.7704 — identical to the gate file's recorded values | MEASURED | `eda_report` + `check_gate` on `/workspace/nytimes-v3/v3_seed42/synthetic_50k_best.npy`, sha256 `0c06596a…`, hash-verified after transfer | 2026-09-21 |
| six PyTorch `ivf_gini` draws, in the order measured | 0.7704, 0.7530, 0.7511, 0.7409, 0.7250, 0.7234 | MEASURED | the two tables above, `ivf_gini` column | 2026-09-21 |
| mean of the six draws | 0.7440 | computed (statistics.mean of the row above) | arithmetic | 2026-09-21 |
| sd of the six draws | 0.0180 | computed (statistics.stdev, sample sd, of the row above) | arithmetic | 2026-09-21 |
| draws clearing the lower gate edge (0.7602) | 1 of 6 (the gate's own sample, 0.7704) | MEASURED (count over the row above) | arithmetic on the six-draw row | 2026-09-21 |
| `v3_best` admission margin over the lower edge | 0.0102 (0.7704 − 0.7602) | computed | arithmetic | 2026-09-21 |
| blob size, `glove_100_v1.bin` | computed 6,973,840 + header; measured 6,973,936 B (header 96 B) | MEASURED (`ls -l`) / computed (spec's float-count arithmetic) | `ls -l` on the box; spec's layer-shape arithmetic | 2026-09-21 |
| blob size, `sift_128_v4.bin` | computed 7,748,612 + header; measured 7,748,764 B (header 152 B) | MEASURED / computed | same | 2026-09-21 |
| blob size, `nytimes_256_v3.bin` | computed 13,124,608 + header; measured 13,124,768 B (header 160 B) | MEASURED / computed | same | 2026-09-21 |

## Test counts

| commit | what | passed / failed | wall | audit ms |
|---|---|---|---|---|
| `9f9aa9d` | baseline before any change | 44 / 0 | 128.70 s | 87 |
| `aa2f0dc` | after the driver refactor (Task 6 A) | 84 / 0 | 132.05 s | 87 |
| `3df8cb2` | + GloVe scenario | 92 / 0 | 194.94 s | 87 |
| `7f659f0` | + NYTimes scenario | 99 / 0 | 366.23 s | 87 |
| `e5ff819` | + structured-gate driver, SIFT on v4 | 105 / 0 | 450.60 s | 88 |
| `4fcd5af` | + clamp test | 106 / 0 | 446.98 s | 87 |

Zero compiler warnings in every run from `3df8cb2` on. Local ungated suite at
`16753ee`: `cargo test -p tig-challenges` 60 passed; `gan_generator` 50 passed.

## Audit width

`audit_is_much_cheaper_than_a_naive_solve`, best-of-three as printed, SIFT_128
workload.

| `AUDIT_MAX_DIMS` | run 1 | run 2 | run 3 | gate |
|---|---|---|---|---|
| 128 | 87 ms | 87 ms | 87 ms | 150 ms |
| 256 | 88 ms | 88 ms | 87 ms | 150 ms |

128 ran as jobs `6c03f4`, `6041bf`, `4f95a5` (whole suite). 256 ran as three
jobs filtered to `recall_audit_tests` at `fc45511`.

Shared memory: 27,720 B at 128 (`s_query` 9,216 + `s_db` 17,408 + `s_red`
1,024 + `s_returned` 72 — this equals the kernel comment's own figure) and
36,936 B at 256 (computed the same way, with `s_query` at 18×256×4 = 18,432).

## Generation time

`print_generation_times` at `16753ee`, `Database::generate`, 700,000 rows,
synchronised, three runs each, ms per run.

| scenario | architecture | run 0 | run 1 | run 2 |
|---|---|---|---|---|
| `sift_128` | v4 structured_gate | 885 | 812 | 811 |
| `glove_100` | mlp | 680 | 669 | 666 |
| `nytimes_256` | spherical | 1585 | 1584 | 1581 |

Run 0 includes weight upload and kernel load. SIFT/GloVe = 1.22x (median 812
over median 669), under the plan's 2x rule, so the gate-noise curand sequence
base `2^40` stays; the plan's fallback (a distinct seed word with the plain
`global_i` sequence) was not needed.

The 2026-08-25 spec's 628 ms figure for SIFT v1 was measured on a different
card and is not comparable to any number in this table.

## SIFT near-threshold gates

`count_sift_gates_near_the_threshold` at `16753ee`, rows=4096, gates=524,288,
CPU reference on GPU-drawn latents and noise.

| margin | count |
|---|---|
| \|logit + smoothed noise\| < 1e-5 | 0 |
| < 1e-4 | 5 |
| < 1e-3 | 55 |

Density: 55 / (2e-3 × 524,288) = 0.052 per unit margin per gate from the
< 1e-3 band; 5 / (2e-4 × 524,288) = 0.048 from the < 1e-4 band. A full
database has 700,000 × 128 = 89.6 million gates. ESTIMATE (extrapolated from
the MEASURED density above): ~9 gates within 1e-6, ~94 within 1e-5, ~940
within 1e-4 per database. NOT measured: how wide the cross-GPU disagreement
window actually is — no second architecture was available.

## NYTimes projected-tangent norm

`measure_nytimes_projected_tangent_norms` at `16753ee`, rows=8192.

| statistic | value |
|---|---|
| min | 1.499242 |
| p0.1 | 1.714236 |
| p1 | 1.9771563 |
| p50 | 3.765099 |
| fraction below 1e-3 | 0 |
| fraction below 1e-5 | 0 |

The spherical forward divides by `max(‖x‖, eps)`; with the measured norm at
or above 1.499, the division scales rounding error by at most 1/1.499 ≈ 0.67x
(computed). The eps clamp, and the error amplification it would imply if the
norm were near zero, never come into play on generated data.

## Output digests

`dump_rows_for_the_wgan_gates` at `16753ee`. Seed `[42u8;32]`, 50,000 rows,
chunk 65,536, block 256, 64-bit FNV-1a over little-endian f32 bytes, on the
RTX 3060 Ti (sm_86). Recorded, NOT asserted: under `--use_fast_math` bytes
may differ across GPU architectures.

| scenario | digest, 50,000 rows | digest, first 1,000 rows |
|---|---|---|
| sift_128 | `51df86ed2098fc4f` | `4cfb568534ce9c28` |
| glove_100 | `1173c3a0f4b57438` | `ff383e83f2915ab3` |
| nytimes_256 | `e2ba765706561abe` | `ebf2344ec89adef1` |

Dump sanity (numpy on the box): all 50,000 rows finite, all three; row norms
within 1e-6 of 1, all three; SIFT exact-zero fraction 0.2390 (WGAN doc: v4
0.239, real SIFT 0.230), SIFT min value 0; NYTimes and GloVe have negative
coordinates and zero fraction 0.0000.

sha256 of the `.npy` files, matched between box and local after a tar
stream: glove `d00c0593…`, nytimes `4f528d1e…`, sift `59341553…`.

## Mutation checks

Each on a throwaway branch, deleted afterwards. "Caught" means the named
test failed.

| task | architecture | mutations tried | caught |
|---|---|---|---|
| Task 6 | GloVe/mlp | normalize ignores `row_offset`; last linear at row 0; no divide | 3 / 3 |
| Task 8 | NYTimes/spherical | `z_skip` at the same curand index; cos/sin swapped; projection skipped; gamma for `1+gamma` | 4 / 4 |
| Task 9 | SIFT/structured_gate | sequence base dropped; threshold inverted; `sparsity[0]`; coupling/smoothing swapped; softplus branch removed; last-maximum argmax; upper `u` clamp removed; lower `u` clamp removed | 8 / 8 (after the fix) |

Detail on Task 6: `normalize ignores row_offset` was caught by both
`unit_norm` and `invariance` (row 65,536 norm 0.534; 3,950,881 values
disagreed); `last linear at row 0` was caught by `unit_norm` and
`invariance`; `no divide` was caught by `gpu_forward` (gpu −0.04766, cpu
−0.05582) and `unit_norm`.

Detail on Task 8: `z_skip` at the same curand index gives independence
`r = 1`, and `gpu_forward` PASSES under it, because the CPU reference is fed
the GPU's own latents — so `independence` is the only guard for that
mutation. The other three mutations were each caught by `gpu_forward` (cos/sin
swap), `gpu_forward` + `unit_norm` at norm 1.198 (projection skipped), and
`gpu_forward` (gamma).

Detail on Task 9: sequence base dropped was caught by `does_not_reuse`
(524,288 of 524,288 draws equal); threshold inverted was caught by zero
fraction 0.7607 plus `gpu_forward` and `synthetic`; `sparsity[0]` was caught
by `gpu_forward` and `invariance`; coupling/smoothing swapped was caught by
`synthetic` and `gpu_forward`; softplus branch removed was caught ONLY by
`synthetic` (gpu NaN, cpu 1); last-maximum argmax was caught ONLY by
`synthetic` (gpu 0, cpu 1); upper `u` clamp removed was NOT caught at
`e5ff819` (10/10 green including the 89.6M-draw test), and was caught after
the fix at `4fcd5af` (output `[inf, 16.635532, …]`); lower `u` clamp removed
was caught at `4fcd5af` (`[…, -inf, -69.077545]`).

## WGAN gate acceptance

Conditions for all runs, taken from the committed summaries the gates were
set from: preprocess l2, seed 42, ann k 100, k_hub 10, max_rows 20000,
nlist 256; metric l2 (sift, real data `sift_1m.npy`) / angular (nytimes,
real data `nytimes_250k_l2_clean.npy`; glove, real data `glove_250k.npy`).
`check_gate` reported `conditions_match True` everywhere; `--allow-condition-mismatch`
was never used. Synthetic input: the 50,000-row GPU dumps from the Output
digests section above.

| dataset | lid_median | relative_contrast_median | hubness_skew | ivf_gini | verdict |
|---|---|---|---|---|---|
| sift | 16.4328 | 2.3217 | 1.7991 | 0.2853 | pass |
| glove | 36.1402 | 1.3758 | 4.5502 | 0.5851 | pass |
| nytimes | 55.6434 | 1.2175 | 3.4091 | 0.7246 | FAIL (ivf_gini below band) |

NYTimes bands: lid [53.2123, 58.8137], contrast [1.2060, 1.3330], hubness
[2.0057, 3.5101], ivf_gini [0.7602, 0.8403]. `gates/nytimes.yaml` records
`v3_best` at lid 55.6884, contrast 1.2183, hubness 3.3920, gini 0.7704.

### The NYTimes investigation

(a) One `eda_report` run, six series: PyTorch CPU samples of the SAME
checkpoint (sha256 `b1acfdda…`, 50,000 rows each via `src.sample.generate`) at
sampling seeds 42, 1, 2, 3, 4, plus the GPU dump `tig`.

| series | lid | contrast | hubness | ivf_gini | verdict |
|---|---|---|---|---|---|
| torch seed 42 | 55.8164 | 1.2165 | 3.2490 | 0.7511 | fail |
| torch seed 1 | 55.6271 | 1.2174 | 3.5301 (out) | 0.7409 | fail |
| torch seed 2 | 55.3722 | 1.2176 | 3.3807 | 0.7250 | fail |
| torch seed 3 | 56.2370 | 1.2155 | 3.1361 | 0.7234 | fail |
| torch seed 4 | 55.6307 | 1.2174 | 3.3608 | 0.7530 | fail |
| tig (GPU port) | 55.6434 | 1.2175 | 3.4091 | 0.7246 | fail |

(b) The gate's ORIGINAL sample file
(`/workspace/nytimes-v3/v3_seed42/synthetic_50k_best.npy`, sha256
`0c06596a5f51d236319ea39434ca62c962699cac2119cacba6b1e1f031b14ee3`,
hash-verified after transfer), measured here in one run with `tig`: lid
55.6884, contrast 1.2183, hubness 3.3920, gini 0.7704 — identical to the
gate file's recorded values, so this environment reproduces the gate's
measurement with no offset.

(c) Six PyTorch draws measured identically: 0.7704, 0.7530, 0.7511, 0.7409,
0.7250, 0.7234. Mean 0.7440, sd 0.0180 (computed). One of six clears the
lower edge 0.7602. `v3_best`'s admission margin over that edge was 0.0102.
"Seed 42" on CPU is not the box's CUDA seed-42 draw: `torch.randn` differs by
device.

Conclusions:

1. The port samples the same distribution as the PyTorch generator as far as
   these four statistics can tell: `tig` is inside the range of five fresh
   PyTorch draws on every statistic.
2. The NYTimes v3 generator does not robustly pass its own gate on
   `ivf_gini`. That is a property of the generator and its gate, upstream of
   this repo.

### Acceptance outcome

The plan's acceptance criterion ("pass on all four statistics for all three
datasets") is MET for sift and glove and NOT MET for nytimes. The port is
accepted on a different, stated criterion: under identical measurement, its
statistics lie inside the range of PyTorch's own samples of the same
checkpoint across sampling seeds. The decision about the NYTimes generator
and its gate band belongs to the WGAN repository's owner, not to this repo.

## Blob sizes (weights)

Computed sizes are the spec's arithmetic from layer shapes (float count × 4);
measured sizes are `ls -l` on the box. The header sizes account for the
whole difference.

| blob | computed (floats × 4) | measured (`ls -l`) | header |
|---|---|---|---|
| `glove_100_v1.bin` | 6,973,840 | 6,973,936 | 96 B |
| `sift_128_v4.bin` | 7,748,612 | 7,748,764 | 152 B |
| `nytimes_256_v3.bin` | 13,124,608 | 13,124,768 | 160 B |

## What was not established

Cross-architecture bit-exactness (one card only). The gate statistics at
N = 700,000 (gates are defined at N = 20,000). `min_recall` for nytimes/glove
(copied from SIFT) and for SIFT on v4 instances (the 0.9 bar was derived on
v1 instances). `AUDIT_TQ` optimality at 256. Any timing on a non-Ampere card.
Fuel budgets at 100 and 256 dims.

## How to reproduce

The four `#[ignore]`d tests, each in its own job so a failure in one does not
hide the others:

```bash
for t in dump_rows_for_the_wgan_gates print_generation_times count_sift_gates_near_the_threshold measure_nytimes_projected_tangent_norms; do
  BOX_ENV="TIG_TEST_EXTRA=--ignored" scripts/box_submit.sh "task10-$t" gpu "$t"
done
```

`dump_rows_for_the_wgan_gates` writes to `TIG_DUMP_DIR` (exported by
`scripts/box_test.sh`), one `.f32` file per scenario, raw little-endian f32,
50,000 rows.

Convert on the box:

```bash
ssh tig-gpu 'cd /workspace/tig-dumps && /venv/main/bin/python - <<PY
import numpy as np
for name, dims in [("sift_128", 128), ("nytimes_256", 256), ("glove_100", 100)]:
    a = np.fromfile(f"{name}.f32", dtype="<f4").reshape(-1, dims)
    assert a.shape[0] == 50000 and np.isfinite(a).all(), (name, a.shape)
    np.save(f"{name}.npy", a)
    print(name, a.shape, "norm min/max", np.linalg.norm(a, axis=1).min(), np.linalg.norm(a, axis=1).max())
PY'
```

Stream the three `.npy` files off the box by the standing transfer rule
(`ssh tig-gpu 'cd /workspace/tig-dumps && sha256sum *.npy'`, then
`ssh tig-gpu 'cd /workspace/tig-dumps && tar czf - sift_128.npy nytimes_256.npy glove_100.npy' | tar xzf - -C <scratch>`,
then compare `sha256sum` locally). Then, per dataset, in
`~/TIG/wgan-synthetic` with its `.venv`:

```bash
python -m src.eval.eda_report --real-path data/glove_250k.npy \
    --synthetic-path tig=/workspace/tig-dumps/glove_100.npy \
    --output-dir runs/glove/tig_port --ann-max-rows 20000 --ann-k 100 --ann-hub-k 10
python -m src.eval.check_gate --dataset glove --run-dir runs/glove/tig_port --stats-name tig
```

with `data/sift_1m.npy` / `sift_128.npy` / `--dataset sift` and
`data/nytimes_250k_l2_clean.npy` / `nytimes_256.npy` / `--dataset nytimes`
substituted for the other two. Do not pass `--allow-condition-mismatch`:
`check_gate` compares the run's N, k and nlist against the conditions pinned
in the gate file (`n: 20000`, `k: 100`, `nlist: 256` in all three), and that
refusal is the guard against measuring under the wrong conditions.
