# c004 index-build fuel per track, measured through `tig-runtime build-index`

**Date:** 2026-09-23
**Hardware:** `tig-gpu`, NVIDIA GeForce RTX 3060 Ti (8 GiB), CUDA 12.8, nightly-2025-02-10
**Code:** branch `c004/build-fuel-probe` at `2b683c65`; gpuq job
`tig-monorepo-20260923T093506Z-0faa80`; results in `/workspace/tig-runs/buildfuel-20260923b/`
on the box (`fuel.tsv` sha256 `228f783b…`).
**Seed:** `buildfuel1`, operator-chosen. **Every number below is MEASURED** in that job.

## What was run

`tig-algorithms/src/vector_search/ivf_kmeans` is an IVF-Flat on the index-build
ABI whose build is the build half of a cuVS IVF-Flat index: strided initial
centroids, `n_iters` Lloyd iterations over the first `train_fraction` of the
database, one assignment pass over the whole database, CSR lists. It was
compiled with the fuel-instrumented toolchain (`INDEX_BUILD=1 build_so`,
`build_ptx`) and driven by `scripts/box_build_fuel.sh`: for each track and
configuration, `tig-runtime build-index --build-fuel 1e14` (unbounded for this
purpose), then one nonce solved against the blob at the mainnet `--fuel 5e12`
and checked by `tig-verifier`. `gpu fuel` and `cpu fuel` are the two numbers
the runtime prints on stderr; `recall` is the verifier's `quality / 1e6`.

The proposed limits being checked: sift_128 **9e11**, glove_100 **11e11**,
nytimes_256 **3e11**.

## Results

`x limit` is `(gpu + cpu fuel) / proposed limit`. n_probe only affects the solve.

| track | n_lists | n_iters | train | n_probe | gpu fuel | cpu fuel | x limit | fits | build ms | solve fuel | recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sift_128 | 256 | 0 | 0.5 | 8 | 1.714e+10 | 9.30e+06 | 0.02 | yes | 2466 | 6.05e+10 | 0.942 |
| sift_128 | 256 | 20 | 0.5 | 8 | 1.920e+11 | 9.36e+06 | 0.21 | yes | 2720 | 4.33e+10 | 0.957 |
| sift_128 | 1024 | 0 | 0.5 | 32 | 6.729e+10 | 1.02e+07 | 0.07 | yes | 2009 | 6.20e+10 | 0.983 |
| sift_128 | 1024 | 10 | 0.5 | 32 | 4.055e+11 | 1.02e+07 | 0.45 | yes | 3837 | 4.48e+10 | 0.989 |
| sift_128 | 1024 | 20 | 0.5 | 32 | 7.437e+11 | 1.02e+07 | 0.83 | yes | 5660 | 4.47e+10 | 0.991 |
| sift_128 | 1024 | 20 | 1.0 | 32 | 1.420e+12 | 1.02e+07 | 1.58 | NO | 9288 | 4.47e+10 | 0.991 |
| sift_128 | 2048 | 20 | 0.5 | 64 | 1.479e+12 | 1.14e+07 | 1.64 | NO | 9604 | 4.65e+10 | 0.993 |
| sift_128 | 4096 | 0 | 0.5 | 128 | 2.679e+11 | 1.38e+07 | 0.30 | yes | 3042 | 6.50e+10 | 0.994 |
| sift_128 | 4096 | 20 | 0.5 | 128 | 2.950e+12 | 1.38e+07 | 3.28 | NO | 17400 | 5.14e+10 | 1.000 |
| glove_100 | 256 | 0 | 0.5 | 8 | 1.619e+10 | 1.10e+07 | 0.01 | yes | 1684 | 8.33e+10 | 0.626 |
| glove_100 | 256 | 20 | 0.5 | 8 | 1.814e+11 | 1.11e+07 | 0.16 | yes | 2743 | 4.30e+10 | 0.692 |
| glove_100 | 1024 | 0 | 0.5 | 32 | 6.347e+10 | 1.17e+07 | 0.06 | yes | 1949 | 7.92e+10 | 0.701 |
| glove_100 | 1024 | 10 | 0.5 | 32 | 3.825e+11 | 1.18e+07 | 0.35 | yes | 3728 | 4.39e+10 | 0.793 |
| glove_100 | 1024 | 20 | 0.5 | 32 | 7.015e+11 | 1.18e+07 | 0.64 | yes | 5464 | 4.38e+10 | 0.788 |
| glove_100 | 1024 | 20 | 1.0 | 32 | 1.339e+12 | 1.18e+07 | 1.22 | NO | 8951 | 4.37e+10 | 0.781 |
| glove_100 | 2048 | 20 | 0.5 | 64 | 1.395e+12 | 1.27e+07 | 1.27 | NO | 9222 | 4.57e+10 | 0.823 |
| glove_100 | 4096 | 0 | 0.5 | 128 | 2.526e+11 | 1.45e+07 | 0.23 | yes | 2923 | 8.35e+10 | 0.810 |
| glove_100 | 4096 | 20 | 0.5 | 128 | 2.782e+12 | 1.46e+07 | 2.53 | NO | 16659 | 5.06e+10 | 0.850 |
| nytimes_256 | 256 | 0 | 0.5 | 8 | 1.014e+10 | 3.30e+06 | 0.03 | yes | 1233 | 2.16e+10 | 0.501 |
| nytimes_256 | 256 | 20 | 0.5 | 8 | 1.136e+11 | 3.35e+06 | 0.38 | yes | 1839 | 1.81e+10 | 0.615 |
| nytimes_256 | 1024 | 0 | 0.5 | 32 | 3.993e+10 | 5.07e+06 | 0.13 | yes | 1367 | 2.24e+10 | 0.605 |
| nytimes_256 | 1024 | 10 | 0.5 | 32 | 2.406e+11 | 5.10e+06 | 0.80 | yes | 2467 | 1.88e+10 | 0.692 |
| nytimes_256 | 1024 | 20 | 0.5 | 32 | 4.413e+11 | 5.13e+06 | 1.47 | NO | 3542 | 1.88e+10 | 0.700 |
| nytimes_256 | 1024 | 20 | 1.0 | 32 | 8.427e+11 | 5.13e+06 | 2.81 | NO | 5695 | 1.87e+10 | 0.681 |
| nytimes_256 | 2048 | 20 | 0.5 | 64 | 8.783e+11 | 7.50e+06 | 2.93 | NO | 5888 | 2.15e+10 | 0.711 |
| nytimes_256 | 4096 | 0 | 0.5 | 128 | 1.591e+11 | 1.22e+07 | 0.53 | yes | 2249 | 3.02e+10 | 0.640 |
| nytimes_256 | 4096 | 20 | 0.5 | 128 | 1.752e+12 | 1.22e+07 | 5.84 | NO | 12594 | 2.88e+10 | 0.729 |

## Fuel is a linear function of the assignment work

Every row fits `gpu_fuel = k * (n_train * n_iters + n_db) * n_lists * dims`
with `k` between 0.506 and 0.537 (0.51 for every row with n_lists >= 1024).
Everything else in the build (the Lloyd update, the CSR lists, centroid pick)
is under 3% of the total. CPU fuel is at most 1.5e7, i.e. under 0.1% of the
GPU figure on every row. So for this family the build cost of a configuration
on a track is predictable from the track's `n_db * dims`:

| track | n_db * dims | proposed limit | limit / (n_db * dims) |
|---|---|---|---|
| sift_128 | 1.280e+08 | 9.0e+11 | 7,031 |
| glove_100 | 1.200e+08 | 1.1e+12 | 9,167 |
| nytimes_256 | 7.680e+07 | 3.0e+11 | 3,906 |

The rightmost column is how many (list x k-means-row-pass) units of work each
limit buys per database row: the proposed nytimes_256 limit is the tightest by
a factor of 1.8 against sift_128 and 2.3 against glove_100.

## What the proposed limits admit

- **cuVS defaults** (1024 lists, 20 iterations, half the rows for training)
  fit on sift_128 at 0.83x and glove_100 at 0.64x, and **do not fit** on
  nytimes_256 at 1.47x.
- On nytimes_256 at 3e11, the largest of the measured configurations that fit
  are 1024 lists with 10 iterations (0.80x) and 4096 lists with no k-means
  (0.53x). From the linear fit, ESTIMATE (not measured): twenty iterations
  fit only with n_lists <= 512, or with 1024 lists and a training fraction of
  about 0.3.
- Training on the full database (train 1.0) or 2048+ lists with 20 iterations
  exceed every proposed limit.
- For the same recipe to be admitted on all three tracks, limits proportional
  to `n_db * dims` would be sift 9e11 : glove 8.4e11 : nytimes 5.4e11
  (arithmetic on the measured fit, not a separate measurement).

## Graph indexes (added 2026-09-23, jobs `…T102149Z-4cd7f0` NSW and `…T102149Z-cde8cb` NN-descent, code `8318425a`)

Two graph builds on the same ABI, same tracks, same seed, same driver
(`scripts/box_build_fuel.sh` with `ALGO=graph_nsw` / `ALGO=graph_nndescent`).
Results: `2026-09-23-c004-index-build-fuel-nsw.tsv`,
`2026-09-23-c004-index-build-fuel-nndescent.tsv`. **All MEASURED.**

- **graph_nsw**: batched single-layer HNSW insertion. Batches double from 1 to
  `batch`; each batch node beam-searches the inserted prefix with
  `ef_construction` candidates and keeps `m` out-edges; reverse edges are added
  under per-node locks, replacing the worst edge when a list is full (no
  diversity heuristic, no hierarchy).
- **graph_nndescent**: the CAGRA k-NN graph build. Random initial lists of
  `degree`, then `rounds` local joins with NN-descent's new/old rule (only
  pairs touching an edge found in the previous round are evaluated). The
  directed k-NN graph is searched as built; CAGRA's reverse-edge and reordering
  passes (about `n * degree` work each) are not included.
- Both use the same greedy beam search for the solve (`ef_search` candidates,
  four hash-chosen entry points), one thread per candidate row with float4
  loads. Fuel counts instructions, and the warp-per-row lane-strided pattern
  costs several times more per row for the same arithmetic.

### NSW

| track | m | ef_construction | batch | ef_search | gpu fuel | cpu fuel | x limit | fits | build ms | solve fuel | recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sift_128 | 16 | 100 | 4096 | 64 | 8.432e+10 | 1.44e+08 | 0.09 | yes | 27833 | 4.11e+08 | 0.851 |
| sift_128 | 16 | 200 | 4096 | 64 | 1.722e+11 | 1.44e+08 | 0.19 | yes | 65389 | 4.10e+08 | 0.865 |
| sift_128 | 32 | 200 | 4096 | 128 | 2.048e+11 | 2.88e+08 | 0.23 | yes | 81429 | 9.44e+08 | 0.973 |
| glove_100 | 16 | 100 | 4096 | 64 | 1.060e+11 | 1.73e+08 | 0.10 | yes | 32719 | 4.22e+08 | 0.292 |
| glove_100 | 16 | 200 | 4096 | 64 | 2.292e+11 | 1.73e+08 | 0.21 | yes | 90598 | 4.24e+08 | 0.296 |
| glove_100 | 32 | 200 | 4096 | 128 | 2.799e+11 | 3.46e+08 | 0.25 | yes | 118258 | 1.03e+09 | 0.616 |
| nytimes_256 | 16 | 100 | 4096 | 64 | 2.950e+10 | 4.33e+07 | 0.10 | yes | 10884 | 4.73e+08 | 0.427 |
| nytimes_256 | 16 | 200 | 4096 | 64 | 6.198e+10 | 4.33e+07 | 0.21 | yes | 26679 | 4.78e+08 | 0.445 |
| nytimes_256 | 32 | 200 | 4096 | 128 | 8.004e+10 | 8.65e+07 | 0.27 | yes | 36699 | 1.21e+09 | 0.696 |

### NN-descent

| track | degree | rounds | ef_search | gpu fuel | cpu fuel | x limit | fits | build ms | solve fuel | recall |
|---|---|---|---|---|---|---|---|---|---|---|
| sift_128 | 32 | 5 | 64 | 8.920e+10 | 2.88e+08 | 0.10 | yes | 15565 | 4.79e+08 | 0.223 |
| sift_128 | 32 | 10 | 64 | 1.330e+11 | 2.88e+08 | 0.15 | yes | 18275 | 4.76e+08 | 0.267 |
| sift_128 | 64 | 10 | 128 | 4.601e+11 | 5.76e+08 | 0.51 | yes | 59778 | 1.25e+09 | 0.731 |
| glove_100 | 32 | 5 | 64 | 9.649e+10 | 3.46e+08 | 0.09 | yes | 11884 | 4.29e+08 | 0.100 |
| glove_100 | 32 | 10 | 64 | 1.451e+11 | 3.46e+08 | 0.13 | yes | 15342 | 4.35e+08 | 0.111 |
| glove_100 | 64 | 10 | 128 | 5.071e+11 | 6.91e+08 | 0.46 | yes | 52667 | 1.24e+09 | 0.331 |
| nytimes_256 | 32 | 5 | 64 | 3.758e+10 | 8.64e+07 | 0.13 | yes | 8221 | 5.29e+08 | 0.278 |
| nytimes_256 | 32 | 10 | 64 | 5.099e+10 | 8.64e+07 | 0.17 | yes | 9324 | 5.32e+08 | 0.297 |
| nytimes_256 | 64 | 10 | 128 | 1.766e+11 | 1.73e+08 | 0.59 | yes | 31081 | 2.23e+09 | 0.546 |

### What the graph rows say

- **Every graph build fits every proposed limit**, with the largest at 0.59x
  (NN-descent, degree 64, nytimes_256) and NSW never above 0.27x. Against the
  cuVS-default IVF build on the same track, NSW at m=16/ef=100 costs 8.8x less on
  sift_128, 6.6x less on glove_100 and 15x less on nytimes_256: there is no
  `n_db * n_lists * dims` assignment pass, and each inserted node touches only
  the rows its search visits.
- **NSW cost scales with ef_construction** (100 -> 200 roughly doubles it) and
  much less with m (16 -> 32 at ef 200 adds 19-29%). Per row it is 380-880
  fuel per dim on the three tracks at m=16/ef=100, against about 5,800 for
  the default IVF.
- **NN-descent cost is dominated by the first round** (a full `n * degree^2`
  join); the new/old rule makes rounds 6-10 cost about half of rounds 1-5 in
  total. Degree 64 costs 3.5x degree 32, close to the 4x of the join size.
- **CPU fuel is no longer negligible for graph indexes** (1.4e8 to 6.9e8): the
  blob is `n_db * degree` ids and the host serialises it element by element.
  It is still under 0.3% of GPU fuel on every row.
- **Recall is a property of the search and of the missing refinement passes,
  not of the build cost.** NSW reaches 0.85-0.97 on sift_128 but 0.29-0.62 on
  glove_100 and 0.43-0.70 on nytimes_256; the raw NN-descent graph navigates
  poorly from random entry points (0.10-0.73). A submission would add the
  reverse-edge/pruning pass (cheap, `n * degree` scale) and a wider search;
  neither changes the build-fuel picture above by more than a few percent.
- Wall clock: NSW builds take 11-118 s and NN-descent 8-60 s on this card,
  inside the runtime's 600 s watchdog, and the beam search's serial thread-0
  merging is what makes NSW slow, not its fuel.

## Decision (2026-09-23): the limit is 1.1e12 for every track

`max_build_fuel_budget` is set to **1.1e12** for c004. The key is per challenge,
so one value covers all three tracks; the proposed 9e11 / 11e11 / 3e11 split is
withdrawn. Against 1.1e12, from the tables above (all MEASURED):

| build | sift_128 | glove_100 | nytimes_256 |
|---|---|---|---|
| IVF 1024 lists, 20 iters, train 0.5 (cuVS default) | 0.68x | 0.64x | 0.40x |
| IVF 1024 lists, 20 iters, train 1.0 | 1.29x, fails | 1.22x, fails | 0.77x |
| IVF 2048 lists, 20 iters, train 0.5 | 1.34x, fails | 1.27x, fails | 0.80x |
| IVF 4096 lists, 0 iters | 0.24x | 0.23x | 0.14x |
| IVF 4096 lists, 20 iters | 2.68x, fails | 2.53x, fails | 1.59x, fails |
| NSW m=32, ef_construction=200 (largest measured) | 0.19x | 0.25x | 0.07x |
| NN-descent degree 64, 10 rounds (largest measured) | 0.42x | 0.46x | 0.16x |

Where the value is recorded: `tig-structs/src/config.rs` (comment on the key),
`tig-challenges/src/vector_search/README.md`, `scripts/test_algorithm`
(`BUILD_FUEL_LIMIT`, the default `--build-fuel`), and the superseded-value notes
in the 2026-08-31 design doc and measurement note. The live protocol config is
not in this repo; setting the key there is a separate step, and
`build_fuel_alpha` remains unset (see the design doc's gate).

## Caveats (IVF section)

- The IVF rows measure one family; the graph section above adds two more.
  A product-quantisation build is still unmeasured.
- The recall column is at a fixed n_probe per n_lists (about 3% of lists) and
  is a property of the *search* settings: the angular corpora (glove, nytimes)
  need more probes than sift to reach the 0.9 bar, at solve fuel, not build
  fuel. Solve fuel here is 1.8e10 to 8.3e10 per nonce against the 5e12 budget.
- Fuel is an instruction count and did not vary with the seed in the earlier
  standalone probe (`wgan-synthetic`, 2026-09-07); it is not re-measured across
  seeds here.

