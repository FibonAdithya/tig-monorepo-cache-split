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
  (0.53x). Twenty iterations fit only with n_lists <= 512 or a training
  fraction of about 0.3.
- Training on the full database (train 1.0) or 2048+ lists with 20 iterations
  exceed every proposed limit.
- For the same recipe to be admitted on all three tracks, limits proportional
  to `n_db * dims` would be sift 9e11 : glove 8.4e11 : nytimes 5.4e11.

## Caveats

- One index family. A build with a different inner loop (product quantisation,
  a graph index) costs differently; this measures IVF-Flat's k-means build,
  which is dominated by the same tiled distance kernel the exact scan uses.
- The recall column is at a fixed n_probe per n_lists (about 3% of lists) and
  is a property of the *search* settings: the angular corpora (glove, nytimes)
  need more probes than sift to reach the 0.9 bar, at solve fuel, not build
  fuel. Solve fuel here is 1.8e10 to 8.3e10 per nonce against the 5e12 budget.
- Fuel is an instruction count and did not vary with the seed in the earlier
  standalone probe (`wgan-synthetic`, 2026-09-07); it is not re-measured across
  seeds here.

