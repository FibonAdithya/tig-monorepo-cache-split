#include <curand_kernel.h>
#include <stdint.h>

// Draw one latent vector per output vector. The global index enters both the
// seed selection and curand sequence, so changing chunk boundaries cannot
// change a vector's random values.
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
        curand_init(((const uint64_t *)(seed))[global_i % 4], global_i, 0, &state);
        float *row = latents + (long long)i * latent_dim;
        for (int j = 0; j < latent_dim; ++j) {
            row[j] = curand_normal(&state);
        }
    }
}

extern "C" __global__ void evaluate_total_distance(
    const uint32_t vector_dims,
    const uint32_t database_size,
    const uint32_t num_queries,
    const float *query_vectors,
    const float *database_vectors,
    const size_t *solution_indexes,
    float *total_distance,
    int *error_flag
)
{
    *total_distance = 0.0f;
    
    for (int idx = 0; idx < num_queries; idx++) {
        size_t search_index = solution_indexes[idx];

        if (search_index >= database_size) {
            *error_flag = 1;
            return;
        }

        const float *search = database_vectors + search_index * vector_dims;
        const float *query = query_vectors + idx * vector_dims;
        
        float dist = 0.0f;
        for (int i = 0; i < vector_dims; ++i) {
            float diff = query[i] - search[i];
            dist += diff * diff;
        }
        dist = sqrtf(dist);

        *total_distance += dist;
    }
}

// Fixed-order dense layer with optional LeakyReLU(0.2).
//
// Determinism is the whole point. Each thread owns complete output elements and
// accumulates over k sequentially, so the summation order does not depend on
// how the scheduler interleaves work. That is the
// property cuBLAS cannot offer: it picks kernels by heuristic, and a different
// architecture picks a different reduction order and a different last bit.
//
// Only fmaf/mul/add/select appear here. build_ptx compiles with --use_fast_math,
// which makes division, sqrt and transcendentals approximate and potentially
// architecture-dependent; these operations are all IEEE-754 exactly rounded, so
// identical PTX yields identical results on any conforming GPU.
//
// A block produces a 128-row by 64-column output tile. Input and weight tiles are
// staged in shared memory, while each thread accumulates an 8-by-4 register tile.
// Each shared-memory operand is reused by four FMAs per thread, and every weight
// is reused across 128 rows. Splitting K across blocks is deliberately avoided:
// every output still visits k=0..in_dim in exactly that order.
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
    constexpr int BM = 128;
    constexpr int BN = 64;
    constexpr int BK = 16;
    constexpr int BLOCK_X = 16;
    constexpr int BLOCK_Y = 16;
    constexpr int TM = BM / BLOCK_Y;
    constexpr int TN = BN / BLOCK_X;

    __shared__ float s_input[BM][BK + 1];
    __shared__ float s_weight[BN][BK + 1];

    const int tid = threadIdx.y * BLOCK_X + threadIdx.x;
    const int row_base = blockIdx.x * BM;
    const int col_base = blockIdx.y * BN;

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int col = col_base + threadIdx.x + j * BLOCK_X;
            acc[i][j] = (col < out_dim) ? bias[col] : 0.0f;
        }
    }

    for (int k_base = 0; k_base < in_dim; k_base += BK) {
        for (int i = tid; i < BM * BK; i += BLOCK_X * BLOCK_Y) {
            const int tile_row = i / BK;
            const int tile_k = i - tile_row * BK;
            const int row = row_base + tile_row;
            const int k = k_base + tile_k;
            s_input[tile_row][tile_k] =
                (row < n && k < in_dim)
                    ? input[(long long)row * in_dim + k]
                    : 0.0f;
        }
        for (int i = tid; i < BN * BK; i += BLOCK_X * BLOCK_Y) {
            const int tile_col = i / BK;
            const int tile_k = i - tile_col * BK;
            const int global_col = blockIdx.y * BN + tile_col;
            const int k = k_base + tile_k;
            s_weight[tile_col][tile_k] =
                (global_col < out_dim && k < in_dim)
                    ? weight[(long long)global_col * in_dim + k]
                    : 0.0f;
        }
        __syncthreads();

        const int tile_k_count = min(BK, in_dim - k_base);
        for (int tile_k = 0; tile_k < tile_k_count; ++tile_k) {
            float x[TM];
            float w[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                x[i] = s_input[threadIdx.y + i * BLOCK_Y][tile_k];
            }
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                w[j] = s_weight[threadIdx.x + j * BLOCK_X][tile_k];
            }
#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] = fmaf(x[i], w[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int row = row_base + threadIdx.y + i * BLOCK_Y;
        if (row < n) {
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int col = col_base + threadIdx.x + j * BLOCK_X;
                if (col >= out_dim) {
                    continue;
                }
                float value = acc[i][j];
                if (apply_activation) {
                    value = (value >= 0.0f) ? value : (value * 0.2f);
                }
                const long long out_row = (long long)(out_row_offset + row);
                output[out_row * out_dim + col] = value;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Row-wise generator steps. One thread owns whole rows and walks a row's
// coordinates in index order, so the result cannot depend on launch geometry.
//
// Unlike gan_linear these use sqrt and division, which --use_fast_math allows
// to be approximate. Verification is recall within a relative 1e-6, not a
// bit-exact integer, so a last-bit difference in a row does not by itself
// change a verdict. See docs/ai/specs/2026-09-21-multi-arch-gan-scenarios-design.md,
// "Determinism", for what this does and does not establish.
// ---------------------------------------------------------------------------

// x / max(||x||, eps), in place, for rows [row_offset, row_offset + n).
extern "C" __global__ void gan_row_normalize(
    float *data,
    const int n,
    const int dim,
    const float eps,
    const int row_offset
)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n;
         i += blockDim.x * gridDim.x)
    {
        float *row = data + (long long)(row_offset + i) * dim;
        float ss = 0.0f;
        for (int j = 0; j < dim; ++j) {
            ss = fmaf(row[j], row[j], ss);
        }
        float norm = sqrtf(ss);
        if (norm < eps) {
            norm = eps;
        }
        for (int j = 0; j < dim; ++j) {
            row[j] = row[j] / norm;
        }
    }
}

// out = leaky(a * (1 + gamma) + beta), element-wise over `count` values.
//
// Element-wise, not row-wise: `count` is rows * width, so one thread owns one
// value rather than a whole row. The largest product this crate launches is
// 131,072 rows (FORWARD_CHUNK is 65,536; the launch-geometry test goes to twice
// that) at width 512, which is 67,108,864 -- well inside `int`, so the index,
// the stride and `count` itself all stay 32-bit as in the kernels above.
extern "C" __global__ void gan_film_leaky(
    const float *a,
    const float *gamma,
    const float *beta,
    float *out,
    const int count
)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < count;
         i += blockDim.x * gridDim.x)
    {
        const float v = fmaf(a[i], 1.0f + gamma[i], beta[i]);
        out[i] = (v >= 0.0f) ? v : (v * 0.2f);
    }
}

// The spherical generator's last step, per row:
//   u = unit(d);  t = unit(v - (v.u) u);  out = cos_r * u + sin_r * t
// `d` and `v` are scratch and are overwritten with u and t.
//
// Every fmaf here has a matching mul_add in `Spherical::forward_cpu`, in the
// same operand order, and the divide by `norm` happens before the multiply by
// `sin_r` for the same reason: `normalize_cpu` stores t/norm back as f32 and
// only then scales it. Keep them in step if either changes.
extern "C" __global__ void gan_sphere_combine(
    float *d,
    float *v,
    float *out,
    const int n,
    const int dim,
    const float cos_r,
    const float sin_r,
    const float eps,
    const int out_row_offset
)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n;
         i += blockDim.x * gridDim.x)
    {
        float *u = d + (long long)i * dim;
        float *t = v + (long long)i * dim;
        float *o = out + (long long)(out_row_offset + i) * dim;

        float ss = 0.0f;
        for (int j = 0; j < dim; ++j) {
            ss = fmaf(u[j], u[j], ss);
        }
        float norm = sqrtf(ss);
        if (norm < eps) {
            norm = eps;
        }
        for (int j = 0; j < dim; ++j) {
            u[j] = u[j] / norm;
        }

        float dot = 0.0f;
        for (int j = 0; j < dim; ++j) {
            dot = fmaf(t[j], u[j], dot);
        }
        for (int j = 0; j < dim; ++j) {
            t[j] = fmaf(-dot, u[j], t[j]);
        }

        ss = 0.0f;
        for (int j = 0; j < dim; ++j) {
            ss = fmaf(t[j], t[j], ss);
        }
        norm = sqrtf(ss);
        if (norm < eps) {
            norm = eps;
        }

        for (int j = 0; j < dim; ++j) {
            o[j] = fmaf(cos_r, u[j], sin_r * (t[j] / norm));
        }
    }
}

// Sequence base for the gate's noise stream. gan_sample_latents passes an
// int index as the curand sequence, so it cannot reach 2^40: the two streams
// are disjoint for the same seed word.
#define GATE_NOISE_SEQUENCE_BASE (1ULL << 40)

// Logistic noise from one uniform draw: log(u) - log1p(-u).
//
// curand_uniform returns (0, 1], and in float32 `1 - 1e-8` IS 1.0, so the
// PyTorch-style clamp(eps, 1 - eps) would let u = 1.0 through: log1p(-1) is
// -inf, the noise +inf, and after smoothing (a linear map) a NaN wherever it
// meets a -inf. At 89.6 million draws per database that is expected several
// times per instance, not a corner case. So the upper clamp is the largest
// float below 1.0, written out.
//
// What makes that clamp worth its own test: the damage is silent, and it spreads.
// The smoothing layer sums over EVERY tap. Under IEEE 754 rules a zero weight
// times an infinity is a NaN, so one corrupted draw is expected to turn the whole
// 128-wide smoothed row into NaN, not just the taps near it. `logit + NaN > 0` is
// false for every coordinate, so `any_open` stays 0 and gan_gate_apply's
// all-closed fallback replaces the row with a one-hot vector: still finite, still
// unit-norm, which is why the 700,000-row unit-norm test cannot see it.
//
// That is what is expected, not something to reason from: this PTX is built with
// --use_fast_math, and what a zero times an infinity does under it is not
// guaranteed by the IEEE rules. The code does not rely on it either way. The
// clamp is needed regardless, because an infinite noise value is wrong whatever
// the smoothing map and the gate comparison then make of it -- NaN across the
// row, or a single coordinate forced open or closed. The argument above says why
// the wrongness is hard to SEE, which is the case for the dedicated test; it is
// not the reason the clamp is there.
//
// Hence a function rather than an expression inlined into each caller: the
// test-only test_gate_noise_from_uniform drives these two clamps directly, and
// it drives THESE, not a copy. A test kernel that repeated the expression would
// still pass with the clamp removed from the shipped kernel, which is exactly
// the mutation that went uncaught.
//
// `__device__ __forceinline__`, the form ref_before in mod.rs's reference kernel
// already uses.
__device__ __forceinline__ float gate_noise_from_uniform(float u, const float eps)
{
    if (u > 0.99999994f) u = 0.99999994f;
    if (u < eps) u = eps;
    return logf(u) - log1pf(-u);
}

// One logistic draw per output coordinate, for the rows of this chunk.
extern "C" __global__ void gan_gate_noise(
    const uint8_t *seed,
    const int n,
    const int dim,
    const float eps,
    float *noise,
    const int index_offset
)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n;
         i += blockDim.x * gridDim.x)
    {
        const int global_i = index_offset + i;
        curandState state;
        curand_init(((const uint64_t *)(seed))[global_i % 4],
                    GATE_NOISE_SEQUENCE_BASE + (unsigned long long)global_i, 0, &state);
        float *row = noise + (long long)i * dim;
        for (int j = 0; j < dim; ++j) {
            row[j] = gate_noise_from_uniform(curand_uniform(&state), eps);
        }
    }
}

// The structured gate's last step, per row. `noise` is ALREADY smoothed.
//   logit_j = clamp * tanh((coupled_j + sparsity) / clamp)
//   open_j  = logit_j + noise_j > 0       (== sigmoid(./T) > 0.5 for any T > 0)
//   none open -> open only the first argmax of logit
//   m_j     = max(softplus(mag_pre_j), floor),  softplus(x) = x > 20 ? x : log1p(exp(x))
//   out     = unit(open * m)
//
// Every step matches `StructuredGate::forward_cpu` in order as well as in
// value: the logit multiplies the clamp back in after the tanh, the softplus
// threshold is the same 20 (PyTorch's F.softplus default), the argmax
// comparison is strict so the FIRST maximum wins as torch.argmax does, and the
// sum of squares accumulates through fmaf in index order, as normalize_cpu
// does. The two differ only on a NaN magnitude, which no finite weight
// produces: `max` in Rust returns the non-NaN operand where the `<` test here
// keeps the NaN. That is the same divergence gan_row_normalize already has
// against normalize_cpu.
extern "C" __global__ void gan_gate_apply(
    const float *mag_pre,
    const float *coupled,
    const float *sparsity,
    const float *noise,
    float *out,
    const int n,
    const int dim,
    const float logit_clamp,
    const float magnitude_floor,
    const float eps,
    const int out_row_offset
)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n;
         i += blockDim.x * gridDim.x)
    {
        const float *m = mag_pre + (long long)i * dim;
        const float *c = coupled + (long long)i * dim;
        const float *z = noise + (long long)i * dim;
        const float s = sparsity[i];
        float *o = out + (long long)(out_row_offset + i) * dim;

        int best = 0;
        // 3.0e38 rather than INFINITY: --use_fast_math permits relaxations
        // around infinities (same choice as the audit kernel). Any logit is
        // bounded by logit_clamp, so the first coordinate always takes this
        // branch and `best` starts at 0 exactly as forward_cpu's does.
        float best_logit = -3.0e38f;
        float best_mag = 0.0f;
        int any_open = 0;
        float ss = 0.0f;
        for (int j = 0; j < dim; ++j) {
            const float logit = logit_clamp * tanhf((c[j] + s) / logit_clamp);
            float mag = (m[j] > 20.0f) ? m[j] : log1pf(expf(m[j]));
            if (mag < magnitude_floor) mag = magnitude_floor;
            if (logit > best_logit) {   // strict: keeps the FIRST maximum, as torch.argmax
                best_logit = logit;
                best = j;
                best_mag = mag;
            }
            const int open = (logit + z[j] > 0.0f) ? 1 : 0;
            const float value = open ? mag : 0.0f;
            o[j] = value;
            ss = fmaf(value, value, ss);
            any_open |= open;
        }
        if (!any_open) {
            o[best] = best_mag;
            // Not the accumulated `ss`: every value written above was 0 in this
            // branch, so the sum of squares is this one coordinate's. fmaf of a
            // single product into 0.0f rounds the same way, which is what
            // normalize_cpu computes over the same vector.
            ss = best_mag * best_mag;
        }
        float norm = sqrtf(ss);
        if (norm < eps) norm = eps;
        for (int j = 0; j < dim; ++j) {
            o[j] = o[j] / norm;
        }
    }
}

#define AUDIT_BLOCK 256
#define AUDIT_MAX_DIMS 256

// Audited queries staged per block. The block streams the database ONCE for all
// AUDIT_TQ of them, so database traffic falls by this factor: auditing 1,000
// queries goes from 1,000 sweeps of the 358 MB database to ceil(1000/18) = 56.
//
// Must equal AUDIT_TQ in mod.rs, which derives grid_dim from it.
//
// Chosen by measurement on an RTX 3060 (28 SMs, sm_86, PTX JIT'd from
// compute_70), SIFT_128 -- 1,000 samples over 700,000 x 128 database vectors --
// best of three under a gpu-claim, with the whole of measure_recall timed (host
// overhead is ~0.3 ms of it; the kernel alone is stable to +/-0.02 ms):
//
//   untiled, one block per query          2623 ms
//   AUDIT_TQ =  8   125 blocks             119 ms
//   AUDIT_TQ = 12    84 blocks             104 ms
//   AUDIT_TQ = 14    72 blocks             130 ms
//   AUDIT_TQ = 16    63 blocks         114-122 ms
//   AUDIT_TQ = 17    59 blocks             124 ms
//   AUDIT_TQ = 18    56 blocks              93 ms   <- chosen
//   AUDIT_TQ = 19    53 blocks              98 ms
//   AUDIT_TQ = 20    50 blocks             103 ms
//   AUDIT_TQ = 22    46 blocks             107 ms
//   AUDIT_TQ = 23    44 blocks             182 ms
//   AUDIT_TQ = 32    32 blocks             234 ms
//
// The sweep above was taken back to back across many rebuilds; the committed
// kernel measures 85-87 ms on five later runs of the full suite, against a
// 150 ms gate. Both figures are the same PTX (56 registers, 27,720 bytes of
// shared memory), so the spread is the card's clocks, not the code. The
// timing table above and the 27,720-byte figure were both measured on an
// RTX 3060 at AUDIT_MAX_DIMS = 128. At AUDIT_MAX_DIMS = 256, shared memory
// is 36,936 bytes (18*256*4 + 256*17*4 + 256*4 + 18*4 = 18,432 + 17,408 +
// 1,024 + 72), computed from the __shared__ declarations below, not read
// from ptxas.
//
// MEASURED on 2026-09-21, on an RTX 3060 Ti (compute capability 8.6, CUDA
// 12.8, PTX built -arch compute_70 -code sm_70 --use_fast_math, the same
// target as above): recall_audit_tests' timed test,
// audit_is_much_cheaper_than_a_naive_solve, over SIFT_128 -- the same
// 1,000-sample, 700,000 x 128 workload as the table above. Each value below
// is one run's own best-of-three, as that test prints it. The three runs at
// AUDIT_MAX_DIMS = 128 ran the whole crate's test suite; the three runs at
// AUDIT_MAX_DIMS = 256 ran the recall_audit_tests module only. The timed
// test and its workload were identical in both:
//
//   AUDIT_MAX_DIMS = 128   87, 87, 87 ms
//   AUDIT_MAX_DIMS = 256   88, 88, 87 ms   <- this commit; 21/21 tests green
//
// Raising the staging width from 128 to 256 dims cost at most 1 ms here: the
// concern that a wider s_query would cost a resident block -- the 107 -> 182
// ms step the table above records between AUDIT_TQ 22 and 23 -- did not
// materialise at AUDIT_TQ = 18. Both rows above were taken on the same
// RTX 3060 Ti on the same day, running the same timed test over the same
// workload, so that pair is a like-for-like comparison on those points -- but
// not on the neighbouring load, since the 128 runs ran the whole suite and
// the 256 runs ran only recall_audit_tests. A 1 ms difference is within what
// that could explain, so read this as no measurable cost, not as exactly
// 1 ms. Neither row is directly comparable to the AUDIT_TQ table above,
// which was taken on a different card, an RTX 3060.
//
// Not measured: AUDIT_TQ was not re-swept at 256, so 18 is known to be
// acceptable at that width, not known to be optimal for it. Also not
// measured: any scenario that actually declares 256 dims, where each staged
// query is twice as wide and the per-row distance loop runs twice as long --
// SIFT_128 stays at 128 dims regardless of AUDIT_MAX_DIMS, so nothing above
// exercises that cost. The first such number comes from the NYTimes
// scenario's box run. Every timing in this file was taken on one of two
// Ampere cards (RTX 3060, RTX 3060 Ti); none was taken on another
// architecture, so none of these figures should be assumed to hold across
// the GPUs miners run.
//
// The curve is not monotonic, and neither end of it is where the cost lives:
//
//   - Above AUDIT_TQ = 22 ptxas needs 96 registers rather than 64 for the
//     per-thread running minima, which drops the SM from three resident blocks
//     to two. That is the 107 -> 182 ms step between 22 and 23, and it is why
//     the obvious "more staged queries is strictly less traffic" reasoning
//     gives the wrong answer.
//   - Below that, what moves the number is how evenly the blocks divide over
//     the 28 SMs. AUDIT_TQ = 18 lands on exactly 56 = 28 x 2. This is the one
//     tuning input here that is a property of the card rather than of the
//     kernel, so on a machine with a different SM count the plateau will sit
//     somewhere slightly different -- the whole 18..22 region measures 93-107
//     ms, well inside the gate, so nothing depends on hitting 18 exactly.
//
// Register blocking over database rows -- each thread owning R rows so that a
// staged query value is reused R times, which is what a GEMM would do -- was
// also measured, and is slower at every setting tried. It needs R times the
// shared memory for the tile, and losing the third resident block costs more
// than the saved shared-memory loads gain. All at AUDIT_TQ = 16, against
// 114-122 ms for one row per thread:
//
//   R = 2, 8-dim chunks, 512-row tile    128 ms
//   R = 2, 16-dim chunks, 512-row tile   141 ms
//   R = 4, 8-dim chunks, 1024-row tile   125 ms
//
// So: one row per thread.
#define AUDIT_TQ 18

// Dims of a database row staged per pass over a tile. A power of two so the
// staging index arithmetic is a shift, and large enough that each row's slice
// is a whole number of fully-used 32-byte sectors.
#define AUDIT_KC 16
// Padded by one so that s_db[row][k] is bank-conflict-free: the row stride is
// odd and therefore coprime with 32, so 32 consecutive rows land on 32 distinct
// banks.
#define AUDIT_DB_STRIDE (AUDIT_KC + 1)
// Rows of the tile covered per pass of the cooperative load.
#define AUDIT_ROWS_PER_PASS (AUDIT_BLOCK / AUDIT_KC)

// The four constants above are not independent, and every relationship between
// them is one a wrong value would break silently -- by skipping candidates and
// reporting recall HIGHER than reality, which is the direction no test here
// catches. So they are checked at compile time rather than left to a comment.
static_assert(AUDIT_BLOCK % AUDIT_KC == 0,
              "AUDIT_ROWS_PER_PASS truncates unless AUDIT_KC divides "
              "AUDIT_BLOCK, and the staging loop then never reaches the last "
              "rows of each tile");
static_assert((AUDIT_BLOCK & (AUDIT_BLOCK - 1)) == 0,
              "the minimum is reduced by repeated halving from AUDIT_BLOCK/2, "
              "which drops the odd slot unless AUDIT_BLOCK is a power of two");
static_assert(AUDIT_TQ <= AUDIT_BLOCK,
              "one thread per audited query computes the returned distance, so "
              "a block must have at least AUDIT_TQ threads");
static_assert(AUDIT_TQ >= 1, "a block must audit at least one query");

// Recall@1 audit, tiled over queries.
//
// One block audits AUDIT_TQ consecutive samples: it stages their query vectors
// in shared memory and then sweeps the database once for all of them, holding
// AUDIT_TQ running minima per thread in registers. grid_dim is
// ceil(num_samples / AUDIT_TQ).
//
// Database rows are read cooperatively and coalesced into shared memory --
// AUDIT_BLOCK threads fetch AUDIT_ROWS_PER_PASS rows of AUDIT_KC contiguous
// floats each, so every warp issues fully-used sectors -- and each thread then
// owns one row of the tile and reads it back out of shared memory. The
// alternative, every thread walking its own 512-byte row in global memory, is
// the defect that makes the reference 1-NN kernel run at a small fraction of
// achievable bandwidth.
//
// Writes hits[s] = 1 when the submitted answer for query sample_query_ids[s] is
// within tolerance of the true nearest neighbour.
//
// tolerance_sq is (1 + tau)^2: the comparison runs in squared distance so that
// no sqrt appears, and --use_fast_math makes sqrt approximate.
//
// Determinism survives the tiling. Every distance still accumulates over
// k = 0..vector_dims-1 in that order through fmaf, exactly as the untiled
// kernel did, so the distances are bit-identical to it. The only reduction is a
// minimum, which is exact at any width, and it is still a fixed-order
// shared-memory tree -- no atomics into an accumulator, and nothing that
// depends on how the scheduler interleaves blocks.
extern "C" __global__ void recall_audit(
    const uint32_t vector_dims,
    const uint32_t database_size,
    const uint32_t num_samples,
    const float *__restrict__ query_vectors,
    const float *__restrict__ database_vectors,
    const size_t *__restrict__ solution_indexes,
    const uint32_t *__restrict__ sample_query_ids,
    const float tolerance_sq,
    uint32_t *__restrict__ hits,
    uint32_t *__restrict__ error_flag)
{
    // blockDim.x == AUDIT_BLOCK is structural, not a preference: one thread
    // owns one row of an AUDIT_BLOCK-row tile, the cooperative staging maps
    // AUDIT_BLOCK threads onto AUDIT_ROWS_PER_PASS x AUDIT_KC elements of it,
    // and the reduction folds AUDIT_BLOCK/2 strides. A launch with fewer
    // threads would leave tile rows unexamined, so the kernel would miss the
    // true minimum and report false HITS -- recall reading HIGHER than reality,
    // the one failure direction a correctness test that checks only that a bad
    // answer scores badly cannot see. Fail closed rather than trust the caller.
    // mod.rs ties block_dim to a Rust const that must equal this #define; this
    // is the belt to that braces.
    //
    // Code 2, not 1: 1 means a solution index out of range, and a future
    // debugger reading "Invalid index in solution" after a block-size mismatch
    // would hunt entirely the wrong bug.
    if (blockDim.x != AUDIT_BLOCK) {
        if (threadIdx.x == 0) atomicExch(error_flag, 2u);
        return;
    }

    const int sample_base = blockIdx.x * AUDIT_TQ;
    if (sample_base >= (int)num_samples) return;
    // The final block gets a short group whenever AUDIT_TQ does not divide
    // num_samples -- 10 of 18 at the default 1,000 samples.
    int n_tq = (int)num_samples - sample_base;
    if (n_tq > AUDIT_TQ) n_tq = AUDIT_TQ;

    __shared__ float s_query[AUDIT_TQ][AUDIT_MAX_DIMS];
    __shared__ float s_db[AUDIT_BLOCK][AUDIT_DB_STRIDE];
    __shared__ float s_red[AUDIT_BLOCK];
    __shared__ float s_returned[AUDIT_TQ];

    // Stage this block's queries. Slots past n_tq are zero-filled rather than
    // left alone: the accumulator loop below runs over the compile-time
    // AUDIT_TQ so that it unrolls, and folding uninitialised shared memory into
    // an arithmetic result -- even one nobody reads -- is the defect class the
    // untiled kernel's reduction had.
    for (int t = 0; t < AUDIT_TQ; ++t) {
        if (t < n_tq) {
            const uint32_t q = sample_query_ids[sample_base + t];
            for (int k = threadIdx.x; k < (int)vector_dims; k += AUDIT_BLOCK) {
                s_query[t][k] = query_vectors[(long long)q * vector_dims + k];
            }
        } else {
            for (int k = threadIdx.x; k < (int)vector_dims; k += AUDIT_BLOCK) {
                s_query[t][k] = 0.0f;
            }
        }
    }
    __syncthreads();

    // One thread per audited query computes the distance to the answer the
    // solution returned. These are the only global reads outside the sweep.
    if ((int)threadIdx.x < n_tq) {
        const int t = threadIdx.x;
        const uint32_t q = sample_query_ids[sample_base + t];
        const size_t idx = solution_indexes[q];
        if (idx >= database_size) {
            // atomicExch, not a plain store: every thread that finds a bad
            // index writes the same constant, so the outcome is deterministic
            // either way, but a concurrent non-atomic write is still a data
            // race under the strict memory model. The host also range-checks
            // every index before launching; this branch is defence in depth for
            // a future caller that bypasses that path.
            atomicExch(error_flag, 1u);
            s_returned[t] = 3.0e38f;
        } else {
            const float *cand = database_vectors + idx * vector_dims;
            float d = 0.0f;
            for (int k = 0; k < (int)vector_dims; ++k) {
                const float diff = s_query[t][k] - cand[k];
                d = fmaf(diff, diff, d);
            }
            s_returned[t] = d;
        }
    }

    float best[AUDIT_TQ];
#pragma unroll
    for (int t = 0; t < AUDIT_TQ; ++t) best[t] = 3.0e38f;

    // Each thread owns one row of the tile; the cooperative load is indexed
    // independently of that ownership, so that consecutive threads read
    // consecutive floats of the same database row.
    const int row = (int)threadIdx.x;
    const int load_row = (int)threadIdx.x / AUDIT_KC;
    const int load_k = (int)threadIdx.x % AUDIT_KC;

    for (int base = 0; base < (int)database_size; base += AUDIT_BLOCK) {
        int rows_in_tile = (int)database_size - base;
        if (rows_in_tile > AUDIT_BLOCK) rows_in_tile = AUDIT_BLOCK;
        const bool active = row < rows_in_tile;

        float acc[AUDIT_TQ];
#pragma unroll
        for (int t = 0; t < AUDIT_TQ; ++t) acc[t] = 0.0f;

        for (int k0 = 0; k0 < (int)vector_dims; k0 += AUDIT_KC) {
            int kc = (int)vector_dims - k0;
            if (kc > AUDIT_KC) kc = AUDIT_KC;

            // Before the store: the previous pass's reads of s_db must have
            // finished. After it: the stores must be visible to every thread.
            __syncthreads();
            if (load_k < kc) {
                for (int r = load_row; r < rows_in_tile; r += AUDIT_ROWS_PER_PASS) {
                    s_db[r][load_k] =
                        database_vectors[(long long)(base + r) * vector_dims + k0 + load_k];
                }
            }
            __syncthreads();

            if (active) {
                for (int k = 0; k < kc; ++k) {
                    const float c = s_db[row][k];
#pragma unroll
                    for (int t = 0; t < AUDIT_TQ; ++t) {
                        const float diff = s_query[t][k0 + k] - c;
                        acc[t] = fmaf(diff, diff, acc[t]);
                    }
                }
            }
        }

        if (active) {
#pragma unroll
            for (int t = 0; t < AUDIT_TQ; ++t) {
                if (acc[t] < best[t]) best[t] = acc[t];
            }
        }
    }

    // Fixed-order tree reduction, one audited query at a time so the scratch
    // buffer stays AUDIT_BLOCK floats rather than AUDIT_TQ times that.
    // AUDIT_TQ * 8 barrier rounds, once, at the end of a sweep over 358 MB, is
    // not measurable.
    for (int t = 0; t < AUDIT_TQ; ++t) {
        __syncthreads();
        s_red[threadIdx.x] = best[t];
        __syncthreads();
        for (int stride = AUDIT_BLOCK / 2; stride > 0; stride >>= 1) {
            if ((int)threadIdx.x < stride) {
                const float other = s_red[threadIdx.x + stride];
                if (other < s_red[threadIdx.x]) s_red[threadIdx.x] = other;
            }
            __syncthreads();
        }
        // hits[s] is written by exactly one thread: thread 0 of the single
        // block that owns sample s.
        if (threadIdx.x == 0 && t < n_tq) {
            hits[sample_base + t] =
                (s_returned[t] <= s_red[0] * tolerance_sq) ? 1u : 0u;
        }
    }
}
