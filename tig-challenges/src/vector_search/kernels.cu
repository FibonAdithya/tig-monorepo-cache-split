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

#define AUDIT_BLOCK 256
#define AUDIT_MAX_DIMS 128

// One block per audited query. Writes hits[s] = 1 when the submitted answer for
// query sample_query_ids[s] is within tolerance of the true nearest neighbour.
//
// tolerance_sq is (1 + tau)^2: the comparison runs in squared distance so that
// no sqrt appears, and --use_fast_math makes sqrt approximate.
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
    const int s = blockIdx.x;
    if (s >= num_samples) return;
    const uint32_t q = sample_query_ids[s];

    __shared__ float s_query[AUDIT_MAX_DIMS];
    for (int i = threadIdx.x; i < vector_dims; i += AUDIT_BLOCK) {
        s_query[i] = query_vectors[(long long)q * vector_dims + i];
    }
    __syncthreads();

    __shared__ float s_returned;
    if (threadIdx.x == 0) {
        const size_t idx = solution_indexes[q];
        if (idx >= database_size) {
            // atomicExch, not a plain store: every block that finds a bad index
            // writes the same constant, so the outcome is deterministic either
            // way, but a concurrent non-atomic write is still a data race under
            // the strict memory model. The host also range-checks every index
            // before launching; this branch is defence in depth for a future
            // caller that bypasses that path.
            atomicExch(error_flag, 1u);
            s_returned = 3.0e38f;
        } else {
            const float *cand = database_vectors + idx * vector_dims;
            float d = 0.0f;
            for (int k = 0; k < vector_dims; ++k) {
                const float diff = s_query[k] - cand[k];
                d = fmaf(diff, diff, d);
            }
            s_returned = d;
        }
    }

    float best = 3.0e38f;
    for (int j = threadIdx.x; j < database_size; j += AUDIT_BLOCK) {
        const float *cand = database_vectors + (long long)j * vector_dims;
        float d = 0.0f;
        for (int k = 0; k < vector_dims; ++k) {
            const float diff = s_query[k] - cand[k];
            d = fmaf(diff, diff, d);
        }
        if (d < best) best = d;
    }

    __shared__ float s_best[AUDIT_BLOCK];
    // Every slot is initialised, not just the blockDim.x of them that ran the
    // scan. The reduction below reads s_best[t + stride] for strides up to
    // AUDIT_BLOCK/2, so a launch with fewer threads than AUDIT_BLOCK would
    // otherwise fold uninitialised shared memory into the minimum and turn
    // every hit into a miss -- silently, with no error anywhere. (The scan's
    // AUDIT_BLOCK stride assumes the same equality, which is why the host ties
    // its block_dim to a constant rather than a literal.)
    for (int i = threadIdx.x; i < AUDIT_BLOCK; i += blockDim.x) {
        s_best[i] = 3.0e38f;
    }
    __syncthreads();
    s_best[threadIdx.x] = best;
    __syncthreads();
    for (int stride = AUDIT_BLOCK / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float other = s_best[threadIdx.x + stride];
            if (other < s_best[threadIdx.x]) s_best[threadIdx.x] = other;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        hits[s] = (s_returned <= s_best[0] * tolerance_sq) ? 1u : 0u;
    }
}
