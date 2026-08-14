#include <curand_kernel.h>
#include <stdint.h>

extern "C" __global__ void generate_clusters(
    const uint8_t *seed,
    const float avg_weight,
    const int vector_dims,
    const float var,
    const float alpha,
    const int num_clusters,
    float *cluster_means,
    float *cluster_stds,
    float *cluster_weights
)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < num_clusters; i += blockDim.x * gridDim.x) 
    {
        curandState state;
        curand_init(((uint64_t *)(seed))[i % 4], i, 0, &state);
    
        cluster_weights[i] = expf(avg_weight + curand_normal(&state) * sqrtf(var));
        float *means = cluster_means + i * vector_dims;
        float *stds = cluster_stds + i * vector_dims;
        float sigma = curand_uniform(&state) * 2 * alpha + 1.05 - alpha;
        float epsilon = curand_uniform(&state) * alpha;
        for (int j = 0; j < vector_dims; ++j)
        {
            means[j] = curand_uniform(&state) * 2.0 - 1.0;
            stds[j] = curand_uniform(&state) * 2 * epsilon + sigma - epsilon;
        }
    }
}

__device__ int binary_search(
    const float *arr,
    const float target,
    const int size
)
{
    int left = 0;
    int right = size - 1;
    int result = -1;

    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] <= target) {
            result = mid;
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return result;
}

__device__ float truncated_normal(
    curandState *state,
    const float mean,
    const float std,
    const float lower,
    const float upper
)
{
    float a = (lower - mean) / std;
    float b = (upper - mean) / std;

    // Uniform sample from truncated CDF range
    float u = curand_uniform(state);
    float p = 0.5f * (1.0f + erff(a / sqrtf(2.0f))) + u * (0.5f * (1.0f + erff(b / sqrtf(2.0f))) - 0.5f * (1.0f + erff(a / sqrtf(2.0f))));

    // Invert CDF using inverse error function
    return mean + std * sqrtf(2.0f) * erfinvf(2.0f * p - 1.0f);
}

extern "C" __global__ void generate_vectors(
    const uint8_t *seed,
    const int database_size,
    const int query_size,
    const int vector_dims,
    const int num_clusters,
    const float *cluster_cum_prob,
    const float *cluster_means,
    const float *cluster_stds,
    float *database_vectors,
    float *query_vectors
)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < database_size; i += blockDim.x * gridDim.x) 
    {   
        curandState state;
        curand_init(((uint64_t *)(seed))[i % 4], i, 0, &state);
        
        int cluster_idx = binary_search(cluster_cum_prob, curand_uniform(&state), num_clusters);
        const float *means = cluster_means + cluster_idx * vector_dims;
        const float *stds = cluster_stds + cluster_idx * vector_dims;

        float *vector = database_vectors + i * vector_dims;
        for (int j = 0; j < vector_dims; ++j) 
        {
            vector[j] = truncated_normal(
                &state, 
                means[j], 
                stds[j],
                -1.0f, 
                1.0f
            );
        }
    }

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < query_size; i += blockDim.x * gridDim.x) 
    {   
        curandState state;
        curand_init(((uint64_t *)(seed))[i % 4], database_size + i, 0, &state);
        
        int cluster_idx = binary_search(cluster_cum_prob, curand_uniform(&state), num_clusters);
        const float *means = cluster_means + cluster_idx * vector_dims;
        const float *stds = cluster_stds + cluster_idx * vector_dims;

        float *vector = query_vectors + i * vector_dims;
        for (int j = 0; j < vector_dims; ++j) 
        {
            vector[j] = truncated_normal(
                &state, 
                means[j], 
                stds[j],
                -1.0f, 
                1.0f
            );
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
