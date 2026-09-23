// IVF-Flat for c004 vector_search on the index-build ABI, written to be
// FUEL-METERED rather than to win a round.
//
// The build half is the way cuVS constructs an IVF-Flat index: k-means (Lloyd)
// over a training subset of the database, then one assignment pass over the
// whole database and CSR inverted lists. The search half is the IVF probe/scan
// from tig-pentesting's fixtures/vector_search_ann, unchanged except that the
// shared-memory padding admits every scenario's dims (up to 256), not only 128.
//
// Determinism, since the verifier regenerates and compares the SOLUTION: no
// float atomics anywhere. Initial centroids are a strided sample; the Lloyd
// update sums each list with one thread per dimension; the CSR scatter uses an
// integer atomic cursor, so ids WITHIN a list are unordered and the update's
// summation order is not reproducible across runs. Fuel is a count of executed
// instructions and does not depend on that order; the emitted solution is an
// argmin with a lowest-index tie-break, which is order-independent.

#include <cuda_runtime.h>
#include <float.h>

#define A_DB_TILE   8
#define A_CENT_TILE 32
// One more than the largest scenario's dims (nytimes_256): lanes of a warp
// index different centroids at the same k, so a power-of-two stride puts all
// 32 lanes in one shared-memory bank. An odd stride rotates them across banks.
// Static shared memory is (8 + 32) * 257 * 4 = 41,120 bytes, under the 48 KiB
// default.
#define A_PAD       257
#define UPD_THREADS 128

// ---------------------------------------------------------------------------
// Initial centroids: row c * (n_rows / n_cent) of the training set. No RNG.
// ---------------------------------------------------------------------------
extern "C" __global__ void ivfk_pick_centroids(
    const float* __restrict__ db, int n_rows, int dims, int n_cent,
    float* __restrict__ cent)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cent * dims) return;
    int c = i / dims, k = i % dims;
    long long stride = (long long)n_rows / (long long)n_cent;
    cent[i] = db[(long long)c * stride * dims + k];
}

// ---------------------------------------------------------------------------
// Assign the first n_rows vectors to their nearest centroid. 256 threads =
// A_DB_TILE(8) vectors x A_CENT_TILE(32) centroids, so the 32 threads sharing
// a vector are one warp and the reduction is a shuffle.
// Work: n_rows * n_cent * dims.
// ---------------------------------------------------------------------------
extern "C" __global__ void ivfk_assign(
    const float* __restrict__ db, int n_rows,
    const float* __restrict__ cent, int n_cent, int dims,
    unsigned int* __restrict__ assign)
{
    __shared__ float s_db[A_DB_TILE][A_PAD];
    __shared__ float s_cent[A_CENT_TILE][A_PAD];

    const int tid = threadIdx.x;
    const int dbi = tid / A_CENT_TILE;
    const int ci  = tid % A_CENT_TILE;
    const int db_base = blockIdx.x * A_DB_TILE;

    for (int i = tid; i < A_DB_TILE * dims; i += blockDim.x) {
        int r = i / dims, k = i % dims;
        int g = db_base + r;
        s_db[r][k] = (g < n_rows) ? db[(long long)g * dims + k] : 0.0f;
    }
    __syncthreads();

    float best = FLT_MAX;
    int   best_c = 0;

    for (int c0 = 0; c0 < n_cent; c0 += A_CENT_TILE) {
        for (int i = tid; i < A_CENT_TILE * dims; i += blockDim.x) {
            int r = i / dims, k = i % dims;
            int g = c0 + r;
            s_cent[r][k] = (g < n_cent) ? cent[(long long)g * dims + k] : 0.0f;
        }
        __syncthreads();

        int cg = c0 + ci;
        if (cg < n_cent) {
            float acc = 0.0f;
            for (int k = 0; k < dims; ++k) {
                float d = s_db[dbi][k] - s_cent[ci][k];
                acc = fmaf(d, d, acc);
            }
            if (acc < best || (acc == best && cg < best_c)) { best = acc; best_c = cg; }
        }
        __syncthreads();
    }

    for (int off = 16; off > 0; off >>= 1) {
        float od = __shfl_down_sync(0xffffffffu, best, off);
        int   oc = __shfl_down_sync(0xffffffffu, best_c, off);
        if (od < best || (od == best && oc < best_c)) { best = od; best_c = oc; }
    }
    if (ci == 0) {
        int g = db_base + dbi;
        if (g < n_rows) assign[g] = (unsigned int)best_c;
    }
}

// ---------------------------------------------------------------------------
// CSR inverted lists: count -> exclusive scan -> scatter. Integer atomics only.
// ---------------------------------------------------------------------------
extern "C" __global__ void ivfk_count(
    const unsigned int* __restrict__ assign, int n_rows,
    unsigned int* __restrict__ counts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_rows) atomicAdd(&counts[assign[i]], 1u);
}

// One block, one thread: n_cent is at most a few thousand, so a serial scan is
// microseconds and not worth the correctness risk of a hand-rolled parallel one.
extern "C" __global__ void ivfk_scan(
    const unsigned int* __restrict__ counts, int n_cent,
    unsigned int* __restrict__ offsets)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    unsigned int run = 0;
    for (int i = 0; i < n_cent; ++i) { offsets[i] = run; run += counts[i]; }
    offsets[n_cent] = run;
}

extern "C" __global__ void ivfk_scatter(
    const unsigned int* __restrict__ assign, int n_rows,
    const unsigned int* __restrict__ offsets,
    unsigned int* __restrict__ cursor,
    unsigned int* __restrict__ ids)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rows) return;
    unsigned int l = assign[i];
    unsigned int p = atomicAdd(&cursor[l], 1u);
    ids[offsets[l] + p] = (unsigned int)i;
}

// ---------------------------------------------------------------------------
// Lloyd update: one block per centroid, thread k sums dimension k over the
// list's members in list order. No float atomics. An empty list keeps its old
// centroid. Work: n_rows * dims.
// ---------------------------------------------------------------------------
extern "C" __global__ void ivfk_update(
    const float* __restrict__ db, int dims,
    const unsigned int* __restrict__ offsets,
    const unsigned int* __restrict__ ids,
    int n_cent, float* __restrict__ cent)
{
    int c = blockIdx.x;
    if (c >= n_cent) return;
    unsigned int s = offsets[c], e = offsets[c + 1];
    if (e == s) return;
    float inv = 1.0f / (float)(e - s);
    for (int k = threadIdx.x; k < dims; k += blockDim.x) {
        float acc = 0.0f;
        for (unsigned int t = s; t < e; ++t)
            acc += db[(long long)ids[t] * dims + k];
        cent[(long long)c * dims + k] = acc * inv;
    }
}

// ---------------------------------------------------------------------------
// Per query: the nprobe nearest centroids. Cost: n_q * n_cent * dims.
// Dynamic shared memory: (dims + n_cent) floats.
// ---------------------------------------------------------------------------
extern "C" __global__ void ivfk_probe(
    const float* __restrict__ q, int n_q,
    const float* __restrict__ cent, int n_cent, int dims,
    int nprobe, unsigned int* __restrict__ probes)
{
    extern __shared__ float smem[];
    float* s_q = smem;            // dims
    float* s_d = smem + dims;     // n_cent

    __shared__ float r_d[256];
    __shared__ int   r_c[256];

    const int qi = blockIdx.x;
    if (qi >= n_q) return;

    for (int k = threadIdx.x; k < dims; k += blockDim.x)
        s_q[k] = q[(long long)qi * dims + k];
    __syncthreads();

    for (int c = threadIdx.x; c < n_cent; c += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < dims; ++k) {
            float d = s_q[k] - cent[(long long)c * dims + k];
            acc = fmaf(d, d, acc);
        }
        s_d[c] = acc;
    }
    __syncthreads();

    for (int p = 0; p < nprobe; ++p) {
        float bd = FLT_MAX; int bc = 0;
        for (int c = threadIdx.x; c < n_cent; c += blockDim.x) {
            float v = s_d[c];
            if (v < bd || (v == bd && c < bc)) { bd = v; bc = c; }
        }
        r_d[threadIdx.x] = bd; r_c[threadIdx.x] = bc;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                if (r_d[threadIdx.x + s] < r_d[threadIdx.x] ||
                    (r_d[threadIdx.x + s] == r_d[threadIdx.x] &&
                     r_c[threadIdx.x + s] < r_c[threadIdx.x])) {
                    r_d[threadIdx.x] = r_d[threadIdx.x + s];
                    r_c[threadIdx.x] = r_c[threadIdx.x + s];
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            probes[(long long)qi * nprobe + p] = (unsigned int)r_c[0];
            s_d[r_c[0]] = FLT_MAX;   // exclude it from the next pass
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Scan the probed lists. One warp per candidate; lane l takes dims l, l+32, ...
// in a FIXED order, so the sum is reproducible.
// Cost: n_q * nprobe * (n_db / n_cent) * dims, on average.
// Dynamic shared memory: dims floats.
// ---------------------------------------------------------------------------
extern "C" __global__ void ivfk_search(
    const float* __restrict__ q, int n_q,
    const float* __restrict__ db, int dims,
    const unsigned int* __restrict__ offsets,
    const unsigned int* __restrict__ ids,
    const unsigned int* __restrict__ probes, int nprobe,
    unsigned long long* __restrict__ out)
{
    extern __shared__ float s_q[];
    __shared__ float        r_d[256];
    __shared__ unsigned int r_i[256];

    const int qi = blockIdx.x;
    if (qi >= n_q) return;

    for (int k = threadIdx.x; k < dims; k += blockDim.x)
        s_q[k] = q[(long long)qi * dims + k];
    __syncthreads();

    const int lane   = threadIdx.x & 31;
    const int warp   = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;

    float best = FLT_MAX;
    unsigned int best_i = 0u;   // a valid index, so an all-empty probe set still
                                // emits a well-formed (merely wrong) answer

    for (int p = 0; p < nprobe; ++p) {
        unsigned int l = probes[(long long)qi * nprobe + p];
        unsigned int s = offsets[l], e = offsets[l + 1];
        for (unsigned int t = s + warp; t < e; t += nwarps) {
            unsigned int id = ids[t];
            float acc = 0.0f;
            for (int k = lane; k < dims; k += 32) {
                float d = s_q[k] - db[(long long)id * dims + k];
                acc = fmaf(d, d, acc);
            }
            for (int off = 16; off > 0; off >>= 1)
                acc += __shfl_down_sync(0xffffffffu, acc, off);
            acc = __shfl_sync(0xffffffffu, acc, 0);
            if (acc < best || (acc == best && id < best_i)) { best = acc; best_i = id; }
        }
    }

    r_d[threadIdx.x] = best; r_i[threadIdx.x] = best_i;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (r_d[threadIdx.x + s] < r_d[threadIdx.x] ||
                (r_d[threadIdx.x + s] == r_d[threadIdx.x] &&
                 r_i[threadIdx.x + s] < r_i[threadIdx.x])) {
                r_d[threadIdx.x] = r_d[threadIdx.x + s];
                r_i[threadIdx.x] = r_i[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[qi] = (unsigned long long)r_i[0];
}
