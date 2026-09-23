// NN-descent k-NN graph build (the CAGRA build path), written to be fuel-metered.
//
// Every node keeps a sorted list of its `k` best neighbours found so far. A
// round performs the local join: for node u, every neighbour-of-neighbour w is
// a candidate, and it is evaluated only when the edge to the neighbour v or v's
// edge to w is NEW (found in the previous round), which is the incremental
// rule that lets NN-descent converge in a handful of rounds instead of paying
// n * k^2 distances every round. Lists are updated in place; another block may
// read a list mid-write, but every value it reads is a valid node id, so the
// only effect is on which candidates that block sees.
//
// Flags ride in the top bits of the id: NEW marks last round's insertions,
// PENDING marks this round's; nnd_flags rotates PENDING -> NEW between rounds
// and nnd_strip removes both before the graph is serialised.
//
// Not deterministic across runs (in-place updates race) -- fuel is a count of
// executed instructions and does not care; a shipped algorithm would need a
// double-buffered update.

#include <cuda_runtime.h>
#include <float.h>

#define ND_THREADS  256
#define ND_MAX_K    64
#define ND_MAX_DIMS 256
#define ND_NEW      0x80000000u
#define ND_PENDING  0x40000000u
#define ND_MASK     0x3FFFFFFFu

__device__ __forceinline__ unsigned long long nd_mix(unsigned long long z)
{
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

__device__ __forceinline__ float nd_dist(const float* __restrict__ s_u, const float* __restrict__ row, int dims)
{
    const float4* q4 = reinterpret_cast<const float4*>(s_u);
    const float4* r4 = reinterpret_cast<const float4*>(row);
    float acc = 0.0f;
    for (int k = 0; k < dims / 4; ++k) {
        float4 a = q4[k];
        float4 b = r4[k];
        float d0 = a.x - b.x, d1 = a.y - b.y, d2 = a.z - b.z, d3 = a.w - b.w;
        acc = fmaf(d0, d0, acc);
        acc = fmaf(d1, d1, acc);
        acc = fmaf(d2, d2, acc);
        acc = fmaf(d3, d3, acc);
    }
    return acc;
}

// ---------------------------------------------------------------------------
// Random initial lists: k hash-chosen distinct-from-self ids per node, with
// their distances, sorted. Every entry is NEW so round 1 is a full join.
// One block per node. Work: n * k * dims.
// ---------------------------------------------------------------------------
extern "C" __global__ void nnd_init(
    const float* __restrict__ db, int n, int dims, int k, unsigned long long seed,
    unsigned int* __restrict__ adj, float* __restrict__ dst)
{
    __shared__ __align__(16) float s_u[ND_MAX_DIMS];
    __shared__ float        s_d[ND_MAX_K];
    __shared__ unsigned int s_i[ND_MAX_K];

    const int u = blockIdx.x;
    if (u >= n) return;
    const int tid = threadIdx.x;

    for (int j = tid; j < dims; j += blockDim.x) s_u[j] = db[(long long)u * dims + j];
    __syncthreads();

    if (tid < k) {
        unsigned int c = (unsigned int)(nd_mix(seed ^ nd_mix(((unsigned long long)u << 32) | (unsigned int)tid)) % (unsigned long long)n);
        if (c == (unsigned int)u) c = (c + 1u) % (unsigned int)n;
        s_d[tid] = nd_dist(s_u, db + (long long)c * dims, dims);
        s_i[tid] = c;
    }
    __syncthreads();

    if (tid == 0) {
        // Insertion sort of k entries, then write with the NEW flag.
        for (int i = 1; i < k; ++i) {
            float d = s_d[i]; unsigned int id = s_i[i];
            int p = i;
            while (p > 0 && s_d[p - 1] > d) { s_d[p] = s_d[p - 1]; s_i[p] = s_i[p - 1]; --p; }
            s_d[p] = d; s_i[p] = id;
        }
        for (int i = 0; i < k; ++i) {
            adj[(long long)u * k + i] = s_i[i] | ND_NEW;
            dst[(long long)u * k + i] = s_d[i];
        }
    }
}

// ---------------------------------------------------------------------------
// One local-join round. One block per node u; thread c evaluates candidates
// c, c+256, ... of the k*k (neighbour j, its neighbour t) pairs, skipping pairs
// where neither edge is NEW; thread 0 merges each batch of 256 results into
// u's sorted list, deduplicating on insertion. Work per node, first round:
// k*k*dims; later rounds: only the pairs that involve a NEW edge.
// ---------------------------------------------------------------------------
extern "C" __global__ void nnd_round(
    const float* __restrict__ db, int n, int dims, int k,
    unsigned int* __restrict__ adj, float* __restrict__ dst)
{
    __shared__ __align__(16) float s_u[ND_MAX_DIMS];
    __shared__ float        s_ld[ND_MAX_K];
    __shared__ unsigned int s_li[ND_MAX_K];   // with flags
    __shared__ float        s_cd[ND_THREADS];
    __shared__ unsigned int s_ci[ND_THREADS];

    const int u = blockIdx.x;
    if (u >= n) return;
    const int tid = threadIdx.x;

    for (int j = tid; j < dims; j += blockDim.x) s_u[j] = db[(long long)u * dims + j];
    if (tid < k) {
        s_li[tid] = adj[(long long)u * k + tid];
        s_ld[tid] = dst[(long long)u * k + tid];
    }
    __syncthreads();

    const int total = k * k;
    for (int base = 0; base < total; base += ND_THREADS) {
        int c = base + tid;
        float d = FLT_MAX;
        unsigned int w = 0;
        if (c < total) {
            int j = c / k, t = c - j * k;
            unsigned int ev = s_li[j];
            unsigned int v = ev & ND_MASK;
            unsigned int ew = adj[(long long)v * k + t];
            w = ew & ND_MASK;
            if (((ev | ew) & ND_NEW) && w != (unsigned int)u)
                d = nd_dist(s_u, db + (long long)w * dims, dims);
        }
        s_cd[tid] = d;
        s_ci[tid] = w;
        __syncthreads();
        if (tid == 0) {
            for (int i = 0; i < ND_THREADS; ++i) {
                float cd = s_cd[i];
                if (cd >= s_ld[k - 1]) continue;
                unsigned int cid = s_ci[i];
                bool dup = false;
                for (int q = 0; q < k; ++q) if ((s_li[q] & ND_MASK) == cid) { dup = true; break; }
                if (dup) continue;
                int p = k - 1;
                while (p > 0 && s_ld[p - 1] > cd) { s_ld[p] = s_ld[p - 1]; s_li[p] = s_li[p - 1]; --p; }
                s_ld[p] = cd;
                s_li[p] = cid | ND_PENDING;
            }
        }
        __syncthreads();
    }

    if (tid < k) {
        adj[(long long)u * k + tid] = s_li[tid];
        dst[(long long)u * k + tid] = s_ld[tid];
    }
}

// PENDING -> NEW, NEW -> old. Work: n * k.
extern "C" __global__ void nnd_flags(unsigned int* __restrict__ adj, int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    unsigned int e = adj[i];
    unsigned int id = e & ND_MASK;
    adj[i] = (e & ND_PENDING) ? (id | ND_NEW) : id;
}

// Remove every flag before serialisation.
extern "C" __global__ void nnd_strip(unsigned int* __restrict__ adj, int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    adj[i] &= ND_MASK;
}
