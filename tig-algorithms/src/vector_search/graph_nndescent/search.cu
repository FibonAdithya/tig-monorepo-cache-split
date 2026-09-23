// Greedy beam search over a fixed-degree adjacency list, shared by the two
// graph-index probes (graph_nndescent, graph_nsw). One block per query, the
// query staged in shared memory, a sorted candidate list of at most `ef`
// entries, and a visited set (open-addressed hash in shared memory).
//
// Distances are computed one THREAD per candidate row with float4 loads: fuel
// counts executed instructions, and a warp-per-row lane-strided reduction
// costs several times more per row (32 lanes of address arithmetic plus a
// shuffle tree) for the same arithmetic. Every scenario's dims is a multiple
// of 4 and device rows are 16-byte aligned, so the float4 view is valid.
//
// Termination is the standard rule: stop when the best unexpanded candidate
// is farther than the ef-th kept result, or when nothing is left to expand.
// Deterministic given the graph: all list mutation is done by thread 0.

#include <cuda_runtime.h>
#include <float.h>

#define GS_THREADS    256
#define GS_MAX_DIMS   256
#define GS_MAX_DEGREE 64
#define GS_MAX_EF     512
#define GS_VIS        8192          // visited-set slots, power of two
#define GS_EMPTY      0xFFFFFFFFu   // empty adjacency slot / empty visited slot
#define GS_STARTS     4             // hash-chosen entry points per query

__device__ __forceinline__ unsigned long long gs_mix(unsigned long long z)
{
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

__device__ __forceinline__ unsigned long long gs_hash(unsigned long long seed, unsigned int a, unsigned int b)
{
    return gs_mix(seed ^ gs_mix(((unsigned long long)a << 32) | b));
}

// Squared L2 between a shared-memory query and a global row, both float4-aligned.
__device__ __forceinline__ float gs_dist(const float* __restrict__ s_q, const float* __restrict__ row, int dims)
{
    const float4* q4 = reinterpret_cast<const float4*>(s_q);
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

// Same, both rows in global memory.
__device__ __forceinline__ float gs_dist_gg(const float* __restrict__ a_row, const float* __restrict__ b_row, int dims)
{
    const float4* a4 = reinterpret_cast<const float4*>(a_row);
    const float4* b4 = reinterpret_cast<const float4*>(b_row);
    float acc = 0.0f;
    for (int k = 0; k < dims / 4; ++k) {
        float4 a = a4[k];
        float4 b = b4[k];
        float d0 = a.x - b.x, d1 = a.y - b.y, d2 = a.z - b.z, d3 = a.w - b.w;
        acc = fmaf(d0, d0, acc);
        acc = fmaf(d1, d1, acc);
        acc = fmaf(d2, d2, acc);
        acc = fmaf(d3, d3, acc);
    }
    return acc;
}

// Insert `w` into the visited set. Returns true when it was not there before.
// A full table rejects everything, which ends the search early rather than
// looping; GS_VIS is sized so that does not happen at the ef values used.
__device__ __forceinline__ bool gs_vis_insert(unsigned int* vis, unsigned int w)
{
    unsigned int h = (w * 2654435761u) & (GS_VIS - 1);
    for (int probe = 0; probe < GS_VIS; ++probe) {
        unsigned int old = atomicCAS(&vis[h], GS_EMPTY, w);
        if (old == GS_EMPTY) return true;
        if (old == w) return false;
        h = (h + 1) & (GS_VIS - 1);
    }
    return false;
}

// Sorted insert into the (cd, ci, cx) list, keeping at most `ef` entries.
// Thread 0 only.
__device__ __forceinline__ void gs_list_insert(
    float* cd, unsigned int* ci, unsigned char* cx, int* size, int ef,
    float d, unsigned int id)
{
    int n = *size;
    if (n >= ef && d >= cd[n - 1]) return;
    int pos = (n < ef) ? n : n - 1;   // slot that will be overwritten/shifted
    while (pos > 0 && cd[pos - 1] > d) {
        cd[pos] = cd[pos - 1];
        ci[pos] = ci[pos - 1];
        cx[pos] = cx[pos - 1];
        --pos;
    }
    cd[pos] = d;
    ci[pos] = id;
    cx[pos] = 0;
    if (n < ef) *size = n + 1;
}

// q: n_q rows of `dims` floats. adj: n_nodes x degree, GS_EMPTY for a missing
// edge. Writes the `out_k` nearest found ids per query (fewer than out_k found
// repeats the best). Only nodes < n_nodes are ever visited, which is what lets
// the NSW build search the already-inserted prefix of the same array.
extern "C" __global__ void graph_search(
    const float* __restrict__ q, int n_q,
    const float* __restrict__ db, int dims,
    const unsigned int* __restrict__ adj, int degree, int n_nodes,
    int ef, int out_k, unsigned long long seed,
    unsigned int* __restrict__ out)
{
    __shared__ __align__(16) float s_q[GS_MAX_DIMS];
    __shared__ float         cd[GS_MAX_EF];
    __shared__ unsigned int  ci[GS_MAX_EF];
    __shared__ unsigned char cx[GS_MAX_EF];
    __shared__ unsigned int  vis[GS_VIS];
    __shared__ float         nd[GS_MAX_DEGREE];
    __shared__ unsigned int  nid[GS_MAX_DEGREE];
    __shared__ int s_size, s_cur, s_stop;

    const int qi = blockIdx.x;
    if (qi >= n_q) return;
    const int tid = threadIdx.x;

    for (int k = tid; k < dims; k += blockDim.x) s_q[k] = q[(long long)qi * dims + k];
    for (int i = tid; i < GS_VIS; i += blockDim.x) vis[i] = GS_EMPTY;
    if (tid == 0) { s_size = 0; s_stop = 0; }
    __syncthreads();

    if (n_nodes <= 0) {
        if (tid < out_k) out[(long long)qi * out_k + tid] = GS_EMPTY;
        return;
    }

    // Entry points: GS_STARTS hash-chosen nodes of the searchable prefix.
    if (tid < GS_STARTS) {
        unsigned int e = (unsigned int)(gs_hash(seed, (unsigned int)qi, (unsigned int)tid) % (unsigned long long)n_nodes);
        if (gs_vis_insert(vis, e)) {
            nd[tid] = gs_dist(s_q, db + (long long)e * dims, dims);
            nid[tid] = e;
        } else {
            nd[tid] = FLT_MAX;
        }
    }
    __syncthreads();
    if (tid == 0) {
        for (int t = 0; t < GS_STARTS; ++t)
            if (nd[t] < FLT_MAX) gs_list_insert(cd, ci, cx, &s_size, ef, nd[t], nid[t]);
    }
    __syncthreads();

    const int max_it = 4 * ef + 8;
    for (int it = 0; it < max_it; ++it) {
        if (tid == 0) {
            int best = -1;
            for (int i = 0; i < s_size; ++i) {
                if (!cx[i]) { best = i; break; }   // list is sorted: first unexpanded is the best
            }
            if (best < 0 || (s_size >= ef && cd[best] > cd[s_size - 1])) {
                s_stop = 1;
            } else {
                cx[best] = 1;
                s_cur = (int)ci[best];
            }
        }
        __syncthreads();
        if (s_stop) break;

        if (tid < degree) {
            unsigned int w = adj[(long long)s_cur * degree + tid];
            if (w != GS_EMPTY && w < (unsigned int)n_nodes && gs_vis_insert(vis, w)) {
                nd[tid] = gs_dist(s_q, db + (long long)w * dims, dims);
                nid[tid] = w;
            } else {
                nd[tid] = FLT_MAX;
            }
        }
        __syncthreads();
        if (tid == 0) {
            for (int t = 0; t < degree; ++t)
                if (nd[t] < FLT_MAX) gs_list_insert(cd, ci, cx, &s_size, ef, nd[t], nid[t]);
        }
        __syncthreads();
    }

    if (tid < out_k) {
        unsigned int v = GS_EMPTY;
        if (s_size > 0) v = (tid < s_size) ? ci[tid] : ci[0];
        out[(long long)qi * out_k + tid] = v;
    }
}
