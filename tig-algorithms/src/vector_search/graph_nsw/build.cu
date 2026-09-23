// Batched NSW (single-layer HNSW) incremental insertion, written to be
// fuel-metered.
//
// Nodes are inserted in batches whose size doubles from 1 up to `batch`. Each
// batch node runs the shared beam search (search.cu) over the already-inserted
// prefix with ef = ef_construction and keeps its m nearest as out-edges; then
// nsw_link adds the reverse edges under a per-node spin lock, replacing the
// worst edge when the target's list is full (the simple variant of HNSW's
// neighbour selection, without the diversity heuristic). The hierarchy of
// HNSW is not modelled: its upper layers hold a vanishing fraction of the
// nodes and of the build work.
//
// Not deterministic across runs (reverse-edge insertion order races); fuel is
// a count of executed instructions and does not care.

#include <cuda_runtime.h>
#include <float.h>

#define NSW_EMPTY 0xFFFFFFFFu

__device__ __forceinline__ float nsw_dist_gg(const float* __restrict__ a_row, const float* __restrict__ b_row, int dims)
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

extern "C" __global__ void nsw_init(
    unsigned int* __restrict__ adj, float* __restrict__ adjd, int total,
    unsigned int* __restrict__ count, unsigned int* __restrict__ lock, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) { adj[i] = NSW_EMPTY; adjd[i] = FLT_MAX; }
    if (i < n) { count[i] = 0; lock[i] = 0; }
}

// One thread per batch node u = start + i. `found` holds the search results
// (n_batch x m, NSW_EMPTY where fewer were found). Forward edges are written
// without a lock (only this thread touches u's list); reverse edges take the
// target's lock. Work: n_batch * m * dims for the distances, plus the list
// scans.
extern "C" __global__ void nsw_link(
    const float* __restrict__ db, int dims, int m, int start, int n_batch,
    const unsigned int* __restrict__ found,
    unsigned int* __restrict__ adj, float* __restrict__ adjd,
    unsigned int* __restrict__ count, unsigned int* __restrict__ lock)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_batch) return;
    const unsigned int u = (unsigned int)(start + i);
    const float* u_row = db + (long long)u * dims;

    unsigned int nf = 0;
    for (int t = 0; t < m; ++t) {
        unsigned int v = found[(long long)i * m + t];
        if (v == NSW_EMPTY) break;
        float d = nsw_dist_gg(u_row, db + (long long)v * dims, dims);
        adj[(long long)u * m + nf] = v;
        adjd[(long long)u * m + nf] = d;
        ++nf;
    }
    count[u] = nf;

    for (unsigned int t = 0; t < nf; ++t) {
        unsigned int v = adj[(long long)u * m + t];
        float d = adjd[(long long)u * m + t];
        bool done = false;
        while (!done) {
            if (atomicCAS(&lock[v], 0u, 1u) == 0u) {
                unsigned int c = count[v];
                if (c < (unsigned int)m) {
                    adj[(long long)v * m + c] = u;
                    adjd[(long long)v * m + c] = d;
                    count[v] = c + 1;
                } else {
                    int worst = 0;
                    float wd = adjd[(long long)v * m];
                    for (int s = 1; s < m; ++s) {
                        float sd = adjd[(long long)v * m + s];
                        if (sd > wd) { wd = sd; worst = s; }
                    }
                    if (d < wd) {
                        adj[(long long)v * m + worst] = u;
                        adjd[(long long)v * m + worst] = d;
                    }
                }
                __threadfence();
                atomicExch(&lock[v], 0u);
                done = true;
            }
        }
    }
}
