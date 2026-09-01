# Vector Range Search

[Vector range search (or vector search engine)](https://en.wikipedia.org/wiki/Vector_database) is the task where, given 2 sets of vectors with the same number of dimensions, a database set and a query set, find for each query vector a nearby vector in the database set, such that the mean distance between the query vectors and their corresponding vector in the database is within a threshold value.

# Example 

> This example illustrates the *task* — finding a near neighbour for each query.
> It is not how TIG scores a solution: see [Quality is recall, not mean
> distance](#quality-is-recall-not-mean-distance) below.

* 10 vectors in the database set.
* 2-dimensional space.
* 3 vectors in the query set.
* Mean distance threshold is $0.2$.
* Distance is Euclidean distance

```
vector_database = [
    [0.05, 0.16],
    [0.31, 0.74],
    [0.32, 0.8 ],
    [0.03, 0.25],
    [0.33, 0.07],
    [0.88, 0.77],
    [0.91, 0.29],
    [0.7 , 0.02],
    [0.53, 0.04],
    [0.72, 0.38]
]

query_vectors = [
    [0.89, 0.86],
    [0.26, 0.88],
    [0.17, 0.41]
]
```

The Euclidean distance from each query vector to the database set is approximately:
```
distances = [
    [1.09, 0.59, 0.57, 1.05, 0.97, 0.09, 0.57, 0.86, 0.9 , 0.51],
    [0.75, 0.15, 0.1 , 0.67, 0.81, 0.63, 0.88, 0.97, 0.88, 0.68],
    [0.28, 0.36, 0.42, 0.21, 0.38, 0.8 , 0.75, 0.66, 0.52, 0.55]
]
```

It can be seen that, for each query vector, if we select the following vectors in the database, the mean Euclidean distance will be below 0.2:
```
indexes = [
    5, // select vector 5 in database as "nearby" to query vector 0
    1, // select vector 1 in database as "nearby" to query vector 1
    0, // select vector 0 in database as "nearby" to query vector 2
]
total_distance = 0.09 + 0.1 + 0.28 = 0.47
mean_distance = 0.47 / 3 = 0.16
```

# Our Challenge

> **This challenge was redesigned on the `vector_search/gan_instance_gen` branch.**
> Three things changed: instances now come from a trained generator over a real
> embedding corpus instead of a synthetic hypercube, quality is **recall**
> instead of mean distance, and index construction is a separate phase that
> costs no solve fuel. The sections below describe the new design. For how to
> run it, see [Running it](#running-it).

In TIG, the vector search challenge uses the Euclidean distance over vectors of
**128 dimensions**. Instances are not sampled from a hypercube. They are
produced by a trained WGAN-GP generator whose weights ship with the challenge,
so that database and query vectors resemble real embedding data (SIFT-like)
rather than uniform noise. The motivation is transfer: an algorithm that wins
here should be good at vector search on real data.

## Scenarios replace a size parameter

A track is no longer a size. It is a **scenario** — one per real corpus:

```rust
Track { s: Scenario }        // tig-challenges/src/vector_search/mod.rs
enum Scenario { SIFT_128 }   // scenarios.rs
```

A scenario fixes every instance parameter, so there is exactly one shape of
instance per track:

| Field | `SIFT_128` |
|---|---|
| `n_queries` | 7,000 |
| `database_size` | 700,000 |
| `vector_dims` | 128 |
| `min_recall` | 0.95 |
| `recall_tolerance` | 1e-6 |
| `audit_samples` | 1,000 |

Adding a corpus means adding a `Scenario` variant, which is a runtime-side
change only — generation kernels come from each algorithm's PTX, so `kernels.cu`
must not change to add one. Changing it would force every algorithm on the
network to be rebuilt and resubmitted.

## Quality is recall, not mean distance

Your algorithm does not return a solution; it calls `save_solution` as it runs,
and the **last** saved solution is evaluated. A valid solution assigns each
query vector a database index.

Quality is **recall@1**, scaled to a fixed-point integer by
`QUALITY_PRECISION = 1_000_000`:

```rust
let recall = self.measure_recall(solution, audit_salt, module, stream, prop)?;
Ok((recall * QUALITY_PRECISION as f32).round() as i32)
```

A returned vector counts as a hit when its distance is within
`recall_tolerance` (1e-6, relative) of the true minimum. That tolerance absorbs
cross-architecture floating-point noise and makes an equidistant alternative a
hit by construction, rather than a spurious miss.

Recall is measured on a **salted subsample** of `audit_samples` (1,000) queries,
not on all 7,000 — verification cost is linear in that number. The subsample is
chosen by an `audit_salt` the solver cannot predict, which is what stops a
solution from being correct only on a known audit set. A solution must declare a
recall that clears `min_recall` (0.95) to qualify.

`min_recall = 0.95` is a floor with measured headroom beneath it, not above it:
measured floors for a viable bar are far lower (0.281 on an oracle worst-case
seed; 0.79 under an unmeasured near-tie hypothesis), so 0.95 clears both with
margin. The **upper** end is unmeasured — no ANN method has been run against it.
A cheap approximate method clearing 0.95 later is grounds to raise the bar, not
to reconsider the design.

## Index building is a separate, unpriced phase

Search structures are built once per precommit rather than once per nonce, and
the build is not charged against solve fuel.

The database is derived from a **database seed that carries no nonce**, so it is
fixed for a whole precommit; only the query set varies per nonce. The build runs
in a process that is never given a nonce at all, and therefore cannot derive the
queries it will later be asked about. It gets its own budget
(`--build-fuel`), its own device-memory cap, and its own wall-clock watchdog;
the resulting index blob is then loaded by each solve.

Two protocol-config keys govern the budget, both `Option`:
`build_fuel_alpha` and `max_build_fuel_budget`.

> **Neither may be set in protocol config yet.** The reference slave still runs
> one process per nonce, and `needs_index_build` is membership-based
> (`"build_fuel_budget" in batch`), so no batch the master produces today
> carries one. Setting `build_fuel_alpha` before the master is wired makes every
> c004 batch fail. See the design doc below.

## Running it

The two-phase flow, mounts, and every new flag are documented in
[`tig-runtime`'s README](../../../tig-runtime/README.md#c004-two-phase-index-build--solve).
The short version: `build-index` writes a blob, the per-nonce solve reads it
with `--index`, and the blob crosses between them **only** through the
`${RESULTS_DIR}:/app/results` bind mount that both containers share.

## Where the rest of the documentation lives

**For reviewing the design** — read in this order, each supersedes parts of the
one before:

1. `docs/superpowers/specs/2026-08-13-gan-instance-generation-design.md` — the
   generator, determinism, and the `## Validation` section with measured
   evidence. Validated on a single GPU architecture only.
2. `docs/superpowers/specs/2026-08-25-per-scenario-gan-tracks-design.md` —
   supersedes the track model above.
3. `docs/superpowers/specs/2026-08-31-c004-index-build-split-design.md` — the
   build/solve split. Carries a blocking warning; read it before touching
   config.
4. `docs/measurements/2026-08-31-c004-post-split-nonce-time.md` — the
   measurements that set `alpha` and the memory cap.
5. `docs/superpowers/2026-08-27-recall-gated-c004-monorepo-followups.md` — what
   the implementation found that the specs got wrong. Short, and the highest
   signal of the five.

The recall-gating design doc itself is **not in this repo** — it lives at
`docs/superpowers/specs/2026-08-27-recall-gated-c004-design.md` in the
`tig-pentesting` repo.

**Not review material:** everything under `docs/superpowers/plans/`. Those are
agent execution artifacts — task-by-task implementation plans, each opening with
a `> **For agentic workers:**` banner and using `- [ ]` checkboxes. They record
how the work was carried out, not what was decided or why. Nothing in them is
required reading to review the challenge.

# Application

Vector search has a wide range of applications an example of which is Threshold-Based Anomaly Detection, where the vector database represents operational data in a high-dimensional space, and query vectors represent new incoming data points to be monitored for anomalies. If the average distance exceeds a predefined threshold, the query vectors are flagged as anomalies. 

See also for example Outlier detection for high dimensional data: https://dl.acm.org/doi/abs/10.1145/375663.375668

Another example application of vector search is in network security, where the vector database corresponds to historical traffic patterns, and query vectors are new traffic data. By tracking the mean distance between sets new data points and historic "regular" data, any deviation exceeding a threshold can indicate a potential intrusion.
