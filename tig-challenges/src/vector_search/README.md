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

In TIG, the vector search challenge uses Euclidean distance. Instances are not
sampled from a hypercube. They are produced by a trained WGAN-GP generator
whose weights ship with the challenge, so that database and query vectors
resemble real embedding data — SIFT, NYTimes or GloVe, depending on the
scenario — rather than uniform noise. The motivation is transfer: an algorithm
that wins here should be good at vector search on real data.

Vector dimensionality is **not fixed** across the challenge: it is 100, 128 or
256, depending on the scenario (see the table below). Algorithms must read
`challenge.vector_dims` at runtime rather than assume a fixed width. A
hardcoded 128 passes the `sift_128` track and fails the other two.

## Scenarios replace a size parameter

A track is no longer a size. It is a **scenario** — one per real corpus:

```rust
Track { s: Scenario }                                    // tig-challenges/src/vector_search/mod.rs
enum Scenario { SIFT_128, GLOVE_100, NYTIMES_256 }        // scenarios.rs
```

A scenario fixes every instance parameter, so there is exactly one shape of
instance per track:

| Field | `sift_128` | `glove_100` | `nytimes_256` |
|---|---|---|---|
| generator architecture | `structured_gate` (SIFT v4) | `mlp` (GloVe v1) | `spherical` (NYTimes v3) |
| `vector_dims` | 128 | 100 | 256 |
| `n_queries` | 7,000 | 7,000 | 7,000 |
| `database_size` | 700,000 | 700,000 | 700,000 |
| `min_recall` | 0.9 | 0.9 | 0.9 |
| `recall_tolerance` | 1e-6 | 1e-6 | 1e-6 |
| `audit_samples` | 1,000 | 1,000 | 1,000 |

The `min_recall = 0.9` bar was derived from measurements taken on SIFT
v1-generated instances (see below). It has **not** been measured for the
`nytimes_256` or `glove_100` corpora, nor re-measured for `sift_128` on the v4
instances this scenario now generates; all three currently share the same
placeholder value. See the spec's Follow-ups.

All three scenarios produce unit-norm rows. `sift_128` rows are additionally
non-negative, with about 24% of coordinates exactly zero (MEASURED 0.2390 on a
50,000-row dump; see the measurement note below) — a property of the
structured-gate architecture's stochastic hard gate. `nytimes_256` and
`glove_100` are angular corpora: what their underlying embeddings compare by is
cosine similarity, not Euclidean distance. Because the generator outputs are
unit-norm for both, Euclidean 1-NN and angular 1-NN return the same neighbour
(for unit vectors `u, v`: `‖u−v‖² = 2 − 2·cos(u,v)`, a decreasing function of
cosine similarity), which is why the challenge's Euclidean audit is valid for
them. `sift_128` is a Euclidean corpus to begin with, so nothing here is
needed to justify it.

Adding a corpus means adding a `Scenario` variant. Whether that touches
`kernels.cu` depends on whether the new corpus's generator uses an
architecture the kernels already support. Generation kernels come from each
algorithm's own PTX, not from the runtime, so any change to `kernels.cu`
forces every algorithm on the network to be rebuilt and resubmitted.

Three kernels never change: `gan_sample_latents` (draws latents),
`gan_linear` (every dense product in every architecture, including the baked
matrices below), and `evaluate_total_distance`. This change adds five more,
plus a device helper, to support the two new architectures:

| kernel | computes | used by |
|---|---|---|
| `gan_row_normalize` | `x / max(‖x‖, eps)` in place | mlp |
| `gan_film_leaky` | `leaky(a · (1 + γ) + β)`, element-wise | spherical |
| `gan_sphere_combine` | project out the tangent component, then `out = cos_r·u + sin_r·t` | spherical |
| `gan_gate_noise` | per-coordinate logistic gate noise from a uniform draw (device helper `gate_noise_from_uniform`) | structured_gate |
| `gan_gate_apply` | the gate's logit, open/closed decision, all-closed fallback, and the final normalise | structured_gate |

Because generation kernels come from each algorithm's PTX, this change forces
every algorithm on the network to be rebuilt and resubmitted **once**. After
that resubmit, a new scenario whose generator uses one of these three
architectures (`mlp`, `structured_gate`, `spherical`) is a runtime-side change
only — a new blob and a new `Scenario` variant, nothing in `kernels.cu`. A new
architecture that needs an operation none of these kernels compute forces
another resubmit.

## Weight blobs

Each scenario's generator weights ship as a committed binary blob under
`weights/`; they are never fetched at runtime, so every verifier regenerates
the same instance from the same bytes. There are two blob formats:

- `TIGGAN01` — the original single-MLP format. Only `v1_sift.bin` uses it now,
  kept as the `TIGGAN01`-parser and MLP-reference-forward-pass test fixture
  (`gan_generator::v1`, under `#[cfg(test)]`). It is no longer wired to any
  scenario.
- `TIGGAN02` — the current format: an architecture tag (`0` = mlp,
  `1` = structured_gate, `2` = spherical), a handful of scalars, and a
  sequence of positional weight tensors whose order and shape relations are
  fixed per architecture. `sift_128_v4.bin`, `glove_100_v1.bin` and
  `nytimes_256_v3.bin` all use it.

Provenance — checkpoint path and hash, blob hash, the WGAN-repo commit, and the
exact exporter command — is recorded per blob in `weights/PROVENANCE.md`.

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
recall that clears `min_recall` (0.9) to qualify.

The audit kernel stages the queries it is checking in a fixed-width
shared-memory buffer. That buffer widened from 128 to 256 dimensions to fit
`nytimes_256`'s rows, at no measurable cost to SIFT's audit: 87, 87, 87 ms at
128 dims against 88, 88, 87 ms at 256 dims, MEASURED on an RTX 3060 Ti (see the
comment above the row-wise kernels in `kernels.cu`, and the measurement note in
the reading order below).

`min_recall = 0.9` is a floor with measured headroom beneath it, not above it:
measured floors for a viable bar are far lower (0.281 on an oracle worst-case
seed; 0.79 under an unmeasured near-tie hypothesis). With the audit's
one-sided 3-sigma tolerance at 1,000 samples, 0.9 has a worst-case pass point
of 0.8715 — still above the unmeasured 0.79 floor, so dropping the bar to 0.9
does not reach either floor. The **upper** end is unmeasured — no ANN method
has been run against it. A cheap approximate method clearing 0.9 later is
grounds to raise the bar, not to reconsider the design.

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

## Known limitation: NYTimes gate acceptance

Measured against the WGAN repository's own realism gates, the SIFT and GloVe
generators pass on all four gate statistics. The NYTimes generator fails one,
`ivf_gini` (0.7246, against the gate's band of [0.7602, 0.8403]). This is not a
defect in this port: five of six PyTorch samples of the very checkpoint being
shipped also fail the same statistic, so the shortfall is a property of the
generator and its gate, not of the GPU port. The decision about the NYTimes
generator and its gate band belongs to the WGAN repository, not to this one.
See `docs/measurements/2026-09-21-c004-multi-arch-generators.md`, "WGAN gate
acceptance" and "The NYTimes investigation", for the full measurement.

## Where the rest of the documentation lives

**For reviewing the design** — read in this order, each supersedes parts of the
one before:

1. `docs/ai/specs/2026-08-13-gan-instance-generation-design.md` — the
   generator, determinism, and the `## Validation` section with measured
   evidence. Validated on a single GPU architecture only.
2. `docs/ai/specs/2026-08-25-per-scenario-gan-tracks-design.md` —
   supersedes the track model above.
3. `docs/ai/specs/2026-08-31-c004-index-build-split-design.md` — the
   build/solve split. Carries a blocking warning; read it before touching
   config.
4. `docs/ai/specs/2026-09-21-multi-arch-gan-scenarios-design.md` —
   supersedes one decision of the 2026-08-25 spec: the single-MLP kernel ABI
   freeze. Adds the `structured_gate` and `spherical` architectures, the
   `TIGGAN02` blob format, and the five kernels listed above.
5. `docs/measurements/2026-09-21-c004-multi-arch-generators.md` — the
   measurements behind item 4: WGAN gate acceptance, generation time, the
   audit-width timing, and the mutation checks.
6. `docs/measurements/2026-08-31-c004-post-split-nonce-time.md` — the
   measurements that set `alpha` and the memory cap.
7. `docs/ai/2026-08-27-recall-gated-c004-monorepo-followups.md` — what
   the implementation found that the specs got wrong. Short, and the highest
   signal of the seven.

The recall-gating design doc itself is **not in this repo** — it lives at
`docs/superpowers/specs/2026-08-27-recall-gated-c004-design.md` in the
`tig-pentesting` repo.

**Not review material:** everything under `docs/ai/plans/`. Those are
agent execution artifacts — task-by-task implementation plans, each opening with
a `> **For agentic workers:**` banner and using `- [ ]` checkboxes. They record
how the work was carried out, not what was decided or why. Nothing in them is
required reading to review the challenge.

# Application

Vector search has a wide range of applications an example of which is Threshold-Based Anomaly Detection, where the vector database represents operational data in a high-dimensional space, and query vectors represent new incoming data points to be monitored for anomalies. If the average distance exceeds a predefined threshold, the query vectors are flagged as anomalies. 

See also for example Outlier detection for high dimensional data: https://dl.acm.org/doi/abs/10.1145/375663.375668

Another example application of vector search is in network security, where the vector database corresponds to historical traffic patterns, and query vectors are new traffic data. By tracking the mean distance between sets new data points and historic "regular" data, any deviation exceeding a threshold can indicate a potential intrusion.
