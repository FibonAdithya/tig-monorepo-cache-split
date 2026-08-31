# c004 post-split per-nonce time, `alpha`, and the memory cap

**Date:** 2026-08-31
**Measures:** `docs/superpowers/specs/2026-08-31-c004-index-build-split-design.md`
**Code under test:** branch `vector_search/gan_instance_gen` at `d0487f97`
(byte-identical in code to `be650a7c`; `be650a7c..d0487f97` is
`docs/superpowers/specs/...design.md` only — `git diff --stat be650a7c d0487f97`
shows exactly one file changed, 519 insertions and 17 deletions, and no source
file at all).

This note replaces the design's **estimate** of 0.05 s/nonce, and the 1.2 s/nonce
figure it is measured against, with measurements. It also revises `alpha` and
reports what the memory and lifespan measurements can and cannot settle.

---

## 0. Summary of what changed

| Quantity | Design said | Measured here | Where |
|---|---|---|---|
| post-split per-nonce (marginal) | 0.05 s (estimated) | **0.0227 s** | §3 |
| post-split per-nonce (bundle-amortised, the number that matters) | — | **0.165 s** | §3.3 |
| pre-split per-nonce | 1.2 s (unsourced, see §2) | **3.24 s** | §2 |
| break-even for a 600 s build | ~520 nonces | **195 nonces** | §4 |
| `alpha` | 0.25 (provisional) | **0.005** | §5 |
| `max_build_fuel_budget` | unset | **1.0e13** suggested | §5.4 |
| build-phase peak device memory | 8 GiB cap assumed | **1010 MiB** with a zero-footprint index | §6 |
| build vs. precommit lifespan | open question | 600 s of **7200 s** (8.3 %) | §7 |

Two findings matter more than any single number:

1. **The verifier is not amortised and now dominates.** The reference slave runs
   one `tig-verifier` process per nonce, and that process regenerates the whole
   700,000-vector database from scratch. Measured at **3.29 s/nonce**, unchanged
   by the split. The benchmarker's end-to-end per-nonce cost therefore falls from
   6.53 s to 3.46 s — **1.9x**, not the ~24x that 1.2 s → 0.05 s implies. See §8.
2. **`alpha` cannot be 0.25.** `build_fuel_budget = alpha * num_nonces *
   fuel_budget` is denominated in `fuel_budget`, which the *player* chooses up to
   `max_fuel_budget` = 5e12 at **zero marginal fee** (`per_nonce_fee` is 0 for
   c004). A real algorithm consumes ~4.67e9 fuel, so the budget can be ~1,071x
   the fuel a nonce actually spends. At `alpha` = 0.25 and `fuel_budget` = 5e12
   the build is authorised 4,791 s of wall-clock at the 80-nonce protocol floor,
   against 246 s of saving. Only the flat 600 s watchdog prevents that. See §5.

---

## 1. Hardware, software, and what was solving

### 1.1 The rig — and the three caveats it forces

| | |
|---|---|
| GPU | **NVIDIA GeForce RTX 3060, 12 GB (12,288 MiB), sm_86** |
| Device memory as `cuMemGetInfo` reports it | 12,488,343,552 B (11,910 MiB) |
| CUDA | 12.6 |
| Rust | `nightly-2025-02-10` |
| Container | `tig-dev-vector_search` (Ubuntu 24.04, glibc 2.39), `--gpus all`, repo bind-mounted |
| Host | `tig-gpu` (vast.ai), glibc 2.35 |

Three consequences, stated plainly rather than buried:

- **This is not the deployment target.** The spec sizes the memory cap against
  the weakest listed `ComputeType`, `AWS_G4dn` — a 16 GB T4. Everything in §6 is
  a 3060 measurement. It can *rule the 8 GiB cap out*; it cannot confirm the cap
  fits a T4's budget. §6.4 says exactly which half is measurement and which is
  arithmetic.
- **This is not the card the old baseline was taken on.** The design's 1165 ms /
  1171 ms figures are attributed to an RTX 4060. **Rather than mix cards, the
  pre-split baseline was re-measured here on the 3060** (§2). Every `alpha` and
  break-even number in this note uses the 3060 baseline against the 3060
  post-split figure. The 4060 numbers are quoted for context only and are used
  in no arithmetic.
- **A T4 is slower than a 3060** (T4 ~8.1 TFLOPS fp32 / 320 GB/s; RTX 3060 ~12.7
  TFLOPS / 360 GB/s). Where a rate measured here is converted into a protocol
  constant, it is derated by 1.25x for the T4. That derate is an estimate from
  published specifications, not a measurement.

### 1.2 What was solving — this is load-bearing

Task 7 established by measurement that **no unmodified mainnet c004 algorithm
runs on this branch at all**: the GAN rewrite of `kernels.cu` renamed the
generation kernels, so every pre-GAN PTX dies with `CUDA_ERROR_NOT_FOUND` before
reaching any timing loop. So two purpose-built subjects were used. Both were
built with `build_so` / `build_ptx` inside the container, so both carry real LLVM
fuel instrumentation.

**`nullstub`** — a do-nothing solver written for this measurement.
`solve_challenge` calls `save_solution(&Solution::new())` and returns;
`build_index` returns a 3-byte blob; `load_index` ignores it. Its measured
`fuel_consumed` is **0** in every output file, which is the evidence that it
contributes no metered work at all.

> **What a `nullstub` number is.** It is the *floor*: everything `tig-runtime`
> and `tig-challenges` do for one nonce with an algorithm that does nothing.
> Per nonce that is the device-to-device copy of the database into `Challenge`
> (decision D5), `generate_queries` for 7,000 queries, `initialize_kernel`,
> `finalize_kernel`, and the output-file write.
>
> **What it is not.** It is not a production nonce. A real algorithm adds its
> search time on top. Crucially, though, the search term cancels out of the
> break-even and `alpha` arithmetic (§4.1), so the floor is the right thing to
> measure for this design's economic question.

**`refsearch`** — the brute-force exact 1-NN reference solver that already
existed on the box as untracked scratch (read, not modified). One kernel,
700,000 x 7,000 x 128 distances. Used for two things only: an anchor for "a real
solver actually solving" (its solutions verify at quality 1,000,000), and the
fuel-to-wall-clock rate in §5. It is **not** representative of a production
algorithm: it is ~112x more expensive in fuel than `there_v10` and uses no index.

Scenario: this branch has exactly **one** scenario, `SIFT_128` (7,000 queries,
700,000 database rows, 128 dims, `min_recall` 0.95), so `track_id` is
`s=sift_128` throughout. It corresponds in size to mainnet's `n_queries=7000`
track, whose `num_nonces_per_bundle` is **20** — the bundle size used in §3.3.
The other live tracks (`n_queries=9000..15000`, bundle sizes 17/15/10/5) have no
counterpart on this branch and were **not** measured; see §9.

### 1.3 How time was measured

`tig-runtime` and `tig-challenges` were given `eprintln!` probes reading
`SystemTime::now()` at: process entry; after `CudaContext::new` + `load_module` +
`get_device_prop`; after `Database::generate`; and at the end of every nonce
iteration. `cuMemGetInfo` was printed at the same points. **These probes exist
only in an isolated `/tmp/task8/repo` copy on the box; nothing was committed and
`/workspace/tig-bench` was never touched** (verified: it is still at `d53ceef4`
with exactly the pre-existing scratch it had before this work).

One measurement per process invocation, looping in the shell — a fresh
`CudaContext` per iteration inside one process exhausts the card after ~10
iterations, and surfaces as `NVRM: Out of memory` in `dmesg`, not as a panic.

---

## 2. The pre-split baseline, re-measured on this card

**The design's 1165 ms / 1171 ms figures cannot be traced to their cited source.**
The spec attributes them to the `## Validation` section of
`2026-08-13-gan-instance-generation-design.md`. `grep -rn "1165\|1171" docs/`
returns hits only in the split design and its plan — **no occurrence anywhere in
the GAN design document**. What that document actually records at
`n_queries=7000` is "combined runtime+verifier time for generation plus
evaluation is ~2.3 s per nonce", and a *pre-GAN* table of 520–643 ms
(`tig-runtime`) / 455–521 ms (`tig-verifier`). 1165 and 1171 look like ~2.3 s
halved, which would make them a derived split of a combined figure rather than a
measurement of the runtime alone. **Treat `t_old = 1.2 s` as unsourced.**

So the baseline was re-measured, on this card, on this branch, in the shape the
production benchmarker actually uses: `tig-benchmarker/slave/main.py:60`
(`run_tig_runtime`) launches **one `tig-runtime` process per nonce** via
`docker exec`, in the legacy four-positional form. Twenty consecutive processes,
`nullstub`, nonces 0–19:

| | mean | sd | min | max |
|---|---|---|---|---|
| full process wall (as the shell sees it, n=20) | **3238.8 ms** | 133.0 | 3131.7 | 3646.2 |
| in-process (entry -> solution written) | 2846.8 ms | 138.4 | 2726.0 | 3263.8 |

Breakdown of the in-process figure: 2055.4 ms of process start + `dlopen` + CUDA
context + PTX patch and JIT; 763.7 ms of `Database::generate`; 27.6 ms of
per-nonce work.

**`t_old` = 3.24 s/nonce** (the shell-observed figure, which is what the slave
pays). This excludes `tig-verifier`, which the slave also runs per nonce; the
verifier is measured separately in §8 and appears identically on both sides of
the comparison.

That is 2.8x the design's 1.2 s. Part of that is the slower card, but 2055 ms of
it is fixed per-process startup dominated by JIT of the challenge PTX, which the
1.2 s figure may never have included. This is the reason the baseline was
re-measured instead of being carried across cards.

---

## 3. Post-split per-nonce time

`tig-runtime batch <settings> <rand_hash> nullstub.so --ptx nullstub.ptx
--start-nonce 0 --num-nonces N --fuel 2000000000 --index /out/idx.blob --output D`

N ∈ {1, 2, 5, 10, 20, 50, 100}, three independent repetitions each (21 processes,
543 marginal per-nonce intervals). Every run exited 0 and wrote N well-formed
output files.

### 3.1 The marginal per-nonce cost

Consecutive `nonce_done` timestamps within one process:

| statistic | value |
|---|---|
| n | 543 |
| **mean** | **22.711 ms** |
| sd | 0.480 ms |
| min | 21.427 ms |
| median | 22.672 ms |
| p95 | 23.671 ms |
| max | 24.773 ms |

The spread is 2.1 % of the mean and shows no drift with N: the per-run means of
the 18 runs that have a marginal interval (N >= 2) lie between 22.36 and
23.21 ms, with no trend in N. An ordinary-least-squares fit
of in-process total against N over all 21 runs gives slope **21.50 ms/nonce**,
intercept 2882 ms — consistent with the direct interval measurement.

**The design's estimate was 0.05 s. The measured marginal cost is 0.0227 s —
2.2x better than estimated.**

### 3.2 The one-off cost, and `Database::generate` separated from it

The brief asks for `Database::generate` separated from the per-nonce cost. The
probes give it directly rather than by subtraction:

| phase | mean | sd | n |
|---|---|---|---|
| process start + `dlopen` + `CudaContext::new` + PTX JIT | 2077.7 ms | 160.5 | 21 |
| **`Database::generate`** (700,000 x 128, + `load_index`) | **765.6 ms** | 5.2 | 21 |
| **total one-off, paid once per batch** | **2843.3 ms** | | |

`Database::generate` is remarkably stable (sd 0.7 %). The startup term is the
variable one, and it is the larger of the two: **73 % of the one-off cost is not
database generation at all, it is process start and PTX JIT.** That matters for
D4 — batching pays for itself on process startup before it pays for anything the
design talks about.

Cross-check from the `build-index` path, which generates the same database in a
different code path: 750.1 ms and 751.2 ms in the two runs of §6. Agrees with
765.6 ms to within 2 %.

### 3.3 The number that should be used: 0.165 s/nonce

D4 puts **one bundle** in one batched query process, not one precommit. On the
`SIFT_128`-sized track `num_nonces_per_bundle` is **20**, so the 2843 ms one-off
is amortised over 20 nonces, not over the whole precommit:

```
t_new = 2.8433 / 20 + 0.022711 = 0.14217 + 0.02271 = 0.16488 s/nonce
```

**`t_new` = 0.165 s/nonce.** The design's 0.05 s estimate is 3.3x optimistic
against this, even though the *marginal* cost it was estimating is 2.2x better
than it guessed. The gap is entirely the per-bundle setup the design does not
account for.

### 3.4 Where the 22.7 ms goes — and what D5's copy really costs

A second instrumented build split `Challenge::for_nonce` into phases with a
`stream.synchronize()` between them. **That added sync is itself expensive: it
breaks pipelining and raises the per-nonce total from 22.7 ms to 31.7 ms.** So
these are attributions under a perturbed build, not a decomposition of the 22.7 ms:

| phase (with the attribution sync) | mean | sd | n |
|---|---|---|---|
| `generate_queries` (7,000 vectors) | 26.795 ms | 0.658 | 60 |
| **database device-to-device alloc + copy (D5)** | **3.894 ms** | 0.198 | 60 |
| `initialize_kernel` + solve + `finalize_kernel` + write | 1.017 ms | 0.143 | 60 |
| for_nonce entry -> output written | 31.705 ms | 0.839 | 60 |

**The spec prices D5's owned copy at "~1.4 ms at T4 bandwidth". Measured here it
is 3.9 ms**, i.e. 358 MB moved at ~92 GB/s of one-way traffic (~184 GB/s counting
read and write, which is a reasonable fraction of the 3060's 360 GB/s). The
1.4 ms figure appears to count only one direction at full advertised bandwidth.
On a T4 (320 GB/s) the same copy would be ~4.4 ms. Against the measured 22.7 ms
nonce that is **~17 % of the per-nonce cost**, not the "~3 % of an estimated
50 ms nonce" the spec claims. This does not by itself overturn D5 — the progress
ledger already records that D5's *rationale* is void and the decision belongs to
the user — but the cost side of that decision is 3x larger and the denominator
2.2x smaller than the spec states.

---

## 4. Break-even, recomputed

### 4.1 Why the algorithm's search time cancels

Let `S` be an algorithm's per-nonce search cost. Then `t_old = 3.2388 + S` and
`t_new = 0.1649 + S`, so

```
delta = t_old - t_new = 3.0739 s/nonce,  independent of S.
```

Verified against `refsearch`: legacy single-nonce wall 28.80 s, batch marginal
25.0006 s; 3.24 + 25.00 = 28.24 s predicted against 28.80 s observed, inside the
startup spread. **So the break-even nonce count below holds for any algorithm
whose search cost the split leaves unchanged**, and is a *conservative* bound for
one whose index makes search cheaper — which is the entire point of the design,
and is unmeasured because no indexed c004 algorithm exists yet.

### 4.2 Break-even

```
B / (t_old - t_new) = 600 / 3.0739 = 195.2 nonces
```

**195 nonces**, against the spec's ~520. The spec's figure is too pessimistic
because its `t_old` was too small; the larger measured saving per nonce pays off
a 600 s build 2.7x sooner.

### 4.3 The "Why not a flat 10 minutes" table, recomputed

600 s build; bundles of 20; `nullstub` floor on both sides, so the numbers are
instance-handling cost only:

| nonces/precommit | today | with a 600 s build | verdict |
|---|---|---|---|
| 80 (protocol floor: `min_num_bundles` 4 x `num_nonces_per_bundle` 20) | 259 s | 613 s | **2.37x worse** |
| 195 | 632 s | 633 s | break-even |
| 500 | 1,619 s | 682 s | 2.37x better |
| 2,000 | 6,478 s | 930 s | 6.97x better |
| 10,000 | 32,388 s | 2,249 s | 14.4x better |

The spec's qualitative conclusion survives — a flat 10-minute cap is a footgun at
the protocol floor — but the floor is 2.4x worse rather than 6.3x worse, and the
crossover is at 195 nonces rather than ~520.

---

## 5. `alpha`

### 5.1 The fuel-to-wall-clock rate

`alpha` is denominated in fuel; every constraint on it is about wall-clock. The
bridge is a measured rate. `refsearch`, one nonce, in a batch:

```
fuel_consumed  = 521,857,948,686      (from /out/C_bat2/0.json; nonce 1 gives 521,857,948,674)
marginal wall  = 25.0006 s            (nonce_done[1] - nonce_done[0], same process)
R              = 2.0874e10 fuel-units per second
```

**Caveats on `R`, which are real:** fuel is a static instruction-cost model, so
`R` is kernel-dependent, not merely hardware-dependent. `refsearch` is a
memory-bound scan, which gives a *low* `R` — and a low `R` is the conservative
direction for sizing `alpha`, because it makes a given fuel budget buy more
seconds. A compute-bound build kernel would show a higher `R` and would therefore
be safer than these numbers assume. `R` is also a single-algorithm, single-card
datum: it is derated 1.25x for a T4 in §5.3 and nowhere else validated.

Sanity check: `there_v10` at ~4.67e9 fuel implies 0.224 s of search on this card,
which is the right order for a production ANN solver.

### 5.2 The two constraints, and why both are N-independent

Because the budget is `alpha * num_nonces * fuel_budget`, the build's wall-clock
grows linearly in `N` exactly as the saving does, so both constraints collapse to
bounds on `alpha` that do not mention `N`. With `F = max_fuel_budget = 5e12`
(the worst case, and a free choice for the player — see §5.5):

| constraint | inequality | bound on `alpha` |
|---|---|---|
| the split is a net win at **every** precommit size | `alpha*F/R <= delta` | **0.01283** |
| the build is a minority of the benchmarker's total per-nonce work (runtime + verifier, §8) | `alpha*F/R <= t_new + t_verify` | **0.01442** |
| the build is a minority of the *runtime-only* query phase | `alpha*F/R <= t_new` | 0.00069 |

The third row is included because it is the reading the spec's own framing
suggests, and it is worth being explicit that it is a *much* harsher test: after
the split the runtime's query phase is only 0.165 s/nonce, so "the build must be
smaller than that" allows a build of just 13 s at the 80-nonce floor. **The
honest accounting is the second row**: `tig-verifier` is mandatory, per-nonce,
and the benchmarker really pays it, so it is part of "total work". Both readings
are reported so the choice is visible rather than hidden in a constant.

### 5.3 The recommended value

Sitting exactly at 0.01283 makes the split *break even* at every size, which is
not a win. Taking half of it leaves a 2x margin, and derating 1.25x for a T4:

```
alpha = delta * R / (2 * F) / 1.25
      = 3.0739 * 2.0874e10 / (2 * 5e12) / 1.25
      = 5.13e-3   ->   alpha = 0.005
```

**Recommended `alpha` = 0.005** (5e-3), against the provisional 0.25 — 50x
smaller. Behaviour at that value, `F` = 5e12, 600 s watchdog:

| N | build fuel | build s (uncapped) | build s (after watchdog) | saving | build as % of query+verify work |
|---|---|---|---|---|---|
| 80 (floor) | 2.00e12 | 95.8 | 95.8 | 245.9 s | 34.7 % |
| 400 | 1.00e13 | 479.1 | 479.1 | 1,229.6 s | 34.7 % |
| 2,000 | 5.00e13 | 2,395.3 | **600** | 6,147.8 s | 8.7 % |
| 10,000 | 2.50e14 | 11,976.7 | **600** | 30,739.2 s | 1.7 % |
| 100,000 | 2.50e15 | 119,767.3 | **600** | 307,392.4 s | 0.2 % |

Both constraints hold at every size, including the 80-nonce floor: the build
costs 95.8 s and saves 245.9 s, a 2.6x net win, and is 34.7 % of the
benchmarker's total per-nonce work. **The break-even footgun of §4.3 disappears
entirely** — it exists only because a *flat* 600 s build is charged to an
80-nonce precommit, which `alpha` at this value never authorises.

For comparison, at the provisional `alpha` = 0.25 and `F` = 5e12: N=80
authorises 1.00e14 fuel = 4,791 s of build against 246 s of saving; N=2,000
authorises 2.50e15 fuel = 119,767 s. In both cases only the flat 600 s watchdog
stops it, which means at `alpha` = 0.25 **the watchdog is the budget and D7's
proportional rule does nothing**.

### 5.4 `max_build_fuel_budget`

The fuel cap and the wall-clock watchdog should agree, or one of them is dead
config. 600 s at the measured rate:

```
600 * 2.0874e10  = 1.252e13        on this RTX 3060
600 * 2.0874e10 / 1.25 = 1.002e13  derated for a T4
```

**Suggested `max_build_fuel_budget` = 1.0e13.** Overflow check the spec asks for:
`1.0e13 * gpu_fuel_scale(20) = 2.0e14`, against `u64::MAX` = 1.84e19 — five
orders of margin. The *unclamped* product still needs `u128`/`checked_mul`: at
`alpha` 0.005, `num_nonces` 100,000, `fuel_budget` 5e12 it is 2.5e15 before the
`min`, which is fine, but the spec's own worst case is unchanged in kind.

### 5.5 A gaming vector the design does not address

`tig-protocol/src/contracts/benchmarks.rs` (`submit_precommit`):

- `fuel_budget` is chosen by the **player**, checked only against
  `max_fuel_budget` (line 89).
- `num_nonces = num_bundles * num_nonces_per_bundle`, and `num_bundles` is chosen
  by the player, checked only against `min_num_bundles`.
- `submission_fee = base_fee + per_nonce_fee * num_bundles`, and c004's
  `per_nonce_fee` is **`"0"`** in the live config. So the fee is **flat** in both
  `num_bundles` and `fuel_budget`.
- D2 makes fuel a spend limit rather than a score term, so there is no
  competitive cost to declaring a huge budget either.

A player therefore maximises their free build budget by setting `fuel_budget` to
`max_fuel_budget` and `num_bundles` as high as they can run, at no marginal cost.
`alpha * num_nonces * fuel_budget` is a product of two player-controlled terms.
Sizing `alpha` against `max_fuel_budget` (as §5.2 does) is what makes the result
safe; sizing it against what an algorithm *actually* consumes would be 1,071x
looser and wrong. **Two alternatives worth considering** — not decided here:
denominate the build budget in observed per-nonce fuel rather than the declared
budget; or make `per_nonce_fee` non-zero so a large `num_bundles` has a price.

---

## 6. Peak device memory during a build

### 6.1 Method, and the trap in it

The brief's `nvidia-smi ... -l 1` sketch cannot be used as written, because
`build_index` **deliberately fills the card**: the balloon is
`free - (memory_cap + 64 MiB)`, so with an 8 GiB cap on a 12 GB device
`nvidia-smi` reports the balloon, not the build. Measured directly — an 8 GiB-cap
run inflates a **3,342,663,680 B** balloon and `nvidia-smi` peaks at 3,482 MiB,
which measures nothing about the build.

So two things were measured instead: `cuMemGetInfo` from inside the process at
each phase (unaffected by the balloon, which is inflated afterwards), and
`nvidia-smi --query-gpu=timestamp,memory.used -lms 100` on the host across a run
whose cap was set to 11 GiB so the balloon is only 116 MiB. The monitor PID was
captured at launch and killed by exact PID.

### 6.2 Results

`cuMemGetInfo`, identical to the byte across all 21 batch runs and both
`build-index` runs (device total 11,910 MiB):

| point | free | used, cumulative | attributable to |
|---|---|---|---|
| after `CudaContext::new` + `load_module` | 12,368,805,888 | 114 MiB | CUDA context + module |
| after `Database::generate` | 11,999,707,136 | 466 MiB | + database, 352 MiB |
| after `Challenge::for_nonce` | 11,630,608,384 | 818 MiB | + D5's owned copy, 352 MiB |

`nvidia-smi` peak, whole card:

| run | peak | note |
|---|---|---|
| `build-index`, 11 GiB cap (balloon 116 MiB) | **1,010 MiB** | ~894 MiB net of the balloon |
| `batch` query process, 20 nonces, no balloon | **1,010 MiB** | 125 non-zero samples |
| `build-index`, 8 GiB cap (the spec value) | 3,482 MiB | balloon-dominated; ran fine, exit 0 |

The two 1,010 MiB peaks agree exactly and reconcile with `cuMemGetInfo`: 818 MiB
of allocations plus ~190 MiB the driver holds outside the reported free pool.
The transient generator scratch (`FORWARD_CHUNK` ping-pong buffers) therefore
never exceeds the post-copy steady state.

### 6.3 What this says about the 8 GiB cap

- **The runtime's own build-phase footprint is 1,010 MiB** with an index that
  allocates nothing: CUDA context, the chunked generator's scratch, and the
  352 MiB database.
- The balloon is sized from `free` measured **after** `Database::generate`, so
  the cap is 8 GiB available *to the algorithm on top of* the context and
  database — not 8 GiB total. That is the right design and worth stating,
  because the opposite reading changes the T4 arithmetic.
- **The measurement rules the cap out only if the runtime's own footprint
  exceeded it. It does not** — 1,010 MiB is 12 % of 8 GiB. So the cap is not
  contradicted.
- **It cannot confirm the cap is right**, because `nullstub`'s `build_index`
  allocates nothing. What a real 700,000-vector ANN index costs to build is
  unmeasured, and no such algorithm exists on this branch to measure.

### 6.4 The T4 question — arithmetic, not measurement

Enforceability of the cap requires `free_after_database >= 8 GiB + 64 MiB`. On
this 3060, `free_after_database` = 11,999,707,136 B = 11.18 GiB, so the cap is
enforceable with 3.11 GiB of balloon — measured. On a 16 GB T4, total usable is
~14.6–15.6 GiB depending on ECC; subtracting the same 466 MiB of context plus
database leaves ~14.2–15.2 GiB, comfortably above the 8.06 GiB required.
**That last step is arithmetic on published T4 capacities, not a measurement:
no T4 was available.** Two things consequently remain open:

1. whether an 8 GiB cap leaves enough for a real index on a T4 (the index's
   build-time footprint is unmeasured); and
2. the query process's peak on a T4 with a real index loaded — measured here at
   1,010 MiB with a zero-footprint index, leaving ~13–14 GiB of T4 headroom for
   the index plus the algorithm's working set, but that headroom has never been
   tested against anything that uses it.

---

## 7. Does the build fit inside a precommit's lifespan?

Live config, fetched 2026-08-31 from `https://mainnet-api.tig.foundation/get-block`:

```
challenges.c004.lifespan_period      = 120
rounds.seconds_between_blocks        = 60
=> lifespan = 120 * 60 = 7200 s (2 hours)
```

(`lifespan_period` has no consumer in this tree — `grep -rn lifespan` finds only
its declaration at `tig-structs/src/config.rs:87` — so the blocks x
seconds-per-block reading is taken from the brief, not verified against protocol
code.)

The build is serial latency ahead of the precommit's first nonce, because it
cannot start until `rand_hash` is known. At the 600 s watchdog ceiling that is
**600 / 7200 = 8.3 % of the lifespan**.

A whole 80-nonce floor precommit, including the per-nonce verifier (§8):

| solver | total | % of lifespan | margin |
|---|---|---|---|
| `nullstub` floor | 876 s | 12.2 % | 6,324 s |
| `refsearch` (25 s/nonce brute force) | 2,876 s | 39.9 % | 4,324 s |

Nonce ceiling within one lifespan, after a 600 s build:

| solver | post-split | pre-split (no build) |
|---|---|---|
| `nullstub` floor | 1,911 nonces | 1,103 nonces |
| `refsearch` | 232 nonces | 228 nonces |

**Finding: the build does not consume a large fraction of the lifespan.** At
8.3 % it is not the binding constraint; amortisation is, and so is the verifier.
Open question 2 can be closed: a whole 80-nonce floor precommit including its
600 s build and its per-nonce verification uses 876 s of the 7,200 s lifespan —
**8.2x headroom** — and even a brute-force solver leaves 4,324 s of margin.

The `refsearch` row is the honest warning attached to that: for an expensive
solver the split barely raises the nonce ceiling (232 vs 228), because the
ceiling is then set by search and verification, not by instance generation.

---

## 8. The finding that reframes the whole argument: the verifier is not amortised

`tig-benchmarker/slave/main.py` runs `tig-verifier` **once per nonce**, as a
separate `docker exec`, immediately after each `tig-runtime` call (lines 95–110).
Its `quality` output is written into the nonce's result file, so it is mandatory
work, not an optional check. The verifier regenerates both halves of the
instance from the seeds — the full 700,000-vector database included — and audits
1,000 salt-sampled queries. **The split does not touch it.**

Measured, `refsearch`'s exact solutions at nonces 0 and 1, three repetitions
each, all six exiting 0 with `quality: 1000000`:

```
3.791  3.308  3.322  3.290  3.278  3.245   (s)
excluding the cold first run: mean 3.289 s, sd 0.026 s
```

So the benchmarker's real per-nonce cost, excluding search:

| | runtime | verifier | total |
|---|---|---|---|
| pre-split | 3.239 s | 3.289 s | **6.53 s** |
| post-split | 0.165 s | 3.289 s | **3.45 s** |

**1.9x, not 24x.** After the split, **95 % of the benchmarker's per-nonce
overhead is the verifier**, and the runtime — the thing this design optimises —
is 5 %. The design's economic case is sound in direction and correct about the
runtime, but the end-to-end benefit it implies is roughly an order of magnitude
optimistic, because the verifier regenerates per nonce exactly what the split
was built to stop regenerating per nonce.

This is a finding about the design, not a defect in the implementation. It is
also the obvious next lever: the same `Database`/`for_nonce` split already exists
in `tig-challenges`, so a batched verifier would take the same 3.289 s down to
roughly `2.8/20 + small`, by the same argument and the same code.

---

## 9. What remains unmeasured

- **Any real indexed algorithm.** No c004 algorithm on this branch builds or uses
  an index. The whole "indexed search is cheaper than the scan it replaces"
  premise — which is what makes the split worth doing at all — is untested. Every
  number here is either a floor (`nullstub`) or an unrepresentative upper anchor
  (`refsearch`, brute force, 112x `there_v10`'s fuel).
- **Any card other than an RTX 3060.** No T4, no 4060. §6.4's T4 fit and §5.3's
  1.25x derate are arithmetic on published specifications.
- **Any scenario other than `SIFT_128`.** This branch has one. Mainnet's live
  config still lists five tracks with `num_nonces_per_bundle` of 20/17/15/10/5.
  If a larger scenario is ever added with a small bundle size, §3.3's
  amortisation gets much worse in both directions at once — a bigger database to
  generate, divided over fewer nonces. At a hypothetical 1.5M-row scenario with a
  5-nonce bundle, linear extrapolation puts `t_new` near 0.8 s/nonce rather than
  0.165 s, which would move `alpha` by roughly 5x. **Extrapolation, not
  measurement** — flagged because the constants in §5 would need revisiting.
- **`build_index` under real memory pressure.** The balloon was never made to
  bite: `nullstub` allocates nothing inside the cap.
- **`R` for anything but one memory-bound kernel.** §5.1.
- **The reaper's effect on longer runs.** Two runs were SIGKILLed at ~23 s
  (`C_leg1`, exit 137, no output; `C_bat4`, killed inside its first nonce) by
  gpuq's orphan-CUDA sweep, which kills any containerised CUDA process alive when
  the 60 s sweep fires (`orphan_cuda_interval_s = 60.0`;
  `reaper.kill_orphan_cuda` exempts only the claim pid's descendants, and a
  container process descends from containerd-shim). Every run reported above
  exited 0 and wrote its expected files; the two reaped runs are recorded here
  and excluded. This is why no single run exceeds ~55 s, and why `refsearch` has
  only one clean marginal-interval sample.

---

## 10. Constants for Task 9

| constant | value | derivation |
|---|---|---|
| `build_fuel_alpha` | **0.005** | `delta * R / (2 * max_fuel_budget) / 1.25` = `3.0739 * 2.0874e10 / 1e13 / 1.25` |
| `max_build_fuel_budget` | **1.0e13** | 600 s watchdog x `R`, derated 1.25x for a T4 |
| memory cap | **8 GiB** unchanged | not contradicted (1,010 MiB measured floor); not confirmed for a real index |
| build watchdog | **600 s** unchanged | 8.3 % of a 7,200 s lifespan |

`build_fuel_alpha * num_nonces * fuel_budget` must still be computed in `u128` or
with `checked_mul`, and `max_build_fuel_budget * gpu_fuel_scale < u64::MAX` must
still be asserted at config load; `1.0e13 * 20 = 2.0e14` passes with five orders
of margin.
