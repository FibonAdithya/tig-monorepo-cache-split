# c004 post-split per-nonce time, `alpha`, and the memory cap

**Date:** 2026-08-31 (revised the same day after review — see §11 for what changed)
**Measures:** `docs/superpowers/specs/2026-08-31-c004-index-build-split-design.md`
**Code under test:** branch `vector_search/gan_instance_gen` at `d0487f97`
(byte-identical in code to `be650a7c`; `git diff --stat be650a7c d0487f97` shows
exactly one file changed, 519 insertions and 17 deletions, and no source file at
all).

This note replaces the design's **estimate** of 0.05 s/nonce, and the 1.2 s/nonce
figure it is measured against, with measurements. It also revises `alpha` and
reports what the memory and lifespan measurements can and cannot settle.

---

## 0. Summary

| Quantity | Design said | Measured here | Where |
|---|---|---|---|
| post-split per-nonce, **marginal** | 0.05 s (estimated) | **0.0227 s** | §3.1 |
| post-split per-nonce, **amortised at the shipped `batch_size` of 8** | — | **0.378 s** | §3.3 |
| `Database::generate` (once per batch) | — | **765.6 ms** | §3.2 |
| per-process startup (once per batch) | — | **2077.7 ms**, of which **2005 ms is `CudaContext::new`** | §2.2 |
| pre-split per-nonce | 1.2 s (unsourced, §2.1) | **3.24 s** (3.31 s in the exact production shape) | §2.1, §2.4 |
| break-even for a *flat* 600 s build | ~520 nonces | **210 nonces** | §4.2 |
| `alpha` | 0.25 (provisional) | **0.003** | §5.4 |
| `max_build_fuel_budget` | unset | **7.0e12** | §5.5 |
| build-phase peak device memory | 8 GiB cap assumed | **1010 MiB** with a zero-footprint index | §6 |
| build vs. precommit lifespan | open question 2 | 600 s of **7200 s** (8.3 %) | §7 |

Three findings matter more than any single number:

1. **The dominant cost on both sides of the comparison is `CudaContext::new`, at
   2.0 s per process — and it is not cacheable.** It is 96 % of per-process
   startup, 62 % of the pre-split per-nonce cost, and therefore the origin of
   most of `delta`. The CUDA JIT cache *is* working (PTX JIT falls from 479 ms to
   4.7 ms once warm), and the same 2.0 s is paid in a long-lived container with
   `docker exec` per nonce — the exact production shape. §2.2, §2.4.
2. **The verifier is not amortised and now dominates.** The reference slave runs
   one `tig-verifier` process per nonce, which regenerates the whole
   700,000-vector database. Measured at **3.25 s/nonce**, unchanged by the split
   — and decomposed: 2006 ms setup, 768 ms regeneration, **91 ms** of actual
   recall audit. The benchmarker's end-to-end per-nonce cost falls from 6.49 s to
   3.63 s — **1.8x**, not the ~24x that 1.2 s → 0.05 s implies. §8.
3. **`alpha` cannot be 0.25.** `build_fuel_budget = alpha * num_nonces *
   fuel_budget` is denominated in `fuel_budget`, which the *player* chooses up to
   `max_fuel_budget` = 5e12 at **zero marginal fee** (`per_nonce_fee` is 0 for
   c004). A real algorithm consumes ~4.67e9 fuel, so the budget can be ~1,071x
   the fuel a nonce actually spends. At `alpha` = 0.25 the build is authorised
   4,791 s of wall-clock at the 80-nonce protocol floor, against 229 s of saving.
   Only the flat 600 s watchdog prevents that. §5.

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
  pre-split baseline was re-measured here on the 3060** (§2.1). Every `alpha` and
  break-even number in this note uses the 3060 baseline against the 3060
  post-split figure. No 4060-derived number enters any arithmetic chain.
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
fuel-to-wall-clock rate in §5.1. It is **not** representative of a production
algorithm: it is ~112x more expensive in fuel than `there_v10` and uses no index.

Scenario: this branch has exactly **one** scenario, `SIFT_128` (7,000 queries,
700,000 database rows, 128 dims, `min_recall` 0.95), so `track_id` is
`s=sift_128` throughout. It corresponds in size to mainnet's `n_queries=7000`
track. The other live tracks (`n_queries=9000..15000`) have no counterpart on
this branch and were **not** measured; see §9.

### 1.3 How time was measured

`tig-runtime`, `tig-challenges` and `tig-verifier` were given `eprintln!` probes
reading `SystemTime::now()` at: process entry; settings parsed; algorithm
`dlopen`ed; PTX read and fuel-patched; `CudaContext::new` returned;
`load_module` returned; `get_device_prop` returned; `Database::generate`
returned; and at the end of every nonce iteration. `cuMemGetInfo` was printed at
the same points. **These probes exist only in an isolated `/tmp/task8/repo` copy
on the box; nothing was committed and `/workspace/tig-bench` was never touched**
(verified: it is still at `d53ceef4` with exactly the pre-existing scratch it had
before this work).

One measurement per process invocation, looping in the shell — a fresh
`CudaContext` per iteration inside one process exhausts the card after ~10
iterations, and surfaces as `NVRM: Out of memory` in `dmesg`, not as a panic.

---

## 2. The pre-split baseline, and where its time actually goes

### 2.1 Re-measured on this card

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
production benchmarker uses: `tig-benchmarker/slave/main.py:60`
(`run_tig_runtime`) launches **one `tig-runtime` process per nonce** via
`docker exec`, in the legacy four-positional form. Twenty consecutive processes,
`nullstub`, nonces 0–19, all in one already-running container:

| | mean | sd | min | max |
|---|---|---|---|---|
| full process wall (as the shell sees it, n=20) | **3238.8 ms** | 133.0 | 3131.7 | 3646.2 |
| in-process (entry -> solution written) | 2846.8 ms | 138.4 | 2726.0 | 3263.8 |

**`t_old` = 3.24 s/nonce.** This excludes `tig-verifier`, which the slave also
runs per nonce; the verifier is measured separately in §8 and appears
identically on both sides of the comparison.

### 2.2 Where the 2,078 ms of startup goes — it is not the JIT

An earlier draft of this note guessed that the startup term was "dominated by JIT
of the challenge PTX". **That is wrong, and the measurement says so.** Probes
between each stage, first run of each container excluded as cold:

| stage | mean | sd |
|---|---|---|
| exec + Rust init -> settings parsed | 0.4 ms | 0.1 |
| `dlopen` of the algorithm `.so` | 2.3 ms | 0.4 |
| PTX read + fuel patch (1,110,707 B) | 3.8 ms | 0.6 |
| **`CudaContext::new` + `set_blocking_synchronize`** | **2005.0 ms** | 33.5 |
| `load_module` — the PTX JIT, cache warm | 4.7 ms | 0.2 |
| `get_device_prop` and the rest | 3.6 ms | 0.5 |
| **startup subtotal** | **2019.9 ms** | |

(n=11, phase G — the production shape of §2.4. Phase F, batch mode in a
throwaway container, gives 2017.9 ms; the phase-A mean of 2077.7 ms over 21 runs
is used elsewhere in this note and agrees to 3 %.)

**96 % of startup is CUDA primary-context creation.** Every other term together
is 15 ms.

### 2.3 The JIT cache works; it is not the explanation

The fuel-patched PTX is byte-identical across every run (constant
`--fuel 2000000000`), so a working cache should collapse the JIT after the first
run. It does. Two containers, eight sequential 5-nonce batches each:

| | `load_module` (warm runs) | cache directory after |
|---|---|---|
| default (cache enabled) | **4.7 ms** | 1,412,463 B in `~/.nv/ComputeCache` |
| `CUDA_CACHE_DISABLE=1` | **479.1 ms** | absent |

So JIT is ~479 ms cold and ~5 ms cached, the cache is populated and hit, and
**`CudaContext::new` is unmoved at 2005 ms / 2017 ms in the two arms.** The
cacheable component is already cached in every measurement in this note.

### 2.4 The exact production shape: one long-lived container, `docker exec` per nonce

The slave `docker exec`s into a container that stays up. Reproduced literally —
`docker run -d ... sleep 3600`, then 12 sequential `docker exec` invocations of
the legacy single-nonce form, first excluded as cold:

| | mean | sd | min | max |
|---|---|---|---|---|
| host-side `docker exec` wall (n=11) | **3310.1 ms** | 186.4 | 2736.0 | 3427.6 |
| in-process entry -> solution (n=10) | 2816.4 ms | 34.4 | | |
| `docker exec` client overhead (difference) | 493.7 ms | | | |

`CudaContext::new` in this shape: 2005.0 ms — identical to the throwaway-container
case. **The 2.0 s is structural, not an artefact of running one container per
measurement.** And `t_old` = 3.2388 s used throughout this note is 2.2 % *below*
the production-shape 3310.1 ms, i.e. conservative: the real pre-split cost is
slightly higher than the figure the arithmetic uses.

### 2.5 Sensitivity: what if startup were smaller?

`delta` exists largely because startup is paid once per nonce pre-split and once
per batch post-split, so it is worth showing the exposure explicitly. Holding
everything else and varying startup `s` (at `batch_size` 8, see §3.3):

```
t_old(s) = 3.2388 - (2.0777 - s)          t_new(s) = (s + 0.7656)/8 + 0.022711
```

| `s` | `t_old` | `t_new` | `delta` | net-win bound | + T4 derate | + 2x margin | |
|---|---|---|---|---|---|---|---|
| 2078 ms | 3.239 s | 0.378 s | 2.861 s | 0.00879 | 0.00703 | 0.00352 | **measured** (warm container, warm JIT cache) |
| 2497 ms | 3.658 s | 0.431 s | 3.228 s | 0.00991 | 0.00793 | 0.00397 | `CUDA_CACHE_DISABLE=1` |
| 1600 ms | 2.761 s | 0.318 s | 2.443 s | 0.00750 | 0.00600 | **0.00300** | full margin exhausted here |
| 1000 ms | 2.161 s | 0.243 s | 1.918 s | 0.00589 | 0.00471 | 0.00236 | hypothetical |
| 500 ms | 1.661 s | 0.181 s | 1.480 s | 0.00455 | 0.00364 | 0.00182 | hypothetical |
| 204 ms | 1.365 s | 0.144 s | 1.221 s | 0.00375 | **0.00300** | 0.00150 | net win with the T4 derate exhausted here |
| 0 ms | 1.161 s | 0.118 s | 1.043 s | 0.00320 | 0.00256 | 0.00128 | floor |

All three bound columns use the conservative `R_low` (§5.1). Read against the
recommended `alpha` = 0.003:

- **The net-win property never breaks.** Even with per-process startup at zero,
  the bare net-win bound is 0.00320 > 0.003. The split still beats the pre-split
  baseline at every precommit size for any startup cost whatsoever.
- **With the 1.25x T4 derate**, 0.003 holds down to `CudaContext::new` ~= 204 ms
  — a 10x reduction from what is measured.
- **The full 2x margin** holds down to ~1,600 ms, i.e. a 21 % reduction. That is
  the term that erodes first.

**Trigger for revisiting:** if `CudaContext::new` is measured below ~1.6 s on the
deployment hardware — a different driver, persistence mode, a bare-metal host —
`alpha` still delivers a net win but no longer with a 2x cushion, and should be
recut. Nothing in this note makes 2.0 s a property of anything but this host.

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
23.21 ms, with no trend in N. An ordinary-least-squares fit of in-process total
against N over all 21 runs gives slope **21.50 ms/nonce**, intercept 2882 ms —
consistent with the direct interval measurement.

**The design's estimate was 0.05 s. The measured marginal cost is 0.0227 s —
2.2x better than estimated.**

### 3.2 The one-off cost, and `Database::generate` separated from it

The brief asks for `Database::generate` separated from the per-nonce cost. The
probes give it directly rather than by subtraction:

| phase | mean | sd | n |
|---|---|---|---|
| process start + `dlopen` + `CudaContext::new` + PTX JIT (decomposed in §2.2) | 2077.7 ms | 160.5 | 21 |
| **`Database::generate`** (700,000 x 128, + `load_index`) | **765.6 ms** | 5.2 | 21 |
| **total one-off, paid once per batch** | **2843.3 ms** | | |

`Database::generate` is remarkably stable (sd 0.7 %). The startup term is the
variable one, and it is the larger of the two: **73 % of the one-off cost is not
database generation at all, and 96 % of *that* is `CudaContext::new`.** So
batching pays for itself on CUDA context creation before it pays for anything the
design talks about.

Cross-checks from two independent code paths that generate the same database:
`build-index` gives 750.1 ms and 751.2 ms; `tig-verifier`'s `generate_instance`
(database + queries) gives 768.3 ms. All within 2 % of 765.6 ms.

### 3.3 The amortisation denominator is an operator config knob, not a protocol constant

D4 says "the bundle's nonces run in one batched query process" — but the
reference benchmarker does not size its batches by `num_nonces_per_bundle`. It
uses a per-algorithm **`batch_size`** chosen by the operator
(`master/master/job_manager.py:66-73` selects it from `algo_selection`;
`slave_manager.py:42` turns it into the batch's `start_nonce`/`num_nonces`), it
is validated to be a **power of two** (`client_manager.py:87-89` rejects 0 and
any non-power-of-two), and the value shipped in `postgres/init.sql` is **8** for
every challenge. The spec's D4 is itself inconsistent on this, saying "the
bundle's nonces" in one place and "once per **precommit**" in another.

So the amortised per-nonce cost is a curve, not a constant:

```
t_new(batch_size) = 2.8433 / batch_size + 0.022711
```

| `batch_size` | `t_new` | `delta` = `t_old` - `t_new` | |
|---|---|---|---|
| 1 | 2.8660 s | 0.3728 s | degenerate — no batching at all |
| 2 | 1.4444 s | 1.7944 s | |
| 4 | 0.7335 s | 2.5053 s | |
| **8** | **0.3781 s** | **2.8607 s** | **shipped default (`init.sql`)** |
| 16 | 0.2004 s | 3.0384 s | |
| 20 | 0.1649 s | 3.0739 s | D4's bundle reading (not a power of two, so not reachable) |
| 64 | 0.0671 s | 3.1717 s | |
| 512 | 0.0283 s | 3.2105 s | |
| ∞ | 0.0227 s | 3.2161 s | the marginal cost of §3.1 |

**This note uses `batch_size` = 8 throughout**, because it is what the reference
benchmarker ships and because it is the conservative choice among realistic
values (smallest `delta`, smallest `alpha`). **`t_new` = 0.378 s/nonce.**

`delta` — which is what break-even and `alpha` actually depend on — moves only
12 % across `batch_size` 8 → 512, and only 7 % from 8 → 20, so the constants in
§5 are insensitive to this choice. `t_new` itself is not: quote it with its
`batch_size`.

### 3.4 Where the 22.7 ms goes — and what D5's copy really costs

A second instrumented build split `Challenge::for_nonce` into phases with a
`stream.synchronize()` between them. **That added sync is itself expensive: it
breaks pipelining and raises the per-nonce total from 22.7 ms to 31.7 ms.** So
these are attributions under a perturbed build, and each is therefore an
**upper bound** on the corresponding phase in the unperturbed 22.7 ms:

| phase (with the attribution sync) | mean | sd | n |
|---|---|---|---|
| `generate_queries` (7,000 vectors) | 26.795 ms | 0.658 | 60 |
| **database device-to-device alloc + copy (D5)** | **<= 3.894 ms** | 0.198 | 60 |
| `initialize_kernel` + solve + `finalize_kernel` + write | 1.017 ms | 0.143 | 60 |
| for_nonce entry -> output written | 31.705 ms | 0.839 | 60 |

**The spec prices D5's owned copy at "~1.4 ms at T4 bandwidth". Measured here it
is at most 3.9 ms.** Two qualifications, both of which cut against over-reading
that number: the phase measured is `alloc` + `clone_dtod`, so the implied
~92 GB/s of one-way traffic (~184 GB/s counting read and write) **understates**
the copy's own bandwidth by however long the 358 MB allocation took; and the
3.894 ms comes from the perturbed build while the 22.7 ms denominator does not,
so the derived **"~17 % of a nonce" is likewise an upper bound**, not a
measurement of the unperturbed ratio. What survives cleanly is that the spec's
1.4 ms counts only one direction at full advertised bandwidth, and that the true
cost is materially larger than 1.4 ms against a denominator materially smaller
than 50 ms. This does not by itself overturn D5 — the progress ledger already
records that D5's *rationale* is void and the decision belongs to the user — but
both sides of that cost/benefit have moved.

---

## 4. Break-even, recomputed

### 4.1 Why the algorithm's search time cancels

Let `S` be an algorithm's per-nonce search cost. Then `t_old = 3.2388 + S` and
`t_new = 0.3781 + S`, so

```
delta = t_old - t_new = 2.8607 s/nonce,  independent of S.
```

Verified against `refsearch`: legacy single-nonce wall 28.80 s, batch marginal
25.0006 s; 3.24 + 25.00 = 28.24 s predicted against 28.80 s observed, inside the
startup spread. **So the break-even nonce count below holds for any algorithm
whose search cost the split leaves unchanged**, and is a *conservative* bound for
one whose index makes search cheaper — which is the entire point of the design,
and is unmeasured because no indexed c004 algorithm exists yet.

### 4.2 Break-even for a flat 600 s build

```
B / (t_old - t_new) = 600 / 2.8607 = 209.7 nonces
```

**210 nonces**, against the spec's ~520. The spec's figure is too pessimistic
because its `t_old` was too small; the larger measured saving per nonce pays off
a 600 s build 2.5x sooner.

### 4.3 The "Why not a flat 10 minutes" table, recomputed

Flat 600 s build; `batch_size` 8; `nullstub` floor on both sides, so the numbers
are instance-handling cost only:

| nonces/precommit | today | with a flat 600 s build | verdict |
|---|---|---|---|
| 80 (protocol floor: `min_num_bundles` 4 x `num_nonces_per_bundle` 20) | 259 s | 630 s | **2.43x worse** |
| 210 | 680 s | 680 s | break-even |
| 500 | 1,619 s | 790 s | 2.05x better |
| 2,000 | 6,478 s | 1,356 s | 4.78x better |
| 10,000 | 32,388 s | 4,381 s | 7.39x better |

The spec's qualitative conclusion survives — a **flat** 10-minute cap is a footgun
at the protocol floor — but the floor is 2.4x worse rather than 6.3x worse, and
the crossover is at 210 nonces rather than ~520. Under the proportional rule at
the recommended `alpha` the footgun does not arise at all (§5.4): the build is
never more than 0.42x the wall-clock it saves, at any precommit size.

---

## 5. `alpha`

### 5.1 The fuel-to-wall-clock rate, and a 36 % discrepancy in it

`alpha` is denominated in fuel; every constraint on it is about wall-clock. The
bridge is a measured rate. `refsearch`, one nonce, in a batch:

```
fuel_consumed  = 521,857,948,686      (/out/C_bat2/0.json; nonce 1 gives 521,857,948,674)
marginal wall  = 25.0006 s            (nonce_done[1] - nonce_done[0], same process)
R_measured     = 2.0874e10 fuel-units per second
```

**This does not agree with the only other recorded figure for the same solver on
the same track.** `2026-08-13-gan-instance-generation-design.md` records
"`fuel_consumed` is 3.84e11 at n_queries=7000" for exact 1-NN — 36 % below
5.2186e11. Fuel is a static instruction-cost model, so it should be reproducible.
What was checked:

- **`refsearch` has not been edited.** `refsearch.rs` and `refsearch/kernels.cu`
  are byte-identical across all four copies on the box
  (`/workspace/keepsafe-2026-08-31/`, `/workspace/refsearch*`,
  `/workspace/tig-bench/`, and the `/tmp` copy built here) and both carry an
  mtime of 2026-08-25 14:49. It is untracked, so it has no git history and the
  version that produced 3.84e11 cannot be recovered.
- **`build_ptx` has not changed** since the datum was recorded
  (`git log 3818f656..HEAD -- tig-binary/scripts/build_ptx` is empty), so the
  instruction cost table and the injection are the same.
- **`kernels.cu` has changed, four times, after the datum.** The datum landed in
  `3818f656` (2026-08-14); `tig-challenges/src/vector_search/kernels.cu` was
  modified on 2026-08-27 by `4481c873`, `f55b7577`, `d56daa91` and `b09a26e3`
  (the recall-audit work). `build_ptx` concatenates `framework.cu` + every
  challenge `.cu` + the algorithm's `.cu` into **one translation unit** and runs
  `nvcc -dopt=on --use_fast_math` once over it, so an algorithm's instrumented
  instruction count is not independent of the challenge's kernels.

That is a plausible mechanism, **not a verified one** — reproducing it would need
a build at the pre-audit `kernels.cu` with the `refsearch` of that date, which no
longer exists. The datum also predates the per-scenario track redesign
(2026-08-25), so it was taken under a different track scheme.

`R` enters `alpha` linearly, and the older figure is the **less** favourable one,
so both are carried through §5.3 and the recommendation is sized against the
lower:

```
R_measured = 5.21858e11 / 25.0006 = 2.0874e10 fuel/s
R_low      = 3.84e11    / 25.0006 = 1.5360e10 fuel/s   (the 2026-08-14 datum)
```

**Other caveats on `R`, which are real.** Fuel is a static instruction-cost
model, so `R` is kernel-dependent, not merely hardware-dependent. `refsearch` is
a memory-bound scan, which gives a *low* `R` — the conservative direction, since
a low `R` makes a given fuel budget buy more seconds. A compute-bound build
kernel would show a higher `R` and would be safer than these numbers assume.

Sanity check: `there_v10` at ~4.67e9 fuel implies 0.22–0.30 s of search on this
card, which is the right order for a production ANN solver.

### 5.2 The two constraints, and why both are N-independent

Because the budget is `alpha * num_nonces * fuel_budget`, the build's wall-clock
grows linearly in `N` exactly as the saving does, so both constraints collapse to
bounds on `alpha` that do not mention `N`. With `F = max_fuel_budget = 5e12`
(the worst case, and a free choice for the player — see §5.6), `batch_size` = 8:

| constraint | inequality | at `R_measured` | at `R_low` |
|---|---|---|---|
| the split is a net win at **every** precommit size | `alpha*F/R <= delta` | 0.01194 | 0.00879 |
| the build is a minority of the benchmarker's total per-nonce work (runtime + verifier, §8) | `alpha*F/R <= t_new + t_verify` | 0.01516 | 0.01115 |
| the build is a minority of the *runtime-only* query phase | `alpha*F/R <= t_new` | 0.00158 | 0.00116 |
| **if the verifier were ever batched too** (§5.3) | `alpha*F/R <= t_new + 0.44` | 0.00341 | 0.00251 |

The third row is included because it is the reading the spec's own framing
suggests, and it is worth being explicit that it is a *much* harsher test: after
the split the runtime's query phase is only 0.378 s/nonce, so "the build must be
smaller than that" allows a build of just 30 s at the 80-nonce floor. **The
honest accounting is the second row**: `tig-verifier` is mandatory, per-nonce,
and the benchmarker really pays it, so it is part of "total work". All readings
are reported so the choice is visible rather than hidden in a constant.

### 5.3 The tension between this note's two recommendations

Row 2 counts the verifier's 3.25 s as legitimate work — while §8 of this same
note recommends **batching the verifier**, which would remove most of it. That is
a real tension and it is stated rather than buried.

Using the verifier's measured decomposition (§8: 2006 ms setup, 768 ms
regeneration, 91 ms audit) and the same batching argument as the runtime, a
batched verifier at `batch_size` 8 would cost roughly
`(2006 + 768)/8 + ~2 + 91 = 440 ms/nonce` — a **derivation from measured parts,
not a measurement of a batched verifier, which does not exist.** At that value
the row-2 bound falls from 0.01516 to **0.00341** (`R_measured`) or **0.00251**
(`R_low`).

**So: `alpha` must be recut if the verifier is ever batched.** The recommended
0.003 survives that change at `R_measured` with 12 % to spare and violates it by
19 % at `R_low`. The net-win bound (row 1), which is the primary constraint, is
unaffected by verifier batching in either case, and 0.003 clears it by 4x.

### 5.4 The recommended value

Sitting exactly at the net-win bound makes the split *break even* at every size,
which is not a win. Taking half of it leaves a 2x margin, derating 1.25x for a
T4, at `batch_size` 8 and the lower of the two rates:

```
alpha = delta * R_low / (2 * F) / 1.25
      = 2.8607 * 1.5360e10 / (2 * 5e12) / 1.25
      = 3.52e-3   ->   alpha = 0.003
```

**Recommended `alpha` = 0.003 (3e-3)**, against the provisional 0.25 — 83x
smaller. Each digit of that reduction is traceable: 0.01194 is the bare net-win
bound at `R_measured`; halving it for margin gives 0.00597; derating for a T4
gives 0.00478; taking the older `R_low` datum instead gives 0.00352; rounding
down to a clean value gives 0.003. A reader who rejects the `R_low` datum (§5.1)
can justify 0.005 instead; a reader who wants the batched-verifier future covered
at `R_low` too needs 0.0025.

Behaviour at `alpha` = 0.003, `F` = 5e12, `max_build_fuel_budget` = 7.0e12
(§5.5), 600 s watchdog, `batch_size` 8 — **this table applies all three limits,
which is what makes it usable for Task 9**:

| N | `alpha*N*F` | after the fuel cap | build s @`R_measured` | build s @`R_low` | what binds | saving | build as % of query+verify |
|---|---|---|---|---|---|---|---|
| 80 (floor) | 1.20e12 | 1.20e12 | 57.5 | 78.1 | `alpha` | 228.9 s | 19.8 % |
| 200 | 3.00e12 | 3.00e12 | 143.7 | 195.3 | `alpha` | 572.1 s | 19.8 % |
| 466 | 6.99e12 | 6.99e12 | 334.9 | 455.1 | `alpha` | 1,333.1 s | 19.8 % |
| 500 | 7.50e12 | **7.00e12** | 335.3 | 455.7 | **fuel cap** | 1,430.3 s | 18.5 % |
| 2,000 | 3.00e13 | **7.00e12** | 335.3 | 455.7 | **fuel cap** | 5,721.4 s | 4.6 % |
| 10,000 | 1.50e14 | **7.00e12** | 335.3 | 455.7 | **fuel cap** | 28,606.8 s | 0.9 % |
| 100,000 | 1.50e15 | **7.00e12** | 335.3 | 455.7 | **fuel cap** | 286,067.6 s | 0.1 % |

Both constraints hold at every size, including the 80-nonce floor: the build
costs 57.5–78.1 s and saves 228.9 s, a 2.9–4.0x net win, and is 19.8 % of the
benchmarker's per-nonce work (16.5 % of the total including the build itself).
**The break-even footgun of §4.3 disappears entirely** — it exists only because a
*flat* 600 s build is charged to an 80-nonce precommit, which `alpha` at this
value never authorises. Note also that **the 600 s watchdog never fires in this
table**: the fuel cap binds first at every N above 466, and the watchdog is a
genuine safety net rather than the budget. That is the intended relationship and
it only holds because §5.5 sizes the two to agree.

For comparison, at the provisional `alpha` = 0.25 and `F` = 5e12: N=80
authorises 1.00e14 fuel = 4,791 s of build against 229 s of saving; N=2,000
authorises 2.50e15 fuel = 119,767 s. In both cases only the flat 600 s watchdog
stops it, which means at `alpha` = 0.25 **the watchdog is the budget and D7's
proportional rule does nothing**.

### 5.5 `max_build_fuel_budget`

The fuel cap and the wall-clock watchdog should agree, or one of them is dead
config — and the fuel cap should bind *first*, so the watchdog stays a safety
net. That means sizing it against the **slowest** modelled rate:

```
R_low / 1.25 (T4 derate) = 1.229e10 fuel/s
600 s * 1.229e10          = 7.37e12   ->   round DOWN to 7.0e12
```

**Suggested `max_build_fuel_budget` = 7.0e12.** It buys 335 s on this 3060 and
570 s at the slowest modelled rate — under the 600 s watchdog in both cases, so
the watchdog never fires first. Overflow check the spec asks for:
`7.0e12 * gpu_fuel_scale(20) = 1.4e14`, against `u64::MAX` = 1.84e19 — five
orders of margin. The *unclamped* product still needs `u128`/`checked_mul`: at
`alpha` 0.003, `num_nonces` 100,000, `fuel_budget` 5e12 it is 1.5e15 before the
`min`.

### 5.6 A gaming vector the design does not address

`tig-protocol/src/contracts/benchmarks.rs` (`submit_precommit`):

- `fuel_budget` is chosen by the **player**, checked only against
  `max_fuel_budget` (line 89).
- `num_nonces = num_bundles * num_nonces_per_bundle`, and `num_bundles` is chosen
  by the player, checked only against `min_num_bundles`.
- `submission_fee = base_fee + per_nonce_fee * num_bundles`, and c004's
  `per_nonce_fee` is **`"0"`** in the live config. So the fee is **flat** in both
  `num_bundles` and `fuel_budget`.
- D2 makes fuel a spend limit rather than a score term, so there is no
  competitive cost to declaring a large budget either.

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

`cuMemGetInfo`, identical to the byte in every run that printed it (device total
11,910 MiB):

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
**600 / 7200 = 8.3 % of the lifespan**; at the recommended `alpha` it is at most
335 s (4.7 %) on this card, or 456 s (6.3 %) at `R_low`.

A whole 80-nonce floor precommit at `alpha` = 0.003 and `batch_size` 8 (10
batches), including the per-nonce verifier (§8):

| solver | total | % of lifespan | margin | headroom |
|---|---|---|---|---|
| `nullstub` floor | 348 s | 4.8 % | 6,852 s | 20.7x |
| `refsearch` (25 s/nonce brute force) | 2,348 s | 32.6 % | 4,852 s | 3.1x |

Nonce ceiling within one lifespan, after the build:

| solver | post-split | pre-split (no build) |
|---|---|---|
| `nullstub` floor | 1,967 nonces | 1,109 nonces |
| `refsearch` | 249 nonces | 229 nonces |

**Finding: the build does not consume a large fraction of the lifespan.** Open
question 2 can be closed: a whole 80-nonce floor precommit including its build
and its per-nonce verification uses 348 s of the 7,200 s lifespan — **20.7x
headroom** — and even a brute-force solver leaves 4,852 s of margin. The lifespan
is not the binding constraint; amortisation is, and so is the verifier.

The `refsearch` row is the honest warning attached to that: for an expensive
solver the split barely raises the nonce ceiling (249 vs 229), because the
ceiling is then set by search and verification, not by instance generation.

---

## 8. The finding that reframes the whole argument: the verifier is not amortised

`tig-benchmarker/slave/main.py` runs `tig-verifier` **once per nonce**, as a
separate `docker exec`, immediately after each `tig-runtime` call (lines 95–110).
Its `quality` output is written into the nonce's result file, so it is mandatory
work, not an optional check. **The split does not touch it.**

Measured, `refsearch`'s exact solutions at nonces 0 and 1, five repetitions each,
all exiting 0 with `quality: 1000000`, first excluded as cold (n=9):

| phase | mean | sd |
|---|---|---|
| process + CUDA context + PTX JIT | 2005.6 ms | 25.9 |
| `generate_instance` — database 700k + queries 7k | 768.3 ms | 3.2 |
| `evaluate_solution` — the recall audit, 1,000 samples | **90.9 ms** | 0.5 |
| in-process total | 2864.7 ms | 28.2 |
| **shell-observed process wall** | **3252.4 ms** | 35.2 |

(An earlier, uninstrumented set of 5 warm runs gave 3288.7 ms, sd 25.6 — the two
agree to 1.1 %.)

**Only 91 ms of the 3.25 s is the audit the verifier exists to do.** The other
97 % is process setup and regenerating an instance the runtime had already
generated moments earlier.

So the benchmarker's real per-nonce cost, excluding search:

| | runtime | verifier | total |
|---|---|---|---|
| pre-split | 3.239 s | 3.252 s | **6.49 s** |
| post-split (`batch_size` 8) | 0.378 s | 3.252 s | **3.63 s** |

**1.8x, not 24x.** After the split, **90 % of the benchmarker's per-nonce
overhead is the verifier**, and the runtime — the thing this design optimises —
is 10 %. The design's economic case is sound in direction and correct about the
runtime, but the end-to-end benefit it implies is roughly an order of magnitude
optimistic, because the verifier regenerates per nonce exactly what the split
was built to stop regenerating per nonce.

This is a finding about the design, not a defect in the implementation. It is
also the obvious next lever: the same `Database`/`for_nonce` split already exists
in `tig-challenges`, and the verifier's amortisable share is now measured
(2005.6 ms setup + the database part of the 768.3 ms regeneration) against a
91 ms irreducible audit. **A batched verifier at `batch_size` 8 would cost
roughly `(2006 + 766)/8 + ~2 + 91 = 440 ms/nonce` — that is a derivation from
measured parts, not a measurement**; no batched verifier exists. If it is ever
built, §5.3 says what happens to `alpha`.

---

## 9. What remains unmeasured

- **Any real indexed algorithm.** No c004 algorithm on this branch builds or uses
  an index. The whole "indexed search is cheaper than the scan it replaces"
  premise — which is what makes the split worth doing at all — is untested. Every
  number here is either a floor (`nullstub`) or an unrepresentative upper anchor
  (`refsearch`, brute force, 112x `there_v10`'s fuel).
- **Any card other than an RTX 3060.** No T4, no 4060. §6.4's T4 fit and §5.4's
  1.25x derate are arithmetic on published specifications. In particular the
  2.0 s `CudaContext::new` (§2.2), which is the single largest term in `delta`,
  is a property of this host and driver and is not known to generalise.
- **Any scenario other than `SIFT_128`.** This branch has one. Mainnet's live
  config still lists five tracks with `num_nonces_per_bundle` of 20/17/15/10/5.
  If a larger scenario is ever added, §3.3's amortisation gets worse: a bigger
  database to generate, divided over the same `batch_size`. At a hypothetical
  1.5M-row scenario, linear extrapolation puts `t_new` near 0.6 s/nonce at
  `batch_size` 8 rather than 0.378 s. **Extrapolation, not measurement** —
  flagged because the constants in §5 would need revisiting.
- **`build_index` under real memory pressure.** The balloon was never made to
  bite: `nullstub` allocates nothing inside the cap.
- **The 36 % fuel discrepancy of §5.1** is explained by a plausible mechanism
  that could not be reproduced, not by a verified cause.
- **`R` for anything but one memory-bound kernel.** §5.1.
- **A batched verifier**, and therefore the 440 ms figure in §8 and the
  batched-verifier row in §5.2.
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
| `build_fuel_alpha` | **0.003** | `delta * R_low / (2 * max_fuel_budget) / 1.25` = `2.8607 * 1.5360e10 / 1e13 / 1.25` = 3.52e-3, rounded down |
| `max_build_fuel_budget` | **7.0e12** | `600 s * R_low / 1.25` = 7.37e12, rounded down so the fuel cap binds before the watchdog |
| memory cap | **8 GiB** unchanged | not contradicted (1,010 MiB measured floor); not confirmed for a real index |
| build watchdog | **600 s** unchanged | safety net; never the binding limit at these constants |

`build_fuel_alpha * num_nonces * fuel_budget` must still be computed in `u128` or
with `checked_mul`, and `max_build_fuel_budget * gpu_fuel_scale < u64::MAX` must
still be asserted at config load; `7.0e12 * 20 = 1.4e14` passes with five orders
of margin.

**Revisit these constants if any of the following changes:** the verifier is
batched (§5.3); `CudaContext::new` falls below ~1.6 s on the deployment hardware
(§2.5); a scenario larger than `SIFT_128` is added (§9); or the 36 % fuel
discrepancy of §5.1 is resolved in favour of the higher rate, in which case 0.005
is justified.

---

## 11. Revision history

**2026-08-31, after review.** Substantive changes:

- **§2.2/§2.3/§2.4 are new.** The first draft attributed the 2,078 ms startup to
  PTX JIT without decomposing it. Measured: it is 2,005 ms of `CudaContext::new`;
  JIT is 4.7 ms with the cache warm and 479 ms with `CUDA_CACHE_DISABLE=1`, and
  the cache is populated and hit. Confirmed in the exact production shape (one
  long-lived container, `docker exec` per nonce), where `t_old` is 3,310 ms — so
  the 3,238.8 ms used in the arithmetic is conservative. §2.5 adds the
  sensitivity of `delta` and `alpha` to that term.
- **§3.3 rewritten.** The amortisation denominator is the operator's
  power-of-two `batch_size` (shipped default **8**), not `num_nonces_per_bundle`.
  The headline `t_new` moves from 0.165 s (`batch_size` 20) to **0.378 s**
  (`batch_size` 8); `delta` moves 7 %.
- **§5.1 extended** with the 36 % discrepancy against the 2026-08-14 `3.84e11`
  figure, what was checked, and the mechanism that plausibly explains it. The
  lower rate is now carried through the recommendation.
- **`alpha` revised 0.005 -> 0.003** and `max_build_fuel_budget` **1.0e13 ->
  7.0e12**, from `batch_size` 8 and `R_low` (both conservative directions).
  §5.4's behaviour table now applies the fuel cap as well as the watchdog.
- **§5.3 is new**: the tension between the `alpha` justification and §8's own
  recommendation to batch the verifier, with the number `alpha` would have to
  become.
- **§8 extended** with the verifier's measured decomposition (2006 / 768 / 91 ms)
  and the batched-verifier figure explicitly labelled a derivation.
- **§3.4** now labels the D5 copy figure and the 17 % ratio as upper bounds, and
  notes the phase includes the allocation.
- Corrected: 6.53 -> 3.46 s became 6.49 -> 3.63 s (`batch_size` 8, and the
  verifier value from the larger sample); "34.7 % of total work" was
  build/(query+verify), now given as both.
