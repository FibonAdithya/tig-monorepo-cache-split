# Recall-gated c004, monorepo side — what the implementation found

Companion to `plans/2026-08-27-recall-gated-c004-monorepo.md`, written after Tasks 1–6 landed
(`f0af13a0..0d2daa62`). Records what the plan got wrong, what remains open, and what was
deliberately parked — the things a reader of the plan alone would not learn.

**The plan's interface-contract section requires that a defect found in one plan be fixed in all
three.** The harness and unification plans were written from the same spec and may carry the same
defects; items marked **[CHECK ALL THREE]** are the candidates.

## The spec's central premise is measurably false

Spec Decision 1 predicted `E[d₂/d₁] − 1 < ~1%` (distance concentration), which would make returning
the 2nd-nearest neighbour nearly free on mean distance. **Measured over 6 seeds: 0.0438** — four
times the threshold. Returning the 2-NN everywhere costs ~7,600 quality units and does *not* qualify.

The redesign survives for a different and stronger reason: the blindness is **skew, not
concentration**. 21% of queries have their 2-NN within 1%, and missing all of them costs ~190 quality
units against a 3,400 margin — inside the exact kernel's own 1,500-unit seed-to-seed spread. Mean
distance cannot distinguish recall 1.00 from recall 0.79.

Measured floors on `r`: **0.281** (what mean-distance quality provably tolerates, worst seed) and
**0.79** (contingent on an unmeasured near-tie-miss model). `r = 0.95` clears both.

Detail: `docs/measurements/2026-08-27-c004-d2-over-d1.md` in the tig-pentesting worktree.
**The spec still carries only the conditional form and is still `Status: proposed`.**

## Open questions for a human

1. **The upper end of `r` is unmeasured.** No ANN method was ever run, so "0.95 is not trivially
   cleared by an approximate method" is judgement, not measurement. If a cheap IVF or graph index
   reaches 0.95 at a large speedup, raise `r` toward 0.99 (worst-case 3σ pass point 0.9806, still far
   above the 0.281 floor) rather than reconsidering the design. One constant plus two assertions.
2. **Spec Decision 1 needs correcting** to the skew mechanism above. Cross-repo; not edited here.
3. **The salt derivation uses 8 of 32 bytes** (`StdRng::seed_from_u64` over `salt[0..8]`), matching how
   tig-protocol draws `sampled_nonces`. 2⁶⁴ is not brute-forceable and the salt post-dates the solve,
   so this is safe — but it is a latent footgun if the protocol's eventual salt derivation puts
   structure in the low 8 bytes. `audit_sampling.rs` carries a test *pinning* the truncation, so
   widening it is necessarily a conscious act. **[CHECK ALL THREE]**

## Plan defects found during implementation

| # | Defect | Where |
|---|---|---|
| 1 | Task 4's tests referenced `tig_challenges::…` from inside the `tig-challenges` crate. A crate cannot name itself — `E0433`. Would have cost a GPU round-trip to discover. | Task 4 Step 3 |
| 2 | Task 1's Files line says the measurement note lands "in this repo"; the Global Constraints say tig-pentesting. `docs/measurements/` does not exist in the monorepo. | Task 1 |
| 3 | **Task 3 is tagged `[local]` but is not locally testable.** `scenarios.rs` is inside the `c004`-gated tree (`c004 = ["cudarc"]`), so the plan's own verification command reports `0 passed; 7 filtered out` — identically before and after any change. A non-discriminating verification step. **The `[local]`/`[GPU]` tags are not trustworthy; file location is the operative rule.** **[CHECK ALL THREE]** | Task 3 Steps 2/4 |
| 4 | The plan says "the cuda arm" throughout; the `dispatch_challenge!` macro arms are named `cpu` and `gpu`. **[CHECK ALL THREE]** | Task 5 |
| 5 | Task 4's out-of-range test mutates `sol.indexes[0]`, but query 0 is not sampled under salt `[9u8;32]` (first sampled id is 3), so it would have failed. | Task 4 Step 3 |
| 6 | The verifier's `--audit-salt [AUDIT_SALT]` makes the *value* optional in clap; a valueless flag silently falls back to the all-zeros salt — a fully predictable audit set. Changed to `<AUDIT_SALT>`. **[CHECK ALL THREE]** | Task 5 Step 5 |
| 7 | **Task 6 Step 4 asserts a guard that does not exist**: "the correctness tests are the guard here: a tiling bug that makes the kernel fast and wrong fails them." Measured false — a kernel scanning half the database passed all 9 recall tests. Closed by a 2nd-NN guard. | Task 6 Step 4 |
| 8 | Task 6's 150 ms gate was derived from a ~1,000 ms estimate; the untiled kernel actually measured **2,623 ms**. The gate was fine, but by luck. | Task 6 Step 1 |

## The dominant defect mode: tests that discriminate nothing

**Eight instances** were found across six tasks and two review rounds. Every one passed while
asserting nothing the code could fail:

1. `is_err()` on the invalid-index test — passed at RED with **no kernel compiled at all**.
2. "every query once" asserting only `len() == 10`, never uniqueness.
3. `corrupting_an_unsampled_query_does_not_move_recall` asserting only `before == after` — satisfied
   by any constant-returning implementation; it demonstrably survived a `tolerance_sq = 1e6` mutation.
4. `err.contains('2')` — satisfied by the "32" in "32 bytes" regardless of the reported length.
5. The whole suite blind to a kernel that skips database candidates (false **hits**, i.e. recall
   reading *higher* than reality — the direction the all-zeros test cannot see).
6. The `audit_sampling` suite blind to a **pinned or range-restricted audit set**: mutations putting
   query 0 in every sample, or restricting the draw to the first 2,000 queries, passed all 7 tests.
7. A `measure_recall_with_samples` delegation test comparing 1.0 to 1.0, because a k-sample draw is a
   **prefix-subset** of the m-sample draw.
8. A proposed tolerance test using **identical** database rows — the hit test is `<=`, so it holds
   even at `tolerance_sq = 1.0`. Replaced with rows 4 f32 ulps apart.

**Lesson for the remaining plans: mutation-check every new assertion, and prefer absolute values over
relations.** Several of these were transcribed verbatim from plan text. **[CHECK ALL THREE]**

## Deliberately parked (real, not blocking)

- `scripts/test_algorithm`'s `--audit-salt` passthrough has **no execution evidence** — two
  *pre-existing* lines need Python 3.12 and both boxes run 3.10 (verified unchanged at `f0af13a0`).
  Parse-checked only. The single edit in this plan with no run behind it.
- `predictable_salt_warning`'s `#[allow(dead_code)]` removes the compiler's backstop, so deleting its
  call site would be silent. The warning is a diagnostic; the control is passing a real salt.
- `an_answer_outside_the_tolerance_is_a_miss` has a 4×-squared sensitivity floor; the middle of that
  range is covered incidentally by `second_nearest_answers_score_a_miss`.
- The `num_samples > num_queries` clamp is not exercised *through* the GPU function. The code is
  correct (it divides by `ids.len()`); this is a test gap. Relevant if `vs-evaluate` calls with
  `u32::MAX`, which the code comment invites. **[CHECK ALL THREE]**
- The constant-linkage test is brittle to reformatting (`const AUDIT_TQ : u32 = 18;` matches nothing)
  — but it fails loudly, never silently.

## Note for unification (Task U1)

`evaluate_solution` gained an `audit_salt` parameter, so `pentest-harness` does not compile until its
Task 12 lands. **This is by design.**

`measure_recall_with_samples` was added so spec contract C6 (`vs-evaluate` reports recall **exact over
all 7,000 queries**) is reachable at all — `measure_recall` alone silently returns a 1,000-sample
estimate, a wrong number that looks right. **The harness must call the `_with_samples` form for C6.**
