# Per-scenario GAN tracks for vector_search

Design, 2026-08-25. Supersedes the track model in
`2026-08-13-gan-instance-generation-design.md`; that document's generator,
kernel, and determinism work all stand unchanged.

## Motivation

Today a `vector_search` track is a size and nothing else:

```rust
impl_kv_string_serde! { Track { n_queries: u32 } }   // mod.rs:13-17
let database_size = 100 * track.n_queries;           // mod.rs:77
```

Five tracks, `n_queries` 7,000 to 15,000, all drawing from one generator
trained on SIFT. An algorithm that handles one track handles them all; the
tracks differ in how much work there is, never in what the data looks like.

The goal of this redesign is **realism and coverage**: each track should draw
from a different real embedding corpus, so that qualifying means handling data
practitioners actually have rather than occupying a size bucket.

Explicit non-goals. This design does **not** attempt to make the competition
discriminating, does not touch the fuel economics, and does not treat
`database_size` as a tuning lever. Those are real problems — see Risks — but
they are governed by challenge parameters, not by the instance distribution,
and conflating them with this change would muddy both.

## The architectural fact everything hangs on

Instance generation is split across two binaries, and the split determines what
a future scenario costs. Verified by reading `tig-runtime/src/main.rs:196-217`:
the CUDA path reads the **algorithm's** PTX file, loads it as a `CudaModule`,
and passes that module into `Challenge::generate_instance`.

| Lives in | Sourced from | Changing it requires |
|---|---|---|
| `gan_linear`, `gan_sample_latents`, `evaluate_total_distance` | the **algorithm's PTX** | every algorithm rebuilt and resubmitted |
| Weight blobs, `ScenarioConfig`, calibration constants, `Track` parsing | **runtime/verifier's own** `tig-challenges` | nothing on the algorithm side |

So adding a scenario after this migration is a runtime-side change only — no
second resubmit — **provided `kernels.cu` does not change**.

That yields the load-bearing constraint of this design:

> **Freeze the kernel ABI.** Every generator must belong to one architecture
> family — an MLP with LeakyReLU(0.2), layer dimensions carried in the weight
> blob — so `kernels.cu` never changes when a scenario is added. `gan_linear`
> already takes its dimensions as kernel arguments, so this holds today.

The cost of the constraint is that a future scenario cannot use a
different generator architecture (a convolutional generator, say) without
forcing another network-wide resubmit. That is accepted deliberately.

## Design

### Types

New file `tig-challenges/src/vector_search/scenarios.rs`, mirroring the
established shape of `job_scheduling/scenarios.rs`:

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Scenario {
    SIFT_128,
}

pub struct ScenarioConfig {
    pub n_queries:      u32,
    pub database_size:  u32,
    pub vector_dims:    usize,          // expected; asserted against the blob
    pub weights:        &'static [u8],
    pub quality_offset: f64,
    pub quality_scale:  f64,
}

impl From<Scenario> for ScenarioConfig { /* one arm per variant */ }
```

`Track` becomes `Track { s: Scenario }`, matching `energy_arbitrage`. The size
axis is removed rather than multiplied — see Track cardinality below.

### What the track macro demands of the enum

`impl_kv_string_serde` (`lib.rs:15-88`) is not derive-based, and imposes two
requirements that are easy to miss:

- **Serialize** builds `format!("{}={}", field, self.field)`, so `Scenario` must
  implement `std::fmt::Display`.
- **Deserialize** calls `.parse::<$ty>()` and maps the error through
  `E::custom`, so `Scenario` must implement `std::str::FromStr` with an error
  type that implements `Display`. `anyhow::Error` satisfies this.

`job_scheduling::Scenario` implements both by hand (`scenarios.rs:54-79`) and is
the template. Its convention is lowercase snake_case on write and
case-insensitive on read, so the wire form here is:

```
s=sift_128
```

This is the literal string the protocol puts in `settings.track_id`, which
`tig-runtime/src/main.rs:110-122` quotes and feeds to `serde_json::from_str`.
Getting it wrong breaks instance generation for the whole track, so the test
below asserts the literal string rather than only a round-trip.

### Why calibration constants must move

`QUALITY_OFFSET` and `QUALITY_SCALE` are currently module-level `const`s
(`mod.rs:56-59`). They move into `ScenarioConfig` because they cannot be shared
across corpora. Quality is linear in mean distance:

```
quality = (offset - avg_dist) / scale
```

and mean inter-point distance scales roughly as `sqrt(vector_dims)` and depends
on the corpus's clustering. A constant pair fitted to SIFT-128 would put another
corpus's quality wildly off the `min_active_quality` band. One pair per
scenario is the minimum workable granularity.

### Why `vector_dims` needs no plumbing

`vector_dims` is already derived from the generator's final layer
(`mod.rs:72`), not from the track. Varying it per scenario therefore requires no
new mechanism — it follows automatically from loading a different blob. The
`vector_dims` field in `ScenarioConfig` exists only as an assertion target, so a
mismatched blob fails at test time rather than network-wide.

### Weight blobs

One blob per scenario, each its own `include_bytes!`, with `ScenarioConfig`
holding a `&'static [u8]`. The existing parser is unchanged — it already parses
exactly this format, and its six hardening tests (bad magic, truncation,
trailing bytes, absurd layer count, element-count overflow, dims/values) keep
passing unmodified.

Rejected: a single concatenated multi-generator blob with an index header. It
would add new parsing code and new failure modes to the security-sensitive path
that was deliberately hardened, in exchange for one checksum instead of two.

`generator.rs` gains `weights_from(blob: &'static [u8]) -> Result<GeneratorWeights>`;
`v1_weights()` becomes a thin wrapper over it so existing callers and tests are
untouched.

### Track cardinality, and `n_queries` for the SIFT track

`Track { s }` means tracks and scenarios are one-to-one. The SIFT-only
milestone therefore ships as a **single track**, replacing today's five.

The single track uses **`n_queries = 7,000`**, chosen on a noise argument rather
than a size one. Bundle quality is the *median* of `num_nonces_per_bundle`
nonces (`tig-protocol/src/contracts/benchmarks.rs:218`), and that count varies
sharply by track in the live config: 20 nonces at `n_queries=7000`, falling to 5
at 15,000. Measured per-nonce sigma is ~500 quality units, the project's central
quality problem, and a median of 20 draws suppresses it far better than a median
of 5. Track 7000 is also the cheapest to generate (628 ms against 1,335 ms at
15,000).

A secondary benefit is available but **not automatic**: with one track,
calibration *can* be an exact fit to a single target rather than a compromise
across five, retiring the 282-unit worst-case residual from the previous spec.
Realising it requires re-fitting the constants against the single track, which
needs a live GPU. Until then the shipped constants remain the inherited joint
fit (residual ≤138 at `n_queries=7000`), which is within tolerance but is not
the exact fit. The ~500-unit per-nonce noise is unaffected either way.

`database_size` stays `100 * n_queries` in value but moves into `ScenarioConfig`
as an explicit field, so a future high-dimensional scenario can override it
without restructuring. This matters: at 960 dimensions a 1.5M-vector database
would need 5.76 GB of VRAM and would not fit an 8 GB card. DEEP-96 is well
inside budget, so the field is precaution, not present need.

## Staging

**Milestone 1 — SIFT only.** `Scenario::SIFT_128` alone, reusing the already
trained and validated v1 generator. This proves the whole mechanism: track
serde, per-scenario config, per-scenario calibration, verification. No new GAN
training is required, so the milestone isolates the structural change from the
modelling work.

**Milestone 2 — add DEEP.** `Scenario::DEEP_96`, trained on the DEEP1B GoogLeNet
descriptor corpus (`deep-image-96-angular` in ann-benchmarks). Runtime-side only
per the split above.

One property of DEEP must be confirmed before training rather than assumed. DEEP
is natively an **angular** corpus while `evaluate_total_distance` computes
Euclidean distance. This is benign if the vectors are unit-norm: for
`‖x‖ = ‖y‖ = 1`, `d² = 2 − 2·cos`, so Euclidean and cosine rank identically and
1-NN results are unchanged. DEEP1B descriptors are normally distributed
L2-normalised, so this is expected to hold — but it should be checked against
the actual corpus, because if it does not hold the scoring would not reflect how
the corpus is used in practice.

Note that DEEP-96 is *lower*-dimensional than SIFT-128. It therefore relieves no
VRAM pressure and, if anything, makes exact 1-NN slightly cheaper. It is chosen
for corpus realism, not for difficulty.

## Testing

The 8 existing `tig-challenges` generator tests must stay green unmodified;
that is the regression bar for the refactor.

New tests:

1. **Track wire format.** Assert that `Track { s: Scenario::SIFT_128 }`
   serializes to the literal string `s=sift_128`, and that that exact string
   deserializes back. A round-trip assertion alone is **not** sufficient: it
   passes for any self-consistent encoding, including a wrong one that the
   protocol cannot produce. The literal assertion is what catches a renamed
   variant or a changed case convention.
2. **Blob/scenario dimension agreement.** For every `Scenario` variant, assert
   that `vector_dims` derived from the loaded blob equals the variant's
   `ScenarioConfig::vector_dims`. This is the test that makes a mismatched blob
   a compile-and-test failure instead of a network-wide verification failure.
   It must iterate over all variants, not the one being added, so a future
   scenario cannot skip it.
3. **Per-scenario calibration lands in band.** For each scenario, solve
   generated instances with the reference exact 1-NN solver over a **fixed,
   explicitly listed set of at least 24 nonces** — never a random or
   wall-clock-derived selection, or the test is flaky by construction. Then
   assert both:
   - every individual nonce clears `min_active_quality` (68,500); and
   - the *mean* over those nonces is within **±400 quality units** of the
     scenario's target (71,862 for SIFT-128 at `n_queries=7000`).

   The band is derived, not guessed: per-nonce sigma is ~500, so the standard
   error of a 24-nonce mean is ~102 and ±400 is close to 4 SE. Tightening it
   below ~±300 would make the test fail on sampling noise alone; loosening it
   past ~±800 would stop catching a genuinely mis-fitted constant. Assert the
   mean, not individual nonces, against the target — a per-nonce assertion at
   this sigma cannot be both meaningful and stable.
4. **Launch-invariance digests.** Re-run the FNV-1a device-buffer digests across
   forward-chunk and latent-block combinations, since the sizing path changes.
   Keep the negative control that proves the digest is input-sensitive.

Test 3 needs a GPU and belongs with the on-device validation, not the unit
suite. Tests 1 and 2 are CPU-only and run locally.

## Migration

One resubmit of all ~105 algorithms. This is unavoidable and is already required
by the 250→128 dimension change; this design adds no separate migration event.
Two facts from the previous spec still apply and are restated because they cost
real debugging time:

- `build_ptx` compiles the challenge's `.cu` into each algorithm's PTX, so every
  existing binary fails at generation with `CUDA_ERROR_NOT_FOUND` before
  reaching its own dimension guard.
- Production algorithms declare `const DIMS: usize = 250` and guard on it; they
  also hardcode track sizes.

The new obligation on algorithm authors is to read `challenge.vector_dims` at
runtime instead of hardcoding. That is precisely what makes Milestone 2, and
every scenario after it, additive.

## Risks

- **Per-nonce noise is untouched.** Sigma remains ~500 quality units against a
  mainnet qualifier spread of ~80–110. This redesign does not address it and
  should not be credited with doing so.
- **Coverage may be nominal.** Whether DEEP-96 instances are distinguishable
  from SIFT-128 instances *to an ANN algorithm* is an empirical question. Two
  GANs can produce data more similar to each other than their source corpora
  are, in which case "different distribution per track" is true of the training
  data but not of the instances. This should be measured once both generators
  exist — a natural test is whether an algorithm tuned on one scenario loses
  measurable quality on the other.
- **Qualifier economics are deferred, not resolved.** The SIFT-only milestone is
  one track, so 100 qualifier slots against today's 500 across five tracks. The
  two incumbent algorithms hold 293 and 207 qualifiers, exactly today's
  capacity. The decision — raise `max_qualifiers_per_track`, accept fewer slots,
  or reintroduce a size axis as `Track { n_queries, s }` — is deliberately left
  until a single-scenario track has been observed running.
- **Kernel ABI freeze is a one-way door.** A future generator needing different
  kernels forces another network-wide resubmit.

## Open questions

1. Qualifier-slot economics, deferred as above.
2. Confirmation that the DEEP corpus is unit-norm, before Milestone 2 training.
3. Cross-architecture determinism remains open from the previous spec and is
   unaffected by this design either way.
