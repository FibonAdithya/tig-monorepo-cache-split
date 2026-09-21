use anyhow::{anyhow, Result};

/// Weights are committed rather than fetched: every verifier regenerates the
/// instance independently, so a single differing byte would fail verification
/// network-wide.
///
/// Each blob gets exactly ONE `include_bytes!`, here, referenced once from its
/// own arm below. Rust does not guarantee that two identical consts in
/// different modules are merged, so a second `include_bytes!` of the same file
/// can embed that blob's 7 MB twice -- in every algorithm .so, since they all
/// link tig-challenges. (`gan_generator`'s tests include this SIFT blob a second
/// time, and `v1_sift.bin` as well, but only under `#[cfg(test)]`, so no shipped
/// build carries either copy.)
const SIFT_128_BLOB: &[u8] = include_bytes!("weights/sift_128_v4.bin");
const GLOVE_100_BLOB: &[u8] = include_bytes!("weights/glove_100_v1.bin");
const NYTIMES_256_BLOB: &[u8] = include_bytes!("weights/nytimes_256_v3.bin");

/// Generates the enum, `Scenario::ALL`, `Display` and `FromStr` from one list.
/// `ALL`'s length is computed from the same list the enum is built from, so a
/// variant cannot exist without being in `ALL`, and the wire name cannot drift
/// between `Display` and `FromStr`.
macro_rules! scenarios {
    ($($variant:ident => $wire:literal),+ $(,)?) => {
        /// One scenario per real embedding corpus. Tracks are one-to-one with
        /// scenarios, so adding a variant adds a track.
        ///
        /// Adding a variant is a runtime-side change only — generation kernels
        /// come from each algorithm's PTX, while blobs and config come from the
        /// runtime's own tig-challenges — PROVIDED the new blob's architecture
        /// is one of the three the kernels already support. Do not change
        /// kernels.cu to add a scenario: that would force every algorithm to be
        /// rebuilt and resubmitted network-wide.
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        #[allow(non_camel_case_types)]
        pub enum Scenario { $($variant),+ }

        impl Scenario {
            pub const ALL: [Scenario; [$($wire),+].len()] = [$(Scenario::$variant),+];
        }

        impl std::fmt::Display for Scenario {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self { $(Scenario::$variant => write!(f, $wire)),+ }
            }
        }

        impl std::str::FromStr for Scenario {
            type Err = anyhow::Error;
            fn from_str(s: &str) -> Result<Self> {
                match s.to_lowercase().as_str() {
                    $($wire => Ok(Scenario::$variant),)+
                    _ => Err(anyhow!("Invalid scenario type: {}", s)),
                }
            }
        }
    };
}

scenarios! {
    SIFT_128 => "sift_128",
    GLOVE_100 => "glove_100",
    NYTIMES_256 => "nytimes_256",
}

pub struct ScenarioConfig {
    pub n_queries: u32,
    pub database_size: u32,
    /// Expected dims. Asserted against the blob's final layer so a mismatched
    /// blob fails in tests rather than network-wide.
    pub vector_dims: usize,
    pub weights: &'static [u8],
    /// Recall@1 a solution must declare to qualify. Per scenario because the
    /// achievable recall/speed frontier depends on the corpus.
    /// Currently 0.9, and deliberately NOT the 0.95 this shipped with.
    ///
    /// The floors, from `docs/measurements/2026-08-27-c004-d2-over-d1.md`
    /// (tig-pentesting repo): 0.281 measured on an oracle worst-case seed, and
    /// 0.79 under an unmeasured near-tie hypothesis. 0.9 clears both — with the
    /// audit's one-sided 3-sigma tolerance at m=1,000 samples it has a
    /// worst-case pass point of 0.8715, still above the unmeasured 0.79 — so
    /// dropping to it does not reach either floor.
    ///
    /// It MUST stay equal to `_MIN_RECALL_BY_TRACK` in the tig-pentesting
    /// repo's `challenges/vector_search.py`. The two live in different repos
    /// with no test that can see both, and they disagreed for a while: this
    /// said 0.95 while the plugin said 0.9. That is worse than either value on
    /// its own, because `build_goal_prompt` PREFERS the bar `vs-generate`
    /// reports from here, while PentestMemory qualifies rounds against the
    /// plugin constant — so the agent aims at one bar and the harness scores
    /// against another, and every round between the two counts as qualifying
    /// here and would be rejected by the network.
    ///
    /// The upper end is still not measured — no ANN method has been run
    /// against it — so "this bar is not trivially cleared by an approximate
    /// method" remains judgement, not measurement. A cheap ANN clearing it is
    /// grounds to raise `r`, not to reconsider the design.
    pub min_recall: f32,
    /// A returned vector counts as a hit when its distance is within this
    /// relative tolerance of the true minimum. Absorbs cross-architecture float
    /// noise and makes an equidistant alternative a hit by construction.
    pub recall_tolerance: f32,
    /// Queries the audit checks. Verification cost is linear in this.
    pub audit_samples: u32,
}

impl From<Scenario> for ScenarioConfig {
    fn from(scenario: Scenario) -> Self {
        match scenario {
            // SIFT v4, the gate-accepted checkpoint `v4_sift1m_x100k`
            // (`/workspace/sift-v4/v4_sift1m_x100k/best_generator.pt` on the
            // training box, EMA weights at step 86,000 of a 100k-step retrain;
            // see weights/PROVENANCE.md for the hashes and the exporter command).
            // Structured-gate architecture: every output coordinate passes a
            // stochastic hard gate, so rows are non-negative and unit-norm with
            // about 24% of coordinates exactly zero -- 0.239 for v4 against
            // 0.230 for the real corpus. A Euclidean corpus, unlike GloVe and
            // NYTimes: nothing here makes Euclidean 1-NN stand in for angular
            // 1-NN, because SIFT descriptors are compared in Euclidean distance
            // to begin with.
            //
            // min_recall's 0.9 was derived from measurements taken on
            // v1-generated SIFT instances, and has NOT been re-derived on the v4
            // instances this arm now generates. The field comment below records
            // where the figure came from.
            //
            // `weights/v1_sift.bin` is no longer wired to any scenario. It
            // survives only as the `TIGGAN01` fixture in `gan_generator::v1`,
            // under `#[cfg(test)]`, where the v1 parser's own tests live.
            Scenario::SIFT_128 => ScenarioConfig {
                n_queries: 7_000,
                database_size: 700_000,
                vector_dims: 128,
                weights: SIFT_128_BLOB,
                min_recall: 0.9,
                recall_tolerance: 1e-6,
                audit_samples: 1_000,
            },
            // GloVe v1, seed 42 (WGAN run probe_spectrum_seed42). Angular
            // corpus: rows are unit-normalised by the mlp driver, which makes
            // Euclidean 1-NN equal to angular 1-NN. min_recall is copied from
            // SIFT and has NOT been measured for this corpus; see the spec's
            // Follow-ups.
            Scenario::GLOVE_100 => ScenarioConfig {
                n_queries: 7_000,
                database_size: 700_000,
                vector_dims: 100,
                weights: GLOVE_100_BLOB,
                min_recall: 0.9,
                recall_tolerance: 1e-6,
                audit_samples: 1_000,
            },
            // NYTimes v3, seed 42, the gate-accepted checkpoint `v3_best`
            // (`/workspace/nytimes-v3/v3_seed42/best_generator.pt` on the
            // training box). Spherical architecture, so rows are unit-norm by
            // construction rather than by a normalise step: the output is
            // cos_r * u + sin_r * t with u and t orthonormal, whose norm is
            // sqrt(cos_r^2 + sin_r^2) = 1. Angular corpus, and on unit vectors
            // Euclidean 1-NN is the same ranking as angular 1-NN. min_recall is
            // copied from SIFT and has NOT been measured for this corpus; see
            // the spec's Follow-ups.
            Scenario::NYTIMES_256 => ScenarioConfig {
                n_queries: 7_000,
                database_size: 700_000,
                vector_dims: 256,
                weights: NYTIMES_256_BLOB,
                min_recall: 0.9,
                recall_tolerance: 1e-6,
                audit_samples: 1_000,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn scenario_display_is_lowercase_snake_case() {
        // The literal string matters: it is what the protocol puts in
        // settings.track_id. A round-trip test alone would pass for any
        // self-consistent encoding, including one the protocol cannot emit.
        assert_eq!(Scenario::SIFT_128.to_string(), "sift_128");
    }

    #[test]
    fn scenario_from_str_is_case_insensitive() {
        assert_eq!(Scenario::from_str("sift_128").unwrap(), Scenario::SIFT_128);
        assert_eq!(Scenario::from_str("SIFT_128").unwrap(), Scenario::SIFT_128);
    }

    #[test]
    fn glove_wire_string_is_literal() {
        // The literal, for the same reason the SIFT test above pins one: this
        // is what the protocol puts in settings.track_id.
        assert_eq!(Scenario::GLOVE_100.to_string(), "glove_100");
        assert_eq!(Scenario::from_str("glove_100").unwrap(), Scenario::GLOVE_100);
        assert_eq!(Scenario::from_str("GLOVE_100").unwrap(), Scenario::GLOVE_100);
    }

    #[test]
    fn nytimes_wire_string_is_literal() {
        // The literal, for the same reason the SIFT and GloVe tests above pin
        // one: this is what the protocol puts in settings.track_id.
        assert_eq!(Scenario::NYTIMES_256.to_string(), "nytimes_256");
        assert_eq!(Scenario::from_str("nytimes_256").unwrap(), Scenario::NYTIMES_256);
        assert_eq!(Scenario::from_str("NYTIMES_256").unwrap(), Scenario::NYTIMES_256);
    }

    #[test]
    fn scenario_from_str_rejects_unknown() {
        // `deep_96`, not `glove_300`: glove_100 is a real scenario now, and a
        // rejection test whose input is one character away from a valid name
        // is one typo away from asserting nothing.
        let err = Scenario::from_str("deep_96").unwrap_err();
        assert!(
            err.to_string().contains("deep_96"),
            "error should name the offending input, got: {}",
            err
        );
    }

    #[test]
    fn every_wire_name_round_trips() {
        // The one thing the macro does NOT guarantee. It keeps `Display` and
        // `FromStr` reading the same literal, so they cannot drift apart, but
        // `from_str` lowercases its input first: an uppercase wire literal like
        // `SIFT_128 => "Sift_128"` would be emitted by `Display` and could
        // never be matched by `FromStr`. A round-trip is the right assertion
        // here precisely because the literal-string tests above already pin
        // what the names are.
        for scenario in Scenario::ALL {
            assert_eq!(Scenario::from_str(&scenario.to_string()).unwrap(), scenario);
        }
    }

    #[test]
    fn every_scenario_blob_matches_its_declared_dims() {
        // Iterate over ALL variants, not just the one being added, so a future
        // scenario cannot silently skip this check. A mismatch here would
        // otherwise surface as network-wide verification failure.
        for scenario in Scenario::ALL {
            let config = ScenarioConfig::from(scenario);
            let generator = crate::gan_generator::Generator::from_blob(config.weights)
                .unwrap_or_else(|e| panic!("scenario {} has an unparseable blob: {}", scenario, e));
            assert_eq!(generator.output_dim(), config.vector_dims, "scenario {}", scenario);
            // `super::super::AUDIT_MAX_DIMS` resolves without any `pub`: a
            // descendant module may name an ancestor's private items. Do not
            // "fix" that constant's visibility -- lib.rs finds it by the exact
            // text `const AUDIT_MAX_DIMS: u32 = `.
            assert!(
                config.vector_dims as u32 <= super::super::AUDIT_MAX_DIMS,
                "scenario {} has {} dims but the audit kernel stages at most {}",
                scenario,
                config.vector_dims,
                super::super::AUDIT_MAX_DIMS
            );
            // The instance shape and the recall bar are shared across
            // scenarios by design; a variant that quietly diverged would move
            // the audit's launch geometry or the qualifying bar without any
            // other test noticing.
            assert_eq!(
                (config.n_queries, config.database_size),
                (7_000, 700_000),
                "scenario {}",
                scenario
            );
            assert_eq!(
                (config.min_recall, config.recall_tolerance, config.audit_samples),
                (0.9, 1e-6, 1_000),
                "scenario {}",
                scenario
            );
        }
    }

    #[test]
    fn sift_config_matches_its_declared_shape() {
        // Renamed from `..._matches_calibrated_constants`: nothing here is
        // calibrated any more. The quality map these numbers were once fitted
        // against (mean distance, via QUALITY_OFFSET/QUALITY_SCALE) is retired,
        // and what this pins now is the instance SHAPE -- how many queries, how
        // many database rows, how many dims -- which is what the audit kernel's
        // launch geometry and the recall bar are reasoned about against.
        let c = ScenarioConfig::from(Scenario::SIFT_128);
        assert_eq!(c.n_queries, 7_000);
        assert_eq!(c.database_size, 700_000);
        assert_eq!(c.vector_dims, 128);
        assert!(!c.weights.is_empty());
    }

    #[test]
    fn sift_128_declares_its_recall_bar() {
        let c = ScenarioConfig::from(Scenario::SIFT_128);
        // Asserted against the value, not `is_finite()` — a bar that silently
        // defaulted to 0.0 would qualify every solution including all-zeros.
        assert_eq!(c.min_recall, 0.9);
        assert_eq!(c.audit_samples, 1_000);
        assert_eq!(c.recall_tolerance, 1e-6);
    }
}
