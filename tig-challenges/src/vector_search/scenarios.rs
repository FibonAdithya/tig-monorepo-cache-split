use anyhow::{anyhow, Result};

/// One scenario per real embedding corpus. Tracks are one-to-one with
/// scenarios, so adding a variant adds a track.
///
/// Adding a variant is a runtime-side change only: generation kernels come from
/// each algorithm's PTX, while blobs and config come from the runtime's own
/// tig-challenges. Do not change kernels.cu to add one — that would force every
/// algorithm to be rebuilt and resubmitted network-wide.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(non_camel_case_types)]
pub enum Scenario {
    SIFT_128,
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
            Scenario::SIFT_128 => ScenarioConfig {
                n_queries: 7_000,
                database_size: 700_000,
                vector_dims: 128,
                // Reuses generator.rs's existing constant. Do NOT add a second
                // include_bytes! of the same file here: Rust does not guarantee
                // that two identical consts in different modules are merged, so
                // it can embed the 7 MB blob twice -- in every algorithm .so,
                // since they all link tig-challenges.
                weights: super::generator::V1_BLOB,
                min_recall: 0.9,
                recall_tolerance: 1e-6,
                audit_samples: 1_000,
            },
        }
    }
}

impl std::fmt::Display for Scenario {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Scenario::SIFT_128 => write!(f, "sift_128"),
        }
    }
}

impl std::str::FromStr for Scenario {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "sift_128" => Ok(Scenario::SIFT_128),
            _ => Err(anyhow!("Invalid scenario type: {}", s)),
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
    fn scenario_from_str_rejects_unknown() {
        let err = Scenario::from_str("glove_300").unwrap_err();
        assert!(
            err.to_string().contains("glove_300"),
            "error should name the offending input, got: {}",
            err
        );
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
