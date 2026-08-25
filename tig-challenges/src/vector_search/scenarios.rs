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
    /// quality = (offset - avg_dist) / scale. Per-scenario because mean
    /// distance scales with dims and clustering; one shared pair cannot span
    /// two corpora.
    pub quality_offset: f64,
    pub quality_scale: f64,
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
                quality_offset: 1.616563,
                quality_scale: 6.399004,
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
    fn sift_config_matches_calibrated_constants() {
        let c = ScenarioConfig::from(Scenario::SIFT_128);
        assert_eq!(c.n_queries, 7_000);
        assert_eq!(c.database_size, 700_000);
        assert_eq!(c.vector_dims, 128);
        assert_eq!(c.quality_offset, 1.616563);
        assert_eq!(c.quality_scale, 6.399004);
        assert!(!c.weights.is_empty());
    }
}
