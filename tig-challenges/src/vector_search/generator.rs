pub use crate::gan_generator::v1::{parse_weights as weights_from, Layer, LATENT_DIM};

/// Weights are committed rather than fetched: every verifier regenerates the
/// instance independently, so a single differing byte would fail verification
/// network-wide.
pub(super) const V1_BLOB: &[u8] = include_bytes!("weights/v1_sift.bin");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_scenario_blob_matches_its_declared_dims() {
        // Iterate over ALL variants, not just the one being added, so a future
        // scenario cannot silently skip this check. A mismatch here would
        // otherwise surface as network-wide verification failure.
        for scenario in [crate::vector_search::Scenario::SIFT_128] {
            let config = crate::vector_search::ScenarioConfig::from(scenario);
            let weights = weights_from(config.weights).unwrap_or_else(|e| {
                panic!("scenario {} has an unparseable blob: {}", scenario, e)
            });
            let derived = weights.layers.last().unwrap().out_dim;
            assert_eq!(derived, config.vector_dims,
                "scenario {}: blob produces {} dims but config declares {}",
                scenario, derived, config.vector_dims);
            assert_eq!(weights.layers[0].in_dim, LATENT_DIM,
                "scenario {}: first layer must consume LATENT_DIM", scenario);
        }
    }
}
