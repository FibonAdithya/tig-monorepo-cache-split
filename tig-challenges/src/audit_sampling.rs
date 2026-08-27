use rand::{rngs::StdRng, Rng, SeedableRng};

/// Pick which queries the recall audit checks.
///
/// Seeded from the audit salt, NOT from the instance seed. The instance seed
/// reaches the algorithm — `Challenge.seed` is public, and even if it were not,
/// `tig-runtime` takes RAND_HASH and NONCE on argv and the algorithm runs
/// in-process, so it can recompute the seed from /proc/self/cmdline. An
/// algorithm that knows the audit set brute-forces only those queries for
/// `num_samples / num_queries` of the honest work.
///
/// `StdRng::seed_from_u64` matches how tig-protocol draws `sampled_nonces`.
pub fn sample_query_ids(
    salt: &[u8; 32],
    num_queries: u32,
    num_samples: u32,
) -> Vec<u32> {
    let n = num_samples.min(num_queries) as usize;
    let mut rng = StdRng::seed_from_u64(u64::from_le_bytes(
        salt[0..8].try_into().expect("salt is 32 bytes"),
    ));
    let mut all: Vec<u32> = (0..num_queries).collect();
    // Partial Fisher-Yates, written out rather than SliceRandom::shuffle:
    // tig-challenges pins rand with `default-features = false` and only the
    // `std_rng` / `small_rng` features, so the `seq` traits are not guaranteed
    // present. `gen_range` needs only `Rng`.
    //
    // `all.len() - i` is at least 1 for every i < n <= all.len(), so the range
    // is never empty and gen_range cannot panic.
    for i in 0..n {
        let j = i + rng.gen_range(0..(all.len() - i));
        all.swap(i, j);
    }
    all.truncate(n);
    all.sort_unstable();
    all
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_is_deterministic_in_the_salt() {
        let salt = [7u8; 32];
        assert_eq!(sample_query_ids(&salt, 7000, 1000), sample_query_ids(&salt, 7000, 1000));
    }

    #[test]
    fn a_different_salt_gives_a_different_sample() {
        let a = sample_query_ids(&[1u8; 32], 7000, 1000);
        let b = sample_query_ids(&[2u8; 32], 7000, 1000);
        assert_ne!(a, b, "sample must depend on the salt, or the audit is fixed");
    }

    #[test]
    fn sample_has_no_duplicates_and_is_in_range() {
        let s = sample_query_ids(&[3u8; 32], 7000, 1000);
        assert_eq!(s.len(), 1000);
        let mut seen = std::collections::HashSet::new();
        for &i in &s {
            assert!(i < 7000, "index {} out of range", i);
            assert!(seen.insert(i), "duplicate index {} — a repeated query is one \
                audited query counted twice, which biases recall toward it", i);
        }
    }

    #[test]
    fn asking_for_more_samples_than_queries_yields_every_query_once() {
        let s = sample_query_ids(&[4u8; 32], 10, 1000);
        assert_eq!(s.len(), 10);
    }

    #[test]
    fn sample_is_sorted_ascending() {
        // The kernel reads sample_query_ids in order; sorted order makes the
        // query-vector reads sequential rather than scattered.
        let s = sample_query_ids(&[5u8; 32], 7000, 1000);
        let mut sorted = s.clone();
        sorted.sort_unstable();
        assert_eq!(s, sorted);
    }
}
