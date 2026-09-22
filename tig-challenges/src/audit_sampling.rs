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
        salt[0..8].try_into().expect("slice is exactly 8 bytes"),
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
        assert_eq!(s, (0..10).collect::<Vec<u32>>());
    }

    #[test]
    fn only_the_first_eight_salt_bytes_select_the_sample() {
        // Deliberate: the seed is drawn from salt[0..8], matching how tig-protocol
        // draws sampled_nonces. This test exists so the truncation is a recorded
        // decision rather than an accident -- if the derivation is ever widened to
        // the full 32 bytes, this test must be updated on purpose.
        let a = [9u8; 32];
        let mut b = [9u8; 32];
        b[8] = 1; // differs only OUTSIDE the read window
        assert_eq!(sample_query_ids(&a, 7000, 1000), sample_query_ids(&b, 7000, 1000));
    }

    #[test]
    fn zero_queries_yields_an_empty_sample() {
        assert!(sample_query_ids(&[1u8; 32], 0, 100).is_empty());
    }

    #[test]
    fn no_query_is_audited_under_every_salt() {
        // The audit set must MOVE with the salt, in full. A sampler that pins
        // even one query -- a partial Fisher-Yates that skips `i == 0`, say,
        // leaving query 0 in slot 0 under every seed -- hands an algorithm a
        // query it can brute-force once and answer for free ever after. None of
        // the tests above can see that: two salts still give different samples,
        // and the sample is still deduplicated, in range and sorted.
        //
        // 64 fixed salts, so this is deterministic. Under the honest sampler a
        // given query survives all 64 intersections with probability (1/7)^64,
        // so the expected size of `common` is 7000 * (1/7)^64 -- zero for any
        // purpose. Under a sampler that pins a query it is exactly 1.
        let mut common: Option<std::collections::HashSet<u32>> = None;
        for salt_byte in 0..64u8 {
            let s: std::collections::HashSet<u32> =
                sample_query_ids(&[salt_byte; 32], 7000, 1000)
                    .into_iter()
                    .collect();
            common = Some(match common {
                None => s,
                Some(prev) => prev.intersection(&s).copied().collect(),
            });
        }
        let mut common: Vec<u32> = common.expect("64 salts were drawn").into_iter().collect();
        common.sort_unstable();
        assert!(
            common.is_empty(),
            "queries {:?} are audited under all 64 salts, so the audit set is \
             partly fixed and an algorithm can brute-force just those",
            common
        );
    }

    #[test]
    fn the_sample_spans_the_whole_query_range() {
        // A sampler that draws only from a PREFIX of the range --
        // `(0..num_queries.min(2000))` instead of `(0..num_queries)`, say --
        // passes every other test in this module: the sample is still
        // salt-dependent, deduplicated, in range, sorted, and 1,000 long. Only
        // its spread gives it away, and only the top of the range does: a
        // prefix-restricted sampler still starts near 0.
        //
        // Fixed salt, so this either always passes or always fails; it cannot
        // flake. The bounds are not tuned to this salt -- drawing 1,000 of
        // 7,000 without replacement, P(nothing below 70) = (6/7)^70 ~= 2e-5,
        // and the same at the top -- so almost every salt satisfies them.
        // If this ever has to change, change the SALT, not the thresholds:
        // widening them is exactly what would blind it to a restricted range.
        let s = sample_query_ids(&[11u8; 32], 7000, 1000);
        assert_eq!(s.len(), 1000);
        assert!(
            s[0] < 70,
            "lowest audited query is {}; the sample does not reach the bottom \
             of the query range",
            s[0]
        );
        let last = *s.last().unwrap();
        assert!(
            last > 6930,
            "highest audited query is {}; the sample does not reach the top of \
             the query range, so most queries are never audited",
            last
        );
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
