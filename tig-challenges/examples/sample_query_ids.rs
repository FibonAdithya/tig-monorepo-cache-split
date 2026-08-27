//! Print `audit_sampling::sample_query_ids(&salt, num_queries, num_samples)`.
//!
//! Exists because the audit sample was previously derivable only by running the
//! GPU lane: `sample_query_ids` is monorepo-only and had no caller outside the
//! crate's own `#[cfg(test)]` module, so unification Task U3 Step 5 (build a
//! solution targeting salt A's sample, score it under A and under B) could not
//! be prepared off the GPU box at all.
//!
//! It is an `example`, not a `bin`, so it is not shipped in any release build
//! and adds nothing to the `.so` an algorithm links.
//!
//! `audit_sampling` is UNGATED in `lib.rs` (`pub mod audit_sampling;` with no
//! `#[cfg(feature = ...)]`) and is pure host-side Rust over `rand`. The `c004`
//! feature pulls in `cudarc`, whose build script needs nvcc; this example needs
//! neither, so it builds and runs on a machine with no CUDA at all.
//!
//! Usage:
//!
//! ```text
//! cargo run --quiet -p tig-challenges --example sample_query_ids -- <64-hex-salt> [num_queries] [num_samples]
//! ```
//!
//! Defaults are the `s=sift_128` scenario's shape: 7000 queries, 1000 samples.
//! Ids go to stdout one per line (so the output pipes into `comm`/`sort`/
//! `python3` unmodified); the postcondition summary goes to stderr.

use tig_challenges::audit_sampling::sample_query_ids;

fn parse_salt(hex: &str) -> [u8; 32] {
    // Hand-rolled rather than pulled from a hex crate: tig-challenges does not
    // depend on one, and adding a dependency to derive a test fixture would be
    // a change to the shipped crate rather than to this example.
    assert_eq!(
        hex.len(),
        64,
        "salt must be exactly 64 hex chars (32 bytes), got {}",
        hex.len()
    );
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16)
            .unwrap_or_else(|e| panic!("byte {} of the salt is not hex: {}", i, e));
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "usage: {} <64-hex-salt> [num_queries=7000] [num_samples=1000]",
            args[0]
        );
        std::process::exit(2);
    }
    let salt = parse_salt(&args[1]);
    let num_queries: u32 = args
        .get(2)
        .map(|s| s.parse().expect("num_queries must be a u32"))
        .unwrap_or(7_000);
    let num_samples: u32 = args
        .get(3)
        .map(|s| s.parse().expect("num_samples must be a u32"))
        .unwrap_or(1_000);

    let ids = sample_query_ids(&salt, num_queries, num_samples);

    // Contract C1's postconditions, asserted here rather than eyeballed: length
    // is min(num_samples, num_queries), strictly ascending (which is sorted AND
    // duplicate-free in one comparison), and every id in range.
    let expected_len = num_samples.min(num_queries) as usize;
    assert_eq!(
        ids.len(),
        expected_len,
        "C1 length: expected {}, got {}",
        expected_len,
        ids.len()
    );
    assert!(
        ids.windows(2).all(|w| w[0] < w[1]),
        "C1 order: sample is not strictly ascending, so it is unsorted or has duplicates"
    );
    assert!(
        ids.iter().all(|&i| i < num_queries),
        "C1 range: some id is >= num_queries"
    );

    for id in &ids {
        println!("{}", id);
    }
    eprintln!(
        "salt={} num_queries={} num_samples={} -> len={} strictly_ascending=true \
         min={} max={}",
        args[1],
        num_queries,
        num_samples,
        ids.len(),
        ids.first().map(|v| v.to_string()).unwrap_or("-".into()),
        ids.last().map(|v| v.to_string()).unwrap_or("-".into()),
    );
}
