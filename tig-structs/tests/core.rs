use tig_structs::core::{BenchmarkSettings, CPUArchitecture, OutputData};
use tig_utils::{jsonify, u8s_from_str, MerkleHash};

#[test]
fn test_calc_solution_signature() {
    let output_data = OutputData {
        nonce: 123,
        runtime_signature: 456,
        fuel_consumed: 789,
        solution: "test".to_string(),
        cpu_arch: CPUArchitecture::AMD64,
    };

    // Assert same as Python version: tig-benchmarker/tests/core.rs
    assert_eq!(output_data.calc_solution_signature(), 11204800550749450632);
}

#[test]
fn test_calc_seed() {
    let settings = BenchmarkSettings {
        player_id: "some_player".to_string(),
        block_id: "some_block".to_string(),
        challenge_id: "some_challenge".to_string(),
        algorithm_id: "some_algorithm".to_string(),
        track_id: "a=1,b=2".to_string(),
    };

    let rand_hash = "random_hash".to_string();
    let nonce = 1337;

    // Assert same as Python version: tig-benchmarker/tests/core.rs
    assert_eq!(
        settings.calc_seed(&rand_hash, nonce),
        [
            84, 136, 44, 57, 142, 50, 248, 37, 94, 195, 254, 190, 222, 27, 136, 115, 229, 136, 19,
            207, 7, 208, 15, 193, 111, 99, 209, 131, 27, 189, 226, 175
        ]
    );
}

#[test]
fn test_calc_db_seed() {
    let settings = BenchmarkSettings {
        player_id: "some_player".to_string(),
        block_id: "some_block".to_string(),
        challenge_id: "some_challenge".to_string(),
        algorithm_id: "some_algorithm".to_string(),
        track_id: "a=1,b=2".to_string(),
    };
    let rand_hash = "random_hash".to_string();

    // Assert same as Python version: tig-benchmarker/tests/data.py
    assert_eq!(
        settings.calc_db_seed(&rand_hash),
        [
            209, 209, 150, 41, 179, 131, 168, 223, 27, 59, 221, 124, 237, 86, 161, 52, 118, 79,
            8, 0, 171, 205, 118, 2, 64, 244, 59, 240, 176, 44, 51, 185
        ]
    );
}

#[test]
fn test_db_seed_carries_its_domain_tag() {
    // Catches dropping the `_db` suffix, which would make this seed the
    // plain hash of "{settings}_{rand_hash}".
    let settings = BenchmarkSettings {
        player_id: "some_player".to_string(),
        block_id: "some_block".to_string(),
        challenge_id: "some_challenge".to_string(),
        algorithm_id: "some_algorithm".to_string(),
        track_id: "a=1,b=2".to_string(),
    };
    let rand_hash = "random_hash".to_string();
    let untagged = u8s_from_str(&format!("{}_{}", jsonify(&settings), rand_hash));
    assert_ne!(settings.calc_db_seed(&rand_hash), untagged);
}

#[test]
fn test_outputdata_to_merklehash() {
    let output_data = OutputData {
        nonce: 123,
        runtime_signature: 456,
        fuel_consumed: 789,
        solution: "test".to_string(),
        cpu_arch: CPUArchitecture::AMD64,
    };

    let merkle_hash: MerkleHash = output_data.into();

    // Assert same as Python version: tig-benchmarker/tests/core.rs
    assert_eq!(
        merkle_hash,
        MerkleHash([
            79, 126, 186, 90, 12, 111, 100, 8, 120, 150, 225, 176, 200, 201, 125, 150, 58, 122,
            214, 216, 68, 6, 125, 247, 248, 4, 165, 185, 157, 44, 13, 151
        ])
    );
}
