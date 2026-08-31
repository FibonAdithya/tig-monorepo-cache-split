use anyhow::Result;
use clap::{arg, Command};
use serde_json::{Map, Value};
use std::{fs, io::Read, panic, path::PathBuf};
use tig_challenges::*;
use tig_structs::core::BenchmarkSettings;
use tig_utils::dejsonify;

#[cfg(feature = "cuda")]
use cudarc::{driver::CudaContext, nvrtc::Ptx, runtime::result::device::get_device_prop};

fn cli() -> Command {
    Command::new("tig-verifier")
        .about("Verifies a solution")
        .arg_required_else_help(true)
        .arg(
            arg!(<SETTINGS> "Settings json string or path to json file")
                .value_parser(clap::value_parser!(String)),
        )
        .arg(
            arg!(<RAND_HASH> "A string used in seed generation")
                .value_parser(clap::value_parser!(String)),
        )
        .arg(arg!(<NONCE> "Nonce value").value_parser(clap::value_parser!(u64)))
        .arg(
            arg!(<SOLUTION> "Solution base64 string, path to json file with solution field, or '-' for stdin")
                .value_parser(clap::value_parser!(String)),
        )
        .arg(arg!(--ptx [PTX] "Path to a CUDA ptx file").value_parser(clap::value_parser!(PathBuf)))
        .arg(arg!(--gpu [GPU] "Which GPU device to use").value_parser(clap::value_parser!(usize)))
        .arg(
            arg!(--"audit-salt" <AUDIT_SALT> "32-byte audit salt as 64 hex chars")
                .value_parser(clap::value_parser!(String)),
        )
        .arg(arg!(--verbose "Enable verbose output").action(clap::ArgAction::SetTrue))
}

fn main() {
    let matches = cli().get_matches();

    if let Err(e) = verify_solution(
        matches.get_one::<String>("SETTINGS").unwrap().clone(),
        matches.get_one::<String>("RAND_HASH").unwrap().clone(),
        *matches.get_one::<u64>("NONCE").unwrap(),
        matches.get_one::<String>("SOLUTION").unwrap().clone(),
        matches.get_one::<PathBuf>("ptx").cloned(),
        matches.get_one::<usize>("gpu").cloned(),
        matches.get_one::<String>("audit-salt").cloned(),
        matches.get_one::<bool>("verbose").cloned().unwrap_or(false),
    ) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

/// Decode the `--audit-salt` value, defaulting to all-zeros when the flag is
/// absent.
///
/// The all-zeros default is deliberate and load-bearing in both directions: the
/// five CPU lanes never pass a salt, and local debugging of a GPU lane should
/// not have to invent 64 hex characters. It is safe only because a *valueless*
/// `--audit-salt` is rejected by the parser (the value is `<AUDIT_SALT>`, not
/// `[AUDIT_SALT]`), so an operator cannot reach the predictable zero salt by
/// accident -- only by omitting the flag entirely.
fn parse_audit_salt(audit_salt_hex: Option<String>) -> Result<[u8; 32]> {
    let Some(h) = audit_salt_hex else {
        return Ok([0u8; 32]);
    };
    let bytes = match hex::decode(&h) {
        Ok(bytes) => bytes,
        // An odd-length run of valid hex digits is a LENGTH problem, not a
        // character problem. hex reports it as `OddLength`, and forwarding that
        // verbatim tells an operator who dropped one character off a 64-char
        // salt to go hunting for a bad character that is not there.
        Err(hex::FromHexError::OddLength) => {
            return Err(anyhow::anyhow!(
                "--audit-salt must be 64 hex chars (32 bytes), got {}",
                h.len()
            ))
        }
        Err(e) => return Err(anyhow::anyhow!("--audit-salt is not hex: {}", e)),
    };
    bytes.try_into().map_err(|v: Vec<u8>| {
        anyhow::anyhow!(
            "--audit-salt must decode to exactly 32 bytes, got {}",
            v.len()
        )
    })
}

/// The warning text for a predictable audit salt on an audited lane, or `None`
/// when there is nothing to say -- either because the salt is one an algorithm
/// could not have known in advance, or because this challenge does not audit on
/// a salt-selected subsample at all.
///
/// `challenge_id` is not decoration. The `gpu` arm of `dispatch_challenge!` is
/// shared by c004, c005 and c006, and only c004's quality is audited on a
/// subsample; the other two take `_audit_salt` and ignore it (see
/// `hypergraph/mod.rs`), and no c005 or c006 run ever passes `--audit-salt`,
/// because for them it means nothing. Warning on all three would put a false
/// security notice on two of the three GPU lanes, and a warning that fires
/// where it does not apply is one operators learn to scroll past -- which costs
/// it precisely the attention the c004 case needs.
///
/// The all-zeros salt is what `parse_audit_salt` returns when `--audit-salt` is
/// omitted, and it is not a secret: `tig_challenges::audit_sampling::
/// sample_query_ids` is `pub` and is NOT hidden by the `hide_verification`
/// feature that tig-algorithms builds tig-challenges with. An algorithm can
/// therefore evaluate `sample_query_ids(&[0u8; 32], 7_000, 1_000)` at build
/// time and bake the answer in. Against a verification run that omitted the
/// flag, it needs to solve only the 1,000 audited queries -- 14% of the honest
/// work -- and can return anything at all for the other 6,000 while still
/// scoring recall 1.0.
///
/// The default stays (it is plan-mandated, contract C7, and the five CPU lanes
/// depend on it); only its silence goes. Returned as a value rather than
/// printed here so the wording is testable without a GPU.
///
/// `#[allow(dead_code)]`: a CPU-only build such as `--features c001` compiles
/// no GPU arm, so nothing calls this outside the tests.
#[allow(dead_code)]
fn predictable_salt_warning(challenge_id: &str, salt: &[u8; 32]) -> Option<String> {
    if challenge_id != "c004" {
        return None;
    }
    if salt.iter().any(|&b| b != 0) {
        return None;
    }
    Some(
        "WARNING: no --audit-salt was given, so this run audits with the \
         all-zeros salt. That salt is predictable: the audited query set is \
         publicly computable before the solve, so a solution can be built to \
         answer only those queries and still score full recall. Safe for local \
         debugging ONLY -- never for verification whose result is trusted. Pass \
         --audit-salt <64 hex chars> drawn after the solution was submitted."
            .to_string(),
    )
}

pub fn verify_solution(
    settings: String,
    rand_hash: String,
    nonce: u64,
    solution_path: String,
    ptx_path: Option<PathBuf>,
    gpu_device: Option<usize>,
    audit_salt_hex: Option<String>,
    verbose: bool,
) -> Result<()> {
    let settings = load_settings(&settings);
    let seeds = Seeds {
        nonce: settings.calc_seed(&rand_hash, nonce),
        db: settings.calc_db_seed(&rand_hash),
    };
    let seed = seeds.nonce;

    // Decoded once, up front, so a malformed salt is a clear error before any
    // GPU work rather than a surprise mid-verification. Only the GPU arm reads
    // it; a CPU-only build (e.g. --features c001) still decodes and validates,
    // so a bad salt is rejected on every lane.
    #[allow(unused_variables)]
    let audit_salt = parse_audit_salt(audit_salt_hex)?;

    let mut err_msg = Option::<String>::None;

    macro_rules! dispatch_challenge {
        ($c:ident, cpu) => {{
            let track_id = if settings.track_id.starts_with('"') && settings.track_id.ends_with('"')
            {
                settings.track_id.clone()
            } else {
                format!(r#""{}""#, settings.track_id)
            };
            let track = serde_json::from_str(&track_id).map_err(|_| {
                anyhow::anyhow!(
                    "Failed to parse track_id '{}' as {}::Track",
                    settings.track_id,
                    stringify!($c)
                )
            })?;
            let challenge = $c::Challenge::generate_instance(&seed, &track).unwrap();
            if verbose {
                println!("{:?}", challenge);
            }

            let solution = load_solution(&solution_path);
            match serde_json::from_str::<$c::Solution>(&solution) {
                Ok(solution) => {
                    if verbose {
                        println!("{:?}", solution);
                    }
                    match challenge.evaluate_solution(&solution) {
                        Ok(quality) => println!("quality: {}", quality),
                        Err(e) => err_msg = Some(format!("Invalid solution: {}", e)),
                    }
                }
                Err(_) => {
                    err_msg = Some(format!(
                        "Invalid solution. Cannot convert to {}::Solution",
                        stringify!($c)
                    ))
                }
            }
        }};

        ($c:ident, gpu) => {{
            let track_id = if settings.track_id.starts_with('"') && settings.track_id.ends_with('"')
            {
                settings.track_id.clone()
            } else {
                format!(r#""{}""#, settings.track_id)
            };
            let track = serde_json::from_str(&track_id).map_err(|_| {
                anyhow::anyhow!(
                    "Failed to parse track_id '{}' as {}::Track",
                    settings.track_id,
                    stringify!($c)
                )
            })?;

            // Not gated behind --verbose: an operator who did not ask for
            // verbose output is exactly the operator who needs to be told that
            // this run's audit set was knowable before the solve. stderr, so it
            // does not contaminate the `quality: N` line on stdout that callers
            // parse. And scoped by challenge, because this arm is shared with
            // c005 and c006, for which the salt is inert -- the helper does the
            // scoping so that "which lanes warn" is testable without a GPU.
            if let Some(w) = predictable_salt_warning(stringify!($c), &audit_salt) {
                eprintln!("{}", w);
            }

            if ptx_path.is_none() {
                panic!("PTX file is required for GPU challenges.");
            }

            let num_gpus = CudaContext::device_count()?;
            if num_gpus == 0 {
                panic!("No CUDA devices found");
            }
            let gpu_device = gpu_device.unwrap_or((nonce % num_gpus as u64) as usize);
            let ptx = Ptx::from_file(ptx_path.unwrap());
            let ctx = CudaContext::new(gpu_device).unwrap();
            ctx.set_blocking_synchronize()?;
            let module = ctx.load_module(ptx).unwrap();
            let stream = ctx.default_stream();
            let prop = get_device_prop(gpu_device as i32).unwrap();

            let challenge = $c::Challenge::generate_instance(
                &seeds,
                &track,
                module.clone(),
                stream.clone(),
                &prop,
            )
            .unwrap();

            let solution = load_solution(&solution_path);
            match serde_json::from_str::<$c::Solution>(&solution) {
                Ok(solution) => {
                    if verbose {
                        println!("{:?}", solution);
                    }
                    match challenge.evaluate_solution(
                        &solution,
                        &audit_salt,
                        module.clone(),
                        stream.clone(),
                        &prop,
                    ) {
                        Ok(quality) => {
                            stream.synchronize()?;
                            ctx.synchronize()?;
                            println!("quality: {}", quality);
                        }
                        Err(e) => err_msg = Some(format!("Invalid solution: {}", e)),
                    }
                }
                Err(_) => {
                    err_msg = Some(format!(
                        "Invalid solution. Cannot convert to {}::Solution",
                        stringify!($c)
                    ))
                }
            }
        }};
    }

    match settings.challenge_id.as_str() {
        "c001" => {
            #[cfg(not(feature = "c001"))]
            panic!("tig-verifier was not compiled with '--features c001'");
            #[cfg(feature = "c001")]
            dispatch_challenge!(c001, cpu)
        }
        "c002" => {
            #[cfg(not(feature = "c002"))]
            panic!("tig-verifier was not compiled with '--features c002'");
            #[cfg(feature = "c002")]
            dispatch_challenge!(c002, cpu)
        }
        "c003" => {
            #[cfg(not(feature = "c003"))]
            panic!("tig-verifier was not compiled with '--features c003'");
            #[cfg(feature = "c003")]
            dispatch_challenge!(c003, cpu)
        }
        "c004" => {
            #[cfg(not(feature = "c004"))]
            panic!("tig-verifier was not compiled with '--features c004'");
            #[cfg(feature = "c004")]
            dispatch_challenge!(c004, gpu)
        }
        "c005" => {
            #[cfg(not(feature = "c005"))]
            panic!("tig-verifier was not compiled with '--features c005'");
            #[cfg(feature = "c005")]
            dispatch_challenge!(c005, gpu)
        }
        "c006" => {
            #[cfg(not(feature = "c006"))]
            panic!("tig-verifier was not compiled with '--features c006'");
            #[cfg(feature = "c006")]
            dispatch_challenge!(c006, gpu)
        }
        "c007" => {
            #[cfg(not(feature = "c007"))]
            panic!("tig-verifier was not compiled with '--features c007'");
            #[cfg(feature = "c007")]
            dispatch_challenge!(c007, cpu)
        }
        "c008" => {
            #[cfg(not(feature = "c008"))]
            panic!("tig-verifier was not compiled with '--features c008'");
            #[cfg(feature = "c008")]
            dispatch_challenge!(c008, cpu)
        }
        _ => panic!("Unsupported challenge"),
    }

    if let Some(err_msg) = err_msg {
        eprintln!("Verification error: {}", err_msg);
        std::process::exit(1);
    }

    Ok(())
}

fn load_settings(settings: &str) -> BenchmarkSettings {
    let settings = if settings.ends_with(".json") {
        fs::read_to_string(settings).unwrap_or_else(|_| {
            eprintln!("Failed to read settings file: {}", settings);
            std::process::exit(1);
        })
    } else {
        settings.to_string()
    };

    dejsonify::<BenchmarkSettings>(&settings).unwrap_or_else(|_| {
        eprintln!("Failed to parse settings");
        std::process::exit(1);
    })
}

fn load_solution(solution: &str) -> String {
    if solution == "-" {
        let mut buffer = String::new();
        std::io::stdin()
            .read_to_string(&mut buffer)
            .unwrap_or_else(|_| {
                eprintln!("Failed to read solution from stdin");
                std::process::exit(1);
            });
        buffer
    } else if solution.ends_with(".json") {
        let d = fs::read_to_string(&solution).unwrap_or_else(|_| {
            eprintln!("Failed to read solution file: {}", solution);
            std::process::exit(1);
        });
        let d = serde_json::from_str::<Map<String, Value>>(&d).unwrap_or_else(|_| {
            eprintln!("Failed to parse solution file: {}", solution);
            std::process::exit(1);
        });
        match d.get("solution") {
            None => {
                eprintln!("json file does not contain 'solution' field: {}", solution);
                std::process::exit(1);
            }
            Some(v) => match v.as_str() {
                None => {
                    eprintln!("invalid 'solution' field in json file. Expecting string");
                    std::process::exit(1);
                }
                Some(s) => s.to_string(),
            },
        }
    } else {
        solution.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// argv[0] plus the four required positionals every invocation needs, so
    /// the tests below isolate the behaviour of `--audit-salt` and nothing else.
    const POSITIONALS: [&str; 5] = [
        "tig-verifier",
        "settings.json",
        "rand_hash",
        "0",
        "sol.json",
    ];

    fn parse(extra: &[&str]) -> Result<clap::ArgMatches, clap::Error> {
        let mut argv: Vec<&str> = POSITIONALS.to_vec();
        argv.extend_from_slice(extra);
        cli().try_get_matches_from(argv)
    }

    #[test]
    fn audit_salt_flag_without_a_value_is_rejected() {
        // The load-bearing assertion. With clap's optional-value form
        // (`[AUDIT_SALT]`) this parses fine and yields None, which the decode
        // then turns into the all-zeros salt -- so a shell-quoting slip or an
        // empty variable would silently audit a fully predictable sample, which
        // is the exact attack the salt exists to prevent. A security parameter
        // must fail closed on operator error, so the value is required
        // (`<AUDIT_SALT>`) and this must be an Err.
        let err = parse(&["--audit-salt"]).expect_err(
            "a valueless --audit-salt must be rejected, not silently defaulted to zeros",
        );
        assert_eq!(
            err.kind(),
            clap::error::ErrorKind::InvalidValue,
            "expected a missing-value parse error, got: {}",
            err
        );
    }

    #[test]
    fn audit_salt_with_a_value_parses() {
        // Pins that requiring the value did not break the supported form.
        let m = parse(&["--audit-salt", &"ab".repeat(32)]).unwrap();
        assert_eq!(
            m.get_one::<String>("audit-salt").map(|s| s.as_str()),
            Some("ab".repeat(32).as_str())
        );
    }

    #[test]
    fn absent_audit_salt_is_still_allowed() {
        // The flag itself stays optional: the five CPU lanes never pass one and
        // local debugging should not have to invent 64 hex characters. Only a
        // valueless flag is rejected.
        let m = parse(&[]).unwrap();
        assert!(m.get_one::<String>("audit-salt").is_none());
    }

    // --- parse_audit_salt: the decode path, including the security-relevant
    // all-zeros default. Previously exercised only by hand against the built
    // binary; those runs do not repeat, these do.

    #[test]
    fn absent_salt_decodes_to_all_zeros() {
        // Pinning the VALUE, not just Ok-ness: this default is what the five
        // CPU lanes and local debugging rely on, and it is also the predictable
        // salt, so a silent change here is a security change.
        assert_eq!(parse_audit_salt(None).unwrap(), [0u8; 32]);
    }

    #[test]
    fn a_real_salt_round_trips_to_its_bytes() {
        assert_eq!(
            parse_audit_salt(Some("ab".repeat(32))).unwrap(),
            [0xabu8; 32]
        );
    }

    #[test]
    fn a_non_hex_salt_reports_a_hex_error() {
        let err = parse_audit_salt(Some("zz".repeat(32)))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("is not hex"),
            "expected the not-hex message, got: {}",
            err
        );
    }

    #[test]
    fn a_short_salt_reports_a_length_error_naming_the_length() {
        // "aabb" is perfectly good hex -- 2 bytes of it. The operator needs to
        // be told it is the wrong SIZE, and told what size it was.
        let err = parse_audit_salt(Some("aabb".to_string()))
            .unwrap_err()
            .to_string();
        assert!(
            // `contains('2')` would NOT do here: the "32" in "32 bytes"
            // satisfies it whatever the reported length is.
            err.contains("32 bytes") && err.contains("got 2"),
            "expected a length message naming the observed length, got: {}",
            err
        );
        assert!(
            !err.contains("is not hex"),
            "a well-formed but short salt is a length problem, not a hex problem: {}",
            err
        );
    }

    // --- the predictable-salt warning.

    #[test]
    fn the_all_zeros_salt_is_warned_about() {
        // `sample_query_ids` is pub and ungated, and tig-algorithms links
        // tig-challenges with `hide_verification`, which does not hide it. So an
        // algorithm can compute sample_query_ids(&[0u8; 32], 7000, 1000) at
        // BUILD time and hardcode the result. A GPU lane verified without
        // --audit-salt hands over exactly that audit set, and the algorithm
        // needs to solve only 1,000 of 7,000 queries -- 14% of the honest work
        // -- to score whatever recall it likes.
        //
        // The all-zeros default itself is plan-mandated (contract C7) and stays.
        // What must not stay is its silence.
        let w = predictable_salt_warning("c004", &[0u8; 32])
            .expect("the all-zeros salt must produce a warning on the c004 lane");
        assert!(
            w.contains("--audit-salt"),
            "the warning must name the flag that fixes it, got: {}",
            w
        );
        assert!(
            w.to_lowercase().contains("predictable")
                || w.to_lowercase().contains("publicly computable"),
            "the warning must say WHY the zero salt is unsafe, got: {}",
            w
        );
    }

    #[test]
    fn a_real_salt_is_not_warned_about() {
        // The other half: a warning that fires on every salt is noise an
        // operator learns to ignore, which is the same as no warning at all.
        assert!(predictable_salt_warning("c004", &[0xabu8; 32]).is_none());
        // One non-zero byte is enough -- the seed is drawn from salt[0..8], but
        // "not the all-zeros default" is the property being reported, and a
        // salt that is zero in its first eight bytes but not elsewhere is not
        // the default and did not come from omitting the flag.
        let mut nearly_zero = [0u8; 32];
        nearly_zero[31] = 1;
        assert!(predictable_salt_warning("c004", &nearly_zero).is_none());
    }

    #[test]
    fn only_the_audited_lane_warns_about_the_zero_salt() {
        // The gpu arm of dispatch_challenge! is shared by c004, c005 and c006,
        // and only c004's quality is audited on a salt-selected subsample:
        // hypergraph/mod.rs takes `_audit_salt` and ignores it, and the
        // parameter exists only so the three challenges share one signature.
        // So a bare `if salt is zero` in that arm fires on every c005 and c006
        // verification ever run -- none of which pass --audit-salt, because for
        // them it means nothing -- and says something false about each.
        //
        // That is the same habituation failure `a_real_salt_is_not_warned_about`
        // guards, on the other axis. A warning that cries wolf on two of the
        // three GPU lanes is a warning operators learn to scroll past, which
        // costs it exactly the attention the c004 case needs.
        for lane in ["c005", "c006"] {
            assert!(
                predictable_salt_warning(lane, &[0u8; 32]).is_none(),
                "{} does not audit on a subsample, so the zero salt is not a \
                 security property of that lane and must not be reported as one",
                lane
            );
        }
        // And the same call that stays silent for those must still speak for
        // c004, or this test is satisfied by a function that never warns.
        assert!(predictable_salt_warning("c004", &[0u8; 32]).is_some());
    }

    #[test]
    fn an_odd_length_salt_is_a_length_error_not_a_hex_error() {
        // 63 chars: one character lost to a shell slip. Every character is a
        // valid hex digit, but hex::decode returns OddLength, and forwarding
        // that verbatim ("is not hex: Odd number of digits") sends the operator
        // hunting for a bad character that does not exist.
        let salt = "a".repeat(63);
        let err = parse_audit_salt(Some(salt)).unwrap_err().to_string();
        assert!(
            !err.contains("is not hex"),
            "an odd-length run of valid hex digits must not be reported as a hex \
             problem, got: {}",
            err
        );
        assert!(
            err.contains("64 hex chars") && err.contains("63"),
            "expected a length message naming the observed 63, got: {}",
            err
        );
    }
}
