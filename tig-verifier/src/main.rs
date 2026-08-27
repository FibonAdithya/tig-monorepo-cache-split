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
    let seed = settings.calc_seed(&rand_hash, nonce);

    // Decoded once, up front, so a malformed salt is a clear error before any
    // GPU work rather than a surprise mid-verification. Defaults to all-zeros
    // when absent: the five CPU lanes never pass one, and local debugging of a
    // GPU lane should not have to invent 64 hex characters.
    // Only the GPU arm reads it; a CPU-only build (e.g. --features c001)
    // still decodes and validates, so a bad salt is rejected on every lane.
    #[allow(unused_variables)]
    let audit_salt: [u8; 32] = match audit_salt_hex {
        Some(h) => hex::decode(&h)
            .map_err(|e| anyhow::anyhow!("--audit-salt is not hex: {}", e))?
            .try_into()
            .map_err(|v: Vec<u8>| {
                anyhow::anyhow!(
                    "--audit-salt must decode to exactly 32 bytes, got {}",
                    v.len()
                )
            })?,
        None => [0u8; 32],
    };

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
                &seed,
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
    const POSITIONALS: [&str; 5] = ["tig-verifier", "settings.json", "rand_hash", "0", "sol.json"];

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
}
