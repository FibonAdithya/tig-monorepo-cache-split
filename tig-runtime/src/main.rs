use anyhow::{anyhow, Result};
use clap::{arg, Command};
use libloading::Library;
use serde_json::{Map, Value};
use std::{fs, panic, path::PathBuf};
use tig_challenges::*;
use tig_structs::core::{BenchmarkSettings, CPUArchitecture, OutputData};
use tig_utils::{dejsonify, jsonify};
#[cfg(feature = "cuda")]
use {
    cudarc::{
        driver::{CudaContext, CudaModule, CudaStream, LaunchConfig, PushKernelArg},
        nvrtc::Ptx,
        runtime::{result::device::get_device_prop, sys::cudaDeviceProp},
    },
    std::sync::Arc,
};

fn cli() -> Command {
    Command::new("tig-runtime")
        .about("Executes an algorithm on a single challenge instance")
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
            arg!(<BINARY> "Path to a shared object (*.so) file")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            arg!(--hyperparameters [HYPERPARAMETERS] "Hyperparameters json string or path to json file")
                .value_parser(clap::value_parser!(String)),
        )
        .arg(
            arg!(--ptx [PTX] "Path to a CUDA ptx file")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            arg!(--fuel [FUEL] "Optional maximum fuel parameter")
                .default_value("2000000000")
                .value_parser(clap::value_parser!(u64)),
        )
        .arg(
            arg!(--output [OUTPUT_FOLDER] "If set, the output data will be saved to this folder (default current directory)")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            arg!(--gpu [GPU] "Which GPU device to use")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            arg!(--index [PATH] "Index blob to load before solving")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        // The legacy four-positional form must keep parsing exactly as before,
        // so the subcommands negate the root's required positionals rather than
        // replacing them.
        .subcommand_negates_reqs(true)
        .args_conflicts_with_subcommands(true)
        .subcommand(
            Command::new("build-index")
                .about("Build an index over the precommit's database and exit. Takes no nonce.")
                .args(shared_args())
                // `.required(true)` is load-bearing: `arg!(--"build-fuel" <FUEL>)`
                // alone makes only the *value* mandatory, leaving the flag
                // optional and `get_one(..).unwrap()` a panic when it is omitted.
                .arg(
                    arg!(--"build-fuel" <FUEL> "Fuel budget for the build phase")
                        .required(true)
                        .value_parser(clap::value_parser!(u64)),
                )
                .arg(
                    arg!(--"index-out" <PATH> "Where to write the index blob")
                        .required(true)
                        .value_parser(clap::value_parser!(PathBuf)),
                )
                .arg(
                    arg!(--"memory-cap" [BYTES] "Device memory the build may use")
                        .value_parser(clap::value_parser!(u64))
                        .default_value("8589934592"),
                )
                .arg(
                    arg!(--"build-timeout" [SECS] "Wall-clock watchdog for the build")
                        .value_parser(clap::value_parser!(u64))
                        .default_value("600"),
                ),
        )
        .subcommand(
            Command::new("batch")
                .about("Solve a contiguous run of nonces in one process")
                .args(shared_args())
                .arg(
                    arg!(--"start-nonce" <N> "First nonce of a batch")
                        .required(true)
                        .value_parser(clap::value_parser!(u64)),
                )
                .arg(
                    arg!(--"num-nonces" <N> "How many nonces to solve")
                        .required(true)
                        .value_parser(clap::value_parser!(u64).range(1..)),
                )
                .arg(
                    arg!(--fuel [FUEL] "Optional maximum fuel parameter")
                        .default_value("2000000000")
                        .value_parser(clap::value_parser!(u64)),
                )
                .arg(
                    arg!(--index [PATH] "Index blob to load before solving")
                        .value_parser(clap::value_parser!(PathBuf)),
                )
                .arg(
                    arg!(--output [OUTPUT_FOLDER] "Folder for the per-nonce output files")
                        .value_parser(clap::value_parser!(PathBuf)),
                ),
        )
}

/// The arguments both new subcommands share with the legacy form. Deliberately
/// does NOT include a nonce: `build-index` must have no way to receive one.
fn shared_args() -> Vec<clap::Arg> {
    vec![
        arg!(<SETTINGS> "Settings json string or path to json file")
            .value_parser(clap::value_parser!(String)),
        arg!(<RAND_HASH> "A string used in seed generation")
            .value_parser(clap::value_parser!(String)),
        arg!(<BINARY> "Path to a shared object (*.so) file")
            .value_parser(clap::value_parser!(PathBuf)),
        arg!(--hyperparameters [HYPERPARAMETERS] "Hyperparameters json string or path to json file")
            .value_parser(clap::value_parser!(String)),
        arg!(--ptx [PTX] "Path to a CUDA ptx file").value_parser(clap::value_parser!(PathBuf)),
        arg!(--gpu [GPU] "Which GPU device to use").value_parser(clap::value_parser!(usize)),
    ]
}

fn main() {
    let matches = cli().get_matches();

    if let Some(sub) = matches.subcommand_matches("build-index") {
        // Same reasoning as the `batch` guard below: an argument that parses
        // and is then ignored is the silent-wrong-answer class. Both of these
        // have a `default_value`, so "supplied" has to mean *explicitly passed
        // on the command line* -- `get_one` cannot tell the difference and
        // would reject every invocation.
        for flag in ["memory-cap", "build-timeout"] {
            if sub.value_source(flag) == Some(clap::parser::ValueSource::CommandLine) {
                eprintln!(
                    "Runtime Error: --{} is parsed but not enforced yet (Task 5 wires it); \
                     refusing rather than silently ignoring it",
                    flag
                );
                std::process::exit(84);
            }
        }

        // Must exit non-zero, not fall through: a `--features c005` build has
        // `cuda` but no `c004`, and a bare `return` here would exit 0 having
        // built nothing.
        #[cfg(not(feature = "c004"))]
        {
            let _ = sub;
            eprintln!("Runtime Error: build-index requires a build with '--features c004'");
            std::process::exit(84);
        }
        #[cfg(feature = "c004")]
        {
            // The `.unwrap()`s below are safe only because `cli()` marks
            // --build-fuel and --index-out `.required(true)`.
            if let Err(e) = build_index(
                sub.get_one::<String>("SETTINGS").unwrap().clone(),
                sub.get_one::<String>("RAND_HASH").unwrap().clone(),
                sub.get_one::<PathBuf>("BINARY").unwrap().clone(),
                sub.get_one("hyperparameters").cloned(),
                sub.get_one::<PathBuf>("ptx").cloned().unwrap_or_else(|| {
                    eprintln!("Runtime Error: --ptx is required for build-index");
                    std::process::exit(84);
                }),
                *sub.get_one::<u64>("build-fuel").unwrap(),
                *sub.get_one::<u64>("memory-cap").unwrap(),
                *sub.get_one::<u64>("build-timeout").unwrap(),
                sub.get_one::<PathBuf>("index-out").unwrap().clone(),
                sub.get_one::<usize>("gpu").cloned(),
            ) {
                eprintln!("Runtime Error: {}", e);
                std::process::exit(84);
            }
            return;
        }
    }

    if matches.subcommand_matches("batch").is_some() {
        // The subcommand's arguments are already parsed and tested; the
        // batched execution path itself lands in a later task. Exiting
        // non-zero here is deliberate -- a silent exit 0 would be
        // indistinguishable from a batch that ran.
        eprintln!("Runtime Error: the 'batch' subcommand is not implemented yet");
        std::process::exit(84);
    }

    // `--index` is declared on the root command but nothing reads it yet, so
    // without this guard `tig-runtime S H 7 lib.so --index blob` would exit 0
    // having solved *without* the index and no caller could tell.
    if matches.value_source("index") == Some(clap::parser::ValueSource::CommandLine) {
        eprintln!(
            "Runtime Error: --index is parsed but not loaded yet (Task 6 wires it); \
             refusing rather than solving without the index"
        );
        std::process::exit(84);
    }

    if let Err(e) = compute_solution(
        matches.get_one::<String>("SETTINGS").unwrap().clone(),
        matches.get_one::<String>("RAND_HASH").unwrap().clone(),
        *matches.get_one::<u64>("NONCE").unwrap(),
        matches.get_one::<PathBuf>("BINARY").unwrap().clone(),
        matches.get_one("hyperparameters").cloned(),
        matches.get_one::<PathBuf>("ptx").cloned(),
        *matches.get_one::<u64>("fuel").unwrap(),
        matches.get_one::<PathBuf>("output").cloned(),
        matches.get_one::<usize>("gpu").cloned(),
    ) {
        eprintln!("Runtime Error: {}", e);
        std::process::exit(84);
    }
}

/// The `track_id` -> `Track` parse that every dispatch arm needs. `challenge`
/// is only used to name the challenge in the error message.
fn parse_track<T: serde::de::DeserializeOwned>(
    settings: &BenchmarkSettings,
    challenge: &str,
) -> Result<T> {
    let track_id = if settings.track_id.starts_with('"') && settings.track_id.ends_with('"') {
        settings.track_id.clone()
    } else {
        format!(r#""{}""#, settings.track_id)
    };
    serde_json::from_str(&track_id).map_err(|_| {
        anyhow!(
            "Failed to parse track_id '{}' as {}::Track",
            settings.track_id,
            challenge
        )
    })
}

fn seeds_for(settings: &BenchmarkSettings, rand_hash: &String, nonce: u64) -> Seeds {
    Seeds {
        nonce: settings.calc_seed(rand_hash, nonce),
        db: settings.calc_db_seed(rand_hash),
    }
}

pub fn compute_solution(
    settings: String,
    rand_hash: String,
    nonce: u64,
    library_path: PathBuf,
    hyperparameters: Option<String>,
    ptx_path: Option<PathBuf>,
    max_fuel: u64,
    output_folder: Option<PathBuf>,
    gpu_device: Option<usize>,
) -> Result<()> {
    let settings = load_settings(&settings);
    let seeds = seeds_for(&settings, &rand_hash, nonce);
    let seed = seeds.nonce;

    let hyperparameters = hyperparameters.map(|x| load_hyperparameters(&x));

    let library = load_module(&library_path)?;
    let fuel_remaining_ptr = unsafe { *library.get::<*mut u64>(b"__fuel_remaining")? };
    unsafe { *fuel_remaining_ptr = max_fuel };
    let runtime_signature_ptr = unsafe { *library.get::<*mut u64>(b"__runtime_signature")? };
    unsafe { *runtime_signature_ptr = u64::from_be_bytes(seed[0..8].try_into().unwrap()) };

    let output_file = match output_folder {
        Some(folder) => {
            fs::create_dir_all(&folder)?;
            folder.join(format!("{}.json", nonce))
        }
        None => format!("{}.json", nonce).into(),
    };

    macro_rules! dispatch_challenge {
        ($c:ident, cpu) => {{
            let track: $c::Track = parse_track(&settings, stringify!($c))?;

            // library function may exit 87 if it runs out of fuel
            let solve_challenge_fn = unsafe {
                library.get::<fn(
                    &$c::Challenge,
                    &dyn Fn(&$c::Solution) -> Result<()>,
                    Option<String>,
                ) -> Result<()>>(b"entry_point")?
            };

            let challenge = $c::Challenge::generate_instance(&seed, &track)?;

            let save_solution_fn = |solution: &$c::Solution| -> Result<()> {
                let fuel_consumed = (max_fuel
                    - unsafe { **library.get::<*const u64>(b"__fuel_remaining")? })
                .min(max_fuel + 1);
                let runtime_signature =
                    unsafe { **library.get::<*const u64>(b"__runtime_signature")? };

                let solution = serde_json::to_string(&solution)?;

                let output_data = OutputData {
                    nonce,
                    runtime_signature,
                    fuel_consumed,
                    solution,
                    #[cfg(target_arch = "x86_64")]
                    cpu_arch: CPUArchitecture::AMD64,
                    #[cfg(target_arch = "aarch64")]
                    cpu_arch: CPUArchitecture::ARM64,
                };
                fs::write(&output_file, jsonify(&output_data))?;
                Ok(())
            };
            let result = solve_challenge_fn(&challenge, &save_solution_fn, hyperparameters);
            if !output_file.exists() {
                save_solution_fn(&$c::Solution::new())?;
            }
            result
        }};

        ($c:ident, gpu) => {{
            let track: $c::Track = parse_track(&settings, stringify!($c))?;

            if ptx_path.is_none() {
                panic!("PTX file is required for GPU challenges.");
            }
            let ptx_path = ptx_path.unwrap();
            // library function may exit 87 if it runs out of fuel
            let solve_challenge_fn = unsafe {
                library.get::<fn(
                    &$c::Challenge,
                    save_solution: &dyn Fn(&$c::Solution) -> anyhow::Result<()>,
                    Option<String>,
                    Arc<CudaModule>,
                    Arc<CudaStream>,
                    &cudaDeviceProp,
                ) -> Result<()>>(b"entry_point")?
            };

            let gpu_fuel_scale = 20; // scale fuel to loosely align with CPU
            let ptx_content = std::fs::read_to_string(&ptx_path)
                .map_err(|e| anyhow!("Failed to read PTX file: {}", e))?;
            let max_fuel_hex = format!("0x{:016x}", max_fuel * gpu_fuel_scale);
            let modified_ptx = ptx_content.replace("0xdeadbeefdeadbeef", &max_fuel_hex);

            let num_gpus = CudaContext::device_count()?;
            if num_gpus == 0 {
                panic!("No CUDA devices found");
            }
            let gpu_device = gpu_device.unwrap_or((nonce % num_gpus as u64) as usize);
            let ptx = Ptx::from_src(modified_ptx);
            let ctx = CudaContext::new(gpu_device)?;
            ctx.set_blocking_synchronize()?;
            let module = ctx.load_module(ptx)?;
            let stream = ctx.fuel_check_stream();
            let prop = get_device_prop(gpu_device as i32)?;

            let challenge = $c::Challenge::generate_instance(
                &seeds,
                &track,
                module.clone(),
                stream.clone(),
                &prop,
            )?;

            let initialize_kernel = module.load_function("initialize_kernel")?;

            let cfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                stream
                    .launch_builder(&initialize_kernel)
                    .arg(&(u64::from_be_bytes(seed[8..16].try_into().unwrap())))
                    .launch(cfg)?;
            }

            let save_solution_fn = |solution: &$c::Solution| -> Result<()> {
                stream.synchronize()?;
                ctx.synchronize()?;

                let mut fuel_usage = stream.alloc_zeros::<u64>(1)?;
                let mut signature = stream.alloc_zeros::<u64>(1)?;
                let mut error_stat = stream.alloc_zeros::<u64>(1)?;

                let finalize_kernel = module.load_function("finalize_kernel")?;

                let cfg = LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                };

                unsafe {
                    stream
                        .launch_builder(&finalize_kernel)
                        .arg(&mut fuel_usage)
                        .arg(&mut signature)
                        .arg(&mut error_stat)
                        .launch(cfg)?;
                }

                let gpu_fuel_consumed = stream.memcpy_dtov(&fuel_usage)?[0] / gpu_fuel_scale;
                let cpu_fuel_consumed =
                    max_fuel - unsafe { **library.get::<*const u64>(b"__fuel_remaining")? };
                let fuel_consumed = (gpu_fuel_consumed + cpu_fuel_consumed).min(max_fuel + 1);

                let gpu_runtime_signature = stream.memcpy_dtov(&signature)?[0];
                let cpu_runtime_signature =
                    unsafe { **library.get::<*const u64>(b"__runtime_signature")? };
                let runtime_signature = gpu_runtime_signature ^ cpu_runtime_signature;

                let solution = serde_json::to_string(&solution)?;

                let output_data = OutputData {
                    nonce,
                    runtime_signature,
                    fuel_consumed,
                    solution,
                    #[cfg(target_arch = "x86_64")]
                    cpu_arch: CPUArchitecture::AMD64,
                    #[cfg(target_arch = "aarch64")]
                    cpu_arch: CPUArchitecture::ARM64,
                };
                fs::write(&output_file, jsonify(&output_data))?;
                Ok(())
            };
            let result = solve_challenge_fn(
                &challenge,
                &save_solution_fn,
                hyperparameters,
                module.clone(),
                stream.clone(),
                &prop,
            );
            if !output_file.exists() {
                save_solution_fn(&$c::Solution::new())?;
            }
            result
        }};
    }

    match settings.challenge_id.as_str() {
        "c001" => {
            #[cfg(not(feature = "c001"))]
            panic!("tig-runtime was not compiled with '--features c001'");
            #[cfg(feature = "c001")]
            dispatch_challenge!(c001, cpu)
        }
        "c002" => {
            #[cfg(not(feature = "c002"))]
            panic!("tig-runtime was not compiled with '--features c002'");
            #[cfg(feature = "c002")]
            dispatch_challenge!(c002, cpu)
        }
        "c003" => {
            #[cfg(not(feature = "c003"))]
            panic!("tig-runtime was not compiled with '--features c003'");
            #[cfg(feature = "c003")]
            dispatch_challenge!(c003, cpu)
        }
        "c004" => {
            #[cfg(not(feature = "c004"))]
            panic!("tig-runtime was not compiled with '--features c004'");
            #[cfg(feature = "c004")]
            dispatch_challenge!(c004, gpu)
        }
        "c005" => {
            #[cfg(not(feature = "c005"))]
            panic!("tig-runtime was not compiled with '--features c005'");
            #[cfg(feature = "c005")]
            dispatch_challenge!(c005, gpu)
        }
        "c006" => {
            #[cfg(not(feature = "c006"))]
            panic!("tig-runtime was not compiled with '--features c006'");
            #[cfg(feature = "c006")]
            dispatch_challenge!(c006, gpu)
        }
        "c007" => {
            #[cfg(not(feature = "c007"))]
            panic!("tig-runtime was not compiled with '--features c007'");
            #[cfg(feature = "c007")]
            dispatch_challenge!(c007, cpu)
        }
        "c008" => {
            #[cfg(not(feature = "c008"))]
            panic!("tig-runtime was not compiled with '--features c008'");
            #[cfg(feature = "c008")]
            dispatch_challenge!(c008, cpu)
        }
        _ => panic!("Unsupported challenge"),
    }
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

fn load_hyperparameters(hyperparameters: &str) -> String {
    let hyperparameters = if hyperparameters.ends_with(".json") {
        fs::read_to_string(hyperparameters).unwrap_or_else(|_| {
            eprintln!("Failed to read hyperparameters file: {}", hyperparameters);
            std::process::exit(1);
        })
    } else {
        hyperparameters.to_string()
    };

    // validate it is valid JSON
    let _ = dejsonify::<Map<String, Value>>(&hyperparameters).unwrap_or_else(|_| {
        eprintln!("Failed to parse hyperparameters as JSON");
        std::process::exit(1);
    });
    hyperparameters
}

pub fn load_module(path: &PathBuf) -> Result<Library> {
    let res = panic::catch_unwind(|| unsafe { Library::new(path) });

    match res {
        Ok(lib_result) => lib_result.map_err(|e| anyhow!(e.to_string())),
        Err(_) => Err(anyhow!("Failed to load module")),
    }
}

/// The path `build_index` writes to before renaming into place. Appends
/// `.tmp` rather than replacing the extension: `with_extension("tmp")` maps
/// both `idx.blob` and `idx.bin` onto the same `idx.tmp`, so two concurrent
/// builds writing different outputs in one directory would collide. Staying in
/// the same directory keeps the rename atomic and same-filesystem.
#[cfg(any(feature = "c004", test))]
fn index_tmp_path(index_out: &std::path::Path) -> PathBuf {
    let mut tmp = index_out.to_path_buf().into_os_string();
    tmp.push(".tmp");
    PathBuf::from(tmp)
}

#[cfg(feature = "c004")]
pub fn build_index(
    settings: String,
    rand_hash: String,
    library_path: PathBuf,
    hyperparameters: Option<String>,
    ptx_path: PathBuf,
    build_fuel: u64,
    _memory_cap_bytes: u64,
    _timeout_secs: u64,
    index_out: PathBuf,
    gpu_device: Option<usize>,
) -> Result<()> {
    let settings = load_settings(&settings);
    // No nonce is derived here and none can be passed -- the `build-index`
    // subcommand has no nonce argument at all. This is the property the design
    // rests on: a build process cannot compute any query set because the
    // material to derive one is not present.
    let db_seed = settings.calc_db_seed(&rand_hash);
    let hyperparameters = hyperparameters.map(|x| load_hyperparameters(&x));

    let library = load_module(&library_path)?;
    let fuel_remaining_ptr = unsafe { *library.get::<*mut u64>(b"__fuel_remaining")? };
    unsafe { *fuel_remaining_ptr = build_fuel };

    let gpu_fuel_scale = 20u64;
    let ptx_content = std::fs::read_to_string(&ptx_path)
        .map_err(|e| anyhow!("Failed to read PTX file: {}", e))?;
    let scaled = build_fuel.checked_mul(gpu_fuel_scale).ok_or_else(|| {
        anyhow!(
            "build fuel {} overflows when scaled by {}",
            build_fuel,
            gpu_fuel_scale
        )
    })?;
    let modified_ptx = ptx_content.replace("0xdeadbeefdeadbeef", &format!("0x{:016x}", scaled));

    let gpu_device = gpu_device.unwrap_or(0);
    let ctx = CudaContext::new(gpu_device)?;
    ctx.set_blocking_synchronize()?;
    let module = ctx.load_module(Ptx::from_src(modified_ptx))?;
    let stream = ctx.fuel_check_stream();
    let prop = get_device_prop(gpu_device as i32)?;

    let build_index_fn = unsafe {
        library.get::<fn(
            &c004::Database,
            Option<String>,
            Arc<CudaModule>,
            Arc<CudaStream>,
            &cudaDeviceProp,
        ) -> Result<Vec<u8>>>(b"build_index")
    }
    .map_err(|_| {
        anyhow!("algorithm does not export `build_index`; it does not support index building")
    })?;

    // Checked before the build, not after: discovering the mismatch after
    // paying for a ten-minute build is a waste with no upside.
    unsafe {
        library.get::<fn(
            &c004::Database,
            &[u8],
            Arc<CudaModule>,
            Arc<CudaStream>,
            &cudaDeviceProp,
        ) -> Result<()>>(b"load_index")
    }
    .map_err(|_| anyhow!("algorithm exports `build_index` but not `load_index`"))?;

    let track: c004::Track = parse_track(&settings, "c004")?;
    let database =
        c004::Database::generate(&db_seed, &track, module.clone(), stream.clone(), &prop)?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    let initialize_kernel = module.load_function("initialize_kernel")?;
    unsafe {
        stream
            .launch_builder(&initialize_kernel)
            .arg(&u64::from_be_bytes(db_seed[8..16].try_into().unwrap()))
            .launch(cfg)?;
    }

    let blob = build_index_fn(&database, hyperparameters, module.clone(), stream.clone(), &prop)?;

    // A GPU fuel trap is asynchronous: `build_index_fn` can return Ok while the
    // device has already trapped and set gbl_ERRORSTAT. Without this, a build
    // that blew its fuel budget writes an index and exits 0, and the spec's
    // "build-fuel exhaustion leaves no index file" is simply false.
    stream.synchronize()?;
    ctx.synchronize()?;
    let mut fuel_usage = stream.alloc_zeros::<u64>(1)?;
    let mut signature = stream.alloc_zeros::<u64>(1)?;
    let mut error_stat = stream.alloc_zeros::<u64>(1)?;
    let finalize_kernel = module.load_function("finalize_kernel")?;
    unsafe {
        stream
            .launch_builder(&finalize_kernel)
            .arg(&mut fuel_usage)
            .arg(&mut signature)
            .arg(&mut error_stat)
            .launch(cfg)?;
    }
    let error_stat = stream.memcpy_dtov(&error_stat)?[0];
    let gpu_fuel_used = stream.memcpy_dtov(&fuel_usage)?[0] / gpu_fuel_scale;
    if error_stat != 0 {
        return Err(anyhow!(
            "build failed on the device (error_stat {}, gpu fuel used {} of {}); no index written",
            error_stat,
            gpu_fuel_used,
            build_fuel
        ));
    }

    // Atomic: a watchdog kill must never leave a partial blob for the query
    // process to load.
    let tmp = index_tmp_path(&index_out);
    fs::write(&tmp, &blob)?;
    fs::rename(&tmp, &index_out)?;
    eprintln!(
        "index written: {} bytes, gpu fuel used {} of {}",
        blob.len(),
        gpu_fuel_used,
        build_fuel
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_settings() -> BenchmarkSettings {
        BenchmarkSettings {
            player_id: "some_player".to_string(),
            block_id: "some_block".to_string(),
            challenge_id: "some_challenge".to_string(),
            algorithm_id: "some_algorithm".to_string(),
            track_id: "a=1,b=2".to_string(),
        }
    }

    #[test]
    fn seeds_for_puts_each_derivation_in_the_right_field() {
        // Catches the swap. If `nonce` and `db` are exchanged, the database
        // becomes per-nonce again and every claim the index-build design makes
        // is false -- with nothing failing and no message naming the cause.
        let settings = test_settings();
        let rand_hash = "random_hash".to_string();

        let seeds = seeds_for(&settings, &rand_hash, 1337);
        assert_eq!(seeds.nonce, settings.calc_seed(&rand_hash, 1337));
        assert_eq!(seeds.db, settings.calc_db_seed(&rand_hash));
    }

    #[test]
    fn seeds_for_holds_the_database_constant_across_nonces() {
        let settings = test_settings();
        let rand_hash = "random_hash".to_string();

        let a = seeds_for(&settings, &rand_hash, 0);
        let b = seeds_for(&settings, &rand_hash, u64::MAX);
        assert_eq!(a.db, b.db, "database seed must not vary with the nonce");
        assert_ne!(a.nonce, b.nonce, "query seed must vary with the nonce");
    }

    #[test]
    fn build_index_mode_has_no_nonce_argument() {
        // The anti-gaming property of D3 is that the build process has no nonce
        // in scope. `build-index` must reject a nonce in every form it could
        // arrive in: as a positional, or as a batch flag.
        for extra in [
            vec!["7"],                  // a positional nonce
            vec!["--start-nonce", "0"], // the batch flag
            vec!["--num-nonces", "5"],
        ] {
            let mut args = vec![
                "tig-runtime",
                "build-index",
                "{}",
                "hash",
                "lib.so",
                "--build-fuel",
                "1000",
                "--index-out",
                "/tmp/i",
            ];
            args.extend(extra.iter().copied());
            let err = cli().try_get_matches_from(&args).unwrap_err();
            assert!(
                err.to_string().contains("unexpected argument"),
                "build-index must reject {:?}, got: {}",
                extra,
                err
            );
        }
    }

    #[test]
    fn build_index_requires_its_outputs() {
        // Without `.required(true)` these parse fine and the mode then panics
        // on `.unwrap()` deep inside build_index, after the process has already
        // opened a CUDA context.
        let base = [
            "tig-runtime",
            "build-index",
            "{}",
            "hash",
            "lib.so",
            "--build-fuel",
            "1000",
            "--index-out",
            "/tmp/i",
        ];
        for missing in ["--build-fuel", "--index-out"] {
            let mut kept = Vec::new();
            let mut skip = false;
            for a in base {
                if skip {
                    skip = false;
                    continue;
                } // drop the flag's value too
                if a == missing {
                    skip = true;
                    continue;
                }
                kept.push(a);
            }
            let err = cli().try_get_matches_from(&kept).unwrap_err();
            assert!(
                err.to_string().contains(missing),
                "missing {} must be an error naming it, got: {}",
                missing,
                err
            );
        }
    }

    #[test]
    fn batched_mode_refuses_zero_nonces() {
        // A batch that produces nothing and exits 0 is indistinguishable from a
        // batch that worked.
        let err = cli()
            .try_get_matches_from(vec![
                "tig-runtime",
                "batch",
                "{}",
                "hash",
                "lib.so",
                "--start-nonce",
                "0",
                "--num-nonces",
                "0",
            ])
            .unwrap_err();
        // `contains("num-nonces")` alone is satisfied by clap's "unexpected
        // argument '--num-nonces'", so deleting the argument outright would
        // leave this test green. Pin the error *kind* to the range check, and
        // pin the positive case too: an argument that rejects everything
        // would also satisfy an assertion about rejection.
        assert_eq!(
            err.kind(),
            clap::error::ErrorKind::ValueValidation,
            "0 must be rejected by the range check (ValueValidation), not by the \
             argument being absent (UnknownArgument); got: {}",
            err
        );
        assert!(err.to_string().contains("num-nonces"), "got: {}", err);

        let m = cli()
            .try_get_matches_from(vec![
                "tig-runtime",
                "batch",
                "{}",
                "hash",
                "lib.so",
                "--start-nonce",
                "0",
                "--num-nonces",
                "1",
            ])
            .expect("a batch of one nonce must parse");
        assert_eq!(*m.subcommand_matches("batch").unwrap().get_one::<u64>("num-nonces").unwrap(), 1);
    }

    #[test]
    fn the_legacy_single_nonce_form_still_parses() {
        // Every existing caller -- tig-verifier's sibling CLI, the slave, and
        // scripts/test_algorithm -- uses this form. Adding subcommands must not
        // move it.
        let m = cli()
            .try_get_matches_from(vec![
                "tig-runtime",
                "{}",
                "hash",
                "7",
                "lib.so",
                "--ptx",
                "p.ptx",
            ])
            .unwrap();
        assert_eq!(m.subcommand_name(), None);
        assert_eq!(*m.get_one::<u64>("NONCE").unwrap(), 7);
    }

    #[test]
    fn build_index_never_derives_a_nonce_seed() {
        // A source-level assertion, deliberately, and it is the right tool for
        // this one property specifically:
        //
        //   * The claim is structural -- "a build process cannot obtain a
        //     nonce" -- not behavioural. Nothing observable at runtime
        //     distinguishes a build that could have derived a query seed from
        //     one that could not.
        //   * `build_index` is `#[cfg(feature = "c004")]` and is therefore not
        //     even compiled in the `--features c001` lane these tests run in,
        //     so no ordinary test can call it or link against it.
        //   * Adding `settings.calc_seed(&rand_hash, 0)` to `build_index`
        //     today leaves every other test in this file green. That single
        //     mutation silently destroys the anti-gaming guarantee the whole
        //     index-build design rests on, and this is the only thing that
        //     catches it.
        //
        // The slicing below fails loudly if it cannot locate the function --
        // a source scan that silently matches nothing is worse than no test.
        const SRC: &str = include_str!("main.rs");

        // The needle is assembled with `concat!` on purpose: written as one
        // literal it would appear verbatim in this test's own source, `find`
        // would match *here* instead of at the definition, and the scan would
        // silently examine the wrong function. Split, the literal never occurs
        // in the file as a contiguous string except at the real definition.
        const NEEDLE: &str = concat!("pub fn ", "build_index(");
        assert_eq!(
            SRC.matches(NEEDLE).count(),
            1,
            "expected exactly one `{}` in main.rs; the scan cannot tell which \
             one to bound",
            NEEDLE
        );
        let start = SRC.find(NEEDLE).expect(
            "could not find the definition of build_index in main.rs -- it was \
             renamed or removed; fix this test rather than letting it scan the \
             wrong slice",
        );
        let rest = &SRC[start..];
        // Inside the function every closing brace is indented; the first
        // brace alone on a line at column 0 is the function's own.
        let end = rest.find("\n}\n").expect(
            "could not find the closing brace of `build_index` -- the source \
             layout changed and this test can no longer bound the function",
        );
        let body = &rest[..end];

        // Guard against a slice that is technically non-empty but does not
        // actually contain the function.
        assert!(
            body.len() > 500 && body.contains("index_out"),
            "sliced {} bytes that do not look like build_index's body",
            body.len()
        );

        assert!(
            body.contains("calc_db_seed("),
            "build_index must derive the nonce-free database seed"
        );
        assert!(
            !body.contains("calc_seed("),
            "build_index must never derive a per-nonce seed: the build process \
             having no way to compute a query set is the anti-gaming property \
             the whole design rests on"
        );
    }

    #[test]
    fn the_index_temp_file_appends_rather_than_replacing_the_extension() {
        // `with_extension("tmp")` collapses idx.blob and idx.bin onto one
        // idx.tmp, so two builds in one directory would race on the same
        // temp path while renaming to different outputs.
        let a = index_tmp_path(std::path::Path::new("/x/idx.blob"));
        let b = index_tmp_path(std::path::Path::new("/x/idx.bin"));
        assert_eq!(a, PathBuf::from("/x/idx.blob.tmp"));
        assert_ne!(a, b, "distinct outputs must not share a temp path");
        // Same directory, or the rename is neither atomic nor guaranteed
        // same-filesystem.
        assert_eq!(a.parent(), std::path::Path::new("/x/idx.blob").parent());
    }
}
