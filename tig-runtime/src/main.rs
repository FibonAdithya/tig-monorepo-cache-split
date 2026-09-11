use anyhow::{anyhow, Result};
use clap::{arg, Command};
use libloading::Library;
use serde_json::{Map, Value};
use std::{
    fs, panic,
    path::{Path, PathBuf},
};
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
            arg!(--"challenge-cache" [PATH] "c004: the challenge's shared input (the database), written if missing and read if present. Deterministic from the seed, so the verifier may read it too")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            arg!(--"algorithm-cache" [PATH] "c004: the algorithm's build_cache output, written if missing and read if present. Opaque; never given to the verifier")
                .value_parser(clap::value_parser!(PathBuf)),
        )
}

fn main() {
    let matches = cli().get_matches();

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
        matches.get_one::<PathBuf>("challenge-cache").cloned(),
        matches.get_one::<PathBuf>("algorithm-cache").cloned(),
    ) {
        eprintln!("Runtime Error: {}", e);
        std::process::exit(84);
    }
}

fn seeds_for(settings: &BenchmarkSettings, rand_hash: &String, nonce: u64) -> Seeds {
    Seeds {
        db: settings.calc_db_seed(rand_hash),
        build: settings.calc_build_seed(rand_hash),
        instance: settings.calc_seed(rand_hash, nonce),
        algo: settings.calc_algo_seed(rand_hash, nonce),
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
    challenge_cache: Option<PathBuf>,
    algorithm_cache: Option<PathBuf>,
) -> Result<()> {
    let settings = load_settings(&settings);

    let seeds = seeds_for(&settings, &rand_hash, nonce);
    // The runtime signature and the device signature are seeded from the
    // algorithm's own seed, which it can read anyway; never from a hidden one.
    let seed = seeds.algo;

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

    /// The cache flags only mean something for a challenge with a shared
    /// input (a `Database`) and the export pair; today that is c004. Refusing
    /// loudly beats solving without the cache the caller asked for.
    macro_rules! refuse_cache {
        ($c:ident) => {
            if challenge_cache.is_some() || algorithm_cache.is_some() {
                return Err(anyhow!(
                    "--challenge-cache and --algorithm-cache are not supported by {}: it has no shared input to cache",
                    stringify!($c)
                ));
            }
        };
    }

    macro_rules! dispatch_challenge {
        // CPU challenges without a shared input: the whole instance from the
        // instance seed, exactly as before.
        ($c:ident, cpu) => {{
            refuse_cache!($c);
            dispatch_challenge!($c, cpu_with, |track: &$c::Track| -> Result<$c::Challenge> {
                $c::Challenge::generate_instance(&seeds.instance, track)
            })
        }};
        // A CPU challenge with a shared input: the same two-stage harness as
        // `gpu_cached`, minus the device. Selected by a challenge that defines
        // `Database::{generate, to_bytes, from_bytes}` and
        // `Challenge::for_nonce`; none does today, so the only expansion is the
        // test-only fake below the match, which is what keeps this arm
        // compiling.
        ($c:ident, cpu_cached) => {{
            dispatch_challenge!($c, cpu_with, |track: &$c::Track| -> Result<$c::Challenge> {
                let build_fn = unsafe {
                    library.get::<fn(&$c::Database, &[u8; 32], Option<String>) -> Result<Vec<u8>>>(
                        b"build_cache",
                    )
                }
                .ok();
                let load_fn = unsafe {
                    library.get::<fn(&$c::Database, &[u8]) -> Result<()>>(b"load_cache")
                }
                .ok();
                let generate = || $c::Database::generate(&seeds.db, track);
                let to_bytes = |db: &$c::Database| db.to_bytes(&seeds.db);
                let from_bytes = |bytes: &[u8]| $c::Database::from_bytes(bytes, &seeds.db, track);
                // `map_err` before `?`: the algorithm's error owns a vtable
                // inside the .so; render it while the library is loaded.
                let build = |db: &$c::Database| -> Result<Vec<u8>> {
                    let f = **build_fn.as_ref().unwrap();
                    f(db, &seeds.build, hyperparameters.clone())
                        .map_err(|e| anyhow!("build_cache: {:#}", e))
                };
                let load = |db: &$c::Database, bytes: &[u8]| -> Result<()> {
                    let f = **load_fn.as_ref().unwrap();
                    f(db, bytes).map_err(|e| anyhow!("load_cache: {:#}", e))
                };
                let db = run_cache_stage(CacheStage {
                    challenge_cache: challenge_cache.as_deref(),
                    algorithm_cache: algorithm_cache.as_deref(),
                    generate: &generate,
                    to_bytes: &to_bytes,
                    from_bytes: &from_bytes,
                    build: build_fn.as_ref().map(|_| &build as &dyn Fn(&$c::Database) -> Result<Vec<u8>>),
                    load: load_fn.as_ref().map(|_| &load as &dyn Fn(&$c::Database, &[u8]) -> Result<()>),
                })?;
                // Neither the build nor the load is this nonce's work.
                unsafe { *fuel_remaining_ptr = max_fuel };
                $c::Challenge::for_nonce(&db, &seeds, track)
            })
        }};
        ($c:ident, cpu_with, $make_challenge:expr) => {{
            let track_id = if settings.track_id.starts_with('"') && settings.track_id.ends_with('"')
            {
                settings.track_id.clone()
            } else {
                format!(r#""{}""#, settings.track_id)
            };
            let track: $c::Track = serde_json::from_str(&track_id).map_err(|_| {
                anyhow::anyhow!(
                    "Failed to parse track_id '{}' as {}::Track",
                    settings.track_id,
                    stringify!($c)
                )
            })?;

            // library function may exit 87 if it runs out of fuel
            let solve_challenge_fn = unsafe {
                library.get::<fn(
                    &$c::Challenge,
                    &dyn Fn(&$c::Solution) -> Result<()>,
                    Option<String>,
                ) -> Result<()>>(b"entry_point")?
            };

            let challenge = $make_challenge(&track)?;

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

        // c005 / c006: no Database, no cache. The whole instance comes from the
        // nonce seed, exactly as before.
        ($c:ident, gpu) => {{
            refuse_cache!($c);
            dispatch_challenge!(
                $c,
                gpu_with,
                |track: &$c::Track,
                 module: Arc<CudaModule>,
                 stream: Arc<CudaStream>,
                 prop: &cudaDeviceProp|
                 -> Result<$c::Challenge> {
                    $c::Challenge::generate_instance(&seeds, track, module, stream, prop)
                }
            )
        }};
        // c004: the database is nonce-free. Generate it, then, if a cache path
        // was given, build the algorithm's index into it on a miss or load it
        // on a hit, and hand the algorithm the loaded index before the nonce's
        // queries exist. Both the build and the load run before
        // `initialize_kernel`, so they sit outside the nonce's device meter, and
        // the CPU meter is re-primed after them for the same reason.
        // c004: the database is nonce-free. Stage 1 reads or generates it,
        // stage 2 reads or builds the algorithm's cache, then the nonce's
        // queries are made from it. Both stages run before `initialize_kernel`,
        // so they sit outside the nonce's device meter, and the CPU meter is
        // re-armed after them for the same reason.
        ($c:ident, gpu_cached) => {{
            dispatch_challenge!(
                $c,
                gpu_with,
                |track: &$c::Track,
                 module: Arc<CudaModule>,
                 stream: Arc<CudaStream>,
                 prop: &cudaDeviceProp|
                 -> Result<$c::Challenge> {
                    let build_fn = unsafe {
                        library.get::<fn(
                            &$c::Database,
                            &[u8; 32],
                            Option<String>,
                            Arc<CudaModule>,
                            Arc<CudaStream>,
                            &cudaDeviceProp,
                        ) -> Result<Vec<u8>>>(b"build_cache")
                    }
                    .ok();
                    let load_fn = unsafe {
                        library.get::<fn(
                            &$c::Database,
                            &[u8],
                            Arc<CudaModule>,
                            Arc<CudaStream>,
                            &cudaDeviceProp,
                        ) -> Result<()>>(b"load_cache")
                    }
                    .ok();
                    let generate = || {
                        $c::Database::generate(&seeds.db, track, module.clone(), stream.clone(), prop)
                    };
                    let to_bytes = |db: &$c::Database| db.to_bytes(&seeds.db, stream.clone());
                    let from_bytes =
                        |bytes: &[u8]| $c::Database::from_bytes(bytes, &seeds.db, track, stream.clone());
                    // `map_err` before `?` on both: the algorithm's error owns a
                    // vtable inside the .so; render it while the library is loaded.
                    let build = |db: &$c::Database| -> Result<Vec<u8>> {
                        let f = **build_fn.as_ref().unwrap();
                        let blob = f(
                            db,
                            &seeds.build,
                            hyperparameters.clone(),
                            module.clone(),
                            stream.clone(),
                            prop,
                        )
                        .map_err(|e| anyhow!("build_cache: {:#}", e))?;
                        // A device fuel trap is asynchronous: check it before
                        // trusting the bytes.
                        stream.synchronize()?;
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
                                .launch(LaunchConfig {
                                    grid_dim: (1, 1, 1),
                                    block_dim: (1, 1, 1),
                                    shared_mem_bytes: 0,
                                })?;
                        }
                        if stream.memcpy_dtov(&error_stat)?[0] != 0 {
                            return Err(anyhow!(
                                "build_cache trapped on the device (out of fuel?); no cache written"
                            ));
                        }
                        Ok(blob)
                    };
                    let load = |db: &$c::Database, bytes: &[u8]| -> Result<()> {
                        let f = **load_fn.as_ref().unwrap();
                        f(db, bytes, module.clone(), stream.clone(), prop)
                            .map_err(|e| anyhow!("load_cache: {:#}", e))
                    };
                    let db = run_cache_stage(CacheStage {
                        challenge_cache: challenge_cache.as_deref(),
                        algorithm_cache: algorithm_cache.as_deref(),
                        generate: &generate,
                        to_bytes: &to_bytes,
                        from_bytes: &from_bytes,
                        build: build_fn.as_ref().map(|_| &build as &dyn Fn(&$c::Database) -> Result<Vec<u8>>),
                        load: load_fn.as_ref().map(|_| &load as &dyn Fn(&$c::Database, &[u8]) -> Result<()>),
                    })?;
                    // Neither the build nor the load is this nonce's work.
                    unsafe { *fuel_remaining_ptr = max_fuel };
                    $c::Challenge::for_nonce(&db, &seeds, track, module, stream, prop)
                }
            )
        }};
        ($c:ident, gpu_with, $make_challenge:expr) => {{
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

            let challenge = $make_challenge(&track, module.clone(), stream.clone(), &prop)?;

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
            dispatch_challenge!(c004, gpu_cached)
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
        // Test-only: a fake CPU challenge with a shared input, so the
        // `cpu_cached` arm is expanded and type-checked by `cargo test` even
        // though no shipped CPU challenge selects it yet.
        #[cfg(test)]
        "c000" => dispatch_challenge!(fake_cpu, cpu_cached),
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

/// Per-process temp name: two processes that both find the cache missing
/// (two batches of one precommit, say) each write their own file and rename,
/// and the last rename wins with a complete file either way.
fn tmp_path(cache: &Path) -> PathBuf {
    let mut s = cache.as_os_str().to_owned();
    s.push(format!(".tmp.{}", std::process::id()));
    PathBuf::from(s)
}

/// The two-stage harness, independent of challenge and device.
///
/// Stage 1 is the challenge's shared input: read from `challenge_cache` when
/// the file exists (decoded and checked by `from_bytes`), otherwise produced by
/// `generate` and written for the next process. Stage 2 is the algorithm's
/// cache: on a hit `load` gets the bytes; on a miss `build` runs, its bytes are
/// written, and the caller continues without reloading, because a build leaves
/// the algorithm ready to solve. If the algorithm exports neither function a
/// note is printed and the solve proceeds without one.
///
/// Every device- or challenge-specific step arrives as a closure, so the GPU
/// and CPU arms differ only in what they pass here, and the branching itself
/// is unit-tested with fakes.
struct CacheStage<'a, D> {
    challenge_cache: Option<&'a Path>,
    algorithm_cache: Option<&'a Path>,
    generate: &'a dyn Fn() -> Result<D>,
    to_bytes: &'a dyn Fn(&D) -> Result<Vec<u8>>,
    from_bytes: &'a dyn Fn(&[u8]) -> Result<D>,
    build: Option<&'a dyn Fn(&D) -> Result<Vec<u8>>>,
    load: Option<&'a dyn Fn(&D, &[u8]) -> Result<()>>,
}

fn run_cache_stage<D>(stage: CacheStage<'_, D>) -> Result<D> {
    let db = match stage.challenge_cache {
        Some(path) if path.exists() => (stage.from_bytes)(&fs::read(path)?)?,
        Some(path) => {
            let db = (stage.generate)()?;
            write_blob(path, &(stage.to_bytes)(&db)?)?;
            db
        }
        None => (stage.generate)()?,
    };
    if let Some(cache) = stage.algorithm_cache {
        match (stage.build, stage.load) {
            (Some(build), Some(load)) => {
                if cache.exists() {
                    load(&db, &fs::read(cache)?)?;
                } else {
                    let blob = build(&db)?;
                    write_blob(cache, &blob)?;
                }
            }
            _ => eprintln!(
                "--algorithm-cache given but the algorithm exports no build_cache/load_cache pair; solving without one"
            ),
        }
    }
    Ok(db)
}

/// Write `bytes` to `path` through a temp file and a rename, so a reader can
/// never see a partial file.
fn write_blob(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() {
            fs::create_dir_all(dir)?;
        }
    }
    let tmp = tmp_path(path);
    fs::write(&tmp, bytes)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

/// A minimal CPU challenge with a shared input, used only to expand the
/// `cpu_cached` arm under `cargo test`. Mirrors what a real one must provide.
#[cfg(test)]
mod fake_cpu {
    use anyhow::Result;
    use serde::{Deserialize, Serialize};
    use tig_challenges::Seeds;

    #[derive(Deserialize)]
    pub struct Track;
    #[derive(Serialize)]
    pub struct Solution;
    impl Solution {
        pub fn new() -> Self {
            Solution
        }
    }
    pub struct Database(pub Vec<u8>);
    impl Database {
        pub fn generate(db_seed: &[u8; 32], _track: &Track) -> Result<Self> {
            Ok(Database(db_seed.to_vec()))
        }
        pub fn to_bytes(&self, _db_seed: &[u8; 32]) -> Result<Vec<u8>> {
            Ok(self.0.clone())
        }
        pub fn from_bytes(bytes: &[u8], _db_seed: &[u8; 32], _track: &Track) -> Result<Self> {
            Ok(Database(bytes.to_vec()))
        }
    }
    pub struct Challenge;
    impl Challenge {
        pub fn for_nonce(_db: &Database, _seeds: &Seeds, _track: &Track) -> Result<Self> {
            Ok(Challenge)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    /// A stage over a fake database that records which closures ran.
    struct Fake {
        generated: Cell<u32>,
        decoded: Cell<u32>,
        built: Cell<u32>,
        loaded: Cell<Option<Vec<u8>>>,
    }
    impl Fake {
        fn new() -> Self {
            Fake {
                generated: Cell::new(0),
                decoded: Cell::new(0),
                built: Cell::new(0),
                loaded: Cell::new(None),
            }
        }
    }

    fn run(
        fake: &Fake,
        challenge: Option<&Path>,
        algorithm: Option<&Path>,
        exports: bool,
    ) -> Vec<u8> {
        let generate = || {
            fake.generated.set(fake.generated.get() + 1);
            Ok(b"db".to_vec())
        };
        let to_bytes = |db: &Vec<u8>| Ok(db.clone());
        let from_bytes = |bytes: &[u8]| {
            fake.decoded.set(fake.decoded.get() + 1);
            Ok(bytes.to_vec())
        };
        let build = |db: &Vec<u8>| {
            fake.built.set(fake.built.get() + 1);
            Ok([db.as_slice(), b"+index"].concat())
        };
        let load = |_db: &Vec<u8>, bytes: &[u8]| {
            fake.loaded.set(Some(bytes.to_vec()));
            Ok(())
        };
        run_cache_stage(CacheStage {
            challenge_cache: challenge,
            algorithm_cache: algorithm,
            generate: &generate,
            to_bytes: &to_bytes,
            from_bytes: &from_bytes,
            build: exports.then_some(&build as &dyn Fn(&Vec<u8>) -> Result<Vec<u8>>),
            load: exports.then_some(&load as &dyn Fn(&Vec<u8>, &[u8]) -> Result<()>),
        })
        .unwrap()
    }

    #[test]
    fn without_cache_flags_the_stage_only_generates() {
        let fake = Fake::new();
        assert_eq!(run(&fake, None, None, true), b"db");
        assert_eq!(
            (fake.generated.get(), fake.decoded.get(), fake.built.get()),
            (1, 0, 0)
        );
        assert!(fake.loaded.take().is_none());
    }

    #[test]
    fn the_challenge_cache_is_written_on_a_miss_and_decoded_on_a_hit() {
        let fake = Fake::new();
        let path = scratch("challenge").join("challenge_cache.bin");
        run(&fake, Some(&path), None, true);
        assert_eq!(fs::read(&path).unwrap(), b"db");
        assert_eq!((fake.generated.get(), fake.decoded.get()), (1, 0));
        run(&fake, Some(&path), None, true);
        assert_eq!(
            (fake.generated.get(), fake.decoded.get()),
            (1, 1),
            "a hit must not regenerate"
        );
    }

    #[test]
    fn the_algorithm_cache_builds_on_a_miss_and_loads_on_a_hit_never_both() {
        let fake = Fake::new();
        let path = scratch("algorithm").join("algorithm_cache.bin");
        run(&fake, None, Some(&path), true);
        assert_eq!(fs::read(&path).unwrap(), b"db+index");
        assert_eq!(fake.built.get(), 1);
        assert!(
            fake.loaded.take().is_none(),
            "a miss continues without reloading"
        );
        run(&fake, None, Some(&path), true);
        assert_eq!(fake.built.get(), 1, "a hit must not rebuild");
        assert_eq!(fake.loaded.take().unwrap(), b"db+index");
    }

    #[test]
    fn missing_exports_skip_the_algorithm_cache_without_writing() {
        let fake = Fake::new();
        let path = scratch("noexports").join("algorithm_cache.bin");
        run(&fake, None, Some(&path), false);
        assert!(!path.exists());
        assert_eq!(fake.built.get(), 0);
    }

    #[test]
    fn a_failed_build_writes_nothing() {
        let path = scratch("failbuild").join("algorithm_cache.bin");
        let generate = || Ok(vec![1u8]);
        let to_bytes = |db: &Vec<u8>| Ok(db.clone());
        let from_bytes = |b: &[u8]| Ok(b.to_vec());
        let build = |_: &Vec<u8>| Err(anyhow!("boom"));
        let load = |_: &Vec<u8>, _: &[u8]| Ok(());
        let err = run_cache_stage(CacheStage {
            challenge_cache: None,
            algorithm_cache: Some(&path),
            generate: &generate,
            to_bytes: &to_bytes,
            from_bytes: &from_bytes,
            build: Some(&build),
            load: Some(&load),
        })
        .unwrap_err();
        assert!(err.to_string().contains("boom"));
        assert!(!path.exists());
    }

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
        // Catches a swap of the two fields, which would make the database
        // per-nonce again with nothing failing.
        let settings = test_settings();
        let rand_hash = "random_hash".to_string();
        let seeds = seeds_for(&settings, &rand_hash, 1337);
        assert_eq!(seeds.db, settings.calc_db_seed(&rand_hash));
        assert_eq!(seeds.build, settings.calc_build_seed(&rand_hash));
        assert_eq!(seeds.instance, settings.calc_seed(&rand_hash, 1337));
        assert_eq!(seeds.algo, settings.calc_algo_seed(&rand_hash, 1337));
    }

    #[test]
    fn seeds_for_holds_the_database_constant_across_nonces() {
        let settings = test_settings();
        let rand_hash = "random_hash".to_string();
        let a = seeds_for(&settings, &rand_hash, 0);
        let b = seeds_for(&settings, &rand_hash, u64::MAX);
        assert_eq!(a.db, b.db, "database seed must not vary with the nonce");
        assert_eq!(a.build, b.build, "build seed must not vary with the nonce");
        assert_ne!(
            a.instance, b.instance,
            "instance seed must vary with the nonce"
        );
        assert_ne!(a.algo, b.algo, "algo seed must vary with the nonce");
        assert_ne!(
            a.instance, a.algo,
            "the algorithm must not receive the query seed"
        );
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "tig-runtime-cache-test-{}-{}",
            std::process::id(),
            name
        ));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn the_two_cache_flags_parse_independently() {
        let m = cli()
            .try_get_matches_from([
                "tig-runtime",
                "s",
                "r",
                "1",
                "b.so",
                "--challenge-cache",
                "db.bin",
                "--algorithm-cache",
                "c.blob",
            ])
            .unwrap();
        assert_eq!(
            m.get_one::<PathBuf>("challenge-cache").unwrap(),
            &PathBuf::from("db.bin")
        );
        assert_eq!(
            m.get_one::<PathBuf>("algorithm-cache").unwrap(),
            &PathBuf::from("c.blob")
        );
        // Either alone is fine: an algorithm cache without a challenge cache
        // just regenerates the database each time.
        let m = cli()
            .try_get_matches_from([
                "tig-runtime",
                "s",
                "r",
                "1",
                "b.so",
                "--algorithm-cache",
                "c.blob",
            ])
            .unwrap();
        assert!(m.get_one::<PathBuf>("challenge-cache").is_none());
        let m = cli()
            .try_get_matches_from(["tig-runtime", "s", "r", "1", "b.so"])
            .unwrap();
        assert!(m.get_one::<PathBuf>("challenge-cache").is_none());
        assert!(m.get_one::<PathBuf>("algorithm-cache").is_none());
    }

    #[test]
    fn write_blob_is_atomic_and_overwrites() {
        let path = scratch("write").join("cache.blob");
        write_blob(&path, b"first").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"first");
        assert!(!tmp_path(&path).exists(), "temp file must be renamed away");
        write_blob(&path, b"second").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"second");
    }
}
