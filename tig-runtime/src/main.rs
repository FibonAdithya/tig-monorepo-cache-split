use anyhow::{anyhow, Result};
use clap::{arg, Command};
use libloading::Library;
use serde_json::{Map, Value};
#[cfg(feature = "c004")]
use std::time::Duration;
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
        // `--memory-cap` and `--build-timeout` were guarded here while they
        // parsed but did nothing. Task 5 wires both -- the balloon and the
        // watchdog inside `build_index` -- so the guards are gone.

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

    // Both forms reach the SAME call below. `batch` is not a second execution
    // path: the legacy single-nonce form is `start_nonce = NONCE,
    // num_nonces = 1`. Two paths is exactly where the batched and single-nonce
    // results would silently diverge, and nothing downstream would notice.
    //
    // Reading every remaining argument off `m` rather than off `matches` is
    // also what wires `--index` in *both* places it is declared -- on the root
    // command and on `batch`. Reading it from `matches` would leave
    // `batch --index blob` solving without the index and exiting 0.
    let (m, start_nonce, num_nonces) = match matches.subcommand_matches("batch") {
        // The `.unwrap()`s are safe only because `cli()` marks both
        // `.required(true)`, and `--num-nonces` is range-limited to `1..`.
        Some(sub) => (
            sub,
            *sub.get_one::<u64>("start-nonce").unwrap(),
            *sub.get_one::<u64>("num-nonces").unwrap(),
        ),
        None => (&matches, *matches.get_one::<u64>("NONCE").unwrap(), 1u64),
    };

    if let Err(e) = compute_solution(
        m.get_one::<String>("SETTINGS").unwrap().clone(),
        m.get_one::<String>("RAND_HASH").unwrap().clone(),
        start_nonce,
        num_nonces,
        m.get_one::<PathBuf>("BINARY").unwrap().clone(),
        m.get_one("hyperparameters").cloned(),
        m.get_one::<PathBuf>("ptx").cloned(),
        *m.get_one::<u64>("fuel").unwrap(),
        m.get_one::<PathBuf>("output").cloned(),
        m.get_one::<usize>("gpu").cloned(),
        m.get_one::<PathBuf>("index").cloned(),
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

/// The half-open range of nonces one process covers.
///
/// Split out of `compute_solution` so both boundaries are testable without a
/// GPU. Each is silent if it is wrong: a batch of zero writes no output at all
/// and exits 0, which is indistinguishable from a batch that worked; and a
/// plain `start_nonce + num_nonces` wraps in a release build, turning a batch
/// near `u64::MAX` into an empty range that also exits 0 having done nothing.
fn nonce_range(start_nonce: u64, num_nonces: u64) -> Result<std::ops::Range<u64>> {
    if num_nonces == 0 {
        return Err(anyhow!(
            "num_nonces must be at least 1; a batch that solves nothing and exits 0 \
             is indistinguishable from one that worked"
        ));
    }
    let end = start_nonce.checked_add(num_nonces).ok_or_else(|| {
        anyhow!(
            "nonce range {}..{}+{} overflows u64",
            start_nonce,
            start_nonce,
            num_nonces
        )
    })?;
    Ok(start_nonce..end)
}

/// Solve `num_nonces` consecutive nonces, starting at `start_nonce`, in this
/// one process.
///
/// The legacy single-nonce form is `start_nonce = NONCE, num_nonces = 1`, so
/// there is one loop and not two code paths. Everything that does not depend on
/// the nonce -- the CUDA context, the module, the c004 `Database` and the index
/// upload -- is hoisted above the loop and therefore paid once per precommit
/// instead of once per nonce.
pub fn compute_solution(
    settings: String,
    rand_hash: String,
    start_nonce: u64,
    num_nonces: u64,
    library_path: PathBuf,
    hyperparameters: Option<String>,
    ptx_path: Option<PathBuf>,
    max_fuel: u64,
    output_folder: Option<PathBuf>,
    gpu_device: Option<usize>,
    index_path: Option<PathBuf>,
) -> Result<()> {
    let settings = load_settings(&settings);
    let nonces = nonce_range(start_nonce, num_nonces)?;

    // Computed once, above the loop: it carries no nonce, so it is constant for
    // the whole precommit. Only the c004 arm reads it, hence the allow -- a
    // `--features c001` build compiles no arm that mentions it.
    #[allow(unused_variables)]
    let db_seed = settings.calc_db_seed(&rand_hash);

    let hyperparameters = hyperparameters.map(|x| load_hyperparameters(&x));

    let library = load_module(&library_path)?;
    // Resolved once, but *written* once per nonce inside the loop. Setting them
    // here only -- which is what the one-process-per-nonce code did -- would
    // carry nonce k-1's spend into nonce k, and a bundle would die partway
    // through with an out-of-fuel exit that names nothing.
    let fuel_remaining_ptr = unsafe { *library.get::<*mut u64>(b"__fuel_remaining")? };
    let runtime_signature_ptr = unsafe { *library.get::<*mut u64>(b"__runtime_signature")? };

    // A directory, not a file. The file name depends on the nonce and is
    // therefore rebuilt inside the loop; computing it once here would make
    // every nonce of a bundle overwrite one file.
    let output_dir = match output_folder {
        Some(folder) => {
            fs::create_dir_all(&folder)?;
            folder
        }
        None => PathBuf::from("."),
    };

    /// `--index` only means something for c004: it is the only challenge with a
    /// `Database` and the only one whose ABI has `load_index`. Refusing loudly
    /// beats the alternative, which is solving without the index the caller
    /// asked for and exiting 0.
    #[allow(unused_macros)]
    macro_rules! refuse_index {
        ($c:ident) => {
            if index_path.is_some() {
                return Err(anyhow!(
                    "--index is only supported by c004; {} has no index ABI, and \
                     solving without the index that was asked for would be a \
                     silently wrong answer",
                    stringify!($c)
                ));
            }
        };
    }

    /// The GPU solve loop, shared by every GPU challenge.
    ///
    /// Exactly two things differ between challenges, and both arrive as
    /// closures so that the loop, the per-nonce resets and the save-solution
    /// closure exist in one copy only:
    ///
    ///   * `$make_db` runs **once, above the loop**. For c004 it generates the
    ///     precommit's `Database` and uploads the index into the algorithm;
    ///     for c005/c006 it returns `()`.
    ///   * `$mk_challenge` builds one nonce's `Challenge` from that value.
    ///
    /// Every reference parameter of both closures is explicitly annotated:
    /// that is what makes them higher-ranked over lifetimes, so the same
    /// closure can be called with a fresh `Seeds` borrow on each iteration.
    #[allow(unused_macros)]
    macro_rules! gpu_common {
        ($c:ident, $make_db:expr, $mk_challenge:expr) => {{
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
            // `start_nonce`, not the loop's nonce: the context is created once,
            // above the loop, so there is no per-nonce device left to pick.
            let gpu_device = gpu_device.unwrap_or((start_nonce % num_gpus as u64) as usize);
            let ptx = Ptx::from_src(modified_ptx);
            let ctx = CudaContext::new(gpu_device)?;
            ctx.set_blocking_synchronize()?;
            let module = ctx.load_module(ptx)?;
            let stream = ctx.fuel_check_stream();
            let prop = get_device_prop(gpu_device as i32)?;

            let cfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            };
            let initialize_kernel = module.load_function("initialize_kernel")?;

            // Above the loop, and this placement is the entire point of the
            // mode: the database generation and the index upload are paid once
            // per precommit rather than once per nonce.
            //
            // Note for the spec: `load_index` runs BEFORE the first
            // `initialize_kernel`, so like `generate_instance` it is outside the
            // fuel meter. That is deliberate and not exploitable -- it receives
            // only the `Database` and the blob, never a query -- but it is a
            // second free phase the enforcement table does not yet mention.
            let db = $make_db(&track, module.clone(), stream.clone(), &prop)?;

            for nonce in nonces {
                let seeds = seeds_for(&settings, &rand_hash, nonce);

                // Both CPU counters, reset per nonce. See the comment where
                // the pointers are resolved.
                unsafe { *fuel_remaining_ptr = max_fuel };
                unsafe {
                    *runtime_signature_ptr =
                        u64::from_be_bytes(seeds.nonce[0..8].try_into().unwrap())
                };

                let challenge =
                    $mk_challenge(&db, &seeds, &track, module.clone(), stream.clone(), &prop)?;

                // The device-side counterpart of the two resets above:
                // `initialize_kernel` zeroes gbl_FUELUSAGE and gbl_ERRORSTAT
                // and sets gbl_SIGNATURE. Hoisting it out of the loop would
                // leave the device's fuel accumulating across the bundle while
                // the CPU's did not -- a half-reset that still dies out of fuel
                // partway through, just later.
                unsafe {
                    stream
                        .launch_builder(&initialize_kernel)
                        .arg(&(u64::from_be_bytes(seeds.nonce[8..16].try_into().unwrap())))
                        .launch(cfg)?;
                }

                // Both of these are rebuilt per nonce, and both must resolve to
                // *this* nonce. A stale `output_file` makes every nonce
                // overwrite one file; a stale `nonce` field stamps every file
                // in the bundle with the batch's first nonce. Neither fails:
                // the files are well-formed, the verifier is never asked about
                // the mismatch, and the slave's Merkle root is silently wrong.
                let output_file = output_dir.join(format!("{}.json", nonce));
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
                    hyperparameters.clone(),
                    module.clone(),
                    stream.clone(),
                    &prop,
                );
                if !output_file.exists() {
                    save_solution_fn(&$c::Solution::new())?;
                }
                result?;
            }
            Ok(())
        }};
    }

    macro_rules! dispatch_challenge {
        ($c:ident, cpu) => {{
            refuse_index!($c);
            let track: $c::Track = parse_track(&settings, stringify!($c))?;

            // library function may exit 87 if it runs out of fuel
            let solve_challenge_fn = unsafe {
                library.get::<fn(
                    &$c::Challenge,
                    &dyn Fn(&$c::Solution) -> Result<()>,
                    Option<String>,
                ) -> Result<()>>(b"entry_point")?
            };

            for nonce in nonces {
                let seeds = seeds_for(&settings, &rand_hash, nonce);
                let seed = seeds.nonce;

                // Reset per nonce, exactly as in the GPU loop.
                unsafe { *fuel_remaining_ptr = max_fuel };
                unsafe {
                    *runtime_signature_ptr = u64::from_be_bytes(seed[0..8].try_into().unwrap())
                };

                let challenge = $c::Challenge::generate_instance(&seed, &track)?;

                let output_file = output_dir.join(format!("{}.json", nonce));
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
                let result =
                    solve_challenge_fn(&challenge, &save_solution_fn, hyperparameters.clone());
                if !output_file.exists() {
                    save_solution_fn(&$c::Solution::new())?;
                }
                result?;
            }
            Ok(())
        }};

        // c005 and c006: no `Database`, no index ABI. They run the same loop,
        // so `batch` works for them too; there is simply nothing to hoist.
        ($c:ident, gpu) => {{
            refuse_index!($c);
            gpu_common!(
                $c,
                |_track: &$c::Track,
                 _module: Arc<CudaModule>,
                 _stream: Arc<CudaStream>,
                 _prop: &cudaDeviceProp|
                 -> Result<()> { Ok(()) },
                |_db: &(),
                 seeds: &Seeds,
                 track: &$c::Track,
                 module: Arc<CudaModule>,
                 stream: Arc<CudaStream>,
                 prop: &cudaDeviceProp|
                 -> Result<$c::Challenge> {
                    $c::Challenge::generate_instance(seeds, track, module, stream, prop)
                }
            )
        }};

        // c004: the database is nonce-free, so it and the index are built once
        // above the loop and every nonce of the bundle reuses them.
        ($c:ident, gpu_db) => {{
            gpu_common!(
                $c,
                |track: &$c::Track,
                 module: Arc<CudaModule>,
                 stream: Arc<CudaStream>,
                 prop: &cudaDeviceProp|
                 -> Result<$c::Database> {
                    let database = $c::Database::generate(
                        &db_seed,
                        track,
                        module.clone(),
                        stream.clone(),
                        prop,
                    )?;

                    if let Some(index_path) = index_path.as_ref() {
                        let blob = fs::read(index_path)?;
                        let load_index_fn = unsafe {
                            library.get::<fn(
                                &$c::Database,
                                &[u8],
                                Arc<CudaModule>,
                                Arc<CudaStream>,
                                &cudaDeviceProp,
                            ) -> Result<()>>(b"load_index")
                        }
                        .map_err(|_| {
                            anyhow!(
                                "--index was given but the algorithm does not export \
                                 `load_index`"
                            )
                        })?;
                        // `map_err` before `?`, not a bare `?`: the algorithm's
                        // `anyhow::Error` owns a vtable that lives inside the
                        // `.so`. Propagating it drops `library`, `dlclose`s the
                        // object, and the caller's `eprintln!` then jumps
                        // through a dangling pointer -- measured on tig-gpu as
                        // `Runtime Error: ` followed by a SIGSEGV. Rendering it
                        // here, while the library is still loaded, hands back a
                        // plain runtime-owned string.
                        load_index_fn(&database, &blob, module, stream, prop)
                            .map_err(|e| anyhow!("{:#}", e))?;
                    }
                    Ok(database)
                },
                |db: &$c::Database,
                 seeds: &Seeds,
                 track: &$c::Track,
                 module: Arc<CudaModule>,
                 stream: Arc<CudaStream>,
                 prop: &cudaDeviceProp|
                 -> Result<$c::Challenge> {
                    $c::Challenge::for_nonce(db, seeds, track, module, stream, prop)
                }
            )
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
            dispatch_challenge!(c004, gpu_db)
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

/// No allocator hands out 100% of reported free memory -- fragmentation and
/// per-allocation bookkeeping always leave a sliver unreachable -- so a balloon
/// of exactly `free - cap` fails on a healthy device and every build errors
/// out. Reserve a fixed, documented sliver instead, and treat the effective cap
/// as `memory_cap_bytes + BALLOON_SLACK`.
#[cfg(any(feature = "c004", test))]
const BALLOON_SLACK: u64 = 64 * 1024 * 1024;

/// How many bytes the balloon must hold so that only `memory_cap_bytes`
/// (+ [`BALLOON_SLACK`]) of device memory is reachable by the algorithm.
///
/// Split out of `build_index` so the arithmetic is testable without a GPU: the
/// refusal is the load-bearing half of the cap, and a cap that silently sizes
/// itself to zero is worse than no build at all.
#[cfg(any(feature = "c004", test))]
fn balloon_size(free: u64, total: u64, memory_cap_bytes: u64) -> Result<u64> {
    // `saturating_add`, not `+`: a nonsense `--memory-cap u64::MAX` would
    // otherwise wrap to a tiny number, pass the check, and hand back a balloon
    // sized from a wrapped subtraction.
    let required = memory_cap_bytes.saturating_add(BALLOON_SLACK);
    if free < required {
        return Err(anyhow!(
            "device has {} bytes free of {} but the memory cap is {} (+{} slack); \
             refusing to run with a cap that would not be enforced",
            free,
            total,
            memory_cap_bytes,
            BALLOON_SLACK
        ));
    }
    Ok(free - required)
}

#[cfg(feature = "c004")]
pub fn build_index(
    settings: String,
    rand_hash: String,
    library_path: PathBuf,
    hyperparameters: Option<String>,
    ptx_path: PathBuf,
    build_fuel: u64,
    memory_cap_bytes: u64,
    timeout_secs: u64,
    index_out: PathBuf,
    gpu_device: Option<usize>,
) -> Result<()> {
    let settings = load_settings(&settings);

    // A flat ceiling that kills a runaway build. It is not the budget -- the
    // budget is `--build-fuel`, which is deterministic and hardware
    // independent. This only catches a build that will never finish.
    //
    // The thread is deliberately detached and never cancelled: it dies with the
    // process. Exit 85 is distinct from 84 (runtime error) and 87 (out of
    // fuel), so a caller can tell a timeout from a failure.
    let timeout = Duration::from_secs(timeout_secs);
    std::thread::spawn(move || {
        std::thread::sleep(timeout);
        eprintln!(
            "build exceeded the {}s wall-clock watchdog",
            timeout.as_secs()
        );
        // `libc::_exit`, not `std::process::exit`. The latter calls libc
        // `exit()`, which runs atexit handlers and shared-object destructors --
        // CUDA's among them -- from *this* thread while the main thread is
        // parked inside `cuStreamSynchronize` (this function calls
        // `set_blocking_synchronize`, so that park is a real blocking wait
        // inside libcuda). Measured on tig-gpu with a kernel genuinely in
        // flight, `std::process::exit(85)` gave 139 (SIGSEGV, core dumped) on
        // one run and 85 on the very next -- nondeterministic, and a 139 is
        // indistinguishable from this box's pre-existing teardown crash. It is
        // exactly the timeout-under-load case the watchdog exists for.
        // `_exit` skips atexit entirely, and Rust's stderr is unbuffered, so
        // the message above has already landed.
        unsafe { libc::_exit(85) };
    });

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

    // Inflated *after* `Database::generate` and *before* `build_index_fn`, so
    // the database and the generator's scratch sit outside the algorithm's cap
    // rather than being charged against it.
    //
    // A hard cap, not a sampled one. Polling `mem_get_info` from the watchdog
    // would miss a spike between samples; holding the surplus makes any
    // allocation past the cap fail as an ordinary cudaMalloc error inside the
    // algorithm.
    let (free, total) = cudarc::driver::result::mem_get_info()?;
    let balloon_bytes = balloon_size(free as u64, total as u64, memory_cap_bytes)?;
    // Deliberately NOT retried at a smaller size on failure: a smaller balloon
    // is a *looser* cap, so a shrink loop silently converts "cannot enforce the
    // cap" into "ran without one".
    let _balloon = stream
        .alloc_zeros::<u8>(balloon_bytes as usize)
        .map_err(|e| {
            anyhow!(
                "could not inflate the {}-byte memory balloon ({}); the cap would be \
             unenforced, so the build is refused rather than run uncapped",
                balloon_bytes,
                e
            )
        })?;

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

    // `map_err` before `?`, not a bare `?`: the `anyhow::Error` the algorithm
    // returns owns a vtable that lives inside the `.so`. Propagating it out of
    // this function drops `library`, `dlclose`s the object, and the caller's
    // `eprintln!("Runtime Error: {}", e)` then jumps through a dangling
    // pointer -- observed as a SIGSEGV that printed `Runtime Error: ` and
    // nothing else. Rendering the message here, while the library is still
    // loaded, hands the caller a plain runtime-owned string instead.
    let blob = build_index_fn(
        &database,
        hyperparameters,
        module.clone(),
        stream.clone(),
        &prop,
    )
    .map_err(|e| anyhow!("{:#}", e))?;

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

    /// The *code* of `build_index`'s body -- comment lines stripped -- bounded
    /// loudly at both ends.
    ///
    /// Three tests below assert on this slice, and each is only as good as the
    /// bounding: a scan that silently matches nothing, or that matches the test
    /// file instead of the function, passes for the wrong reason. Comments are
    /// dropped because the prose in `build_index` names `Database::generate`
    /// and `build_index_fn` verbatim, and an ordering assertion that can be
    /// satisfied by a comment is not an ordering assertion.
    fn build_index_code() -> String {
        const SRC: &str = include_str!("main.rs");
        // `concat!` on purpose: as one literal the needle would occur in this
        // helper's own source and `find` would match here, not at the
        // definition.
        const NEEDLE: &str = concat!("pub fn ", "build_index(");
        assert_eq!(
            SRC.matches(NEEDLE).count(),
            1,
            "expected exactly one `{}` in main.rs; the scan cannot tell which \
             one to bound",
            NEEDLE
        );
        let rest = &SRC[SRC.find(NEEDLE).unwrap()..];
        // Inside the function every closing brace is indented; the first brace
        // alone at column 0 is the function's own.
        let end = rest.find("\n}\n").expect(
            "could not find the closing brace of `build_index` -- the source \
             layout changed and these tests can no longer bound the function",
        );
        let body = &rest[..end];
        assert!(
            body.len() > 500 && body.contains("index_out"),
            "sliced {} bytes that do not look like build_index's body",
            body.len()
        );
        body.lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// `find` the one occurrence of `needle` in `code`, failing loudly if it is
    /// not there exactly once.
    fn offset_of(code: &str, needle: &str) -> usize {
        assert_eq!(
            code.matches(needle).count(),
            1,
            "expected exactly one `{}` in build_index's code; found {}. A \
             source scan that matches zero or several places proves nothing",
            needle,
            code.matches(needle).count()
        );
        code.find(needle).unwrap()
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
        assert_eq!(
            *m.subcommand_matches("batch")
                .unwrap()
                .get_one::<u64>("num-nonces")
                .unwrap(),
            1
        );
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

    #[test]
    fn the_balloon_absorbs_everything_the_cap_does_not_allow() {
        // `free` deliberately differs from `total`: on a real device the
        // context, the module and the database are already resident when the
        // balloon goes up. Sizing from `total` would hand the algorithm
        // everything they hold *on top of* its cap, and with free == total
        // that mutation is invisible.
        let total = 12 * 1024 * 1024 * 1024u64;
        let free = 11 * 1024 * 1024 * 1024u64;
        let cap = 2 * 1024 * 1024 * 1024u64;
        // The balloon must hold the whole remainder minus the documented
        // slack, not some fraction of it: a smaller balloon is a looser cap
        // than the caller asked for.
        assert_eq!(
            balloon_size(free, total, cap).unwrap(),
            free - cap - BALLOON_SLACK
        );
    }

    #[test]
    fn a_cap_larger_than_free_memory_is_refused_not_silently_shrunk() {
        // 32 GiB cap on a 12 GiB card. The dangerous outcome is a zero-sized
        // (or wrapped) balloon and a build that runs completely uncapped, so
        // this must be an error, never an `Ok(0)`.
        let free = 12 * 1024 * 1024 * 1024u64;
        let err = balloon_size(free, free, 32 * 1024 * 1024 * 1024)
            .expect_err("a cap above free memory must be refused, not honoured with no balloon");
        assert!(
            err.to_string().contains("would not be enforced"),
            "the refusal must say the cap is unenforceable, got: {}",
            err
        );
    }

    #[test]
    fn the_cap_is_refused_when_only_the_slack_is_missing() {
        // The boundary, both sides of it. `free == cap + slack` is exactly
        // enough (a zero-byte balloon is correct there -- nothing is
        // reachable beyond the cap); one byte less is not, and an off-by-one
        // in the comparison would let a build run a byte over its ceiling.
        let cap = 2 * 1024 * 1024 * 1024u64;
        assert_eq!(
            balloon_size(cap + BALLOON_SLACK, cap + BALLOON_SLACK, cap).unwrap(),
            0
        );
        assert!(balloon_size(cap + BALLOON_SLACK - 1, cap + BALLOON_SLACK - 1, cap).is_err());
    }

    #[test]
    fn an_absurd_cap_cannot_wrap_into_a_tiny_balloon() {
        // `cap + BALLOON_SLACK` overflows u64 here. With a plain `+` in a
        // release build this wraps to a small number, passes the `free <`
        // check, and `free - required` then hands back a nearly-free-sized
        // balloon while claiming to enforce a 16-exabyte cap.
        let free = 12 * 1024 * 1024 * 1024u64;
        assert!(balloon_size(free, free, u64::MAX).is_err());
        assert!(balloon_size(free, free, u64::MAX - BALLOON_SLACK + 1).is_err());
    }

    #[test]
    fn the_watchdog_exits_85_and_covers_the_whole_build() {
        // Two independent properties. (a) 85 must stay distinct from 84
        // (runtime error) and 87 (out of fuel), or a caller cannot tell a
        // build that will never finish from one that failed. (b) The watchdog
        // must be armed before any of the expensive work, or a build that
        // wedges inside CudaContext::new or Database::generate -- neither of
        // which is the algorithm's code, and both of which can hang -- runs
        // forever uncovered. The thread is unreachable from a unit test
        // without a GPU, so both are asserted on the source.
        let code = build_index_code();
        assert!(
            code.contains(concat!("libc::_", "exit(85)")),
            "build_index must install a watchdog that exits 85"
        );
        // Not a restatement of the line above: `std::process::exit(85)` also
        // "exits 85", and it is what was there until a run with a kernel in
        // flight produced 139 instead. libc `exit()` runs CUDA's atexit
        // handlers from the watchdog thread while the main thread is inside
        // cuStreamSynchronize; `_exit` does not.
        assert!(
            !code.contains(concat!("std::process::", "exit(")),
            "the watchdog must not use std::process::exit: it runs atexit \
             handlers, and under GPU load that was measured crashing to 139 \
             instead of exiting 85"
        );
        let settings = offset_of(&code, concat!("load_", "settings("));
        let spawn = offset_of(&code, concat!("thread::", "spawn("));
        let ctx = offset_of(&code, concat!("CudaContext::", "new("));
        assert!(
            settings < spawn && spawn < ctx,
            "the watchdog must be armed after the settings load and before the \
             CUDA context is created (settings@{} spawn@{} ctx@{})",
            settings,
            spawn,
            ctx
        );
    }

    #[test]
    fn the_algorithms_error_is_rendered_before_the_library_is_unloaded() {
        // The algorithm's `anyhow::Error` carries a vtable that lives in the
        // `.so`. Propagating it with a bare `?` drops `library`, dlclose()s the
        // object, and the caller's `eprintln!` then jumps through a dangling
        // pointer -- measured on tig-gpu as `Runtime Error: ` followed by a
        // SIGSEGV, with the actual allocation-failure message never printed.
        let code = build_index_code();
        let call = offset_of(&code, concat!("build_index_", "fn("));
        // Bound the window at the `?` that propagates, not at the first `;`:
        // rustfmt splits the call across lines, so a `;`-bounded window can
        // stop before the conversion it is looking for.
        let stmt = &code[call..];
        let end = stmt.find("?;").expect("the call must propagate with `?`");
        // The needle is the *formatting*, not `.map_err(`. A no-op
        // `.map_err(|e| e)` restores the segfault exactly while still
        // containing `.map_err(`, so that token discriminates nothing.
        assert!(
            stmt[..end].contains(concat!("anyhow!(\"{", ":#}\"")),
            "the algorithm's error must be rendered to an owned string while the \
             library is still loaded; a bare `?`, or a map_err that hands back \
             the same error object, segfaults the error path"
        );
    }

    #[test]
    fn the_balloon_is_held_and_not_dropped_on_the_spot() {
        // `let _balloon = ...` binds; `let _ = ...` drops the allocation
        // immediately. The second frees the surplus the instant it is
        // reserved, so the cap is silently not in force -- and nothing is
        // observable at runtime: the build simply succeeds where it should
        // have failed. This is the single mutation the whole task exists to
        // prevent, and it is invisible to every other test here.
        let code = build_index_code();
        assert!(
            code.contains(concat!("let _", "balloon = ")),
            "the balloon must be bound to a live binding that outlives \
             build_index_fn; `let _ = ...` drops it at once and disables the \
             cap with no runtime symptom"
        );
    }

    #[test]
    fn the_balloon_is_inflated_between_the_database_and_the_algorithm() {
        // The ordering is the cap's whole meaning, and both ways of getting it
        // wrong are silent:
        //   * above `Database::generate` -- the database and the generator's
        //     scratch are charged against the algorithm's cap, so the
        //     algorithm gets less than the cap promises and a build that
        //     should pass fails;
        //   * below `build_index_fn` -- the cap is never in force while the
        //     algorithm runs, so a build that should fail passes.
        let code = build_index_code();
        let generate = offset_of(&code, concat!("Database::", "generate("));
        let balloon = offset_of(&code, concat!("let _", "balloon = "));
        let call = offset_of(&code, concat!("build_index_", "fn("));
        assert!(
            generate < balloon,
            "the balloon must go up AFTER Database::generate, or the database \
             is charged against the algorithm's cap (generate@{} balloon@{})",
            generate,
            balloon
        );
        assert!(
            balloon < call,
            "the balloon must go up BEFORE build_index_fn, or the cap is never \
             in force while the algorithm runs (balloon@{} call@{})",
            balloon,
            call
        );
    }

    #[test]
    fn the_build_index_flags_are_no_longer_guarded_as_unwired() {
        // Task 4 rejected explicit `--memory-cap` / `--build-timeout` so the
        // flags could not lie about being enforced. Task 5 enforces them, so
        // the guards must be gone -- otherwise every capped build exits 84.
        let src = include_str!("main.rs");
        assert!(
            !src.contains(concat!("parsed but not ", "enforced yet")),
            "the unwired-flag guard for --memory-cap / --build-timeout is still \
             present; every capped build would exit 84 instead of running"
        );
    }

    /// The code of the `gpu_common!` macro -- comment lines stripped -- bounded
    /// at both ends by the two macro definitions that surround it.
    ///
    /// Four tests below assert on offsets inside this slice, and every one of
    /// them is only as good as the bounding, so a slice that silently matched
    /// nothing (or matched this helper's own source) would be worse than no
    /// test. Comments are stripped because the prose inside `gpu_common!`
    /// names `fuel_remaining_ptr`, `initialize_kernel` and `output_file`
    /// verbatim, and an ordering assertion a comment can satisfy is not an
    /// ordering assertion.
    fn gpu_common_code() -> String {
        const SRC: &str = include_str!("main.rs");
        // `concat!` throughout: written as one literal, each needle would occur
        // in this helper's own source and `find` would match here rather than
        // at the definition.
        const OPEN: &str = concat!("macro_rules! gpu", "_common {");
        const CLOSE: &str = concat!("macro_rules! dispatch", "_challenge {");
        for needle in [OPEN, CLOSE] {
            assert_eq!(
                SRC.matches(needle).count(),
                1,
                "expected exactly one `{}` in main.rs; the scan cannot tell which \
                 one to bound",
                needle
            );
        }
        let start = SRC.find(OPEN).unwrap();
        let end = SRC.find(CLOSE).unwrap();
        assert!(
            start < end,
            "gpu_common! must be defined before dispatch_challenge! or the slice \
             runs backwards"
        );
        let body = &SRC[start..end];
        assert!(
            body.len() > 2000 && body.contains("solve_challenge_fn"),
            "sliced {} bytes that do not look like gpu_common!'s body",
            body.len()
        );
        body.lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn batch_and_index_are_no_longer_guarded_as_unwired() {
        // Task 4 refused `batch` outright and Task 5 kept the `--index` guard,
        // precisely so neither flag could lie about what it does. Task 6 wires
        // both; leaving either guard in place makes every batched or indexed
        // run exit 84 instead of running.
        let src = include_str!("main.rs");
        assert!(
            !src.contains(concat!("not implemented", " yet")),
            "the batch subcommand's unwired guard is still present; every \
             batched run would exit 84 instead of running"
        );
        assert!(
            !src.contains(concat!("parsed but not ", "loaded yet")),
            "the --index guard is still present; every indexed run would exit 84"
        );
        // The positive half. Deleting the guard without wiring the flag is the
        // dangerous outcome, not a loud one: `--index blob` would then solve
        // *without* the index and exit 0.
        assert!(
            src.contains(concat!("load_index", "_fn(&database, &blob")),
            "--index is unguarded but nothing calls `load_index` on the query \
             side; the flag would be silently ignored and the run would solve \
             without the index it was given"
        );
    }

    #[test]
    fn both_forms_reach_one_dispatch_that_reads_index_from_the_subcommand() {
        // `--index` is declared in TWO places -- on the root command and on
        // `batch`. Reading arguments off `matches` rather than off the matched
        // subcommand leaves `batch --index blob` solving without the index and
        // exiting 0, which is the exact silent-wrong-answer the guard existed
        // to prevent. And a second, batch-only call site is where the batched
        // and single-nonce paths would quietly drift apart.
        const SRC: &str = include_str!("main.rs");
        const CALL: &str = concat!("= compute_", "solution(");
        assert_eq!(
            SRC.matches(CALL).count(),
            1,
            "expected exactly one call site for compute_solution; two means the \
             legacy and batch forms are separate code paths"
        );
        let start = SRC.find(CALL).unwrap();
        let end = start
            + SRC[start..]
                .find(") {")
                .expect("the call must be the scrutinee of an `if let Err`");
        let args = &SRC[start..end];
        assert!(
            args.contains(concat!("m.get_one::<PathBuf>(\"", "index\")")),
            "the index path must be read off the matched form `m`, not off the \
             root `matches`; got: {}",
            args
        );
        assert!(
            args.contains("start_nonce") && args.contains("num_nonces"),
            "the single call site must be given the range, got: {}",
            args
        );
    }

    #[test]
    fn a_batch_of_one_is_the_legacy_form() {
        // The plan's "one code path, not two" rests on this: the legacy form is
        // start_nonce = NONCE, num_nonces = 1.
        assert_eq!(nonce_range(7, 1).unwrap(), 7..8);
        assert_eq!(nonce_range(3, 5).unwrap(), 3..8);
        assert_eq!(nonce_range(0, 5).unwrap().collect::<Vec<_>>(), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn a_nonce_range_that_would_wrap_is_refused_not_silently_emptied() {
        // With a plain `start + num` in a release build this wraps: the range
        // becomes empty (or absurd), the process writes nothing and exits 0,
        // and the caller cannot tell it from a batch that worked.
        assert!(nonce_range(u64::MAX, 2).is_err());
        assert!(nonce_range(u64::MAX - 1, 3).is_err());
        // The boundary on the other side: exactly reaching u64::MAX is fine.
        assert_eq!(nonce_range(u64::MAX - 1, 1).unwrap(), (u64::MAX - 1)..u64::MAX);
        // Zero is rejected here too, not only by clap's range parser: a caller
        // that constructs the arguments itself must not get an empty batch.
        assert!(nonce_range(0, 0).is_err());
        assert!(nonce_range(5, 0).is_err());
    }

    #[test]
    fn every_per_nonce_step_lives_inside_the_loop() {
        // The single highest-value structural property in this task, and every
        // way of getting it wrong is silent:
        //
        //   * a CPU fuel/signature reset hoisted above the loop carries nonce
        //     k-1's spend into nonce k, and the bundle dies partway through
        //     with an out-of-fuel exit that names nothing;
        //   * `initialize_kernel` hoisted above the loop does the same to the
        //     device's gbl_FUELUSAGE;
        //   * `output_file` hoisted above the loop makes every nonce overwrite
        //     one file -- five nonces, one result, exit 0;
        //   * the save closure hoisted above the loop stamps every file in the
        //     bundle with the batch's first nonce. The files are well formed,
        //     the verifier is never asked about the mismatch, and the slave's
        //     Merkle root is silently wrong.
        //
        // The GPU test for the first two lives on tig-gpu and cannot run here.
        let code = gpu_common_code();
        let loop_start = offset_of(&code, concat!("for nonce in ", "nonces {"));
        for needle in [
            concat!("*fuel_remaining", "_ptr = max_fuel"),
            concat!("*runtime_signature", "_ptr ="),
            concat!("launch_builder(&initialize", "_kernel)"),
            concat!("output_dir.join(format!(\"{}", ".json\", nonce))"),
            concat!("let save_solution", "_fn = |solution:"),
        ] {
            let at = offset_of(&code, needle);
            assert!(
                at > loop_start,
                "`{}` must be inside the per-nonce loop (loop@{} found@{})",
                needle,
                loop_start,
                at
            );
        }
    }

    #[test]
    fn the_database_and_the_index_are_paid_once_above_the_loop() {
        // The whole point of the mode. If `$make_db` -- which generates the
        // database and uploads the index -- ends up inside the loop, the batch
        // still produces correct output and still exits 0; it is simply as slow
        // as one process per nonce, and the task has bought nothing.
        let code = gpu_common_code();
        let loop_start = offset_of(&code, concat!("for nonce in ", "nonces {"));
        let make_db = offset_of(&code, concat!("$make", "_db(&track,"));
        assert!(
            make_db < loop_start,
            "the database and the index upload must happen ONCE, above the loop \
             (make_db@{} loop@{})",
            make_db,
            loop_start
        );
    }

    #[test]
    fn the_gpu_device_is_picked_from_the_start_nonce() {
        // The context is created once, above the loop, so there is no per-nonce
        // device left to pick. `nonce % num_gpus` inside the loop would not
        // even compile against a context that already exists -- but written
        // above the loop with a stale outer `nonce` it compiles fine and
        // silently pins the whole bundle by the wrong index.
        let code = gpu_common_code();
        assert!(
            code.contains(concat!("start_nonce % num", "_gpus")),
            "the device must be chosen from start_nonce"
        );
        assert!(
            !code.contains(concat!("(nonce % num", "_gpus")),
            "the device must not be chosen from a per-nonce value"
        );
    }

    #[test]
    fn the_index_load_error_is_rendered_before_the_library_is_unloaded() {
        // Same dlclose/vtable trap as `build_index`: the algorithm's
        // `anyhow::Error` owns a vtable inside the `.so`, and a bare `?` drops
        // `library` while the error is still alive. Measured on tig-gpu as
        // `Runtime Error: ` followed by a SIGSEGV with rc 139.
        const SRC: &str = include_str!("main.rs");
        const CALL: &str = concat!("load_index", "_fn(&database,");
        assert_eq!(
            SRC.matches(CALL).count(),
            1,
            "expected exactly one call to load_index_fn"
        );
        let start = SRC.find(CALL).unwrap();
        let stmt = &SRC[start..];
        let end = stmt.find("?;").expect("the call must propagate with `?`");
        // The needle is the *formatting*, not `.map_err(`: a no-op
        // `.map_err(|e| e)` restores the segfault exactly while still
        // containing `.map_err(`, so that token discriminates nothing.
        assert!(
            stmt[..end].contains(concat!("anyhow!(\"{", ":#}\"")),
            "the algorithm's load_index error must be rendered to an owned string \
             while the library is still loaded"
        );
    }

    #[test]
    fn index_parses_on_both_the_legacy_form_and_the_batch_subcommand() {
        // `--index` is declared twice. A test that only exercises one of them
        // passes while the other is silently dropped.
        let m = cli()
            .try_get_matches_from(vec![
                "tig-runtime",
                "{}",
                "hash",
                "7",
                "lib.so",
                "--index",
                "/tmp/idx.blob",
            ])
            .expect("the legacy form must accept --index");
        assert_eq!(
            m.get_one::<PathBuf>("index"),
            Some(&PathBuf::from("/tmp/idx.blob"))
        );

        let m = cli()
            .try_get_matches_from(vec![
                "tig-runtime",
                "batch",
                "{}",
                "hash",
                "lib.so",
                "--start-nonce",
                "3",
                "--num-nonces",
                "5",
                "--index",
                "/tmp/idx.blob",
                "--output",
                "/tmp/out",
            ])
            .expect("batch must accept --index");
        let sub = m.subcommand_matches("batch").unwrap();
        assert_eq!(
            sub.get_one::<PathBuf>("index"),
            Some(&PathBuf::from("/tmp/idx.blob"))
        );
        assert_eq!(
            sub.get_one::<PathBuf>("output"),
            Some(&PathBuf::from("/tmp/out"))
        );
        assert_eq!(*sub.get_one::<u64>("start-nonce").unwrap(), 3);
        assert_eq!(*sub.get_one::<u64>("num-nonces").unwrap(), 5);
    }
}
