import glob
import os
from threading import Condition, Event, Lock


# ---------------------------------------------------------------------------
# Where the caches live
# ---------------------------------------------------------------------------
#
# Both are keyed by the precommit (benchmark_id), because that is what pins
# the database seed: every batch of a precommit shares one database and one
# algorithm build. They sit at the results root, not in a batch directory, so
# they outlive any one batch and are purged once every batch of the precommit
# is gone (see purge_cache_files).

# Challenge ids whose runtime takes the two cache flags: those with a shared
# input (a `Database`) and the build_cache/load_cache export pair. Add a
# challenge here when its arm in tig-runtime becomes `cpu_cached` or
# `gpu_cached`.
CACHED_CHALLENGES = {"c004"}


def has_cache(batch):
    return batch["settings"]["challenge_id"] in CACHED_CHALLENGES


def cache_paths(results_dir, batch):
    """(challenge cache path, algorithm cache path) for a batch."""
    b = batch["benchmark_id"]
    s = batch["settings"]
    return (
        f"{results_dir}/{b}_{s['challenge_id']}_challenge_cache.bin",
        f"{results_dir}/{b}_{s['algorithm_id']}_algorithm_cache.bin",
    )


def benchmark_id_of(batch_id):
    """A batch id is `<benchmark_id>_<batch_idx>`."""
    return batch_id.rsplit("_", 1)[0]


def purge_cache_files(results_dir, benchmark_id):
    """Delete the precommit's cache files. Call only once no batch of it is
    pending, processing, ready or awaiting purge."""
    removed = []
    for path in glob.glob(f"{results_dir}/{glob.escape(benchmark_id)}_*_cache.bin"):
        try:
            os.remove(path)
            removed.append(path)
        except FileNotFoundError:
            pass
    return removed


# ---------------------------------------------------------------------------
# One builder per precommit
# ---------------------------------------------------------------------------

_GATES = {}
_GATES_LOCK = Lock()


def new_cache_gate():
    """State for one cache key: only one worker builds, the rest wait."""
    return {"ready": Event(), "lock": Lock(), "cv": Condition()}


def gate_for(key):
    """The gate shared by every batch with this key (the benchmark_id), so
    two batches of one precommit on the same slave build once, not twice."""
    with _GATES_LOCK:
        gate = _GATES.get(key)
        if gate is None:
            gate = _GATES[key] = new_cache_gate()
        return gate


def forget_gate(key):
    with _GATES_LOCK:
        _GATES.pop(key, None)


def run_gated(gate, run, cache_exists=None):
    """Run `run()` for one nonce so that only the first worker builds a
    missing cache and the rest proceed once it exists.

    The runtime writes its caches when the paths it is given are missing, so
    the first run to start is the build. While it runs, every other worker
    waits; they are released as soon as `cache_exists()` reports the files on
    disk, which the runtime writes before it starts its own solve, so they do
    not wait for the builder's solve as well. If the builder fails without
    producing the cache, the next worker to wake takes the lock and builds.
    Once the cache is known to exist nothing is gated any more.
    """
    if gate is None or gate["ready"].is_set():
        return run()
    while True:
        if cache_exists is not None and cache_exists():
            gate["ready"].set()
            return run()
        if gate["lock"].acquire(blocking=False):
            try:
                result = run()
                gate["ready"].set()
                return result
            finally:
                gate["lock"].release()
                with gate["cv"]:
                    gate["cv"].notify_all()
        with gate["cv"]:
            gate["cv"].wait(timeout=0.25)
        if gate["ready"].is_set():
            return run()
