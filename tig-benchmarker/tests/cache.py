import unittest
import sys
import os
import threading
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tempfile

from common.cache import (
    benchmark_id_of, cache_paths, forget_gate, gate_for, has_cache, new_cache_gate, purge_cache_files, run_gated,
)


class TestCachePaths(unittest.TestCase):
    BATCH = {
        "id": "abc123_4",
        "benchmark_id": "abc123",
        "settings": {"challenge_id": "c004", "algorithm_id": "c004_a100"},
    }

    def test_paths_are_keyed_by_precommit_not_batch(self):
        challenge, algorithm = cache_paths("/r", self.BATCH)
        self.assertEqual(challenge, "/r/abc123_c004_challenge_cache.bin")
        self.assertEqual(algorithm, "/r/abc123_c004_a100_algorithm_cache.bin")
        other_batch = dict(self.BATCH, id="abc123_5")
        self.assertEqual(cache_paths("/r", other_batch), (challenge, algorithm))

    def test_only_listed_challenges_have_caches(self):
        self.assertTrue(has_cache(self.BATCH))
        self.assertFalse(has_cache(dict(self.BATCH, settings={"challenge_id": "c001", "algorithm_id": "x"})))

    def test_benchmark_id_of_strips_the_batch_index(self):
        self.assertEqual(benchmark_id_of("abc123_4"), "abc123")
        self.assertEqual(benchmark_id_of("with_underscores_12"), "with_underscores")

    def test_gate_is_shared_across_batches_of_one_precommit(self):
        try:
            self.assertIs(gate_for("p1"), gate_for("p1"))
            self.assertIsNot(gate_for("p1"), gate_for("p2"))
        finally:
            forget_gate("p1")
            forget_gate("p2")

    def test_purge_removes_only_that_precommits_cache_files(self):
        with tempfile.TemporaryDirectory() as d:
            mine = cache_paths(d, self.BATCH)
            theirs = cache_paths(d, dict(self.BATCH, benchmark_id="zzz"))
            for path in mine + theirs:
                open(path, "wb").close()
            open(f"{d}/abc123_4.json", "wb").close()
            removed = purge_cache_files(d, "abc123")
            self.assertEqual(sorted(removed), sorted(mine))
            for path in theirs:
                self.assertTrue(os.path.exists(path))
            self.assertTrue(os.path.exists(f"{d}/abc123_4.json"))
            self.assertEqual(purge_cache_files(d, "abc123"), [])


class TestCacheGate(unittest.TestCase):
    def test_no_gate_runs_immediately(self):
        self.assertEqual(run_gated(None, lambda: 42), 42)

    def test_only_one_worker_runs_until_the_first_success(self):
        gate = new_cache_gate()
        running = []
        overlap = []
        lock = threading.Lock()

        def run():
            with lock:
                running.append(1)
                if len(running) > 1 and not gate["ready"].is_set():
                    overlap.append(1)
            time.sleep(0.05)
            with lock:
                running.pop()
            return True

        threads = [threading.Thread(target=lambda: run_gated(gate, run)) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(overlap, [], "two workers ran while the cache was not ready")
        self.assertTrue(gate["ready"].is_set())

    def test_waiters_proceed_as_soon_as_the_cache_exists(self):
        # The builder writes the cache, then keeps solving for a while. The
        # other workers must start during that solve, not after it.
        gate = new_cache_gate()
        cache = threading.Event()
        builder_finished = threading.Event()
        started_before_builder_finished = []
        lock = threading.Lock()

        def builder():
            time.sleep(0.05)
            cache.set()             # caches are on disk
            time.sleep(0.4)         # ... but the builder's own solve continues
            builder_finished.set()
            return True

        def waiter():
            with lock:
                started_before_builder_finished.append(not builder_finished.is_set())
            return True

        first = threading.Thread(target=lambda: run_gated(gate, builder, cache.is_set))
        first.start()
        time.sleep(0.01)  # let the builder take the lock
        rest = [threading.Thread(target=lambda: run_gated(gate, waiter, cache.is_set)) for _ in range(4)]
        for t in rest:
            t.start()
        for t in rest:
            t.join()
        first.join()
        self.assertEqual(len(started_before_builder_finished), 4)
        self.assertTrue(all(started_before_builder_finished),
                        "a waiter waited for the builder's solve instead of the cache")

    def test_a_failed_first_run_lets_the_next_worker_build(self):
        gate = new_cache_gate()
        attempts = []

        def failing():
            attempts.append("fail")
            raise RuntimeError("boom")

        def ok():
            attempts.append("ok")
            return True

        with self.assertRaises(RuntimeError):
            run_gated(gate, failing, lambda: False)
        self.assertFalse(gate["ready"].is_set(), "a failure must not mark the cache ready")
        self.assertTrue(run_gated(gate, ok, lambda: False))
        self.assertTrue(gate["ready"].is_set())
        self.assertEqual(attempts, ["fail", "ok"])

    def test_a_concurrent_failure_hands_the_build_to_a_waiter(self):
        gate = new_cache_gate()
        order = []
        lock = threading.Lock()

        def failing():
            with lock:
                order.append("fail")
            time.sleep(0.05)
            raise RuntimeError("boom")

        def ok():
            with lock:
                order.append("ok")
            return True

        def guarded_fail():
            try:
                run_gated(gate, failing, lambda: False)
            except RuntimeError:
                pass

        first = threading.Thread(target=guarded_fail)
        first.start()
        time.sleep(0.01)
        second = threading.Thread(target=lambda: run_gated(gate, ok, lambda: False))
        second.start()
        first.join()
        second.join()
        self.assertEqual(order, ["fail", "ok"])
        self.assertTrue(gate["ready"].is_set())

    def test_after_ready_workers_do_not_serialise(self):
        gate = new_cache_gate()
        gate["ready"].set()
        started = threading.Barrier(3, timeout=2)

        def run():
            # Every worker must reach the barrier at once; if runs were
            # serialised the barrier would time out.
            started.wait()
            return True

        threads = [threading.Thread(target=lambda: run_gated(gate, run)) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertFalse(started.broken)


if __name__ == '__main__':
    unittest.main()
