import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from bench.config import Config
from bench.engines.base import EngineError, ModelRef
from bench.safety import Reading
from bench.store import ResultStore
from bench.suites import RunAborted, RunContext, key
from bench.suites import concurrency
from fake_engine import FakeEngine

CORPUS = " ".join(f"w{i}" for i in range(120000))
MIB = 1024 * 1024


class Guard:
    gpu_index = 0
    vram_total_mb = 24000

    def __init__(self, refuse_at=None, reset=False, reset_error=False, after_refuse_at=None):
        self.refuse_at = refuse_at
        self.after_refuse_at = after_refuse_at
        self.reset = reset
        self.reset_error = reset_error
        self.required = []
        self.after_calls = 0

    def before_step(self, required_mb, reading):
        self.required.append(required_mb)
        if self.refuse_at is not None and len(self.required) > self.refuse_at:
            return False, "not enough VRAM"
        return True, "ok"

    def after_step(self, prev, now):
        self.after_calls += 1
        if self.after_refuse_at is not None and self.after_calls > self.after_refuse_at:
            return False, "spill"
        return True, "ok"

    def driver_reset_since(self, ts):
        if self.reset_error:
            raise RuntimeError("probe broke")
        return self.reset


class ConcurrencyEngine(FakeEngine):
    """Fake engine that can hold requests until all of a level are in flight."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.lock = threading.Lock()
        self.call_lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.hang = None  # Event: requests wait on it
        self.barrier = None  # Barrier: requests wait until all of a level are in flight

    def complete(self, model, prompt, sampling):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            if self.barrier is not None:
                self.barrier.wait(5)
            if self.hang is not None:
                self.hang.wait(30)
            with self.call_lock:  # FakeEngine bookkeeping is not thread-safe
                return super().complete(model, prompt, sampling)
        finally:
            with self.lock:
                self.active -= 1


class ConcurrencyTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(concurrency, "take_reading",
                                    lambda idx, frac=None: Reading(20000, frac, None, None))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def ctx(self, engine=None, cfg=None, guard=None):
        model = ModelRef("m1", "S", "q4", None, None, "sha-1", 4 * 1024**3)
        store = ResultStore(Path(self.tmp.name) / "results.json")
        return RunContext(cfg or Config(), engine or FakeEngine(), model, store,
                          guard or Guard(), CORPUS, Path(self.tmp.name) / "raw")

    def run_levels(self, engine=None, levels=None, **kw):
        ctx = self.ctx(engine, **kw)
        if levels is None:
            concurrency.run_concurrency(ctx)
        else:
            concurrency.run_concurrency(ctx, levels)
        return ctx

    def entry(self, ctx, n):
        return ctx.store.get(key(ctx, "concurrency", f"p{n}"))

    def test_levels_and_keys(self):
        ctx = self.run_levels(ConcurrencyEngine())
        for n in (1, 2, 4, 8):
            entry = self.entry(ctx, n)
            self.assertEqual(entry["status"], "ok")
            values = entry["values"]
            for name in ("total_tps", "per_request_tps", "ttft_p50_s"):
                self.assertIsNotNone(values[name])
            self.assertEqual(len(values["requests"]), n)
            self.assertEqual(values["parallel"], n)

    def test_engine_reloaded_with_parallel(self):
        engine = ConcurrencyEngine()
        self.run_levels(engine)
        self.assertEqual(engine.parallels, [1, 2, 4, 8])
        loads = [c for c in engine.calls if c.startswith("load:")]
        self.assertEqual(loads, [f"load:{concurrency.SLOT_CTX}"] * 4)
        self.assertGreaterEqual(concurrency.SLOT_CTX, 8192 + concurrency.OUTPUT_TOKENS)
        self.assertEqual(engine.calls[-1], "unload")

    def test_requests_run_at_the_same_time(self):
        engine = ConcurrencyEngine()
        engine.barrier = threading.Barrier(4)
        ctx = self.run_levels(engine, [4])
        self.assertEqual(self.entry(ctx, 4)["status"], "ok")
        self.assertEqual(engine.peak, 4)

    def test_each_request_has_its_own_prompt_and_no_cache_hits(self):
        engine = ConcurrencyEngine()
        ctx = self.run_levels(engine)
        self.assertEqual(len(engine.prompts), 15)
        self.assertEqual(len(set(engine.prompts)), 15)
        self.assertEqual(len({p.split()[0] for p in engine.prompts}), 15)
        for n in (1, 2, 4, 8):
            values = self.entry(ctx, n)["values"]
            self.assertFalse(values["cache_contaminated"])
            self.assertEqual(values["cached_tokens"], 0)

    def test_cache_contamination_flagged(self):
        ctx = self.run_levels(ConcurrencyEngine(always_cached=2000), [2])
        values = self.entry(ctx, 2)["values"]
        self.assertTrue(values["cache_contaminated"])
        self.assertEqual(values["cached_tokens"], 4000)

    def test_throughput_numbers(self):
        engine = ConcurrencyEngine(gen_tps=(50.0, 100.0, 1000.0), ttft=0.4)
        ticks = iter([10.0, 14.0])  # start, end of the batch: 4 s
        with mock.patch.object(concurrency, "_clock", lambda: next(ticks)):
            ctx = self.run_levels(engine, [3])
        values = self.entry(ctx, 3)["values"]
        self.assertEqual(values["wall_s"], 4.0)
        self.assertEqual(values["total_tps"], 3 * 256 / 4.0)
        self.assertEqual(values["per_request_tps"], 100.0)  # median, the mean would be 383
        self.assertEqual(values["ttft_p50_s"], 0.4)
        self.assertEqual(values["gen_tokens"], 768)
        self.assertEqual(sorted(r["gen_tps"] for r in values["requests"]), [50.0, 100.0, 1000.0])

    def test_decode_tps_total_formula(self):
        requests = [{"gen_tokens": 200, "wall_s": 10.0, "ttft_s": 6.0},
                    {"gen_tokens": 100, "wall_s": 5.0, "ttft_s": 1.0},
                    {"gen_tokens": 50, "wall_s": 1.0, "ttft_s": 2.0}]  # no decode window: ignored
        self.assertEqual(concurrency._decode_tps_total(requests), 200 / 4.0 + 100 / 4.0)
        self.assertIsNone(concurrency._decode_tps_total(requests[2:]))

    def test_decode_tps_total_is_stored(self):
        ctx = self.run_levels(ConcurrencyEngine(), [2])
        self.assertIn("decode_tps_total", self.entry(ctx, 2)["values"])

    def test_foreign_ollama_ignores_parallel_so_higher_levels_are_skipped(self):
        engine = ConcurrencyEngine(name="ollama")
        engine.parallel = None  # a server given by URL
        ctx = self.run_levels(engine)
        self.assertEqual(self.entry(ctx, 1)["status"], "ok")
        for n in (2, 4, 8):
            entry = self.entry(ctx, n)
            self.assertEqual(entry["status"], "skipped")
            self.assertEqual(entry["values"]["reason"], "foreign Ollama server ignores parallel")
        self.assertEqual(engine.parallels, [1])

    def test_own_ollama_server_runs_all_levels(self):
        engine = ConcurrencyEngine(name="ollama")
        engine.parallel = 1
        ctx = self.run_levels(engine, [1, 2])
        self.assertEqual(self.entry(ctx, 2)["status"], "ok")

    def test_serialized_level_stays_ok_and_is_flagged(self):
        engine = ConcurrencyEngine()
        engine.meta = {"effective_parallel": 1, "effective_ctx": 10240}
        ctx = self.run_levels(engine, [1, 2, 4])
        self.assertEqual(self.entry(ctx, 1)["values"]["serialized"], False)
        for n in (2, 4):
            entry = self.entry(ctx, n)
            self.assertEqual(entry["status"], "ok")
            values = entry["values"]
            self.assertIs(values["serialized"], True)
            self.assertEqual(values["effective_parallel"], 1)
            self.assertEqual(values["effective_ctx"], 10240)
            self.assertEqual(values["serialized_note"],
                             f"engine ran {n} requests with parallel 1")
        self.assertEqual(engine.parallels, [1, 2, 4])  # the ladder is not stopped

    def test_effective_parallel_as_requested_is_not_serialized(self):
        engine = ConcurrencyEngine()
        engine.meta = {"effective_parallel": 2}
        ctx = self.run_levels(engine, [2])
        values = self.entry(ctx, 2)["values"]
        self.assertIs(values["serialized"], False)
        self.assertNotIn("serialized_note", values)

    def test_unknown_effective_parallel_is_not_judged(self):
        ctx = self.run_levels(ConcurrencyEngine(), [2])
        values = self.entry(ctx, 2)["values"]
        self.assertIsNone(values["effective_parallel"])
        self.assertIsNone(values["serialized"])

    def test_guard_stops_levels(self):
        engine = ConcurrencyEngine()
        guard = Guard(refuse_at=2)
        ctx = self.run_levels(engine, guard=guard)
        self.assertEqual(self.entry(ctx, 1)["status"], "ok")
        self.assertEqual(self.entry(ctx, 2)["status"], "ok")
        self.assertEqual(self.entry(ctx, 4)["status"], "does_not_fit")
        self.assertIn("VRAM", self.entry(ctx, 4)["values"]["reason"])
        skipped = self.entry(ctx, 8)
        self.assertEqual(skipped["status"], "skipped")
        self.assertIn("p4", skipped["values"]["reason"])
        self.assertEqual(engine.parallels, [1, 2])

    def test_guard_estimate_from_computed_curve(self):
        ctx = self.ctx()
        meta = {}
        ctx.store.save(key(ctx, "memory", "computed"), "ok",
                       {"weights_bytes": 4 * 1024**3, "kv_bytes_per_token": 1000}, meta)
        guard = Guard()
        ctx.guard = guard
        concurrency.run_concurrency(ctx, [1, 2])
        weights_mb = 4096
        kv = lambda n: int(1000 * concurrency.SLOT_CTX * n // MIB)  # noqa: E731
        self.assertEqual(guard.required, [weights_mb + kv(1), weights_mb + kv(2)])

    def test_guard_without_curve_checks_with_zero_and_notes_it(self):
        guard = Guard()
        ctx = self.run_levels(guard=guard, levels=[1])
        self.assertEqual(guard.required, [0])
        self.assertIn("no memory estimate", self.entry(ctx, 1)["values"]["guard_note"])

    def test_after_step_refusal_stops_higher_levels(self):
        ctx = self.run_levels(ConcurrencyEngine(), guard=Guard(after_refuse_at=1))
        self.assertEqual(self.entry(ctx, 1)["status"], "ok")
        self.assertEqual(self.entry(ctx, 2)["status"], "does_not_fit")
        self.assertEqual(self.entry(ctx, 4)["status"], "skipped")
        self.assertEqual(self.entry(ctx, 8)["status"], "skipped")

    def test_load_engine_error_is_does_not_fit_and_stops(self):
        class Failing(ConcurrencyEngine):
            def load(self, model, ctx, parallel=1):
                if parallel == 4:
                    raise EngineError("out of memory")
                return super().load(model, ctx, parallel)

        ctx = self.run_levels(Failing())
        self.assertEqual(self.entry(ctx, 2)["status"], "ok")
        self.assertEqual(self.entry(ctx, 4)["status"], "does_not_fit")
        self.assertEqual(self.entry(ctx, 8)["status"], "skipped")

    def test_missing_model_is_error_not_does_not_fit(self):
        class Missing(ConcurrencyEngine):
            def load(self, model, ctx, parallel=1):
                raise EngineError('ollama /api/generate: HTTP 404: {"error":"model \'qwen3.5:4b\' not found"}')

        ctx = self.run_levels(Missing(), [1, 2])
        self.assertEqual(self.entry(ctx, 1)["status"], "error")
        self.assertIsNone(self.entry(ctx, 2))

    def test_request_failure_does_not_hang_and_is_error(self):
        def fail(kind, index):
            if kind == "complete" and index == 1:
                return ValueError("boom")
        engine = ConcurrencyEngine(fail_on=fail)
        started = time.monotonic()
        ctx = self.run_levels(engine, levels=[1, 2, 4])
        self.assertLess(time.monotonic() - started, 10)
        self.assertEqual(self.entry(ctx, 1)["status"], "ok")
        self.assertEqual(self.entry(ctx, 2)["status"], "error")
        self.assertIn("boom", self.entry(ctx, 2)["values"]["error"])
        self.assertIsNone(self.entry(ctx, 4))  # error is retried, later levels wait
        self.assertEqual(engine.calls[-1], "unload")

    def test_engine_error_in_request_is_does_not_fit(self):
        def fail(kind, index):
            if kind == "complete" and index == 0:
                return EngineError("slot failed")
        ctx = self.run_levels(ConcurrencyEngine(fail_on=fail), [2, 4])
        self.assertEqual(self.entry(ctx, 2)["status"], "does_not_fit")
        self.assertEqual(self.entry(ctx, 4)["status"], "skipped")

    def test_unknown_reset_check_after_request_failure_is_does_not_fit(self):
        def fail(kind, index):
            if kind == "complete":
                return ValueError("died")
        ctx = self.run_levels(ConcurrencyEngine(fail_on=fail), [2], guard=Guard(reset_error=True))
        entry = self.entry(ctx, 2)
        self.assertEqual(entry["status"], "does_not_fit")
        self.assertEqual(entry["values"]["driver_reset_check"], "unknown")

    def test_driver_reset_aborts_and_raises(self):
        engine = ConcurrencyEngine()
        ctx = self.ctx(engine, guard=Guard(reset=True))
        with self.assertRaises(RunAborted):
            concurrency.run_concurrency(ctx)
        self.assertEqual(self.entry(ctx, 1)["status"], "aborted")
        self.assertIsNone(self.entry(ctx, 2))
        self.assertEqual(engine.calls[-1], "unload")

    def test_aborted_level_not_rerun_automatically(self):
        ctx = self.ctx(ConcurrencyEngine())
        ctx.store.save(key(ctx, "concurrency", "p1"), "aborted", {"reason": "reset"})
        concurrency.run_concurrency(ctx, [1])
        self.assertEqual(self.entry(ctx, 1)["status"], "aborted")
        self.assertEqual(ctx.engine.calls, [])
        ctx.cfg.force = True
        concurrency.run_concurrency(ctx, [1])
        self.assertEqual(self.entry(ctx, 1)["status"], "ok")

    def test_unknown_reset_check_after_success_keeps_ok_and_stops(self):
        ctx = self.run_levels(ConcurrencyEngine(), guard=Guard(reset_error=True))
        entry = self.entry(ctx, 1)
        self.assertEqual(entry["status"], "ok")
        self.assertEqual(entry["values"]["driver_reset_check"], "unknown")
        self.assertEqual(self.entry(ctx, 2)["status"], "skipped")

    def test_final_levels_are_not_repeated(self):
        engine = ConcurrencyEngine()
        ctx = self.run_levels(engine)
        count = engine.complete_count
        concurrency.run_concurrency(ctx)
        self.assertEqual(engine.complete_count, count)

    def test_scope_warm_only_runs_nothing(self):
        engine = ConcurrencyEngine()
        self.run_levels(engine, cfg=Config(scope="warm-only"))
        self.assertEqual(engine.calls, [])

    def test_hung_requests_are_bounded(self):
        engine = ConcurrencyEngine()
        engine.hang = threading.Event()
        self.addCleanup(engine.hang.set)
        with mock.patch.object(concurrency, "batch_timeout", lambda e: 0.3):
            ctx = self.run_levels(engine, [2])
        entry = self.entry(ctx, 2)
        self.assertEqual(entry["status"], "does_not_fit")
        self.assertIn("did not finish", entry["values"]["reason"])
        self.assertEqual(engine.calls[-1], "unload")


if __name__ == "__main__":
    unittest.main()
