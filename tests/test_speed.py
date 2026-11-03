import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bench.config import Config
from bench.engines.base import EngineError, ModelRef
from bench.safety import Reading
from bench.store import ResultStore
from bench.suites import RunAborted, RunContext, key
from bench.suites import speed
from fake_engine import FakeEngine

CORPUS = " ".join(f"w{i}" for i in range(60000))


class Guard:
    gpu_index = 0
    vram_total_mb = 24000

    def __init__(self, reset=False):
        self.reset = reset

    def driver_reset_since(self, ts):
        return self.reset

    def before_step(self, required_mb, reading):
        return (self.refuse is None, self.refuse or "ok")

    refuse = None


def make_ctx(tmp, engine, cfg=None, guard=None):
    model = ModelRef("m1", "S", "q4", None, None, "sha-1", 1000)
    store = ResultStore(Path(tmp) / "results.json")
    return RunContext(cfg or Config(), engine, model, store, guard or Guard(), CORPUS,
                      Path(tmp) / "raw")


class SpeedTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(speed, "take_reading",
                                    lambda idx, frac=None: Reading(20000, frac, None, None))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def run_speed(self, engine, **kw):
        ctx = make_ctx(self.tmp.name, engine, **kw)
        speed.run_speed(ctx)
        return ctx

    def test_speed_median_differs_from_mean(self):
        ctx = self.run_speed(FakeEngine(gen_tps=(90, 100, 200)))
        entry = ctx.store.get(key(ctx, "speed", "warm"))
        self.assertEqual(entry["values"]["gen_tps"], 100)

    def test_cold_starts_with_unload_before_every_load(self):
        engine = FakeEngine()
        self.run_speed(engine)
        self.assertEqual(engine.calls[0], "unload")
        for i, call in enumerate(engine.calls[:12]):
            if call.startswith("load:"):
                self.assertEqual(engine.calls[i - 1], "unload")

    def test_cold_cache_contamination_flagged(self):
        ctx = self.run_speed(FakeEngine(always_cached=400))
        self.assertTrue(ctx.store.get(key(ctx, "speed", "cold"))["values"]["cache_contaminated"])

    def test_failing_step_with_driver_reset_aborts(self):
        def fail(kind, index):
            if kind == "complete":
                return EngineError("connection refused")
        ctx = make_ctx(self.tmp.name, FakeEngine(fail_on=fail), guard=Guard(reset=True))
        with self.assertRaises(RunAborted):
            speed.run_speed(ctx)
        entry = ctx.store.get(key(ctx, "speed", "cold"))
        self.assertEqual(entry["status"], "aborted")

    def test_failing_step_records_unknown_reset_check(self):
        def fail(kind, index):
            if kind == "complete" and index == 0:
                return EngineError("boom")
        ctx = make_ctx(self.tmp.name, FakeEngine(fail_on=fail), guard=Guard(reset=None))
        speed.run_speed(ctx)
        entry = ctx.store.get(key(ctx, "speed", "cold"))
        self.assertEqual(entry["status"], "error")
        self.assertEqual(entry["values"]["driver_reset_check"], "unknown")

    def test_aborted_step_is_not_rerun_automatically(self):
        engine = FakeEngine()
        ctx = make_ctx(self.tmp.name, engine)
        ctx.store.save(key(ctx, "speed", "cold"), "aborted", {"reason": "reset"})
        speed.run_speed(ctx)
        self.assertEqual(ctx.store.get(key(ctx, "speed", "cold"))["status"], "aborted")
        self.assertEqual(ctx.store.get(key(ctx, "speed", "warm"))["status"], "ok")
        ctx.cfg.force_tests = ["*/speed/cold"]
        speed.run_speed(ctx)
        self.assertEqual(ctx.store.get(key(ctx, "speed", "cold"))["status"], "ok")

    def test_rerun_uses_fresh_prompts(self):
        engine = FakeEngine()
        ctx = self.run_speed(engine)
        first = set(engine.prompts)
        engine.prompts.clear()
        ctx.cfg.force = True
        speed.run_speed(ctx)
        self.assertFalse(first & set(engine.prompts))

    def test_speed_median(self):
        ctx = self.run_speed(FakeEngine(gen_tps=(100, 120, 110)))
        entry = ctx.store.get(key(ctx, "speed", "warm"))
        self.assertEqual(entry["status"], "ok")
        self.assertEqual(entry["values"]["gen_tps"], 110)
        self.assertEqual(len(entry["values"]["runs"]), 3)

    def test_meta_recorded(self):
        ctx = self.run_speed(FakeEngine())
        meta = ctx.store.get(key(ctx, "speed", "warm"))["meta"]
        self.assertEqual(meta["engine_version"], "fake-1")
        self.assertEqual(meta["model_digest"], "sha-1")
        self.assertIn("bench_version", meta)

    def test_speed_idempotent(self):
        engine = FakeEngine()
        ctx = self.run_speed(engine)
        first = engine.complete_count
        self.assertGreater(first, 0)
        speed.run_speed(ctx)
        self.assertEqual(engine.complete_count, first)

    def test_speed_force_pattern(self):
        engine = FakeEngine()
        ctx = self.run_speed(engine)
        before = engine.complete_count
        ctx.cfg.force_tests = ["*/speed/warm"]
        speed.run_speed(ctx)
        self.assertEqual(engine.complete_count - before, 3)

    def test_scope_warm_only(self):
        engine = FakeEngine()
        cfg = Config(scope="warm-only")
        ctx = self.run_speed(engine, cfg=cfg)
        keys = sorted(ctx.store.data["tests"])
        self.assertEqual(keys, [key(ctx, "speed", "warm")])
        self.assertNotIn("unload", engine.calls)

    def test_cold_unloads_first_and_includes_load(self):
        engine = FakeEngine(load_s=3.0)
        ctx = self.run_speed(engine)
        values = ctx.store.get(key(ctx, "speed", "cold"))["values"]
        self.assertEqual(values["load_s"], 3.0)
        self.assertIn("ttft_s", values)
        cold_start = engine.calls.index("unload")
        self.assertTrue(engine.calls[cold_start + 1].startswith("load:"))

    def test_prompts_are_distinct_except_cache_pair(self):
        engine = FakeEngine()
        self.run_speed(engine)
        counts = {}
        for prompt in engine.prompts:
            counts[prompt] = counts.get(prompt, 0) + 1
        repeated = [c for c in counts.values() if c > 1]
        self.assertEqual(repeated, [2, 2, 2])  # exactly the three cache pairs

    def test_cache_step_reports_second_request(self):
        engine = FakeEngine(ttft=1.0)
        ctx = self.run_speed(engine)
        values = ctx.store.get(key(ctx, "speed", "cache"))["values"]
        self.assertAlmostEqual(values["ttft_s"], 0.1)
        self.assertGreater(values["cached_tokens"], 0)
        self.assertTrue(values["cache_hit"])

    def test_prompt_sizes(self):
        engine = FakeEngine()
        self.run_speed(engine)
        sizes = sorted({len(p.split()) for p in engine.prompts})
        self.assertTrue(any(500 <= s <= 512 for s in sizes))
        self.assertTrue(any(8000 <= s <= 8192 for s in sizes))

    def test_error_is_stored_scrubbed_and_run_continues(self):
        def fail(kind, index):
            if kind == "complete" and index == 0:
                return EngineError(r"boom at C:\Users\michael\x\file.gguf")
        engine = FakeEngine(fail_on=fail)
        ctx = self.run_speed(engine)
        entry = ctx.store.get(key(ctx, "speed", "cold"))
        self.assertEqual(entry["status"], "error")
        self.assertNotIn("michael", entry["values"]["error"])
        self.assertEqual(ctx.store.get(key(ctx, "speed", "warm"))["status"], "ok")

    def test_llama_bench_only_for_llamacpp(self):
        ctx = self.run_speed(FakeEngine())
        self.assertIsNone(ctx.store.get(key(ctx, "speed", "llama-bench")))

    BENCH = Path("/x/llama-bench")
    ROWS = [{"n_prompt": 512, "n_gen": 0, "avg_ts": 900.0},
            {"n_prompt": 8192, "n_gen": 0, "avg_ts": 700.0},
            {"n_prompt": 0, "n_gen": 256, "avg_ts": 80.0}]

    def bench_ctx(self, suites=("speed",), guard=None):
        cfg = Config(llama_bench=self.BENCH, suites=list(suites))
        return make_ctx(self.tmp.name, FakeEngine(name="llamacpp"), cfg=cfg, guard=guard)

    def fake_bench(self, calls):
        def run(binary, model, args, **kw):
            calls.append(args)
            if "-d" in args:
                depth = int(args[args.index("-d") + 1])
                return [{"n_prompt": 512, "n_gen": 0, "n_depth": depth, "avg_ts": 300.0},
                        {"n_prompt": 0, "n_gen": 256, "n_depth": depth, "avg_ts": 50.0}]
            return self.ROWS
        return run

    def test_llama_bench_runs_for_llamacpp(self):
        ctx = self.bench_ctx()
        calls = []
        with mock.patch.object(speed, "run_llama_bench", side_effect=self.fake_bench(calls)):
            speed.run_speed(ctx)
        self.assertEqual(len(calls), 1)  # no memory ladder: deep stays pending
        values = ctx.store.get(key(ctx, "speed", "llama-bench"))["values"]
        self.assertEqual(values["pp512_tps"], 900.0)
        self.assertEqual(values["pp8192_tps"], 700.0)
        self.assertEqual(values["tg256_tps"], 80.0)
        self.assertNotIn("deep", values)
        self.assertIsNone(ctx.store.get(key(ctx, "speed", "llama-bench-deep")))

    def test_deep_pending_then_filled_after_memory(self):
        ctx = self.bench_ctx(suites=("speed", "memory"))
        calls = []
        with mock.patch.object(speed, "run_llama_bench", side_effect=self.fake_bench(calls)):
            speed.run_speed(ctx)
            self.assertIsNone(ctx.store.get(key(ctx, "speed", "llama-bench-deep")))
            ctx.store.save(key(ctx, "memory", "ctx-8192"), "ok", {"vram_used_mb": 5000})
            speed.run_speed(ctx)
        self.assertEqual(len(calls), 2)  # llama-bench ran once, deep once
        deep = ctx.store.get(key(ctx, "speed", "llama-bench-deep"))
        self.assertEqual(deep["status"], "ok")
        self.assertEqual(deep["values"]["depth"], 8192)

    def test_deep_pending_when_ladder_has_error(self):
        ctx = self.bench_ctx()
        ctx.store.save(key(ctx, "memory", "ctx-8192"), "error", {"error": "x"})
        with mock.patch.object(speed, "run_llama_bench", side_effect=self.fake_bench([])):
            speed.run_speed(ctx)
        self.assertIsNone(ctx.store.get(key(ctx, "speed", "llama-bench-deep")))

    def test_deep_skipped_when_no_step_fits(self):
        ctx = self.bench_ctx()
        ctx.store.save(key(ctx, "memory", "ctx-8192"), "does_not_fit", {})
        ctx.store.save(key(ctx, "memory", "ctx-16384"), "skipped", {})
        with mock.patch.object(speed, "run_llama_bench", side_effect=self.fake_bench([])):
            speed.run_speed(ctx)
        deep = ctx.store.get(key(ctx, "speed", "llama-bench-deep"))
        self.assertEqual(deep["status"], "skipped")
        self.assertEqual(deep["values"]["reason"], "no context step fits")

    def test_deep_refused_by_guard(self):
        guard = Guard()
        guard.refuse = "low VRAM"
        ctx = self.bench_ctx(guard=guard)
        ctx.store.save(key(ctx, "memory", "ctx-8192"), "ok", {"vram_used_mb": 5000})
        calls = []
        with mock.patch.object(speed, "run_llama_bench", side_effect=self.fake_bench(calls)):
            speed.run_speed(ctx)
        self.assertEqual(len(calls), 1)
        deep = ctx.store.get(key(ctx, "speed", "llama-bench-deep"))
        self.assertEqual((deep["status"], deep["values"]["reason"]), ("skipped", "low VRAM"))

    def test_deep_without_vram_reading_uses_computed_estimate(self):
        guard = Guard()
        required = []
        guard.before_step = lambda mb, reading: required.append(mb) or (True, "ok")
        ctx = self.bench_ctx(guard=guard)
        ctx.store.save(key(ctx, "memory", "ctx-8192"), "ok", {"vram_used_mb": None})
        ctx.store.save(key(ctx, "memory", "computed"), "ok",
                       {"curve": [{"ctx": 8192, "bytes": 6 * 1024 * 1024 * 1024}]})
        with mock.patch.object(speed, "run_llama_bench", side_effect=self.fake_bench([])):
            speed.run_speed(ctx)
        self.assertEqual(required, [6144])

    def test_deep_skipped_without_any_memory_figure(self):
        ctx = self.bench_ctx()
        ctx.store.save(key(ctx, "memory", "ctx-8192"), "ok", {})
        calls = []
        with mock.patch.object(speed, "run_llama_bench", side_effect=self.fake_bench(calls)):
            speed.run_speed(ctx)
        self.assertEqual(len(calls), 1)
        deep = ctx.store.get(key(ctx, "speed", "llama-bench-deep"))
        self.assertEqual((deep["status"], deep["values"]["reason"]),
                         ("skipped", "no memory figure for depth 8192"))

    def test_deep_uses_largest_ok_memory_step(self):
        ctx = self.bench_ctx()
        for n in (8192, 16384):
            ctx.store.save(key(ctx, "memory", f"ctx-{n}"), "ok", {"vram_used_mb": 1000})
        ctx.store.save(key(ctx, "memory", "ctx-32768"), "does_not_fit", {})
        calls = []
        with mock.patch.object(speed, "run_llama_bench", side_effect=self.fake_bench(calls)):
            speed.run_speed(ctx)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1][calls[1].index("-d") + 1], "16384")
        deep = ctx.store.get(key(ctx, "speed", "llama-bench-deep"))["values"]
        self.assertEqual(deep["depth"], 16384)
        self.assertEqual(deep["pp512_tps"], 300.0)
        self.assertEqual(deep["tg256_tps"], 50.0)

    def test_driver_reset_aborts_run(self):
        engine = FakeEngine()
        ctx = make_ctx(self.tmp.name, engine, guard=Guard(reset=True))
        with self.assertRaises(RunAborted):
            speed.run_speed(ctx)
        self.assertEqual(ctx.store.get(key(ctx, "speed", "cold"))["status"], "aborted")

    def test_unknown_reset_check_is_recorded(self):
        ctx = make_ctx(self.tmp.name, FakeEngine(), guard=Guard(reset=None))
        speed.run_speed(ctx)
        values = ctx.store.get(key(ctx, "speed", "warm"))["values"]
        self.assertEqual(values["driver_reset_check"], "unknown")


if __name__ == "__main__":
    unittest.main()
