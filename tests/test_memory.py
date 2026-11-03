import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bench.config import Config
from bench.engines.base import EngineError, ModelRef
from bench.safety import Reading
from bench.store import ResultStore
from bench.suites import RunAborted, RunContext, key
from bench.suites import memory
from fake_engine import FakeEngine
from gguf_writer import write_gguf

CORPUS = " ".join(f"w{i}" for i in range(80000))
MIB = 1024 * 1024
DENSE = {"general.architecture": "llama", "llama.block_count": 32,
         "llama.attention.head_count_kv": 8, "llama.attention.key_length": 128}
STEPS = [8192, 16384, 32768, 65536, 131072, 262144]


class Guard:
    gpu_index = 0
    vram_total_mb = 24000

    def __init__(self, refuse_at=None, reset=False):
        self.refuse_at = refuse_at
        self.reset = reset
        self.required = []

    def before_step(self, required_mb, reading):
        self.required.append(required_mb)
        if self.refuse_at is not None and len(self.required) > self.refuse_at:
            return False, "not enough VRAM"
        return True, "ok"

    def after_step(self, prev, now):
        return True, "ok"

    def driver_reset_since(self, ts):
        return self.reset


class MemoryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.gguf = write_gguf(Path(self.tmp.name) / "m.gguf", dict(DENSE))
        patcher = mock.patch.object(
            memory, "take_reading", lambda idx, frac=None: Reading(20000, frac, None, None))
        patcher.start()
        self.addCleanup(patcher.stop)

    def ctx(self, engine=None, cfg=None, guard=None, gguf=True):
        model = ModelRef("m1", "S", "q4", None, self.gguf if gguf else None, "sha-1", 4 * 1024**3)
        store = ResultStore(Path(self.tmp.name) / "results.json")
        cfg = cfg or Config(ctx_steps=list(STEPS))
        return RunContext(cfg, engine or FakeEngine(), model, store, guard or Guard(), CORPUS,
                          Path(self.tmp.name) / "raw")

    def test_memory_computed_without_engine(self):
        engine = FakeEngine()
        ctx = self.ctx(engine, cfg=Config(scope="no-context"))
        memory.run_memory(ctx)
        entry = ctx.store.get(key(ctx, "memory", "computed"))
        self.assertEqual(entry["status"], "ok")
        self.assertEqual(engine.calls, [])
        values = entry["values"]
        self.assertEqual(values["kv_bytes_per_token"], 8 * 256 * 2 * 32)
        self.assertEqual(len(values["curve"]), len(STEPS))
        self.assertEqual(values["curve"][0]["ctx"], 8192)
        self.assertEqual(values["curve"][0]["bytes"], 4 * 1024**3 + 8192 * values["kv_bytes_per_token"])
        self.assertFalse(values["sliding_window"])
        self.assertFalse(values["recurrent_state_ignored"])
        self.assertIsNone(ctx.store.get(key(ctx, "memory", "ctx-8192")))

    def test_computed_flags_upper_bound_for_sliding_window_and_recurrent(self):
        kv = dict(DENSE, **{"llama.attention.sliding_window": 4096, "llama.ssm.state_size": 128,
                            "llama.full_attention_interval": 4})
        write_gguf(self.gguf, kv)
        ctx = self.ctx(cfg=Config(scope="no-context"))
        memory.run_memory(ctx)
        values = ctx.store.get(key(ctx, "memory", "computed"))["values"]
        self.assertTrue(values["sliding_window"])
        self.assertTrue(values["recurrent_state_ignored"])
        self.assertIn("upper bound", values["note"])

    def test_computed_without_gguf_is_skipped(self):
        ctx = self.ctx(cfg=Config(scope="no-context"), gguf=False)
        memory.run_memory(ctx)
        entry = ctx.store.get(key(ctx, "memory", "computed"))
        self.assertEqual(entry["status"], "skipped")

    def test_ladder_runs_all_steps(self):
        engine = FakeEngine()
        ctx = self.ctx(engine)
        memory.run_memory(ctx)
        for n in STEPS:
            entry = ctx.store.get(key(ctx, "memory", f"ctx-{n}"))
            self.assertEqual(entry["status"], "ok", n)
        values = ctx.store.get(key(ctx, "memory", "ctx-8192"))["values"]
        self.assertEqual(values["vram_used_mb"], 4000)
        self.assertEqual(values["kv_buffer_mb"], 100.0)
        self.assertIn("gen_tps", values)
        self.assertEqual(values["driver_reset_check"], "none")
        self.assertEqual(engine.calls[-1], "unload")

    def test_rung_records_vram_baseline_after_unload(self):
        # free VRAM is 22000 MB after the unload and 20000 MB after the run: baseline 2000 MB
        readings = iter([Reading(22000, None, None, None), Reading(20000, None, None, None)] * len(STEPS))
        with mock.patch.object(memory, "take_reading", lambda idx, frac=None: next(readings)):
            ctx = self.ctx()
            memory.run_memory(ctx)
        values = ctx.store.get(key(ctx, "memory", "ctx-8192"))["values"]
        self.assertEqual(values["vram_baseline_mb"], 2000)
        self.assertEqual(values["vram_used_mb"], 4000)

    def test_refused_rung_keeps_the_baseline(self):
        ctx = self.ctx(guard=Guard(refuse_at=0))
        memory.run_memory(ctx)
        values = ctx.store.get(key(ctx, "memory", "ctx-8192"))["values"]
        self.assertEqual(values["vram_baseline_mb"], 4000)

    def test_memory_ladder_stops(self):
        # computed has no guard call; the guard sees one call per ladder step
        guard = Guard(refuse_at=3)  # steps 8k, 16k, 32k pass, 64k refused
        ctx = self.ctx(guard=guard)
        memory.run_memory(ctx)
        status = {n: ctx.store.get(key(ctx, "memory", f"ctx-{n}"))["status"] for n in STEPS}
        self.assertEqual(status, {8192: "ok", 16384: "ok", 32768: "ok", 65536: "does_not_fit",
                                  131072: "skipped", 262144: "skipped"})

    def test_load_failure_is_does_not_fit_and_stops(self):
        def fail(kind, arg):
            if kind == "load" and arg == 16384:
                return EngineError("out of memory")
        ctx = self.ctx(FakeEngine(fail_on=fail))
        memory.run_memory(ctx)
        self.assertEqual(ctx.store.get(key(ctx, "memory", "ctx-16384"))["status"], "does_not_fit")
        self.assertEqual(ctx.store.get(key(ctx, "memory", "ctx-32768"))["status"], "skipped")

    def test_refusal_after_another_process_grabbed_vram_is_a_retryable_error(self):
        readings = iter([23000, 23000, 17000, 17000, 17000, 17000])  # baseline 1000, then 7000
        patch = mock.patch.object(memory, "take_reading",
                                  lambda idx, frac=None: Reading(next(readings), frac, None, None))
        with patch:
            ctx = self.ctx(guard=Guard(refuse_at=1), cfg=Config(ctx_steps=[8192, 16384, 32768]))
            memory.run_memory(ctx)
        self.assertEqual(ctx.store.get(key(ctx, "memory", "ctx-8192"))["status"], "ok")
        entry = ctx.store.get(key(ctx, "memory", "ctx-16384"))
        self.assertEqual(entry["status"], "error")
        self.assertIn("VRAM taken by another process", entry["values"]["reason"])
        self.assertIsNone(ctx.store.get(key(ctx, "memory", "ctx-32768")))  # retried together

    def test_refusal_with_an_unchanged_baseline_is_still_does_not_fit(self):
        ctx = self.ctx(guard=Guard(refuse_at=1), cfg=Config(ctx_steps=[8192, 16384]))
        memory.run_memory(ctx)
        self.assertEqual(ctx.store.get(key(ctx, "memory", "ctx-16384"))["status"], "does_not_fit")

    def test_missing_model_is_error_not_does_not_fit(self):
        def fail(kind, arg):
            if kind == "load":
                return EngineError('ollama /api/generate: HTTP 404: {"error":"model \'qwen3.5:4b\' not found"}')
        ctx = self.ctx(FakeEngine(fail_on=fail))
        memory.run_memory(ctx)
        entry = ctx.store.get(key(ctx, "memory", "ctx-8192"))
        self.assertEqual(entry["status"], "error")
        self.assertIn("not found", entry["values"]["error"])
        self.assertIsNone(ctx.store.get(key(ctx, "memory", "ctx-16384")))  # retried later

    def test_unexpected_exception_is_error_and_stops(self):
        def fail(kind, arg):
            if kind == "complete" and arg == 1:
                return ValueError("bug")
        ctx = self.ctx(FakeEngine(fail_on=fail))
        memory.run_memory(ctx)
        self.assertEqual(ctx.store.get(key(ctx, "memory", "ctx-8192"))["status"], "ok")
        self.assertEqual(ctx.store.get(key(ctx, "memory", "ctx-16384"))["status"], "error")
        self.assertIsNone(ctx.store.get(key(ctx, "memory", "ctx-32768")))

    def test_crashed_step_blocks_rerun(self):
        ctx = self.ctx()
        ctx.store.save(key(ctx, "memory", "ctx-8192"), "ok", {})
        ctx.store.mark_running(key(ctx, "memory", "ctx-16384"), {})
        ctx.store.recover_crashed()
        engine = ctx.engine
        memory.run_memory(ctx)
        self.assertEqual(engine.complete_count, 0)
        self.assertEqual(ctx.store.get(key(ctx, "memory", "ctx-16384"))["status"], "does_not_fit")
        self.assertEqual(ctx.store.get(key(ctx, "memory", "ctx-32768"))["status"], "skipped")

    def test_failing_rung_with_driver_reset_aborts(self):
        def fail(kind, arg):
            if kind == "complete":
                return EngineError("connection reset")
        ctx = self.ctx(FakeEngine(fail_on=fail), guard=Guard(reset=True))
        with self.assertRaises(RunAborted):
            memory.run_memory(ctx)
        self.assertEqual(ctx.store.get(key(ctx, "memory", "ctx-8192"))["status"], "aborted")

    def test_failing_rung_with_unknown_reset_is_not_retried(self):
        def fail(kind, arg):
            if kind == "complete":
                return ValueError("weird")
        engine = FakeEngine(fail_on=fail)
        ctx = self.ctx(engine, guard=Guard(reset=None))
        memory.run_memory(ctx)
        entry = ctx.store.get(key(ctx, "memory", "ctx-8192"))
        self.assertEqual(entry["status"], "does_not_fit")
        self.assertEqual(entry["values"]["driver_reset_check"], "unknown")
        self.assertEqual(ctx.store.get(key(ctx, "memory", "ctx-16384"))["status"], "skipped")
        attempts = engine.attempts
        memory.run_memory(ctx)
        self.assertEqual(engine.attempts, attempts)

    def test_idempotent(self):
        engine = FakeEngine()
        ctx = self.ctx(engine)
        memory.run_memory(ctx)
        count = engine.complete_count
        memory.run_memory(ctx)
        self.assertEqual(engine.complete_count, count)

    def test_driver_reset_aborts_and_raises(self):
        ctx = self.ctx(guard=Guard(reset=True))
        with self.assertRaises(RunAborted):
            memory.run_memory(ctx)
        self.assertEqual(ctx.store.get(key(ctx, "memory", "ctx-8192"))["status"], "aborted")

    def test_unknown_reset_check_stops_climbing(self):
        ctx = self.ctx(guard=Guard(reset=None))
        memory.run_memory(ctx)
        first = ctx.store.get(key(ctx, "memory", "ctx-8192"))
        self.assertEqual(first["status"], "ok")
        self.assertEqual(first["values"]["driver_reset_check"], "unknown")
        self.assertEqual(ctx.store.get(key(ctx, "memory", "ctx-16384"))["status"], "skipped")

    def test_recurrent_state_recorded(self):
        engine = FakeEngine()
        engine.rs_mb = 75.0
        ctx = self.ctx(engine, cfg=Config(ctx_steps=[8192]))
        memory.run_memory(ctx)
        self.assertEqual(ctx.store.get(key(ctx, "memory", "ctx-8192"))["values"]["rs_buffer_mb"], 75.0)

    def test_scope_warm_only_skips_memory(self):
        engine = FakeEngine()
        ctx = self.ctx(engine, cfg=Config(scope="warm-only"))
        memory.run_memory(ctx)
        self.assertEqual(ctx.store.data["tests"], {})

    def test_force_pattern_reruns_one_step(self):
        engine = FakeEngine()
        ctx = self.ctx(engine)
        memory.run_memory(ctx)
        before = engine.complete_count
        ctx.cfg.force_tests = ["*/memory/ctx-16384"]
        memory.run_memory(ctx)
        self.assertEqual(engine.complete_count - before, 1)


if __name__ == "__main__":
    unittest.main()
