import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bench.config import Config
from bench.engines.base import ChatResult, EngineError, ModelRef, Timing
from bench.store import ResultStore
from bench.suites import RunAborted, RunContext, key
from bench.suites import smoke
from fake_engine import FakeEngine

CORPUS = " ".join(f"w{i}" for i in range(80000))
TIMING = Timing(None, 0.1, 10, 100.0, 5, 50.0, None)


class ChatEngine(FakeEngine):
    """FakeEngine that also answers chat requests; `reply(index, messages, ...)` decides."""

    def __init__(self, reply=None, **kw):
        super().__init__(**kw)
        self.reply = reply or (lambda i, m, tools, schema: ChatResult("hello", [], TIMING, {}))
        self.chats = []

    def chat(self, model, messages, sampling, tools=None, json_schema=None, thinking=False):
        index = len(self.chats)
        self.chats.append({"messages": messages, "sampling": sampling, "tools": tools,
                           "json_schema": json_schema, "thinking": thinking})
        if self.fail_on is not None:
            err = self.fail_on("chat", index)
            if err:
                raise err
        out = self.reply(index, messages, tools, json_schema)
        return out if isinstance(out, ChatResult) else ChatResult(out, [], TIMING, {})


class Guard:
    gpu_index = 0
    vram_total_mb = 24000

    def __init__(self, reset=False):
        self.reset = reset

    def driver_reset_since(self, ts):
        return self.reset


class SmokeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def ctx(self, engine=None, cfg=None, guard=None, model_id="qwen3:14b", rungs=()):
        model = ModelRef(model_id, "S", "q4", None, None, "sha-1", 4 * 1024**3)
        store = ResultStore(Path(self.tmp.name) / "results.json")
        ctx = RunContext(cfg or Config(), engine or ChatEngine(), model, store, guard or Guard(),
                         CORPUS, Path(self.tmp.name) / "raw")
        for n, status in rungs:
            store.save(key(ctx, "memory", f"ctx-{n}"), status, {"ctx": n})
        return ctx

    def smoke_keys(self, ctx):
        return {k: v for k, v in ctx.store.data["tests"].items() if "/smoke/" in k}

    def test_smoke_idempotent(self):
        engine = ChatEngine()
        ctx = self.ctx(engine, Config(no_code_exec=True), rungs=[(8192, "ok")])
        smoke.run_smoke(ctx)
        first = len(engine.chats)
        self.assertEqual(first, 25)
        stored = {k: v["finished_at"] for k, v in self.smoke_keys(ctx).items()}
        self.assertEqual(len(stored), 31)
        smoke.run_smoke(ctx)
        self.assertEqual(len(engine.chats), first)
        self.assertEqual({k: v["finished_at"] for k, v in self.smoke_keys(ctx).items()}, stored)

    def test_judged_tasks_are_ok_with_passed_flag(self):
        def reply(i, messages, tools, schema):
            text = messages[-1]["content"]
            if "average speed" in text:
                return "ANSWER: 75"
            return "I do not know"
        ctx = self.ctx(ChatEngine(reply), Config(tasks=["math-1", "math-2", "creative-1"]))
        smoke.run_smoke(ctx)
        good = ctx.store.get(key(ctx, "smoke", "math-1"))
        self.assertEqual(good["status"], "ok")
        self.assertIs(good["values"]["passed"], True)
        bad = ctx.store.get(key(ctx, "smoke", "math-2"))
        self.assertEqual(bad["status"], "ok")
        self.assertIs(bad["values"]["passed"], False)
        self.assertTrue(bad["values"]["reason"])
        free = ctx.store.get(key(ctx, "smoke", "creative-1"))
        self.assertEqual(free["status"], "ok")
        self.assertIsNone(free["values"]["passed"])
        self.assertIsNone(ctx.store.get(key(ctx, "smoke", "math-3")))

    def test_raw_answer_is_scrubbed_and_name_is_safe(self):
        ctx = self.ctx(ChatEngine(lambda *a: "see C:\\Users\\michael\\secret.txt ok"),
                       Config(tasks=["math-1"]))
        smoke.run_smoke(ctx)
        values = ctx.store.get(key(ctx, "smoke", "math-1"))["values"]
        self.assertEqual(values["raw"], "raw/fake-qwen3_14b-math-1.txt")
        text = (Path(self.tmp.name) / values["raw"]).read_text(encoding="utf-8")
        self.assertNotIn("michael", text)
        self.assertNotIn(":", values["raw"])

    def test_thinking_off_and_recorded(self):
        engine = ChatEngine()
        ctx = self.ctx(engine, Config(tasks=["math-1"]))
        smoke.run_smoke(ctx)
        self.assertIs(engine.chats[0]["thinking"], False)
        entry = ctx.store.get(key(ctx, "smoke", "math-1"))
        self.assertIs(entry["meta"]["thinking"], False)
        engine2 = ChatEngine()
        ctx2 = self.ctx(engine2, Config(tasks=["math-1"], thinking=True), model_id="other")
        smoke.run_smoke(ctx2)
        self.assertIs(engine2.chats[0]["thinking"], True)
        self.assertIs(ctx2.store.get(key(ctx2, "smoke", "math-1"))["meta"]["thinking"], True)

    def test_thinking_text_not_counted_as_answer(self):
        ctx = self.ctx(ChatEngine(lambda *a: "<think>ANSWER: 75</think>no idea"),
                       Config(tasks=["math-1"]))
        smoke.run_smoke(ctx)
        self.assertIs(ctx.store.get(key(ctx, "smoke", "math-1"))["values"]["passed"], False)

    def test_sampling_and_inputs_passed_to_engine(self):
        engine = ChatEngine()
        ctx = self.ctx(engine, Config(tasks=["creative-1", "json-1", "tool-1"]))
        smoke.run_smoke(ctx)
        by_tools = {bool(c["tools"]): c for c in engine.chats}
        self.assertEqual(by_tools[True]["tools"][0]["function"]["name"], "get_weather")
        creative = next(c for c in engine.chats if not c["tools"] and not c["json_schema"])
        self.assertEqual(creative["sampling"].temperature, 0.8)
        self.assertEqual(creative["sampling"].top_p, 0.95)
        self.assertEqual(creative["sampling"].seed, 42)
        self.assertTrue(any(c["json_schema"] for c in engine.chats))

    def test_code_exec_disabled_is_skipped(self):
        ctx = self.ctx(ChatEngine(), Config(no_code_exec=True, tasks=["code-1"]))
        smoke.run_smoke(ctx)
        entry = ctx.store.get(key(ctx, "smoke", "code-1"))
        self.assertEqual(entry["status"], "skipped")

    def test_code_task_passes_with_correct_function(self):
        good = "```python\nimport re\ndef is_palindrome(s):\n    t = re.sub('[^a-z0-9]', '', s.lower())\n    return t == t[::-1]\n```"
        ctx = self.ctx(ChatEngine(lambda *a: good), Config(tasks=["code-1"]))
        smoke.run_smoke(ctx)
        entry = ctx.store.get(key(ctx, "smoke", "code-1"))
        self.assertEqual((entry["status"], entry["values"]["passed"]), ("ok", True))

    def test_long_context_pending_without_memory_rung(self):
        engine = ChatEngine()
        ctx = self.ctx(engine, Config(tasks=["needle-*", "math-1"]))
        smoke.run_smoke(ctx)
        self.assertIsNone(ctx.store.get(key(ctx, "smoke", "needle-10")))
        self.assertEqual(ctx.store.get(key(ctx, "smoke", "math-1"))["status"], "ok")
        self.assertEqual(len(engine.chats), 1)

    def test_needle_tasks_are_skipped_when_the_scope_has_no_ladder(self):
        for scope in ("no-context", "warm-only"):
            with self.subTest(scope=scope):
                (Path(self.tmp.name) / "results.json").unlink(missing_ok=True)
                ctx = self.ctx(ChatEngine(), Config(tasks=["needle-*"], scope=scope))
                smoke.run_smoke(ctx)
                entry = ctx.store.get(key(ctx, "smoke", "needle-10"))
                self.assertEqual(entry["status"], "skipped")
                self.assertEqual(entry["values"]["reason"], f"not in scope {scope}")

    def test_scope_skipped_needles_run_in_a_later_full_scope_run(self):
        engine = ChatEngine(lambda i, m, t, s: "7391-QZ")
        ctx = self.ctx(engine, Config(tasks=["needle-10"], scope="no-context"))
        smoke.run_smoke(ctx)
        self.assertEqual(ctx.store.get(key(ctx, "smoke", "needle-10"))["status"], "skipped")
        ctx.cfg.scope = "full"
        smoke.run_smoke(ctx)  # still no ladder: pending, the old scope skip is not final
        ctx.store.save(key(ctx, "memory", "ctx-8192"), "ok", {"ctx": 8192})
        smoke.run_smoke(ctx)
        self.assertEqual(ctx.store.get(key(ctx, "smoke", "needle-10"))["status"], "ok")

    def test_a_real_skip_stays_final_in_every_scope(self):
        ctx = self.ctx(ChatEngine(), Config(tasks=["needle-10"]),
                       rungs=[(8192, "does_not_fit")])
        smoke.run_smoke(ctx)
        first = ctx.store.get(key(ctx, "smoke", "needle-10"))["finished_at"]
        ctx.store.save(key(ctx, "memory", "ctx-8192"), "ok", {"ctx": 8192})
        smoke.run_smoke(ctx)
        self.assertEqual(ctx.store.get(key(ctx, "smoke", "needle-10"))["status"], "skipped")
        self.assertEqual(ctx.store.get(key(ctx, "smoke", "needle-10"))["finished_at"], first)

    def test_long_context_pending_while_a_rung_is_retryable(self):
        ctx = self.ctx(ChatEngine(), Config(tasks=["needle-*"]),
                       rungs=[(8192, "does_not_fit"), (16384, "error")])
        smoke.run_smoke(ctx)
        self.assertIsNone(ctx.store.get(key(ctx, "smoke", "needle-10")))

    def test_long_context_skipped_when_nothing_fits(self):
        ctx = self.ctx(ChatEngine(), Config(tasks=["needle-*"]),
                       rungs=[(8192, "does_not_fit"), (16384, "skipped")])
        smoke.run_smoke(ctx)
        entry = ctx.store.get(key(ctx, "smoke", "needle-10"))
        self.assertEqual(entry["status"], "skipped")

    def test_long_context_length_is_min_of_32k_and_largest_ok_rung(self):
        def answer(i, messages, tools, schema):
            return "7391-QZ"
        engine = ChatEngine(answer)
        ctx = self.ctx(engine, Config(tasks=["needle-10"]),
                       rungs=[(8192, "ok"), (16384, "ok"), (32768, "does_not_fit")])
        smoke.run_smoke(ctx)
        entry = ctx.store.get(key(ctx, "smoke", "needle-10"))
        self.assertEqual(entry["status"], "ok")
        self.assertIs(entry["values"]["passed"], True)
        self.assertEqual(entry["values"]["context_tokens"], 16384)
        self.assertIn("load:16384", engine.calls)
        prompt = engine.chats[0]["messages"][-1]["content"]
        self.assertIn("7391-QZ", prompt)
        self.assertLessEqual(len(prompt.split()), 16384)
        self.assertGreater(len(prompt.split()), 14000)

    def test_long_context_capped_at_32k(self):
        engine = ChatEngine()
        ctx = self.ctx(engine, Config(tasks=["needle-50"]),
                       rungs=[(65536, "ok"), (131072, "ok")])
        smoke.run_smoke(ctx)
        self.assertEqual(ctx.store.get(key(ctx, "smoke", "needle-50"))["values"]["context_tokens"],
                         32768)

    def test_engine_error_is_error_and_retried(self):
        state = {"fail": True}

        def fail(kind, index):
            if kind == "chat" and state["fail"]:
                return EngineError("connection refused C:\\Users\\michael\\x")
        engine = ChatEngine(fail_on=fail)
        ctx = self.ctx(engine, Config(tasks=["math-1"]), guard=Guard(reset=False))
        smoke.run_smoke(ctx)
        entry = ctx.store.get(key(ctx, "smoke", "math-1"))
        self.assertEqual(entry["status"], "error")
        self.assertNotIn("michael", entry["values"]["error"])
        state["fail"] = False
        smoke.run_smoke(ctx)
        self.assertEqual(ctx.store.get(key(ctx, "smoke", "math-1"))["status"], "ok")

    def test_failure_with_driver_reset_aborts(self):
        def fail(kind, index):
            if kind == "chat":
                return EngineError("connection refused")
        ctx = self.ctx(ChatEngine(fail_on=fail), Config(tasks=["math-1", "math-2"]),
                       guard=Guard(reset=True))
        with self.assertRaises(RunAborted):
            smoke.run_smoke(ctx)
        self.assertEqual(ctx.store.get(key(ctx, "smoke", "math-1"))["status"], "aborted")
        self.assertIsNone(ctx.store.get(key(ctx, "smoke", "math-2")))
        # an aborted task is not repeated automatically
        ctx.guard = Guard(reset=False)
        smoke.run_smoke(ctx)
        self.assertEqual(ctx.store.get(key(ctx, "smoke", "math-1"))["status"], "aborted")

    def test_reset_after_success_aborts(self):
        ctx = self.ctx(ChatEngine(), Config(tasks=["math-1"]), guard=Guard(reset=True))
        with self.assertRaises(RunAborted):
            smoke.run_smoke(ctx)
        self.assertEqual(ctx.store.get(key(ctx, "smoke", "math-1"))["status"], "aborted")

    def test_unknown_reset_check_is_recorded(self):
        ctx = self.ctx(ChatEngine(), Config(tasks=["math-1"]), guard=Guard(reset=None))
        smoke.run_smoke(ctx)
        entry = ctx.store.get(key(ctx, "smoke", "math-1"))
        self.assertEqual(entry["status"], "ok")
        self.assertEqual(entry["values"]["driver_reset_check"], "unknown")

    def test_checker_exception_is_judged_not_an_engine_error(self):
        ctx = self.ctx(ChatEngine(), Config(tasks=["math-1"]))
        with mock.patch.object(smoke, "check", side_effect=RecursionError("deep")):
            smoke.run_smoke(ctx)
        entry = ctx.store.get(key(ctx, "smoke", "math-1"))
        self.assertEqual(entry["status"], "ok")
        self.assertIs(entry["values"]["passed"], False)
        self.assertEqual(entry["values"]["reason"], "checker error: RecursionError")

    def test_needle_text_gets_the_task_depth(self):
        seen = []
        real = smoke.needle_text

        def spy(text, n, depth, needle, count):
            seen.append(depth)
            return real(text, n, depth, needle, count)
        ctx = self.ctx(ChatEngine(), Config(tasks=["needle-*"]), rungs=[(8192, "ok")])
        with mock.patch.object(smoke, "needle_text", spy):
            smoke.run_smoke(ctx)
        self.assertEqual(seen, [0.1, 0.5, 0.9])

    def test_force_test_reruns(self):
        engine = ChatEngine()
        ctx = self.ctx(engine, Config(tasks=["math-1"]))
        smoke.run_smoke(ctx)
        ctx.cfg.force_tests = ["*/smoke/math-1"]
        smoke.run_smoke(ctx)
        self.assertEqual(len(engine.chats), 2)


if __name__ == "__main__":
    unittest.main()
