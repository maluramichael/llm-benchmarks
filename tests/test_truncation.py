"""Prompts that Ollama truncates silently must never be stored as valid measurements.

The fake server behaves like Ollama 0.40.13: a prompt with at least num_ctx tokens is cut to
num_ctx/2+2 tokens (the head is kept for 4 tokens, then the tail), a WARN line goes to the
server log and prompt_eval_count is the reduced number. `inflate` makes the real token count
of measured prompts larger than the estimate from the calibration text.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bench.config import Config
from bench.engines.base import ModelRef
from bench.engines.ollama import OllamaEngine
from bench.safety import Reading
from bench.store import ResultStore
from bench.suites import RunContext, key, retry_truncated
from bench.suites import memory, smoke, speed
from bench.engines.base import PromptTruncated
from fake_http import FakeServer
from test_cache_free import CORPUS, Guard


class TruncatingOllama:
    def __init__(self, log_path, inflate=1.0, always=False, write_log=True):
        self.log_path = log_path
        self.inflate = inflate
        self.always = always
        self.write_log = write_log
        self.ctx = 0
        self.truncations = 0
        self.sent: list[int] = []  # real token counts of the prompts that were not truncated
        self.chat_prompts: list[str] = []

    def routes(self):
        return {"/api/version": {"version": "0.40.13"}, "/api/ps": {"models": []},
                "/api/generate": self.generate, "/api/chat": self.chat}

    def real(self, text):
        words = len(text.split())
        return words if text.startswith("Calibration") else int(words * self.inflate)

    def cut(self, real, options):
        """Reduced count when Ollama would truncate, else None."""
        if not (real >= self.ctx or (self.always and real > 1500)):
            return None
        self.truncations += 1
        limit = self.ctx // 2 + 2
        if self.write_log:
            with open(self.log_path, "a", encoding="utf-8") as handle:
                handle.write(f'level=WARN source=llama_server.go:344 msg="truncating input '
                             f'prompt" limit={limit} prompt={real} keep=4 new={limit}\n')
        return limit

    def generate(self, body):
        options = body.get("options") or {}
        if body.get("keep_alive") == 0:
            return {"done": True}
        self.ctx = options.get("num_ctx", self.ctx)
        prompt = body.get("prompt", "")
        if prompt == "":
            return {"done": True, "load_duration": 10**9}
        real = self.real(prompt)
        kept = None if prompt.startswith("Calibration") else self.cut(real, options)
        if kept is None:
            kept = real
            if body.get("stream"):  # measured requests only, not exact counts of short texts
                self.sent.append(real)
        reply = {"done": True, "response": "x", "prompt_eval_count": kept,
                 "prompt_eval_cached_count": 0, "prompt_eval_duration": 10**9,
                 "eval_count": 5, "eval_duration": 10**8}
        return json.dumps(reply) + "\n" if body.get("stream") else reply

    def chat(self, body):
        text = " ".join(m["content"] for m in body["messages"])
        self.chat_prompts.append(text)
        real = self.real(text) + 20  # template overhead
        kept = self.cut(real, body.get("options") or {})
        if kept is None:
            kept = real
            self.sent.append(real)
        return {"done": True, "message": {"role": "assistant", "content": "x"},
                "prompt_eval_count": kept, "eval_count": 1, "eval_duration": 10**8}


class TruncationSuiteTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        for module in (speed, memory):
            patcher = mock.patch.object(
                module, "take_reading", lambda idx, frac=None: Reading(20000, frac, None, None))
            patcher.start()
            self.addCleanup(patcher.stop)

    def start(self, **fake_kw):
        log = Path(self.tmp.name) / "ollama.log"
        log.write_text("", encoding="utf-8")
        self.fake = TruncatingOllama(log, **fake_kw)
        server = FakeServer(self.fake.routes())
        server.__enter__()
        self.addCleanup(server.__exit__)
        engine = OllamaEngine(Config(ollama_url=server.url), log_dir=Path(self.tmp.name))
        self.addCleanup(engine.stop)
        engine.start()
        engine._log_path = log  # as if we had started (and owned) the server
        return engine

    def ctx(self, engine, **cfg):
        model = ModelRef("S", "S", "q4", "qwen3.5:4b", None, "sha", 4 * 1024**3)
        store = ResultStore(Path(self.tmp.name) / "results.json")
        return RunContext(Config(**cfg), engine, model, store, Guard(), CORPUS,
                          Path(self.tmp.name) / "raw")

    # -- memory ladder ---------------------------------------------------------------

    def test_ladder_rung_is_retried_once_and_then_ok(self):
        ctx = self.ctx(self.start(inflate=1.15), ctx_steps=[4096, 8192])
        memory.run_memory(ctx)
        self.assertEqual(self.fake.truncations, 2)  # once per rung
        for n in (4096, 8192):
            entry = ctx.store.get(key(ctx, "memory", f"ctx-{n}"))
            self.assertEqual(entry["status"], "ok", entry["values"])
            target = memory._fill_tokens(n, exact=False)
            self.assertGreaterEqual(entry["values"]["prompt_tokens"], 0.9 * target)
            self.assertLess(entry["values"]["prompt_tokens"], n)

    def test_without_a_log_the_fallback_check_still_retries(self):
        ctx = self.ctx(self.start(inflate=1.10, write_log=False), ctx_steps=[4096])
        memory.run_memory(ctx)
        entry = ctx.store.get(key(ctx, "memory", "ctx-4096"))
        self.assertEqual(entry["status"], "ok", entry["values"])
        self.assertGreaterEqual(entry["values"]["prompt_tokens"], 0.9 * 0.94 * 4096 * 0.9)

    def test_double_truncation_is_a_retryable_error(self):
        ctx = self.ctx(self.start(always=True), ctx_steps=[4096, 8192])
        memory.run_memory(ctx)
        entry = ctx.store.get(key(ctx, "memory", "ctx-4096"))
        self.assertEqual(entry["status"], "error")  # retryable, not does_not_fit, not ok
        self.assertEqual(entry["values"]["reason"], "prompt truncated by the engine")
        self.assertEqual(self.fake.truncations, 2)
        self.assertIsNone(ctx.store.get(key(ctx, "memory", "ctx-8192")))  # ladder stopped

    # -- speed -----------------------------------------------------------------------

    def test_warm_step_is_retried_and_double_truncation_is_an_error(self):
        ctx = self.ctx(self.start(inflate=1.30), reps=1, scope="warm-only")
        speed.run_speed(ctx)
        warm = ctx.store.get(key(ctx, "speed", "warm"))
        self.assertEqual(warm["status"], "ok", warm["values"])
        self.assertGreaterEqual(warm["values"]["runs"][0]["prompt_tokens"], 0.9 * 8192)
        ctx = self.ctx(self.start(always=True), reps=1, scope="warm-only")
        ctx.store = ResultStore(Path(self.tmp.name) / "again.json")
        speed.run_speed(ctx)
        self.assertEqual(ctx.store.get(key(ctx, "speed", "warm"))["status"], "error")

    # -- smoke needles ---------------------------------------------------------------

    def run_needles(self, engine):
        ctx = self.ctx(engine, tasks=["needle-*"])
        ctx.store.save(key(ctx, "memory", "ctx-8192"), "ok", {"ctx": 8192})
        smoke.run_smoke(ctx)
        return ctx

    def test_needle_is_retried_with_a_corrected_document(self):
        ctx = self.run_needles(self.start(inflate=1.25))
        self.assertGreaterEqual(self.fake.truncations, 1)
        for task in ("needle-10", "needle-50", "needle-90"):
            entry = ctx.store.get(key(ctx, "smoke", task))
            self.assertEqual(entry["status"], "ok", entry["values"])
        # the last request of every needle task fitted the context
        self.assertTrue(self.fake.sent)
        self.assertTrue(all(n < 8192 for n in self.fake.sent))

    def test_needle_is_not_judged_on_a_truncated_prompt(self):
        ctx = self.run_needles(self.start(always=True))
        for task in ("needle-10", "needle-50", "needle-90"):
            entry = ctx.store.get(key(ctx, "smoke", task))
            self.assertEqual(entry["status"], "error")
            self.assertNotIn("passed", entry["values"])
        self.assertFalse(self.fake.sent)

    def test_needle_document_leaves_room_for_template_and_answer(self):
        # exact engine: document + 512 + max_tokens (64) fits the loaded context
        from test_smoke import ChatEngine
        engine = ChatEngine()
        ctx = self.ctx(engine, tasks=["needle-10"])
        ctx.store.save(key(ctx, "memory", "ctx-8192"), "ok", {"ctx": 8192})
        smoke.run_smoke(ctx)
        words = len(engine.chats[0]["messages"][-1]["content"].split())
        question = 30  # the task text around the document
        self.assertLessEqual(words - question + 512 + 64, 8192)


class RetryHelperTest(unittest.TestCase):
    def test_first_success_is_not_retried(self):
        built = []
        result, payload = retry_truncated(lambda scale, retry: built.append((scale, retry)) or "p",
                                          lambda p: "r")
        self.assertEqual((result, payload, built), ("r", "p", [(1.0, False)]))

    def test_known_count_rebuilds_at_full_scale_unknown_shrinks(self):
        for real, scale in ((9000, 1.0), (None, 0.9)):
            built = []
            calls = []

            def send(payload):
                calls.append(payload)
                if len(calls) == 1:
                    raise PromptTruncated(real_tokens=real, kept_tokens=10)
                return "r"
            retry_truncated(lambda s, r: built.append((s, r)) or f"p{len(built)}", send)
            self.assertEqual(built, [(1.0, False), (scale, True)])
            self.assertEqual(calls, ["p1", "p2"])

    def test_second_truncation_raises(self):
        def send(payload):
            raise PromptTruncated(real_tokens=5, kept_tokens=1)
        with self.assertRaises(PromptTruncated) as caught:
            retry_truncated(lambda s, r: "p", send)
        self.assertIn("prompt truncated by the engine", str(caught.exception))
