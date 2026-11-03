"""Token counting must never warm the prompt cache with a prompt that is measured later.

A fake Ollama server tokenizes by words, caches whatever it evaluated (prefix reuse like
the real slot cache) and reports the cached part in prompt_eval_cached_count.
"""

import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bench.config import Config
from bench.engines.base import ModelRef
from bench.engines.ollama import OllamaEngine
from bench.safety import Reading
from bench.store import ResultStore
from bench.suites import RunContext, key
from bench.suites import memory, speed
from fake_http import FakeServer
from gguf_writer import write_gguf

WORDS = random.Random(7).choices(
    ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
     "lambda", "omega", "river", "mountain", "city", "bridge", "castle", "forest"], k=90000)
CORPUS = " ".join(WORDS)
DENSE = {"general.architecture": "llama", "llama.block_count": 32,
         "llama.attention.head_count_kv": 8, "llama.attention.key_length": 128}


class Guard:
    gpu_index = 0
    vram_total_mb = 24000

    def before_step(self, required_mb, reading):
        return True, "ok"

    def after_step(self, prev, now):
        return True, "ok"

    def driver_reset_since(self, ts):
        return False


class CachingOllama:
    """Routes for FakeServer: whole-word tokens, one prompt cache slot."""

    def __init__(self):
        self.last: list[str] = []
        self.ctx = 0

    def routes(self) -> dict:
        return {"/api/version": {"version": "0.40.13"}, "/api/generate": self.generate,
                "/api/ps": {"models": []}}

    def generate(self, body):
        prompt = body.get("prompt", "")
        options = body.get("options") or {}
        if body.get("keep_alive") == 0:
            self.last = []
            return {"done": True}
        self.ctx = options.get("num_ctx", self.ctx)
        if prompt == "":
            return {"done": True, "load_duration": 1_000_000_000}
        tokens = prompt.split()
        if len(tokens) + options.get("num_predict", 0) > self.ctx:
            return {"error": "prompt too long for the context"}
        shared = 0
        for old, new in zip(self.last, tokens):
            if old != new:
                break
            shared += 1
        self.last = tokens
        reply = {"done": True, "response": "x", "prompt_eval_count": len(tokens),
                 "prompt_eval_cached_count": shared, "prompt_eval_duration": 10**9,
                 "eval_count": 5, "eval_duration": 10**8}
        if body.get("stream"):
            return json.dumps(reply) + "\n"
        return reply


class CacheFreeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fake = CachingOllama()
        server = FakeServer(self.fake.routes())
        server.__enter__()
        self.addCleanup(server.__exit__)
        self.server = server
        for module in (speed, memory):
            patcher = mock.patch.object(
                module, "take_reading", lambda idx, frac=None: Reading(20000, frac, None, None))
            patcher.start()
            self.addCleanup(patcher.stop)
        self.gguf = write_gguf(Path(self.tmp.name) / "m.gguf", dict(DENSE))
        self.model = ModelRef("S", "S", "q4", "qwen3.5:4b", self.gguf, "sha", 4 * 1024**3)

    def ctx(self, **cfg):
        engine = OllamaEngine(Config(ollama_url=self.server.url), log_dir=Path(self.tmp.name))
        self.addCleanup(engine.stop)
        engine.start()
        store = ResultStore(Path(self.tmp.name) / "results.json")
        return RunContext(Config(**cfg), engine, self.model, store, Guard(), CORPUS,
                          Path(self.tmp.name) / "raw")

    def test_warm_and_cold_prompts_start_uncached(self):
        ctx = self.ctx(reps=3)
        speed.run_speed(ctx)
        for step, goal in (("cold", 512), ("warm", 8192)):
            values = ctx.store.get(key(ctx, "speed", step))["values"]
            self.assertFalse(values["cache_contaminated"], step)
            for run in values["runs"]:
                self.assertEqual(run["cached_tokens"], 0, step)
                self.assertAlmostEqual(run["prompt_tokens"] / goal, 1.0, delta=0.03)
        cache = ctx.store.get(key(ctx, "speed", "cache"))["values"]
        self.assertGreater(cache["cached_tokens"], 0)  # the repeat is the point of this step

    def test_cold_step_loads_the_unloaded_model_once_at_the_speed_context(self):
        ctx = self.ctx(reps=1)
        speed.run_speed(ctx)
        first = [b for _, p, b in self.server.requests if p == "/api/generate"][0]
        self.assertEqual(first["options"]["num_ctx"], speed.SPEED_CTX)
        self.assertEqual(first["prompt"], "")  # the load, before any counting

    def test_every_memory_rung_starts_uncached_and_fits(self):
        steps = [4096, 8192, 16384]
        ctx = self.ctx(ctx_steps=steps)
        memory.run_memory(ctx)
        for n in steps:
            entry = ctx.store.get(key(ctx, "memory", f"ctx-{n}"))
            self.assertEqual(entry["status"], "ok", (n, entry["values"]))
            values = entry["values"]
            self.assertEqual(values["cached_tokens"], 0, n)
            target = min(n - memory.OUTPUT_TOKENS - memory.FILL_MARGIN, 0.97 * n)
            self.assertAlmostEqual(values["prompt_tokens"] / target, 1.0, delta=0.03, msg=n)
            self.assertLess(values["prompt_tokens"] + memory.OUTPUT_TOKENS, n)

    def test_no_prompt_is_sent_twice_before_it_is_measured(self):
        ctx = self.ctx(reps=3, scope="warm-only")
        speed.run_speed(ctx)
        counted = [b["prompt"] for _, p, b in self.server.requests
                   if p == "/api/generate" and b.get("prompt") and not b.get("stream")]
        measured = [b["prompt"] for _, p, b in self.server.requests
                    if p == "/api/generate" and b.get("stream")]
        self.assertEqual(len(measured), 3)
        self.assertTrue(counted)
        self.assertFalse(set(counted) & set(measured))
        for text in counted:
            self.assertTrue(text.startswith("Calibration "))
