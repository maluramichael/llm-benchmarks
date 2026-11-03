"""Test helper: an engine that returns fixed timings and counts its calls."""

from bench.engines.base import Completion, EngineError, Timing


class FakeEngine:
    name = "fake"

    def __init__(self, gen_tps=(100.0,), ttft=0.5, load_s=2.0, cached_on_repeat=True,
                 fail_on=None, name=None, always_cached=0):
        if name:
            self.name = name
        self.gen_tps = list(gen_tps)
        self.ttft = ttft
        self.load_s = load_s
        self.cached_on_repeat = cached_on_repeat
        self.fail_on = fail_on  # callable(call_index, prompt) -> exception or None
        self.prompts: list[str] = []
        self.calls: list[str] = []  # "load:<ctx>", "unload", "complete"
        self.rs_mb = None
        self.fraction = 1.0
        self.always_cached = always_cached
        self.attempts = 0

    # engine interface
    def start(self): pass
    def stop(self): pass
    def version(self): return "fake-1"
    def backend(self): return None

    def load(self, model, ctx, parallel=1):
        self.calls.append(f"load:{ctx}")
        if self.fail_on is not None:
            err = self.fail_on("load", ctx)
            if err:
                raise err
        return self.load_s

    def unload(self, model):
        self.calls.append("unload")

    def count_tokens(self, model, text):
        return len(text.split())

    def gpu_fraction(self, model): return self.fraction
    def kv_buffer_mb(self): return 100.0
    def rs_buffer_mb(self): return self.rs_mb

    def complete(self, model, prompt, sampling):
        index = self.attempts
        self.attempts += 1
        if self.fail_on is not None:
            err = self.fail_on("complete", index)
            if err:
                raise err
        cached = len(prompt.split()) if (prompt in self.prompts and self.cached_on_repeat) else 0
        cached = cached or self.always_cached
        self.prompts.append(prompt)
        self.calls.append("complete")
        tps = self.gen_tps[index % len(self.gen_tps)]
        n = len(prompt.split())
        timing = Timing(load_s=None, ttft_s=self.ttft if not cached else self.ttft / 10,
                        prompt_tokens=n, prompt_tps=1000.0, gen_tokens=256, gen_tps=tps,
                        cached_tokens=cached)
        return Completion(text="ok", timing=timing)

    @property
    def complete_count(self):
        return self.calls.count("complete")
