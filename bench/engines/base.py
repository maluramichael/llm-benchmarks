"""Engine interface and the data types every engine adapter returns.

Timing semantics (identical for all engines, so results are comparable):

* load_s         seconds the engine spent loading the model for this request, or None
                 when it does not say (model was already resident: a few milliseconds).
* ttft_s         time to first token. For `Engine.complete` this is wall-clock time
                 measured by the harness from sending the request to receiving the first
                 streamed chunk with non-empty text. It includes model load and prompt
                 processing when the model is not resident yet, so load first for speed
                 tests. If the model produced no text at all, it is the time until the
                 final chunk. For `Engine.chat` (not streamed) it is the engine-reported
                 load plus prompt evaluation time, a server-side approximation.
* prompt_tokens  total prompt length in tokens, cached tokens included (Ollama's
                 prompt_eval_count includes them, verified on 0.40.13).
* prompt_tps     (prompt_tokens - cached_tokens) divided by the engine-reported prompt
                 evaluation time, because that time covers only the evaluated (uncached)
                 part. None when either is zero (for example a fully cached prompt).
* gen_tokens, gen_tps  generated tokens and tokens per second of generation time.
* cached_tokens  prompt tokens served from the engine's prompt cache (part of
                 prompt_tokens), None if unknown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


class EngineError(RuntimeError):
    """An engine call failed (server not reachable, error reply, bad data)."""


class PromptTruncated(EngineError):
    """The engine cut the prompt because it did not fit the context (Ollama does that
    silently: it keeps the first 4 tokens and the tail, and only logs a warning).

    `real_tokens` is the length the prompt really had when the engine said so (from its
    log), else None; `kept_tokens` is what the engine evaluated (prompt_eval_count). A
    subclass of EngineError so that code that does not know it still treats it as a failed
    call; suites that know it rebuild the prompt once (bench.suites.retry_truncated).
    """

    def __init__(self, message: str = "prompt truncated by the engine",
                 real_tokens: int | None = None, kept_tokens: int | None = None):
        detail = []
        if real_tokens is not None:
            detail.append(f"{real_tokens} tokens sent")
        if kept_tokens is not None:
            detail.append(f"{kept_tokens} kept")
        super().__init__(message + (f" ({', '.join(detail)})" if detail else ""))
        self.real_tokens = real_tokens
        self.kept_tokens = kept_tokens


@dataclass
class ModelRef:
    id: str
    cls: str
    quant: str
    ollama_tag: str | None
    gguf_path: Path | None
    digest: str | None
    size_bytes: int | None
    # Where the weights are loaded from, a category only (never a path): "local store",
    # "model dir" or "explicit". None while they are still to be copied or downloaded.
    weights_source: str | None = None


@dataclass
class Sampling:
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None
    seed: int = 42
    max_tokens: int = 256


@dataclass
class Timing:
    load_s: float | None
    ttft_s: float
    prompt_tokens: int
    prompt_tps: float | None
    gen_tokens: int
    gen_tps: float | None
    cached_tokens: int | None


@dataclass
class Completion:
    text: str
    timing: Timing
    raw: dict = field(default_factory=dict)


@dataclass
class ChatResult:
    text: str
    tool_calls: list[dict]  # entries {"name": str, "arguments": dict}
    timing: Timing
    raw: dict = field(default_factory=dict)


class Engine(Protocol):
    name: str

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def version(self) -> str: ...

    def backend(self) -> str | None: ...

    def load(self, model: ModelRef, ctx: int, parallel: int = 1) -> float:
        """Load the model with a context of `ctx` tokens; returns the load time in s."""
        ...

    def unload(self, model: ModelRef) -> None: ...

    def complete(self, model: ModelRef, prompt: str, sampling: Sampling) -> Completion: ...

    def chat(self, model: ModelRef, messages: list[dict], sampling: Sampling,
             tools: list[dict] | None = None, json_schema: dict | None = None,
             thinking: bool = False) -> ChatResult: ...

    def count_tokens(self, model: ModelRef, text: str) -> int: ...

    def gpu_fraction(self, model: ModelRef) -> float | None: ...

    def kv_buffer_mb(self) -> float | None: ...
