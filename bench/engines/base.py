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
* prompt_tokens  tokens the engine actually evaluated for the prompt (cached tokens are
                 not included, see cached_tokens).
* prompt_tps     prompt_tokens divided by the engine-reported prompt evaluation time, or
                 None when either is zero (for example a fully cached prompt).
* gen_tokens, gen_tps  generated tokens and tokens per second of generation time.
* cached_tokens  prompt tokens served from the engine's prompt cache, None if unknown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


class EngineError(RuntimeError):
    """An engine call failed (server not reachable, error reply, bad data)."""


@dataclass
class ModelRef:
    id: str
    cls: str
    quant: str
    ollama_tag: str | None
    gguf_path: Path | None
    digest: str | None
    size_bytes: int | None


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
