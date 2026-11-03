"""Shared pieces of the test suites: run context, step keys, prompts, step bookkeeping.

Distinct prompts: both engines keep a prompt cache (llama.cpp `cache_prompt`, Ollama
prefix reuse), so repeating a prompt would be served warm and distort the numbers. Every
measured request therefore gets its own `slot`: a corpus offset plus a unique header line,
so the very first tokens differ. Only the speed step `cache` repeats a prompt on purpose.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from bench import VERSION
from bench.corpus import cut_to_tokens
from bench.privacy import local_secrets, scrub

SLOT_STRIDE_CHARS = 1009  # corpus offset between two slots (prime: no alignment with lines)
CHARS_PER_TOKEN_MAX = 8  # generous: enough characters for n tokens
MIN_FILL = 0.9  # a prompt shorter than this share of the wanted length is an error
ERROR_LIMIT = 500


class RunAborted(RuntimeError):
    """The whole run must end now (GPU driver reset detected)."""


@dataclass
class RunContext:
    cfg: object
    engine: object
    model: object
    store: object
    guard: object
    corpus_text: str
    raw_dir: Path
    prompts: dict = field(default_factory=dict)  # (slot, n_tokens) -> prompt, memoised
    nonce: str = ""  # per run_speed call; makes a rerun use fresh prompts


def key(ctx: RunContext, suite: str, step: str) -> str:
    return f"{ctx.engine.name}/{ctx.model.id}/{suite}/{step}"


def forced(ctx: RunContext, test_key: str) -> bool:
    cfg = ctx.cfg
    return bool(cfg.force) or any(
        fnmatch.fnmatchcase(test_key, p) for p in cfg.force_tests)


def step_meta(ctx: RunContext) -> dict:
    try:
        version = ctx.engine.version()
    except Exception:
        version = None
    return {"engine_version": version, "model_digest": ctx.model.digest,
            "bench_version": VERSION}


def clean_error(err: BaseException) -> str:
    """Error text safe to store: personal data scrubbed, length limited."""
    text = f"{type(err).__name__}: {err}"
    return scrub(text, local_secrets())[:ERROR_LIMIT]


def now() -> datetime:
    return datetime.now(timezone.utc)


def make_prompt(ctx: RunContext, n_tokens: int, slot: int) -> str:
    """A prompt of about `n_tokens` tokens, different for every `slot`.

    The tokenizer of the loaded model measures the length. The corpus is repeated
    when it is shorter than needed (long context steps).
    """
    memo = (slot, n_tokens)
    if memo in ctx.prompts:
        return ctx.prompts[memo]
    text = ctx.corpus_text
    if not text.strip():
        raise ValueError("corpus text is empty")
    need = n_tokens * CHARS_PER_TOKEN_MAX
    offset = (slot * SLOT_STRIDE_CHARS) % len(text)
    start = offset
    while start < len(text) and not text[start].isspace():
        start += 1  # begin at a word boundary
    body = text[start:]
    while len(body) < need:
        body += "\n\n" + text
    tag = f"{ctx.nonce}-{slot}" if ctx.nonce else str(slot)
    header = f"Document {tag}.\n\n"
    count = lambda t: ctx.engine.count_tokens(ctx.model, t)  # noqa: E731
    prompt = cut_to_tokens(header + body[:need], n_tokens, count)
    used = count(prompt)
    if used < n_tokens * MIN_FILL:
        raise ValueError(f"could only build a prompt of {used} tokens (wanted {n_tokens})")
    ctx.prompts[memo] = prompt
    return prompt


def reset_state(ctx: RunContext, since: datetime) -> str:
    """Driver reset check as text: "none", "reset" or "unknown" (probe failed).

    "unknown" is never read as "none": callers decide what is safe for their step.
    """
    try:
        result = ctx.guard.driver_reset_since(since)
    except Exception:
        return "unknown"
    if result is True:
        return "reset"
    return "none" if result is False else "unknown"


def median(values) -> float | int | None:
    """Median of the values that are not None; None when there are none."""
    import statistics
    present = [v for v in values if v is not None]
    return statistics.median(present) if present else None


def context_plan(ctx: RunContext) -> tuple[str, int | str | None]:
    """What a step that needs a context size can do now: ("pending", None),
    ("skip", reason) or ("run", largest ok memory rung).

    pending: no `memory/ctx-*` entry yet, or no rung is ok and a rung is still retryable
    (error, aborted, running): the step is neither run nor stored.
    """
    prefix = key(ctx, "memory", "ctx-")
    rungs = {int(k[len(prefix):]): v["status"] for k, v in ctx.store.data["tests"].items()
             if k.startswith(prefix) and k[len(prefix):].isdigit()}
    if not rungs:
        return "pending", None
    ok = [n for n, status in rungs.items() if status == "ok"]
    if ok:
        return "run", max(ok)
    if any(status not in ("does_not_fit", "skipped") for status in rungs.values()):
        return "pending", None  # memory ladder has errors, rerun memory first
    return "skip", "no context step fits"
