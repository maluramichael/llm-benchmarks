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
from bench.engines.base import EngineError, PromptTruncated
from bench.corpus import cut_to_tokens
from bench.gguf import read_metadata
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
    facts: dict = field(default_factory=dict)  # memoised facts about the model


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
            "bench_version": VERSION,
            "weights_source": getattr(ctx.model, "weights_source", None)}


def clean_error(err: BaseException) -> str:
    """Error text safe to store: personal data scrubbed, length limited."""
    text = f"{type(err).__name__}: {err}"
    return scrub(text, local_secrets())[:ERROR_LIMIT]


def model_missing(err: BaseException) -> bool:
    """An engine error that says the model is not there (Ollama: HTTP 404, "not found").
    That is no fit problem: the step is stored `error` (retried), never `does_not_fit`."""
    text = str(err).lower()
    return isinstance(err, EngineError) and "404" in text and "not found" in text


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
    # The unique tag comes first: the very first tokens differ between prompts, so a
    # prompt cache has no shared prefix to reuse even between neighbouring slots.
    header = f"{tag} Document.\n\n"
    count = lambda t: ctx.engine.count_tokens(ctx.model, t)  # noqa: E731
    prompt = cut_to_tokens(header + body[:need], n_tokens, count)
    used = count(prompt)
    if used < n_tokens * MIN_FILL:
        raise ValueError(f"could only build a prompt of {used} tokens (wanted {n_tokens})")
    ctx.prompts[memo] = prompt
    return prompt


TRUNCATED_REASON = "prompt truncated by the engine"
SHRINK_UNKNOWN = 0.9  # new length share when the engine did not say how long the prompt was


def exact_counts(ctx: RunContext) -> bool:
    """True when the engine counts tokens exactly (llama.cpp), False for estimates (Ollama)."""
    return bool(getattr(ctx.engine, "exact_token_counts", True))


def fresh_prompt(ctx: RunContext, n_tokens: int, slot: int, scale: float, retry: bool) -> str:
    """`make_prompt` for a `retry_truncated` builder: on a retry the memoised prompt of
    this slot is dropped (the engine has learned a better ratio since) and `scale` applies."""
    if retry:
        ctx.prompts.pop((slot, n_tokens), None)
    return make_prompt(ctx, max(1, int(n_tokens * scale)), slot)


def retry_truncated(build, send):
    """Send a prompt; if the engine truncated it, rebuild it once and send it again.

    `build(scale, retry)` makes the payload (prompt or messages), `send(payload)` sends it
    and returns the result; both are called as `build(1.0, False)` first. After a
    PromptTruncated the engine has corrected its token estimate from the real count in the
    error, so the payload is rebuilt at the same target (scale 1.0); when the real count is
    unknown the target shrinks to 90 % (`SHRINK_UNKNOWN`). A second PromptTruncated is
    raised with the reason "prompt truncated by the engine": callers store `error`
    (retryable), never `ok` and never `does_not_fit`. Returns (result, payload).
    """
    scale, retry = 1.0, False
    while True:
        payload = build(scale, retry)
        try:
            return send(payload), payload
        except PromptTruncated as err:
            if retry:
                raise PromptTruncated(TRUNCATED_REASON, err.real_tokens, err.kept_tokens) \
                    from None
            scale = 1.0 if err.real_tokens else SHRINK_UNKNOWN
            retry = True


def prepare_counting(ctx: RunContext, n_ctx: int) -> None:
    """Tell an engine that counts by calibration (Ollama) which context the prompts that
    are built next will run with, so it loads an unloaded model once at that size."""
    prepare = getattr(ctx.engine, "prepare_counting", None)
    if prepare is not None:
        prepare(ctx.model, n_ctx)


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


def model_context_length(ctx: RunContext) -> int | None:
    """The context length the model was trained for (GGUF `<arch>.context_length`), read
    from the model file once; None when there is no file or no such key."""
    if "context_length" not in ctx.facts:
        value = None
        path = getattr(ctx.model, "gguf_path", None)
        if path is not None:
            try:
                meta = read_metadata(path)
                found = meta.get(f"{meta.get('general.architecture')}.context_length")
                if isinstance(found, int) and not isinstance(found, bool) and found > 0:
                    value = found
            except (OSError, ValueError, EOFError):
                pass
        ctx.facts["context_length"] = value
    return ctx.facts["context_length"]


def beyond_reason(limit: int) -> str:
    return f"beyond the model's context length ({limit})"


def beyond_values(limit: int) -> dict:
    return {"reason": beyond_reason(limit), "beyond_model_context": True,
            "context_length": limit}


def is_beyond_skip(entry) -> bool:
    return bool(entry and entry["status"] == "skipped"
                and (entry.get("values") or {}).get("beyond_model_context"))


def skip_beyond(ctx: RunContext, test_keys: list[str], limit: int, meta: dict,
                extra: dict | None = None) -> None:
    """Store steps that need more context than the model has as `skipped` with the flag
    `beyond_model_context`. Such a step is no limit of the hardware, so it never counts as
    a failure; an earlier error or does_not_fit of it is replaced."""
    for test_key in test_keys:
        if not is_beyond_skip(ctx.store.get(test_key)):
            ctx.store.save(test_key, "skipped", {**(extra or {}), **beyond_values(limit)}, meta)


PARTIAL = 0.999  # below this share on the GPU a model is partly offloaded to the CPU


def offload_fields(ctx: RunContext) -> dict:
    """gpu_fraction of the loaded model and, when known and below 100 %, partial_offload
    True. The step result is real, but not "whole model on the GPU": charts and README mark
    it. Call it while the model is still loaded; an unknown fraction gives {}."""
    try:
        fraction = ctx.engine.gpu_fraction(ctx.model)
    except Exception:
        return {}
    if not isinstance(fraction, (int, float)):
        return {}
    fields: dict = {"gpu_fraction": fraction}
    if fraction < PARTIAL:
        fields["partial_offload"] = True
    return fields


def stored_no_fit(ctx: RunContext, n_ctx: int) -> str | None:
    """Why the model cannot be loaded with `n_ctx` tokens when the memory suite already
    found out: `memory/fit` or a rung of the same or a smaller context is `does_not_fit`."""
    if (ctx.store.get(key(ctx, "memory", "fit")) or {}).get("status") == "does_not_fit":
        return "does not fit (guard): memory/fit is does_not_fit"
    prefix = key(ctx, "memory", "ctx-")
    for test_key, entry in ctx.store.data["tests"].items():
        rest = test_key[len(prefix):]
        if (test_key.startswith(prefix) and rest.isdigit() and int(rest) <= n_ctx
                and entry.get("status") == "does_not_fit"):
            return f"does not fit (guard): memory/ctx-{rest} is does_not_fit"
    return None


def memory_estimate_mb(ctx: RunContext, n_ctx: int, parallel: int = 1) -> int | None:
    """Weights plus KV cache for `n_ctx` tokens per slot, from memory/computed; None if
    that step is missing or has no figures."""
    values = (ctx.store.get(key(ctx, "memory", "computed")) or {}).get("values") or {}
    weights, per_token = values.get("weights_bytes"), values.get("kv_bytes_per_token")
    if not weights or per_token is None:
        return None
    return int((weights + per_token * n_ctx * parallel) // (1024 * 1024))


def measured_rung(ctx: RunContext, needed: int) -> int | None:
    """The smallest memory rung of this engine and model that is stored `ok` and has a
    context of at least `needed` tokens: measured proof that the model loads that big."""
    prefix = key(ctx, "memory", "ctx-")
    found = [int(k[len(prefix):]) for k, e in ctx.store.data["tests"].items()
             if k.startswith(prefix) and k[len(prefix):].isdigit()
             and e.get("status") == "ok" and int(k[len(prefix):]) >= needed]
    return min(found) if found else None


def guard_check(ctx: RunContext, n_ctx: int, read_vram,
                parallel: int = 1) -> tuple[str | None, str | None]:
    """May the model be loaded for `n_ctx` tokens per slot and `parallel` slots?

    Returns (refusal, basis). refusal is None when it may, else "does not fit (guard): ...".
    basis says what the answer rests on: "measured ctx-<n>" (a memory rung of at least the
    needed context is ok: evidence beats the estimate), "estimate" (weights plus KV cache
    from memory/computed, asked of the guard with a VRAM reading after an unload), or None
    (nothing known, nothing checked). A rung at or below the needed context that did not
    fit refuses without asking the guard.
    """
    needed = n_ctx * parallel
    if (ctx.store.get(key(ctx, "memory", "fit")) or {}).get("status") == "does_not_fit":
        return stored_no_fit(ctx, needed), None
    rung = measured_rung(ctx, needed)
    if rung is not None:
        return None, f"measured ctx-{rung}"
    stored = stored_no_fit(ctx, needed)
    if stored:
        return stored, None
    estimate = memory_estimate_mb(ctx, n_ctx, parallel)
    if estimate is None:
        return None, None
    try:
        ctx.engine.unload(ctx.model)
    except Exception:
        pass
    allowed, why = ctx.guard.before_step(estimate, read_vram(ctx.guard.gpu_index))
    if not allowed:
        return f"does not fit (guard): {why}", None
    return None, "estimate"


def skip_for_guard(ctx: RunContext, test_keys: list[str], reason: str, meta: dict,
                   extra: dict | None = None) -> None:
    for test_key in test_keys:
        ctx.store.save(test_key, "skipped", {**(extra or {}), "reason": reason}, meta)


def scope_skip(ctx: RunContext, test_key: str, extra: dict | None = None, meta=None) -> None:
    """Store a step that the chosen --scope leaves out (no memory ladder, so the context size
    it needs will never exist). The entry is `skipped` with `scope_skipped` True: unlike a
    real skip it is not final, a later run with --scope full measures the step."""
    ctx.store.save(test_key, "skipped",
                   {**(extra or {}), "reason": f"not in scope {ctx.cfg.scope}",
                    "scope_skipped": True}, meta)


def wants_run(ctx: RunContext, test_key: str) -> bool:
    """store.should_run, except that a step skipped only because of the scope runs again
    once the scope is full."""
    entry = ctx.store.get(test_key)
    if (entry and entry["status"] == "skipped" and ctx.cfg.scope == "full"
            and _dict_of(entry.get("values")).get("scope_skipped") is True):
        return True
    return ctx.store.should_run(test_key, ctx.cfg.force, ctx.cfg.force_tests)


def _dict_of(value) -> dict:
    return value if isinstance(value, dict) else {}


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
