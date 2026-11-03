"""Memory suite: a computed memory curve and the measured context ladder.

computed    weights plus f16 KV cache per context step, from the GGUF header only. It
            needs no running engine. The curve is an upper bound for sliding-window
            attention (the real cache is smaller) and ignores recurrent state of hybrid
            models; both cases are flagged in the values.
ctx-<n>     one measured rung per context step: unload, VRAM guard, load with n tokens,
            fill the context with a prompt, measure. A rung that does not fit stops the
            ladder; higher rungs are stored as `skipped` (see safety.ladder_start, which
            also stops a second run from repeating a rung that crashed).

Decisions for failures:
* engine error while loading or filling a rung (EngineError) = `does_not_fit`.
* any other exception = `error` (retried next run) and the ladder stops for this run, unless
  the reset probe says "unknown": then `does_not_fit`, so a rung that may have reset the
  driver is never repeated. Every failure first asks the reset probe; "reset" = `aborted`.
* guard refuses (VRAM reserve, partial offload, shared memory or swap growth) =
  `does_not_fit`.
* driver reset = rung `aborted`, RunAborted is raised, the run ends.
* reset probe failed (None) = the rung is kept as `ok` with `driver_reset_check:
  "unknown"`, but the ladder does not climb further (rest `skipped`), because a rung that
  may have reset the driver must not be followed by a bigger one.
"""

from __future__ import annotations

from bench.engines.base import EngineError, Sampling
from bench.gguf import attention_layer_count, kv_bytes_per_token, memory_curve, read_metadata
from bench.safety import ladder_start, take_reading
from bench.engines.base import PromptTruncated
from bench.suites import (RunAborted, RunContext, SHRINK_UNKNOWN, TRUNCATED_REASON, clean_error,
                          exact_counts, forced, fresh_prompt, key, model_context_length, model_missing,
                          now, reset_state, retry_truncated, skip_beyond, step_meta)

MIB = 1024 * 1024
OUTPUT_TOKENS = 128
FILL_MARGIN = 256  # tokens kept free for output and chat template
MAX_FILL = 0.97  # the prompt targets at most this share of the rung (exact token counts)
MAX_FILL_ESTIMATED = 0.94  # the same for engines that estimate token counts (Ollama)
MEMORY_SLOT = 1000
CACHE_TYPE_BYTES = 2.0  # f16 KV cache
FINAL_STATUSES = ("ok", "does_not_fit", "skipped")
NOTE_PLAIN = "weights plus f16 KV cache; compute buffers and other overhead are not included"
NOTE_FLAGGED = ("upper bound: sliding-window attention keeps a smaller cache, and recurrent "
                "state of hybrid models is not included; " + NOTE_PLAIN)


def _fill_tokens(n: int, exact: bool = True) -> int:
    """Prompt length that fills rung `n`: leaves room for output and template, at most 97 %
    (94 % when the engine only estimates token counts)."""
    share = MAX_FILL if exact else MAX_FILL_ESTIMATED
    return max(1, min(n - OUTPUT_TOKENS - FILL_MARGIN, int(n * share)))


def _steps(ctx: RunContext) -> list[int]:
    return sorted({int(s) for s in ctx.cfg.ctx_steps if int(s) > 0})


def _flags(meta: dict) -> tuple[bool, bool]:
    sliding = any(k.endswith(".attention.sliding_window") or k.endswith(".sliding_window_pattern")
                  for k in meta)
    recurrent = any(".ssm." in k or k.endswith(".full_attention_interval")
                    or k.endswith(".attention.recurrent_layers") for k in meta)
    return sliding, recurrent


def _computed(ctx: RunContext) -> tuple[str, dict]:
    path = ctx.model.gguf_path
    if path is None:
        return "skipped", {"reason": "model has no GGUF file; the curve needs the GGUF header"}
    meta = read_metadata(path)
    weights = ctx.model.size_bytes or path.stat().st_size
    steps = _steps(ctx)
    curve = memory_curve(meta, weights, steps, CACHE_TYPE_BYTES)
    sliding, recurrent = _flags(meta)
    return "ok", {
        "weights_bytes": weights,
        "context_length": model_context_length(ctx),
        "kv_bytes_per_token": kv_bytes_per_token(meta, CACHE_TYPE_BYTES),
        "attention_layers": attention_layer_count(meta),
        "cache_type_bytes": CACHE_TYPE_BYTES,
        "curve": [{"ctx": n, "bytes": b} for n, b in curve],
        "sliding_window": sliding,
        "recurrent_state_ignored": recurrent,
        "note": NOTE_FLAGGED if (sliding or recurrent) else NOTE_PLAIN,
    }


def _estimates(ctx: RunContext) -> dict[int, int]:
    """Expected bytes per step for the VRAM guard; {} when the model has no GGUF."""
    try:
        status, values = _computed(ctx)
    except Exception:
        return {}
    if status != "ok":
        return {}
    return {row["ctx"]: row["bytes"] for row in values["curve"]}


def _run_computed(ctx: RunContext, meta: dict) -> None:
    test_key = key(ctx, "memory", "computed")
    if not ctx.store.should_run(test_key, ctx.cfg.force, ctx.cfg.force_tests):
        return
    ctx.store.mark_running(test_key, meta)
    try:
        status, values = _computed(ctx)
    except Exception as err:
        ctx.store.save(test_key, "error", {"error": clean_error(err)}, meta)
        return
    ctx.store.save(test_key, status, values, meta)


def _optional(call):
    try:
        return call()
    except Exception:
        return None


FOREIGN_VRAM_MB = 1024  # baseline growth since the first rung that points to a foreign process


def _first_baseline(ctx: RunContext, n: int) -> int | None:
    """vram_baseline_mb of the lowest other rung of this ladder that has one."""
    prefix = key(ctx, "memory", "ctx-")
    found = {}
    for test_key, entry in ctx.store.data["tests"].items():
        rest = test_key[len(prefix):]
        if test_key.startswith(prefix) and rest.isdigit() and int(rest) != n:
            baseline = (entry.get("values") or {}).get("vram_baseline_mb")
            if isinstance(baseline, (int, float)):
                found[int(rest)] = baseline
    return found[min(found)] if found else None


def _refused(ctx: RunContext, n: int, test_key: str, values: dict, why: str, meta: dict) -> str:
    """The guard said no. If the VRAM in use before the load (baseline) grew a lot since the
    first rung, a foreign process took it: that is no limit of this model, so the rung is a
    retryable `error`. Otherwise it is a final `does_not_fit`."""
    first, baseline = _first_baseline(ctx, n), values.get("vram_baseline_mb")
    if first is not None and baseline is not None and baseline - first > FOREIGN_VRAM_MB:
        ctx.store.save(test_key, "error",
                       {**values, "reason": f"VRAM taken by another process (in use before the "
                                            f"load: {baseline} MiB, {first} MiB at the first "
                                            f"rung); guard said: {why}"}, meta)
        return "error"
    ctx.store.save(test_key, "does_not_fit", {**values, "reason": why}, meta)
    return "does_not_fit"


def _rung(ctx: RunContext, n: int, index: int, estimates: dict, meta: dict) -> str:
    """Run one rung. Returns ok, stop (ok, but do not climb), does_not_fit or error."""
    engine, guard, store = ctx.engine, ctx.guard, ctx.store
    test_key = key(ctx, "memory", f"ctx-{n}")
    store.mark_running(test_key, meta)
    started = now()
    values: dict = {"ctx": n}
    status = "ok"
    fill_target = _fill_tokens(n, exact_counts(ctx))
    try:
        try:
            engine.unload(ctx.model)
            before = take_reading(guard.gpu_index)
            # Used VRAM with this model unloaded (desktop, other programs): the chart subtracts it.
            if before.vram_free_mb is not None and guard.vram_total_mb:
                values["vram_baseline_mb"] = guard.vram_total_mb - before.vram_free_mb
            allowed, why = guard.before_step(int(estimates.get(n, 0) // MIB), before)
            if not allowed:
                return _refused(ctx, n, test_key, values, why, meta)
            values["load_s"] = engine.load(ctx.model, n)
            fill = fill_target
            slot = MEMORY_SLOT + index
            sampling = Sampling(temperature=0.0, seed=42, max_tokens=OUTPUT_TOKENS)
            result, _ = retry_truncated(
                lambda scale, retry: fresh_prompt(ctx, fill, slot, scale, retry),
                lambda prompt: engine.complete(ctx.model, prompt, sampling))
            timing = result.timing
            fraction = _optional(lambda: engine.gpu_fraction(ctx.model))
            after = take_reading(guard.gpu_index, fraction)
            values.update({
                "gpu_fraction": fraction,
                "kv_buffer_mb": _optional(engine.kv_buffer_mb),
                "rs_buffer_mb": _optional(getattr(engine, "rs_buffer_mb", lambda: None)),
                "vram_free_mb": after.vram_free_mb,
                "vram_used_mb": (guard.vram_total_mb - after.vram_free_mb
                                 if after.vram_free_mb is not None and guard.vram_total_mb
                                 else None),
                "shared_mb": after.shared_mb, "swap_mb": after.swap_mb,
                "prompt_tokens": timing.prompt_tokens, "ttft_s": timing.ttft_s,
                "prompt_tps": timing.prompt_tps, "gen_tokens": timing.gen_tokens,
                "gen_tps": timing.gen_tps, "cached_tokens": timing.cached_tokens})
            allowed, why = guard.after_step(before, after)
            if not allowed:
                return _refused(ctx, n, test_key, values, why, meta)
        except RunAborted:
            raise
        except Exception as err:
            # A dead engine is how a driver reset usually looks: check before judging.
            check = reset_state(ctx, started)
            if check == "reset":
                store.save(test_key, "aborted",
                           {**values, "reason": "GPU driver reset detected during this rung",
                            "error": clean_error(err)}, meta)
                raise RunAborted(f"GPU driver reset detected during {test_key}") from None
            values["driver_reset_check"] = check
            if isinstance(err, PromptTruncated):
                # The prompt was cut twice: nothing was measured, the rung is retried later.
                store.save(test_key, "error",
                           {**values, "reason": TRUNCATED_REASON, "error": clean_error(err)}, meta)
                return "error"
            if model_missing(err):  # not a fit problem: retried by the next run
                store.save(test_key, "error", {**values, "error": clean_error(err)}, meta)
                return "error"
            if isinstance(err, EngineError) or check == "unknown":
                # unknown: never retry a rung that may have reset the driver
                store.save(test_key, "does_not_fit", {**values, "reason": clean_error(err)}, meta)
                return "does_not_fit"
            store.save(test_key, "error", {**values, "error": clean_error(err)}, meta)
            return "error"
        check = reset_state(ctx, started)
        if check == "reset":
            store.save(test_key, "aborted",
                       {**values, "reason": "GPU driver reset detected during this rung"}, meta)
            raise RunAborted(f"GPU driver reset detected during {test_key}")
        values["driver_reset_check"] = check
        store.save(test_key, status, values, meta)
        return "ok" if check == "none" else "stop"
    finally:
        for scale in (1.0, SHRINK_UNKNOWN):
            ctx.prompts.pop((MEMORY_SLOT + index, max(1, int(fill_target * scale))), None)


def _skip_above(ctx: RunContext, n: int, steps: list[int], reason: str) -> None:
    for step in steps:
        test_key = key(ctx, "memory", f"ctx-{step}")
        entry = ctx.store.get(test_key)
        if step > n and (entry is None or entry["status"] not in FINAL_STATUSES):
            ctx.store.save(test_key, "skipped", {"reason": reason})


def _ladder(ctx: RunContext, meta: dict) -> None:
    steps = _steps(ctx)
    limit = model_context_length(ctx)
    if limit:  # rungs above the trained context are no limit of the hardware: not measured
        skip_beyond(ctx, [key(ctx, "memory", f"ctx-{n}") for n in steps if n > limit], limit, meta)
        steps = [n for n in steps if n <= limit]
    prefix = key(ctx, "memory", "")
    estimates = _estimates(ctx)
    forced_steps = [n for n in steps if forced(ctx, key(ctx, "memory", f"ctx-{n}"))]
    try:
        if forced_steps:
            queue = iter(forced_steps)
            next_step = lambda: next(queue, None)  # noqa: E731
        else:
            next_step = lambda: ladder_start(ctx.store, prefix, steps)  # noqa: E731
        while (n := next_step()) is not None:
            outcome = _rung(ctx, n, steps.index(n), estimates, meta)
            if outcome == "stop":
                _skip_above(ctx, n, steps, f"driver reset check unavailable after step {n}")
                return
            if outcome == "does_not_fit":
                ladder_start(ctx.store, prefix, steps)  # marks the higher rungs as skipped
                return
            if outcome == "error":
                return
    finally:
        _optional(lambda: ctx.engine.unload(ctx.model))


def run_memory(ctx: RunContext) -> None:
    """Computed curve (always) and measured ladder (unless scope is no-context).

    Scope `warm-only` runs no memory step at all. Raises RunAborted on a driver reset.
    """
    scope = ctx.cfg.scope
    if scope == "warm-only":
        return
    meta = step_meta(ctx)
    _run_computed(ctx, meta)
    if scope == "no-context":
        return
    _ladder(ctx, meta)
