"""Concurrency suite: 1, 2, 4 and 8 simultaneous requests, total and per-request throughput.

Per level `p<n>` (key `<engine>/<model>/concurrency/p<n>`):
unload, VRAM guard, load with `parallel=n` and a context of SLOT_CTX tokens per slot,
build n distinct 8,192 token prompts, send them all at once from n threads, measure.

* total_tps        generated tokens of all requests divided by the wall-clock time of the
                   whole batch (first send to last answer).
* decode_tps_total sum over requests of gen_tokens / (wall_s - ttft_s): aggregate decode speed
                   without the prefill and queue wait that total_tps includes.
* effective_parallel / effective_ctx / serialized  what the engine reports it really got
                   (meta of the adapter). Fewer slots than requested keeps the level `ok`, sets
                   `serialized` and `serialized_note`; the ladder goes on.
* per_request_tps  median of the per-request generation speeds the engine reports.
* ttft_p50_s       median time to first token. Under load a request may wait for a slot.
* requests         the raw numbers of every request.

No request shares a prompt (unique tag first, see bench.suites.make_prompt), so no request
can hit another's prompt cache. `cached_tokens` is summed per level and `cache_contaminated`
is set when a request had more than 5 % of its prompt cached.

Threads: ThreadPoolExecutor, one worker per request, released together by an event. Both
engine adapters are safe for this: `complete` opens its own HTTP connection per call through
a shared urllib opener (stateless handlers, no shared connection pool) and reads only
attributes that do not change while a level runs. Ollama serialises or parallelises inside the
server according to OLLAMA_NUM_PARALLEL, which the adapter pins from `parallel`.
Every future is awaited with a bound (`batch_timeout`); a request that raises never blocks
the others. When the bound passes, the level fails and the final unload ends the hung
requests (the server process is stopped or the model unloaded).

Decisions for failures (same as the memory ladder):
* guard refuses before or after a level = `does_not_fit`; higher levels are stored `skipped`.
  The VRAM estimate is weights plus f16 KV cache for SLOT_CTX * n tokens from
  `memory/computed`; without that entry the check runs with 0 MB and says so in `guard_note`.
* EngineError (load or request) = `does_not_fit`, higher levels `skipped`.
* other exception = `error` (retried next run), higher levels stay pending, unless the reset
  probe says "unknown": then `does_not_fit` so a level that may have reset the driver is never
  repeated. Every failure first asks the reset probe; "reset" = `aborted` + RunAborted.
* a reset probe that failed after a good level keeps the level `ok` but stops climbing.
* a level stored `aborted` is not repeated automatically (only --force / --force-test).
* a foreign Ollama server (given by URL, `parallel` is None) ignores `parallel`: levels above
  1 are stored `skipped` ("foreign Ollama server ignores parallel"), p1 still runs.
* scope `warm-only` runs nothing.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, wait
from threading import Event

from bench.engines.base import EngineError, Sampling
from bench.safety import take_reading
from bench.suites import (RunAborted, RunContext, clean_error, forced, key, make_prompt, median,
                          guard_check, model_context_length, model_missing, now, offload_fields,
                          prepare_counting, reset_state, skip_beyond, step_meta)

MIB = 1024 * 1024
DEFAULT_LEVELS = [1, 2, 4, 8]
PROMPT_TOKENS = 8192
OUTPUT_TOKENS = 256
SLOT_CTX = 10240  # per slot: an 8,192 token prompt plus 256 output tokens, with margin
CONC_SLOT = 300  # corpus slots 300 and up; speed uses 0-299, memory 1000 and up
CONTAMINATION = 0.05
FINAL_STATUSES = ("ok", "does_not_fit", "skipped")
_clock = time.perf_counter


def batch_timeout(engine) -> float:
    """Seconds the whole batch of a level may take at most."""
    base = getattr(engine, "timeout", None)
    return float(base) * 2 if isinstance(base, (int, float)) and base > 0 else 1800.0


def _estimate_mb(ctx: RunContext, n: int) -> int | None:
    values = (ctx.store.get(key(ctx, "memory", "computed")) or {}).get("values", {})
    weights, per_token = values.get("weights_bytes"), values.get("kv_bytes_per_token")
    if not weights or per_token is None:
        return None
    return int((weights + per_token * SLOT_CTX * n) // MIB)


def _one_request(engine, model, prompt: str, go: Event) -> dict:
    go.wait()
    sampling = Sampling(temperature=0.0, seed=42, max_tokens=OUTPUT_TOKENS)
    started = time.perf_counter()
    timing = engine.complete(model, prompt, sampling).timing
    return {"ttft_s": timing.ttft_s, "gen_tps": timing.gen_tps,
            "gen_tokens": timing.gen_tokens, "prompt_tokens": timing.prompt_tokens,
            "cached_tokens": timing.cached_tokens, "wall_s": time.perf_counter() - started}


def _decode_tps_total(requests: list[dict]) -> float | None:
    """Aggregate decode speed: sum of gen_tokens / (wall_s - ttft_s) over the requests.

    Unlike total_tps it leaves out each request's wait and prefill (up to its first token).
    Requests without a positive decode window are ignored; None when none is left.
    """
    rates = [r["gen_tokens"] / (r["wall_s"] - r["ttft_s"]) for r in requests
             if r["gen_tokens"] and r["ttft_s"] is not None and r["wall_s"] - r["ttft_s"] > 0]
    return sum(rates) if rates else None


def _effective(engine, n: int) -> dict:
    """What the engine says it really ran with (meta effective_parallel / effective_ctx)."""
    try:
        meta = dict(getattr(engine, "meta", None) or {})
    except Exception:
        meta = {}
    parallel = meta.get("effective_parallel")
    info = {"effective_parallel": parallel, "effective_ctx": meta.get("effective_ctx"),
            "serialized": None if parallel is None else parallel < n}
    if info["serialized"]:
        info["serialized_note"] = f"engine ran {n} requests with parallel {parallel}"
    return info


def _foreign_ollama(engine) -> bool:
    """An Ollama server given by URL: its NUM_PARALLEL is not ours to set."""
    return getattr(engine, "name", "") == "ollama" and getattr(engine, "parallel", 0) is None


def _measure(ctx: RunContext, prompts: list[str]) -> dict:
    n = len(prompts)
    go = Event()
    pool = ThreadPoolExecutor(max_workers=n, thread_name_prefix="bench-conc")
    try:
        futures = [pool.submit(_one_request, ctx.engine, ctx.model, p, go) for p in prompts]
        started = _clock()
        go.set()
        done, pending = wait(futures, timeout=batch_timeout(ctx.engine))
        wall = _clock() - started
    finally:
        go.set()
        pool.shutdown(wait=False, cancel_futures=True)
    if pending:
        raise EngineError(f"{len(pending)} of {n} requests did not finish in time")
    for future in futures:  # first failure in request order
        error = future.exception()
        if error is not None:
            raise error
    requests = [f.result() for f in futures]
    gen_tokens = sum(r["gen_tokens"] or 0 for r in requests)
    cached = sum(r["cached_tokens"] or 0 for r in requests)
    return {
        "parallel": n, "wall_s": wall, "gen_tokens": gen_tokens,
        "total_tps": gen_tokens / wall if wall > 0 else None,
        "decode_tps_total": _decode_tps_total(requests),
        "per_request_tps": median(r["gen_tps"] for r in requests),
        "ttft_p50_s": median(r["ttft_s"] for r in requests),
        "cached_tokens": cached,
        "cache_contaminated": any((r["cached_tokens"] or 0) > CONTAMINATION * r["prompt_tokens"]
                                  for r in requests),
        "requests": requests}


def _level(ctx: RunContext, n: int, base_slot: int, meta: dict) -> str:
    """Run one level. Returns ok, stop (ok, but do not climb), does_not_fit or error."""
    engine, guard, store = ctx.engine, ctx.guard, ctx.store
    test_key = key(ctx, "concurrency", f"p{n}")
    store.mark_running(test_key, meta)
    started = now()
    values: dict = {"parallel": n}
    slots = [base_slot + i for i in range(n)]
    try:
        try:
            engine.unload(ctx.model)
            before = take_reading(guard.gpu_index)
            # Measured ladder rungs beat the estimate (see bench.suites.guard_check); the
            # reading above stays: after_step compares against it.
            refusal, basis = guard_check(ctx, SLOT_CTX, lambda index: before, parallel=n)
            if refusal:
                store.save(test_key, "does_not_fit", {**values, "reason": refusal}, meta)
                return "does_not_fit"
            if basis:
                values["guard_basis"] = basis
            else:  # nothing known about the size: still check the reserve against 0
                values["guard_note"] = ("no memory estimate (memory/computed missing), "
                                        "VRAM not pre-checked")
                allowed, why = guard.before_step(0, before)
                if not allowed:
                    store.save(test_key, "does_not_fit", {**values, "reason": why}, meta)
                    return "does_not_fit"
            values["load_s"] = engine.load(ctx.model, SLOT_CTX, parallel=n)
            values["ctx_per_slot"] = SLOT_CTX
            prepare_counting(ctx, SLOT_CTX * n)
            prompts = [make_prompt(ctx, PROMPT_TOKENS, slot) for slot in slots]
            values.update(_measure(ctx, prompts))
            values.update(offload_fields(ctx))
            values.update(_effective(engine, n))  # a serialised level is real: stays ok
            after = take_reading(guard.gpu_index, _gpu_fraction(engine, ctx.model))
            allowed, why = guard.after_step(before, after)
            if not allowed:
                store.save(test_key, "does_not_fit", {**values, "reason": why}, meta)
                return "does_not_fit"
        except RunAborted:
            raise
        except Exception as err:
            check = reset_state(ctx, started)
            if check == "reset":
                store.save(test_key, "aborted",
                           {**values, "reason": "GPU driver reset detected during this level",
                            "error": clean_error(err)}, meta)
                raise RunAborted(f"GPU driver reset detected during {test_key}") from None
            values["driver_reset_check"] = check
            if model_missing(err):  # not a fit problem: retried by the next run
                store.save(test_key, "error", {**values, "error": clean_error(err)}, meta)
                return "error"
            if isinstance(err, EngineError) or check == "unknown":
                store.save(test_key, "does_not_fit", {**values, "reason": clean_error(err)}, meta)
                return "does_not_fit"
            store.save(test_key, "error", {**values, "error": clean_error(err)}, meta)
            return "error"
        check = reset_state(ctx, started)
        if check == "reset":
            store.save(test_key, "aborted",
                       {**values, "reason": "GPU driver reset detected during this level"}, meta)
            raise RunAborted(f"GPU driver reset detected during {test_key}")
        values["driver_reset_check"] = check
        store.save(test_key, "ok", values, meta)
        return "ok" if check == "none" else "stop"
    finally:
        for slot in slots:
            ctx.prompts.pop((slot, PROMPT_TOKENS), None)


def _gpu_fraction(engine, model):
    try:
        return engine.gpu_fraction(model)
    except Exception:
        return None


def _skip(ctx: RunContext, n: int, reason: str) -> None:
    test_key = key(ctx, "concurrency", f"p{n}")
    entry = ctx.store.get(test_key)
    if entry is None or entry["status"] not in FINAL_STATUSES:
        ctx.store.save(test_key, "skipped", {"reason": reason})


def run_concurrency(ctx: RunContext, levels: list[int] | None = None) -> None:
    """Run the levels that have no final result yet. Raises RunAborted on a driver reset."""
    if ctx.cfg.scope == "warm-only":
        return
    levels = sorted({int(n) for n in (levels or DEFAULT_LEVELS) if int(n) > 0})
    meta = step_meta(ctx)
    limit = model_context_length(ctx)
    if limit and SLOT_CTX > limit:  # one slot is longer than the model can handle
        keys = [key(ctx, "concurrency", f"p{n}") for n in levels]
        skip_beyond(ctx, [k for k in keys if (ctx.store.get(k) or {}).get("status") != "ok"],
                    limit, meta)
        return
    ctx.nonce = ctx.nonce or "c"  # prompts are tagged; speed sets its own nonce per run
    blocked: str | None = None
    base = CONC_SLOT
    ran = False
    try:
        for n in levels:
            test_key = key(ctx, "concurrency", f"p{n}")
            slot_base, base = base, base + n
            entry = ctx.store.get(test_key)
            is_forced = forced(ctx, test_key)
            if entry and entry["status"] in ("does_not_fit", "skipped") and not is_forced:
                blocked = blocked or f"level p{n} {entry['status']}"
                continue
            if n > 1 and _foreign_ollama(ctx.engine):
                _skip(ctx, n, "foreign Ollama server ignores parallel")
                continue
            if blocked and not is_forced:
                _skip(ctx, n, f"not run: {blocked}")
                continue
            if not ctx.store.should_run(test_key, ctx.cfg.force, ctx.cfg.force_tests):
                continue
            if entry and entry["status"] == "aborted" and not is_forced:
                blocked = blocked or f"level p{n} aborted"
                continue  # may have reset the driver: only --force/--force-test repeats it
            ran = True
            outcome = _level(ctx, n, slot_base, meta)
            if outcome == "error":
                return  # retried next run, higher levels stay pending
            if outcome == "does_not_fit":
                blocked = f"level p{n} did not fit"
            elif outcome == "stop":
                blocked = f"driver reset check unavailable after p{n}"
    finally:
        if ran:
            try:
                ctx.engine.unload(ctx.model)
            except Exception:
                pass
