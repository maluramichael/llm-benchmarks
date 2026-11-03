"""Speed suite: cold start, warm run, prompt cache and (llama.cpp only) llama-bench.

Fixed parameters: temperature 0, seed 42, 256 output tokens, `cfg.reps` repetitions
(default 3), median of the repetitions.

* cold   per repetition: unload, load (load time counts), then one 512 token request.
* warm   model resident, `reps` requests with new 8,192 token prompts.
* cache  per repetition: the same new 8,192 token prompt twice; the second request is
         the result (TTFT, cached tokens), the first one is kept as `first_ttft_s`.
* llama-bench  pp512, pp8192, tg256 at depth 0 (llama.cpp only, final `ok`).
* llama-bench-deep  pp512 and tg256 at depth = the largest `ok` memory rung.

Every request uses its own prompt (see bench.suites.make_prompt) so no engine prompt cache
can serve it by accident; only `cache` repeats a prompt on purpose.

Order: the memory suite should run before speed (runner order memory -> speed). The deep
step is its own step: it is neither run nor stored while no `memory/ctx-*` entry exists, or
while no rung is ok and a rung is retryable (error, aborted); a later run fills it. It is
stored `skipped` with a reason when all rungs are does_not_fit/skipped ("no context step
fits"), when no VRAM figure exists for the depth ("no memory figure for depth N", taken
from the rung, else from memory/computed), or when the guard refuses.
With --scope no-context or warm-only the deep step is stored `skipped` ("not in scope
<scope>", flag `scope_skipped`); a later --scope full run measures it.
A step stored `aborted` (driver reset, or a crash recovered as aborted) is not repeated
automatically; only --force / --force-test run it again.
Every run_speed call uses a fresh prompt nonce, so a rerun against a warm server stays cold.

A driver reset (Windows) after a step, or a failing step while the reset probe says
reset, stores the step as `aborted` and raises RunAborted.
When the reset probe itself fails, the step is kept and marked `driver_reset_check:
"unknown"` (a speed step is short and not the risky one; the context ladder is stricter).
"""

from __future__ import annotations

import secrets

from bench.engines.base import Sampling
from bench.engines.llamacpp import run_llama_bench
from bench.safety import take_reading
from bench.suites import (RunAborted, RunContext, clean_error, context_plan, forced, key,
                          make_prompt, median, now, prepare_counting, reset_state, retry_truncated,
                          fresh_prompt, scope_skip, step_meta, wants_run)

SPEED_CTX = 10240  # holds an 8,192 token prompt plus 256 output tokens
COLD_TOKENS = 512
WARM_TOKENS = 8192
OUTPUT_TOKENS = 256
# Corpus slots: every repetition of every step gets its own, so no prompt is ever repeated.
COLD_SLOT, WARM_SLOT, CACHE_SLOT = 0, 100, 200
CONTAMINATION = 0.05  # warm run with more cached tokens than this share is flagged

STEPS = ("cold", "warm", "cache", "llama-bench", "llama-bench-deep")


def _sampling() -> Sampling:
    return Sampling(temperature=0.0, seed=42, max_tokens=OUTPUT_TOKENS)


def _run_fields(timing) -> dict:
    return {"ttft_s": timing.ttft_s, "prompt_tps": timing.prompt_tps,
            "gen_tps": timing.gen_tps, "prompt_tokens": timing.prompt_tokens,
            "gen_tokens": timing.gen_tokens, "cached_tokens": timing.cached_tokens}


def _summary(runs: list[dict], exclude: tuple[str, ...] = ()) -> dict:
    fields = tuple(f for f in ("ttft_s", "prompt_tps", "gen_tps", "prompt_tokens", "gen_tokens",
                               "cached_tokens") if f not in exclude)
    values = {name: median(run.get(name) for run in runs) for name in fields}
    values["runs"] = runs
    return values


def _send(ctx: RunContext, tokens: int, slot: int):
    """One measured request with its own prompt; a prompt the engine truncated is rebuilt
    once (see bench.suites.retry_truncated). Returns (completion, prompt)."""
    return retry_truncated(
        lambda scale, retry: fresh_prompt(ctx, tokens, slot, scale, retry),
        lambda prompt: ctx.engine.complete(ctx.model, prompt, _sampling()))


def _ensure_loaded(ctx: RunContext, state: dict) -> None:
    if not state["loaded"]:
        state["load_s"] = ctx.engine.load(ctx.model, SPEED_CTX)
        state["loaded"] = True


def _contaminated(runs: list[dict]) -> bool:
    return any((r["cached_tokens"] or 0) > CONTAMINATION * r["prompt_tokens"] for r in runs)


def _cold(ctx: RunContext, state: dict) -> dict:
    # Prompts first (the tokenizer may need the model); then every repetition starts
    # with an unload. Unloading an engine that holds nothing is a no-op.
    reps = ctx.cfg.reps
    prepare_counting(ctx, SPEED_CTX)
    for i in range(reps):  # built first, while the model is loaded for counting
        make_prompt(ctx, COLD_TOKENS, COLD_SLOT + i)
    runs = []
    for i in range(reps):
        state["loaded"] = False
        ctx.engine.unload(ctx.model)
        load_s = ctx.engine.load(ctx.model, SPEED_CTX)
        state["loaded"] = True
        timing = _send(ctx, COLD_TOKENS, COLD_SLOT + i)[0].timing
        run = _run_fields(timing)
        run["load_s"] = load_s
        run["ttft_incl_load_s"] = load_s + timing.ttft_s
        runs.append(run)
    values = _summary(runs)
    values["load_s"] = median(r["load_s"] for r in runs)
    values["ttft_incl_load_s"] = median(r["ttft_incl_load_s"] for r in runs)
    values["cache_contaminated"] = _contaminated(runs)
    return values


def _warm(ctx: RunContext, state: dict) -> dict:
    _ensure_loaded(ctx, state)
    runs = [_run_fields(_send(ctx, WARM_TOKENS, WARM_SLOT + i)[0].timing)
            for i in range(ctx.cfg.reps)]
    values = _summary(runs)
    values["cache_contaminated"] = _contaminated(runs)
    return values


def _cache(ctx: RunContext, state: dict) -> dict:
    _ensure_loaded(ctx, state)
    runs, firsts = [], []
    for i in range(ctx.cfg.reps):
        first_result, prompt = _send(ctx, WARM_TOKENS, CACHE_SLOT + i)
        first = first_result.timing
        second = ctx.engine.complete(ctx.model, prompt, _sampling()).timing
        firsts.append(first.ttft_s)
        run = _run_fields(second)
        # Only the few uncached tail tokens are evaluated: the figure would not be
        # comparable with the cold and warm prompt_tps.
        del run["prompt_tps"]
        runs.append(run)
    values = _summary(runs, exclude=("prompt_tps",))
    values["first_ttft_s"] = median(firsts)
    values["cache_hit"] = (values["cached_tokens"] or 0) > 0
    return values


def _bench_binary(ctx: RunContext):
    return ctx.cfg.llama_bench or getattr(ctx.engine, "bench_bin", None)


def _bench_rows(rows: list[dict], depth: int) -> dict:
    """pp512 / pp8192 / tg256 tokens per second out of llama-bench rows at one depth."""
    out = {"pp512_tps": None, "pp8192_tps": None, "tg256_tps": None}
    for row in rows:
        if row.get("n_depth", 0) != depth:
            continue
        n_prompt, n_gen, speed = row.get("n_prompt", 0), row.get("n_gen", 0), row.get("avg_ts")
        if n_prompt == 512 and not n_gen:
            out["pp512_tps"] = speed
        elif n_prompt == 8192 and not n_gen:
            out["pp8192_tps"] = speed
        elif not n_prompt and n_gen == 256:
            out["tg256_tps"] = speed
    return out


class _Skip(Exception):
    """A step that is final without being measured (stored as `skipped`)."""


_deep_plan = context_plan  # shared with the smoke suite (long context needle)


def _required_mb(ctx: RunContext, depth: int) -> int | None:
    """VRAM the deep run needs: the measured rung, else the computed estimate, else None."""
    rung = (ctx.store.get(key(ctx, "memory", f"ctx-{depth}")) or {}).get("values", {})
    if rung.get("vram_used_mb"):
        return int(rung["vram_used_mb"])
    computed = (ctx.store.get(key(ctx, "memory", "computed")) or {}).get("values", {})
    for row in computed.get("curve", []):
        if row.get("ctx") == depth:
            return int(row["bytes"] // (1024 * 1024))
    return None


def _bench_common(ctx: RunContext, state: dict):
    binary = _bench_binary(ctx)
    if binary is None:
        raise RuntimeError("no llama-bench binary known (use --llama-bench)")
    fa = "0" if getattr(ctx.engine, "meta", {}).get("flash_attn") == "off" else "1"
    # The server holds the GPU; llama-bench needs it for itself.
    state["loaded"] = False
    ctx.engine.unload(ctx.model)
    return binary, ["-r", str(ctx.cfg.reps), "-ngl", "99", "-fa", fa]


def _llama_bench(ctx: RunContext, state: dict) -> dict:
    binary, common = _bench_common(ctx, state)
    rows = run_llama_bench(binary, ctx.model, ["-p", "512,8192", "-n", "256", *common])
    return _bench_rows(rows, 0)


def _llama_bench_deep(ctx: RunContext, state: dict) -> dict:
    kind, depth = _deep_plan(ctx)
    if kind == "skip":
        raise _Skip(depth)
    required = _required_mb(ctx, depth)
    if required is None:
        raise _Skip(f"no memory figure for depth {depth}")
    binary, common = _bench_common(ctx, state)
    allowed, why = ctx.guard.before_step(required, take_reading(ctx.guard.gpu_index))
    if not allowed:
        raise _Skip(why)
    rows = run_llama_bench(
        binary, ctx.model, ["-p", "512", "-n", "256", "-d", str(depth), *common])
    return {"depth": depth, **_bench_rows(rows, depth)}


_STEP_FUNCTIONS = {"cold": _cold, "warm": _warm, "cache": _cache,
                   "llama-bench": _llama_bench, "llama-bench-deep": _llama_bench_deep}


def run_speed(ctx: RunContext) -> None:
    """Run the speed steps that have no final result yet. Raises RunAborted on a driver reset."""
    steps = ["warm"] if ctx.cfg.scope == "warm-only" else list(STEPS)
    if ctx.engine.name != "llamacpp":
        steps = [s for s in steps if not s.startswith("llama-bench")]
    if ctx.cfg.scope != "full" and ctx.engine.name == "llamacpp" and "llama-bench-deep" not in steps:
        steps.append("llama-bench-deep")  # only to be stored as out of scope below
    state = {"loaded": False}
    ctx.prompts.clear()
    ctx.nonce = secrets.token_hex(3)  # fresh prompts even if this is a rerun on a warm server
    meta = step_meta(ctx)
    for step in steps:
        test_key = key(ctx, "speed", step)
        if not wants_run(ctx, test_key):
            continue
        entry = ctx.store.get(test_key)
        if entry and entry["status"] == "aborted" and not forced(ctx, test_key):
            continue  # may have reset the driver: only --force/--force-test repeats it
        if step == "llama-bench-deep" and ctx.cfg.scope == "warm-only":
            scope_skip(ctx, test_key, meta=meta)
            continue
        if step == "llama-bench-deep" and _deep_plan(ctx)[0] == "pending":
            if ctx.cfg.scope != "full":  # the ladder will not run in this scope
                scope_skip(ctx, test_key, meta=meta)
            continue  # else pending until the memory ladder gives a depth
        ctx.store.mark_running(test_key, meta)
        started = now()
        try:
            values = _STEP_FUNCTIONS[step](ctx, state)
            check = reset_state(ctx, started)
        except RunAborted:
            raise
        except _Skip as skip:
            ctx.store.save(test_key, "skipped", {"reason": str(skip)}, meta)
            continue
        except Exception as err:
            state["loaded"] = False  # unknown state: load again for the next step
            check = reset_state(ctx, started)  # a dead engine is how a driver reset looks
            if check == "reset":
                ctx.store.save(test_key, "aborted",
                               {"reason": "GPU driver reset detected during this step",
                                "error": clean_error(err)}, meta)
                raise RunAborted(f"GPU driver reset detected during {test_key}") from None
            ctx.store.save(test_key, "error",
                           {"error": clean_error(err), "driver_reset_check": check}, meta)
            continue
        if check == "reset":
            ctx.store.save(test_key, "aborted",
                           {"reason": "GPU driver reset detected during this step"}, meta)
            raise RunAborted(f"GPU driver reset detected during {test_key}")
        values["driver_reset_check"] = check
        ctx.store.save(test_key, "ok", values, meta)
