"""Speed suite: cold start, warm run, prompt cache and (llama.cpp only) llama-bench.

Fixed parameters: temperature 0, seed 42, 256 output tokens, `cfg.reps` repetitions
(default 3), median of the repetitions.

* cold   per repetition: unload, load (load time counts), then one 512 token request.
* warm   model resident, `reps` requests with new 8,192 token prompts.
* cache  per repetition: the same new 8,192 token prompt twice; the second request is
         the result (TTFT, cached tokens), the first one is kept as `first_ttft_s`.
* llama-bench  pp512, pp8192, tg256 and, when a memory step has proven it fits, a run at
         depth = the largest `ok` memory context step.

Every request uses its own prompt (see bench.suites.make_prompt) so no engine prompt cache
can serve it by accident; only `cache` repeats a prompt on purpose.

Order: the memory suite must run before speed (runner order memory -> speed). The deep
llama-bench run needs a measured ladder rung; while no `memory/ctx-*` entry exists (full
scope, memory suite selected) the `llama-bench` step is left pending, not stored. Once the
ladder ran without any fitting rung, the step is stored with `deep: null`.
A step stored `aborted` (driver reset, or a crash recovered as aborted) is not repeated
automatically; only --force / --force-test run it again.
Every run_speed call uses a fresh prompt nonce, so a rerun against a warm server stays cold.

A driver reset (Windows) after a step, or a failing step while the reset probe says reset, stores the step as `aborted` and raises RunAborted.
When the reset probe itself fails, the step is kept and marked `driver_reset_check:
"unknown"` (a speed step is short and not the risky one; the context ladder is stricter).
"""

from __future__ import annotations

import secrets

from bench.engines.base import Sampling
from bench.engines.llamacpp import run_llama_bench
from bench.safety import take_reading
from bench.suites import (RunAborted, RunContext, clean_error, forced, key, make_prompt, median,
                          now, reset_state, step_meta)

SPEED_CTX = 10240  # holds an 8,192 token prompt plus 256 output tokens
COLD_TOKENS = 512
WARM_TOKENS = 8192
OUTPUT_TOKENS = 256
# Corpus slots: every repetition of every step gets its own, so no prompt is ever repeated.
COLD_SLOT, WARM_SLOT, CACHE_SLOT = 0, 100, 200
CONTAMINATION = 0.05  # warm run with more cached tokens than this share is flagged

STEPS = ("cold", "warm", "cache", "llama-bench")


def _sampling() -> Sampling:
    return Sampling(temperature=0.0, seed=42, max_tokens=OUTPUT_TOKENS)


def _run_fields(timing) -> dict:
    return {"ttft_s": timing.ttft_s, "prompt_tps": timing.prompt_tps,
            "gen_tps": timing.gen_tps, "prompt_tokens": timing.prompt_tokens,
            "gen_tokens": timing.gen_tokens, "cached_tokens": timing.cached_tokens}


def _summary(runs: list[dict]) -> dict:
    fields = ("ttft_s", "prompt_tps", "gen_tps", "prompt_tokens", "gen_tokens", "cached_tokens")
    values = {name: median(run.get(name) for run in runs) for name in fields}
    values["runs"] = runs
    return values


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
    prompts = [make_prompt(ctx, COLD_TOKENS, COLD_SLOT + i) for i in range(reps)]
    runs = []
    for prompt in prompts:
        state["loaded"] = False
        ctx.engine.unload(ctx.model)
        load_s = ctx.engine.load(ctx.model, SPEED_CTX)
        state["loaded"] = True
        timing = ctx.engine.complete(ctx.model, prompt, _sampling()).timing
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
    prompts = [make_prompt(ctx, WARM_TOKENS, WARM_SLOT + i) for i in range(ctx.cfg.reps)]
    runs = [_run_fields(ctx.engine.complete(ctx.model, p, _sampling()).timing) for p in prompts]
    values = _summary(runs)
    values["cache_contaminated"] = _contaminated(runs)
    return values


def _cache(ctx: RunContext, state: dict) -> dict:
    _ensure_loaded(ctx, state)
    prompts = [make_prompt(ctx, WARM_TOKENS, CACHE_SLOT + i) for i in range(ctx.cfg.reps)]
    runs, firsts = [], []
    for prompt in prompts:
        first = ctx.engine.complete(ctx.model, prompt, _sampling()).timing
        second = ctx.engine.complete(ctx.model, prompt, _sampling()).timing
        firsts.append(first.ttft_s)
        runs.append(_run_fields(second))
    values = _summary(runs)
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


def _deep_depth(ctx: RunContext) -> int | None:
    """Largest context step the memory ladder has measured as `ok` (never a guess)."""
    prefix = key(ctx, "memory", "ctx-")
    depths = [int(k[len(prefix):]) for k, v in ctx.store.data["tests"].items()
              if k.startswith(prefix) and k[len(prefix):].isdigit() and v["status"] == "ok"]
    return max(depths) if depths else None


def _llama_bench(ctx: RunContext, state: dict) -> dict:
    binary = _bench_binary(ctx)
    if binary is None:
        raise RuntimeError("no llama-bench binary known (use --llama-bench)")
    fa = "0" if getattr(ctx.engine, "meta", {}).get("flash_attn") == "off" else "1"
    common = ["-r", str(ctx.cfg.reps), "-ngl", "99", "-fa", fa]
    # The server holds the GPU; llama-bench needs it for itself.
    state["loaded"] = False
    ctx.engine.unload(ctx.model)
    rows = run_llama_bench(binary, ctx.model, ["-p", "512,8192", "-n", "256", *common])
    values = _bench_rows(rows, 0)
    values["deep"] = None
    depth = _deep_depth(ctx)
    if depth:
        used = (ctx.store.get(key(ctx, "memory", f"ctx-{depth}")) or {}).get("values", {})
        allowed, why = ctx.guard.before_step(
            int(used.get("vram_used_mb") or 0), take_reading(ctx.guard.gpu_index))
        if not allowed:
            values["deep"] = {"depth": depth, "skipped": why}
        else:
            deep_rows = run_llama_bench(
                binary, ctx.model, ["-p", "512", "-n", "256", "-d", str(depth), *common])
            values["deep"] = {"depth": depth, **_bench_rows(deep_rows, depth)}
    else:
        values["deep_reason"] = "no memory ladder step fits"
    return values


def _waits_for_memory(ctx: RunContext) -> bool:
    """True while the deep llama-bench run cannot know its depth yet (memory not run)."""
    cfg = ctx.cfg
    if cfg.scope != "full" or "memory" not in cfg.suites:
        return False
    prefix = key(ctx, "memory", "ctx-")
    return not any(k.startswith(prefix) for k in ctx.store.data["tests"])


_STEP_FUNCTIONS = {"cold": _cold, "warm": _warm, "cache": _cache, "llama-bench": _llama_bench}


def run_speed(ctx: RunContext) -> None:
    """Run the speed steps that have no final result yet. Raises RunAborted on a driver reset."""
    steps = ["warm"] if ctx.cfg.scope == "warm-only" else list(STEPS)
    if ctx.engine.name != "llamacpp":
        steps = [s for s in steps if s != "llama-bench"]
    state = {"loaded": False}
    ctx.prompts.clear()
    ctx.nonce = secrets.token_hex(3)  # fresh prompts even if this is a rerun on a warm server
    meta = step_meta(ctx)
    for step in steps:
        test_key = key(ctx, "speed", step)
        if not ctx.store.should_run(test_key, ctx.cfg.force, ctx.cfg.force_tests):
            continue
        entry = ctx.store.get(test_key)
        if entry and entry["status"] == "aborted" and not forced(ctx, test_key):
            continue  # may have reset the driver: only --force/--force-test repeats it
        if step == "llama-bench" and _waits_for_memory(ctx):
            continue  # stays pending until the memory suite has run
        ctx.store.mark_running(test_key, meta)
        started = now()
        try:
            values = _STEP_FUNCTIONS[step](ctx, state)
            check = reset_state(ctx, started)
        except RunAborted:
            raise
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
