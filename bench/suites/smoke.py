"""Smoke suite: 31 small tasks per model, judged by the rules in bench.checkers.

Tasks live in `bench/data/tasks/<category>.toml` (see `load_tasks`). Every task runs once
with seed 42 and the sampling of its file; stored under `smoke/<task-id>` as
`values = {passed, reason, raw, category, ...}` and the raw answer in
`raw/<engine>-<model>-<task>.txt` (scrubbed, file name safe on Windows).

Status (ruling R1): a task that was run and judged is `ok`, with `passed` True or False
(None for categories nobody judges: creative, code_explain). A wrong answer, and a timeout
of generated code (reason "timeout"), are `ok` with `passed` False. `error` only when the
engine call failed (retried next run), `aborted` after a driver reset (never repeated
automatically, only with --force / --force-test), `skipped` for code tasks under
--no-code-exec (the engine is not even asked) and for long context when no context step fits.

Before the first load the guard is asked for the memory estimate at 8,192 tokens: refused
(or a smaller memory rung is does_not_fit) means every task still to run is stored `skipped`
with "does not fit (guard): <reason>". Each judged task records gpu_fraction and, below
100 %, partial_offload.

Thinking is off unless `--thinking on`: Ollama `think=false`, llama.cpp
`chat_template_kwargs.enable_thinking=false` (both done by the engine adapters). The state is
stored in `meta.thinking`. Only the adapter's content field is judged, never thinking text.

Long context: the context is loaded with `min(32768, largest ok memory rung)` tokens; the
needle document gets that minus 512 (question, template) minus the answer's max_tokens, and
94 % of it for engines that only estimate token counts (Ollama). A prompt the engine
truncated is rebuilt once, truncated again it is stored `error` (never judged). The needle tasks stay pending (not stored) while
no `memory/ctx-*` entry exists or a rung is still retryable, so the runner order is
memory -> speed -> smoke; a later run fills them in. When all rungs are does_not_fit or
skipped the needle tasks are stored `skipped`. With --scope no-context or warm-only there
is no ladder: the needle tasks are stored `skipped` ("not in scope <scope>", flag
`scope_skipped`) and a later run with --scope full measures them after all.

Code tasks execute model generated Python (not a sandbox, see bench.checkers).
"""

from __future__ import annotations

import fnmatch
import json
import re
import tomllib
from pathlib import Path

from bench.checkers import check, to_store
from bench.corpus import needle_text
from bench.engines.base import Sampling
from bench.privacy import local_secrets, scrub
from bench.safety import take_reading
from bench.suites import (RunAborted, RunContext, clean_error, context_plan, exact_counts,
                          forced, guard_check, key, model_context_length, now, offload_fields,
                          reset_state, retry_truncated, scope_skip, skip_beyond, skip_for_guard,
                          step_meta, wants_run)

TASKS_DIR = Path(__file__).resolve().parent.parent / "data" / "tasks"
CATEGORY_COUNTS = {"math": 8, "code": 6, "json": 5, "tool_calls": 5, "long_context": 3,
                   "creative": 2, "code_explain": 2}
SMOKE_CTX = 8192  # context for all tasks except long context
MAX_NEEDLE_CTX = 32768
NEEDLE_MARGIN = 512  # tokens kept free for question and chat template (plus max_tokens)
NEEDLE_SHARE_ESTIMATED = 0.94  # share of the document budget used when counts are estimates
SEED = 42


def load_tasks(pattern: str | None) -> list[dict]:
    """All tasks in category order; `pattern` (comma separated fnmatch on task id or
    category) keeps only the matching ones."""
    patterns = [p.strip() for p in (pattern or "").split(",") if p.strip()]
    tasks = []
    for category in CATEGORY_COUNTS:
        path = TASKS_DIR / f"{category}.toml"
        with path.open("rb") as handle:
            tasks.extend(tomllib.load(handle)["task"])
    if not patterns:
        return tasks
    return [t for t in tasks if any(fnmatch.fnmatchcase(t["id"], p)
                                    or fnmatch.fnmatchcase(t["category"], p) for p in patterns)]


def _clean(text: str) -> str:
    return scrub(text, local_secrets())


def raw_name(ctx: RunContext, task_id: str) -> str:
    """`<engine>-<model>-<task>.txt` with every character outside [A-Za-z0-9._-] replaced."""
    name = f"{ctx.engine.name}-{ctx.model.id}-{task_id}.txt"
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def _write_raw(ctx: RunContext, task_id: str, result) -> str:
    text = result.text
    if result.tool_calls:
        text += "\n\n[tool_calls]\n" + json.dumps(result.tool_calls, indent=2, sort_keys=True)
    name = raw_name(ctx, task_id)
    ctx.raw_dir.mkdir(parents=True, exist_ok=True)
    (ctx.raw_dir / name).write_text(_clean(text), encoding="utf-8", newline="\n")
    return f"raw/{name}"


def _ensure_loaded(ctx: RunContext, state: dict, wanted: int) -> None:
    if state["ctx"] is None or state["ctx"] < wanted:
        state["ctx"] = None  # unknown if the load fails half way
        ctx.engine.load(ctx.model, wanted)
        state["ctx"] = wanted


def _document_tokens(ctx: RunContext, task: dict, length: int, scale: float = 1.0) -> int:
    """Length of the needle document for a context of `length` tokens: the context holds
    the document, NEEDLE_MARGIN for question and template, and the answer (max_tokens).
    Engines that estimate token counts (Ollama) get a further 6 % off, `scale` shrinks it
    after a truncation whose real length was unknown."""
    room = length - NEEDLE_MARGIN - task["sampling"]["max_tokens"]
    share = 1.0 if exact_counts(ctx) else NEEDLE_SHARE_ESTIMATED
    return max(1, int(room * share * scale))


def _messages(ctx: RunContext, task: dict, length: int | None,
              scale: float = 1.0) -> list[dict]:
    messages = [dict(m) for m in task["messages"]]
    if task["category"] == "long_context":
        count = lambda t: ctx.engine.count_tokens(ctx.model, t)  # noqa: E731
        document = needle_text(ctx.corpus_text, _document_tokens(ctx, task, length, scale),
                               task["depth"], task["needle"], count)
        messages[-1]["content"] = messages[-1]["content"].replace("{document}", document)
    return messages


def _sampling(task: dict) -> Sampling:
    given = task["sampling"]
    return Sampling(temperature=given["temperature"], top_p=given.get("top_p", 1.0),
                    seed=SEED, max_tokens=given["max_tokens"])


def run_smoke(ctx: RunContext) -> None:
    """Run the smoke tasks that have no final result yet. Raises RunAborted on a driver reset."""
    cfg = ctx.cfg
    tasks = load_tasks(",".join(cfg.tasks) or None)
    meta = {**step_meta(ctx), "thinking": bool(cfg.thinking)}
    state = {"ctx": None}
    pending = []
    guard_basis = None
    for task in tasks:
        test_key = key(ctx, "smoke", task["id"])
        entry = ctx.store.get(test_key)
        if (wants_run(ctx, test_key) and not (task["check"]["type"] == "code" and cfg.no_code_exec)
                and not (entry and entry["status"] == "aborted" and not forced(ctx, test_key))):
            pending.append(task)
    limit = model_context_length(ctx)
    if pending and limit and SMOKE_CTX > limit:  # the model cannot run this context at all
        for task in pending:
            skip_beyond(ctx, [key(ctx, "smoke", task["id"])], limit, meta,
                        {"category": task["category"]})
        pending_ids = {t["id"] for t in pending}
        tasks = [t for t in tasks if t["id"] not in pending_ids]
        pending = []
    if pending:  # the guard applies to every suite, not only to memory
        refusal, guard_basis = guard_check(ctx, SMOKE_CTX, take_reading)
        if refusal:
            for task in pending:
                skip_for_guard(ctx, [key(ctx, "smoke", task["id"])], refusal, meta,
                               {"category": task["category"]})
            pending_ids = {t["id"] for t in pending}
            tasks = [t for t in tasks if t["id"] not in pending_ids]
    for task in tasks:
        test_key = key(ctx, "smoke", task["id"])
        if not wants_run(ctx, test_key):
            continue
        entry = ctx.store.get(test_key)
        if entry and entry["status"] == "aborted" and not forced(ctx, test_key):
            continue  # may have reset the driver: only --force/--force-test repeats it
        if task["check"]["type"] == "code" and cfg.no_code_exec:
            ctx.store.save(test_key, "skipped",
                           {"reason": "code execution disabled (--no-code-exec)",
                            "category": task["category"]}, meta)
            continue
        length = None
        if task["category"] == "long_context":
            kind, detail = context_plan(ctx)
            if kind == "pending":
                if cfg.scope != "full":  # the ladder will not run in this scope
                    scope_skip(ctx, test_key, {"category": task["category"]}, meta)
                continue  # else pending until the memory ladder gives a context size
            if kind == "skip":
                ctx.store.save(test_key, "skipped",
                               {"reason": detail, "category": task["category"]}, meta)
                continue
            length = min(MAX_NEEDLE_CTX, detail, limit or MAX_NEEDLE_CTX)
        _run_task(ctx, task, test_key, length, state, meta, guard_basis)


def _run_task(ctx: RunContext, task: dict, test_key: str, length: int | None, state: dict,
              meta: dict, guard_basis: str | None = None) -> None:
    ctx.store.mark_running(test_key, meta)
    started = now()
    try:
        _ensure_loaded(ctx, state, length or SMOKE_CTX)
        # A prompt the engine truncated is rebuilt once; truncated twice is an error, so
        # a needle task is never judged on a document with its front part cut off.
        result, _ = retry_truncated(
            lambda scale, retry: _messages(ctx, task, length, scale),
            lambda messages: ctx.engine.chat(
                ctx.model, messages, _sampling(task), tools=task.get("tools"),
                json_schema=task.get("json_schema"), thinking=bool(ctx.cfg.thinking)))
        check_state = reset_state(ctx, started)
    except RunAborted:
        raise
    except Exception as err:
        state["ctx"] = None  # unknown engine state: load again for the next task
        check_state = reset_state(ctx, started)  # a dead engine is how a driver reset looks
        if check_state == "reset":
            ctx.store.save(test_key, "aborted",
                           {"reason": "GPU driver reset detected during this task",
                            "error": clean_error(err), "category": task["category"]}, meta)
            raise RunAborted(f"GPU driver reset detected during {test_key}") from None
        ctx.store.save(test_key, "error",
                       {"error": clean_error(err), "driver_reset_check": check_state,
                        "category": task["category"]}, meta)
        return
    if check_state == "reset":
        ctx.store.save(test_key, "aborted",
                       {"reason": "GPU driver reset detected during this task",
                        "category": task["category"]}, meta)
        raise RunAborted(f"GPU driver reset detected during {test_key}")
    try:
        outcome, reason = check(task, result, ctx.cfg)
    except Exception as err:  # e.g. RecursionError in compile(): judged, not an engine error
        outcome, reason = "fail", f"checker error: {type(err).__name__}"
    status, passed = to_store(outcome, reason)
    values = {"passed": passed, "reason": _clean(reason), "category": task["category"],
              "raw": _write_raw(ctx, task["id"], result), "gen_tokens": result.timing.gen_tokens,
              "driver_reset_check": check_state}
    values.update(offload_fields(ctx))
    if guard_basis:
        values["guard_basis"] = guard_basis
    if length is not None:
        values["context_tokens"] = length
    ctx.store.save(test_key, status, values, meta)
