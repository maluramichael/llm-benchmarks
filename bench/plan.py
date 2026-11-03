"""Plan: what fits into memory, what has to be downloaded, how long it takes.

`make_plan` returns one entry per engine and model:

    engine, model, cls, quant     what it is
    available                     the model could be found or resolved at all
    source                        local, mirror (copied by the run) or download
    fits, reason                  fits at 8k (None: unknown, VRAM figure unreliable) context plus the safety reserve? and why
    need_gb, available_gb         the numbers behind it in GiB (None when unknown)
    download_gb                   0 when the file is there, None when it cannot be resolved
    needs_consent                 download above 20 GB and no --yes
    minutes_low, minutes_high     time estimate for the selected suites (rough)
    new                           nothing measured for it in this run's results yet
    notes                         caveats of the computed memory curve

The fit check uses the computed memory curve (GGUF header: weights plus f16 KV cache) at 8k
context. That curve is an upper bound for sliding-window attention and leaves out the
recurrent state of hybrid models; the notes say so. Without a GGUF file only the weights are
judged. Nothing is downloaded or copied: models are resolved in dry-run mode.
"""

from __future__ import annotations

import dataclasses
import math
import sys
from typing import Callable

from bench import models
from bench.gguf import memory_curve, read_metadata
from bench.safety import Guard
from bench.store import FINAL
from bench.suites.memory import CACHE_TYPE_BYTES, _flags

MIB = 1024 * 1024
GIB = 1024**3
FIT_CTX = 8192
CONSENT_GB = models.CONSENT_BYTES / 1e9

NOTE_UPPER_BOUND = ("memory curve is an upper bound: sliding-window attention keeps a smaller "
                    "KV cache than the curve assumes")
NOTE_RECURRENT = ("recurrent state of this hybrid model is not part of the memory curve; "
                  "expect some hundred MB more")

# Minutes per step, measured on an RTX 3090 with a 4B model and scaled with the weights
# (GB). low: fast card, small model. high: slow card, large model, cold caches.
LOAD_LOW, LOAD_HIGH = 0.15, 0.4  # minutes per GB of weights per model load
FILL_TPS_LOW, FILL_TPS_HIGH = 8000.0, 1200.0  # prompt tokens per second for a ladder rung
SPEED_LOW, SPEED_HIGH = 2.5, 7.0
LLAMA_BENCH_LOW, LLAMA_BENCH_HIGH = 1.0, 4.0
SMOKE_LOW, SMOKE_HIGH = 1.5, 6.0
CONCURRENCY_LOW, CONCURRENCY_HIGH = 2.0, 8.0
PER_GB_LOW, PER_GB_HIGH = 0.05, 0.3  # extra minutes per GB of weights for speed and smoke


def _gib(value) -> str:
    """A size in binary units (GiB, 1024^3 bytes). Download sizes stay decimal GB."""
    if value is None:
        return "unknown"
    text = f"{value:.1f}"
    return (text[:-2] if text.endswith(".0") else text) + " GiB"


def vram_is_reliable(gpu: dict | None) -> bool:
    """Windows reports at most 4 GB for AMD and Intel cards (WMI): that figure is no reason
    to say a model does not fit."""
    return bool(gpu) and (gpu.get("vendor") in ("nvidia", "apple") or sys.platform != "win32")


def _gpu(system: dict, index: int) -> dict | None:
    gpus = [g for g in system.get("gpus") or [] if isinstance(g, dict)]
    for gpu in gpus:
        if gpu.get("index") == index:
            return gpu
    return gpus[index] if 0 <= index < len(gpus) else None


def _available_mb(system: dict, index: int) -> tuple[int | None, int | None]:
    """(available MiB, total MiB) for models: free VRAM of the card when known, else its
    total; system memory without a GPU; (None, None) when nothing is known."""
    gpu = _gpu(system, index)
    if gpu and (gpu.get("vram_total_mb") or 0) > 0:
        total = int(gpu["vram_total_mb"])
        free = gpu.get("vram_free_mb")
        return (int(free) if isinstance(free, (int, float)) else total), total
    if not system.get("gpus"):
        ram = system.get("memory_for_models_gb") or system.get("ram_gb")
        if ram:
            return int(ram * 1024), int(ram * 1024)
    return None, None


def _needs(ref, ctx_steps: list[int]) -> tuple[dict[int, int] | None, int, list[str]]:
    """Bytes per context step from the GGUF header, the weights, and the caveats.

    The curve is None when there is no readable GGUF (then only the weights count)."""
    weights = ref.size_bytes or 0
    if ref.gguf_path is None:
        return None, weights, []
    try:
        meta = read_metadata(ref.gguf_path)
        curve = dict(memory_curve(meta, weights, ctx_steps, CACHE_TYPE_BYTES))
    except (OSError, ValueError, KeyError, EOFError):
        return None, weights, []
    sliding, recurrent = _flags(meta)
    notes = []
    if sliding:
        notes.append(NOTE_UPPER_BOUND)
    if recurrent:
        notes.append(NOTE_RECURRENT)
    return curve, weights, notes


def _minutes(cfg, engine: str, weights_gb: float, rungs: list[int]) -> tuple[float, float]:
    low = high = 0.0
    suites = set(cfg.suites)
    load_low, load_high = LOAD_LOW * weights_gb, LOAD_HIGH * weights_gb
    if "memory" in suites and cfg.scope == "full":
        for ctx in rungs:
            low += load_low + ctx / FILL_TPS_LOW / 60
            high += load_high + ctx / FILL_TPS_HIGH / 60
    if "speed" in suites:
        factor = 0.4 if cfg.scope == "warm-only" else 1.0
        low += (SPEED_LOW + PER_GB_LOW * weights_gb) * factor
        high += (SPEED_HIGH + PER_GB_HIGH * weights_gb * 4) * factor
        if engine == "llamacpp" and cfg.scope != "warm-only":
            low += LLAMA_BENCH_LOW
            high += LLAMA_BENCH_HIGH
    if "smoke" in suites:
        low += SMOKE_LOW + PER_GB_LOW * weights_gb
        high += SMOKE_HIGH + PER_GB_HIGH * weights_gb * 2
    if "concurrency" in suites and cfg.scope != "warm-only":
        low += CONCURRENCY_LOW + load_low
        high += CONCURRENCY_HIGH + load_high
    return low, high


def _is_new(store, engine: str, model_id: str) -> bool:
    if store is None:
        return True
    prefix = f"{engine}/{model_id}/"
    return not any(k.startswith(prefix) and v.get("status") in FINAL
                   for k, v in store.data["tests"].items())


def make_plan(cfg, system: dict, *, store=None,
              resolve: Callable | None = None) -> list[dict]:
    """The plan for `cfg.engines` x `cfg.models`, in that order (engines outer)."""
    resolve = resolve or models.resolve
    dry = dataclasses.replace(cfg, dry_run=True)
    index = cfg.gpu or 0
    available, total = _available_mb(system, index)
    reserve = Guard(cfg, index, total or 0).reserve_mb()
    steps = sorted({int(s) for s in cfg.ctx_steps if int(s) > 0} | {FIT_CTX})
    entries = []
    for engine in cfg.engines:
        for model_id in cfg.models:
            entry = {"engine": engine, "model": model_id, "cls": model_id, "quant": None,
                     "available": True, "source": None, "fits": False, "reason": "", "need_gb": None, "available_gb": None,
                     "download_gb": None, "needs_consent": False, "minutes_low": 0,
                     "minutes_high": 0, "new": _is_new(store, engine, model_id), "notes": []}
            entries.append(entry)
            try:
                ref = resolve(model_id, engine, dry)
            except models.ModelError as err:
                entry["available"] = False
                entry["reason"] = str(err)
                continue
            entry["cls"], entry["quant"] = ref.cls, ref.quant
            download = 0.0 if ref.gguf_path is not None else (ref.size_bytes or 0) / 1e9
            entry["source"] = ("local" if ref.gguf_path is not None else
                               "mirror" if cfg.mirror and ref.digest else "download")
            entry["download_gb"] = round(download, 1)
            entry["needs_consent"] = download > CONSENT_GB and not cfg.yes
            curve, weights, notes = _needs(ref, steps)
            entry["notes"] = notes
            weights_gb = weights / GIB
            need_bytes = curve[FIT_CTX] if curve else weights
            need_gb = need_bytes / GIB
            entry["need_gb"] = round(need_gb, 1)
            what = (f"{_gib(need_gb)} at 8k context (weights {_gib(weights_gb)} plus KV cache "
                    f"{_gib(need_gb - weights_gb)})") if curve else (
                f"{_gib(weights_gb)} (weights only, KV cache unknown)")
            gpu = _gpu(system, index)
            if available is not None and gpu and not vram_is_reliable(gpu):
                entry["fits"] = None
                entry["available_gb"] = round(available / 1024, 1)
                entry["reason"] = (f"unknown: needs {what}, but the VRAM figure of this card "
                                   f"({_gib(available / 1024)}) is unreliable on Windows; the run "
                                   "finds out")
                rungs = list(cfg.ctx_steps)
            elif available is None:
                entry["fits"] = True
                entry["reason"] = f"needs {what}; memory size unknown, not checked"
                rungs = list(cfg.ctx_steps)
            else:
                entry["available_gb"] = round(available / 1024, 1)
                fits = need_bytes / MIB + reserve <= available
                entry["fits"] = fits
                verdict = "fits" if fits else "does not fit"
                entry["reason"] = (f"{verdict}: needs {what} plus {_gib(reserve / 1024)} safety "
                                   f"reserve, {_gib(available / 1024)} available")
                rungs = [c for c in cfg.ctx_steps
                         if ((curve[c] if curve and c in curve else need_bytes) / MIB
                             + reserve <= available)] if curve else (
                    list(cfg.ctx_steps) if fits else [])
            low, high = _minutes(cfg, engine, weights_gb, rungs)
            entry["minutes_low"] = max(1, math.ceil(low))
            entry["minutes_high"] = max(entry["minutes_low"], math.ceil(high))
    return entries


def format_plan(entries: list[dict]) -> str:
    """The plan as plain text lines for the console."""
    lines = [f"{'engine':9} {'model':6} {'quant':8} {'result':12} {'download':>21} "
             f"{'time':>11}  {'state'}"]
    low = high = 0
    for e in entries:
        result = ("unavailable" if not e.get("available", True) else "unknown" if e["fits"] is None
                  else "fits" if e["fits"] else "does not fit")
        if e["download_gb"] is None:
            download = "-"
        elif e["download_gb"] == 0:
            download = "local"
        else:
            download = ("mirror copy " if e.get("source") == "mirror" else "") + \
                f"{e['download_gb']:.1f} GB"
        runs = e["fits"] is not False and e.get("available", True)
        time = f"{e['minutes_low']}-{e['minutes_high']} min" if runs else "-"
        state = "new" if e["new"] else "measured"
        if e["needs_consent"]:
            state += ", needs --yes"
        lines.append(f"{e['engine']:9} {e['model']:6} {(e['quant'] or '-'):8} {result:12} "
                     f"{download:>21} {time:>11}  {state}")
        if runs:
            low += e["minutes_low"]
            high += e["minutes_high"]
    lines.append("")
    for e in entries:
        lines.append(f"{e['engine']}/{e['model']}: {e['reason']}")
        for note in e["notes"]:
            lines.append(f"  note: {note}")
    lines.append("")
    lines.append(f"Estimated total for what fits: {low}-{high} minutes "
                 "(rough; a rerun skips everything already measured).")
    return "\n".join(lines)
