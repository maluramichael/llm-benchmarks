"""Run report: README.md of one run, rendered from its results.json.

`render_run_readme` is a pure function of the results (same input, same text; no clock, no
paths). `write_report` reads `<run_dir>/results.json`, checks everything that would be
published (results.json, the README text, every file in raw/) for personal data and raises
`PrivacyError` before writing anything when it finds some.

All text is plain English without em-dashes: dashes in data are replaced.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from bench import VERSION
from bench.privacy import PrivacyError, find_leaks, local_secrets

ENGINE_ORDER = ("llamacpp", "ollama")  # the chart order
CLASS_ORDER = ("S", "M", "L", "XL", "F")
CATEGORY_ORDER = ("math", "code", "json", "tool_calls", "long_context", "creative",
                  "code_explain")
STATUS_ORDER = ("ok", "does_not_fit", "skipped", "error", "aborted", "running")
SAFE_RAW = re.compile(r"^raw/[A-Za-z0-9._-]+$")
EM_DASH = "\u2014"
GIB = 1024**3


# --- small helpers -----------------------------------------------------------------------

def _cell(value) -> str:
    """One table cell: numbers rounded to three decimals, None as a plain dash."""
    if value is None or value == "":
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return str(round(value, 3))
    if isinstance(value, (list, tuple)):
        return ", ".join(_cell(v) for v in value) or "-"
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ").strip() or "-"


def _table(header: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(":--" for _ in header) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_cell(c) for c in row) + " |")
    return lines


def _gb(n_bytes) -> str:
    if not isinstance(n_bytes, (int, float)):
        return "-"
    return f"{n_bytes / GIB:.2f}"


def _natural(text: str):
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", text)]


def _dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _note(entry: dict) -> str:
    """Why a step has no figures: reason, then error, then the flags worth seeing."""
    values = _dict(entry.get("values"))
    parts = []
    for name in ("reason", "error"):
        if values.get(name):
            parts.append(str(values[name]))
    if values.get("serialized_note"):
        parts.append(str(values["serialized_note"]))
    if values.get("cache_contaminated"):
        parts.append("prompt cache contaminated")
    if values.get("driver_reset_check") == "unknown":
        parts.append("driver reset check unavailable")
    if values.get("previous_status"):
        parts.append(f"was {values['previous_status']}")
    return "; ".join(parts)


def _group(tests: dict) -> dict:
    """engine -> model -> suite -> step -> entry."""
    grouped: dict = {}
    for key, entry in tests.items():
        parts = key.split("/", 3)
        if len(parts) != 4 or not isinstance(entry, dict):
            continue
        engine, model, suite, step = parts
        grouped.setdefault(engine, {}).setdefault(model, {}).setdefault(suite, {})[step] = entry
    return grouped


def _engine_key(name: str):
    return (ENGINE_ORDER.index(name) if name in ENGINE_ORDER else len(ENGINE_ORDER), name)


def _model_key(name: str):
    return (CLASS_ORDER.index(name) if name in CLASS_ORDER else len(CLASS_ORDER), name)


def _date(results: dict) -> str:
    started = _dict(results.get("run")).get("started_at")
    if not started:
        times = sorted(e.get("started_at") for e in _dict(results.get("tests")).values()
                       if isinstance(e, dict) and e.get("started_at"))
        started = times[0] if times else None
    return str(started)[:10] if started else "unknown date"


# --- sections ----------------------------------------------------------------------------

def _system_section(results: dict) -> list[str]:
    system = _dict(results.get("system"))
    run = _dict(results.get("run"))
    rows = [("Device", run.get("device")), ("Started", run.get("started_at")),
            ("bench version", run.get("bench_version")),
            ("Operating system", " ".join(str(x) for x in (system.get("os"),
                                                           system.get("os_version")) if x)),
            ("Kernel", system.get("kernel")), ("Hardware model", system.get("model")),
            ("CPU", system.get("cpu")),
            ("Cores / threads", f"{system['cores']} / {system.get('threads')}"
             if system.get("cores") else system.get("threads")),
            ("RAM", f"{system['ram_gb']} GB" if system.get("ram_gb") else None),
            ("RAM speed", f"{system['ram_speed_mts']} MT/s" if system.get("ram_speed_mts")
             else None)]
    for position, gpu in enumerate(g for g in system.get("gpus") or [] if isinstance(g, dict)):
        text = f"{gpu.get('name')}, {gpu.get('vram_total_mb')} MiB VRAM"
        if gpu.get("driver"):
            text += f", driver {gpu['driver']}"
        if gpu.get("backend"):
            text += f", backend {gpu['backend']}"
        rows.append((f"GPU {position}", text))
    rows += [("Memory usable for models", f"{system['memory_for_models_gb']} GB"
              if system.get("memory_for_models_gb") else None),
             ("Uptime at start", f"{system['uptime_min']} min"
              if system.get("uptime_min") is not None else None),
             ("Monitors on the GPU", system.get("monitors_on_gpu")),
             ("Power", system.get("power")), ("Laptop", system.get("laptop")),
             ("Free disk", f"{system['disk_free_gb']} GB" if system.get("disk_free_gb")
              else None)]
    return ["## System", ""] + _table(["Item", "Value"], [[k, v] for k, v in rows
                                                          if v not in (None, "")]) + [""]


def _engines_section(results: dict, grouped: dict) -> list[str]:
    info = _dict(results.get("engines"))
    names = sorted(set(info) | set(grouped), key=_engine_key)
    if not names:
        return []
    shown = {"version", "backend", "kv_cache_type", "flash_attn", "thinking",
             "llama_arg_env_removed"}
    rows = []
    for name in names:
        meta = _dict(info.get(name))
        other = ", ".join(f"{k}={_cell(v)}" for k, v in sorted(meta.items())
                          if k not in shown)
        rows.append([name, meta.get("version"), meta.get("backend"), meta.get("kv_cache_type"),
                     meta.get("flash_attn"), meta.get("thinking"),
                     meta.get("llama_arg_env_removed") or "none", other])
    return ["## Engines", ""] + _table(
        ["Engine", "Version", "Backend", "KV cache type", "Flash attention", "Thinking",
         "Removed LLAMA_ARG_* variables", "Other"], rows) + [""]


def _memory_section(steps: dict) -> list[str]:
    if not steps:
        return []
    out = ["### Memory", ""]
    fit = steps.get("fit")
    if fit:
        out += [f"Not measured: {_cell(_dict(fit.get('values')).get('reason'))}", ""]
    computed = steps.get("computed")
    if computed:
        values = _dict(computed.get("values"))
        if computed.get("status") == "ok":
            out += ["Computed from the GGUF header: weights plus f16 KV cache "
                    f"({_cell(values.get('kv_bytes_per_token'))} bytes per token), "
                    "per context length. This is an upper bound where the model uses "
                    "sliding-window attention.", ""]
            out += _table(["Context", "Computed GiB"],
                          [[row.get("ctx"), _gb(row.get("bytes"))]
                           for row in values.get("curve") or [] if isinstance(row, dict)])
            if values.get("note"):
                out += ["", f"Note: {_cell(values['note'])}"]
        else:
            out += [f"Computed curve: {computed.get('status')}: {_note(computed)}"]
        out.append("")
    rungs = sorted(((int(k[4:]), v) for k, v in steps.items()
                    if k.startswith("ctx-") and k[4:].isdigit()))
    if rungs:
        out += ["Measured context ladder:", ""]
        rows = []
        for ctx, entry in rungs:
            v = _dict(entry.get("values"))
            rows.append([ctx, entry.get("status"), v.get("load_s"), v.get("vram_used_mb"),
                         v.get("kv_buffer_mb"), v.get("rs_buffer_mb"), v.get("prompt_tps"),
                         v.get("gen_tps"), v.get("ttft_s"), v.get("gpu_fraction"),
                         _note(entry)])
        out += _table(["Context", "Status", "Load s", "VRAM used MB", "KV buffer MB",
                       "RS buffer MB", "Prompt tok/s", "Gen tok/s", "TTFT s", "GPU share",
                       "Note"], rows)
        out.append("")
    return out


def _speed_section(steps: dict) -> list[str]:
    if not steps:
        return []
    out = ["### Speed", ""]
    rows = []
    for name in ("cold", "warm", "cache"):
        entry = steps.get(name)
        if not entry:
            continue
        v = _dict(entry.get("values"))
        extra = []
        if name == "cache":
            if v.get("first_ttft_s") is not None:
                extra.append(f"first request TTFT {_cell(v['first_ttft_s'])} s")
            if "cache_hit" in v:
                extra.append("cache hit" if v["cache_hit"] else "no cache hit")
        note = "; ".join(x for x in [_note(entry)] + extra if x)
        rows.append([name, entry.get("status"), v.get("ttft_s"), v.get("load_s"),
                     v.get("ttft_incl_load_s"), v.get("prompt_tps"), v.get("gen_tps"),
                     v.get("cached_tokens"), note])
    if rows:
        out += _table(["Step", "Status", "TTFT s", "Load s", "TTFT incl. load s",
                       "Prompt tok/s", "Gen tok/s", "Cached tokens", "Note"], rows) + [""]
    bench_rows = []
    for name in ("llama-bench", "llama-bench-deep"):
        entry = steps.get(name)
        if entry:
            v = _dict(entry.get("values"))
            bench_rows.append([name, entry.get("status"), v.get("depth", 0), v.get("pp512_tps"),
                               v.get("pp8192_tps"), v.get("tg256_tps"), _note(entry)])
    if bench_rows:
        out += ["llama-bench:", ""]
        out += _table(["Step", "Status", "Depth", "pp512 tok/s", "pp8192 tok/s", "tg256 tok/s",
                       "Note"], bench_rows) + [""]
    return out if len(out) > 2 else []


def _concurrency_section(steps: dict) -> list[str]:
    levels = sorted(((int(k[1:]), v) for k, v in steps.items()
                     if re.fullmatch(r"p\d+", k)), key=lambda kv: kv[0])
    if not levels:
        return []
    include_decode = any("decode_tps_total" in _dict(e.get("values")) for _, e in levels)
    header = ["Parallel", "Status", "Total tok/s"]
    if include_decode:
        header.append("decode_tps_total")
    header += ["Per request tok/s", "TTFT p50 s"]
    include_effective = any("effective_parallel" in _dict(e.get("values")) for _, e in levels)
    if include_effective:
        header += ["Effective parallel", "Effective context", "Serialized"]
    header.append("Note")
    rows = []
    for n, entry in levels:
        v = _dict(entry.get("values"))
        row = [n, entry.get("status"), v.get("total_tps")]
        if include_decode:
            row.append(v.get("decode_tps_total"))
        row += [v.get("per_request_tps"), v.get("ttft_p50_s")]
        if include_effective:
            row += [v.get("effective_parallel"), v.get("effective_ctx"), v.get("serialized")]
        row.append(_note(entry))
        rows.append(row)
    return ["### Concurrency", ""] + _table(header, rows) + [""]


def _result(entry: dict) -> str:
    status = entry.get("status")
    if status == "ok":
        passed = _dict(entry.get("values")).get("passed")
        return "not judged" if passed is None else "pass" if passed else "fail"
    return str(status)


def _smoke_rows(steps: dict) -> list[tuple]:
    rows = []
    for task, entry in steps.items():
        category = _dict(entry.get("values")).get("category") or task.rsplit("-", 1)[0]
        order = CATEGORY_ORDER.index(category) if category in CATEGORY_ORDER else len(
            CATEGORY_ORDER)
        rows.append((order, _natural(task), task, category, entry))
    return sorted(rows, key=lambda r: (r[0], r[1]))


def _smoke_totals(steps: dict) -> dict[str, list[int]]:
    totals: dict[str, list[int]] = {}
    for _, _, _, category, entry in _smoke_rows(steps):
        passed = _dict(entry.get("values")).get("passed")
        if entry.get("status") == "ok" and passed is not None:
            counts = totals.setdefault(category, [0, 0])
            counts[0] += 1 if passed else 0
            counts[1] += 1
    return totals


def _smoke_section(steps: dict) -> list[str]:
    if not steps:
        return []
    totals = _smoke_totals(steps)
    out = ["### Smoke", ""]
    if totals:
        shown = ", ".join(f"{c} {p}/{n}" for c, (p, n) in totals.items())
        out += [f"Judged tasks passed: {shown}.", ""]
    rows = []
    for _, _, task, category, entry in _smoke_rows(steps):
        v = _dict(entry.get("values"))
        raw = v.get("raw")
        link = f"[raw]({raw})" if isinstance(raw, str) and SAFE_RAW.match(raw) else "-"
        detail = _note(entry)
        if _result(entry) == "fail" and not detail:
            detail = "no reason given"
        rows.append([task, category, _result(entry), detail, link])
    # Raw cells are links: keep them unescaped by building the rows by hand.
    out += ["| Task | Category | Result | Detail | Raw answer |", "|:--|:--|:--|:--|:--|"]
    for row in rows:
        out.append("| " + " | ".join(_cell(c) if not str(c).startswith("[raw](") else c
                                     for c in row) + " |")
    out.append("")
    return out


def _model_sections(results: dict, grouped: dict) -> list[str]:
    models = _dict(results.get("models"))
    out = []
    for engine in sorted(grouped, key=_engine_key):
        for model in sorted(grouped[engine], key=_model_key):
            suites = grouped[engine][model]
            out += [f"## {engine} / {model}", ""]
            info = _dict(models.get(f"{engine}/{model}"))
            facts = [f"class {info['cls']}" if info.get("cls") else None,
                     f"quantisation {info['quant']}" if info.get("quant") else None,
                     f"{_gb(info.get('size_bytes'))} GiB weights"
                     if info.get("size_bytes") else None,
                     f"digest {info['digest']}" if info.get("digest") else None]
            facts = [f for f in facts if f]
            if facts:
                text = ", ".join(facts)
                out += [text[0].upper() + text[1:] + ".", ""]
            out += _memory_section(suites.get("memory", {}))
            out += _speed_section(suites.get("speed", {}))
            out += _concurrency_section(suites.get("concurrency", {}))
            out += _smoke_section(suites.get("smoke", {}))
    return out


def _notes_section(results: dict) -> list[str]:
    notes = results.get("notes")
    notes = [notes] if isinstance(notes, str) else [n for n in notes or [] if isinstance(n, str)]
    checks = [r for r in results.get("doctor") or [] if isinstance(r, dict)]
    out = ["## Notes", ""]
    out += [f"- {n}" for n in notes] or ["No notes."]
    if checks:
        out += ["", "Environment checks before the run:", ""]
        out += [f"- {r.get('check')} ({r.get('status')}): {r.get('text')}" for r in checks]
    return out + [""]


def _summary_section(results: dict, grouped: dict) -> list[str]:
    tests = _dict(results.get("tests"))
    counts: dict[str, int] = {}
    for entry in tests.values():
        status = _dict(entry).get("status", "?")
        counts[status] = counts.get(status, 0) + 1
    order = [s for s in STATUS_ORDER if s in counts] + sorted(set(counts) - set(STATUS_ORDER))
    out = ["## Summary", "",
           f"Tests: {len(tests)}" + (": " + ", ".join(f"{s} {counts[s]}" for s in order)
                                     if order else "."), ""]
    for engine in sorted(grouped, key=_engine_key):
        for model in sorted(grouped[engine], key=_model_key):
            suites = grouped[engine][model]
            facts = []
            rungs = [int(k[4:]) for k, v in suites.get("memory", {}).items()
                     if k.startswith("ctx-") and k[4:].isdigit() and v.get("status") == "ok"]
            if rungs:
                facts.append(f"max context on the GPU {max(rungs)} tokens")
            warm = _dict(suites.get("speed", {}).get("warm", {}).get("values"))
            if warm.get("gen_tps") is not None:
                facts.append(f"warm generation {_cell(warm['gen_tps'])} tok/s")
            if warm.get("prompt_tps") is not None:
                facts.append(f"warm prompt processing {_cell(warm['prompt_tps'])} tok/s")
            totals = _smoke_totals(suites.get("smoke", {}))
            if totals:
                passed = sum(p for p, _ in totals.values())
                judged = sum(n for _, n in totals.values())
                facts.append(f"smoke {passed}/{judged} judged tasks passed")
            out.append(f"- {engine} / {model}: " + ("; ".join(facts) if facts else "no figures"))
    return out + [""]


# --- public ------------------------------------------------------------------------------

def render_run_readme(results: dict) -> str:
    """The README of a run as Markdown text."""
    grouped = _group(_dict(results.get("tests")))
    run = _dict(results.get("run"))
    title = f"# Run: {run.get('device') or 'unknown device'} ({_date(results)})"
    lines = [title, "",
             f"Generated by bench {run.get('bench_version') or VERSION} from results.json. "
             "Do not edit this file: add notes to `notes` in results.json and run "
             "`bench report` again.", ""]
    lines += _system_section(results)
    lines += _engines_section(results, grouped)
    lines += _model_sections(results, grouped)
    lines += _notes_section(results)
    lines += _summary_section(results, grouped)
    return "\n".join(lines).replace(EM_DASH, "-").rstrip("\n") + "\n"


def _write_text(path: Path, text: str) -> None:
    from bench.store import retry_on_lock
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    retry_on_lock(lambda: os.replace(tmp, path))


def write_report(run_dir: Path) -> None:
    """Render `<run_dir>/README.md` from `<run_dir>/results.json`.

    Raises FileNotFoundError without results.json, ValueError for unreadable JSON and
    PrivacyError (nothing written) when results.json, the README text or a raw answer
    holds personal data (host or user name, user path, IP or MAC address).
    """
    run_dir = Path(run_dir)
    results_path = run_dir / "results.json"
    text = results_path.read_text(encoding="utf-8-sig")
    try:
        results = json.loads(text)
    except ValueError as err:
        raise ValueError(f"{results_path.name} is not valid JSON: {err}") from None
    if not isinstance(results, dict):
        raise ValueError(f"{results_path.name} does not contain a JSON object")
    readme = render_run_readme(results)
    secrets = local_secrets()
    leaks = [f"results.json: {leak}" for leak in find_leaks(text, secrets)]
    leaks += [f"README.md: {leak}" for leak in find_leaks(readme, secrets)]
    raw_dir = run_dir / "raw"
    if raw_dir.is_dir():
        for path in sorted(raw_dir.iterdir()):
            if path.is_file():
                content = path.read_text(encoding="utf-8-sig", errors="replace")
                leaks += [f"raw/{path.name}: {leak}" for leak in find_leaks(content, secrets)]
    if leaks:
        raise PrivacyError(leaks)
    _write_text(run_dir / "README.md", readme)
