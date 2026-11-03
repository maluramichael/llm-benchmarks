"""Build the README charts and the run list from runs/*/results.json.

The result files are the only source of truth. This script reads every
`runs/<date>-<device>/results.json` (and lists the old `runs/*.md` reports as
"old prompt"), then

- rewrites the parts of README.md between the <!-- generated:... --> markers,
- writes one light and one dark SVG per chart into charts/.

Same input, same output: no timestamps, no randomness, stable sort orders.
Standard library only (the privacy scan comes from bench/privacy.py, which is
standard library only too), so CI needs nothing but python3.

    python3 scripts/build.py          # write files
    python3 scripts/build.py --check  # exit 1 if anything is out of date
"""

from __future__ import annotations

import json
import math
import re
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.privacy import find_leaks, local_secrets  # noqa: E402

SCHEMA_VERSION = 1
STATUSES = {"ok", "does_not_fit", "skipped", "error", "aborted", "running"}
CLASS_ORDER = ["S", "M", "L", "XL", "F"]
CATEGORY_ORDER = ["math", "code", "json", "tool_calls", "long_context", "creative", "code_explain"]

# Color follows the engine, never its rank. New engines get the next free slot
# here, by hand, so existing colors never move.
ENGINES = ["llamacpp", "ollama", "mlx"]
ENGINE_NAMES = {"llamacpp": "llama.cpp", "ollama": "Ollama", "mlx": "MLX"}
SERIES = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a"],
    "dark": ["#3987e5", "#d95926", "#199e70"],
}
# One blue ramp for the smoke grid (five bins of pass rate), checked with the
# dataviz validator in --ordinal mode against the chart surface of each theme.
RAMP = {
    "light": ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"],
    "dark": ["#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4"],
}
# Text on a ramp step: True means dark ink, False means white.
RAMP_DARK_INK = {"light": [True, True, False, False, False], "dark": [False, False, True, True, True]}
THEMES = {
    "light": {
        "surface": "#ffffff",
        "border": "#d1d9e0",
        "primary": "#1f2328",
        "secondary": "#59636e",
        "muted": "#818b98",
        "grid": "#eaeef2",
        "baseline": "#c8d1da",
        "limit": "#b0bac5",
    },
    "dark": {
        "surface": "#0d1117",
        "border": "#3d444d",
        "primary": "#f0f6fc",
        "secondary": "#9198a1",
        "muted": "#6e7681",
        "grid": "#1d232b",
        "baseline": "#3d444d",
        "limit": "#4d5560",
    },
}
FONT = 'system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif'
STATUS_LABEL = {"does_not_fit": "does not fit", "skipped": "skipped", "error": "error",
                "aborted": "aborted", "running": "not finished"}
MEMORY_LIMITS_GB = [8, 12, 16, 24]
DASHES = {"S": "", "M": "7 4", "L": "2 4", "XL": "9 3 2 3", "F": "12 4"}
GIB_MB = 1024.0


# --------------------------------------------------------------------- text helpers


def clean(value) -> str:
    """Text that ends up in a generated file: no em-dashes, whatever the data says."""
    return str(value).replace("\u2014", "-")


def esc(value) -> str:
    return escape(clean(value), quote=True)


def md_text(value) -> str:
    return clean(value).replace("[", "(").replace("]", ")").replace("\n", " ").strip()


def text_width(s: str, size: float) -> float:
    """Rough width of system-ui text, good enough for layout decisions."""
    return len(s) * size * 0.56


def wrap(text: str, size: float, width: float) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if line and text_width(trial, size) > width:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)
    return lines


def engine_name(engine: str) -> str:
    return ENGINE_NAMES.get(engine, engine)


def engine_color(engine: str, theme: str) -> str:
    return SERIES[theme][ENGINES.index(engine)] if engine in ENGINES else THEMES[theme]["muted"]


def engine_rank(engine: str) -> tuple:
    return (ENGINES.index(engine), engine) if engine in ENGINES else (len(ENGINES), engine)


def model_rank(model: str) -> tuple:
    return (CLASS_ORDER.index(model), model) if model in CLASS_ORDER else (len(CLASS_ORDER), model)


# --------------------------------------------------------------------- loading


def validate(results) -> list[str]:
    """Schema errors of one results.json, empty list when it is usable."""
    if not isinstance(results, dict):
        return ["top level is not an object"]
    errors = []
    if results.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version is {results.get('schema_version')!r}, expected {SCHEMA_VERSION}")
    if results.get("legacy") is True:
        # Written by the migration of the old reports: nothing but these four fields.
        for field in ("device", "date"):
            if not isinstance(results.get(field), str) or not results[field].strip():
                errors.append(f"legacy file needs a text field '{field}'")
        return errors
    run = results.get("run")
    if not isinstance(run, dict) or not isinstance(run.get("device"), str) or not run["device"].strip():
        errors.append("run.device is missing")
    tests = results.get("tests")
    if not isinstance(tests, dict):
        errors.append("'tests' is missing or not an object")
        return errors
    for key, entry in tests.items():
        if len(key.split("/", 3)) != 4:
            errors.append(f"test key {key!r} is not engine/model/suite/step")
            continue
        if key.split("/", 1)[0] not in ENGINES:
            errors.append(f"test {key!r}: unknown engine (add it to ENGINES with a color)")
        if not isinstance(entry, dict):
            errors.append(f"test {key!r} is not an object")
        elif entry.get("status") not in STATUSES:
            errors.append(f"test {key!r}: unknown status {entry.get('status')!r}")
        elif not isinstance(entry.get("values", {}), dict):
            errors.append(f"test {key!r}: 'values' is not an object")
    return errors


def _date_of(name: str, results: dict) -> str:
    m = re.match(r"^(\d{4}-\d{2}-\d{2})-", name)
    if m:
        return m.group(1)
    started = (results.get("run") or {}).get("started_at") or results.get("date") or ""
    return str(started)[:10]


def load_runs(root: Path = ROOT) -> list[dict]:
    """Every runs/*/results.json as a normalized dict, oldest first.

    Keys: dir, date, device, legacy, tests. Raises SystemExit naming the folder
    (never the content) when a file does not pass validate().
    """
    runs = []
    for path in sorted((root / "runs").glob("*/results.json")):
        name = path.parent.name
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise SystemExit(f"runs/{name}/results.json: not readable JSON") from None
        errors = validate(data)
        if errors:
            raise SystemExit(f"runs/{name}/results.json: " + "; ".join(errors[:5]))
        legacy = data.get("legacy") is True
        runs.append({
            "dir": name,
            "date": _date_of(name, data),
            "device": clean(data["device"] if legacy else data["run"]["device"]).strip(),
            "legacy": legacy,
            "tests": {} if legacy else data["tests"],
        })
    runs.sort(key=lambda r: (r["date"], r["dir"]))
    return runs


def old_reports(root: Path) -> list[dict]:
    """The pre-harness runs/*.md reports, listed as 'old prompt'."""
    found = []
    for path in sorted((root / "runs").glob("*.md")):
        lines = path.read_text(encoding="utf-8").splitlines()
        m = re.match(r"^# (\d{4}-\d{2}-\d{2}): (.+)$", lines[0] if lines else "")
        if m:
            found.append({"dir": path.name, "date": m.group(1), "device": clean(m.group(2)).strip(),
                          "legacy": True, "tests": {}, "link": f"runs/{path.name}"})
    return found


# --------------------------------------------------------------------- facts


class Facts:
    """The newest entry per (device, engine, model, suite, step) over all runs.

    A newer run replaces an older one. A newer `running` or `aborted` entry
    (a crash leftover) does not hide an older final result.
    """

    def __init__(self, runs: list[dict]):
        self.entries: dict[tuple, dict] = {}
        for run in sorted((r for r in runs if not r["legacy"]), key=lambda r: (r["date"], r["dir"])):
            for key, entry in run["tests"].items():
                engine, model, suite, step = key.split("/", 3)
                full = (run["device"], engine, model, suite, step)
                if full in self.entries and entry.get("status") in ("running", "aborted"):
                    continue
                self.entries[full] = entry

    def get(self, device, engine, model, suite, step):
        return self.entries.get((device, engine, model, suite, step))

    def steps(self, device, engine, model, suite) -> dict:
        return {k[4]: v for k, v in self.entries.items() if k[:4] == (device, engine, model, suite)}

    def combos(self) -> list[tuple]:
        return sorted({k[:3] for k in self.entries}, key=lambda c: (c[0], model_rank(c[2]), engine_rank(c[1])))

    def devices(self) -> list[str]:
        """Fastest device first, by its best warm generation speed on class S."""
        best: dict[str, float] = {}
        for (device, _e, model, suite, step), entry in self.entries.items():
            best.setdefault(device, 0.0)
            if (model, suite, step) == ("S", "speed", "warm") and entry.get("status") == "ok":
                best[device] = max(best[device], number((entry.get("values") or {}).get("gen_tps")) or 0.0)
        return sorted(best, key=lambda d: (-best[d], d))


def number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def gap_label(entry) -> str:
    if entry is None:
        return "not measured"
    return STATUS_LABEL.get(entry.get("status"), "not measured")


class Cell:
    def __init__(self, value=None, label="", hatched=False, suffix="", tip=""):
        self.value, self.label, self.hatched, self.suffix, self.tip = value, label, hatched, suffix, tip


def fit_gap(facts: Facts, device, engine, model) -> Cell | None:
    fit = facts.get(device, engine, model, "memory", "fit")
    if fit and fit.get("status") == "does_not_fit":
        return Cell(label="does not fit")
    return None


def speed_cell(field: str, step: str = "warm"):
    def build(facts: Facts, device, model, engine):
        entry = facts.get(device, engine, model, "speed", step)
        if entry is None:
            return fit_gap(facts, device, engine, model)
        value = number((entry.get("values") or {}).get(field)) if entry.get("status") == "ok" else None
        return Cell(value) if value is not None else Cell(label=gap_label(entry))
    return build


def context_cell(facts: Facts, device, model, engine):
    rungs = {}
    for step, entry in facts.steps(device, engine, model, "memory").items():
        if step.startswith("ctx-") and step[4:].isdigit():
            rungs[int(step[4:])] = entry
    if not rungs:
        return fit_gap(facts, device, engine, model)
    ok = [n for n, e in rungs.items() if e.get("status") == "ok"]
    if not ok:
        worst = rungs[min(rungs)]
        return Cell(label=gap_label(worst))
    top = max(ok)
    plus = not any(n > top and e.get("status") != "ok" for n, e in rungs.items())
    return Cell(float(top), suffix="+" if plus else "")


def concurrency_levels(facts: Facts, device) -> list[int]:
    levels = set()
    for (d, _e, _m, suite, step) in facts.entries:
        if d == device and suite == "concurrency" and re.fullmatch(r"p\d+", step):
            levels.add(int(step[1:]))
    return sorted(levels)


# --------------------------------------------------------------------- SVG pieces


def bar_path(x: float, y: float, w: float, h: float, r: float = 4) -> str:
    """Square at the baseline (left), rounded at the data end (right)."""
    if w <= 0:
        return ""
    r = min(r, w, h / 2)
    return (f"M{x:.1f},{y:.1f}H{x + w - r:.1f}"
            f"A{r},{r} 0 0 1 {x + w:.1f},{y + r:.1f}"
            f"V{y + h - r:.1f}"
            f"A{r},{r} 0 0 1 {x + w - r:.1f},{y + h:.1f}"
            f"H{x:.1f}Z")


def nice_top(max_value: float, unit: float = 1.0, target_ticks: int = 3) -> tuple[float, float]:
    """(step, top) so that top >= max_value and the ticks are round numbers."""
    raw = max(max_value, unit) / target_ticks / unit
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    step = 10 * mag
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            step = m * mag
            break
    step *= unit
    return step, step * math.ceil(max_value / step - 1e-9)


def fmt_tick(v: float) -> str:
    return f"{v / 1000:g}K" if v >= 10_000 else f"{v:,.0f}"


def fmt_ctx(v: float) -> str:
    n = int(v)
    if n == 0:
        return "0"
    return f"{n // 1024}K" if n % 1024 == 0 else str(n)


def style_block(t: dict) -> str:
    return f"""
    text {{ font-family: {FONT}; fill: {t['secondary']}; }}
    .title {{ font-size: 20px; font-weight: 600; fill: {t['primary']}; }}
    .section {{ font-size: 15px; font-weight: 600; fill: {t['primary']}; }}
    .panel {{ font-size: 13px; font-weight: 600; fill: {t['primary']}; }}
    .device {{ font-size: 13px; fill: {t['primary']}; }}
    .sub {{ font-size: 13px; }}
    .legend {{ font-size: 12px; fill: {t['primary']}; }}
    .value {{ font-size: 11px; font-variant-numeric: tabular-nums; }}
    .tick {{ font-size: 10px; fill: {t['muted']}; font-variant-numeric: tabular-nums; }}
    .muted, .note, .foot {{ fill: {t['muted']}; }}
    .note {{ font-size: 11px; }}
    .foot {{ font-size: 11px; }}
    .cell {{ font-size: 12px; font-weight: 600; font-variant-numeric: tabular-nums; }}
    .limitlabel {{ font-size: 10px; fill: {t['muted']}; }}
    .grid {{ stroke: {t['grid']}; stroke-width: 1; }}
    .row {{ stroke: {t['grid']}; stroke-width: 1; }}
    .base {{ stroke: {t['baseline']}; stroke-width: 1; }}
    .limit {{ stroke: {t['limit']}; stroke-width: 1; stroke-dasharray: 2 3; }}
    .gapbox {{ fill: none; stroke: {t['muted']}; stroke-width: 1; stroke-dasharray: 3 2; }}
    """


def svg_document(width: int, height: int, theme: str, label: str, body: list[str], hatch: bool) -> str:
    t = THEMES[theme]
    defs = ""
    if hatch:
        defs = (f'<defs><pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" '
                f'patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="6" '
                f'stroke="{t["surface"]}" stroke-width="2.5" opacity="0.75"/></pattern></defs>\n')
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{esc(label)}">\n'
        f"<style>{style_block(t)}</style>\n{defs}"
        f'<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="8" '
        f'fill="{t["surface"]}" stroke="{t["border"]}"/>\n'
        + "\n".join(body) + "\n</svg>\n"
    )


def legend_items(items: list[tuple[str, str, str]], x0: float, y: float, max_x: float, t: dict
                 ) -> tuple[list[str], float]:
    """items = (kind, color, text) with kind swatch|gap|hatch. Wraps lines; returns (elements, last y)."""
    el, x = [], x0
    for kind, color, text in items:
        need = 16 + text_width(text, 12) + 22
        if x > x0 and x + need > max_x:
            x, y = x0, y + 20
        if kind == "swatch":
            el.append(f'<rect x="{x}" y="{y - 9}" width="10" height="10" rx="2" fill="{color}"/>')
        elif kind == "hatch":
            el.append(f'<rect x="{x}" y="{y - 9}" width="10" height="10" rx="2" fill="{color}"/>')
            el.append(f'<rect x="{x}" y="{y - 9}" width="10" height="10" rx="2" fill="url(#hatch)"/>')
        else:
            el.append(f'<rect x="{x + 0.5}" y="{y - 8.5}" width="9" height="9" rx="2" class="gapbox"/>')
        el.append(f'<text x="{x + 16}" y="{y}" class="legend">{esc(text)}</text>')
        x += need
    return el, y


def header(title: str, subtitle: str, pad: int, width: int) -> tuple[list[str], float]:
    el = [f'<text x="{pad}" y="{pad + 14}" class="title">{esc(title)}</text>']
    y = pad + 14
    for line in wrap(subtitle, 13, width - 2 * pad):
        y += 22
        el.append(f'<text x="{pad}" y="{y}" class="sub">{esc(line)}</text>')
    return el, y


def footer(lines: list[str], y: float, pad: int, width: int) -> tuple[list[str], float]:
    el = []
    for text in lines:
        for line in wrap(text, 11, width - 2 * pad):
            y += 16
            el.append(f'<text x="{pad}" y="{y}" class="foot">{esc(line)}</text>')
    return el, y


# --------------------------------------------------------------------- grouped bar charts


def bar_chart(runs: list[dict], theme: str, *, title: str, subtitle: str, unit: str, cell_fn,
              fmt, notes: list[str], rows_fn=None, ctx_axis: bool = False, aria: str) -> str:
    """Engines side by side, one panel per model class, one row per device (or device and level)."""
    facts = Facts(runs)
    t = THEMES[theme]
    devices = facts.devices()
    rows_fn = rows_fn or (lambda f, d: [(d, "", None)])
    rows = [(d, sub, tag) for d in devices for (_n, sub, tag) in rows_fn(facts, d)]
    models = sorted({k[2] for k in facts.entries}, key=model_rank)
    engines = sorted({k[1] for k in facts.entries}, key=engine_rank)

    grid = {}
    for device, sub, tag in rows:
        for model in models:
            for engine in engines:
                cell = cell_fn(facts, device, model, engine) if tag is None else cell_fn(facts, device, model, engine, tag)
                grid[(device, sub, model, engine)] = cell
    # Keep only rows, panels and engines that carry something to show.
    present = [(k, c) for k, c in grid.items() if c is not None]
    if not any(c.value is not None for _k, c in present):
        return ""
    rows = [r for r in rows if any(grid[(r[0], r[1], m, e)] is not None for m in models for e in engines)]
    models = [m for m in models if any(c is not None for (_d, _s, mm, _e), c in grid.items() if mm == m)]
    engines = [e for e in engines if any(c is not None for (_d, _s, _m, ee), c in grid.items() if ee == e)]

    pad, label_w, gap = 28, 196, 28
    n = len(models)
    panel_w = max(240.0, (960 - 2 * pad - label_w - gap * (n - 1)) / n)
    width = int(2 * pad + label_w + gap * (n - 1) + panel_w * n)
    bar_h, bar_gap = 12, 4
    row_h = len(engines) * bar_h + (len(engines) - 1) * bar_gap + 16
    value_room = 70 if any(c and c.hatched for _k, c in present) else 52

    values = [c.value for _k, c in present if c.value is not None]
    if ctx_axis:
        step, top = nice_top(max(values), unit=32768)
        tick_text = fmt_ctx
    else:
        step, top = nice_top(max(values))
        tick_text = fmt_tick
    scale = (panel_w - value_room) / top

    el, y = header(title, subtitle, pad, width)
    any_gap = any(c.value is None for _k, c in present) or any(grid[(r[0], r[1], m, e)] is None
                                                                for r in rows for m in models for e in engines)
    any_hatch = any(c.hatched for _k, c in present)
    items = [("swatch", engine_color(e, theme), engine_name(e)) for e in engines]
    if any_gap:
        items.append(("gap", "", "No bar: does not fit, skipped, failed or not measured"))
    if any_hatch:
        items.append(("hatch", t["muted"], "Hatched: requests were served one after another, not in parallel"))
    y += 26
    leg, y = legend_items(items, pad, y, width - pad, t)
    el += leg

    y += 34
    for m, model in enumerate(models):
        px = pad + label_w + m * (panel_w + gap)
        el.append(f'<text x="{px}" y="{y}" class="panel">Class {esc(model)}</text>')
    body_top = y + 12
    body_bottom = body_top + row_h * len(rows)

    for m, model in enumerate(models):
        px = pad + label_w + m * (panel_w + gap)
        tick = 0.0
        while tick <= top + 1e-9:
            gx = px + tick * scale
            if tick > 0:
                el.append(f'<line x1="{gx:.1f}" y1="{body_top}" x2="{gx:.1f}" y2="{body_bottom}" class="grid"/>')
            el.append(f'<text x="{gx:.1f}" y="{body_bottom + 16}" class="tick" '
                      f'text-anchor="{"start" if tick == 0 else "middle"}">{tick_text(tick)}</text>')
            tick += step
        el.append(f'<line x1="{px}" y1="{body_top}" x2="{px}" y2="{body_bottom}" class="base"/>')
        el.append(f'<text x="{px + top * scale + value_room - 6:.1f}" y="{body_bottom + 30}" class="tick" '
                  f'text-anchor="end">{esc(unit)}</text>')

        for r, (device, sub, _tag) in enumerate(rows):
            ry = body_top + r * row_h + 8
            cells = [(e, grid[(device, sub, model, e)]) for e in engines]
            labels = {c.label if c else "not measured" for _e, c in cells if c is None or c.value is None}
            if len(labels) == 1 and all(c is None or c.value is None for _e, c in cells):
                el.append(f'<text x="{px + 8}" y="{ry + row_h / 2 - 4:.1f}" class="note">{esc(labels.pop())}</text>')
                continue
            for b, (engine, cell) in enumerate(cells):
                by = ry + b * (bar_h + bar_gap)
                if cell is None or cell.value is None:
                    label = cell.label if cell else "not measured"
                    el.append(f'<text x="{px + 8}" y="{by + bar_h - 1}" class="note">{esc(label)}</text>')
                    continue
                w = cell.value * scale
                tip = f"{device}{', ' + sub if sub else ''}, class {model}, {engine_name(engine)}: {fmt(cell.value)}{cell.suffix} {unit}"
                if cell.hatched:
                    tip += " (serialized)"
                path = bar_path(px, by, w, bar_h)
                color = engine_color(engine, theme)
                el.append(f'<path d="{path}" fill="{color}"><title>{esc(tip)}</title></path>')
                if cell.hatched:
                    el.append(f'<path d="{path}" fill="url(#hatch)"/>')
                text = fmt(cell.value) + cell.suffix + (" serialized" if cell.hatched else "")
                el.append(f'<text x="{px + w + 6:.1f}" y="{by + bar_h - 1}" class="value">{esc(text)}</text>')

    shown_devices = set()
    for r, (device, sub, _tag) in enumerate(rows):
        top_y = body_top + r * row_h
        if r and rows[r - 1][0] != device:
            el.append(f'<line x1="{pad}" y1="{top_y}" x2="{width - pad}" y2="{top_y}" class="row"/>')
        if device not in shown_devices:
            shown_devices.add(device)
            el.append(f'<text x="{pad}" y="{top_y + (16 if sub else 20):.1f}" class="device">{esc(device)}</text>')
        if sub:
            first = r == 0 or rows[r - 1][0] != device
            el.append(f'<text x="{pad + 12}" y="{top_y + (38 if first else 28):.1f}" class="note">{esc(sub)}</text>')
    y = body_bottom + 30
    foot, y = footer(notes, y, pad, width)
    el += foot
    return svg_document(width, int(y + pad - 6), theme, aria, el, any_hatch)


# --------------------------------------------------------------------- the seven charts


def fmt_gen(v: float) -> str:
    return f"{v:,.1f}"


def fmt_int(v: float) -> str:
    return f"{v:,.0f}"


def fmt_sec(v: float) -> str:
    return f"{v:,.2f}"


SPEED_NOTE = "Warm run, 8,192 token prompt, 256 token answer, median of 3 repeats."


def chart_generation(runs: list[dict], theme: str) -> str:
    return bar_chart(runs, theme, title="Generation speed",
                     subtitle="Tokens per second while the answer is written. Higher is better.",
                     unit="tok/s", cell_fn=speed_cell("gen_tps"), fmt=fmt_gen,
                     notes=[SPEED_NOTE], aria="Generation speed in tokens per second per device, model class and engine")


def chart_prompt_processing(runs: list[dict], theme: str) -> str:
    return bar_chart(runs, theme, title="Prompt processing",
                     subtitle="Tokens per second while the prompt is read. Higher is better.",
                     unit="tok/s", cell_fn=speed_cell("prompt_tps"), fmt=fmt_int,
                     notes=[SPEED_NOTE], aria="Prompt processing speed in tokens per second per device, model class and engine")


def chart_ttft(runs: list[dict], theme: str) -> str:
    return bar_chart(runs, theme, title="Time to first token",
                     subtitle="Seconds from sending the prompt to the first token. Lower is better.",
                     unit="s", cell_fn=speed_cell("ttft_s"), fmt=fmt_sec,
                     notes=["Warm run, model already loaded, 8,192 token prompt, median of 3 repeats."],
                     aria="Time to first token in seconds per device, model class and engine")


def chart_max_context(runs: list[dict], theme: str) -> str:
    return bar_chart(runs, theme, title="Largest context that fits",
                     subtitle="Highest context size in tokens that loaded and answered on this device.",
                     unit="tokens", cell_fn=context_cell, fmt=fmt_ctx, ctx_axis=True,
                     notes=["A plus sign means the context ladder ended before a limit was found, so the real "
                            "limit is higher."],
                     aria="Largest context size that fits per device, model class and engine")


def _concurrency_rows(facts: Facts, device: str):
    return [(device, f"{n} at once", n) for n in concurrency_levels(facts, device)]


def _concurrency_cell(facts: Facts, device, model, engine, level):
    entry = facts.get(device, engine, model, "concurrency", f"p{level}")
    if entry is None:
        return None
    values = entry.get("values") or {}
    value = number(values.get("total_tps")) if entry.get("status") == "ok" else None
    if value is None:
        return Cell(label=gap_label(entry))
    return Cell(value, hatched=values.get("serialized") is True)


def chart_concurrency(runs: list[dict], theme: str) -> str:
    return bar_chart(runs, theme, title="Concurrent requests",
                     subtitle="Total tokens per second across all requests at once. Higher is better.",
                     unit="tok/s", cell_fn=_concurrency_cell, fmt=fmt_gen, rows_fn=_concurrency_rows,
                     notes=["Hatched bars: the engine served the requests one after another, so the figure is "
                            "not parallel throughput."],
                     aria="Total throughput with several parallel requests per device, model class and engine")


def memory_series(facts: Facts, device: str) -> list[dict]:
    series = []
    for (d, engine, model) in facts.combos():
        if d != device:
            continue
        ok, nofit = [], []
        for step, entry in facts.steps(d, engine, model, "memory").items():
            if not (step.startswith("ctx-") and step[4:].isdigit()):
                continue
            n = int(step[4:])
            used = number((entry.get("values") or {}).get("vram_used_mb"))
            if entry.get("status") == "ok" and used is not None:
                ok.append((n, used / GIB_MB))
            elif entry.get("status") == "does_not_fit":
                nofit.append(n)
        if ok:
            series.append({"engine": engine, "model": model, "points": sorted(ok), "nofit": sorted(nofit)})
    return series


def chart_memory(runs: list[dict], theme: str) -> str:
    facts = Facts(runs)
    t = THEMES[theme]
    panels = [(d, memory_series(facts, d)) for d in facts.devices()]
    panels = [(d, s) for d, s in panels if s]
    if not panels:
        return ""
    all_ctx = sorted({n for _d, s in panels for ser in s for n, _g in ser["points"]} |
                     {n for _d, s in panels for ser in s for n in ser["nofit"]})
    peak = max(g for _d, s in panels for ser in s for _n, g in ser["points"])
    top = max(float(MEMORY_LIMITS_GB[-1]), math.ceil(peak / 4) * 4.0)
    lo, hi = math.log2(all_ctx[0]), math.log2(all_ctx[-1])
    span = (hi - lo) or 1.0

    pad, width = 28, 960
    plot_l, plot_r = pad + 44, width - pad - 130
    plot_h, panel_gap = 220, 64
    engines = sorted({ser["engine"] for _d, s in panels for ser in s}, key=engine_rank)
    models = sorted({ser["model"] for _d, s in panels for ser in s}, key=model_rank)

    def px(ctx):
        return plot_l + (math.log2(ctx) - lo) / span * (plot_r - plot_l)

    def py(gb, y0):
        return y0 + plot_h - gb / top * plot_h

    el, y = header("GPU memory by context size",
                   "GPU memory in use after loading the model with this context, in GB (1024 MiB). One chart "
                   "per device; the dotted lines mark common card sizes.", pad, width)
    items = [("swatch", engine_color(e, theme), engine_name(e)) for e in engines]
    y += 26
    leg, y = legend_items(items, pad, y, width - pad, t)
    el += leg
    # line style per model class, drawn as short samples
    lx = pad
    y += 22
    for model in models:
        dash = DASHES.get(model, "")
        attr = f' stroke-dasharray="{dash}"' if dash else ""
        el.append(f'<line x1="{lx}" y1="{y - 4}" x2="{lx + 28}" y2="{y - 4}" stroke="{t["secondary"]}" '
                  f'stroke-width="2"{attr}/>')
        el.append(f'<text x="{lx + 34}" y="{y}" class="legend">Class {esc(model)}</text>')
        lx += 34 + text_width(f"Class {model}", 12) + 22
    if any(ser["nofit"] for _d, s in panels for ser in s):
        el.append(f'<text x="{lx}" y="{y}" class="legend" fill="{t["primary"]}">x  did not fit at this context</text>')

    y += 34
    for device, series in panels:
        el.append(f'<text x="{pad}" y="{y}" class="panel panel-title">{esc(device)}</text>')
        y0 = y + 14
        for gb in range(0, int(top) + 1, 4):
            gy = py(gb, y0)
            if gb:
                el.append(f'<line x1="{plot_l}" y1="{gy:.1f}" x2="{plot_r}" y2="{gy:.1f}" class="grid"/>')
            el.append(f'<text x="{plot_l - 8}" y="{gy + 3:.1f}" class="tick" text-anchor="end">{gb}</text>')
        el.append(f'<line x1="{plot_l}" y1="{y0}" x2="{plot_l}" y2="{y0 + plot_h}" class="base"/>')
        el.append(f'<line x1="{plot_l}" y1="{y0 + plot_h}" x2="{plot_r}" y2="{y0 + plot_h}" class="base"/>')
        el.append(f'<text x="{pad}" y="{y0 + plot_h / 2:.1f}" class="tick" text-anchor="middle" '
                  f'transform="rotate(-90 {pad + 4} {y0 + plot_h / 2:.1f})">GB</text>')
        for ctx in all_ctx:
            el.append(f'<text x="{px(ctx):.1f}" y="{y0 + plot_h + 16}" class="tick" text-anchor="middle">{fmt_ctx(ctx)}</text>')
        for gb in MEMORY_LIMITS_GB:
            gy = py(gb, y0)
            el.append(f'<line x1="{plot_l}" y1="{gy:.1f}" x2="{plot_r}" y2="{gy:.1f}" class="limit"/>')
            el.append(f'<text x="{plot_l + 8}" y="{gy - 4:.1f}" class="limitlabel">{gb} GB</text>')
        ends = []
        for index, ser in enumerate(series):
            color = engine_color(ser["engine"], theme)
            dash = DASHES.get(ser["model"], "")
            attr = f' stroke-dasharray="{dash}"' if dash else ""
            pts = " ".join(f"{px(n):.1f},{py(g, y0):.1f}" for n, g in ser["points"])
            name = f"{engine_name(ser['engine'])}, class {ser['model']}"
            el.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"{attr}>'
                      f'<title>{esc(device + ", " + name)}</title></polyline>')
            for n, g in ser["points"]:
                el.append(f'<circle cx="{px(n):.1f}" cy="{py(g, y0):.1f}" r="4" fill="{color}" '
                          f'stroke="{t["surface"]}" stroke-width="2"><title>{esc(name)}: {fmt_ctx(n)}, {g:.1f} GB</title></circle>')
            for n in ser["nofit"]:
                cx, cy = px(n), y0 + 10 + index * 11
                el.append(f'<text x="{cx:.1f}" y="{cy + 4:.1f}" class="legend" text-anchor="middle" '
                          f'fill="{color}" style="fill:{color}">x<title>{esc(name)}: does not fit at {fmt_ctx(n)}</title></text>')
            last_n, last_g = ser["points"][-1]
            ends.append([py(last_g, y0), f"{engine_name(ser['engine'])} {ser['model']}", color, px(last_n)])
        # direct labels at the line ends, pushed apart so they never overlap
        ends.sort(key=lambda e: (e[0], e[1]))
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + 12)
        for ey, text, _color, ex in ends:
            el.append(f'<text x="{ex + 9:.1f}" y="{ey + 3:.1f}" class="note">{esc(text)}</text>')
        y = y0 + plot_h + panel_gap - 14
    foot, y = footer(["Measured with the context ladder, f16 KV cache, whole model on the GPU."], y - 30, pad, width)
    el += foot
    return svg_document(width, int(y + pad - 6), theme, "GPU memory by context size per device", el, False)


def smoke_table(runs: list[dict]) -> dict:
    """(device, engine, model) -> {category: (passed, judged)} from the newest result of every task."""
    facts = Facts(runs)
    table: dict = {}
    for (device, engine, model, suite, _task), entry in sorted(facts.entries.items()):
        if suite != "smoke" or entry.get("status") != "ok":
            continue
        values = entry.get("values") or {}
        passed = values.get("passed")
        if not isinstance(passed, bool):
            continue
        category = values.get("category")
        if not isinstance(category, str):
            continue
        done, total = table.setdefault((device, engine, model), {}).get(category, (0, 0))
        table[(device, engine, model)][category] = (done + (1 if passed else 0), total + 1)
    return table


def chart_smoke(runs: list[dict], theme: str) -> str:
    table = smoke_table(runs)
    if not table:
        return ""
    t = THEMES[theme]
    facts = Facts(runs)
    order = {d: i for i, d in enumerate(facts.devices())}
    keys = sorted(table, key=lambda k: (order.get(k[0], len(order)), model_rank(k[2]), engine_rank(k[1])))
    cats = [c for c in CATEGORY_ORDER if any(c in table[k] for k in keys)]
    cats += sorted({c for k in keys for c in table[k]} - set(cats))

    pad, label_w, cw, ch, cg = 28, 170, 76, 28, 3
    width = max(960, 2 * pad + label_w + len(cats) * (cw + cg))
    el, y = header("Smoke tests passed",
                   "Passed tasks out of judged tasks per category. Only the newest result per device, engine "
                   "and model counts.", pad, width)
    items = [("swatch", RAMP[theme][i], label) for i, label in enumerate(["0-19%", "20-39%", "40-59%", "60-79%", "80-100%"])]
    items.append(("gap", "", "No judged task in this category"))
    y += 26
    leg, y = legend_items(items, pad, y, width - pad, t)
    el += leg
    y += 30
    for c, cat in enumerate(cats):
        cx = pad + label_w + c * (cw + cg) + cw / 2
        el.append(f'<text x="{cx:.1f}" y="{y}" class="panel" text-anchor="middle">{esc(cat.replace("_", " "))}</text>')
    y += 10
    device = None
    for key in keys:
        d, engine, model = key
        if d != device:
            device = d
            y += 10
            el.append(f'<text x="{pad}" y="{y + 16}" class="device">{esc(d)}</text>')
            y += 24
        el.append(f'<text x="{pad + 14}" y="{y + ch / 2 + 4:.1f}" class="note">{esc(engine_name(engine))}, class {esc(model)}</text>')
        for c, cat in enumerate(cats):
            cx = pad + label_w + c * (cw + cg)
            if cat not in table[key]:
                el.append(f'<rect x="{cx}" y="{y}" width="{cw}" height="{ch}" rx="4" class="gapbox"/>')
                el.append(f'<text x="{cx + cw / 2}" y="{y + ch / 2 + 4:.1f}" class="note" text-anchor="middle">-</text>')
                continue
            done, total = table[key][cat]
            ratio = done / total
            step = min(4, int(ratio * 5))
            ink = t["primary"] if RAMP_DARK_INK[theme][step] and theme == "light" else (
                "#0d1117" if RAMP_DARK_INK[theme][step] else "#ffffff")
            el.append(f'<rect x="{cx}" y="{y}" width="{cw}" height="{ch}" rx="4" fill="{RAMP[theme][step]}">'
                      f'<title>{esc(d)}, {esc(engine_name(engine))}, class {esc(model)}, {esc(cat)}: {done} of {total} passed</title></rect>')
            el.append(f'<text x="{cx + cw / 2}" y="{y + ch / 2 + 4:.1f}" class="cell" text-anchor="middle" '
                      f'style="fill:{ink}">{done}/{total}</text>')
        y += ch + cg
    foot, y = footer(["Tasks the checker cannot judge, and tasks that were skipped or failed to run, are not counted."],
                     y + 6, pad, width)
    el += foot
    return svg_document(width, int(y + pad - 6), theme, "Smoke tests passed per device, engine, model class and category", el, False)


# name, heading, description sentence, renderer
CHARTS = [
    ("generation", "Generation speed",
     "Tokens per second while the answer is written, warm run, 8,192 token prompt.", chart_generation),
    ("prompt-processing", "Prompt processing",
     "Tokens per second while the prompt is read, warm run, 8,192 token prompt.", chart_prompt_processing),
    ("ttft", "Time to first token",
     "Seconds until the first token, warm run, 8,192 token prompt. Lower is better.", chart_ttft),
    ("max-context", "Largest context",
     "Highest context size that loaded and answered. A plus sign means the ladder ended before a limit was found.",
     chart_max_context),
    ("concurrency", "Concurrent requests",
     "Total tokens per second with several requests at once. Hatched bars were served one after another and are "
     "not parallel throughput.", chart_concurrency),
    ("memory", "Memory",
     "GPU memory in use per context size and device. Dotted lines mark 8, 12, 16 and 24 GB cards.", chart_memory),
    ("smoke", "Smoke tests",
     "Passed tasks out of judged tasks per category. The newest result per device, engine and model counts.",
     chart_smoke),
]
CHARTS_BY_NAME = {name: fn for name, _h, _d, fn in CHARTS}


# --------------------------------------------------------------------- README


def picture(name: str, heading: str) -> str:
    return (f'<picture>\n  <source media="(prefers-color-scheme: dark)" srcset="charts/{name}-dark.svg">\n'
            f'  <img alt="{esc(heading)}" src="charts/{name}-light.svg">\n</picture>')


def build_chart_block(runs: list[dict], svgs: dict[str, str]) -> str:
    parts = []
    for name, heading, description, _fn in CHARTS:
        if svgs.get(name):
            parts.append(f"### {heading}\n\n{description}\n\n{picture(name, heading)}")
    return "\n\n".join(parts) if parts else "No measured runs yet."


def build_run_list(runs: list[dict], root: Path) -> str:
    entries = list(runs) + old_reports(root)
    entries.sort(key=lambda r: (r["date"], r["dir"]))
    out = []
    for run in entries:
        link = run.get("link") or f"runs/{run['dir']}/"
        line = f"- [{run['date']}: {md_text(run['device'])}]({link})"
        if run["legacy"]:
            line += ", old prompt"
        else:
            engines = sorted({k.split("/")[0] for k in run["tests"]}, key=engine_rank)
            models = sorted({k.split("/")[1] for k in run["tests"]}, key=model_rank)
            suites = [s for s in ("speed", "memory", "smoke", "concurrency")
                      if any(k.split("/")[2] == s for k in run["tests"])]
            line += (f", {' and '.join(engine_name(e) for e in engines)}; classes {', '.join(models)}; "
                     f"{', '.join(suites)}")
        out.append(line)
    return "\n".join(out)


def replace_block(text: str, name: str, body: str) -> str:
    start, end = f"<!-- generated:{name} -->", f"<!-- /generated:{name} -->"
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
    if not pattern.search(text):
        raise SystemExit(f"README.md: markers for '{name}' missing")
    return pattern.sub(lambda _: f"{start}\n{body}\n{end}", text)


# --------------------------------------------------------------------- main


def privacy_scan(outputs: dict[Path, str]) -> None:
    secrets = local_secrets()
    for path, text in outputs.items():
        leaks = find_leaks(text, secrets)
        if leaks:
            kinds = sorted({leak.split(":", 1)[0] for leak in leaks})
            raise SystemExit(f"privacy check failed for {path.name}: {len(leaks)} finding(s) of kind "
                             f"{', '.join(kinds)}; fix the device name or the data, nothing was written")


def build_outputs(root: Path = ROOT) -> dict[Path, str]:
    runs = load_runs(root)
    svgs = {}
    outputs: dict[Path, str] = {}
    for name, _heading, _desc, fn in CHARTS:
        light, dark = fn(runs, "light"), fn(runs, "dark")
        svgs[name] = light
        if light:
            outputs[root / "charts" / f"{name}-light.svg"] = light
            outputs[root / "charts" / f"{name}-dark.svg"] = dark
    readme_path = root / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    readme = replace_block(readme, "charts", build_chart_block(runs, svgs))
    readme = replace_block(readme, "runs", build_run_list(runs, root))
    outputs[readme_path] = readme
    for path, text in outputs.items():
        if "\u2014" in text:
            raise SystemExit(f"{path.name}: generated text contains an em-dash")
    privacy_scan(outputs)
    return outputs


def run(root: Path = ROOT, check: bool = False) -> int:
    outputs = build_outputs(root)
    charts = root / "charts"
    stale = [p for p, text in outputs.items() if not p.exists() or p.read_text(encoding="utf-8") != text]
    orphans = [p for p in sorted(charts.glob("*.svg")) if p not in outputs] if charts.is_dir() else []
    if check:
        for p in stale:
            print(f"out of date: {p.relative_to(root)}")
        for p in orphans:
            print(f"obsolete: {p.relative_to(root)}")
        return 1 if stale or orphans else 0
    charts.mkdir(exist_ok=True)
    for p in stale:
        p.write_text(outputs[p], encoding="utf-8", newline="\n")
        print(f"wrote {p.relative_to(root)}")
    for p in orphans:
        p.unlink()
        print(f"removed {p.relative_to(root)}")
    if not stale and not orphans:
        print("everything up to date")
    return 0


if __name__ == "__main__":
    sys.exit(run(ROOT, check="--check" in sys.argv))
