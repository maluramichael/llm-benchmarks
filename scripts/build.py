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

import argparse
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
# here, by hand, so existing colors never move. The four slots (blue, orange, aqua,
# yellow) pass the dataviz validator on the adjacent pairs, light on #ffffff and dark on
# #0d1117 (worst color-vision separation delta E 8.4). The light aqua and yellow are
# below 3:1 contrast, so every bar carries a direct value label and every chart a legend.
ENGINES = ["llamacpp", "ollama", "mlx", "vllm"]
ENGINE_NAMES = {"llamacpp": "llama.cpp", "ollama": "Ollama", "mlx": "MLX", "vllm": "vLLM"}
SERIES = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500"],
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


def _gpu_names(data: dict) -> tuple:
    gpus = (data.get("system") or {}).get("gpus") if isinstance(data.get("system"), dict) else None
    names = [clean(g["name"]).strip() for g in gpus or [] if isinstance(g, dict) and isinstance(g.get("name"), str)]
    return tuple(sorted(names))


def scan_run_files(folder: Path, root: Path) -> None:
    """Fail on personal data in a contributed run: results.json, README.md and raw/*.

    The secret list is empty on purpose: a CI machine has its own host and user names, which
    would match ordinary words. Only the generic findings (user paths, IPs, MACs) count.
    The message names file, count and kind, never the finding itself.
    """
    files = [folder / "results.json", folder / "README.md"]
    raw = folder / "raw"
    if raw.is_dir():
        files += sorted(p for p in raw.rglob("*") if p.is_file())
    for path in files:
        if not path.is_file():
            continue
        leaks = find_leaks(path.read_text(encoding="utf-8", errors="replace"), [])
        if leaks:
            kinds = sorted({leak.split(":", 1)[0] for leak in leaks})
            raise SystemExit(f"{path.relative_to(root).as_posix()}: {len(leaks)} finding(s) of kind "
                             f"{', '.join(kinds)}; remove personal data from the run before building")


def load_runs(root: Path = ROOT) -> list[dict]:
    """Every runs/*/results.json as a normalized dict, oldest first.

    Keys: dir, date, device, legacy, tests, gpus. Raises SystemExit naming the folder
    (never the content) when a file does not pass validate().
    """
    runs = []
    for path in sorted((root / "runs").glob("*/results.json")):
        name = path.parent.name
        scan_run_files(path.parent, root)
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
            "gpus": () if legacy else _gpu_names(data),
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

HAS_DATA = {"ok", "does_not_fit", "skipped", "error"}


def device_labels(runs: list[dict]) -> dict[tuple, str]:
    """(device, gpus) -> label. Two machines that share a device name but have different GPUs
    must not merge, so such names get the GPU appended."""
    gpus_by_name: dict[str, set] = {}
    for run in runs:
        if not run["legacy"]:
            gpus_by_name.setdefault(run["device"], set()).add(run["gpus"])
    labels = {}
    for run in runs:
        if run["legacy"]:
            continue
        name, gpus = run["device"], run["gpus"]
        split = len(gpus_by_name[name]) > 1
        labels[(name, gpus)] = f"{name} ({', '.join(gpus) or 'unknown GPU'})" if split else name
    return labels


class Facts:
    """The results behind every chart.

    For every (device, engine, model, suite) all steps come from ONE run: the newest run that
    has at least one final entry (ok, does_not_fit, skipped, error) for that suite. A newer run
    that only left `running` or `aborted` entries behind (a crash) does not hide it. Mixing
    steps of several runs would give impossible tables (an old ladder with a new rung).
    """

    def __init__(self, runs: list[dict]):
        live = sorted((r for r in runs if not r["legacy"]), key=lambda r: (r["date"], r["dir"]))
        labels = device_labels(live)
        groups: dict[tuple, list[dict]] = {}
        for run in live:
            label = labels[(run["device"], run["gpus"])]
            per: dict[tuple, dict] = {}
            for key, entry in run["tests"].items():
                engine, model, suite, step = key.split("/", 3)
                per.setdefault((engine, model, suite), {})[step] = entry
            for group, steps in per.items():
                groups.setdefault((label,) + group, []).append(steps)
        self.entries: dict[tuple, dict] = {}
        for (label, engine, model, suite), candidates in groups.items():
            chosen = candidates[-1]
            for steps in reversed(candidates):
                if any(e.get("status") in HAS_DATA for e in steps.values()):
                    chosen = steps
                    break
            for step, entry in chosen.items():
                self.entries[(label, engine, model, suite, step)] = entry

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
    def __init__(self, value=None, label="", hatched=False, suffix="", partial: str | None = None):
        # partial: None, or the text for a step that ran partly on the CPU ("62% GPU")
        self.value, self.label, self.hatched, self.suffix, self.partial = value, label, hatched, suffix, partial


def partial_text(values: dict) -> str | None:
    """None for a step that ran fully on the GPU, else a short label such as '62% GPU'."""
    if not isinstance(values, dict) or values.get("partial_offload") is not True:
        return None
    fraction = number(values.get("gpu_fraction"))
    return f"{round(fraction * 100)}% GPU" if fraction is not None else "partial GPU"


def fit_gap(facts: Facts, device, engine, model) -> Cell | None:
    fit = facts.get(device, engine, model, "memory", "fit")
    if fit and fit.get("status") == "does_not_fit":
        return Cell(label="does not fit")
    return None


def speed_cell(field: str, step: str = "warm"):
    def build(facts: Facts, device, model, engine, variant="", tag=None):
        entry = facts.get(device, engine, model, "speed", step)
        if entry is None:
            return fit_gap(facts, device, engine, model) if step == "warm" else None
        value = number((entry.get("values") or {}).get(field)) if entry.get("status") == "ok" else None
        if value is None:
            return Cell(label=gap_label(entry))
        return Cell(value, partial=partial_text(entry.get("values")))
    return build


_WARM_TTFT = speed_cell("ttft_s", "warm")
_CACHE_TTFT = speed_cell("ttft_s", "cache")


def ttft_cell(facts: Facts, device, model, engine, variant="", tag=None):
    return (_CACHE_TTFT if variant == "cache" else _WARM_TTFT)(facts, device, model, engine)


def context_cell(facts: Facts, device, model, engine, variant="", tag=None):
    rungs = {}
    for step, entry in facts.steps(device, engine, model, "memory").items():
        if step.startswith("ctx-") and step[4:].isdigit():
            rungs[int(step[4:])] = entry
    if not rungs:
        return fit_gap(facts, device, engine, model)
    ok = [n for n, e in rungs.items() if e.get("status") == "ok"]
    if not ok:
        return Cell(label=gap_label(rungs[min(rungs)]))
    top = max(ok)
    above = {e.get("status") for n, e in rungs.items() if n > top}
    # A rung that ladder_start turned into does_not_fit after a crash, or a rung skipped because the
    # driver reset check was unavailable, is not a measured limit.
    unsure = any(n > top and ((e.get("values") or {}).get("previous_status") in ("aborted", "running")
                              or "driver reset check unavailable" in str((e.get("values") or {}).get("reason", "")))
                 for n, e in rungs.items())
    if unsure or above & {"error", "aborted", "running"}:
        suffix = "*"  # the ladder stopped by accident: the real limit is unknown
    elif above:
        suffix = ""  # a rung above did not fit or was skipped: this is the limit
    else:
        suffix = "+"  # the ladder ended with a working rung: the limit is higher
    return Cell(float(top), suffix=suffix)


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
        elif kind == "faint":
            el.append(f'<rect x="{x}" y="{y - 9}" width="10" height="10" rx="2" fill="{color}" fill-opacity="{FAINT}" stroke="{color}" stroke-width="1.5"/>')
        elif kind == "partial":
            el.append(f'<rect x="{x + 0.75}" y="{y - 8.25}" width="8.5" height="8.5" rx="2" fill="none" '
                      f'stroke="{color}" stroke-width="1.5" stroke-dasharray="2 2"/>')
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

PAD, LABEL_W, COL_GAP, WIDTH = 28, 170, 24, 960
BAR_H, BAR_GAP = 12, 4
FAINT = 0.4  # opacity of the second bar of a pair (cache request)


def chunks(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def bar_chart(runs: list[dict], theme: str, *, title: str, subtitle: str, unit: str, cell_fn,
              fmt, notes: list[str], rows_fn=None, ctx_axis: bool = False, aria: str,
              variants=(("", ""),), suffix_notes=None) -> str:
    """Engines side by side, one panel per model class, one row per device (or device and level).

    The chart is always 960 px wide: more than four classes (three with hatched bars, which need
    wider labels) wrap into further bands of panels below.
    """
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
                for variant, _text in variants:
                    grid[(device, sub, model, engine, variant)] = cell_fn(facts, device, model, engine, variant, tag)
    present = [c for c in grid.values() if c is not None]
    if not any(c.value is not None for c in present):
        return ""
    rows = [r for r in rows if any(grid[(r[0], r[1], m, e, v)] is not None
                                   for m in models for e in engines for v, _ in variants)]
    models = [m for m in models if any(c is not None for k, c in grid.items() if k[2] == m)]
    engines = [e for e in engines if any(c is not None for k, c in grid.items() if k[3] == e)]
    series = [(e, v) for e in engines for v, _ in variants]

    any_hatch = any(c.hatched for c in present)
    any_partial = any(c.partial for c in present)
    cols = min(len(models), 3 if (any_hatch or any_partial) else 4)
    panel_w = (WIDTH - 2 * PAD - LABEL_W - COL_GAP * (cols - 1)) / cols
    value_room = (92 if any_hatch else 54) + (50 if any_partial else 0)
    block = len(series) * BAR_H + (len(series) - 1) * BAR_GAP
    row_h = max(block + 16, 40)
    has_sub = any(sub for _d, sub, _t in rows)

    values = [c.value for c in present if c.value is not None]
    if ctx_axis:
        step, top = nice_top(max(values), unit=32768)
        tick_text = fmt_ctx
    else:
        step, top = nice_top(max(values))
        tick_text = fmt_tick
    scale = (panel_w - value_room) / top

    # vertical layout of one band: optional device header rows, then the data rows
    layout, offset = [], 0
    for r, (device, sub, tag) in enumerate(rows):
        first = r == 0 or rows[r - 1][0] != device
        if has_sub and first:
            layout.append(("header", r, offset, 22))
            offset += 22
        layout.append(("row", r, offset, row_h))
        offset += row_h
    band_h = offset

    el, y = header(title, subtitle, PAD, WIDTH)
    items = [("swatch", engine_color(e, theme), engine_name(e)) for e in engines]
    if len(variants) > 1:
        items.append(("swatch", t["muted"], f"Solid bar: {variants[0][1]}"))
        items.append(("faint", t["muted"], f"Lighter bar: {variants[1][1]}"))
    if any(c is None or c.value is None for k, c in grid.items()
           if any(k[0] == r[0] and k[1] == r[1] for r in rows)):
        items.append(("gap", "", "No bar: does not fit, skipped, failed or not measured"))
    if any_hatch:
        items.append(("hatch", t["muted"], "Hatched: requests were served one after another, not in parallel"))
    if any_partial:
        items.append(("partial", t["primary"], "Dotted outline: part of the model ran on the CPU (GPU share shown)"))
    y += 26
    leg, y = legend_items(items, PAD, y, WIDTH - PAD, t)
    el += leg

    for band in chunks(models, cols):
        y += 34
        for m, model in enumerate(band):
            px = PAD + LABEL_W + m * (panel_w + COL_GAP)
            el.append(f'<text x="{px:.1f}" y="{y}" class="panel">Class {esc(model)}</text>')
        body_top = y + 12
        body_bottom = body_top + band_h
        for m, model in enumerate(band):
            px = PAD + LABEL_W + m * (panel_w + COL_GAP)
            tick = 0.0
            while tick <= top + 1e-9:
                gx = px + tick * scale
                if tick > 0:
                    el.append(f'<line x1="{gx:.1f}" y1="{body_top}" x2="{gx:.1f}" y2="{body_bottom}" class="grid"/>')
                el.append(f'<text x="{gx:.1f}" y="{body_bottom + 16}" class="tick" '
                          f'text-anchor="{"start" if tick == 0 else "middle"}">{tick_text(tick)}</text>')
                tick += step
            el.append(f'<line x1="{px:.1f}" y1="{body_top}" x2="{px:.1f}" y2="{body_bottom}" class="base"/>')
            el.append(f'<text x="{px + panel_w:.1f}" y="{body_bottom + 30}" class="tick" '
                      f'text-anchor="end">{esc(unit)}</text>')
            for kind, r, off, height in layout:
                if kind != "row":
                    continue
                device, sub, _tag = rows[r]
                ry = body_top + off + (height - block) / 2
                # a missing second bar (no cache step) is simply absent, not a gap worth a label
                cells = [((e, v), grid[(device, sub, model, e, v)]) for e, v in series
                         if v == "" or grid[(device, sub, model, e, v)] is not None]
                if all(c is None or c.value is None for _s, c in cells):
                    label = {c.label if c else "not measured" for _s, c in cells}
                    if len(label) == 1:
                        el.append(f'<text x="{px + 8:.1f}" y="{ry + block / 2 + 4:.1f}" class="note">{esc(label.pop())}</text>')
                        continue
                for (engine, variant), cell in cells:
                    b = series.index((engine, variant))
                    by = ry + b * (BAR_H + BAR_GAP)
                    if cell is None or cell.value is None:
                        label = cell.label if cell else "not measured"
                        el.append(f'<text x="{px + 8:.1f}" y="{by + BAR_H - 1}" class="note">{esc(label)}</text>')
                        continue
                    w = cell.value * scale
                    what = dict(variants).get(variant, "")
                    tip = (f"{device}{', ' + sub if sub else ''}, class {model}, {engine_name(engine)}"
                           f"{', ' + what if variant else ''}: {fmt(cell.value)}{cell.suffix} {unit}")
                    if cell.hatched:
                        tip += " (serialized)"
                    path = bar_path(px, by, w, BAR_H)
                    opacity = (f' fill-opacity="{FAINT}" stroke="{engine_color(engine, theme)}" stroke-width="1.5"'
                               if variant else "")
                    if cell.partial:
                        tip += f" (partial offload, {cell.partial})"
                        opacity += (' stroke-dasharray="2 2"' if variant else
                                    f' stroke="{t["primary"]}" stroke-width="1.5" stroke-dasharray="2 2"')
                    el.append(f'<path d="{path}" fill="{engine_color(engine, theme)}"{opacity}>'
                              f'<title>{esc(tip)}</title></path>')
                    if cell.hatched:
                        el.append(f'<path d="{path}" fill="url(#hatch)"/>')
                    text = (fmt(cell.value) + cell.suffix + (" serialized" if cell.hatched else "")
                            + (f" ({cell.partial})" if cell.partial else ""))
                    el.append(f'<text x="{px + w + 6:.1f}" y="{by + BAR_H - 1}" class="value">{esc(text)}</text>')
        # row labels and separators of this band
        for kind, r, off, height in layout:
            device, sub, _tag = rows[r]
            top_y = body_top + off
            first = r == 0 or rows[r - 1][0] != device
            if first and r and kind == ("header" if has_sub else "row"):
                el.append(f'<line x1="{PAD}" y1="{top_y}" x2="{WIDTH - PAD}" y2="{top_y}" class="row"/>')
            if kind == "header":
                el.append(f'<text x="{PAD}" y="{top_y + 15}" class="device">{esc(device)}</text>')
            elif has_sub:
                el.append(f'<text x="{PAD + 12}" y="{top_y + height / 2 + 4:.1f}" class="note">{esc(sub)}</text>')
            elif first:
                lines = wrap(clean(device), 13, LABEL_W - 4)[:2]
                base = top_y + height / 2 + 4 - 8 * (len(lines) - 1)
                for i, line in enumerate(lines):
                    el.append(f'<text x="{PAD}" y="{base + 16 * i:.1f}" class="device">{esc(line)}</text>')
        y = body_bottom + 30

    shown = []
    for suffix, text in (suffix_notes or {}).items():
        if any(c.suffix == suffix for c in present):
            shown.append(text)
    foot, y = footer(list(notes) + shown, y, PAD, WIDTH)
    el += foot
    return svg_document(WIDTH, int(y + PAD - 6), theme, aria, el, any_hatch)


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
                     unit="s", cell_fn=ttft_cell, fmt=fmt_sec,
                     variants=(("", "warm run, prompt read in full"),
                               ("cache", "second request, prompt cache hit")),
                     notes=["8,192 token prompt, model already loaded, median of 3 repeats. The second "
                            "request repeats the same prompt and shows what the prompt cache saves."],
                     aria="Time to first token in seconds per device, model class and engine")


def chart_max_context(runs: list[dict], theme: str) -> str:
    return bar_chart(runs, theme, title="Largest context that fits",
                     subtitle="Highest context size in tokens that loaded and answered on this device.",
                     unit="tokens", cell_fn=context_cell, fmt=fmt_ctx, ctx_axis=True, notes=[],
                     suffix_notes={"+": "+ The context ladder ended on a working size, so the real limit is higher.",
                                   "*": "* The ladder stopped on an error or abort above this size, so the limit is unknown."},
                     aria="Largest context size that fits per device, model class and engine")


def _concurrency_rows(facts: Facts, device: str):
    return [(device, f"{n} at once", n) for n in concurrency_levels(facts, device)]


def _concurrency_cell(facts: Facts, device, model, engine, variant, level):
    entry = facts.get(device, engine, model, "concurrency", f"p{level}")
    if entry is None:
        return None
    values = entry.get("values") or {}
    value = number(values.get("total_tps")) if entry.get("status") == "ok" else None
    if value is None:
        return Cell(label=gap_label(entry))
    return Cell(value, hatched=values.get("serialized") is True, partial=partial_text(values))


def chart_concurrency(runs: list[dict], theme: str) -> str:
    return bar_chart(runs, theme, title="Concurrent requests",
                     subtitle="Total tokens per second across all requests at once. Higher is better.",
                     unit="tok/s", cell_fn=_concurrency_cell, fmt=fmt_gen, rows_fn=_concurrency_rows,
                     notes=["Hatched bars: the engine served the requests one after another, so the figure is "
                            "not parallel throughput."],
                     aria="Total throughput with several parallel requests per device, model class and engine")


def memory_series(facts: Facts, device: str) -> list[dict]:
    """Per engine and model: points (ctx, total GiB, model GiB or None), rungs that did not fit, rungs without a figure."""
    series = []
    for (d, engine, model) in facts.combos():
        if d != device:
            continue
        ok, nofit, nofigure, partial = [], [], [], []
        for step, entry in facts.steps(d, engine, model, "memory").items():
            if not (step.startswith("ctx-") and step[4:].isdigit()):
                continue
            n = int(step[4:])
            values = entry.get("values") or {}
            used = number(values.get("vram_used_mb"))
            if entry.get("status") == "ok":
                if used is None:
                    nofigure.append(n)
                    continue
                base = number(values.get("vram_baseline_mb"))
                if base is not None and used - base <= 0:
                    nofigure.append(n)  # the baseline is not below the reading: no usable figure
                    continue
                net = (used - base) / GIB_MB if base is not None else None
                ok.append((n, used / GIB_MB, net))
                if partial_text(values):
                    partial.append(n)
            elif entry.get("status") == "does_not_fit":
                nofit.append(n)
        if ok or nofit or nofigure:
            series.append({"engine": engine, "model": model, "points": sorted(ok),
                           "nofit": sorted(nofit), "nofigure": sorted(nofigure), "partial": sorted(partial)})
    return series


def chart_memory(runs: list[dict], theme: str) -> str:
    facts = Facts(runs)
    t = THEMES[theme]
    panels = []
    for d in facts.devices():
        series = memory_series(facts, d)
        if any(s["points"] for s in series):
            points = [p for s in series for p in s["points"]]
            net = all(p[2] is not None for p in points)
            panels.append((d, series, net))
    if not panels:
        return ""
    value = lambda p, net: p[2] if net else p[1]  # noqa: E731
    all_ctx = sorted({n for _d, s, _n in panels for ser in s for n in
                      [p[0] for p in ser["points"]] + ser["nofit"] + ser["nofigure"]})
    peak = max(value(p, net) for _d, s, net in panels for ser in s for p in ser["points"])
    top = max(float(MEMORY_LIMITS_GB[-1]), math.ceil(peak / 4) * 4.0)
    lo, hi = math.log2(all_ctx[0]), math.log2(all_ctx[-1])
    span = (hi - lo) or 1.0

    plot_l, plot_r, lim_r = PAD + 44, WIDTH - PAD - 190, WIDTH - PAD - 48
    plot_h, panel_gap = 220, 36
    engines = sorted({ser["engine"] for _d, s, _n in panels for ser in s}, key=engine_rank)
    models = sorted({ser["model"] for _d, s, _n in panels for ser in s}, key=model_rank)
    nets = {net for _d, _s, net in panels}
    if nets == {True}:
        what = ("GPU memory used by the model: in use after loading it minus in use before loading it, "
                "in GiB.")
    elif nets == {False}:
        what = "Total GPU memory in use after loading the model, in GiB. It includes the desktop and other programs."
    else:
        what = ("GPU memory in GiB. Each chart says whether it shows the model alone (in use after "
                "loading minus in use before) or the total card usage, which includes the desktop and other programs.")

    def px(ctx):
        return plot_l + (math.log2(ctx) - lo) / span * (plot_r - plot_l)

    def py(gb, y0):
        return y0 + plot_h - gb / top * plot_h

    el, y = header("GPU memory by context size", what + " One chart per device; dotted lines mark common card sizes (a 24 GB card has 24 GiB).",
                   PAD, WIDTH)
    y += 26
    leg, y = legend_items([("swatch", engine_color(e, theme), engine_name(e)) for e in engines],
                          PAD, y, WIDTH - PAD, t)
    el += leg
    lx = PAD
    y += 22
    for model in models:
        dash = DASHES.get(model, "")
        attr = f' stroke-dasharray="{dash}"' if dash else ""
        el.append(f'<line x1="{lx}" y1="{y - 4}" x2="{lx + 28}" y2="{y - 4}" stroke="{t["secondary"]}" '
                  f'stroke-width="2"{attr}/>')
        el.append(f'<text x="{lx + 34}" y="{y}" class="legend">Class {esc(model)}</text>')
        lx += 34 + text_width(f"Class {model}", 12) + 22
    if any(ser["nofit"] for _d, s, _n in panels for ser in s):
        el.append(f'<text x="{lx + 4}" y="{y}" class="legend" style="fill:{t["primary"]}">x</text>')
        el.append(f'<text x="{lx + 16}" y="{y}" class="legend">did not fit</text>')
        lx += 16 + text_width("did not fit", 12) + 22
    if any(ser["nofigure"] for _d, s, _n in panels for ser in s):
        el.append(f'<circle cx="{lx + 5}" cy="{y - 4}" r="4" fill="none" stroke="{t["primary"]}" stroke-width="1.5"/>')
        el.append(f'<text x="{lx + 16}" y="{y}" class="legend">ran, no memory figure</text>')
        lx += 16 + text_width("ran, no memory figure", 12) + 22
    if any(ser["partial"] for _d, s, _n in panels for ser in s):
        el.append(f'<circle cx="{lx + 5}" cy="{y - 4}" r="5" fill="{t["muted"]}" stroke="{t["primary"]}" '
                  f'stroke-width="1.5" stroke-dasharray="2 1.5"/>')
        el.append(f'<text x="{lx + 18}" y="{y}" class="legend">part of the model on the CPU</text>')

    y += 34
    for device, series, net in panels:
        mode = "model memory" if net else "total card usage"
        lines = wrap(clean(device), 13, plot_r - PAD - 130) or [""]
        for i, line in enumerate(lines):
            el.append(f'<text x="{PAD}" y="{y + 16 * i}" class="panel panel-title">{esc(line)}</text>')
        y += 16 * (len(lines) - 1)
        el.append(f'<text x="{PAD + text_width(lines[-1], 13) + 14:.0f}" y="{y}" class="sub">{mode}</text>')
        y0 = y + 14
        for gb in range(0, int(top) + 1, 4):
            gy = py(gb, y0)
            if gb:
                el.append(f'<line x1="{plot_l}" y1="{gy:.1f}" x2="{plot_r}" y2="{gy:.1f}" class="grid"/>')
            el.append(f'<text x="{plot_l - 8}" y="{gy + 3:.1f}" class="tick" text-anchor="end">{gb}</text>')
        el.append(f'<line x1="{plot_l}" y1="{y0}" x2="{plot_l}" y2="{y0 + plot_h}" class="base"/>')
        el.append(f'<line x1="{plot_l}" y1="{y0 + plot_h}" x2="{plot_r}" y2="{y0 + plot_h}" class="base"/>')
        el.append(f'<text x="{PAD}" y="{y0 + plot_h / 2:.1f}" class="tick" text-anchor="middle" '
                  f'transform="rotate(-90 {PAD + 4} {y0 + plot_h / 2:.1f})">GiB</text>')
        for ctx in all_ctx:
            el.append(f'<text x="{px(ctx):.1f}" y="{y0 + plot_h + 16}" class="tick" text-anchor="middle">{fmt_ctx(ctx)}</text>')
        for gb in MEMORY_LIMITS_GB:
            gy = py(gb, y0)
            el.append(f'<line x1="{plot_l}" y1="{gy:.1f}" x2="{lim_r}" y2="{gy:.1f}" class="limit"/>')
            el.append(f'<text x="{lim_r + 6}" y="{gy + 3:.1f}" class="limitlabel">{gb} GB</text>')
        ends = []
        for ser in series:
            if not ser["points"]:
                continue
            color = engine_color(ser["engine"], theme)
            dash = DASHES.get(ser["model"], "")
            attr = f' stroke-dasharray="{dash}"' if dash else ""
            name = f"{engine_name(ser['engine'])}, class {ser['model']}"
            pts = " ".join(f"{px(p[0]):.1f},{py(value(p, net), y0):.1f}" for p in ser["points"])
            el.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"{attr}>'
                      f'<title>{esc(device + ", " + name)}</title></polyline>')
            for p in ser["points"]:
                part = p[0] in ser["partial"]
                ring = (f'stroke="{t["primary"]}" stroke-width="1.5" stroke-dasharray="2 1.5"' if part
                        else f'stroke="{t["surface"]}" stroke-width="2"')
                el.append(f'<circle cx="{px(p[0]):.1f}" cy="{py(value(p, net), y0):.1f}" r="{5 if part else 4}" '
                          f'fill="{color}" {ring}{" class=" + chr(34) + "partial" + chr(34) if part else ""}>'
                          f'<title>{esc(name)}: {fmt_ctx(p[0])}, {value(p, net):.1f} GiB'
                          f'{", part of the model on the CPU" if part else ""}</title></circle>')
            last = ser["points"][-1]
            ends.append([py(value(last, net), y0), f"{engine_name(ser['engine'])} {ser['model']}", px(last[0])])
        ends.sort(key=lambda e: (e[0], e[1]))
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + 12)
        for ey, text, ex in ends:
            el.append(f'<text x="{ex + 9:.1f}" y="{ey + 3:.1f}" class="note">{esc(text)}</text>')
        # strip below the axis: rungs that did not fit (x) or ran without a memory figure (hollow circle)
        sy = y0 + plot_h + 32
        for ser in series:
            if not (ser["nofit"] or ser["nofigure"]):
                continue
            color = engine_color(ser["engine"], theme)
            name = f"{engine_name(ser['engine'])}, class {ser['model']}"
            for n in ser["nofit"]:
                el.append(f'<text x="{px(n):.1f}" y="{sy + 4}" class="nofit" text-anchor="middle" '
                          f'style="fill:{color};font-size:13px;font-weight:700">x'
                          f'<title>{esc(name)}: does not fit at {fmt_ctx(n)}</title></text>')
            for n in ser["nofigure"]:
                el.append(f'<circle cx="{px(n):.1f}" cy="{sy}" r="4" fill="{t["surface"]}" stroke="{color}" '
                          f'stroke-width="1.5" class="nofigure"><title>{esc(name)}: ran at {fmt_ctx(n)}, '
                          f'no memory figure</title></circle>')
            last_x = max(px(n) for n in ser["nofit"] + ser["nofigure"])
            el.append(f'<text x="{last_x + 12:.1f}" y="{sy + 4}" class="note">{esc(engine_name(ser["engine"]))} {esc(ser["model"])}</text>')
            sy += 14
        y = sy + panel_gap - 14
    any_partial = any(ser["partial"] for _d, s, _n in panels for ser in s)
    notes = ["Measured with the context ladder, f16 KV cache. "
             + ("The whole model was on the GPU except at points with a dotted ring, where part of it ran on the CPU."
                if any_partial else "The whole model was on the GPU.")]
    if not all(net for _d, _s, net in panels):
        notes.append("Total card usage includes the desktop and other programs.")
    foot, y = footer(notes, y - 24, PAD, WIDTH)
    el += foot
    return svg_document(WIDTH, int(y + PAD - 6), theme, "GPU memory by context size per device", el, False)


def smoke_table(runs: list[dict]) -> dict:
    """(device, engine, model) -> {category: (passed, judged)} from the newest run that has smoke results."""
    facts = Facts(runs)
    table: dict = {}
    for (device, engine, model, suite, _task), entry in sorted(facts.entries.items()):
        if suite != "smoke" or entry.get("status") != "ok":
            continue
        values = entry.get("values") or {}
        passed = values.get("passed")
        category = values.get("category")
        if not isinstance(passed, bool) or not isinstance(category, str):
            continue
        done, total = table.setdefault((device, engine, model), {}).get(category, (0, 0))
        table[(device, engine, model)][category] = (done + (1 if passed else 0), total + 1)
    return table


def smoke_partial(runs: list[dict]) -> set:
    """(device, engine, model, category) cells where at least one counted task ran partly on the CPU."""
    found = set()
    for (device, engine, model, suite, _task), entry in Facts(runs).entries.items():
        values = entry.get("values") or {}
        if (suite == "smoke" and entry.get("status") == "ok" and isinstance(values.get("passed"), bool)
                and isinstance(values.get("category"), str) and partial_text(values)):
            found.add((device, engine, model, values["category"]))
    return found


def ramp_step(done: int, total: int) -> int:
    """Five bins of pass rate: 0-19, 20-39, 40-59, 60-79, 80-100 percent."""
    return min(4, int(done / total * 5))


def chart_smoke(runs: list[dict], theme: str) -> str:
    table = smoke_table(runs)
    if not table:
        return ""
    t = THEMES[theme]
    partial = smoke_partial(runs)
    facts = Facts(runs)
    order = {d: i for i, d in enumerate(facts.devices())}
    keys = sorted(table, key=lambda k: (order.get(k[0], len(order)), model_rank(k[2]), engine_rank(k[1])))
    cats = [c for c in CATEGORY_ORDER if any(c in table[k] for k in keys)]
    cats += sorted({c for k in keys for c in table[k]} - set(cats))

    cg, label_w = 3, 170
    cw = min(76, (WIDTH - 2 * PAD - label_w) // len(cats) - cg)
    ch = 28
    el, y = header("Smoke tests passed",
                   "Passed tasks out of judged tasks per category. All results of a device, engine and model "
                   "come from its newest run.", PAD, WIDTH)
    items = [("swatch", RAMP[theme][i], label) for i, label in enumerate(["0-19%", "20-39%", "40-59%", "60-79%", "80-100%"])]
    items.append(("gap", "", "No judged task in this category"))
    if partial:
        items.append(("partial", t["primary"], "Dot in a corner: a task ran with part of the model on the CPU"))
    y += 26
    leg, y = legend_items(items, PAD, y, WIDTH - PAD, t)
    el += leg
    y += 30
    for c, cat in enumerate(cats):
        cx = PAD + label_w + c * (cw + cg) + cw / 2
        el.append(f'<text x="{cx:.1f}" y="{y}" class="panel" text-anchor="middle" style="font-size:12px">'
                  f'{esc(cat.replace("_", " "))}</text>')
    y += 10
    device = None
    for key in keys:
        d, engine, model = key
        if d != device:
            device = d
            y += 10
            el.append(f'<text x="{PAD}" y="{y + 16}" class="device">{esc(d)}</text>')
            y += 24
        el.append(f'<text x="{PAD + 14}" y="{y + ch / 2 + 4:.1f}" class="note">{esc(engine_name(engine))}, class {esc(model)}</text>')
        for c, cat in enumerate(cats):
            cx = PAD + label_w + c * (cw + cg)
            if cat not in table[key]:
                el.append(f'<rect x="{cx}" y="{y}" width="{cw}" height="{ch}" rx="4" class="gapbox"/>')
                el.append(f'<text x="{cx + cw / 2}" y="{y + ch / 2 + 4:.1f}" class="note" text-anchor="middle">-</text>')
                continue
            done, total = table[key][cat]
            step = ramp_step(done, total)
            dark_ink = RAMP_DARK_INK[theme][step]
            ink = (t["primary"] if theme == "light" else "#0d1117") if dark_ink else "#ffffff"
            el.append(f'<rect x="{cx}" y="{y}" width="{cw}" height="{ch}" rx="4" fill="{RAMP[theme][step]}">'
                      f'<title>{esc(d)}, {esc(engine_name(engine))}, class {esc(model)}, {esc(cat)}: {done} of {total} passed</title></rect>')
            el.append(f'<text x="{cx + cw / 2}" y="{y + ch / 2 + 4:.1f}" class="cell" text-anchor="middle" '
                      f'style="fill:{ink}">{done}/{total}</text>')
            if (d, engine, model, cat) in partial:
                el.append(f'<circle cx="{cx + cw - 7}" cy="{y + 7}" r="3" fill="{ink}" class="partial">'
                          f'<title>at least one task ran with part of the model on the CPU</title></circle>')
        y += ch + cg
    foot, y = footer(["Tasks the checker cannot judge, and tasks that were skipped or failed to run, are not counted."],
                     y + 6, PAD, WIDTH)
    el += foot
    return svg_document(WIDTH, int(y + PAD - 6), theme, "Smoke tests passed per device, engine, model class and category", el, False)


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the README charts and run list from runs/*/results.json.")
    parser.add_argument("--check", action="store_true", help="exit 1 if a generated file is out of date")
    args = parser.parse_args(argv)
    return run(ROOT, check=args.check)


if __name__ == "__main__":
    sys.exit(main())
