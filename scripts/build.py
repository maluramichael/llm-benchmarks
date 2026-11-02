"""Build the results table, the run list and the charts from runs/*.md.

The run files are the only source of truth. This script reads every test card
("### <engine>, <model> (<class>)" plus its field list), then

- rewrites the parts of README.md between the <!-- generated:... --> markers,
- writes charts/reference-light.svg and charts/reference-dark.svg.

Same input, same output: no timestamps, no randomness, stable sort orders.
Standard library only, so CI needs nothing but python3.

    python3 scripts/build.py          # write files
    python3 scripts/build.py --check  # exit 1 if anything is out of date
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
README = ROOT / "README.md"
CHARTS = ROOT / "charts"

CLASS_ORDER = ["S", "M", "L", "XL", "F", "Extra"]
CHART_CLASSES = ["S", "M", "L"]

# Color follows the engine, never its rank. New engines get the next free slot
# here, by hand, so existing colors never move.
ENGINES = ["llama.cpp", "Ollama", "MLX", "Strata"]
SERIES = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#9085e9"],
}
THEMES = {
    "light": {
        "surface": "#ffffff",
        "border": "#d1d9e0",
        "primary": "#1f2328",
        "secondary": "#59636e",
        "muted": "#818b98",
        "grid": "#eaeef2",
        "baseline": "#c8d1da",
    },
    "dark": {
        "surface": "#0d1117",
        "border": "#3d444d",
        "primary": "#f0f6fc",
        "secondary": "#9198a1",
        "muted": "#6e7681",
        "grid": "#1d232b",
        "baseline": "#3d444d",
    },
}
FONT = 'system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif'


@dataclass
class Result:
    date: str
    device: str
    file: str
    anchor: str
    engine: str
    model: str
    cls: str
    pp: float | None
    gen: float | None
    status: str  # "ok", "does not fit", "not measured"

    @property
    def family(self) -> str:
        for name in ENGINES:
            if self.engine.lower().startswith(name.lower()):
                return name
        return self.engine.split(" ")[0]

    @property
    def is_variant(self) -> bool:
        # "llama.cpp ... with MTP" and the like: shown in the table, not in the chart
        return " with " in self.engine


@dataclass
class Run:
    date: str
    device: str
    file: str
    sections: list[str]
    results: list[Result]


# --------------------------------------------------------------------- parsing


def split_heading(text: str) -> tuple[str, str, str] | None:
    """'<engine>, <model> (<class>)' -> (engine, model, class).

    The engine may contain commas inside parentheses ("build 12024, Vulkan"),
    the model may contain commas outside ("Q4_K_M, abliterated"), so split at
    the first comma on paren depth 0.
    """
    m = re.match(r"^(.*) \(([^()]+)\)$", text.strip())
    if not m:
        return None
    body, cls = m.groups()
    depth = 0
    for i, ch in enumerate(body):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            return body[:i].strip(), body[i + 1 :].strip(), cls.strip()
    return None


def parse_number(value: str) -> float | None:
    m = re.match(r"^([\d,]+(?:\.\d+)?)", value.strip())
    return float(m.group(1).replace(",", "")) if m else None


def github_slug(heading: str) -> str:
    """The id GitHub gives a heading, used when a card has no explicit anchor."""
    slug = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return slug.replace(" ", "-")


def parse_run(path: Path) -> Run:
    lines = path.read_text(encoding="utf-8").splitlines()
    m = re.match(r"^# (\d{4}-\d{2}-\d{2}): (.+)$", lines[0] if lines else "")
    if not m:
        raise SystemExit(f"{path.name}: first line must be '# YYYY-MM-DD: <device>'")
    date, device = m.groups()
    if not path.name.startswith(date + "-"):
        raise SystemExit(f"{path.name}: file name must start with the date from the title ({date}-)")
    sections = [line[3:].strip() for line in lines if line.startswith("## ")]

    results: list[Result] = []
    anchor = None
    current: dict | None = None

    def flush() -> None:
        if current is None:
            return
        fields = current["fields"]
        pp = parse_number(fields.get("Prompt processing tok/s", ""))
        gen = parse_number(fields.get("Generation tok/s", ""))
        if pp is not None and gen is not None:
            status = "ok"
        elif "does not fit" in fields.get("Max context", "").lower():
            status = "does not fit"
        else:
            status = "not measured"
        results.append(
            Result(date, device, path.name, current["anchor"], current["engine"],
                   current["model"], current["cls"], pp, gen, status)
        )

    section = None
    for line in lines:
        a = re.match(r'^<a id="([^"]+)"></a>$', line)
        if a:
            anchor = a.group(1)
            continue
        if line.startswith("## ") or line.startswith("### "):
            flush()
            current = None
            if line.startswith("## "):
                section = line[3:].strip()
            # test cards only live under "## Tests"; other ### headings are free text
            elif section == "Tests":
                parts = split_heading(line[4:])
                if parts:
                    engine, model, cls = parts
                    if cls not in CLASS_ORDER:
                        raise SystemExit(f"{path.name}: unknown class '{cls}' in '{line}', "
                                         f"use one of {', '.join(CLASS_ORDER)}")
                    current = {"anchor": anchor or github_slug(line[4:]), "engine": engine,
                               "model": model, "cls": cls, "fields": {}}
            anchor = None
            continue
        f = re.match(r"^- \*\*(.+?):\*\* (.+)$", line)
        if f and current is not None:
            current["fields"][f.group(1)] = f.group(2)
    flush()
    if not results:
        raise SystemExit(f"{path.name}: no test cards found under '## Tests' "
                         f"(expected '### <engine> <version>, <model> <quantization> (<class>)')")
    return Run(date, device, path.name, sections, results)


def load_runs() -> list[Run]:
    return sorted((parse_run(p) for p in RUNS.glob("*.md")), key=lambda r: (r.date, r.file))


# --------------------------------------------------------------------- ordering


def class_rank(cls: str) -> int:
    return CLASS_ORDER.index(cls) if cls in CLASS_ORDER else len(CLASS_ORDER)


def device_order(results: list[Result]) -> list[str]:
    """Fastest device first, by its best generation speed on class S."""
    best: dict[str, float] = {}
    for r in results:
        best.setdefault(r.device, 0.0)
        if r.cls == "S" and r.gen is not None:
            best[r.device] = max(best[r.device], r.gen)
    return sorted(best, key=lambda d: (-best[d], d))


def chart_cell(results: list[Result], device: str, cls: str, family: str) -> Result | None:
    """Latest non-variant result for one device, class and engine family."""
    hits = [r for r in results
            if r.device == device and r.cls == cls and r.family == family and not r.is_variant]
    if not hits:
        return None
    # newest run wins; on the same day prefer a measured value over "does not fit"
    return max(hits, key=lambda r: (r.date, r.status == "ok", r.file))


# --------------------------------------------------------------------- README


def fmt_pp(v: float) -> str:
    return f"{v:,.0f}"


def fmt_gen(v: float) -> str:
    return f"{v:,.1f}"


def build_table(results: list[Result]) -> str:
    out = [
        "| Device | Model | Engine | Prompt processing tok/s | Generation tok/s | Details |",
        "|:------|:-------|:-------|-----------------------:|-----------------:|:------|",
    ]
    for device in device_order(results):
        rows = [r for r in results if r.device == device]
        rows.sort(key=lambda r: (class_rank(r.cls), r.model,
                                 -(r.gen or -1), r.engine, r.date))
        for i, r in enumerate(rows):
            name = f"**{device}**" if i == 0 else ""
            if r.status == "ok":
                pp, gen = fmt_pp(r.pp), fmt_gen(r.gen)
            else:
                pp = gen = r.status
            link = f"[details](runs/{r.file}#{r.anchor})"
            out.append(f"| {name} | {r.model} | {r.engine} | {pp} | {gen} | {link} |")
    return "\n".join(out)


EXTRA_SECTIONS = {"Logic and coding test": "logic and coding test",
                  "Concurrent requests": "concurrent requests"}


def build_run_list(runs: list[Run]) -> str:
    out = []
    for run in runs:
        notes = []
        extra_models = []
        for r in run.results:
            if r.cls not in CHART_CLASSES and r.model.split(" ")[0] not in extra_models:
                extra_models.append(r.model.split(" ")[0])
        notes += extra_models
        notes += [EXTRA_SECTIONS[s] for s in run.sections if s in EXTRA_SECTIONS]
        suffix = ", " + ", ".join(notes) if notes else ""
        out.append(f"- [{run.date}: {run.device}](runs/{run.file}){suffix}")
    return "\n".join(out)


CHART_MD = """<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/reference-dark.svg">
  <img alt="Prompt processing and generation speed per device for the reference models Qwen3.5 4B, Qwen3 14B and Qwen3.8 27B, split by engine" src="charts/reference-light.svg">
</picture>"""


def replace_block(text: str, name: str, body: str) -> str:
    start, end = f"<!-- generated:{name} -->", f"<!-- /generated:{name} -->"
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
    if not pattern.search(text):
        raise SystemExit(f"README.md: markers for '{name}' missing")
    return pattern.sub(lambda _: f"{start}\n{body}\n{end}", text)


# --------------------------------------------------------------------- SVG


def nice_step(max_value: float, target_ticks: int = 3) -> float:
    raw = max_value / target_ticks
    mag = 10 ** len(str(int(raw))) / 10
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def fmt_tick(v: float) -> str:
    if v >= 10_000:
        return f"{v / 1000:g}K"
    return f"{v:,.0f}"


def text_width(s: str, size: float) -> float:
    """Rough width of system-ui text, good enough for layout decisions."""
    return len(s) * size * 0.56


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


def build_svg(results: list[Result], theme: str) -> str:
    t = THEMES[theme]
    colors = dict(zip(ENGINES, SERIES[theme]))
    chart = [r for r in results if r.cls in CHART_CLASSES and not r.is_variant]
    families = [e for e in ENGINES if any(r.family == e for r in chart)]
    devices = [d for d in device_order(results) if any(r.device == d for r in chart)]
    panels = []
    for cls in CHART_CLASSES:
        models = sorted({r.model for r in chart if r.cls == cls})
        if models:
            panels.append((cls, " / ".join(models)))

    W, pad = 960, 28
    label_w, gap = 196, 28
    panel_w = (W - 2 * pad - label_w - gap * (len(panels) - 1)) / len(panels)
    bar_h, bar_gap = 12, 4
    row_h = len(families) * bar_h + (len(families) - 1) * bar_gap + 16
    value_room = 48

    el: list[str] = []
    y = pad + 14
    el.append(f'<text x="{pad}" y="{y}" class="title">Same model, every device</text>')
    y += 22
    el.append(f'<text x="{pad}" y="{y}" class="sub">8,192 token German prompt, 256 token answer, '
              f'median of 3 runs. Higher is better.</text>')

    # legend
    y += 26
    lx = pad
    for fam in families:
        el.append(f'<rect x="{lx}" y="{y - 9}" width="10" height="10" rx="2" fill="{colors[fam]}"/>')
        el.append(f'<text x="{lx + 16}" y="{y}" class="legend">{escape(fam)}</text>')
        lx += 16 + text_width(fam, 12) + 22

    metrics = [("gen", "Generation", "tok/s, how fast the answer is written", fmt_gen),
               ("pp", "Prompt processing", "tok/s, how fast the document is read", fmt_pp)]

    for key, name, hint, fmt in metrics:
        y += 40
        el.append(f'<text x="{pad}" y="{y}" class="section">{name}</text>')
        el.append(f'<text x="{pad + text_width(name, 15) + 12:.0f}" y="{y}" class="sub">{hint}</text>')
        values = [getattr(r, key) for r in chart if getattr(r, key) is not None]
        step = nice_step(max(values))
        top = step * -(-max(values) // step)
        scale = (panel_w - value_room) / top

        y += 30
        header_y = y
        body_top = header_y + 12
        body_bottom = body_top + row_h * len(devices)

        for p, (cls, model) in enumerate(panels):
            px = pad + label_w + p * (panel_w + gap)
            el.append(f'<text x="{px}" y="{header_y}" class="panel">{escape(model)}'
                      f'<tspan class="muted" dx="6">{cls}</tspan></text>')
            # gridlines + ticks
            tick = 0.0
            while tick <= top + 1e-9:
                gx = px + tick * scale
                if tick > 0:
                    el.append(f'<line x1="{gx:.1f}" y1="{body_top}" x2="{gx:.1f}" y2="{body_bottom}" class="grid"/>')
                el.append(f'<text x="{gx:.1f}" y="{body_bottom + 16}" class="tick" '
                          f'text-anchor="{"start" if tick == 0 else "middle"}">{fmt_tick(tick)}</text>')
                tick += step
            el.append(f'<line x1="{px}" y1="{body_top}" x2="{px}" y2="{body_bottom}" class="base"/>')

            for d, device in enumerate(devices):
                ry = body_top + d * row_h + 8
                cells = [(fam, chart_cell(chart, device, cls, fam)) for fam in families]
                states = {"not measured" if c is None else c.status for _, c in cells}
                if len(states) == 1 and "ok" not in states:
                    # one label per row instead of one per engine
                    el.append(f'<text x="{px + 8}" y="{ry + row_h / 2 - 4:.1f}" class="note">{states.pop()}</text>')
                    continue
                for b, (fam, r) in enumerate(cells):
                    by = ry + b * (bar_h + bar_gap)
                    if r is None or r.status != "ok":
                        label = "does not fit" if r and r.status == "does not fit" else "not measured"
                        el.append(f'<text x="{px + 8}" y="{by + bar_h - 1}" class="note">{label}</text>')
                        continue
                    v = getattr(r, key)
                    w = v * scale
                    tip = f"{device}, {r.model}, {r.engine}: {fmt(v)} tok/s"
                    el.append(f'<path d="{bar_path(px, by, w, bar_h)}" fill="{colors[fam]}">'
                              f'<title>{escape(tip)}</title></path>')
                    el.append(f'<text x="{px + w + 6:.1f}" y="{by + bar_h - 1}" class="value">{fmt(v)}</text>')

        for d, device in enumerate(devices):
            cy = body_top + d * row_h + row_h / 2 + 4
            el.append(f'<text x="{pad}" y="{cy:.1f}" class="device">{escape(device)}</text>')
            if d:
                el.append(f'<line x1="{pad}" y1="{body_top + d * row_h}" x2="{W - pad}" '
                          f'y2="{body_top + d * row_h}" class="row"/>')
        y = body_bottom + 16

    y += 30
    el.append(f'<text x="{pad}" y="{y}" class="foot">Engines without a bar did not run this model on that '
              f'device. Generated from runs/*.md, full numbers in the table below.</text>')
    H = y + pad - 6

    style = f"""
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
    .muted {{ font-weight: 400; }}
    .grid {{ stroke: {t['grid']}; stroke-width: 1; }}
    .row {{ stroke: {t['grid']}; stroke-width: 1; }}
    .base {{ stroke: {t['baseline']}; stroke-width: 1; }}
    """
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H:.0f}" '
        f'viewBox="0 0 {W} {H:.0f}" role="img" aria-label="LLM benchmark reference chart">\n'
        f"<style>{style}</style>\n"
        f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1:.0f}" rx="8" '
        f'fill="{t["surface"]}" stroke="{t["border"]}"/>\n'
        + "\n".join(el)
        + "\n</svg>\n"
    )


# --------------------------------------------------------------------- main


def main() -> int:
    check = "--check" in sys.argv
    runs = load_runs()
    results = [r for run in runs for r in run.results]

    readme = README.read_text(encoding="utf-8")
    readme = replace_block(readme, "chart", CHART_MD)
    readme = replace_block(readme, "results", build_table(results))
    readme = replace_block(readme, "runs", build_run_list(runs))

    outputs = {README: readme}
    for theme in THEMES:
        outputs[CHARTS / f"reference-{theme}.svg"] = build_svg(results, theme)

    stale = [p for p, content in outputs.items()
             if not p.exists() or p.read_text(encoding="utf-8") != content]
    if check:
        for p in stale:
            print(f"out of date: {p.relative_to(ROOT)}")
        return 1 if stale else 0
    CHARTS.mkdir(exist_ok=True)
    for p in stale:
        p.write_text(content := outputs[p], encoding="utf-8")
        print(f"wrote {p.relative_to(ROOT)}")
    if not stale:
        print("everything up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
