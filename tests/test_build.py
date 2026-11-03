import contextlib
import importlib.util
import io
import json
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "runs"

_spec = importlib.util.spec_from_file_location("build_script", ROOT / "scripts" / "build.py")
build = importlib.util.module_from_spec(_spec)
sys.modules["build_script"] = build
_spec.loader.exec_module(build)

README_TEMPLATE = """# Title

<!-- generated:charts -->
old
<!-- /generated:charts -->

<!-- generated:runs -->
old
<!-- /generated:runs -->
"""


def make_root(extra=None):
    """A temporary repo root with the fixture runs and a README with markers."""
    tmp = Path(tempfile.mkdtemp())
    shutil.copytree(FIXTURES, tmp / "runs")
    (tmp / "README.md").write_text(README_TEMPLATE, encoding="utf-8")
    for name, data in (extra or {}).items():
        (tmp / "runs" / name).mkdir()
        (tmp / "runs" / name / "results.json").write_text(json.dumps(data), encoding="utf-8")
    return tmp


class BuildTests(unittest.TestCase):
    def setUp(self):
        self.root = make_root()
        self.addCleanup(shutil.rmtree, self.root, True)
        self.runs = build.load_runs(self.root)

    def test_load_runs_sorted_and_normalized(self):
        names = [r["dir"] for r in self.runs]
        self.assertEqual(names, sorted(names))
        self.assertEqual(len(self.runs), 4)
        legacy = [r for r in self.runs if r["legacy"]]
        self.assertEqual([r["device"] for r in legacy], ["Fixture Old Device"])

    def test_legacy_skipped(self):
        outputs = build.build_outputs(self.root)
        readme = outputs[self.root / "README.md"]
        line = next(l for l in readme.splitlines() if "Fixture Old Device" in l)
        self.assertIn("old prompt", line)
        for path, text in outputs.items():
            if path.suffix == ".svg":
                self.assertNotIn("Fixture Old Device", text)

    def test_validate_accepts_legacy_and_full(self):
        legacy = {"schema_version": 1, "legacy": True, "device": "X", "date": "2026-10-26"}
        self.assertEqual(build.validate(legacy), [])
        full = json.loads((FIXTURES / "2026-11-03-fixture-big-gpu" / "results.json").read_text())
        self.assertEqual(build.validate(full), [])

    def test_validate_rejects(self):
        full = json.loads((FIXTURES / "2026-11-03-fixture-big-gpu" / "results.json").read_text())
        no_version = {k: v for k, v in full.items() if k != "schema_version"}
        self.assertTrue(build.validate(no_version))
        bad = json.loads(json.dumps(full))
        bad["tests"]["llamacpp/S/speed/warm"]["status"] = "great"
        errors = build.validate(bad)
        self.assertTrue(any("great" in e for e in errors))
        self.assertTrue(build.validate({"schema_version": 1, "legacy": True, "device": "X"}))
        self.assertTrue(build.validate([]))

    def test_invalid_file_stops_build(self):
        make = make_root({"2026-11-04-broken": {"tests": {}}})
        self.addCleanup(shutil.rmtree, make, True)
        with self.assertRaises(SystemExit) as ctx:
            build.load_runs(make)
        self.assertIn("2026-11-04-broken", str(ctx.exception))

    def test_smoke_latest_wins(self):
        table = build.smoke_table(self.runs)
        cells = table[("Fixture Big GPU 24 GB", "llamacpp", "S")]
        self.assertEqual(cells["code"], (1, 3))  # the older run had 3/3
        self.assertEqual(cells["math"], (2, 3))
        self.assertNotIn("creative", cells)  # never judged

    def test_deterministic(self):
        first = build.build_outputs(self.root)
        second = build.build_outputs(self.root)
        self.assertEqual({p: t.encode() for p, t in first.items()},
                         {p: t.encode() for p, t in second.items()})

    def test_write_then_check_clean(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(build.run(self.root, check=False), 0)
            self.assertEqual(build.run(self.root, check=True), 0)
            (self.root / "charts" / "generation-light.svg").write_text("x")
            self.assertEqual(build.run(self.root, check=True), 1)

    def test_no_em_dash(self):
        extra = make_root({"2026-11-05-dash": {
            "schema_version": 1, "run": {"device": "Dash \u2014 Device"},
            "tests": {"llamacpp/S/speed/warm": {"status": "ok", "values": {"gen_tps": 5, "prompt_tps": 6, "ttft_s": 1}}}}})
        self.addCleanup(shutil.rmtree, extra, True)
        for root in (self.root, extra):
            for path, text in build.build_outputs(root).items():
                self.assertNotIn("\u2014", text, path.name)
        out = build.build_outputs(extra)
        self.assertIn("Dash - Device", "".join(out.values()))

    def test_memory_chart_lines(self):
        svg = build.chart_memory(self.runs, "light")
        panels = svg.count("panel-title")
        self.assertEqual(panels, 2)  # one chart per device with memory data, each with its own y axis
        self.assertEqual(len(re.findall(r'class="limit"', svg)), 4 * panels)
        for gb in (8, 12, 16, 24):
            self.assertEqual(svg.count(f">{gb} GB<"), panels)
        self.assertIn("<polyline", svg)

    def test_serialized_marked_not_plain(self):
        svg = build.chart_concurrency(self.runs, "light")
        self.assertIn("url(#hatch)", svg)
        self.assertIn("one after another", svg)

    def test_missing_values_are_gaps_with_legend(self):
        svg = build.chart_generation(self.runs, "light")
        self.assertIn("does not fit", svg)
        self.assertIn("No bar", svg)  # legend entry for gaps
        values = re.findall(r'class="value"[^>]*>(0|0\.0)[ <]', svg)
        self.assertEqual(values, [])

    def test_all_charts_valid_xml_both_themes(self):
        import xml.etree.ElementTree as ET
        for name, fn in build.CHARTS_BY_NAME.items():
            for theme in ("light", "dark"):
                svg = fn(self.runs, theme)
                ET.fromstring(svg)

    def test_picture_markup_in_readme(self):
        readme = build.build_outputs(self.root)[self.root / "README.md"]
        self.assertIn("prefers-color-scheme: dark", readme)
        self.assertIn("charts/memory-dark.svg", readme)
        self.assertIn("<!-- generated:charts -->", readme)

    def test_privacy_scan_blocks_leak(self):
        leaky = make_root({"2026-11-06-leak": {
            "schema_version": 1, "run": {"device": "Box at 203.0.113.7"}, "tests": {
                "llamacpp/S/speed/warm": {"status": "ok", "values": {"gen_tps": 5, "prompt_tps": 6, "ttft_s": 1}}}}})
        self.addCleanup(shutil.rmtree, leaky, True)
        with self.assertRaises(SystemExit) as ctx:
            build.build_outputs(leaky)
        self.assertNotIn("203.0.113.7", str(ctx.exception))

    def test_empty_runs_build(self):
        empty = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, empty, True)
        (empty / "runs").mkdir()
        (empty / "README.md").write_text(README_TEMPLATE, encoding="utf-8")
        out = build.build_outputs(empty)
        self.assertIn("No measured runs yet", out[empty / "README.md"])


if __name__ == "__main__":
    unittest.main()
