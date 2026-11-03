import contextlib
import io
import json
import re
import shutil
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
import xml.etree.ElementTree as ET

import test_build
from test_build import build, make_root
import test_runner


def ent(status, values=None):
    return {"status": status, "values": values or {}, "started_at": "2026-11-03T10:00:00Z",
            "finished_at": "2026-11-03T10:01:00Z", "meta": {}}


def run_doc(device, tests, gpu=None):
    doc = {"schema_version": 1, "run": {"device": device, "started_at": "2026-11-03T09:00:00Z"},
           "tests": tests}
    if gpu:
        doc["system"] = {"gpus": [{"name": gpu, "driver": "1.0"}]}
    return doc


def runs_from(extra):
    root = make_root(extra, fixtures=False)
    return root, build.load_runs(root)


def smoke(task, passed, category="code"):
    return ent("ok", {"passed": passed, "category": category})


class OneRunPerSuiteTest(unittest.TestCase):
    def tearDown(self):
        for r in getattr(self, "roots", []):
            shutil.rmtree(r, True)

    def make(self, extra):
        root, runs = runs_from(extra)
        self.roots = [root]
        return runs

    def test_partial_newer_run_replaces_the_whole_suite(self):
        old = {f"llamacpp/S/smoke/code-{i}": smoke(i, True) for i in (1, 2, 3)}
        new = {"llamacpp/S/smoke/code-1": smoke(1, False), "llamacpp/S/smoke/code-9": smoke(9, False)}
        runs = self.make({"2026-12-01-a": run_doc("Dev", old), "2026-12-02-a": run_doc("Dev", new)})
        self.assertEqual(build.smoke_table(runs)[("Dev", "llamacpp", "S")]["code"], (0, 2))

    def test_crash_leftover_does_not_hide_older_run(self):
        old = {f"llamacpp/S/smoke/code-{i}": smoke(i, True) for i in (1, 2, 3)}
        new = {"llamacpp/S/smoke/code-1": ent("running"), "llamacpp/S/smoke/code-2": ent("aborted")}
        runs = self.make({"2026-12-01-a": run_doc("Dev", old), "2026-12-02-a": run_doc("Dev", new)})
        self.assertEqual(build.smoke_table(runs)[("Dev", "llamacpp", "S")]["code"], (3, 3))

    def test_memory_ladder_comes_from_one_run(self):
        mem = lambda n: ent("ok", {"vram_used_mb": 4096, "ctx": n})
        old = {f"llamacpp/S/memory/ctx-{n}": mem(n) for n in (2048, 8192, 32768)}
        new = {"llamacpp/S/memory/ctx-4096": mem(4096)}
        runs = self.make({"2026-12-01-a": run_doc("Dev", old), "2026-12-02-a": run_doc("Dev", new)})
        series = build.memory_series(build.Facts(runs), "Dev")
        self.assertEqual([n for n, *_ in series[0]["points"]], [4096])

    def test_concurrency_levels_come_from_one_run(self):
        lv = lambda n: ent("ok", {"total_tps": 10.0 * n, "serialized": False})
        old = {f"llamacpp/S/concurrency/p{n}": lv(n) for n in (1, 2, 4)}
        new = {"llamacpp/S/concurrency/p8": lv(8)}
        runs = self.make({"2026-12-01-a": run_doc("Dev", old), "2026-12-02-a": run_doc("Dev", new)})
        self.assertEqual(build.concurrency_levels(build.Facts(runs), "Dev"), [8])

    def test_same_device_name_different_gpu_does_not_merge(self):
        t = lambda gen: {"llamacpp/S/speed/warm": ent("ok", {"gen_tps": gen, "prompt_tps": 1.0, "ttft_s": 1.0})}
        runs = self.make({"2026-12-01-a": run_doc("Dev", t(11.0), gpu="GPU One"),
                          "2026-12-02-a": run_doc("Dev", t(22.0), gpu="GPU Two")})
        svg = build.chart_generation(runs, "light")
        self.assertIn("Dev (GPU One)", svg)
        self.assertIn("Dev (GPU Two)", svg)
        self.assertIn("11.0", svg)
        self.assertIn("22.0", svg)

    def test_same_device_same_gpu_merges(self):
        t = lambda gen: {"llamacpp/S/speed/warm": ent("ok", {"gen_tps": gen, "prompt_tps": 1.0, "ttft_s": 1.0})}
        runs = self.make({"2026-12-01-a": run_doc("Dev", t(11.0), gpu="GPU One"),
                          "2026-12-02-a": run_doc("Dev", t(22.0), gpu="GPU One")})
        svg = build.chart_generation(runs, "light")
        self.assertNotIn("GPU One", svg)
        self.assertIn("22.0", svg)
        self.assertNotIn("11.0", svg)


class WidthTest(unittest.TestCase):
    def test_bar_charts_stay_960_wide(self):
        tests = {}
        for m in ("S", "M", "L", "XL"):
            for e in ("llamacpp", "ollama"):
                tests[f"{e}/{m}/speed/warm"] = ent("ok", {"gen_tps": 50.0, "prompt_tps": 900.0, "ttft_s": 1.0})
                tests[f"{e}/{m}/speed/cache"] = ent("ok", {"ttft_s": 0.1})
                tests[f"{e}/{m}/memory/ctx-8192"] = ent("ok", {"vram_used_mb": 5000})
                for n in (1, 2):
                    tests[f"{e}/{m}/concurrency/p{n}"] = ent("ok", {"total_tps": 5.0 * n, "serialized": n == 2})
        extra = {f"2026-12-{i + 1:02d}-d{i}": run_doc(f"Device number {i} with a long name", tests) for i in range(12)}
        root, runs = runs_from(extra)
        self.addCleanup(shutil.rmtree, root, True)
        for name in ("generation", "prompt-processing", "ttft", "max-context", "concurrency", "memory", "smoke"):
            for theme in ("light", "dark"):
                svg = build.CHARTS_BY_NAME[name](runs, theme)
                if not svg:
                    continue
                ET.fromstring(svg)
                self.assertIn('width="960"', svg.split(">", 1)[0], name)

    def test_five_classes_wrap_into_a_second_band(self):
        tests = {f"llamacpp/{m}/speed/warm": ent("ok", {"gen_tps": 50.0, "prompt_tps": 9.0, "ttft_s": 1.0})
                 for m in ("S", "M", "L", "XL", "F")}
        root, runs = runs_from({"2026-12-01-a": run_doc("Dev", tests)})
        self.addCleanup(shutil.rmtree, root, True)
        svg = build.chart_generation(runs, "light")
        self.assertIn('width="960"', svg)
        for m in ("S", "M", "L", "XL", "F"):
            self.assertIn(f">Class {m}<", svg)


class ValueLevelTest(unittest.TestCase):
    def setUp(self):
        warm = ent("ok", {"gen_tps": 123.4, "prompt_tps": 7777.0, "ttft_s": 0.55})
        cache = ent("ok", {"ttft_s": 0.07, "first_ttft_s": 0.55})
        tests = {"llamacpp/S/speed/warm": warm, "llamacpp/S/speed/cache": cache,
                 "ollama/S/speed/warm": ent("error", {"gen_tps": 999.0, "prompt_tps": 999.0, "ttft_s": 9.0, "error": "x"})}
        root, self.runs = runs_from({"2026-12-01-a": run_doc("Dev", tests)})
        self.addCleanup(shutil.rmtree, root, True)

    def test_each_chart_reads_its_own_field(self):
        gen = build.chart_generation(self.runs, "light")
        self.assertIn("123.4", gen)
        self.assertNotIn("7,777", gen)
        pp = build.chart_prompt_processing(self.runs, "light")
        self.assertIn("7,777", pp)
        self.assertNotIn("123.4", pp)

    def test_non_ok_entries_are_never_plotted(self):
        for fn in (build.chart_generation, build.chart_prompt_processing, build.chart_ttft):
            svg = fn(self.runs, "light")
            self.assertNotIn("999", svg)
            self.assertIn("error", svg)

    def test_ttft_chart_has_warm_and_cache_bars(self):
        svg = build.chart_ttft(self.runs, "light")
        self.assertIn("0.55", svg)
        self.assertIn("0.07", svg)
        self.assertIn("prompt cache", svg)

    def test_ramp_bins(self):
        self.assertEqual([build.ramp_step(d, t) for d, t in [(0, 1), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (1, 2)]],
                         [0, 1, 2, 3, 4, 4, 2])

    def test_smoke_cell_fill_follows_ratio(self):
        tests = {f"llamacpp/S/smoke/math-{i}": smoke(i, i <= 4, "math") for i in range(1, 6)}
        root, runs = runs_from({"2026-12-01-a": run_doc("Dev", tests)})
        self.addCleanup(shutil.rmtree, root, True)
        svg = build.chart_smoke(runs, "light")
        self.assertIn(f'fill="{build.RAMP["light"][4]}"', svg)  # 4/5 = 80%
        self.assertIn(">4/5<", svg)


class ContextCellTest(unittest.TestCase):
    def cell(self, rungs):
        tests = {f"llamacpp/S/memory/ctx-{n}": ent(status, {}) for n, status in rungs.items()}
        root, runs = runs_from({"2026-12-01-a": run_doc("Dev", tests)})
        self.addCleanup(shutil.rmtree, root, True)
        return build.context_cell(build.Facts(runs), "Dev", "S", "llamacpp", "", None)

    def test_max_of_ok_rungs(self):
        c = self.cell({2048: "ok", 8192: "error", 32768: "ok"})
        self.assertEqual(c.value, 32768.0)

    def test_plus_only_when_ladder_ended_ok_on_top(self):
        self.assertEqual(self.cell({2048: "ok", 8192: "ok"}).suffix, "+")

    def test_error_or_abort_above_gets_star(self):
        self.assertEqual(self.cell({2048: "ok", 8192: "error"}).suffix, "*")
        self.assertEqual(self.cell({2048: "ok", 8192: "aborted", 32768: "skipped"}).suffix, "*")

    def test_does_not_fit_above_is_a_found_limit(self):
        c = self.cell({2048: "ok", 8192: "does_not_fit"})
        self.assertEqual((c.value, c.suffix), (2048.0, ""))


class MemoryChartTest(unittest.TestCase):
    def runs(self, with_baseline=True, extra_steps=None):
        v = {"vram_used_mb": 6144}
        if with_baseline:
            v["vram_baseline_mb"] = 3072
        tests = {"llamacpp/S/memory/ctx-8192": ent("ok", v),
                 "llamacpp/S/memory/ctx-32768": ent("ok", {**v, "vram_used_mb": 9216}),
                 "llamacpp/S/memory/ctx-131072": ent("does_not_fit", {"reason": "x"}),
                 "ollama/S/memory/ctx-8192": ent("ok", {"ctx": 8192})}
        tests.update(extra_steps or {})
        root, runs = runs_from({"2026-12-01-a": run_doc("Dev", tests)})
        self.addCleanup(shutil.rmtree, root, True)
        return runs

    def test_gib_and_baseline_subtraction(self):
        series = build.memory_series(build.Facts(self.runs()), "Dev")
        ser = next(s for s in series if s["engine"] == "llamacpp")
        self.assertEqual([(n, net) for n, _total, net in ser["points"]], [(8192, 3.0), (32768, 6.0)])
        self.assertEqual([total for _n, total, _net in ser["points"]], [6.0, 9.0])

    def test_baseline_panel_is_labelled_model_memory(self):
        svg = build.chart_memory(self.runs(), "light")
        self.assertIn("model memory", svg)
        self.assertNotIn("includes the desktop", svg)

    def test_without_baseline_total_is_labelled(self):
        svg = build.chart_memory(self.runs(with_baseline=False), "light")
        self.assertIn("total card usage", svg)
        self.assertIn("includes the desktop and other programs", svg)
        series = build.memory_series(build.Facts(self.runs(with_baseline=False)), "Dev")
        ser = next(s for s in series if s["engine"] == "llamacpp")
        self.assertEqual(ser["points"][0][1], 6.0)

    def test_limit_labels_sit_right_of_all_data(self):
        svg = build.chart_memory(self.runs(), "light")
        xs = [float(x) for x in re.findall(r'<circle cx="([\d.]+)"', svg)]
        label_x = [float(m) for m in re.findall(r'<text x="([\d.]+)" y="[\d.]+" class="limitlabel">', svg)]
        self.assertEqual(len(label_x), 4)
        self.assertGreater(min(label_x), max(xs) + 60)

    def test_markers_for_gaps_are_on_the_axis_strip(self):
        svg = build.chart_memory(self.runs(), "light")
        self.assertIn('class="nofit"', svg)      # does_not_fit at 131072
        self.assertIn('class="nofigure"', svg)   # ollama rung ok without vram figure


class MiscTest(unittest.TestCase):
    def test_engine_colors_distinct_and_not_grey(self):
        for theme in ("light", "dark"):
            colors = [build.engine_color(e, theme) for e in build.ENGINES]
            self.assertGreaterEqual(len(build.ENGINES), 4)
            self.assertEqual(len(set(colors)), len(colors))
            self.assertNotIn(build.THEMES[theme]["muted"], colors)

    def test_argparse_rejects_unknown_flag(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as ctx:
            build.main(["--nope"])
        self.assertEqual(ctx.exception.code, 2)


class RunnerProducedTest(unittest.TestCase):
    setUp = test_runner.RunnerTest.setUp
    make_engine = test_runner.RunnerTest.make_engine
    resolve = test_runner.RunnerTest.resolve
    env = test_runner.RunnerTest.env
    cfg = test_runner.RunnerTest.cfg
    run_dir = test_runner.RunnerTest.run_dir
    results = test_runner.RunnerTest.results
    text = test_runner.RunnerTest.text

    def test_runner_output_validates_and_charts(self):
        import itertools
        from unittest import mock
        from bench.safety import Reading
        from bench.suites import memory
        readings = itertools.cycle([Reading(22000, None, None, None), Reading(20000, None, None, None)])
        with mock.patch.object(memory, "take_reading", lambda idx, frac=None: next(readings)):
            code = test_runner.runner.run(self.cfg(engines=["ollama"], suites=["memory", "speed"]), self.env())
        self.assertEqual(code, 0, self.text)
        data = self.results()
        self.assertEqual(build.validate(data), [])
        root = make_root()
        self.addCleanup(shutil.rmtree, root, True)
        shutil.rmtree(root / "runs")
        shutil.copytree(self.run_dir().parent, root / "runs")
        outputs = build.build_outputs(root)
        names = {p.name for p in outputs}
        self.assertIn("generation-light.svg", names)
        self.assertIn("memory-light.svg", names)
        mem = next(t for p, t in outputs.items() if p.name == "memory-light.svg")
        self.assertIn("model memory", mem)  # the memory suite now records vram_baseline_mb


if __name__ == "__main__":
    unittest.main()


class FinalReviewTest(unittest.TestCase):
    def tmp_root(self, extra):
        root = make_root(extra, fixtures=False)
        self.addCleanup(shutil.rmtree, root, True)
        return root

    def good(self):
        return run_doc("Dev", {"llamacpp/S/speed/warm": ent("ok", {"gen_tps": 5.0, "prompt_tps": 6.0, "ttft_s": 1.0})})

    def test_scan_results_json(self):
        doc = self.good()
        doc["tests"]["llamacpp/S/speed/warm"]["values"]["error"] = "connect to 192.168.1.23 failed"
        root = self.tmp_root({"2026-12-01-leak": doc})
        with self.assertRaises(SystemExit) as ctx:
            build.load_runs(root)
        msg = str(ctx.exception)
        self.assertIn("2026-12-01-leak/results.json", msg)
        self.assertIn("ip", msg)
        self.assertNotIn("192.168.1.23", msg)

    def test_scan_run_readme_and_raw(self):
        root = self.tmp_root({"2026-12-01-ok": self.good()})
        run = root / "runs" / "2026-12-01-ok"
        (run / "README.md").write_text("opened C:\\Users\\alice\\models\\x.gguf\n", encoding="utf-8")
        with self.assertRaises(SystemExit) as ctx:
            build.load_runs(root)
        self.assertIn("2026-12-01-ok/README.md", str(ctx.exception))
        self.assertNotIn("alice", str(ctx.exception))
        (run / "README.md").write_text("fine\n", encoding="utf-8")
        (run / "raw").mkdir()
        (run / "raw" / "a.txt").write_text("host 10.0.0.7 answered", encoding="utf-8")
        with self.assertRaises(SystemExit) as ctx:
            build.load_runs(root)
        self.assertIn("raw/a.txt", str(ctx.exception))
        self.assertNotIn("10.0.0.7", str(ctx.exception))

    def test_clean_run_files_load(self):
        root = self.tmp_root({"2026-12-01-ok": self.good()})
        (root / "runs" / "2026-12-01-ok" / "README.md").write_text("all fine\n", encoding="utf-8")
        self.assertEqual(len(build.load_runs(root)), 1)

    def cell(self, rungs):
        tests = {f"llamacpp/S/memory/ctx-{n}": ent(st, vals) for n, (st, vals) in rungs.items()}
        root = self.tmp_root({"2026-12-01-a": run_doc("Dev", tests)})
        return build.context_cell(build.Facts(build.load_runs(root)), "Dev", "S", "llamacpp", "", None)

    def test_crash_converted_rung_is_starred(self):
        c = self.cell({2048: ("ok", {}), 8192: ("does_not_fit", {"previous_status": "aborted"}),
                       32768: ("skipped", {"reason": "lower step 8192 did not fit or crashed"})})
        self.assertEqual((c.value, c.suffix), (2048.0, "*"))
        c = self.cell({2048: ("ok", {}), 8192: ("does_not_fit", {"previous_status": "running"})})
        self.assertEqual(c.suffix, "*")

    def test_driver_reset_unavailable_skip_is_starred(self):
        c = self.cell({2048: ("ok", {}), 8192: ("skipped", {"reason": "driver reset check unavailable after step 2048"})})
        self.assertEqual(c.suffix, "*")

    def test_real_limit_stays_plain(self):
        c = self.cell({2048: ("ok", {}), 8192: ("does_not_fit", {"reason": "only 900 MB would stay free"})})
        self.assertEqual(c.suffix, "")

    def test_has_data_name(self):
        self.assertTrue(hasattr(build, "HAS_DATA"))
        self.assertFalse(hasattr(build, "FINAL_STATUSES"))

    def test_one_bad_rung_does_not_flip_panel_to_total(self):
        tests = {"llamacpp/S/memory/ctx-2048": ent("ok", {"vram_used_mb": 4096, "vram_baseline_mb": 1024}),
                 "llamacpp/S/memory/ctx-8192": ent("ok", {"vram_used_mb": 1000, "vram_baseline_mb": 1024})}
        root = self.tmp_root({"2026-12-01-a": run_doc("Dev", tests)})
        runs = build.load_runs(root)
        ser = build.memory_series(build.Facts(runs), "Dev")[0]
        self.assertEqual([p[0] for p in ser["points"]], [2048])
        self.assertEqual(ser["nofigure"], [8192])
        svg = build.chart_memory(runs, "light")
        self.assertIn("model memory", svg)
        self.assertIn('class="nofigure"', svg)

    def test_units_are_gib(self):
        tests = {"llamacpp/S/memory/ctx-2048": ent("ok", {"vram_used_mb": 4096, "vram_baseline_mb": 1024})}
        root = self.tmp_root({"2026-12-01-a": run_doc("Dev", tests)})
        svg = build.chart_memory(build.load_runs(root), "light")
        self.assertIn("GiB", svg)
        self.assertNotIn("(1024 MiB)", svg)
        self.assertIn("3.0 GiB", svg)

    def test_long_device_name_wraps_in_memory_title(self):
        name = "Fixture Workstation With A Very Long Device Name And Dual Cards Inside Plus Extra Words To Force A Wrap 48 GB"
        tests = {"llamacpp/S/memory/ctx-2048": ent("ok", {"vram_used_mb": 4096, "vram_baseline_mb": 1024})}
        root = self.tmp_root({"2026-12-01-a": run_doc(name, tests)})
        svg = build.chart_memory(build.load_runs(root), "light")
        ET.fromstring(svg)
        self.assertGreaterEqual(svg.count("panel-title"), 2)
