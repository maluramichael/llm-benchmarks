import contextlib
import json
import re
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

from bench import runner
from bench.config import Config
from bench.engines.base import EngineError, ModelRef
from bench.models import ModelNotFound, NeedsConsent
from bench.privacy import PrivacyError
from bench.safety import Reading
from bench.store import ResultStore
from bench.suites import RunAborted, memory, speed
from fake_engine import FakeEngine

CORPUS = " ".join(f"w{i}" for i in range(60000))
SYSTEM = {"os": "Test OS", "cpu": "Test CPU", "gpus": [
    {"index": 0, "name": "NVIDIA GeForce RTX 3090", "vram_total_mb": 24576, "vram_free_mb": 23000,
     "driver": "1.0", "backend": "cuda", "vendor": "nvidia"}], "ram_gb": 64,
    "memory_for_models_gb": 24, "laptop": False}
DEVICE = "PC RTX 3090 24 GB"


def io_stringio():
    import io
    return io.StringIO()


class FakeGuard:
    def __init__(self, cfg, gpu_index, vram_total_mb, reset=False):
        self.cfg, self.gpu_index, self.vram_total_mb = cfg, gpu_index, vram_total_mb
        self.reset = reset

    def reserve_mb(self):
        return 1536

    def before_step(self, required_mb, reading):
        return True, "ok"

    def after_step(self, prev, now):
        return True, "ok"

    def driver_reset_since(self, ts):
        return self.reset


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root / "runs"
        self.messages: list[str] = []
        self.engines: dict[str, list[FakeEngine]] = {}
        for module in (memory, speed):
            patcher = mock.patch.object(module, "take_reading",
                                        lambda idx, frac=None: Reading(20000, frac, None, None))
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch.object(runner, "local_secrets", return_value=[])
        patcher.start()
        self.addCleanup(patcher.stop)

    # helpers
    def make_engine(self, name, cfg):
        engine = FakeEngine(name=name)
        self.engines.setdefault(name, []).append(engine)
        engine.stopped = 0
        original = engine.stop

        def stop():
            engine.stopped += 1
            original()
        engine.stop = stop
        return engine

    def resolve(self, model_id, engine, cfg, say=None):
        return ModelRef(model_id, model_id, "Q4_K_M", "tag", None, "sha256:" + model_id,
                        2_000_000_000)

    def env(self, **kw):
        values = dict(
            system=lambda: json.loads(json.dumps(SYSTEM)),
            doctor=lambda cfg: [{"check": "uptime", "status": "ok", "text": "up for 5 min"}],
            preflight=lambda cfg, gpu: [],
            make_engine=self.make_engine, resolve=self.resolve, guard=FakeGuard,
            corpus=lambda cfg: CORPUS, today=lambda: date(2026, 11, 3),
            keep_awake=lambda: contextlib.nullcontext(), say=self.messages.append)
        values.update(kw)
        return runner.Env(**values)

    def cfg(self, **kw):
        values = dict(engines=["ollama"], models=["S"], suites=["memory", "speed"],
                      ctx_steps=[8192], reps=2, output=self.out, device=DEVICE)
        values.update(kw)
        return Config(**values)

    def run_dir(self):
        found = sorted(self.out.iterdir())
        self.assertEqual(len(found), 1, found)
        return found[0]

    def results(self):
        return json.loads((self.run_dir() / "results.json").read_text(encoding="utf-8"))

    @property
    def text(self):
        return "\n".join(self.messages)

    # tests from the brief
    def test_run_end_to_end_fake(self):
        code = runner.run(self.cfg(), self.env())
        self.assertEqual(code, 0, self.text)
        folder = self.run_dir()
        self.assertEqual(folder.name, "2026-11-03-pc-rtx-3090-24gb")
        self.assertTrue((folder / "results.json").is_file())
        self.assertTrue((folder / "README.md").is_file())
        self.assertTrue((folder / "raw").is_dir())
        data = self.results()
        self.assertIn("ollama/S/speed/warm", data["tests"])
        self.assertIn("ollama/S/memory/ctx-8192", data["tests"])
        self.assertEqual(data["tests"]["ollama/S/speed/warm"]["status"], "ok")
        readme = (folder / "README.md").read_text(encoding="utf-8")
        self.assertIn("## ollama / S", readme)
        self.assertNotIn("\u2014", readme)

    def test_run_resume(self):
        self.assertEqual(runner.run(self.cfg(), self.env()), 0)
        before = (self.run_dir() / "results.json").read_text(encoding="utf-8")
        later = self.env(today=lambda: date(2026, 11, 4))
        self.engines.clear()
        self.assertEqual(runner.run(self.cfg(), later), 0, self.text)
        engine = self.engines["ollama"][0]
        self.assertEqual([c for c in engine.calls if c.startswith("load") or c == "complete"], [])
        self.assertEqual(self.run_dir().name, "2026-11-03-pc-rtx-3090-24gb")
        after = json.loads((self.run_dir() / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(after["tests"], json.loads(before)["tests"])
        self.assertEqual(after["run"]["started_at"], json.loads(before)["run"]["started_at"])

    def test_engines_stopped_on_error(self):
        def boom(ctx):
            raise ValueError("suite exploded")
        env = self.env()
        env.suites = {**env.suites, "speed": boom}
        code = runner.run(self.cfg(), env)
        self.assertEqual(code, 1)
        self.assertEqual(self.engines["ollama"][0].stopped, 1)
        self.assertIn("suite exploded", self.text)
        self.assertTrue((self.run_dir() / "README.md").is_file())

    def test_run_dir_name(self):
        self.assertEqual(runner.run_dir_name(date(2026, 11, 3), "PC RTX 3080 Ti 12 GB"),
                         "2026-11-03-pc-rtx-3080-ti-12gb")
        self.assertEqual(runner.run_dir_name(date(2026, 11, 3), "MacBook Pro M5 Max 128 GB"),
                         "2026-11-03-macbook-pro-m5-max-128gb")
        self.assertEqual(runner.run_dir_name(date(2026, 1, 2), "  Weird / Name!! "),
                         "2026-01-02-weird-name")

    def test_run_dir_is_continued_for_the_same_device(self):
        runs = self.out
        (runs / "2026-11-01-pc-rtx-3090-24gb").mkdir(parents=True)
        (runs / "2026-11-01-pc-rtx-3090-24gb" / "results.json").write_text(
            json.dumps({"schema_version": 1, "tests": {}}), encoding="utf-8")
        (runs / "2026-10-26-pc-rtx-3090-24gb").mkdir()
        (runs / "2026-10-26-pc-rtx-3090-24gb" / "results.json").write_text(
            json.dumps({"schema_version": 1, "legacy": True, "device": "x", "date": "y"}),
            encoding="utf-8")
        (runs / "2026-11-02-pc-rtx-3090-24gb-extra").mkdir()
        found = runner.find_run_dir(runs, DEVICE, date(2026, 11, 3))
        self.assertEqual(found.name, "2026-11-01-pc-rtx-3090-24gb")

    def test_new_device_gets_a_new_folder_and_legacy_is_never_continued(self):
        runs = self.out
        legacy = runs / "2026-10-26-pc-rtx-3090-24gb"
        legacy.mkdir(parents=True)
        (legacy / "results.json").write_text(
            json.dumps({"schema_version": 1, "legacy": True, "device": "x", "date": "y"}),
            encoding="utf-8")
        self.assertEqual(runner.find_run_dir(runs, DEVICE, date(2026, 11, 3)).name,
                         "2026-11-03-pc-rtx-3090-24gb")
        self.assertEqual(runner.find_run_dir(runs, "Other PC", date(2026, 11, 3)).name,
                         "2026-11-03-other-pc")
        self.assertEqual(runner.find_run_dir(self.root / "missing", "Other PC",
                                             date(2026, 11, 3)).name, "2026-11-03-other-pc")

    # results content
    def test_results_carry_system_engines_models_and_run(self):
        env = self.env()
        env.make_engine = lambda name, cfg: self._with_meta(self.make_engine(name, cfg))
        self.assertEqual(runner.run(self.cfg(thinking=False), env), 0, self.text)
        data = self.results()
        self.assertEqual(data["system"]["os"], "Test OS")
        self.assertEqual(data["run"]["device"], DEVICE)
        self.assertEqual(data["run"]["bench_version"], "1.0.0")
        self.assertRegex(data["run"]["started_at"], r"^\d{4}-\d\d-\d\dT")
        engine = data["engines"]["ollama"]
        self.assertEqual(engine["version"], "fake-1")
        self.assertEqual(engine["kv_cache_type"], "f16")
        self.assertEqual(engine["llama_arg_env_removed"], ["LLAMA_ARG_X"])
        self.assertIs(engine["thinking"], False)
        self.assertEqual(data["models"]["ollama/S"]["quant"], "Q4_K_M")
        self.assertEqual(data["doctor"], [{"check": "uptime", "status": "ok",
                                           "text": "up for 5 min"}])

    @staticmethod
    def _with_meta(engine):
        engine.meta = {"kv_cache_type": "f16", "flash_attn": "on",
                       "llama_arg_env_removed": ["LLAMA_ARG_X"]}
        return engine

    def test_doctor_detail_is_printed_but_never_stored(self):
        rows = [{"check": "processes", "status": "ok", "text": "no big process",
                 "detail": "ComfyUI 9000 MB"}]
        self.assertEqual(runner.run(self.cfg(), self.env(doctor=lambda cfg: rows)), 0)
        self.assertIn("ComfyUI 9000 MB", self.text)
        raw = (self.run_dir() / "results.json").read_text(encoding="utf-8")
        self.assertNotIn("ComfyUI", raw)
        self.assertNotIn("detail", raw)

    def test_process_names_in_a_doctor_warning_are_not_stored(self):
        rows = [{"check": "processes", "status": "warn",
                 "text": "large processes: ComfyUI 9000 MB"}]
        runner.run(self.cfg(), self.env(doctor=lambda cfg: rows))
        raw = (self.run_dir() / "results.json").read_text(encoding="utf-8")
        self.assertNotIn("ComfyUI", raw)
        self.assertIn('"check": "processes"', raw)

    def test_device_name_is_scrubbed_before_it_becomes_a_folder_name(self):
        with mock.patch.object(runner, "local_secrets", return_value=["nero"]):
            runner.run(self.cfg(device="PC on nero"), self.env())
        self.assertNotIn("nero", self.run_dir().name)
        self.assertNotIn("nero", self.results()["run"]["device"])

    def test_device_defaults_to_the_suggested_name(self):
        runner.run(self.cfg(device=None), self.env())
        self.assertIn("rtx-3090-24gb", self.run_dir().name)

    # suites
    def test_suite_order_is_memory_speed_smoke_concurrency(self):
        order = []
        env = self.env()
        env.suites = {name: (lambda ctx, n=name: order.append(n))
                      for name in ("memory", "speed", "smoke", "concurrency")}
        cfg = self.cfg(suites=["concurrency", "smoke", "speed", "memory"])
        self.assertEqual(runner.run(cfg, env), 0)
        self.assertEqual(order, ["memory", "speed", "smoke", "concurrency"])

    def test_only_selected_suites_run(self):
        order = []
        env = self.env()
        env.suites = {name: (lambda ctx, n=name: order.append(n))
                      for name in ("memory", "speed", "smoke", "concurrency")}
        runner.run(self.cfg(suites=["smoke"]), env)
        self.assertEqual(order, ["smoke"])

    def test_every_engine_and_model_in_order(self):
        seen = []
        env = self.env()
        env.suites = {"memory": lambda ctx: seen.append((ctx.engine.name, ctx.model.id))}
        runner.run(self.cfg(engines=["llamacpp", "ollama"], models=["S", "M"],
                           suites=["memory"]), env)
        self.assertEqual(seen, [("llamacpp", "S"), ("llamacpp", "M"), ("ollama", "S"),
                                ("ollama", "M")])
        self.assertEqual([e.stopped for es in self.engines.values() for e in es], [1, 1])

    def test_suites_get_a_complete_context(self):
        captured = {}
        env = self.env()

        def grab(ctx):
            captured.update(guard=ctx.guard, raw=ctx.raw_dir, corpus=ctx.corpus_text,
                            store=ctx.store)
        env.suites = {"memory": grab}
        runner.run(self.cfg(suites=["memory"]), env)
        self.assertEqual(captured["guard"].gpu_index, 0)
        self.assertEqual(captured["guard"].vram_total_mb, 24576)
        self.assertEqual(captured["raw"], self.run_dir() / "raw")
        self.assertEqual(captured["corpus"], CORPUS)

    def test_gpu_option_picks_the_guard_card(self):
        system = json.loads(json.dumps(SYSTEM))
        system["gpus"].append({"index": 1, "name": "Second", "vram_total_mb": 8192,
                               "vram_free_mb": 8000})
        captured = {}
        env = self.env(system=lambda: system)
        env.suites = {"memory": lambda ctx: captured.update(g=ctx.guard)}
        runner.run(self.cfg(suites=["memory"], gpu=1), env)
        self.assertEqual((captured["g"].gpu_index, captured["g"].vram_total_mb), (1, 8192))

    # failures
    def test_run_aborted_stops_saves_and_reports(self):
        def abort(ctx):
            ctx.store.save("ollama/S/memory/ctx-8192", "aborted", {"reason": "reset"})
            raise RunAborted("GPU driver reset detected during ollama/S/memory/ctx-8192")
        env = self.env()
        env.suites = {**env.suites, "memory": abort}
        code = runner.run(self.cfg(engines=["ollama", "llamacpp"]), env)
        self.assertEqual(code, runner.EXIT_ABORTED)
        self.assertNotEqual(code, 0)
        self.assertEqual(self.engines["ollama"][0].stopped, 1)
        self.assertNotIn("llamacpp", self.engines)  # nothing else is started after a reset
        self.assertIn("driver reset", self.text)
        self.assertIn("run the same command again", self.text.lower())
        self.assertTrue((self.run_dir() / "README.md").is_file())
        self.assertEqual(self.results()["tests"]["ollama/S/memory/ctx-8192"]["status"], "aborted")

    def test_keyboard_interrupt_stops_engines_and_keeps_the_store_consistent(self):
        def interrupt(ctx):
            ctx.store.mark_running("ollama/S/speed/warm", {})
            raise KeyboardInterrupt
        env = self.env()
        env.suites = {**env.suites, "speed": interrupt}
        code = runner.run(self.cfg(), env)
        self.assertEqual(code, 130)
        self.assertEqual(self.engines["ollama"][0].stopped, 1)
        entry = self.results()["tests"]["ollama/S/speed/warm"]
        self.assertEqual(entry["status"], "error")  # retried by the next run, not a crash
        self.assertIn("interrupted", entry["values"]["reason"])

    def test_leftover_running_entry_is_recovered_and_noted(self):
        run_dir = self.out / "2026-11-03-pc-rtx-3090-24gb"
        store = ResultStore(run_dir / "results.json")
        store.mark_running("ollama/S/memory/ctx-16384", {})
        self.assertEqual(runner.run(self.cfg(ctx_steps=[8192, 16384]), self.env()), 0)
        data = self.results()
        self.assertIn(data["tests"]["ollama/S/memory/ctx-16384"]["status"],
                      ("does_not_fit", "aborted"))
        self.assertTrue(any("ollama/S/memory/ctx-16384" in note for note in data["notes"]))

    def test_unresolved_model_is_reported_and_the_rest_continues(self):
        def resolve(model_id, engine, cfg, say=None):
            if model_id == "M":
                raise ModelNotFound("no M here")
            return self.resolve(model_id, engine, cfg)
        seen = []
        env = self.env(resolve=resolve)
        env.suites = {"memory": lambda ctx: seen.append(ctx.model.id)}
        code = runner.run(self.cfg(models=["M", "S"], suites=["memory"]), env)
        self.assertEqual(seen, ["S"])
        self.assertEqual(code, 1)
        self.assertIn("no M here", self.text)

    def test_download_consent_is_explained(self):
        def resolve(model_id, engine, cfg, say=None):
            raise NeedsConsent("Ollama model x", 23_000_000_000)
        code = runner.run(self.cfg(), self.env(resolve=resolve))
        self.assertEqual(code, 1)
        self.assertIn("--yes", self.text)
        self.assertEqual(self.engines, {})  # nothing to measure: engine never started

    def test_engine_start_failure_does_not_stop_the_other_engine(self):
        def make(name, cfg):
            engine = self.make_engine(name, cfg)
            if name == "llamacpp":
                engine.start = mock.Mock(side_effect=EngineError("no binaries"))
            return engine
        seen = []
        env = self.env(make_engine=make)
        env.suites = {"memory": lambda ctx: seen.append(ctx.engine.name)}
        code = runner.run(self.cfg(engines=["llamacpp", "ollama"], suites=["memory"]), env)
        self.assertEqual(seen, ["ollama"])
        self.assertEqual(code, 1)
        self.assertIn("no binaries", self.text)
        self.assertEqual(self.engines["llamacpp"][0].stopped, 1)

    def test_model_that_cannot_fit_is_recorded_and_not_run(self):
        def resolve(model_id, engine, cfg, say=None):
            return ModelRef(model_id, model_id, "Q4_K_M", "t", None, None, 30 * 1024**3)
        seen = []
        env = self.env(resolve=resolve)
        env.suites = {"memory": lambda ctx: seen.append(1)}
        code = runner.run(self.cfg(suites=["memory"]), env)
        self.assertEqual(seen, [])
        entry = self.results()["tests"]["ollama/S/memory/fit"]
        self.assertEqual(entry["status"], "does_not_fit")
        self.assertIn("weights", entry["values"]["reason"])
        self.assertEqual(code, 0)

    def test_failed_test_makes_the_exit_code_nonzero(self):
        def fail(kind, index):
            if kind == "complete":
                return EngineError("boom")
        env = self.env()
        env.make_engine = lambda name, cfg: self._failing(name, fail)
        code = runner.run(self.cfg(suites=["speed"]), env)
        self.assertEqual(code, 1)
        self.assertIn("error", self.text)

    def _failing(self, name, fail):
        engine = FakeEngine(name=name, fail_on=fail)
        self.engines.setdefault(name, []).append(engine)
        return engine

    # checks before the run
    def test_preflight_warns_and_notes_without_names(self):
        rows = [{"kind": "gpu_busy", "text": "1 other process uses GPU 0: ComfyUI.exe",
                 "note": "another process was using the benchmark GPU at the start"}]
        code = runner.run(self.cfg(), self.env(preflight=lambda cfg, gpu: rows))
        self.assertEqual(code, 0)
        self.assertIn("ComfyUI.exe", self.text)
        data = self.results()
        self.assertIn("another process was using the benchmark GPU at the start", data["notes"])
        self.assertNotIn("ComfyUI", json.dumps(data))

    def test_preflight_aborts_with_the_flag_and_touches_nothing(self):
        rows = [{"kind": "foreign_ollama", "text": "foreign Ollama on 11434", "note": "n"}]
        code = runner.run(self.cfg(abort_if_busy=True), self.env(preflight=lambda c, g: rows))
        self.assertEqual(code, 1)
        self.assertEqual(self.engines, {})
        self.assertFalse(self.out.exists())
        self.assertIn("foreign Ollama on 11434", self.text)

    def test_unavailable_reset_detection_is_a_warning(self):
        class Blind(FakeGuard):
            def driver_reset_since(self, ts):
                return None
        code = runner.run(self.cfg(), self.env(guard=Blind))
        self.assertEqual(code, 0)
        self.assertIn("reset detection is unavailable", self.text)
        self.assertTrue(any("reset detection" in n for n in self.results()["notes"]))

    def test_notes_are_not_duplicated_on_resume(self):
        rows = [{"kind": "gpu_busy", "text": "t", "note": "busy at start"}]
        env = self.env(preflight=lambda cfg, gpu: rows)
        runner.run(self.cfg(), env)
        runner.run(self.cfg(), env)
        self.assertEqual(self.results()["notes"].count("busy at start"), 1)

    def test_hand_written_notes_survive(self):
        runner.run(self.cfg(), self.env())
        path = self.run_dir() / "results.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["notes"] = ["written by hand"]
        path.write_text(json.dumps(data), encoding="utf-8")
        runner.run(self.cfg(), self.env())
        self.assertIn("written by hand", self.results()["notes"])

    def test_dry_run_shows_the_plan_and_changes_nothing(self):
        code = runner.run(self.cfg(dry_run=True), self.env())
        self.assertEqual(code, 0)
        self.assertFalse(self.out.exists())
        self.assertEqual(self.engines, {})
        self.assertIn("S", self.text)

    def test_folder_without_readable_results_is_never_continued(self):
        folder = self.out / "2026-11-03-pc-rtx-3090-24gb"
        folder.mkdir(parents=True)
        (folder / "results.json").write_text("{not json", encoding="utf-8")
        empty = self.out / "2026-11-02-pc-rtx-3090-24gb"
        empty.mkdir()
        self.assertEqual(runner.run(self.cfg(), self.env()), 0, self.text)
        self.assertEqual((folder / "results.json").read_text(encoding="utf-8"), "{not json")
        self.assertTrue((self.out / "2026-11-03-pc-rtx-3090-24gb-2" / "results.json").is_file())
        self.assertFalse((empty / "results.json").exists())

    def test_find_run_dir_picks_a_free_name(self):
        for name in ("2026-11-03-pc", "2026-11-03-pc-2"):
            (self.out / name).mkdir(parents=True)
        self.assertEqual(runner.find_run_dir(self.out, "PC", date(2026, 11, 3)).name,
                         "2026-11-03-pc-3")

    def test_corpus_failure_is_a_clear_error(self):
        def corpus(cfg):
            raise ValueError("corpus broken")
        code = runner.run(self.cfg(), self.env(corpus=corpus))
        self.assertEqual(code, 1)
        self.assertIn("corpus broken", self.text)
        self.assertEqual(self.engines, {})

    def test_privacy_error_in_the_report_is_visible_and_fails_the_run(self):
        with mock.patch.object(runner.report, "write_report",
                               side_effect=PrivacyError(["ip: 10.0.0.1"])):
            code = runner.run(self.cfg(), self.env())
        self.assertEqual(code, 1)
        self.assertIn("10.0.0.1", self.text)
        self.assertTrue((self.run_dir() / "results.json").is_file())

    def test_keep_awake_wraps_the_whole_run(self):
        events = []

        @contextlib.contextmanager
        def awake():
            events.append("on")
            yield
            events.append("off")
        env = self.env(keep_awake=awake)
        env.suites = {"memory": lambda ctx: events.append("suite")}
        runner.run(self.cfg(suites=["memory"]), env)
        self.assertEqual(events, ["on", "suite", "off"])

    def test_keep_awake_is_released_after_an_interrupt(self):
        events = []

        @contextlib.contextmanager
        def awake():
            events.append("on")
            try:
                yield
            finally:
                events.append("off")

        def interrupt(ctx):
            raise KeyboardInterrupt
        env = self.env(keep_awake=awake)
        env.suites = {"memory": interrupt}
        self.assertEqual(runner.run(self.cfg(suites=["memory"]), env), 130)
        self.assertEqual(events, ["on", "off"])

    def test_force_is_passed_to_the_suites(self):
        runner.run(self.cfg(), self.env())
        self.engines.clear()
        runner.run(self.cfg(force=True), self.env())
        engine = self.engines["ollama"][0]
        self.assertGreater(engine.complete_count, 0)

    # fix round 1
    def test_aborted_steps_are_listed_but_do_not_fail_a_later_clean_run(self):
        def abort(ctx):
            ctx.store.save("ollama/S/memory/ctx-8192", "aborted", {"reason": "reset"})
            raise RunAborted("GPU driver reset detected")
        first = self.env()
        first.suites = {**first.suites, "memory": abort}
        self.assertEqual(runner.run(self.cfg(), first), runner.EXIT_ABORTED)
        self.messages.clear()
        second = self.env()
        second.suites = {name: (lambda ctx: None) for name in second.suites}
        self.assertEqual(runner.run(self.cfg(), second), 0, self.text)
        self.assertIn("--force-test ollama/S/memory/ctx-8192", self.text)
        self.assertIn("driver reset", self.text)
        self.assertNotIn("retry them", self.text)

    def test_error_steps_still_fail_with_the_retry_advice(self):
        def err(ctx):
            ctx.store.save("ollama/S/speed/warm", "error", {"error": "x"})
        env = self.env()
        env.suites = {**env.suites, "speed": err}
        self.assertEqual(runner.run(self.cfg(), env), 1)
        self.assertIn("run the same command again to retry", self.text)

    def test_resolve_crash_is_a_problem_not_a_traceback(self):
        def resolve(model_id, engine, cfg, say=None):
            if model_id == "M":
                raise OSError("the share is gone")
            return self.resolve(model_id, engine, cfg)
        seen = []
        env = self.env(resolve=resolve)
        env.suites = {"memory": lambda ctx: seen.append(ctx.model.id)}
        code = runner.run(self.cfg(models=["M", "S"], suites=["memory"]), env)
        self.assertEqual((code, seen), (1, ["S"]))
        self.assertIn("the share is gone", self.text)
        self.assertTrue((self.run_dir() / "README.md").is_file())

    def test_engine_construction_failure_does_not_skip_the_other_engine(self):
        def make(name, cfg):
            if name == "llamacpp":
                raise RuntimeError("cannot build engine")
            return self.make_engine(name, cfg)
        seen = []
        env = self.env(make_engine=make)
        env.suites = {"memory": lambda ctx: seen.append(ctx.engine.name)}
        code = runner.run(self.cfg(engines=["llamacpp", "ollama"], suites=["memory"]), env)
        self.assertEqual((code, seen), (1, ["ollama"]))
        self.assertIn("cannot build engine", self.text)
        self.assertTrue((self.run_dir() / "README.md").is_file())

    def test_unexpected_suite_error_turns_running_entries_into_retryable_errors(self):
        def boom(ctx):
            ctx.store.mark_running("ollama/S/speed/warm", {})
            raise ValueError("broken suite")
        env = self.env()
        env.suites = {**env.suites, "speed": boom}
        runner.run(self.cfg(), env)
        entry = self.results()["tests"]["ollama/S/speed/warm"]
        self.assertEqual(entry["status"], "error")
        self.runs_again = runner.run(self.cfg(), self.env())
        self.assertNotEqual(self.results()["tests"]["ollama/S/speed/warm"]["status"], "aborted")

    def test_continuing_a_run_is_announced_and_the_first_system_is_kept(self):
        runner.run(self.cfg(), self.env())
        first = self.results()
        self.messages.clear()
        changed = json.loads(json.dumps(SYSTEM))
        changed["gpus"][0]["driver"] = "2.0"
        runner.run(self.cfg(), self.env(system=lambda: changed))
        self.assertIn("continuing run", self.text)
        self.assertIn(self.run_dir().name, self.text)
        self.assertRegex(self.text, r"started 2026-\d\d-\d\dT")
        self.assertRegex(self.text, r"with \d+ results")
        data = self.results()
        self.assertEqual(data["system_first"], first["system"])
        self.assertEqual(data["system"]["gpus"][0]["driver"], "2.0")
        self.assertIn("driver", self.text)
        self.assertTrue(any("driver" in n and "1.0" in n and "2.0" in n for n in data["notes"]))

    def test_first_run_does_not_claim_to_continue(self):
        runner.run(self.cfg(), self.env())
        self.assertNotIn("continuing run", self.text)

    def test_changed_engine_and_bench_versions_are_noted(self):
        runner.run(self.cfg(), self.env())
        self.messages.clear()

        def make(name, cfg):
            engine = self.make_engine(name, cfg)
            engine.version = lambda: "fake-2"
            return engine
        with mock.patch.object(runner, "VERSION", "9.9.9"):
            runner.run(self.cfg(), self.env(make_engine=make))
        notes = " ".join(self.results()["notes"])
        self.assertIn("engine version", notes)
        self.assertIn("fake-1", notes)
        self.assertIn("fake-2", notes)
        self.assertIn("bench version", notes)
        self.assertIn("warning", self.text)

    def test_same_setup_adds_no_change_warnings(self):
        runner.run(self.cfg(), self.env())
        self.messages.clear()
        runner.run(self.cfg(), self.env())
        self.assertNotIn("changed since", self.text)

    def test_skip_rule_includes_the_reserve(self):
        def resolve_mib(mib):
            return lambda model_id, engine, cfg, say=None: ModelRef(
                model_id, model_id, "Q4_K_M", "t", None, None, mib * 1024 * 1024)
        seen = []
        env = self.env(resolve=resolve_mib(24576 - 1536 + 100))  # fits the card, not with reserve
        env.suites = {"memory": lambda ctx: seen.append(1)}
        runner.run(self.cfg(suites=["memory"]), env)
        self.assertEqual(seen, [])
        self.assertEqual(self.results()["tests"]["ollama/S/memory/fit"]["status"], "does_not_fit")

    def test_skip_rule_lets_a_model_through_that_fits_with_reserve(self):
        seen = []
        env = self.env(resolve=lambda model_id, engine, cfg, say=None: ModelRef(
            model_id, model_id, "Q4_K_M", "t", None, None, (24576 - 1536 - 100) * 1024 * 1024))
        env.suites = {"memory": lambda ctx: seen.append(1)}
        runner.run(self.cfg(suites=["memory"]), env)
        self.assertEqual(seen, [1])

    def test_preflight_detail_is_printed_but_not_stored(self):
        rows = [{"kind": "gpu_busy", "text": "1 process", "detail": "ComfyUI pid 5 C 9000MiB",
                 "note": "busy"}]
        runner.run(self.cfg(), self.env(preflight=lambda cfg, gpu: rows))
        self.assertIn("ComfyUI pid 5", self.text)
        self.assertNotIn("ComfyUI", json.dumps(self.results()))

    def test_every_step_prints_a_start_and_an_end_line(self):
        def steps(ctx):
            ctx.store.mark_running("ollama/S/memory/ctx-8192", {})
            ctx.store.save("ollama/S/memory/ctx-8192", "ok", {})
            ctx.store.save("ollama/S/memory/ctx-16384", "does_not_fit", {"reason": "x"})
        env = self.env()
        env.suites = {"memory": steps}
        runner.run(self.cfg(suites=["memory"]), env)
        lines = self.messages
        self.assertIn("ollama/S/memory/ctx-8192 ...", lines)
        self.assertTrue(any(re.fullmatch(r"ollama/S/memory/ctx-8192 \.\.\. ok \d+ s", l)
                            for l in lines), lines)
        self.assertTrue(any(l.startswith("ollama/S/memory/ctx-16384 ... does_not_fit")
                            for l in lines))

    def test_default_output_is_flushed_per_line(self):
        with mock.patch("builtins.print") as printed:
            runner.Env().say("hello")
        self.assertEqual(printed.call_args.kwargs.get("flush"), True)

    # fix round 2
    def test_a_numbered_folder_is_continued(self):
        for name in ("2026-11-03-pc-rtx-3090-24gb", "2026-11-03-pc-rtx-3090-24gb-2"):
            (self.out / name).mkdir(parents=True)
        (self.out / "2026-11-03-pc-rtx-3090-24gb" / "results.json").write_text("{bad", "utf-8")
        ResultStore(self.out / "2026-11-03-pc-rtx-3090-24gb-2" / "results.json").save(
            "ollama/S/speed/warm", "ok", {})
        found = runner.find_run_dir(self.out, DEVICE, date(2026, 11, 9))
        self.assertEqual(found.name, "2026-11-03-pc-rtx-3090-24gb-2")
        self.assertEqual(runner.run(self.cfg(), self.env()), 0, self.text)
        self.assertEqual(sorted(p.name for p in self.out.iterdir()),
                         ["2026-11-03-pc-rtx-3090-24gb", "2026-11-03-pc-rtx-3090-24gb-2"])
        self.assertIn("continuing run 2026-11-03-pc-rtx-3090-24gb-2", self.text)

    def test_numbered_folders_sort_by_number_and_slugs_with_digits_are_not_misparsed(self):
        for name in ("2026-11-03-pc-2", "2026-11-03-pc-10", "2026-11-03-pc-3090"):
            (self.out / name).mkdir(parents=True)
            ResultStore(self.out / name / "results.json").save("a/b/c/d", "ok", {})
        self.assertEqual(runner.find_run_dir(self.out, "PC", date(2026, 11, 4)).name,
                         "2026-11-03-pc-10")
        self.assertEqual(runner.find_run_dir(self.out, "PC 3090", date(2026, 11, 4)).name,
                         "2026-11-03-pc-3090")
        self.assertEqual(runner.find_run_dir(self.out, "PC 30", date(2026, 11, 4)).name,
                         "2026-11-04-pc-30")

    def test_report_and_plan_find_a_numbered_folder(self):
        from bench import cli
        folder = self.out / "2026-11-03-pc-test-2"
        ResultStore(folder / "results.json").save("ollama/S/speed/warm", "ok", {"gen_tps": 7.5})
        with contextlib.redirect_stdout(io_stringio()):
            self.assertEqual(cli.main(["report", "--output", str(self.out),
                                       "--device", "PC Test"]), 0)
        self.assertTrue((folder / "README.md").is_file())

    def test_report_failure_is_the_last_line_even_after_an_abort(self):
        def abort(ctx):
            raise RunAborted("GPU driver reset detected")
        env = self.env()
        env.suites = {**env.suites, "memory": abort}
        with mock.patch.object(runner.report, "write_report",
                               side_effect=PrivacyError(["ip: 10.0.0.1"])):
            code = runner.run(self.cfg(), env)
        self.assertEqual(code, runner.EXIT_ABORTED)
        self.assertIn("report was not written", self.messages[-1])
        self.assertIn("10.0.0.1", self.messages[-1])

    def test_interrupt_keeps_130_and_report_failure_last(self):
        def interrupt(ctx):
            raise KeyboardInterrupt
        env = self.env()
        env.suites = {**env.suites, "memory": interrupt}
        with mock.patch.object(runner.report, "write_report", side_effect=OSError("disk")):
            code = runner.run(self.cfg(), env)
        self.assertEqual(code, 130)
        self.assertIn("report was not written", self.messages[-1])

    def test_resolve_and_engine_failures_leave_a_note(self):
        def resolve(model_id, engine, cfg, say=None):
            if model_id == "M":
                raise OSError("the share is gone")
            if model_id == "L":
                raise ModelNotFound("no L here")
            return self.resolve(model_id, engine, cfg)

        def make(name, cfg):
            if name == "llamacpp":
                raise RuntimeError("cannot build engine")
            return self.make_engine(name, cfg)
        seen = []
        env = self.env(resolve=resolve, make_engine=make)
        env.suites = {"memory": lambda ctx: seen.append((ctx.engine.name, ctx.model.id))}
        code = runner.run(self.cfg(engines=["llamacpp", "ollama"], models=["S", "M", "L"],
                                   suites=["memory"]), env)
        self.assertEqual((code, seen), (1, [("ollama", "S")]))
        notes = "\n".join(self.results()["notes"])
        for expected in ("ollama/M", "the share is gone", "ollama/L", "no L here",
                         "llamacpp: cannot create the engine", "cannot build engine"):
            self.assertIn(expected, notes)
        self.assertIn("cannot create the engine", self.text)  # the inner catch, not the outer

    def test_unreliable_vram_does_not_skip_a_model_in_the_runner(self):
        system = json.loads(json.dumps(SYSTEM))
        system["gpus"][0].update(vendor="amd", vram_total_mb=4096)
        big = lambda model_id, engine, cfg, say=None: ModelRef(  # noqa: E731
            model_id, model_id, "Q4_K_M", "t", None, None, 20 * 1024**3)
        seen = []
        for platform, expected in (("win32", [1]), ("linux", [])):
            seen.clear()
            env = self.env(system=lambda: system, resolve=big)
            env.suites = {"memory": lambda ctx: seen.append(1)}
            with mock.patch.object(runner.plan.sys, "platform", platform):
                runner.run(self.cfg(suites=["memory"], force=True), env)
            self.assertEqual(seen, expected, platform)

    def test_save_failures_surface_as_a_problem(self):
        def save_one(ctx):
            ctx.store.save("ollama/S/speed/warm", "ok", {})
        original = ResultStore._persist
        calls = {"n": 0}

        def flaky(store):
            calls["n"] += 1
            if store.data["tests"] and "ollama/S/speed/warm" in store.data["tests"]:
                raise OSError("disk full")
            return original(store)
        env = self.env()
        env.suites = {"speed": save_one}
        with mock.patch.object(ResultStore, "_persist", flaky):
            code = runner.run(self.cfg(suites=["speed"]), env)
        self.assertEqual(code, 1)
        self.assertIn("disk full", self.text)
        self.assertIn("speed suite failed", self.text)

    def test_failing_save_while_cleaning_up_running_entries_is_no_traceback(self):
        original = ResultStore._persist
        state = {"broken": False}

        def flaky(store):
            if state["broken"]:
                raise OSError("disk full")
            return original(store)

        def suite(ctx):
            ctx.store.mark_running("ollama/S/speed/warm", {})
            state["broken"] = True
            raise ValueError("suite broke")
        env = self.env()
        env.suites = {"speed": suite}
        with mock.patch.object(ResultStore, "_persist", flaky):
            code = runner.run(self.cfg(suites=["speed"]), env)
            state["broken"] = False
        self.assertEqual(code, 1)
        self.assertIn("disk full", self.text)
        self.assertIn("suite broke", self.text)
        self.assertTrue((self.run_dir() / "README.md").is_file())

    def test_run_summary_is_printed(self):
        runner.run(self.cfg(), self.env())
        self.assertIn("results.json", self.text)
        self.assertIn("README.md", self.text)


if __name__ == "__main__":
    unittest.main()
