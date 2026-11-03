import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import bench
from bench.cli import main
from bench.privacy import find_leaks, local_secrets
from bench.store import ResultStore

ROOT = Path(__file__).resolve().parent.parent
COMMANDS = ["system", "doctor", "plan", "setup", "run", "report"]


class CliTest(unittest.TestCase):
    def test_version(self):
        self.assertEqual(bench.VERSION, "1.0.0")

    def test_help_lists_all_commands(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            with self.assertRaises(SystemExit) as raised:
                main(["--help"])
        self.assertEqual(raised.exception.code, 0)
        for command in COMMANDS:
            self.assertIn(command, out.getvalue())

    def test_setup_is_not_implemented_yet(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["setup"])
        self.assertEqual(code, 0)
        self.assertIn("not implemented yet", out.getvalue())

    def run_main(self, argv, **patches):
        out = io.StringIO()
        with contextlib.ExitStack() as stack:
            for target, value in patches.items():
                stack.enter_context(mock.patch(target, value))
            stack.enter_context(contextlib.redirect_stdout(out))
            code = main(argv)
        return code, out.getvalue()

    def test_run_passes_the_config_and_returns_the_exit_code(self):
        seen = []

        def fake_run(cfg):
            seen.append(cfg)
            return 3
        code, _ = self.run_main(["run", "--engines", "ollama", "--models", "S", "--yes"],
                                **{"bench.cli.runner.run": fake_run})
        self.assertEqual(code, 3)
        self.assertEqual((seen[0].engines, seen[0].models, seen[0].yes), (["ollama"], ["S"], True))

    def test_plan_prints_the_plan(self):
        system = {"gpus": [{"index": 0, "name": "Test", "vram_total_mb": 24576}], "cpu": "x"}
        entries = [{"engine": "ollama", "model": "S", "cls": "S", "quant": "Q4_K_M", "fits": True,
                    "reason": "fits: needs 3 GB", "need_gb": 3.0, "available_gb": 24.0,
                    "download_gb": 0.0, "needs_consent": False, "minutes_low": 3,
                    "minutes_high": 9, "new": True, "notes": []}]
        with tempfile.TemporaryDirectory() as tmp:
            code, text = self.run_main(
                ["plan", "--output", tmp],
                **{"bench.cli.collect_system": lambda: system,
                   "bench.cli.planning.make_plan": lambda cfg, sys_, store=None: entries})
        self.assertEqual(code, 0)
        self.assertIn("ollama", text)
        self.assertIn("fits", text)
        self.assertIn("3-9 min", text)

    def test_doctor_shows_rows_and_findings(self):
        rows = [{"check": "uptime", "status": "ok", "text": "up for 5 min"},
                {"check": "processes", "status": "ok", "text": "fine", "detail": "x 1 MB"}]
        finding = [{"kind": "gpu_busy", "text": "busy", "note": "n", "detail": "tool pid 9"}]
        code, text = self.run_main(["doctor"], **{"bench.cli.safety.doctor": lambda cfg: rows,
                                                  "bench.cli.preflight.check":
                                                      lambda cfg, gpu: finding})
        self.assertEqual(code, 0)
        for expected in ("up for 5 min", "x 1 MB", "gpu_busy", "busy", "tool pid 9"):
            self.assertIn(expected, text)

    def test_report_renders_the_run_folder_given_as_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ResultStore(Path(tmp) / "results.json")
            store.save("ollama/S/speed/warm", "ok", {"gen_tps": 42.5})
            code, text = self.run_main(["report", "--output", tmp])
            self.assertEqual(code, 0, text)
            self.assertIn("42.5", (Path(tmp) / "README.md").read_text(encoding="utf-8"))

    def test_report_finds_the_run_of_this_device(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "2026-11-03-pc-test"
            ResultStore(folder / "results.json").save("ollama/S/speed/warm", "ok", {})
            code, _ = self.run_main(["report", "--output", tmp, "--device", "PC Test"])
            self.assertEqual(code, 0)
            self.assertTrue((folder / "README.md").is_file())

    def test_report_without_results_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, text = self.run_main(["report", "--output", tmp, "--device", "PC Test"])
        self.assertEqual(code, 1)
        self.assertIn("results.json", text)

    def test_report_stops_on_personal_data_and_says_what(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ResultStore(Path(tmp) / "results.json")
            store.set_section("notes", ["server at 192.168.2.10"])
            code, text = self.run_main(["report", "--output", tmp])
            self.assertEqual(code, 1)
            self.assertIn("192.168.2.10", text)
            self.assertFalse((Path(tmp) / "README.md").exists())

    def test_system_prints_scrubbed_json_with_device_name(self):
        info = {"os": "Test OS", "cpu": "Some CPU on nero", "gpus": [],
                "path": "/home/michael/x", "ip": "192.168.2.10"}
        out = io.StringIO()
        with mock.patch("bench.cli.collect_system", return_value=info), \
                mock.patch("bench.cli.local_secrets", return_value=["nero", "michael"]), \
                contextlib.redirect_stdout(out):
            code = main(["system"])
        self.assertEqual(code, 0)
        text = out.getvalue()
        data = json.loads(text)
        self.assertEqual(data["suggested_device_name"], "PC Some CPU on <redacted>")
        self.assertEqual(data["system"]["path"], "<home>/x")
        self.assertEqual(data["system"]["ip"], "<ip>")
        self.assertEqual(find_leaks(text, ["nero", "michael"]), [])

    def test_system_command_runs_for_real(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["system"])
        self.assertEqual(code, 0)
        data = json.loads(out.getvalue())
        self.assertIn("system", data)
        self.assertIn("suggested_device_name", data)
        self.assertEqual(find_leaks(out.getvalue(), local_secrets()), [])

    def test_missing_command_is_usage_error(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                main([])
        self.assertEqual(raised.exception.code, 2)

    def test_launchers_hand_over_the_original_directory_before_the_cd(self):
        sh = (ROOT / "bench.sh").read_text(encoding="utf-8")
        self.assertLess(sh.index("BENCH_CWD="), sh.index('cd "$(dirname'))
        self.assertIn("export BENCH_CWD", sh)
        ps = (ROOT / "bench.ps1").read_text(encoding="utf-8")
        self.assertLess(ps.index("$env:BENCH_CWD"), ps.index("uv run"))

    def test_powershell_launcher_puts_back_what_it_changed(self):
        ps = (ROOT / "bench.ps1").read_text(encoding="utf-8")
        for name in ("BENCH_CWD", "UV_INSTALL_DIR", "UV_NO_MODIFY_PATH"):
            self.assertIn(name, ps.split("finally", 1)[0].split("foreach", 1)[1])
        finally_block = ps.split("} finally {", 1)[1]
        self.assertIn("Set-Item", finally_block)
        self.assertIn("Pop-Location", finally_block)
        self.assertIn("Push-Location", ps)

    def test_stdout_is_line_buffered_at_start(self):
        stream = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
        self.assertFalse(stream.line_buffering)
        with mock.patch.object(sys, "stdout", stream):
            main(["setup"])
            self.assertTrue(stream.line_buffering)

    def test_launchers_run_python_unbuffered(self):
        self.assertIn("export PYTHONUNBUFFERED=1", (ROOT / "bench.sh").read_text(encoding="utf-8"))
        ps = (ROOT / "bench.ps1").read_text(encoding="utf-8")
        self.assertIn("$env:PYTHONUNBUFFERED = '1'", ps)
        self.assertIn("'PYTHONUNBUFFERED'", ps.split("$saved = @{}")[0])  # restored afterwards

    def test_powershell_launcher_joins_array_arguments(self):
        ps = (ROOT / "bench.ps1").read_text(encoding="utf-8")
        self.assertIn("-is [array]", ps)
        self.assertIn("-join ','", ps)
        self.assertIn("@forward", ps)
        self.assertNotIn("bench.py @args", ps)

    def test_powershell_launcher_removes_variables_that_did_not_exist(self):
        finally_block = (ROOT / "bench.ps1").read_text(encoding="utf-8").split("} finally {")[1]
        self.assertIn("Remove-Item", finally_block)
        self.assertIn("UV_", (ROOT / "bench.ps1").read_text(encoding="utf-8"))

    @unittest.skipUnless(shutil.which("pwsh") and os.name != "nt", "needs pwsh on macOS/Linux")
    def test_powershell_launcher_for_real(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_uv = Path(tmp) / "uv"
            fake_uv.write_text('#!/bin/sh\nfor a in "$@"; do echo "ARG:$a"; done\n',
                               encoding="utf-8")
            fake_uv.chmod(0o755)
            script = (f"$env:PATH = '{tmp}' + [IO.Path]::PathSeparator + $env:PATH; "
                      f"& '{ROOT / 'bench.ps1'}' run --engines ollama,llamacpp --models S,M; "
                      "'UV=[' + $env:UV_INSTALL_DIR + '][' + $env:UV_NO_MODIFY_PATH + ']"
                      "[' + $env:BENCH_CWD + ']'; (Get-Location).Path")
            result = subprocess.run(["pwsh", "-NoProfile", "-Command", script], cwd=tmp,
                                    capture_output=True, text=True, check=False)
        out = result.stdout
        self.assertIn("ARG:ollama,llamacpp", out, out + result.stderr)
        self.assertIn("ARG:S,M", out)
        self.assertNotIn("ARG:llamacpp\n", out)
        self.assertIn("UV=[][][]", out)
        self.assertIn(os.path.realpath(tmp), os.path.realpath(out.strip().splitlines()[-1]))

    def test_bench_py_entry_point(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "bench.py"), "--help"],
            capture_output=True, text=True, cwd=ROOT, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        for command in COMMANDS:
            self.assertIn(command, result.stdout)


if __name__ == "__main__":
    unittest.main()
