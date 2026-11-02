import contextlib
import io
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import bench
from bench.cli import main
from bench.privacy import find_leaks, local_secrets

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

    def test_commands_not_implemented_yet(self):
        for command in [c for c in COMMANDS if c != "system"]:
            with self.subTest(command=command):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    code = main([command])
                self.assertEqual(code, 0)
                self.assertIn("not implemented yet", out.getvalue())

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

    def test_bench_py_entry_point(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "bench.py"), "--help"],
            capture_output=True, text=True, cwd=ROOT, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        for command in COMMANDS:
            self.assertIn(command, result.stdout)


if __name__ == "__main__":
    unittest.main()
