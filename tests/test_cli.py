import contextlib
import io
import subprocess
import sys
import unittest
from pathlib import Path

import bench
from bench.cli import main

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
        for command in COMMANDS:
            with self.subTest(command=command):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    code = main([command])
                self.assertEqual(code, 0)
                self.assertIn("not implemented yet", out.getvalue())

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
