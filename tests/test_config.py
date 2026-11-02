import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from bench.config import Config, load_config


def write_toml(directory: str, text: str) -> str:
    path = Path(directory) / "bench.toml"
    path.write_text(text, encoding="utf-8")
    return str(path)


class ConfigTest(unittest.TestCase):
    def test_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = write_toml(tmp, "")
            command, cfg = load_config(["run", "--config", cfg_path])
        self.assertEqual(command, "run")
        self.assertEqual(cfg, Config())
        self.assertEqual(cfg.engines, ["ollama", "llamacpp"])
        self.assertEqual(cfg.models, ["S", "M", "L"])
        self.assertEqual(cfg.suites, ["speed", "memory", "smoke"])
        self.assertEqual(cfg.scope, "full")
        self.assertEqual(cfg.reps, 3)
        self.assertEqual(cfg.ctx_steps, [8192, 16384, 32768, 65536, 131072, 262144])
        self.assertEqual(cfg.output, Path("runs"))
        self.assertEqual(cfg.force_tests, [])
        for flag in ("force", "yes", "dry_run", "no_install", "no_download",
                     "no_code_exec", "thinking"):
            self.assertIs(getattr(cfg, flag), False, flag)

    def test_toml_then_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = write_toml(tmp, 'reps = 5\nengines = ["llamacpp"]\n')
            _, from_file = load_config(["plan", "--config", cfg_path])
            _, from_cli = load_config(["plan", "--config", cfg_path, "--reps", "2"])
        self.assertEqual(from_file.reps, 5)
        self.assertEqual(from_file.engines, ["llamacpp"])
        self.assertEqual(from_cli.reps, 2)
        self.assertEqual(from_cli.engines, ["llamacpp"])

    def test_force_test_repeatable(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = write_toml(tmp, "")
            _, cfg = load_config(["run", "--config", cfg_path,
                                  "--force-test", "a", "--force-test", "b"])
        self.assertEqual(cfg.force_tests, ["a", "b"])

    def test_ctx_steps_parse(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = write_toml(tmp, "")
            _, cfg = load_config(["run", "--config", cfg_path,
                                  "--ctx-steps", "4k,8k,128k"])
        self.assertEqual(cfg.ctx_steps, [4096, 8192, 131072])

    def test_unknown_engine_rejected(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                load_config(["run", "--engines", "strata"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("strata", stderr.getvalue())

    def test_unknown_toml_key_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = write_toml(tmp, "repz = 5\n")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    load_config(["run", "--config", cfg_path])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("repz", stderr.getvalue())

    def test_missing_explicit_config_rejected(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                load_config(["run", "--config", "/nonexistent/bench.toml"])
        self.assertEqual(raised.exception.code, 2)

    def test_toml_ctx_steps_and_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = write_toml(
                tmp, 'ctx_steps = ["4k", 8192]\noutput = "out"\nmirror = "//host/share"\n')
            _, cfg = load_config(["run", "--config", cfg_path])
        self.assertEqual(cfg.ctx_steps, [4096, 8192])
        self.assertEqual(cfg.output, Path("out"))
        self.assertEqual(cfg.mirror, "//host/share")

    def test_gguf_repeatable(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = write_toml(tmp, "")
            _, cfg = load_config(["setup", "--config", cfg_path,
                                  "--gguf", "S=/m/s.gguf", "--gguf", "M=/m/m.gguf"])
        self.assertEqual(cfg.gguf, {"S": Path("/m/s.gguf"), "M": Path("/m/m.gguf")})

    def test_flags_and_comma_lists(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = write_toml(tmp, "")
            _, cfg = load_config(["run", "--config", cfg_path, "--yes", "--dry-run",
                                  "--no-code-exec", "--thinking",
                                  "--suites", "speed,concurrency", "--models", "S,XL"])
        self.assertTrue(cfg.yes and cfg.dry_run and cfg.no_code_exec and cfg.thinking)
        self.assertEqual(cfg.suites, ["speed", "concurrency"])
        self.assertEqual(cfg.models, ["S", "XL"])


if __name__ == "__main__":
    unittest.main()
