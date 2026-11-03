import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bench.config import Config, load_config


def write_toml(directory: str, text: str) -> str:
    path = Path(directory) / "bench.toml"
    path.write_text(text, encoding="utf-8")
    return str(path)


class RelativePathTest(unittest.TestCase):
    """Launchers cd into the repo; user given relative paths mean the user's directory."""

    def load(self, argv, base, toml=None):
        env = {"BENCH_CWD": str(base)}
        with mock.patch.dict(os.environ, env):
            return load_config(argv)[1]

    def test_cli_paths_resolve_against_the_original_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            cfg = self.load(["run", "--output", "out", "--ollama-models-dir", "models/x",
                             "--gguf", "S=ggufs/s.gguf", "--model-dir", "m",
                             "--llama-server", "bin/llama-server", "--mirror", "mir"], base)
        self.assertEqual(cfg.output, base / "out")
        self.assertEqual(cfg.ollama_models_dir, base / "models" / "x")
        self.assertEqual(cfg.gguf["S"], base / "ggufs" / "s.gguf")
        self.assertEqual(cfg.model_dir, base / "m")
        self.assertEqual(cfg.llama_server, base / "bin" / "llama-server")
        self.assertEqual(cfg.mirror, str(base / "mir"))

    def test_absolute_paths_urls_and_bare_commands_stay(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            absolute = str(base / "abs")
            cfg = self.load(["run", "--output", absolute, "--mirror", "https://example.org/m",
                             "--llama-server", "llama-server"], base / "elsewhere")
        self.assertEqual(cfg.output, Path(absolute))
        self.assertEqual(cfg.mirror, "https://example.org/m")
        self.assertEqual(cfg.llama_server, Path("llama-server"))

    def test_default_output_is_not_touched(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.load(["run"], Path(tmp))
        self.assertEqual(cfg.output, Path("runs"))

    def test_config_file_option_is_found_in_the_original_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            (base / "my.toml").write_text('reps = 7\noutput = "o"\n', encoding="utf-8")
            cfg = self.load(["run", "--config", "my.toml"], base)
        self.assertEqual(cfg.reps, 7)
        self.assertEqual(cfg.output, Path("o"))  # paths inside the file stay as written

    def test_without_launcher_variable_the_current_directory_is_used(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BENCH_CWD", None)
            _, cfg = load_config(["run", "--output", "o"])
        self.assertEqual(cfg.output, Path.cwd() / "o")

    def test_abort_if_busy_flag(self):
        self.assertFalse(Config().abort_if_busy)
        _, cfg = load_config(["run", "--abort-if-busy"])
        self.assertTrue(cfg.abort_if_busy)


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
            _, cfg = load_config(["run", "--config", cfg_path,
                                  "--gguf", "S=/m/s.gguf", "--gguf", "M=/m/m.gguf"])
        self.assertEqual(cfg.gguf, {"S": Path("/m/s.gguf"), "M": Path("/m/m.gguf")})

    def test_gguf_ids_are_validated(self):
        for bad in ("a/b=/m/x.gguf", "a b=/m/x.gguf", "a:b=/m/x.gguf", "..\\x=/m/x.gguf",
                    "x;y=/m/x.gguf"):
            with self.subTest(bad=bad), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    load_config(["run", "--gguf", bad])
        _, cfg = load_config(["run", "--gguf", "XL.v2_a-b=/m/x.gguf"])
        self.assertIn("XL.v2_a-b", cfg.gguf)

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
