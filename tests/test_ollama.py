import copy
import http.client
import json
import os
import socket
import stat
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from bench import proc
from bench.config import Config
from bench.engines.base import (ChatResult, Completion, EngineError, ModelRef, PromptTruncated,
                                Sampling, Timing)
from bench.engines.ollama import OllamaEngine
from fake_http import FakeServer

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ollama"


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def stream_lines(name: str = "generate-stream.ndjson") -> list[str]:
    return (FIXTURES / name).read_text(encoding="utf-8").splitlines(keepends=True)


MODEL = ModelRef(id="S1", cls="S", quant="Q8_0", ollama_tag="qwen3.5:0.8b",
                 gguf_path=None, digest=None, size_bytes=None)

# A stand-in for `ollama serve`: serves /api/version on the port from OLLAMA_HOST and
# records the environment it was started with.
FAKE_OLLAMA = """#!{python}
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
host, port = os.environ["OLLAMA_HOST"].rsplit(":", 1)
GPU = ("GGML_VK_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
       "ROCR_VISIBLE_DEVICES", "HSA_OVERRIDE_GFX_VERSION", "GPU_DEVICE_ORDINAL")
env = {{k: v for k, v in os.environ.items() if k.startswith("OLLAMA_") or k in GPU}}
open(os.environ["FAKE_OUT"], "w").write(json.dumps({{"argv": sys.argv[1:], "env": env}}))
if os.environ.get("FAKE_MODE") == "hang":  # accepts connections, never answers
    import socket, time
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, int(port))); srv.listen(); held = []
    while True:
        held.append(srv.accept()[0])
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        self.send_response(200); self.end_headers()
        self.wfile.write(b'{{"version":"9.9.9"}}')
HTTPServer((host, int(port)), H).serve_forever()
"""


def config(**kw) -> Config:
    return Config(**kw)


class EngineTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def engine_for(self, server: FakeServer, **kw) -> OllamaEngine:
        engine = OllamaEngine(config(ollama_url=server.url, **kw), log_dir=self.dir)
        self.addCleanup(engine.stop)
        engine.start()
        return engine


class ProcessOwnershipTestBase(EngineTestCase):
    def fake_binary(self) -> Path:
        path = self.dir / "ollama"
        path.write_text(FAKE_OLLAMA.format(python=sys.executable), encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return path


class ProcessOwnershipTest(ProcessOwnershipTestBase):

    @unittest.skipIf(os.name == "nt", "the fake binary is a POSIX script")
    def test_own_server_on_free_port(self):
        # Something else already listens on the default port (best effort: if the port
        # is taken by a real server, that is the same situation).
        occupied = socket.socket()
        try:
            occupied.bind(("127.0.0.1", 11434))
            occupied.listen()
        except OSError:
            pass
        self.addCleanup(occupied.close)
        out = self.dir / "env.json"
        before = dict(os.environ)
        os.environ["FAKE_OUT"] = str(out)
        self.addCleanup(os.environ.pop, "FAKE_OUT", None)
        before["FAKE_OUT"] = str(out)

        models_dir = self.dir / "models"
        engine = OllamaEngine(config(ollama_bin=self.fake_binary(), ollama_models_dir=models_dir),
                              log_dir=self.dir / "logs")
        engine.start()
        try:
            self.assertEqual(engine.version(), "9.9.9")
            port = int(engine.base_url.rsplit(":", 1)[1])
            self.assertNotEqual(port, 11434)
            self.assertTrue(engine.base_url.startswith("http://127.0.0.1:"))
            seen = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(seen["argv"], ["serve"])
            self.assertEqual(seen["env"]["OLLAMA_HOST"], f"127.0.0.1:{port}")
            self.assertEqual(seen["env"]["OLLAMA_NUM_PARALLEL"], "1")
            self.assertEqual(seen["env"]["OLLAMA_MODELS"], str(models_dir))
            self.assertIn("OLLAMA_KEEP_ALIVE", seen["env"])
            owned = engine._owned
            pid = owned.pid
        finally:
            engine.stop()
        # The process is gone, the engine can be stopped again, our environment is intact.
        self.assertFalse(owned.running)
        engine.stop()
        with self.assertRaises(OSError):
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
        self.assertEqual(dict(os.environ), before)
        self.assertIsInstance(pid, int)
        # The listener on the default port was never touched.
        with socket.create_connection(("127.0.0.1", 11434), timeout=1):
            pass

    POLLUTION = {
        "OLLAMA_KV_CACHE_TYPE": "q8_0", "OLLAMA_FLASH_ATTENTION": "true",
        "OLLAMA_KEEP_ALIVE": "-1", "OLLAMA_MAX_LOADED_MODELS": "3",
        "OLLAMA_NUM_PARALLEL": "8", "OLLAMA_CONTEXT_LENGTH": "4096",
        "OLLAMA_GPU_OVERHEAD": "2147483648", "OLLAMA_LLM_LIBRARY": "cpu",
        "OLLAMA_SCHED_SPREAD": "1", "OLLAMA_NUM_GPU": "1",
        "OLLAMA_MODELS": "/user/store", "OLLAMA_HOST": "0.0.0.0:11434"}

    def start_with_pollution(self, **cfg):
        out = self.dir / "env.json"
        patcher = mock.patch.dict(os.environ, {**self.POLLUTION, "FAKE_OUT": str(out)})
        patcher.start()
        self.addCleanup(patcher.stop)
        engine = OllamaEngine(config(ollama_bin=self.fake_binary(), **cfg),
                              log_dir=self.dir / "logs")
        engine.start()
        self.addCleanup(engine.stop)
        return engine, json.loads(out.read_text(encoding="utf-8"))["env"]

    @unittest.skipIf(os.name == "nt", "the fake binary is a POSIX script")
    def test_polluted_parent_environment_does_not_reach_the_server(self):
        engine, env = self.start_with_pollution()
        port = engine.base_url.rsplit(":", 1)[1]
        self.assertEqual(env["OLLAMA_KV_CACHE_TYPE"], "f16")
        self.assertEqual(env["OLLAMA_FLASH_ATTENTION"], "1")
        self.assertEqual(env["OLLAMA_MAX_LOADED_MODELS"], "1")
        self.assertEqual(env["OLLAMA_NUM_PARALLEL"], "1")
        self.assertEqual(env["OLLAMA_KEEP_ALIVE"], "30m")
        self.assertEqual(env["OLLAMA_HOST"], f"127.0.0.1:{port}")
        for name in ("OLLAMA_CONTEXT_LENGTH", "OLLAMA_GPU_OVERHEAD", "OLLAMA_LLM_LIBRARY",
                     "OLLAMA_SCHED_SPREAD", "OLLAMA_NUM_GPU"):
            self.assertNotIn(name, env)
        self.assertEqual(os.environ["OLLAMA_KV_CACHE_TYPE"], "q8_0")  # parent untouched

    GPU_ENV = {"OLLAMA_VULKAN": "1", "GGML_VK_VISIBLE_DEVICES": "0", "CUDA_VISIBLE_DEVICES": "1",
               "HIP_VISIBLE_DEVICES": "2", "ROCR_VISIBLE_DEVICES": "3",
               "HSA_OVERRIDE_GFX_VERSION": "8.0.3", "GPU_DEVICE_ORDINAL": "0"}

    def start_gpu(self, parent: dict, backend: str | None, probe_fails=False):
        out = self.dir / "env.json"
        patcher = mock.patch.dict(os.environ, {**parent, "FAKE_OUT": str(out)})
        patcher.start()
        self.addCleanup(patcher.stop)

        def probe(cfg):
            if probe_fails:
                raise RuntimeError("no gpu tools")
            return backend
        with mock.patch("bench.engines.ollama.gpu_backend", probe):
            engine = OllamaEngine(config(ollama_bin=self.fake_binary()),
                                  log_dir=self.dir / "logs")
            engine.start()
        self.addCleanup(engine.stop)
        return engine, json.loads(out.read_text(encoding="utf-8"))["env"]

    @unittest.skipIf(os.name == "nt", "the fake binary is a POSIX script")
    def test_gpu_selection_variables_are_kept(self):
        engine, env = self.start_gpu(self.GPU_ENV, "cuda")
        for name, value in self.GPU_ENV.items():
            self.assertEqual(env[name], value)
        self.assertEqual(engine.meta["ollama_vulkan"], True)
        self.assertEqual(engine.meta["gpu_env"]["GGML_VK_VISIBLE_DEVICES"], "0")

    @unittest.skipIf(os.name == "nt", "the fake binary is a POSIX script")
    def test_vulkan_backend_sets_ollama_vulkan_even_if_the_parent_lacks_it(self):
        engine, env = self.start_gpu({}, "vulkan")
        self.assertEqual(env["OLLAMA_VULKAN"], "1")
        self.assertIs(engine.meta["ollama_vulkan"], True)

    @unittest.skipIf(os.name == "nt", "the fake binary is a POSIX script")
    def test_other_backends_do_not_set_ollama_vulkan(self):
        for backend in ("cuda", "rocm", None):
            engine, env = self.start_gpu({}, backend)
            self.assertNotIn("OLLAMA_VULKAN", env, backend)
            self.assertIs(engine.meta["ollama_vulkan"], False)

    @unittest.skipIf(os.name == "nt", "the fake binary is a POSIX script")
    def test_failing_gpu_probe_keeps_the_parent_setting(self):
        engine, env = self.start_gpu({"OLLAMA_VULKAN": "1"}, None, probe_fails=True)
        self.assertEqual(env["OLLAMA_VULKAN"], "1")
        self.assertIs(engine.meta["ollama_vulkan"], True)

    def test_foreign_server_vulkan_is_unknown(self):
        with FakeServer({"/api/version": fixture("version.json")}) as server:
            self.assertEqual(self.engine_for(server).meta["ollama_vulkan"], "unknown")

    @unittest.skipIf(os.name == "nt", "the fake binary is a POSIX script")
    def test_meta_reports_the_backend_from_our_log(self):
        engine, _ = self.start_gpu({}, "vulkan")
        engine._log_path.write_text('level=INFO msg="inference compute" library=Vulkan\n',
                                    encoding="utf-8")
        self.assertEqual(engine.meta["backend_used"], "vulkan")

    @unittest.skipIf(os.name == "nt", "the fake binary is a POSIX script")
    def test_models_dir_from_the_user_environment_is_kept_unless_configured(self):
        _, env = self.start_with_pollution()
        self.assertEqual(env["OLLAMA_MODELS"], "/user/store")
        _, env = self.start_with_pollution(ollama_models_dir=self.dir / "mine")
        self.assertEqual(env["OLLAMA_MODELS"], str(self.dir / "mine"))

    @unittest.skipIf(os.name == "nt", "the fake binary is a POSIX script")
    def test_proxies_are_passed_through(self):
        with mock.patch.dict(os.environ, {"HTTPS_PROXY": "http://proxy:3128",
                                          "NO_PROXY": "localhost"}):
            _, env = self.start_with_pollution()
        # the fake only reports OLLAMA_*; the proxy variables must simply not be dropped
        self.assertEqual(env["OLLAMA_KV_CACHE_TYPE"], "f16")
        self.assertFalse(any(n.startswith("OLLAMA_") and "PROXY" in n for n in env))

    @unittest.skipIf(os.name == "nt", "the fake binary is a POSIX script")
    def test_meta_shows_the_pinned_settings(self):
        engine, _ = self.start_with_pollution()
        meta = engine.meta
        self.assertEqual(meta["kv_cache_type"], "f16")
        self.assertEqual(meta["flash_attn"], "on")
        self.assertEqual(meta["max_loaded_models"], 1)
        self.assertEqual(meta["num_parallel"], 1)
        self.assertEqual(meta["settings_source"], "pinned")

    @unittest.skipIf(os.name == "nt", "the fake binary is a POSIX script")
    def test_meta_follows_the_server_config_line_of_our_own_log(self):
        engine, _ = self.start_with_pollution()
        log = engine._log_path
        log.write_text('time=x level=INFO msg="server config" env="map[CUDA_VISIBLE_DEVICES: '
                       'OLLAMA_FLASH_ATTENTION:false OLLAMA_KV_CACHE_TYPE:q4_0 '
                       'OLLAMA_NUM_PARALLEL:1]"\n', encoding="utf-8")
        meta = engine.meta
        self.assertEqual(meta["kv_cache_type"], "q4_0")
        self.assertEqual(meta["flash_attn"], "off")
        self.assertEqual(meta["settings_source"], "server log")

    @unittest.skipIf(os.name == "nt", "the fake binary is a POSIX script")
    def test_parallel_is_pinned_to_the_requested_value(self):
        engine, _ = self.start_with_pollution()
        real_post = engine._post

        def post(path, payload, timeout=None):
            if path == "/api/generate":
                return {"load_duration": 1_000_000_000}
            return real_post(path, payload, timeout)

        with mock.patch.object(engine, "_post", post):
            engine.load(MODEL, 4096, parallel=4)
        self.assertEqual(engine.meta["num_parallel"], 4)

    RUNNER_LINE = ('time=2026-11-03T10:00:00.000+01:00 level=INFO source=llama_server.go:467 '
                   'msg="starting llama-server" cmd="/opt/ollama/llama-server --model /m/blob '
                   '-c {c} -np {np} --n-gpu-layers 99"\n')

    def test_parse_runner_start_takes_the_newest_line(self):
        from bench.engines import ollama as ol
        text = self.RUNNER_LINE.format(c=4096, np=1) + "x\n" + self.RUNNER_LINE.format(c=16384, np=4)
        self.assertEqual(ol.parse_runner_start(text), (4, 16384))
        self.assertEqual(ol.parse_runner_start(self.RUNNER_LINE.format(c=10240, np=1)),
                         (1, 10240))
        self.assertIsNone(ol.parse_runner_start('msg="starting llama-server" cmd="x --model y"'))
        self.assertIsNone(ol.parse_runner_start("nothing"))

    def test_meta_reports_the_effective_parallel_of_our_own_log(self):
        engine, _ = self.start_with_pollution()
        engine._log_path.write_text(self.RUNNER_LINE.format(c=10240, np=1), encoding="utf-8")
        engine._log_mark = 0
        meta = engine.meta
        self.assertEqual(meta["effective_parallel"], 1)
        self.assertEqual(meta["effective_ctx"], 10240)

    def test_effective_parallel_only_counts_the_latest_load(self):
        engine, _ = self.start_with_pollution()
        log = engine._log_path
        log.write_text(self.RUNNER_LINE.format(c=16384, np=4), encoding="utf-8")
        engine._log_mark = log.stat().st_size
        self.assertIsNone(engine.meta["effective_parallel"])
        with log.open("a", encoding="utf-8") as handle:
            handle.write(self.RUNNER_LINE.format(c=10240, np=1))
        self.assertEqual(engine.meta["effective_parallel"], 1)

    def test_effective_parallel_unknown_for_foreign_server(self):
        with FakeServer({"/api/version": fixture("version.json")}) as server:
            engine = self.engine_for(server)
            engine._log_path = self.dir / "other.log"
            (self.dir / "other.log").write_text(self.RUNNER_LINE.format(c=1, np=1),
                                                encoding="utf-8")
            meta = engine.meta
        self.assertIsNone(meta["effective_parallel"])
        self.assertIsNone(meta["effective_ctx"])

    def test_foreign_server_settings_are_unknown(self):
        with FakeServer({"/api/version": fixture("version.json")}) as server:
            engine = self.engine_for(server)
            meta = engine.meta
            # even when a log of ours exists, a foreign server did not write it
            engine._log_path = self.dir / "other.log"
            (self.dir / "other.log").write_text(
                'msg="server config" env="map[OLLAMA_KV_CACHE_TYPE:q8_0]"', encoding="utf-8")
            self.assertEqual(engine.meta, meta)
        self.assertEqual(meta["kv_cache_type"], "unknown")
        self.assertEqual(meta["flash_attn"], "unknown")
        self.assertEqual(meta["settings_source"], "foreign server")

    @unittest.skipIf(os.name == "nt", "the fake binary is a POSIX script")
    def test_parallel_change_restarts_own_server_with_new_env(self):
        out = self.dir / "env.json"
        os.environ["FAKE_OUT"] = str(out)
        self.addCleanup(os.environ.pop, "FAKE_OUT", None)
        engine = OllamaEngine(config(ollama_bin=self.fake_binary()), log_dir=self.dir / "logs")
        engine.start()
        self.addCleanup(engine.stop)
        first = engine._owned
        real_post = engine._post

        def post(path, payload, timeout=None):
            if path == "/api/generate":
                return {"load_duration": 1_000_000_000}
            return real_post(path, payload, timeout)

        with mock.patch.object(engine, "_post", post):
            engine.load(MODEL, 4096, parallel=4)
        self.assertIsNot(engine._owned, first)
        self.assertFalse(first.running)
        self.assertEqual(json.loads(out.read_text())["env"]["OLLAMA_NUM_PARALLEL"], "4")

    def test_foreign_server_untouched(self):
        with FakeServer({"/api/version": fixture("version.json")}) as server:
            with mock.patch("bench.proc.subprocess.Popen") as popen, \
                    mock.patch("bench.proc.os.killpg", create=True) as killpg, \
                    mock.patch("bench.proc.subprocess.run") as run:
                engine = OllamaEngine(config(ollama_url=server.url), log_dir=self.dir)
                engine.start()
                self.assertEqual(engine.version(), "0.40.13")
                engine.stop()
                engine.stop()
            popen.assert_not_called()
            killpg.assert_not_called()
            run.assert_not_called()
            self.assertIsNone(engine._owned)
            # Still serving after stop().
            again = OllamaEngine(config(ollama_url=server.url), log_dir=self.dir)
            again.start()
            self.assertEqual(again.version(), "0.40.13")

    def test_trailing_slash_in_url(self):
        with FakeServer({"/api/version": fixture("version.json")}) as server:
            engine = OllamaEngine(config(ollama_url=server.url + "/"), log_dir=self.dir)
            engine.start()
            self.assertEqual(engine.version(), "0.40.13")

    def test_unreachable_foreign_server_is_an_error_not_a_start(self):
        port = socket.socket()
        port.bind(("127.0.0.1", 0))
        free = port.getsockname()[1]
        port.close()
        with mock.patch("bench.proc.subprocess.Popen") as popen:
            engine = OllamaEngine(config(ollama_url=f"http://127.0.0.1:{free}"),
                                  log_dir=self.dir)
            with self.assertRaises(EngineError):
                engine.start()
        popen.assert_not_called()

    def test_no_start_without_url_is_refused(self):
        with mock.patch("bench.proc.subprocess.Popen") as popen:
            engine = OllamaEngine(config(no_ollama_start=True), log_dir=self.dir)
            with self.assertRaises(EngineError) as ctx:
                engine.start()
        popen.assert_not_called()
        self.assertIn("--ollama-url", str(ctx.exception))

    def test_missing_binary(self):
        engine = OllamaEngine(config(ollama_bin=self.dir / "nope"), log_dir=self.dir)
        with self.assertRaises(EngineError):
            engine.start()

    def test_never_picks_default_port(self):
        with mock.patch("bench.engines.ollama.proc.free_port", side_effect=[11434, 11434, 5555]):
            from bench.engines import ollama
            self.assertEqual(ollama.pick_port(), 5555)


class CompleteTest(EngineTestCase):
    def test_complete_timing(self):
        final = {"model": "qwen3.5:0.8b", "response": "", "done": True, "done_reason": "length",
                 "load_duration": 2_000_000_000, "prompt_eval_count": 8192,
                 "prompt_eval_cached_count": 0, "prompt_eval_duration": 1_000_000_000,
                 "eval_count": 256, "eval_duration": 2_000_000_000}
        lines = [json.dumps({"response": "Hello", "done": False}) + "\n",
                 json.dumps({"response": " world", "done": False}) + "\n",
                 json.dumps(final) + "\n"]
        with FakeServer({"/api/version": fixture("version.json"),
                         "/api/generate": lines}) as server:
            engine = self.engine_for(server)
            result = engine.complete(MODEL, "Hi", Sampling())
        self.assertIsInstance(result, Completion)
        self.assertEqual(result.text, "Hello world")
        t = result.timing
        self.assertIsInstance(t, Timing)
        self.assertEqual(t.prompt_tokens, 8192)
        self.assertEqual(t.prompt_tps, 8192.0)
        self.assertEqual(t.gen_tokens, 256)
        self.assertEqual(t.gen_tps, 128.0)
        self.assertEqual(t.load_s, 2.0)
        self.assertEqual(t.cached_tokens, 0)
        self.assertGreater(t.ttft_s, 0)
        self.assertEqual(result.raw, final)

    def test_request_body(self):
        with FakeServer({"/api/version": fixture("version.json"),
                         "/api/generate": stream_lines()}) as server:
            engine = self.engine_for(server)
            engine.complete(MODEL, "Hi", Sampling(temperature=0.2, top_p=0.9, top_k=40,
                                                  seed=7, max_tokens=64))
            body = [b for _, p, b in server.requests if p == "/api/generate"][0]
        self.assertEqual(body["model"], "qwen3.5:0.8b")
        self.assertEqual(body["prompt"], "Hi")
        self.assertIs(body["raw"], True)
        self.assertIs(body["stream"], True)
        self.assertEqual(body["options"], {"temperature": 0.2, "top_p": 0.9, "top_k": 40,
                                           "seed": 7, "num_predict": 64})

    def test_top_k_omitted_when_none_and_num_ctx_follows_load(self):
        with FakeServer({"/api/version": fixture("version.json"),
                         "/api/generate": lambda body: stream_lines()
                         if body["prompt"] else fixture("generate-raw.json")}) as server:
            engine = self.engine_for(server)
            engine.load(MODEL, 16384)
            engine.complete(MODEL, "Hi", Sampling())
            load_body, body = [b for _, p, b in server.requests if p == "/api/generate"]
        self.assertNotIn("top_k", body["options"])
        self.assertEqual(body["options"]["num_ctx"], 16384)
        self.assertEqual(load_body["options"]["num_ctx"], 16384)

    def test_real_stream_fixture(self):
        with FakeServer({"/api/version": fixture("version.json"),
                         "/api/generate": stream_lines()}) as server:
            engine = self.engine_for(server)
            result = engine.complete(MODEL, "Hi", Sampling())
        self.assertEqual(result.text, "\n1. 2019")
        self.assertEqual(result.timing.prompt_tokens, 4)
        self.assertEqual(result.timing.gen_tokens, 8)
        self.assertEqual(result.timing.cached_tokens, 0)
        self.assertAlmostEqual(result.timing.load_s, 3.1065473)
        self.assertAlmostEqual(result.timing.gen_tps, 8 / 0.039167)

    def test_ttft_is_time_to_first_non_empty_chunk(self):
        final = json.dumps(fixture("generate-raw.json") | {"response": "", "done": True}) + "\n"
        lines = [json.dumps({"response": "", "done": False}) + "\n",
                 (0.4, json.dumps({"response": "A", "done": False}) + "\n"),
                 (0.3, json.dumps({"response": "B", "done": False}) + "\n"),
                 final]
        with FakeServer({"/api/version": fixture("version.json"),
                         "/api/generate": lines}) as server:
            engine = self.engine_for(server)
            started = time.perf_counter()
            result = engine.complete(MODEL, "Hi", Sampling())
            total = time.perf_counter() - started
        self.assertGreaterEqual(result.timing.ttft_s, 0.4)
        self.assertLess(result.timing.ttft_s, total - 0.25)

    def test_cached_tokens_and_zero_duration(self):
        final = {"response": "", "done": True, "prompt_eval_count": 512,
                 "prompt_eval_cached_count": 512, "prompt_eval_duration": 0,
                 "eval_count": 0, "eval_duration": 0}
        with FakeServer({"/api/version": fixture("version.json"),
                         "/api/generate": [json.dumps(final) + "\n"]}) as server:
            engine = self.engine_for(server)
            t = engine.complete(MODEL, "Hi", Sampling()).timing
        self.assertEqual(t.prompt_tokens, 512)
        self.assertEqual(t.cached_tokens, 512)
        self.assertIsNone(t.prompt_tps)
        self.assertIsNone(t.gen_tps)
        self.assertIsNone(t.load_s)

    def test_error_chunk_raises(self):
        with FakeServer({"/api/version": fixture("version.json"),
                         "/api/generate": ['{"error":"model requires more memory"}\n']}) as server:
            engine = self.engine_for(server)
            with self.assertRaises(EngineError) as ctx:
                engine.complete(MODEL, "Hi", Sampling())
        self.assertIn("more memory", str(ctx.exception))

    def test_truncated_stream_raises(self):
        with FakeServer({"/api/version": fixture("version.json"),
                         "/api/generate": ['{"response":"A","done":false}\n']}) as server:
            engine = self.engine_for(server)
            with self.assertRaises(EngineError):
                engine.complete(MODEL, "Hi", Sampling())

    def test_http_error_raises(self):
        with FakeServer({"/api/version": fixture("version.json")}) as server:
            engine = self.engine_for(server)
            with self.assertRaises(EngineError) as ctx:
                engine.complete(MODEL, "Hi", Sampling())
        self.assertIn("404", str(ctx.exception))

    def test_model_without_ollama_tag(self):
        bare = ModelRef("X", "S", "Q8_0", None, None, None, None)
        with FakeServer({"/api/version": fixture("version.json")}) as server:
            engine = self.engine_for(server)
            with self.assertRaises(EngineError):
                engine.complete(bare, "Hi", Sampling())


class ChatTest(EngineTestCase):
    def test_chat_tool_calls(self):
        with FakeServer({"/api/version": fixture("version.json"),
                         "/api/chat": fixture("chat-tools.json")}) as server:
            engine = self.engine_for(server)
            tools = [{"type": "function", "function": {"name": "get_weather"}}]
            result = engine.chat(MODEL, [{"role": "user", "content": "Weather in Berlin?"}],
                                 Sampling(), tools=tools)
            body = [b for _, p, b in server.requests if p == "/api/chat"][0]
        self.assertIsInstance(result, ChatResult)
        self.assertEqual(result.tool_calls, [{"name": "get_weather",
                                              "arguments": {"city": "Berlin"}}])
        self.assertEqual(result.text, "")
        self.assertEqual(body["tools"], tools)
        self.assertIs(body["stream"], False)
        self.assertEqual(result.timing.prompt_tokens, 280)
        self.assertEqual(result.timing.gen_tokens, 26)
        self.assertEqual(result.raw["done_reason"], "stop")

    def test_chat_arguments_as_json_string_are_decoded(self):
        reply = copy.deepcopy(fixture("chat-tools.json"))
        reply["message"]["tool_calls"][0]["function"]["arguments"] = '{"city": "Berlin"}'
        with FakeServer({"/api/version": fixture("version.json"), "/api/chat": reply}) as server:
            engine = self.engine_for(server)
            result = engine.chat(MODEL, [], Sampling())
        self.assertEqual(result.tool_calls[0]["arguments"], {"city": "Berlin"})

    def test_chat_json_schema_is_sent_as_format(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        with FakeServer({"/api/version": fixture("version.json"),
                         "/api/chat": fixture("chat-json.json")}) as server:
            engine = self.engine_for(server)
            result = engine.chat(MODEL, [{"role": "user", "content": "x"}], Sampling(),
                                 json_schema=schema)
            body = [b for _, p, b in server.requests if p == "/api/chat"][0]
        self.assertEqual(body["format"], schema)
        self.assertNotIn("tools", body)
        self.assertEqual(json.loads(result.text), {"name": "Alice", "age": 30})
        self.assertEqual(result.tool_calls, [])

    def test_thinking_flag(self):
        with FakeServer({"/api/version": fixture("version.json"),
                         "/api/chat": fixture("chat-json.json")}) as server:
            engine = self.engine_for(server)
            engine.chat(MODEL, [], Sampling())
            engine.chat(MODEL, [], Sampling(), thinking=True)
            bodies = [b for _, p, b in server.requests if p == "/api/chat"]
        self.assertIs(bodies[0]["think"], False)
        self.assertIs(bodies[1]["think"], True)

    def test_chat_ttft_is_server_side_first_token_latency(self):
        with FakeServer({"/api/version": fixture("version.json"),
                         "/api/chat": fixture("chat-json.json")}) as server:
            engine = self.engine_for(server)
            t = engine.chat(MODEL, [], Sampling()).timing
        self.assertAlmostEqual(t.ttft_s, (4_129_800 + 53_848_000) / 1e9)


class MetricsTest(EngineTestCase):
    def test_gpu_fraction(self):
        ps = {"models": [{"name": "qwen3.5:0.8b", "model": "qwen3.5:0.8b",
                          "size": 10, "size_vram": 10}]}
        with FakeServer({"/api/version": fixture("version.json"), "/api/ps": ps}) as server:
            engine = self.engine_for(server)
            self.assertEqual(engine.gpu_fraction(MODEL), 1.0)

    def test_gpu_fraction_partial_missing_and_real_fixture(self):
        ps = {"models": [{"name": "other:1b", "size": 10, "size_vram": 1},
                         {"name": "qwen3.5:0.8b", "size": 10, "size_vram": 4}]}
        with FakeServer({"/api/version": fixture("version.json"), "/api/ps": ps}) as server:
            engine = self.engine_for(server)
            self.assertEqual(engine.gpu_fraction(MODEL), 0.4)
            server.routes["/api/ps"] = {"models": []}
            self.assertIsNone(engine.gpu_fraction(MODEL))
            server.routes["/api/ps"] = fixture("ps.json")
            self.assertEqual(engine.gpu_fraction(MODEL), 1.0)

    def test_load_returns_load_duration_and_unload(self):
        replies = {"/api/version": fixture("version.json"),
                   "/api/generate": lambda body: fixture("unload.json")
                   if body.get("keep_alive") == 0 else {"done": True, "load_duration": 2.5e9}}
        with FakeServer(replies) as server:
            engine = self.engine_for(server)
            self.assertEqual(engine.load(MODEL, 8192), 2.5)
            engine.unload(MODEL)
            load_body, unload_body = [b for _, p, b in server.requests if p == "/api/generate"]
        self.assertEqual(load_body["prompt"], "")
        self.assertEqual(load_body["options"], {"num_ctx": 8192})
        self.assertEqual(unload_body, {"model": "qwen3.5:0.8b", "keep_alive": 0,
                                       "stream": False})

    def test_show(self):
        with FakeServer({"/api/version": fixture("version.json"),
                         "/api/show": fixture("show.json")}) as server:
            engine = self.engine_for(server)
            info = engine.show(MODEL)
            body = [b for _, p, b in server.requests if p == "/api/show"][0]
        self.assertEqual(body, {"model": "qwen3.5:0.8b"})
        self.assertEqual(info["details"]["quantization_level"], "Q8_0")

    def test_backend_and_kv_buffer_unknown_for_foreign_server(self):
        with FakeServer({"/api/version": fixture("version.json")}) as server:
            engine = self.engine_for(server)
            self.assertIsNone(engine.backend())
            self.assertIsNone(engine.kv_buffer_mb())

    def test_backend_and_kv_buffer_from_own_log(self):
        log = (self.dir / "server.log")
        log.write_text(
            'time=x level=INFO source=types.go msg="inference compute" id=GPU-1 library=CUDA '
            'compute=8.9\n'
            'llama_kv_cache:      CUDA0 KV buffer size =   384.00 MiB\n'
            'llama_kv_cache:      CUDA1 KV buffer size =   128.50 MiB\n', encoding="utf-8")
        engine = OllamaEngine(config(), log_dir=self.dir)
        engine._log_path = log
        self.assertEqual(engine.backend(), "cuda")
        self.assertEqual(engine.kv_buffer_mb(), 512.5)

    def test_kv_buffer_only_counts_the_latest_load(self):
        log = self.dir / "server.log"
        log.write_text("llama_kv_cache: CUDA0 KV buffer size = 100.00 MiB\n", encoding="utf-8")
        engine = OllamaEngine(config(), log_dir=self.dir)
        engine._log_path = log
        engine._log_mark = log.stat().st_size
        with log.open("a", encoding="utf-8") as handle:
            handle.write("llama_kv_cache: CUDA0 KV buffer size = 200.00 MiB\n")
        self.assertEqual(engine.kv_buffer_mb(), 200.0)


class CountTokensTest(EngineTestCase):
    """Ollama has no tokenizer endpoint. Counting a text on the server would put exactly
    that text into the prompt cache, so counts come from a chars-per-token ratio that is
    calibrated once with a text that starts with a unique nonce line."""

    TEXT = "The Rhine flows north through many cities. " * 100  # 4300 chars

    def reply(self, count, cached=0):
        return {"done": True, "response": "x", "prompt_eval_count": count,
                "prompt_eval_cached_count": cached}

    def generates(self, server):
        return [b for _, p, b in server.requests if p == "/api/generate"]

    def test_calibration_request_starts_with_a_nonce_line(self):
        with FakeServer({"/api/version": fixture("version.json"),
                         "/api/generate": self.reply(1000)}) as server:
            engine = self.engine_for(server)
            engine.load(MODEL, 9000)
            engine.count_tokens(MODEL, self.TEXT)
            engine.count_tokens(MODEL, self.TEXT)
            bodies = self.generates(server)[1:]  # the first request is the load
        self.assertEqual(len(bodies), 1)  # calibrated once
        body = bodies[0]
        first_line = body["prompt"].split("\n", 1)[0]
        self.assertRegex(first_line, r"^Calibration [0-9a-f]{8,}$")
        self.assertNotIn(self.TEXT, [b["prompt"] for b in bodies])
        self.assertIs(body["raw"], True)
        self.assertIs(body["stream"], False)
        self.assertEqual(body["options"]["num_predict"], 1)
        self.assertEqual(body["options"]["num_ctx"], 9000)  # the loaded context

    def test_nonce_differs_between_calibrations(self):
        with FakeServer({"/api/version": fixture("version.json"),
                         "/api/generate": self.reply(1000)}) as server:
            engine = self.engine_for(server)
            engine.load(MODEL, 9000)
            engine.count_tokens(MODEL, self.TEXT)
            engine.unload(MODEL)
            engine.load(MODEL, 10000)
            engine.count_tokens(MODEL, self.TEXT)
            lines = [b["prompt"].split("\n", 1)[0] for b in self.generates(server)
                     if b.get("prompt", "").startswith("Calibration")]
        self.assertEqual(len(lines), 2)
        self.assertNotEqual(lines[0], lines[1])

    def test_count_scales_with_the_calibrated_ratio(self):
        # 1000 tokens for the calibration text: the ratio is len(text) / 1000.
        with FakeServer({"/api/version": fixture("version.json"),
                         "/api/generate": self.reply(1000)}) as server:
            engine = self.engine_for(server)
            engine.load(MODEL, 9000)
            calibration_chars = None

            def spy(body):
                nonlocal calibration_chars
                calibration_chars = len(body["prompt"])
                return self.reply(1000)

            server.routes["/api/generate"] = spy
            small = engine.count_tokens(MODEL, self.TEXT)
            double = engine.count_tokens(MODEL, self.TEXT + self.TEXT[:len(self.TEXT) // 2])
        ratio = calibration_chars / 1000
        self.assertAlmostEqual(small, len(self.TEXT) / ratio, delta=2)
        self.assertAlmostEqual(double, 1.5 * len(self.TEXT) / ratio, delta=2)

    def test_ratio_is_kept_per_model_and_context(self):
        other = ModelRef(id="S2", cls="S", quant="Q8_0", ollama_tag="other:1b",
                         gguf_path=None, digest=None, size_bytes=None)
        with FakeServer({"/api/version": fixture("version.json"),
                         "/api/generate": self.reply(1000)}) as server:
            engine = self.engine_for(server)
            engine.load(MODEL, 9000)
            engine.count_tokens(MODEL, self.TEXT)
            engine.load(other, 9000)
            engine.count_tokens(other, self.TEXT)
            engine.unload(MODEL)
            engine.load(MODEL, 12000)
            engine.count_tokens(MODEL, self.TEXT)
            calibrations = [b for b in self.generates(server)
                            if b.get("prompt", "").startswith("Calibration")]
        self.assertEqual([c["model"] for c in calibrations],
                         ["qwen3.5:0.8b", "other:1b", "qwen3.5:0.8b"])

    def test_calibration_does_not_reload_a_loaded_model(self):
        with FakeServer({"/api/version": fixture("version.json"),
                         "/api/generate": self.reply(1000)}) as server:
            engine = self.engine_for(server)
            engine.load(MODEL, 20000)
            engine.count_tokens(MODEL, self.TEXT * 20)
            ctxs = {b["options"]["num_ctx"] for b in self.generates(server)}
        self.assertEqual(ctxs, {20000})

    def test_unloaded_model_is_loaded_at_the_context_the_caller_needs(self):
        with FakeServer({"/api/version": fixture("version.json"),
                         "/api/generate": self.reply(1000)}) as server:
            engine = self.engine_for(server)
            engine.prepare_counting(MODEL, 10240)
            engine.count_tokens(MODEL, self.TEXT)
            ctxs = [b["options"]["num_ctx"] for b in self.generates(server)]
            engine.prepare_counting(MODEL, 4096)  # loaded already: no second load
            after = len(self.generates(server))
        self.assertEqual(ctxs, [10240, 10240])
        self.assertEqual(after, 2)

    def test_unloaded_model_without_a_hint_is_not_forced_to_2048(self):
        with FakeServer({"/api/version": fixture("version.json"),
                         "/api/generate": self.reply(1000)}) as server:
            engine = self.engine_for(server)
            engine.count_tokens(MODEL, self.TEXT * 20)  # 86000 chars
            ctxs = {b["options"]["num_ctx"] for b in self.generates(server)}
        self.assertEqual(len(ctxs), 1)
        self.assertNotIn(2048, ctxs)

    def test_calibration_sample_is_limited_to_the_context(self):
        with FakeServer({"/api/version": fixture("version.json"),
                         "/api/generate": self.reply(1000)}) as server:
            engine = self.engine_for(server)
            engine.load(MODEL, 2048)
            engine.count_tokens(MODEL, "word " * 50000)
            calibration = [b for b in self.generates(server)
                           if b["prompt"].startswith("Calibration")][0]
        self.assertLessEqual(len(calibration["prompt"]), 2 * 2048 + 100)

    def test_short_text_is_counted_exactly(self):
        with FakeServer({"/api/version": fixture("version.json"),
                         "/api/generate": self.reply(4)}) as server:
            engine = self.engine_for(server)
            engine.load(MODEL, 9000)
            self.assertEqual(engine.count_tokens(MODEL, "The Rhine flows"), 4)
            body = self.generates(server)[-1]
        self.assertEqual(body["prompt"], "The Rhine flows")

    def test_count_detects_truncation(self):
        with FakeServer({"/api/version": fixture("version.json"),
                         "/api/generate": self.reply(2048)}) as server:
            engine = self.engine_for(server)
            engine.load(MODEL, 2048)
            with self.assertRaises(EngineError):
                engine.count_tokens(MODEL, self.TEXT * 5)

    def test_count_real_fixture(self):
        with FakeServer({"/api/version": fixture("version.json"),
                         "/api/generate": fixture("generate-raw.json")}) as server:
            engine = self.engine_for(server)
            self.assertEqual(engine.count_tokens(MODEL, "The Rhine"), 10)


class CachedTimingTest(EngineTestCase):
    def test_prompt_tps_uses_only_uncached_tokens(self):
        warm = fixture("cache-probe.json")["replies"][1]
        lines = [json.dumps({"response": "x", "done": False}) + "\n",
                 json.dumps(warm | {"response": "", "done": True}) + "\n"]
        with FakeServer({"/api/version": fixture("version.json"),
                         "/api/generate": lines}) as server:
            engine = self.engine_for(server)
            t = engine.complete(MODEL, "Hi", Sampling()).timing
        self.assertEqual(t.prompt_tokens, 601)  # the total, cached included
        self.assertEqual(t.cached_tokens, 597)
        self.assertAlmostEqual(t.prompt_tps, (601 - 597) / 0.030052)

    def test_cold_call_tps(self):
        cold = fixture("cache-probe.json")["replies"][0]
        with FakeServer({"/api/version": fixture("version.json"),
                         "/api/generate": [json.dumps(cold | {"response": "x",
                                                              "done": True}) + "\n"]}) as server:
            engine = self.engine_for(server)
            t = engine.complete(MODEL, "Hi", Sampling()).timing
        self.assertAlmostEqual(t.prompt_tps, 601 / 13.752908)

    def test_chat_timing_uses_uncached_tokens_too(self):
        reply = fixture("chat-tools.json") | {"prompt_eval_count": 280,
                                              "prompt_eval_cached_count": 80,
                                              "prompt_eval_duration": 1_000_000_000}
        with FakeServer({"/api/version": fixture("version.json"), "/api/chat": reply}) as server:
            engine = self.engine_for(server)
            t = engine.chat(MODEL, [], Sampling()).timing
        self.assertEqual(t.prompt_tokens, 280)
        self.assertEqual(t.prompt_tps, 200.0)


def raw_server(test: unittest.TestCase, reply: bytes | None, close: bool = False) -> int:
    """A socket server that sends `reply` (nothing if None), then closes the connection
    if `close`, else keeps it open.

    Sockets are closed when the test ends. Returns the port.
    """
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    held: list[socket.socket] = []

    def serve():
        while True:
            try:
                conn = sock.accept()[0]
            except OSError:
                return
            held.append(conn)
            if reply is not None:
                conn.sendall(reply)
            if close:
                conn.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()

    def close():
        sock.close()
        for conn in held:
            conn.close()

    test.addCleanup(close)
    return sock.getsockname()[1]


def silent_server(test: unittest.TestCase) -> int:
    """Accepts connections and never answers."""
    return raw_server(test, None)


def garbage_server(test: unittest.TestCase) -> int:
    """Answers every connection with something that is not HTTP."""
    return raw_server(test, b"not http at all\r\n\r\n")


def truncated_500_server(test: unittest.TestCase) -> int:
    """A 500 whose Content-Length promises more than is sent."""
    return raw_server(test, b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 100\r\n"
                            b"Connection: close\r\n\r\nshort", close=True)


class RobustnessTest(ProcessOwnershipTestBase):
    def test_hanging_foreign_server_fails_fast(self):
        port = silent_server(self)
        engine = OllamaEngine(config(ollama_url=f"http://127.0.0.1:{port}"),
                              log_dir=self.dir, quick_timeout=0.3)
        started = time.monotonic()
        with self.assertRaises(EngineError):
            engine.start()
        self.assertLess(time.monotonic() - started, 5)

    def test_connection_closed_mid_response_is_an_engine_error(self):
        # 200 with a promised body of 1000 bytes, 5 sent, then the connection closes
        port = raw_server(self, b"HTTP/1.1 200 OK\r\nContent-Length: 1000\r\n"
                                b"Connection: close\r\n\r\nabcde", close=True)
        engine = OllamaEngine(config(ollama_url=f"http://127.0.0.1:{port}"), log_dir=self.dir)
        engine.base_url = f"http://127.0.0.1:{port}"
        for call in (engine.version, lambda: engine.load(MODEL, 2048),
                     lambda: engine.complete(MODEL, "x", Sampling()),
                     lambda: engine.chat(MODEL, [{"role": "user", "content": "x"}], Sampling()),
                     lambda: engine.count_tokens(MODEL, "x")):
            with self.assertRaises(EngineError):
                call()

    def test_inference_timeout_is_an_engine_error(self):
        port = silent_server(self)
        engine = OllamaEngine(config(ollama_url=f"http://127.0.0.1:{port}"), log_dir=self.dir,
                              timeout=0.3, quick_timeout=0.3)
        engine.base_url = f"http://127.0.0.1:{port}"
        started = time.monotonic()
        for call in (lambda: engine.complete(MODEL, "x", Sampling()),
                     lambda: engine.chat(MODEL, [{"role": "user", "content": "x"}], Sampling()),
                     lambda: engine.load(MODEL, 2048)):
            with self.assertRaises(EngineError):
                call()
        self.assertLess(time.monotonic() - started, 10)

    def test_timeout_while_streaming_is_an_engine_error(self):
        head = (b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
                b'1d\r\n{"response":"a","done":false}\n\r\n')
        port = raw_server(self, head)  # then silence
        engine = OllamaEngine(config(ollama_url=f"http://127.0.0.1:{port}"), log_dir=self.dir,
                              timeout=0.5)
        engine.base_url = f"http://127.0.0.1:{port}"
        with self.assertRaises(EngineError):
            engine.complete(MODEL, "x", Sampling())

    def test_truncated_error_body_becomes_engine_error(self):
        port = truncated_500_server(self)
        engine = OllamaEngine(config(ollama_url=f"http://127.0.0.1:{port}"), log_dir=self.dir)
        engine.base_url = f"http://127.0.0.1:{port}"
        for call in (engine.version, lambda: engine.complete(MODEL, "x", Sampling())):
            with self.assertRaises(EngineError) as ctx:
                call()
            self.assertNotIsInstance(ctx.exception, http.client.HTTPException)

    def test_failed_foreign_start_clears_base_url(self):
        port = garbage_server(self)
        engine = OllamaEngine(config(ollama_url=f"http://127.0.0.1:{port}"), log_dir=self.dir)
        with self.assertRaises(EngineError):
            engine.start()
        self.assertIsNone(engine.base_url)

    def test_quick_calls_use_the_short_timeout_and_inference_the_long_one(self):
        engine = OllamaEngine(config(ollama_url="http://127.0.0.1:1"), log_dir=self.dir,
                              quick_timeout=3, timeout=77)
        engine.base_url = "http://127.0.0.1:1"
        seen = {}

        def fake_open(request, timeout=None):
            seen[request.full_url.rsplit("/", 1)[-1] + ":" + str(
                json.loads(request.data).get("keep_alive") if request.data else "-")] = timeout
            raise urllib.error.URLError("stop")

        engine._opener = mock.Mock(open=fake_open)
        for call in (lambda: engine.version(), lambda: engine.gpu_fraction(MODEL),
                     lambda: engine.show(MODEL), lambda: engine.unload(MODEL),
                     lambda: engine.load(MODEL, 4096),
                     lambda: engine.complete(MODEL, "x", Sampling()),
                     lambda: engine.chat(MODEL, [], Sampling()),
                     lambda: engine.count_tokens(MODEL, "x")):
            with self.assertRaises(EngineError):
                call()
        self.assertEqual(seen["version:-"], 3)
        self.assertEqual(seen["ps:-"], 3)
        self.assertEqual(seen["show:None"], 3)
        self.assertEqual(seen["generate:0"], 3)  # unload
        self.assertEqual(seen["generate:30m"], 77)  # load, complete, count_tokens
        self.assertEqual(seen["chat:30m"], 77)

    def test_http_exceptions_become_engine_errors(self):
        port = garbage_server(self)
        engine = OllamaEngine(config(ollama_url=f"http://127.0.0.1:{port}"), log_dir=self.dir)
        engine.base_url = f"http://127.0.0.1:{port}"
        with self.assertRaises(EngineError):
            engine.version()
        with self.assertRaises(EngineError):
            engine.complete(MODEL, "x", Sampling())

    def test_incomplete_read_during_stream_and_reply(self):
        engine = OllamaEngine(config(ollama_url="http://x"), log_dir=self.dir)
        engine.base_url = "http://x"

        class Broken:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def __iter__(self):
                raise http.client.IncompleteRead(b"")

            def read(self):
                raise http.client.IncompleteRead(b"")

        engine._opener = mock.Mock(open=lambda *a, **k: Broken())
        with self.assertRaises(EngineError):
            engine.complete(MODEL, "x", Sampling())
        with self.assertRaises(EngineError):
            engine.version()

    def test_non_dict_chunk_and_reply(self):
        with FakeServer({"/api/version": fixture("version.json"),
                         "/api/generate": ["[1, 2]\n"], "/api/ps": "[1]"}) as server:
            engine = self.engine_for(server)
            with self.assertRaises(EngineError):
                engine.complete(MODEL, "x", Sampling())
            with self.assertRaises(EngineError):
                engine.gpu_fraction(MODEL)

    def test_http_proxy_variables_do_not_apply(self):
        with FakeServer({"/api/version": fixture("version.json")}) as server:
            proxy = {"HTTP_PROXY": "http://127.0.0.1:9", "http_proxy": "http://127.0.0.1:9",
                     "ALL_PROXY": "http://127.0.0.1:9"}
            with mock.patch.dict(os.environ, proxy):
                engine = OllamaEngine(config(ollama_url=server.url), log_dir=self.dir)
                engine.start()
                self.assertEqual(engine.version(), "0.40.13")

    def test_keep_alive_is_sent_with_inference_requests(self):
        with FakeServer({"/api/version": fixture("version.json"),
                         "/api/generate": lambda body: stream_lines() if body.get("prompt")
                         else fixture("generate-raw.json"),
                         "/api/chat": fixture("chat-json.json")}) as server:
            engine = self.engine_for(server)
            engine.load(MODEL, 4096)
            engine.complete(MODEL, "Hi", Sampling())
            engine.chat(MODEL, [], Sampling())
            engine.unload(MODEL)
            bodies = [b for _, p, b in server.requests if p in ("/api/generate", "/api/chat")]
        self.assertEqual([b["keep_alive"] for b in bodies], ["30m", "30m", "30m", 0])

    def test_gpu_fraction_latest_tag(self):
        ps = {"models": [{"name": "qwen3.5:latest", "size": 10, "size_vram": 5}]}
        bare = ModelRef("S1", "S", "Q8_0", "qwen3.5", None, None, None)
        with FakeServer({"/api/version": fixture("version.json"), "/api/ps": ps}) as server:
            engine = self.engine_for(server)
            self.assertEqual(engine.gpu_fraction(bare), 0.5)
            self.assertEqual(engine.gpu_fraction(
                ModelRef("S1", "S", "Q8_0", "qwen3.5:latest", None, None, None)), 0.5)
            self.assertIsNone(engine.gpu_fraction(MODEL))

    def test_parallel_property(self):
        with FakeServer({"/api/version": fixture("version.json")}) as server:
            self.assertIsNone(self.engine_for(server).parallel)


@unittest.skipIf(os.name == "nt", "the fake binary is a POSIX script")
class OwnedServerFailureTest(ProcessOwnershipTestBase):
    def setUp(self):
        super().setUp()
        self.created = []
        real_owned = proc.Owned
        created = self.created

        class Recording(real_owned):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                created.append(self)

        patcher = mock.patch("bench.engines.ollama.proc.Owned", Recording)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(lambda: [o.stop() for o in self.created])
        os.environ["FAKE_OUT"] = str(self.dir / "env.json")
        self.addCleanup(os.environ.pop, "FAKE_OUT", None)

    def engine(self, **kw) -> OllamaEngine:
        return OllamaEngine(config(ollama_bin=self.fake_binary()), log_dir=self.dir / "logs",
                            **kw)

    def assert_no_child(self):
        self.assertEqual(len(self.created), 1)
        self.assertFalse(self.created[0].running)

    def test_keyboard_interrupt_during_startup_stops_the_child(self):
        engine = self.engine()
        with mock.patch.object(proc.Owned, "wait_until", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                engine.start()
        self.assert_no_child()
        self.assertIsNone(engine._owned)
        self.assertIsNone(engine.base_url)

    def test_http_exception_during_startup_stops_the_child(self):
        engine = self.engine(start_timeout=1.5)
        engine._opener = mock.Mock(open=mock.Mock(side_effect=http.client.RemoteDisconnected()))
        with self.assertRaises(EngineError):
            engine.start()
        self.assert_no_child()
        self.assertIsNone(engine._owned)

    def test_pure_http_exception_during_startup_stops_the_child(self):
        # BadStatusLine is an HTTPException but not an OSError (RemoteDisconnected is both).
        self.assertFalse(issubclass(http.client.BadStatusLine, OSError))
        engine = self.engine(start_timeout=1.5)
        engine._opener = mock.Mock(open=mock.Mock(side_effect=http.client.BadStatusLine("x")))
        with self.assertRaises(EngineError):
            engine.start()
        self.assert_no_child()
        self.assertIsNone(engine._owned)

    def test_log_directory_that_cannot_be_created_is_an_engine_error(self):
        blocker = self.dir / "blocker"
        blocker.write_text("a file, not a directory", encoding="utf-8")
        engine = OllamaEngine(config(ollama_bin=self.fake_binary()), log_dir=blocker / "logs")
        with self.assertRaises(EngineError):
            engine.start()
        self.assertIsNone(engine._owned)
        self.assertIsNone(engine.base_url)

    def test_failed_restart_leaves_no_stale_state(self):
        engine = self.engine()
        engine.start()
        self.addCleanup(engine.stop)
        first = engine._owned
        with mock.patch.object(engine, "_binary", side_effect=EngineError("gone")):
            with self.assertRaises(EngineError):
                engine.load(MODEL, 4096, parallel=4)
        self.assertFalse(first.running)
        self.assertIsNone(engine._owned)
        self.assertIsNone(engine.base_url)
        self.assertIsNone(engine.parallel)

    def test_start_assigns_ownership_before_spawning(self):
        engine = self.engine()
        seen = []
        real_start = proc.Owned.start

        def start(self_):
            seen.append(engine._owned is self_)
            return real_start(self_)

        with mock.patch.object(proc.Owned, "start", start):
            engine.start()
        self.addCleanup(engine.stop)
        self.assertEqual(seen, [True])

    def test_unexpected_exception_during_startup_stops_the_child(self):
        engine = self.engine()
        with mock.patch.object(proc.Owned, "wait_until", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                engine.start()
        self.assert_no_child()

    def test_server_that_never_answers_fails_within_the_deadline(self):
        os.environ["FAKE_MODE"] = "hang"
        self.addCleanup(os.environ.pop, "FAKE_MODE", None)
        engine = self.engine(start_timeout=1.5, quick_timeout=0.3)
        started = time.monotonic()
        with self.assertRaises(EngineError):
            engine.start()
        self.assertLess(time.monotonic() - started, 10)
        self.assert_no_child()

    def test_second_start_is_refused_and_keeps_the_first_server(self):
        engine = self.engine()
        engine.start()
        self.addCleanup(engine.stop)
        first = engine._owned
        with self.assertRaises(EngineError):
            engine.start()
        self.assertIs(engine._owned, first)
        self.assertTrue(first.running)
        self.assertEqual(len(self.created), 1)

    def test_stop_resets_state(self):
        engine = self.engine()
        engine.start()
        self.assertEqual(engine.parallel, 1)
        engine.stop()
        self.assertIsNone(engine._owned)
        self.assertIsNone(engine.base_url)
        self.assertIsNone(engine.parallel)
        engine.stop()

    def test_somebody_else_answering_on_the_port_is_not_trusted(self):
        # The probe is answered, but our child is gone by then: the answer came from
        # another server on that port, which must not be used any further.
        engine = self.engine()

        def probe_then_child_dies(self_, predicate, timeout, interval=0.2):
            self.assertTrue(predicate())  # really reaches the fake server
            self_.stop()

        with FakeServer({"/api/version": fixture("version.json"),
                         "/api/generate": fixture("generate-raw.json")}) as server:
            port = int(server.url.rsplit(":", 1)[1])
            with mock.patch("bench.engines.ollama.pick_port", return_value=port), \
                    mock.patch.object(proc.Owned, "wait_until", probe_then_child_dies):
                with self.assertRaises(EngineError):
                    engine.start()
            self.assertEqual(server.requests, [("GET", "/api/version", None)])
            with self.assertRaises(EngineError):  # engine is disconnected
                engine.version()
            self.assertEqual(len(server.requests), 1)
        self.assertIsNone(engine.base_url)
        self.assertIsNone(engine._owned)


if __name__ == "__main__":
    unittest.main()


class TruncationTest(EngineTestCase):
    """Ollama 0.40.13 silently cuts a prompt longer than num_ctx to num_ctx/2+2 tokens and
    only says so in its log: `msg="truncating input prompt" limit=L prompt=N keep=4 new=L`."""

    TEXT = "The Rhine flows north through many cities. " * 400  # 16800 chars

    @staticmethod
    def warn(limit, prompt):
        return (f'time=2026-11-03T10:00:00 level=WARN source=llama_server.go:344 '
                f'msg="truncating input prompt" limit={limit} prompt={prompt} keep=4 '
                f'new={limit}\n')

    def ready(self, server, ctx=8192, ratio_count=4000):
        engine = self.engine_for(server)
        engine.load(MODEL, ctx)
        engine.count_tokens(MODEL, self.TEXT)  # calibrate (server answers ratio_count)
        engine._log_path = self.dir / "ollama.log"
        engine._log_path.write_text("old " + self.warn(10, 999999), encoding="utf-8")
        return engine

    def generate_route(self, engine_box, count, log_line=None):
        def route(body):
            if body.get("prompt") and not str(body["prompt"]).startswith("Calibration") \
                    and log_line:
                with engine_box[0]._log_path.open("a", encoding="utf-8") as handle:
                    handle.write(log_line)
            if body.get("stream"):
                return json.dumps({"response": "x", "done": True,
                                   "prompt_eval_count": count}) + "\n"
            return {"done": True, "response": "x", "prompt_eval_count": count}
        return route

    def test_log_line_written_during_the_request_raises_with_the_real_count(self):
        box = []
        with FakeServer({"/api/version": fixture("version.json")}) as server:
            server.routes["/api/generate"] = lambda body: {"done": True, "prompt_eval_count": 4000}
            engine = self.ready(server)
            box.append(engine)
            server.routes["/api/generate"] = self.generate_route(
                box, 4098, self.warn(4098, 9000))
            with self.assertRaises(PromptTruncated) as caught:
                engine.complete(MODEL, self.TEXT, Sampling())
        self.assertEqual(caught.exception.real_tokens, 9000)  # not the old line's 999999
        self.assertEqual(caught.exception.kept_tokens, 4098)
        self.assertIsInstance(caught.exception, EngineError)

    def test_real_count_is_fed_back_into_the_ratio(self):
        box = []
        with FakeServer({"/api/version": fixture("version.json")}) as server:
            server.routes["/api/generate"] = lambda body: {"done": True, "prompt_eval_count": 4000}
            engine = self.ready(server)
            box.append(engine)
            before = engine.count_tokens(MODEL, self.TEXT)
            server.routes["/api/generate"] = self.generate_route(
                box, 4098, self.warn(4098, 8000))
            with self.assertRaises(PromptTruncated):
                engine.complete(MODEL, self.TEXT, Sampling())
            after = engine.count_tokens(MODEL, self.TEXT)
        self.assertNotEqual(before, after)
        self.assertEqual(after, 8000)

    def test_small_count_without_a_log_line_is_truncation_too(self):
        # fallback: the reply has fewer than 90 % of the tokens we expect
        with FakeServer({"/api/version": fixture("version.json")}) as server:
            server.routes["/api/generate"] = lambda body: {"done": True, "prompt_eval_count": 4000}
            engine = self.ready(server)
            expected = engine.count_tokens(MODEL, self.TEXT)
            server.routes["/api/generate"] = self.generate_route([engine], int(expected * 0.8))
            with self.assertRaises(PromptTruncated) as caught:
                engine.complete(MODEL, self.TEXT, Sampling())
        self.assertIsNone(caught.exception.real_tokens)

    def test_good_reply_passes_and_teaches_the_ratio(self):
        with FakeServer({"/api/version": fixture("version.json")}) as server:
            server.routes["/api/generate"] = lambda body: {"done": True, "prompt_eval_count": 4000}
            engine = self.ready(server)
            expected = engine.count_tokens(MODEL, self.TEXT)
            server.routes["/api/generate"] = self.generate_route([engine], int(expected * 1.05))
            result = engine.complete(MODEL, self.TEXT, Sampling())
            after = engine.count_tokens(MODEL, self.TEXT)
        self.assertEqual(result.timing.prompt_tokens, int(expected * 1.05))
        self.assertEqual(after, int(expected * 1.05))

    def test_short_prompts_are_never_checked(self):
        with FakeServer({"/api/version": fixture("version.json"),
                         "/api/generate": {"done": True, "response": "x",
                                           "prompt_eval_count": 3}}) as server:
            engine = self.engine_for(server)
            engine.load(MODEL, 4096)
            server.routes["/api/generate"] = [json.dumps(
                {"response": "x", "done": True, "prompt_eval_count": 3}) + "\n"]
            engine.complete(MODEL, "Hi there", Sampling())

    def test_chat_is_checked_too(self):
        box = []
        with FakeServer({"/api/version": fixture("version.json")}) as server:
            server.routes["/api/generate"] = lambda body: {"done": True, "prompt_eval_count": 4000}
            engine = self.ready(server)
            box.append(engine)

            def chat(body):
                with engine._log_path.open("a", encoding="utf-8") as handle:
                    handle.write(self.warn(4098, 9500))
                return {"message": {"content": "x"}, "done": True, "prompt_eval_count": 4098}

            server.routes["/api/chat"] = chat
            with self.assertRaises(PromptTruncated) as caught:
                engine.chat(MODEL, [{"role": "user", "content": self.TEXT}], Sampling())
        self.assertEqual(caught.exception.real_tokens, 9500)
