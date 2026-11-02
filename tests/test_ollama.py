import copy
import json
import os
import socket
import stat
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from bench.config import Config
from bench.engines.base import ChatResult, Completion, EngineError, ModelRef, Sampling, Timing
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
env = {{k: v for k, v in os.environ.items() if k.startswith("OLLAMA_")}}
open(os.environ["FAKE_OUT"], "w").write(json.dumps({{"argv": sys.argv[1:], "env": env}}))
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


class ProcessOwnershipTest(EngineTestCase):
    def fake_binary(self) -> Path:
        path = self.dir / "ollama"
        path.write_text(FAKE_OLLAMA.format(python=sys.executable), encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return path

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
            pid = engine._owned.pid
        finally:
            engine.stop()
        # The process is gone, the engine can be stopped again, our environment is intact.
        self.assertFalse(engine._owned.running)
        engine.stop()
        with self.assertRaises(OSError):
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
        self.assertEqual(dict(os.environ), before)
        self.assertIsInstance(pid, int)
        # The listener on the default port was never touched.
        with socket.create_connection(("127.0.0.1", 11434), timeout=1):
            pass

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

        def post(path, payload):
            if path == "/api/generate":
                return {"load_duration": 1_000_000_000}
            return real_post(path, payload)

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
        final = {"response": "", "done": True, "prompt_eval_count": 0,
                 "prompt_eval_cached_count": 512, "prompt_eval_duration": 0,
                 "eval_count": 0, "eval_duration": 0}
        with FakeServer({"/api/version": fixture("version.json"),
                         "/api/generate": [json.dumps(final) + "\n"]}) as server:
            engine = self.engine_for(server)
            t = engine.complete(MODEL, "Hi", Sampling()).timing
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
        engine._owned = mock.Mock(log_path=log)
        self.assertEqual(engine.backend(), "cuda")
        self.assertEqual(engine.kv_buffer_mb(), 512.5)

    def test_kv_buffer_only_counts_the_latest_load(self):
        log = self.dir / "server.log"
        log.write_text("llama_kv_cache: CUDA0 KV buffer size = 100.00 MiB\n", encoding="utf-8")
        engine = OllamaEngine(config(), log_dir=self.dir)
        engine._owned = mock.Mock(log_path=log)
        engine._log_mark = log.stat().st_size
        with log.open("a", encoding="utf-8") as handle:
            handle.write("llama_kv_cache: CUDA0 KV buffer size = 200.00 MiB\n")
        self.assertEqual(engine.kv_buffer_mb(), 200.0)


class CountTokensTest(EngineTestCase):
    def reply(self, count, cached=0):
        return {"done": True, "response": "x", "prompt_eval_count": count,
                "prompt_eval_cached_count": cached}

    def test_count_tokens_request_and_cache(self):
        with FakeServer({"/api/version": fixture("version.json"),
                         "/api/generate": self.reply(10)}) as server:
            engine = self.engine_for(server)
            self.assertEqual(engine.count_tokens(MODEL, "The Rhine flows"), 10)
            self.assertEqual(engine.count_tokens(MODEL, "The Rhine flows"), 10)
            self.assertEqual(len([r for r in server.requests if r[1] == "/api/generate"]), 1)
            body = [b for _, p, b in server.requests if p == "/api/generate"][0]
            server.routes["/api/generate"] = self.reply(3)
            self.assertEqual(engine.count_tokens(MODEL, "Other text"), 3)
        self.assertEqual(body["prompt"], "The Rhine flows")
        self.assertIs(body["raw"], True)
        self.assertIs(body["stream"], False)
        self.assertEqual(body["options"]["num_predict"], 1)
        self.assertGreaterEqual(body["options"]["num_ctx"], 2048)

    def test_count_tokens_adds_cached_tokens(self):
        with FakeServer({"/api/version": fixture("version.json"),
                         "/api/generate": self.reply(6, cached=4)}) as server:
            engine = self.engine_for(server)
            self.assertEqual(engine.count_tokens(MODEL, "same prompt again"), 10)

    def test_count_tokens_uses_loaded_ctx(self):
        with FakeServer({"/api/version": fixture("version.json"),
                         "/api/generate": self.reply(5)}) as server:
            engine = self.engine_for(server)
            engine.load(MODEL, 9000)
            engine.count_tokens(MODEL, "abc")
            body = [b for _, p, b in server.requests if p == "/api/generate"][-1]
        self.assertEqual(body["options"]["num_ctx"], 9000)

    def test_count_tokens_detects_truncation(self):
        with FakeServer({"/api/version": fixture("version.json"),
                         "/api/generate": self.reply(2048)}) as server:
            engine = self.engine_for(server)
            with self.assertRaises(EngineError):
                engine.count_tokens(MODEL, "long text")

    def test_count_tokens_real_fixture(self):
        with FakeServer({"/api/version": fixture("version.json"),
                         "/api/generate": fixture("generate-raw.json")}) as server:
            engine = self.engine_for(server)
            self.assertEqual(engine.count_tokens(MODEL, "The Rhine"), 10)


if __name__ == "__main__":
    unittest.main()
