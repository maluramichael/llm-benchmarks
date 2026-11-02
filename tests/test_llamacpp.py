import json
import os
import socket
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from bench import proc
from bench.config import Config
from bench.engines import llamacpp
from bench.engines.base import ChatResult, Completion, EngineError, ModelRef, Sampling, Timing
from bench.engines.llamacpp import (LlamaCppEngine, parse_backend, parse_kv_buffer_mb,
                                    parse_llama_bench, parse_offloaded_layers,
                                    parse_rs_buffer_mb, parse_version, run_llama_bench)
from fake_http import FakeServer

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "llamacpp"


def text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def fixture(name: str) -> dict:
    return json.loads(text(name))


# -- log and output parsers ----------------------------------------------------------------

class ParserTest(unittest.TestCase):
    def test_kv_buffer_from_log(self):
        self.assertEqual(parse_kv_buffer_mb(text("server-lv4.log")), 256.0)

    def test_kv_buffer_sums_all_devices(self):
        log = ("llama_kv_cache:      CUDA0 KV buffer size =   256.00 MiB\n"
               "llama_kv_cache:      CUDA1 KV buffer size =   128.50 MiB\n"
               "llama_kv_cache:  CUDA_Host KV buffer size =     0.50 MiB\n"
               "llama_kv_cache: size =  385.00 MiB (  8192 cells)\n")
        self.assertEqual(parse_kv_buffer_mb(log), 385.0)

    def test_kv_buffer_unknown(self):
        self.assertIsNone(parse_kv_buffer_mb(""))
        self.assertIsNone(parse_kv_buffer_mb("llama_kv_cache: size = 256.00 MiB\n"))

    def test_recurrent_state_is_separate_from_kv(self):
        log = text("server-lv4.log")
        self.assertEqual(parse_rs_buffer_mb(log), 50.25)
        self.assertEqual(parse_kv_buffer_mb(log), 256.0)  # RS is not added to it
        self.assertIsNone(parse_rs_buffer_mb("llama_kv_cache: CUDA0 KV buffer size = 1 MiB"))

    def test_offloaded_layers(self):
        self.assertEqual(parse_offloaded_layers(text("server-lv4.log")), (33, 33))
        self.assertEqual(parse_offloaded_layers("load_tensors: offloaded 20/33 layers to GPU"),
                         (20, 33))
        self.assertIsNone(parse_offloaded_layers("nothing"))

    def test_backend(self):
        self.assertEqual(parse_backend(text("server-lv4.log")), "cuda")
        self.assertEqual(parse_backend("llama_prepare_model_devices: using device Vulkan0 "
                                       "(NVIDIA GeForce GTX 1080 Ti) - 8000 MiB free"),
                         "vulkan")
        self.assertEqual(parse_backend("llama_prepare_model_devices: using device MTL0 "
                                       "(Apple M2) - 1 MiB free"), "metal")
        self.assertEqual(parse_backend("llama_prepare_model_devices: using device ROCm0 "
                                       "(AMD) - 1 MiB free"), "rocm")
        self.assertEqual(parse_backend("load_tensors: offloaded 0/33 layers to GPU"), "cpu")
        self.assertIsNone(parse_backend(""))

    def test_version(self):
        self.assertEqual(parse_version(text("server-version.txt")), "b12061 (d02b4c0d6)")
        self.assertEqual(parse_version("version: 4000 (abc1234)\nbuilt with gcc"),
                         "b4000 (abc1234)")
        self.assertEqual(parse_version("version: something odd"), "something odd")
        self.assertIsNone(parse_version("no version here"))

    def test_version_takes_the_last_match(self):
        old_and_new = ("version: 0.7.0 (build 11000, commit aaaa111)\n"
                       "version: 0.8.0-dev (build 12061, commit d02b4c0d6)\n")
        self.assertEqual(parse_version(old_and_new), "b12061 (d02b4c0d6)")

    def test_kv_unified(self):
        self.assertIs(llamacpp.parse_kv_unified(text("server-lv4.log")), False)
        self.assertIs(llamacpp.parse_kv_unified("... n_slots = 4, kv_unified = 'true'"), True)
        self.assertIs(llamacpp.parse_kv_unified("llama_context: kv_unified = false"), False)
        self.assertIsNone(llamacpp.parse_kv_unified("nothing"))

    def test_flash_attention_failure_detection(self):
        failing = ("llama_init_from_model: flash_attn requested but not supported by the "
                   "backend\n")
        self.assertIn("not supported", llamacpp.flash_attn_failure(failing))
        self.assertIsNotNone(llamacpp.flash_attn_failure(
            "llama_init_from_model: layer 3 is assigned to device CUDA0 but the Flash "
            "Attention tensor is assigned to device CPU (usually due to missing support)"))
        # The context dump mentions flash_attn on every start, also when memory ran out.
        oom = ("llama_context: flash_attn            = enabled\n"
               "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 9000 MiB failed\n"
               "llama_init_from_model: failed to initialize the context\n")
        self.assertIsNone(llamacpp.flash_attn_failure(oom))
        self.assertIsNone(llamacpp.flash_attn_failure(text("server-lv4.log")))


# -- llama-bench ---------------------------------------------------------------------------

FAKE_BENCH = """#!{python}
import json, os, sys, time
open(os.environ["FAKE_OUT"], "w").write(json.dumps(sys.argv[1:]))
mode = os.environ.get("FAKE_MODE", "ok")
print("ggml_cuda_init: found 1 CUDA devices", file=sys.stderr)
if mode == "hang":
    time.sleep(60)
if mode == "fail":
    print("error: failed to load model " + sys.argv[sys.argv.index("-m") + 1], file=sys.stderr)
    sys.exit(2)
if mode == "garbage":
    print("this is not json")
    sys.exit(0)
sys.stdout.write(open(os.environ["FAKE_JSON"], encoding="utf-8").read())
"""

MODEL = ModelRef(id="S1", cls="S", quant="Q4_K_M", ollama_tag=None,
                 gguf_path=Path("C:/models/model.gguf"), digest=None, size_bytes=None)


class TempCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def setenv(self, **values):
        patcher = mock.patch.dict(os.environ, values)
        patcher.start()
        self.addCleanup(patcher.stop)

    def script(self, name: str, body: str) -> Path:
        path = self.dir / name
        path.write_text(body.format(python=sys.executable), encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return path


class LlamaBenchTest(TempCase):
    def test_llama_bench_json(self):
        rows = parse_llama_bench(text("llama-bench.json"))
        self.assertEqual([(r["n_prompt"], r["n_gen"], r["n_depth"]) for r in rows],
                         [(512, 0, 0), (0, 64, 0)])
        self.assertAlmostEqual(rows[0]["avg_ts"], 8122.196212)
        self.assertAlmostEqual(rows[1]["avg_ts"], 183.7, delta=1)
        depth = parse_llama_bench(text("llama-bench-depth.json"))
        self.assertEqual([r["n_depth"] for r in depth], [4096, 4096])

    def test_parse_rejects_non_arrays(self):
        for bad in ("", "oops", "{}", "[1, 2]", "[]"):
            with self.subTest(bad=bad), self.assertRaises(EngineError):
                parse_llama_bench(bad)

    def test_parse_tolerates_a_byte_order_mark(self):
        self.assertEqual(len(parse_llama_bench("\ufeff" + text("llama-bench.json"))), 2)

    def run_bench(self, mode="ok", args=("-p", "512", "-n", "64"), **kw):
        out = self.dir / "args.json"
        self.setenv(FAKE_OUT=str(out), FAKE_MODE=mode,
                    FAKE_JSON=str(FIXTURES / "llama-bench.json"))
        binary = self.script("llama-bench", FAKE_BENCH)
        return run_llama_bench(binary, MODEL, list(args), log_dir=self.dir / "logs", **kw), out

    @unittest.skipIf(os.name == "nt", "the fake binary is a POSIX script")
    def test_run_llama_bench(self):
        rows, out = self.run_bench()
        self.assertEqual(len(rows), 2)
        # stdout is data, stderr is only log: the noise did not end up in the result.
        self.assertEqual(json.loads(out.read_text()),
                         ["-m", str(MODEL.gguf_path), "-o", "json", "-p", "512", "-n", "64"])
        # The local model path is not passed on to results.
        self.assertTrue(all("model_filename" not in row for row in rows))

    @unittest.skipIf(os.name == "nt", "the fake binary is a POSIX script")
    def test_failures_are_engine_errors(self):
        for mode, needle in (("fail", "exit"), ("garbage", "JSON")):
            with self.subTest(mode=mode), self.assertRaises(EngineError) as ctx:
                self.run_bench(mode)
            self.assertIn(needle, str(ctx.exception))

    @unittest.skipIf(os.name == "nt", "the fake binary is a POSIX script")
    def test_bench_error_message_hides_the_model_path(self):
        gguf = self.dir / "private" / "model.gguf"
        model = ModelRef("S1", "S", "Q4", None, gguf, None, None)
        out = self.dir / "args.json"
        self.setenv(FAKE_OUT=str(out), FAKE_MODE="fail")
        with self.assertRaises(EngineError) as ctx:
            run_llama_bench(self.script("llama-bench", FAKE_BENCH), model, [],
                            log_dir=self.dir / "logs")
        self.assertNotIn(str(gguf), str(ctx.exception))
        self.assertNotIn(str(gguf.parent), str(ctx.exception))
        self.assertIn("<model>", str(ctx.exception))

    @unittest.skipIf(os.name == "nt", "the fake binary is a POSIX script")
    def test_timeout_stops_the_process(self):
        created = []
        real = proc.Owned

        class Recording(real):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                created.append(self)

        with mock.patch("bench.engines.llamacpp.proc.Owned", Recording):
            with self.assertRaises(EngineError) as ctx:
                self.run_bench("hang", timeout=0.5)
        self.assertIn("after", str(ctx.exception))
        self.assertEqual(len(created), 1)
        self.assertFalse(created[0].running)

    def test_missing_binary_and_model(self):
        with self.assertRaises(EngineError):
            run_llama_bench(self.dir / "nope", MODEL, [], log_dir=self.dir)
        no_model = ModelRef("x", "S", "Q4", None, None, None, None)
        with self.assertRaises(EngineError):
            run_llama_bench(self.dir / "nope", no_model, [], log_dir=self.dir)


# -- timing --------------------------------------------------------------------------------

class TimingTest(unittest.TestCase):
    def test_timings(self):
        # Cold: prompt_n is the whole prompt.
        cold = fixture("completion-cold.json")
        t = LlamaCppEngine._timing(cold["timings"], 0.5)
        self.assertIsInstance(t, Timing)
        self.assertEqual(t.prompt_tokens, 601)
        self.assertEqual(t.cached_tokens, 0)
        self.assertAlmostEqual(t.prompt_tps, 601 / 0.13105)
        self.assertEqual(t.gen_tokens, 32)
        self.assertAlmostEqual(t.gen_tps, 32 / 0.182195)
        self.assertIsNone(t.load_s)
        self.assertEqual(t.ttft_s, 0.5)

    def test_prompt_n_excludes_cached_tokens(self):
        # Warm: llama.cpp reports 4 evaluated + 597 cached; the total is their sum
        # (Ollama is the other way round), the speed covers only the evaluated part.
        warm = fixture("completion-warm.json")
        self.assertEqual(warm["tokens_evaluated"], 601)
        t = LlamaCppEngine._timing(warm["timings"], 0.1)
        self.assertEqual(t.prompt_tokens, 601)
        self.assertEqual(t.cached_tokens, 597)
        self.assertAlmostEqual(t.prompt_tps, 4 / 0.029288)

    def test_missing_cache_and_zero_values(self):
        t = LlamaCppEngine._timing({"prompt_n": 10, "prompt_ms": 5.0, "predicted_n": 0,
                                    "predicted_ms": 0.0}, 0.0)
        self.assertEqual((t.prompt_tokens, t.cached_tokens), (10, None))
        self.assertAlmostEqual(t.prompt_tps, 2000.0)
        self.assertIsNone(t.gen_tps)
        full = LlamaCppEngine._timing({"prompt_n": 0, "cache_n": 50, "prompt_ms": 0.0,
                                       "predicted_n": 8, "predicted_ms": 80.0}, 0.0)
        self.assertEqual(full.prompt_tokens, 50)
        self.assertIsNone(full.prompt_tps)
        self.assertAlmostEqual(full.gen_tps, 100.0)


# -- server process ------------------------------------------------------------------------

FAKE_SERVER = """#!{python}
import json, os, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer
argv = sys.argv[1:]
if argv == ["--version"]:
    sys.stderr.write(open(os.environ["FAKE_VERSION"], encoding="utf-8").read())
    sys.exit(0)
with open(os.environ["FAKE_OUT"], "a", encoding="utf-8") as out:
    out.write(json.dumps(argv) + "\\n")
mode = os.environ.get("FAKE_MODE", "ok")
fa = None
for i, a in enumerate(argv):
    if a in ("-fa", "--flash-attn"):
        fa = argv[i + 1]
    elif a.startswith("--flash-attn="):
        fa = a.split("=", 1)[1]
log = open(os.environ["FAKE_LOG"], encoding="utf-8").read()
sys.stderr.write(log)
sys.stderr.flush()
if fa == "on" and mode == "fa_unsupported":
    sys.stderr.write("llama_init_from_model: flash_attn requested but not supported\\n")
    sys.exit(1)
if mode == "always_fa_error":
    sys.stderr.write("llama_init_from_model: flash_attn requested but not supported\\n")
    sys.exit(1)
if mode == "oom":
    sys.stderr.write("srv load_model: loading model '" + argv[argv.index("-m") + 1] + "'\\n")
    sys.stderr.write("ggml_backend_cuda_buffer_type_alloc_buffer: allocating failed\\n")
    sys.exit(1)
if mode == "hang":
    time.sleep(120)
port = int(argv[argv.index("--port") + 1])
host = argv[argv.index("--host") + 1]
hits = [0]
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        hits[0] += 1
        if mode == "slow_health" and hits[0] <= 3:  # still loading: 503, like the real one
            self.send_response(503); self.end_headers()
            self.wfile.write(b'{{"error":{{"code":503,"message":"Loading model"}}}}')
            return
        if "FAKE_HEALTH" in os.environ:
            open(os.environ["FAKE_HEALTH"], "w").write(str(hits[0]))
        self.send_response(200); self.end_headers()
        self.wfile.write(b'{{"status":"ok"}}')
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
        self.send_response(200); self.end_headers()
        words = body["content"].split()
        self.wfile.write(json.dumps({{"tokens": list(range(len(words)))}}).encode())
HTTPServer((host, port), H).serve_forever()
"""


class ServerCase(TempCase):
    def setUp(self):
        super().setUp()
        self.out = self.dir / "launches.jsonl"
        self.setenv(FAKE_OUT=str(self.out), FAKE_LOG=str(FIXTURES / "server-lv4.log"),
                    FAKE_VERSION=str(FIXTURES / "server-version.txt"), FAKE_MODE="ok")
        self.server_bin = self.script("llama-server", FAKE_SERVER)
        self.bench_bin = self.script("llama-bench", FAKE_BENCH)
        self.gguf = self.dir / "model.gguf"
        self.gguf.write_bytes(b"GGUF")
        self.model = ModelRef("S1", "S", "Q4_K_M", None, self.gguf, None, None)
        self.created = []
        real = proc.Owned
        created = self.created

        class Recording(real):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                created.append(self)

        patcher = mock.patch("bench.engines.llamacpp.proc.Owned", Recording)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(lambda: [o.stop() for o in self.created])

    def engine(self, **cfg) -> LlamaCppEngine:
        engine = LlamaCppEngine(Config(**cfg), log_dir=self.dir / "logs",
                                binaries=(self.server_bin, self.bench_bin), start_timeout=20)
        self.addCleanup(engine.stop)
        engine.start()
        return engine

    def launches(self) -> list[list[str]]:
        if not self.out.exists():
            return []
        return [json.loads(line) for line in self.out.read_text().splitlines()]

    @staticmethod
    def opt(argv: list[str], name: str) -> str:
        return argv[argv.index(name) + 1]


@unittest.skipIf(os.name == "nt", "the fake binary is a POSIX script")
class ServerProcessTest(ServerCase):
    def test_launch_flags(self):
        engine = self.engine()
        seconds = engine.load(self.model, 8192)
        self.assertIsInstance(seconds, float)
        self.assertGreater(seconds, 0)
        (argv,) = self.launches()
        self.assertEqual(self.opt(argv, "-m"), str(self.gguf))
        self.assertEqual(self.opt(argv, "-ngl"), "99")
        self.assertEqual(self.opt(argv, "-fa"), "on")
        self.assertEqual(self.opt(argv, "-c"), "8192")
        self.assertEqual(self.opt(argv, "-np"), "1")
        self.assertEqual(self.opt(argv, "--host"), "127.0.0.1")
        self.assertEqual(self.opt(argv, "-fit"), "off")
        self.assertEqual(self.opt(argv, "-cram"), "0")
        self.assertEqual(self.opt(argv, "-lv"), "4")
        self.assertIn("--jinja", argv)
        port = int(self.opt(argv, "--port"))
        self.assertNotIn(port, (0, 8080))
        self.assertEqual(engine.base_url, f"http://127.0.0.1:{port}")

    def test_parallel_gives_every_slot_the_full_context(self):
        engine = self.engine()
        engine.load(self.model, 4096, parallel=4)
        (argv,) = self.launches()
        self.assertEqual(self.opt(argv, "-np"), "4")
        self.assertEqual(self.opt(argv, "-c"), "16384")

    def test_llama_args_come_last(self):
        engine = self.engine(llama_args='--threads 4 --alias "my model" -ngl 20')
        engine.load(self.model, 2048)
        (argv,) = self.launches()
        self.assertEqual(argv[-6:], ["--threads", "4", "--alias", "my model", "-ngl", "20"])
        self.assertLess(argv.index("-ngl"), argv.index("--threads"))  # ours precede

    def test_user_flash_attention_setting_is_kept_and_never_retried(self):
        self.setenv(FAKE_MODE="fa_unsupported")
        for args in ("-fa on", "--flash-attn on", "--flash-attn=on"):
            self.out.unlink(missing_ok=True)
            engine = self.engine(llama_args=args)
            with self.subTest(args=args), self.assertRaises(EngineError):
                engine.load(self.model, 2048)
            (argv,) = self.launches()  # exactly one start: no automatic fallback
            self.assertEqual(sum(a in ("-fa", "--flash-attn") or a.startswith("--flash-attn=")
                                 for a in argv), 1)
            self.assertEqual(engine.meta["flash_attn"], "user")
            engine.stop()

    def test_meta_and_metrics_from_the_log(self):
        engine = self.engine()
        engine.load(self.model, 8192)
        self.assertEqual(engine.kv_buffer_mb(), 256.0)
        self.assertEqual(engine.rs_buffer_mb(), 50.25)
        self.assertEqual(engine.backend(), "cuda")
        self.assertEqual(engine.gpu_fraction(self.model), 1.0)
        self.assertEqual(engine.version(), "b12061 (d02b4c0d6)")
        self.assertEqual(engine.meta["flash_attn"], "on")
        self.assertIs(engine.meta["flash_attn_fallback"], False)
        self.assertEqual(engine.meta["ctx"], 8192)
        self.assertEqual(engine.meta["parallel"], 1)
        other = ModelRef("S2", "S", "Q4", None, self.gguf, None, None)
        self.assertIsNone(engine.gpu_fraction(other))

    def test_meta_records_kv_unified_and_the_effective_context(self):
        engine = self.engine()
        engine.load(self.model, 4096, parallel=2)
        self.assertIs(engine.meta["kv_unified"], False)
        self.assertEqual(engine.meta["ctx"], 4096)
        self.assertEqual(engine.meta["ctx_arg"], 8192)  # the -c given to llama-server

    def test_version_is_not_stale_across_engines(self):
        version_file = self.dir / "version.txt"
        self.setenv(FAKE_VERSION=str(version_file))
        version_file.write_text("version: 0.7.0 (build 11000, commit aaaa111)\n",
                                encoding="utf-8")
        self.assertEqual(self.engine().version(), "b11000 (aaaa111)")
        version_file.write_text("version: 0.8.0 (build 12061, commit d02b4c0d6)\n",
                                encoding="utf-8")
        self.assertEqual(self.engine().version(), "b12061 (d02b4c0d6)")  # same log_dir

    def test_every_start_has_its_own_log_so_metrics_are_not_summed(self):
        engine = self.engine()
        port = proc.free_port()
        with mock.patch("bench.engines.llamacpp.pick_port", return_value=port):
            engine.load(self.model, 8192)
            first = engine._log_path
            engine.load(self.model, 8192)  # the same port again
        self.assertNotEqual(engine._log_path, first)
        self.assertEqual(engine.kv_buffer_mb(), 256.0)  # not 512
        self.assertEqual(engine.rs_buffer_mb(), 50.25)
        self.assertEqual(parse_offloaded_layers(engine._log_text()), (33, 33))

    def test_a_log_left_at_the_old_name_is_not_read(self):
        engine = self.engine()
        port = proc.free_port()
        stale = self.dir / "logs" / f"llamacpp-{port}.log"
        stale.parent.mkdir(parents=True)
        stale.write_text("llama_kv_cache: CUDA0 KV buffer size = 1000.00 MiB\n",
                         encoding="utf-8")
        with mock.patch("bench.engines.llamacpp.pick_port", return_value=port):
            engine.load(self.model, 8192)
        self.assertEqual(engine.kv_buffer_mb(), 256.0)

    def test_fallback_restart_gets_its_own_log(self):
        self.setenv(FAKE_MODE="fa_unsupported")
        engine = self.engine()
        engine.load(self.model, 8192)
        self.assertEqual(len({o.log_path for o in self.created}), 2)
        self.assertEqual(engine.kv_buffer_mb(), 256.0)  # only the second start counts

    def test_error_messages_do_not_contain_the_model_path(self):
        private = self.dir / "private"
        private.mkdir()
        gguf = private / "secret-model.gguf"
        gguf.write_bytes(b"GGUF")
        model = ModelRef("S1", "S", "Q4", None, gguf, None, None)
        self.setenv(FAKE_MODE="oom")
        engine = self.engine()
        with self.assertRaises(EngineError) as ctx:
            engine.load(model, 8192)
        message = str(ctx.exception)
        self.assertNotIn(str(gguf), message)
        self.assertNotIn(str(private), message)
        self.assertIn("<model>", message)

    def test_health_503_while_loading_is_waited_out(self):
        health = self.dir / "health-hits"
        self.setenv(FAKE_MODE="slow_health", FAKE_HEALTH=str(health))
        engine = self.engine()
        self.assertGreater(engine.load(self.model, 2048), 0)
        self.assertEqual(len(self.launches()), 1)  # no restart for a 503
        self.assertEqual(health.read_text(), "4")  # three 503, then ok
        self.assertTrue(engine._owned.running)

    def test_flash_attention_fallback_restarts_once_with_fa_off(self):
        self.setenv(FAKE_MODE="fa_unsupported")
        engine = self.engine()
        seconds = engine.load(self.model, 8192)
        first, second = self.launches()
        self.assertEqual(self.opt(first, "-fa"), "on")
        self.assertEqual(self.opt(second, "-fa"), "off")
        self.assertNotEqual(self.opt(first, "--port"), "")
        self.assertEqual(self.opt(first, "-c"), self.opt(second, "-c"))
        self.assertEqual(engine.meta["flash_attn"], "off")
        self.assertIs(engine.meta["flash_attn_fallback"], True)
        self.assertIn("flash_attn", engine.meta["flash_attn_fallback_reason"])
        self.assertNotIn(str(self.gguf), json.dumps(engine.meta))
        self.assertGreater(seconds, 0)
        self.assertTrue(engine._owned.running)
        self.assertEqual(len(self.created), 2)
        self.assertFalse(self.created[0].running)  # the failed first attempt is gone

    def test_no_fallback_when_the_failure_is_not_about_flash_attention(self):
        self.setenv(FAKE_MODE="oom")
        engine = self.engine()
        with self.assertRaises(EngineError) as ctx:
            engine.load(self.model, 8192)
        self.assertIn("exited", str(ctx.exception))
        self.assertEqual(len(self.launches()), 1)
        self.assertIs(engine.meta["flash_attn_fallback"], False)

    def test_fallback_that_fails_too_is_an_error_not_a_loop(self):
        self.setenv(FAKE_MODE="always_fa_error")
        engine = self.engine()
        with self.assertRaises(EngineError):
            engine.load(self.model, 8192)
        self.assertEqual(len(self.launches()), 2)
        self.assertTrue(all(not o.running for o in self.created))
        self.assertIsNone(engine.base_url)

    def test_unload_stop_and_reload(self):
        engine = self.engine()
        engine.load(self.model, 2048)
        first = self.created[-1]
        port = int(self.opt(self.launches()[0], "--port"))
        engine.unload(self.model)
        self.assertFalse(first.running)
        with self.assertRaises(OSError):
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
        with self.assertRaises(EngineError):
            engine.complete(self.model, "Hi", Sampling())
        engine.load(self.model, 4096)  # a fresh server for every load
        self.assertEqual(len(self.launches()), 2)
        self.assertNotEqual(self.created[-1].pid, first.pid)
        engine.stop()
        engine.stop()
        self.assertFalse(self.created[-1].running)
        self.assertIsNone(engine.base_url)

    def test_only_our_own_process_is_touched(self):
        bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        self.addCleanup(bystander.wait)
        self.addCleanup(bystander.kill)
        engine = self.engine()
        engine.load(self.model, 2048)
        engine.stop()
        engine.start()
        engine.load(self.model, 2048)
        engine.unload(self.model)
        self.assertIsNone(bystander.poll())

    def test_load_needs_start_and_a_model_file(self):
        engine = LlamaCppEngine(Config(), log_dir=self.dir / "logs",
                                binaries=(self.server_bin, self.bench_bin))
        with self.assertRaises(EngineError):
            engine.load(self.model, 2048)
        engine.start()
        for bad in (ModelRef("x", "S", "Q4", "tag", None, None, None),
                    ModelRef("x", "S", "Q4", None, self.dir / "missing.gguf", None, None)):
            with self.subTest(gguf=bad.gguf_path), self.assertRaises(EngineError):
                engine.load(bad, 2048)
        self.assertEqual(self.launches(), [])

    def test_missing_binaries(self):
        cfg = Config(llama_server=self.dir / "nope")
        engine = LlamaCppEngine(cfg, log_dir=self.dir / "logs")
        with self.assertRaises(EngineError):
            engine.start()

    def test_startup_that_never_finishes_fails_within_the_deadline(self):
        self.setenv(FAKE_MODE="hang")
        engine = LlamaCppEngine(Config(), log_dir=self.dir / "logs", start_timeout=1.0,
                                binaries=(self.server_bin, self.bench_bin))
        engine.start()
        with self.assertRaises(EngineError) as ctx:
            engine.load(self.model, 2048)
        self.assertIn("not ready", str(ctx.exception))
        self.assertFalse(self.created[0].running)
        self.addCleanup(engine.stop)

    def test_keyboard_interrupt_during_startup_stops_the_child(self):
        engine = self.engine()
        with mock.patch.object(proc.Owned, "wait_until", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                engine.load(self.model, 2048)
        self.assertEqual(len(self.created), 1)
        self.assertFalse(self.created[0].running)
        self.assertIsNone(engine._owned)
        self.assertIsNone(engine.base_url)

    def test_log_directory_that_cannot_be_created_is_an_engine_error(self):
        blocker = self.dir / "blocker"
        blocker.write_text("a file", encoding="utf-8")
        engine = LlamaCppEngine(Config(), log_dir=blocker / "logs",
                                binaries=(self.server_bin, self.bench_bin))
        engine.start()
        with self.assertRaises(EngineError):
            engine.load(self.model, 2048)
        self.assertIsNone(engine._owned)

    def test_count_tokens_loads_the_model_when_needed_and_caches(self):
        engine = self.engine()
        self.assertEqual(engine.count_tokens(self.model, "one two three"), 3)
        self.assertEqual(len(self.launches()), 1)
        self.assertEqual(self.opt(self.launches()[0], "-c"), "2048")
        self.assertEqual(engine.count_tokens(self.model, "one two three"), 3)
        self.assertEqual(len(self.created), 1)

    def test_the_environment_is_not_changed(self):
        before = dict(os.environ)
        engine = self.engine()
        engine.load(self.model, 2048)
        engine.stop()
        self.assertEqual(dict(os.environ), before)


# -- HTTP calls against a fake server ------------------------------------------------------

class HttpCase(TempCase):
    def engine_for(self, server: FakeServer) -> LlamaCppEngine:
        engine = LlamaCppEngine(Config(), log_dir=self.dir, binaries=(Path("s"), Path("b")))
        engine.start()
        engine.base_url = server.url
        engine._model_id = MODEL.id  # as if load() had run
        self.addCleanup(engine.stop)
        return engine

    @staticmethod
    def body(server: FakeServer, path: str) -> dict:
        return [b for _, p, b in server.requests if p == path][-1]


class CompleteTest(HttpCase):
    def stream(self):
        return text("completion-stream.txt").replace("\n\n", "\n\n").splitlines(keepends=True)

    def test_real_stream_fixture(self):
        with FakeServer({"/completion": self.stream()}) as server:
            result = self.engine_for(server).complete(MODEL, "Count from one to ten:",
                                                      Sampling(max_tokens=24))
        self.assertIsInstance(result, Completion)
        self.assertTrue(result.text.startswith(" 1, 2,"))
        t = result.timing
        self.assertEqual((t.prompt_tokens, t.cached_tokens, t.gen_tokens), (6, 0, 24))
        self.assertAlmostEqual(t.prompt_tps, 6 / 0.031867)
        self.assertAlmostEqual(t.gen_tps, 24 / 0.132236)
        self.assertIsNone(t.load_s)
        self.assertGreater(t.ttft_s, 0)
        # The echoed prompt and the settings are not kept in the raw result.
        self.assertNotIn("prompt", result.raw)
        self.assertNotIn("generation_settings", result.raw)
        self.assertNotIn("model", result.raw)  # holds the local GGUF path
        self.assertEqual(result.raw["timings"]["prompt_n"], 6)

    def test_request_body(self):
        with FakeServer({"/completion": self.stream()}) as server:
            self.engine_for(server).complete(
                MODEL, "Hi", Sampling(temperature=0.2, top_p=0.9, top_k=40, seed=7,
                                      max_tokens=64))
            body = self.body(server, "/completion")
        self.assertEqual(body["prompt"], "Hi")
        self.assertIs(body["stream"], True)
        self.assertEqual((body["temperature"], body["top_p"], body["top_k"], body["seed"],
                          body["n_predict"]), (0.2, 0.9, 40, 7, 64))

    def test_top_k_omitted_when_none(self):
        with FakeServer({"/completion": self.stream()}) as server:
            self.engine_for(server).complete(MODEL, "Hi", Sampling())
            self.assertNotIn("top_k", self.body(server, "/completion"))

    def test_warm_prompt_uses_the_total_and_evaluated_parts(self):
        warm = fixture("completion-warm.json")
        chunks = ['data: {"content":"x","stop":false}\n\n',
                  "data: " + json.dumps({**warm, "content": "", "stop": True}) + "\n\n"]
        with FakeServer({"/completion": chunks}) as server:
            t = self.engine_for(server).complete(MODEL, "Hi", Sampling()).timing
        self.assertEqual((t.prompt_tokens, t.cached_tokens), (601, 597))
        self.assertAlmostEqual(t.prompt_tps, 4 / 0.029288)

    def test_ttft_is_time_to_first_non_empty_chunk(self):
        final = {"content": "", "stop": True,
                 "timings": {"prompt_n": 1, "prompt_ms": 1.0, "predicted_n": 1,
                             "predicted_ms": 1.0, "cache_n": 0}}
        chunks = [(0, 'data: {"content":"","stop":false}\n\n'),
                  (0.3, 'data: {"content":"Hi","stop":false}\n\n'),
                  (1.0, "data: " + json.dumps(final) + "\n\n")]
        with FakeServer({"/completion": chunks}) as server:
            engine = self.engine_for(server)
            started = time.perf_counter()
            t = engine.complete(MODEL, "Hi", Sampling()).timing
            total = time.perf_counter() - started
        self.assertGreaterEqual(t.ttft_s, 0.29)
        self.assertLess(t.ttft_s, total - 0.5)  # clearly before the last chunk

    def test_error_and_truncated_streams_raise(self):
        cases = {"error": ['data: {"content":"a","stop":false}\n\n',
                           'error: {"code":500,"message":"boom"}\n\n'],
                 "truncated": ['data: {"content":"a","stop":false}\n\n'],
                 "no timings": ['data: {"content":"a","stop":true}\n\n'],
                 "not json": ["data: {oops\n\n"]}
        for name, chunks in cases.items():
            with self.subTest(name=name), FakeServer({"/completion": chunks}) as server:
                with self.assertRaises(EngineError):
                    self.engine_for(server).complete(MODEL, "Hi", Sampling())

    def test_http_error_raises_and_unloaded_model_is_refused(self):
        with FakeServer({}) as server:
            engine = self.engine_for(server)
            with self.assertRaises(EngineError):
                engine.complete(MODEL, "Hi", Sampling())
            other = ModelRef("S2", "S", "Q4", None, None, None, None)
            with self.assertRaises(EngineError):
                engine.complete(other, "Hi", Sampling())


class ChatTest(HttpCase):
    def test_chat_tool_calls_arguments_become_a_dict(self):
        reply = fixture("chat-tools.json")
        with FakeServer({"/v1/chat/completions": reply}) as server:
            tools = [{"type": "function", "function": {"name": "get_weather"}}]
            result = self.engine_for(server).chat(
                MODEL, [{"role": "user", "content": "Weather in Berlin?"}], Sampling(),
                tools=tools)
            body = self.body(server, "/v1/chat/completions")
        self.assertIsInstance(result, ChatResult)
        self.assertEqual(result.text, "")
        self.assertEqual(result.tool_calls, [{"name": "get_weather",
                                              "arguments": {"city": "Berlin"}}])
        self.assertEqual(body["tools"], tools)
        self.assertIs(body["stream"], False)
        self.assertEqual(body["messages"][0]["content"], "Weather in Berlin?")
        self.assertEqual(body["seed"], 42)
        self.assertEqual(body["max_tokens"], 256)
        t = result.timing
        self.assertEqual((t.prompt_tokens, t.cached_tokens, t.gen_tokens), (282, 0, 26))
        self.assertAlmostEqual(t.prompt_tps, 282 / 0.080484)
        # No load in llama.cpp: first-token latency is the server's prompt time.
        self.assertAlmostEqual(t.ttft_s, 0.080484)
        self.assertIsNone(t.load_s)

    def test_raw_reply_has_no_model_path(self):
        with FakeServer({"/v1/chat/completions": fixture("chat-json.json")}) as server:
            result = self.engine_for(server).chat(MODEL, [], Sampling())
        self.assertNotIn("model", result.raw)
        self.assertNotIn("models", json.dumps(result.raw))
        self.assertIn("choices", result.raw)

    def test_tool_arguments_must_decode_to_an_object(self):
        def reply(arguments):
            call = {"type": "function", "function": {"name": "f", "arguments": arguments}}
            return {"choices": [{"message": {"content": "", "tool_calls": [call]}}],
                    "timings": {}}
        for arguments in ("[1, 2]", '"text"', "5", "null", [1, 2], 7):
            with self.subTest(arguments=arguments), \
                    FakeServer({"/v1/chat/completions": reply(arguments)}) as server:
                with self.assertRaises(EngineError):
                    self.engine_for(server).chat(MODEL, [], Sampling())
        for arguments, expected in (("", {}), ("{}", {}), (None, {}), ('{"a": 1}', {"a": 1})):
            with self.subTest(arguments=arguments), \
                    FakeServer({"/v1/chat/completions": reply(arguments)}) as server:
                result = self.engine_for(server).chat(MODEL, [], Sampling())
                self.assertEqual(result.tool_calls[0]["arguments"], expected)

    def test_arguments_given_as_an_object_and_bad_json(self):
        call = {"type": "function", "function": {"name": "f", "arguments": {"a": 1}}}
        reply = {"choices": [{"message": {"content": "", "tool_calls": [call]}}],
                 "timings": {}}
        with FakeServer({"/v1/chat/completions": reply}) as server:
            result = self.engine_for(server).chat(MODEL, [], Sampling())
        self.assertEqual(result.tool_calls[0]["arguments"], {"a": 1})
        call["function"]["arguments"] = "{not json"
        with FakeServer({"/v1/chat/completions": reply}) as server:
            with self.assertRaises(EngineError):
                self.engine_for(server).chat(MODEL, [], Sampling())

    def test_json_schema_response_format(self):
        schema = {"type": "object", "properties": {"capital": {"type": "string"}}}
        with FakeServer({"/v1/chat/completions": fixture("chat-json.json")}) as server:
            result = self.engine_for(server).chat(MODEL, [], Sampling(), json_schema=schema)
            body = self.body(server, "/v1/chat/completions")
        self.assertEqual(json.loads(result.text), {"capital": "Paris"})
        self.assertEqual(body["response_format"]["type"], "json_schema")
        self.assertEqual(body["response_format"]["json_schema"]["schema"], schema)
        self.assertIn("name", body["response_format"]["json_schema"])
        self.assertNotIn("tools", body)

    def test_thinking_flag_and_top_k(self):
        with FakeServer({"/v1/chat/completions": fixture("chat-json.json")}) as server:
            engine = self.engine_for(server)
            engine.chat(MODEL, [], Sampling())
            off = self.body(server, "/v1/chat/completions")
            engine.chat(MODEL, [], Sampling(top_k=20), thinking=True)
            on = self.body(server, "/v1/chat/completions")
        self.assertEqual(off["chat_template_kwargs"], {"enable_thinking": False})
        self.assertNotIn("top_k", off)
        self.assertEqual(on["chat_template_kwargs"], {"enable_thinking": True})
        self.assertEqual(on["top_k"], 20)

    def test_usage_is_the_fallback_without_timings(self):
        reply = {"choices": [{"message": {"content": "ok"}}],
                 "usage": {"prompt_tokens": 30, "completion_tokens": 5,
                           "prompt_tokens_details": {"cached_tokens": 10}}}
        with FakeServer({"/v1/chat/completions": reply}) as server:
            t = self.engine_for(server).chat(MODEL, [], Sampling()).timing
        self.assertEqual((t.prompt_tokens, t.cached_tokens, t.gen_tokens), (30, 10, 5))
        self.assertIsNone(t.prompt_tps)

    def test_error_replies(self):
        with FakeServer({"/v1/chat/completions": {"error": {"message": "bad"}}}) as server:
            with self.assertRaises(EngineError) as ctx:
                self.engine_for(server).chat(MODEL, [], Sampling())
        self.assertIn("bad", str(ctx.exception))
        with FakeServer({"/v1/chat/completions": {"choices": []}}) as server:
            with self.assertRaises(EngineError):
                self.engine_for(server).chat(MODEL, [], Sampling())


class CountTokensTest(HttpCase):
    def test_count_tokens_uses_tokenize_without_inference(self):
        with FakeServer({"/tokenize": fixture("tokenize.json")}) as server:
            engine = self.engine_for(server)
            count = engine.count_tokens(MODEL, "The quick brown fox")
            body = self.body(server, "/tokenize")
            paths = [p for _, p, _ in server.requests]
        self.assertEqual(count, 40)
        self.assertEqual(body["content"], "The quick brown fox")
        self.assertIs(body["add_special"], True)  # what /completion would tokenize
        self.assertEqual(paths, ["/tokenize"])

    def test_count_tokens_is_cached_per_model_and_text(self):
        with FakeServer({"/tokenize": {"tokens": [1, 2]}}) as server:
            engine = self.engine_for(server)
            engine.count_tokens(MODEL, "a")
            engine.count_tokens(MODEL, "a")
            engine.count_tokens(MODEL, "b")
            self.assertEqual(len(server.requests), 2)

    def test_bad_tokenize_replies(self):
        for reply in ({"tokens": "x"}, {}, {"error": "no"}):
            with self.subTest(reply=reply), FakeServer({"/tokenize": reply}) as server:
                with self.assertRaises(EngineError):
                    self.engine_for(server).count_tokens(MODEL, "a")


class RobustnessTest(HttpCase):
    def test_no_proxy_is_used(self):
        engine = LlamaCppEngine(Config(), log_dir=self.dir, binaries=(Path("s"), Path("b")))
        with mock.patch.dict(os.environ, {"http_proxy": "http://127.0.0.1:9",
                                          "HTTP_PROXY": "http://127.0.0.1:9"}):
            with FakeServer({"/tokenize": {"tokens": [1]}}) as server:
                engine.base_url = server.url
                engine._model_id = MODEL.id
                self.assertEqual(engine.count_tokens(MODEL, "a"), 1)

    def test_unreachable_server_is_an_engine_error(self):
        engine = LlamaCppEngine(Config(), log_dir=self.dir, binaries=(Path("s"), Path("b")),
                                quick_timeout=1)
        engine.base_url = f"http://127.0.0.1:{proc.free_port()}"
        engine._model_id = MODEL.id
        with self.assertRaises(EngineError):
            engine.count_tokens(MODEL, "a")

    def test_metrics_are_unknown_before_any_load(self):
        engine = LlamaCppEngine(Config(), log_dir=self.dir, binaries=(Path("s"), Path("b")))
        self.assertIsNone(engine.kv_buffer_mb())
        self.assertIsNone(engine.rs_buffer_mb())
        self.assertIsNone(engine.backend())
        self.assertIsNone(engine.gpu_fraction(MODEL))
        self.assertEqual(engine.name, "llamacpp")


if __name__ == "__main__":
    unittest.main()
