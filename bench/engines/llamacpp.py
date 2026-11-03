"""llama.cpp adapter: `llama-server` (HTTP) and `llama-bench`, standard library only.

Verified against build b12061 (see tests/fixtures/llamacpp, captured on a real machine).

Process ownership: the adapter always starts its own `llama-server` on a free port bound to
127.0.0.1 (never assuming 8080 is free) and stops only that process tree. There is no way to
attach to a foreign server. A model is loaded by starting the server with that model, so
`load` always starts a fresh server and `unload`/`stop` end it.

Launch flags: `-m <gguf> -ngl 99 -fa on -c <ctx*parallel> -np <parallel> --port <p>
--host 127.0.0.1 --jinja -fit off -cram 0 -lv 4`, then `--llama-args` (last one wins).

* `-fit off`: llama.cpp must never silently shrink the context or the offload to fit.
* `-cram 0`: no 8 GiB host-RAM prompt cache.
* `-lv 4`: without it the KV/RS buffer lines are not logged and `kv_buffer_mb` has nothing.
* `-c` is the total over all slots, so it is multiplied by `parallel`: every slot gets `ctx`
  tokens, like the context allocation of the Ollama adapter.

Flash attention: started with `-fa on`. If the server exits during startup with a log that
blames flash attention (for example on Pascal GPUs) it is restarted once with `-fa off`;
`meta` records that. Any other startup failure is an error. A `-fa` in `--llama-args` is
the user's choice: it is used as is and never retried.

Timing semantics differ from Ollama: `timings.prompt_n` EXCLUDES cached tokens, so the total
prompt length is `prompt_n + cache_n`, and the prompt speed is `prompt_n / prompt_ms`.
llama.cpp has no load time in a reply (the model is resident), so `Timing.load_s` is None;
`load()` returns the time from process start to `/health` ok instead.

Memory: `kv_buffer_mb` sums the `KV buffer size` lines of all devices. The recurrent state of
hybrid models (`RS buffer size`) is exposed separately by `rs_buffer_mb`, never added.

Token counting uses `/tokenize` (no inference). It needs the model: if it is not loaded,
`count_tokens` loads it with a small context first.

Timeouts: quick calls (health, version) use a short timeout, only inference uses the long one.

Known limits, on purpose:

* `cache_prompt` stays on, so repeating an identical prompt is warm (`cached_tokens` > 0 and a
  tiny `prompt_tokens - cached_tokens`). A speed suite that wants cold prompt numbers needs
  its own strategy (distinct prompts or a fresh `load`).
* `--gpu` is not forwarded to llama-server or llama-bench; llama.cpp picks its own devices.
* Every server start writes its own log file (port and time in the name), so the metrics
  parsed from the log (KV, RS, backend, offload, kv_unified) always belong to the last start.
* Model paths are replaced by `<model>` in error messages and `meta`, and the reply's `model`
  field (the local GGUF path) is dropped from `raw`.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import shlex
import time
import urllib.error
import urllib.request
from pathlib import Path

from bench import proc
from bench.config import Config
from bench.engines.base import (ChatResult, Completion, EngineError, ModelRef, Sampling,
                                Timing)
from bench.engines.release import ReleaseError, ensure_llamacpp

DEFAULT_PORT = 8080
MS = 1000.0
START_TIMEOUT_S = 900
REQUEST_TIMEOUT_S = 1800
QUICK_TIMEOUT_S = 5
BENCH_TIMEOUT_S = 3600
VERSION_TIMEOUT_S = 30
MIN_COUNT_CTX = 2048
LOG_READ_BYTES = 8 << 20
DEFAULT_LOG_DIR = Path(".bench") / "logs"
DROPPED_RAW_KEYS = ("prompt", "generation_settings", "tokens", "model")

KV_RE = re.compile(r"\bKV buffer size\s*=\s*([\d.]+)\s*MiB")
RS_RE = re.compile(r"\bRS buffer size\s*=\s*([\d.]+)\s*MiB")
OFFLOAD_RE = re.compile(r"offloaded\s+(\d+)/(\d+)\s+layers to GPU")
DEVICE_RE = re.compile(r"using device\s+([A-Za-z_]+)")
KV_UNIFIED_RE = re.compile(r"kv_unified\s*=\s*'?(true|false)'?")
VERSION_LINE_RE = re.compile(r"^version:\s*(.+)$", re.MULTILINE)
BUILD_RE = re.compile(r"build\s+(\d+).*?commit\s+(\w+)")
OLD_BUILD_RE = re.compile(r"^(\d+)\s*\((\w+)\)")
FA_FLAGS = ("-fa", "--flash-attn")
# llama-server reads LLAMA_ARG_* variables as if they were command line options (cache
# types, context size, fit mode, ...). A user environment must not change what we measure,
# so the children never see them; the removed names are recorded in `meta`.
LLAMA_ARG_ENV = ("LLAMA_ARG_*",)
# Words that, in a line that also says "flash", mean flash attention is the problem.
# The wording of llama.cpp's messages is not verified on a failing GPU (see the report).
FA_PROBLEM_WORDS = ("not supported", "unsupported", "missing support", "requested but",
                    "required but", "not available", "cannot", "can't", "failed", "error")
BACKEND_PREFIXES = (("cuda", "cuda"), ("vulkan", "vulkan"), ("mtl", "metal"),
                    ("metal", "metal"), ("rocm", "rocm"), ("hip", "rocm"), ("cpu", "cpu"))


# -- parsers -------------------------------------------------------------------------------

def parse_kv_buffer_mb(log_text: str) -> float | None:
    """Sum of all `KV buffer size` lines (every device); None if there are none."""
    sizes = [float(v) for v in KV_RE.findall(log_text)]
    return sum(sizes) if sizes else None


def parse_rs_buffer_mb(log_text: str) -> float | None:
    """Sum of the recurrent state buffers of hybrid models; None if there are none."""
    sizes = [float(v) for v in RS_RE.findall(log_text)]
    return sum(sizes) if sizes else None


def parse_offloaded_layers(log_text: str) -> tuple[int, int] | None:
    """(layers on the GPU, layers in total) from the last `offloaded N/M layers` line."""
    found = OFFLOAD_RE.findall(log_text)
    return (int(found[-1][0]), int(found[-1][1])) if found else None


def parse_backend(log_text: str) -> str | None:
    """cuda, vulkan, metal, rocm or cpu from the devices the model was placed on."""
    match = DEVICE_RE.search(log_text)
    if match:
        name = match.group(1).lower()
        for prefix, backend in BACKEND_PREFIXES:
            if name.startswith(prefix):
                return backend
        return name
    layers = parse_offloaded_layers(log_text)
    if layers is not None and layers[0] == 0:
        return "cpu"
    return None


def parse_version(text: str) -> str | None:
    """`b12061 (d02b4c0d6)` from the output of `llama-server --version`."""
    matches = VERSION_LINE_RE.findall(text)  # the last one: a log may hold several runs
    if not matches:
        return None
    line = matches[-1].strip()
    for pattern in (BUILD_RE, OLD_BUILD_RE):
        found = pattern.search(line)
        if found:
            return f"b{found.group(1)} ({found.group(2)})"
    return line


def removed_llama_arg_env() -> list[str]:
    """Names of the LLAMA_ARG_* variables in our environment (not passed to children)."""
    return sorted(name for name in os.environ if proc.dropped(name, LLAMA_ARG_ENV))


def parse_kv_unified(log_text: str) -> bool | None:
    """Whether all slots share one KV buffer (`kv_unified = 'false'`), from the last mention."""
    found = KV_UNIFIED_RE.findall(log_text)
    return found[-1] == "true" if found else None


def hide_model_path(text: str, gguf: Path | str | None) -> str:
    """Replace the model file and its folder by `<model>` (privacy: no local paths)."""
    if gguf is None:
        return text
    path = Path(gguf)
    forms = {str(path), path.as_posix(), str(path.parent), path.parent.as_posix()}
    for form in sorted(forms, key=len, reverse=True):
        if len(form) > 1:
            text = text.replace(form, "<model>")
    return text


def flash_attn_failure(log_text: str) -> str | None:
    """The log line that blames flash attention for a failed start, or None.

    The context dump prints `flash_attn = enabled` on every start (also when memory ran out),
    so a line must name flash attention AND a problem word to count.
    """
    for line in log_text.splitlines():
        low = line.lower()
        if "flash" in low and any(word in low for word in FA_PROBLEM_WORDS):
            return line.strip()[:200]
    return None


def parse_llama_bench(output: str) -> list[dict]:
    """The JSON array `llama-bench -o json` prints: one object per pp/tg test."""
    try:
        rows = json.loads(output.lstrip("\ufeff"))
    except ValueError:
        raise EngineError("llama-bench output is not JSON") from None
    if not isinstance(rows, list) or not rows or not all(isinstance(r, dict) for r in rows):
        raise EngineError("llama-bench output is not a JSON array of results")
    return rows


def run_llama_bench(bench_bin: Path, model: ModelRef, args: list[str], *,
                    log_dir: Path | None = None, timeout: float = BENCH_TIMEOUT_S) -> list[dict]:
    """Run `llama-bench -m <gguf> -o json <args>` and return its result rows.

    `args` are the llama-bench options of the caller (for example `-p 512 -n 64 -d 4096
    -r 3 -ngl 99 -fa 1`). stdout (the JSON) and stderr (logs) are kept apart. The local
    model path (`model_filename`) is removed from the rows. The process is ours and is
    stopped on timeout and on any error.
    """
    if model.gguf_path is None:
        raise EngineError(f"model {model.id} has no GGUF file for llama-bench")
    bench_bin = Path(bench_bin)
    if not bench_bin.is_file():
        raise EngineError(f"llama-bench executable not found: {bench_bin}")
    directory = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
    stamp = f"{os.getpid()}-{time.time_ns()}"  # unique: logs are never reused
    out_path = directory / f"llama-bench-{stamp}.json"
    log_path = directory / f"llama-bench-{stamp}.log"
    owned = proc.Owned([bench_bin, "-m", model.gguf_path, "-o", "json", *args],
                       log_path=log_path, stdout_path=out_path, drop_env=LLAMA_ARG_ENV)
    try:
        try:
            owned.start()
            code = owned.wait(timeout)
        except proc.ProcError as err:
            if owned.running:
                raise EngineError(f"llama-bench did not finish after {timeout:g} s") \
                    from None
            raise EngineError("cannot run llama-bench: "
                              f"{hide_model_path(str(err), model.gguf_path)}") from None
        if code != 0:
            raise EngineError(f"llama-bench exited with code {code}:\n"
                              f"{hide_model_path(owned.log_tail(), model.gguf_path)}")
    finally:
        owned.stop()
    try:
        output = out_path.read_text(encoding="utf-8-sig")
    except OSError as err:
        raise EngineError(f"cannot read llama-bench output: {err.strerror or err}") from None
    rows = parse_llama_bench(output)
    for row in rows:
        row.pop("model_filename", None)
    return rows


def _user_args(text: str) -> list[str]:
    try:
        tokens = shlex.split(text, posix=os.name != "nt")
    except ValueError as err:
        raise EngineError(f"cannot read --llama-args: {err}") from None
    result = []
    for token in tokens:  # non-POSIX splitting keeps the quotes
        if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
            token = token[1:-1]
        result.append(token)
    return result


def _sets_flash_attn(args: list[str]) -> bool:
    return any(a in FA_FLAGS or a.startswith("--flash-attn=") for a in args)


def pick_port() -> int:
    """A free port that is not llama-server's default port."""
    for _ in range(20):
        port = proc.free_port()
        if port != DEFAULT_PORT:
            return port
    raise EngineError("no free port found")


class _StartFailed(EngineError):
    """llama-server did not become ready; carries what the fallback logic needs."""

    def __init__(self, message: str, exited: bool, log: str):
        super().__init__(message)
        self.exited = exited
        self.log = log


class LlamaCppEngine:
    name = "llamacpp"

    def __init__(self, cfg: Config, log_dir: Path | None = None,
                 timeout: float = REQUEST_TIMEOUT_S, quick_timeout: float = QUICK_TIMEOUT_S,
                 start_timeout: float = START_TIMEOUT_S,
                 binaries: tuple[Path, Path] | None = None):
        self.cfg = cfg
        self.log_dir = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
        self.timeout = timeout
        self.quick_timeout = quick_timeout
        self.start_timeout = start_timeout
        self.base_url: str | None = None
        # No proxy: the server is local.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self._binaries = binaries
        self._started = False
        self._owned: proc.Owned | None = None
        self._log_path: Path | None = None
        self._model_id: str | None = None
        self._version: str | None = None
        self._meta: dict = {}
        self._counts: dict[tuple[str, str], int] = {}

    # -- lifecycle ---------------------------------------------------------------------

    @property
    def meta(self) -> dict:
        """What happened during the last load (flash attention choice and fallback, and
        the names of the LLAMA_ARG_* variables kept away from the server)."""
        return dict(self._meta)

    @property
    def bench_bin(self) -> Path | None:
        """Path of llama-bench once start() has found the binaries, else None."""
        return self._binaries[1] if self._binaries else None

    def start(self) -> None:
        """Find (or download, unless disabled) the binaries. No server runs yet."""
        if self._binaries is None:
            try:
                self._binaries = ensure_llamacpp(self.cfg)
            except ReleaseError as err:
                raise EngineError(str(err)) from None
        self._started = True

    def stop(self) -> None:
        """Stop the server this engine started; idempotent. Call start() to use it again."""
        self._release()
        self._started = False

    def _release(self) -> None:
        """Stop our server (if any) and forget it and its address."""
        owned, self._owned = self._owned, None
        self.base_url = None
        self._model_id = None
        if owned is not None:
            owned.stop()

    def _require_started(self) -> tuple[Path, Path]:
        if not self._started or self._binaries is None:
            raise EngineError("engine not started; call start() first")
        return self._binaries

    # -- loading -----------------------------------------------------------------------

    def load(self, model: ModelRef, ctx: int, parallel: int = 1) -> float:
        """Start a server for `model` with context `ctx` per slot; returns seconds to ready."""
        server_bin, _ = self._require_started()
        if model.gguf_path is None:
            raise EngineError(f"model {model.id} has no GGUF file for llama.cpp")
        gguf = Path(model.gguf_path)
        if not gguf.is_file():
            raise EngineError(f"GGUF file not found for model {model.id}")
        user_args = _user_args(self.cfg.llama_args)
        user_fa = _sets_flash_attn(user_args)
        self._release()
        self._meta = {"flash_attn": "user" if user_fa else "on",
                      "flash_attn_fallback": False, "ctx": ctx, "parallel": parallel,
                      "ctx_arg": ctx * parallel,
                      "llama_arg_env_removed": removed_llama_arg_env()}
        fa = None if user_fa else "on"
        try:
            seconds = self._start_server(server_bin, gguf, ctx, parallel, fa, user_args)
        except _StartFailed as err:
            reason = flash_attn_failure(err.log) if fa == "on" and err.exited else None
            if reason is None:
                raise EngineError(str(err)) from None
            self._meta.update({
                "flash_attn": "off", "flash_attn_fallback": True,
                "flash_attn_fallback_reason": self._sanitize(reason, gguf)})
            try:
                seconds = self._start_server(server_bin, gguf, ctx, parallel, "off",
                                             user_args)
            except _StartFailed as again:
                raise EngineError(str(again)) from None  # once only: no loop
        self._model_id = model.id
        self._meta["kv_unified"] = parse_kv_unified(self._log_text())
        return seconds

    @staticmethod
    def _sanitize(line: str, gguf: Path) -> str:
        return hide_model_path(line, gguf)

    def _start_server(self, server_bin: Path, gguf: Path, ctx: int, parallel: int,
                      fa: str | None, user_args: list[str]) -> float:
        port = pick_port()
        argv = [server_bin, "-m", gguf, "-ngl", "99"]
        if fa:
            argv += ["-fa", fa]
        argv += ["-c", str(ctx * parallel), "-np", str(parallel), "--port", str(port),
                 "--host", "127.0.0.1", "--jinja", "-fit", "off", "-cram", "0", "-lv", "4",
                 *user_args]
        # A new file for every start: Owned appends, and old lines must never be counted.
        log_path = self.log_dir / f"llamacpp-{port}-{time.time_ns()}.log"
        owned = proc.Owned(argv, log_path=log_path, drop_env=LLAMA_ARG_ENV)
        # Owned from here on: whatever happens next (Ctrl-C included), only stop() ends it.
        # stop() does nothing if the process was never spawned.
        self._owned = owned
        # Point at the new log before spawning: after a failed spawn the probes must not
        # read the previous start's KV or backend lines.
        self._log_path = log_path
        started = time.perf_counter()
        try:
            owned.start()
            self.base_url = f"http://127.0.0.1:{port}"
            owned.wait_until(self._healthy, self.start_timeout)
            if not owned.running:
                # Our child is gone, so the answer came from another server on that port.
                raise EngineError("llama-server exited right after start; "
                                  "another server may be using the port")
        except BaseException as err:
            exited = owned.returncode is not None
            log = self._log_text()
            self._release()
            if isinstance(err, proc.ProcError):
                message = hide_model_path(f"cannot start llama-server: {err}", gguf)
                raise _StartFailed(message, exited, log) from None
            raise
        return time.perf_counter() - started

    def _healthy(self) -> bool:
        try:
            return self._get("/health").get("status") == "ok"
        except EngineError:
            return False

    def unload(self, model: ModelRef) -> None:
        """Unloading means ending our server; the next load starts a fresh one."""
        if self._model_id in (None, model.id):
            self._release()

    # -- HTTP --------------------------------------------------------------------------

    def _open(self, path: str, payload: dict | None, timeout: float | None = None):
        if not self.base_url:
            raise EngineError("no llama-server running (load a model first)")
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path, data=data,
            headers={"Content-Type": "application/json"} if data else {},
            method="GET" if data is None else "POST")
        try:
            return self._opener.open(request, timeout=timeout or self.timeout)
        except urllib.error.HTTPError as err:
            try:
                body = err.read().decode("utf-8", errors="replace")[:300]
            except (OSError, http.client.HTTPException):  # truncated error body
                body = ""
            finally:
                err.close()
            raise EngineError(f"llama.cpp {path}: HTTP {err.code}: {body}") from None
        except (urllib.error.URLError, OSError, http.client.HTTPException) as err:
            raise EngineError(f"llama.cpp {path}: {getattr(err, 'reason', None) or err!r}") \
                from None

    def _get(self, path: str) -> dict:
        return self._post(path, None, self.quick_timeout)

    def _post(self, path: str, payload: dict | None, timeout: float | None = None) -> dict:
        with self._open(path, payload, timeout) as response:
            try:
                reply = json.loads(response.read())
            except (OSError, ValueError, http.client.HTTPException) as err:
                raise EngineError(f"llama.cpp {path}: unreadable reply: {err!r}") from None
        if not isinstance(reply, dict):
            raise EngineError(f"llama.cpp {path}: unexpected reply {str(reply)[:100]}")
        if reply.get("error"):
            raise EngineError(f"llama.cpp {path}: {self._error_text(reply['error'])}")
        return reply

    @staticmethod
    def _error_text(error) -> str:
        if isinstance(error, dict):
            return str(error.get("message") or error)[:300]
        return str(error)[:300]

    def _require_loaded(self, model: ModelRef) -> None:
        if not self.base_url or self._model_id != model.id:
            raise EngineError(f"model {model.id} is not loaded in llama-server")

    @staticmethod
    def _sampling(sampling: Sampling) -> dict:
        body = {"temperature": sampling.temperature, "top_p": sampling.top_p,
                "seed": sampling.seed}
        if sampling.top_k is not None:
            body["top_k"] = sampling.top_k
        return body

    # -- Engine interface --------------------------------------------------------------

    def version(self) -> str:
        """Build of the llama-server binary, `b12061 (d02b4c0d6)`."""
        server_bin, _ = self._require_started()
        if self._version is None:
            owned = proc.Owned(
                [server_bin, "--version"], drop_env=LLAMA_ARG_ENV,
                log_path=self.log_dir / f"llamacpp-version-{os.getpid()}-{time.time_ns()}.log")
            try:
                try:
                    owned.start()
                    owned.wait(VERSION_TIMEOUT_S)
                except proc.ProcError as err:
                    raise EngineError(f"cannot run llama-server --version: {err}") from None
                output = owned.log_tail(8000)
            finally:
                owned.stop()
            version = parse_version(output)
            if version is None:
                raise EngineError(f"unexpected llama-server --version output: {output[:100]}")
            self._version = version
        return self._version

    def backend(self) -> str | None:
        """GPU backend as logged by the last server we started."""
        return parse_backend(self._log_text())

    def complete(self, model: ModelRef, prompt: str, sampling: Sampling) -> Completion:
        self._require_loaded(model)
        payload = {"prompt": prompt, "n_predict": sampling.max_tokens, "stream": True,
                   "cache_prompt": True, **self._sampling(sampling)}
        parts: list[str] = []
        ttft = None
        final = None
        sent = time.perf_counter()
        with self._open("/completion", payload) as response:
            try:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if line.startswith("error:"):
                        raise EngineError(f"llama.cpp /completion: "
                                          f"{self._error_text(self._json(line[6:]))}")
                    if not line.startswith("data:"):
                        continue  # blank separators, comments, event names
                    data = line[5:].strip()
                    if data == "[DONE]":
                        continue
                    chunk = self._json(data)
                    if not isinstance(chunk, dict):
                        raise EngineError(f"llama.cpp /completion: unexpected chunk "
                                          f"{str(chunk)[:100]}")
                    if chunk.get("error"):
                        raise EngineError(f"llama.cpp /completion: "
                                          f"{self._error_text(chunk['error'])}")
                    if chunk.get("content"):
                        if ttft is None:
                            ttft = time.perf_counter() - sent
                        parts.append(chunk["content"])
                    if chunk.get("stop") is True:
                        final = chunk
                        break
            except (OSError, http.client.HTTPException) as err:
                raise EngineError(f"llama.cpp /completion: broken stream: {err!r}") from None
        if final is None:
            raise EngineError("llama.cpp /completion: stream ended without a final chunk")
        timings = final.get("timings")
        if not isinstance(timings, dict) or not timings:
            raise EngineError("llama.cpp /completion: final chunk has no timings")
        if ttft is None:
            ttft = time.perf_counter() - sent
        raw = {k: v for k, v in final.items() if k not in DROPPED_RAW_KEYS}
        return Completion("".join(parts), self._timing(timings, ttft), raw)

    @staticmethod
    def _json(text: str):
        try:
            return json.loads(text)
        except ValueError:
            raise EngineError(f"llama.cpp: unreadable stream data: {text[:100]!r}") from None

    def chat(self, model: ModelRef, messages: list[dict], sampling: Sampling,
             tools: list[dict] | None = None, json_schema: dict | None = None,
             thinking: bool = False) -> ChatResult:
        self._require_loaded(model)
        payload = {"messages": messages, "max_tokens": sampling.max_tokens, "stream": False,
                   "cache_prompt": True, "chat_template_kwargs": {"enable_thinking": thinking},
                   **self._sampling(sampling)}
        if tools:
            payload["tools"] = tools
        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema", "json_schema": {"name": "response", "schema": json_schema}}
        reply = self._post("/v1/chat/completions", payload)
        choices = reply.get("choices")
        if not isinstance(choices, list) or not choices:
            raise EngineError("llama.cpp /v1/chat/completions: reply has no choices")
        message = choices[0].get("message") or {}
        calls = [self._tool_call(c) for c in message.get("tool_calls") or []]
        timings = reply.get("timings")
        if isinstance(timings, dict) and timings:
            # Not streamed: first token latency is approximated by the server's prompt time.
            timing = self._timing(timings, timings.get("prompt_ms", 0.0) / MS)
        else:
            timing = self._usage_timing(reply.get("usage") or {})
        raw = {k: v for k, v in reply.items() if k != "model"}  # "model" is the local path
        return ChatResult(message.get("content") or "", calls, timing, raw)

    @staticmethod
    def _tool_call(call: dict) -> dict:
        function = call.get("function") or {}
        arguments = function.get("arguments")
        if arguments is None or arguments == "":
            arguments = {}
        if isinstance(arguments, str):  # OpenAI shape: a JSON string (Ollama gives an object)
            try:
                arguments = json.loads(arguments)
            except ValueError:
                raise EngineError(f"tool call arguments are not JSON: {arguments[:100]}") \
                    from None
        if not isinstance(arguments, dict):
            raise EngineError(f"tool call arguments are not an object: {str(arguments)[:100]}")
        return {"name": function.get("name", ""), "arguments": arguments}

    @staticmethod
    def _timing(timings: dict, ttft: float) -> Timing:
        evaluated = timings.get("prompt_n", 0)  # excludes cached tokens (unlike Ollama)
        cached = timings.get("cache_n")
        prompt_ms = timings.get("prompt_ms", 0.0)
        gen_tokens = timings.get("predicted_n", 0)
        gen_ms = timings.get("predicted_ms", 0.0)
        return Timing(
            load_s=None,
            ttft_s=ttft,
            prompt_tokens=evaluated + (cached or 0),
            prompt_tps=evaluated / (prompt_ms / MS) if evaluated > 0 and prompt_ms > 0
            else None,
            gen_tokens=gen_tokens,
            gen_tps=gen_tokens / (gen_ms / MS) if gen_tokens and gen_ms > 0 else None,
            cached_tokens=cached)

    @staticmethod
    def _usage_timing(usage: dict) -> Timing:
        """Token counts only (OpenAI usage: prompt_tokens includes cached ones), no speeds."""
        details = usage.get("prompt_tokens_details") or {}
        return Timing(load_s=None, ttft_s=0.0, prompt_tokens=usage.get("prompt_tokens", 0),
                      prompt_tps=None, gen_tokens=usage.get("completion_tokens", 0),
                      gen_tps=None, cached_tokens=details.get("cached_tokens"))

    def count_tokens(self, model: ModelRef, text: str) -> int:
        """Exact token count from /tokenize (no inference), as /completion would tokenize."""
        key = (model.id, hashlib.sha256(text.encode("utf-8")).hexdigest())
        if key in self._counts:
            return self._counts[key]
        if self._model_id != model.id or not self.base_url:
            self.load(model, MIN_COUNT_CTX)  # tokenizing needs the model, not a large context
        reply = self._post("/tokenize", {"content": text, "add_special": True},
                           self.quick_timeout)
        tokens = reply.get("tokens")
        if not isinstance(tokens, list):
            raise EngineError("llama.cpp /tokenize: reply has no token list")
        self._counts[key] = len(tokens)
        return len(tokens)

    def gpu_fraction(self, model: ModelRef) -> float | None:
        """Share of the model's layers on the GPU (1.0 = all), from the load log."""
        if self._model_id != model.id:
            return None
        layers = parse_offloaded_layers(self._log_text())
        if layers is None or layers[1] <= 0:
            return None
        return min(1.0, layers[0] / layers[1])

    def kv_buffer_mb(self) -> float | None:
        """KV cache size of the last load (sum over devices), from the server log."""
        return parse_kv_buffer_mb(self._log_text())

    def rs_buffer_mb(self) -> float | None:
        """Recurrent state size of hybrid models, kept apart from the KV figure."""
        return parse_rs_buffer_mb(self._log_text())

    # -- server log --------------------------------------------------------------------

    def _log_text(self) -> str:
        if self._log_path is None:
            return ""
        try:
            with self._log_path.open("rb") as handle:
                data = handle.read(LOG_READ_BYTES)
        except OSError:
            return ""
        return data.decode("utf-8", errors="replace")
