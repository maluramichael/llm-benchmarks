"""Ollama adapter (native HTTP API, standard library only).

Process ownership: with `--ollama-url` the adapter only connects and never stops or
signals anything. Without it, it starts its own `ollama serve` on a free port (never
assuming 11434 is free: other servers may already own it) and stops only that process
tree. The environment for the server (OLLAMA_HOST, OLLAMA_NUM_PARALLEL, OLLAMA_MODELS,
OLLAMA_KEEP_ALIVE) is given to the child only.

Field names were checked against Ollama 0.40.13. All durations are nanoseconds.
Speed tests use /api/generate with raw=true (no chat template) and streaming, so the
time to first token can be measured here.

Token counting: Ollama has no tokenizer endpoint, so `count_tokens` sends the text
with raw=true and num_predict=1 and reads the prompt token count. Ollama can answer
part of a prompt from its prompt cache and then reports the evaluated tokens in
`prompt_eval_count` and the cached ones in `prompt_eval_cached_count`; the prompt length
is the sum of both (it equals `prompt_eval_count` on a cold cache). Results are cached
per (model, text hash).
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

from bench import proc
from bench.config import Config
from bench.engines.base import (ChatResult, Completion, EngineError, ModelRef, Sampling,
                                Timing)

DEFAULT_PORT = 11434
NS = 1e9
START_TIMEOUT_S = 90
REQUEST_TIMEOUT_S = 1800
MIN_COUNT_CTX = 2048
KEEP_ALIVE = "30m"
DEFAULT_LOG_DIR = Path(".bench") / "logs"
BACKENDS = {"cuda", "rocm", "metal", "vulkan", "cpu"}

LIBRARY_RE = re.compile(r"\blibrary=(\w+)")
KV_RES = (
    re.compile(r"KV buffer size\s*=\s*([\d.]+)\s*MiB"),  # llama.cpp runner
    re.compile(r'msg="kv cache" device=\S+ size="([\d.]+) MiB"'),  # Ollama engine runner
)


def pick_port() -> int:
    """A free port that is not Ollama's default port."""
    for _ in range(20):
        port = proc.free_port()
        if port != DEFAULT_PORT:
            return port
    raise EngineError("no free port found")


class OllamaEngine:
    name = "ollama"

    def __init__(self, cfg: Config, log_dir: Path | None = None,
                 timeout: float = REQUEST_TIMEOUT_S):
        self.cfg = cfg
        self.log_dir = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
        self.timeout = timeout
        self.base_url: str | None = None
        self._owned: proc.Owned | None = None
        self._parallel = 1
        self._ctx: dict[str, int] = {}
        self._counts: dict[tuple[str, str], int] = {}
        self._log_mark = 0

    # -- lifecycle ---------------------------------------------------------------------

    def start(self) -> None:
        if self.cfg.ollama_url:
            self.base_url = self.cfg.ollama_url.rstrip("/")
            self.version()  # fails early when nothing answers; we never start one then
            return
        if self.cfg.no_ollama_start:
            raise EngineError("no Ollama server to use: --no-ollama-start is set "
                              "but --ollama-url is not")
        self._start_own(self._parallel)

    def _binary(self) -> str:
        if self.cfg.ollama_bin:
            path = Path(self.cfg.ollama_bin)
            if not path.exists():
                raise EngineError(f"ollama executable not found: {path}")
            return str(path)
        found = shutil.which("ollama")
        if not found:
            raise EngineError("ollama executable not found (use --ollama-bin or --ollama-url)")
        return found

    def _start_own(self, parallel: int) -> None:
        binary = self._binary()
        port = pick_port()
        env = {"OLLAMA_HOST": f"127.0.0.1:{port}",
               "OLLAMA_NUM_PARALLEL": str(parallel),
               "OLLAMA_KEEP_ALIVE": KEEP_ALIVE}
        if self.cfg.ollama_models_dir:
            env["OLLAMA_MODELS"] = str(self.cfg.ollama_models_dir)
        log_path = self.log_dir / f"ollama-{port}.log"
        owned = proc.Owned([binary, "serve"], env=env, log_path=log_path)
        try:
            owned.start()
            self.base_url = f"http://127.0.0.1:{port}"
            owned.wait_until(self._answers, START_TIMEOUT_S)
        except proc.ProcError as err:
            owned.stop()
            self.base_url = None
            raise EngineError(f"cannot start ollama serve: {err}") from None
        self._owned = owned
        self._parallel = parallel
        self._log_mark = 0

    def _answers(self) -> bool:
        try:
            self.version()
        except EngineError:
            return False
        return True

    def stop(self) -> None:
        """Stop the server this engine started. A server given by URL is left alone."""
        if self._owned is not None:
            self._owned.stop()

    # -- HTTP --------------------------------------------------------------------------

    def _open(self, path: str, payload: dict | None):
        if not self.base_url:
            raise EngineError("engine not started")
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path, data=data,
            headers={"Content-Type": "application/json"} if data else {},
            method="GET" if data is None else "POST")
        try:
            return urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as err:
            body = err.read().decode("utf-8", errors="replace")[:300]
            raise EngineError(f"ollama {path}: HTTP {err.code}: {body}") from None
        except (urllib.error.URLError, OSError) as err:
            raise EngineError(f"ollama {path}: {getattr(err, 'reason', err)}") from None

    def _get(self, path: str) -> dict:
        return self._post(path, None)

    def _post(self, path: str, payload: dict | None) -> dict:
        with self._open(path, payload) as response:
            try:
                reply = json.loads(response.read())
            except (OSError, ValueError) as err:
                raise EngineError(f"ollama {path}: unreadable reply: {err}") from None
        if isinstance(reply, dict) and reply.get("error"):
            raise EngineError(f"ollama {path}: {reply['error']}")
        return reply

    def _tag(self, model: ModelRef) -> str:
        if not model.ollama_tag:
            raise EngineError(f"model {model.id} has no Ollama tag")
        return model.ollama_tag

    def _options(self, model: ModelRef, sampling: Sampling) -> dict:
        options = {"temperature": sampling.temperature, "top_p": sampling.top_p,
                   "seed": sampling.seed, "num_predict": sampling.max_tokens}
        if sampling.top_k is not None:
            options["top_k"] = sampling.top_k
        if model.id in self._ctx:
            options["num_ctx"] = self._ctx[model.id]  # a different value reloads the model
        return options

    # -- Engine interface --------------------------------------------------------------

    def version(self) -> str:
        reply = self._get("/api/version")
        if not isinstance(reply, dict) or "version" not in reply:
            raise EngineError(f"ollama /api/version: unexpected reply {str(reply)[:100]}")
        return str(reply["version"])

    def show(self, model: ModelRef) -> dict:
        return self._post("/api/show", {"model": self._tag(model)})

    def backend(self) -> str | None:
        """GPU backend as logged by a server we started; None for foreign servers.

        The Ollama API does not expose the backend, only the server log does.
        """
        found = LIBRARY_RE.findall(self._log_text(whole=True))
        for name in reversed(found):
            if name.lower() in BACKENDS:
                return name.lower()
        return None

    def load(self, model: ModelRef, ctx: int, parallel: int = 1) -> float:
        """Load the model with context `ctx`; returns the load time in seconds.

        OLLAMA_NUM_PARALLEL is a server setting: a server we own is restarted when
        `parallel` changes. A server given by URL keeps its own setting.
        """
        if self._owned is not None and parallel != self._parallel:
            self._owned.stop()
            self._ctx.clear()
            self._start_own(parallel)
        self._log_mark = self._log_size()
        started = time.perf_counter()
        reply = self._post("/api/generate", {
            "model": self._tag(model), "prompt": "", "stream": False,
            "options": {"num_ctx": ctx}})
        self._ctx[model.id] = ctx
        if "load_duration" in reply:
            return reply["load_duration"] / NS
        return time.perf_counter() - started

    def unload(self, model: ModelRef) -> None:
        self._post("/api/generate", {"model": self._tag(model), "keep_alive": 0,
                                     "stream": False})
        self._ctx.pop(model.id, None)

    def complete(self, model: ModelRef, prompt: str, sampling: Sampling) -> Completion:
        payload = {"model": self._tag(model), "prompt": prompt, "raw": True, "stream": True,
                   "options": self._options(model, sampling)}
        parts: list[str] = []
        ttft = None
        final = None
        sent = time.perf_counter()
        with self._open("/api/generate", payload) as response:
            try:
                for line in response:
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    if chunk.get("error"):
                        raise EngineError(f"ollama /api/generate: {chunk['error']}")
                    if chunk.get("response"):
                        if ttft is None:
                            ttft = time.perf_counter() - sent
                        parts.append(chunk["response"])
                    if chunk.get("done"):
                        final = chunk
                        break
            except (OSError, ValueError) as err:
                raise EngineError(f"ollama /api/generate: broken stream: {err}") from None
        if final is None:
            raise EngineError("ollama /api/generate: stream ended without a final chunk")
        if ttft is None:
            ttft = time.perf_counter() - sent
        return Completion("".join(parts), self._timing(final, ttft), final)

    def chat(self, model: ModelRef, messages: list[dict], sampling: Sampling,
             tools: list[dict] | None = None, json_schema: dict | None = None,
             thinking: bool = False) -> ChatResult:
        payload = {"model": self._tag(model), "messages": messages, "stream": False,
                   "think": bool(thinking), "options": self._options(model, sampling)}
        if tools:
            payload["tools"] = tools
        if json_schema is not None:
            payload["format"] = json_schema
        reply = self._post("/api/chat", payload)
        message = reply.get("message") or {}
        calls = [self._tool_call(c) for c in message.get("tool_calls") or []]
        # Not streamed: first token latency is approximated by the server's own numbers.
        ttft = (reply.get("load_duration", 0) + reply.get("prompt_eval_duration", 0)) / NS
        return ChatResult(message.get("content") or "", calls, self._timing(reply, ttft), reply)

    @staticmethod
    def _tool_call(call: dict) -> dict:
        function = call.get("function") or {}
        arguments = function.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                raise EngineError(f"tool call arguments are not JSON: {arguments[:100]}") \
                    from None
        return {"name": function.get("name", ""), "arguments": arguments}

    @staticmethod
    def _timing(reply: dict, ttft: float) -> Timing:
        prompt_tokens = reply.get("prompt_eval_count", 0)
        prompt_ns = reply.get("prompt_eval_duration", 0)
        gen_tokens = reply.get("eval_count", 0)
        gen_ns = reply.get("eval_duration", 0)
        load_ns = reply.get("load_duration")
        return Timing(
            load_s=None if load_ns is None else load_ns / NS,
            ttft_s=ttft,
            prompt_tokens=prompt_tokens,
            prompt_tps=prompt_tokens / (prompt_ns / NS) if prompt_tokens and prompt_ns > 0
            else None,
            gen_tokens=gen_tokens,
            gen_tps=gen_tokens / (gen_ns / NS) if gen_tokens and gen_ns > 0 else None,
            cached_tokens=reply.get("prompt_eval_cached_count"))

    def count_tokens(self, model: ModelRef, text: str) -> int:
        key = (model.id, hashlib.sha256(text.encode("utf-8")).hexdigest())
        if key in self._counts:
            return self._counts[key]
        # Never sends fewer context tokens than the model was loaded with, so this does
        # not make Ollama reload it; unloaded models get a context that fits the text.
        ctx = self._ctx.get(model.id) or max(MIN_COUNT_CTX, len(text) // 2 + 64)
        reply = self._post("/api/generate", {
            "model": self._tag(model), "prompt": text, "raw": True, "stream": False,
            "options": {"num_predict": 1, "num_ctx": ctx, "temperature": 0, "seed": 42}})
        count = reply.get("prompt_eval_count", 0) + reply.get("prompt_eval_cached_count", 0)
        if count >= ctx:
            raise EngineError(f"text of {count} tokens does not fit the context of {ctx}; "
                              "the count would be wrong")
        self._counts[key] = count
        return count

    def gpu_fraction(self, model: ModelRef) -> float | None:
        """Share of the loaded model that sits in GPU memory (1.0 = fully on the GPU)."""
        tag = self._tag(model)
        for entry in self._get("/api/ps").get("models") or []:
            if tag in (entry.get("name"), entry.get("model")):
                size = entry.get("size") or 0
                if size <= 0:
                    return None
                return min(1.0, (entry.get("size_vram") or 0) / size)
        return None

    def kv_buffer_mb(self) -> float | None:
        """KV cache size of the last load, from the log of a server we started."""
        text = self._log_text()
        for pattern in KV_RES:
            sizes = [float(v) for v in pattern.findall(text)]
            if sizes:
                return sum(sizes)
        return None

    # -- server log --------------------------------------------------------------------

    def _log_size(self) -> int:
        if self._owned is None:
            return 0
        try:
            return self._owned.log_path.stat().st_size
        except OSError:
            return 0

    def _log_text(self, whole: bool = False) -> str:
        if self._owned is None:
            return ""
        try:
            data = self._owned.log_path.read_bytes()
        except OSError:
            return ""
        return data[0 if whole else self._log_mark:].decode("utf-8", errors="replace")
