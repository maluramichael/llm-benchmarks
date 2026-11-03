"""Pre-flight check: is something else using the benchmark GPU?

Read-only. Two things are looked for:

* other compute workloads on the benchmark GPU: processes of type C in the process table of
  plain `nvidia-smi` (NVIDIA only), and known engines (ollama, llama-server, python, ...)
  whatever their type. Ordinary desktop programs have a graphics context (G, C+G) and are
  ignored,
* an Ollama server that already listens on the default port 11434 and was not started by
  bench. bench never uses it (unless --ollama-url is given) and never stops it, but it can
  hold GPU memory or answer for a model that bench thinks it loaded.

Every finding is a dict `{kind, text, note}`. `text` is for the console and names at most a count and engines from KNOWN_ENGINES;
`detail` (console only) lists every process; `note` is generic and is what gets stored in
results.json.
On the office PCs the shared OllamaServe is stopped by the Ansible playbook before a run;
this check only detects and warns.
"""

from __future__ import annotations

import json
import os
import re
import socket
import urllib.request

from bench.gpu import find_nvidia_smi, run_text

OLLAMA_PORT = 11434
CONNECT_TIMEOUT_S = 1.0
PS_TIMEOUT_S = 3.0


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=CONNECT_TIMEOUT_S):
            return True
    except OSError:
        return False


def _fetch_ps(url: str):
    """The /api/ps reply of a running Ollama, or None."""
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url, timeout=PS_TIMEOUT_S) as reply:
            return json.loads(reply.read().decode("utf-8", errors="replace"))
    except (OSError, ValueError):
        return None


# Process table of plain `nvidia-smi`: GPU, GI ID, CI ID, PID, Type, name, memory.
# Type is C (compute), G (graphics) or C+G. Under Windows (WDDM) every desktop program has a
# graphics context, so only real compute workloads count. The name may contain spaces.
_PROCESS_ROW = re.compile(
    r"^\|\s*(?P<gpu>\d+)\s+\S+\s+\S+\s+(?P<pid>\d+)\s+(?P<type>C\+G|C|G)\s+"
    r"(?P<name>.+?)\s+(?P<memory>N/A|\d+\s*MiB)\s*\|\s*$")
# Names (lower case, part of the program name) of engines and compute tools worth naming.
KNOWN_ENGINES = ("ollama", "llama-server", "llama-cli", "python", "comfyui", "koboldcpp",
                 "lm studio", "lm-studio")


def parse_processes(text: str | None) -> list[dict]:
    """The rows of the process table of `nvidia-smi` as dicts gpu, pid, type, name, memory."""
    rows = []
    for line in (text or "").splitlines():
        match = _PROCESS_ROW.match(line)
        if match:
            rows.append({"gpu": int(match["gpu"]), "pid": int(match["pid"]),
                         "type": match["type"], "name": match["name"].strip(),
                         "memory": match["memory"].replace(" ", "")})
    return rows


def _known_engine(name: str) -> str | None:
    base = re.split(r"[\\/]", name.lower())[-1]
    for engine in KNOWN_ENGINES:
        if engine in base:
            return engine
    return None


def _gpu_workloads(gpu_index: int, run) -> list[dict]:
    """Processes on GPU `gpu_index` that run compute (type C) or are a known engine."""
    exe = find_nvidia_smi()
    if not exe:
        return []
    rows = parse_processes(run([exe]))
    return [r for r in rows
            if r["gpu"] == gpu_index and r["pid"] != os.getpid()
            and (r["type"] == "C" or _known_engine(r["name"]))]


def check(cfg, gpu_index: int, *, run=run_text, port_open=_port_open,
          fetch_ps=_fetch_ps) -> list[dict]:
    findings: list[dict] = []
    try:
        workloads = _gpu_workloads(gpu_index, run)
    except Exception:
        workloads = []
    if workloads:
        names = sorted({n for n in (_known_engine(r["name"]) for r in workloads) if n})
        named = f" ({', '.join(names)})" if names else ""
        detail = ", ".join(f"{r['name']} pid {r['pid']} type {r['type']} {r['memory']}"
                           for r in workloads)
        findings.append({
            "kind": "gpu_busy",
            "text": f"{len(workloads)} other process(es) run compute work on GPU "
                    f"{gpu_index}{named}. They distort the numbers and take VRAM; close them "
                    "for a clean run.",
            "detail": detail,
            "note": f"another process was using the benchmark GPU at the start "
                    f"(GPU {gpu_index})"})
    if "ollama" in cfg.engines and not cfg.ollama_url:
        try:
            listening = port_open("127.0.0.1", OLLAMA_PORT)
        except Exception:
            listening = False
        if listening:
            loaded = None
            try:
                reply = fetch_ps(f"http://127.0.0.1:{OLLAMA_PORT}/api/ps")
                if isinstance(reply, dict) and isinstance(reply.get("models"), list):
                    loaded = len(reply["models"])
            except Exception:
                loaded = None
            extra = f" with {loaded} model(s) loaded" if loaded is not None else ""
            findings.append({
                "kind": "foreign_ollama",
                "text": f"an Ollama server that bench did not start is listening on port "
                        f"{OLLAMA_PORT}{extra}. bench starts its own server on a free port and "
                        "never stops yours, but yours may hold GPU memory. Stop it before the "
                        "run (on the office PCs the playbook stops OllamaServe).",
                "note": f"a foreign Ollama server was listening on port {OLLAMA_PORT} "
                        "at the start"})
    return findings
