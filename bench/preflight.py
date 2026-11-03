"""Pre-flight check: is something else using the benchmark GPU?

Read-only. Two things are looked for:

* other processes with a compute context on the benchmark GPU (nvidia-smi
  --query-compute-apps; NVIDIA only),
* an Ollama server that already listens on the default port 11434 and was not started by
  bench. bench never uses it (unless --ollama-url is given) and never stops it, but it can
  hold GPU memory or answer for a model that bench thinks it loaded.

Every finding is a dict `{kind, text, note}`. `text` is for the console and may name
processes; `note` is generic and is what gets stored in results.json.
On the office PCs the shared OllamaServe is stopped by the Ansible playbook before a run;
this check only detects and warns.
"""

from __future__ import annotations

import json
import os
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


def _csv_rows(text: str | None) -> list[list[str]]:
    rows = []
    for line in (text or "").splitlines():
        if line.strip():
            rows.append([part.strip() for part in line.split(",")])
    return rows


def _gpu_processes(gpu_index: int, run) -> list[tuple[str, str]]:
    """(pid, process name) of the other processes on GPU `gpu_index`."""
    exe = find_nvidia_smi()
    if not exe:
        return []
    apps = run([exe, "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits"])
    if not apps:
        return []
    uuids = {row[0]: row[1] for row in _csv_rows(
        run([exe, "--query-gpu=index,uuid", "--format=csv,noheader"])) if len(row) >= 2}
    wanted = uuids.get(str(gpu_index))  # None: mapping unknown, count every process
    found = []
    for row in _csv_rows(apps):
        if len(row) < 3:
            continue
        uuid, pid, name = row[0], row[1], row[2]
        if wanted is not None and uuid != wanted:
            continue
        if pid == str(os.getpid()):
            continue
        found.append((pid, name))
    return found


def check(cfg, gpu_index: int, *, run=run_text, port_open=_port_open,
          fetch_ps=_fetch_ps) -> list[dict]:
    findings: list[dict] = []
    try:
        processes = _gpu_processes(gpu_index, run)
    except Exception:
        processes = []
    if processes:
        names = ", ".join(sorted({name for _pid, name in processes}))
        findings.append({
            "kind": "gpu_busy",
            "text": f"{len(processes)} other process(es) use GPU {gpu_index}: {names}. "
                    "They distort the numbers and take VRAM; close them for a clean run.",
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
