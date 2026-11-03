"""Rules that judge smoke test answers. Each checker returns `(outcome, reason)`.

Outcomes:
* pass       the answer is correct
* fail       the answer is wrong (a timeout of generated code is a fail with reason "timeout")
* skipped    the task was not judged on purpose (generated code with --no-code-exec)
* unchecked  the category has no rule (creative, code_explain); the answer is only stored

`to_store` is the single place that maps an outcome to the store status and the `passed`
flag: judged tasks are always stored `ok` (pass/fail lives in `passed`), because a store
status `error` means "run again" and a wrong answer must not be repeated forever.

Only the answer counts. Thinking text (`<think>...</think>` inside the content, or an
unterminated `<think>` when the answer was cut off) is removed first; engines that return
thinking in a separate field never put it into `ChatResult.text` at all.

Code execution: `check_code` runs model generated Python on this machine. THIS IS NOT A
SANDBOX. The measures are only these: own temporary directory as working directory and
HOME/TMP, `python -I` (isolated mode), a minimal environment (no inherited tokens or other
secrets), stdin closed, a 10 second time limit that kills the whole process tree, and a
cap on the captured output. The code can still read files the user can read and use the
network. Use `--no-code-exec` to skip the code tasks on a machine where that is not
acceptable.
"""

from __future__ import annotations

import json
import math
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from decimal import Decimal, InvalidOperation
from pathlib import Path

CODE_TIMEOUT = 10  # seconds
OUTPUT_CAP = 64 * 1024  # bytes kept per stream; the rest is read and thrown away
REASON_LIMIT = 300

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_NUMBER = re.compile(r"ANSWER:[ \t]*(-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)")
_FENCE = re.compile(r"```[ \t]*([\w+-]*)[ \t]*\n(.*?)(?:```|\Z)", re.DOTALL)

# Runs solution and hidden tests in one interpreter. A pass is only believed when the
# runner got to its end and wrote the token file, so sys.exit(0) or os._exit(0) inside the
# model code cannot fake a pass.
_RUNNER = '''\
import sys
token = sys.argv[1]
namespace = {"__name__": "solution"}
with open("solution.py", encoding="utf-8") as handle:
    solution = handle.read()
with open("tests.py", encoding="utf-8") as handle:
    tests = handle.read()
exec(compile(solution, "solution.py", "exec"), namespace)
exec(compile(tests, "tests.py", "exec"), namespace)
with open("pass-" + token, "w", encoding="utf-8") as handle:
    handle.write("ok")
'''


def answer_text(text: str) -> str:
    """The answer without thinking text."""
    if "</think>" in text and "<think>" not in text.rsplit("</think>", 1)[0]:
        text = text.rsplit("</think>", 1)[1]  # the template opened the block already
    text = _THINK_BLOCK.sub("", text)
    if "<think>" in text:
        text = text.split("<think>", 1)[0]  # cut off while thinking
    return text.strip()


def _clip(text, limit: int = REASON_LIMIT) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 3] + "..."


# ---------------------------------------------------------------- number

def check_number(spec: dict, result, cfg=None) -> tuple[str, str]:
    matches = _NUMBER.findall(answer_text(result.text))
    if not matches:
        return "fail", "no line of the form ANSWER: <number>"
    try:
        given = Decimal(matches[-1].replace(",", ""))
        wanted = Decimal(str(spec["answer"]))
    except InvalidOperation:
        return "fail", "answer is not a number"
    if given == wanted:
        return "pass", ""
    return "fail", f"expected {spec['answer']}, got {matches[-1]}"


# ---------------------------------------------------------------- json

def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


_TYPES = {
    "object": lambda v: isinstance(v, dict), "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str), "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None, "integer": _is_int, "number": _is_number}


def validate_schema(value, schema: dict, path: str = "$") -> str | None:
    """Subset of JSON schema: type, properties, required, enum, items, minimum, maximum.

    Returns the first problem as text, or None when the value is valid.
    """
    kind = schema.get("type")
    if kind is not None:
        kinds = kind if isinstance(kind, list) else [kind]
        if not any(_TYPES[k](value) for k in kinds):
            return f"{path}: expected type {kind}"
    if "enum" in schema and value not in schema["enum"]:
        return f"{path}: value not in enum"
    if _is_number(value):
        if "minimum" in schema and value < schema["minimum"]:
            return f"{path}: below minimum"
        if "maximum" in schema and value > schema["maximum"]:
            return f"{path}: above maximum"
    if isinstance(value, dict):
        for name in schema.get("required", []):
            if name not in value:
                return f"{path}: missing required field {name}"
        for name, sub in schema.get("properties", {}).items():
            if name in value:
                problem = validate_schema(value[name], sub, f"{path}.{name}")
                if problem:
                    return problem
    if isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            problem = validate_schema(item, schema["items"], f"{path}[{index}]")
            if problem:
                return problem
    return None


def _parse_json(text: str):
    text = answer_text(text)
    for fence in _FENCE.finditer(text):
        try:
            return json.loads(fence.group(2))
        except ValueError:
            pass
    try:
        return json.loads(text)
    except ValueError:
        pass
    start = text.find("{")
    if start >= 0:
        try:
            return json.JSONDecoder().raw_decode(text[start:])[0]
        except ValueError:
            pass
    raise ValueError("no valid JSON in the answer")


def same_value(actual, wanted) -> bool:
    """Equality for checked values: numbers by value, strings trimmed and case-insensitive."""
    if isinstance(wanted, bool) or isinstance(actual, bool):
        return isinstance(wanted, bool) and isinstance(actual, bool) and actual == wanted
    if _is_number(wanted):
        return _is_number(actual) and math.isclose(actual, wanted, rel_tol=1e-9, abs_tol=1e-12)
    if isinstance(wanted, str):
        return isinstance(actual, str) and actual.strip().casefold() == wanted.strip().casefold()
    if isinstance(wanted, list):
        return (isinstance(actual, list) and len(actual) == len(wanted)
                and all(same_value(a, w) for a, w in zip(actual, wanted)))
    if isinstance(wanted, dict):
        return (isinstance(actual, dict) and set(actual) == set(wanted)
                and all(same_value(actual[k], w) for k, w in wanted.items()))
    return actual == wanted


def _lookup(value, path: str):
    for part in path.split("."):
        if isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        elif isinstance(value, dict) and part in value:
            value = value[part]
        else:
            raise KeyError(path)
    return value


def check_json(spec: dict, result, task: dict | None = None, cfg=None) -> tuple[str, str]:
    try:
        value = _parse_json(result.text)
    except ValueError as err:
        return "fail", str(err)
    schema = (task or {}).get("json_schema")
    if schema:
        problem = validate_schema(value, schema)
        if problem:
            return "fail", problem
    for path, wanted in spec.get("expect", {}).items():
        try:
            actual = _lookup(value, path)
        except KeyError:
            return "fail", f"{path}: missing"
        if not same_value(actual, wanted):
            return "fail", (f"{path}: expected {json.dumps(wanted)}, "
                            f"got {_clip(json.dumps(actual), 80)}")
    return "pass", ""


# ---------------------------------------------------------------- tool calls

def check_tool_call(spec: dict, result, cfg=None) -> tuple[str, str]:
    calls = result.tool_calls
    if spec.get("expect_none"):
        if calls:
            return "fail", f"unexpected tool call {calls[0].get('name')}"
        return "pass", ""
    if not calls:
        return "fail", "no tool call"
    if len(calls) != 1:
        return "fail", f"{len(calls)} tool calls, expected 1"
    call = calls[0]
    if call.get("name") != spec["name"]:
        return "fail", f"wrong tool {call.get('name')}"
    arguments = call.get("arguments") or {}
    for name, wanted in spec.get("args", {}).items():
        if name not in arguments:
            return "fail", f"argument {name} missing"
        if not same_value(arguments[name], wanted):
            return "fail", f"argument {name}: expected {json.dumps(wanted)}"
    return "pass", ""


# ---------------------------------------------------------------- needle

def check_needle(spec: dict, result, cfg=None) -> tuple[str, str]:
    if spec["answer"].casefold() in answer_text(result.text).casefold():
        return "pass", ""
    return "fail", "needle not found in the answer"


# ---------------------------------------------------------------- code

def extract_code(text: str) -> str | None:
    text = answer_text(text)
    blocks = [body for _lang, body in _FENCE.findall(text)]
    for body in blocks:
        if re.search(r"^\s*(def|class)\s", body, re.MULTILINE):
            return body
    if blocks:
        return blocks[0]
    return text if re.search(r"^\s*def\s", text, re.MULTILINE) else None


def _minimal_env(home: str) -> dict:
    env = {"PYTHONIOENCODING": "utf-8", "HOME": home, "TMPDIR": home, "TEMP": home,
           "TMP": home, "USERPROFILE": home}
    for name in ("PATH", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "PATHEXT", "LANG"):
        if name in os.environ:
            env[name] = os.environ[name]
    return env


def _drain(stream, sink: list) -> None:
    kept = 0
    try:
        while chunk := stream.read(4096):
            if kept < OUTPUT_CAP:
                sink.append(chunk[:OUTPUT_CAP - kept])
                kept += len(sink[-1])
    except (OSError, ValueError):
        pass


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, timeout=10, check=False)
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        proc.kill()
    except OSError:
        pass


def _run_python(workdir: str, token: str, timeout: float) -> tuple[int | None, str]:
    """Run runner.py in `workdir`; returns (exit code or None on timeout, stderr text)."""
    flags = {}
    if os.name == "nt":
        flags["creationflags"] = (subprocess.CREATE_NEW_PROCESS_GROUP
                                  | subprocess.CREATE_NO_WINDOW)
    else:
        flags["start_new_session"] = True  # own process group, killed as a whole
    proc = subprocess.Popen(
        [sys.executable, "-I", "runner.py", token], cwd=workdir, env=_minimal_env(workdir),
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **flags)
    out: list = []
    err: list = []
    threads = [threading.Thread(target=_drain, args=(proc.stdout, out), daemon=True),
               threading.Thread(target=_drain, args=(proc.stderr, err), daemon=True)]
    for thread in threads:
        thread.start()
    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_tree(proc)
        proc.wait()
    for thread in threads:
        thread.join(timeout=2)
    for stream in (proc.stdout, proc.stderr):
        try:
            stream.close()
        except OSError:
            pass
    return (None if timed_out else proc.returncode), b"".join(err).decode("utf-8", "replace")


def check_code(spec: dict, result, cfg=None, timeout: float = CODE_TIMEOUT) -> tuple[str, str]:
    if cfg is not None and getattr(cfg, "no_code_exec", False):
        return "skipped", "code execution disabled (--no-code-exec)"
    code = extract_code(result.text)
    if code is None:
        return "fail", "no code block in the answer"
    try:
        compile(code, "solution.py", "exec")
    except (SyntaxError, ValueError) as err:
        return "fail", f"syntax error: {_clip(err, 120)}"
    token = secrets.token_hex(8)
    workdir = tempfile.mkdtemp(prefix="bench-code-")
    try:
        root = Path(workdir)
        (root / "runner.py").write_text(_RUNNER, encoding="utf-8")
        (root / "solution.py").write_text(code, encoding="utf-8")
        (root / "tests.py").write_text(spec["tests"], encoding="utf-8")
        returncode, stderr = _run_python(workdir, token, timeout)
        passed = returncode == 0 and (root / f"pass-{token}").exists()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    if returncode is None:
        return "fail", "timeout"
    if passed:
        return "pass", ""
    lines = [line for line in stderr.replace(workdir, "<tmp>").splitlines() if line.strip()]
    detail = lines[-1] if lines else "no error output"
    return "fail", _clip(f"tests failed (exit {returncode}): {detail}")


# ---------------------------------------------------------------- dispatch

CHECKERS = {"number": check_number, "json": check_json, "tool_call": check_tool_call,
            "code": check_code, "needle": check_needle}


def check(task: dict, result, cfg) -> tuple[str, str]:
    """Judge one answer: `(outcome, reason)` with outcome pass, fail, skipped or unchecked."""
    kind = task["check"]["type"]
    if kind == "none":
        return "unchecked", ""
    if kind not in CHECKERS:
        raise ValueError(f"unknown check type {kind!r} in task {task.get('id')}")
    spec = task["check"]
    if kind == "json":
        return check_json(spec, result, task, cfg)
    return CHECKERS[kind](spec, result, cfg)


def to_store(outcome: str, reason: str = "") -> tuple[str, bool | None]:
    """Store status and `passed` flag for an outcome (ruling R1, the only mapping)."""
    if outcome == "pass":
        return "ok", True
    if outcome == "fail":
        return "ok", False
    if outcome == "unchecked":
        return "ok", None
    if outcome == "skipped":
        return "skipped", None
    raise ValueError(f"unknown outcome {outcome!r}")
