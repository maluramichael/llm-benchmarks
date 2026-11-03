"""Processes the harness starts itself, and nothing else.

`Owned` wraps exactly one child process (started by us, in its own process group or
session) and stops exactly that process tree. It never looks up processes by name or
port. POSIX: the child leads its own session, `stop` signals the whole group, also when
the leader has exited already (grandchildren such as Ollama's runner processes survive a
dead leader). Windows: the child is put into a kill-on-close Job Object. `stop` acts once;
a second call does nothing, because the group id or pid may belong to somebody else by then.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Callable, Sequence

from bench.proctree import attach_job

IS_WINDOWS = os.name == "nt"
TAIL_CHARS = 800


class ProcError(RuntimeError):
    """A process could not be started or did not become ready."""


def free_port() -> int:
    """A TCP port on 127.0.0.1 that is free right now (asked from the OS)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


DEFAULT_PORTS = (11434, 8080)  # Ollama and llama-server: never ours, a foreign server may sit there


def pick_port() -> int:
    """A free port that is neither Ollama's nor llama-server's default port."""
    for _ in range(20):
        port = free_port()
        if port not in DEFAULT_PORTS:
            return port
    raise ProcError("no free port found")


def dropped(name: str, patterns: Sequence[str]) -> bool:
    """True if the environment variable `name` matches a pattern: an exact name, or a
    prefix when the pattern ends with `*`. Case-insensitive on Windows, like the OS."""
    if IS_WINDOWS:
        name, patterns = name.upper(), [p.upper() for p in patterns]
    return any(name.startswith(p[:-1]) if p.endswith("*") else name == p for p in patterns)


class Owned:
    """A child process of this program: `with Owned(argv, env=..., log_path=...) as p:`.

    The child gets a copy of the current environment without the variables matching
    `drop_env` (exact names, or prefixes ending in `*`), plus the `env` entries. Only the
    child's copy changes, never the environment of this program.
    stdout and stderr go to `log_path` (parent directories are created). With
    `stdout_path`, stdout goes to that file instead and `log_path` keeps only stderr,
    for programs whose stdout is data (for example JSON) and must not be mixed with logs.
    """

    def __init__(self, argv: Sequence[str | os.PathLike], *, log_path: Path,
                 env: dict[str, str] | None = None, cwd: Path | None = None,
                 stdout_path: Path | None = None, drop_env: Sequence[str] = ()):
        self.argv = [str(a) for a in argv]
        self.env = dict(env or {})
        self.drop_env = tuple(drop_env)
        self.log_path = Path(log_path)
        self.stdout_path = Path(stdout_path) if stdout_path else None
        self.cwd = cwd
        self._proc: subprocess.Popen | None = None
        self._log = None
        self._out = None
        self._job = None
        self._stopped = False

    def start(self) -> "Owned":
        if self._proc is not None:
            raise ProcError("process already started")
        child_env = {k: v for k, v in os.environ.items() if not dropped(k, self.drop_env)}
        child_env.update(self.env)
        options: dict = {}
        if IS_WINDOWS:
            options["creationflags"] = (subprocess.CREATE_NEW_PROCESS_GROUP
                                        | getattr(subprocess, "CREATE_NO_WINDOW", 0))
        else:
            options["start_new_session"] = True  # own process group, killable as a unit
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = self.log_path.open("ab")
            if self.stdout_path is not None:
                self.stdout_path.parent.mkdir(parents=True, exist_ok=True)
                self._out = self.stdout_path.open("wb")
            self._proc = subprocess.Popen(
                self.argv, stdin=subprocess.DEVNULL, stdout=self._out or self._log,
                stderr=self._log if self._out else subprocess.STDOUT, env=child_env,
                cwd=self.cwd, **options)
            self._stopped = False
            if IS_WINDOWS:
                self._job = attach_job(self._proc)
        except (OSError, ValueError) as err:  # ValueError: for example a NUL in an argument
            self._close_log()
            raise ProcError(
                f"cannot start {self.argv[0]}: {getattr(err, 'strerror', None) or err}") from None
        return self

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc else None

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def returncode(self) -> int | None:
        return self._proc.poll() if self._proc else None

    def log_tail(self, chars: int = TAIL_CHARS) -> str:
        try:
            data = self.log_path.read_bytes()[-chars:]
        except OSError:
            return ""
        return data.decode("utf-8", errors="replace").strip()

    def wait(self, timeout: float) -> int:
        """Wait for the process to exit; returns its exit code. Does not stop it on timeout."""
        if self._proc is None:
            raise ProcError("process not started")
        try:
            return self._proc.wait(timeout)
        except subprocess.TimeoutExpired:
            raise ProcError(f"still running after {timeout:g} s") from None

    def wait_until(self, predicate: Callable[[], bool], timeout: float,
                   interval: float = 0.2) -> None:
        """Poll until `predicate()` is true; fail if the process exits or time runs out."""
        deadline = time.monotonic() + timeout
        while True:
            if self._proc is None or self._proc.poll() is not None:
                code = self.returncode
                raise ProcError(f"process exited with code {code} before it was ready:\n"
                                f"{self.log_tail()}")
            try:
                if predicate():
                    return
            except OSError:
                pass
            if time.monotonic() >= deadline:
                raise ProcError(f"not ready after {timeout:g} s:\n{self.log_tail()}")
            time.sleep(interval)

    def stop(self, grace: float = 5.0) -> None:
        """End the process tree. Idempotent: only the first call signals anything."""
        proc = self._proc
        if proc is None:
            return
        try:
            if not self._stopped:
                self._stopped = True
                if IS_WINDOWS:
                    self._stop_windows(proc, grace)
                else:
                    self._stop_posix(proc, grace)
        finally:
            self._close_log()

    def _stop_windows(self, proc: subprocess.Popen, grace: float) -> None:
        job, self._job = self._job, None
        if job is not None and job.kill():  # every process of the job, leader gone or not
            if proc.poll() is None:
                try:
                    proc.wait(grace)
                except subprocess.TimeoutExpired:
                    proc.kill()
            return
        if proc.poll() is not None:
            return  # no job and no leader: its pid may belong to someone else by now
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                       capture_output=True, check=False)
        try:
            proc.wait(grace)
        except subprocess.TimeoutExpired:
            proc.kill()  # taskkill failed; at least end the direct child
            proc.wait(grace)

    def _stop_posix(self, proc: subprocess.Popen, grace: float) -> None:
        group = proc.pid  # the child is its own group leader (start_new_session)
        if proc.poll() is None:
            self._signal_group(group, signal.SIGTERM)
            try:
                proc.wait(grace)
            except subprocess.TimeoutExpired:
                pass
        # Sweep what is left of the group: runner children that ignored SIGTERM, and
        # grandchildren that outlived their leader.
        self._signal_group(group, signal.SIGKILL)
        proc.wait(grace)

    @staticmethod
    def _signal_group(group: int, sig: int) -> None:
        try:
            os.killpg(group, sig)
        except (ProcessLookupError, PermissionError):
            pass

    def _close_log(self) -> None:
        for name in ("_log", "_out"):
            handle = getattr(self, name)
            if handle is not None:
                handle.close()
                setattr(self, name, None)

    def __enter__(self) -> "Owned":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
