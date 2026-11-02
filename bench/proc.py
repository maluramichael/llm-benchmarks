"""Processes the harness starts itself, and nothing else.

`Owned` wraps exactly one child process (started by us, in its own process group or
session) and stops exactly that process tree. It never looks up processes by name or
port, and it does nothing when the child is already gone, because its pid may belong to
somebody else by then.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Callable, Sequence

IS_WINDOWS = os.name == "nt"
TAIL_CHARS = 800


class ProcError(RuntimeError):
    """A process could not be started or did not become ready."""


def free_port() -> int:
    """A TCP port on 127.0.0.1 that is free right now (asked from the OS)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Owned:
    """A child process of this program: `with Owned(argv, env=..., log_path=...) as p:`.

    `env` entries are added to a copy of the current environment for the child only.
    stdout and stderr go to `log_path` (parent directories are created).
    """

    def __init__(self, argv: Sequence[str | os.PathLike], *, log_path: Path,
                 env: dict[str, str] | None = None, cwd: Path | None = None):
        self.argv = [str(a) for a in argv]
        self.env = dict(env or {})
        self.log_path = Path(log_path)
        self.cwd = cwd
        self._proc: subprocess.Popen | None = None
        self._log = None

    def start(self) -> "Owned":
        if self._proc is not None:
            raise ProcError("process already started")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("ab")
        child_env = {**os.environ, **self.env}
        options: dict = {}
        if IS_WINDOWS:
            options["creationflags"] = (subprocess.CREATE_NEW_PROCESS_GROUP
                                        | getattr(subprocess, "CREATE_NO_WINDOW", 0))
        else:
            options["start_new_session"] = True  # own process group, killable as a unit
        try:
            self._proc = subprocess.Popen(
                self.argv, stdin=subprocess.DEVNULL, stdout=self._log,
                stderr=subprocess.STDOUT, env=child_env, cwd=self.cwd, **options)
        except OSError as err:
            self._close_log()
            raise ProcError(f"cannot start {self.argv[0]}: {err.strerror or err}") from None
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
        """End the process tree. Idempotent; does nothing if the child already exited."""
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.poll() is None:
                if IS_WINDOWS:
                    self._stop_windows(proc, grace)
                else:
                    self._stop_posix(proc, grace)
        finally:
            self._close_log()

    def _stop_windows(self, proc: subprocess.Popen, grace: float) -> None:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                       capture_output=True, check=False)
        try:
            proc.wait(grace)
        except subprocess.TimeoutExpired:
            proc.kill()  # taskkill failed; at least end the direct child
            proc.wait(grace)

    def _stop_posix(self, proc: subprocess.Popen, grace: float) -> None:
        group = proc.pid  # the child is its own group leader (start_new_session)
        self._signal_group(group, signal.SIGTERM)
        try:
            proc.wait(grace)
        except subprocess.TimeoutExpired:
            pass
        # Sweep what is left of the group (runner children that ignored SIGTERM).
        self._signal_group(group, signal.SIGKILL)
        proc.wait(grace)

    @staticmethod
    def _signal_group(group: int, sig: int) -> None:
        try:
            os.killpg(group, sig)
        except (ProcessLookupError, PermissionError):
            pass

    def _close_log(self) -> None:
        if self._log is not None:
            self._log.close()
            self._log = None

    def __enter__(self) -> "Owned":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
