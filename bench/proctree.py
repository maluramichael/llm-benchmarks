"""Ending a process together with everything it started.

Shared by the code runner of the smoke tests (bench.checkers) and the engine processes
(bench.proc). POSIX: the process is its own session and group leader, `killpg` reaches the
whole group, also when the leader is already gone. Windows: a Job Object with
kill-on-close (ctypes); `taskkill /F /T` is the fallback when the job API fails. A child
started in the first microseconds after the spawn could escape the job (subprocess cannot
start suspended); that race is accepted.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys

IS_WINDOWS = sys.platform == "win32"


class WinJob:
    """Windows Job Object that kills every process in it when terminated or closed."""

    KILL_ON_JOB_CLOSE = 0x2000
    EXTENDED_LIMIT_INFORMATION = 9

    def __init__(self, kernel32, handle):
        self.k32 = kernel32
        self.handle = handle

    @classmethod
    def open(cls, kernel32=None):
        """A new job set to kill on close, or None when the API fails."""
        try:
            if kernel32 is None:
                import ctypes
                kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            handle = kernel32.CreateJobObjectW(None, None)
            if not handle:
                return None
            info = cls._limit_info()
            if not kernel32.SetInformationJobObject(
                    handle, cls.EXTENDED_LIMIT_INFORMATION, info, cls._size(info)):
                kernel32.CloseHandle(handle)
                return None
            return cls(kernel32, handle)
        except (OSError, AttributeError, ValueError):
            return None

    @classmethod
    def _limit_info(cls):
        import ctypes

        class Basic(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                        ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", ctypes.c_uint32),
                        ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t),
                        ("ActiveProcessLimit", ctypes.c_uint32),
                        ("Affinity", ctypes.c_size_t),
                        ("PriorityClass", ctypes.c_uint32),
                        ("SchedulingClass", ctypes.c_uint32)]

        class Extended(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", Basic),
                        ("IoInfo", ctypes.c_uint64 * 6),
                        ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        info = Extended()
        info.BasicLimitInformation.LimitFlags = cls.KILL_ON_JOB_CLOSE
        return info

    @staticmethod
    def _size(info) -> int:
        import ctypes
        return ctypes.sizeof(info)

    def assign(self, process_handle) -> bool:
        try:
            return bool(self.k32.AssignProcessToJobObject(self.handle, process_handle))
        except (OSError, ValueError):
            return False

    def kill(self) -> bool:
        """Terminate every process of the job and close it (once)."""
        if self.handle is None:
            return True
        handle, self.handle = self.handle, None
        try:
            ok = bool(self.k32.TerminateJobObject(handle, 1))
            self.k32.CloseHandle(handle)
            return ok
        except (OSError, ValueError):
            return False


def spawn_kwargs() -> dict:
    if IS_WINDOWS:
        # numeric fallbacks: CREATE_NEW_PROCESS_GROUP, CREATE_NO_WINDOW
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200)
                | getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)}
    return {"start_new_session": True}  # own process group, killed as a whole


def attach_job(proc: subprocess.Popen):
    """Windows: put the new process into a kill-on-close job. None elsewhere or on failure."""
    if not IS_WINDOWS:
        return None
    job = WinJob.open()
    if job is None:
        return None
    if not job.assign(int(proc._handle)):
        job.kill()
        return None
    return job


def kill_tree(proc: subprocess.Popen, job=None) -> None:
    """Kill the process and everything it started, also when the parent is already gone."""
    try:
        if IS_WINDOWS:
            if job is None or not job.kill():
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                               capture_output=True, timeout=10, check=False)
        else:
            # The group id is the pid of the leader; members outlive a dead leader.
            os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        proc.kill()
    except OSError:
        pass
