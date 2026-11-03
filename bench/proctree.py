"""Ending a process together with everything it started.

Shared by the code runner of the smoke tests (bench.checkers) and the engine processes
(bench.proc). POSIX: the process is its own session and group leader, `killpg` reaches the
whole group, also when the leader is already gone. Windows: a Job Object with
kill-on-close (ctypes); `taskkill /F /T` is the fallback when the job API fails. The child is
created suspended (CREATE_SUSPENDED), assigned to the job and then resumed with
NtResumeProcess, so it cannot start a child that escapes the job.
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


CREATE_SUSPENDED = 0x4


def spawn_kwargs(suspended: bool = False) -> dict:
    """Popen options. Windows with `suspended`: the child is created suspended, so that
    `attach_job(proc, suspended=True)` can put it into the job before it runs a single
    instruction (it cannot start a child that escapes the job)."""
    if IS_WINDOWS:
        # numeric fallbacks: CREATE_NEW_PROCESS_GROUP, CREATE_NO_WINDOW
        flags = (getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200)
                 | getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000))
        return {"creationflags": flags | (CREATE_SUSPENDED if suspended else 0)}
    return {"start_new_session": True}  # own process group, killed as a whole


def resume_process(proc: subprocess.Popen, ntdll=None) -> bool:
    """Windows: resume a process created with CREATE_SUSPENDED (NtResumeProcess on its
    process handle; subprocess does not expose the thread handle). True on success."""
    try:
        if ntdll is None:
            import ctypes
            ntdll = ctypes.WinDLL("ntdll")
        import ctypes
        status = ntdll.NtResumeProcess(ctypes.c_void_p(int(proc._handle)))
        return status == 0
    except (OSError, AttributeError, ValueError, TypeError):
        return False


def attach_job(proc: subprocess.Popen, suspended: bool = False):
    """Windows: put the new process into a kill-on-close job. None elsewhere or when the job
    API is unavailable (callers then fall back to `taskkill /T`).

    With `suspended` the process was created suspended: it is assigned first and resumed
    afterwards. If assigning or resuming fails, the process (and the job) is killed and
    OSError is raised, so a child never runs outside the job it was meant to be in. When the
    job API itself is unavailable the child is resumed (the taskkill fallback applies)."""
    if not IS_WINDOWS:
        return None
    job = WinJob.open()
    if job is None:
        if suspended and not resume_process(proc):
            _kill_quietly(proc)
            raise OSError("cannot resume the suspended process")
        return None
    if not job.assign(int(proc._handle)):
        job.kill()
        if suspended:
            _kill_quietly(proc)
            raise OSError("cannot assign the process to its job")
        return None
    if suspended and not resume_process(proc):
        job.kill()
        _kill_quietly(proc)
        raise OSError("cannot resume the suspended process")
    return job


def _kill_quietly(proc: subprocess.Popen) -> None:
    try:
        proc.kill()
    except OSError:
        pass


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
