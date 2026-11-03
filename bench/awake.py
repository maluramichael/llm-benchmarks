"""Keep the computer awake while a run is going, and only then.

Windows   SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) from the calling
          thread; put back to ES_CONTINUOUS at the end. The state belongs to the thread, so
          the caller must stay in the same thread for the whole block.
macOS     `caffeinate -i -w <pid>`: no idle sleep; it ends by itself when this process ends.
Linux     `systemd-inhibit` around `tail --pid=<pid>`: blocks idle and sleep, and ends with
          this process. Without systemd-inhibit nothing is done.

Only helper processes this module starts are ever stopped. No persistent power setting is
read or changed. Failure to keep the machine
awake is a warning, never an error.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
from typing import Callable, Iterator

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
STOP_WAIT_S = 5


def _windows_kernel32():
    import ctypes
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    kernel32.SetThreadExecutionState.argtypes = [ctypes.c_uint32]
    kernel32.SetThreadExecutionState.restype = ctypes.c_uint32
    return kernel32


@contextlib.contextmanager
def keep_awake(*, platform: str | None = None, kernel32=None, popen=None, which=None,
               pid: int | None = None,
               say: Callable[[str], None] | None = None) -> Iterator[str | None]:
    """Context manager; yields the method used, or None when nothing could be done."""
    platform = platform or sys.platform
    say = say or (lambda message: print(message, file=sys.stderr))
    popen = popen or subprocess.Popen
    which = which or shutil.which
    pid = pid or os.getpid()
    method: str | None = None
    helper = None
    kernel = None
    try:
        if platform == "win32":
            try:
                kernel = kernel32 or _windows_kernel32()
                if kernel.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED):
                    method = "SetThreadExecutionState"
                else:
                    kernel = None
                    say("warning: could not keep the computer awake "
                        "(SetThreadExecutionState failed)")
            except Exception as err:
                kernel = None
                say(f"warning: could not keep the computer awake ({type(err).__name__})")
        elif platform == "darwin":
            argv = ["{exe}", "-i", "-w", str(pid)]
            helper, method = _start_helper(popen, which, "caffeinate", argv, "caffeinate", say)
        elif platform.startswith("linux"):
            argv = ["{exe}", "--what=idle:sleep", "--who=bench", "--why=benchmark run",
                    "--mode=block", "tail", f"--pid={pid}", "-f", "/dev/null"]
            helper, method = _start_helper(popen, which, "systemd-inhibit", argv,
                                           "systemd-inhibit", say)
        yield method
    finally:
        if kernel is not None:
            with contextlib.suppress(Exception):
                kernel.SetThreadExecutionState(ES_CONTINUOUS)
        if helper is not None:
            _stop_helper(helper)


def _start_helper(popen, which, command: str, argv: list[str], label: str, say):
    exe = which(command)
    if not exe:
        return None, None
    argv = [exe if part == "{exe}" else part for part in argv]
    try:
        helper = popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    except (OSError, ValueError) as err:
        say(f"warning: could not start {label} ({type(err).__name__}); "
            "the computer may go to sleep during the run")
        return None, None
    return helper, label


def _stop_helper(helper) -> None:
    with contextlib.suppress(Exception):
        helper.terminate()
        try:
            helper.wait(timeout=STOP_WAIT_S)
        except subprocess.TimeoutExpired:
            helper.kill()
            helper.wait(timeout=STOP_WAIT_S)
