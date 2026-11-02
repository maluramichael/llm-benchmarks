import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from bench import proc

PY = sys.executable
SLEEP = [PY, "-c", "import time; time.sleep(60)"]
# Starts a grandchild, writes both pids to the file given as argv[1], then sleeps.
TREE = """
import os, subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
open(sys.argv[1], "w").write(f"{os.getpid()} {child.pid}")
time.sleep(60)
"""


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    # A zombie still answers signal 0; a reaped or adopted-and-reaped process does not.
    try:
        status = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)],
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        return True
    return bool(status) and not status.startswith("Z")


class ProcTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.log = self.dir / "logs" / "proc.log"


class FreePortTest(unittest.TestCase):
    def test_port_is_usable(self):
        port = proc.free_port()
        self.assertTrue(0 < port < 65536)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))

    def test_two_calls_differ_while_first_is_held(self):
        first = proc.free_port()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", first))
            self.assertNotEqual(proc.free_port(), first)


class OwnedTest(ProcTestCase):
    def test_env_and_log(self):
        script = "import os, sys; print(os.environ['BENCH_X']); print('err', file=sys.stderr)"
        owned = proc.Owned([PY, "-c", script], env={"BENCH_X": "hello"}, log_path=self.log)
        owned.start()
        owned._proc.wait(10)
        owned.stop()
        text = self.log.read_text(encoding="utf-8")
        self.assertIn("hello", text)
        self.assertIn("err", text)

    def test_env_does_not_leak_into_parent(self):
        self.assertNotIn("BENCH_LEAK", os.environ)
        with proc.Owned(SLEEP, env={"BENCH_LEAK": "1"}, log_path=self.log):
            self.assertNotIn("BENCH_LEAK", os.environ)
        self.assertNotIn("BENCH_LEAK", os.environ)

    def test_context_manager_stops(self):
        with proc.Owned(SLEEP, log_path=self.log) as owned:
            pid = owned.pid
            self.assertTrue(owned.running)
        self.assertFalse(owned.running)
        self.assertFalse(alive(pid))

    @unittest.skipIf(os.name == "nt", "process groups are POSIX only")
    def test_stop_kills_whole_tree(self):
        pids = self.dir / "pids"
        owned = proc.Owned([PY, "-c", TREE, str(pids)], log_path=self.log).start()
        deadline = time.monotonic() + 10
        while not pids.exists() or len(pids.read_text().split()) < 2:
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.05)
        parent, child = map(int, pids.read_text().split())
        self.assertTrue(alive(child))
        owned.stop()
        deadline = time.monotonic() + 5
        while alive(child) and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertFalse(alive(parent))
        self.assertFalse(alive(child))

    def test_stop_is_idempotent(self):
        owned = proc.Owned(SLEEP, log_path=self.log).start()
        owned.stop()
        owned.stop()
        proc.Owned(SLEEP, log_path=self.log).stop()  # never started

    def test_stop_after_process_died_on_its_own(self):
        owned = proc.Owned([PY, "-c", "pass"], log_path=self.log).start()
        owned._proc.wait(10)
        with mock.patch("bench.proc.os.killpg", create=True) as killpg, \
                mock.patch("bench.proc.subprocess.run") as run:
            owned.stop()
            owned.stop()
        killpg.assert_not_called()
        run.assert_not_called()  # no taskkill either: the pid may belong to someone else now

    def test_windows_uses_taskkill_for_the_tree(self):
        owned = proc.Owned(SLEEP, log_path=self.log).start()
        pid = owned.pid
        real_run = subprocess.run
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            if cmd[0] == "taskkill":
                owned._proc.kill()  # stand in for taskkill so the test leaves nothing behind
                return subprocess.CompletedProcess(cmd, 0)
            return real_run(cmd, **kwargs)

        with mock.patch.object(proc, "IS_WINDOWS", True), \
                mock.patch("bench.proc.subprocess.run", fake_run):
            owned.stop()
        self.assertEqual(calls, [["taskkill", "/T", "/F", "/PID", str(pid)]])

    def test_wait_until_reports_early_exit_with_log(self):
        owned = proc.Owned([PY, "-c", "print('boom'); raise SystemExit(3)"],
                           log_path=self.log).start()
        with self.assertRaises(proc.ProcError) as ctx:
            owned.wait_until(lambda: False, timeout=10)
        self.assertIn("boom", str(ctx.exception))
        self.assertIn("3", str(ctx.exception))
        owned.stop()

    def test_wait_until_times_out(self):
        with proc.Owned(SLEEP, log_path=self.log) as owned:
            with self.assertRaises(proc.ProcError):
                owned.wait_until(lambda: False, timeout=0.3, interval=0.05)

    def test_wait_until_returns_when_true(self):
        with proc.Owned(SLEEP, log_path=self.log) as owned:
            owned.wait_until(lambda: True, timeout=1)

    def test_invalid_argument_closes_the_log(self):
        owned = proc.Owned([PY, "bad\0arg"], log_path=self.log)
        with self.assertRaises(proc.ProcError):
            owned.start()
        self.assertIsNone(owned._log)
        owned.stop()

    def test_unusable_log_location_is_a_proc_error(self):
        blocker = self.dir / "blocker"
        blocker.write_text("file", encoding="utf-8")
        owned = proc.Owned(SLEEP, log_path=blocker / "logs" / "x.log")
        with self.assertRaises(proc.ProcError):
            owned.start()
        self.assertIsNone(owned.pid)
        owned.stop()

    def test_missing_binary(self):
        with self.assertRaises(proc.ProcError):
            proc.Owned([str(self.dir / "nope")], log_path=self.log).start()


if __name__ == "__main__":
    unittest.main()
