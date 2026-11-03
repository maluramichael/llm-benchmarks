import unittest
from unittest import mock

from bench import proctree


class FakeKernel32:
    def __init__(self, set_ok=1, assign_ok=1):
        self.calls = []
        self.set_ok, self.assign_ok = set_ok, assign_ok

    def CreateJobObjectW(self, attrs, name):
        self.calls.append("create")
        return 77

    def SetInformationJobObject(self, handle, info_class, info, size):
        self.calls.append(("set", handle, info_class))
        return self.set_ok

    def AssignProcessToJobObject(self, handle, process):
        self.calls.append(("assign", handle, process))
        return self.assign_ok

    def TerminateJobObject(self, handle, code):
        self.calls.append(("terminate", handle))
        return 1

    def CloseHandle(self, handle):
        self.calls.append(("close", handle))
        return 1


class WindowsJobTest(unittest.TestCase):
    def test_job_is_created_assigned_and_killed(self):
        k32 = FakeKernel32()
        job = proctree.WinJob.open(k32)
        self.assertIsNotNone(job)
        self.assertEqual(k32.calls[:2], ["create", ("set", 77, 9)])
        self.assertTrue(job.assign(1234))
        self.assertTrue(job.kill())
        job.kill()  # idempotent: handle closed once
        self.assertEqual(k32.calls[2:], [("assign", 77, 1234), ("terminate", 77), ("close", 77)])

    def test_job_setup_failure_gives_no_job(self):
        k32 = FakeKernel32(set_ok=0)
        self.assertIsNone(proctree.WinJob.open(k32))
        self.assertIn(("close", 77), k32.calls)

    def test_assign_failure_is_reported(self):
        job = proctree.WinJob.open(FakeKernel32(assign_ok=0))
        self.assertFalse(job.assign(5))

    def test_kill_tree_uses_job_on_windows(self):
        proc, job = mock.Mock(pid=5), mock.Mock()
        job.kill.return_value = True
        with mock.patch.object(proctree, "IS_WINDOWS", True), \
                mock.patch.object(proctree.subprocess, "run") as run:
            proctree.kill_tree(proc, job)
        job.kill.assert_called_once()
        run.assert_not_called()

    def test_kill_tree_falls_back_to_taskkill(self):
        proc = mock.Mock(pid=5)
        with mock.patch.object(proctree, "IS_WINDOWS", True), \
                mock.patch.object(proctree.subprocess, "run") as run:
            proctree.kill_tree(proc, None)
        self.assertEqual(run.call_args[0][0], ["taskkill", "/F", "/T", "/PID", "5"])

    def test_spawn_kwargs_per_platform(self):
        with mock.patch.object(proctree, "IS_WINDOWS", True):
            self.assertIn("creationflags", proctree.spawn_kwargs())
        with mock.patch.object(proctree, "IS_WINDOWS", False):
            self.assertEqual(proctree.spawn_kwargs(), {"start_new_session": True})


if __name__ == "__main__":
    unittest.main()
