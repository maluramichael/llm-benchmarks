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


class SuspendedStartTest(unittest.TestCase):
    """The child starts suspended, joins the job, and only then runs (no escape race)."""

    def child(self):
        child = mock.Mock(pid=77)
        child._handle = 1234
        return child

    def test_suspended_flag_is_added_to_the_creation_flags(self):
        with mock.patch.object(proctree, "IS_WINDOWS", True), \
                mock.patch.object(proctree.subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200,
                                  create=True):
            self.assertFalse(proctree.spawn_kwargs()["creationflags"] & 0x4)
            flags = proctree.spawn_kwargs(suspended=True)["creationflags"]
        self.assertTrue(flags & 0x4)
        self.assertTrue(flags & 0x200)  # the other flags stay

    def test_assign_happens_before_resume(self):
        order = []
        job = mock.Mock()
        job.assign.side_effect = lambda handle: order.append(("assign", handle)) or True
        child = self.child()
        with mock.patch.object(proctree, "IS_WINDOWS", True), \
                mock.patch.object(proctree.WinJob, "open", return_value=job), \
                mock.patch.object(proctree, "resume_process",
                                  side_effect=lambda p: order.append(("resume", p)) or True):
            self.assertIs(proctree.attach_job(child, suspended=True), job)
        self.assertEqual(order, [("assign", 1234), ("resume", child)])

    def test_failed_assign_kills_the_suspended_child_and_raises(self):
        job = mock.Mock()
        job.assign.return_value = False
        child = self.child()
        with mock.patch.object(proctree, "IS_WINDOWS", True), \
                mock.patch.object(proctree.WinJob, "open", return_value=job), \
                mock.patch.object(proctree, "resume_process") as resume:
            with self.assertRaises(OSError):
                proctree.attach_job(child, suspended=True)
        resume.assert_not_called()  # it never runs outside the job
        child.kill.assert_called_once()
        job.kill.assert_called_once()

    def test_failed_resume_kills_job_and_child_and_raises(self):
        job = mock.Mock()
        job.assign.return_value = True
        child = self.child()
        with mock.patch.object(proctree, "IS_WINDOWS", True), \
                mock.patch.object(proctree.WinJob, "open", return_value=job), \
                mock.patch.object(proctree, "resume_process", return_value=False):
            with self.assertRaises(OSError):
                proctree.attach_job(child, suspended=True)
        job.kill.assert_called_once()
        child.kill.assert_called_once()

    def test_without_the_job_api_the_child_is_resumed_and_taskkill_stays_the_fallback(self):
        child = self.child()
        with mock.patch.object(proctree, "IS_WINDOWS", True), \
                mock.patch.object(proctree.WinJob, "open", return_value=None), \
                mock.patch.object(proctree, "resume_process", return_value=True) as resume:
            self.assertIsNone(proctree.attach_job(child, suspended=True))
        resume.assert_called_once_with(child)

    def test_unsuspended_use_is_unchanged(self):
        job = mock.Mock()
        job.assign.return_value = True
        with mock.patch.object(proctree, "IS_WINDOWS", True), \
                mock.patch.object(proctree.WinJob, "open", return_value=job), \
                mock.patch.object(proctree, "resume_process") as resume:
            self.assertIs(proctree.attach_job(self.child()), job)
        resume.assert_not_called()

    def test_resume_calls_ntresumeprocess_with_the_process_handle(self):
        ntdll = mock.Mock()
        ntdll.NtResumeProcess.return_value = 0  # STATUS_SUCCESS
        self.assertTrue(proctree.resume_process(self.child(), ntdll=ntdll))
        ntdll.NtResumeProcess.assert_called_once()
        self.assertEqual(ntdll.NtResumeProcess.call_args.args[0].value, 1234)
        ntdll.NtResumeProcess.return_value = 0xC0000008  # an error status
        self.assertFalse(proctree.resume_process(self.child(), ntdll=ntdll))
        ntdll.NtResumeProcess.side_effect = OSError
        self.assertFalse(proctree.resume_process(self.child(), ntdll=ntdll))


if __name__ == "__main__":
    unittest.main()
