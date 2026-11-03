import unittest
from pathlib import Path
from unittest import mock

from bench import awake

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


class WindowsTest(unittest.TestCase):
    def test_sets_and_resets_execution_state(self):
        kernel32 = mock.Mock()
        kernel32.SetThreadExecutionState.return_value = ES_CONTINUOUS
        with awake.keep_awake(platform="win32", kernel32=kernel32) as method:
            self.assertEqual(method, "SetThreadExecutionState")
            kernel32.SetThreadExecutionState.assert_called_once_with(
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        self.assertEqual(kernel32.SetThreadExecutionState.call_args_list[-1],
                         mock.call(ES_CONTINUOUS))

    def test_reset_also_after_an_error(self):
        kernel32 = mock.Mock()
        kernel32.SetThreadExecutionState.return_value = 1
        with self.assertRaises(RuntimeError):
            with awake.keep_awake(platform="win32", kernel32=kernel32):
                raise RuntimeError("boom")
        self.assertEqual(kernel32.SetThreadExecutionState.call_args_list[-1],
                         mock.call(ES_CONTINUOUS))

    def test_failed_call_warns_and_runs_on(self):
        kernel32 = mock.Mock()
        kernel32.SetThreadExecutionState.return_value = 0
        said = []
        with awake.keep_awake(platform="win32", kernel32=kernel32, say=said.append) as method:
            self.assertIsNone(method)
        self.assertTrue(said)


class MacTest(unittest.TestCase):
    def test_caffeinate_waits_for_our_process_and_is_terminated(self):
        child = mock.Mock()
        popen = mock.Mock(return_value=child)
        with awake.keep_awake(platform="darwin", popen=popen, pid=4242,
                              which=lambda name: "/usr/bin/caffeinate") as method:
            self.assertEqual(method, "caffeinate")
            argv = popen.call_args.args[0]
            self.assertEqual(argv, ["/usr/bin/caffeinate", "-i", "-w", "4242"])
            child.terminate.assert_not_called()
        child.terminate.assert_called_once()
        child.wait.assert_called()

    def test_missing_caffeinate_is_not_an_error(self):
        with awake.keep_awake(platform="darwin", which=lambda name: None) as method:
            self.assertIsNone(method)


class LinuxTest(unittest.TestCase):
    def test_systemd_inhibit_blocks_idle_and_sleep(self):
        child = mock.Mock()
        popen = mock.Mock(return_value=child)
        with awake.keep_awake(platform="linux", popen=popen, pid=77,
                              which=lambda name: "/usr/bin/" + name) as method:
            self.assertEqual(method, "systemd-inhibit")
            argv = popen.call_args.args[0]
        self.assertEqual(argv[0], "/usr/bin/systemd-inhibit")
        self.assertIn("--what=idle:sleep", argv)
        self.assertIn("--pid=77", argv)  # the helper ends with us even if we are killed
        child.terminate.assert_called_once()

    def test_without_systemd_inhibit_nothing_happens(self):
        popen = mock.Mock()
        with awake.keep_awake(platform="linux", popen=popen, which=lambda name: None) as method:
            self.assertIsNone(method)
        popen.assert_not_called()

    def test_start_failure_is_a_warning(self):
        said = []
        with awake.keep_awake(platform="linux", popen=mock.Mock(side_effect=OSError("no")),
                              which=lambda name: "/x/" + name, say=said.append) as method:
            self.assertIsNone(method)
        self.assertTrue(said)

    def test_other_platforms_do_nothing(self):
        with awake.keep_awake(platform="freebsd") as method:
            self.assertIsNone(method)


class NoPersistentChangeTest(unittest.TestCase):
    def test_module_never_touches_power_settings(self):
        source = Path(awake.__file__).read_text(encoding="utf-8").lower()
        for forbidden in ("powercfg", "pmset", "gsettings", "registry", "winreg"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
