import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from bench import checkers
from bench.checkers import (check, check_code, check_json, check_needle, check_number,
                            check_tool_call, to_store)
from bench.config import Config
from bench.engines.base import ChatResult, Timing

TIMING = Timing(None, 0.1, 10, 100.0, 5, 50.0, None)


def result(text="", calls=None):
    return ChatResult(text, calls or [], TIMING, {})


def task(spec):
    return {"id": "t", "category": "x", "check": spec}


class NumberTest(unittest.TestCase):
    def test_number_pass_fail(self):
        spec = {"type": "number", "answer": "42"}
        self.assertEqual(check_number(spec, result("The answer is 42."))[0], "fail")
        self.assertEqual(check_number(spec, result("ANSWER: 42"))[0], "pass")
        self.assertEqual(check_number(spec, result("ANSWER: 41"))[0], "fail")

    def test_number_formats(self):
        spec = {"type": "number", "answer": "2418"}
        self.assertEqual(check_number(spec, result("so\nANSWER: 2,418"))[0], "pass")
        self.assertEqual(check_number(spec, result("ANSWER: 2418.0"))[0], "pass")
        self.assertEqual(check_number(spec, result("ANSWER: 1\nANSWER: 2418"))[0], "pass")

    def test_thinking_text_is_not_the_answer(self):
        spec = {"type": "number", "answer": "42"}
        text = "<think>maybe ANSWER: 42</think>I do not know."
        self.assertEqual(check_number(spec, result(text))[0], "fail")
        self.assertEqual(check_number(spec, result("<think>ANSWER: 7 and then"))[0], "fail")


class JsonTest(unittest.TestCase):
    SCHEMA = {"type": "object", "required": ["name", "age", "kind"],
              "properties": {"name": {"type": "string"}, "age": {"type": "integer", "minimum": 0},
                             "kind": {"type": "string", "enum": ["a", "b"]},
                             "tags": {"type": "array", "items": {"type": "string"}}}}

    def run_check(self, text, expect=None):
        t = {"json_schema": self.SCHEMA}
        spec = {"type": "json", "expect": expect or {}}
        return check_json(spec, result(text), t)

    def test_json_schema(self):
        self.assertEqual(self.run_check('{"name": "x", "age": 3, "kind": "a"}')[0], "pass")
        self.assertEqual(self.run_check('{"name": "x", "age": 3}')[0], "fail")  # required
        self.assertEqual(self.run_check('{"name": "x", "age": "3", "kind": "a"}')[0], "fail")
        self.assertEqual(self.run_check('{"name": "x", "age": 3, "kind": "c"}')[0], "fail")
        self.assertEqual(self.run_check('{"name": "x", "age": -1, "kind": "a"}')[0], "fail")
        self.assertEqual(
            self.run_check('{"name": "x", "age": 3, "kind": "a", "tags": [1]}')[0], "fail")
        self.assertEqual(self.run_check('{"name": "x", "age": true, "kind": "a"}')[0], "fail")
        self.assertEqual(self.run_check("not json")[0], "fail")

    def test_json_in_code_fence_and_values(self):
        text = 'Here:\n```json\n{"name": "x", "age": 3, "kind": "a"}\n```'
        self.assertEqual(self.run_check(text, {"name": "x"})[0], "pass")
        self.assertEqual(self.run_check(text, {"name": "y"})[0], "fail")

    def test_json_paths(self):
        schema = {"type": "object"}
        spec = {"type": "json", "expect": {"items.1.q": 3, "l": ["a", "b"]}}
        text = '{"items": [{"q": 1}, {"q": 3.0}], "l": ["a", "b"]}'
        self.assertEqual(check_json(spec, result(text), {"json_schema": schema})[0], "pass")
        bad = '{"items": [{"q": 1}], "l": ["a", "b"]}'
        self.assertEqual(check_json(spec, result(bad), {"json_schema": schema})[0], "fail")


class ToolCallTest(unittest.TestCase):
    SPEC = {"type": "tool_call", "name": "get_weather", "args": {"city": "Berlin"}}

    def test_tool_call_args(self):
        ok = result("", [{"name": "get_weather", "arguments": {"city": "Berlin"}}])
        self.assertEqual(check_tool_call(self.SPEC, ok)[0], "pass")
        loose = result("", [{"name": "get_weather", "arguments": {"city": " berlin "}}])
        self.assertEqual(check_tool_call(self.SPEC, loose)[0], "pass")
        wrong_arg = result("", [{"name": "get_weather", "arguments": {"city": "Paris"}}])
        self.assertEqual(check_tool_call(self.SPEC, wrong_arg)[0], "fail")
        wrong_name = result("", [{"name": "other", "arguments": {"city": "Berlin"}}])
        self.assertEqual(check_tool_call(self.SPEC, wrong_name)[0], "fail")
        self.assertEqual(check_tool_call(self.SPEC, result("It is sunny."))[0], "fail")
        two = result("", [{"name": "get_weather", "arguments": {"city": "Berlin"}}] * 2)
        self.assertEqual(check_tool_call(self.SPEC, two)[0], "fail")

    def test_numbers_compare_by_value(self):
        spec = {"type": "tool_call", "name": "c", "args": {"amount": 250}}
        call = result("", [{"name": "c", "arguments": {"amount": 250.0}}])
        self.assertEqual(check_tool_call(spec, call)[0], "pass")
        call = result("", [{"name": "c", "arguments": {"amount": "250"}}])
        self.assertEqual(check_tool_call(spec, call)[0], "fail")

    def test_expect_none(self):
        spec = {"type": "tool_call", "expect_none": True}
        self.assertEqual(check_tool_call(spec, result("Hello!"))[0], "pass")
        extra = result("Hello", [{"name": "get_weather", "arguments": {"city": "x"}}])
        self.assertEqual(check_tool_call(spec, extra)[0], "fail")


class CodeTest(unittest.TestCase):
    SPEC = {"type": "code", "tests": "assert add(2, 3) == 5\n"}

    def code(self, body):
        return result("Sure:\n```python\n" + body + "\n```\nDone.")

    def test_code_runs_hidden_tests(self):
        cfg = Config()
        self.assertEqual(check_code(self.SPEC, self.code("def add(a, b):\n    return a + b"),
                                    cfg)[0], "pass")
        self.assertEqual(check_code(self.SPEC, self.code("def add(a, b):\n    return a - b"),
                                    cfg)[0], "fail")
        self.assertEqual(check_code(self.SPEC, result("no code here"), cfg)[0], "fail")
        self.assertEqual(check_code(self.SPEC, self.code("def add(a, b:"), cfg)[0], "fail")

    def test_code_timeout(self):
        loop = self.code("def add(a, b):\n    while True:\n        pass")
        outcome, reason = check_code(self.SPEC, loop, Config(), timeout=1)
        self.assertEqual(outcome, "fail")
        self.assertEqual(reason, "timeout")
        self.assertEqual(to_store(outcome, reason), ("ok", False))

    def test_code_writes_only_in_its_own_temp_dir(self):
        marker = Path(tempfile.gettempdir()) / "bench_checker_marker_x"
        marker.unlink(missing_ok=True)
        self.addCleanup(lambda: marker.unlink(missing_ok=True))
        body = ("def add(a, b):\n    open('rel_out.txt', 'w').write('x')\n"
                "    open(%r, 'w').write('x')\n    return a + b" % str(Path("/tmp/x_bench_unused")))
        with tempfile.TemporaryDirectory() as cwd:
            import os
            old = os.getcwd()
            os.chdir(cwd)
            try:
                outcome, _ = check_code(self.SPEC, self.code(body), Config())
                self.assertEqual(os.listdir(cwd), [])
            finally:
                os.chdir(old)
        self.assertEqual(outcome, "pass")
        Path("/tmp/x_bench_unused").unlink(missing_ok=True)

    def test_code_does_not_see_secrets_and_closes_stdin(self):
        import os
        os.environ["BENCH_FAKE_TOKEN"] = "s3cret"
        self.addCleanup(os.environ.pop, "BENCH_FAKE_TOKEN", None)
        body = ("import os, sys\ndef add(a, b):\n"
                "    assert 'BENCH_FAKE_TOKEN' not in os.environ\n"
                "    assert sys.stdin.read() == ''\n    return a + b")
        self.assertEqual(check_code(self.SPEC, self.code(body), Config())[0], "pass")

    def test_code_output_is_capped_in_reason(self):
        body = "def add(a, b):\n    print('x' * 10**7)\n    raise ValueError('boom ' + 'y' * 5000)"
        outcome, reason = check_code(self.SPEC, self.code(body), Config())
        self.assertEqual(outcome, "fail")
        self.assertLessEqual(len(reason), 400)

    def test_code_exec_disabled(self):
        cfg = Config(no_code_exec=True)
        outcome, reason = check_code(self.SPEC, self.code("def add(a, b):\n    return a+b"), cfg)
        self.assertEqual(outcome, "skipped")
        self.assertEqual(to_store(outcome, reason)[0], "skipped")


class CodeHardeningTest(unittest.TestCase):
    SPEC = {"type": "code", "tests": "assert add(2, 3) == 5\n"}
    SPAWN = ("import subprocess, sys\n"
             "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n")

    def code(self, body):
        return result("```python\n" + body + "\n```")

    def run_recording(self, body, timeout=10):
        made = []
        real = tempfile.mkdtemp

        def record(*a, **kw):
            made.append(real(*a, **kw))
            return made[-1]
        started = time.monotonic()
        with mock.patch("bench.checkers.tempfile.mkdtemp", side_effect=record):
            outcome = check_code(self.SPEC, self.code(body), Config(), timeout=timeout)
        return outcome, time.monotonic() - started, made

    def test_surviving_grandchild_does_not_hang_normal_exit(self):
        body = self.SPAWN + "def add(a, b):\n    return a + b"
        outcome, elapsed, made = self.run_recording(body)
        self.assertEqual(outcome[0], "pass")
        self.assertLess(elapsed, 10)
        self.assertEqual(len(made), 1)
        self.assertFalse(Path(made[0]).exists())

    def test_surviving_grandchild_does_not_hang_timeout(self):
        body = self.SPAWN + "def add(a, b):\n    while True:\n        pass"
        outcome, elapsed, made = self.run_recording(body, timeout=1)
        self.assertEqual(outcome, ("fail", "timeout"))
        self.assertLess(elapsed, 10)
        self.assertFalse(Path(made[0]).exists())

    def test_temp_dir_is_removed_after_timeout(self):
        body = "def add(a, b):\n    while True:\n        pass"
        _outcome, _elapsed, made = self.run_recording(body, timeout=1)
        self.assertFalse(Path(made[0]).exists())

    def test_pass_token_is_not_readable_by_model_code(self):
        body = ("import os, sys\n"
                "def add(a, b):\n"
                "    open('pass-' + sys.argv[1], 'w').write('ok')\n"
                "    os._exit(0)")
        self.assertEqual(check_code(self.SPEC, self.code(body), Config())[0], "fail")

    def test_os_exit_does_not_fake_a_pass(self):
        body = "import os\nos._exit(0)\ndef add(a, b):\n    return 0"
        self.assertEqual(check_code(self.SPEC, self.code(body), Config())[0], "fail")

    def test_runs_isolated(self):
        body = ("import os, sys\n"
                "assert sys.flags.isolated == 1\n"
                "assert os.getcwd() not in sys.path and '' not in sys.path\n"
                "def add(a, b):\n    return a + b")
        self.assertEqual(check_code(self.SPEC, self.code(body), Config())[0], "pass")


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
        job = checkers._WinJob.open(k32)
        self.assertIsNotNone(job)
        self.assertEqual(k32.calls[:2], ["create", ("set", 77, 9)])
        self.assertTrue(job.assign(1234))
        self.assertTrue(job.kill())
        job.kill()  # idempotent: handle closed once
        self.assertEqual(k32.calls[2:], [("assign", 77, 1234), ("terminate", 77), ("close", 77)])

    def test_job_setup_failure_gives_no_job(self):
        k32 = FakeKernel32(set_ok=0)
        self.assertIsNone(checkers._WinJob.open(k32))
        self.assertIn(("close", 77), k32.calls)

    def test_assign_failure_is_reported(self):
        job = checkers._WinJob.open(FakeKernel32(assign_ok=0))
        self.assertFalse(job.assign(5))

    def test_kill_tree_uses_job_on_windows(self):
        proc, job = mock.Mock(pid=5), mock.Mock()
        job.kill.return_value = True
        with mock.patch.object(checkers, "IS_WINDOWS", True), \
                mock.patch.object(checkers.subprocess, "run") as run:
            checkers._kill_tree(proc, job)
        job.kill.assert_called_once()
        run.assert_not_called()

    def test_kill_tree_falls_back_to_taskkill(self):
        proc = mock.Mock(pid=5)
        with mock.patch.object(checkers, "IS_WINDOWS", True), \
                mock.patch.object(checkers.subprocess, "run") as run:
            checkers._kill_tree(proc, None)
        self.assertEqual(run.call_args[0][0], ["taskkill", "/F", "/T", "/PID", "5"])

    def test_spawn_kwargs_per_platform(self):
        with mock.patch.object(checkers, "IS_WINDOWS", True):
            self.assertIn("creationflags", checkers._spawn_kwargs())
        with mock.patch.object(checkers, "IS_WINDOWS", False):
            self.assertEqual(checkers._spawn_kwargs(), {"start_new_session": True})


class FixTest(unittest.TestCase):
    def test_malformed_numbers_fail(self):
        spec = {"type": "number", "answer": "12"}
        for text in ("ANSWER: 12,34", "ANSWER: 12%", "ANSWER: 12abc", "ANSWER: 12,5"):
            self.assertEqual(check_number(spec, result(text))[0], "fail", text)
        self.assertEqual(check_number(spec, result("ANSWER: 12."))[0], "pass")
        self.assertEqual(check_number(spec, result("ANSWER: 12 dollars"))[0], "pass")

    def test_enum_is_type_aware(self):
        from bench.checkers import validate_schema
        self.assertIsNotNone(validate_schema(1, {"enum": [True]}))
        self.assertIsNotNone(validate_schema(True, {"enum": [1]}))
        self.assertIsNone(validate_schema(True, {"enum": [True]}))
        self.assertIsNone(validate_schema("a", {"enum": ["a", "b"]}))
        self.assertIsNone(validate_schema(2, {"enum": [1, 2.0]}))


class NeedleTest(unittest.TestCase):
    def test_needle(self):
        spec = {"type": "needle", "answer": "7391-QZ"}
        self.assertEqual(check_needle(spec, result("The code is 7391-qz."))[0], "pass")
        self.assertEqual(check_needle(spec, result("I could not find it."))[0], "fail")
        self.assertEqual(check_needle(spec, result("<think>7391-QZ?</think>unknown"))[0], "fail")


class DispatchTest(unittest.TestCase):
    def test_check_dispatch_and_unchecked(self):
        cfg = Config()
        t = task({"type": "none"})
        self.assertEqual(check(t, result("a story"), cfg)[0], "unchecked")
        t = task({"type": "number", "answer": "1"})
        self.assertEqual(check(t, result("ANSWER: 1"), cfg)[0], "pass")
        with self.assertRaises(ValueError):
            check(task({"type": "bogus"}), result(""), cfg)

    def test_to_store_mapping(self):
        self.assertEqual(to_store("pass", ""), ("ok", True))
        self.assertEqual(to_store("fail", "x"), ("ok", False))
        self.assertEqual(to_store("unchecked", ""), ("ok", None))
        self.assertEqual(to_store("skipped", "x"), ("skipped", None))


if __name__ == "__main__":
    unittest.main()
