import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bench.store import FINAL, RETRY, ResultStore, StoreError

KEY = "ollama/S/speed/warm"


class StoreTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.path = self.dir / "results.json"

    def on_disk(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8"))


class StatusTest(StoreTestCase):
    def test_constants(self):
        self.assertEqual(FINAL, {"ok", "does_not_fit", "skipped"})
        self.assertEqual(RETRY, {"error", "aborted"})

    def test_missing_file_is_empty_store(self):
        store = ResultStore(self.path)
        self.assertEqual(store.data["schema_version"], 1)
        self.assertEqual(store.data["tests"], {})
        self.assertIsNone(store.get(KEY))
        self.assertTrue(store.should_run(KEY))
        self.assertFalse(self.path.exists())

    def test_skip_final(self):
        for status in sorted(FINAL):
            with self.subTest(status=status):
                store = ResultStore(self.path)
                store.save(KEY, status, {"tps": 1.0})
                self.assertFalse(store.should_run(KEY))
                self.assertFalse(ResultStore(self.path).should_run(KEY))

    def test_retry_error(self):
        for status in sorted(RETRY):
            with self.subTest(status=status):
                store = ResultStore(self.path)
                store.save(KEY, status, {})
                self.assertTrue(store.should_run(KEY))

    def test_running_is_not_final(self):
        store = ResultStore(self.path)
        store.mark_running(KEY, {})
        self.assertTrue(store.should_run(KEY))

    def test_force_and_pattern(self):
        store = ResultStore(self.path)
        small, medium = "ollama/S/speed/warm", "ollama/M/speed/warm"
        store.save(small, "ok", {})
        store.save(medium, "ok", {})
        self.assertFalse(store.should_run(small))
        self.assertTrue(store.should_run(small, force=True))
        patterns = ["*/S/speed/*"]
        self.assertTrue(store.should_run(small, force_patterns=patterns))
        self.assertFalse(store.should_run(medium, force_patterns=patterns))
        self.assertFalse(store.should_run(medium, False, []))

    def test_save_rejects_running_and_unknown_status(self):
        store = ResultStore(self.path)
        for status in ("running", "bogus"):
            with self.assertRaises(ValueError):
                store.save(KEY, status, {})
        self.assertIsNone(store.get(KEY))


class EntryTest(StoreTestCase):
    def test_entry_shape(self):
        store = ResultStore(self.path)
        store.mark_running(KEY, {"model": "S"})
        running = store.get(KEY)
        self.assertEqual(running["status"], "running")
        self.assertEqual(running["meta"], {"model": "S"})
        self.assertIsNone(running["finished_at"])
        self.assertRegex(running["started_at"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
        store.save(KEY, "ok", {"tps": 12.5}, {"engine": "x"})
        done = store.get(KEY)
        self.assertEqual(done["status"], "ok")
        self.assertEqual(done["values"], {"tps": 12.5})
        self.assertEqual(done["meta"], {"model": "S", "engine": "x"})
        self.assertEqual(done["started_at"], running["started_at"])
        self.assertRegex(done["finished_at"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")

    def test_save_without_mark_running(self):
        store = ResultStore(self.path)
        store.save(KEY, "skipped", {}, None)
        entry = store.get(KEY)
        self.assertEqual(entry["meta"], {})
        self.assertIsNotNone(entry["started_at"])

    def test_rerun_starts_fresh(self):
        store = ResultStore(self.path)
        store.save(KEY, "ok", {"tps": 1}, {"old": True})
        store.mark_running(KEY, {"new": True})
        entry = store.get(KEY)
        self.assertEqual(entry["status"], "running")
        self.assertEqual(entry["values"], {})
        self.assertEqual(entry["meta"], {"new": True})

    def test_set_section(self):
        store = ResultStore(self.path)
        store.set_section("system", {"os": "linux"})
        store.set_section("notes", ["a"])
        self.assertEqual(self.on_disk()["system"], {"os": "linux"})
        self.assertEqual(ResultStore(self.path).data["notes"], ["a"])
        for name in ("tests", "schema_version"):
            with self.assertRaises(ValueError):
                store.set_section(name, {})

    def test_values_are_normalized_like_disk(self):
        store = ResultStore(self.path)
        store.save(KEY, "ok", {"t": (1, 2)})
        self.assertEqual(store.get(KEY)["values"], {"t": [1, 2]})

    def test_save_does_not_alias_caller_objects(self):
        store = ResultStore(self.path)
        values = {"a": [1]}
        store.save(KEY, "ok", values)
        values["a"].append(2)
        self.assertEqual(store.get(KEY)["values"], {"a": [1]})


class CrashRecoveryTest(StoreTestCase):
    def test_crash_recovery(self):
        ResultStore(self.path).mark_running(KEY, {})
        fresh = ResultStore(self.path)
        self.assertEqual(fresh.get(KEY)["status"], "running")
        self.assertEqual(fresh.recover_crashed(), [KEY])
        self.assertEqual(fresh.get(KEY)["status"], "aborted")
        self.assertEqual(self.on_disk()["tests"][KEY]["status"], "aborted")
        self.assertTrue(fresh.should_run(KEY))
        self.assertEqual(fresh.recover_crashed(), [])

    def test_recover_leaves_finished_entries(self):
        store = ResultStore(self.path)
        store.save("a", "ok", {"x": 1})
        store.mark_running("b", {})
        self.assertEqual(store.recover_crashed(), ["b"])
        self.assertEqual(store.get("a")["status"], "ok")

    def test_recover_without_crash_does_not_write(self):
        store = ResultStore(self.path)
        self.assertEqual(store.recover_crashed(), [])
        self.assertFalse(self.path.exists())


class PersistenceTest(StoreTestCase):
    def test_persist_after_each_save(self):
        store = ResultStore(self.path)
        store.mark_running("a", {})
        self.assertEqual(self.on_disk()["tests"]["a"]["status"], "running")
        store.save("a", "ok", {"n": 1})
        self.assertEqual(self.on_disk()["tests"]["a"]["values"], {"n": 1})
        store.save("b", "error", {"reason": "boom"})
        disk = self.on_disk()
        self.assertEqual(sorted(disk["tests"]), ["a", "b"])
        self.assertEqual(disk["schema_version"], 1)
        self.assertEqual(disk, store.data)

    def test_file_format(self):
        store = ResultStore(self.path)
        store.save("b", "ok", {"z": 1, "a": 2})
        text = self.path.read_text(encoding="utf-8")
        self.assertTrue(text.endswith("\n"))
        self.assertIn('\n  "schema_version": 1', text)
        self.assertLess(text.index('"a": 2'), text.index('"z": 1'))

    def test_creates_parent_directory(self):
        nested = self.dir / "runs" / "x" / "results.json"
        ResultStore(nested).save(KEY, "ok", {})
        self.assertTrue(nested.is_file())

    def test_no_tmp_file_left_behind(self):
        ResultStore(self.path).save(KEY, "ok", {})
        self.assertEqual([p.name for p in self.dir.iterdir()], ["results.json"])

    def test_fsync_before_replace(self):
        calls = []
        real_fsync, real_replace = os.fsync, os.replace
        with mock.patch("bench.store.os.fsync",
                        side_effect=lambda fd: (calls.append("fsync"), real_fsync(fd))[1]), \
             mock.patch("bench.store.os.replace",
                        side_effect=lambda a, b: (calls.append("replace"), real_replace(a, b))[1]):
            ResultStore(self.path).save(KEY, "ok", {})
        self.assertGreaterEqual(calls.count("fsync"), 1)
        self.assertEqual(calls.index("fsync") < calls.index("replace"), True)


class AtomicWriteTest(StoreTestCase):
    def test_atomic_write(self):
        store = ResultStore(self.path)
        store.save("a", "ok", {"n": 1})
        before = self.path.read_bytes()
        with mock.patch("bench.store.os.replace", side_effect=OSError("power lost")):
            with self.assertRaises(OSError):
                store.save("b", "ok", {"n": 2})
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(sorted(self.on_disk()["tests"]), ["a"])
        # No half written leftovers next to the result file.
        self.assertEqual([p.name for p in self.dir.iterdir()], ["results.json"])

    def test_failure_while_writing_keeps_old_file(self):
        store = ResultStore(self.path)
        store.save("a", "ok", {"n": 1})
        before = self.path.read_bytes()
        with mock.patch("bench.store.os.fsync", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                store.save("b", "ok", {"n": 2})
        self.assertEqual(self.path.read_bytes(), before)

    def test_failed_write_loses_nothing_later(self):
        store = ResultStore(self.path)
        with mock.patch("bench.store.os.replace", side_effect=OSError("busy")):
            with self.assertRaises(OSError):
                store.save("a", "ok", {"n": 1})
        # The result stays in memory and goes to disk with the next write.
        self.assertEqual(store.get("a")["status"], "ok")
        store.save("b", "ok", {"n": 2})
        self.assertEqual(sorted(self.on_disk()["tests"]), ["a", "b"])

    def test_unserializable_value_is_rejected_without_poisoning_store(self):
        store = ResultStore(self.path)
        store.save("a", "ok", {"n": 1})
        for bad in ({"x": object()}, {"x": float("nan")}):
            with self.assertRaises(ValueError):
                store.save("b", "ok", bad)
        with self.assertRaises(ValueError):
            store.set_section("system", {"x": object()})
        self.assertIsNone(store.get("b"))
        self.assertNotIn("system", store.data)
        store.save("c", "ok", {})
        self.assertEqual(sorted(self.on_disk()["tests"]), ["a", "c"])


class CorruptFileTest(StoreTestCase):
    def assert_refused(self, content: str):
        self.path.write_text(content, encoding="utf-8")
        with self.assertRaises(StoreError) as ctx:
            ResultStore(self.path)
        self.assertIn(str(self.path), str(ctx.exception))
        self.assertEqual(self.path.read_text(encoding="utf-8"), content)

    def test_truncated_json(self):
        self.assert_refused('{"schema_version": 1, "tests": {"a": ')

    def test_empty_file(self):
        self.assert_refused("")

    def test_not_an_object(self):
        self.assert_refused("[1, 2]")

    def test_wrong_schema_version(self):
        self.assert_refused('{"schema_version": 2, "tests": {}}')

    def test_tests_not_an_object(self):
        self.assert_refused('{"schema_version": 1, "tests": []}')

    def test_unreadable_bytes(self):
        self.path.write_bytes(b"\xff\xfe\x00bad")
        with self.assertRaises(StoreError) as ctx:
            ResultStore(self.path)
        self.assertIn(str(self.path), str(ctx.exception))

    def test_missing_tests_key_is_filled(self):
        self.path.write_text('{"schema_version": 1, "system": {"a": 1}}', encoding="utf-8")
        store = ResultStore(self.path)
        self.assertEqual(store.data["tests"], {})
        self.assertEqual(store.data["system"], {"a": 1})


if __name__ == "__main__":
    unittest.main()
