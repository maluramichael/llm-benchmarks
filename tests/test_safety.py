import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from bench import safety
from bench.config import Config
from bench.safety import Guard, Reading, doctor, ladder_start
from bench.store import ResultStore

NS = "xmlns='http://schemas.microsoft.com/win/2004/08/events/event'"


def event(provider, event_id, when):
    return (f"<Event {NS}><System><Provider Name='{provider}'/><EventID Qualifiers='0'>{event_id}"
            f"</EventID><TimeCreated SystemTime='{when}'/><Channel>System</Channel>"
            f"<Computer>host</Computer></System><EventData><Data>x</Data></EventData></Event>")


def reading(**kw):
    base = dict(vram_free_mb=10000, gpu_fraction=1.0, shared_mb=100, swap_mb=500)
    base.update(kw)
    return Reading(**base)


class GuardTest(unittest.TestCase):
    def test_reserve_small_card(self):
        self.assertEqual(Guard(Config(), 0, 4096).reserve_mb(), 512)
        self.assertEqual(Guard(Config(), 0, 24576).reserve_mb(), 1536)

    def test_before_step_blocks(self):
        ok, why = Guard(Config(), 0, 8192).before_step(2000, reading(vram_free_mb=3000))
        self.assertFalse(ok)
        self.assertIn("reserve", why)
        ok, _ = Guard(Config(), 0, 8192).before_step(2000, reading(vram_free_mb=4000))
        self.assertTrue(ok)

    def test_after_step_shared_growth(self):
        guard = Guard(Config(), 0, 8192)
        ok, why = guard.after_step(reading(shared_mb=100), reading(shared_mb=700))
        self.assertFalse(ok)
        self.assertIn("shared", why.lower())
        self.assertTrue(guard.after_step(reading(shared_mb=100), reading(shared_mb=500))[0])

    def test_after_step_swap(self):
        ok, why = Guard(Config(), 0, 8192).after_step(reading(swap_mb=0), reading(swap_mb=1200))
        self.assertFalse(ok)
        self.assertIn("swap", why.lower())

    def test_partial_gpu(self):
        ok, why = Guard(Config(), 0, 8192).after_step(reading(), reading(gpu_fraction=0.94))
        self.assertFalse(ok)
        self.assertIn("GPU", why)

    def test_unknown_values_do_not_abort(self):
        none = Reading(None, None, None, None)
        self.assertTrue(Guard(Config(), 0, 8192).after_step(none, none)[0])

    def test_driver_reset_events(self):
        out = event("nvlddmkm", 153, "2026-10-30T10:00:00.1Z") + event("Display", 4101, "2026-10-31T10:00:00Z")
        guard = Guard(Config(), 0, 8192)
        ts = datetime(2026, 10, 1, tzinfo=timezone.utc)
        with mock.patch.object(safety, "IS_WINDOWS", True), \
                mock.patch.object(safety, "run_text", return_value=out) as run, \
                mock.patch.object(safety, "WATCHDOG_DIR", Path("/nonexistent")):
            self.assertTrue(guard.driver_reset_since(ts))
        args = run.call_args[0][0]
        self.assertEqual(args[:3], ["wevtutil", "qe", "System"])
        self.assertIn("2026-10-01T00:00:00", " ".join(args))

    def test_driver_reset_none_or_failure(self):
        guard = Guard(Config(), 0, 8192)
        ts = datetime(2026, 10, 1, tzinfo=timezone.utc)
        for out in ("", None, "<garbage"):
            with mock.patch.object(safety, "IS_WINDOWS", True), \
                    mock.patch.object(safety, "run_text", return_value=out), \
                    mock.patch.object(safety, "WATCHDOG_DIR", Path("/nonexistent")):
                self.assertFalse(guard.driver_reset_since(ts))

    def test_driver_reset_watchdog_dump(self):
        ts = datetime(2026, 10, 1, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            dump = Path(tmp, "WATCHDOG-20261030-1000.dmp")
            dump.write_bytes(b"x")
            with mock.patch.object(safety, "IS_WINDOWS", True), \
                    mock.patch.object(safety, "run_text", return_value=""), \
                    mock.patch.object(safety, "WATCHDOG_DIR", Path(tmp)):
                self.assertTrue(Guard(Config(), 0, 8192).driver_reset_since(ts))
                later = datetime.fromtimestamp(dump.stat().st_mtime + 100, timezone.utc)
                self.assertFalse(Guard(Config(), 0, 8192).driver_reset_since(later))

    def test_driver_reset_not_windows(self):
        with mock.patch.object(safety, "IS_WINDOWS", False):
            self.assertFalse(Guard(Config(), 0, 8192).driver_reset_since(datetime(2020, 1, 1)))


class ParseTest(unittest.TestCase):
    def test_parse_adapter_shared(self):
        mb = 1024 * 1024
        rows = [
            {"Path": "x\\gpu adapter memory(luid_0x0_0xa)\\shared usage",
             "InstanceName": "luid_0x0_0xa", "CookedValue": 80 * mb},
            {"Path": "x\\gpu adapter memory(luid_0x0_0xb)\\shared usage",
             "InstanceName": "luid_0x0_0xb", "CookedValue": 8192},
            {"Path": "x\\gpu adapter memory(luid_0x0_0xa)\\dedicated usage",
             "InstanceName": "luid_0x0_0xa", "CookedValue": 1500 * mb},
            {"Path": "x\\gpu adapter memory(luid_0x0_0xb)\\dedicated usage",
             "InstanceName": "luid_0x0_0xb", "CookedValue": 0},
        ]
        self.assertEqual(safety.parse_adapter_shared_mb(json.dumps(rows)), 80)
        self.assertIsNone(safety.parse_adapter_shared_mb("nope"))
        self.assertIsNone(safety.parse_adapter_shared_mb("[]"))


class LadderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = ResultStore(Path(self.tmp.name) / "results.json")

    def test_fresh_store_starts_at_lowest_sorted_deduplicated(self):
        self.assertEqual(ladder_start(self.store, "m/", [16384, 8192, 8192]), 8192)

    def test_ok_steps_are_passed(self):
        self.store.save("m/ctx-8192", "ok", {})
        self.assertEqual(ladder_start(self.store, "m/", [16384, 8192]), 16384)

    def test_all_done(self):
        self.store.save("m/ctx-8192", "ok", {})
        self.assertIsNone(ladder_start(self.store, "m/", [8192]))

    def test_ladder_after_crash(self):
        self.store.save("m/ctx-32768", "aborted", {})
        steps = [8192, 16384, 32768, 65536, 131072]
        self.assertIsNone(ladder_start(self.store, "m/", steps))
        self.assertEqual(self.store.get("m/ctx-32768")["status"], "does_not_fit")
        for n in (65536, 131072):
            entry = self.store.get(f"m/ctx-{n}")
            self.assertEqual(entry["status"], "skipped")
            self.assertTrue(entry["values"]["reason"])
        self.assertIsNone(self.store.get("m/ctx-16384"))
        # second call stays blocked and is stable
        self.assertIsNone(ladder_start(self.store, "m/", steps))

    def test_does_not_fit_blocks(self):
        self.store.save("m/ctx-16384", "does_not_fit", {})
        self.assertIsNone(ladder_start(self.store, "m/", [8192, 16384, 32768]))
        self.assertEqual(self.store.get("m/ctx-32768")["status"], "skipped")

    def test_other_prefix_unaffected(self):
        self.store.save("a/ctx-8192", "aborted", {})
        self.assertEqual(ladder_start(self.store, "b/", [8192]), 8192)


def win_probe(uptime=20.0):
    return {
        "uptime_min": uptime,
        "tdr": {"TdrDelay": None, "TdrDdiDelay": None, "TdrLevel": None, "HwSchMode": None},
        "monitors": [{"Active": True}],
        "processes": [{"ProcessName": "big", "WorkingSet64": 6 * 1024 ** 3},
                      {"ProcessName": "small", "WorkingSet64": 1024 ** 2}],
        "pagefile": {"CurrentUsage": 490, "PeakUsage": 1697},
    }


class DoctorTest(unittest.TestCase):
    def by_check(self, rows):
        return {row["check"]: row for row in rows}

    def test_doctor_uptime(self):
        rows = self.by_check(doctor(Config(), platform="win32", probe=lambda: win_probe(300)))
        self.assertEqual(rows["uptime"]["status"], "warn")
        rows = self.by_check(doctor(Config(), platform="win32", probe=lambda: win_probe(20)))
        self.assertEqual(rows["uptime"]["status"], "ok")

    def test_windows_rows(self):
        rows = self.by_check(doctor(Config(), platform="win32", probe=win_probe))
        self.assertEqual(rows["processes"]["status"], "warn")
        self.assertIn("big", rows["processes"]["text"])
        self.assertEqual(rows["tdr"]["status"], "info")
        self.assertIn("default", rows["tdr"]["text"])
        self.assertIn("monitors", rows)
        self.assertIn("sysmem_fallback", rows)
        self.assertIn("power_supply", rows)
        for row in rows.values():
            self.assertIn(row["status"], ("ok", "warn", "info"))
            self.assertNotIn("\u2014", row["text"])

    def test_probe_failure_is_info(self):
        for probe in (lambda: None, mock.Mock(side_effect=RuntimeError("boom"))):
            rows = self.by_check(doctor(Config(), platform="win32", probe=probe))
            self.assertEqual(rows["uptime"]["status"], "info")
            self.assertIn("could not", rows["uptime"]["text"])

    def test_not_windows(self):
        rows = self.by_check(doctor(Config(), platform="darwin"))
        self.assertEqual(rows["uptime"]["status"], "info")
        self.assertIn("not available on this platform", rows["uptime"]["text"])

    def test_parse_probe(self):
        raw = json.dumps({"uptime_min": 22.5, "tdr": {"TdrDelay": 10}, "monitors": {"Active": True},
                          "processes": {"ProcessName": "a", "WorkingSet64": 5},
                          "pagefile": None})
        parsed = safety.parse_probe(raw)
        self.assertEqual(parsed["monitors"], [{"Active": True}])
        self.assertEqual(len(parsed["processes"]), 1)
        self.assertIsNone(safety.parse_probe("garbage"))


if __name__ == "__main__":
    unittest.main()
