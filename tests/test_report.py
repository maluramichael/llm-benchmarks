import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bench import report
from bench.privacy import PrivacyError
from bench.store import ResultStore


def sample_results():
    """A results.json as the runner writes it (hand built, no engine needed)."""
    def ok(values, status="ok"):
        return {"status": status, "values": values, "started_at": "2026-11-03T10:00:00Z",
                "finished_at": "2026-11-03T10:01:00Z", "meta": {}}
    tests = {
        "llamacpp/S/memory/computed": ok({
            "weights_bytes": 2_648_593_216, "kv_bytes_per_token": 32768,
            "curve": [{"ctx": 8192, "bytes": 2_916_000_000}, {"ctx": 32768, "bytes": 3_722_000_000}],
            "note": "upper bound: sliding-window attention keeps a smaller cache, and recurrent "
                    "state of hybrid models is not included; weights plus f16 KV cache",
            "recurrent_state_ignored": True, "sliding_window": False}),
        "llamacpp/S/memory/ctx-8192": ok({"ctx": 8192, "load_s": 3.2, "vram_used_mb": 3100,
                                           "kv_buffer_mb": 256, "gen_tps": 170.5,
                                           "prompt_tps": 7500.2, "ttft_s": 1.1}),
        "llamacpp/S/memory/ctx-32768": ok({"ctx": 32768, "load_s": 4.0, "vram_used_mb": 4200,
                                            "kv_buffer_mb": 1024, "gen_tps": 160.0,
                                            "prompt_tps": 7000.0, "ttft_s": 4.8}),
        "llamacpp/S/memory/ctx-65536": ok({"reason": "only 900 MB would stay free"},
                                           "does_not_fit"),
        "llamacpp/S/speed/cold": ok({"ttft_s": 0.4, "load_s": 2.5, "ttft_incl_load_s": 2.9,
                                      "gen_tps": 178.0, "prompt_tps": 5200.0}),
        "llamacpp/S/speed/warm": ok({"ttft_s": 1.0, "gen_tps": 168.0, "prompt_tps": 7558.0,
                                      "cache_contaminated": False}),
        "llamacpp/S/speed/cache": ok({"ttft_s": 0.065, "first_ttft_s": 1.1, "cached_tokens": 8100,
                                       "cache_hit": True}),
        "llamacpp/S/speed/llama-bench": ok({"pp512_tps": 8115.0, "pp8192_tps": 7844.0,
                                             "tg256_tps": 179.0}),
        "llamacpp/S/speed/llama-bench-deep": ok({"depth": 32768, "pp512_tps": 1457.0,
                                                  "tg256_tps": 63.0}),
        "llamacpp/S/concurrency/p1": ok({"parallel": 1, "total_tps": 98.0, "decode_tps_total": 170.0,
                                          "per_request_tps": 169.0, "ttft_p50_s": 1.1}),
        "llamacpp/S/concurrency/p2": ok({"parallel": 2, "total_tps": 126.0,
                                          "per_request_tps": 124.0, "ttft_p50_s": 2.0}),
        "ollama/S/concurrency/p4": ok({"parallel": 4, "total_tps": 60.0, "per_request_tps": 15.0,
                                        "ttft_p50_s": 3.0, "effective_parallel": 1,
                                        "effective_ctx": 16384, "serialized": True,
                                        "serialized_note": "engine ran 4 requests with parallel 1"}),
        "llamacpp/S/smoke/math-1": ok({"passed": True, "category": "math", "reason": "answer 42",
                                        "raw": "raw/llamacpp-S-math-1.txt"}),
        "llamacpp/S/smoke/math-2": ok({"passed": False, "category": "math",
                                        "reason": "expected 7 | got 8",
                                        "raw": "raw/llamacpp-S-math-2.txt"}),
        "llamacpp/S/smoke/creative-1": ok({"passed": None, "category": "creative",
                                            "reason": "stored only", "raw": "raw/c.txt"}),
        "llamacpp/S/smoke/code-1": ok({"reason": "code execution disabled", "category": "code"},
                                       "skipped"),
        "ollama/S/speed/warm": ok({"error": "EngineError: server gone"}, "error"),
    }
    return {
        "schema_version": 1, "tests": tests,
        "run": {"device": "PC RTX 3090 24 GB", "started_at": "2026-11-03T09:00:00Z",
                "bench_version": "1.0.0"},
        "system": {"os": "Windows 11", "os_version": "10.0.26100", "cpu": "AMD Ryzen 9",
                   "cores": 16, "threads": 32, "ram_gb": 64, "ram_speed_mts": 3600,
                   "gpus": [{"name": "NVIDIA GeForce RTX 3090", "vram_total_mb": 24576,
                             "driver": "591.59", "backend": "cuda"}],
                   "memory_for_models_gb": 24, "uptime_min": 12, "monitors_on_gpu": 1,
                   "power": "ac", "disk_free_gb": 500.5, "laptop": False},
        "engines": {"llamacpp": {"version": "b12061", "backend": "cuda", "flash_attn": "on",
                                  "kv_cache_type": "f16", "thinking": False,
                                  "llama_arg_env_removed": ["LLAMA_ARG_CTX_SIZE"]},
                    "ollama": {"version": "0.40.13", "backend": "cuda", "kv_cache_type": "f16",
                               "flash_attn": "on", "thinking": False}},
        "models": {"llamacpp/S": {"engine": "llamacpp", "id": "S", "cls": "S", "quant": "Q4_K_M",
                                   "size_bytes": 2648593216, "digest": "sha256:abc"},
                   "ollama/S": {"engine": "ollama", "id": "S", "cls": "S", "quant": "Q4_K_M",
                                 "size_bytes": 2648593216, "digest": "sha256:abc"}},
        "notes": ["Measured right after boot.", "Second note"],
        "doctor": [{"check": "uptime", "status": "ok", "text": "up for 12 min"}],
    }


class RenderTest(unittest.TestCase):
    def setUp(self):
        self.text = report.render_run_readme(sample_results())

    def test_sections_in_order(self):
        positions = [self.text.index(h) for h in (
            "# ", "## System", "## Engines", "## llamacpp / S", "### Memory", "### Speed",
            "### Concurrency", "### Smoke", "## ollama / S", "## Notes", "## Summary")]
        self.assertEqual(positions, sorted(positions))

    def test_title_has_device_and_date(self):
        first = self.text.splitlines()[0]
        self.assertIn("PC RTX 3090 24 GB", first)
        self.assertIn("2026-11-03", first)

    def test_system_table(self):
        for expected in ("Windows 11", "AMD Ryzen 9", "64 GB", "NVIDIA GeForce RTX 3090",
                         "24576", "591.59", "10.0.26100"):
            self.assertIn(expected, self.text)

    def test_engine_meta_is_listed(self):
        for expected in ("b12061", "0.40.13", "f16", "LLAMA_ARG_CTX_SIZE", "Flash"):
            self.assertIn(expected, self.text)

    def test_memory_tables_show_every_rung_and_the_failures(self):
        for expected in ("8192", "32768", "65536", "does_not_fit", "only 900 MB would stay free",
                         "3100", "1024"):
            self.assertIn(expected, self.text)

    def test_computed_curve_is_marked_as_upper_bound(self):
        self.assertIn("upper bound", self.text)
        self.assertIn("recurrent", self.text)

    def test_speed_and_llama_bench_values(self):
        for expected in ("178.0", "7558.0", "0.065", "8115.0", "1457.0", "cold", "warm",
                         "llama-bench"):
            self.assertIn(expected, self.text)

    def test_concurrency_includes_the_decode_total_when_present(self):
        self.assertIn("decode_tps_total", self.text)
        self.assertIn("170.0", self.text)
        self.assertIn("126.0", self.text)

    def test_concurrency_shows_effective_parallel_and_serialization(self):
        self.assertIn("Effective parallel", self.text)
        self.assertIn("Serialized", self.text)
        self.assertIn("Effective context", self.text)
        self.assertIn("engine ran 4 requests with parallel 1", self.text)
        row = [l for l in self.text.splitlines() if l.startswith("| 4 |")][0]
        self.assertIn("| 1 | 16384 | yes |", row.replace("  ", " "))

    def test_failed_smoke_task_shows_its_reason(self):
        row = [l for l in self.text.splitlines() if l.startswith("| math-2 |")][0]
        self.assertIn("expected 7", row)
        data = sample_results()
        del data["tests"]["llamacpp/S/smoke/math-2"]["values"]["reason"]
        row = [l for l in report.render_run_readme(data).splitlines()
               if l.startswith("| math-2 |")][0]
        self.assertIn("no reason", row)

    def test_smoke_table_has_results_and_raw_links(self):
        self.assertIn("[raw](raw/llamacpp-S-math-1.txt)", self.text)
        self.assertRegex(self.text, r"math-1 \| math \| pass")
        self.assertRegex(self.text, r"math-2 \| math \| fail")
        self.assertRegex(self.text, r"creative-1 \| creative \| not judged")
        self.assertRegex(self.text, r"code-1 \| code \| skipped")

    def test_table_cells_escape_pipes(self):
        self.assertIn("expected 7 \\| got 8", self.text)

    def test_smoke_totals_per_category(self):
        self.assertIn("math 1/2", self.text)

    def test_errors_are_visible(self):
        self.assertIn("EngineError: server gone", self.text)

    def test_notes_and_summary(self):
        self.assertIn("Measured right after boot.", self.text)
        self.assertIn("Second note", self.text)
        summary = self.text[self.text.index("## Summary"):]
        self.assertIn("32768", summary)  # max context on the GPU
        self.assertIn("168.0", summary)  # warm generation speed

    def test_doctor_rows_are_in_the_notes_section(self):
        self.assertIn("up for 12 min", self.text)

    def test_no_em_dashes_even_when_data_has_them(self):
        data = sample_results()
        data["notes"] = ["a \u2014 b"]
        data["tests"]["llamacpp/S/smoke/math-1"]["values"]["reason"] = "x \u2014 y"
        text = report.render_run_readme(data)
        self.assertNotIn("\u2014", text)

    def test_deterministic(self):
        self.assertEqual(self.text, report.render_run_readme(sample_results()))

    def test_unsafe_raw_links_are_not_linked(self):
        data = sample_results()
        data["tests"]["llamacpp/S/smoke/math-1"]["values"]["raw"] = "../../etc/passwd"
        self.assertNotIn("passwd)", report.render_run_readme(data))

    def test_empty_results_render(self):
        text = report.render_run_readme({"schema_version": 1, "tests": {}})
        self.assertIn("## Summary", text)
        self.assertNotIn("\u2014", text)

    def test_notes_may_be_a_single_string(self):
        data = sample_results()
        data["notes"] = "one note"
        self.assertIn("one note", report.render_run_readme(data))


class WriteReportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run_dir = Path(self.tmp.name) / "2026-11-03-pc"
        self.run_dir.mkdir()
        self.results = self.run_dir / "results.json"

    def write(self, data, bom=False):
        text = json.dumps(data, indent=2)
        self.results.write_text(("\ufeff" if bom else "") + text, encoding="utf-8")

    def test_writes_readme(self):
        self.write(sample_results())
        report.write_report(self.run_dir)
        text = (self.run_dir / "README.md").read_text(encoding="utf-8")
        self.assertIn("## Summary", text)

    def test_bom_is_accepted(self):
        self.write(sample_results(), bom=True)
        report.write_report(self.run_dir)
        self.assertTrue((self.run_dir / "README.md").is_file())

    def test_report_privacy(self):
        data = sample_results()
        data["notes"] = ["measured on nero"]
        self.write(data)
        with mock.patch.object(report, "local_secrets", return_value=["nero"]):
            with self.assertRaises(PrivacyError) as raised:
                report.write_report(self.run_dir)
        self.assertTrue(any("nero" in leak for leak in raised.exception.leaks))
        self.assertFalse((self.run_dir / "README.md").exists())

    def test_leak_only_visible_in_the_rendered_text_is_found(self):
        # "ner\u006f" is "nero" for the renderer but not for a scan of the raw JSON text
        text = json.dumps(sample_results()).replace(
            "Measured right after boot.", "host ner\\u006f")
        self.results.write_text(text, encoding="utf-8")
        self.assertNotIn("nero", text)
        with mock.patch.object(report, "local_secrets", return_value=["nero"]):
            with self.assertRaises(PrivacyError) as raised:
                report.write_report(self.run_dir)
        self.assertTrue(any(leak.startswith("README.md") for leak in raised.exception.leaks))

    def test_leak_in_a_raw_answer(self):
        self.write(sample_results())
        raw = self.run_dir / "raw"
        raw.mkdir()
        (raw / "a.txt").write_text('File "C:\\Users\\somebody\\x.py", line 3', encoding="utf-8")
        with self.assertRaises(PrivacyError) as raised:
            report.write_report(self.run_dir)
        self.assertTrue(any("a.txt" in leak and "somebody" in leak
                            for leak in raised.exception.leaks), raised.exception.leaks)

    def test_leak_in_an_error_value(self):
        data = sample_results()
        data["tests"]["ollama/S/speed/warm"]["values"]["error"] = "failed at 192.168.2.10"
        self.write(data)
        with self.assertRaises(PrivacyError):
            report.write_report(self.run_dir)

    def test_loopback_in_engine_errors_is_fine(self):
        data = sample_results()
        data["tests"]["ollama/S/speed/warm"]["values"]["error"] = (
            "EngineError: cannot reach http://127.0.0.1:11434/api/ps")
        self.write(data)
        with mock.patch.object(report, "local_secrets", return_value=[]):
            report.write_report(self.run_dir)

    def test_old_readme_survives_a_leak(self):
        (self.run_dir / "README.md").write_text("old", encoding="utf-8")
        data = sample_results()
        data["notes"] = ["see 10.1.2.3"]
        self.write(data)
        with self.assertRaises(PrivacyError):
            report.write_report(self.run_dir)
        self.assertEqual((self.run_dir / "README.md").read_text(encoding="utf-8"), "old")

    def test_missing_results_is_a_clear_error(self):
        with self.assertRaises(FileNotFoundError):
            report.write_report(self.run_dir)

    def test_reads_what_the_store_wrote(self):
        store = ResultStore(self.results)
        store.save("llamacpp/S/speed/warm", "ok", {"gen_tps": 12.5})
        report.write_report(self.run_dir)
        self.assertIn("12.5", (self.run_dir / "README.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
