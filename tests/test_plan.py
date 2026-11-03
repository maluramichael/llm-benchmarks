import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bench import plan
from bench.config import Config
from bench.engines.base import ModelRef
from bench.models import ModelNotFound
from bench.store import ResultStore
from gguf_writer import write_gguf

GIB = 1024**3
# 32 attention layers, 8 KV heads of 128: 128 KiB of KV cache per token, 1 GiB at 8k.
DENSE = {"general.architecture": "llama", "llama.block_count": 32,
         "llama.attention.head_count_kv": 8, "llama.attention.key_length": 128}
SIZES = {"S": 2_648_593_216, "M": 9_276_184_896, "L": 16_810_714_464}


def card(total_mb, free_mb=None):
    return {"gpus": [{"index": 0, "name": "Test GPU", "vram_total_mb": total_mb,
                      "vram_free_mb": free_mb}], "memory_for_models_gb": total_mb / 1024,
            "ram_gb": 64}


class PlanTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.gguf = write_gguf(self.root / "m.gguf", dict(DENSE))
        self.calls = []

    def resolver(self, with_file=True, extra=None):
        def resolve(model_id, engine, cfg, **_):
            self.calls.append((model_id, engine, cfg.dry_run))
            if model_id not in SIZES:
                raise ModelNotFound(f"model {model_id} is not in the catalog")
            path = (extra or self.gguf) if with_file else None
            return ModelRef(model_id, model_id, "Q4_K_M", "tag:" + model_id.lower(), path,
                            "sha256:x" if path else None, SIZES[model_id])
        return resolve

    def make(self, system, models=("S", "M", "L"), engines=("llamacpp",), store=None, **cfg):
        cfg = Config(engines=list(engines), models=list(models), **cfg)
        return plan.make_plan(cfg, system, store=store, resolve=self.resolver())

    def test_plan_fit(self):
        big = {e["model"]: e for e in self.make(card(24576))}
        self.assertTrue(big["L"]["fits"], big["L"]["reason"])
        small = {e["model"]: e for e in self.make(card(8192))}
        self.assertTrue(small["S"]["fits"])
        self.assertFalse(small["M"]["fits"])
        self.assertIn("8k", small["M"]["reason"])
        self.assertIn("8 GiB", small["M"]["reason"])  # what the card has (binary unit)
        self.assertNotIn(" GB", small["M"]["reason"].replace("23.4 GB", ""))
        self.assertIn("reserve", small["M"]["reason"])

    def test_reserve_decides_the_edge(self):
        # M needs 8.64 GiB weights + 1 GiB KV at 8k = 9.64 GiB; the 1.5 GiB reserve on top.
        edge = int(9.64 * 1024 + 1536)
        self.assertTrue(self.make(card(edge + 40), models=("M",))[0]["fits"])
        self.assertFalse(self.make(card(edge - 40), models=("M",))[0]["fits"])

    def test_free_vram_is_used_when_known(self):
        entry = self.make(card(24576, free_mb=8000), models=("L",))[0]
        self.assertFalse(entry["fits"])

    def test_small_card_uses_the_small_reserve(self):
        entry = self.make(card(4096), models=("S",))[0]  # 2.47 + 1 + 0.5 < 4
        self.assertTrue(entry["fits"], entry["reason"])

    def test_entry_shape_and_order(self):
        entries = self.make(card(24576), models=("S", "M"), engines=("llamacpp", "ollama"))
        self.assertEqual([(e["engine"], e["model"]) for e in entries],
                         [("llamacpp", "S"), ("llamacpp", "M"), ("ollama", "S"),
                          ("ollama", "M")])
        for entry in entries:
            for field in ("fits", "reason", "download_gb", "minutes_low", "minutes_high",
                          "new", "quant", "needs_consent", "notes"):
                self.assertIn(field, entry)
            self.assertGreater(entry["minutes_low"], 0)
            self.assertGreaterEqual(entry["minutes_high"], entry["minutes_low"])
        self.assertTrue(all(call[2] for call in self.calls), "plan must resolve in dry-run mode")

    def test_no_download_when_the_file_is_there(self):
        self.assertEqual(self.make(card(24576), models=("S",))[0]["download_gb"], 0)

    def test_download_size_and_consent(self):
        cfg = Config(engines=["ollama"], models=["L"])
        entries = plan.make_plan(cfg, card(24576), resolve=self.resolver(with_file=False))
        self.assertAlmostEqual(entries[0]["download_gb"], 16.8, places=1)
        self.assertFalse(entries[0]["needs_consent"])
        big = ModelRef("F", "F", "Q4_K_M", "t", None, None, 23_387_911_136)
        entries = plan.make_plan(Config(engines=["ollama"], models=["F"]), card(24576),
                                 resolve=lambda *a, **k: big)
        self.assertTrue(entries[0]["needs_consent"])
        entries = plan.make_plan(Config(engines=["ollama"], models=["F"], yes=True), card(24576),
                                 resolve=lambda *a, **k: big)
        self.assertFalse(entries[0]["needs_consent"])

    def test_without_a_gguf_only_the_weights_are_judged(self):
        entries = plan.make_plan(Config(engines=["ollama"], models=["M"]), card(8192),
                                 resolve=self.resolver(with_file=False))
        self.assertFalse(entries[0]["fits"])  # 8.6 GiB of weights alone
        self.assertIn("weights only", entries[0]["reason"])

    def test_new_means_not_measured_in_the_store(self):
        store = ResultStore(self.root / "results.json")
        store.save("llamacpp/S/speed/warm", "ok", {})
        store.save("llamacpp/M/speed/warm", "error", {})
        entries = {(e["engine"], e["model"]): e
                   for e in self.make(card(24576), store=store, engines=("llamacpp", "ollama"))}
        self.assertFalse(entries[("llamacpp", "S")]["new"])
        self.assertTrue(entries[("llamacpp", "M")]["new"])
        self.assertTrue(entries[("ollama", "S")]["new"])
        self.assertTrue(all(e["new"] for e in self.make(card(24576)))
                        )

    def test_selected_suites_change_the_time(self):
        everything = self.make(card(24576), models=("L",),
                               suites=["memory", "speed", "smoke", "concurrency"])[0]
        one = self.make(card(24576), models=("L",), suites=["smoke"])[0]
        self.assertGreater(everything["minutes_high"], one["minutes_high"])
        no_context = self.make(card(24576), models=("L",), suites=["memory"],
                               scope="no-context")[0]
        full = self.make(card(24576), models=("L",), suites=["memory"])[0]
        self.assertGreater(full["minutes_high"], no_context["minutes_high"])

    def test_rungs_that_cannot_fit_cost_no_time(self):
        fits_all = self.make(card(48000), models=("S",), suites=["memory"])[0]
        tight = self.make(card(4600), models=("S",), suites=["memory"])[0]
        self.assertGreater(fits_all["minutes_high"], tight["minutes_high"])

    def test_upper_bound_and_recurrent_notes(self):
        plain = self.make(card(24576), models=("S",))[0]
        self.assertEqual(plain["notes"], [])
        meta = dict(DENSE, **{"llama.attention.sliding_window": 4096, "llama.ssm.state_size": 128,
                                      "llama.full_attention_interval": 4})
        path = write_gguf(self.root / "hybrid.gguf", meta)
        entries = plan.make_plan(Config(engines=["llamacpp"], models=["S"]), card(24576),
                                 resolve=self.resolver(extra=path))
        text = " ".join(entries[0]["notes"])
        self.assertIn("upper bound", text)
        self.assertIn("recurrent", text)

    def test_unknown_model_does_not_fit_and_says_why(self):
        entries = plan.make_plan(Config(engines=["ollama"], models=["XL"]), card(24576),
                                 resolve=self.resolver())
        self.assertFalse(entries[0]["fits"])
        self.assertFalse(entries[0]["available"])
        self.assertIn("unavailable", plan.format_plan(entries))
        self.assertIn("catalog", entries[0]["reason"])
        self.assertIsNone(entries[0]["download_gb"])

    def test_unknown_memory_size_is_not_a_failure(self):
        entry = self.make({"gpus": [], "memory_for_models_gb": None, "ram_gb": None},
                          models=("S",))[0]
        self.assertTrue(entry["fits"])
        self.assertIn("unknown", entry["reason"])

    def test_cpu_only_uses_system_memory(self):
        entry = self.make({"gpus": [], "memory_for_models_gb": 64, "ram_gb": 64},
                          models=("L",))[0]
        self.assertTrue(entry["fits"])

    def test_mirror_copy_is_reported_as_such(self):
        ref = ModelRef("F", "F", "Q4_K_M", "t", None, "sha256:mirror", 23_387_911_136)
        entries = plan.make_plan(Config(engines=["ollama"], models=["F"], mirror="/m"),
                                 card(48000), resolve=lambda *a, **k: ref)
        self.assertEqual(entries[0]["source"], "mirror")
        self.assertTrue(entries[0]["fits"])
        self.assertTrue(entries[0]["needs_consent"])
        self.assertIn("mirror copy 23.4 GB", plan.format_plan(entries))
        pulled = ModelRef("F", "F", "Q4_K_M", "t", None, None, 23_387_911_136)
        entries = plan.make_plan(Config(engines=["ollama"], models=["F"], mirror="/m"),
                                 card(48000), resolve=lambda *a, **k: pulled)
        self.assertEqual(entries[0]["source"], "download")

    def test_unreliable_vram_is_unknown_not_a_failure(self):
        amd = card(4096)
        amd["gpus"][0]["vendor"] = "amd"
        with mock.patch.object(plan.sys, "platform", "win32"):
            entry = self.make(amd, models=("L",))[0]
            nvidia = card(4096)
            nvidia["gpus"][0]["vendor"] = "nvidia"
            sure = self.make(nvidia, models=("L",))[0]
        self.assertIsNone(entry["fits"])
        self.assertIn("unreliable", entry["reason"])
        self.assertIn("unknown", plan.format_plan([entry]))
        self.assertFalse(sure["fits"])
        with mock.patch.object(plan.sys, "platform", "linux"):
            self.assertFalse(self.make(amd, models=("L",))[0]["fits"])

    def test_format_plan_is_plain_english_without_em_dashes(self):
        entries = self.make(card(8192), engines=("llamacpp", "ollama"))
        text = plan.format_plan(entries)
        for model in ("S", "M", "L"):
            self.assertIn(model, text)
        self.assertIn("does not fit", text)
        self.assertNotIn("\u2014", text)
        self.assertIn("min", text)


if __name__ == "__main__":
    unittest.main()
