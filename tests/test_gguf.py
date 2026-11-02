import tempfile
import unittest
from pathlib import Path

from bench.gguf import (attention_layer_count, kv_bytes_per_token, memory_curve,
                        read_metadata)
from gguf_writer import (ARRAY, BOOL, FLOAT32, FLOAT64, INT8, INT16, INT32, INT64, STRING, UINT8,
                         UINT16, UINT32, UINT64, Array, Typed, write_gguf)

REAL = Path(__file__).resolve().parent.parent / ".bench" / "fixtures-local"

DENSE = {
    "general.architecture": "llama",
    "llama.block_count": 32,
    "llama.attention.head_count_kv": 8,
    "llama.attention.key_length": 128,
}


def dense_meta(**overrides) -> dict:
    meta = dict(DENSE)
    meta.update(overrides)
    return meta


class ReaderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "m.gguf"

    def test_read_roundtrip(self):
        write_gguf(self.path, DENSE)
        meta = read_metadata(self.path)
        for key, value in DENSE.items():
            self.assertEqual(meta[key], value)

    def test_all_types_roundtrip(self):
        kv = {
            "t.u8": Typed(UINT8, 200), "t.i8": Typed(INT8, -5), "t.u16": Typed(UINT16, 65000),
            "t.i16": Typed(INT16, -300), "t.u32": Typed(UINT32, 4_000_000_000),
            "t.i32": Typed(INT32, -70000), "t.f32": Typed(FLOAT32, 0.5), "t.bool": True,
            "t.str": "héllo", "t.u64": Typed(UINT64, 2**40), "t.i64": Typed(INT64, -2**40),
            "t.f64": Typed(FLOAT64, 1.25), "t.arr": [1, 2, 3], "t.flags": [True, False],
            "t.tags": ["a", "b"], "t.empty": Array(UINT32, []),
            "t.nested": Array(ARRAY, [[1, 2], [3]]),
        }
        write_gguf(self.path, kv)
        meta = read_metadata(self.path)
        self.assertEqual(meta["t.u8"], 200)
        self.assertEqual(meta["t.i16"], -300)
        self.assertEqual(meta["t.u32"], 4_000_000_000)
        self.assertEqual(meta["t.f32"], 0.5)
        self.assertIs(meta["t.bool"], True)
        self.assertEqual(meta["t.str"], "héllo")
        self.assertEqual(meta["t.i64"], -2**40)
        self.assertEqual(meta["t.f64"], 1.25)
        self.assertEqual(meta["t.arr"], [1, 2, 3])
        self.assertEqual(meta["t.flags"], [True, False])
        self.assertEqual(meta["t.tags"], ["a", "b"])
        self.assertEqual(meta["t.empty"], [])
        self.assertEqual(meta["t.nested"], [[1, 2], [3]])

    def test_header_counts(self):
        write_gguf(self.path, DENSE, tensor_count=291)
        meta = read_metadata(self.path)
        self.assertEqual(meta["gguf.version"], 3)
        self.assertEqual(meta["gguf.tensor_count"], 291)
        self.assertEqual(meta["gguf.kv_count"], len(DENSE))

    def test_tensor_data_is_not_needed(self):
        # Tensor infos and data follow the KV pairs; the reader must not touch them.
        write_gguf(self.path, DENSE, tensor_count=10**9, tail=b"\xff" * 64)
        self.assertEqual(read_metadata(self.path)["llama.block_count"], 32)

    def test_large_arrays_are_skipped_and_reading_continues(self):
        kv = {
            "general.architecture": "llama",
            "tokenizer.ggml.tokens": [f"tok{i}" for i in range(5000)],
            "tokenizer.ggml.token_type": [1] * 5000,
            "tokenizer.ggml.scores": [0.0] * 5000,
            "llama.block_count": 32,
            "llama.rope.dimension_sections": [11, 11, 10, 0],
            "general.tags": ["text-generation"],
        }
        write_gguf(self.path, kv)
        meta = read_metadata(self.path)
        for key in ("tokenizer.ggml.tokens", "tokenizer.ggml.token_type", "tokenizer.ggml.scores"):
            self.assertNotIn(key, meta)
        self.assertEqual(meta["gguf.skipped_arrays"], {
            "tokenizer.ggml.tokens": 5000, "tokenizer.ggml.token_type": 5000,
            "tokenizer.ggml.scores": 5000})
        self.assertEqual(meta["llama.block_count"], 32)
        self.assertEqual(meta["llama.rope.dimension_sections"], [11, 11, 10, 0])
        self.assertEqual(meta["general.tags"], ["text-generation"])

    def test_per_layer_array_of_a_large_model_is_kept(self):
        counts = [0, 8] * 100
        write_gguf(self.path, {"x.attention.head_count_kv": counts})
        self.assertEqual(read_metadata(self.path)["x.attention.head_count_kv"], counts)

    def test_bad_magic(self):
        self.path.write_bytes(b"NOPE" + b"\x00" * 40)
        with self.assertRaisesRegex(ValueError, "not a GGUF"):
            read_metadata(self.path)

    def test_unsupported_version(self):
        write_gguf(self.path, DENSE, version=1)
        with self.assertRaisesRegex(ValueError, "version"):
            read_metadata(self.path)

    def test_truncated_inside_kv_raises(self):
        write_gguf(self.path, DENSE)
        data = self.path.read_bytes()
        self.path.write_bytes(data[:-6])
        with self.assertRaisesRegex(ValueError, "truncated"):
            read_metadata(self.path)

    def test_truncated_inside_skipped_array_raises(self):
        write_gguf(self.path, {"tokenizer.ggml.token_type": [1] * 5000})
        data = self.path.read_bytes()
        self.path.write_bytes(data[:-100])
        with self.assertRaisesRegex(ValueError, "truncated"):
            read_metadata(self.path)

    def test_empty_file(self):
        self.path.write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "truncated|not a GGUF"):
            read_metadata(self.path)


class KvBytesTest(unittest.TestCase):
    def test_kv_dense(self):
        self.assertEqual(kv_bytes_per_token(DENSE), 2 * 32 * 8 * 128 * 2)
        self.assertEqual(kv_bytes_per_token(DENSE), 131072)
        self.assertEqual(attention_layer_count(DENSE), 32)

    def test_cache_type_bytes(self):
        self.assertEqual(kv_bytes_per_token(DENSE, 1.0), 65536)
        self.assertEqual(kv_bytes_per_token(DENSE, 4.0), 262144)

    def test_key_length_falls_back_to_embedding_over_head_count(self):
        meta = dense_meta()
        del meta["llama.attention.key_length"]
        meta["llama.embedding_length"] = 4096
        meta["llama.attention.head_count"] = 32
        self.assertEqual(kv_bytes_per_token(meta), 131072)

    def test_head_count_kv_falls_back_to_head_count(self):
        meta = dense_meta()
        del meta["llama.attention.head_count_kv"]
        meta["llama.attention.head_count"] = 16
        self.assertEqual(kv_bytes_per_token(meta), 2 * 32 * 16 * 128 * 2)

    def test_value_length_is_used_when_it_differs(self):
        meta = dense_meta(**{"llama.attention.value_length": 64})
        self.assertEqual(kv_bytes_per_token(meta), (128 + 64) * 32 * 8 * 2)

    def test_kv_hybrid_array(self):
        counts = ([0, 0, 0, 8] * 8)
        meta = dense_meta(**{"llama.attention.head_count_kv": counts})
        self.assertEqual(attention_layer_count(meta), 8)
        self.assertEqual(kv_bytes_per_token(meta), 2 * 8 * 8 * 128 * 2)

    def test_kv_array_with_varying_heads(self):
        meta = dense_meta(**{"llama.block_count": 4, "llama.attention.head_count_kv": [2, 4, 0, 8]})
        self.assertEqual(kv_bytes_per_token(meta), 2 * (2 + 4 + 8) * 128 * 2)

    def test_kv_array_length_must_match_block_count(self):
        meta = dense_meta(**{"llama.attention.head_count_kv": [0, 8]})
        with self.assertRaisesRegex(ValueError, "head_count_kv"):
            kv_bytes_per_token(meta)

    def test_kv_interval(self):
        meta = dense_meta(**{"llama.full_attention_interval": 4})
        self.assertEqual(attention_layer_count(meta), 8)
        self.assertEqual(kv_bytes_per_token(meta), 2 * 8 * 8 * 128 * 2)

    def test_kv_recurrent_layers_array(self):
        flags = [True, True, False, True] * 8
        meta = dense_meta(**{"llama.attention.recurrent_layers": flags})
        self.assertEqual(attention_layer_count(meta), 8)
        self.assertEqual(kv_bytes_per_token(meta), 2 * 8 * 8 * 128 * 2)

    def test_recurrent_layers_array_wins_over_interval(self):
        flags = [False] * 4 + [True] * 28
        meta = dense_meta(**{"llama.attention.recurrent_layers": flags,
                             "llama.full_attention_interval": 4})
        self.assertEqual(attention_layer_count(meta), 4)

    def test_nextn_layers_are_not_counted(self):
        # 65 blocks of which one is the multi-token-prediction head (its own context).
        meta = dense_meta(**{"llama.block_count": 65, "llama.nextn_predict_layers": 1,
                             "llama.full_attention_interval": 4})
        self.assertEqual(attention_layer_count(meta), 16)
        meta = dense_meta(**{"llama.block_count": 65, "llama.nextn_predict_layers": 1})
        self.assertEqual(attention_layer_count(meta), 64)

    def test_kv_mla(self):
        # Multi-head latent attention caches one compressed vector plus the rope key per layer.
        meta = {"general.architecture": "deepseek2", "deepseek2.block_count": 10,
                "deepseek2.attention.kv_lora_rank": 512, "deepseek2.rope.dimension_count": 64}
        self.assertEqual(kv_bytes_per_token(meta), 10 * (512 + 64) * 2)

    def test_unknown_architecture_names_missing_keys(self):
        meta = {"general.architecture": "mystery", "mystery.embedding_length": 1024}
        with self.assertRaises(ValueError) as caught:
            kv_bytes_per_token(meta)
        message = str(caught.exception)
        self.assertIn("mystery.block_count", message)
        self.assertIn("mystery.attention.head_count_kv", message)

    def test_missing_architecture(self):
        with self.assertRaisesRegex(ValueError, "general.architecture"):
            kv_bytes_per_token({"llama.block_count": 32})

    def test_missing_key_length_names_keys(self):
        meta = dense_meta()
        del meta["llama.attention.key_length"]
        with self.assertRaisesRegex(ValueError, "llama.attention.key_length"):
            kv_bytes_per_token(meta)

    def test_recurrent_model_without_layer_marker_is_refused(self):
        meta = dense_meta(**{"llama.ssm.state_size": 128})
        with self.assertRaisesRegex(ValueError, "full_attention_interval"):
            kv_bytes_per_token(meta)

    def test_interval_must_be_positive(self):
        meta = dense_meta(**{"llama.full_attention_interval": 0})
        with self.assertRaisesRegex(ValueError, "full_attention_interval"):
            kv_bytes_per_token(meta)


class MemoryCurveTest(unittest.TestCase):
    def test_curve(self):
        self.assertEqual(memory_curve(DENSE, 4_000_000_000, [8192]),
                         [(8192, 4_000_000_000 + 8192 * 131072)])

    def test_curve_keeps_step_order_and_cache_type(self):
        curve = memory_curve(DENSE, 1000, [32768, 8192], cache_type_bytes=1.0)
        self.assertEqual(curve, [(32768, 1000 + 32768 * 65536), (8192, 1000 + 8192 * 65536)])

    def test_curve_empty(self):
        self.assertEqual(memory_curve(DENSE, 1000, []), [])


@unittest.skipUnless((REAL / "qwen3-14b-head.gguf").exists(), "local real GGUF sample missing")
class RealQwen3Test(unittest.TestCase):
    def test_metadata_and_kv(self):
        meta = read_metadata(REAL / "qwen3-14b-head.gguf")
        self.assertEqual(meta["general.architecture"], "qwen3")
        self.assertEqual(meta["qwen3.block_count"], 40)
        self.assertEqual(meta["qwen3.attention.head_count_kv"], 8)
        self.assertEqual(meta["qwen3.attention.key_length"], 128)
        self.assertEqual(meta["qwen3.context_length"], 40960)
        self.assertIn("tokenizer.ggml.tokens", meta["gguf.skipped_arrays"])
        self.assertGreater(meta["gguf.skipped_arrays"]["tokenizer.ggml.tokens"], 100_000)
        self.assertEqual(kv_bytes_per_token(meta), 163840)  # 160 KiB per token at f16

    def test_matches_measured_vram_growth(self):
        # Measured on an RTX 4090: 12.4 GB at 16k context, 16.3 GB at 40,960 (about 3.9 GB).
        meta = read_metadata(REAL / "qwen3-14b-head.gguf")
        (_, low), (_, high) = memory_curve(meta, 0, [16384, 40960])
        growth_gb = (high - low) / 1e9
        self.assertAlmostEqual(growth_gb, 3.9, delta=0.3)


@unittest.skipUnless((REAL / "qwen3.5-4b-head.gguf").exists(), "local real GGUF sample missing")
class RealQwen35Test(unittest.TestCase):
    def test_hybrid_recurrent_layers(self):
        meta = read_metadata(REAL / "qwen3.5-4b-head.gguf")
        self.assertEqual(meta["general.architecture"], "qwen35")
        self.assertEqual(meta["qwen35.block_count"], 32)
        self.assertEqual(meta["qwen35.attention.head_count_kv"], 4)
        self.assertEqual(meta["qwen35.attention.key_length"], 256)
        self.assertEqual(meta["qwen35.full_attention_interval"], 4)
        self.assertEqual(sum(1 for r in meta["qwen35.attention.recurrent_layers"] if not r), 8)
        self.assertEqual(attention_layer_count(meta), 8)
        self.assertEqual(kv_bytes_per_token(meta), 8 * 2 * 4 * 256 * 2)  # 32 KiB


@unittest.skipUnless((REAL / "qwen3.8-27b-head.gguf").exists(), "local real GGUF sample missing")
class RealQwen38Test(unittest.TestCase):
    def test_hybrid_interval(self):
        meta = read_metadata(REAL / "qwen3.8-27b-head.gguf")
        self.assertEqual(meta["general.architecture"], "qwen35")
        self.assertEqual(meta["qwen35.block_count"], 65)
        self.assertEqual(meta["qwen35.nextn_predict_layers"], 1)
        self.assertEqual(meta["qwen35.attention.head_count_kv"], 4)
        self.assertEqual(meta["qwen35.attention.key_length"], 256)
        self.assertEqual(meta["qwen35.full_attention_interval"], 4)
        self.assertNotIn("qwen35.attention.recurrent_layers", meta)
        self.assertEqual(attention_layer_count(meta), 16)
        self.assertEqual(kv_bytes_per_token(meta), 16 * 2 * 4 * 256 * 2)  # 64 KiB


if __name__ == "__main__":
    unittest.main()
