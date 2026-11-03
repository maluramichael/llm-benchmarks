import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bench import models
from bench.config import Config
from bench.models import ModelError, ModelNotFound, NeedsConsent
from tests.fake_http import FakeServer
from tests.gguf_writer import write_gguf


def digest_of(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def blob_file(store: Path, data: bytes) -> str:
    digest = digest_of(data)
    path = store / "blobs" / digest.replace(":", "-")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return digest


def make_model(store: Path, tag: str, model_data: bytes, *, extra_layers=True,
               absent_variant=True) -> str:
    """Write a manifest shaped like Ollama 0.40 plus blobs; returns the model digest."""
    name, _, version = tag.partition(":")
    layers = []
    if extra_layers:  # the model layer must not be the first one
        data = b"{{ .Prompt }}"
        layers.append({"mediaType": "application/vnd.ollama.image.template",
                       "digest": blob_file(store, data), "size": len(data)})
    model_digest = blob_file(store, model_data)
    layers.append({"mediaType": models.MODEL_LAYER, "digest": model_digest,
                   "size": len(model_data), "from": "model.gguf"})
    if extra_layers:
        data = b'{"stop":[]}'
        layers.append({"mediaType": "application/vnd.ollama.image.params",
                       "digest": blob_file(store, data), "size": len(data)})
        # manifest list pointing at a variant blob the mirror does not have
        listing = json.dumps({"manifests": [
            {"digest": "sha256:" + "ab" * 32, "platform": {"os": "mlx"}}]}).encode()
        layers.append({"mediaType": "application/vnd.ollama.manifest.list.v2+json",
                       "digest": blob_file(store, listing), "size": len(listing)})
    config = b'{"model_format":"gguf"}'
    manifest = {"schemaVersion": 2,
                "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
                "config": {"mediaType": "application/vnd.docker.container.image.v1+json",
                           "digest": blob_file(store, config), "size": len(config)},
                "layers": layers}
    path = store / "manifests" / "registry.ollama.ai" / "library" / name / version
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return model_digest


class Base(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.store = self.tmp / "store"
        self.mirror = self.tmp / "mirror"
        self.model_dir = self.tmp / "gguf-dir"

    def cfg(self, **kw):
        base = dict(ollama_models_dir=self.store, model_dir=self.model_dir)
        base.update(kw)
        return Config(**base)


class CatalogTest(unittest.TestCase):
    def test_classes_and_pins(self):
        cat = models.catalog()
        self.assertEqual(set(cat), {"S", "M", "L", "F"})
        self.assertEqual([cat[k]["ollama_tag"] for k in "SML"],
                         ["qwen3.5:4b", "qwen3:14b", "qwen3.8:27b"])
        self.assertTrue(all(cat[k]["reuse_ollama_blob"] for k in "SML"))
        self.assertEqual(cat["F"]["ollama_tag"], "flint:40b")
        pin = cat["F"]["gguf"]
        self.assertEqual(pin["repo"], "ggml-org/Flint-GGUF")
        self.assertEqual(pin["file"], "Flint-Q4_K_M.gguf")
        self.assertRegex(pin["revision"], r"^[0-9a-f]{40}$")
        self.assertRegex(pin["sha256"], r"^[0-9a-f]{64}$")
        self.assertGreater(pin["size_bytes"], 20 * 10**9)
        self.assertTrue(all(cat[k]["quant"] for k in cat))
        self.assertEqual([cat[k]["quant"] for k in "SML"], ["Q4_K_M"] * 3)
        self.assertEqual(cat["F"]["quant"], "NVFP4")  # the Ollama layer
        self.assertEqual(pin["quant"], "Q4_K_M")  # the Hugging Face file

    def test_catalog_is_a_fresh_copy(self):
        models.catalog()["S"]["quant"] = "x"
        self.assertNotEqual(models.catalog()["S"]["quant"], "x")


class BlobLookupTest(Base):
    def test_blob_lookup(self):
        digest = make_model(self.store, "qwen3:14b", b"GGUF-14b")
        path = models.ollama_blob_for("qwen3:14b", self.store)
        self.assertEqual(path, self.store / "blobs" / digest.replace(":", "-"))
        self.assertEqual(path.read_bytes(), b"GGUF-14b")

    def test_missing_manifest_and_blob(self):
        with self.assertRaises(ModelNotFound):
            models.ollama_blob_for("qwen3:14b", self.store)
        digest = make_model(self.store, "qwen3:14b", b"GGUF-14b")
        (self.store / "blobs" / digest.replace(":", "-")).unlink()
        with self.assertRaises(ModelNotFound):
            models.ollama_blob_for("qwen3:14b", self.store)

    def test_wrong_blob_size_is_an_error(self):
        digest = make_model(self.store, "qwen3:14b", b"GGUF-14b")
        (self.store / "blobs" / digest.replace(":", "-")).write_bytes(b"short")
        with self.assertRaises(ModelError):
            models.ollama_blob_for("qwen3:14b", self.store)

    def test_tag_cannot_leave_the_store(self):
        for tag in ("../../etc/passwd:x", "a/b/c/d:1", "qwen3:../x", "x:y/z",
                    "registry.example:5000/ns/name:1"):
            with self.assertRaises(ModelError, msg=tag):
                models.ollama_blob_for(tag, self.store)

    def test_errors_do_not_contain_the_store_path(self):
        with self.assertRaises(ModelNotFound) as ctx:
            models.ollama_blob_for("qwen3:14b", self.store)
        self.assertNotIn(str(self.tmp), str(ctx.exception))
        digest = make_model(self.store, "qwen3:14b", b"GGUF-14b")
        (self.store / "blobs" / digest.replace(":", "-")).unlink()
        with self.assertRaises(ModelNotFound) as ctx:
            models.ollama_blob_for("qwen3:14b", self.store)
        self.assertNotIn(str(self.tmp), str(ctx.exception))
        with self.assertRaises(ModelNotFound) as ctx:
            models.import_from_mirror(self.tmp / "nomirror", "qwen3:14b", self.store)
        self.assertNotIn(str(self.tmp), str(ctx.exception))

    def test_no_model_layer(self):
        make_model(self.store, "qwen3:14b", b"x")
        path = self.store / "manifests/registry.ollama.ai/library/qwen3/14b"
        data = json.loads(path.read_text())
        data["layers"] = [l for l in data["layers"] if l["mediaType"] != models.MODEL_LAYER]
        path.write_text(json.dumps(data))
        with self.assertRaises(ModelError):
            models.ollama_blob_for("qwen3:14b", self.store)


class MirrorImportTest(Base):
    def test_mirror_import_no_overwrite(self):
        mirror_store = self.mirror / "ollama"
        digest = make_model(mirror_store, "qwen3:14b", b"GGUF-from-mirror")
        # target already has the model blob, with other content, and one other layer
        target_blob = self.store / "blobs" / digest.replace(":", "-")
        target_blob.parent.mkdir(parents=True)
        target_blob.write_bytes(b"different")
        warnings = []
        models.import_from_mirror(self.mirror, "qwen3:14b", self.store, warnings.append)
        self.assertEqual(target_blob.read_bytes(), b"different")
        self.assertTrue(any("unexpected size" in w for w in warnings))
        # a missing blob was copied, the manifest too, and nothing is left over
        for blob in (mirror_store / "blobs").iterdir():
            if blob.name != target_blob.name:
                self.assertEqual((self.store / "blobs" / blob.name).read_bytes(),
                                 blob.read_bytes())
        self.assertTrue((self.store / "manifests/registry.ollama.ai/library/qwen3/14b").exists())
        self.assertEqual([p for p in self.store.rglob("*") if p.name.endswith(".part")], [])

    def test_manifest_list_variant_not_required(self):
        make_model(self.mirror / "ollama", "qwen3.5:4b", b"gguf")
        models.import_from_mirror(self.mirror, "qwen3.5:4b", self.store)
        self.assertEqual(models.ollama_blob_for("qwen3.5:4b", self.store).read_bytes(), b"gguf")
        self.assertFalse((self.store / "blobs" / ("sha256-" + "ab" * 32)).exists())

    def test_existing_manifest_is_kept(self):
        make_model(self.mirror / "ollama", "qwen3:14b", b"mirror")
        make_model(self.store, "qwen3:14b", b"local")
        manifest = self.store / "manifests/registry.ollama.ai/library/qwen3/14b"
        before = manifest.read_bytes()
        models.import_from_mirror(self.mirror, "qwen3:14b", self.store)
        self.assertEqual(manifest.read_bytes(), before)
        self.assertEqual(models.ollama_blob_for("qwen3:14b", self.store).read_bytes(), b"local")

    def test_missing_mirror_blob_copies_nothing(self):
        mirror_store = self.mirror / "ollama"
        make_model(mirror_store, "qwen3:14b", b"mirror")
        manifest = json.loads((mirror_store / "manifests/registry.ollama.ai/library/qwen3/14b")
                              .read_text())
        last = manifest["layers"][-1]["digest"]  # the last blob that would be copied
        (mirror_store / "blobs" / last.replace(":", "-")).unlink()
        with self.assertRaises(ModelError):
            models.import_from_mirror(self.mirror, "qwen3:14b", self.store)
        self.assertEqual(list((self.store / "blobs").glob("*")) if self.store.exists() else [], [])
        self.assertFalse((self.store / "manifests").exists())

    def test_wrong_size_mirror_blob_copies_nothing(self):
        mirror_store = self.mirror / "ollama"
        make_model(mirror_store, "qwen3:14b", b"mirror")
        manifest = json.loads((mirror_store / "manifests/registry.ollama.ai/library/qwen3/14b")
                              .read_text())
        last = manifest["layers"][-1]["digest"]
        with (mirror_store / "blobs" / last.replace(":", "-")).open("ab") as handle:
            handle.write(b"x")
        with self.assertRaises(ModelError):
            models.import_from_mirror(self.mirror, "qwen3:14b", self.store)
        self.assertEqual(list((self.store / "blobs").glob("*")) if self.store.exists() else [], [])
        self.assertFalse((self.store / "manifests").exists())

    def test_transient_lock_while_publishing_is_retried(self):
        make_model(self.mirror / "ollama", "qwen3:14b", b"mirror")
        real = os.link
        calls = []

        def flaky(src, dst):
            calls.append(dst)
            if len(calls) == 1:
                raise PermissionError("locked")
            return real(src, dst)
        with mock.patch("bench.models.os.link", flaky):
            models.import_from_mirror(self.mirror, "qwen3:14b", self.store)
        self.assertEqual(models.ollama_blob_for("qwen3:14b", self.store).read_bytes(), b"mirror")

    def test_publish_without_hard_links_falls_back(self):
        make_model(self.mirror / "ollama", "qwen3:14b", b"mirror")
        with mock.patch("bench.models.os.link", side_effect=OSError(95, "not supported")):
            models.import_from_mirror(self.mirror, "qwen3:14b", self.store)
        self.assertEqual(models.ollama_blob_for("qwen3:14b", self.store).read_bytes(), b"mirror")
        self.assertEqual([p for p in self.store.rglob("*") if ".part" in p.name], [])

    def test_publish_keeps_a_file_that_appeared_meanwhile(self):
        dest = self.tmp / "final"
        dest.write_bytes(b"theirs")
        part = self.tmp / "scratch"
        part.write_bytes(b"mine")
        models._publish(part, dest)
        self.assertEqual(dest.read_bytes(), b"theirs")
        self.assertFalse(part.exists())

    def test_scratch_names_are_unique_per_call(self):
        names = {models._scratch(self.tmp / "blob").name for _ in range(3)}
        self.assertEqual(len(names), 3)

    def test_not_enough_disk_space(self):
        make_model(self.mirror / "ollama", "qwen3:14b", b"mirror")
        usage = mock.Mock(free=1)
        with mock.patch("bench.models.shutil.disk_usage", return_value=usage):
            with self.assertRaises(ModelError) as ctx:
                models.import_from_mirror(self.mirror, "qwen3:14b", self.store)
        self.assertIn("disk space", str(ctx.exception))
        self.assertFalse((self.store / "manifests").exists())

    def test_corrupt_blob_is_rejected_and_manifest_not_written(self):
        mirror_store = self.mirror / "ollama"
        digest = make_model(mirror_store, "qwen3:14b", b"mirror-data")
        (mirror_store / "blobs" / digest.replace(":", "-")).write_bytes(b"mirror-dxta")
        with self.assertRaises(ModelError):
            models.import_from_mirror(self.mirror, "qwen3:14b", self.store)
        self.assertFalse((self.store / "manifests").exists())
        self.assertFalse(any(p.name.endswith(".part") for p in self.store.rglob("*")))

    def test_mirror_root_may_be_the_store_itself(self):
        make_model(self.mirror, "qwen3:14b", b"x")
        models.import_from_mirror(self.mirror, "qwen3:14b", self.store)
        self.assertTrue(models.ollama_blob_for("qwen3:14b", self.store).exists())

    def test_unknown_tag_in_mirror(self):
        make_model(self.mirror / "ollama", "qwen3:14b", b"x")
        with self.assertRaises(ModelNotFound):
            models.import_from_mirror(self.mirror, "qwen3:32b", self.store)


class ResolveLlamacppTest(Base):
    def test_resolve_order(self):
        explicit = self.tmp / "my.gguf"
        explicit.write_bytes(b"explicit")
        mirror_digest = make_model(self.mirror / "ollama", "qwen3.5:4b", b"from-mirror")
        local_digest = make_model(self.store, "qwen3.5:4b", b"from-local-store")
        # explicit wins
        ref = models.resolve("S", "llamacpp",
                             self.cfg(mirror=str(self.mirror), gguf={"S": explicit}))
        self.assertEqual(ref.gguf_path, explicit)
        self.assertIsNone(ref.digest)
        # then the mirror, used in place
        ref = models.resolve("S", "llamacpp", self.cfg(mirror=str(self.mirror)))
        self.assertEqual(ref.gguf_path.read_bytes(), b"from-mirror")
        self.assertEqual(ref.digest, mirror_digest)
        self.assertTrue(str(ref.gguf_path).startswith(str(self.mirror)))
        # then the local store
        ref = models.resolve("S", "llamacpp", self.cfg())
        self.assertEqual(ref.gguf_path.read_bytes(), b"from-local-store")
        self.assertEqual(ref.digest, local_digest)
        self.assertEqual((ref.id, ref.cls, ref.ollama_tag), ("S", "S", "qwen3.5:4b"))
        # nothing anywhere: no download entry for S
        empty = self.cfg(ollama_models_dir=self.tmp / "nowhere")
        with self.assertRaises(ModelNotFound):
            models.resolve("S", "llamacpp", empty)

    def test_flint_order_and_download_last(self):
        entries = models.catalog()
        data = b"flint-bytes"
        entries["F"]["gguf"].update(size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        # mirror file wins over a file in the model dir
        (self.mirror / "gguf" / "flint").mkdir(parents=True)
        (self.mirror / "gguf/flint/Flint-Q4_K_M.gguf").write_bytes(data)
        self.model_dir.mkdir()
        (self.model_dir / "Flint-Q4_K_M.gguf").write_bytes(data)
        ref = models.resolve("F", "llamacpp", self.cfg(mirror=str(self.mirror)), entries=entries)
        self.assertEqual(ref.gguf_path, self.mirror / "gguf/flint/Flint-Q4_K_M.gguf")
        self.assertIsNone(ref.digest)  # never hashed, so no digest is claimed
        self.assertEqual(ref.quant, "Q4_K_M")
        ref = models.resolve("F", "llamacpp", self.cfg(), entries=entries)
        self.assertEqual(ref.gguf_path, self.model_dir / "Flint-Q4_K_M.gguf")
        self.assertIsNone(ref.digest)
        self.assertEqual(ref.quant, "Q4_K_M")
        (self.model_dir / "Flint-Q4_K_M.gguf").unlink()
        with FakeServer({}) as server:  # server has nothing: the download is attempted
            with self.assertRaises(ModelError):
                models.resolve("F", "llamacpp", self.cfg(), entries=entries,
                               hf_base=server.url)
            self.assertEqual(len(server.requests), 1)

    def test_wrong_size_file_is_never_replaced(self):
        entries = models.catalog()
        entries["F"]["gguf"].update(size_bytes=99)
        self.model_dir.mkdir()
        existing = self.model_dir / "Flint-Q4_K_M.gguf"
        existing.write_bytes(b"partial")
        with self.assertRaises(ModelError):
            models.resolve("F", "llamacpp", self.cfg(), entries=entries)
        self.assertEqual(existing.read_bytes(), b"partial")

    def test_reuse_ollama_blobs_override_uses_the_nvfp4_blob(self):
        digest = make_model(self.store, "flint:40b", b"nvfp4-blob")
        ref = models.resolve("F", "llamacpp", self.cfg(reuse_ollama_blobs=True))
        self.assertEqual(ref.gguf_path.read_bytes(), b"nvfp4-blob")
        self.assertEqual((ref.quant, ref.digest), ("NVFP4", digest))
        # without the override the blob is ignored and the (large) download needs consent
        with self.assertRaises(NeedsConsent):
            models.resolve("F", "llamacpp", self.cfg())

    def test_quant_follows_the_source(self):
        make_model(self.store, "qwen3:14b", b"blob")
        self.assertEqual(models.resolve("M", "llamacpp", self.cfg()).quant, "Q4_K_M")
        make_model(self.store, "flint:40b", b"blob")
        self.assertEqual(models.resolve("F", "ollama", self.cfg()).quant, "NVFP4")
        ref = models.resolve("F", "llamacpp", self.cfg(dry_run=True))  # pinned HF file
        self.assertEqual(ref.quant, "Q4_K_M")

    def test_download_ref_has_the_pinned_quant(self):
        entries = models.catalog()
        pin = entries["F"]["gguf"]
        data = b"flint"
        pin.update(size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        path = f"/{pin['repo']}/resolve/{pin['revision']}/{pin['file']}"
        with FakeServer({path: data}) as server:
            ref = models.resolve("F", "llamacpp", self.cfg(yes=True), entries=entries,
                                 hf_base=server.url)
        self.assertEqual(ref.quant, "Q4_K_M")

    def test_explicit_gguf_quant_comes_from_the_header(self):
        for file_type, quant in ((15, "Q4_K_M"), (39, "NVFP4"), (7, "Q8_0"), (1, "F16"),
                                 (9999, "unknown")):
            path = write_gguf(self.tmp / f"t{file_type}.gguf",
                              {"general.architecture": "llama", "general.file_type": file_type})
            ref = models.resolve("F", "llamacpp", self.cfg(gguf={"F": path}))
            self.assertEqual(ref.quant, quant, file_type)
        junk = self.tmp / "junk.gguf"
        junk.write_bytes(b"not a gguf")
        self.assertEqual(models.resolve("S", "llamacpp", self.cfg(gguf={"S": junk})).quant,
                         "unknown")
        nokey = write_gguf(self.tmp / "nokey.gguf", {"general.architecture": "llama"})
        self.assertEqual(models.resolve("XL", "llamacpp", self.cfg(gguf={"XL": nokey})).quant,
                         "unknown")

    def test_explicit_gguf_error_hides_the_path(self):
        missing = self.tmp / "private" / "x.gguf"
        for model_id in ("S", "XL"):
            with self.assertRaises(ModelError) as ctx:
                models.resolve(model_id, "llamacpp", self.cfg(gguf={model_id: missing}))
            self.assertNotIn(str(self.tmp), str(ctx.exception))

    def test_wrong_size_error_hides_the_path(self):
        entries = models.catalog()
        entries["F"]["gguf"].update(size_bytes=99)
        self.model_dir.mkdir()
        (self.model_dir / "Flint-Q4_K_M.gguf").write_bytes(b"partial")
        with self.assertRaises(ModelError) as ctx:
            models.resolve("F", "llamacpp", self.cfg(), entries=entries)
        self.assertNotIn(str(self.tmp), str(ctx.exception))

    def test_large_download_needs_consent(self):
        with FakeServer({}) as server:
            with self.assertRaises(NeedsConsent) as ctx:
                models.resolve("F", "llamacpp", self.cfg(), hf_base=server.url)
            self.assertEqual(server.requests, [])
        self.assertGreater(ctx.exception.size_bytes, 20 * 10**9)

    def test_no_download_flag(self):
        with self.assertRaises(ModelNotFound):
            models.resolve("F", "llamacpp", self.cfg(no_download=True, yes=True))

    def test_dry_run_downloads_nothing(self):
        with FakeServer({}) as server:
            ref = models.resolve("F", "llamacpp", self.cfg(dry_run=True), hf_base=server.url)
            self.assertEqual(server.requests, [])
        self.assertIsNone(ref.gguf_path)
        self.assertFalse(self.model_dir.exists())

    def test_download_with_consent_verifies_and_pins_revision(self):
        data = b"flint-download" * 100
        entries = models.catalog()
        pin = entries["F"]["gguf"]
        pin.update(size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        path = f"/{pin['repo']}/resolve/{pin['revision']}/{pin['file']}"
        with FakeServer({path: data}) as server:
            ref = models.resolve("F", "llamacpp", self.cfg(yes=True), entries=entries,
                                 hf_base=server.url)
        self.assertEqual(ref.gguf_path, self.model_dir / "Flint-Q4_K_M.gguf")
        self.assertEqual(ref.gguf_path.read_bytes(), data)
        self.assertEqual(ref.digest, "sha256:" + pin["sha256"])
        self.assertEqual(ref.size_bytes, len(data))

    def test_download_with_wrong_content_leaves_nothing(self):
        entries = models.catalog()
        pin = entries["F"]["gguf"]
        pin.update(size_bytes=4, sha256="00" * 32)
        path = f"/{pin['repo']}/resolve/{pin['revision']}/{pin['file']}"
        with FakeServer({path: b"abcd"}) as server:
            with self.assertRaises(ModelError):
                models.resolve("F", "llamacpp", self.cfg(yes=True), entries=entries,
                               hf_base=server.url)
        self.assertEqual(list(self.model_dir.glob("*")), [])

    def test_xl_only_with_gguf(self):
        with self.assertRaises(ModelNotFound):
            models.resolve("XL", "llamacpp", self.cfg())
        with self.assertRaises(ModelError):
            models.resolve("XL", "ollama", self.cfg())
        big = self.tmp / "big.gguf"
        big.write_bytes(b"1234")
        ref = models.resolve("XL", "llamacpp", self.cfg(gguf={"XL": big}))
        self.assertEqual((ref.id, ref.gguf_path, ref.size_bytes), ("XL", big, 4))
        with self.assertRaises(ModelError):
            models.resolve("XL", "llamacpp", self.cfg(gguf={"XL": self.tmp / "missing"}))

    def test_url_mirror_is_refused(self):
        with self.assertRaises(ModelError):
            models.resolve("S", "llamacpp", self.cfg(mirror="https://example.org/m"))


class ResolveOllamaTest(Base):
    def test_imports_from_mirror(self):
        digest = make_model(self.mirror / "ollama", "qwen3:14b", b"mirror-14b")
        ref = models.resolve("M", "ollama", self.cfg(mirror=str(self.mirror)))
        self.assertEqual((ref.ollama_tag, ref.digest, ref.quant), ("qwen3:14b", digest, "Q4_K_M"))
        self.assertEqual(ref.size_bytes, len(b"mirror-14b"))
        self.assertTrue((self.store / "blobs" / digest.replace(":", "-")).exists())

    def test_dry_run_does_not_import(self):
        make_model(self.mirror / "ollama", "qwen3:14b", b"mirror-14b")
        models.resolve("M", "ollama", self.cfg(mirror=str(self.mirror), dry_run=True))
        self.assertFalse(self.store.exists())

    def test_local_store_without_mirror(self):
        digest = make_model(self.store, "qwen3:14b", b"local")
        ref = models.resolve("M", "ollama", self.cfg())
        self.assertEqual(ref.digest, digest)

    def test_missing_means_pull_by_the_adapter(self):
        ref = models.resolve("M", "ollama", self.cfg())
        self.assertIsNone(ref.digest)
        self.assertEqual(ref.size_bytes, models.catalog()["M"]["size_bytes"])
        with self.assertRaises(ModelNotFound):
            models.resolve("M", "ollama", self.cfg(no_download=True))

    def test_large_pull_needs_consent(self):
        with self.assertRaises(NeedsConsent):
            models.resolve("F", "ollama", self.cfg())
        ref = models.resolve("F", "ollama", self.cfg(yes=True))
        self.assertEqual(ref.ollama_tag, "flint:40b")


if __name__ == "__main__":
    unittest.main()
