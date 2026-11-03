import hashlib
import io
import json
import os
import stat
import tarfile
import tempfile
import time
import unittest
import urllib.request
import zipfile
from pathlib import Path
from unittest import mock

from bench.config import Config
from bench.engines import release
from bench.engines.release import (DEFAULT_RELEASE, ReleaseError, ensure_llamacpp,
                                   select_asset, select_runtime_asset)
from fake_http import FakeServer

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "llamacpp"
REAL = json.loads((FIXTURES / "release-b12061.json").read_text(encoding="utf-8"))
TAG = "b12061"


class SelectAssetTest(unittest.TestCase):
    def pick(self, os_name, arch, backend):
        return select_asset(REAL, os_name, arch, backend)

    def test_select_asset_win_cuda(self):
        # "cuda" means the 12.x build: CUDA 13 dropped Pascal (GTX 10xx).
        self.assertEqual(self.pick("windows", "x64", "cuda"),
                         "llama-b12061-bin-win-cuda-12.4-x64.zip")

    def test_cuda13_only_when_asked(self):
        self.assertEqual(self.pick("windows", "x64", "cuda13"),
                         "llama-b12061-bin-win-cuda-13.4-x64.zip")
        self.assertEqual(self.pick("linux", "arm64", "cuda13"),
                         "llama-b12061-bin-ubuntu-cuda-13.4-arm64.tar.gz")

    def test_select_asset_linux_vulkan(self):
        self.assertEqual(self.pick("linux", "x64", "vulkan"),
                         "llama-b12061-bin-ubuntu-vulkan-x64.tar.gz")

    def test_select_asset_macos_arm(self):
        self.assertEqual(self.pick("macos", "arm64", "metal"),
                         "llama-b12061-bin-macos-arm64.tar.gz")
        self.assertEqual(self.pick("darwin", "arm64", "cpu"),
                         "llama-b12061-bin-macos-arm64.tar.gz")
        self.assertEqual(self.pick("macos", "x64", "metal"),
                         "llama-b12061-bin-macos-x64.tar.gz")

    def test_cpu_and_other_backends(self):
        self.assertEqual(self.pick("linux", "x64", "cpu"),
                         "llama-b12061-bin-ubuntu-x64.tar.gz")
        self.assertEqual(self.pick("linux", "arm64", "cpu"),
                         "llama-b12061-bin-ubuntu-arm64.tar.gz")
        self.assertEqual(self.pick("windows", "x64", "cpu"),
                         "llama-b12061-bin-win-cpu-x64.zip")
        self.assertEqual(self.pick("windows", "arm64", "vulkan"),
                         "llama-b12061-bin-win-vulkan-arm64.zip")
        self.assertEqual(self.pick("linux", "x64", "cuda"),
                         "llama-b12061-bin-ubuntu-cuda-12.8-x64.tar.gz")
        self.assertEqual(self.pick("linux", "x64", "rocm"),
                         "llama-b12061-bin-ubuntu-rocm-10.0-x64.tar.gz")
        self.assertEqual(self.pick("windows", "x64", "rocm"),
                         "llama-b12061-bin-win-rocm-10.0-x64.zip")

    def test_platform_spellings_are_normalised(self):
        self.assertEqual(self.pick("Windows", "AMD64", "cuda"),
                         "llama-b12061-bin-win-cuda-12.4-x64.zip")
        self.assertEqual(self.pick("Linux", "x86_64", "cpu"),
                         "llama-b12061-bin-ubuntu-x64.tar.gz")
        self.assertEqual(self.pick("Darwin", "aarch64", "metal"),
                         "llama-b12061-bin-macos-arm64.tar.gz")

    def test_missing_combinations_are_errors(self):
        for args in (("windows", "arm64", "cuda"),   # only CUDA 13 exists for arm64
                     ("linux", "arm64", "cuda"),
                     ("macos", "arm64", "cuda"),
                     ("linux", "x64", "metal"),
                     ("linux", "riscv", "cpu"),
                     ("plan9", "x64", "cpu"),
                     ("linux", "x64", "opencl")):
            with self.subTest(args=args), self.assertRaises(ReleaseError):
                self.pick(*args)

    def test_picks_highest_minor_of_a_major(self):
        names = ["llama-b1-bin-win-cuda-12.4-x64.zip", "llama-b1-bin-win-cuda-12.9-x64.zip",
                 "llama-b1-bin-win-cuda-12.10-x64.zip", "llama-b1-bin-win-cuda-13.1-x64.zip"]
        data = {"tag_name": "b1", "assets": [{"name": n} for n in names]}
        self.assertEqual(select_asset(data, "windows", "x64", "cuda"),
                         "llama-b1-bin-win-cuda-12.10-x64.zip")

    def test_ignores_runtime_ui_and_other_bundles(self):
        for backend in ("cpu", "cuda", "vulkan"):
            for os_name in ("windows", "linux"):
                name = self.pick(os_name, "x64", backend)
                self.assertTrue(name.startswith("llama-b12061-bin-"), name)
                self.assertNotIn("cudart", name)

    def test_release_without_tag_is_an_error(self):
        with self.assertRaises(ReleaseError):
            select_asset({"assets": []}, "linux", "x64", "cpu")

    def test_runtime_asset_only_for_windows_cuda(self):
        self.assertEqual(
            select_runtime_asset(REAL, "llama-b12061-bin-win-cuda-12.4-x64.zip"),
            "cudart-llama-bin-win-cuda-12.4-x64.zip")
        self.assertEqual(
            select_runtime_asset(REAL, "llama-b12061-bin-win-cuda-13.4-x64.zip"),
            "cudart-llama-bin-win-cuda-13.4-x64.zip")
        self.assertIsNone(select_runtime_asset(REAL, "llama-b12061-bin-win-vulkan-x64.zip"))
        self.assertIsNone(select_runtime_asset(
            REAL, "llama-b12061-bin-ubuntu-cuda-12.8-x64.tar.gz"))

    def test_runtime_asset_missing_from_release_is_an_error(self):
        data = {"tag_name": "b1", "assets": [{"name": "llama-b1-bin-win-cuda-12.4-x64.zip"}]}
        with self.assertRaises(ReleaseError):
            select_runtime_asset(data, "llama-b1-bin-win-cuda-12.4-x64.zip")

    def test_default_backend(self):
        gpu = lambda backend: mock.Mock(backend=backend)  # noqa: E731
        self.assertEqual(release.default_backend("macos", []), "metal")
        self.assertEqual(release.default_backend("windows", [gpu("cuda")]), "cuda")
        self.assertEqual(release.default_backend("linux", [gpu("vulkan")]), "vulkan")
        self.assertEqual(release.default_backend("linux", [gpu(None), gpu("rocm")]), "rocm")
        self.assertEqual(release.default_backend("linux", []), "cpu")

    def test_pinned_default_tag(self):
        self.assertEqual(DEFAULT_RELEASE, "b12061")


# -- archives ------------------------------------------------------------------------------

def make_zip(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def make_tar(entries: list) -> bytes:
    """entries: (name, data) for files, (name, None, target) for symlinks,
    (name, None, target, "hard") for hard links."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for entry in entries:
            info = tarfile.TarInfo(entry[0])
            if len(entry) >= 3:
                info.type = tarfile.LNKTYPE if len(entry) == 4 else tarfile.SYMTYPE
                info.linkname = entry[2]
                archive.addfile(info)
            else:
                info.size = len(entry[1])
                info.mode = 0o755
                archive.addfile(info, io.BytesIO(entry[1]))
    return buffer.getvalue()


class TempDirCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)


class ExtractTest(TempDirCase):
    def extract(self, data: bytes, suffix: str) -> Path:
        archive = self.dir / ("archive" + suffix)
        archive.write_bytes(data)
        out = Path(tempfile.mkdtemp(dir=self.dir))  # evil members would land in self.dir
        release.safe_extract(archive, out)
        return out

    def test_zip_roundtrip(self):
        out = self.extract(make_zip({"llama-server.exe": b"x", "sub/dir/a.dll": b"y"}), ".zip")
        self.assertEqual((out / "llama-server.exe").read_bytes(), b"x")
        self.assertEqual((out / "sub" / "dir" / "a.dll").read_bytes(), b"y")

    def test_tar_roundtrip_with_internal_symlink(self):
        out = self.extract(make_tar([("top/llama-server", b"bin"),
                                     ("top/libllama.so.0", b"lib"),
                                     ("top/libllama.so", None, "libllama.so.0")]), ".tar.gz")
        self.assertEqual((out / "top" / "libllama.so").read_bytes(), b"lib")
        self.assertTrue((out / "top" / "libllama.so").is_symlink())
        if os.name != "nt":
            self.assertTrue((out / "top" / "llama-server").stat().st_mode & stat.S_IXUSR)

    def test_zip_slip_is_refused(self):
        for name in ("../evil.txt", "a/../../evil.txt", "/abs/evil.txt", "C:/evil.txt",
                     "..\\evil.txt"):
            with self.subTest(name=name), self.assertRaises(ReleaseError):
                self.extract(make_zip({name: b"x"}), ".zip")
        self.assertFalse((self.dir / "evil.txt").exists())

    def test_zip_symlink_is_refused(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            info = zipfile.ZipInfo("link")
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, "/etc/passwd")
        with self.assertRaises(ReleaseError):
            self.extract(buffer.getvalue(), ".zip")

    def test_tar_traversal_is_refused(self):
        for entry in (("../evil.txt", b"x"), ("/abs/evil.txt", b"x"),
                      ("a/../../evil.txt", b"x"),
                      ("link", None, "../../outside"),
                      ("link", None, "/etc/passwd"),
                      ("hard", None, "../outside", "hard"),
                      ("hard", None, "/etc/passwd", "hard")):
            with self.subTest(entry=entry[0] + str(entry[2:])), self.assertRaises(ReleaseError):
                self.extract(make_tar([entry]), ".tar.gz")
        self.assertFalse((self.dir / "evil.txt").exists())

    def test_symlink_chain_that_only_escapes_physically_is_refused(self):
        # a/b/s -> ../.. is the root, so a/b/s/l -> ../../../x points outside although
        # it looks harmless name by name. Python before 3.11.4 has no data_filter to catch it.
        entries = [("a/b/s", None, "../.."), ("a/b/s/l", None, "../../../x")]
        with mock.patch.object(tarfile, "data_filter", None):
            with self.assertRaises(ReleaseError):
                self.extract(make_tar(entries), ".tar.gz")
        with self.assertRaises(ReleaseError):  # and with the filter
            self.extract(make_tar(entries), ".tar.gz")

    def test_harmless_symlinks_still_work_without_data_filter(self):
        with mock.patch.object(tarfile, "data_filter", None):
            out = self.extract(make_tar([("top/lib.so.0", b"lib"),
                                         ("top/lib.so", None, "lib.so.0")]), ".tar.gz")
        self.assertEqual((out / "top" / "lib.so").read_bytes(), b"lib")

    def test_tar_special_files_are_refused(self):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            info = tarfile.TarInfo("dev")
            info.type = tarfile.CHRTYPE
            archive.addfile(info)
        with self.assertRaises(ReleaseError):
            self.extract(buffer.getvalue(), ".tar.gz")

    def test_symlink_into_a_symlinked_dir_that_escapes_is_refused(self):
        entries = [("top/out", None, "../.."), ("top/out/evil.txt", b"x")]
        with self.assertRaises(ReleaseError):
            self.extract(make_tar(entries), ".tar.gz")
        self.assertFalse((self.dir / "evil.txt").exists())

    def test_unknown_archive_and_corrupt_archive(self):
        with self.assertRaises(ReleaseError):
            self.extract(b"hello", ".rar")
        with self.assertRaises(ReleaseError):
            self.extract(b"not a zip", ".zip")
        with self.assertRaises(ReleaseError):
            self.extract(b"not a tar", ".tar.gz")


# -- download ------------------------------------------------------------------------------

def make_release(server_url: str, assets: dict[str, bytes], tag: str = TAG,
                 digest: bool = True) -> dict:
    items = []
    for name, data in assets.items():
        item = {"name": name, "size": len(data),
                "browser_download_url": f"{server_url}/dl/{name}"}
        if digest:
            item["digest"] = "sha256:" + hashlib.sha256(data).hexdigest()
        items.append(item)
    return {"tag_name": tag, "prerelease": True, "assets": items}


def local_opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


class DownloadTest(TempDirCase):
    def serve(self, data: bytes, **asset):
        server = FakeServer({"/dl/a.zip": data})
        server.__enter__()
        self.addCleanup(server.__exit__)
        entry = {"name": "a.zip", "size": len(data),
                 "browser_download_url": server.url + "/dl/a.zip",
                 "digest": "sha256:" + hashlib.sha256(data).hexdigest()}
        entry.update(asset)
        return server, entry

    def fetch(self, entry) -> Path:
        dest = self.dir / "a.zip"
        release.download_asset(entry, dest, local_opener())
        return dest

    def test_good_download(self):
        _, entry = self.serve(b"payload")
        self.assertEqual(self.fetch(entry).read_bytes(), b"payload")

    def test_final_replace_retries_on_a_transient_lock(self):
        _, entry = self.serve(b"payload")
        real = os.replace
        calls = []

        def flaky(src, dst):
            calls.append(src)
            if len(calls) == 1:
                raise PermissionError("locked")
            return real(src, dst)
        with mock.patch("bench.engines.release.os.replace", flaky):
            self.assertEqual(self.fetch(entry).read_bytes(), b"payload")
        self.assertEqual(len(calls), 2)

    def test_size_mismatch_is_refused(self):
        _, entry = self.serve(b"payload", size=100)
        with self.assertRaises(ReleaseError) as ctx:
            self.fetch(entry)
        self.assertIn("size", str(ctx.exception))
        self.assertFalse((self.dir / "a.zip").exists())

    def test_oversized_download_is_cut_off(self):
        _, entry = self.serve(b"x" * 5000, size=10)
        with self.assertRaises(ReleaseError):
            self.fetch(entry)
        self.assertFalse((self.dir / "a.zip").exists())

    def test_digest_mismatch_is_refused(self):
        _, entry = self.serve(b"payload", digest="sha256:" + "0" * 64)
        with self.assertRaises(ReleaseError) as ctx:
            self.fetch(entry)
        self.assertIn("digest", str(ctx.exception))
        self.assertFalse((self.dir / "a.zip").exists())

    def test_missing_digest_still_checks_size_and_warns(self):
        _, entry = self.serve(b"payload")
        del entry["digest"]
        said = []
        dest = self.dir / "a.zip"
        info = release.download_asset(entry, dest, local_opener(), said.append)
        self.assertEqual(dest.read_bytes(), b"payload")
        self.assertIs(info["digest_verified"], False)
        warning = [m for m in said if "no sha256 published for a.zip" in m]
        self.assertEqual(len(warning), 1)
        self.assertIn("verified size only", warning[0])
        entry["size"] = 3
        with self.assertRaises(ReleaseError):  # the size is still enforced
            release.download_asset(entry, self.dir / "b.zip", local_opener(), said.append)

    def test_verified_digest_is_reported_without_a_warning(self):
        _, entry = self.serve(b"payload")
        said = []
        info = release.download_asset(entry, self.dir / "a.zip", local_opener(), said.append)
        self.assertEqual(info, {"name": "a.zip", "size": 7, "digest_verified": True})
        self.assertFalse([m for m in said if "no sha256" in m])

    def test_redirect_to_a_forbidden_url_is_refused(self):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}),
                                             release.CheckedRedirectHandler)
        for target in ("http://example.com/a.zip", "ftp://example.com/a.zip",
                       "file:///etc/passwd"):
            with self.subTest(target=target), FakeServer() as server:
                server.routes["/dl/a.zip"] = (302, target)
                entry = {"name": "a.zip", "size": 7, "browser_download_url":
                         server.url + "/dl/a.zip"}
                # The message tells our refusal apart from a DNS or connect error.
                # urllib itself already stops file:// before our handler (HTTP 302).
                expected = "HTTP 302" if target.startswith("file:") else "only https"
                with self.assertRaisesRegex(ReleaseError, expected):
                    release.download_asset(entry, self.dir / "a.zip", opener)
                self.assertFalse((self.dir / "a.zip").exists())

    def test_redirect_handler_refuses_before_any_connection_and_closes_the_reply(self):
        handler = release.CheckedRedirectHandler()
        request = urllib.request.Request("https://github.com/x/a.zip")
        for target in ("http://example.com/a.zip", "ftp://example.com/a.zip",
                       "file:///etc/passwd"):
            with self.subTest(target=target):
                reply = io.BytesIO(b"moved")
                with self.assertRaisesRegex(ReleaseError, "only https is allowed"):
                    handler.redirect_request(request, reply, 302, "Found", {}, target)
                self.assertTrue(reply.closed)
        followed = handler.redirect_request(request, io.BytesIO(), 302, "Found", {},
                                            "https://objects.example.com/a.zip")
        self.assertEqual(followed.full_url, "https://objects.example.com/a.zip")

    def test_redirect_to_an_allowed_url_is_followed(self):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}),
                                             release.CheckedRedirectHandler)
        with FakeServer({"/real": b"payload"}) as server:
            server.routes["/dl/a.zip"] = (302, server.url + "/real")
            entry = {"name": "a.zip", "size": 7,
                     "browser_download_url": server.url + "/dl/a.zip"}
            release.download_asset(entry, self.dir / "a.zip", opener)
        self.assertEqual((self.dir / "a.zip").read_bytes(), b"payload")

    def test_default_opener_checks_redirects(self):
        handlers = release.default_opener().handlers
        self.assertTrue(any(isinstance(h, release.CheckedRedirectHandler) for h in handlers))

    def test_unsupported_digest_algorithm_is_refused(self):
        _, entry = self.serve(b"payload", digest="md5:abc")
        with self.assertRaises(ReleaseError):
            self.fetch(entry)

    def test_missing_size_is_refused(self):
        _, entry = self.serve(b"payload")
        del entry["size"]
        with self.assertRaises(ReleaseError):
            self.fetch(entry)

    def test_urls_must_be_https_or_loopback(self):
        for url in ("file:///etc/passwd", "ftp://example.com/a.zip",
                    "http://example.com/a.zip", "http://127.0.0.1.evil.com/a.zip"):
            entry = {"name": "a.zip", "size": 1, "browser_download_url": url}
            with self.subTest(url=url), self.assertRaises(ReleaseError):
                self.fetch(entry)

    def test_http_error_is_a_release_error(self):
        server, entry = self.serve(b"payload")
        entry["browser_download_url"] = server.url + "/missing"
        with self.assertRaises(ReleaseError):
            self.fetch(entry)

    def test_release_json_is_fetched_by_tag_never_latest(self):
        data = make_release("http://x", {"a.zip": b"1"})
        with FakeServer({f"/repos/ggml-org/llama.cpp/releases/tags/{TAG}": data}) as server:
            got = release.fetch_release_json(TAG, local_opener(), server.url)
        self.assertEqual(got["tag_name"], TAG)
        self.assertEqual([p for _, p, _ in server.requests],
                         [f"/repos/ggml-org/llama.cpp/releases/tags/{TAG}"])

    def test_release_json_errors(self):
        with FakeServer({"/repos/ggml-org/llama.cpp/releases/tags/b1": "not json"}) as server:
            for tag in ("b1", "b2"):  # unreadable reply, then 404
                with self.subTest(tag=tag), self.assertRaises(ReleaseError):
                    release.fetch_release_json(tag, local_opener(), server.url)
            wrong = {"/repos/ggml-org/llama.cpp/releases/tags/b3": {"tag_name": "b9"}}
            server.routes.update(wrong)
            with self.assertRaises(ReleaseError):  # asked for b3, got b9
                release.fetch_release_json("b3", local_opener(), server.url)


# -- ensure_llamacpp -----------------------------------------------------------------------

LINUX_TAR = make_tar([
    (f"llama-{TAG}/llama-server", b"#!server"),
    (f"llama-{TAG}/llama-bench", b"#!bench"),
    (f"llama-{TAG}/libllama.so.0", b"lib"),
    (f"llama-{TAG}/libllama.so", None, "libllama.so.0"),
])
WIN_CUDA_ZIP = make_zip({"llama-server.exe": b"s", "llama-bench.exe": b"b", "ggml-cuda.dll": b"c"})
CUDART_ZIP = make_zip({"cudart64_12.dll": b"rt", "cublas64_12.dll": b"bl"})


class EnsureTest(TempDirCase):
    def setUp(self):
        super().setUp()
        self.engines = self.dir / "engines"
        self.cfg = Config()

    def serve(self, assets: dict[str, bytes], tag: str = TAG, digest: bool = True):
        server = FakeServer()
        server.__enter__()
        self.addCleanup(server.__exit__)
        server.routes[f"/repos/ggml-org/llama.cpp/releases/tags/{tag}"] = make_release(
            server.url, assets, tag, digest)
        for name, data in assets.items():
            server.routes[f"/dl/{name}"] = data
        return server

    def ensure(self, server, os_name="linux", arch="x64", **kw):
        kw.setdefault("say", lambda _: None)
        return ensure_llamacpp(self.cfg, opener=local_opener(), engines_dir=self.engines,
                               os_name=os_name, arch=arch, api_base=server.url, **kw)

    def downloads(self, server):
        return [p for _, p, _ in server.requests if p.startswith("/dl/")]

    def test_linux_download_extract_and_reuse(self):
        self.cfg.llama_backend = "cpu"
        asset = f"llama-{TAG}-bin-ubuntu-x64.tar.gz"
        server = self.serve({asset: LINUX_TAR})
        server_bin, bench_bin = self.ensure(server)
        self.assertEqual(server_bin.name, "llama-server")
        self.assertEqual(bench_bin.name, "llama-bench")
        self.assertEqual(server_bin.read_bytes(), b"#!server")
        # Only under .bench/engines/llama.cpp/<tag>/ (here: below the injected root).
        self.assertTrue(str(server_bin).startswith(str(self.engines / TAG)))
        if os.name != "nt":
            self.assertTrue(os.access(server_bin, os.X_OK))
        self.assertEqual(self.downloads(server), ["/dl/" + asset])
        # No scratch files left behind next to the result.
        self.assertEqual(sorted(p.name for p in (self.engines / TAG).iterdir()),
                         ["llama-b12061-bin-ubuntu-x64", "release.json"])
        # Second call: nothing is fetched again, not even the release JSON.
        before = len(server.requests)
        self.assertEqual(self.ensure(server), (server_bin, bench_bin))
        self.assertEqual(len(server.requests), before)

    def test_install_metadata_records_whether_digests_were_verified(self):
        self.cfg.llama_backend = "cpu"
        asset = f"llama-{TAG}-bin-ubuntu-x64.tar.gz"
        said = []
        server = self.serve({asset: LINUX_TAR}, digest=False)
        self.ensure(server, say=said.append)
        folder = self.engines / TAG / f"llama-{TAG}-bin-ubuntu-x64"
        meta = json.loads((folder / "install.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["tag"], TAG)
        self.assertEqual(meta["assets"], [{"name": asset, "size": len(LINUX_TAR),
                                           "digest_verified": False}])
        self.assertTrue([m for m in said if "no sha256 published" in m])

    def test_install_metadata_with_digests(self):
        self.cfg.llama_backend = "cuda"
        main = f"llama-{TAG}-bin-win-cuda-12.4-x64.zip"
        server = self.serve({main: WIN_CUDA_ZIP,
                             "cudart-llama-bin-win-cuda-12.4-x64.zip": CUDART_ZIP})
        self.ensure(server, os_name="windows")
        folder = self.engines / TAG / f"llama-{TAG}-bin-win-cuda-12.4-x64"
        meta = json.loads((folder / "install.json").read_text(encoding="utf-8"))
        self.assertEqual([a["digest_verified"] for a in meta["assets"]], [True, True])

    def test_old_partial_folders_are_removed_but_young_ones_stay(self):
        self.cfg.llama_backend = "cpu"
        tag_dir = self.engines / TAG
        old, young = tag_dir / ".partial-old", tag_dir / ".partial-young"
        other = self.engines / "b1" / ".partial-other-tag"
        for folder in (old, young, other):
            (folder / "out").mkdir(parents=True)
            (folder / "out" / "x").write_bytes(b"x")
        busy = tag_dir / ".partial-busy"  # old folder, but a file inside is being written
        (busy / "out").mkdir(parents=True)
        (busy / "out" / "x").write_bytes(b"x")
        stale = time.time() - 2 * 3600
        for folder in (old, other):
            for path in (folder, folder / "out", folder / "out" / "x"):
                os.utime(path, (stale, stale))
        os.utime(busy, (stale, stale))
        os.utime(busy / "out", (stale, stale))
        server = self.serve({f"llama-{TAG}-bin-ubuntu-x64.tar.gz": LINUX_TAR})
        self.ensure(server)
        self.assertFalse(old.exists())
        self.assertTrue(young.exists())  # maybe another run is working in it
        self.assertTrue(busy.exists())  # a download is still writing into it
        self.assertTrue(other.exists())  # other tags are not touched

    def test_windows_cuda_gets_the_runtime_in_the_same_folder(self):
        self.cfg.llama_backend = "cuda"
        main = f"llama-{TAG}-bin-win-cuda-12.4-x64.zip"
        server = self.serve({main: WIN_CUDA_ZIP,
                             "cudart-llama-bin-win-cuda-12.4-x64.zip": CUDART_ZIP,
                             "cudart-llama-bin-win-cuda-13.4-x64.zip": b"not wanted"})
        server_bin, bench_bin = self.ensure(server, os_name="windows")
        self.assertEqual(server_bin.name, "llama-server.exe")
        self.assertEqual(bench_bin.name, "llama-bench.exe")
        self.assertEqual(server_bin.parent, bench_bin.parent)
        self.assertEqual((server_bin.parent / "cudart64_12.dll").read_bytes(), b"rt")
        self.assertEqual(sorted(self.downloads(server)),
                         sorted(["/dl/" + main, "/dl/cudart-llama-bin-win-cuda-12.4-x64.zip"]))

    def test_cuda13_is_explicit(self):
        self.cfg.llama_backend = "cuda13"
        main = f"llama-{TAG}-bin-win-cuda-13.4-x64.zip"
        server = self.serve({main: WIN_CUDA_ZIP,
                             "cudart-llama-bin-win-cuda-13.4-x64.zip": CUDART_ZIP})
        self.ensure(server, os_name="windows")
        self.assertIn("/dl/" + main, self.downloads(server))

    def test_default_tag_is_pinned_and_latest_is_never_used(self):
        self.cfg.llama_backend = "cpu"
        asset = f"llama-{DEFAULT_RELEASE}-bin-ubuntu-x64.tar.gz"
        server = self.serve({asset: LINUX_TAR}, tag=DEFAULT_RELEASE)
        self.ensure(server)
        for _, path, _ in server.requests:
            self.assertNotIn("latest", path)
        self.assertTrue((self.engines / DEFAULT_RELEASE).is_dir())

    def test_release_override(self):
        self.cfg.llama_backend = "cpu"
        self.cfg.llama_release = "b777"
        asset = "llama-b777-bin-ubuntu-x64.tar.gz"
        server = self.serve({asset: LINUX_TAR}, tag="b777")
        server_bin, _ = self.ensure(server)
        self.assertIn("/b777/", str(server_bin).replace(os.sep, "/"))

    def test_backend_is_detected_when_not_given(self):
        asset = f"llama-{TAG}-bin-ubuntu-vulkan-x64.tar.gz"
        server = self.serve({asset: LINUX_TAR})
        self.ensure(server, gpus=[mock.Mock(backend="vulkan")])
        self.assertEqual(self.downloads(server), ["/dl/" + asset])

    def test_different_backends_do_not_collide(self):
        cpu = f"llama-{TAG}-bin-ubuntu-x64.tar.gz"
        vk = f"llama-{TAG}-bin-ubuntu-vulkan-x64.tar.gz"
        server = self.serve({cpu: LINUX_TAR, vk: LINUX_TAR})
        self.cfg.llama_backend = "cpu"
        first = self.ensure(server)
        self.cfg.llama_backend = "vulkan"
        second = self.ensure(server)
        self.assertNotEqual(first[0].parent, second[0].parent)
        self.assertTrue(first[0].exists() and second[0].exists())

    def test_tag_must_be_a_plain_name(self):
        self.cfg.llama_backend = "cpu"
        for tag in ("../x", "a/b", "a\\b", ".", "..", "b1 2", "latest/../../x"):
            self.cfg.llama_release = tag
            with self.subTest(tag=tag), self.assertRaises(ReleaseError):
                ensure_llamacpp(self.cfg, opener=mock.Mock(), engines_dir=self.engines,
                                os_name="linux", arch="x64")
        self.assertFalse(self.engines.exists())

    def test_failed_download_leaves_nothing_behind(self):
        self.cfg.llama_backend = "cpu"
        asset = f"llama-{TAG}-bin-ubuntu-x64.tar.gz"
        server = self.serve({asset: LINUX_TAR})
        server.routes["/dl/" + asset] = LINUX_TAR[:-10]  # truncated
        with self.assertRaises(ReleaseError):
            self.ensure(server)
        leftovers = [p.name for p in (self.engines / TAG).iterdir()] \
            if (self.engines / TAG).exists() else []
        self.assertEqual([n for n in leftovers if n != "release.json"], [])

    def test_failed_extract_leaves_nothing_behind(self):
        self.cfg.llama_backend = "cpu"
        asset = f"llama-{TAG}-bin-ubuntu-x64.tar.gz"
        evil = make_tar([(f"llama-{TAG}/llama-server", b"s"), ("../evil", b"x")])
        server = self.serve({asset: evil})
        with self.assertRaises(ReleaseError):
            self.ensure(server)
        names = [p.name for p in (self.engines / TAG).iterdir()]
        self.assertEqual([n for n in names if n != "release.json"], [])
        self.assertFalse((self.engines / "evil").exists())

    def test_archive_without_binaries_is_an_error_and_not_kept(self):
        self.cfg.llama_backend = "cpu"
        asset = f"llama-{TAG}-bin-ubuntu-x64.tar.gz"
        server = self.serve({asset: make_tar([("readme.txt", b"hi")])})
        with self.assertRaises(ReleaseError):
            self.ensure(server)
        names = [p.name for p in (self.engines / TAG).iterdir()]
        self.assertEqual([n for n in names if n != "release.json"], [])

    def test_no_download_without_cache_is_an_error_without_network(self):
        self.cfg.llama_backend = "cpu"
        self.cfg.no_download = True
        opener = mock.Mock()
        with self.assertRaises(ReleaseError):
            ensure_llamacpp(self.cfg, opener=opener, engines_dir=self.engines,
                            os_name="linux", arch="x64")
        opener.open.assert_not_called()

    def test_no_install_also_forbids_fetching(self):
        self.cfg.llama_backend = "cpu"
        self.cfg.no_install = True
        opener = mock.Mock()
        with self.assertRaises(ReleaseError):
            ensure_llamacpp(self.cfg, opener=opener, engines_dir=self.engines,
                            os_name="linux", arch="x64")
        opener.open.assert_not_called()

    def test_no_download_uses_the_cache(self):
        self.cfg.llama_backend = "cpu"
        server = self.serve({f"llama-{TAG}-bin-ubuntu-x64.tar.gz": LINUX_TAR})
        expected = self.ensure(server)
        self.cfg.no_download = True
        opener = mock.Mock()
        got = ensure_llamacpp(self.cfg, opener=opener, engines_dir=self.engines,
                              os_name="linux", arch="x64")
        self.assertEqual(got, expected)
        opener.open.assert_not_called()

    def test_given_binaries_are_used_without_network(self):
        server_bin = self.dir / "my" / "llama-server"
        bench_bin = self.dir / "my" / "llama-bench"
        server_bin.parent.mkdir()
        server_bin.write_bytes(b"s")
        bench_bin.write_bytes(b"b")
        self.cfg.llama_server = server_bin
        opener = mock.Mock()
        # llama-bench is looked up next to the given server.
        self.assertEqual(ensure_llamacpp(self.cfg, opener=opener, engines_dir=self.engines),
                         (server_bin, bench_bin))
        other = self.dir / "elsewhere-bench"
        other.write_bytes(b"b")
        self.cfg.llama_bench = other
        self.assertEqual(ensure_llamacpp(self.cfg, opener=opener, engines_dir=self.engines),
                         (server_bin, other))
        opener.open.assert_not_called()
        self.assertFalse(self.engines.exists())

    def test_given_binaries_must_exist(self):
        self.cfg.llama_server = self.dir / "nope"
        with self.assertRaises(ReleaseError):
            ensure_llamacpp(self.cfg, engines_dir=self.engines)
        (self.dir / "llama-server").write_bytes(b"s")
        self.cfg.llama_server = self.dir / "llama-server"  # no bench next to it
        with self.assertRaises(ReleaseError):
            ensure_llamacpp(self.cfg, engines_dir=self.engines)

    def test_default_engines_dir_is_inside_dot_bench(self):
        self.assertEqual(release.ENGINES_DIR, Path(".bench") / "engines" / "llama.cpp")

    def test_cached_release_json_allows_offline_selection(self):
        self.cfg.llama_backend = "cpu"
        asset = f"llama-{TAG}-bin-ubuntu-x64.tar.gz"
        server = self.serve({asset: LINUX_TAR})
        self.ensure(server)
        # Remove the extracted build only; the release JSON stays and no API call is needed.
        import shutil
        for child in (self.engines / TAG).iterdir():
            if child.is_dir():
                shutil.rmtree(child)
        before = [p for _, p, _ in server.requests if "/releases/" in p]
        self.ensure(server)
        after = [p for _, p, _ in server.requests if "/releases/" in p]
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
