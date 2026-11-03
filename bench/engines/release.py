"""Select, download and unpack an official llama.cpp release.

Pinned, never "latest": `DEFAULT_RELEASE` is the default and `--llama-release <tag>`
overrides it. `releases/latest` on GitHub now points to a versioned release without
binaries; the binaries live on the `bNNNNN` tags (marked prerelease).

Backends: `cuda` means the CUDA 12.x build because CUDA 13 dropped Pascal (GTX 10xx);
`cuda13` picks the 13.x build explicitly. On Windows the CUDA runtime is a separate
asset (`cudart-llama-bin-win-cuda-<ver>-<arch>.zip`) and is unpacked into the same folder.

Linux CUDA builds get no separate runtime bundle (the `cudart-llama-<tag>-bin-ubuntu-*`
archives are not fetched): they rely on the CUDA libraries of the system.

Everything lands in `.bench/engines/llama.cpp/<tag>/<asset name>/`, never in a system path.
Download and unpacking happen in a scratch directory next to the target and the finished
folder is renamed into place, so an interrupted run leaves no half-installed engine.
Size (from the API) and the SHA-256 digest (when the API offers one) are verified; without
a digest only the size is checked, a warning is printed and `install.json` in the installed
folder records `digest_verified: false` per asset. Redirects are checked like the first URL
(no downgrade from https), archive members that would leave the target folder are refused,
and `.partial-*` scratch folders older than an hour (hard-killed runs) are removed.

Archive links: symlink and hard link members are checked by name before extraction and by
real path (after resolving earlier links) after extraction, on every Python version; on
3.11.4 and later tarfile's "data" filter is applied on top. A failing `safe_extract` may leave
a partly filled target, so callers extract into a scratch folder (`ensure_llamacpp` does).
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import platform
import posixpath
import re
import shutil
import stat
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

from bench.config import Config
from bench.store import retry_on_lock

DEFAULT_RELEASE = "b12061"
REPO = "ggml-org/llama.cpp"
GITHUB_API = "https://api.github.com"
ENGINES_DIR = Path(".bench") / "engines" / "llama.cpp"
USER_AGENT = "llm-benchmarks"
API_TIMEOUT_S = 30
DOWNLOAD_TIMEOUT_S = 60
CHUNK = 1 << 20
PARTIAL_MAX_AGE_S = 3600
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
DRIVE_RE = re.compile(r"^[A-Za-z]:")
SERVER_NAMES = ("llama-server", "llama-server.exe")
BENCH_NAMES = ("llama-bench", "llama-bench.exe")

OS_ALIASES = {"windows": "windows", "win32": "windows", "win": "windows",
              "linux": "linux", "darwin": "macos", "macos": "macos", "mac": "macos"}
ARCH_ALIASES = {"x64": "x64", "x86_64": "x64", "amd64": "x64",
                "arm64": "arm64", "aarch64": "arm64"}
OS_PREFIX = {"windows": "win", "linux": "ubuntu", "macos": "macos"}
OS_SUFFIX = {"windows": r"\.zip", "linux": r"\.tar\.gz", "macos": r"\.tar\.gz"}
# Asset name part between "bin-<os>-" and "-<arch>", per backend, as a regex.
# The first group (if any) is the version used to pick the newest of several builds.
BACKEND_VARIANTS = {
    "cpu": {"windows": r"cpu", "linux": None},
    "vulkan": {"windows": r"vulkan", "linux": r"vulkan"},
    "cuda": {"windows": r"cuda-(12\.\d+)", "linux": r"cuda-(12\.\d+)"},
    "cuda13": {"windows": r"cuda-(13\.\d+)", "linux": r"cuda-(13\.\d+)"},
    "rocm": {"windows": r"rocm-(\d+\.\d+)", "linux": r"rocm-(\d+\.\d+)"},
}


class ReleaseError(RuntimeError):
    """A release cannot be selected, downloaded, verified or unpacked."""


def _normalise(table: dict[str, str], value: str, what: str) -> str:
    try:
        return table[str(value).strip().lower()]
    except KeyError:
        raise ReleaseError(f"unsupported {what} {value!r}") from None


def _version_key(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


def select_asset(release_json: dict, os_name: str, arch: str, backend: str) -> str:
    """Name of the release asset for this platform and backend (see module docstring)."""
    tag = release_json.get("tag_name")
    if not isinstance(tag, str) or not tag:
        raise ReleaseError("release data has no tag_name")
    os_key = _normalise(OS_ALIASES, os_name, "operating system")
    arch_key = _normalise(ARCH_ALIASES, arch, "architecture")
    if backend not in BACKEND_VARIANTS and backend != "metal":
        raise ReleaseError(f"unsupported backend {backend!r}")

    if os_key == "macos":
        if backend not in ("metal", "cpu"):
            raise ReleaseError(f"backend {backend!r} is not available for macOS")
        variant = ""
    else:
        if backend == "metal":
            raise ReleaseError(f"backend 'metal' is not available for {os_key}")
        variant = BACKEND_VARIANTS[backend][os_key]
        variant = "" if variant is None else variant + "-"
    pattern = re.compile(
        rf"^llama-{re.escape(tag)}-bin-{OS_PREFIX[os_key]}-{variant}{arch_key}"
        rf"{OS_SUFFIX[os_key]}$")
    found: list[tuple[tuple[int, ...], str]] = []
    for asset in release_json.get("assets") or []:
        name = asset.get("name", "")
        match = pattern.match(name)
        if match:
            version = _version_key(match.group(1)) if match.groups() else ()
            found.append((version, name))
    if not found:
        wanted = f"{os_key}/{arch_key}/{backend}"
        available = sorted(a.get("name", "") for a in release_json.get("assets") or []
                           if a.get("name", "").startswith("llama-"))
        raise ReleaseError(f"release {tag} has no build for {wanted}; "
                           f"assets: {', '.join(available) or 'none'}")
    return max(found)[1]


def select_runtime_asset(release_json: dict, asset_name: str) -> str | None:
    """The separate CUDA runtime asset for a Windows CUDA build; None for all others."""
    match = re.match(r"^llama-.+-bin-win-cuda-(\d+\.\d+)-(\w+)\.zip$", asset_name)
    if not match:
        return None
    runtime = f"cudart-llama-bin-win-cuda-{match.group(1)}-{match.group(2)}.zip"
    names = {a.get("name") for a in release_json.get("assets") or []}
    if runtime not in names:
        raise ReleaseError(f"release has no CUDA runtime asset {runtime}")
    return runtime


def default_backend(os_name: str, gpus: list) -> str:
    """Backend to use when none is given: Metal on macOS, else the first GPU's best one."""
    if _normalise(OS_ALIASES, os_name, "operating system") == "macos":
        return "metal"
    for gpu in gpus:
        if getattr(gpu, "backend", None) in BACKEND_VARIANTS:
            return gpu.backend
    return "cpu"


# -- network -------------------------------------------------------------------------------

def _check_url(url: str) -> None:
    parts = urllib.parse.urlsplit(url)
    if parts.scheme == "https" or (parts.scheme == "http"
                                   and parts.hostname in LOOPBACK_HOSTS):
        return
    raise ReleaseError(f"refusing to download from {url!r}: only https is allowed")


class CheckedRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Applies the URL rules to every redirect target, so https cannot be downgraded."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        try:
            _check_url(newurl)
        except ReleaseError:
            fp.close()  # urllib only closes the 3xx response when we follow it
            raise
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def default_opener():
    """URL opener for GitHub: system proxy settings, redirects checked."""
    return urllib.request.build_opener(CheckedRedirectHandler)


def _http_error(url: str, err: BaseException) -> ReleaseError:
    if isinstance(err, urllib.error.HTTPError):
        return ReleaseError(f"{url}: HTTP {err.code}")
    return ReleaseError(f"{url}: {getattr(err, 'reason', None) or err!r}")


def fetch_release_json(tag: str, opener, api_base: str = GITHUB_API) -> dict:
    """The release record of one tag from the GitHub API (never `releases/latest`)."""
    url = f"{api_base.rstrip('/')}/repos/{REPO}/releases/tags/{tag}"
    _check_url(url)
    request = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json", "User-Agent": USER_AGENT})
    try:
        with opener.open(request, timeout=API_TIMEOUT_S) as response:
            data = json.loads(response.read())
    except (urllib.error.URLError, OSError, http.client.HTTPException) as err:
        raise _http_error(url, err) from None
    except ValueError:
        raise ReleaseError(f"{url}: reply is not JSON") from None
    if not isinstance(data, dict) or data.get("tag_name") != tag \
            or not isinstance(data.get("assets"), list):
        raise ReleaseError(f"{url}: unexpected release data for tag {tag!r}")
    return data


def download_asset(asset: dict, dest: Path, opener,
                   say: Callable[[str], None] | None = None) -> dict:
    """Download one asset to `dest`, checking size and digest; `dest` only appears when ok.

    Returns {"name", "size", "digest_verified"}. Without a published digest only the size
    is verified: `say` gets a warning and `digest_verified` is False.
    """
    url = asset.get("browser_download_url", "")
    _check_url(url)
    size = asset.get("size")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ReleaseError(f"{asset.get('name')}: release data has no usable size")
    expected_digest = None
    digest = asset.get("digest")
    if digest:
        algorithm, _, value = str(digest).partition(":")
        if algorithm != "sha256" or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
            raise ReleaseError(f"{asset.get('name')}: unsupported digest {digest!r}")
        expected_digest = value.lower()
    if say:
        say(f"downloading {asset.get('name')} ({size / 1e6:.0f} MB)")
        if expected_digest is None:
            say(f"warning: no sha256 published for {asset.get('name')}; verified size only")
    part = dest.with_name(dest.name + ".part")
    hasher = hashlib.sha256()
    total = 0
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with opener.open(request, timeout=DOWNLOAD_TIMEOUT_S) as response, \
                part.open("wb") as out:
            _check_url(response.geturl() if hasattr(response, "geturl") else url)
            while True:
                chunk = response.read(CHUNK)
                if not chunk:
                    break
                total += len(chunk)
                if total > size:
                    raise ReleaseError(f"{asset.get('name')}: more data than the "
                                       f"announced size of {size} bytes")
                hasher.update(chunk)
                out.write(chunk)
        if total != size:
            raise ReleaseError(f"{asset.get('name')}: size {total} does not match "
                               f"the announced {size} bytes")
        if expected_digest and hasher.hexdigest() != expected_digest:
            raise ReleaseError(f"{asset.get('name')}: digest does not match the release data")
        retry_on_lock(lambda: os.replace(part, dest))
        return {"name": asset.get("name"), "size": size,
                "digest_verified": expected_digest is not None}
    except (urllib.error.URLError, OSError, http.client.HTTPException) as err:
        raise _http_error(url, err) from None
    finally:
        part.unlink(missing_ok=True)


# -- unpacking -----------------------------------------------------------------------------

def _check_member_name(name: str) -> str:
    """The member name in POSIX form; raises for anything that could leave the target."""
    clean = name.replace("\\", "/")
    if not clean or clean.startswith("/") or DRIVE_RE.match(clean) \
            or ".." in clean.split("/") or "\0" in clean:
        raise ReleaseError(f"archive member {name!r} would leave the target folder")
    return clean


def _inside(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _check_link_target(member: str, link: str, hard: bool) -> None:
    clean = link.replace("\\", "/")
    if not clean or clean.startswith("/") or DRIVE_RE.match(clean) or "\0" in clean:
        raise ReleaseError(f"archive link {member!r} points to {link!r}")
    base = "" if hard else posixpath.dirname(member)  # hard links are relative to the root
    resolved = posixpath.normpath(posixpath.join(base, clean))
    if resolved == ".." or resolved.startswith("../"):
        raise ReleaseError(f"archive link {member!r} points outside the target folder")


def _extract_zip(archive: Path, dest: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        infos = bundle.infolist()
        for info in infos:  # validate everything before writing anything
            _check_member_name(info.filename)
            if stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK:
                raise ReleaseError(f"archive member {info.filename!r} is a symbolic link")
        for info in infos:
            name = _check_member_name(info.filename)
            target = dest / name
            if not _inside(dest, target.parent):
                raise ReleaseError(f"archive member {name!r} would leave the target folder")
            if name.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)
            base = target.name
            if os.name != "nt" and base.startswith("llama-") and "." not in base:
                target.chmod(0o755)


def _extract_tar(archive: Path, dest: Path) -> None:
    with tarfile.open(archive, "r:*") as bundle:
        members = bundle.getmembers()
        for member in members:
            name = _check_member_name(member.name)
            if member.issym() or member.islnk():
                _check_link_target(name, member.linkname, hard=member.islnk())
            elif not (member.isreg() or member.isdir()):
                raise ReleaseError(f"archive member {name!r} is a special file")
        options = {"filter": "data"} if getattr(tarfile, "data_filter", None) else {}
        for member in members:
            target = dest / _check_member_name(member.name)
            # An earlier symlink may already point outside: check where we really write.
            if not _inside(dest, target.parent):
                raise ReleaseError(f"archive member {member.name!r} would leave "
                                   "the target folder")
            if member.islnk() and not _inside(dest, dest / member.linkname):
                raise ReleaseError(f"archive link {member.name!r} points outside "
                                   "the target folder")
            try:
                bundle.extract(member, dest, **options)
            except tarfile.FilterError as err:
                raise ReleaseError(f"archive member {member.name!r} refused: {err}") from None
            if (member.issym() or member.islnk()) and not _inside(dest, target):
                # Names looked harmless, but resolved through earlier links it leads out.
                raise ReleaseError(f"archive link {member.name!r} points outside "
                                   "the target folder")


def safe_extract(archive: Path, dest: Path) -> None:
    """Unpack a .zip or .tar.gz into `dest`; refuses members that would escape it."""
    name = archive.name.lower()
    try:
        if name.endswith(".zip"):
            _extract_zip(archive, dest)
        elif name.endswith((".tar.gz", ".tgz")):
            _extract_tar(archive, dest)
        else:
            raise ReleaseError(f"unknown archive type: {archive.name}")
    except (zipfile.BadZipFile, tarfile.TarError, EOFError) as err:
        raise ReleaseError(f"cannot unpack {archive.name}: {err}") from None
    except OSError as err:
        raise ReleaseError(f"cannot unpack {archive.name}: {err.strerror or err}") from None


def find_binaries(folder: Path) -> tuple[Path, Path] | None:
    """`llama-server` and `llama-bench` inside an unpacked release (shallowest match)."""
    found: dict[str, list[Path]] = {"server": [], "bench": []}
    for current, _, files in os.walk(folder):
        for file in files:
            if file in SERVER_NAMES:
                found["server"].append(Path(current) / file)
            elif file in BENCH_NAMES:
                found["bench"].append(Path(current) / file)
    if not found["server"]:
        return None
    server = min(found["server"], key=lambda p: (len(p.parts), str(p)))
    beside = [p for p in found["bench"] if p.parent == server.parent]
    candidates = beside or found["bench"]
    if not candidates:
        return None
    return server, min(candidates, key=lambda p: (len(p.parts), str(p)))


# -- ensure --------------------------------------------------------------------------------

def _newest_mtime(folder: Path) -> float:
    """Latest change anywhere in `folder`: writing into a file does not touch the folder."""
    newest = folder.stat().st_mtime
    for path in folder.rglob("*"):
        try:
            newest = max(newest, path.lstat().st_mtime)
        except OSError:
            pass
    return newest


def _remove_stale_partials(tag_dir: Path) -> None:
    """Delete scratch folders of hard-killed runs (untouched for an hour), this tag only."""
    cutoff = time.time() - PARTIAL_MAX_AGE_S
    for folder in tag_dir.glob(".partial-*"):
        try:
            if folder.is_dir() and not folder.is_symlink() \
                    and _newest_mtime(folder) < cutoff:
                shutil.rmtree(folder, ignore_errors=True)
        except OSError:
            pass


def _given_binaries(cfg: Config) -> tuple[Path, Path]:
    server = Path(cfg.llama_server)
    if not server.is_file():
        raise ReleaseError(f"llama-server not found: {server}")
    if cfg.llama_bench:
        bench = Path(cfg.llama_bench)
    else:
        bench = next((server.with_name(n) for n in BENCH_NAMES
                      if server.with_name(n).is_file()), server.with_name(BENCH_NAMES[0]))
    if not bench.is_file():
        raise ReleaseError(f"llama-bench not found: {bench} (use --llama-bench)")
    return server, bench


def ensure_llamacpp(cfg: Config, *, opener=None, engines_dir: Path | None = None,
                    os_name: str | None = None, arch: str | None = None,
                    api_base: str = GITHUB_API, gpus: list | None = None,
                    say: Callable[[str], None] | None = None) -> tuple[Path, Path]:
    """Paths of `llama-server` and `llama-bench`: the given ones, else the pinned release.

    The release is downloaded once into `.bench/engines/llama.cpp/<tag>/` and reused.
    With --no-download or --no-install nothing is fetched and a missing build is an error.
    `opener`, `engines_dir`, `os_name`, `arch`, `api_base`, `gpus` and `say` exist for
    tests; the defaults are the real system.
    """
    if cfg.llama_server:
        return _given_binaries(cfg)
    tag = cfg.llama_release or DEFAULT_RELEASE
    if not TAG_RE.match(tag):
        raise ReleaseError(f"invalid llama.cpp release tag {tag!r}")
    os_name = os_name or platform.system()
    arch = arch or platform.machine()
    backend = cfg.llama_backend
    if backend is None:
        if gpus is None:
            from bench import gpu
            gpus = gpu.list_gpus()
        backend = default_backend(os_name, gpus)
    say = say or (lambda message: print(message, file=sys.stderr))
    fetch_allowed = not (cfg.no_download or cfg.no_install)
    tag_dir = Path(engines_dir or ENGINES_DIR) / tag
    cache = tag_dir / "release.json"

    data = None
    try:
        data = json.loads(cache.read_text(encoding="utf-8"))
        if data.get("tag_name") != tag:
            data = None
    except (OSError, ValueError, AttributeError):
        data = None
    if data is None:
        if not fetch_allowed:
            raise ReleaseError(f"llama.cpp {tag} is not installed and downloading is "
                               "disabled (--no-download/--no-install)")
        data = fetch_release_json(tag, opener or default_opener(), api_base)

    asset_name = select_asset(data, os_name, arch, backend)
    stem = re.sub(r"(\.zip|\.tar\.gz)$", "", asset_name)
    final = tag_dir / stem
    if final.is_dir():
        found = find_binaries(final)
        if found is None:
            raise ReleaseError(f"{final} has no llama-server/llama-bench; "
                               "delete that folder to fetch it again")
        return found
    if not fetch_allowed:
        raise ReleaseError(f"llama.cpp {tag} ({asset_name}) is not installed and "
                           "downloading is disabled (--no-download/--no-install)")

    wanted = [asset_name]
    runtime = select_runtime_asset(data, asset_name)
    if runtime:
        wanted.append(runtime)
    by_name = {a.get("name"): a for a in data["assets"]}
    opener = opener or default_opener()
    try:
        tag_dir.mkdir(parents=True, exist_ok=True)
        _remove_stale_partials(tag_dir)
        cache.write_text(json.dumps(data), encoding="utf-8")
        scratch = Path(tempfile.mkdtemp(prefix=".partial-", dir=tag_dir))
    except OSError as err:
        raise ReleaseError(f"cannot prepare {tag_dir}: {err.strerror or err}") from None
    try:
        out = scratch / "out"
        out.mkdir()
        installed = []
        for name in wanted:
            archive = scratch / name
            installed.append(download_asset(by_name[name], archive, opener, say))
            safe_extract(archive, out)
            archive.unlink()
        if find_binaries(out) is None:
            raise ReleaseError(f"{asset_name} contains no llama-server and llama-bench")
        (out / "install.json").write_text(
            json.dumps({"tag": tag, "asset": asset_name, "assets": installed}, indent=2),
            encoding="utf-8")
        try:
            os.replace(out, final)  # atomic: the folder appears complete or not at all
        except OSError:
            if not final.is_dir():
                raise
    except OSError as err:
        raise ReleaseError(f"cannot install {asset_name}: {err.strerror or err}") from None
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    found = find_binaries(final)
    if found is None:
        raise ReleaseError(f"{final} has no llama-server/llama-bench")
    return found
