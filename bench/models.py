"""Model catalog and resolution.

`catalog()` reads `bench/data/models.toml` (size classes S, M, L, F). `resolve()` turns a
model id into a `ModelRef` for one engine. For llama.cpp the order is:

    explicit --gguf  >  mirror  >  local store  >  download

The mirror is a directory (a network share is fine) with two parts: `ollama/` is a plain
Ollama store (`manifests/...`, `blobs/sha256-<hex>`) and `gguf/<family>/<file>.gguf` holds
GGUF files only llama.cpp needs. A mirror root that is itself an Ollama store is accepted
too. For Ollama the mirror is imported into the Ollama store (`import_from_mirror`), the
local store comes next, and what is still missing is left to `ollama pull` by the engine
adapter (the returned ref then has no digest).

Safety rules: nothing in a target store is ever overwritten or deleted (existing blobs and
manifests stay as they are), imports verify every digest while copying and write the
manifest last so a model never appears half copied, and downloads of more than 20 GB need
`--yes` (`NeedsConsent`).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tomllib
from pathlib import Path
from typing import Callable

from bench.config import Config
from bench.engines import release
from bench.engines.base import ModelRef
from bench.store import _retry_on_lock

CATALOG_FILE = Path(__file__).parent / "data" / "models.toml"
CONSENT_BYTES = 20 * 10**9
HF_BASE = "https://huggingface.co"
DEFAULT_MODEL_DIR = Path(".bench") / "models"
DEFAULT_REGISTRY = "registry.ollama.ai"
MODEL_LAYER = "application/vnd.ollama.image.model"
PART_PREFIX = ".bench-"
COPY_CHUNK = 1 << 20
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
HOST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]*(:[0-9]+)?$")
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
URL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
ENGINE_NAMES = ("ollama", "llamacpp")


class ModelError(RuntimeError):
    """A model could not be found, checked or copied."""


class ModelNotFound(ModelError):
    """The model is not at the place that was looked at."""


class NeedsConsent(ModelError):
    """A large download was asked for without `--yes`."""

    def __init__(self, what: str, size_bytes: int):
        self.size_bytes = size_bytes
        super().__init__(f"{what} is {size_bytes / 1e9:.1f} GB to download; "
                         "pass --yes to allow downloads of more than 20 GB")


# -- catalog -------------------------------------------------------------------------------

def catalog() -> dict[str, dict]:
    """Size class -> catalog entry (a fresh copy on every call)."""
    try:
        data = tomllib.loads(CATALOG_FILE.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as err:
        raise ModelError(f"cannot read model catalog {CATALOG_FILE.name}: {err}") from None
    return data


def default_ollama_models_dir() -> Path:
    env = os.environ.get("OLLAMA_MODELS")
    return Path(env) if env else Path.home() / ".ollama" / "models"


# -- Ollama store --------------------------------------------------------------------------

def _manifest_path(tag: str, models_dir: Path) -> Path:
    name, sep, version = tag.partition(":")
    if not sep:
        version = "latest"
    parts = name.split("/")
    if len(parts) == 1:
        host, namespace, model = DEFAULT_REGISTRY, "library", parts[0]
    elif len(parts) == 2:
        host, (namespace, model) = DEFAULT_REGISTRY, parts
    elif len(parts) == 3:
        host, namespace, model = parts
    else:
        raise ModelError(f"invalid Ollama tag {tag!r}")
    if not HOST_RE.match(host) or not all(NAME_RE.match(p) for p in (namespace, model, version)):
        raise ModelError(f"invalid Ollama tag {tag!r}")
    return Path(models_dir) / "manifests" / host / namespace / model / version


def _blob_path(models_dir: Path, digest: str) -> Path:
    return Path(models_dir) / "blobs" / digest.replace(":", "-")


def _read_manifest(path: Path, tag: str) -> dict:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ModelNotFound(f"no manifest for {tag} in {path.parent}") from None
    except (OSError, ValueError) as err:
        raise ModelError(f"unreadable manifest for {tag}: {err}") from None
    entries = []
    if isinstance(manifest, dict):
        entries = [manifest.get("config"), *(manifest.get("layers") or [])]
    layers = manifest.get("layers") if isinstance(manifest, dict) else None
    if not isinstance(layers, list) or not isinstance(manifest.get("config"), dict):
        raise ModelError(f"manifest for {tag} has no config and layers")
    for entry in entries:
        if not isinstance(entry, dict) or not DIGEST_RE.match(str(entry.get("digest"))) \
                or not isinstance(entry.get("size"), int) or isinstance(entry.get("size"), bool):
            raise ModelError(f"manifest for {tag} has an invalid layer entry")
    return manifest


def _model_layer(manifest: dict, tag: str) -> dict:
    for layer in manifest["layers"]:
        if layer.get("mediaType") == MODEL_LAYER:
            return layer
    raise ModelError(f"manifest for {tag} has no {MODEL_LAYER} layer")


def _model_blob(tag: str, models_dir: Path) -> tuple[Path, dict]:
    manifest = _read_manifest(_manifest_path(tag, models_dir), tag)
    layer = _model_layer(manifest, tag)
    blob = _blob_path(models_dir, layer["digest"])
    try:
        actual = blob.stat().st_size
    except OSError:
        raise ModelNotFound(f"blob {blob.name} of {tag} is missing in {blob.parent}") from None
    if actual != layer["size"]:
        raise ModelError(f"blob {blob.name} of {tag} has {actual} bytes, "
                         f"the manifest says {layer['size']}")
    return blob, layer


def ollama_blob_for(tag: str, models_dir: Path) -> Path:
    """Path of the GGUF blob (layer `application/vnd.ollama.image.model`) of `tag`."""
    return _model_blob(tag, Path(models_dir))[0]


def _mirror_ollama_dir(mirror: Path) -> Path:
    mirror = Path(mirror)
    if (mirror / "ollama" / "manifests").is_dir():
        return mirror / "ollama"
    if (mirror / "manifests").is_dir():
        return mirror
    raise ModelNotFound(f"no Ollama store (ollama/manifests) in mirror {mirror}")


def _copy_verified(src: Path, dest: Path, digest: str, size: int) -> None:
    """Copy `src` to `dest` through a scratch file, checking size and sha256 on the way."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(PART_PREFIX + dest.name + ".part")
    hasher = hashlib.sha256()
    total = 0
    try:
        with src.open("rb") as source, part.open("wb") as out:
            while True:
                chunk = source.read(COPY_CHUNK)
                if not chunk:
                    break
                total += len(chunk)
                hasher.update(chunk)
                out.write(chunk)
        if total != size:
            raise ModelError(f"{src.name}: copied {total} bytes, expected {size}")
        if "sha256:" + hasher.hexdigest() != digest:
            raise ModelError(f"{src.name}: sha256 does not match {digest}")
        if dest.exists():  # appeared meanwhile: keep what is there
            return
        _retry_on_lock(lambda: os.replace(part, dest))
    finally:
        part.unlink(missing_ok=True)


def import_from_mirror(mirror: Path, tag: str, ollama_models_dir: Path,
                       say: Callable[[str], None] | None = None) -> None:
    """Copy the manifest and the missing blobs of `tag` from the mirror into an Ollama store.

    Copies the config blob and every blob in `layers[]` (manifest-list entries are not
    followed). Everything the mirror must provide is checked before the first byte is
    copied. Existing blobs and an existing manifest are never overwritten; a present blob
    of unexpected size only triggers a warning through `say`. The manifest comes last.
    """
    source = _mirror_ollama_dir(mirror)
    target = Path(ollama_models_dir)
    manifest_src = _manifest_path(tag, source)
    manifest = _read_manifest(manifest_src, tag)
    wanted: dict[str, int] = {}
    for entry in [manifest["config"], *manifest["layers"]]:
        wanted.setdefault(entry["digest"], entry["size"])
    missing = []
    for digest, size in wanted.items():
        src = _blob_path(source, digest)
        try:
            actual = src.stat().st_size
        except OSError:
            raise ModelError(f"mirror lacks blob {src.name} needed by {tag}") from None
        if actual != size:
            raise ModelError(f"mirror blob {src.name} has {actual} bytes, "
                             f"the manifest says {size}")
        dest = _blob_path(target, digest)
        if dest.exists():
            if say and dest.stat().st_size != size:
                say(f"warning: existing blob {dest.name} has an unexpected size; kept as is")
        else:
            missing.append((src, dest, digest, size))
    needed = sum(item[3] for item in missing)
    if needed:
        target.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(target).free
        if free < needed:
            raise ModelError(f"not enough disk space in {target}: need "
                             f"{needed / 1e9:.1f} GB, {free / 1e9:.1f} GB free")
    for src, dest, digest, size in missing:
        if say:
            say(f"copying {dest.name} ({size / 1e6:.0f} MB)")
        _copy_verified(src, dest, digest, size)
    manifest_dest = _manifest_path(tag, target)
    if not manifest_dest.exists():
        manifest_dest.parent.mkdir(parents=True, exist_ok=True)
        part = manifest_dest.with_name(PART_PREFIX + manifest_dest.name + ".part")
        try:
            part.write_bytes(manifest_src.read_bytes())
            _retry_on_lock(lambda: os.replace(part, manifest_dest))
        finally:
            part.unlink(missing_ok=True)


# -- resolution ----------------------------------------------------------------------------

def _gguf_ref(model_id: str, entry: dict | None, path: Path, digest: str | None,
              size: int | None) -> ModelRef:
    return ModelRef(id=model_id, cls=model_id,
                    quant=(entry or {}).get("quant", "unknown"),
                    ollama_tag=(entry or {}).get("ollama_tag"),
                    gguf_path=path, digest=digest, size_bytes=size)


def _check_consent(what: str, size: int, cfg: Config) -> None:
    if cfg.no_download:
        raise ModelNotFound(f"{what} is not available locally and --no-download is set")
    if size > CONSENT_BYTES and not cfg.yes and not cfg.dry_run:
        raise NeedsConsent(what, size)


def _pinned_file(path: Path, pin: dict) -> Path | None:
    """`path` if it is a file of the pinned size; None when absent; error when wrong size."""
    if not path.is_file():
        return None
    if path.stat().st_size != pin["size_bytes"]:
        raise ModelError(f"{path.name} has {path.stat().st_size} bytes, expected "
                         f"{pin['size_bytes']}; remove or replace it by hand")
    return path


def _download(model_id: str, entry: dict, cfg: Config, opener, hf_base: str,
              say) -> ModelRef:
    pin = entry["gguf"]
    what = f"{pin['repo']}/{pin['file']}"
    _check_consent(what, pin["size_bytes"], cfg)
    digest = "sha256:" + pin["sha256"]
    if cfg.dry_run:
        return ModelRef(model_id, model_id, entry["quant"], entry["ollama_tag"], None,
                        digest, pin["size_bytes"])
    dest = Path(cfg.model_dir or DEFAULT_MODEL_DIR) / pin["file"]
    asset = {"name": pin["file"], "size": pin["size_bytes"], "digest": digest,
             "browser_download_url":
                 f"{hf_base.rstrip('/')}/{pin['repo']}/resolve/{pin['revision']}/{pin['file']}"}
    try:
        release.download_asset(asset, dest, opener or release.default_opener(), say)
    except release.ReleaseError as err:
        raise ModelError(str(err)) from None
    return _gguf_ref(model_id, entry, dest, digest, pin["size_bytes"])


def _resolve_llamacpp(model_id: str, entry: dict, cfg: Config, opener, hf_base: str,
                      say) -> ModelRef:
    pin = entry.get("gguf")
    reuse = bool(entry.get("reuse_ollama_blob") or cfg.reuse_ollama_blobs)
    tag = entry["ollama_tag"]
    store = Path(cfg.ollama_models_dir or default_ollama_models_dir())
    # 1. explicit file
    if model_id in cfg.gguf:
        path = Path(cfg.gguf[model_id])
        if not path.is_file():
            raise ModelError(f"--gguf file for {model_id} not found: {path}")
        return _gguf_ref(model_id, entry, path, None, path.stat().st_size)
    # 2. mirror
    if cfg.mirror:
        if URL_RE.match(cfg.mirror):
            raise ModelError("URL mirrors are not supported yet; use a directory")
        mirror = Path(cfg.mirror)
        if pin and pin.get("mirror_path"):
            found = _pinned_file(mirror / "gguf" / pin["mirror_path"], pin)
            if found:
                return _gguf_ref(model_id, entry, found, "sha256:" + pin["sha256"],
                                 pin["size_bytes"])
        if reuse:
            try:
                blob, layer = _model_blob(tag, _mirror_ollama_dir(mirror))
                return _gguf_ref(model_id, entry, blob, layer["digest"], layer["size"])
            except ModelNotFound:
                pass
    # 3. local store
    if reuse:
        try:
            blob, layer = _model_blob(tag, store)
            return _gguf_ref(model_id, entry, blob, layer["digest"], layer["size"])
        except ModelNotFound:
            pass
    if pin:
        for folder in (Path(cfg.model_dir or DEFAULT_MODEL_DIR),):
            found = _pinned_file(folder / pin["file"], pin)
            if found:
                return _gguf_ref(model_id, entry, found, "sha256:" + pin["sha256"],
                                 pin["size_bytes"])
    # 4. download
    if pin:
        return _download(model_id, entry, cfg, opener, hf_base, say)
    raise ModelNotFound(
        f"no GGUF for {model_id} ({tag}) found; pull it with Ollama, give --mirror, "
        f"or pass --gguf {model_id}=<path>")


def _resolve_ollama(model_id: str, entry: dict, cfg: Config, say) -> ModelRef:
    tag = entry["ollama_tag"]
    store = Path(cfg.ollama_models_dir or default_ollama_models_dir())
    if cfg.mirror:
        if URL_RE.match(cfg.mirror):
            raise ModelError("URL mirrors are not supported yet; use a directory")
        try:
            mirror_dir = _mirror_ollama_dir(Path(cfg.mirror))
            _read_manifest(_manifest_path(tag, mirror_dir), tag)
        except ModelNotFound:
            pass
        else:
            if not cfg.dry_run:
                import_from_mirror(Path(cfg.mirror), tag, store, say)
    try:
        _, layer = _model_blob(tag, store)
        return ModelRef(model_id, model_id, entry["quant"], tag, None, layer["digest"],
                        layer["size"])
    except ModelNotFound:
        pass
    # Not in the store: the Ollama adapter pulls it.
    _check_consent(f"Ollama model {tag}", entry["size_bytes"], cfg)
    return ModelRef(model_id, model_id, entry["quant"], tag, None, None, entry["size_bytes"])


def resolve(model_id: str, engine: str, cfg: Config, *, opener=None, hf_base: str = HF_BASE,
            entries: dict | None = None,
            say: Callable[[str], None] | None = None) -> ModelRef:
    """Find the model `model_id` (S, M, L, F, XL or an id given with --gguf) for `engine`.

    With `cfg.dry_run` nothing is copied or downloaded. `opener`, `hf_base` and `entries`
    exist for tests.
    """
    if engine not in ENGINE_NAMES:
        raise ModelError(f"unknown engine {engine!r}")
    entries = catalog() if entries is None else entries
    entry = entries.get(model_id)
    if entry is None:
        if engine != "llamacpp":
            raise ModelError(f"model {model_id} is only available for llama.cpp "
                             f"with --gguf {model_id}=<path>")
        path = cfg.gguf.get(model_id)
        if path is None:
            raise ModelNotFound(f"model {model_id} is not in the catalog; "
                                f"pass --gguf {model_id}=<path> (llama.cpp only)")
        path = Path(path)
        if not path.is_file():
            raise ModelError(f"--gguf file for {model_id} not found: {path}")
        return _gguf_ref(model_id, None, path, None, path.stat().st_size)
    if engine == "ollama":
        return _resolve_ollama(model_id, entry, cfg, say)
    return _resolve_llamacpp(model_id, entry, cfg, opener, hf_base, say)
