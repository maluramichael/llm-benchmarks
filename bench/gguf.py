"""GGUF metadata reader, KV cache size per token and memory curve.

read_metadata() parses only the header and the key/value section of a GGUF file (v2 or
v3, little endian). Tensor infos are not read (only their count is reported), so a file
truncated after the metadata is fine. The returned dict is flat and keyed by GGUF key,
plus these synthetic keys:

    gguf.version, gguf.tensor_count, gguf.kv_count   header fields
    gguf.skipped_arrays                              {key: length} of arrays not kept

Arrays kept: those with at most MAX_KEPT_ARRAY_ITEMS elements that are numbers, bools or
(at most MAX_KEPT_STRING_ITEMS) strings. That covers per-layer arrays such as
head_count_kv or recurrent_layers, rope sections and tags. Larger arrays (tokenizer
tokens, merges, scores, token types, ~150k entries) are skipped without being decoded
into Python objects and only listed in gguf.skipped_arrays.

kv_bytes_per_token() models the cache that grows with the context: key and value vectors
of every layer that has full attention. Layers without a KV cache (linear attention,
gated delta net, SSM) keep a fixed-size state that does not grow with the context and is
not part of the result. Sliding-window caches are not modelled, so for such models the
result is an upper bound. A model whose attention layers cannot be determined raises
ValueError naming the missing keys instead of guessing.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import BinaryIO

MAGIC = b"GGUF"
SUPPORTED_VERSIONS = (2, 3)
MAX_KEPT_ARRAY_ITEMS = 1024
MAX_KEPT_STRING_ITEMS = 64
MAX_STRING_BYTES = 64 * 1024 * 1024  # guard against a corrupt length field

# GGUF value types 0 to 12.
UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, UINT64, INT64, FLOAT64 = range(13)
SCALARS = {
    UINT8: struct.Struct("<B"), INT8: struct.Struct("<b"), UINT16: struct.Struct("<H"),
    INT16: struct.Struct("<h"), UINT32: struct.Struct("<I"), INT32: struct.Struct("<i"),
    FLOAT32: struct.Struct("<f"), BOOL: struct.Struct("<?"), UINT64: struct.Struct("<Q"),
    INT64: struct.Struct("<q"), FLOAT64: struct.Struct("<d"),
}


class _Reader:
    def __init__(self, handle: BinaryIO, size: int):
        self.handle = handle
        self.size = size

    def take(self, count: int) -> bytes:
        data = self.handle.read(count)
        if len(data) != count:
            raise ValueError("GGUF file truncated inside the metadata section")
        return data

    def unpack(self, fmt: struct.Struct):
        return fmt.unpack(self.take(fmt.size))[0]

    def skip(self, count: int) -> None:
        target = self.handle.tell() + count
        if target > self.size:
            raise ValueError("GGUF file truncated inside the metadata section")
        self.handle.seek(target)

    def string(self) -> str:
        length = self.unpack(SCALARS[UINT64])
        if length > MAX_STRING_BYTES:
            raise ValueError(f"GGUF string length {length} is implausible, file is corrupt")
        return self.take(length).decode("utf-8", errors="replace")

    def skip_string(self) -> None:
        length = self.unpack(SCALARS[UINT64])
        self.skip(length)

    def value(self, type_id: int):
        if type_id in SCALARS:
            return self.unpack(SCALARS[type_id])
        if type_id == STRING:
            return self.string()
        if type_id == ARRAY:
            element_type, count = self._array_header()
            return [self.value(element_type) for _ in range(count)]
        raise ValueError(f"unknown GGUF value type {type_id}")

    def _array_header(self) -> tuple[int, int]:
        element_type = self.unpack(SCALARS[UINT32])
        count = self.unpack(SCALARS[UINT64])
        if element_type not in SCALARS and element_type not in (STRING, ARRAY):
            raise ValueError(f"unknown GGUF array element type {element_type}")
        return element_type, count

    def skip_value(self, type_id: int) -> None:
        if type_id in SCALARS:
            self.skip(SCALARS[type_id].size)
        elif type_id == STRING:
            self.skip_string()
        elif type_id == ARRAY:
            element_type, count = self._array_header()
            self.skip_items(element_type, count)
        else:
            raise ValueError(f"unknown GGUF value type {type_id}")

    def skip_items(self, element_type: int, count: int) -> None:
        if element_type in SCALARS:
            self.skip(SCALARS[element_type].size * count)
        else:
            for _ in range(count):
                self.skip_value(element_type)

    def array(self, key: str, skipped: dict) -> list | None:
        """Read the array if small enough, else skip it and record its length."""
        element_type, count = self._array_header()
        limit = MAX_KEPT_STRING_ITEMS if element_type in (STRING, ARRAY) else MAX_KEPT_ARRAY_ITEMS
        if count > limit:
            self.skip_items(element_type, count)
            skipped[key] = count
            return None
        return [self.value(element_type) for _ in range(count)]


def read_metadata(path: Path) -> dict:
    """Read header and KV pairs of a GGUF file. Tensor infos and data are never read."""
    path = Path(path)
    with path.open("rb") as handle:
        reader = _Reader(handle, os.fstat(handle.fileno()).st_size)
        magic = handle.read(4)
        if magic != MAGIC:
            if len(magic) < 4 and not magic:
                raise ValueError(f"{path.name} is not a GGUF file (empty)")
            raise ValueError(f"{path.name} is not a GGUF file (bad magic)")
        version = reader.unpack(SCALARS[UINT32])
        if version not in SUPPORTED_VERSIONS:
            raise ValueError(f"unsupported GGUF version {version} (supported: 2, 3)")
        tensor_count = reader.unpack(SCALARS[UINT64])
        kv_count = reader.unpack(SCALARS[UINT64])
        meta: dict = {"gguf.version": version, "gguf.tensor_count": tensor_count,
                      "gguf.kv_count": kv_count}
        skipped: dict[str, int] = {}
        for _ in range(kv_count):
            key = reader.string()
            type_id = reader.unpack(SCALARS[UINT32])
            if type_id == ARRAY:
                items = reader.array(key, skipped)
                if items is not None:
                    meta[key] = items
            else:
                meta[key] = reader.value(type_id)
        meta["gguf.skipped_arrays"] = skipped
    return meta


# --- KV cache model ----------------------------------------------------------------------

def _arch(meta: dict) -> str:
    arch = meta.get("general.architecture")
    if not isinstance(arch, str) or not arch:
        raise ValueError("cannot model the KV cache: missing key general.architecture")
    return arch


def _per_layer(value, n_layers: int, key: str) -> list:
    """Expand a scalar to one value per layer, or check that an array has one per layer."""
    if isinstance(value, list):
        if len(value) != n_layers:
            raise ValueError(f"{key} has {len(value)} entries but the model has {n_layers} layers")
        return value
    return [value] * n_layers


def _check_keys(meta: dict, arch: str) -> None:
    """Raise a ValueError naming every key that is needed but absent."""
    prefix = f"{arch}.attention"
    missing = [] if f"{arch}.block_count" in meta else [f"{arch}.block_count"]
    if f"{prefix}.kv_lora_rank" in meta:  # MLA
        if f"{arch}.rope.dimension_count" not in meta:
            missing.append(f"{arch}.rope.dimension_count")
    else:
        if f"{prefix}.head_count_kv" not in meta and f"{prefix}.head_count" not in meta:
            missing.append(f"{prefix}.head_count_kv (or {prefix}.head_count)")
        if f"{prefix}.key_length" not in meta and not (
                f"{arch}.embedding_length" in meta and f"{prefix}.head_count" in meta):
            missing.append(f"{prefix}.key_length (or {arch}.embedding_length with "
                           f"{prefix}.head_count)")
    if missing:
        raise ValueError(f"cannot model the KV cache of '{arch}': missing keys "
                         + ", ".join(missing))


def _layer_plan(meta: dict) -> tuple[str, list[bool]]:
    """Return (arch, flags) where flags[i] says whether layer i keeps a growing KV cache.

    Markers, in order of precedence: a per-layer head_count_kv array (0 = no attention),
    attention.recurrent_layers (bool array), full_attention_interval (every N-th layer has
    full attention, layer i when (i + 1) % N == 0). Without any marker all layers attend,
    unless the model has SSM keys, which means a hybrid whose layers we cannot tell apart.
    Layers of multi-token-prediction heads (nextn_predict_layers) are not counted: they
    run in a separate context.
    """
    arch = _arch(meta)
    _check_keys(meta, arch)
    n_total = meta[f"{arch}.block_count"]
    n_layers = n_total - meta.get(f"{arch}.nextn_predict_layers", 0)
    kv_key = f"{arch}.attention.head_count_kv"
    recurrent_key = f"{arch}.attention.recurrent_layers"
    interval_key = f"{arch}.full_attention_interval"

    kv_heads = meta.get(kv_key)
    if isinstance(kv_heads, list):
        flags = [heads > 0 for heads in _per_layer(kv_heads, n_total, kv_key)[:n_layers]]
    elif recurrent_key in meta:
        recurrent = _per_layer(meta[recurrent_key], n_total, recurrent_key)[:n_layers]
        flags = [not bool(r) for r in recurrent]
    elif interval_key in meta:
        interval = meta[interval_key]
        if not isinstance(interval, int) or interval < 1:
            raise ValueError(f"{interval_key} must be a positive integer, got {interval!r}")
        flags = [(i + 1) % interval == 0 for i in range(n_layers)]
    elif any(key.startswith(f"{arch}.ssm.") for key in meta):
        raise ValueError(
            f"cannot model the KV cache of '{arch}': it has SSM layers but no marker for which "
            f"layers use attention; missing keys {recurrent_key} or {interval_key} "
            f"(or a per-layer {kv_key} array)")
    else:
        flags = [True] * n_layers
    return arch, flags


def attention_layer_count(meta: dict) -> int:
    """Number of layers that keep a KV cache growing with the context."""
    return sum(_layer_plan(meta)[1])


def kv_bytes_per_token(meta: dict, cache_type_bytes: float = 2.0) -> int:
    """Bytes of KV cache per context token (f16 cache: 2.0 bytes per element).

    Dense/GQA layers store key_length + value_length elements per KV head. MLA models
    (attention.kv_lora_rank present) store one latent vector of kv_lora_rank plus the
    rope part (rope.dimension_count) per layer.
    """
    arch, flags = _layer_plan(meta)
    prefix = f"{arch}.attention"
    if f"{prefix}.kv_lora_rank" in meta:
        per_layer = meta[f"{prefix}.kv_lora_rank"] + meta[f"{arch}.rope.dimension_count"]
        return round(sum(per_layer for attends in flags if attends) * cache_type_bytes)

    n_total = meta[f"{arch}.block_count"]
    kv_key = f"{prefix}.head_count_kv" if f"{prefix}.head_count_kv" in meta else f"{prefix}.head_count"
    kv_heads = _per_layer(meta[kv_key], n_total, kv_key)[:len(flags)]
    if f"{prefix}.key_length" in meta:
        key_length = meta[f"{prefix}.key_length"]
    else:
        heads = meta[f"{prefix}.head_count"]
        if isinstance(heads, list):
            raise ValueError(f"cannot derive {prefix}.key_length from a per-layer head_count; "
                             f"missing key {prefix}.key_length")
        key_length = meta[f"{arch}.embedding_length"] // heads
    value_length = meta.get(f"{prefix}.value_length", key_length)

    elements = sum(n_heads * (key_length + value_length)
                   for n_heads, attends in zip(kv_heads, flags) if attends)
    return round(elements * cache_type_bytes)


def memory_curve(meta: dict, weights_bytes: int, ctx_steps: list[int],
                 cache_type_bytes: float = 2.0) -> list[tuple[int, int]]:
    """Expected model memory (weights plus KV cache) in bytes for each context length."""
    per_token = kv_bytes_per_token(meta, cache_type_bytes)
    return [(ctx, weights_bytes + ctx * per_token) for ctx in ctx_steps]
