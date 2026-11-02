"""Test helper: writes minimal GGUF v3 files (header and KV pairs, optional raw tail).

Python values map to GGUF types as follows: bool -> BOOL, int -> UINT32, float -> FLOAT32,
str -> STRING, list -> ARRAY of the type of its first element. Wrap a value in Typed() to
force another scalar type, or in Array(element_type, items) to force an array element type
(needed for empty and nested arrays).
"""

import struct
from pathlib import Path

UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, UINT64, INT64, FLOAT64 = range(13)

SCALAR_FORMATS = {UINT8: "<B", INT8: "<b", UINT16: "<H", INT16: "<h", UINT32: "<I", INT32: "<i",
                  FLOAT32: "<f", BOOL: "<?", UINT64: "<Q", INT64: "<q", FLOAT64: "<d"}


class Typed:
    def __init__(self, type_id: int, value):
        self.type_id = type_id
        self.value = value


class Array:
    def __init__(self, element_type: int, items: list):
        self.element_type = element_type
        self.items = items


def _string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _infer(value) -> int:
    if isinstance(value, bool):
        return BOOL
    if isinstance(value, int):
        return UINT32
    if isinstance(value, float):
        return FLOAT32
    if isinstance(value, str):
        return STRING
    if isinstance(value, (list, Array)):
        return ARRAY
    raise TypeError(f"unsupported value {value!r}")


def _payload(type_id: int, value) -> bytes:
    if type_id in SCALAR_FORMATS:
        return struct.pack(SCALAR_FORMATS[type_id], value)
    if type_id == STRING:
        return _string(value)
    if type_id == ARRAY:
        if isinstance(value, Array):
            element_type, items = value.element_type, value.items
        else:
            items = value
            element_type = _infer(items[0]) if items else UINT32
        body = b"".join(_payload(element_type, item) for item in items)
        return struct.pack("<IQ", element_type, len(items)) + body
    raise ValueError(f"bad type {type_id}")


def write_gguf(path: Path, kv: dict, tensor_count: int = 0, version: int = 3,
               tail: bytes = b"") -> Path:
    """Write header, KV pairs (in dict order) and an optional raw tail (stand-in for tensors)."""
    out = bytearray(b"GGUF" + struct.pack("<IQQ", version, tensor_count, len(kv)))
    for key, value in kv.items():
        if isinstance(value, Typed):
            type_id, raw = value.type_id, value.value
        else:
            type_id, raw = _infer(value), value
        out += _string(key) + struct.pack("<I", type_id) + _payload(type_id, raw)
    Path(path).write_bytes(bytes(out) + tail)
    return Path(path)
