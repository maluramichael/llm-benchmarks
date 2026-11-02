"""Result storage: idempotent, crash-safe, one JSON file per run.

Layout of the file::

    {"schema_version": 1, "tests": {<key>: {status, values, started_at,
     finished_at, meta}}, "system": ..., "engines": ..., "notes": ...}

Durability: every change rewrites the whole file through a temporary file
(`<file>.tmp`), which is flushed and fsynced before `os.replace` swaps it in.
A crash (power loss, blue screen) therefore leaves either the old or the new
complete file, never a half written one.

Failed writes: when the write itself fails (I/O error), the change stays in
memory and the exception is raised. Nothing is lost, because the next
successful write persists all pending changes. Values that cannot be
serialized (objects, NaN) are rejected before anything is changed, so one bad
value can never make every later write fail.
"""

from __future__ import annotations

import contextlib
import fnmatch
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

SCHEMA_VERSION = 1
FINAL = {"ok", "does_not_fit", "skipped"}  # never measured again without force
RETRY = {"error", "aborted"}  # measured again on the next run
RUNNING = "running"  # in progress, or left behind by a crash
RESERVED_SECTIONS = {"schema_version", "tests"}
# Windows virus scanners and indexers can hold a file for a moment.
LOCK_ATTEMPTS = 10
LOCK_BACKOFF_START = 0.05
LOCK_BACKOFF_MAX = 0.5
WINDOWS_LOCK_ERRORS = {5, 32}  # access denied, sharing violation


class StoreError(RuntimeError):
    """The result file exists but cannot be used. A human has to decide."""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _dumps(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False,
                      allow_nan=False) + "\n"


def _is_transient_lock(err: OSError) -> bool:
    return isinstance(err, PermissionError) or getattr(err, "winerror", None) in WINDOWS_LOCK_ERRORS


def _retry_on_lock(action):
    """Run `action`, retrying a few times on transient file lock errors."""
    delay = LOCK_BACKOFF_START
    for attempt in range(1, LOCK_ATTEMPTS + 1):
        try:
            return action()
        except OSError as err:
            if attempt == LOCK_ATTEMPTS or not _is_transient_lock(err):
                raise
            time.sleep(delay)
            delay = min(delay * 2, LOCK_BACKOFF_MAX)


def _normalize(value: Any, what: str) -> Any:
    """Return a deep copy shaped exactly like it will be after reload from disk.

    Raises ValueError for anything JSON cannot hold (objects, NaN, Infinity).
    """
    try:
        return json.loads(_dumps(value))
    except (TypeError, ValueError) as err:
        raise ValueError(f"{what} is not JSON serializable: {err}") from None


class ResultStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.data: dict = self._load()

    # Reading

    def _load(self) -> dict:
        try:
            raw = self.path.read_text(encoding="utf-8-sig")
        except FileNotFoundError:
            return {"schema_version": SCHEMA_VERSION, "tests": {}}
        except (OSError, UnicodeDecodeError) as err:
            raise StoreError(f"cannot read result file {self.path}: {err}") from None
        try:
            data = json.loads(raw)
        except ValueError as err:
            raise StoreError(
                f"result file {self.path} is not valid JSON ({err}); "
                "it was left untouched, fix or remove it by hand") from None
        if not isinstance(data, dict):
            raise StoreError(f"result file {self.path} does not contain a JSON object")
        if data.get("schema_version") != SCHEMA_VERSION:
            raise StoreError(
                f"result file {self.path} has schema_version "
                f"{data.get('schema_version')!r}, expected {SCHEMA_VERSION}")
        data.setdefault("tests", {})
        if not isinstance(data["tests"], dict):
            raise StoreError(f"result file {self.path}: 'tests' is not a JSON object")
        return data

    def get(self, key: str) -> dict | None:
        return self.data["tests"].get(key)

    def should_run(self, key: str, force: bool = False,
                   force_patterns: Sequence[str] = ()) -> bool:
        """True unless the test already has a final result.

        `force` or a matching `force_patterns` entry (fnmatch, case sensitive)
        always runs. Missing, `error`, `aborted` and stale `running` entries run.
        """
        if force:
            return True
        if any(fnmatch.fnmatchcase(key, pattern) for pattern in force_patterns):
            return True
        entry = self.get(key)
        return entry is None or entry["status"] not in FINAL

    # Writing

    def mark_running(self, key: str, meta: dict) -> None:
        """Record that a test starts. Replaces any earlier entry for the key."""
        meta = _normalize(meta, "meta")
        self.data["tests"][key] = {
            "status": RUNNING,
            "values": {},
            "started_at": _now(),
            "finished_at": None,
            "meta": meta,
        }
        self._persist()

    def save(self, key: str, status: str, values: dict, meta: dict | None = None) -> None:
        """Record the result of a test. `meta` is merged into the one from mark_running."""
        if status not in FINAL and status not in RETRY:
            raise ValueError(
                f"invalid status {status!r} (allowed: {', '.join(sorted(FINAL | RETRY))})")
        values = _normalize(values, "values")
        meta = _normalize(meta, "meta") if meta is not None else {}
        previous = self.get(key)
        now = _now()
        if previous is not None and previous["status"] == RUNNING:
            started_at = previous["started_at"]
            meta = {**previous.get("meta", {}), **meta}
        else:
            started_at = now
        self.data["tests"][key] = {
            "status": status,
            "values": values,
            "started_at": started_at,
            "finished_at": now,
            "meta": meta,
        }
        self._persist()

    def recover_crashed(self) -> list[str]:
        """Turn leftover `running` entries into `aborted`. Returns their keys, sorted."""
        crashed = sorted(k for k, v in self.data["tests"].items() if v["status"] == RUNNING)
        for key in crashed:
            self.data["tests"][key]["status"] = "aborted"
        if crashed:
            self._persist()
        return crashed

    def set_section(self, name: str, value: Any) -> None:
        """Set a top level section such as `system`, `engines` or `notes`."""
        if name in RESERVED_SECTIONS:
            raise ValueError(f"section name {name!r} is reserved")
        self.data[name] = _normalize(value, f"section {name!r}")
        self._persist()

    def _persist(self) -> None:
        """Atomically replace the file: write `.tmp`, flush, fsync, os.replace.

        Transient lock errors (PermissionError, Windows sharing violations) are
        retried with backoff; the last error is raised after LOCK_ATTEMPTS tries.
        """
        text = _dumps(self.data)
        tmp = self.path.with_name(self.path.name + ".tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)

        def write_tmp() -> None:
            with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())

        try:
            _retry_on_lock(write_tmp)
            _retry_on_lock(lambda: os.replace(tmp, self.path))
        except BaseException:
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise
        self._sync_directory()

    def _sync_directory(self) -> None:
        """Make the rename itself durable. Best effort, not possible on Windows."""
        if os.name == "nt":
            return
        with contextlib.suppress(OSError):
            fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
