"""Privacy: remove local names, paths and addresses from data, and find what is left.

scrub() cleans objects before they are written, find_leaks() is the last check
before anything is published (bench report aborts when it finds something).
"""

from __future__ import annotations

import getpass
import os
import re
import socket
from pathlib import Path

HOME = "<home>"
REDACTED = "<redacted>"
IP = "<ip>"
MAC = "<mac>"

# Profile folders that are shared by all users and not personal.
_SHARED = {"public", "default", "all users", "shared", "default user"}
_NAME = r"[^\\/\s\"'<>|:*?,;()\[\]{}]+"
_USER_PATH = re.compile(
    r"(?:[A-Za-z]:[\\/]+Users[\\/]+|(?<![\w.])/home/|(?<![\w.])/Users/)(?P<name>" + _NAME + ")")
_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
_IPV4 = re.compile(r"(?<![\d.])" + _OCTET + r"(?:\." + _OCTET + r"){3}(?![\d]|\.\d)")
_IPV6 = re.compile(
    r"(?<![\w:])(?:[0-9A-Fa-f]{1,4}:){1,7}:(?:[0-9A-Fa-f]{1,4}(?::[0-9A-Fa-f]{1,4}){0,6})?(?![\w:])"
    r"|(?<![\w:])::(?:[0-9A-Fa-f]{1,4}:){0,6}[0-9A-Fa-f]{1,4}(?![\w:])")
_MAC = re.compile(r"(?<![0-9A-Fa-f:-])(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}(?![0-9A-Fa-f:-])")
# Names that are not worth redacting everywhere (too generic).
_GENERIC_USERS = {"root"}


def _is_shared(name: str) -> bool:
    return name.lower() in _SHARED or name.startswith("<")


def _secret_list(secrets) -> list[str]:
    seen: dict[str, None] = {}
    for secret in secrets or []:
        if isinstance(secret, str) and secret.strip():
            seen[secret.strip()] = None
    return sorted(seen, key=len, reverse=True)


def _is_path(secret: str) -> bool:
    return "\\" in secret or "/" in secret


def _path_pattern(path: str) -> re.Pattern:
    """Match a home path with / or \\ (also doubled, as in JSON) between the parts."""
    parts = [re.escape(part) for part in re.split(r"[\\/]+", path.strip()) if part]
    lead = r"[\\/]+" if path.strip().startswith(("/", "\\")) else ""
    return re.compile(lead + r"[\\/]+".join(parts), re.IGNORECASE)


def _word_pattern(word: str) -> re.Pattern:
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(word) + r"(?![A-Za-z0-9])", re.IGNORECASE)


def _has_digit_ipv6(match: re.Match) -> bool:
    return any(ch.isdigit() for ch in match.group(0))


def _scrub_text(text: str, paths: list[re.Pattern], words: list[re.Pattern]) -> str:
    for pattern in paths:
        text = pattern.sub(HOME, text)
    text = _USER_PATH.sub(lambda m: m.group(0) if _is_shared(m.group("name")) else HOME, text)
    for pattern in words:
        text = pattern.sub(REDACTED, text)
    text = _MAC.sub(MAC, text)
    text = _IPV4.sub(IP, text)
    text = _IPV6.sub(lambda m: IP if _has_digit_ipv6(m) else m.group(0), text)
    return text


def scrub(obj, secrets: list[str]):
    """Return a copy of obj with home paths, secrets, IP and MAC addresses replaced.

    Strings inside dicts, lists and tuples (and dict keys) are cleaned; other
    values are returned unchanged. Home paths become <home>, secrets <redacted>.
    """
    items = _secret_list(secrets)
    paths = [_path_pattern(s) for s in items if _is_path(s)]
    words = [_word_pattern(s) for s in items if not _is_path(s)]

    def walk(value):
        if isinstance(value, str):
            return _scrub_text(value, paths, words)
        if isinstance(value, dict):
            return {walk(k) if isinstance(k, str) else k: walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v) for v in value]
        if isinstance(value, tuple):
            return tuple(walk(v) for v in value)
        return value

    return walk(obj)


def find_leaks(text: str, secrets: list[str]) -> list[str]:
    """List everything in text that looks personal; an empty list means clean."""
    leaks: dict[str, str] = {}

    def report(kind: str, value: str) -> None:
        leaks.setdefault(f"{kind}: {value}".lower(), f"{kind}: {value}")

    for secret in _secret_list(secrets):
        pattern = _path_pattern(secret) if _is_path(secret) else _word_pattern(secret)
        for match in pattern.finditer(text):
            report("secret", match.group(0))
    for match in _USER_PATH.finditer(text):
        if not _is_shared(match.group("name")):
            report("user path", match.group(0))
    for match in _IPV4.finditer(text):
        report("ip", match.group(0))
    for match in _IPV6.finditer(text):
        if _has_digit_ipv6(match):
            report("ip", match.group(0))
    for match in _MAC.finditer(text):
        report("mac", match.group(0))
    return list(leaks.values())


def local_secrets() -> list[str]:
    """Host name, user name and home path (both slash spellings) of this machine."""
    found: dict[str, None] = {}

    def add(value) -> None:
        if isinstance(value, str) and value.strip() and value.strip().lower() not in _GENERIC_USERS:
            found[value.strip()] = None

    try:
        host = socket.gethostname()
        add(host)
        add(host.split(".")[0])
    except Exception:
        pass
    try:
        add(getpass.getuser())
    except Exception:
        pass
    for variable in ("USER", "USERNAME", "LOGNAME"):
        add(os.environ.get(variable))
    try:
        home = str(Path.home())
        if home.strip("/\\"):
            add(home)
            add(home.replace("\\", "/"))
            add(home.replace("/", "\\"))
    except Exception:
        pass
    return list(found)
