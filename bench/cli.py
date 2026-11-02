"""Command line interface: bench system|doctor|plan|setup|run|report."""

from __future__ import annotations

import json
import sys

from bench.config import load_config
from bench.privacy import local_secrets, scrub
from bench.system import collect_system, suggest_device_name


def _system_command() -> int:
    info = collect_system()
    output = {"suggested_device_name": suggest_device_name(info), "system": info}
    print(json.dumps(scrub(output, local_secrets()), indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    command, _config = load_config(sys.argv[1:] if argv is None else argv)
    if command == "system":
        return _system_command()
    print(f"bench {command}: not implemented yet")
    return 0
