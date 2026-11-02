"""Command line interface: bench system|doctor|plan|setup|run|report."""

from __future__ import annotations

import sys

from bench.config import load_config


def main(argv: list[str] | None = None) -> int:
    command, _config = load_config(sys.argv[1:] if argv is None else argv)
    print(f"bench {command}: not implemented yet")
    return 0
