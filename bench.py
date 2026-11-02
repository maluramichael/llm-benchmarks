"""Entry point: python bench.py <command> [options]. Prefer ./bench.sh or bench.ps1."""

import sys

from bench.cli import main

if __name__ == "__main__":
    sys.exit(main())
