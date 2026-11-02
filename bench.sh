#!/bin/sh
# Launcher for macOS and Linux: finds uv or fetches it into ./.bench/uv, then runs bench.py.
# No admin rights, no PATH or shell profile changes.
set -eu

cd "$(dirname "$0")"

if command -v uv >/dev/null 2>&1; then
    UV=uv
elif [ -x ./.bench/uv/uv ]; then
    UV=./.bench/uv/uv
else
    echo "uv not found, installing it into ./.bench/uv (official installer from astral.sh)" >&2
    mkdir -p ./.bench/uv
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$PWD/.bench/uv" UV_NO_MODIFY_PATH=1 sh
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$PWD/.bench/uv" UV_NO_MODIFY_PATH=1 sh
    else
        echo "Neither curl nor wget found. Install uv manually: https://docs.astral.sh/uv/" >&2
        exit 1
    fi
    UV=./.bench/uv/uv
fi

exec "$UV" run --no-project --python 3.12 bench.py "$@"
