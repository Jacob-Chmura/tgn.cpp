#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
BUILD_MODE="${1:-build}"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not installed. Install it first: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

cd "$PROJECT_ROOT"

if [ "$BUILD_MODE" == "serve" ]; then
    uv run \
        --with mkdocs-material \
        --with mkdocstrings[python] \
        --with mkdoxy \
        --with-editable "$PROJECT_ROOT/python" \
        mkdocs serve
else
    uv run \
        --with mkdocs-material \
        --with mkdocstrings[python] \
        --with mkdoxy \
        --with-editable "$PROJECT_ROOT/python" \
        mkdocs build
fi
