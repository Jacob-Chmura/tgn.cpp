#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
BUILD_MODE="${1:-build}"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not installed. Install it first: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

if ! command -v doxygen >/dev/null 2>&1; then
    echo "doxygen is not installed. Install it first: https://www.doxygen.nl/"
    exit 1
fi

if ! command -v doxybook2 >/dev/null 2>&1; then
    echo "doxybook2 is not installed. Install it first: https://github.com/matusnovak/doxybook2/releases/tag/v1.5.0"
    exit 1
fi

cd "$PROJECT_ROOT"
mkdir -p build/docs
doxygen Doxyfile

mkdir -p docs/api/cpp
doxybook2 --input build/docs/xml \
          --output docs/api/cpp

if [ "$BUILD_MODE" == "serve" ]; then
    uv run \
        --with mkdocs-material \
        --with mkdocstrings[python] \
        --with-editable "$PROJECT_ROOT/python" \
        mkdocs serve
else
    uv run \
        --with mkdocs-material \
        --with mkdocstrings[python] \
        --with-editable "$PROJECT_ROOT/python" \
        mkdocs build
fi
