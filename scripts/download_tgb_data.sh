#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <dataset_name>"
    echo "Example: $0 tgbl-wiki"
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not installed. Install it first: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

DATASET_NAME="$1"
OUTPUT_PATH="$PROJECT_ROOT/data/$DATASET_NAME.tguf"

echo "Streaming $DATASET_NAME directly to $OUTPUT_PATH..."
uv run --no-project \
    --with py-tgb \
    --with numpy \
    --with tqdm \
    --with pandas \
    python "$PROJECT_ROOT/tools/download_tgb_to_tguf.py" \
    --name "$DATASET_NAME" \
    --output "$OUTPUT_PATH"
