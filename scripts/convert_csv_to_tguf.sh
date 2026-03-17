#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <edges_csv> <output_tguf> [labels_csv] [static_node_feats_csv]" >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not installed. Install it first: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

EDGES="$1"
OUTPUT="$2"
LABELS="${3:-}"
NODE_FEATS="${4:-}"

PY_ARGS=(--edges "$EDGES" --output "$OUTPUT")
[[ -n "$LABELS" ]] && PY_ARGS+=(--labels "$LABELS")
[[ -n "$NODE_FEATS" ]] && PY_ARGS+=(--node-feats "$NODE_FEATS")

uv run --no-project \
    --with numpy \
    --with tqdm \
    --with pandas==2.2.3 \
    --with-editable "$PROJECT_ROOT/python" \
    python "$PROJECT_ROOT/tools/convert_csv_to_tguf.py" "${PY_ARGS[@]}"
