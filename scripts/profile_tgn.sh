#!/usr/bin/env bash
set -euo pipefail

BINARY="$1"
DATASET="$2"
BIN_NAME=$(basename "$BINARY")
DS_NAME=$(basename "$DATASET" .tguf)
OUT_DIR="perf_results"
THIRD_PARTY="third_party/FlameGraph"

if [[ ! -x "$BINARY" ]]; then
    echo "Error: Binary $BINARY not found."
    exit 1
fi

if [[ ! -d "$THIRD_PARTY" ]]; then
    echo "Cloning FlameGraph tools to third_party/..."
    git clone --depth 1 https://github.com/brendangregg/FlameGraph "$THIRD_PARTY"
fi

mkdir -p "$OUT_DIR"
echo "PROFILING: $BIN_NAME on $DS_NAME (1 Epoch)"

sudo perf record -g --call-graph dwarf -F 999 -o "$OUT_DIR/perf.data" \
    -- "$BINARY" "$DATASET" --epochs 1

echo "Generating FlameGraph..."
SVG_OUT="$OUT_DIR/${DS_NAME}_${BIN_NAME}.svg"

sudo perf script -i "$OUT_DIR/perf.data" | \
    "$THIRD_PARTY/stackcollapse-perf.pl" | \
    "$THIRD_PARTY/flamegraph.pl" --title="TGN: $BIN_NAME ($DS_NAME)" > "$SVG_OUT"

echo "Done. Check: $SVG_OUT"
