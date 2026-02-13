#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
BUILD_DIR="$PROJECT_ROOT/build"
BUILD_TYPE="Debug"

echo "Configuring CMake project (Build type: $BUILD_TYPE) in $BUILD_DIR"
mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"

cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DTGN_BUILD_TESTS=ON \
      -DCMAKE_PREFIX_PATH=/home/kuba/repo/tgn.cpp/libtorch/ \
      -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
      ..

cmake --build . --parallel "$(nproc)"

ctest --output-on-failure
