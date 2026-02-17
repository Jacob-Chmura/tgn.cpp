#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
BUILD_DIR="$PROJECT_ROOT/build"
BUILD_TYPE="Debug"

if [[ $# -ge 1 ]]; then
  BUILD_TYPE="$1"
fi

case "$BUILD_TYPE" in
  Debug|Release|RelWithDebInfo|MinSizeRel) ;;
  *)
    echo "Invalid build type: $BUILD_TYPE, try 'Debug|Release|RelWithDebInfo|MinSizeRel'"
    exit 1
    ;;
esac

echo "Configuring CMake project (Build type: $BUILD_TYPE) in $BUILD_DIR"
mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"

cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
      ..

cmake --build . --parallel "$(nproc)"
