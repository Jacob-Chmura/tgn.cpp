#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
BUILD_DIR="$PROJECT_ROOT/build"

BUILD_TYPE="Release"
BUILD_EXAMPLES="OFF"

for arg in "$@"; do
  case $arg in
    --examples)
      BUILD_EXAMPLES="ON"
      shift
      ;;
    Debug|Release|RelWithDebInfo|MinSizeRel)
      BUILD_TYPE="$arg"
      shift
      ;;
    *)
      ;;
  esac
done

echo "Configuring CMake project"
echo "  Build Type: $BUILD_TYPE"
echo "  Build Examples: $BUILD_EXAMPLES"
echo "  Build Directory: $BUILD_DIR"

mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"

cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
      -DTGN_BUILD_EXAMPLES="${BUILD_EXAMPLES}" \
      ..

cmake --build . --parallel "$(nproc)"
