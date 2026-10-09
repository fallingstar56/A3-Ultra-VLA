#!/usr/bin/env bash

# Run the complete checkpoint -> ONNX -> TensorRT conversion on the target
# Thor. Generated files remain under work/ and never overwrite validated
# checkpoint artifacts.

set -euo pipefail

EDGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${EDGE_ROOT}/env.sh"

EXPORT_ROOT="${A3_EXPORT_OUTPUT_DIR:-${EDGE_ROOT}/work/export}"
BUILD_ROOT="${A3_BUILD_OUTPUT_DIR:-${EDGE_ROOT}/work/build}"

echo "[convert] checkpoint: ${A3_MODEL_PATH}"
echo "[convert] ONNX output: ${EXPORT_ROOT}/onnx"
echo "[convert] engine output: ${BUILD_ROOT}/engines"

A3_EXPORT_OUTPUT_DIR="$EXPORT_ROOT" \
  bash "${EDGE_ROOT}/export_onnx.sh" "$@"

A3_EXPORT_OUTPUT_DIR="$EXPORT_ROOT" \
A3_ONNX_SOURCE_DIR="${EXPORT_ROOT}/onnx" \
A3_BUILD_OUTPUT_DIR="$BUILD_ROOT" \
  bash "${EDGE_ROOT}/build_engines.sh" "$@"

OUTPUT_ENV="${EDGE_ROOT}/work/conversion.env"
mkdir -p -- "$(dirname -- "$OUTPUT_ENV")"
{
  printf 'export A3_MODEL_PATH=%q\n' "$A3_MODEL_PATH"
  printf 'export A3_ONNX_SOURCE_DIR=%q\n' "${EXPORT_ROOT}/onnx"
  printf 'export A3_ENGINE_DIR=%q\n' "${BUILD_ROOT}/engines"
} > "$OUTPUT_ENV"

echo "[convert] complete"
echo "[convert] runtime engines are in: ${BUILD_ROOT}/engines"
echo "[convert] before check/run: source ${OUTPUT_ENV}"

