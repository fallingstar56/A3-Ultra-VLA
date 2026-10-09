#!/usr/bin/env bash

set -euo pipefail

EDGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${EDGE_ROOT}/env.sh"

# Keep generated files separate from the validated baseline artifacts.
OUTPUT_ROOT="${A3_EXPORT_OUTPUT_DIR:-${EDGE_ROOT}/work/export}"
EMBODIMENT_TAG="${A3_EMBODIMENT_TAG:-${EMBODIMENT:-NEW_EMBODIMENT}}"
mkdir -p -- "${OUTPUT_ROOT}"

echo "[export] checkpoint: ${A3_MODEL_PATH}"
echo "[export] embodiment: ${EMBODIMENT_TAG}"
echo "[export] ONNX output: ${OUTPUT_ROOT}/onnx"

exec bash "${A3_GR00T_ROOT}/scripts/deployment/export_a3_onnx_on_adu.sh" \
  --model-path "${A3_MODEL_PATH}" \
  --dataset-path dummy \
  --embodiment-tag "${EMBODIMENT_TAG}" \
  --output-dir "${OUTPUT_ROOT}" \
  --export-mode full_pipeline \
  --precision bf16 \
  --steps export \
  "$@"
