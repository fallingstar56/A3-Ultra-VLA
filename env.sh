#!/usr/bin/env bash

# Shared environment for A3 whole-body edge inference and deployment tools.
# Source this file; do not execute it directly.

set -euo pipefail

EDGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT="${A3_MODEL_PATH:-${EDGE_ROOT}/models/checkpoint-50000}"
TORCH_SITE="${EDGE_ROOT}/torch_venv/lib/python3.12/site-packages"
TRT_SITE="${EDGE_ROOT}/trt_venv/lib/python3.12/site-packages"
EXPLICIT_A3_TASK="${A3_TASK:-}"

if [[ -f "${CHECKPOINT}/deploy.env" ]]; then
  # shellcheck disable=SC1090
  source "${CHECKPOINT}/deploy.env"
fi

# An operator's task choice must take precedence over checkpoint defaults.
# In particular, grasp-stop requires the exact task text "抓瓶子".
if [[ -n "${EXPLICIT_A3_TASK}" ]]; then
  export A3_TASK="${EXPLICIT_A3_TASK}"
fi

export A3_EDGE_ROOT="${EDGE_ROOT}"
export A3_MODEL_PATH="${CHECKPOINT}"
export A3_ENGINE_DIR="${A3_ENGINE_DIR:-${CHECKPOINT}/engines}"
export A3_RECORD_CHUNKS_DIR="${A3_RECORD_CHUNKS_DIR:-${CHECKPOINT}/chunks}"

export A3_PYTHON_BIN="${A3_PYTHON_BIN:-${EDGE_ROOT}/torch_venv/bin/python}"
export A3_TORCH_SITE="${A3_TORCH_SITE:-${TORCH_SITE}}"
export A3_TRT_SITE="${A3_TRT_SITE:-${TRT_SITE}}"
export A3_WORKER_PYTHON="${A3_WORKER_PYTHON:-/usr/bin/python3}"

export A3_GR00T_ROOT="${A3_GR00T_ROOT:-${EDGE_ROOT}/gr00t_code}"
export A3_GR00T_DIR="${A3_GR00T_DIR:-${A3_GR00T_ROOT}}"
export A3_DEPLOYMENT_SCRIPTS_DIR="${A3_DEPLOYMENT_SCRIPTS_DIR:-${A3_GR00T_ROOT}/scripts/deployment}"
export A3_ROBOINTERFACE_DIR="${A3_ROBOINTERFACE_DIR:-${EDGE_ROOT}/RoboInterface}"
export A3_ROS_SERVER_DIR="${A3_ROS_SERVER_DIR:-${EDGE_ROOT}/ros_server}"

export A3_ROBOT_TRANSPORT="${A3_ROBOT_TRANSPORT:-subproc}"
export A3_ROBOT_IP="${A3_ROBOT_IP:-192.168.100.100}"
export A3_TRT_MODE="${A3_TRT_MODE:-vit_llm_only}"
export A3_SDPA_BACKEND="${A3_SDPA_BACKEND:-cudnn}"
export A3_DIT_CUDA_GRAPH="${A3_DIT_CUDA_GRAPH:-1}"
export A3_EXEC_STEPS="${A3_EXEC_STEPS:-3}"
export A3_MODE="${A3_MODE:-rtc_chunk}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
