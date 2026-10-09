#!/usr/bin/env bash
# A3-ADU whole-body local-engine RTC launcher.
#
# It intentionally follows the protocol in:
#   gr00t/dev: gr00t/eval/real_robot/A3/infer_a3_rtc_zmq.py
#   robotinterface/master_wholebody_human
#
# Usage:
#   bash edge_infer/run_a3_adu_wholebody_rtc.sh
#   bash edge_infer/run_a3_adu_wholebody_rtc.sh --check
#   A3_TASK='exact training prompt' bash edge_infer/run_a3_adu_wholebody_rtc.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

if [[ -n "${A3_PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$A3_PYTHON_BIN"
elif [[ -x /agibot/torch_build/venv/bin/python ]]; then
  PYTHON_BIN=/agibot/torch_build/venv/bin/python
else
  PYTHON_BIN=/agibot/edge_deploy/torch_venv/bin/python
fi
MODEL_PATH="${A3_MODEL_PATH:-/agibot/models/a3_60000/ckpt}"
ENGINE_DIR="${A3_ENGINE_DIR:-/agibot/models/a3_60000/engines}"
RECORD_DIR="${A3_RECORD_CHUNKS_DIR:-/agibot/models/a3_60000/chunks}"
TRANSPORT="${A3_ROBOT_TRANSPORT:-subproc}"
if [[ -n "${A3_ROBOT_IP:-}" ]]; then
  ROBOT_IP="$A3_ROBOT_IP"
elif [[ "$TRANSPORT" == "http" ]]; then
  ROBOT_IP=127.0.0.1
else
  ROBOT_IP=192.168.100.100
fi
ROBOT_PORT="${A3_ROBOT_PORT:-5050}"
WORKER_PYTHON="${A3_WORKER_PYTHON:-/usr/bin/python3}"
HAND_KIND="${A3_HAND_KIND:-hand}"
TASK="${A3_TASK:-A3U hands over the water bottle}"
MODE="${A3_MODE:-rtc_chunk}"
# Correctness-first default: keep the PyTorch action head so gr00t/dev's exact
# per-token train-time RTC implementation is used. Set n17_full_pipeline only
# after validating its compatibility guard on your exported engines.
TRT_MODE="${A3_TRT_MODE:-vit_llm_only}"
EXEC_STEPS="${A3_EXEC_STEPS:-15}"
SERVER_READY_TIMEOUT="${A3_SERVER_READY_TIMEOUT_SEC:-120}"
SONIC_READY_TIMEOUT="${A3_SONIC_READY_TIMEOUT_SEC:-60}"
CHECK_ONLY=0

case "$MODE" in
  standard|rtc_chunk) ;;
  *)
    echo "[a3-rtc] A3_MODE must be standard or rtc_chunk, got: $MODE" >&2
    exit 64
    ;;
esac

case "$TRANSPORT" in
  subproc|http) ;;
  *)
    echo "[a3-rtc] A3_ROBOT_TRANSPORT must be subproc or http, got: $TRANSPORT" >&2
    exit 64
    ;;
esac

if [[ ! "$EXEC_STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "[a3-rtc] A3_EXEC_STEPS must be a positive integer, got: $EXEC_STEPS" >&2
  exit 64
fi

if [[ "${1:-}" == "--check" ]]; then
  CHECK_ONLY=1
  shift
fi

first_dir_with_file() {
  local relative="$1"
  shift
  local candidate
  for candidate in "$@"; do
    if [[ -f "$candidate/$relative" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

ROBOINTERFACE_DIR="${A3_ROBOINTERFACE_DIR:-}"
if [[ -z "$ROBOINTERFACE_DIR" ]]; then
  ROBOINTERFACE_DIR="$(first_dir_with_file interface.py \
    "$DEPLOY_DIR/robotinterface/RoboInterface" \
    "$DEPLOY_DIR/RoboInterface" \
    /agibot/edge_deploy/RoboInterface \
    /agibot/robotinterface/RoboInterface \
    /agibot/RoboInterface || true)"
fi

GR00T_DIR="${A3_GR00T_DIR:-}"
if [[ -z "$GR00T_DIR" ]]; then
  GR00T_DIR="$(first_dir_with_file gr00t/policy/gr00t_policy.py \
    "$DEPLOY_DIR/gr00t_code" \
    "$DEPLOY_DIR" \
    /agibot/edge_deploy/gr00t_code \
    /agibot/gr00t || true)"
fi

ROS_SERVER_DIR="${A3_ROS_SERVER_DIR:-}"
if [[ -z "$ROS_SERVER_DIR" ]]; then
  ROS_SERVER_DIR="$(first_dir_with_file install/setup.bash \
    "$DEPLOY_DIR/ros_server" \
    /agibot/ros_server || true)"
fi

TRT_SITE="${A3_TRT_SITE:-}"
if [[ -z "$TRT_SITE" ]]; then
  PYTHON_VENV="$(cd -- "$(dirname -- "$PYTHON_BIN")/.." 2>/dev/null && pwd || true)"
  TRT_SITE="$(first_dir_with_file tensorrt/__init__.py \
    "$DEPLOY_DIR/trt_venv/lib/python3.12/site-packages" \
    /agibot/edge_deploy/trt_venv/lib/python3.12/site-packages \
    "$PYTHON_VENV/lib/python3.12/site-packages" \
    /agibot/torch_build/venv/lib/python3.12/site-packages || true)"
fi

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -e "$path" ]]; then
    echo "[a3-rtc] missing ${label}: $path" >&2
    exit 66
  fi
}

require_file "$PYTHON_BIN" "Python interpreter"
require_file "$MODEL_PATH" "checkpoint"
require_file "$ENGINE_DIR" "TensorRT engine directory"
require_file "$SCRIPT_DIR/infer_a3_edge.py" "inference client"
if [[ "$TRANSPORT" == "subproc" ]]; then
  require_file "$SCRIPT_DIR/a3_ros_worker.py" "shared-memory ROS worker"
  require_file "$WORKER_PYTHON" "ROS worker Python"
  require_file "$ROS_SERVER_DIR/install/setup.bash" "ros_server setup"
  require_file "$ROS_SERVER_DIR/_pb_gen" "ros_server generated protobufs"
  require_file \
    "$ROS_SERVER_DIR/install/a3_server/lib/python3.12/site-packages/a3_server/server_node.py" \
    "a3_server Python package"
fi
if [[ -z "$ROBOINTERFACE_DIR" ]]; then
  echo "[a3-rtc] cannot find robotinterface/RoboInterface; set A3_ROBOINTERFACE_DIR" >&2
  exit 66
fi
if [[ -z "$GR00T_DIR" ]]; then
  echo "[a3-rtc] cannot find the gr00t source checkout; set A3_GR00T_DIR" >&2
  exit 66
fi
if [[ -z "$TRT_SITE" ]]; then
  echo "[a3-rtc] cannot find TensorRT Python bindings; set A3_TRT_SITE" >&2
  exit 66
fi

INTERFACE_PY="$ROBOINTERFACE_DIR/interface.py"
for token in '"mode": "whole_body"' 'payload["source_fps"]' \
             'payload["emit_mode"]' 'def set_state_source' \
             'def get_observation_with_progress'; do
  if ! grep -Fq "$token" "$INTERFACE_PY"; then
    echo "[a3-rtc] incompatible RoboInterface: $INTERFACE_PY" >&2
    echo "[a3-rtc] missing protocol token: $token" >&2
    echo "[a3-rtc] deploy robotinterface/master_wholebody_human" >&2
    exit 65
  fi
done

# Load the ADU/ROS environment if present. These are read-only setup scripts.
# ROS Jazzy's generated setup.bash reads AMENT_TRACE_SETUP_FILES before it is
# defined, so it cannot be sourced while this launcher has `set -u` enabled.
source_optional_setup() {
  local setup_path="$1"
  set +u
  # Vendor setup scripts can leave a non-zero status when an optional overlay
  # is absent even though their required exports were installed successfully.
  # Keep the later concrete file/topic checks as the source of truth.
  source "$setup_path" || true
  set -u
}

if [[ -f /agibot/software/v0/entry/env/env.sh ]]; then
  # shellcheck disable=SC1091
  source_optional_setup /agibot/software/v0/entry/env/env.sh 2>/dev/null
fi
if [[ -f /opt/ros/jazzy/setup.bash ]]; then
  # shellcheck disable=SC1091
  source_optional_setup /opt/ros/jazzy/setup.bash 2>/dev/null
fi
PROTO_SETUP=/agibot/software/v0/share/ros2_package/aimrt_protocol_ros2_package/share/ros2_plugin_proto/local_setup.bash
if [[ -f "$PROTO_SETUP" ]]; then
  # shellcheck disable=SC1090
  source_optional_setup "$PROTO_SETUP"
fi
if [[ -f "$ROS_SERVER_DIR/install/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source_optional_setup "$ROS_SERVER_DIR/install/setup.bash"
fi

export A3_ROBOINTERFACE_DIR="$ROBOINTERFACE_DIR"
export A3_TRT_SITE="$TRT_SITE"
export A3_ROS_SERVER_DIR="$ROS_SERVER_DIR"
export A3_DEPLOYMENT_SCRIPTS_DIR="$GR00T_DIR/scripts/deployment"
export PYTHONPATH="$ROS_SERVER_DIR/_pb_gen:$GR00T_DIR:$SCRIPT_DIR:$ROBOINTERFACE_DIR:${PYTHONPATH:-}"

CUDA_LIB_PATHS=()
for cuda_lib in \
  /usr/local/cuda/thor/targets/aarch64-linux/lib \
  /usr/local/cuda/targets/aarch64-linux/lib \
  /usr/local/cuda-12.8/thor/targets/aarch64-linux/lib \
  /usr/local/cuda-12.8/targets/aarch64-linux/lib \
  /usr/lib/aarch64-linux-gnu \
  "$TRT_SITE/nvidia/cufile/lib" \
  /agibot/edge_deploy/trt_venv/lib/python3.12/site-packages/nvidia/cufile/lib; do
  if [[ -d "$cuda_lib" ]]; then
    CUDA_LIB_PATHS+=("$cuda_lib")
  fi
done
if (( ${#CUDA_LIB_PATHS[@]} > 0 )); then
  CUDA_PREFIX="$(IFS=:; printf '%s' "${CUDA_LIB_PATHS[*]}")"
  export LD_LIBRARY_PATH="$CUDA_PREFIX${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

echo "[a3-rtc] python:          $PYTHON_BIN"
echo "[a3-rtc] checkpoint:      $MODEL_PATH"
echo "[a3-rtc] engines:         $ENGINE_DIR"
"$PYTHON_BIN" - <<'PY'
import os
import sys
import torch

site = os.environ["A3_TRT_SITE"]
sys.path.insert(0, site)
import tensorrt as trt
import cv2
import numpy as np

print(f"[a3-rtc] TensorRT:       {trt.__version__} ({trt.__file__})")
if not trt.__version__.startswith("10.13"):
    raise SystemExit(
        f"[a3-rtc] TensorRT 10.13 is required by the deployed engines; "
        f"loaded {trt.__version__} from {trt.__file__}"
    )

# Thor requires its architecture-specific cuBLAS build.  A CUDA allocation can
# succeed with the generic SBSA library while cublasCreate fails only on the
# first model linear layer, so exercise one tiny GEMM during preflight.
x = torch.ones((16, 16), device="cuda", dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()
if float(y[0, 0]) != 16.0:
    raise SystemExit("[a3-rtc] Thor cuBLAS preflight returned an invalid result")
print("[a3-rtc] Thor cuBLAS:     ready")

frame = np.zeros((4, 6, 3), dtype=np.uint8)
resized = cv2.resize(frame, (3, 2))
if resized.shape != (2, 3, 3):
    raise SystemExit(f"[a3-rtc] OpenCV resize returned invalid shape: {resized.shape}")
print(f"[a3-rtc] OpenCV:          {cv2.__version__} ({cv2.__file__})")
print(f"[a3-rtc] NumPy:           {np.__version__} ({np.__file__})")
PY

# Import the complete edge client before touching robot state.  This catches
# missing runtime modules during --check instead of after the operator starts
# a real inference run.
"$PYTHON_BIN" "$SCRIPT_DIR/infer_a3_edge.py" --help >/dev/null
echo "[a3-rtc] edge client imports: ready"
echo "[a3-rtc] RoboInterface:   $INTERFACE_PY"
echo "[a3-rtc] gr00t:           $GR00T_DIR"
echo "[a3-rtc] ros_server:      $ROS_SERVER_DIR"
echo "[a3-rtc] transport:       $TRANSPORT"
if [[ "$TRANSPORT" == "http" ]]; then
  echo "[a3-rtc] robot HTTP:      http://${ROBOT_IP}:${ROBOT_PORT}"
else
  echo "[a3-rtc] ROS worker:      $WORKER_PYTHON (robot_ip=$ROBOT_IP)"
  echo "[a3-rtc] observation IPC: POSIX shared memory (raw BGR + WBC state)"
  echo "[a3-rtc] action IPC:      local pipe -> server-atomic whole-body swap"
fi
echo "[a3-rtc] route:           whole_body_state -> reference_window"
echo "[a3-rtc] mode:            $MODE"
echo "[a3-rtc] task:            $TASK"
echo "[a3-rtc] min exec steps:  $EXEC_STEPS policy frames"

if [[ "$TRANSPORT" == "subproc" ]]; then
  STANDALONE_SERVER=0
  if pgrep -f '/a3_server/lib/a3_server/server|ros2 run a3_server server' \
      >/dev/null 2>&1; then
    STANDALONE_SERVER=1
  elif command -v ros2 >/dev/null 2>&1 \
      && ros2 node list 2>/dev/null | grep -Fxq /a3_server; then
    STANDALONE_SERVER=1
  fi
  if (( STANDALONE_SERVER )); then
    echo "[a3-rtc] standalone /a3_server is already running" >&2
    echo "[a3-rtc] subproc transport owns A3ServerNode; stop the standalone " \
         "a3_server first to avoid duplicate reference_window publishers" >&2
    exit 70
  fi
fi

# HTTP uses the standalone a3_server and therefore probes its endpoints here.
# Subproc owns A3ServerNode itself; infer_a3_edge performs the equivalent
# shared-memory sensor readiness check after starting the worker.
if [[ "$TRANSPORT" == "http" ]]; then
"$PYTHON_BIN" - "$ROBOT_IP" "$ROBOT_PORT" "$SERVER_READY_TIMEOUT" <<'PY'
import json
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

host, port, timeout_s = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
base = f"http://{host}:{port}"
deadline = time.monotonic() + timeout_s
last_error = None

print(
    "[a3-rtc] waiting for whole_body_state; before a3_server/inference, "
    "the robot must be in AVATAR / WHOLE_BODY_TRACKING (Sonic) mode",
    flush=True,
)

def get(path):
    with urlopen(base + path, timeout=3.0) as response:
        return response.status, json.loads(response.read().decode("utf-8"))

while time.monotonic() < deadline:
    try:
        request = Request(
            base + "/set_state_source",
            data=json.dumps({"source": "whole_body"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=3.0) as response:
            if response.status != 200:
                raise RuntimeError(f"/set_state_source HTTP {response.status}")

        status, state = get(
            "/get_observation_with_progress"
            "?cameras=__preflight_none__"
            "&include_imu=true"
            "&state_source=whole_body_state"
        )
        if status != 200:
            raise RuntimeError(
                f"/get_observation_with_progress HTTP {status}"
            )
        if state.get("state_source") != "whole_body_state":
            raise RuntimeError("observation did not use whole_body_state")
        if not isinstance(state.get("chunk_progress"), dict):
            raise RuntimeError("observation has no chunk_progress object")
        for progress_key in ("wb", "arm"):
            if progress_key not in state["chunk_progress"]:
                raise RuntimeError(
                    f"chunk progress missing key: {progress_key}"
                )
        joints = state.get("joints") or {}
        missing = []
        for name, dim in (("leg", 12), ("waist", 3), ("arm", 14)):
            pos = ((joints.get(name) or {}).get("position") or [])
            if len(pos) < dim:
                missing.append(f"{name}({len(pos)}/{dim})")
        if not missing:
            print(
                "[a3-rtc] whole-body observation ready: "
                "progress + leg=12 waist=3 arm=14"
            )
            break
        last_error = "incomplete state: " + ", ".join(missing)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, RuntimeError) as exc:
        last_error = str(exc)
    time.sleep(1.0)
else:
    raise SystemExit(
        f"a3_server/whole_body_state not ready after {timeout_s:.0f}s: {last_error}; "
        "first verify the robot is in AVATAR / WHOLE_BODY_TRACKING (Sonic) mode, "
        "then ensure a3_server was built from master_wholebody_human and the "
        "decorated /wbc/whole_body_state protobuf topic is publishing"
    )
PY
fi

REF_TOPIC=/wbc/infer/reference_window/pb_3Aaimdk_2Eprotocol_2ETaWholeBodyReferenceWindow
if command -v ros2 >/dev/null 2>&1; then
  SONIC_DEADLINE=$((SECONDS + SONIC_READY_TIMEOUT))
  TOPIC_INFO=""
  while (( SECONDS < SONIC_DEADLINE )); do
    TOPIC_INFO="$(ros2 topic info "$REF_TOPIC" 2>&1 || true)"
    if printf '%s\n' "$TOPIC_INFO" | grep -Eq 'Subscription count: [1-9][0-9]*'; then
      break
    fi
    sleep 1
  done
  printf '%s\n' "$TOPIC_INFO"
  if ! printf '%s\n' "$TOPIC_INFO" | grep -Eq 'Subscription count: [1-9][0-9]*'; then
    echo "[a3-rtc] no sonic/WBC subscriber on $REF_TOPIC" >&2
    echo "[a3-rtc] enable gr00t_inference/sonic before starting inference" >&2
    exit 69
  fi
fi

INFER_ARGS=(
  --mode "$MODE" \
  --transport "$TRANSPORT" \
  --worker-python "$WORKER_PYTHON" \
  --robot_ip "$ROBOT_IP" \
  --robot_port "$ROBOT_PORT" \
  --model-path "$MODEL_PATH" \
  --engine-dir "$ENGINE_DIR" \
  --trt-mode "$TRT_MODE" \
  --embodiment-tag NEW_EMBODIMENT \
  --hand_kind "$HAND_KIND" \
  --state-source whole_body_state \
  --task "$TASK" \
  --exec_steps "$EXEC_STEPS" \
  --no-reset \
  --record-chunks-dir "$RECORD_DIR"
)

if (( CHECK_ONLY )); then
  if [[ "$TRANSPORT" == "subproc" ]]; then
    "$PYTHON_BIN" -u "$SCRIPT_DIR/infer_a3_edge.py" \
      "${INFER_ARGS[@]}" --sensor-check-only "$@"
  fi
  echo "[a3-rtc] preflight passed; no action chunk was sent"
  exit 0
fi

mkdir -p -- "$RECORD_DIR"

exec "$PYTHON_BIN" -u "$SCRIPT_DIR/infer_a3_edge.py" \
  "${INFER_ARGS[@]}" "$@"
