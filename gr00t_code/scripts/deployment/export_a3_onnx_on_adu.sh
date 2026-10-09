#!/usr/bin/env bash

set -euo pipefail

# Export a GR00T N1.7 checkpoint to ONNX directly on an A3 ADU.
#
# The validated A3 environment intentionally uses two venvs:
#   - torch_venv: Thor sm_101 PyTorch
#   - trt_venv:  ONNX and the remaining deployment dependencies
#
# Do not prepend the complete trt_venv site-packages directory through
# PYTHONPATH: it contains another PyTorch build.  The bootstrap below appends
# it after torch_venv has been selected, so the Thor PyTorch always wins.

EDGE_ROOT="${A3_EDGE_ROOT:-/agibot/edge_deploy}"
GR00T_ROOT="${A3_GR00T_ROOT:-${EDGE_ROOT}/gr00t_code}"
PYTHON_BIN="${A3_PYTHON_BIN:-${EDGE_ROOT}/torch_venv/bin/python}"
TRT_SITE="${A3_TRT_SITE:-${EDGE_ROOT}/trt_venv/lib/python3.12/site-packages}"
PIPELINE="${GR00T_ROOT}/scripts/deployment/build_trt_pipeline.py"
CUFILE_LIB="${TRT_SITE}/nvidia/cufile/lib"

for path in "$PYTHON_BIN" "$PIPELINE" "$TRT_SITE/onnx" \
  "$TRT_SITE/diffusers" "$CUFILE_LIB/libcufile.so.0"; do
  if [[ ! -e "$path" ]]; then
    echo "[adu-onnx] missing required path: $path" >&2
    exit 1
  fi
done

export PYTHONNOUSERSITE=1
export GR00T_ONNX_EXPORTER_MODE=legacy
export LD_LIBRARY_PATH="/usr/local/cuda/thor/targets/aarch64-linux/lib:/usr/local/cuda/targets/aarch64-linux/lib:/usr/local/cuda-12.8/thor/targets/aarch64-linux/lib:/usr/local/cuda-12.8/targets/aarch64-linux/lib:${CUFILE_LIB}:/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"

# Remove an inherited PYTHONPATH so an operator's shell cannot accidentally
# select trt_venv's incompatible PyTorch before torch_venv.
env -u PYTHONPATH "$PYTHON_BIN" - \
  "$GR00T_ROOT" "$TRT_SITE" "$PIPELINE" "$@" <<'PY'
from __future__ import annotations

import importlib
from pathlib import Path
import runpy
import sys

gr00t_root = Path(sys.argv[1]).resolve()
trt_site = Path(sys.argv[2]).resolve()
pipeline = Path(sys.argv[3]).resolve()
pipeline_args = sys.argv[4:]

sys.path.insert(0, str(gr00t_root))
sys.path.append(str(trt_site))

torch = importlib.import_module("torch")
onnx = importlib.import_module("onnx")
# TensorRT must be loaded before the first PyTorch CUDA context is created on
# Thor.  Loading it only when the build step imports build_tensorrt_engine.py
# can select its bundled CUDA runtime too late and fail with cudaError 35.
trt_site_str = str(trt_site)
while trt_site_str in sys.path:
    sys.path.remove(trt_site_str)
sys.path.insert(0, trt_site_str)
tensorrt = importlib.import_module("tensorrt")
sys.path.remove(trt_site_str)
sys.path.append(trt_site_str)
for module_name in ("diffusers", "peft", "accelerate", "tyro"):
    importlib.import_module(module_name)

if not tensorrt.__version__.startswith("10.13"):
    raise RuntimeError(
        f"TensorRT 10.13 is required, got {tensorrt.__version__} "
        f"from {tensorrt.__file__}"
    )

torch_path = Path(torch.__file__).resolve()
python_prefix = Path(sys.prefix).resolve()
if python_prefix not in torch_path.parents:
    raise RuntimeError(
        f"wrong torch selected: {torch_path}; expected it below {python_prefix}"
    )

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available in the Thor torch_venv")
capability = torch.cuda.get_device_capability()
if capability != (10, 1):
    raise RuntimeError(f"expected Thor sm_101, got capability={capability}")

# A real GEMM catches the common false-positive where CUDA is visible but the
# wheel has no sm_101 kernels, or the CUDA/cuBLAS library order is wrong.
x = torch.ones((16, 16), device="cuda", dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()
if float(y[0, 0]) != 16.0:
    raise RuntimeError("Thor BF16 GEMM preflight returned an unexpected result")

print(f"[adu-onnx] python: {sys.executable}")
print(f"[adu-onnx] torch:  {torch.__version__} ({torch.__file__})")
print(f"[adu-onnx] onnx:   {onnx.__version__} ({onnx.__file__})")
print(f"[adu-onnx] TRT:    {tensorrt.__version__} ({tensorrt.__file__})")
print(f"[adu-onnx] GPU:    sm_{capability[0]}{capability[1]}")
print("[adu-onnx] exporter: legacy")

sys.argv = [str(pipeline), *pipeline_args]
runpy.run_path(str(pipeline), run_name="__main__")
PY
