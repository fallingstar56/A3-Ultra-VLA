#!/usr/bin/env bash

# Prepare a newly copied GR00T checkpoint for this edge deployment.
#
# Usage:
#   bash prepare_checkpoint.sh \
#     --checkpoint /agibot/edge_deploy_minimal/models/checkpoint-60000 \
#     --cosmos /agibot/edge_deploy_minimal/Cosmos-Reason2-2B

set -euo pipefail

EDGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT="${A3_MODEL_PATH:-}"
COSMOS_DIR="${A3_COSMOS_PATH:-${EDGE_ROOT}/Cosmos-Reason2-2B}"
RESOLVER="${EDGE_ROOT}/gr00t_code/scripts/deployment/resolve_a3_checkpoint_config.py"
PYTHON_BIN="${A3_WORKER_PYTHON:-/usr/bin/python3}"

usage() {
  sed -n '3,9p' "${BASH_SOURCE[0]}"
}

while (( $# > 0 )); do
  case "$1" in
    --checkpoint)
      CHECKPOINT="${2:?--checkpoint requires a path}"
      shift 2
      ;;
    --cosmos)
      COSMOS_DIR="${2:?--cosmos requires a path}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[prepare] unknown argument: $1" >&2
      usage >&2
      exit 64
      ;;
  esac
done

if [[ -z "$CHECKPOINT" ]]; then
  echo "[prepare] set A3_MODEL_PATH or pass --checkpoint" >&2
  exit 64
fi

CHECKPOINT="$(readlink -f -- "$CHECKPOINT")"
COSMOS_DIR="$(readlink -f -- "$COSMOS_DIR")"

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "[prepare] missing ${label}: $path" >&2
    exit 66
  fi
}

require_file "$PYTHON_BIN" "system Python"
require_file "$RESOLVER" "checkpoint resolver"
for name in config.json processor_config.json statistics.json \
            embodiment_id.json model.safetensors.index.json; do
  require_file "$CHECKPOINT/$name" "checkpoint file"
done
for name in config.json model.safetensors tokenizer.json; do
  require_file "$COSMOS_DIR/$name" "Cosmos file"
done

# Validate every shard referenced by the Hugging Face index before editing
# configuration files. This catches partial rsync/copy operations early.
"$PYTHON_BIN" - "$CHECKPOINT" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
index = json.loads((root / "model.safetensors.index.json").read_text())
weight_map = index.get("weight_map") or {}
if not weight_map:
    raise SystemExit("[prepare] model.safetensors.index.json has no weight_map")
shards = sorted(set(weight_map.values()))
missing = [name for name in shards if not (root / name).is_file()]
if missing:
    raise SystemExit(f"[prepare] missing checkpoint shards: {missing}")
print(f"[prepare] checkpoint shards ready: {len(shards)}")
PY

echo "[prepare] resolving embodiment, hand kind and training prompt"
"$PYTHON_BIN" "$RESOLVER" --checkpoint "$CHECKPOINT"

DEPLOY_ENV_TMP="${CHECKPOINT}/.deploy.env.tmp.$$"
trap 'rm -f -- "${DEPLOY_ENV_TMP:-}"' EXIT
"$PYTHON_BIN" "$RESOLVER" \
  --checkpoint "$CHECKPOINT" \
  --shell > "$DEPLOY_ENV_TMP"
chmod 0644 "$DEPLOY_ENV_TMP"
mv -f -- "$DEPLOY_ENV_TMP" "$CHECKPOINT/deploy.env"

# Rewrite only the runtime checkpoint copies. Preserve the first pre-deploy
# version so the training-time paths remain auditable.
"$PYTHON_BIN" - "$CHECKPOINT" "$COSMOS_DIR" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

root = Path(sys.argv[1])
cosmos = str(Path(sys.argv[2]).resolve())

for name in ("config.json", "processor_config.json"):
    path = root / name
    backup = path.with_name(path.name + ".pre_edge_deploy.bak")
    if not backup.exists():
        shutil.copy2(path, backup)

    data = json.loads(path.read_text())
    changed = data.get("model_name") != cosmos
    data["model_name"] = cosmos
    if name == "processor_config.json":
        kwargs = data.setdefault("processor_kwargs", {})
        changed = changed or kwargs.get("model_name") != cosmos
        kwargs["model_name"] = cosmos
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(f"[prepare] {path}: model_name -> {cosmos}"
          + ("" if changed else " (already set)"))
PY

mkdir -p -- "$CHECKPOINT/onnx" "$CHECKPOINT/engines" "$CHECKPOINT/chunks"

# shellcheck disable=SC1090
source "$CHECKPOINT/deploy.env"
printf '[prepare] ready: checkpoint=%s\n' "$CHECKPOINT"
printf '[prepare] embodiment=%s hand_kind=%s task=%s\n' \
  "${EMBODIMENT:-}" "${HAND_KIND:-}" "${TASK:-}"
printf '[prepare] next: A3_MODEL_PATH=%q bash %q\n' \
  "$CHECKPOINT" "$EDGE_ROOT/convert_model.sh"

