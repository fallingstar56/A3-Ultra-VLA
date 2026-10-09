#!/usr/bin/env bash

set -euo pipefail

EDGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${EDGE_ROOT}/run.sh" --check "$@"

