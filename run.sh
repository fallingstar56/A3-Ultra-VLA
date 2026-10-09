#!/usr/bin/env bash

set -euo pipefail

EDGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${EDGE_ROOT}/env.sh"

exec bash "${EDGE_ROOT}/edge_infer/run_a3_adu_wholebody_rtc.sh" "$@"

