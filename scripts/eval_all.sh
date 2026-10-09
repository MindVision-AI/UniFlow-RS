#!/usr/bin/env bash
# Optional shorthand; all launch settings are owned by eval.sh.
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/eval_all.sh CHECKPOINT [OUTPUT_DIR]" >&2
  exit 2
fi
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${ROOT_DIR}/scripts/eval.sh" all "$1" "${2:-${ROOT_DIR}/outputs/all_benchmarks}"
