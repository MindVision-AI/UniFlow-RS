#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: scripts/eval.sh DATASET_OR_all CHECKPOINT [OUTPUT_DIR]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="$1"
CHECKPOINT="$2"
OUTPUT_DIR="${3:-${ROOT_DIR}/outputs/${DATASET}}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"

cd "${ROOT_DIR}"
python scripts/evaluate_dataset.py "${DATASET}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-processes "${NUM_PROCESSES}" \
  --num-steps 2 \
  --seed 12345 \
  ${METRICS_ONLY:+--metrics-only}
