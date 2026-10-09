#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/uniflow_rs_lora32_tmas_60k.yaml}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

accelerate launch \
  --multi_gpu \
  --num_processes "${NUM_PROCESSES}" \
  --num_machines 1 \
  --mixed_precision "${MIXED_PRECISION}" \
  --dynamo_backend no \
  train.py \
  --config "${CONFIG}"
