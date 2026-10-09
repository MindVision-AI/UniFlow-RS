# Evaluation Guide

## Environment

Use Python 3.10, a CUDA-capable GPU, and the dependencies in
`requirements.txt`. The reference environment uses PyTorch 2.8.0 with
CUDA 12.8, torchvision 0.23.0, and Transformers 4.57.6.
See the [README](../README.md) for installation and model downloads.

## Configuration

- Evaluation: `configs/uniflow_rs_512_2step_eval.yaml`.
- Training: `configs/uniflow_rs_lora32_tmas_60k.yaml`; see
  [TRAINING.md](TRAINING.md) for usage.

Evaluation uses FLUX.2 Klein Base 4B, the frozen FLUX.2 autoencoder,
Qwen3-4B-FP8, and the shared rank-32 LoRA adapter. Defaults are
512-pixel model resolution, two flow steps, BF16, seed 12345,
and batch size 1 per GPU.

Keep the checkpoint, prompts, palettes, preprocessing, post-processing,
batch size, and GPU count fixed when comparing repeated runs.

## Checkpoint

Place the released adapter at `checkpoints/uniflow_rs.pt`.

Expected SHA-256:

```text
26d097a9567f55636f8c1fef17c94ce16be13befdc12b5f384b983840a136dac
```

The evaluator loads the adapter into the base transformer and uses the frozen
pretrained VAE. The adapter contains model weights, not optimizer or scheduler
state.

## Data and Text Features

Follow [DATASETS.md](DATASETS.md) to prepare the required dataset layouts,
splits, and referring-expression caches.

Fixed task prompts are encoded and cached on first use. The local text encoder
can be selected with `--text-model-dir pretrained/qwen3-4b-fp8` in
`evaluate.py`. If the encoder weights change, regenerate the associated caches.

RRSIS-D and LaSeRS use separate per-expression caches prepared with
`tools/prepare_rrsisd_text_embeddings.py --qwen-model-path`. Changing
`--text-model-dir` does not regenerate these caches.

## Evaluation

Run commands from the repository root. For one benchmark:

```bash
METRICS_ONLY=1 bash scripts/eval.sh whu_cd checkpoints/uniflow_rs.pt outputs/whu_cd
```

For all thirteen benchmarks:

```bash
METRICS_ONLY=1 bash scripts/eval.sh all checkpoints/uniflow_rs.pt outputs/all_benchmarks
```

Evaluation defaults to one GPU. Set `NUM_PROCESSES=2` for two GPUs.
Leave `METRICS_ONLY` unset to also save predictions and comparison panels.

## Outputs

- `metrics_summary.json`: aggregate evaluation results.
- `per_dataset_metrics.json` and `per_dataset_metrics/`: per-benchmark results.
- Task-specific `metrics.json` files.
- Inputs, ground truth, probabilities, predictions, and comparison panels
  when image export is enabled, as applicable to each task.

Metrics are computed from model predictions and dataset labels, not from
exported visualization images.
