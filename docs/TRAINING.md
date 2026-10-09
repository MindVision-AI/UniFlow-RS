# Training components retained in the anonymous release

This release includes the training path required to understand and reproduce
the method described in the manuscript:

- the frozen FLUX.2 Klein 4B Base MMDiT, VAE, and Qwen text conditioner;
- one shared rank-32 LoRA adapter;
- deterministic structured-target construction for all six task families;
- task-dependent source-state and conditioning construction;
- the common latent Flow Matching regression objective;
- task/dataset mixture sampling and paired RGB/SAR batches;
- Task--Modality Adaptive Timestep Sampling (TMAS), including its canonical
  task-family-by-physical-modality grouping, dual EMAs, distributed
  sample-count-weighted aggregation, warm-up, exploration mixture, state
  serialization, and restart behavior;
- AdamW, MultiStepLR, gradient accumulation, and scheduled checkpointing.

## Code layout

`train.py` is a thin CLI entry point. Training code is split by responsibility:

| Module | Responsibility |
| --- | --- |
| `uniflow_rs/training/engine.py` | Shared-LoRA initialization, optimizer updates, and training orchestration |
| `uniflow_rs/training/tmas.py` | Canonical task/modality grouping, dual EMAs, probabilities, sampling, distributed statistics, and state restoration |
| `uniflow_rs/training/data.py` | Training data loaders and required prompt-key selection |
| `uniflow_rs/training/checkpoints.py` | Run directories, resume-path lookup, pruning, and compatible adapter serialization |
| `uniflow_rs/training/cli.py` | Arguments, overrides, and supported-method checks |
| `uniflow_rs/task_runtime.py` | Shared detection backend, task metrics, and OpenEarthMap prediction |
| `uniflow_rs/text_conditions.py` | Shared fixed-prompt encoding and referring-expression cache loading |
| `uniflow_rs/text_config.py` | Text-model selection and fixed-prompt cache identity |

Training and evaluation share text conditioning without importing each other's
loops. Evaluation directly uses `task_runtime.py` for metrics and OpenEarthMap
prediction. Its sliding-window temperature falls back to
`evaluation.palette_temperature` (0.1 if unset); an explicit sliding-window
temperature overrides that value.

The model implementation and paper evaluation settings are unchanged.
DOTA-only CLI options have been removed; use the DIOR-R detection interface.
Evaluation modules live under `uniflow_rs/evaluation/`, and dataset modules under
`uniflow_rs/data/`.
TMAS state
dictionary keys and model parameter names are preserved. The supported timestep
modes are `task_adaptive_progress` (paper TMAS), `uniform`, and fixed `beta`.
Older single-EMA adaptive samplers, task-family-head training, and trainable-VAE
experiments are excluded. Requests to enable removed modes fail explicitly.

## Manuscript-declared recipe

`configs/uniflow_rs_lora32_tmas_60k.yaml` retains the model and optimizer recipe
and now disables training-time validation entirely. This revised recipe is not a
record of how the released checkpoint was trained and does not retroactively
change any existing result:

- 60,000 optimizer updates;
- global batch size 32 on four GPUs (per-device batch 4 and gradient
  accumulation 2);
- AdamW with initial learning rate `5e-5`;
- learning-rate halving at 20k, 35k, and 50k updates;
- frozen base model, VAE, and text encoder;
- rank-32 shared LoRA;
- TMAS with 10 bins, exploration weight 0.2, fast/slow EMA coefficients
  0.90/0.99, and a 10% uniform warm-up;
- complete training partitions, with no validation or best-checkpoint selection;
- the DIOR-R instruction is `Generate a class-coded mask of all DIOR-R object
  classes.`, matching the single class-coded target canvas used by the public
  protocol.

Run it with:

```bash
NUM_PROCESSES=4 bash scripts/train.sh \
  configs/uniflow_rs_lora32_tmas_60k.yaml
```

Auxiliary binary, semantic, and height endpoint-supervision branches and
forced-t=1 sampling have been removed from this release, including their
configuration fields and loss helpers. The default recipe uses the single
Flow Matching loss. This cleanup does not change existing checkpoint weights
or establish how those weights were historically trained.

## Fixed-budget training: validation disabled

The public recipe sets `training.validation_enabled: false`. All thirteen
sources retain their complete configured training partitions. No validation
partition is read, no training-only holdout is created, and no validation or
test score is used for scheduling, early stopping, or checkpoint selection.

Training runs to the configured budget (60,000 optimizer steps, subject to the
epoch cap). Periodic `checkpoint_epoch_*.pt` files are still saved for recovery;
use the checkpoint saved at the end of the scheduled run for final evaluation.
No validation-selected `best.pt` or task-specific best files are generated.
The supplied `checkpoints/uniflow_rs.pt` is an existing released artifact,
not a file newly selected by this training recipe.

All source configurations have `val_split: null` and a disabled validation
policy. Metric-selection mappings are omitted. The optional training-validation
loop and best-score selection have been removed; enabling them now fails
explicitly before model allocation. No validation-partition report is written.
The dataset builder also rejects validation requests rather than silently
rewriting them. Split checks retain test-as-training rejection and check
physical-image overlap between training and test in prepared manifests;
they do not create validation readers or remove training samples.
Resume restores LoRA, optimizer, scheduler, step/epoch, and TMAS state, without
using historical best scores. New checkpoints omit unused per-task best-score
metadata; the existing adapter loading format and parameter keys are unchanged.

Checkpoint weights and inference computations are unchanged. The evaluation-only
palette decoding temperature is now named `evaluation.palette_temperature`; its
value remains 0.1. Independent post-training benchmark evaluation remains available.

## What remains excluded

Private infrastructure paths, credentials, training logs, TensorBoard event
files, generated predictions, dataset files, checkpoint binaries, and
standalone exploratory scripts are not redistributed. The public training path
uses one shared LoRA and a frozen VAE. Alternate task-family heads, decoder
fine-tuning objectives, and superseded adaptive sampling modes are not included.
