# Training

UniFlow-RS training uses:

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

Evaluation modules live under `uniflow_rs/evaluation/`, and dataset modules under
`uniflow_rs/data/`. Rotated detection uses the DIOR-R interface. Supported timestep
modes are `task_adaptive_progress` (paper TMAS), `uniform`, and fixed `beta`.

## Training configuration

`configs/uniflow_rs_lora32_tmas_60k.yaml` specifies the following training settings:

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

The training objective is a single latent Flow Matching loss.

## Checkpointing and resume

The configuration sets `training.validation_enabled: false` and uses a fixed
training budget (60,000 optimizer steps, subject to the epoch cap).
Periodic `checkpoint_epoch_*.pt` files are saved for recovery;
use the checkpoint saved at the end of the scheduled run for final evaluation.

Resume restores LoRA, optimizer, scheduler, step/epoch, and TMAS state from a
training checkpoint. Use `evaluate.py` for post-training benchmark evaluation.
