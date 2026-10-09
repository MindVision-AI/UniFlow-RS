"""Public training arguments and shared-method configuration checks."""

from __future__ import annotations
import argparse
from pathlib import Path
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train UniFlow-RS jointly on the paper benchmarks"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/uniflow_rs_lora32_tmas_60k.yaml"),
    )
    parser.add_argument(
        "--flux-model-dir",
        "--root-model-dir",
        dest="flux_model_dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--text-model-dir", type=Path, default=None,
        help="Local Qwen encoder directory for fixed task prompts; overrides config. "
             "Referring-expression caches are prepared separately.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume a multitask checkpoint path or use 'latest'.",
    )
    parser.add_argument(
        "--init-adapter",
        type=Path,
        default=None,
        help=(
            "Initialize LoRA from a compatible UniFlow-RS adapter. "
            "Optimizer/epoch state is intentionally not resumed."
        ),
    )
    parser.add_argument(
        "--timestep-sampling",
        choices=(
            "config",
            "uniform",
            "task_adaptive_progress",
        ),
        default="config",
        help=(
            "Override only the random Flow-Matching timestep distribution. "
            "'uniform' sets flow.beta_alpha=flow.beta_beta=1 without "
            "adding auxiliary supervision; 'task_adaptive_progress' uses "
            "the dual-EMA task/modality TMAS sampler."
        ),
    )
    parser.add_argument(
        "--max-optimizer-steps",
        type=int,
        default=None,
        help="Override training.max_optimizer_steps for controlled pilots.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Override output.run_name for an independent run directory.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def apply_experiment_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Apply explicit CLI overrides before saving the resolved run config."""

    if args.timestep_sampling == "uniform":
        cfg.setdefault("flow", {})["beta_alpha"] = 1.0
        cfg["flow"]["beta_beta"] = 1.0
        cfg["flow"].setdefault("timestep_sampling", {})["mode"] = "uniform"
    elif args.timestep_sampling in {
        "task_adaptive_progress",
    }:
        sampling_cfg = cfg.setdefault("flow", {}).setdefault(
            "timestep_sampling",
            {},
        )
        sampling_cfg["mode"] = args.timestep_sampling
        adaptive_cfg = sampling_cfg.setdefault(args.timestep_sampling, {})
        adaptive_cfg.setdefault("num_bins", 10)
        adaptive_cfg.setdefault("uniform_mix", 0.2)
        adaptive_cfg.setdefault("epsilon", 1.0e-6)
        adaptive_cfg.setdefault("warmup_ratio", 0.1)
        adaptive_cfg.setdefault("beta_fast", 0.90)
        adaptive_cfg.setdefault("beta_slow", 0.99)
        adaptive_cfg.setdefault("gamma_difficulty", 1.0)
        adaptive_cfg.setdefault("gamma_progress", 1.0)
        adaptive_cfg.setdefault("progress_mode", "absolute")
        adaptive_cfg.setdefault("progress_floor", 0.1)
        adaptive_cfg.setdefault("progress_clip", 1.0)
    if args.max_optimizer_steps is not None:
        if int(args.max_optimizer_steps) < 1:
            raise ValueError("--max-optimizer-steps must be positive")
        cfg.setdefault("training", {})["max_optimizer_steps"] = int(
            args.max_optimizer_steps
        )
    if args.run_name is not None:
        run_name = str(args.run_name).strip()
        if not run_name:
            raise ValueError("--run-name cannot be empty")
        cfg.setdefault("output", {})["run_name"] = run_name
    return cfg


def _resolved_optional_path(
    cli_value: Path | None,
    config_value: str | None,
) -> Path | None:
    value = cli_value if cli_value is not None else config_value
    if value in (None, ""):
        return None
    return Path(value).expanduser().resolve()


def validate_method_config(cfg: dict) -> None:
    """Reject removed experimental modes before allocating models or GPUs."""
    training = cfg.get("training", {})
    if training.get("validation_enabled", False) or training.get("validation_primary_metrics") or training.get("validation_dataset_metrics"):
        raise ValueError(
            "This reviewer release uses fixed-budget training without validation or best-checkpoint selection; run evaluate.py after training"
        )
    if (cfg.get("model", {}).get("task_family_heads") or {}).get("enabled", False):
        raise ValueError(
            "Task-family-head training is not part of this shared-predictor release"
        )
    if (training.get("vae_decoder") or {}).get("enabled", False) or training.get(
        "joint_decoder_train", False
    ):
        raise ValueError(
            "This release trains shared LoRA with a frozen VAE; decoder fine-tuning is not supported"
        )
    sampling = cfg.get("flow", {}).get("timestep_sampling", {}).get("mode", "beta")
    if str(sampling).lower() not in {"beta", "uniform", "task_adaptive_progress"}:
        raise ValueError(
            "Supported timestep modes: beta, uniform, task_adaptive_progress"
        )
