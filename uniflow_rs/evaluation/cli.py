"""Evaluation components: cli."""

from __future__ import annotations

import argparse
import math
from pathlib import (
    Path,
)
from uniflow_rs._compat.multitask import (
    ALL_TASKS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Evaluate UniFlow-RS on remote-sensing benchmarks")
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/uniflow_rs_512_2step_eval.yaml"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--flux-model-dir",
        "--root-model-dir",
        dest="flux_model_dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--text-model-dir",
        type=Path,
        default=None,
        help="Local Qwen encoder directory for fixed task prompts; overrides config. "
        "Referring-expression caches are prepared separately.",
    )
    parser.add_argument(
        "--task",
        choices=(*ALL_TASKS, "all"),
        default="all",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
    )
    parser.add_argument(
        "--referring-split",
        type=str,
        default=None,
        help=(
            "Override the physical referring manifest split. This exposes "
            "LaSeRS views such as test_implicit, test_multi_cate, and "
            "test_part_level without changing the YAML."
        ),
    )
    parser.add_argument(
        "--referring-suite",
        action="store_true",
        help=(
            "Evaluate all nine official LaSeRS test views in one process, "
            "loading FLUX only once. Requires --task referring --split test."
        ),
    )
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Deterministic VAE sampling seed. Defaults to "
            "evaluation.validation_seed from the config."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/paper_benchmarks"),
    )
    parser.add_argument(
        "--detection-submission-dir",
        dest="detection_submission_dir",
        type=Path,
        default=None,
        help=(
            "Write oriented-detection per-class text files and a zip here. "
            "For an "
            "unlabeled test split, export is automatic under "
            "<output-dir>/detection/test/task1_submission."
        ),
    )
    parser.add_argument(
        "--height-ground-threshold",
        type=float,
        default=None,
        help=(
            "Override inference.height_postprocess.ground_threshold_meters. "
            "Predicted heights below this value are set exactly to zero."
        ),
    )
    parser.add_argument(
        "--height-min-component-area",
        type=int,
        default=None,
        help=(
            "Override inference.height_postprocess.min_component_area. "
            "Smaller non-ground connected components are removed."
        ),
    )
    parser.add_argument(
        "--height-median-kernel",
        type=int,
        default=None,
        help=(
            "Override inference.height_postprocess.median_kernel. "
            "Use 1 to disable median smoothing."
        ),
    )
    parser.add_argument("--metrics-only", action="store_true")
    return parser.parse_args()


def _optional_path(cli_value: Path | None, config_value) -> Path | None:
    value = cli_value if cli_value is not None else config_value
    if value in (None, ""):
        return None
    return Path(value).expanduser().resolve()


def _height_postprocess_config(
    cfg: dict,
    args: argparse.Namespace,
) -> dict[str, bool | float | int]:
    """Resolve and validate metric-height postprocessing parameters."""

    configured = dict(cfg.get("inference", {}).get("height_postprocess", {}) or {})
    enabled = bool(configured.get("enabled", False))
    threshold = float(
        configured.get("ground_threshold_meters", 0.0)
        if args.height_ground_threshold is None
        else args.height_ground_threshold
    )
    minimum_area = int(
        configured.get("min_component_area", 0)
        if args.height_min_component_area is None
        else args.height_min_component_area
    )
    median_kernel = int(
        configured.get("median_kernel", 1)
        if args.height_median_kernel is None
        else args.height_median_kernel
    )
    component_median_weight = float(configured.get("component_median_weight", 0.0))
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("Height ground threshold must be finite and >= 0")
    if minimum_area < 0:
        raise ValueError("Height minimum component area must be >= 0")
    if median_kernel < 1 or median_kernel % 2 == 0:
        raise ValueError("Height median kernel must be a positive odd integer")
    if not 0.0 <= component_median_weight <= 1.0:
        raise ValueError("Height component median weight must be in [0, 1]")
    # Any explicit CLI override is an intentional request to run the
    # postprocessor, even when the YAML default is disabled.
    enabled = enabled or any(
        value is not None
        for value in (
            args.height_ground_threshold,
            args.height_min_component_area,
            args.height_median_kernel,
        )
    )
    return {
        "enabled": enabled,
        "ground_threshold_meters": threshold,
        "min_component_area": minimum_area,
        "median_kernel": median_kernel,
        "component_median_weight": component_median_weight,
    }
