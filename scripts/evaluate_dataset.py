#!/usr/bin/env python3
"""Launch one benchmark or all benchmarks with the same evaluation settings."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


DATASETS = {
    "whu_building": ("building", None),
    "spacenet6": ("sar_building", None),
    "whu_cd": ("change", "whu_cd"),
    "levir_cd_plus": ("change", "levir_cd_plus"),
    "urbansar_floods": ("sar_change", None),
    "loveda": ("semantic", None),
    "openearthmap_rgb": ("sar_semantic", "openearthmap_optical"),
    "openearthmap_sar": ("sar_semantic", "openearthmap_sar"),
    "potsdam": ("height", "potsdam"),
    "dfc23_track2": ("height", "dfc23_track2"),
    "rrsisd": ("referring", "rrsisd"),
    "lasers": ("referring", "lasers"),
    "dior_r": ("detection", None),
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("all", *DATASETS))
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs" / "uniflow_rs_512_2step_eval.yaml",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--metrics-only", action="store_true")
    parser.add_argument("--flux-model-dir", type=Path, default=None)
    parser.add_argument("--text-model-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.num_processes < 1:
        raise ValueError("--num-processes must be positive")
    task, source = ("all", None) if args.dataset == "all" else DATASETS[args.dataset]
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else root / "outputs" / args.dataset
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with args.config.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if source is not None:
        sources = cfg["data"][task]["sources"]
        if source not in sources:
            raise KeyError(f"Missing source {source!r} under data.{task}.sources")
        cfg["data"][task]["sources"] = {source: sources[source]}
    resolved_config = output_dir / "resolved_config.yaml"
    with resolved_config.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(cfg, stream, sort_keys=False, width=88)

    eval_args = [
        str(root / "evaluate.py"),
        "--config",
        str(resolved_config),
        "--checkpoint",
        str(args.checkpoint),
        "--task",
        task,
        "--split",
        "test",
        "--num-steps",
        str(args.num_steps),
        "--seed",
        str(args.seed),
        "--output-dir",
        str(output_dir),
    ]
    if args.metrics_only:
        eval_args.append("--metrics-only")
    if args.flux_model_dir is not None:
        eval_args.extend(("--flux-model-dir", str(args.flux_model_dir)))
    if args.text_model_dir is not None:
        eval_args.extend(("--text-model-dir", str(args.text_model_dir)))

    if args.num_processes == 1:
        command = [sys.executable, *eval_args]
    else:
        command = [
            "accelerate",
            "launch",
            "--multi_gpu",
            "--num_processes",
            str(args.num_processes),
            "--num_machines",
            "1",
            "--mixed_precision",
            "bf16",
            "--dynamo_backend",
            "no",
            *eval_args,
        ]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
