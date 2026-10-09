"""Evaluation components: engine."""

from __future__ import annotations

import json
import logging
import yaml
from accelerate import (
    Accelerator,
)
from uniflow_rs.model import UniFlowRS
from uniflow_rs._compat.dota import (
    DETECTION_TASK,
)
from uniflow_rs._compat.multitask import (
    HEIGHT_TASK,
    REFERRING_TASK,
    SAR_BUILDING_TASK,
    active_tasks_from_config,
    build_task_datasets,
)
from uniflow_rs.evaluation.cli import (
    _optional_path,
    parse_args,
)
from uniflow_rs.evaluation.runner import (
    LOGGER,
    evaluate_task,
)
from uniflow_rs.evaluation.distributed import evaluation_dataloader_config

LASERS_REFERRING_TEST_SPLITS = (
    "test_explicit",
    "test_implicit",
    "test_instance_level",
    "test_part_level",
    "test_semantic_level",
    "test_long_query",
    "test_short_query",
    "test_single_cate",
    "test_multi_cate",
)


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    accelerator = Accelerator(
        mixed_precision="bf16", dataloader_config=evaluation_dataloader_config(),
    )
    logging.basicConfig(
        level=logging.INFO if accelerator.is_main_process else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    pretrained_cfg = cfg.get("pretrained", {})
    flux_model_dir = _optional_path(
        args.flux_model_dir,
        pretrained_cfg.get("local_dir"),
    )
    text_model_dir = _optional_path(
        args.text_model_dir,
        pretrained_cfg.get("text_model_dir"),
    )
    checkpoint = args.checkpoint.expanduser().resolve()
    model = UniFlowRS.from_checkpoint(
        cfg,
        checkpoint,
        accelerator,
        flux_model_dir=flux_model_dir,
        text_model_dir=text_model_dir,
    )
    backbone = model.backbone

    configured_tasks = active_tasks_from_config(cfg)
    selected_tasks = configured_tasks if args.task == "all" else (args.task,)
    unavailable = sorted(set(selected_tasks) - set(configured_tasks))
    if unavailable:
        raise ValueError(
            f"Task(s) {unavailable} are disabled by "
            f"evaluation.active_tasks={configured_tasks}"
        )
    if args.referring_suite:
        if args.referring_split is not None:
            raise ValueError(
                "Use either --referring-suite or --referring-split, not both"
            )
        if selected_tasks != (REFERRING_TASK,) or args.split != "test":
            raise ValueError("--referring-suite requires --task referring --split test")
        suite_metrics = {}
        for physical_split in LASERS_REFERRING_TEST_SPLITS:
            cfg["data"][REFERRING_TASK]["test_split"] = physical_split
            dataset = build_task_datasets(
                cfg["data"],
                logical_split="test",
                augment=False,
                tasks=(REFERRING_TASK,),
            )[REFERRING_TASK]
            suite_metrics[physical_split] = evaluate_task(
                REFERRING_TASK,
                accelerator,
                model,
                dataset,
                cfg,
                args,
            )
        if accelerator.is_main_process:
            summary = {
                "checkpoint": str(checkpoint),
                "base_model": backbone.base_model,
                "model_variant": backbone.model_variant,
                "dataset": "LaSeRS",
                "test_splits": list(LASERS_REFERRING_TEST_SPLITS),
                "metrics": suite_metrics,
            }
            args.output_dir.mkdir(parents=True, exist_ok=True)
            with (args.output_dir / "lasers_metrics_summary.json").open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(summary, file, indent=2)
        return
    if args.referring_split is not None:
        if REFERRING_TASK not in selected_tasks:
            raise ValueError(
                "--referring-split requires --task referring or an active "
                "referring task with --task all"
            )
        cfg["data"][REFERRING_TASK][f"{args.split}_split"] = str(args.referring_split)
    datasets = build_task_datasets(
        cfg["data"],
        logical_split=args.split,
        augment=False,
        tasks=selected_tasks,
    )
    all_metrics = {
        task: evaluate_task(
            task,
            accelerator,
            model,
            datasets[task],
            cfg,
            args,
        )
        for task in selected_tasks
    }
    if accelerator.is_main_process:
        task_ious = {}
        task_scores = {}
        for task in selected_tasks:
            iou_key = f"{task}_iou"
            if task != HEIGHT_TASK and iou_key in all_metrics[task]:
                task_ious[task] = float(all_metrics[task][iou_key])
            score_key = (
                "height_score"
                if task == HEIGHT_TASK
                else (
                    "detection_map50"
                    if task == DETECTION_TASK
                    else (
                        "sar_building_iou"
                        if task == SAR_BUILDING_TASK
                        else (
                            str(
                                cfg["evaluation"].get(
                                    "referring_best_metric",
                                    "referring_iou",
                                )
                            )
                            if task == REFERRING_TASK
                            else iou_key
                        )
                    )
                )
            )
            if score_key in all_metrics[task]:
                task_scores[task] = float(all_metrics[task][score_key])
        segmentation_mean_iou = (
            sum(task_ious.values()) / len(task_ious) if task_ious else None
        )
        active_task_mean_score = (
            sum(task_scores.values()) / len(task_scores) if task_scores else None
        )
        LOGGER.info(
            "Task IoUs: %s | scores: %s | active-task mean score %s",
            task_ious,
            task_scores,
            (
                "n/a"
                if active_task_mean_score is None
                else f"{active_task_mean_score:.4f}"
            ),
        )
        per_dataset_metrics: dict[str, dict[str, object]] = {}
        for task in selected_tasks:
            task_dataset_metrics = all_metrics[task].get(
                "per_dataset",
                {},
            )
            for source_name, source_payload in task_dataset_metrics.items():
                if source_name in per_dataset_metrics:
                    raise ValueError(
                        "Duplicate evaluation source name across tasks: "
                        f"{source_name!r}"
                    )
                per_dataset_metrics[source_name] = source_payload
        summary = {
            "checkpoint": str(checkpoint),
            "base_model": backbone.base_model,
            "model_variant": backbone.model_variant,
            "active_tasks": list(configured_tasks),
            "selected_tasks": list(selected_tasks),
            "segmentation_mean_iou": segmentation_mean_iou,
            "active_task_mean_score": active_task_mean_score,
            "per_dataset_metrics": per_dataset_metrics,
            "metrics": all_metrics,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / "metrics_summary.json").open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(summary, file, indent=2)
        with (args.output_dir / "per_dataset_metrics.json").open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(per_dataset_metrics, file, indent=2)
        per_dataset_dir = args.output_dir / "per_dataset_metrics"
        per_dataset_dir.mkdir(parents=True, exist_ok=True)
        for source_name, source_payload in per_dataset_metrics.items():
            source_summary = {
                "checkpoint": str(checkpoint),
                "base_model": backbone.base_model,
                "model_variant": backbone.model_variant,
                "source": source_name,
                **source_payload,
            }
            with (per_dataset_dir / f"{source_name}.json").open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(source_summary, file, indent=2)
        LOGGER.info(
            "Saved independent metrics for %d data sources: %s",
            len(per_dataset_metrics),
            ", ".join(per_dataset_metrics),
        )
