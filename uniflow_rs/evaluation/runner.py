"""Evaluation components: runner."""

from __future__ import annotations

import argparse
import json
import logging
import torch
from accelerate import (
    Accelerator,
)
from accelerate.utils import (
    gather_object,
)
from torch.utils.data import (
    DataLoader,
)
from tqdm.auto import (
    tqdm,
)
from uniflow_rs._compat.deventer_flow import (
    fixed_torch_seed,
    task_condition_modes,
    task_flow_sources,
)
from uniflow_rs._compat.dota import (
    DETECTION_TASK,
    DOTAFullImageStitcher,
    dota_batch_ground_truth,
    dota_batch_stitch_records,
)
from uniflow_rs._compat.multitask import (
    BUILDING_TASK,
    CHANGE_TASK,
    HEIGHT_TASK,
    HEIGHT_METRIC_COUNT_SIZE,
    LOVEDA_CODE_PALETTE_NAME,
    OPENEARTHMAP_SAR_DISPLAY_PALETTE,
    REFERRING_TASK,
    SAR_BUILDING_TASK,
    SAR_CHANGE_TASK,
    SAR_SEMANTIC_TASK,
    SEMANTIC_TASK,
    decoded_rgb_to_height,
    decode_loveda_palette_postprocessed,
    decode_urbansar_flood_palette,
    height_error_counts,
    loveda_confusion,
    loveda_metrics,
    openearthmap_sar_confusion,
    postprocess_binary_prediction,
    postprocess_height_prediction,
    urbansar_flood_confusion,
    spacenet_instance_counts,
)
from uniflow_rs.task_runtime import (
    detection_backend,
    task_metrics,
)
from uniflow_rs.evaluation.cli import (
    _height_postprocess_config,
)
from uniflow_rs.evaluation.height_outputs import (
    save_height_comparison,
    save_height_tile,
    stitch_height_predictions,
)
from uniflow_rs.evaluation.visualization import (
    save_prediction,
    save_referring_prediction,
    save_sar_change_prediction,
    save_semantic_prediction,
)

from uniflow_rs.evaluation.distributed import (
    collect_stitched_batch,
    real_batch_size,
    trim_evaluation_batch,
    validate_evaluation_sharding,
)
from uniflow_rs.model import UniFlowRS

LOGGER = logging.getLogger("uniflow_rs.evaluate")


# Tasks backed by a single dataset do not need SourceTaggedDataset, so their
# batches have no dataset_source field. Give them stable
# source identities here so standalone evaluation can expose the same
# per-dataset contract as multi-source tasks.
SINGLE_DATASET_SOURCE_BY_TASK = {
    BUILDING_TASK: "whu_building",
    SEMANTIC_TASK: "loveda",
    SAR_BUILDING_TASK: "spacenet6",
    SAR_CHANGE_TASK: "urbansar_floods",
    DETECTION_TASK: "dior_r",
}


def evaluation_source_names(cfg: dict, task: str) -> tuple[str, ...]:
    """Return stable dataset-source names for one evaluation task."""

    configured_sources = cfg.get("data", {}).get(task, {}).get("sources", {}) or {}
    if configured_sources:
        return tuple(str(name) for name in configured_sources)
    return (SINGLE_DATASET_SOURCE_BY_TASK.get(task, task),)


def referring_source_image_count(dataset) -> int:
    """Count physical referring images through composition wrappers."""

    children = getattr(dataset, "datasets", None)
    if children is not None:
        iterable = children.values() if isinstance(children, dict) else children
        return sum(referring_source_image_count(child) for child in iterable)

    child = getattr(dataset, "dataset", None)
    if child is not None and child is not dataset:
        return referring_source_image_count(child)

    samples = getattr(dataset, "samples", None)
    if samples is None:
        return len(dataset)

    source_images = {
        str(sample["image"])
        for sample in samples
        if isinstance(sample, dict) and "image" in sample
    }
    return len(source_images) if source_images else len(dataset)


def evaluate_task(
    task: str,
    accelerator: Accelerator,
    model: UniFlowRS,
    dataset,
    cfg: dict,
    args: argparse.Namespace,
) -> dict[str, object]:
    evaluation_split = str(getattr(dataset, "split", args.split))
    detection_has_ground_truth = not (
        task == DETECTION_TASK and not bool(getattr(dataset, "has_labels", True))
    )
    validate_evaluation_sharding(accelerator)
    batch_size = int(cfg["evaluation"].get("batch_size_per_gpu", 1))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(cfg["evaluation"].get("num_workers", 4)),
        pin_memory=True,
        drop_last=False,
    )
    loader = accelerator.prepare(loader)
    threshold = float(cfg["inference"].get("mask_threshold", 0.5))
    detection_impl = detection_backend(cfg)
    change_postprocess = dict(cfg["inference"].get("change_postprocess", {}) or {})
    semantic_code_palette = str(
        cfg["data"]
        .get(SEMANTIC_TASK, {})
        .get(
            "code_palette",
            LOVEDA_CODE_PALETTE_NAME,
        )
    )
    num_steps = int(
        cfg["inference"]["num_steps"] if args.num_steps is None else args.num_steps
    )
    counts = torch.zeros(
        (
            (
                int(detection_impl["class_count"]) + 1,
                int(detection_impl["class_count"]) + 1,
            )
            if task == DETECTION_TASK
            else (
                (7, 7)
                if task == SEMANTIC_TASK
                else (
                    (9, 9)
                    if task == SAR_SEMANTIC_TASK
                    else (
                        (3, 3)
                        if task == SAR_CHANGE_TASK
                        else (
                            HEIGHT_METRIC_COUNT_SIZE
                            if task == HEIGHT_TASK
                            else (
                                9
                                if task == SAR_BUILDING_TASK
                                else 11 if task == REFERRING_TASK else 6
                            )
                        )
                    )
                )
            )
        ),
        device=accelerator.device,
        dtype=torch.float64,
    )
    raw_semantic_counts = torch.zeros_like(counts) if task == SEMANTIC_TASK else None
    raw_change_counts = (
        torch.zeros_like(counts)
        if task == CHANGE_TASK and bool(change_postprocess.get("enabled", False))
        else None
    )
    raw_height_counts = torch.zeros_like(counts) if task == HEIGHT_TASK else None
    reported_modalities = (
        ("rgb", "sar") if task in {HEIGHT_TASK, SAR_BUILDING_TASK} else ()
    )
    modality_counts = {
        modality: torch.zeros_like(counts) for modality in reported_modalities
    }
    source_names = evaluation_source_names(cfg, task)
    source_counts = {
        source_name: torch.zeros_like(counts) for source_name in source_names
    }
    source_record_counts = {
        source_name: torch.zeros(
            (),
            device=accelerator.device,
            dtype=torch.int64,
        )
        for source_name in source_names
    }
    source_modality_counts = {
        source_name: {
            modality: torch.zeros_like(counts) for modality in reported_modalities
        }
        for source_name in source_names
    }
    source_modality_record_counts = {
        source_name: {
            modality: torch.zeros(
                (),
                device=accelerator.device,
                dtype=torch.int64,
            )
            for modality in reported_modalities
        }
        for source_name in source_names
    }
    output_dir = args.output_dir / task / evaluation_split
    validation_seed = int(
        cfg["evaluation"].get("validation_seed", 12345)
        if args.seed is None
        else args.seed
    )
    task_seed_offset = {
        BUILDING_TASK: 0,
        CHANGE_TASK: 100_000,
        SEMANTIC_TASK: 200_000,
        HEIGHT_TASK: 300_000,
        SAR_BUILDING_TASK: 400_000,
        SAR_SEMANTIC_TASK: 500_000,
        REFERRING_TASK: 600_000,
        SAR_CHANGE_TASK: 700_000,
        DETECTION_TASK: 800_000,
    }[task]
    local_detections: list[dict] = []
    local_detection_ground_truth: list[dict] = []
    detection_postprocess = dict(
        cfg["inference"].get("detection_postprocess", {}) or {}
    )
    detection_postprocess.setdefault(
        "class_count",
        int(detection_impl["class_count"]),
    )
    stitch_detection_images = task == DETECTION_TASK and bool(
        detection_postprocess.get("stitch_full_images", True)
    )
    detection_stitcher = (
        DOTAFullImageStitcher(detection_postprocess)
        if stitch_detection_images
        else None
    )
    height_max_meters = float(
        cfg["data"]
        .get(HEIGHT_TASK, {})
        .get(
            "max_height_meters",
            25.5,
        )
    )
    height_codec = str(
        cfg["data"]
        .get(HEIGHT_TASK, {})
        .get(
            "height_codec",
            "linear",
        )
    )
    height_postprocess = _height_postprocess_config(cfg, args)
    task_seed = validation_seed + task_seed_offset + accelerator.process_index
    with fixed_torch_seed(task_seed, accelerator.device):
        for batch_index, batch in enumerate(tqdm(
            loader,
            desc=f"Evaluating {task}/{evaluation_split}",
            disable=not accelerator.is_local_main_process,
        )):
            real_size = real_batch_size(
                len(dataset), batch_size, accelerator.num_processes,
                accelerator.process_index, batch_index,
            )
            if real_size == 0:
                # Do not infer, score or save repeated padding records. Detection
                # stitching still requires this rank to join both collectives.
                if stitch_detection_images:
                    detections, targets = collect_stitched_batch(
                        detection_stitcher, [],
                        [] if detection_has_ground_truth else None, gather_object,
                    )
                    local_detections.extend(detections)
                    local_detection_ground_truth.extend(targets)
                continue
            batch = trim_evaluation_batch(batch, real_size)
            before = batch["before"].to(accelerator.device, non_blocking=True)
            after = batch["after"].to(accelerator.device, non_blocking=True)
            target = batch["mask"].to(accelerator.device, non_blocking=True)
            valid_mask = batch["loss_mask"].to(
                accelerator.device,
                non_blocking=True,
            )
            batch_tasks = list(batch["task"])
            batch_sources = [str(value) for value in batch.get("dataset_source", ())]
            if not batch_sources and len(source_names) == 1:
                batch_sources = [source_names[0]] * before.shape[0]
            if len(batch_sources) != before.shape[0]:
                raise ValueError(
                    f"Task {task!r} evaluates sources {source_names}, but "
                    "the batch has missing dataset_source metadata"
                )
            unknown_batch_sources = sorted(set(batch_sources) - set(source_names))
            if unknown_batch_sources:
                raise ValueError(
                    f"Task {task!r} batch contains unknown data sources: "
                    f"{unknown_batch_sources}"
                )
            source_indices = {
                source_name: torch.as_tensor(
                    [
                        index
                        for index, value in enumerate(batch_sources)
                        if value == source_name
                    ],
                    device=accelerator.device,
                    dtype=torch.long,
                )
                for source_name in source_names
            }
            for source_name, indices in source_indices.items():
                source_record_counts[source_name] += indices.numel()
            batch_modalities = [
                str(value).strip().casefold()
                for value in batch.get("input_modality", ())
            ]
            if batch_modalities and len(batch_modalities) != before.shape[0]:
                raise ValueError(
                    f"Task {task!r} has inconsistent input_modality metadata"
                )
            modality_indices = {
                modality: torch.as_tensor(
                    [
                        index
                        for index, value in enumerate(batch_modalities)
                        if value == modality
                    ],
                    device=accelerator.device,
                    dtype=torch.long,
                )
                for modality in reported_modalities
            }
            source_modality_indices = {
                source_name: {
                    modality: torch.as_tensor(
                        [
                            index
                            for index, (source, selected_modality) in enumerate(
                                zip(batch_sources, batch_modalities)
                            )
                            if source == source_name and selected_modality == modality
                        ],
                        device=accelerator.device,
                        dtype=torch.long,
                    )
                    for modality in reported_modalities
                }
                for source_name in source_names
            }
            for source_name, by_modality in source_modality_indices.items():
                for modality, indices in by_modality.items():
                    source_modality_record_counts[source_name][
                        modality
                    ] += indices.numel()
            txt, vec, _ = model.text_condition(
                list(batch.get("prompt_key", batch_tasks)),
                accelerator.device,
                prompt_indices=batch.get("prompt_index"),
            )
            with torch.no_grad(), accelerator.autocast():
                if task == SAR_SEMANTIC_TASK:
                    decoded, prediction = model.predict_openearthmap(
                        before=before,
                        after=after,
                        txt=txt,
                        vec=vec,
                        batch_tasks=batch_tasks,
                        cfg=cfg,
                        num_steps=num_steps,
                        use_evaluation_temperature=True,
                    )
                else:
                    common_arguments = dict(
                        before=before,
                        after=after,
                        txt=txt,
                        vec=vec,
                        num_steps=num_steps,
                        resolution=int(cfg["data"]["resolution"]),
                        beta_alpha=float(cfg["flow"]["beta_alpha"]),
                        beta_beta=float(cfg["flow"]["beta_beta"]),
                        flow_source=task_flow_sources(
                            batch_tasks,
                            cfg["flow"],
                        ),
                        condition_mode=task_condition_modes(
                            batch_tasks,
                            cfg["flow"],
                        ),
                        latent_difference_scale=float(
                            cfg["flow"].get(
                                "latent_difference_scale",
                                1.0,
                            )
                        ),
                    )
                if task in {
                    DETECTION_TASK,
                    SEMANTIC_TASK,
                    HEIGHT_TASK,
                    SAR_CHANGE_TASK,
                }:
                    decoded = model.predict_target(**common_arguments)
                elif task != SAR_SEMANTIC_TASK:
                    probability = model.predict_probability(**common_arguments)
            if task == DETECTION_TASK:
                prediction, detection_probabilities = detection_impl["decode"](
                    decoded,
                    float(
                        cfg["inference"]
                        .get("detection_postprocess", {})
                        .get("palette_temperature", 0.1)
                    ),
                    query_classes=batch.get("detection_query_class"),
                    mask_threshold=threshold,
                )
                if detection_has_ground_truth:
                    counts += detection_impl["confusion"](
                        prediction,
                        target[:, 0],
                        valid_mask,
                    )
                    for source_name, indices in source_indices.items():
                        if indices.numel() > 0:
                            source_counts[source_name] += detection_impl["confusion"](
                                prediction.index_select(0, indices),
                                target[:, 0].index_select(0, indices),
                                valid_mask.index_select(0, indices),
                            )
                if stitch_detection_images:
                    assert detection_stitcher is not None
                    detections, targets = collect_stitched_batch(
                        detection_stitcher,
                        dota_batch_stitch_records(
                            prediction,
                            detection_probabilities,
                            batch,
                        ),
                        dota_batch_ground_truth(batch) if detection_has_ground_truth else None,
                        gather_object,
                    )
                    local_detections.extend(detections)
                    local_detection_ground_truth.extend(targets)
                else:
                    local_detections.extend(
                        detection_impl["batch_detections"](
                            prediction,
                            detection_probabilities,
                            batch,
                            detection_postprocess,
                        )
                    )
                    if detection_has_ground_truth:
                        local_detection_ground_truth.extend(
                            dota_batch_ground_truth(batch)
                        )
                if not args.metrics_only:
                    for index, name in enumerate(batch["name"]):
                        save_semantic_prediction(
                            output_dir,
                            str(name),
                            after[index],
                            decoded[index],
                            prediction[index],
                            prediction[index],
                            target[index, 0],
                            palette=detection_impl["display_palette"],
                            # DIOR-R's resized protocol has no padded image
                            # region: label 0 is the real background.  Do not
                            # mask it with GT, otherwise false-positive
                            # detections disappear from saved qualitative
                            # predictions even though metrics still count
                            # them.
                            ignore_zero=False,
                            zero_label_name="background",
                        )
            elif task == SEMANTIC_TASK:
                raw_prediction, prediction = decode_loveda_palette_postprocessed(
                    decoded,
                    cfg["inference"].get(
                        "semantic_postprocess",
                        {},
                    ),
                    code_palette=semantic_code_palette,
                )
                assert raw_semantic_counts is not None
                raw_semantic_counts += loveda_confusion(
                    raw_prediction,
                    target[:, 0],
                )
                counts += loveda_confusion(prediction, target[:, 0])
                for source_name, indices in source_indices.items():
                    if indices.numel() > 0:
                        source_counts[source_name] += loveda_confusion(
                            prediction.index_select(0, indices),
                            target[:, 0].index_select(0, indices),
                        )
                if not args.metrics_only:
                    for index, name in enumerate(batch["name"]):
                        save_semantic_prediction(
                            output_dir,
                            str(name),
                            after[index],
                            decoded[index],
                            raw_prediction[index],
                            prediction[index],
                            target[index, 0],
                        )
            elif task == SAR_SEMANTIC_TASK:
                counts += openearthmap_sar_confusion(
                    prediction,
                    target[:, 0],
                )
                for source_name, indices in source_indices.items():
                    if indices.numel() > 0:
                        source_counts[source_name] += openearthmap_sar_confusion(
                            prediction.index_select(0, indices),
                            target[:, 0].index_select(0, indices),
                        )
                if not args.metrics_only:
                    for index, name in enumerate(batch["name"]):
                        save_semantic_prediction(
                            output_dir,
                            str(name),
                            after[index],
                            decoded[index],
                            prediction[index],
                            prediction[index],
                            target[index, 0],
                            # Decode with the VAE codebook, but present the
                            # benchmark's familiar official class colors.
                            palette=OPENEARTHMAP_SAR_DISPLAY_PALETTE,
                            ignore_zero=False,
                            zero_label_name="background",
                        )
            elif task == SAR_CHANGE_TASK:
                prediction = decode_urbansar_flood_palette(decoded)
                sar_change_counts = urbansar_flood_confusion(
                    prediction,
                    target[:, 0],
                    valid_mask[:, 0],
                )
                counts += sar_change_counts
                for source_name, indices in source_indices.items():
                    if indices.numel() > 0:
                        source_counts[source_name] += urbansar_flood_confusion(
                            prediction.index_select(0, indices),
                            target[:, 0].index_select(0, indices),
                            valid_mask[:, 0].index_select(0, indices),
                        )
                if not args.metrics_only:
                    for index, name in enumerate(batch["name"]):
                        save_sar_change_prediction(
                            output_dir,
                            str(name),
                            before[index],
                            after[index],
                            decoded[index],
                            prediction[index],
                            target[index, 0],
                        )
            elif task == HEIGHT_TASK:
                raw_prediction = decoded_rgb_to_height(
                    decoded,
                    max_height_meters=height_max_meters,
                    height_codec=height_codec,
                )
                assert raw_height_counts is not None
                raw_height_counts += height_error_counts(
                    raw_prediction,
                    target,
                    valid_mask=valid_mask,
                )
                prediction = postprocess_height_prediction(
                    raw_prediction,
                    height_postprocess,
                    max_height_meters=height_max_meters,
                )
                counts += height_error_counts(
                    prediction,
                    target,
                    valid_mask=valid_mask,
                )
                for source_name, indices in source_indices.items():
                    if indices.numel() > 0:
                        source_counts[source_name] += height_error_counts(
                            prediction.index_select(0, indices),
                            target.index_select(0, indices),
                            valid_mask=valid_mask.index_select(0, indices),
                        )
                for modality, indices in modality_indices.items():
                    if indices.numel() > 0:
                        modality_counts[modality] += height_error_counts(
                            prediction.index_select(0, indices),
                            target.index_select(0, indices),
                            valid_mask=valid_mask.index_select(0, indices),
                        )
                for source_name, by_modality in source_modality_indices.items():
                    for modality, indices in by_modality.items():
                        if indices.numel() > 0:
                            source_modality_counts[source_name][
                                modality
                            ] += height_error_counts(
                                prediction.index_select(0, indices),
                                target.index_select(0, indices),
                                valid_mask=valid_mask.index_select(
                                    0,
                                    indices,
                                ),
                            )
                if not args.metrics_only:
                    for index, name in enumerate(batch["name"]):
                        save_height_tile(
                            output_dir,
                            str(name),
                            prediction[index, 0],
                            height_max_meters,
                        )
                        save_height_comparison(
                            output_dir,
                            str(name),
                            before[index],
                            after[index],
                            prediction[index, 0],
                            target[index, 0],
                            height_max_meters,
                            raw_prediction=raw_prediction[index, 0],
                        )
            else:
                valid = valid_mask[:, 0] >= 0.5
                raw_prediction = (probability[:, 0] >= threshold) & valid
                prediction = (
                    postprocess_binary_prediction(
                        raw_prediction,
                        valid,
                        change_postprocess,
                    )
                    if task == CHANGE_TASK
                    else raw_prediction
                )
                display_probability = probability[:, 0] * valid
                truth = target[:, 0] >= 0.5
                binary_counts = torch.stack(
                    (
                        (prediction & truth & valid).sum(),
                        (prediction & ~truth & valid).sum(),
                        (~prediction & truth & valid).sum(),
                        (~prediction & ~truth & valid).sum(),
                        ((probability[:, 0] - target[:, 0]).abs() * valid).sum(),
                        valid.sum(),
                    )
                ).to(torch.float64)
                counts[:6] += binary_counts
                for source_name, indices in source_indices.items():
                    if indices.numel() < 1:
                        continue
                    source_prediction = prediction.index_select(0, indices)
                    source_truth = truth.index_select(0, indices)
                    source_valid = valid.index_select(0, indices)
                    source_probability = probability[:, 0].index_select(
                        0,
                        indices,
                    )
                    source_target = target[:, 0].index_select(0, indices)
                    source_counts[source_name][:6] += torch.stack(
                        (
                            (source_prediction & source_truth & source_valid).sum(),
                            (source_prediction & ~source_truth & source_valid).sum(),
                            (~source_prediction & source_truth & source_valid).sum(),
                            (~source_prediction & ~source_truth & source_valid).sum(),
                            (
                                (source_probability - source_target).abs()
                                * source_valid
                            ).sum(),
                            source_valid.sum(),
                        )
                    ).to(torch.float64)
                if raw_change_counts is not None:
                    raw_change_counts[:6] += torch.stack(
                        (
                            (raw_prediction & truth & valid).sum(),
                            (raw_prediction & ~truth & valid).sum(),
                            (~raw_prediction & truth & valid).sum(),
                            (~raw_prediction & ~truth & valid).sum(),
                            ((probability[:, 0] - target[:, 0]).abs() * valid).sum(),
                            valid.sum(),
                        )
                    ).to(torch.float64)
                if task == REFERRING_TASK:
                    per_sample_intersection = (
                        (prediction & truth & valid).flatten(1).sum(dim=1).double()
                    )
                    per_sample_union = (
                        ((prediction | truth) & valid).flatten(1).sum(dim=1).double()
                    )
                    per_sample_iou = (
                        per_sample_intersection / per_sample_union.clamp_min(1.0)
                    )
                    counts[6] += per_sample_iou.sum()
                    counts[7] += per_sample_iou.numel()
                    counts[8] += (per_sample_iou >= 0.5).sum()
                    counts[9] += (per_sample_iou >= 0.7).sum()
                    counts[10] += (per_sample_iou >= 0.9).sum()
                    for source_name, indices in source_indices.items():
                        if indices.numel() < 1:
                            continue
                        source_iou = per_sample_iou.index_select(0, indices)
                        source_counts[source_name][6] += source_iou.sum()
                        source_counts[source_name][7] += source_iou.numel()
                        source_counts[source_name][8] += (source_iou >= 0.5).sum()
                        source_counts[source_name][9] += (source_iou >= 0.7).sum()
                        source_counts[source_name][10] += (source_iou >= 0.9).sum()
                if task == SAR_BUILDING_TASK:
                    instance_mask = batch["instance_mask"].to(
                        accelerator.device,
                        non_blocking=True,
                    )
                    counts[6:9] += spacenet_instance_counts(
                        prediction,
                        instance_mask,
                        valid_mask=valid_mask,
                    )
                    for source_name, indices in source_indices.items():
                        if indices.numel() > 0:
                            source_counts[source_name][6:9] += spacenet_instance_counts(
                                prediction.index_select(0, indices),
                                instance_mask.index_select(0, indices),
                                valid_mask=valid_mask.index_select(
                                    0,
                                    indices,
                                ),
                            )
                    for modality, indices in modality_indices.items():
                        if indices.numel() < 1:
                            continue
                        modality_prediction = prediction.index_select(0, indices)
                        modality_truth = truth.index_select(0, indices)
                        modality_valid = valid.index_select(0, indices)
                        modality_probability = probability[:, 0].index_select(
                            0,
                            indices,
                        )
                        modality_target = target[:, 0].index_select(0, indices)
                        modality_counts[modality][:6] += torch.stack(
                            (
                                (
                                    modality_prediction
                                    & modality_truth
                                    & modality_valid
                                ).sum(),
                                (
                                    modality_prediction
                                    & ~modality_truth
                                    & modality_valid
                                ).sum(),
                                (
                                    ~modality_prediction
                                    & modality_truth
                                    & modality_valid
                                ).sum(),
                                (
                                    ~modality_prediction
                                    & ~modality_truth
                                    & modality_valid
                                ).sum(),
                                (
                                    (modality_probability - modality_target).abs()
                                    * modality_valid
                                ).sum(),
                                modality_valid.sum(),
                            )
                        ).to(torch.float64)
                        modality_counts[modality][6:9] += spacenet_instance_counts(
                            modality_prediction,
                            instance_mask.index_select(0, indices),
                            valid_mask=valid_mask.index_select(0, indices),
                        )
                    for source_name, by_modality in source_modality_indices.items():
                        for modality, indices in by_modality.items():
                            if indices.numel() < 1:
                                continue
                            selected_prediction = prediction.index_select(
                                0,
                                indices,
                            )
                            selected_truth = truth.index_select(0, indices)
                            selected_valid = valid.index_select(0, indices)
                            selected_probability = probability[:, 0].index_select(
                                0, indices
                            )
                            selected_target = target[:, 0].index_select(
                                0,
                                indices,
                            )
                            selected_counts = source_modality_counts[source_name][
                                modality
                            ]
                            selected_counts[:6] += torch.stack(
                                (
                                    (
                                        selected_prediction
                                        & selected_truth
                                        & selected_valid
                                    ).sum(),
                                    (
                                        selected_prediction
                                        & ~selected_truth
                                        & selected_valid
                                    ).sum(),
                                    (
                                        ~selected_prediction
                                        & selected_truth
                                        & selected_valid
                                    ).sum(),
                                    (
                                        ~selected_prediction
                                        & ~selected_truth
                                        & selected_valid
                                    ).sum(),
                                    (
                                        (selected_probability - selected_target).abs()
                                        * selected_valid
                                    ).sum(),
                                    selected_valid.sum(),
                                )
                            ).to(torch.float64)
                            selected_counts[6:9] += spacenet_instance_counts(
                                selected_prediction,
                                instance_mask.index_select(0, indices),
                                valid_mask=valid_mask.index_select(0, indices),
                            )
                if not args.metrics_only:
                    for index, name in enumerate(batch["name"]):
                        if task == REFERRING_TASK:
                            save_referring_prediction(
                                output_dir,
                                str(name),
                                after[index],
                                display_probability[index],
                                target[index, 0],
                                threshold,
                                str(batch["expression"][index]),
                            )
                        else:
                            save_prediction(
                                output_dir,
                                str(name),
                                before[index],
                                after[index],
                                display_probability[index],
                                target[index, 0],
                                threshold,
                                prediction=(
                                    prediction[index] if task == CHANGE_TASK else None
                                ),
                                raw_prediction=(
                                    raw_prediction[index]
                                    if task == CHANGE_TASK
                                    else None
                                ),
                            )
    can_stitch_height = getattr(dataset, "samples", None) is not None and all(
        "parent" in sample and "row" in sample and "column" in sample
        for sample in getattr(dataset, "samples", ())
    )
    if task == HEIGHT_TASK and not args.metrics_only and can_stitch_height:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            stitch_height_predictions(
                output_dir,
                dataset,
                max_height_meters=height_max_meters,
            )
        accelerator.wait_for_everyone()
    counts = accelerator.reduce(counts, reduction="sum")
    if detection_stitcher is not None:
        local_detections.extend(detection_stitcher.finish())
    aggregate_task_metrics = (
        {}
        if task == DETECTION_TASK and not detection_has_ground_truth
        else task_metrics(
            task,
            counts,
            height_max_meters=height_max_meters,
        )
    )
    metrics: dict[str, object] = dict(aggregate_task_metrics)
    if task == DETECTION_TASK:
        all_detection_predictions = (
            local_detections
            if stitch_detection_images
            else gather_object(local_detections)
        )
        all_detection_ground_truth = (
            local_detection_ground_truth
            if stitch_detection_images
            else gather_object(local_detection_ground_truth)
        )
        metrics.update(
            {
                "detection_has_ground_truth": detection_has_ground_truth,
                "detection_predictions_before_submission_nms": len(
                    all_detection_predictions
                ),
            }
        )
        if detection_has_ground_truth:
            metrics.update(
                detection_impl["oriented_metrics"](
                    all_detection_predictions,
                    all_detection_ground_truth,
                    nms_threshold=float(
                        detection_postprocess.get("nms_iou_threshold", 0.1)
                    ),
                )
            )
        if accelerator.is_main_process and (
            args.detection_submission_dir is not None or not detection_has_ground_truth
        ):
            submission_dir = (
                args.detection_submission_dir
                if args.detection_submission_dir is not None
                else output_dir / "task1_submission"
            )
            metrics.update(
                detection_impl["write_submission"](
                    all_detection_predictions,
                    submission_dir,
                    image_names=getattr(dataset, "parent_names", ()),
                    nms_threshold=float(
                        detection_postprocess.get("nms_iou_threshold", 0.1)
                    ),
                )
            )
    if task == SAR_BUILDING_TASK:
        metrics.update(
            {
                name.replace(
                    "sar_building_",
                    "sar_building_combined_",
                    1,
                ): value
                for name, value in aggregate_task_metrics.items()
            }
        )
    modality_metrics_by_name = {}
    for modality, modality_task_counts in modality_counts.items():
        modality_task_counts = accelerator.reduce(
            modality_task_counts,
            reduction="sum",
        )
        modality_metrics = task_metrics(
            task,
            modality_task_counts,
            height_max_meters=height_max_meters,
        )
        modality_metrics_by_name[modality] = modality_metrics
        metrics.update(
            {
                f"modality/{modality}/{name}": value
                for name, value in modality_metrics.items()
            }
        )
        if task == SAR_BUILDING_TASK:
            for name, value in modality_metrics.items():
                suffix = name.removeprefix("sar_building_")
                metrics[f"sar_building_{modality}_{suffix}"] = value
        elif task == HEIGHT_TASK:
            for name, value in modality_metrics.items():
                suffix = name.removeprefix("height_")
                metrics[f"height_{modality}_{suffix}"] = value
    if task == SAR_BUILDING_TASK:
        # Keep legacy/top-level metrics comparable to the official SAR-only
        # protocol while retaining the RGB and combined diagnostics.
        metrics.update(modality_metrics_by_name["sar"])
    if raw_height_counts is not None:
        raw_height_counts = accelerator.reduce(
            raw_height_counts,
            reduction="sum",
        )
        raw_height_metrics = task_metrics(
            HEIGHT_TASK,
            raw_height_counts,
            height_max_meters=height_max_meters,
        )
        metrics.update(
            {
                name.replace("height_", "height_raw_", 1): value
                for name, value in raw_height_metrics.items()
            }
        )
    if raw_semantic_counts is not None:
        raw_semantic_counts = accelerator.reduce(
            raw_semantic_counts,
            reduction="sum",
        )
        raw_metrics = loveda_metrics(raw_semantic_counts)
        metrics.update(
            {
                name.replace("semantic_", "semantic_raw_", 1): value
                for name, value in raw_metrics.items()
            }
        )
    if raw_change_counts is not None:
        raw_change_counts = accelerator.reduce(
            raw_change_counts,
            reduction="sum",
        )
        raw_metrics = task_metrics(CHANGE_TASK, raw_change_counts)
        metrics.update(
            {
                name.replace("change_", "change_raw_", 1): value
                for name, value in raw_metrics.items()
            }
        )
    per_dataset_metrics: dict[str, dict[str, object]] = {}
    for source_name in source_names:
        source_task_counts = accelerator.reduce(
            source_counts[source_name],
            reduction="sum",
        )
        source_records_tensor = accelerator.reduce(
            source_record_counts[source_name],
            reduction="sum",
        )
        source_records = int(source_records_tensor.item())
        if source_records < 1:
            continue

        # A singleton source is exactly the aggregate task evaluation, so
        # retain all additional protocol metrics (e.g. DIOR-R oriented AP,
        # LoveDA raw/post metrics, and SpaceNet6 RGB/SAR aliases).  Multi-source
        # tasks are recomputed from their own sufficient statistics.
        if len(source_names) == 1:
            source_metrics = dict(metrics)
        else:
            source_metrics = task_metrics(
                task,
                source_task_counts,
                height_max_meters=height_max_meters,
            )

        source_modalities: dict[str, dict[str, object]] = {}
        for modality in reported_modalities:
            selected_counts = accelerator.reduce(
                source_modality_counts[source_name][modality],
                reduction="sum",
            )
            selected_records_tensor = accelerator.reduce(
                source_modality_record_counts[source_name][modality],
                reduction="sum",
            )
            selected_records = int(selected_records_tensor.item())
            if selected_records < 1:
                continue
            selected_metrics = task_metrics(
                task,
                selected_counts,
                height_max_meters=height_max_meters,
            )
            source_modalities[modality] = {
                "records": selected_records,
                "metrics": selected_metrics,
            }
            for name, value in selected_metrics.items():
                metrics[f"source/{source_name}/modality/{modality}/{name}"] = value
                if task == SAR_BUILDING_TASK:
                    suffix = name.removeprefix("sar_building_")
                    metrics[
                        f"source/{source_name}/" f"sar_building_{modality}_{suffix}"
                    ] = value
                elif task == HEIGHT_TASK:
                    suffix = name.removeprefix("height_")
                    metrics[f"source/{source_name}/" f"height_{modality}_{suffix}"] = (
                        value
                    )

        per_dataset_metrics[source_name] = {
            "task": task,
            "split": evaluation_split,
            "records": source_records,
            "metrics": source_metrics,
            "modalities": source_modalities,
        }
        for name, value in source_metrics.items():
            metrics[f"source/{source_name}/{name}"] = value

    metrics["per_dataset"] = per_dataset_metrics
    evaluated_records = sum(item["records"] for item in per_dataset_metrics.values())
    if evaluated_records != len(dataset):
        raise RuntimeError(
            f"Evaluation counted {evaluated_records} records, expected {len(dataset)}"
        )
    metrics.update(
        {
            "task": task,
            "dataset": getattr(dataset, "dataset_name", None),
            "split": evaluation_split,
            "images": len(dataset),
            "evaluated_records": evaluated_records,
            "patches": len(dataset) if task == HEIGHT_TASK else None,
            "source_images": (
                len({str(sample["parent"]) for sample in dataset.samples})
                if task == HEIGHT_TASK and can_stitch_height
                else (
                    len(dataset)
                    if task == HEIGHT_TASK
                    else (
                        referring_source_image_count(dataset)
                        if task == REFERRING_TASK
                        else (
                            len(getattr(dataset, "parent_names", ()))
                            if task == DETECTION_TASK
                            else len(getattr(dataset, "samples", dataset))
                        )
                    )
                )
            ),
            "num_steps": num_steps,
            "seed": validation_seed,
            "semantic_code_palette": (
                semantic_code_palette if task == SEMANTIC_TASK else None
            ),
            "height_postprocess": (height_postprocess if task == HEIGHT_TASK else None),
            "change_postprocess": (change_postprocess if task == CHANGE_TASK else None),
        }
    )
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2)
        if task == HEIGHT_TASK:
            LOGGER.info(
                "Height raw -> post | MAE %.3f -> %.3f m | "
                "RMSE %.3f -> %.3f m | Delta1 %.2f -> %.2f%% | "
                "post Delta2/3 %.2f/%.2f%% | "
                "above-ground Delta1 %.2f%%",
                metrics["height_raw_mae"],
                metrics["height_mae"],
                metrics["height_raw_rmse"],
                metrics["height_rmse"],
                metrics["height_raw_delta1_percent"],
                metrics["height_delta1_percent"],
                metrics["height_delta2_percent"],
                metrics["height_delta3_percent"],
                metrics["height_above_ground_delta1_percent"],
            )
            if "height_rgb_delta1_percent" in metrics:
                LOGGER.info(
                    "Height by modality | RGB delta1 %.2f%% | " "SAR delta1 %.2f%%",
                    metrics["height_rgb_delta1_percent"],
                    metrics["height_sar_delta1_percent"],
                )
        elif task == SAR_BUILDING_TASK:
            LOGGER.info(
                "SpaceNet6 by modality | RGB instance F1 %.4f raster IoU "
                "%.4f | SAR instance F1 %.4f raster IoU %.4f",
                metrics["sar_building_rgb_instance_f1"],
                metrics["sar_building_rgb_iou"],
                metrics["sar_building_sar_instance_f1"],
                metrics["sar_building_sar_iou"],
            )
        elif task == DETECTION_TASK:
            if detection_has_ground_truth:
                LOGGER.info(
                    "Oriented-detection all-class validation | mask mIoU %.4f | "
                    "oriented mAP50 %.4f mAP50:95 %.4f | geometry "
                    "P/R/F1@0.5 %.4f/%.4f/%.4f | "
                    "P/R/F1@0.75 %.4f/%.4f/%.4f | "
                    "matched IoU@0.5 %.4f | merge/split %.4f/%.4f",
                    metrics["detection_miou"],
                    metrics["detection_map50"],
                    metrics["detection_map50_95"],
                    metrics["detection_geometry_precision_at_0_5"],
                    metrics["detection_geometry_recall_at_0_5"],
                    metrics["detection_geometry_f1_at_0_5"],
                    metrics["detection_geometry_precision_at_0_75"],
                    metrics["detection_geometry_recall_at_0_75"],
                    metrics["detection_geometry_f1_at_0_75"],
                    metrics["detection_geometry_matched_mean_iou_at_0_5"],
                    metrics["detection_instance_merge_rate"],
                    metrics["detection_instance_split_rate"],
                )
            else:
                LOGGER.info(
                    "Unlabeled oriented-detection test | exported %d "
                    "predictions to %s",
                    metrics.get("detection_submission_predictions", 0),
                    metrics.get("detection_submission_zip", "n/a"),
                )
    return metrics
