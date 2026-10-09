"""Task backends, metrics, and prediction helpers shared by train and eval."""

from __future__ import annotations


import torch

from uniflow_rs._compat.deventer import binary_metrics
from uniflow_rs._compat.deventer_flow import (
    predict_multitask_decoded,
    predict_multitask_palette_sliding,
    task_condition_modes,
    task_flow_sources,
)
from uniflow_rs._compat.dior import (
    DIOR_BINARY_DISPLAY_PALETTE,
    DIOR_CODE_PALETTE,
    DIOR_CLASS_COUNT,
    DIOR_CLASS_NAMES,
    decode_dior_palette,
    dior_batch_detections,
    dior_confusion,
    dior_oriented_metrics,
    dior_raster_metrics,
    load_dior_code_palette,
    write_dior_task1_submission,
)
from uniflow_rs._compat.dota import (
    DETECTION_TASK,
)
from uniflow_rs._compat.multitask import (
    BUILDING_TASK,
    CHANGE_TASK,
    HEIGHT_TASK,
    OPENEARTHMAP_SAR_CODE_PALETTE_NAME,
    REFERRING_TASK,
    SAR_BUILDING_TASK,
    SAR_CHANGE_TASK,
    SAR_SEMANTIC_TASK,
    SEMANTIC_TASK,
    URBANSAR_FLOOD_REGIONS,
    decode_openearthmap_sar_palette,
    height_metrics,
    loveda_metrics,
    openearthmap_sar_metrics,
    resolve_openearthmap_sar_code_palette,
    urbansar_flood_benchmark_metrics,
    urbansar_flood_metrics,
    urbansar_flood_official_metrics,
)


def detection_backend(cfg: dict) -> dict[str, object]:
    """Resolve the configured oriented-detection implementation."""

    detection_cfg = cfg.get("data", {}).get(DETECTION_TASK, {})
    dataset_format = str(detection_cfg.get("dataset_format", "dior")).strip().casefold()
    if dataset_format == "dior":
        target_mode = str(detection_cfg.get("target_mode", "binary")).strip().casefold()
        if target_mode not in {"binary", "palette"}:
            raise ValueError(
                "data.detection.target_mode must be 'binary' or 'palette', "
                f"got {target_mode!r}"
            )
        code_palette = (
            load_dior_code_palette(detection_cfg["code_palette_file"])
            if detection_cfg.get("code_palette_file")
            else DIOR_CODE_PALETTE
        )

        def decode(
            decoded,
            temperature=0.1,
            query_classes=None,
            mask_threshold=0.5,
        ):
            return decode_dior_palette(
                decoded,
                temperature=temperature,
                query_classes=query_classes,
                mask_threshold=mask_threshold,
                palette=code_palette,
            )

        return {
            "format": dataset_format,
            "target_mode": target_mode,
            "class_count": DIOR_CLASS_COUNT,
            "class_names": DIOR_CLASS_NAMES,
            "decode": decode,
            "batch_detections": dior_batch_detections,
            "confusion": dior_confusion,
            "oriented_metrics": dior_oriented_metrics,
            "raster_metrics": dior_raster_metrics,
            "display_palette": (
                code_palette
                if target_mode == "palette"
                else DIOR_BINARY_DISPLAY_PALETTE
            ),
            "write_submission": write_dior_task1_submission,
        }
    raise ValueError(
        "This release supports data.detection.dataset_format='dior' only, got "
        f"{dataset_format!r}"
    )


def task_metrics(
    task: str,
    counts: torch.Tensor,
    height_max_meters: float = 25.5,
) -> dict[str, float]:
    if task == DETECTION_TASK:
        if counts.shape[-1] != DIOR_CLASS_COUNT + 1:
            raise ValueError("DIOR-R confusion must contain background plus 20 classes")
        return dior_raster_metrics(counts)
    if task == SEMANTIC_TASK:
        return loveda_metrics(counts)
    if task == SAR_SEMANTIC_TASK:
        return openearthmap_sar_metrics(counts)
    if task == SAR_CHANGE_TASK:
        if counts.ndim == 2:
            # Compatibility with validation on the legacy train/val split,
            # which has category folders rather than official test regions.
            metrics = urbansar_flood_metrics(counts)
            empty_regions = torch.zeros(
                (len(URBANSAR_FLOOD_REGIONS), 3, 3),
                dtype=counts.dtype,
                device=counts.device,
            )
            metrics.update(urbansar_flood_official_metrics(empty_regions))
            metrics["sar_change_official_macro_f1"] = 0.5 * (
                metrics["sar_change_open_flood_dice"]
                + metrics["sar_change_urban_flood_dice"]
            )
            metrics["sar_change_official_regions"] = 0.0
            return metrics
        if tuple(counts.shape) != (
            len(URBANSAR_FLOOD_REGIONS) + 1,
            3,
            3,
        ):
            raise ValueError(
                "UrbanSARFloods validation counts must contain one global "
                "and three region confusion matrices, got "
                f"{tuple(counts.shape)}"
            )
        metrics = urbansar_flood_metrics(counts[0])
        # The disaster-mapping benchmark reports the arithmetic mean of the
        # three class IoUs (NF/FO/FU) from the accumulated global confusion
        # matrix.  Keep this as an explicit field, and also expose the
        # unweighted mean of the three region-level mIoUs as a diagnostic.
        metrics.update(
            urbansar_flood_benchmark_metrics(
                counts[0],
                counts[1:],
            )
        )
        metrics.update(urbansar_flood_official_metrics(counts[1:]))
        return metrics
    if task == HEIGHT_TASK:
        return height_metrics(
            counts,
            max_height_meters=height_max_meters,
        )
    # SpaceNet appends instance TP/FP/FN after the six raster counters.
    generic = binary_metrics(counts[:6].detach().cpu())
    prefixes = {
        BUILDING_TASK: "building",
        CHANGE_TASK: "change",
        SAR_BUILDING_TASK: "sar_building",
        REFERRING_TASK: "referring",
    }
    if task not in prefixes:
        raise ValueError(f"Unsupported binary task: {task}")
    prefix = prefixes[task]
    tp, fp, fn = (float(counts[index]) for index in range(3))
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    metrics = {
        f"{prefix}_iou": generic["building_iou"],
        f"{prefix}_dice": generic["building_dice"],
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_background_iou": generic["background_iou"],
        f"{prefix}_mean_iou": generic["mean_iou"],
        f"{prefix}_pixel_accuracy": generic["pixel_accuracy"],
        f"{prefix}_mask_mae": generic["mask_mae"],
    }
    if task == REFERRING_TASK:
        sample_count = max(float(counts[7]), 1.0)
        referring_giou = float(counts[6]) / sample_count
        referring_ciou = generic["building_iou"]
        metrics.update(
            {
                # LaSeRS/SegEarth-R2 terminology: gIoU averages the IoU of
                # each query, whereas cIoU accumulates intersections and
                # unions over the complete evaluation set. Keep the older
                # RRSIS-D names as exact aliases for checkpoint compatibility.
                "referring_giou": referring_giou,
                "referring_ciou": referring_ciou,
                "referring_overall_iou": referring_ciou,
                "referring_mean_iou": referring_giou,
                "referring_precision_at_0_5": (float(counts[8]) / sample_count),
                "referring_precision_at_0_7": (float(counts[9]) / sample_count),
                "referring_precision_at_0_9": (float(counts[10]) / sample_count),
            }
        )
    if task == SAR_BUILDING_TASK:
        instance_tp, instance_fp, instance_fn = (
            float(counts[index]) for index in range(6, 9)
        )
        instance_precision = instance_tp / max(
            instance_tp + instance_fp,
            1.0,
        )
        instance_recall = instance_tp / max(
            instance_tp + instance_fn,
            1.0,
        )
        metrics.update(
            {
                "sar_building_instance_precision": instance_precision,
                "sar_building_instance_recall": instance_recall,
                "sar_building_instance_f1": (
                    2.0
                    * instance_precision
                    * instance_recall
                    / max(instance_precision + instance_recall, 1.0e-12)
                ),
                "sar_building_instance_iou_threshold": 0.5,
            }
        )
    return metrics


@torch.no_grad()
def predict_sar_semantic_output(
    transformer,
    autoencoder,
    before: torch.Tensor,
    after: torch.Tensor,
    txt: torch.Tensor,
    vec: torch.Tensor,
    batch_tasks: list[str],
    cfg: dict,
    num_steps: int,
    *,
    use_evaluation_temperature: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict OpenEarthMap-SAR at native size with overlap blending.

    The evaluation wrapper opts into its existing evaluation-config fallback;
    training validation keeps its historical 0.1 fallback. An explicit sliding
    temperature has precedence in either path.
    """

    resolution = int(
        cfg["data"]
        .get(SAR_SEMANTIC_TASK, {})
        .get(
            "resolution",
            cfg["data"]["resolution"],
        )
    )
    flow_sources = task_flow_sources(batch_tasks, cfg["flow"])
    condition_modes = task_condition_modes(batch_tasks, cfg["flow"])
    common_arguments = dict(
        transformer=transformer,
        autoencoder=autoencoder,
        before=before,
        after=after,
        txt=txt,
        vec=vec,
        num_steps=int(num_steps),
        beta_alpha=float(cfg["flow"]["beta_alpha"]),
        beta_beta=float(cfg["flow"]["beta_beta"]),
        flow_source=flow_sources,
        condition_mode=condition_modes,
        latent_difference_scale=float(cfg["flow"].get("latent_difference_scale", 1.0)),
    )
    sar_code_palette_name = str(
        cfg["data"]
        .get(SAR_SEMANTIC_TASK, {})
        .get(
            "code_palette",
            OPENEARTHMAP_SAR_CODE_PALETTE_NAME,
        )
    )
    sar_code_palette = resolve_openearthmap_sar_code_palette(sar_code_palette_name)
    if after.shape[-2:] == (resolution, resolution):
        decoded = predict_multitask_decoded(
            **common_arguments,
            resolution=resolution,
        )
        return decoded, decode_openearthmap_sar_palette(
            decoded,
            code_palette=sar_code_palette_name,
        )

    sliding_cfg = dict(cfg["inference"].get("sar_semantic_sliding_window", {}))
    if not bool(sliding_cfg.get("enabled", True)):
        raise ValueError(
            "OpenEarthMap-SAR evaluation images exceed the model resolution; "
            "enable inference.sar_semantic_sliding_window"
        )
    stride = int(sliding_cfg.get("stride", resolution // 2))
    palette_temperature = float(
        sliding_cfg.get(
            "palette_temperature",
            (
                cfg["evaluation"].get("palette_temperature", 0.1)
                if use_evaluation_temperature
                else 0.1
            ),
        )
    )
    decoded, probabilities = predict_multitask_palette_sliding(
        **common_arguments,
        palette=sar_code_palette,
        tile_size=resolution,
        stride=stride,
        palette_temperature=palette_temperature,
    )
    return decoded, probabilities.argmax(dim=1)
