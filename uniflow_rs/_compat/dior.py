"""DIOR oriented-object detection support for UniFlow-RS."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from uniflow_rs._compat.dota import (
    DOTAFullImageStitcher,
    DOTADetectionDataset,
    dota_batch_detections,
    dota_batch_ground_truth,
    dota_batch_stitch_records,
    dota_confusion,
    dota_oriented_metrics,
    dota_raster_metrics,
    decode_dota_palette,
    write_dota_task1_submission,
)


DIOR_CLASS_NAMES = (
    "airplane",
    "airport",
    "baseballfield",
    "basketballcourt",
    "bridge",
    "chimney",
    "dam",
    "expressway-service-area",
    "expressway-toll-station",
    "golffield",
    "groundtrackfield",
    "harbor",
    "overpass",
    "ship",
    "stadium",
    "storagetank",
    "tenniscourt",
    "trainstation",
    "vehicle",
    "windmill",
)
DIOR_CLASS_COUNT = len(DIOR_CLASS_NAMES)
DIOR_BINARY_DISPLAY_PALETTE = np.asarray(
    ((0, 0, 0), (0, 0, 0), *((255, 255, 255),) * DIOR_CLASS_COUNT),
    dtype=np.uint8,
)

# Index 0 is ignore/padding and index 1 is valid background. Indices 2..21
# follow DIOR_CLASS_NAMES. Shared DIOR-R/DOTA categories deliberately reuse
# the DOTA code colours; DIOR-only categories use additional well-separated
# RGB-cube colours. This is an initial codebook and must pass the frozen-VAE
# round-trip check before it is used for training.
DIOR_CODE_PALETTE = np.asarray(
    (
        (0, 0, 0),
        (0, 0, 0),
        (255, 0, 0),       # airplane / DOTA plane
        (166, 0, 0),       # airport
        (255, 255, 0),     # baseballfield / DOTA baseball-diamond
        (255, 0, 255),     # basketballcourt / DOTA basketball-court
        (128, 0, 255),     # bridge
        (0, 166, 0),       # chimney
        (0, 0, 166),       # dam
        (166, 166, 0),     # expressway-service-area
        (0, 166, 166),     # expressway-toll-station
        (166, 0, 166),     # golffield
        (255, 128, 0),     # groundtrackfield / DOTA ground-track-field
        (0, 255, 128),     # harbor
        (255, 128, 128),   # overpass
        (0, 255, 0),       # ship / DOTA ship
        (128, 255, 128),   # stadium
        (0, 0, 255),       # storagetank / DOTA storage-tank
        (0, 255, 255),     # tenniscourt / DOTA tennis-court
        (128, 128, 255),   # trainstation
        (64, 192, 0),      # vehicle
        (192, 64, 0),      # windmill
    ),
    dtype=np.uint8,
)
DIOR_PALETTE_NAME = "dior_detection_v1"


def resolve_dior_code_palette(
    palette: np.ndarray | Sequence[Sequence[int]] | None = None,
) -> np.ndarray:
    """Validate and return a DIOR palette with ignore/background entries."""

    resolved = DIOR_CODE_PALETTE if palette is None else np.asarray(
        palette,
        dtype=np.uint8,
    )
    expected_shape = (DIOR_CLASS_COUNT + 2, 3)
    if resolved.shape != expected_shape:
        raise ValueError(
            "DIOR code palette must have shape "
            f"{expected_shape}, got {tuple(resolved.shape)}"
        )
    if not np.array_equal(resolved[0], (0, 0, 0)):
        raise ValueError("DIOR palette index 0 must be black ignore/padding")
    if not np.array_equal(resolved[1], (0, 0, 0)):
        raise ValueError("DIOR palette index 1 must be black background")
    return resolved


def load_dior_code_palette(path: str | Path) -> np.ndarray:
    """Load a DIOR palette from an optimization or round-trip JSON file."""

    palette_path = Path(path).expanduser()
    with palette_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, list):
        palette = payload
    elif isinstance(payload, dict):
        palette = payload.get("best_palette_rgb")
        if palette is None:
            palette = payload.get("palette_rgb")
        if palette is None and isinstance(payload.get("protocol"), dict):
            palette = payload["protocol"].get("palette_rgb")
    else:
        palette = None
    if palette is None:
        raise ValueError(
            f"DIOR palette JSON '{palette_path}' has no palette RGB array"
        )
    return resolve_dior_code_palette(palette).copy()


class DIORDetectionDataset(DOTADetectionDataset):
    """DIOR-R images with binary-query or multi-class palette targets."""

    CLASS_NAMES = DIOR_CLASS_NAMES
    MANIFEST_FORMAT = "dior_uniflow_rs_oriented_resize_v1"
    MANIFEST_FORMATS = (
        MANIFEST_FORMAT,
        "dior_uniflow_rs_oriented_v1",
    )
    DATASET_NAME = "DIOR-R"

    def __init__(self, *args, target_mode: str = "binary", **kwargs):
        mode = str(target_mode).strip().casefold()
        if mode == "palette":
            palette_name = kwargs.pop("code_palette", DIOR_PALETTE_NAME)
            if isinstance(palette_name, str):
                if palette_name.strip().casefold() != DIOR_PALETTE_NAME:
                    raise ValueError(
                        "Unknown DIOR code palette "
                        f"{palette_name!r}; expected {DIOR_PALETTE_NAME!r}"
                    )
                palette_name = DIOR_CODE_PALETTE
            kwargs["code_palette"] = resolve_dior_code_palette(palette_name)
        super().__init__(*args, target_mode=mode, **kwargs)


def decode_dior_palette(
    decoded: Tensor,
    temperature: float = 0.1,
    query_classes: Tensor | Sequence[int] | None = None,
    mask_threshold: float = 0.5,
    palette: np.ndarray | Sequence[Sequence[int]] | None = None,
) -> tuple[Tensor, Tensor]:
    if query_classes is None:
        if decoded.ndim != 4 or decoded.shape[1] != 3:
            raise ValueError(
                "decoded must have shape [B,3,H,W], got "
                f"{tuple(decoded.shape)}"
            )
        temperature = float(temperature)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("DIOR palette temperature must be positive")
        code_palette = resolve_dior_code_palette(palette)
        colors = torch.as_tensor(
            code_palette[1:],
            device=decoded.device,
            dtype=torch.float32,
        ).div(127.5).sub(1.0)
        pixels = decoded.float().clamp(-1.0, 1.0).permute(0, 2, 3, 1)
        distance = (
            pixels.unsqueeze(-2)
            - colors.view(1, 1, 1, len(colors), 3)
        ).square().sum(dim=-1)
        logits = (-distance / temperature).permute(0, 3, 1, 2)
        probabilities = logits.softmax(dim=1)
        return probabilities.argmax(dim=1).add(1), probabilities
    return decode_dota_palette(
        decoded,
        temperature=temperature,
        query_classes=query_classes,
        mask_threshold=mask_threshold,
        class_count=DIOR_CLASS_COUNT,
    )






def dior_confusion(
    prediction: Tensor,
    target: Tensor,
    valid_mask: Tensor | None = None,
) -> Tensor:
    return dota_confusion(
        prediction,
        target,
        valid_mask=valid_mask,
        class_count=DIOR_CLASS_COUNT,
    )


def dior_raster_metrics(confusion: Tensor) -> dict[str, float]:
    return dota_raster_metrics(confusion, class_names=DIOR_CLASS_NAMES)


def dior_batch_detections(
    labels: Tensor,
    probabilities: Tensor,
    batch: Mapping,
    config: Mapping | None = None,
) -> list[dict]:
    options = dict(config or {})
    options["class_count"] = DIOR_CLASS_COUNT
    return dota_batch_detections(labels, probabilities, batch, options)


def dior_oriented_metrics(
    detections: Sequence[dict],
    ground_truth: Sequence[dict],
    nms_threshold: float = 0.1,
) -> dict[str, float]:
    return dota_oriented_metrics(
        detections,
        ground_truth,
        nms_threshold=nms_threshold,
        class_names=DIOR_CLASS_NAMES,
    )


def write_dior_task1_submission(
    detections: Sequence[dict],
    output_dir: str | Path,
    image_names: Sequence[str],
    nms_threshold: float = 0.1,
) -> dict[str, object]:
    return write_dota_task1_submission(
        detections,
        output_dir,
        image_names=image_names,
        nms_threshold=nms_threshold,
        class_names=DIOR_CLASS_NAMES,
    )


DIORFullImageStitcher = DOTAFullImageStitcher
dior_batch_ground_truth = dota_batch_ground_truth
dior_batch_stitch_records = dota_batch_stitch_records
