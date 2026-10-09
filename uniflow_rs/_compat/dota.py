"""Shared oriented-detection implementation used by the DIOR-R adapter.

Internal DOTA-derived names are retained to avoid restructuring the dataset
base and geometry routines required by DIOR-R.
"""

from __future__ import annotations

import json
import math
import random
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

DETECTION_TASK = "detection"
DOTA_CLASS_NAMES = (
    "plane",
    "ship",
    "storage-tank",
    "baseball-diamond",
    "tennis-court",
    "basketball-court",
    "ground-track-field",
    "harbor",
    "bridge",
    "large-vehicle",
    "small-vehicle",
    "helicopter",
    "roundabout",
    "soccer-ball-field",
    "swimming-pool",
)

# Index 0 is reserved for ignore/padding. Index 1 is the generated background
# and indices 2..16 are the official DOTA classes above. The object colours
# lie on RGB-cube edges/faces and remain well separated after a frozen VAE
# round trip; background is explicitly black.
DOTA_CODE_PALETTE = np.asarray(
    (
        (0, 0, 0),
        (0, 0, 0),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 128, 0),
        (0, 255, 128),
        (128, 0, 255),
        (255, 0, 128),
        (128, 255, 0),
        (0, 128, 255),
        (128, 0, 128),
        (0, 128, 128),
        (128, 128, 0),
    ),
    dtype=np.uint8,
)

# Referring detection generates only a white binary mask. Class IDs are still
# retained internally for per-category metrics and oriented-box recovery, but
# visualisations must not imply that the model is asked to synthesize 15
# different colours.
DOTA_BINARY_DISPLAY_PALETTE = np.asarray(
    ((0, 0, 0), (0, 0, 0), *((255, 255, 255),) * len(DOTA_CLASS_NAMES)),
    dtype=np.uint8,
)


def _image_tensor(array: np.ndarray) -> Tensor:
    return (
        torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
        .float()
        .mul(2.0)
        .sub(1.0)
    )


def _sample_transform() -> tuple[int, bool, bool]:
    return (
        random.randrange(4),
        random.random() < 0.5,
        random.random() < 0.5,
    )


def _transform(array: np.ndarray, transform: tuple[int, bool, bool]) -> np.ndarray:
    rotation, horizontal_flip, vertical_flip = transform
    result = np.rot90(array, rotation, axes=(0, 1))
    if horizontal_flip:
        result = np.fliplr(result)
    if vertical_flip:
        result = np.flipud(result)
    return result


class DOTADetectionDataset(Dataset):
    """Class-query DOTA tiles represented as binary masks.

    Training keeps one record per non-empty tile and samples either a present
    (positive) or absent (negative) class. Evaluation enumerates every DOTA
    class for every tile, which removes ground-truth class-presence leakage and
    makes the predictions exportable to the official hidden-label test server.
    The text prompt carries class identity; the generated target is always a
    white foreground mask on black, and a negative query is entirely black.
    """

    CLASS_NAMES = DOTA_CLASS_NAMES
    MANIFEST_FORMAT = "dota_uniflow_rs_oriented_v1"
    DATASET_NAME = "DOTA"

    def __init__(
        self,
        root: str | Path,
        split: str,
        resolution: int = 512,
        manifest: str = "splits.json",
        augment: bool = False,
        prompt_key: str | None = None,
        negative_query_probability: float = 0.5,
        evaluation_query_mode: str = "all",
        training: bool | None = None,
        target_mode: str = "binary",
        code_palette: np.ndarray | None = None,
    ) -> None:
        self.root = Path(root).expanduser()
        self.split = str(split).strip().casefold()
        self.resolution = int(resolution)
        self.class_names = tuple(type(self).CLASS_NAMES)
        self.dataset_name = str(type(self).DATASET_NAME)
        self.augment = bool(augment)
        self.prompt_key = None if prompt_key is None else str(prompt_key)
        self.negative_query_probability = float(negative_query_probability)
        if not 0.0 <= self.negative_query_probability <= 1.0:
            raise ValueError("negative_query_probability must be in [0,1]")
        self.evaluation_query_mode = str(evaluation_query_mode).strip().casefold()
        if self.evaluation_query_mode not in {"all", "present"}:
            raise ValueError("evaluation_query_mode must be 'all' or 'present'")
        self.target_mode = str(target_mode).strip().casefold()
        if self.target_mode not in {"binary", "palette"}:
            raise ValueError(
                "target_mode must be 'binary' or 'palette', got " f"{target_mode!r}"
            )
        self.code_palette = (
            None
            if self.target_mode != "palette" or code_palette is None
            else np.asarray(code_palette, dtype=np.uint8)
        )
        if self.target_mode == "palette":
            if self.code_palette is None:
                raise ValueError(
                    f"{self.dataset_name} palette mode requires code_palette"
                )
            expected_palette_length = len(self.class_names) + 2
            if self.code_palette.shape != (expected_palette_length, 3):
                raise ValueError(
                    f"{self.dataset_name} code_palette must have shape "
                    f"[{expected_palette_length}, 3], got "
                    f"{tuple(self.code_palette.shape)}"
                )
        # A logical training split may concatenate several prepared physical
        # folders (for example DIOR-R train + val = official trainval). Keep
        # query sampling tied to the logical role rather than the folder name.
        self.training = self.split == "train" if training is None else bool(training)
        manifest_path = self.root / str(manifest)
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing prepared {self.dataset_name} manifest: {manifest_path}\n"
                "Prepare the dataset manifest first; for DIOR-R use tools/prepare_dior_detection.py."
            )
        with manifest_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        manifest_formats = tuple(
            getattr(
                type(self),
                "MANIFEST_FORMATS",
                (type(self).MANIFEST_FORMAT,),
            )
        )
        if payload.get("format") not in manifest_formats:
            raise ValueError(
                f"Unsupported {self.dataset_name} manifest format: "
                f"{payload.get('format')!r}; expected one of "
                f"{manifest_formats!r}"
            )
        if int(payload["resolution"]) != self.resolution:
            raise ValueError(
                f"Prepared {self.dataset_name} resolution does not match config: "
                f"manifest={payload['resolution']}, config={self.resolution}"
            )
        if tuple(payload["class_names"]) != self.class_names:
            raise ValueError(
                f"Prepared {self.dataset_name} class order does not match the code"
            )
        if self.split not in payload["splits"]:
            raise KeyError(f"DOTA manifest has no split {self.split!r}")
        self.tile_size = int(payload["tile_size"])
        self.overlap = int(payload["overlap"])
        self.max_objects_per_tile = int(payload["max_objects_per_tile"])
        base_samples = list(payload["splits"][self.split])
        if not base_samples:
            raise ValueError(f"Prepared DOTA split {self.split!r} is empty")
        labeled_flags = {bool(sample.get("labeled")) for sample in base_samples}
        if len(labeled_flags) != 1:
            raise ValueError(
                f"DOTA split {self.split!r} mixes labeled and unlabeled tiles"
            )
        self.has_labels = labeled_flags == {True}
        if not self.has_labels and self.split != "test":
            raise ValueError(
                f"Only the DOTA test split may be unlabeled, got {self.split!r}"
            )

        positive_samples: list[dict] = []
        for sample in base_samples:
            query_classes = sorted(
                {
                    int(item["class_id"])
                    for item in sample.get("objects", ())
                    if not bool(item.get("difficult", False))
                }
            )
            copied = dict(sample)
            copied["_query_classes"] = query_classes
            copied["_negative_query_classes"] = [
                class_id
                for class_id in range(len(self.class_names))
                if class_id not in query_classes
            ]
            if query_classes:
                positive_samples.append(copied)
        if self.training and not positive_samples and self.target_mode == "binary":
            raise ValueError(
                f"Prepared DOTA split {self.split!r} has no positive tiles"
            )

        if self.training:
            if not self.has_labels:
                raise ValueError("DOTA training requires labels")
            self.samples = (
                base_samples if self.target_mode == "palette" else positive_samples
            )
            self.random_query = self.target_mode == "binary"
        else:
            expanded: list[dict] = []
            if self.target_mode == "palette":
                expanded = [
                    dict(sample, _parent_query_count=1) for sample in base_samples
                ]
            else:
                for sample in (
                    dict(
                        sample,
                        _query_classes=sorted(
                            {
                                int(item["class_id"])
                                for item in sample.get("objects", ())
                                if not bool(item.get("difficult", False))
                            }
                        ),
                    )
                    for sample in base_samples
                ):
                    query_classes = (
                        range(len(self.class_names))
                        if self.evaluation_query_mode == "all"
                        else sample["_query_classes"]
                    )
                    for class_id in query_classes:
                        record = dict(sample)
                        record["_query_class"] = int(class_id)
                        record["_query_is_positive"] = (
                            int(class_id) in sample["_query_classes"]
                        )
                        expanded.append(record)
            if not expanded:
                raise ValueError(
                    f"DOTA split {self.split!r} produced no evaluation queries"
                )
            parent_counts: dict[str, int] = defaultdict(int)
            for record in expanded:
                parent_counts[str(record["parent"])] += 1
            for record in expanded:
                record["_parent_query_count"] = parent_counts[str(record["parent"])]
            self.samples = expanded
            self.random_query = False
        self.parent_names = tuple(
            sorted({str(sample["parent"]) for sample in base_samples})
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor | str | int | float]:
        sample = self.samples[index]
        query_class: int | None = None
        query_is_positive = True
        if self.target_mode == "palette":
            query_name = "all_classes"
        elif self.random_query:
            positive_classes = sample["_query_classes"]
            negative_classes = sample["_negative_query_classes"]
            choose_negative = (
                bool(negative_classes)
                and random.random() < self.negative_query_probability
            )
            selected_classes = negative_classes if choose_negative else positive_classes
            query_class = int(random.choice(selected_classes))
            query_is_positive = not choose_negative
        else:
            query_class = int(sample["_query_class"])
            query_is_positive = bool(sample["_query_is_positive"])
        if self.target_mode == "binary":
            query_name = self.class_names[query_class]
        with Image.open(self.root / sample["image"]) as source:
            image = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
        expected = (self.resolution, self.resolution)
        if self.has_labels:
            with Image.open(self.root / sample["target"]) as source:
                labels = np.asarray(source.convert("L"), dtype=np.uint8)
            with Image.open(self.root / sample["valid_mask"]) as source:
                valid = np.asarray(source.convert("L"), dtype=np.uint8) > 127
        else:
            valid = np.zeros(expected, dtype=bool)
            valid_height = min(
                self.resolution,
                math.ceil(
                    int(sample["valid_height"]) * self.resolution / self.tile_size
                ),
            )
            valid_width = min(
                self.resolution,
                math.ceil(
                    int(sample["valid_width"]) * self.resolution / self.tile_size
                ),
            )
            valid[:valid_height, :valid_width] = True
            labels = np.where(valid, 1, 0).astype(np.uint8)
        if (
            image.shape[:2] != expected
            or labels.shape != expected
            or valid.shape != expected
        ):
            raise RuntimeError(
                f"Prepared DOTA tile has invalid dimensions: {sample['name']}"
            )
        if labels.size and (
            labels.min() < 0 or labels.max() > len(self.class_names) + 1
        ):
            raise ValueError(
                f"{self.dataset_name} class IDs must be in "
                f"[0,{len(self.class_names) + 1}]: {sample['name']}"
            )
        if self.augment:
            transform = _sample_transform()
            image = _transform(image, transform)
            labels = _transform(labels, transform)
            valid = _transform(valid, transform)

        if self.target_mode == "palette":
            query_labels = labels.astype(np.uint8, copy=False)
            target_rgb = self.code_palette[query_labels].astype(np.float32) / 255.0
        else:
            # Keep ignore/padding at 0, valid background at 1, and use the
            # original DOTA label ID for the queried foreground. The RGB
            # target is class-agnostic black/white; class identity comes from
            # the prompt.
            query_label = int(query_class) + 2
            query_labels = np.where(
                labels == 0,
                0,
                np.where(labels == query_label, query_label, 1),
            ).astype(np.uint8)
            foreground = query_labels == query_label
            target_rgb = np.repeat(foreground[..., None], 3, axis=2).astype(np.float32)
        image_tensor = _image_tensor(image)
        mask_tensor = torch.from_numpy(
            np.ascontiguousarray(query_labels[None].astype(np.float32))
        )
        metric_valid = valid if self.has_labels else np.zeros_like(valid)
        valid_tensor = torch.from_numpy(
            np.ascontiguousarray(metric_valid[None].astype(np.float32))
        )
        result: dict[str, Tensor | str | int | float] = {
            "before": image_tensor,
            "after": image_tensor,
            "mask": mask_tensor,
            "loss_mask": valid_tensor,
            "target_rgb": _image_tensor(target_rgb),
            "task": DETECTION_TASK,
            "name": (
                str(sample["name"])
                if self.target_mode == "palette"
                else f"{sample['name']}__query_{query_name.replace('-', '_')}"
            ),
            "input_modality": "rgb",
            "detection_has_labels": self.has_labels,
            "parent_name": str(sample["parent"]),
            "parent_query_count": int(sample.get("_parent_query_count", 1)),
            "tile_x": int(sample["x"]),
            "tile_y": int(sample["y"]),
            "source_width": int(sample["source_width"]),
            "source_height": int(sample["source_height"]),
            "valid_width": int(sample["valid_width"]),
            "valid_height": int(sample["valid_height"]),
            "tile_size": self.tile_size,
            "tile_overlap": self.overlap,
        }
        if self.target_mode == "binary":
            result.update(
                {
                    "detection_query_class": int(query_class),
                    "detection_query_name": query_name,
                    "detection_query_is_positive": query_is_positive,
                }
            )
        if self.prompt_key is not None:
            result["prompt_key"] = (
                str(self.prompt_key)
                if self.target_mode == "palette"
                else f"{self.prompt_key}_{query_name.replace('-', '_')}"
            )

        boxes = torch.zeros((self.max_objects_per_tile, 8), dtype=torch.float32)
        classes = torch.full((self.max_objects_per_tile,), -1, dtype=torch.long)
        object_ids = torch.full((self.max_objects_per_tile,), -1, dtype=torch.long)
        difficult = torch.zeros((self.max_objects_per_tile,), dtype=torch.bool)
        object_valid = torch.zeros((self.max_objects_per_tile,), dtype=torch.bool)
        objects = [
            item
            for item in sample.get("objects", ())
            if (
                self.target_mode == "palette"
                or int(item["class_id"]) == int(query_class)
            )
            and not bool(item.get("difficult", False))
        ]
        if len(objects) > self.max_objects_per_tile:
            raise RuntimeError(
                f"DOTA tile {sample['name']} has {len(objects)} objects, above "
                f"manifest max {self.max_objects_per_tile}"
            )
        for row, item in enumerate(objects):
            boxes[row] = torch.as_tensor(item["polygon"], dtype=torch.float32)
            classes[row] = int(item["class_id"])
            object_ids[row] = int(item["object_id"])
            difficult[row] = bool(item.get("difficult", False))
            object_valid[row] = True
        result.update(
            {
                "detection_gt_boxes": boxes,
                "detection_gt_classes": classes,
                "detection_gt_ids": object_ids,
                "detection_gt_difficult": difficult,
                "detection_gt_valid": object_valid,
            }
        )
        return result


def dota_palette_logits(decoded: Tensor, temperature: float = 0.1) -> Tensor:
    if decoded.ndim != 4 or decoded.shape[1] != 3:
        raise ValueError(
            f"decoded must have shape [B,3,H,W], got {tuple(decoded.shape)}"
        )
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("DOTA palette temperature must be finite and positive")
    colors = (
        torch.as_tensor(
            DOTA_CODE_PALETTE[1:], device=decoded.device, dtype=torch.float32
        )
        .div(127.5)
        .sub(1.0)
    )
    pixels = decoded.float().clamp(-1.0, 1.0).permute(0, 2, 3, 1)
    distance = (
        (pixels.unsqueeze(-2) - colors.view(1, 1, 1, len(colors), 3))
        .square()
        .sum(dim=-1)
    )
    return (-distance / temperature).permute(0, 3, 1, 2)


def dota_referring_probability(decoded: Tensor) -> Tensor:
    """Convert a decoded black/white RGB target to foreground probability."""

    if decoded.ndim != 4 or decoded.shape[1] != 3:
        raise ValueError(
            f"decoded must have shape [B,3,H,W], got {tuple(decoded.shape)}"
        )
    return decoded.float().mean(dim=1).add(1.0).mul(0.5).clamp(0.0, 1.0)


def decode_dota_palette(
    decoded: Tensor,
    temperature: float = 0.1,
    query_classes: Tensor | Sequence[int] | None = None,
    mask_threshold: float = 0.5,
    class_count: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Decode a legacy colour map or a class-conditioned binary mask.

    Query classes use zero-based DOTA IDs. Returned labels retain the legacy
    convention (1=background, 2..16=object classes), so the existing raster,
    stitching, and oriented-box metrics remain reusable.
    """

    if query_classes is None:
        probabilities = dota_palette_logits(decoded, temperature).softmax(dim=1)
        return probabilities.argmax(dim=1).add(1), probabilities

    class_count = len(DOTA_CLASS_NAMES) if class_count is None else int(class_count)
    if class_count < 1:
        raise ValueError("class_count must be positive")

    query = torch.as_tensor(
        query_classes,
        device=decoded.device,
        dtype=torch.long,
    ).reshape(-1)
    if query.numel() != decoded.shape[0]:
        raise ValueError(
            "query_classes must contain one class per decoded sample, got "
            f"{query.numel()} for batch {decoded.shape[0]}"
        )
    if bool(((query < 0) | (query >= class_count)).any()):
        raise ValueError(f"Detection query classes must be in [0,{class_count - 1}]")

    foreground = dota_referring_probability(decoded)
    probabilities = torch.zeros(
        (decoded.shape[0], class_count + 1, decoded.shape[2], decoded.shape[3]),
        device=decoded.device,
        dtype=foreground.dtype,
    )
    probabilities[:, 0] = 1.0 - foreground
    query_channels = (
        query.add(1).view(-1, 1, 1, 1).expand(-1, 1, decoded.shape[2], decoded.shape[3])
    )
    probabilities.scatter_(1, query_channels, foreground.unsqueeze(1))
    labels = torch.ones_like(foreground, dtype=torch.long)
    labels = torch.where(
        foreground >= float(mask_threshold),
        query.add(2).view(-1, 1, 1),
        labels,
    )
    return labels, probabilities


def dota_confusion(
    prediction: Tensor,
    target: Tensor,
    valid_mask: Tensor | None = None,
    class_count: int | None = None,
) -> Tensor:
    class_count = len(DOTA_CLASS_NAMES) if class_count is None else int(class_count)
    if prediction.ndim == 4 and prediction.shape[1] == 1:
        prediction = prediction[:, 0]
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    prediction = prediction.long().clamp(1, class_count + 1)
    target = target.long()
    valid = (target >= 1) & (target <= class_count + 1)
    if valid_mask is not None:
        if valid_mask.ndim == 4:
            valid_mask = valid_mask[:, 0]
        valid &= valid_mask.bool()
    bins = torch.bincount(
        (target[valid] - 1) * (class_count + 1) + (prediction[valid] - 1),
        minlength=(class_count + 1) ** 2,
    )
    return bins.reshape(class_count + 1, class_count + 1).to(torch.float64)


def dota_raster_metrics(
    confusion: Tensor,
    class_names: Sequence[str] = DOTA_CLASS_NAMES,
) -> dict[str, float]:
    confusion = confusion.detach().cpu().double()
    true_positive = confusion.diag()
    ground_truth = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)
    union = ground_truth + predicted - true_positive
    iou = true_positive / union.clamp_min(1.0)
    dice = 2.0 * true_positive / (ground_truth + predicted).clamp_min(1.0)
    object_present = ground_truth[1:] > 0
    object_miou = (
        float(iou[1:][object_present].mean()) if bool(object_present.any()) else 0.0
    )
    object_mdice = (
        float(dice[1:][object_present].mean()) if bool(object_present.any()) else 0.0
    )
    metrics = {
        "detection_iou": object_miou,
        "detection_dice": object_mdice,
        "detection_miou": object_miou,
        "detection_mdice": object_mdice,
        "detection_background_iou": float(iou[0]),
        "detection_pixel_accuracy": float(
            true_positive.sum() / confusion.sum().clamp_min(1.0)
        ),
    }
    for class_id, name in enumerate(class_names, start=1):
        metrics[f"detection_{name.replace('-', '_')}_iou"] = float(iou[class_id])
    return metrics


def _polygon_overlap_geometry(
    first: Sequence[float],
    second: Sequence[float],
) -> tuple[float, float, float]:
    """Return intersection and individual areas for two convex polygons."""

    first_array = np.asarray(first, dtype=np.float32).reshape(-1, 2)
    second_array = np.asarray(second, dtype=np.float32).reshape(-1, 2)
    first_hull = cv2.convexHull(first_array)
    second_hull = cv2.convexHull(second_array)
    first_area = abs(float(cv2.contourArea(first_hull)))
    second_area = abs(float(cv2.contourArea(second_hull)))
    if first_area <= 0.0 or second_area <= 0.0:
        return 0.0, first_area, second_area
    intersection, _ = cv2.intersectConvexConvex(first_hull, second_hull)
    return float(intersection), first_area, second_area


def polygon_iou(first: Sequence[float], second: Sequence[float]) -> float:
    intersection, first_area, second_area = _polygon_overlap_geometry(
        first,
        second,
    )
    return intersection / max(
        first_area + second_area - intersection,
        1.0e-8,
    )


def _detection_class_ids(options: Mapping) -> list[int]:
    class_count = int(options.get("class_count", len(DOTA_CLASS_NAMES)))
    included = options.get("include_class_ids")
    if included is None:
        return list(range(class_count))
    class_ids = sorted({int(value) for value in included})
    invalid = [value for value in class_ids if value < 0 or value >= class_count]
    if invalid:
        raise ValueError(
            f"include_class_ids contains values outside [0, {class_count}): "
            f"{invalid}"
        )
    return class_ids


def _watershed_component_masks(
    component: np.ndarray,
    options: Mapping,
) -> list[np.ndarray]:
    """Split a connected semantic region at distance-transform basins."""

    area = int(component.sum())
    min_component_area = int(options.get("watershed_min_component_area", 32))
    if area < min_component_area:
        return [component]

    distance = cv2.distanceTransform(
        component.astype(np.uint8),
        cv2.DIST_L2,
        5,
    )
    blur_sigma = float(options.get("watershed_distance_blur_sigma", 1.0))
    if blur_sigma > 0.0:
        distance = cv2.GaussianBlur(distance, (0, 0), blur_sigma)
    maximum = float(distance.max())
    if maximum <= 0.0:
        return [component]

    peak_min_distance = max(
        int(options.get("watershed_peak_min_distance", 3)),
        1,
    )
    kernel_size = 2 * peak_min_distance + 1
    local_maximum = cv2.dilate(
        distance,
        np.ones((kernel_size, kernel_size), dtype=np.uint8),
    )
    relative_threshold = float(options.get("watershed_peak_rel_threshold", 0.25))
    if not 0.0 <= relative_threshold <= 1.0:
        raise ValueError("watershed_peak_rel_threshold must be in [0, 1]")
    peaks = (
        component
        & (distance >= local_maximum - 1.0e-6)
        & (distance >= maximum * relative_threshold)
    ).astype(np.uint8)
    marker_count, _ = cv2.connectedComponents(peaks, 8)
    if marker_count <= 2:
        return [component]

    # Assign every foreground pixel to its nearest distance peak. Unlike
    # cv2.watershed with a background marker, this marker-controlled basin
    # split cannot let the background consume foreground pixels.
    marker_surface = np.ones(component.shape, dtype=np.uint8)
    marker_surface[peaks > 0] = 0
    _, basin_labels = cv2.distanceTransformWithLabels(
        marker_surface,
        cv2.DIST_L2,
        5,
        labelType=cv2.DIST_LABEL_CCOMP,
    )
    segments = [
        (basin_labels == marker_id) & component for marker_id in range(1, marker_count)
    ]
    min_segment_area = int(
        options.get(
            "watershed_min_segment_area",
            options.get("min_component_area", 4),
        )
    )
    if len(segments) <= 1 or any(
        int(segment.sum()) < min_segment_area for segment in segments
    ):
        return [component]
    return segments


def _instance_component_masks(
    binary: np.ndarray,
    class_id: int,
    options: Mapping,
) -> list[np.ndarray]:
    component_count, components, _, _ = cv2.connectedComponentsWithStats(
        binary,
        8,
    )
    watershed_class_ids = {
        int(value) for value in options.get("watershed_class_ids", ())
    }
    masks: list[np.ndarray] = []
    for component_id in range(1, component_count):
        component = components == component_id
        if class_id in watershed_class_ids:
            masks.extend(_watershed_component_masks(component, options))
        else:
            masks.append(component)
    return masks


def _component_detections(
    labels: np.ndarray,
    probabilities: np.ndarray,
    metadata: Mapping[str, int | str],
    config: Mapping | None,
) -> list[dict]:
    options = dict(config or {})
    min_area = int(options.get("min_component_area", 4))
    score_threshold = float(options.get("score_threshold", 0.05))
    opening_kernel = int(options.get("opening_kernel", 0))
    tile_size = float(metadata["tile_size"])
    scale = labels.shape[1] / tile_size
    detections: list[dict] = []
    for class_id in _detection_class_ids(options):
        binary = (labels == class_id + 2).astype(np.uint8)
        if opening_kernel > 1:
            kernel = np.ones((opening_kernel, opening_kernel), dtype=np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        for component in _instance_component_masks(binary, class_id, options):
            area = int(component.sum())
            if area < min_area:
                continue
            score = float(probabilities[class_id + 1][component].mean())
            if score < score_threshold:
                continue
            contours, _ = cv2.findContours(
                component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            if len(contour) < 3:
                continue
            moments = cv2.moments(contour)
            if abs(float(moments["m00"])) < 1.0e-8:
                continue
            center_x = float(moments["m10"] / moments["m00"]) / scale
            center_y = float(moments["m01"] / moments["m00"]) / scale
            # Assign each overlapping-window prediction to exactly one core
            # region. This removes cut-off duplicates before oriented NMS.
            margin = 0.5 * float(metadata.get("tile_overlap", 0))
            tile_x = float(metadata["tile_x"])
            tile_y = float(metadata["tile_y"])
            source_width = float(metadata["source_width"])
            source_height = float(metadata["source_height"])
            if tile_x > 0.0 and center_x < margin:
                continue
            if tile_y > 0.0 and center_y < margin:
                continue
            if tile_x + tile_size < source_width and center_x >= tile_size - margin:
                continue
            if tile_y + tile_size < source_height and center_y >= tile_size - margin:
                continue
            box = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float64)
            box[:, 0] = box[:, 0] / scale + float(metadata["tile_x"])
            box[:, 1] = box[:, 1] / scale + float(metadata["tile_y"])
            box[:, 0] = np.clip(box[:, 0], 0.0, float(metadata["source_width"]) - 1.0)
            box[:, 1] = np.clip(box[:, 1], 0.0, float(metadata["source_height"]) - 1.0)
            detections.append(
                {
                    "parent": str(metadata["parent_name"]),
                    "class_id": class_id,
                    "score": score,
                    "component_area": area,
                    "polygon": box.reshape(-1).tolist(),
                }
            )
    return detections


def dota_batch_detections(
    labels: Tensor,
    probabilities: Tensor,
    batch: Mapping,
    config: Mapping | None = None,
) -> list[dict]:
    labels_np = labels.detach().cpu().numpy()
    probabilities_np = probabilities.detach().float().cpu().numpy()
    config = dict(config or {})
    config.setdefault("class_count", len(DOTA_CLASS_NAMES))
    records: list[dict] = []
    for index in range(labels_np.shape[0]):
        metadata = {
            "parent_name": batch["parent_name"][index],
            "tile_x": int(batch["tile_x"][index]),
            "tile_y": int(batch["tile_y"][index]),
            "source_width": int(batch["source_width"][index]),
            "source_height": int(batch["source_height"][index]),
            "tile_size": int(batch["tile_size"][index]),
            "tile_overlap": int(batch["tile_overlap"][index]),
        }
        records.extend(
            _component_detections(
                labels_np[index], probabilities_np[index], metadata, config
            )
        )
    return records


def _png_bytes(array: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(
        ".png",
        np.ascontiguousarray(array),
        [cv2.IMWRITE_PNG_COMPRESSION, 3],
    )
    if not success:
        raise RuntimeError("Could not encode DOTA stitch record as PNG")
    return encoded.tobytes()


def _decode_png(payload: bytes) -> np.ndarray:
    decoded = cv2.imdecode(
        np.frombuffer(payload, dtype=np.uint8),
        cv2.IMREAD_UNCHANGED,
    )
    if decoded is None:
        raise RuntimeError("Could not decode DOTA stitch record")
    return decoded


def dota_batch_stitch_records(
    labels: Tensor,
    probabilities: Tensor,
    batch: Mapping,
) -> list[dict]:
    """Pack tile predictions for cheap distributed full-image stitching.

    Only the winning class and its confidence are transferred.  Quantising
    confidence to uint8 and PNG-compressing both maps keeps per-step
    ``gather_object`` traffic far below transferring the 16-channel tensor.
    """

    query_classes = batch.get("detection_query_class")
    if query_classes is None:
        class_count = max(int(probabilities.shape[1]) - 1, 1)
        label_indices = labels.long().sub(1).clamp(0, class_count - 1)
        confidence = probabilities.gather(
            1,
            label_indices.unsqueeze(1),
        ).squeeze(1)
        query_tensor = None
    else:
        query_tensor = torch.as_tensor(
            query_classes,
            device=probabilities.device,
            dtype=torch.long,
        ).reshape(-1)
        query_channels = (
            query_tensor.add(1)
            .view(-1, 1, 1, 1)
            .expand(-1, 1, probabilities.shape[2], probabilities.shape[3])
        )
        # For query-mode stitching we need foreground probability even when
        # background wins the binary threshold.
        confidence = probabilities.gather(1, query_channels).squeeze(1)
    labels_np = labels.detach().to(torch.uint8).cpu().numpy()
    confidence_np = (
        confidence.detach()
        .float()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    records: list[dict] = []
    for index in range(labels_np.shape[0]):
        record = {
            "parent_name": str(batch["parent_name"][index]),
            "tile_x": int(batch["tile_x"][index]),
            "tile_y": int(batch["tile_y"][index]),
            "source_width": int(batch["source_width"][index]),
            "source_height": int(batch["source_height"][index]),
            "valid_width": int(batch["valid_width"][index]),
            "valid_height": int(batch["valid_height"][index]),
            "tile_size": int(batch["tile_size"][index]),
            "tile_overlap": int(batch["tile_overlap"][index]),
            "labels_png": _png_bytes(labels_np[index]),
            "confidence_png": _png_bytes(confidence_np[index]),
        }
        if query_tensor is not None:
            record["query_class"] = int(query_tensor[index])
            record["expected_query_records"] = int(batch["parent_query_count"][index])
        records.append(record)
    return records


def _tile_count(length: int, tile_size: int, overlap: int) -> int:
    if length <= tile_size:
        return 1
    stride = tile_size - overlap
    count = len(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    last = (count - 1) * stride
    return count + int(last != final)


def _stitched_component_detections(
    labels: np.ndarray,
    confidence: np.ndarray,
    parent: str,
    config: Mapping | None,
) -> list[dict]:
    options = dict(config or {})
    min_area = int(options.get("min_component_area", 4))
    score_threshold = float(options.get("score_threshold", 0.05))
    opening_kernel = int(options.get("opening_kernel", 0))
    detections: list[dict] = []
    for class_id in _detection_class_ids(options):
        class_y, class_x = np.nonzero(labels == class_id + 2)
        if class_y.size == 0:
            continue
        x0 = max(0, int(class_x.min()) - 1)
        x1 = min(labels.shape[1], int(class_x.max()) + 2)
        y0 = max(0, int(class_y.min()) - 1)
        y1 = min(labels.shape[0], int(class_y.max()) + 2)
        binary = (labels[y0:y1, x0:x1] == class_id + 2).astype(np.uint8)
        if opening_kernel > 1:
            kernel = np.ones((opening_kernel, opening_kernel), dtype=np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        for component in _instance_component_masks(binary, class_id, options):
            area = int(component.sum())
            if area < min_area:
                continue
            score = float(confidence[y0:y1, x0:x1][component].mean())
            if score < score_threshold:
                continue
            contours, _ = cv2.findContours(
                component.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            if len(contour) < 3:
                continue
            box = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float64)
            box[:, 0] += x0
            box[:, 1] += y0
            box[:, 0] = np.clip(box[:, 0], 0.0, labels.shape[1] - 1.0)
            box[:, 1] = np.clip(box[:, 1], 0.0, labels.shape[0] - 1.0)
            detections.append(
                {
                    "parent": str(parent),
                    "class_id": class_id,
                    "score": score,
                    "component_area": area,
                    "polygon": box.reshape(-1).tolist(),
                }
            )
    return detections


class DOTAFullImageStitcher:
    """Fuse overlapping tile labels before recovering oriented instances."""

    def __init__(self, config: Mapping | None = None) -> None:
        self.config = dict(config or {})
        self._states: dict[str, dict] = {}
        self._completed: set[str] = set()

    @staticmethod
    def _new_state(record: Mapping) -> dict:
        width = int(record["source_width"])
        height = int(record["source_height"])
        tile_size = int(record["tile_size"])
        overlap = int(record["tile_overlap"])
        query_mode = "query_class" in record
        return {
            "labels": np.ones((height, width), dtype=np.uint8),
            "confidence": np.zeros((height, width), dtype=np.uint8),
            "owner_weight": np.zeros((height, width), dtype=np.uint16),
            "query_confidence": {},
            "query_owner_weight": {},
            "query_mode": query_mode,
            "seen": set(),
            "expected": (
                int(record["expected_query_records"])
                if query_mode
                else _tile_count(width, tile_size, overlap)
                * _tile_count(height, tile_size, overlap)
            ),
        }

    @staticmethod
    def _spatial_weight(record: Mapping, width: int, height: int) -> np.ndarray:
        tile_x = int(record["tile_x"])
        tile_y = int(record["tile_y"])
        tile_size = int(record["tile_size"])
        source_width = int(record["source_width"])
        source_height = int(record["source_height"])
        x = np.arange(width, dtype=np.int32)
        y = np.arange(height, dtype=np.int32)
        left = x + 1 if tile_x > 0 else np.full_like(x, tile_size)
        right = (
            width - x if tile_x + width < source_width else np.full_like(x, tile_size)
        )
        top = y + 1 if tile_y > 0 else np.full_like(y, tile_size)
        bottom = (
            height - y
            if tile_y + height < source_height
            else np.full_like(y, tile_size)
        )
        horizontal = np.minimum(left, right)
        vertical = np.minimum(top, bottom)
        return np.minimum(vertical[:, None], horizontal[None, :]).astype(np.uint16)

    def _finish_parent(self, parent: str) -> list[dict]:
        state = self._states.pop(parent)
        self._completed.add(parent)
        if bool(state["query_mode"]):
            detections: list[dict] = []
            mask_threshold = float(self.config.get("mask_threshold", 0.5))
            for class_id, confidence_u8 in state["query_confidence"].items():
                confidence = confidence_u8.astype(np.float32) / 255.0
                labels = np.ones(confidence.shape, dtype=np.uint8)
                labels[confidence >= mask_threshold] = int(class_id) + 2
                detections.extend(
                    _stitched_component_detections(
                        labels,
                        confidence,
                        parent,
                        self.config,
                    )
                )
            return detections
        return _stitched_component_detections(
            state["labels"],
            state["confidence"].astype(np.float32) / 255.0,
            parent,
            self.config,
        )

    def add(self, records: Sequence[Mapping]) -> list[dict]:
        detections: list[dict] = []
        for record in records:
            parent = str(record["parent_name"])
            if parent in self._completed:
                # Distributed DataLoader padding can repeat its last sample.
                continue
            if parent not in self._states:
                self._states[parent] = self._new_state(record)
            state = self._states[parent]
            tile_key = (
                int(record["tile_x"]),
                int(record["tile_y"]),
                int(record.get("query_class", -1)),
            )
            if tile_key in state["seen"]:
                continue
            labels = _decode_png(record["labels_png"])
            confidence = _decode_png(record["confidence_png"])
            tile_size = int(record["tile_size"])
            if labels.shape != (tile_size, tile_size):
                labels = cv2.resize(
                    labels,
                    (tile_size, tile_size),
                    interpolation=cv2.INTER_NEAREST,
                )
                confidence = cv2.resize(
                    confidence,
                    (tile_size, tile_size),
                    interpolation=cv2.INTER_LINEAR,
                )
            valid_width = int(record["valid_width"])
            valid_height = int(record["valid_height"])
            labels = labels[:valid_height, :valid_width]
            confidence = confidence[:valid_height, :valid_width]
            weight = self._spatial_weight(record, valid_width, valid_height)
            x = int(record["tile_x"])
            y = int(record["tile_y"])
            region = np.s_[y : y + valid_height, x : x + valid_width]
            if bool(state["query_mode"]):
                class_id = int(record["query_class"])
                if class_id not in state["query_confidence"]:
                    shape = state["confidence"].shape
                    state["query_confidence"][class_id] = np.zeros(
                        shape, dtype=np.uint8
                    )
                    state["query_owner_weight"][class_id] = np.zeros(
                        shape, dtype=np.uint16
                    )
                class_confidence = state["query_confidence"][class_id][region]
                class_weight = state["query_owner_weight"][class_id][region]
                update = (weight > class_weight) | (
                    (weight == class_weight) & (confidence > class_confidence)
                )
                class_confidence[update] = confidence[update]
                class_weight[update] = weight[update]
            else:
                previous_weight = state["owner_weight"][region]
                previous_confidence = state["confidence"][region]
                update = (weight > previous_weight) | (
                    (weight == previous_weight) & (confidence > previous_confidence)
                )
                state["labels"][region][update] = labels[update]
                state["confidence"][region][update] = confidence[update]
                state["owner_weight"][region][update] = weight[update]
            state["seen"].add(tile_key)
            if len(state["seen"]) >= int(state["expected"]):
                detections.extend(self._finish_parent(parent))
        return detections

    def finish(self) -> list[dict]:
        detections: list[dict] = []
        for parent in list(self._states):
            detections.extend(self._finish_parent(parent))
        return detections


def dota_batch_ground_truth(batch: Mapping) -> list[dict]:
    boxes = batch["detection_gt_boxes"].detach().cpu().numpy()
    classes = batch["detection_gt_classes"].detach().cpu().numpy()
    object_ids = batch["detection_gt_ids"].detach().cpu().numpy()
    difficult = batch["detection_gt_difficult"].detach().cpu().numpy()
    valid = batch["detection_gt_valid"].detach().cpu().numpy()
    records: list[dict] = []
    for batch_index, parent in enumerate(batch["parent_name"]):
        for row in np.flatnonzero(valid[batch_index]):
            records.append(
                {
                    "parent": str(parent),
                    "object_id": int(object_ids[batch_index, row]),
                    "class_id": int(classes[batch_index, row]),
                    "difficult": bool(difficult[batch_index, row]),
                    "polygon": boxes[batch_index, row].tolist(),
                }
            )
    return records


def _oriented_nms(detections: Sequence[dict], threshold: float) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for detection in detections:
        grouped[(str(detection["parent"]), int(detection["class_id"]))].append(
            detection
        )
    kept: list[dict] = []
    for current in grouped.values():
        current = sorted(current, key=lambda item: float(item["score"]), reverse=True)
        while current:
            selected = current.pop(0)
            kept.append(selected)
            current = [
                candidate
                for candidate in current
                if polygon_iou(selected["polygon"], candidate["polygon"]) < threshold
            ]
    return kept


def write_dota_task1_submission(
    detections: Sequence[dict],
    output_dir: str | Path,
    image_names: Sequence[str],
    nms_threshold: float = 0.1,
    ensure_nonempty: bool = True,
    class_names: Sequence[str] = DOTA_CLASS_NAMES,
) -> dict[str, object]:
    """Write official DOTA-v1.0 Task1 text files and a ready-to-upload zip.

    The official server rejects empty per-class files. If a class has no
    prediction, a single 1e-8-confidence one-pixel placeholder is emitted;
    this is ranked last and cannot conceal that the class AP is effectively
    zero, but keeps the submission structurally valid.
    """

    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    parents = tuple(sorted({str(name) for name in image_names}))
    if not parents:
        raise ValueError("DOTA Task1 export requires at least one image name")
    predictions = _oriented_nms(detections, float(nms_threshold))
    by_class: dict[int, list[dict]] = defaultdict(list)
    for item in predictions:
        class_id = int(item["class_id"])
        if not 0 <= class_id < len(class_names):
            raise ValueError(f"Invalid DOTA class ID in prediction: {class_id}")
        by_class[class_id].append(item)

    text_paths: list[Path] = []
    placeholder_classes: list[str] = []
    for class_id, class_name in enumerate(class_names):
        rows = sorted(
            by_class.get(class_id, ()),
            key=lambda item: (
                str(item["parent"]),
                -float(item["score"]),
            ),
        )
        lines = []
        for item in rows:
            polygon = np.asarray(item["polygon"], dtype=np.float64).reshape(8)
            coordinates = " ".join(f"{value:.2f}" for value in polygon)
            score = min(max(float(item["score"]), 0.0), 1.0)
            lines.append(f"{item['parent']} {score:.8f} {coordinates}")
        if not lines and ensure_nonempty:
            placeholder_classes.append(class_name)
            lines.append(
                f"{parents[0]} 0.00000001 0.00 0.00 1.00 0.00 " "1.00 1.00 0.00 1.00"
            )
        path = destination / f"Task1_{class_name}.txt"
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        text_paths.append(path)

    archive = destination.parent / f"{destination.name}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in text_paths:
            bundle.write(path, arcname=path.name)
    return {
        "detection_submission_dir": str(destination.resolve()),
        "detection_submission_zip": str(archive.resolve()),
        "detection_submission_predictions": len(predictions),
        "detection_submission_placeholder_classes": placeholder_classes,
    }


def _average_precision(
    recall: np.ndarray,
    precision: np.ndarray,
    use_07_metric: bool = False,
) -> float:
    if use_07_metric:
        # The official DOTA-v1.0 devkit follows VOC-2007 11-point AP.
        return float(
            sum(
                (
                    precision[recall >= threshold].max()
                    if np.any(recall >= threshold)
                    else 0.0
                )
                for threshold in np.arange(0.0, 1.1, 0.1)
            )
            / 11.0
        )
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    changed = np.flatnonzero(recall[1:] != recall[:-1]) + 1
    return float(np.sum((recall[changed] - recall[changed - 1]) * precision[changed]))


def _class_ap(
    detections: Sequence[dict],
    ground_truth: Sequence[dict],
    class_id: int,
    threshold: float,
    use_07_metric: bool = False,
) -> tuple[float, float]:
    gt_by_parent: dict[str, list[dict]] = defaultdict(list)
    for item in ground_truth:
        if int(item["class_id"]) == class_id:
            gt_by_parent[str(item["parent"])].append(item)
    positive_count = sum(
        not bool(item.get("difficult", False))
        for items in gt_by_parent.values()
        for item in items
    )
    if positive_count == 0:
        return float("nan"), float("nan")
    matched = {
        parent: np.zeros(len(items), dtype=bool)
        for parent, items in gt_by_parent.items()
    }
    selected = sorted(
        (item for item in detections if int(item["class_id"]) == class_id),
        key=lambda item: float(item["score"]),
        reverse=True,
    )
    true_positive: list[float] = []
    false_positive: list[float] = []
    for detection in selected:
        parent = str(detection["parent"])
        candidates = gt_by_parent.get(parent, ())
        overlaps = [
            polygon_iou(detection["polygon"], item["polygon"]) for item in candidates
        ]
        best_index = int(np.argmax(overlaps)) if overlaps else -1
        best_overlap = overlaps[best_index] if overlaps else 0.0
        if best_overlap >= threshold and bool(
            candidates[best_index].get("difficult", False)
        ):
            continue
        is_match = (
            best_overlap >= threshold
            and best_index >= 0
            and not matched[parent][best_index]
        )
        true_positive.append(float(is_match))
        false_positive.append(float(not is_match))
        if is_match:
            matched[parent][best_index] = True
    if not true_positive:
        return 0.0, 0.0
    tp = np.cumsum(np.asarray(true_positive))
    fp = np.cumsum(np.asarray(false_positive))
    recall = tp / max(positive_count, 1)
    precision = tp / np.maximum(tp + fp, 1.0)
    return _average_precision(
        recall,
        precision,
        use_07_metric=use_07_metric,
    ), float(recall[-1])


def _group_geometry_records(
    records: Sequence[dict],
) -> dict[tuple[str, int], list[dict]]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for item in records:
        grouped[(str(item["parent"]), int(item["class_id"]))].append(item)
    return grouped


def _geometry_overlap_groups(
    detections: Sequence[dict],
    ground_truth: Sequence[dict],
) -> list[dict]:
    """Cache pairwise geometry once for all score-neutral diagnostics."""

    prediction_groups = _group_geometry_records(detections)
    truth_groups = _group_geometry_records(ground_truth)
    groups: list[dict] = []
    for key in prediction_groups.keys() | truth_groups.keys():
        predictions = list(prediction_groups.get(key, ()))
        targets = list(truth_groups.get(key, ()))
        iou = np.zeros((len(predictions), len(targets)), dtype=np.float32)
        prediction_coverage = np.zeros_like(iou)
        target_coverage = np.zeros_like(iou)
        for prediction_index, prediction in enumerate(predictions):
            for target_index, target in enumerate(targets):
                intersection, prediction_area, target_area = _polygon_overlap_geometry(
                    prediction["polygon"],
                    target["polygon"],
                )
                iou[prediction_index, target_index] = intersection / max(
                    prediction_area + target_area - intersection,
                    1.0e-8,
                )
                prediction_coverage[prediction_index, target_index] = (
                    intersection / max(prediction_area, 1.0e-8)
                )
                target_coverage[prediction_index, target_index] = intersection / max(
                    target_area, 1.0e-8
                )
        groups.append(
            {
                "predictions": predictions,
                "targets": targets,
                "regular_indices": np.asarray(
                    [
                        index
                        for index, item in enumerate(targets)
                        if not bool(item.get("difficult", False))
                    ],
                    dtype=np.int64,
                ),
                "difficult_indices": np.asarray(
                    [
                        index
                        for index, item in enumerate(targets)
                        if bool(item.get("difficult", False))
                    ],
                    dtype=np.int64,
                ),
                "iou": iou,
                "prediction_coverage": prediction_coverage,
                "target_coverage": target_coverage,
            }
        )
    return groups


def _score_neutral_geometry_at_threshold(
    geometry_groups: Sequence[dict],
    threshold: float,
) -> dict[str, float]:
    """Match boxes geometrically without reading prediction confidence.

    Candidate pairs are sorted only by polygon IoU and greedily assigned
    one-to-one inside each source-image/class group. Predictions whose best
    overlap belongs to a difficult object are ignored, matching the intent of
    the official DOTA protocol. These diagnostics deliberately run before
    score-based NMS so confidence cannot alter their ordering.
    """

    true_positive = 0
    prediction_count = 0
    ground_truth_count = 0
    matched_ious: list[float] = []
    for group in geometry_groups:
        iou = group["iou"]
        regular_indices = group["regular_indices"]
        difficult_indices = group["difficult_indices"]
        ground_truth_count += int(regular_indices.size)
        best_regular = (
            iou[:, regular_indices].max(axis=1)
            if regular_indices.size
            else np.zeros(iou.shape[0], dtype=np.float32)
        )
        best_difficult = (
            iou[:, difficult_indices].max(axis=1)
            if difficult_indices.size
            else np.zeros(iou.shape[0], dtype=np.float32)
        )
        valid_mask = ~((best_difficult >= threshold) & (best_difficult > best_regular))
        valid_indices = np.flatnonzero(valid_mask)
        prediction_count += int(valid_indices.size)
        regular_iou = iou[np.ix_(valid_indices, regular_indices)]
        candidate_predictions, candidate_truth = np.nonzero(regular_iou >= threshold)
        candidates = sorted(
            (
                (
                    float(regular_iou[prediction_index, truth_index]),
                    int(prediction_index),
                    int(truth_index),
                )
                for prediction_index, truth_index in zip(
                    candidate_predictions,
                    candidate_truth,
                )
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        used_predictions: set[int] = set()
        used_truth: set[int] = set()
        for overlap, prediction_index, truth_index in candidates:
            if prediction_index in used_predictions or truth_index in used_truth:
                continue
            used_predictions.add(prediction_index)
            used_truth.add(truth_index)
            matched_ious.append(float(overlap))
        true_positive += len(used_predictions)

    false_positive = prediction_count - true_positive
    false_negative = ground_truth_count - true_positive
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "matched_mean_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "true_positive": float(true_positive),
        "false_positive": float(false_positive),
        "false_negative": float(false_negative),
        "predictions": float(prediction_count),
        "ground_truth": float(ground_truth_count),
    }


def _instance_merge_split_metrics(
    geometry_groups: Sequence[dict],
    association_threshold: float = 0.5,
) -> dict[str, float]:
    """Measure instance topology errors independently of confidence.

    A prediction is a merge when it covers at least ``association_threshold``
    of two or more non-difficult GT instances. A GT instance is split when
    two or more predicted boxes lie inside at least that fraction of their
    own area. The asymmetric coverage definitions still detect a large box
    containing two objects and two small fragments contained by one object.
    """

    prediction_count = sum(len(group["predictions"]) for group in geometry_groups)
    ground_truth_count = sum(
        int(group["regular_indices"].size) for group in geometry_groups
    )
    merged_predictions = 0
    split_ground_truth = 0
    for group in geometry_groups:
        regular_indices = group["regular_indices"]
        if not regular_indices.size:
            continue
        target_coverage = group["target_coverage"][:, regular_indices]
        prediction_coverage = group["prediction_coverage"][:, regular_indices]
        merged_predictions += int(
            np.sum(np.sum(target_coverage >= association_threshold, axis=1) >= 2)
        )
        split_ground_truth += int(
            np.sum(np.sum(prediction_coverage >= association_threshold, axis=0) >= 2)
        )
    return {
        "detection_instance_merge_rate": float(
            merged_predictions / max(prediction_count, 1)
        ),
        "detection_instance_split_rate": float(
            split_ground_truth / max(ground_truth_count, 1)
        ),
        "detection_merged_predictions": float(merged_predictions),
        "detection_split_ground_truth": float(split_ground_truth),
        "detection_instance_association_threshold": float(association_threshold),
    }


def dota_oriented_metrics(
    detections: Sequence[dict],
    ground_truth: Sequence[dict],
    nms_threshold: float = 0.1,
    class_names: Sequence[str] = DOTA_CLASS_NAMES,
) -> dict[str, float]:
    # DataLoader sharding may repeat its final sample; de-duplicate GT by the
    # stable parent/object identifier before calculating official AP.
    unique_ground_truth = {
        (str(item["parent"]), int(item["object_id"])): item for item in ground_truth
    }
    truth = list(unique_ground_truth.values())
    predictions = _oriented_nms(detections, float(nms_threshold))
    thresholds = [round(0.5 + 0.05 * index, 2) for index in range(10)]
    per_threshold: dict[float, list[float]] = {}
    recalls50: list[float] = []
    metrics: dict[str, float] = {
        "detection_predictions": float(len(predictions)),
        "detection_ground_truth": float(
            sum(not item.get("difficult", False) for item in truth)
        ),
    }
    for threshold in thresholds:
        class_aps: list[float] = []
        for class_id, name in enumerate(class_names):
            ap, recall = _class_ap(
                predictions,
                truth,
                class_id,
                threshold,
                use_07_metric=threshold == 0.5,
            )
            if math.isfinite(ap):
                class_aps.append(ap)
                if threshold == 0.5:
                    metrics[f"detection_{name.replace('-', '_')}_ap50"] = ap
                    recalls50.append(recall)
        per_threshold[threshold] = class_aps
    metrics["detection_map50"] = float(np.mean(per_threshold[0.5]))
    metrics["detection_map75"] = float(np.mean(per_threshold[0.75]))
    metrics["detection_map50_95"] = float(
        np.mean([np.mean(per_threshold[threshold]) for threshold in thresholds])
    )
    metrics["detection_recall50"] = float(np.mean(recalls50)) if recalls50 else 0.0
    geometry_groups = _geometry_overlap_groups(detections, truth)
    for threshold, suffix in ((0.5, "0_5"), (0.75, "0_75")):
        geometry = _score_neutral_geometry_at_threshold(
            geometry_groups,
            threshold,
        )
        for name in (
            "precision",
            "recall",
            "f1",
            "matched_mean_iou",
            "true_positive",
            "false_positive",
            "false_negative",
        ):
            metrics[f"detection_geometry_{name}_at_{suffix}"] = geometry[name]
    metrics.update(
        _instance_merge_split_metrics(
            geometry_groups,
            association_threshold=0.5,
        )
    )
    return metrics
