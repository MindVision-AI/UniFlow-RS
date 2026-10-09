"""Data components: transforms."""

from __future__ import annotations

import random
from pathlib import (
    Path,
)
from typing import (
    Mapping,
)
import cv2
import numpy as np
import torch
from PIL import (
    Image,
)
from torch import (
    Tensor,
)
from uniflow_rs.data.contracts import (
    _PAIRED_VIEW_INDEX,
    _RASTER_EXTENSIONS,
)


def postprocess_binary_prediction(
    prediction: Tensor,
    valid_mask: Tensor | None = None,
    config: Mapping | None = None,
) -> Tensor:
    """Conservatively clean binary masks during validation and inference.

    The operation is deliberately non-differentiable and CPU based because it
    is intended only for metrics and exported predictions.  It can close
    one-pixel gaps, remove tiny foreground islands, and fill small background
    holes that are fully enclosed by foreground.  Components connected to an
    image boundary or an invalid region are never interpreted as holes.
    """

    if prediction.ndim != 3:
        raise ValueError(
            "prediction must have shape [B,H,W], got " f"{tuple(prediction.shape)}"
        )
    options = dict(config or {})
    binary = prediction.bool()
    if valid_mask is None:
        valid = torch.ones_like(binary, dtype=torch.bool)
    else:
        if valid_mask.ndim == 4 and valid_mask.shape[1] == 1:
            valid_mask = valid_mask[:, 0]
        if valid_mask.shape != binary.shape:
            raise ValueError(
                "valid_mask must match prediction shape, got "
                f"{tuple(valid_mask.shape)} and {tuple(binary.shape)}"
            )
        valid = valid_mask.bool()
    binary = binary & valid
    if not bool(options.get("enabled", False)):
        return binary

    min_component_area = int(options.get("min_component_area", 0))
    max_hole_area = int(options.get("max_hole_area", 0))
    closing_kernel = int(options.get("closing_kernel", 0))
    if min_component_area < 0:
        raise ValueError("min_component_area must be non-negative")
    if max_hole_area < 0:
        raise ValueError("max_hole_area must be non-negative")
    if closing_kernel < 0 or (closing_kernel > 1 and closing_kernel % 2 == 0):
        raise ValueError("closing_kernel must be 0, 1, or a positive odd value")

    binary_np = binary.detach().cpu().numpy().astype(np.uint8)
    valid_np = valid.detach().cpu().numpy().astype(np.uint8)
    result = np.zeros_like(binary_np, dtype=np.uint8)
    closing_structure = (
        np.ones((closing_kernel, closing_kernel), dtype=np.uint8)
        if closing_kernel > 1
        else None
    )
    adjacency_structure = np.ones((3, 3), dtype=np.uint8)

    for index in range(binary_np.shape[0]):
        current = binary_np[index] & valid_np[index]
        current_valid = valid_np[index]

        if closing_structure is not None:
            current = cv2.morphologyEx(
                current,
                cv2.MORPH_CLOSE,
                closing_structure,
            )
            current &= current_valid

        if min_component_area > 1:
            component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
                current, connectivity=8
            )
            cleaned = np.zeros_like(current)
            for component in range(1, component_count):
                if stats[component, cv2.CC_STAT_AREA] >= min_component_area:
                    cleaned[labels == component] = 1
            current = cleaned

        if max_hole_area > 0:
            background = ((current == 0) & (current_valid > 0)).astype(np.uint8)
            component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
                background, connectivity=8
            )
            invalid_neighbourhood = cv2.dilate(
                (current_valid == 0).astype(np.uint8),
                adjacency_structure,
            ).astype(bool)
            height, width = current.shape
            for component in range(1, component_count):
                area = int(stats[component, cv2.CC_STAT_AREA])
                if area > max_hole_area:
                    continue
                x = int(stats[component, cv2.CC_STAT_LEFT])
                y = int(stats[component, cv2.CC_STAT_TOP])
                component_width = int(stats[component, cv2.CC_STAT_WIDTH])
                component_height = int(stats[component, cv2.CC_STAT_HEIGHT])
                touches_image_boundary = (
                    x == 0
                    or y == 0
                    or x + component_width == width
                    or y + component_height == height
                )
                component_mask = labels == component
                touches_invalid = bool(np.any(component_mask & invalid_neighbourhood))
                if not touches_image_boundary and not touches_invalid:
                    current[component_mask] = 1

        result[index] = current & current_valid

    return torch.from_numpy(result.astype(bool)).to(prediction.device)


def _paths_by_stem(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    paths = {
        path.stem.casefold(): path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.casefold() in _RASTER_EXTENSIONS
    }
    if not paths:
        raise FileNotFoundError(f"No raster images found in '{directory}'")
    return paths


def _normalized_rgb(path: Path, resolution: int) -> np.ndarray:
    with Image.open(path) as source:
        image = source.convert("RGB")
        if image.size != (resolution, resolution):
            image = image.resize(
                (resolution, resolution),
                Image.Resampling.BILINEAR,
            )
        return np.asarray(image, dtype=np.float32) / 255.0


def _binary_mask(path: Path, resolution: int) -> np.ndarray:
    with Image.open(path) as source:
        mask = source.convert("L")
        if mask.size != (resolution, resolution):
            mask = mask.resize(
                (resolution, resolution),
                Image.Resampling.NEAREST,
            )
        return np.asarray(mask, dtype=np.uint8) > 127


def _augment_pair(
    before: np.ndarray,
    after: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transform = _sample_geometric_augmentation()
    return tuple(
        _apply_geometric_augmentation(array, transform)
        for array in (before, after, target)
    )


def _sample_geometric_augmentation() -> tuple[int, bool, bool]:
    return (
        int(np.random.randint(0, 4)),
        bool(np.random.rand() < 0.5),
        bool(np.random.rand() < 0.5),
    )


def _paired_view_index(
    index: int | tuple[str, int, int],
) -> tuple[int, tuple[int, bool, bool] | None]:
    """Resolve a sampler-provided view index and shared augmentation.

    Pair-aware training batches attach one transform code to both the RGB and
    SAR view of a physical scene.  Ordinary integer indexing keeps the older
    independently sampled augmentation behavior used by validation and tests.
    """

    if not isinstance(index, tuple):
        return int(index), None
    if len(index) != 3 or index[0] != _PAIRED_VIEW_INDEX:
        raise TypeError(f"Unsupported paired dataset index: {index!r}")
    transform_code = int(index[2])
    if not 0 <= transform_code < 16:
        raise ValueError(
            f"Paired augmentation code must be in [0, 15], got {transform_code}"
        )
    return (
        int(index[1]),
        (
            transform_code % 4,
            bool((transform_code // 4) % 2),
            bool((transform_code // 8) % 2),
        ),
    )


def _paired_modality_batch_indices(
    physical_count: int,
    batch_size: int,
    rng: random.Random,
) -> list[tuple[str, int, int]]:
    """Return complete RGB/SAR pairs for one per-device training batch."""

    if batch_size < 2 or batch_size % 2 != 0:
        raise ValueError(
            "Paired RGB/SAR batches require an even batch_size_per_gpu >= 2; "
            f"got {batch_size}"
        )
    if physical_count < 1:
        raise ValueError("Paired RGB/SAR dataset cannot be empty")
    indices: list[tuple[str, int, int]] = []
    for _ in range(batch_size // 2):
        physical_index = rng.randrange(physical_count)
        transform_code = rng.randrange(16)
        views = [2 * physical_index, 2 * physical_index + 1]
        if rng.random() < 0.5:
            views.reverse()
        indices.extend(
            (_PAIRED_VIEW_INDEX, view_index, transform_code) for view_index in views
        )
    return indices


def _apply_geometric_augmentation(
    array: np.ndarray,
    transform: tuple[int, bool, bool],
) -> np.ndarray:
    """Apply one shared right-angle rotation/flip transform."""

    rotation, horizontal_flip, vertical_flip = transform
    transformed = np.rot90(array, rotation, axes=(0, 1))
    if horizontal_flip:
        transformed = np.fliplr(transformed)
    if vertical_flip:
        transformed = np.flipud(transformed)
    return transformed


def _image_tensor(array: np.ndarray) -> Tensor:
    return (
        torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
        .float()
        .mul(2.0)
        .sub(1.0)
    )


def _mask_tensor(array: np.ndarray) -> Tensor:
    return torch.from_numpy(np.ascontiguousarray(array[None].astype(np.float32)))


def _binary_target_rgb(mask: Tensor) -> Tensor:
    return mask.repeat(3, 1, 1).mul(2.0).sub(1.0)
