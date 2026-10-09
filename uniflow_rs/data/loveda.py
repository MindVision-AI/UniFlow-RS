"""Data components: loveda."""

from __future__ import annotations

import math
from pathlib import (
    Path,
)
from typing import (
    Mapping,
    Sequence,
)
import cv2
import numpy as np
import torch
from PIL import (
    Image,
)
from scipy.ndimage import (
    distance_transform_edt,
)
from torch import (
    Tensor,
)
from torch.nn import (
    functional as F,
)
from torch.utils.data import (
    Dataset,
)
from uniflow_rs.data.contracts import (
    LOVEDA_CLASS_NAMES,
    LOVEDA_CODE_PALETTE,
    LOVEDA_CODE_PALETTE_NAME,
    SEMANTIC_TASK,
    resolve_loveda_code_palette,
)
from uniflow_rs.data.transforms import (
    _augment_pair,
    _image_tensor,
    _paths_by_stem,
)


def fill_loveda_ignore_labels(
    labels: np.ndarray,
    fallback_label: int = 1,
) -> np.ndarray:
    """Fill no-data ID 0 only for constructing the RGB generation endpoint.

    The original labels stay unchanged for the validity mask and standard
    LoveDA evaluation, where ID 0 is ignored.
    """

    labels = np.asarray(labels, dtype=np.int64)
    if labels.size and (labels.min() < 0 or labels.max() >= len(LOVEDA_CODE_PALETTE)):
        raise ValueError(
            "LoveDA labels must be in [0, 7], got range "
            f"[{labels.min()}, {labels.max()}]"
        )
    if not 1 <= int(fallback_label) <= 7:
        raise ValueError(f"fallback_label must be in [1, 7], got {fallback_label}")
    ignored = labels == 0
    if not bool(ignored.any()):
        return labels.copy()
    if bool(ignored.all()):
        return np.full_like(labels, int(fallback_label))

    # Non-zero inputs receive indices of their nearest zero input. Here the
    # ignored pixels are True/1 and valid class pixels are False/0, so these
    # coordinates point to the nearest evaluated class.
    nearest_indices = distance_transform_edt(
        ignored,
        return_distances=False,
        return_indices=True,
    )
    filled = labels.copy()
    filled[ignored] = labels[
        nearest_indices[0][ignored],
        nearest_indices[1][ignored],
    ]
    if bool((filled == 0).any()):
        raise RuntimeError("Failed to fill every LoveDA ignore pixel")
    return filled


def loveda_labels_to_rgb(
    labels: np.ndarray,
    code_palette: str = LOVEDA_CODE_PALETTE_NAME,
) -> np.ndarray:
    """Convert class IDs to the internal VAE/flow RGB codebook."""

    labels = np.asarray(labels, dtype=np.int64)
    if labels.size and (labels.min() < 1 or labels.max() > 7):
        raise ValueError(
            "LoveDA RGB generation labels must be in [1, 7], got range "
            f"[{labels.min()}, {labels.max()}]. Fill ignore ID 0 first."
        )
    return resolve_loveda_code_palette(code_palette)[labels]


def _loveda_palette_distance(
    decoded: Tensor,
    code_palette: str = LOVEDA_CODE_PALETTE_NAME,
) -> Tensor:
    if decoded.ndim != 4 or decoded.shape[1] != 3:
        raise ValueError(
            "Expected decoded LoveDA RGB tensor with shape [B,3,H,W], "
            f"got {tuple(decoded.shape)}"
        )
    palette = (
        torch.as_tensor(
            resolve_loveda_code_palette(code_palette)[1:],
            device=decoded.device,
            dtype=torch.float32,
        )
        .div(127.5)
        .sub(1.0)
    )
    # The original FLUX decoder has no final tanh, so raw values are not
    # mathematically bounded even though training images use [-1, 1].
    # Binary UniFlow-RS inference already clamps before converting to a mask;
    # semantic palette matching must apply the same range contract.
    pixels = decoded.float().clamp(-1.0, 1.0).permute(0, 2, 3, 1)
    distance = (pixels.unsqueeze(-2) - palette.view(1, 1, 1, 7, 3)).square().sum(dim=-1)
    return distance


def decode_loveda_palette(
    decoded: Tensor,
    code_palette: str = LOVEDA_CODE_PALETTE_NAME,
) -> Tensor:
    """Map a decoded normalized RGB image to original LoveDA IDs 1..7."""

    return (
        _loveda_palette_distance(decoded, code_palette=code_palette)
        .argmin(dim=-1)
        .add(1)
    )


def _clean_small_loveda_islands(
    labels: np.ndarray,
    uncertain: np.ndarray,
    *,
    min_area: int,
    min_uncertain_fraction: float,
    surround_min_fraction: float,
) -> np.ndarray:
    """Replace tiny components only when one neighboring class dominates."""

    source = np.asarray(labels, dtype=np.int64)
    result = source.copy()
    uncertain = np.asarray(uncertain, dtype=bool)
    kernel = np.ones((3, 3), dtype=np.uint8)
    for class_id in range(1, 8):
        binary = (source == class_id).astype(np.uint8)
        component_count, component_map, statistics, _ = (
            cv2.connectedComponentsWithStats(binary, connectivity=8)
        )
        for component_id in range(1, component_count):
            area = int(statistics[component_id, cv2.CC_STAT_AREA])
            if area >= min_area:
                continue
            component = component_map == component_id
            if float(uncertain[component].mean()) < min_uncertain_fraction:
                continue
            ring = (
                cv2.dilate(component.astype(np.uint8), kernel, iterations=1).astype(
                    bool
                )
                & ~component
            )
            surrounding = result[ring]
            surrounding = surrounding[surrounding != class_id]
            if surrounding.size == 0:
                continue
            values, counts = np.unique(surrounding, return_counts=True)
            dominant_index = int(counts.argmax())
            if (
                float(counts[dominant_index]) / float(surrounding.size)
                < surround_min_fraction
            ):
                continue
            result[component] = int(values[dominant_index])
    return result


def decode_loveda_palette_postprocessed(
    decoded: Tensor,
    config: Mapping[str, object] | None = None,
    code_palette: str = LOVEDA_CODE_PALETTE_NAME,
) -> tuple[Tensor, Tensor]:
    """Return raw and conservatively postprocessed LoveDA predictions.

    The model still generates RGB and the raw result is always the ordinary
    nearest-palette prediction. Postprocessing can reject low-confidence
    barren predictions, changes ambiguous pixels supported by a strong local
    majority, and removes configurable tiny connected components surrounded
    by one dominant class.
    """

    distance = _loveda_palette_distance(
        decoded,
        code_palette=code_palette,
    )
    raw = distance.argmin(dim=-1).add(1)
    options = dict(config or {})
    if not bool(options.get("enabled", False)):
        return raw, raw

    barren_options = options.get("barren_filter", {})
    if barren_options is None:
        barren_options = {}
    if not isinstance(barren_options, Mapping):
        raise ValueError("semantic_postprocess.barren_filter must be a mapping")
    barren_options = dict(barren_options)
    barren_filter_enabled = bool(barren_options.get("enabled", False))
    barren_confident = None
    barren_fallback = None
    if barren_filter_enabled:
        barren_max_distance = float(barren_options.get("max_color_distance", 0.08))
        barren_min_margin = float(barren_options.get("min_color_margin", 0.25))
        if not math.isfinite(barren_max_distance) or barren_max_distance < 0.0:
            raise ValueError(
                "semantic_postprocess.barren_filter.max_color_distance "
                "must be finite and >= 0"
            )
        if not math.isfinite(barren_min_margin) or barren_min_margin < 0.0:
            raise ValueError(
                "semantic_postprocess.barren_filter.min_color_margin "
                "must be finite and >= 0"
            )

        # LoveDA class ID 5 is barren, corresponding to palette index 4.
        # If its decoded RGB is not both close to the canonical orange code
        # and clearly ahead of every alternative, exclude barren and select
        # the nearest of the remaining six classes. Distances are squared in
        # normalized RGB.
        barren_distance = distance[..., 4]
        non_barren_distance = torch.cat(
            (distance[..., :4], distance[..., 5:]),
            dim=-1,
        )
        best_non_barren_distance, best_non_barren_index = non_barren_distance.min(
            dim=-1
        )
        barren_fallback = torch.where(
            best_non_barren_index < 4,
            best_non_barren_index + 1,
            best_non_barren_index + 2,
        ).to(dtype=raw.dtype)
        barren_confident = (barren_distance <= barren_max_distance) & (
            best_non_barren_distance - barren_distance >= barren_min_margin
        )

    nearest = distance.topk(k=2, dim=-1, largest=False).values
    uncertain = (nearest[..., 0] > float(options.get("max_color_distance", 0.15))) | (
        nearest[..., 1] - nearest[..., 0] < float(options.get("min_color_margin", 0.15))
    )

    kernel_size = int(options.get("majority_kernel_size", 3))
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(
            "semantic_postprocess.majority_kernel_size must be a positive "
            f"odd integer, got {kernel_size}"
        )
    majority_min_count = int(options.get("majority_min_count", 5))
    majority_iterations = int(options.get("majority_iterations", 1))
    processed = raw
    if barren_filter_enabled:
        assert barren_confident is not None
        assert barren_fallback is not None
        processed = torch.where(
            processed.eq(5) & ~barren_confident,
            barren_fallback,
            processed,
        )
    kernel = torch.ones(
        7,
        1,
        kernel_size,
        kernel_size,
        device=decoded.device,
        dtype=torch.float32,
    )
    for _ in range(max(majority_iterations, 0)):
        one_hot = (
            F.one_hot(
                processed.sub(1),
                num_classes=7,
            )
            .permute(0, 3, 1, 2)
            .float()
        )
        neighbor_counts = F.conv2d(
            one_hot,
            kernel,
            padding=kernel_size // 2,
            groups=7,
        )
        majority_count, majority_index = neighbor_counts.max(dim=1)
        majority = majority_index.add(1)
        replace = (
            uncertain & (majority_count >= majority_min_count) & (majority != processed)
        )
        processed = torch.where(replace, majority, processed)

    min_island_area = int(options.get("min_island_area", 0))
    if min_island_area > 1:
        min_uncertain_fraction = float(
            options.get("island_min_uncertain_fraction", 0.0)
        )
        surround_min_fraction = float(options.get("island_surround_min_fraction", 0.6))
        if not 0.0 <= min_uncertain_fraction <= 1.0:
            raise ValueError(
                "semantic_postprocess.island_min_uncertain_fraction must "
                "be in [0, 1]"
            )
        if not 0.0 <= surround_min_fraction <= 1.0:
            raise ValueError(
                "semantic_postprocess.island_surround_min_fraction must " "be in [0, 1]"
            )
        processed_cpu = processed.detach().cpu().numpy()
        uncertain_cpu = uncertain.detach().cpu().numpy()
        cleaned = [
            _clean_small_loveda_islands(
                labels,
                sample_uncertain,
                min_area=min_island_area,
                min_uncertain_fraction=min_uncertain_fraction,
                surround_min_fraction=surround_min_fraction,
            )
            for labels, sample_uncertain in zip(
                processed_cpu,
                uncertain_cpu,
                strict=True,
            )
        ]
        processed = torch.from_numpy(np.stack(cleaned)).to(
            device=raw.device,
            dtype=raw.dtype,
        )
    # Majority voting or island replacement must not reintroduce a barren
    # label at a pixel that failed the class-specific confidence gate.
    if barren_filter_enabled:
        assert barren_confident is not None
        assert barren_fallback is not None
        processed = torch.where(
            processed.eq(5) & ~barren_confident,
            barren_fallback,
            processed,
        )
    return raw, processed


def loveda_confusion(
    prediction: Tensor,
    target: Tensor,
) -> Tensor:
    """Return a 7x7 confusion matrix, ignoring original LoveDA label 0."""

    prediction = prediction.long()
    target = target.long()
    valid = (target >= 1) & (target <= 7)
    if not bool(valid.any()):
        return torch.zeros(
            7,
            7,
            device=target.device,
            dtype=torch.float64,
        )
    target_index = target[valid] - 1
    prediction_index = prediction[valid].clamp(1, 7) - 1
    bins = torch.bincount(
        target_index * 7 + prediction_index,
        minlength=49,
    )
    return bins.reshape(7, 7).to(torch.float64)


def loveda_metrics(confusion: Tensor) -> dict[str, float]:
    confusion = confusion.detach().cpu().double()
    true_positive = confusion.diag()
    ground_truth = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)
    union = ground_truth + predicted - true_positive
    class_iou = true_positive / union.clamp_min(1.0)
    class_dice = 2.0 * true_positive / (ground_truth + predicted).clamp_min(1.0)
    class_accuracy = true_positive / ground_truth.clamp_min(1.0)
    valid_classes = ground_truth > 0
    mean_iou = float(class_iou[valid_classes].mean())
    mean_dice = float(class_dice[valid_classes].mean())
    pixel_accuracy = float(true_positive.sum() / confusion.sum().clamp_min(1.0))
    mean_accuracy = float(class_accuracy[valid_classes].mean())
    metrics = {
        "semantic_iou": mean_iou,
        "semantic_dice": mean_dice,
        "semantic_miou": mean_iou,
        "semantic_mdice": mean_dice,
        "semantic_pixel_accuracy": pixel_accuracy,
        "semantic_mean_accuracy": mean_accuracy,
    }
    for index, name in enumerate(LOVEDA_CLASS_NAMES):
        metrics[f"semantic_{name}_iou"] = float(class_iou[index])
        metrics[f"semantic_{name}_dice"] = float(class_dice[index])
    return metrics


def _resolve_loveda_split(root: Path, split: str) -> Path:
    direct = root / str(split)
    if direct.is_dir():
        return direct
    matches = [
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.casefold() == str(split).casefold()
    ]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Could not resolve LoveDA split '{split}' below '{root}'")


class LoveDASemanticDataset(Dataset):
    """LoveDA images and labels resized together to the model resolution."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        resolution: int = 512,
        domains: Sequence[str] = ("Urban", "Rural"),
        augment: bool = False,
        source_size: int = 1024,
        code_palette: str = LOVEDA_CODE_PALETTE_NAME,
    ):
        self.root = Path(root).expanduser()
        self.split_dir = _resolve_loveda_split(self.root, split)
        self.resolution = int(resolution)
        self.augment = bool(augment)
        self.source_size = int(source_size)
        self.code_palette = str(code_palette).strip().casefold()
        resolve_loveda_code_palette(self.code_palette)
        if self.resolution < 1 or self.source_size < 1:
            raise ValueError("LoveDA source_size and resolution must both be positive")
        self.samples: list[dict[str, Path | str]] = []
        for configured_domain in domains:
            domain_candidates = [
                path
                for path in self.split_dir.iterdir()
                if path.is_dir()
                and path.name.casefold() == str(configured_domain).casefold()
            ]
            if len(domain_candidates) != 1:
                raise FileNotFoundError(
                    f"Could not resolve LoveDA domain '{configured_domain}' "
                    f"below '{self.split_dir}'"
                )
            domain_dir = domain_candidates[0]
            images = _paths_by_stem(domain_dir / "images_png")
            masks = _paths_by_stem(domain_dir / "masks_png")
            if set(images) != set(masks):
                raise RuntimeError(
                    "LoveDA image/mask filenames are not paired for "
                    f"{self.split_dir.name}/{domain_dir.name}: "
                    f"missing={len(set(images) - set(masks))}, "
                    f"extra={len(set(masks) - set(images))}"
                )
            self.samples.extend(
                {
                    "image": images[stem],
                    "mask": masks[stem],
                    "domain": domain_dir.name,
                }
                for stem in sorted(images)
            )
        if not self.samples:
            raise FileNotFoundError(
                f"No paired LoveDA samples found below '{self.split_dir}'"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        sample = self.samples[index]
        image_path = Path(sample["image"])
        mask_path = Path(sample["mask"])
        with Image.open(image_path) as source:
            image_source = source.convert("RGB")
            source_image_size = image_source.size
            if source_image_size != (self.resolution, self.resolution):
                image_source = image_source.resize(
                    (self.resolution, self.resolution),
                    Image.Resampling.BILINEAR,
                )
            image = np.asarray(image_source, dtype=np.float32) / 255.0
        with Image.open(mask_path) as source:
            mask_source = source.convert("L")
            source_mask_size = mask_source.size
            if source_mask_size != (self.resolution, self.resolution):
                mask_source = mask_source.resize(
                    (self.resolution, self.resolution),
                    Image.Resampling.NEAREST,
                )
            labels = np.asarray(mask_source, dtype=np.uint8)
        if source_image_size != source_mask_size:
            raise RuntimeError(
                f"LoveDA image/mask size mismatch: {image_path}, {mask_path}"
            )
        if source_image_size != (self.source_size, self.source_size):
            raise RuntimeError(
                f"Expected LoveDA {self.source_size}x{self.source_size} input, "
                f"got {source_image_size[0]}x{source_image_size[1]} "
                f"for '{image_path}'"
            )
        if self.augment:
            image, _, labels = _augment_pair(image, image, labels)
        if labels.size and (labels.min() < 0 or labels.max() > 7):
            raise ValueError(f"LoveDA labels must be in [0, 7]: '{mask_path}'")
        label_tensor = torch.from_numpy(
            np.ascontiguousarray(labels[None].astype(np.float32))
        )
        loss_mask = torch.from_numpy(
            np.ascontiguousarray((labels != 0)[None].astype(np.float32))
        )
        generation_labels = fill_loveda_ignore_labels(labels)
        target_rgb_array = (
            loveda_labels_to_rgb(
                generation_labels,
                code_palette=self.code_palette,
            ).astype(np.float32)
            / 255.0
        )
        target_rgb = _image_tensor(target_rgb_array)
        if not bool((target_rgb >= -1.0).all() and (target_rgb <= 1.0).all()):
            raise RuntimeError(
                f"Normalized LoveDA palette target is outside [-1, 1]: "
                f"'{mask_path}'"
            )
        name = f"{sample['domain']}_{image_path.stem}"
        image_tensor = _image_tensor(image)
        return {
            "before": image_tensor,
            "after": image_tensor,
            "mask": label_tensor,
            "loss_mask": loss_mask,
            "target_rgb": target_rgb,
            "task": SEMANTIC_TASK,
            "name": name,
        }
