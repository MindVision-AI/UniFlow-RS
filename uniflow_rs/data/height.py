"""Data components: height."""

from __future__ import annotations

import json
import math
import random
import re
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
from torch import (
    Tensor,
)
from torch.utils.data import (
    Dataset,
)
from uniflow_rs.data.contracts import (
    HEIGHT_TASK,
)
from uniflow_rs.data.transforms import (
    _apply_geometric_augmentation,
    _augment_pair,
    _image_tensor,
    _mask_tensor,
    _normalized_rgb,
    _paired_modality_batch_indices,
    _paired_view_index,
    _paths_by_stem,
    _sample_geometric_augmentation,
)


def height_to_target_rgb(
    height_meters: Tensor,
    max_height_meters: float = 25.5,
    height_codec: str = "linear",
) -> Tensor:
    """Encode a metric height map as the grayscale RGB expected by FLUX VAE."""

    maximum = float(max_height_meters)
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError(
            "max_height_meters must be finite and > 0, got " f"{max_height_meters!r}"
        )
    if height_meters.ndim != 3 or height_meters.shape[0] != 1:
        raise ValueError(
            "Expected one height channel [1,H,W], got " f"{tuple(height_meters.shape)}"
        )
    codec = str(height_codec).casefold()
    normalized = height_meters.clamp(0.0, maximum)
    if codec == "linear":
        normalized = normalized.div(maximum)
    elif codec == "log1p":
        normalized = torch.log1p(normalized).div(math.log1p(maximum))
    else:
        raise ValueError(
            "height_codec must be 'linear' or 'log1p', got " f"{height_codec!r}"
        )
    return normalized.mul(2.0).sub(1.0).repeat(3, 1, 1)


def decoded_rgb_to_height(
    decoded: Tensor,
    max_height_meters: float = 25.5,
    straight_through_clamp: bool = False,
    height_codec: str = "linear",
) -> Tensor:
    """Decode a FLUX-VAE grayscale RGB result into a metric height map."""

    maximum = float(max_height_meters)
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError(
            "max_height_meters must be finite and > 0, got " f"{max_height_meters!r}"
        )
    if decoded.ndim != 4 or decoded.shape[1] != 3:
        raise ValueError(f"Expected decoded RGB [B,3,H,W], got {tuple(decoded.shape)}")
    bounded = decoded.clamp(-1.0, 1.0)
    if straight_through_clamp:
        # Keep physical values bounded in the forward pass while avoiding
        # zero gradients when an early decoded endpoint overshoots [-1, 1].
        bounded = decoded + (bounded - decoded).detach()
    codec = str(height_codec).casefold()
    if codec == "linear":
        # Preserve the original operation order for bitwise-compatible
        # Potsdam checkpoints and exact zero-loss round-trip tests.
        return bounded.mean(dim=1, keepdim=True).add(1.0).mul(0.5 * maximum)
    if codec == "log1p":
        normalized = bounded.mean(dim=1, keepdim=True).add(1.0).mul(0.5)
        return torch.expm1(normalized.mul(math.log1p(maximum)))
    raise ValueError(
        "height_codec must be 'linear' or 'log1p', got " f"{height_codec!r}"
    )


def postprocess_height_prediction(
    prediction: Tensor,
    settings: Mapping | None,
    max_height_meters: float,
) -> Tensor:
    """Apply the shared non-differentiable metric-height cleanup."""

    if prediction.ndim != 4 or prediction.shape[1] != 1:
        raise ValueError(
            "Expected height prediction [B,1,H,W], got " f"{tuple(prediction.shape)}"
        )
    options = dict(settings or {})
    if not bool(options.get("enabled", False)):
        return prediction
    maximum = float(max_height_meters)
    threshold = float(options.get("ground_threshold_meters", 0.0))
    minimum_area = int(options.get("min_component_area", 0))
    median_kernel = int(options.get("median_kernel", 1))
    median_weight = float(options.get("component_median_weight", 0.0))
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("max_height_meters must be finite and > 0")
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("Height ground threshold must be finite and >= 0")
    if minimum_area < 0:
        raise ValueError("Height minimum component area must be >= 0")
    if median_kernel < 1 or median_kernel % 2 == 0:
        raise ValueError("Height median kernel must be a positive odd integer")
    if not 0.0 <= median_weight <= 1.0:
        raise ValueError("Height component median weight must be in [0, 1]")

    # The calibrated joint-training configuration only applies a scalar
    # ground threshold. Keep that common path entirely on the accelerator;
    # CPU/OpenCV is needed only for component or spatial operations.
    if minimum_area <= 1 and median_kernel == 1 and median_weight == 0.0:
        bounded = torch.nan_to_num(
            prediction,
            nan=0.0,
            posinf=maximum,
            neginf=0.0,
        ).clamp(0.0, maximum)
        return torch.where(
            bounded >= threshold if threshold > 0.0 else bounded > 0.0,
            bounded,
            torch.zeros_like(bounded),
        )

    processed_batch: list[np.ndarray] = []
    for sample in prediction.detach().float().cpu().numpy()[:, 0]:
        raw = (
            np.nan_to_num(
                sample,
                nan=0.0,
                posinf=maximum,
                neginf=0.0,
            )
            .clip(0.0, maximum)
            .astype(np.float32, copy=False)
        )
        support = raw >= threshold if threshold > 0.0 else raw > 0.0
        if minimum_area > 1 and support.any():
            component_count, labels, statistics, _ = cv2.connectedComponentsWithStats(
                support.astype(np.uint8),
                connectivity=8,
            )
            keep = np.zeros_like(support, dtype=bool)
            for component in range(1, component_count):
                if statistics[component, cv2.CC_STAT_AREA] >= minimum_area:
                    keep |= labels == component
            support = keep
        smoothed = (
            cv2.medianBlur(raw, median_kernel) if median_kernel > 1 else raw.copy()
        )
        output = np.where(support, smoothed, 0.0).astype(np.float32)
        if median_weight > 0.0 and support.any():
            component_count, labels = cv2.connectedComponents(
                support.astype(np.uint8),
                connectivity=8,
            )
            for component in range(1, component_count):
                region = labels == component
                if not region.any():
                    continue
                median_height = float(np.median(output[region]))
                output[region] = (1.0 - median_weight) * output[
                    region
                ] + median_weight * median_height
        processed_batch.append(output)
    result = torch.from_numpy(np.stack(processed_batch)[:, None])
    return result.to(device=prediction.device, dtype=prediction.dtype)


def height_error_counts(
    prediction: Tensor,
    target: Tensor,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Accumulate height errors and DFC23-style delta threshold counts.

    Delta metrics use ``max(pred / target, target / pred) < 1.25 ** n``.
    Heights are non-negative. To make the official ratio finite at ground
    pixels, values are clamped to a small positive epsilon: a zero/zero pair
    therefore counts as correct, while a one-sided zero pair counts as wrong.
    Both all-valid-pixel and above-ground (target > 0.1 m) counts are kept.
    """

    prediction = prediction.float()
    target = target.float()
    if prediction.shape != target.shape:
        raise ValueError(
            "Height prediction and target must match, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    valid = (
        torch.ones_like(target, dtype=torch.bool)
        if valid_mask is None
        else valid_mask.to(
            device=target.device,
            dtype=torch.float32,
        )
        > 0.5
    )
    if valid.shape != target.shape:
        raise ValueError(
            "Height valid_mask must match target, got "
            f"{tuple(valid.shape)} and {tuple(target.shape)}"
        )
    full_error = prediction - target
    error = full_error[valid]
    elevated = (target > 0.1) & valid
    elevated_error = full_error[elevated]
    epsilon = torch.finfo(torch.float32).eps
    safe_prediction = prediction.clamp_min(0.0).clamp_min(epsilon)
    safe_target = target.clamp_min(0.0).clamp_min(epsilon)
    ratio = torch.maximum(
        safe_prediction / safe_target,
        safe_target / safe_prediction,
    )
    valid_ratio = ratio[valid]
    elevated_ratio = ratio[elevated]
    delta_thresholds = (1.25, 1.25**2, 1.25**3)
    delta_counts = tuple(
        (valid_ratio < threshold).sum() for threshold in delta_thresholds
    )
    elevated_delta_counts = tuple(
        (elevated_ratio < threshold).sum() for threshold in delta_thresholds
    )
    return torch.stack(
        (
            error.abs().sum(),
            error.square().sum(),
            error.sum(),
            torch.tensor(
                error.numel(),
                device=error.device,
                dtype=error.dtype,
            ),
            elevated_error.abs().sum(),
            elevated_error.square().sum(),
            torch.tensor(
                elevated_error.numel(),
                device=error.device,
                dtype=error.dtype,
            ),
            *delta_counts,
            *elevated_delta_counts,
        )
    ).to(torch.float64)


def height_metrics(
    counts: Tensor,
    max_height_meters: float = 25.5,
) -> dict[str, float]:
    """Convert accumulated height errors into metric validation scores."""

    (
        absolute_error,
        squared_error,
        signed_error,
        pixels,
        elevated_absolute_error,
        elevated_squared_error,
        elevated_pixels,
        delta1_pixels,
        delta2_pixels,
        delta3_pixels,
        elevated_delta1_pixels,
        elevated_delta2_pixels,
        elevated_delta3_pixels,
    ) = [float(value) for value in counts.detach().cpu()]
    pixels = max(pixels, 1.0)
    elevated_pixels = max(elevated_pixels, 1.0)
    mae = absolute_error / pixels
    rmse = math.sqrt(max(squared_error / pixels, 0.0))
    elevated_mae = elevated_absolute_error / elevated_pixels
    elevated_rmse = math.sqrt(max(elevated_squared_error / elevated_pixels, 0.0))
    delta1 = delta1_pixels / pixels
    delta2 = delta2_pixels / pixels
    delta3 = delta3_pixels / pixels
    elevated_delta1 = elevated_delta1_pixels / elevated_pixels
    elevated_delta2 = elevated_delta2_pixels / elevated_pixels
    elevated_delta3 = elevated_delta3_pixels / elevated_pixels
    maximum = float(max_height_meters)
    return {
        "height_mae": mae,
        "height_rmse": rmse,
        "height_bias": signed_error / pixels,
        "height_above_ground_mae": elevated_mae,
        "height_above_ground_rmse": elevated_rmse,
        "height_delta1": delta1,
        "height_delta2": delta2,
        "height_delta3": delta3,
        "height_delta1_percent": 100.0 * delta1,
        "height_delta2_percent": 100.0 * delta2,
        "height_delta3_percent": 100.0 * delta3,
        "height_above_ground_delta1": elevated_delta1,
        "height_above_ground_delta2": elevated_delta2,
        "height_above_ground_delta3": elevated_delta3,
        "height_above_ground_delta1_percent": 100.0 * elevated_delta1,
        "height_above_ground_delta2_percent": 100.0 * elevated_delta2,
        "height_above_ground_delta3_percent": 100.0 * elevated_delta3,
        "height_score": max(0.0, 1.0 - rmse / maximum),
    }


class PotsdamHeightDataset(Dataset):
    """Aligned non-overlapping Potsdam RGB/nDSM patches.

    The normalized DSM is distributed as an 8-bit image in decimeters.
    Pixel value 255 therefore represents 25.5 m or higher.
    """

    _PATCH_PATTERN = re.compile(r"^(?P<parent>.+)__r(?P<row>\d+)_c(?P<column>\d+)$")

    def __init__(
        self,
        root: str | Path,
        split: str,
        resolution: int = 512,
        tile_ids: Sequence[str] | None = None,
        augment: bool = False,
        image_dir: str = "images",
        height_dir: str = "heights",
        max_height_meters: float = 25.5,
        source_max_height_meters: float | None = None,
        height_codec: str = "linear",
        source_size: int = 6000,
    ):
        self.root = Path(root).expanduser()
        self.split = str(split)
        self.resolution = int(resolution)
        self.augment = bool(augment)
        self.max_height_meters = float(max_height_meters)
        self.source_max_height_meters = float(
            self.max_height_meters
            if source_max_height_meters is None
            else source_max_height_meters
        )
        self.height_codec = str(height_codec).casefold()
        self.source_size = int(source_size)
        if self.resolution < 1 or self.source_size < 1:
            raise ValueError("Potsdam source_size and resolution must both be positive")
        if not math.isfinite(self.max_height_meters) or self.max_height_meters <= 0.0:
            raise ValueError("Potsdam max_height_meters must be > 0")
        if (
            not math.isfinite(self.source_max_height_meters)
            or self.source_max_height_meters <= 0.0
        ):
            raise ValueError("Potsdam source_max_height_meters must be > 0")
        if self.height_codec not in {"linear", "log1p"}:
            raise ValueError("Potsdam height_codec must be 'linear' or 'log1p'")
        images = _paths_by_stem(self.root / image_dir)
        heights = _paths_by_stem(self.root / height_dir)
        if set(images) != set(heights):
            raise RuntimeError(
                "Prepared Potsdam image/height filenames are not paired: "
                f"missing={len(set(images) - set(heights))}, "
                f"extra={len(set(heights) - set(images))}"
            )
        sample_metadata: dict[str, dict[str, int | str]] = {}
        for stem in images:
            match = self._PATCH_PATTERN.fullmatch(stem)
            if match is None:
                parent = stem
                row = 0
                column = 0
            else:
                parent = match.group("parent").casefold()
                row = int(match.group("row"))
                column = int(match.group("column"))
            valid_height = max(
                min(
                    self.resolution,
                    self.source_size - row * self.resolution,
                ),
                0,
            )
            valid_width = max(
                min(
                    self.resolution,
                    self.source_size - column * self.resolution,
                ),
                0,
            )
            if valid_height < 1 or valid_width < 1:
                raise ValueError(f"Potsdam patch '{stem}' starts outside source bounds")
            sample_metadata[stem] = {
                "parent": parent,
                "row": row,
                "column": column,
                "valid_height": valid_height,
                "valid_width": valid_width,
            }
        available_parents = {
            str(metadata["parent"]) for metadata in sample_metadata.values()
        }
        requested_parents = (
            sorted(available_parents)
            if tile_ids is None
            else [str(tile_id).casefold() for tile_id in tile_ids]
        )
        if len(set(requested_parents)) != len(requested_parents):
            raise ValueError(
                f"Duplicate Potsdam tile IDs for {split}: {requested_parents}"
            )
        missing = sorted(set(requested_parents) - available_parents)
        if missing:
            raise FileNotFoundError(
                f"Missing prepared Potsdam {split} tiles: {missing}"
            )
        expected_grid = math.ceil(self.source_size / self.resolution)
        expected_coordinates = {
            (row, column)
            for row in range(expected_grid)
            for column in range(expected_grid)
        }
        for parent in requested_parents:
            parent_stems = [
                stem
                for stem, metadata in sample_metadata.items()
                if metadata["parent"] == parent
            ]
            if any(
                self._PATCH_PATTERN.fullmatch(stem) is not None for stem in parent_stems
            ):
                actual_coordinates = {
                    (
                        int(sample_metadata[stem]["row"]),
                        int(sample_metadata[stem]["column"]),
                    )
                    for stem in parent_stems
                }
                if actual_coordinates != expected_coordinates:
                    raise RuntimeError(
                        f"Incomplete Potsdam patch grid for '{parent}': "
                        f"expected {len(expected_coordinates)}, got "
                        f"{len(actual_coordinates)}"
                    )
        selected = [
            stem
            for stem in sorted(images)
            if str(sample_metadata[stem]["parent"]) in requested_parents
        ]
        self.samples = [
            {
                "image": images[stem],
                "height": heights[stem],
                "tile_id": stem,
                **sample_metadata[stem],
            }
            for stem in selected
        ]
        if not self.samples:
            raise FileNotFoundError(
                f"No prepared Potsdam samples selected below '{self.root}'"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        sample = self.samples[index]
        image_path = Path(sample["image"])
        height_path = Path(sample["height"])
        with Image.open(image_path) as source:
            image_source = source.convert("RGB")
            image_size = image_source.size
            if image_size != (self.resolution, self.resolution):
                image_source = image_source.resize(
                    (self.resolution, self.resolution),
                    Image.Resampling.BILINEAR,
                )
            image = np.asarray(image_source, dtype=np.float32) / 255.0
        with Image.open(height_path) as source:
            height_source = source.convert("L")
            height_size = height_source.size
            if height_size != (self.resolution, self.resolution):
                height_source = height_source.resize(
                    (self.resolution, self.resolution),
                    Image.Resampling.BILINEAR,
                )
            height_decimeters = np.asarray(
                height_source,
                dtype=np.float32,
            )
        if image_size != height_size:
            raise RuntimeError(
                f"Potsdam image/nDSM size mismatch: {image_path}, {height_path}"
            )
        valid_source_sizes = {
            (self.source_size, self.source_size),
            (self.resolution, self.resolution),
        }
        if image_size not in valid_source_sizes:
            raise RuntimeError(
                f"Expected prepared {self.resolution}x{self.resolution} or "
                f"raw {self.source_size}x{self.source_size} Potsdam input, "
                f"got {image_size[0]}x{image_size[1]} for '{image_path}'"
            )
        height_meters = (
            height_decimeters * (self.source_max_height_meters / 255.0)
        ).clip(0.0, self.max_height_meters)
        valid_height = int(sample["valid_height"])
        valid_width = int(sample["valid_width"])
        valid_array = np.zeros(
            (self.resolution, self.resolution),
            dtype=np.float32,
        )
        valid_array[:valid_height, :valid_width] = 1.0
        if self.augment:
            joint_target = np.stack(
                (height_meters, valid_array),
                axis=-1,
            )
            image, _, joint_target = _augment_pair(
                image,
                image,
                joint_target,
            )
            height_meters = joint_target[..., 0]
            valid_array = joint_target[..., 1]
        height_tensor = torch.from_numpy(
            np.ascontiguousarray(height_meters[None])
        ).float()
        loss_mask = torch.from_numpy(np.ascontiguousarray(valid_array[None])).float()
        image_tensor = _image_tensor(image)
        return {
            "before": image_tensor,
            "after": image_tensor,
            "mask": height_tensor,
            "loss_mask": loss_mask,
            "target_rgb": height_to_target_rgb(
                height_tensor,
                self.max_height_meters,
                height_codec=self.height_codec,
            ),
            "task": HEIGHT_TASK,
            "name": str(sample["tile_id"]),
            "parent_name": str(sample["parent"]),
            "tile_row": int(sample["row"]),
            "tile_column": int(sample["column"]),
            "valid_height": valid_height,
            "valid_width": valid_width,
            "source_height": self.source_size,
            "source_width": self.source_size,
        }


class DFC23Track2HeightDataset(Dataset):
    """Single- or paired-modality DFC23 surface-height samples.

    ``input_modality='rgb_or_sar'`` exposes two views of every physical
    sample.  One view contains only RGB and the other only SAR; both predict
    the same released metric DSM.  This lets DFC23 follow the same
    single-image condition protocol as the other dense-prediction tasks.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        resolution: int = 512,
        augment: bool = False,
        manifest: str = "splits.json",
        max_height_meters: float = 200.0,
        height_codec: str = "log1p",
        input_modality: str = "rgb_or_sar",
        rgb_prompt_key: str | None = None,
        sar_prompt_key: str | None = None,
    ):
        self.root = Path(root).expanduser()
        self.split = str(split)
        self.resolution = int(resolution)
        self.augment = bool(augment)
        self.max_height_meters = float(max_height_meters)
        self.height_codec = str(height_codec).casefold()
        self.input_modality = str(input_modality).strip().casefold()
        self.rgb_prompt_key = None if rgb_prompt_key is None else str(rgb_prompt_key)
        self.sar_prompt_key = None if sar_prompt_key is None else str(sar_prompt_key)
        if self.resolution < 1:
            raise ValueError("DFC23 resolution must be positive")
        if self.height_codec not in {"linear", "log1p"}:
            raise ValueError("DFC23 height_codec must be 'linear' or 'log1p'")
        if self.input_modality not in {
            "rgb",
            "sar",
            "rgb_or_sar",
            "paired",
        }:
            raise ValueError(
                "DFC23 input_modality must be 'rgb', 'sar', "
                "'rgb_or_sar', or 'paired'"
            )
        manifest_path = self.root / manifest
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing prepared DFC23 manifest: {manifest_path}. "
                "Run prepare_dfc23_track2.py first."
            )
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        split_map = payload.get("splits", payload)
        if self.split not in split_map:
            raise KeyError(
                f"DFC23 split {self.split!r} is not present in " f"{manifest_path}"
            )
        self.samples = list(split_map[self.split])
        if not self.samples:
            raise FileNotFoundError(f"DFC23 split {self.split!r} contains no samples")
        required = {"name", "rgb", "sar", "height"}
        for sample in self.samples:
            missing = required - set(sample)
            if missing:
                raise KeyError(
                    f"DFC23 sample is missing fields {sorted(missing)}: " f"{sample}"
                )

    def __len__(self) -> int:
        multiplier = 2 if self.input_modality == "rgb_or_sar" else 1
        return len(self.samples) * multiplier

    def paired_batch_indices(
        self,
        batch_size: int,
        rng: random.Random,
        occurrence: int = 0,
    ) -> list[int | tuple[str, int, int]]:
        del occurrence
        if self.input_modality == "rgb_or_sar":
            return _paired_modality_batch_indices(
                len(self.samples),
                batch_size,
                rng,
            )
        return [rng.randrange(len(self)) for _ in range(batch_size)]

    def _path(self, relative: str) -> Path:
        path = self.root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def __getitem__(
        self,
        index: int | tuple[str, int, int],
    ) -> dict[str, Tensor | str]:
        index, paired_transform = _paired_view_index(index)
        if self.input_modality == "rgb_or_sar":
            sample = self.samples[index // 2]
            selected_modality = "rgb" if index % 2 == 0 else "sar"
        else:
            sample = self.samples[index]
            selected_modality = self.input_modality
        rgb = None
        sar = None
        if selected_modality in {"rgb", "paired"}:
            rgb = _normalized_rgb(
                self._path(str(sample["rgb"])),
                self.resolution,
            )
        if selected_modality in {"sar", "paired"}:
            # Prepared SAR is one fixed-dB grayscale PNG. Replication to RGB
            # is only the frozen FLUX-VAE interface; it is still one input.
            with Image.open(self._path(str(sample["sar"]))) as source:
                sar_source = source.convert("L")
                if sar_source.size != (self.resolution, self.resolution):
                    sar_source = sar_source.resize(
                        (self.resolution, self.resolution),
                        Image.Resampling.BILINEAR,
                    )
                sar_gray = (
                    np.asarray(
                        sar_source,
                        dtype=np.float32,
                    )
                    / 255.0
                )
            sar = np.repeat(sar_gray[..., None], 3, axis=-1)

        target = np.load(
            self._path(str(sample["height"])),
            allow_pickle=False,
        ).astype(np.float32, copy=False)
        if target.ndim == 3 and target.shape[0] == 1:
            target = target[0]
        if target.ndim != 2:
            raise ValueError(f"Expected DFC23 height [H,W], got {target.shape}")
        if target.shape != (self.resolution, self.resolution):
            target = cv2.resize(
                target,
                (self.resolution, self.resolution),
                interpolation=cv2.INTER_LINEAR,
            )
        target = np.nan_to_num(
            target,
            nan=0.0,
            posinf=self.max_height_meters,
            neginf=0.0,
        ).clip(0.0, self.max_height_meters)

        if self.augment:
            if selected_modality == "paired":
                assert sar is not None and rgb is not None
                transform = (
                    paired_transform
                    if paired_transform is not None
                    else _sample_geometric_augmentation()
                )
                sar = _apply_geometric_augmentation(sar, transform)
                rgb = _apply_geometric_augmentation(rgb, transform)
                target = _apply_geometric_augmentation(target, transform)
            else:
                selected_image = rgb if selected_modality == "rgb" else sar
                assert selected_image is not None
                transform = (
                    paired_transform
                    if paired_transform is not None
                    else _sample_geometric_augmentation()
                )
                selected_image = _apply_geometric_augmentation(
                    selected_image,
                    transform,
                )
                target = _apply_geometric_augmentation(target, transform)
                if selected_modality == "rgb":
                    rgb = selected_image
                else:
                    sar = selected_image

        target_tensor = _mask_tensor(target)
        target_rgb = height_to_target_rgb(
            target_tensor,
            max_height_meters=self.max_height_meters,
            height_codec=self.height_codec,
        )
        if selected_modality == "paired":
            assert sar is not None and rgb is not None
            before_tensor = _image_tensor(sar)
            after_tensor = _image_tensor(rgb)
            sample_name = str(sample["name"])
            prompt_key = None
        else:
            selected_image = rgb if selected_modality == "rgb" else sar
            assert selected_image is not None
            after_tensor = _image_tensor(selected_image)
            # The common data interface still carries both fields, but
            # single_after mode encodes only ``after`` and creates the null
            # condition latent internally.  No second image reaches the VAE.
            before_tensor = after_tensor
            sample_name = f"{sample['name']}__{selected_modality}"
            prompt_key = (
                self.rgb_prompt_key
                if selected_modality == "rgb"
                else self.sar_prompt_key
            )
        result = {
            "before": before_tensor,
            "after": after_tensor,
            "mask": target_tensor,
            "loss_mask": torch.ones_like(target_tensor),
            "target_rgb": target_rgb,
            "task": HEIGHT_TASK,
            "name": sample_name,
            "input_modality": selected_modality,
        }
        if prompt_key is not None:
            result["prompt_key"] = prompt_key
        return result
