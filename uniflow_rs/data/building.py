"""Data components: building."""

from __future__ import annotations

import random
from pathlib import (
    Path,
)
import numpy as np
import torch
from PIL import (
    Image,
)
from scipy.ndimage import (
    label as connected_components,
)
from torch import (
    Tensor,
)
from torch.utils.data import (
    Dataset,
)
from uniflow_rs.data.contracts import (
    BUILDING_TASK,
    SAR_BUILDING_TASK,
)
from uniflow_rs.data.transforms import (
    _apply_geometric_augmentation,
    _binary_target_rgb,
    _image_tensor,
    _mask_tensor,
    _paired_modality_batch_indices,
    _paired_view_index,
    _sample_geometric_augmentation,
)


def spacenet_instance_counts(
    prediction: Tensor,
    target_instances: Tensor,
    valid_mask: Tensor | None = None,
    iou_threshold: float = 0.5,
) -> Tensor:
    """Return SpaceNet-style instance TP/FP/FN from rasterized footprints."""

    if prediction.ndim == 4 and prediction.shape[1] == 1:
        prediction = prediction[:, 0]
    if target_instances.ndim == 4 and target_instances.shape[1] == 1:
        target_instances = target_instances[:, 0]
    if prediction.ndim != 3 or target_instances.shape != prediction.shape:
        raise ValueError(
            "Expected matching [B,H,W] prediction and instance targets, got "
            f"{tuple(prediction.shape)} and {tuple(target_instances.shape)}"
        )
    threshold = float(iou_threshold)
    if not 0.0 < threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")
    if valid_mask is None:
        valid = torch.ones_like(prediction, dtype=torch.bool)
    else:
        if valid_mask.ndim == 4 and valid_mask.shape[1] == 1:
            valid_mask = valid_mask[:, 0]
        if valid_mask.shape != prediction.shape:
            raise ValueError(
                "SpaceNet valid mask must match predictions, got "
                f"{tuple(valid_mask.shape)} and {tuple(prediction.shape)}"
            )
        valid = valid_mask > 0.5

    true_positive = 0
    false_positive = 0
    false_negative = 0
    prediction_cpu = (prediction.detach().bool() & valid.detach().bool()).cpu().numpy()
    target_cpu = (
        (target_instances.detach().long() * valid.detach().long()).cpu().numpy()
    )
    connectivity = np.ones((3, 3), dtype=np.uint8)
    for predicted_mask, target_labels in zip(
        prediction_cpu,
        target_cpu,
        strict=True,
    ):
        predicted_labels, predicted_count = connected_components(
            predicted_mask,
            structure=connectivity,
        )
        target_values, target_inverse = np.unique(
            target_labels,
            return_inverse=True,
        )
        target_labels = target_inverse.reshape(target_labels.shape)
        target_count = len(target_values) - int(target_values[0] == 0)
        if target_values[0] != 0:
            target_labels += 1
            target_count = len(target_values)
        if predicted_count == 0 or target_count == 0:
            true_positive += 0
            false_positive += predicted_count
            false_negative += target_count
            continue

        pair_width = predicted_count + 1
        intersections = np.bincount(
            (
                target_labels.astype(np.int64).ravel() * pair_width
                + predicted_labels.astype(np.int64).ravel()
            ),
            minlength=(target_count + 1) * pair_width,
        ).reshape(target_count + 1, pair_width)
        predicted_areas = np.bincount(
            predicted_labels.ravel(),
            minlength=predicted_count + 1,
        )
        target_areas = np.bincount(
            target_labels.ravel(),
            minlength=target_count + 1,
        )
        candidates: list[tuple[float, int, int]] = []
        for target_id in range(1, target_count + 1):
            for predicted_id in range(1, predicted_count + 1):
                intersection = int(intersections[target_id, predicted_id])
                if intersection == 0:
                    continue
                union = (
                    int(target_areas[target_id])
                    + int(predicted_areas[predicted_id])
                    - intersection
                )
                iou = intersection / max(union, 1)
                if iou >= threshold:
                    candidates.append((iou, target_id, predicted_id))
        matched_targets: set[int] = set()
        matched_predictions: set[int] = set()
        for _, target_id, predicted_id in sorted(
            candidates,
            reverse=True,
        ):
            if target_id in matched_targets or predicted_id in matched_predictions:
                continue
            matched_targets.add(target_id)
            matched_predictions.add(predicted_id)
        matches = len(matched_targets)
        true_positive += matches
        false_positive += predicted_count - matches
        false_negative += target_count - matches
    return torch.tensor(
        (true_positive, false_positive, false_negative),
        device=prediction.device,
        dtype=torch.float64,
    )


class BuildingTaskDataset(Dataset):
    """Expose any single-image building dataset in the shared task format."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        sample = self.dataset[index]
        image = sample["image"]
        mask = sample["mask"].float()
        return {
            "before": image,
            "after": image,
            "mask": mask,
            "loss_mask": torch.ones_like(mask),
            "target_rgb": _binary_target_rgb(mask),
            "task": BUILDING_TASK,
            "name": sample["name"],
        }


class SpaceNet6SARBuildingDataset(Dataset):
    """Prepared SpaceNet 6 RGB/SAR views and building instances.

    ``rgb_or_sar`` exposes the co-registered optical and SAR views as two
    single-image samples.  The pair-aware batch sampler keeps those views in
    the same optimizer microbatch; neither view is concatenated with the
    other at the model input.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        resolution: int = 512,
        manifest: str = "splits.json",
        augment: bool = False,
        input_modality: str = "sar",
        rgb_prompt_key: str | None = None,
        sar_prompt_key: str | None = None,
    ):
        self.root = Path(root).expanduser()
        self.split = str(split)
        self.resolution = int(resolution)
        self.augment = bool(augment)
        self.input_modality = str(input_modality).strip().casefold()
        self.rgb_prompt_key = None if rgb_prompt_key is None else str(rgb_prompt_key)
        self.sar_prompt_key = None if sar_prompt_key is None else str(sar_prompt_key)
        if self.input_modality not in {"sar", "rgb", "rgb_or_sar"}:
            raise ValueError(
                "SpaceNet 6 input_modality must be sar, rgb, or rgb_or_sar; "
                f"got {input_modality!r}"
            )
        manifest_path = self.root / str(manifest)
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing prepared SpaceNet 6 manifest: '{manifest_path}'. "
                "Run prepare_spacenet6_sar.py before training."
            )
        import json

        with manifest_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        prepared_resolution = int(metadata.get("resolution", -1))
        if prepared_resolution != self.resolution:
            raise ValueError(
                "Prepared SpaceNet 6 resolution does not match the config: "
                f"manifest={prepared_resolution}, config={self.resolution}"
            )
        split_names = metadata.get("splits", {}).get(self.split)
        if split_names is None:
            raise KeyError(
                f"SpaceNet 6 manifest has no split '{self.split}': "
                f"'{manifest_path}'"
            )
        self.samples: list[dict[str, Path | str]] = []
        for name in split_names:
            name = str(name)
            sample = {
                "name": name,
                "sar": self.root / "images" / f"{name}.png",
                "rgb": self.root / "optical_images" / f"{name}.png",
                "mask": self.root / "masks" / f"{name}.png",
                "instance": self.root / "instances" / f"{name}.png",
                "valid_mask": self.root / "valid_masks" / f"{name}.png",
            }
            required_keys = {"mask", "instance", "valid_mask"}
            if self.input_modality in {"sar", "rgb_or_sar"}:
                required_keys.add("sar")
            if self.input_modality in {"rgb", "rgb_or_sar"}:
                required_keys.add("rgb")
            missing = [
                path
                for key, path in sample.items()
                if key in required_keys and not Path(path).is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    "Prepared SpaceNet 6 sample is incomplete: "
                    + ", ".join(str(path) for path in missing)
                )
            self.samples.append(sample)
        if not self.samples:
            raise ValueError(f"Prepared SpaceNet 6 split '{self.split}' is empty")

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
        with Image.open(Path(sample[selected_modality])) as source:
            image = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
        with Image.open(Path(sample["mask"])) as source:
            mask = np.asarray(source.convert("L"), dtype=np.uint8) > 127
        with Image.open(Path(sample["instance"])) as source:
            instance = np.asarray(source, dtype=np.int64)
        with Image.open(Path(sample["valid_mask"])) as source:
            valid = np.asarray(source.convert("L"), dtype=np.uint8) > 127
        expected = (self.resolution, self.resolution)
        if (
            image.shape[:2] != expected
            or mask.shape != expected
            or instance.shape != expected
            or valid.shape != expected
        ):
            raise RuntimeError(
                "Prepared SpaceNet 6 sample has inconsistent dimensions: "
                f"'{sample['name']}'"
            )
        if self.augment:
            combined = np.stack(
                (
                    mask.astype(np.int64),
                    instance,
                    valid.astype(np.int64),
                ),
                axis=-1,
            )
            transform = (
                paired_transform
                if paired_transform is not None
                else _sample_geometric_augmentation()
            )
            image = _apply_geometric_augmentation(image, transform)
            combined = _apply_geometric_augmentation(combined, transform)
            mask = combined[..., 0] > 0
            instance = combined[..., 1]
            valid = combined[..., 2] > 0
        mask_tensor = _mask_tensor(mask)
        valid_tensor = _mask_tensor(valid)
        image_tensor = _image_tensor(image)
        result = {
            "before": image_tensor,
            "after": image_tensor,
            "mask": mask_tensor,
            "loss_mask": valid_tensor,
            "target_rgb": _binary_target_rgb(mask_tensor),
            "instance_mask": torch.from_numpy(
                np.ascontiguousarray(instance[None])
            ).long(),
            "task": SAR_BUILDING_TASK,
            "name": f"{sample['name']}__{selected_modality}",
            "input_modality": selected_modality,
        }
        prompt_key = (
            self.rgb_prompt_key if selected_modality == "rgb" else self.sar_prompt_key
        )
        if prompt_key is not None:
            result["prompt_key"] = prompt_key
        return result
