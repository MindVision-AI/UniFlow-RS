"""Data components: referring."""

from __future__ import annotations

import json
from pathlib import (
    Path,
)
import torch
from torch import (
    Tensor,
)
from torch.utils.data import (
    Dataset,
)
from uniflow_rs.data.contracts import (
    REFERRING_TASK,
)
from uniflow_rs.data.transforms import (
    _binary_mask,
    _binary_target_rgb,
    _image_tensor,
    _mask_tensor,
    _normalized_rgb,
)


class ReferringSegmentationDataset(Dataset):
    """Prepared image/expression/target-mask triplets.

    Spatial flips and rotations are deliberately disabled: changing an image
    while leaving expressions such as "the airplane on the left" unchanged
    would introduce direct language/label contradictions.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        resolution: int = 512,
        manifest: str = "splits.json",
        augment: bool = False,
    ):
        self.root = Path(root).expanduser()
        self.split = str(split).casefold()
        self.resolution = int(resolution)
        if self.resolution < 1:
            raise ValueError("Referring-segmentation resolution must be positive")
        if augment:
            raise ValueError(
                "Geometric augmentation is disabled for referring data because "
                "its expressions contain spatial directions."
            )
        manifest_path = self.root / str(manifest)
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing prepared referring manifest: {manifest_path}\n"
                "Run the matching prepare_rrsisd.py or prepare_lasers.py "
                "before training."
            )
        with manifest_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        supported_formats = {
            "rrsisd_uniflow_rs_v1",
            "referring_uniflow_rs_v2",
        }
        if payload.get("format") not in supported_formats:
            raise ValueError(
                f"Unsupported referring manifest format in '{manifest_path}': "
                f"{payload.get('format')!r}"
            )
        self.dataset_name = str(payload.get("dataset_name", "RRSIS-D"))
        manifest_resolution = int(payload["resolution"])
        if manifest_resolution != self.resolution:
            raise ValueError(
                f"Prepared {self.dataset_name} resolution is "
                f"{manifest_resolution}, but "
                f"the config requests {self.resolution}."
            )
        if self.split not in payload["splits"]:
            raise KeyError(
                f"{self.dataset_name} split '{self.split}' is not in "
                f"'{manifest_path}'"
            )
        self.samples = list(payload["splits"][self.split])
        self.prompt_count = int(payload["prompt_count"])
        self.prompt_sha256 = str(payload["prompt_sha256"])
        if not self.samples:
            raise FileNotFoundError(
                f"No {self.dataset_name} samples for split '{self.split}'"
            )
        prompt_indices = [int(sample["prompt_index"]) for sample in self.samples]
        if len(set(prompt_indices)) != len(prompt_indices):
            raise ValueError(
                f"Duplicate {self.dataset_name} prompt_index values in split "
                f"'{self.split}'"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor | str | int]:
        sample = self.samples[index]
        image_path = self.root / str(sample["image"])
        mask_path = self.root / str(sample["mask"])
        image = _normalized_rgb(image_path, self.resolution)
        target = _binary_mask(mask_path, self.resolution)
        image_tensor = _image_tensor(image)
        mask = _mask_tensor(target)
        return {
            "before": image_tensor,
            "after": image_tensor,
            "mask": mask,
            "loss_mask": torch.ones_like(mask),
            "target_rgb": _binary_target_rgb(mask),
            "task": REFERRING_TASK,
            "name": str(sample["name"]),
            "prompt_index": int(sample["prompt_index"]),
            "expression": str(sample["expression"]),
            "category": str(sample["category"]),
            "ref_id": int(sample["ref_id"]),
        }


# Backward-compatible import name for older scripts and checkpoints.
RRSISDReferringDataset = ReferringSegmentationDataset
