"""Data components: openearthmap."""

from __future__ import annotations

import math
import random
from pathlib import (
    Path,
)
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
    OPENEARTHMAP_SAR_CLASS_NAMES,
    OPENEARTHMAP_SAR_CODE_PALETTE_NAME,
    SAR_SEMANTIC_TASK,
    _RASTER_EXTENSIONS,
    resolve_openearthmap_sar_code_palette,
)
from uniflow_rs.data.transforms import (
    _apply_geometric_augmentation,
    _image_tensor,
    _paired_modality_batch_indices,
    _paired_view_index,
    _paths_by_stem,
    _sample_geometric_augmentation,
)


def openearthmap_sar_labels_to_rgb(
    labels: np.ndarray,
    code_palette: str = OPENEARTHMAP_SAR_CODE_PALETTE_NAME,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size and (labels.min() < 0 or labels.max() > 8):
        raise ValueError(
            "OpenEarthMap-SAR RGB labels must be in [0, 8], got range "
            f"[{labels.min()}, {labels.max()}]"
        )
    return resolve_openearthmap_sar_code_palette(code_palette)[labels]


def decode_openearthmap_sar_palette(
    decoded: Tensor,
    code_palette: str = OPENEARTHMAP_SAR_CODE_PALETTE_NAME,
) -> Tensor:
    """Map decoded VAE code colors to official OpenEarthMap IDs 0..8."""

    colors = (
        torch.as_tensor(
            resolve_openearthmap_sar_code_palette(code_palette),
            device=decoded.device,
            dtype=torch.float32,
        )
        .div(127.5)
        .sub(1.0)
    )
    pixels = decoded.float().clamp(-1.0, 1.0).permute(0, 2, 3, 1)
    squared_distance = (
        (pixels.unsqueeze(-2) - colors.view(1, 1, 1, len(colors), 3))
        .square()
        .sum(dim=-1)
    )
    return squared_distance.argmin(dim=-1)


def openearthmap_sar_palette_logits(
    decoded: Tensor,
    temperature: float = 0.1,
    code_palette: str = OPENEARTHMAP_SAR_CODE_PALETTE_NAME,
) -> Tensor:
    """Return differentiable logits for the nine VAE code colors."""

    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("OpenEarthMap-SAR palette temperature must be finite and > 0")
    colors = (
        torch.as_tensor(
            resolve_openearthmap_sar_code_palette(code_palette),
            device=decoded.device,
            dtype=torch.float32,
        )
        .div(127.5)
        .sub(1.0)
    )
    squared_distance = (
        (
            decoded.float().permute(0, 2, 3, 1).unsqueeze(-2)
            - colors.view(1, 1, 1, len(colors), 3)
        )
        .square()
        .sum(dim=-1)
    )
    return (-squared_distance / temperature).permute(0, 3, 1, 2)


def openearthmap_sar_confusion(
    prediction: Tensor,
    target: Tensor,
) -> Tensor:
    """Return 9x9 confusion used for official eight-class mIoU."""

    prediction = prediction.long().clamp(0, 8)
    target = target.long()
    valid = (target >= 0) & (target <= 8)
    bins = torch.bincount(
        target[valid] * 9 + prediction[valid],
        minlength=81,
    )
    return bins.reshape(9, 9).to(torch.float64)


def openearthmap_sar_metrics(confusion: Tensor) -> dict[str, float]:
    confusion = confusion.detach().cpu().double()
    if confusion.shape != (9, 9):
        raise ValueError(
            "OpenEarthMap-SAR confusion must have shape [9,9], got "
            f"{tuple(confusion.shape)}"
        )
    true_positive = confusion.diag()[1:]
    ground_truth = confusion.sum(dim=1)[1:]
    # Column sums include ID-0 ground-truth pixels predicted as an evaluated
    # class, matching the official baseline's per-class union calculation.
    predicted = confusion.sum(dim=0)[1:]
    union = ground_truth + predicted - true_positive
    class_iou = true_positive / union.clamp_min(1.0)
    class_dice = 2.0 * true_positive / (ground_truth + predicted).clamp_min(1.0)
    class_accuracy = true_positive / ground_truth.clamp_min(1.0)
    valid_classes = ground_truth > 0
    mean_iou = float(class_iou[valid_classes].mean())
    mean_dice = float(class_dice[valid_classes].mean())
    pixel_accuracy = float(confusion.diag().sum() / confusion.sum().clamp_min(1.0))
    mean_accuracy = float(class_accuracy[valid_classes].mean())
    metrics = {
        "sar_semantic_iou": mean_iou,
        "sar_semantic_dice": mean_dice,
        "sar_semantic_miou": mean_iou,
        "sar_semantic_mdice": mean_dice,
        "sar_semantic_pixel_accuracy": pixel_accuracy,
        "sar_semantic_mean_accuracy": mean_accuracy,
        "sar_semantic_background_iou": float(
            confusion[0, 0]
            / (confusion[0].sum() + confusion[:, 0].sum() - confusion[0, 0]).clamp_min(
                1.0
            )
        ),
    }
    for index, name in enumerate(OPENEARTHMAP_SAR_CLASS_NAMES):
        metrics[f"sar_semantic_{name}_iou"] = float(class_iou[index])
        metrics[f"sar_semantic_{name}_dice"] = float(class_dice[index])
    return metrics


def _resolve_openearthmap_sar_split(
    root: Path,
    split: str,
    trainval_dir: str,
) -> Path:
    split_name = str(split).strip()
    candidates = []
    if split_name.casefold() in {"train", "val"}:
        candidates.append(root / str(trainval_dir) / split_name)
    candidates.extend((root / split_name, root / split_name.casefold()))
    for candidate in candidates:
        if (candidate / "sar_images").is_dir() and (candidate / "labels").is_dir():
            return candidate
    attempted = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        f"Could not resolve OpenEarthMap-SAR split '{split}'. Tried:\n" f"{attempted}"
    )


class OpenEarthMapSARSemanticDataset(Dataset):
    """DFC 2025 Track 1 land-cover semantic segmentation.

    Every split resizes the complete source image to ``resolution``. SAR uses
    bilinear interpolation while the discrete label map uses nearest-neighbor
    interpolation. The paired optical RGB image can be selected without
    changing the model input contract. Training may additionally apply paired
    flips/rotations.
    """

    dataset_name = "OpenEarthMap-SAR"

    def __init__(
        self,
        root: str | Path,
        split: str,
        resolution: int = 512,
        source_size: int = 1024,
        trainval_dir: str = "dfc25_track1_trainval",
        augment: bool = False,
        input_modality: str = "sar",
        subset: str = "all",
        holdout_fraction: float = 0.0,
        holdout_seed: int = 12345,
        code_palette: str = OPENEARTHMAP_SAR_CODE_PALETTE_NAME,
        rgb_prompt_key: str | None = None,
        sar_prompt_key: str | None = None,
    ):
        self.root = Path(root).expanduser()
        self.split = str(split)
        self.split_dir = _resolve_openearthmap_sar_split(
            self.root,
            self.split,
            trainval_dir,
        )
        self.resolution = int(resolution)
        self.source_size = int(source_size)
        self.augment = bool(augment)
        self.input_modality = str(input_modality).strip().casefold()
        self.rgb_prompt_key = None if rgb_prompt_key is None else str(rgb_prompt_key)
        self.sar_prompt_key = None if sar_prompt_key is None else str(sar_prompt_key)
        self.subset = str(subset).strip().casefold()
        self.holdout_fraction = float(holdout_fraction)
        self.holdout_seed = int(holdout_seed)
        self.code_palette = str(code_palette).strip().casefold()
        resolve_openearthmap_sar_code_palette(self.code_palette)
        if self.input_modality not in {"sar", "rgb", "rgb_or_sar"}:
            raise ValueError(
                "OpenEarthMap input_modality must be sar, rgb, or "
                "rgb_or_sar, got "
                f"'{input_modality}'"
            )
        if self.subset not in {"all", "fit", "holdout"}:
            raise ValueError(
                "OpenEarthMap subset must be all, fit, or holdout, got " f"'{subset}'"
            )
        if self.resolution < 1 or self.source_size < 1:
            raise ValueError(
                "OpenEarthMap-SAR source_size and resolution must be positive"
            )
        if self.resolution > self.source_size:
            raise ValueError(
                "OpenEarthMap-SAR resolution cannot exceed source_size: "
                f"{self.resolution} > {self.source_size}"
            )
        required_modalities = (
            ("sar", "rgb")
            if self.input_modality == "rgb_or_sar"
            else (self.input_modality,)
        )
        images_by_modality = {
            modality: _paths_by_stem(
                self.split_dir / ("sar_images" if modality == "sar" else "rgb_images")
            )
            for modality in required_modalities
        }
        labels = _paths_by_stem(self.split_dir / "labels")
        for modality, images in images_by_modality.items():
            if set(images) != set(labels):
                raise RuntimeError(
                    "OpenEarthMap image/label filenames are not paired for "
                    f"{modality} below '{self.split_dir}': "
                    f"missing={len(set(images)-set(labels))}, "
                    f"extra={len(set(labels)-set(images))}"
                )
        self.samples = [
            {
                "name": stem,
                **{
                    modality: images_by_modality[modality][stem]
                    for modality in required_modalities
                },
                "mask": labels[stem],
            }
            for stem in sorted(labels)
        ]
        if not self.samples:
            raise FileNotFoundError(
                f"No OpenEarthMap {self.input_modality.upper()} samples below "
                f"'{self.split_dir}'"
            )
        if self.subset != "all":
            if not 0.0 < self.holdout_fraction < 1.0:
                raise ValueError(
                    "OpenEarthMap fit/holdout subsets require "
                    "0 < holdout_fraction < 1"
                )
            indices = list(range(len(self.samples)))
            random.Random(self.holdout_seed).shuffle(indices)
            holdout_count = max(
                1,
                min(
                    len(indices) - 1,
                    int(round(len(indices) * self.holdout_fraction)),
                ),
            )
            holdout_indices = set(indices[:holdout_count])
            keep_holdout = self.subset == "holdout"
            self.samples = [
                sample
                for sample_index, sample in enumerate(self.samples)
                if (sample_index in holdout_indices) == keep_holdout
            ]

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
        image_path = Path(sample[selected_modality])
        mask_path = Path(sample["mask"])
        with Image.open(image_path) as source:
            if source.size != (self.source_size, self.source_size):
                raise RuntimeError(
                    f"Expected OpenEarthMap-SAR {self.source_size}x"
                    f"{self.source_size}, got {source.size} for '{image_path}'"
                )
            if selected_modality == "sar":
                input_image = source.convert("L")
            else:
                input_image = source.convert("RGB")
            if input_image.size != (self.resolution, self.resolution):
                input_image = input_image.resize(
                    (self.resolution, self.resolution),
                    Image.Resampling.BILINEAR,
                )
            image = np.asarray(input_image, dtype=np.float32) / 255.0
            if selected_modality == "sar":
                image = np.repeat(image[..., None], 3, axis=2)
        with Image.open(mask_path) as source:
            if source.size != (self.source_size, self.source_size):
                raise RuntimeError(
                    "OpenEarthMap-SAR image/label size mismatch: "
                    f"'{image_path}', '{mask_path}'"
                )
            label_image = source.convert("L")
            if label_image.size != (self.resolution, self.resolution):
                label_image = label_image.resize(
                    (self.resolution, self.resolution),
                    Image.Resampling.NEAREST,
                )
            labels = np.asarray(label_image, dtype=np.uint8)
        if labels.size and (labels.min() < 0 or labels.max() > 8):
            raise ValueError(
                f"OpenEarthMap-SAR labels must be in [0, 8]: '{mask_path}'"
            )
        if self.augment:
            transform = (
                paired_transform
                if paired_transform is not None
                else _sample_geometric_augmentation()
            )
            image = _apply_geometric_augmentation(image, transform)
            labels = _apply_geometric_augmentation(labels, transform)
        label_tensor = torch.from_numpy(
            np.ascontiguousarray(labels[None].astype(np.float32))
        )
        target_rgb_array = (
            openearthmap_sar_labels_to_rgb(
                labels,
                code_palette=self.code_palette,
            ).astype(np.float32)
            / 255.0
        )
        image_tensor = _image_tensor(image)
        result = {
            # Single-image protocol: both generic condition slots point to the
            # selected modality; single_after prevents encoding the null slot.
            "before": image_tensor,
            "after": image_tensor,
            "mask": label_tensor,
            "loss_mask": torch.ones_like(label_tensor),
            "target_rgb": _image_tensor(target_rgb_array),
            "task": SAR_SEMANTIC_TASK,
            "name": f"{sample['name']}__{selected_modality}",
            "input_modality": selected_modality,
        }
        prompt_key = (
            self.rgb_prompt_key if selected_modality == "rgb" else self.sar_prompt_key
        )
        if prompt_key is not None:
            result["prompt_key"] = prompt_key
        return result


class OpenEarthMapOpticalSemanticDataset(Dataset):
    """Original OpenEarthMap optical-RGB land-cover segmentation.

    The public benchmark stores every region in its own ``images`` and
    ``labels`` directories and lists sample basenames in ``train.txt``,
    ``val.txt`` and ``test.txt``. Native image sizes vary by source region, so
    each image/label pair is checked at its original size and then resized as
    a whole to the configured model resolution. The official test split has
    no public labels and is therefore intentionally rejected by this training
    dataset; configurations should use the labeled validation split for local
    evaluation.

    OpenEarthMap and OpenEarthMap-SAR use the same IDs 0..8 and VAE codebook.
    This dataset consequently reuses the existing ``sar_semantic`` task
    contract, losses, decoder and metrics without changing the MMDiT shape.
    """

    dataset_name = "OpenEarthMap"

    def __init__(
        self,
        root: str | Path,
        split: str,
        resolution: int = 512,
        augment: bool = False,
        code_palette: str = OPENEARTHMAP_SAR_CODE_PALETTE_NAME,
    ):
        self.root = Path(root).expanduser()
        self.split = str(split)
        self.resolution = int(resolution)
        self.augment = bool(augment)
        self.code_palette = str(code_palette).strip().casefold()
        resolve_openearthmap_sar_code_palette(self.code_palette)
        if self.resolution < 1:
            raise ValueError("OpenEarthMap resolution must be positive")

        split_path = self.root / f"{self.split}.txt"
        if not split_path.is_file():
            matches = [
                path
                for path in self.root.glob("*.txt")
                if path.stem.casefold() == self.split.casefold()
            ]
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"Could not resolve OpenEarthMap split {self.split!r} "
                    f"below '{self.root}'"
                )
            split_path = matches[0]

        names = [
            line.strip()
            for line in split_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not names:
            raise RuntimeError(f"OpenEarthMap split is empty: {split_path}")
        if len(names) != len(set(names)):
            raise RuntimeError(f"Duplicate OpenEarthMap names in split: {split_path}")

        images: dict[str, Path] = {}
        labels: dict[str, Path] = {}
        regions = sorted(path for path in self.root.iterdir() if path.is_dir())
        for region in regions:
            for kind, index in (("images", images), ("labels", labels)):
                directory = region / kind
                if not directory.is_dir():
                    continue
                for path in directory.iterdir():
                    if path.suffix.casefold() not in _RASTER_EXTENSIONS:
                        continue
                    if path.name in index:
                        raise RuntimeError(
                            f"Duplicate OpenEarthMap {kind} basename: {path.name}"
                        )
                    index[path.name] = path

        missing_images = [name for name in names if name not in images]
        missing_labels = [name for name in names if name not in labels]
        if missing_images:
            preview = ", ".join(missing_images[:5])
            raise FileNotFoundError(
                f"Missing {len(missing_images)} OpenEarthMap images for "
                f"split {self.split!r}. Examples: {preview}"
            )
        if missing_labels:
            preview = ", ".join(missing_labels[:5])
            raise FileNotFoundError(
                f"Missing {len(missing_labels)} OpenEarthMap labels for "
                f"split {self.split!r}. The official test labels are not "
                f"public; use val for local evaluation. Examples: {preview}"
            )
        self.samples = [
            {
                "name": Path(name).stem,
                "image": images[name],
                "mask": labels[name],
            }
            for name in names
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        sample = self.samples[index]
        image_path = Path(sample["image"])
        mask_path = Path(sample["mask"])
        with Image.open(image_path) as source:
            image_source = source.convert("RGB")
            source_image_size = image_source.size
            if image_source.size != (self.resolution, self.resolution):
                image_source = image_source.resize(
                    (self.resolution, self.resolution),
                    Image.Resampling.BILINEAR,
                )
            image = np.asarray(image_source, dtype=np.float32) / 255.0
        with Image.open(mask_path) as source:
            mask_source = source.convert("L")
            source_mask_size = mask_source.size
            if mask_source.size != (self.resolution, self.resolution):
                mask_source = mask_source.resize(
                    (self.resolution, self.resolution),
                    Image.Resampling.NEAREST,
                )
            labels = np.asarray(mask_source, dtype=np.uint8)
        if source_image_size != source_mask_size:
            raise RuntimeError(
                f"OpenEarthMap image/label size mismatch: '{image_path}', "
                f"'{mask_path}'"
            )
        if labels.size and (labels.min() < 0 or labels.max() > 8):
            raise ValueError(f"OpenEarthMap labels must be in [0, 8]: '{mask_path}'")
        if self.augment:
            transform = _sample_geometric_augmentation()
            image = _apply_geometric_augmentation(image, transform)
            labels = _apply_geometric_augmentation(labels, transform)

        label_tensor = torch.from_numpy(
            np.ascontiguousarray(labels[None].astype(np.float32))
        )
        target_rgb_array = (
            openearthmap_sar_labels_to_rgb(
                labels,
                code_palette=self.code_palette,
            ).astype(np.float32)
            / 255.0
        )
        image_tensor = _image_tensor(image)
        return {
            "before": image_tensor,
            "after": image_tensor,
            "mask": label_tensor,
            # The official implementation trains nine logits including
            # background ID 0, while reporting land-cover mIoU over IDs 1..8.
            "loss_mask": torch.ones_like(label_tensor),
            "target_rgb": _image_tensor(target_rgb_array),
            "task": SAR_SEMANTIC_TASK,
            "name": str(sample["name"]),
            "input_modality": "rgb",
        }
