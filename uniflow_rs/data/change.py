"""Data components: change."""

from __future__ import annotations

import random
import re
from pathlib import (
    Path,
)
import numpy as np
import tifffile
import torch
from torch import (
    Tensor,
)
from torch.utils.data import (
    Dataset,
)
from uniflow_rs.data.contracts import (
    CHANGE_TASK,
    SAR_CHANGE_TASK,
    URBANSAR_FLOOD_CLASS_NAMES,
    URBANSAR_FLOOD_PALETTE,
    URBANSAR_FLOOD_REGIONS,
)
from uniflow_rs.data.semantic_common import (
    _semantic_confusion,
    _semantic_metrics,
    _semantic_palette_distance,
)
from uniflow_rs.data.transforms import (
    _apply_geometric_augmentation,
    _augment_pair,
    _binary_mask,
    _binary_target_rgb,
    _image_tensor,
    _mask_tensor,
    _normalized_rgb,
    _paths_by_stem,
    _sample_geometric_augmentation,
)


def _resolve_whu_change_root(root: str | Path) -> Path:
    configured = Path(root).expanduser()
    candidates = (
        configured,
        configured / "1. The two-period image data",
        configured
        / "Building change detection dataset_add"
        / "1. The two-period image data",
    )
    for candidate in candidates:
        if (candidate / "2012" / "splited_images").is_dir():
            return candidate
    attempted = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "Could not find the WHU change-detection 'splited_images' folders. "
        f"Tried:\n{attempted}"
    )


def _resolve_official_change_tile_dir(
    root: Path,
    split: str,
    configured: str | Path | None,
) -> Path:
    if configured is not None:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = root / path
        candidates = (path,)
    else:
        candidates = (
            root / "change_label" / "splited_images" / split / "label",
            root / "change_label" / split / "tiles",
        )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    attempted = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "Official WHU change-label tiles have not been prepared. Tried:\n"
        f"{attempted}\n"
        "Run `python prepare_whu_change_labels.py --config "
        "<your-config.yaml>` once before training."
    )


def urbansar_flood_labels_to_rgb(labels: np.ndarray) -> np.ndarray:
    """Convert internal UrbanSARFloods IDs 1..3 to the VAE RGB codebook."""

    labels = np.asarray(labels, dtype=np.int64)
    if labels.size and (labels.min() < 1 or labels.max() > 3):
        raise ValueError(
            "UrbanSARFloods internal labels must be in [1, 3], got range "
            f"[{labels.min()}, {labels.max()}]"
        )
    return URBANSAR_FLOOD_PALETTE[labels]


def decode_urbansar_flood_palette(decoded: Tensor) -> Tensor:
    """Decode a VAE RGB endpoint to internal NF/FO/FU IDs 1..3."""

    squared_distance = _semantic_palette_distance(
        decoded,
        URBANSAR_FLOOD_PALETTE,
        "UrbanSARFloods",
    )
    return squared_distance.argmin(dim=-1) + 1


def urbansar_flood_confusion(
    prediction: Tensor,
    target: Tensor,
    valid_mask: Tensor | None = None,
) -> Tensor:
    if valid_mask is not None:
        valid_mask = valid_mask.to(device=target.device)
        if valid_mask.ndim == target.ndim + 1 and valid_mask.shape[1] == 1:
            valid_mask = valid_mask[:, 0]
        if valid_mask.shape != target.shape:
            raise ValueError(
                "UrbanSARFloods valid mask must match the target shape, got "
                f"{tuple(valid_mask.shape)} and {tuple(target.shape)}"
            )
        # _semantic_confusion ignores IDs below one.  Mark pixels not owned by
        # this 512x512 test tile as zero so overlapping edge tiles are counted
        # exactly once in the aggregate confusion matrix.
        target = torch.where(
            valid_mask.bool(),
            target,
            torch.zeros_like(target),
        )
    return _semantic_confusion(prediction, target, class_count=3)


def urbansar_flood_metrics(confusion: Tensor) -> dict[str, float]:
    """Return three-class and flood-vs-background UrbanSARFloods metrics."""

    confusion = confusion.detach().cpu().double()
    metrics = _semantic_metrics(
        confusion,
        prefix=SAR_CHANGE_TASK,
        class_names=URBANSAR_FLOOD_CLASS_NAMES,
    )
    true_flood = confusion[1:, :].sum()
    predicted_flood = confusion[:, 1:].sum()
    true_positive = confusion[1:, 1:].sum()
    union = true_flood + predicted_flood - true_positive
    metrics[f"{SAR_CHANGE_TASK}_binary_flood_iou"] = float(
        true_positive / union.clamp_min(1.0)
    )
    metrics[f"{SAR_CHANGE_TASK}_binary_flood_dice"] = float(
        2.0 * true_positive / (true_flood + predicted_flood).clamp_min(1.0)
    )
    metrics[f"{SAR_CHANGE_TASK}_flood_miou"] = 0.5 * (
        metrics[f"{SAR_CHANGE_TASK}_open_flood_iou"]
        + metrics[f"{SAR_CHANGE_TASK}_urban_flood_iou"]
    )
    # Keep the benchmark convention explicit.  AnyDisasterMapping reports
    # mIoU over all three UrbanSARFloods classes (NF, FO, FU), including the
    # non-flood/background class, from one accumulated confusion matrix.  The
    # legacy ``sar_change_miou`` above uses the same value whenever all classes
    # are present, but its present-class filtering can differ on a region that
    # contains no pixels of one class.  This field is therefore the stable
    # cross-paper comparison column and always averages exactly three IoUs.
    metrics.update(urbansar_flood_benchmark_metrics(confusion))
    return metrics


def urbansar_flood_benchmark_metrics(
    confusion: Tensor,
    region_confusions: Tensor | None = None,
) -> dict[str, float]:
    """Compute the UrbanSARFloods benchmark-compatible three-class mIoU.

    The benchmark implementation used for the disaster-mapping comparison
    constructs one confusion matrix over labels ``0=NF, 1=FO, 2=FU`` and
    reports the arithmetic mean of the three per-class IoUs. UniFlow-RS stores
    these same classes as the three rows/columns of its internal confusion
    matrix, so no foreground-only conversion is applied here.

    ``region_confusions`` is optional.  When supplied, the function also
    reports the mean of the per-region three-class mIoUs as a diagnostic.  It
    is not used for the primary benchmark value, which is computed from the
    accumulated global matrix just like the reference evaluator.
    """

    confusion = confusion.detach().cpu().double()
    if tuple(confusion.shape) != (3, 3):
        raise ValueError(
            "UrbanSARFloods benchmark confusion must have shape [3,3], got "
            f"{tuple(confusion.shape)}"
        )

    true_positive = confusion.diag()
    union = confusion.sum(dim=1) + confusion.sum(dim=0) - true_positive
    class_iou = true_positive / union.clamp_min(1.0)
    metrics = {
        f"{SAR_CHANGE_TASK}_benchmark_miou": float(class_iou.mean()),
        f"{SAR_CHANGE_TASK}_benchmark_nf_iou": float(class_iou[0]),
        f"{SAR_CHANGE_TASK}_benchmark_fo_iou": float(class_iou[1]),
        f"{SAR_CHANGE_TASK}_benchmark_fu_iou": float(class_iou[2]),
    }

    if region_confusions is not None:
        region_confusions = region_confusions.detach().cpu().double()
        expected = (len(URBANSAR_FLOOD_REGIONS), 3, 3)
        if tuple(region_confusions.shape) != expected:
            raise ValueError(
                "UrbanSARFloods region benchmark confusion must have shape "
                f"{expected}, got {tuple(region_confusions.shape)}"
            )
        region_tp = region_confusions.diagonal(dim1=1, dim2=2)
        region_union = (
            region_confusions.sum(dim=2) + region_confusions.sum(dim=1) - region_tp
        )
        region_iou = region_tp / region_union.clamp_min(1.0)
        metrics[f"{SAR_CHANGE_TASK}_benchmark_region_miou"] = float(
            region_iou.mean(dim=1).mean()
        )
        for index, region in enumerate(URBANSAR_FLOOD_REGIONS):
            metrics[f"{SAR_CHANGE_TASK}_benchmark_{region.casefold()}_miou"] = float(
                region_iou[index].mean()
            )
    return metrics


def urbansar_flood_official_metrics(
    region_confusions: Tensor,
) -> dict[str, float]:
    """Return the official six UrbanSARFloods region/class F1 scores.

    The public benchmark reports open-area flood (FO) and urban flood (FU)
    precision, recall and F1 independently for Weihui, Jubba and Nova.  Its
    single checkpoint-selection score is the unweighted mean of those six F1
    values.  ``region_confusions`` must follow ``URBANSAR_FLOOD_REGIONS`` and
    use the internal class order ``[NF, FO, FU]``.
    """

    confusion = region_confusions.detach().cpu().double()
    expected_shape = (len(URBANSAR_FLOOD_REGIONS), 3, 3)
    if tuple(confusion.shape) != expected_shape:
        raise ValueError(
            "UrbanSARFloods region confusion must have shape "
            f"{expected_shape}, got {tuple(confusion.shape)}"
        )

    metrics: dict[str, float] = {}
    f1_scores: list[float] = []
    evaluated_regions = 0
    for region_index, region_name in enumerate(URBANSAR_FLOOD_REGIONS):
        current = confusion[region_index]
        region_key = region_name.casefold()
        if float(current.sum()) <= 0.0:
            for class_key in ("fo", "fu"):
                metrics[f"sar_change_{region_key}_{class_key}_precision"] = 0.0
                metrics[f"sar_change_{region_key}_{class_key}_recall"] = 0.0
                metrics[f"sar_change_{region_key}_{class_key}_f1"] = 0.0
            continue

        evaluated_regions += 1
        for class_index, class_key in ((1, "fo"), (2, "fu")):
            true_positive = float(current[class_index, class_index])
            predicted = float(current[:, class_index].sum())
            ground_truth = float(current[class_index, :].sum())
            precision = true_positive / max(predicted, 1.0)
            recall = true_positive / max(ground_truth, 1.0)
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if precision + recall > 0.0
                else 0.0
            )
            metrics[f"sar_change_{region_key}_{class_key}_precision"] = precision
            metrics[f"sar_change_{region_key}_{class_key}_recall"] = recall
            metrics[f"sar_change_{region_key}_{class_key}_f1"] = f1
            f1_scores.append(f1)

    metrics["sar_change_official_macro_f1"] = (
        sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    )
    metrics["sar_change_official_regions"] = float(evaluated_regions)
    return metrics


class WHUChangeDataset(Dataset):
    """Paired WHU tiles with official change-detection targets."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        resolution: int = 512,
        before_year: str = "2012",
        after_year: str = "2016",
        augment: bool = False,
        target_source: str = "official",
        official_label_dir: str | Path | None = None,
    ):
        self.root = _resolve_whu_change_root(root)
        self.split = str(split)
        self.resolution = int(resolution)
        self.before_year = str(before_year)
        self.after_year = str(after_year)
        self.augment = bool(augment)
        self.target_source = str(target_source).casefold()
        if self.target_source != "official":
            raise ValueError(
                "WHU-CD target_source must be 'official', got " f"'{target_source}'"
            )

        def directory(year: str, kind: str) -> Path:
            return self.root / year / "splited_images" / self.split / kind

        before_images = _paths_by_stem(directory(self.before_year, "image"))
        after_images = _paths_by_stem(directory(self.after_year, "image"))
        reference = set(before_images)
        if set(after_images) != reference:
            raise RuntimeError(
                "WHU before/after image filenames are not paired: "
                f"split='{self.split}', "
                f"missing={len(reference - set(after_images))}, "
                f"extra={len(set(after_images) - reference)}"
            )

        official_dir = _resolve_official_change_tile_dir(
            self.root,
            self.split,
            official_label_dir,
        )
        official_labels = _paths_by_stem(official_dir)
        target_collections = (official_labels,)
        for paths in target_collections:
            if set(paths) != reference:
                raise RuntimeError(
                    "WHU change-detection target filenames are not paired: "
                    f"split='{self.split}', missing={len(reference - set(paths))}, "
                    f"extra={len(set(paths) - reference)}"
                )

        self.samples = [
            {
                "before_image": before_images[stem],
                "after_image": after_images[stem],
                "official_label": official_labels.get(stem),
            }
            for stem in sorted(reference)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        paths = self.samples[index]
        before_path = paths["before_image"]
        after_path = paths["after_image"]
        before = _normalized_rgb(before_path, self.resolution)
        after = _normalized_rgb(after_path, self.resolution)
        target = _binary_mask(paths["official_label"], self.resolution)
        if self.augment:
            before, after, target = _augment_pair(before, after, target)
        mask = _mask_tensor(target)
        return {
            "before": _image_tensor(before),
            "after": _image_tensor(after),
            "mask": mask,
            "loss_mask": torch.ones_like(mask),
            "target_rgb": _binary_target_rgb(mask),
            "task": CHANGE_TASK,
            "name": before_path.stem,
        }


def _resolve_levir_cd_plus_root(root: str | Path) -> Path:
    """Resolve the original LEVIR-CD+ train/test/A/B/label layout."""

    configured = Path(root).expanduser()
    candidates = (
        configured,
        configured / "LEVIR-CD+",
        configured / "extracted" / "LEVIR-CD+",
    )
    for candidate in candidates:
        if all(
            (candidate / "train" / directory).is_dir()
            for directory in ("A", "B", "label")
        ):
            return candidate
    attempted = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "Could not find the LEVIR-CD+ train/A, train/B and train/label "
        f"directories. Tried:\n{attempted}"
    )


class LEVIRCDPlusDataset(Dataset):
    """LEVIR-CD+ before/after images and binary building-change masks.

    The released data has no validation directory. A deterministic subset of
    the 637 training pairs is therefore reserved for validation before any
    image is resized. The official 348-pair test split remains untouched.
    Images and labels are resized directly from 1024 to ``resolution``; RGB
    uses bilinear interpolation and masks use nearest-neighbour interpolation.
    """

    dataset_name = "LEVIR-CD+"

    def __init__(
        self,
        root: str | Path,
        split: str,
        resolution: int = 512,
        augment: bool = False,
        validation_fraction: float = 0.1,
        split_seed: int = 12345,
    ):
        self.root = _resolve_levir_cd_plus_root(root)
        self.split = str(split).strip().casefold()
        self.resolution = int(resolution)
        self.augment = bool(augment)
        self.validation_fraction = float(validation_fraction)
        self.split_seed = int(split_seed)
        if self.split not in {"train", "val", "test", "small_train"}:
            raise ValueError(
                "LEVIR-CD+ split must be train, val, test or small_train, "
                f"got {split!r}"
            )
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError(
                "validation_fraction must be in [0, 1), got "
                f"{self.validation_fraction}"
            )

        physical_split = "train" if self.split == "val" else self.split
        split_root = self.root / physical_split
        before_images = _paths_by_stem(split_root / "A")
        after_images = _paths_by_stem(split_root / "B")
        labels = _paths_by_stem(split_root / "label")
        reference = set(before_images)
        for name, paths in (("B", after_images), ("label", labels)):
            if set(paths) != reference:
                raise RuntimeError(
                    "LEVIR-CD+ filenames are not paired: "
                    f"split={physical_split!r}, directory={name!r}, "
                    f"missing={len(reference - set(paths))}, "
                    f"extra={len(set(paths) - reference)}"
                )

        stems = sorted(reference)
        if self.split in {"train", "val"} and self.validation_fraction > 0:
            shuffled = list(stems)
            random.Random(self.split_seed).shuffle(shuffled)
            validation_count = max(
                1,
                int(round(len(shuffled) * self.validation_fraction)),
            )
            validation_stems = set(shuffled[:validation_count])
            stems = [
                stem
                for stem in stems
                if (stem in validation_stems) == (self.split == "val")
            ]
        if not stems:
            raise RuntimeError(f"LEVIR-CD+ split {self.split!r} contains no samples")

        self.samples = [
            {
                "before_image": before_images[stem],
                "after_image": after_images[stem],
                "label": labels[stem],
            }
            for stem in stems
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        paths = self.samples[index]
        before_path = paths["before_image"]
        before = _normalized_rgb(before_path, self.resolution)
        after = _normalized_rgb(paths["after_image"], self.resolution)
        target = _binary_mask(paths["label"], self.resolution)
        if self.augment:
            before, after, target = _augment_pair(before, after, target)
        mask = _mask_tensor(target)
        return {
            "before": _image_tensor(before),
            "after": _image_tensor(after),
            "mask": mask,
            "loss_mask": torch.ones_like(mask),
            "target_rgb": _binary_target_rgb(mask),
            "task": CHANGE_TASK,
            "name": before_path.stem,
        }


def _resolve_urbansar_flood_root(root: str | Path) -> Path:
    configured = Path(root).expanduser()
    candidates = (configured, configured / "urban_sar_floods")
    for candidate in candidates:
        if (
            (candidate / "Train_dataset.txt").is_file()
            and (candidate / "Valid_dataset.txt").is_file()
            and (candidate / "01_NF" / "SAR").is_dir()
        ):
            return candidate
    attempted = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "Could not resolve the extracted UrbanSARFloods directory. Tried:\n"
        f"{attempted}"
    )


def _urbansar_norm_statistics(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing UrbanSARFloods normalization statistics: '{path}'"
        )
    text = path.read_text(encoding="utf-8", errors="replace")
    statistics: dict[str, np.ndarray] = {}
    for name in ("mean_in", "min_in", "max_in"):
        match = re.search(rf"{name}\s*:\s*\[([^\]]+)\]", text)
        if match is None:
            raise ValueError(f"Missing {name} in '{path}'")
        values = np.fromstring(match.group(1), sep=" ", dtype=np.float32)
        if values.shape != (8,) or not np.isfinite(values).all():
            raise ValueError(f"UrbanSARFloods {name} must contain eight finite values")
        statistics[name] = values
    if bool((statistics["max_in"] <= statistics["min_in"]).any()):
        raise ValueError("UrbanSARFloods max_in must be greater than min_in")
    return statistics


class UrbanSARFloodChangeDataset(Dataset):
    """UrbanSARFloods intensity/coherence pairs and NF/FO/FU labels.

    The released TIFFs use two layouts in practice (H,W,8 and 8,H,W), so the
    reader canonicalizes them explicitly. Band order follows the official
    release: pre/co-event coherence VH,VV, then pre/post intensity VH,VV.
    Each temporal condition is compressed to three channels for the frozen
    FLUX VAE as [intensity VV, intensity VH, mean coherence].
    """

    dataset_name = "UrbanSARFloods"
    categories = ("01_NF", "02_FO", "03_FU")

    def __init__(
        self,
        root: str | Path,
        split: str,
        resolution: int = 512,
        augment: bool = False,
        balance_categories: bool = False,
        normalization_file: str = "data_norm.txt",
    ):
        self.root = _resolve_urbansar_flood_root(root)
        self.split = str(split).strip().casefold()
        self.resolution = int(resolution)
        self.augment = bool(augment)
        self.balance_categories = bool(balance_categories)
        if self.resolution != 512:
            raise ValueError(
                "UrbanSARFloods is natively 512x512; this implementation "
                f"requires resolution=512, got {self.resolution}"
            )
        split_files = {
            "train": "Train_dataset.txt",
            "val": "Valid_dataset.txt",
            "valid": "Valid_dataset.txt",
            "validation": "Valid_dataset.txt",
            "test": "Test_dataset.txt",
        }
        if self.split not in split_files:
            raise ValueError(
                "UrbanSARFloods supports train/val/test manifests. Run "
                "prepare_urbansar_test_512.py once to create Test_dataset.txt "
                "from testing_case_orig. "
                f"Unsupported split '{split}'."
            )
        split_path = self.root / split_files[self.split]
        if self.split == "test" and not split_path.is_file():
            raise FileNotFoundError(
                f"Missing official UrbanSARFloods 512 test manifest: "
                f"'{split_path}'. Run prepare_urbansar_test_512.py first."
            )
        samples: list[dict[str, Path | str]] = []
        for raw_line in split_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            relative = raw_line.strip().replace("\\", "/")
            while relative.startswith("../"):
                relative = relative[3:]
            if not relative:
                continue
            gt_path = self.root / relative
            if "/GT/" not in f"/{relative}":
                raise ValueError(
                    f"Unexpected UrbanSARFloods manifest entry: {raw_line!r}"
                )
            sar_relative = relative.replace("/GT/", "/SAR/").replace(
                "_GT.tif", "_SAR.tif"
            )
            sar_path = self.root / sar_relative
            relative_parts = relative.split("/")
            if self.split == "test" and len(relative_parts) >= 2:
                category = relative_parts[1]
            else:
                category = relative_parts[0]
            if self.split != "test" and category not in self.categories:
                raise ValueError(f"Unknown UrbanSARFloods category in {raw_line!r}")
            valid_relative = relative.replace("/GT/", "/VALID/").replace(
                "_GT.tif", "_VALID.tif"
            )
            valid_path = self.root / valid_relative
            if self.split != "test" or not valid_path.is_file():
                valid_path = None
            if not gt_path.is_file() or not sar_path.is_file():
                raise FileNotFoundError(
                    f"Incomplete UrbanSARFloods pair: '{sar_path}', " f"'{gt_path}'"
                )
            samples.append(
                {
                    "name": (f"{category}_" f"{sar_path.stem.removesuffix('_SAR')}"),
                    "sar": sar_path,
                    "gt": gt_path,
                    "valid": valid_path,
                    "category": category,
                }
            )
        if not samples:
            raise ValueError(f"UrbanSARFloods split is empty: '{split_path}'")
        self.unique_samples = tuple(samples)
        if self.balance_categories and self.split == "train":
            grouped = {
                category: [
                    sample for sample in samples if sample["category"] == category
                ]
                for category in self.categories
            }
            if any(not group for group in grouped.values()):
                raise ValueError("Cannot balance UrbanSARFloods with an empty category")
            balanced_count = max(len(group) for group in grouped.values())
            samples = [
                group[index % len(group)]
                for category in self.categories
                for group in (grouped[category],)
                for index in range(balanced_count)
            ]
        self.samples = samples
        statistics = _urbansar_norm_statistics(self.root / str(normalization_file))
        self.band_mean = statistics["mean_in"]
        self.band_min = statistics["min_in"]
        self.band_max = statistics["max_in"]

    def __len__(self) -> int:
        return len(self.samples)

    def _read_sar(self, path: Path) -> np.ndarray:
        array = np.asarray(tifffile.imread(path), dtype=np.float32)
        if array.shape == (8, 512, 512):
            array = np.moveaxis(array, 0, -1)
        if array.shape != (512, 512, 8):
            raise RuntimeError(
                f"Expected UrbanSARFloods TIFF shape HxWx8 or 8xHxW, got "
                f"{array.shape} for '{path}'"
            )
        finite = np.isfinite(array)
        if not bool(finite.all()):
            array = np.where(
                finite,
                array,
                self.band_mean.reshape(1, 1, 8),
            )
        normalized = np.clip(
            (array - self.band_min.reshape(1, 1, 8))
            / (self.band_max - self.band_min).reshape(1, 1, 8),
            0.0,
            1.0,
        ).astype(np.float32)
        return normalized

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        sample = self.samples[index]
        bands = self._read_sar(Path(sample["sar"]))
        # Official order: [pre coh VH,VV, co coh VH,VV,
        #                  pre intensity VH,VV, post intensity VH,VV].
        before = np.stack(
            (bands[..., 5], bands[..., 4], bands[..., :2].mean(axis=-1)),
            axis=-1,
        )
        after = np.stack(
            (bands[..., 7], bands[..., 6], bands[..., 2:4].mean(axis=-1)),
            axis=-1,
        )
        labels = np.asarray(tifffile.imread(Path(sample["gt"])))
        labels = np.squeeze(labels)
        if labels.shape != (512, 512):
            raise RuntimeError(
                f"Expected UrbanSARFloods label shape 512x512, got "
                f"{labels.shape} for '{sample['gt']}'"
            )
        if not np.isfinite(labels).all():
            raise ValueError(f"UrbanSARFloods label contains NaN/Inf: '{sample['gt']}'")
        labels = labels.astype(np.int64)
        if labels.size and (labels.min() < 0 or labels.max() > 2):
            raise ValueError(f"UrbanSARFloods labels must be 0/1/2: '{sample['gt']}'")
        if self.augment:
            transform = _sample_geometric_augmentation()
            before = _apply_geometric_augmentation(before, transform)
            after = _apply_geometric_augmentation(after, transform)
            labels = _apply_geometric_augmentation(labels, transform)
        # Shift official 0/1/2 labels to internal 1/2/3 semantic IDs.
        internal_labels = labels + 1
        label_tensor = torch.from_numpy(
            np.ascontiguousarray(internal_labels[None].astype(np.float32))
        )
        valid_path = sample.get("valid")
        if valid_path is None:
            loss_mask = torch.ones_like(label_tensor)
        else:
            valid = np.squeeze(np.asarray(tifffile.imread(Path(valid_path))))
            if valid.shape != (512, 512):
                raise RuntimeError(
                    "Expected UrbanSARFloods validity mask shape 512x512, "
                    f"got {valid.shape} for '{valid_path}'"
                )
            loss_mask = torch.from_numpy(
                np.ascontiguousarray((valid > 0)[None].astype(np.float32))
            )
        target_rgb = (
            urbansar_flood_labels_to_rgb(internal_labels).astype(np.float32) / 255.0
        )
        return {
            "before": _image_tensor(before),
            "after": _image_tensor(after),
            "mask": label_tensor,
            "loss_mask": loss_mask,
            "target_rgb": _image_tensor(target_rgb),
            "task": SAR_CHANGE_TASK,
            "input_modality": "sar",
            "name": str(sample["name"]),
            "category": str(sample["category"]),
        }
