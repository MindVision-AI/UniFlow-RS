"""Binary building-mask data and latent helpers for UniFlow-RS fine-tuning."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from einops import rearrange, repeat
from PIL import Image, ImageDraw
from torch import Tensor
from torch.utils.data import Dataset

_IMAGE_DIRECTORY_NAMES = ("images", "image", "img")
_MASK_DIRECTORY_NAMES = ("masks", "mask", "labels", "label", "gt")


def _child_directory(
    parent: Path,
    configured_name: str | None,
    candidate_names: tuple[str, ...],
) -> Path | None:
    """Resolve a child directory without depending on filename case."""

    if not parent.is_dir():
        return None
    children = {
        child.name.casefold(): child for child in parent.iterdir() if child.is_dir()
    }
    if configured_name:
        configured_path = Path(configured_name)
        if configured_path.is_absolute():
            return configured_path if configured_path.is_dir() else None
        direct = parent / configured_path
        if direct.is_dir():
            return direct
        if len(configured_path.parts) == 1:
            return children.get(configured_path.name.casefold())
        return None
    return next(
        (
            children[name.casefold()]
            for name in candidate_names
            if name.casefold() in children
        ),
        None,
    )


def _resolve_dataset_directories(
    root: Path,
    split: str,
    image_dir_name: str | None,
    mask_dir_name: str | None,
) -> tuple[Path, Path, Path]:
    """Support both official and commonly repackaged WHU layouts."""

    split_candidates = (
        root / split,
        root / "AerialImageDataset" / split,
        root / "WHU Building Dataset" / "AerialImageDataset" / split,
        root,
    )
    attempted: list[Path] = []
    for split_root in split_candidates:
        if split_root in attempted:
            continue
        attempted.append(split_root)
        if split_root == root and root.name.casefold() != split.casefold():
            continue
        image_dir = _child_directory(
            split_root,
            image_dir_name,
            _IMAGE_DIRECTORY_NAMES,
        )
        mask_dir = _child_directory(
            split_root,
            mask_dir_name,
            _MASK_DIRECTORY_NAMES,
        )
        if image_dir is not None and mask_dir is not None:
            return split_root, image_dir, mask_dir

    attempted_text = "\n".join(f"  - {path}" for path in attempted)
    raise FileNotFoundError(
        "Could not find paired image/mask directories for split "
        f"'{split}'. Tried:\n{attempted_text}\n"
        "Supported names are Image/Mask, image/label, or images/masks. "
        "You can also set data.image_dir_name and data.mask_dir_name."
    )


class BuildingMaskDataset(Dataset):
    """Read RGB tiles and binary building masks from a dataset split.

    Supported out of the box:

    - Generic paired tiles: ``root/split/images`` and ``root/split/masks``.
    - WHU official: ``root/AerialImageDataset/split/image`` and ``label``.
    - WHU repackaged: ``root/split/Image`` and ``root/split/Mask``.

    Class-ID masks use the configured building ID. Binary WHU masks use
    255 for buildings and 0 for background.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        resolution: int = 512,
        building_id: int = 0,
        mask_format: str = "class_ids",
        augment: bool = False,
        image_dir_name: str | None = None,
        mask_dir_name: str | None = None,
    ):
        dataset_root = Path(root).expanduser()
        self.root, self.image_dir, self.mask_dir = _resolve_dataset_directories(
            dataset_root,
            str(split),
            image_dir_name,
            mask_dir_name,
        )
        self.resolution = int(resolution)
        self.building_id = int(building_id)
        self.mask_format = str(mask_format)
        self.augment = bool(augment)

        if self.mask_format not in {
            "class_ids",
            "binary_white",
            "binary_nonzero",
        }:
            raise ValueError(
                "mask_format must be 'class_ids', 'binary_white', or "
                "'binary_nonzero', got "
                f"'{self.mask_format}'"
            )
        extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        self.image_paths = sorted(
            path
            for path in self.image_dir.iterdir()
            if path.suffix.lower() in extensions
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in '{self.image_dir}'")

        masks_by_stem = {
            path.stem.casefold(): path
            for path in self.mask_dir.iterdir()
            if path.suffix.lower() in extensions
        }
        self.mask_paths: list[Path] = []
        for image_path in self.image_paths:
            candidates = [
                self.mask_dir / image_path.name,
                self.mask_dir / f"{image_path.stem}.png",
                self.mask_dir / f"{image_path.stem}.tif",
            ]
            mask_path = next(
                (path for path in candidates if path.is_file()),
                masks_by_stem.get(image_path.stem.casefold()),
            )
            if mask_path is None:
                raise FileNotFoundError(
                    f"No mask found for '{image_path.name}' in '{self.mask_dir}'"
                )
            self.mask_paths.append(mask_path)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        if image.size != (self.resolution, self.resolution):
            image = image.resize(
                (self.resolution, self.resolution),
                Image.Resampling.BILINEAR,
            )
        if mask.size != (self.resolution, self.resolution):
            mask = mask.resize(
                (self.resolution, self.resolution),
                Image.Resampling.NEAREST,
            )

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        mask_np = np.asarray(mask)
        if mask_np.ndim == 3:
            mask_np = mask_np[..., 0]
        if self.mask_format == "class_ids":
            building = mask_np == self.building_id
        elif self.mask_format == "binary_nonzero":
            building = mask_np != 0
        else:
            building = mask_np > 127

        if self.augment:
            rotation = int(np.random.randint(0, 4))
            image_np = np.rot90(image_np, rotation, axes=(0, 1))
            building = np.rot90(building, rotation, axes=(0, 1))
            if np.random.rand() < 0.5:
                image_np = np.fliplr(image_np)
                building = np.fliplr(building)
            if np.random.rand() < 0.5:
                image_np = np.flipud(image_np)
                building = np.flipud(building)

        image_t = torch.from_numpy(
            np.ascontiguousarray(image_np.transpose(2, 0, 1))
        ).float()
        image_t = image_t.mul(2.0).sub(1.0)
        mask_t = torch.from_numpy(
            np.ascontiguousarray(building[None].astype(np.float32))
        )
        return {
            "image": image_t,
            "mask": mask_t,
            "name": image_path.stem,
        }


class CocoBuildingDataset(Dataset):
    """Rasterize COCO building polygons and resize or tile large images."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        annotation_file: str | Path,
        resolution: int = 512,
        augment: bool = False,
        category_id: int = 1,
        spatial_mode: str = "resize",
    ):
        self.dataset_root = Path(root).expanduser()
        self.image_dir = self.dataset_root / split
        annotation_path = Path(annotation_file)
        if not annotation_path.is_absolute():
            annotation_path = self.dataset_root / annotation_path
        self.annotation_file = annotation_path
        self.resolution = int(resolution)
        self.augment = bool(augment)
        self.category_id = int(category_id)
        self.spatial_mode = str(spatial_mode).casefold()
        if self.spatial_mode not in {"resize", "grid"}:
            raise ValueError(
                "spatial_mode must be 'resize' or 'grid', got " f"'{self.spatial_mode}'"
            )

        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.annotation_file.is_file():
            raise FileNotFoundError(
                f"COCO annotation file not found: {self.annotation_file}"
            )
        with self.annotation_file.open("r", encoding="utf-8") as file:
            coco = json.load(file)

        annotations_by_image: dict[int, list[dict]] = defaultdict(list)
        for annotation in coco.get("annotations", []):
            if int(annotation.get("category_id", self.category_id)) != self.category_id:
                continue
            segmentation = annotation.get("segmentation")
            if isinstance(segmentation, dict):
                raise ValueError(
                    "RLE COCO masks are not supported by the WHU polygon loader"
                )
            annotations_by_image[int(annotation["image_id"])].append(annotation)
        self.annotations_by_image = dict(annotations_by_image)

        # Each sample is (COCO image record, left, top). Resize mode produces
        # one 512x512 sample per source image. Grid mode is retained as an
        # optional alternative for preserving native spatial resolution.
        self.samples: list[tuple[dict, int, int]] = []
        for record in sorted(coco.get("images", []), key=lambda item: int(item["id"])):
            image_path = self.image_dir / str(record["file_name"])
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"COCO image '{record['file_name']}' not found in "
                    f"'{self.image_dir}'"
                )
            width = int(record["width"])
            height = int(record["height"])
            if width < self.resolution or height < self.resolution:
                raise ValueError(
                    f"Image '{image_path}' is {width}x{height}, smaller than "
                    f"the configured output size {self.resolution}"
                )
            if self.spatial_mode == "resize":
                self.samples.append((record, 0, 0))
                continue
            x_positions = list(range(0, width - self.resolution + 1, self.resolution))
            y_positions = list(range(0, height - self.resolution + 1, self.resolution))
            if x_positions[-1] != width - self.resolution:
                x_positions.append(width - self.resolution)
            if y_positions[-1] != height - self.resolution:
                y_positions.append(height - self.resolution)
            for top in y_positions:
                for left in x_positions:
                    self.samples.append((record, left, top))

        if not self.samples:
            raise FileNotFoundError(f"No COCO images found in '{self.annotation_file}'")

    def __len__(self) -> int:
        return len(self.samples)

    def _rasterize_mask(
        self,
        record: dict,
        left: int,
        top: int,
        width: int,
        height: int,
    ) -> Image.Image:
        mask = Image.new("L", (width, height), color=0)
        draw = ImageDraw.Draw(mask)
        right = left + width
        bottom = top + height
        for annotation in self.annotations_by_image.get(int(record["id"]), []):
            bbox = annotation.get("bbox")
            if bbox is not None:
                box_left, box_top, box_width, box_height = map(float, bbox)
                if (
                    box_left + box_width < left
                    or box_left >= right
                    or box_top + box_height < top
                    or box_top >= bottom
                ):
                    continue
            for polygon in annotation.get("segmentation", []):
                if len(polygon) < 6:
                    continue
                points = [
                    (
                        float(polygon[index]) - left,
                        float(polygon[index + 1]) - top,
                    )
                    for index in range(0, len(polygon) - 1, 2)
                ]
                draw.polygon(points, fill=255)
        return mask

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        record, left, top = self.samples[index]
        image_path = self.image_dir / str(record["file_name"])
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            if self.spatial_mode == "resize":
                image = image.resize(
                    (self.resolution, self.resolution),
                    Image.Resampling.BILINEAR,
                )
            else:
                image = image.crop(
                    (
                        left,
                        top,
                        left + self.resolution,
                        top + self.resolution,
                    )
                )
        if self.spatial_mode == "resize":
            mask = self._rasterize_mask(
                record,
                left=0,
                top=0,
                width=int(record["width"]),
                height=int(record["height"]),
            ).resize(
                (self.resolution, self.resolution),
                Image.Resampling.NEAREST,
            )
        else:
            mask = self._rasterize_mask(
                record,
                left=left,
                top=top,
                width=self.resolution,
                height=self.resolution,
            )

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        building = np.asarray(mask, dtype=np.uint8) > 127
        if self.augment:
            rotation = int(np.random.randint(0, 4))
            image_np = np.rot90(image_np, rotation, axes=(0, 1))
            building = np.rot90(building, rotation, axes=(0, 1))
            if np.random.rand() < 0.5:
                image_np = np.fliplr(image_np)
                building = np.fliplr(building)
            if np.random.rand() < 0.5:
                image_np = np.flipud(image_np)
                building = np.flipud(building)

        image_t = torch.from_numpy(
            np.ascontiguousarray(image_np.transpose(2, 0, 1))
        ).float()
        image_t = image_t.mul(2.0).sub(1.0)
        mask_t = torch.from_numpy(
            np.ascontiguousarray(building[None].astype(np.float32))
        )
        return {
            "image": image_t,
            "mask": mask_t,
            "name": (
                Path(str(record["file_name"])).stem
                if self.spatial_mode == "resize"
                else (
                    f"{Path(str(record['file_name'])).stem}" f"_y{top:04d}_x{left:04d}"
                )
            ),
        }


def dataset_kwargs_from_config(data: dict) -> dict:
    """Build shared dataset arguments from a YAML ``data`` section."""

    return {
        "root": data["root"],
        "resolution": int(data.get("resolution", 512)),
        "building_id": int(data.get("building_id", 0)),
        "mask_format": str(data.get("mask_format", "class_ids")),
        "image_dir_name": data.get("image_dir_name"),
        "mask_dir_name": data.get("mask_dir_name"),
    }


def build_dataset_from_config(
    data: dict,
    logical_split: str,
    augment: bool,
) -> Dataset:
    """Create either a paired-raster or COCO-polygon building dataset."""

    split = str(data.get(f"{logical_split}_split", logical_split))
    dataset_format = str(data.get("dataset_format", "paired_raster")).casefold()
    if dataset_format == "coco_polygons":
        annotation_dir = Path(str(data.get("annotation_dir", "annotation")))
        annotation_files = data.get("annotation_files", {})
        annotation_file = annotation_files.get(
            logical_split,
            annotation_dir / f"{split}.json",
        )
        return CocoBuildingDataset(
            root=data["root"],
            split=split,
            annotation_file=annotation_file,
            resolution=int(data.get("resolution", 512)),
            augment=augment,
            category_id=int(data.get("category_id", 1)),
            spatial_mode=str(data.get("spatial_mode", "resize")),
        )
    if dataset_format != "paired_raster":
        raise ValueError(
            "data.dataset_format must be 'paired_raster' or "
            f"'coco_polygons', got '{dataset_format}'"
        )
    return BuildingMaskDataset(
        split=split,
        augment=augment,
        **dataset_kwargs_from_config(data),
    )


def pack_latent(latent: Tensor) -> Tensor:
    if latent.ndim != 4 or latent.shape[-2] % 2 or latent.shape[-1] % 2:
        raise ValueError(
            "UniFlow-RS latent must be [B,C,H,W] with even H and W, got "
            f"{tuple(latent.shape)}"
        )
    return rearrange(
        latent,
        "b c (h ph) (w pw) -> b (h w) (c ph pw)",
        ph=2,
        pw=2,
    )


def unpack_latent(tokens: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        tokens,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )


def make_image_ids(
    batch_size: int,
    latent_height: int,
    latent_width: int,
    device: torch.device,
) -> Tensor:
    ids = torch.zeros(
        latent_height // 2,
        latent_width // 2,
        3,
        dtype=torch.float32,
        device=device,
    )
    ids[..., 1] += torch.arange(latent_height // 2, device=device, dtype=torch.float32)[
        :, None
    ]
    ids[..., 2] += torch.arange(latent_width // 2, device=device, dtype=torch.float32)[
        None, :
    ]
    return repeat(ids, "h w c -> b (h w) c", b=batch_size)


def decoded_rgb_to_probability(decoded: Tensor) -> Tensor:
    if decoded.ndim != 4 or decoded.shape[1] != 3:
        raise ValueError(f"Expected decoded RGB [B,3,H,W], got {tuple(decoded.shape)}")
    return decoded.clamp(-1.0, 1.0).mean(dim=1, keepdim=True).add(1.0).mul(0.5)


def binary_metrics(counts: Tensor) -> dict[str, float]:
    """Convert ``[TP, FP, FN, TN, absolute_error, pixels]`` to metrics."""

    tp, fp, fn, tn, absolute_error, pixels = [float(value) for value in counts]
    building_iou = tp / max(tp + fp + fn, 1.0)
    building_dice = 2.0 * tp / max(2.0 * tp + fp + fn, 1.0)
    background_iou = tn / max(tn + fp + fn, 1.0)
    background_dice = 2.0 * tn / max(2.0 * tn + fp + fn, 1.0)
    return {
        "pixel_accuracy": (tp + tn) / max(tp + fp + fn + tn, 1.0),
        "building_iou": building_iou,
        "building_dice": building_dice,
        "background_iou": background_iou,
        "background_dice": background_dice,
        "mean_iou": 0.5 * (building_iou + background_iou),
        "mean_dice": 0.5 * (building_dice + background_dice),
        "mask_mae": absolute_error / max(pixels, 1.0),
    }
