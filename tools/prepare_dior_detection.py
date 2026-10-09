"""Prepare DIOR-R for the UniFlow-RS rotated-detection task.

Each 800x800 DIOR-R image is resized directly to the model resolution. There
is exactly one prepared sample per source image; no crop, overlap, or visible
fraction filtering is applied. OBB coordinates and raster targets are written
in the configured model coordinate system so detection recovery and evaluation
use the same geometry as the model output. Setting ``--resolution 800`` keeps
the native DIOR-R spatial resolution without tiling.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from uniflow_rs._compat.dior import DIOR_CLASS_NAMES


MANIFEST_FORMAT = "dior_uniflow_rs_oriented_resize_v1"
CLASS_TO_ID = {name: index for index, name in enumerate(DIOR_CLASS_NAMES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resize DIOR-R full images for UniFlow-RS evaluation"
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-size", type=int, default=800)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--boundary-width", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _normalise_class_name(value: str) -> str:
    return value.strip().casefold().replace("_", "-").replace(" ", "-")


def parse_annotation(path: Path) -> list[dict]:
    root = ET.parse(path).getroot()
    aliases = {
        "baseball-field": "baseballfield",
        "basketball-court": "basketballcourt",
        "golf-field": "golffield",
        "ground-track-field": "groundtrackfield",
        "storage-tank": "storagetank",
        "tennis-court": "tenniscourt",
        "train-station": "trainstation",
    }
    objects: list[dict] = []
    for object_node in root.findall("object"):
        raw_name = object_node.findtext("name", "")
        class_name = aliases.get(
            _normalise_class_name(raw_name),
            _normalise_class_name(raw_name),
        )
        if class_name not in CLASS_TO_ID:
            raise ValueError(f"Unknown DIOR-R class {raw_name!r} in {path}")
        box = object_node.find("robndbox")
        if box is None:
            raise ValueError(f"DIOR-R annotation has no robndbox in {path}")
        polygon = np.asarray(
            [
                [
                    float(box.findtext(x_name, "0")),
                    float(box.findtext(y_name, "0")),
                ]
                for x_name, y_name in (
                    ("x_left_top", "y_left_top"),
                    ("x_right_top", "y_right_top"),
                    ("x_right_bottom", "y_right_bottom"),
                    ("x_left_bottom", "y_left_bottom"),
                )
            ],
            dtype=np.float32,
        )
        area = abs(float(cv2.contourArea(polygon)))
        if area <= 0.0:
            continue
        objects.append(
            {
                "object_id": len(objects),
                "class_id": CLASS_TO_ID[class_name],
                "class_name": class_name,
                "polygon": polygon,
                "difficult": object_node.findtext("difficult", "0") == "1",
                "area": area,
            }
        )
    return objects


def _scale_polygon(
    polygon: np.ndarray,
    width: int,
    height: int,
    resolution: int,
) -> np.ndarray:
    scale = np.asarray(
        (resolution / float(width), resolution / float(height)),
        dtype=np.float32,
    )
    return polygon * scale


def _render_labels(
    objects: list[dict],
    width: int,
    height: int,
    resolution: int,
    boundary_width: int,
) -> tuple[Image.Image, list[dict]]:
    labels = Image.new("L", (resolution, resolution), 1)
    draw = ImageDraw.Draw(labels)
    prepared_objects: list[dict] = []
    for item in sorted(objects, key=lambda value: value["area"], reverse=True):
        scaled = _scale_polygon(
            item["polygon"],
            width=width,
            height=height,
            resolution=resolution,
        )
        points = [tuple(point) for point in scaled.tolist()]
        draw.polygon(points, fill=int(item["class_id"]) + 2)
        if boundary_width > 0:
            draw.line(
                points + [points[0]],
                fill=1,
                width=boundary_width,
                joint="curve",
            )
        prepared_objects.append(
            {
                "object_id": int(item["object_id"]),
                "class_id": int(item["class_id"]),
                "difficult": bool(item["difficult"]),
                "polygon": scaled.reshape(-1).astype(float).tolist(),
            }
        )
    return labels, prepared_objects


def resolve_image(root: Path, split: str, image_id: str) -> Path:
    directory = root / (
        "JPEGImages-test" if split == "test" else "JPEGImages-trainval"
    )
    for suffix in (".jpg", ".jpeg", ".png"):
        path = directory / f"{image_id}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Missing DIOR-R image for {split}/{image_id} in {directory}"
    )


def process_image(
    image_path: Path,
    annotation_path: Path,
    split: str,
    output_root: Path,
    source_size: int,
    resolution: int,
    boundary_width: int,
    overwrite: bool,
) -> dict:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        width, height = image.size
    if (width, height) != (source_size, source_size):
        raise ValueError(
            f"DIOR-R expects {source_size}x{source_size} images, got "
            f"{width}x{height} for {image_path}"
        )
    objects = parse_annotation(annotation_path)
    labels, prepared_objects = _render_labels(
        objects,
        width=width,
        height=height,
        resolution=resolution,
        boundary_width=boundary_width,
    )
    name = image_path.stem
    image_relative = Path("images") / split / f"{name}.jpg"
    target_relative = Path("targets") / split / f"{name}.png"
    valid_relative = Path("valid_masks") / split / f"{name}.png"
    image_output = output_root / image_relative
    target_output = output_root / target_relative
    valid_output = output_root / valid_relative

    resized = image.resize((resolution, resolution), Image.Resampling.BILINEAR)
    valid = Image.new("L", (resolution, resolution), 255)
    image_output.parent.mkdir(parents=True, exist_ok=True)
    target_output.parent.mkdir(parents=True, exist_ok=True)
    valid_output.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not image_output.is_file():
        resized.save(image_output, quality=95, optimize=True)
    if overwrite or not target_output.is_file():
        labels.save(target_output, compress_level=2)
    if overwrite or not valid_output.is_file():
        valid.save(valid_output, compress_level=2)

    return {
        "name": name,
        "parent": name,
        "image": image_relative.as_posix(),
        "target": target_relative.as_posix(),
        "valid_mask": valid_relative.as_posix(),
        "labeled": True,
        "x": 0,
        "y": 0,
        "valid_width": resolution,
        "valid_height": resolution,
        "source_width": resolution,
        "source_height": resolution,
        "original_width": width,
        "original_height": height,
        "objects": prepared_objects,
    }


def main() -> None:
    args = parse_args()
    if args.source_size < 1 or args.resolution < 1:
        raise ValueError("source-size and resolution must be positive")
    if args.boundary_width < 0:
        raise ValueError("boundary-width must be non-negative")
    root = args.input_root.expanduser().resolve()
    annotation_root = root / "Annotations" / "Oriented Bounding Boxes"
    split_root = root / "ImageSets" / "Main"
    if not annotation_root.is_dir() or not split_root.is_dir():
        raise FileNotFoundError(f"Invalid DIOR-R root: {root}")
    output_root = args.output_root.expanduser().resolve()
    for directory in ("images", "targets", "valid_masks"):
        for split in ("train", "val", "test"):
            (output_root / directory / split).mkdir(
                parents=True,
                exist_ok=True,
            )

    manifest: dict = {
        "format": MANIFEST_FORMAT,
        "source_root": str(root),
        "source_size": int(args.source_size),
        "resolution": int(args.resolution),
        "resize_mode": "direct_bilinear",
        "target_rendering": "scaled_source_polygons",
        "boundary_width": int(args.boundary_width),
        "tile_size": int(args.resolution),
        "overlap": 0,
        "class_names": list(DIOR_CLASS_NAMES),
        "splits": {},
    }
    max_objects = 0
    for split in ("train", "val", "test"):
        split_file = split_root / f"{split}.txt"
        image_ids = [
            line.strip()
            for line in split_file.read_text().splitlines()
            if line.strip()
        ]
        if args.max_images is not None:
            image_ids = image_ids[: int(args.max_images)]
        jobs = []
        for image_id in image_ids:
            annotation = annotation_root / f"{image_id}.xml"
            if not annotation.is_file():
                raise FileNotFoundError(annotation)
            jobs.append((resolve_image(root, split, image_id), annotation))

        def run(job: tuple[Path, Path]) -> dict:
            return process_image(
                job[0],
                job[1],
                split,
                output_root,
                args.source_size,
                args.resolution,
                args.boundary_width,
                args.overwrite,
            )

        samples: list[dict] = []
        with ThreadPoolExecutor(
            max_workers=max(1, int(args.workers))
        ) as executor:
            iterator = executor.map(run, jobs)
            for sample in tqdm(
                iterator,
                total=len(jobs),
                desc=f"Preparing DIOR-R {split}",
            ):
                samples.append(sample)
        for sample in samples:
            max_objects = max(max_objects, len(sample["objects"]))
        manifest["splits"][split] = samples

    manifest["max_objects_per_tile"] = max_objects
    manifest["counts"] = {
        split: len(samples)
        for split, samples in manifest["splits"].items()
    }
    manifest_path = output_root / "splits.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_root),
                **manifest["counts"],
                "max_objects_per_image": max_objects,
            },
            indent=2,
        )
    )
    print(f"Saved: {manifest_path}")


if __name__ == "__main__":
    main()
