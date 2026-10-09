"""Prepare the official RRSIS-D release for UniFlow-RS evaluation.

The public archive stores one COCO-style compressed RLE mask per referring
expression in ``instances.json`` and the corresponding language metadata in
``refs(unc).p``.  This script safely reads the built-in-only pickle, decodes
the masks without requiring pycocotools, resizes image/mask pairs together,
and writes a deterministic manifest consumed by ``RRSISDReferringDataset``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw
from tqdm.auto import tqdm


class _BuiltinsOnlyUnpickler(pickle.Unpickler):
    """Reject arbitrary globals while loading the official plain-data pickle."""

    def find_class(self, module: str, name: str):
        raise pickle.UnpicklingError(
            f"RRSIS-D metadata unexpectedly requested {module}.{name}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare RRSIS-D image/expression/mask samples at 512x512"
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/RRSIS-D"),
        help="Root of the downloaded official RRSIS-D archive.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/rrsisd_512"),
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Diagnostic only: keep at most this many samples per split.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate image and mask files that already exist.",
    )
    return parser.parse_args()


def _find_file(root: Path, candidates: Iterable[Path], label: str) -> Path:
    attempted: list[Path] = []
    for relative in candidates:
        path = root / relative
        attempted.append(path)
        if path.is_file():
            return path
    details = "\n".join(f"  - {path}" for path in attempted)
    raise FileNotFoundError(f"Could not find RRSIS-D {label}. Tried:\n{details}")


def _find_directory(
    root: Path,
    candidates: Iterable[Path],
    label: str,
) -> Path:
    attempted: list[Path] = []
    for relative in candidates:
        path = root / relative
        attempted.append(path)
        if path.is_dir():
            return path
    details = "\n".join(f"  - {path}" for path in attempted)
    raise FileNotFoundError(f"Could not find RRSIS-D {label}. Tried:\n{details}")


def resolve_release(root: Path) -> tuple[Path, Path, Path]:
    """Resolve both the official and the observed extracted folder layouts."""

    root = root.expanduser().resolve()
    refs = _find_file(
        root,
        (
            Path("rrsisd/refs(unc).p"),
            Path("refs(unc).p"),
        ),
        "refs(unc).p",
    )
    instances = _find_file(
        root,
        (
            Path("rrsisd/instances.json"),
            Path("instances.json"),
        ),
        "instances.json",
    )
    images = _find_directory(
        root,
        (
            Path("images/JPEGImages"),
            Path("images/rrsisd/JPEGImages"),
            Path("rrsisd/images/rrsisd/JPEGImages"),
            Path("JPEGImages"),
        ),
        "JPEGImages directory",
    )
    return refs, instances, images


def load_refs(path: Path) -> list[dict]:
    with path.open("rb") as file:
        refs = _BuiltinsOnlyUnpickler(file).load()
    if not isinstance(refs, list) or not all(
        isinstance(item, dict) for item in refs
    ):
        raise TypeError(f"Unexpected RRSIS-D refs payload in '{path}'")
    return refs


def _compressed_rle_counts(encoded: str | bytes) -> list[int]:
    """Decode COCO's compressed ASCII run lengths.

    This follows the public COCO maskApi delta/variable-length representation.
    The returned runs alternate background and foreground in Fortran order.
    """

    data = encoded.encode("ascii") if isinstance(encoded, str) else encoded
    counts: list[int] = []
    position = 0
    while position < len(data):
        value = 0
        shift = 0
        more = True
        last = 0
        while more:
            if position >= len(data):
                raise ValueError("Truncated compressed COCO RLE")
            last = int(data[position]) - 48
            position += 1
            value |= (last & 0x1F) << (5 * shift)
            more = bool(last & 0x20)
            shift += 1
        if last & 0x10:
            value |= -1 << (5 * shift)
        if len(counts) > 2:
            value += counts[-2]
        if value < 0:
            raise ValueError(f"Negative COCO RLE run length: {value}")
        counts.append(value)
    return counts


def _decode_rle(rle: dict) -> np.ndarray:
    size = rle.get("size")
    if (
        not isinstance(size, list)
        or len(size) != 2
        or min(int(value) for value in size) < 1
    ):
        raise ValueError(f"Invalid COCO RLE size: {size!r}")
    height, width = (int(value) for value in size)
    raw_counts = rle.get("counts")
    if isinstance(raw_counts, list):
        counts = [int(value) for value in raw_counts]
    elif isinstance(raw_counts, (str, bytes)):
        counts = _compressed_rle_counts(raw_counts)
    else:
        raise TypeError(f"Unsupported COCO RLE counts: {type(raw_counts)}")

    flat = np.zeros(height * width, dtype=np.uint8)
    offset = 0
    foreground = False
    for run in counts:
        end = offset + int(run)
        if end > flat.size:
            raise ValueError(
                f"COCO RLE exceeds {height}x{width}: {end} > {flat.size}"
            )
        if foreground:
            flat[offset:end] = 1
        offset = end
        foreground = not foreground
    if offset != flat.size:
        raise ValueError(
            f"COCO RLE covers {offset} pixels, expected {flat.size}"
        )
    return flat.reshape((height, width), order="F")


def decode_segmentation(
    segmentation,
    height: int,
    width: int,
) -> np.ndarray:
    """Decode RLE or polygon COCO segmentation into one binary mask."""

    if isinstance(segmentation, dict):
        segments = [segmentation]
    elif isinstance(segmentation, list):
        segments = segmentation
    else:
        raise TypeError(
            f"Unsupported segmentation type: {type(segmentation).__name__}"
        )
    rle_shapes = {
        (int(segment["size"][0]), int(segment["size"][1]))
        for segment in segments
        if isinstance(segment, dict)
    }
    if len(rle_shapes) > 1:
        raise ValueError(
            f"One annotation contains inconsistent RLE sizes: {rle_shapes}"
        )
    canvas_height, canvas_width = (
        next(iter(rle_shapes))
        if rle_shapes
        else (height, width)
    )
    result = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    for segment in segments:
        if isinstance(segment, dict):
            decoded = _decode_rle(segment)
            if decoded.shape != result.shape:
                raise ValueError(
                    f"RLE shape {decoded.shape} != expected {result.shape}"
                )
            result |= decoded
            continue
        if not isinstance(segment, list) or len(segment) < 6:
            raise ValueError("Invalid COCO polygon segmentation")
        polygon = [
            (float(segment[index]), float(segment[index + 1]))
            for index in range(0, len(segment), 2)
        ]
        canvas = Image.new("L", (canvas_width, canvas_height), color=0)
        ImageDraw.Draw(canvas).polygon(polygon, fill=1)
        result |= np.asarray(canvas, dtype=np.uint8)
    return result


def _atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def main() -> None:
    args = _parse_args()
    if args.resolution < 1:
        raise ValueError("--resolution must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be in [1, 100]")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be positive")

    refs_path, instances_path, image_dir = resolve_release(args.input_root)
    refs = load_refs(refs_path)
    with instances_path.open("r", encoding="utf-8") as file:
        instances = json.load(file)
    annotations = {
        int(annotation["id"]): annotation
        for annotation in instances["annotations"]
    }
    categories = {
        int(category["id"]): str(category["name"])
        for category in instances["categories"]
    }
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    splits: dict[str, list[dict]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    prompt_hasher = hashlib.sha256()
    kept_per_split = {split: 0 for split in splits}
    empty_masks_per_split = {split: 0 for split in splits}
    mask_size_mismatches_per_split = {split: 0 for split in splits}

    for prompt_index, ref in enumerate(
        tqdm(refs, desc="Preparing RRSIS-D", unit="sample")
    ):
        split = str(ref["split"]).casefold()
        if split not in splits:
            raise ValueError(f"Unsupported RRSIS-D split: {split!r}")
        if (
            args.max_samples is not None
            and kept_per_split[split] >= args.max_samples
        ):
            continue
        sentences = ref.get("sentences", [])
        if len(sentences) != 1:
            raise ValueError(
                f"Expected one RRSIS-D expression for ref {ref.get('ref_id')}"
            )
        expression = str(
            sentences[0].get("raw") or sentences[0].get("sent")
        ).strip()
        if not expression:
            raise ValueError(f"Empty expression for ref {ref.get('ref_id')}")
        annotation_id = int(ref["ann_id"])
        annotation = annotations.get(annotation_id)
        if annotation is None:
            raise KeyError(f"Missing annotation id {annotation_id}")
        file_name = str(ref["file_name"])
        source_path = image_dir / file_name
        if not source_path.is_file():
            raise FileNotFoundError(source_path)

        relative_image = Path("images") / split / file_name
        relative_mask = (
            Path("masks") / split / f"{Path(file_name).stem}.png"
        )
        output_image = output_root / relative_image
        output_mask = output_root / relative_mask
        output_image.parent.mkdir(parents=True, exist_ok=True)
        output_mask.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(source_path) as source:
            image = source.convert("RGB")
            source_width, source_height = image.size
            if args.overwrite or not output_image.is_file():
                resized = image.resize(
                    (args.resolution, args.resolution),
                    Image.Resampling.BILINEAR,
                )
                resized.save(
                    output_image,
                    quality=args.jpeg_quality,
                    subsampling=0,
                )
        mask = decode_segmentation(
            annotation["segmentation"],
            source_height,
            source_width,
        )
        if mask.shape != (source_height, source_width):
            # Fourteen official samples have an RLE canvas of 800x800 while
            # the JPEG is one to sixteen rows shorter. Both are mapped to the
            # same 512x512 coordinates, matching the official resize protocol.
            mask_size_mismatches_per_split[split] += 1
        # The official release contains three zero-area RLEs (two train, one
        # test). Preserve them instead of silently replacing precise masks
        # with bounding boxes or changing the published split cardinalities.
        if not bool(mask.any()):
            empty_masks_per_split[split] += 1
        if args.overwrite or not output_mask.is_file():
            mask_image = Image.fromarray(mask * 255, mode="L").resize(
                (args.resolution, args.resolution),
                Image.Resampling.NEAREST,
            )
            mask_image.save(output_mask)

        category_id = int(ref["category_id"])
        entry = {
            "prompt_index": int(prompt_index),
            "ref_id": int(ref["ref_id"]),
            "image_id": int(ref["image_id"]),
            "annotation_id": annotation_id,
            "name": Path(file_name).stem,
            "image": relative_image.as_posix(),
            "mask": relative_mask.as_posix(),
            "expression": expression,
            "category_id": category_id,
            "category": categories.get(category_id, str(category_id)),
            "source_width": source_width,
            "source_height": source_height,
            "mask_source_width": int(mask.shape[1]),
            "mask_source_height": int(mask.shape[0]),
        }
        splits[split].append(entry)
        kept_per_split[split] += 1
        prompt_hasher.update(
            f"{prompt_index}\t{expression}\n".encode("utf-8")
        )

    manifest = {
        "format": "rrsisd_uniflow_rs_v1",
        "resolution": int(args.resolution),
        "source_refs": str(refs_path),
        "source_instances": str(instances_path),
        "prompt_count": sum(len(entries) for entries in splits.values()),
        "prompt_sha256": prompt_hasher.hexdigest(),
        "split_counts": {
            split: len(entries) for split, entries in splits.items()
        },
        "empty_mask_counts": empty_masks_per_split,
        "mask_size_mismatch_counts": mask_size_mismatches_per_split,
        "splits": splits,
    }
    _atomic_json(manifest, output_root / "splits.json")
    print(json.dumps(manifest["split_counts"], indent=2))
    if any(empty_masks_per_split.values()):
        print(
            "Official zero-area masks preserved: "
            + json.dumps(empty_masks_per_split)
        )
    if any(mask_size_mismatches_per_split.values()):
        print(
            "Official JPEG/RLE size mismatches aligned by common resize: "
            + json.dumps(mask_size_mismatches_per_split)
        )
    print(f"Saved: {output_root / 'splits.json'}")


if __name__ == "__main__":
    main()
