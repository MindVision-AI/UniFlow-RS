"""Prepare LaSeRS for the UniFlow-RS referring-segmentation evaluator.

LaSeRS differs from RRSIS-D in three important ways:

* one expression may refer to several categories and therefore contains
  several compressed COCO RLE masks;
* the official training release has no validation split;
* the nine test JSON files are overlapping benchmark views, not nine
  disjoint test partitions.

This script unions every sample's RLE masks, creates an image-disjoint and
source-stratified validation split from the official training images, and
writes both a deduplicated ``test`` split and all nine official test views.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from prepare_rrsisd import decode_segmentation


TEST_SUBSETS = {
    "test_explicit": "test_explicit.json",
    "test_implicit": "test_implicit.json",
    "test_instance_level": "test_instance_level.json",
    "test_part_level": "test_part_level.json",
    # The released filename contains the typo ``sematic``.  The manifest
    # exposes the correctly spelled split name.
    "test_semantic_level": "test_sematic_level.json",
    "test_long_query": "test_long_query.json",
    "test_short_query": "test_short_query.json",
    "test_single_cate": "test_single_cate.json",
    "test_multi_cate": "test_multi_cate.json",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare LaSeRS expressions and union masks at 512x512"
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/LaSeRS"),
        help="Root containing train/ and test/ from the LaSeRS release.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/lasers_512"),
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--validation-seed", type=int, default=12345)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Diagnostic only: retain at most this many unique expressions "
            "in train, val, and the deduplicated test split."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate prepared image and mask files that already exist.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list) or not all(
        isinstance(item, dict) for item in payload
    ):
        raise TypeError(f"Expected a list of annotations in '{path}'")
    return payload


def _atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def _canonical_sample_key(item: dict) -> str:
    """Identify the same released annotation across overlapping test JSONs."""

    identity = {
        "image_source": item.get("image_source"),
        "image_name": item.get("image_name"),
        "description": item.get("description"),
        "category": item.get("category"),
        "mask": item.get("mask"),
    }
    encoded = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _deduplicate(items: list[dict]) -> list[dict]:
    unique: dict[str, dict] = {}
    for item in items:
        unique.setdefault(_canonical_sample_key(item), item)
    return list(unique.values())


def _validation_images(
    items: list[dict],
    fraction: float,
    seed: int,
) -> set[tuple[str, str]]:
    """Select whole images per source so expressions never cross splits."""

    images_by_source: dict[str, set[str]] = defaultdict(set)
    for item in items:
        images_by_source[str(item["image_source"])].add(
            str(item["image_name"])
        )
    selected: set[tuple[str, str]] = set()
    for source, image_names in sorted(images_by_source.items()):
        ordered = sorted(
            image_names,
            key=lambda name: hashlib.sha256(
                f"{seed}\t{source}\t{name}".encode("utf-8")
            ).digest(),
        )
        count = max(1, int(round(len(ordered) * fraction)))
        if count >= len(ordered):
            raise ValueError(
                f"Validation split would consume every {source} image"
            )
        selected.update((source, name) for name in ordered[:count])
    return selected


def _prepare_image(
    source_path: Path,
    output_path: Path,
    resolution: int,
    jpeg_quality: int,
    overwrite: bool,
) -> tuple[int, int]:
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    with Image.open(source_path) as source:
        image = source.convert("RGB")
        source_width, source_height = image.size
        if overwrite or not output_path.is_file():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            resized = image.resize(
                (resolution, resolution),
                Image.Resampling.BILINEAR,
            )
            save_arguments = {}
            if output_path.suffix.casefold() in {".jpg", ".jpeg"}:
                save_arguments = {
                    "quality": jpeg_quality,
                    "subsampling": 0,
                }
            resized.save(output_path, **save_arguments)
    return source_width, source_height


def _prepare_entry(
    item: dict,
    prompt_index: int,
    origin: str,
    input_root: Path,
    output_root: Path,
    resolution: int,
    jpeg_quality: int,
    overwrite: bool,
    image_sizes: dict[tuple[str, str], tuple[int, int]],
    benchmark_subsets: list[str] | None = None,
) -> dict:
    expression = str(item.get("description", "")).strip()
    if not expression:
        raise ValueError(f"Empty LaSeRS expression at source id {item.get('id')}")
    image_name = str(item["image_name"])
    source_tag = str(item.get("image_source", "unknown"))
    source_path = input_root / origin / "images" / image_name
    relative_image = Path("images") / origin / image_name
    output_image = output_root / relative_image
    image_key = (origin, image_name)
    if image_key not in image_sizes:
        image_sizes[image_key] = _prepare_image(
            source_path,
            output_image,
            resolution,
            jpeg_quality,
            overwrite,
        )
    source_width, source_height = image_sizes[image_key]

    declared_size = item.get("size")
    if not isinstance(declared_size, list) or len(declared_size) != 2:
        raise ValueError(
            f"Invalid LaSeRS size at source id {item.get('id')}: "
            f"{declared_size!r}"
        )
    declared_height, declared_width = (int(value) for value in declared_size)
    if (declared_width, declared_height) != (source_width, source_height):
        raise ValueError(
            f"LaSeRS image/annotation size mismatch for '{source_path}': "
            f"image={(source_width, source_height)}, "
            f"annotation={(declared_width, declared_height)}"
        )

    # ``mask`` is a list with one compressed RLE per referred category.
    # decode_segmentation ORs every entry, producing the required union target.
    mask = decode_segmentation(
        item["mask"],
        source_height,
        source_width,
    )
    if mask.shape != (source_height, source_width):
        raise ValueError(
            f"LaSeRS RLE shape {mask.shape} != image shape "
            f"{(source_height, source_width)} for '{source_path}'"
        )
    relative_mask = (
        Path("masks") / origin / f"{prompt_index:08d}.png"
    )
    output_mask = output_root / relative_mask
    if overwrite or not output_mask.is_file():
        output_mask.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask * 255, mode="L").resize(
            (resolution, resolution),
            Image.Resampling.NEAREST,
        ).save(output_mask)

    categories = [str(value) for value in item.get("category", [])]
    return {
        "prompt_index": int(prompt_index),
        # Generated ids are unique even though six ids in train_data.json are
        # duplicated. The released id is retained separately for provenance.
        "ref_id": int(prompt_index),
        "source_annotation_id": int(item.get("id", -1)),
        "name": f"{Path(image_name).stem}_{prompt_index:08d}",
        "image": relative_image.as_posix(),
        "mask": relative_mask.as_posix(),
        "expression": expression,
        "category": ", ".join(categories),
        "categories": categories,
        "image_source": source_tag,
        "is_implicit": bool(item.get("is_implit", False)),
        "query_type": str(item.get("query_type", "")),
        "benchmark_subsets": sorted(benchmark_subsets or []),
        "source_width": source_width,
        "source_height": source_height,
        "target_count": len(item.get("mask", [])),
        "target_area": int(np.asarray(mask, dtype=np.uint8).sum()),
    }


def main() -> None:
    args = _parse_args()
    if args.resolution < 1:
        raise ValueError("--resolution must be positive")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be in (0, 1)")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be in [1, 100]")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be positive")

    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    settings = {
        "input_root": str(input_root),
        "resolution": int(args.resolution),
        "validation_fraction": float(args.validation_fraction),
        "validation_seed": int(args.validation_seed),
        "max_samples": args.max_samples,
    }
    settings_path = output_root / "preparation_settings.json"
    if settings_path.is_file() and not args.overwrite:
        with settings_path.open("r", encoding="utf-8") as file:
            previous_settings = json.load(file)
        if previous_settings != settings:
            raise RuntimeError(
                f"Existing LaSeRS preparation settings in '{settings_path}' "
                "do not match this run. Use a new --output-root or rerun "
                "with --overwrite."
            )
    if args.overwrite or not settings_path.is_file():
        _atomic_json(settings, settings_path)
    train_path = input_root / "train" / "annotations" / "train_data.json"
    train_items = _deduplicate(_read_json(train_path))
    validation_images = _validation_images(
        train_items,
        fraction=float(args.validation_fraction),
        seed=int(args.validation_seed),
    )
    raw_splits = {"train": [], "val": []}
    for item in train_items:
        image_key = (
            str(item["image_source"]),
            str(item["image_name"]),
        )
        raw_splits[
            "val" if image_key in validation_images else "train"
        ].append(item)
    if args.max_samples is not None:
        raw_splits = {
            split: items[: args.max_samples]
            for split, items in raw_splits.items()
        }

    test_annotations = input_root / "test" / "annotations"
    test_by_key: dict[str, dict] = {}
    test_subset_keys: dict[str, list[str]] = {}
    membership: dict[str, set[str]] = defaultdict(set)
    source_test_files: dict[str, str] = {}
    for split, filename in TEST_SUBSETS.items():
        path = test_annotations / filename
        source_test_files[split] = str(path)
        keys: list[str] = []
        for item in _read_json(path):
            key = _canonical_sample_key(item)
            previous = test_by_key.setdefault(key, item)
            if _canonical_sample_key(previous) != key:
                raise RuntimeError(f"Conflicting LaSeRS duplicate for {key}")
            if key not in membership or split not in membership[key]:
                keys.append(key)
            membership[key].add(split)
        test_subset_keys[split] = keys
    test_keys = list(test_by_key)
    if args.max_samples is not None:
        test_keys = test_keys[: args.max_samples]
        retained = set(test_keys)
        test_subset_keys = {
            split: [key for key in keys if key in retained]
            for split, keys in test_subset_keys.items()
        }

    splits: dict[str, list[dict]] = {
        "train": [],
        "val": [],
        "test": [],
        **{split: [] for split in TEST_SUBSETS},
    }
    prompt_hasher = hashlib.sha256()
    image_sizes: dict[tuple[str, str], tuple[int, int]] = {}
    prompt_index = 0

    for split in ("train", "val"):
        for item in tqdm(
            raw_splits[split],
            desc=f"Preparing LaSeRS {split}",
            unit="expression",
        ):
            entry = _prepare_entry(
                item,
                prompt_index,
                origin="train",
                input_root=input_root,
                output_root=output_root,
                resolution=int(args.resolution),
                jpeg_quality=int(args.jpeg_quality),
                overwrite=bool(args.overwrite),
                image_sizes=image_sizes,
            )
            splits[split].append(entry)
            prompt_hasher.update(
                f"{prompt_index}\t{entry['expression']}\n".encode("utf-8")
            )
            prompt_index += 1

    test_entries_by_key: dict[str, dict] = {}
    for key in tqdm(test_keys, desc="Preparing LaSeRS test", unit="expression"):
        item = test_by_key[key]
        entry = _prepare_entry(
            item,
            prompt_index,
            origin="test",
            input_root=input_root,
            output_root=output_root,
            resolution=int(args.resolution),
            jpeg_quality=int(args.jpeg_quality),
            overwrite=bool(args.overwrite),
            image_sizes=image_sizes,
            benchmark_subsets=list(membership[key]),
        )
        test_entries_by_key[key] = entry
        splits["test"].append(entry)
        prompt_hasher.update(
            f"{prompt_index}\t{entry['expression']}\n".encode("utf-8")
        )
        prompt_index += 1
    for split, keys in test_subset_keys.items():
        splits[split] = [test_entries_by_key[key] for key in keys]

    train_images = {
        (str(item["image_source"]), str(item["image_name"]))
        for item in raw_splits["train"]
    }
    val_images = {
        (str(item["image_source"]), str(item["image_name"]))
        for item in raw_splits["val"]
    }
    if train_images & val_images:
        raise RuntimeError("LaSeRS train/val image leakage detected")

    manifest = {
        "format": "referring_uniflow_rs_v2",
        "dataset_name": "LaSeRS",
        "resolution": int(args.resolution),
        "source_train": str(train_path),
        "source_test_files": source_test_files,
        "validation_fraction": float(args.validation_fraction),
        "validation_seed": int(args.validation_seed),
        "prompt_count": int(prompt_index),
        "prompt_sha256": prompt_hasher.hexdigest(),
        "split_counts": {
            split: len(entries) for split, entries in splits.items()
        },
        "unique_image_counts": {
            split: len({str(entry["image"]) for entry in entries})
            for split, entries in splits.items()
        },
        "splits": splits,
    }
    manifest_path = output_root / "splits.json"
    _atomic_json(manifest, manifest_path)
    print(json.dumps(manifest["split_counts"], indent=2))
    print(json.dumps(manifest["unique_image_counts"], indent=2))
    print(f"Saved: {manifest_path}")


if __name__ == "__main__":
    main()
