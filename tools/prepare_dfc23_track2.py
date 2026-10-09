"""Prepare paired DFC23 Track 2 RGB/SAR samples for height estimation."""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import zipfile
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/dfc23_track2_raw"),
        help="Directory containing the DFC23 Track 2 train.zip archive.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/dfc23_track2_512"),
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--sar-db-min", type=float, default=-16.0)
    parser.add_argument("--sar-db-max", type=float, default=4.5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _find_one(root: Path, candidates: tuple[str, ...]) -> Path:
    for name in candidates:
        path = root / name
        if path.is_file():
            return path
    recursive = [
        path
        for name in candidates
        for path in root.rglob(name)
        if path.is_file()
    ]
    if len(recursive) == 1:
        return recursive[0]
    raise FileNotFoundError(
        f"Could not uniquely locate any of {candidates} below '{root}'"
    )


def _read_tiff(archive: zipfile.ZipFile, member: str) -> np.ndarray:
    try:
        payload = archive.read(member)
    except KeyError as error:
        raise FileNotFoundError(
            f"Missing '{member}' in {archive.filename}"
        ) from error
    return np.asarray(tifffile.imread(io.BytesIO(payload)))


def _rgb_uint8(array: np.ndarray) -> np.ndarray:
    if array.ndim == 3 and array.shape[0] in {1, 3, 4}:
        array = np.moveaxis(array, 0, -1)
    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError(f"Expected RGB TIFF, got shape {array.shape}")
    array = array[..., :3]
    if array.dtype == np.uint8:
        return array
    finite = np.nan_to_num(array.astype(np.float32), nan=0.0)
    if finite.max(initial=0.0) <= 1.0:
        finite *= 255.0
    return np.clip(np.rint(finite), 0, 255).astype(np.uint8)


def _sar_uint8(
    array: np.ndarray,
    db_min: float,
    db_max: float,
) -> np.ndarray:
    array = np.squeeze(array).astype(np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected one SAR channel, got shape {array.shape}")
    linear = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    db = 10.0 * np.log10(np.maximum(linear, 1.0e-6))
    normalized = (db - db_min) / (db_max - db_min)
    return np.clip(np.rint(normalized * 255.0), 0, 255).astype(np.uint8)


def _height_float(array: np.ndarray) -> np.ndarray:
    array = np.squeeze(array).astype(np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected one DSM channel, got shape {array.shape}")
    return np.nan_to_num(array, nan=0.0, posinf=200.0, neginf=0.0).clip(
        0.0
    )


def _sensor_city(name: str) -> tuple[str, str]:
    pieces = Path(name).stem.split("_")
    sensor = pieces[0] if pieces else "unknown"
    city = pieces[1] if len(pieces) > 1 else "unknown"
    return sensor, city


def _stratified_split(
    samples: list[dict],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for sample in samples:
        groups[(sample["sensor"], sample["city"])].append(sample)
    rng = random.Random(seed)
    train: list[dict] = []
    val: list[dict] = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda item: item["name"])
        rng.shuffle(group)
        count = int(round(len(group) * val_ratio))
        if len(group) > 1:
            count = min(max(count, 1), len(group) - 1)
        else:
            count = 0
        val.extend(group[:count])
        train.extend(group[count:])
    train.sort(key=lambda item: item["name"])
    val.sort(key=lambda item: item["name"])
    return train, val


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not 0.0 < float(args.val_ratio) < 1.0:
        raise ValueError("--val-ratio must be strictly between 0 and 1")
    if not (
        math.isfinite(args.sar_db_min)
        and math.isfinite(args.sar_db_max)
        and args.sar_db_min < args.sar_db_max
    ):
        raise ValueError("SAR dB bounds must be finite with min < max")
    extracted_train = next(
        (
            candidate
            for candidate in (input_root / "train", input_root)
            if all((candidate / name).is_dir() for name in ("rgb", "sar", "dsm"))
        ),
        None,
    )
    archive_path = (
        None
        if extracted_train is not None
        else _find_one(input_root, ("train.zip", "track2.zip"))
    )
    directories = {
        name: output_root / name
        for name in ("rgb", "sar", "heights")
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    prepared: list[dict] = []
    height_min = float("inf")
    height_max = float("-inf")
    archive_context = (
        zipfile.ZipFile(archive_path)
        if archive_path is not None
        else nullcontext(None)
    )
    with archive_context as archive:
        if extracted_train is not None:
            filenames = sorted(
                path.name
                for path in (extracted_train / "rgb").iterdir()
                if path.is_file()
                and path.suffix.casefold() in {".tif", ".tiff"}
            )

            def read_source(kind: str, filename: str) -> np.ndarray:
                path = extracted_train / kind / filename
                if not path.is_file():
                    raise FileNotFoundError(path)
                return np.asarray(tifffile.imread(path))

        else:
            assert archive is not None
            members = set(archive.namelist())
            prefixes = {
                member.split("/", 1)[0]
                for member in members
                if "/" in member
            }
            required_prefixes = {"rgb", "sar", "dsm"}
            if not required_prefixes.issubset(prefixes):
                raise RuntimeError(
                    f"{archive_path} does not contain "
                    f"{sorted(required_prefixes)}"
                )
            filenames = sorted(
                Path(member).name
                for member in members
                if member.startswith("rgb/")
                and Path(member).suffix.casefold() in {".tif", ".tiff"}
            )

            def read_source(kind: str, filename: str) -> np.ndarray:
                return _read_tiff(archive, f"{kind}/{filename}")

        if not filenames:
            raise FileNotFoundError(
                f"No RGB TIFF images found below {input_root}"
            )
        for filename in tqdm(
            filenames,
            desc="Preparing DFC23 Track 2",
        ):
            stem = Path(filename).stem
            rgb_path = directories["rgb"] / f"{stem}.png"
            sar_path = directories["sar"] / f"{stem}.png"
            height_path = directories["heights"] / f"{stem}.npy"
            outputs = (rgb_path, sar_path, height_path)
            if args.overwrite or not all(path.is_file() for path in outputs):
                rgb = _rgb_uint8(read_source("rgb", filename))
                sar = _sar_uint8(
                    read_source("sar", filename),
                    float(args.sar_db_min),
                    float(args.sar_db_max),
                )
                height = _height_float(
                    read_source("dsm", filename)
                )
                expected = rgb.shape[:2]
                if expected != (512, 512):
                    raise RuntimeError(
                        f"Expected native 512x512 DFC23 input, got "
                        f"{expected} for {filename}"
                    )
                if rgb.shape[:2] != expected or sar.shape != expected:
                    raise RuntimeError(
                        f"DFC23 size mismatch for {filename}: "
                        f"RGB={rgb.shape}, SAR={sar.shape}, expected={expected}"
                    )
                if height.shape != expected:
                    raise RuntimeError(
                        f"DFC23 height size mismatch for {filename}: "
                        f"{height.shape} versus {expected}"
                    )
                Image.fromarray(rgb).save(rgb_path)
                Image.fromarray(sar).save(sar_path)
                np.save(height_path, height, allow_pickle=False)
            height_values = np.load(height_path, mmap_mode="r")
            height_min = min(height_min, float(np.min(height_values)))
            height_max = max(height_max, float(np.max(height_values)))
            sensor, city = _sensor_city(stem)
            prepared.append(
                {
                    "name": stem,
                    "rgb": str(rgb_path.relative_to(output_root)),
                    "sar": str(sar_path.relative_to(output_root)),
                    "height": str(height_path.relative_to(output_root)),
                    "sensor": sensor,
                    "city": city,
                }
            )

    train, val = _stratified_split(
        prepared,
        float(args.val_ratio),
        int(args.seed),
    )
    manifest = {
        "format": "dfc23_track2_rgb_sar_v1",
        "source": str(
            extracted_train if extracted_train is not None else archive_path
        ),
        "modalities": ["rgb", "sar"],
        "sar_normalization": {
            "transform": "10*log10(max(value,1e-6))",
            "db_min": float(args.sar_db_min),
            "db_max": float(args.sar_db_max),
        },
        "height_range_meters": [height_min, height_max],
        "split_seed": int(args.seed),
        "val_ratio": float(args.val_ratio),
        "splits": {
            "train": train,
            "val": val,
            # Public DFC23 validation/test images have no local GT. Keep the
            # held-out labeled subset as the reproducible local test split.
            "test": val,
        },
    }
    manifest_path = output_root / "splits.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "train": len(train),
                "val": len(val),
                "test": len(val),
                "height_min_m": height_min,
                "height_max_m": height_max,
                "sar_db_window": [args.sar_db_min, args.sar_db_max],
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
