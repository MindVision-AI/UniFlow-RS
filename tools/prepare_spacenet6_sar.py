"""Prepare co-registered SpaceNet 6 RGB/SAR building tiles for UniFlow-RS."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

try:
    import tifffile
except ImportError as error:
    raise ImportError(
        "prepare_spacenet6_sar.py requires tifffile. Run "
        "`python -m pip install tifffile` or reinstall requirements.txt."
    ) from error


SAR_PREFIX = "SN6_Train_AOI_11_Rotterdam_SAR-Intensity_"
OPTICAL_PREFIX = "SN6_Train_AOI_11_Rotterdam_PS-RGB_"
BUILDING_CSV = "SN6_Train_AOI_11_Rotterdam_Buildings.csv"
ORIENTATION_FILE = "SAR_orientations.txt"
POLYGON_PREFIX = "POLYGON"
SOURCE_BANDS = ("HH", "HV", "VH", "VV")
PSEUDO_RGB = ("HH", "VV", "mean(HV,VH)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert co-registered SpaceNet 6 PS-RGB/four-polarization SAR "
            "GeoTIFFs and pixel-WKT labels into deterministic 80/20 "
            "train/test 512x512 PNG samples."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/SpaceNet6"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/spacenet6_sar_512"),
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--normalization-images", type=int, default=256)
    parser.add_argument("--normalization-stride", type=int, default=16)
    parser.add_argument("--lower-percentile", type=float, default=1.0)
    parser.add_argument("--upper-percentile", type=float, default=99.5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resplit-only",
        action="store_true",
        help=(
            "Only rewrite splits.json from already prepared images; do not "
            "read or convert the original GeoTIFFs."
        ),
    )
    return parser.parse_args()


def resolve_aoi_root(root: Path) -> Path:
    root = root.expanduser().resolve()
    candidates = (
        root,
        root / "train" / "AOI_11_Rotterdam",
        root
        / "SN6_buildings"
        / "train"
        / "AOI_11_Rotterdam",
    )
    for candidate in candidates:
        if (
            (candidate / "SAR-Intensity").is_dir()
            and (candidate / "PS-RGB").is_dir()
            and (candidate / "geojson_buildings").is_dir()
            and (candidate / "SummaryData" / BUILDING_CSV).is_file()
        ):
            return candidate
    attempted = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Could not find the extracted SpaceNet 6 Rotterdam training data. "
        f"Tried:\n{attempted}"
    )


def image_id_from_sar(path: Path) -> str:
    if not path.stem.startswith(SAR_PREFIX):
        raise ValueError(f"Unexpected SpaceNet 6 SAR filename: '{path.name}'")
    return path.stem[len(SAR_PREFIX) :]


def image_id_from_optical(path: Path) -> str:
    if not path.stem.startswith(OPTICAL_PREFIX):
        raise ValueError(
            f"Unexpected SpaceNet 6 optical filename: '{path.name}'"
        )
    return path.stem[len(OPTICAL_PREFIX) :]


def collection_id_from_image_id(image_id: str) -> str:
    marker = "_tile_"
    if marker not in image_id:
        raise ValueError(f"SpaceNet 6 image ID has no tile suffix: {image_id}")
    return image_id.split(marker, 1)[0]


def load_sar_orientations(path: Path) -> dict[str, int]:
    orientations: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 2 or fields[1] not in {"0", "1"}:
                raise ValueError(
                    f"Invalid SAR orientation at {path}:{line_number}: "
                    f"{line.rstrip()!r}"
                )
            orientations[fields[0]] = int(fields[1])
    if not orientations:
        raise ValueError(f"No SAR orientations found in '{path}'")
    return orientations


def read_sar(path: Path) -> np.ndarray:
    array = np.asarray(tifffile.imread(path), dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(
            f"Expected four-band SpaceNet 6 TIFF, got {array.shape}: '{path}'"
        )
    if array.shape[-1] == 4:
        pass
    elif array.shape[0] == 4:
        array = np.moveaxis(array, 0, -1)
    else:
        raise ValueError(
            f"Could not identify four SAR bands in {array.shape}: '{path}'"
        )
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    return array


def read_optical(path: Path) -> np.ndarray:
    array = np.asarray(tifffile.imread(path), dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(
            f"Expected three-band SpaceNet 6 PS-RGB TIFF, got "
            f"{array.shape}: '{path}'"
        )
    if array.shape[-1] == 3:
        pass
    elif array.shape[0] == 3:
        array = np.moveaxis(array, 0, -1)
    else:
        raise ValueError(
            f"Could not identify three optical bands in {array.shape}: "
            f"'{path}'"
        )
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def pseudo_rgb_from_sar(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if array.ndim != 3 or array.shape[-1] != 4:
        raise ValueError(f"Expected SAR [H,W,4], got {array.shape}")
    valid = np.any(array > 0.0, axis=-1)
    pseudo_rgb = np.stack(
        (
            array[..., 0],
            array[..., 3],
            0.5 * (array[..., 1] + array[..., 2]),
        ),
        axis=-1,
    )
    return pseudo_rgb, valid


def random_train_test_split(
    names: list[str],
    train_ratio: float,
    seed: int,
) -> dict[str, list[str]]:
    """Return an exact deterministic tile-level train/test partition.

    ``val`` intentionally aliases ``test`` because this project uses the
    held-out 20% for both per-epoch best-checkpoint selection and final local
    evaluation.  It is not a third independently sampled subset.
    """

    if not 0.0 < train_ratio < 1.0:
        raise ValueError("--train-ratio must be strictly between 0 and 1")
    unique_names = sorted({str(name) for name in names})
    if len(unique_names) != len(names):
        raise ValueError("SpaceNet 6 sample names must be unique")
    if len(unique_names) < 2:
        raise ValueError("At least two samples are required for an 80/20 split")
    shuffled = list(unique_names)
    random.Random(int(seed)).shuffle(shuffled)
    train_count = min(
        max(int(round(len(shuffled) * float(train_ratio))), 1),
        len(shuffled) - 1,
    )
    train_names = sorted(shuffled[:train_count])
    test_names = sorted(shuffled[train_count:])
    return {
        "train": train_names,
        "val": list(test_names),
        "test": test_names,
    }


def resplit_prepared_output(
    output_root: Path,
    train_ratio: float,
    seed: int,
) -> dict:
    """Rewrite only the prepared manifest using the 80/20 protocol."""

    output_root = output_root.expanduser().resolve()
    manifest_path = output_root / "splits.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing existing SpaceNet 6 manifest: '{manifest_path}'"
        )
    names = sorted(path.stem for path in (output_root / "images").glob("*.png"))
    if len(names) != 3401:
        raise RuntimeError(
            "Expected 3401 prepared SpaceNet 6 images before resplitting, "
            f"got {len(names)}"
        )
    for directory_name in (
        "optical_images",
        "masks",
        "instances",
        "valid_masks",
    ):
        missing = [
            name
            for name in names
            if not (output_root / directory_name / f"{name}.png").is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Prepared SpaceNet 6 {directory_name} is incomplete; "
                f"first missing sample: {missing[0]}"
            )
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = random_train_test_split(names, train_ratio, seed)
    metadata["format"] = "uniflow_rs_spacenet6_rgb_sar_v3"
    metadata.pop("spatial_split", None)
    metadata.pop("centers", None)
    metadata["split_protocol"] = {
        "method": "deterministic_random_tile_80_20",
        "train_ratio": float(train_ratio),
        "test_ratio": float(1.0 - train_ratio),
        "validation_alias": "test",
        "seed": int(seed),
    }
    metadata["counts"] = {
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "test": len(splits["test"]),
    }
    metadata["splits"] = splits
    manifest_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return metadata


def estimate_normalization(
    paths: list[Path],
    number_of_images: int,
    stride: int,
    lower_percentile: float,
    upper_percentile: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 <= lower_percentile < upper_percentile <= 100.0:
        raise ValueError("Normalization percentiles are invalid")
    count = min(max(int(number_of_images), 1), len(paths))
    indices = np.linspace(0, len(paths) - 1, count, dtype=np.int64)
    sampled: list[np.ndarray] = []
    for index in tqdm(indices, desc="Estimating SAR normalization"):
        pseudo_rgb, valid = pseudo_rgb_from_sar(read_sar(paths[int(index)]))
        values = pseudo_rgb[::stride, ::stride]
        sampled_valid = valid[::stride, ::stride]
        values = values[sampled_valid]
        if values.size:
            sampled.append(values)
    if not sampled:
        raise RuntimeError("No valid SAR pixels were available for normalization")
    values = np.concatenate(sampled, axis=0)
    lower = np.percentile(values, lower_percentile, axis=0).astype(np.float32)
    upper = np.percentile(values, upper_percentile, axis=0).astype(np.float32)
    if bool(np.any(upper <= lower)):
        raise RuntimeError(
            f"Degenerate SAR normalization bounds: {lower}, {upper}"
        )
    return lower, upper


def estimate_optical_normalization(
    paths: list[Path],
    number_of_images: int,
    stride: int,
    lower_percentile: float,
    upper_percentile: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 <= lower_percentile < upper_percentile <= 100.0:
        raise ValueError("Normalization percentiles are invalid")
    count = min(max(int(number_of_images), 1), len(paths))
    indices = np.linspace(0, len(paths) - 1, count, dtype=np.int64)
    sampled: list[np.ndarray] = []
    for index in tqdm(indices, desc="Estimating optical normalization"):
        optical = read_optical(paths[int(index)])
        values = optical[::stride, ::stride]
        valid = np.any(values > 0.0, axis=-1)
        values = values[valid]
        if values.size:
            sampled.append(values)
    if not sampled:
        raise RuntimeError(
            "No valid optical pixels were available for normalization"
        )
    values = np.concatenate(sampled, axis=0)
    lower = np.percentile(values, lower_percentile, axis=0).astype(np.float32)
    upper = np.percentile(values, upper_percentile, axis=0).astype(np.float32)
    if bool(np.any(upper <= lower)):
        raise RuntimeError(
            f"Degenerate optical normalization bounds: {lower}, {upper}"
        )
    return lower, upper


def load_pixel_wkts(csv_path: Path) -> dict[str, list[str]]:
    polygons: dict[str, list[str]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        for row in tqdm(csv.DictReader(file), desc="Reading building WKT"):
            polygons.setdefault(row["ImageId"], []).append(
                row["PolygonWKT_Pix"]
            )
    return polygons


def polygon_rings(wkt: str) -> list[np.ndarray]:
    text = str(wkt).strip()
    if text == "POLYGON EMPTY":
        return []
    if not text.startswith("POLYGON ((") or not text.endswith("))"):
        raise ValueError(f"Unsupported SpaceNet 6 WKT: {text[:80]!r}")
    inner = text[len("POLYGON ((") : -2]
    rings: list[np.ndarray] = []
    for ring_text in re.split(r"\)\s*,\s*\(", inner):
        coordinates = np.fromstring(
            ring_text.replace(",", " "),
            sep=" ",
            dtype=np.float64,
        )
        if coordinates.size % 2 != 0 or coordinates.size < 6:
            raise ValueError(f"Malformed SpaceNet 6 polygon ring: {wkt[:80]!r}")
        rings.append(coordinates.reshape(-1, 2))
    return rings


def rasterize_instances(
    wkts: list[str],
    height: int,
    width: int,
) -> np.ndarray:
    instances = np.zeros((height, width), dtype=np.int32)
    instance_id = 0
    for wkt in wkts:
        rings = polygon_rings(wkt)
        if not rings:
            continue
        instance_id += 1
        pixel_rings = [
            np.rint(ring)
            .clip((0, 0), (width - 1, height - 1))
            .astype(np.int32)
            .reshape(-1, 1, 2)
            for ring in rings
        ]
        cv2.fillPoly(instances, [pixel_rings[0]], instance_id)
        if len(pixel_rings) > 1:
            cv2.fillPoly(instances, pixel_rings[1:], 0)
    if instance_id > np.iinfo(np.uint16).max:
        raise RuntimeError(
            f"Tile contains too many building instances: {instance_id}"
        )
    return instances


def main() -> None:
    args = parse_args()
    if args.resolution < 1:
        raise ValueError("--resolution must be positive")
    if args.normalization_stride < 1:
        raise ValueError("--normalization-stride must be positive")
    if args.resplit_only:
        manifest = resplit_prepared_output(
            args.output_root,
            train_ratio=float(args.train_ratio),
            seed=int(args.seed),
        )
        print(json.dumps(
            {
                "output": str(args.output_root.expanduser().resolve()),
                "counts": manifest["counts"],
                "split_protocol": manifest["split_protocol"],
            },
            indent=2,
        ))
        return
    aoi_root = resolve_aoi_root(args.input_root)
    sar_paths = sorted((aoi_root / "SAR-Intensity").glob("*.tif"))
    optical_paths = sorted((aoi_root / "PS-RGB").glob("*.tif"))
    if len(sar_paths) != 3401:
        raise RuntimeError(
            f"Expected 3401 SpaceNet 6 training SAR tiles, got {len(sar_paths)}"
        )
    if len(optical_paths) != 3401:
        raise RuntimeError(
            "Expected 3401 SpaceNet 6 training PS-RGB tiles, got "
            f"{len(optical_paths)}"
        )
    optical_by_id = {
        image_id_from_optical(path): path for path in optical_paths
    }
    sar_image_ids = {image_id_from_sar(path) for path in sar_paths}
    if set(optical_by_id) != sar_image_ids:
        raise RuntimeError(
            "SpaceNet 6 PS-RGB and SAR image IDs do not match: "
            f"missing={len(sar_image_ids - set(optical_by_id))}, "
            f"extra={len(set(optical_by_id) - sar_image_ids)}"
        )
    splits = random_train_test_split(
        [image_id_from_sar(path) for path in sar_paths],
        train_ratio=float(args.train_ratio),
        seed=int(args.seed),
    )
    lower, upper = estimate_normalization(
        sar_paths,
        number_of_images=int(args.normalization_images),
        stride=int(args.normalization_stride),
        lower_percentile=float(args.lower_percentile),
        upper_percentile=float(args.upper_percentile),
    )
    optical_lower, optical_upper = estimate_optical_normalization(
        optical_paths,
        number_of_images=int(args.normalization_images),
        stride=int(args.normalization_stride),
        lower_percentile=float(args.lower_percentile),
        upper_percentile=float(args.upper_percentile),
    )
    wkts = load_pixel_wkts(
        aoi_root / "SummaryData" / BUILDING_CSV
    )
    orientations = load_sar_orientations(
        aoi_root / "SummaryData" / ORIENTATION_FILE
    )
    image_ids = {image_id_from_sar(path) for path in sar_paths}
    if set(wkts) != image_ids:
        raise RuntimeError(
            "SpaceNet 6 SAR and building CSV IDs do not match: "
            f"missing={len(image_ids - set(wkts))}, "
            f"extra={len(set(wkts) - image_ids)}"
        )
    collection_ids = {
        collection_id_from_image_id(image_id) for image_id in image_ids
    }
    missing_orientations = sorted(collection_ids - set(orientations))
    if missing_orientations:
        raise RuntimeError(
            "SpaceNet 6 orientation metadata is incomplete: "
            f"{missing_orientations[:5]}"
        )

    output_root = args.output_root.expanduser().resolve()
    directories = {
        name: output_root / name
        for name in (
            "images",
            "optical_images",
            "masks",
            "instances",
            "valid_masks",
        )
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    def prepare_one(path: Path) -> tuple[int, int]:
        name = image_id_from_sar(path)
        outputs = {
            key: directory / f"{name}.png"
            for key, directory in directories.items()
        }
        if (
            not args.overwrite
            and all(output.is_file() for output in outputs.values())
        ):
            with Image.open(outputs["mask"]) as source:
                mask = np.asarray(source.convert("L"), dtype=np.uint8) > 127
            with Image.open(outputs["valid_masks"]) as source:
                valid = np.asarray(source.convert("L"), dtype=np.uint8) > 127
            return int(mask.sum()), int(valid.sum())

        sar = read_sar(path)
        optical = read_optical(optical_by_id[name])
        if optical.shape[:2] != sar.shape[:2]:
            raise RuntimeError(
                f"SpaceNet 6 RGB/SAR size mismatch for {name}: "
                f"{optical.shape[:2]} vs {sar.shape[:2]}"
            )
        pseudo_rgb, valid = pseudo_rgb_from_sar(sar)
        normalized = np.clip(
            (pseudo_rgb - lower.reshape(1, 1, 3))
            / (upper - lower).reshape(1, 1, 3),
            0.0,
            1.0,
        )
        normalized[~valid] = 0.0
        optical_valid = np.any(optical > 0.0, axis=-1)
        optical_normalized = np.clip(
            (optical - optical_lower.reshape(1, 1, 3))
            / (optical_upper - optical_lower).reshape(1, 1, 3),
            0.0,
            1.0,
        )
        optical_normalized[~optical_valid] = 0.0
        instances = rasterize_instances(
            wkts[name],
            height=sar.shape[0],
            width=sar.shape[1],
        )
        # Canonicalize the side-looking direction. SAR layover and shadow
        # reverse between the official north/south collection orientations.
        orientation = orientations[collection_id_from_image_id(name)]
        if orientation == 1:
            normalized = np.ascontiguousarray(
                np.rot90(normalized, 2, axes=(0, 1))
            )
            valid = np.ascontiguousarray(
                np.rot90(valid, 2, axes=(0, 1))
            )
            instances = np.ascontiguousarray(
                np.rot90(instances, 2, axes=(0, 1))
            )
            optical_normalized = np.ascontiguousarray(
                np.rot90(optical_normalized, 2, axes=(0, 1))
            )
        image = cv2.resize(
            normalized,
            (args.resolution, args.resolution),
            interpolation=cv2.INTER_AREA,
        )
        optical_image = cv2.resize(
            optical_normalized,
            (args.resolution, args.resolution),
            interpolation=cv2.INTER_AREA,
        )
        valid_resized = cv2.resize(
            valid.astype(np.uint8),
            (args.resolution, args.resolution),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        instances = cv2.resize(
            instances,
            (args.resolution, args.resolution),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.uint16)
        instances[~valid_resized] = 0
        mask = instances > 0

        Image.fromarray(
            np.rint(image * 255.0).astype(np.uint8),
            mode="RGB",
        ).save(outputs["images"])
        Image.fromarray(
            np.rint(optical_image * 255.0).astype(np.uint8),
            mode="RGB",
        ).save(outputs["optical_images"])
        Image.fromarray((mask * 255).astype(np.uint8)).save(outputs["masks"])
        Image.fromarray(instances).save(outputs["instances"])
        Image.fromarray(
            (valid_resized * 255).astype(np.uint8)
        ).save(outputs["valid_masks"])
        return int(mask.sum()), int(valid_resized.sum())

    positive_pixels = 0
    valid_pixels = 0
    workers = max(int(args.workers), 1)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        iterator = executor.map(prepare_one, sar_paths)
        for positive, valid in tqdm(
            iterator,
            total=len(sar_paths),
            desc="Preparing SpaceNet 6 SAR",
        ):
            positive_pixels += positive
            valid_pixels += valid

    manifest = {
        "format": "uniflow_rs_spacenet6_rgb_sar_v3",
        "source": str(aoi_root),
        "resolution": int(args.resolution),
        "source_size": 900,
        "source_bands": list(SOURCE_BANDS),
        "pseudo_rgb": list(PSEUDO_RGB),
        "orientation_alignment": {
            "enabled": True,
            "canonical_orientation": 0,
            "rotated_orientation": 1,
            "operation": "rotate_180_degrees",
        },
        "normalization": {
            "lower_percentile": float(args.lower_percentile),
            "upper_percentile": float(args.upper_percentile),
            "lower": [float(value) for value in lower],
            "upper": [float(value) for value in upper],
        },
        "optical_normalization": {
            "lower_percentile": float(args.lower_percentile),
            "upper_percentile": float(args.upper_percentile),
            "lower": [float(value) for value in optical_lower],
            "upper": [float(value) for value in optical_upper],
        },
        "split_protocol": {
            "method": "deterministic_random_tile_80_20",
            "train_ratio": float(args.train_ratio),
            "test_ratio": float(
                1.0 - args.train_ratio
            ),
            "validation_alias": "test",
            "seed": int(args.seed),
        },
        "counts": {
            split: len(names) for split, names in splits.items()
        },
        "positive_ratio_valid_pixels": (
            positive_pixels / max(valid_pixels, 1)
        ),
        "splits": splits,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "splits.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    print(json.dumps(
        {
            "output": str(output_root),
            "counts": manifest["counts"],
            "normalization": manifest["normalization"],
            "positive_ratio_valid_pixels": (
                manifest["positive_ratio_valid_pixels"]
            ),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
