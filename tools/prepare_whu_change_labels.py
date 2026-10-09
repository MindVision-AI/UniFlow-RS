"""Split WHU's official whole-image change labels into aligned 512 tiles."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - progress bars are optional
    def tqdm(iterable, **_kwargs):
        return iterable


Image.MAX_IMAGE_PIXELS = None
_GRID_NAME = re.compile(r"^(?P<first>\d+)_(?P<second>\d+)$")
_RASTER_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


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


def _paths_by_stem(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    paths = {
        path.stem.casefold(): path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.casefold() in _RASTER_EXTENSIONS
    }
    if not paths:
        raise FileNotFoundError(f"No raster images found in '{directory}'")
    return paths


def _find_tif(directory: Path, preferred_name: str) -> Path:
    preferred = directory / preferred_name
    if preferred.is_file():
        return preferred
    candidates = sorted(
        path
        for path in directory.glob("*.tif")
        if path.is_file()
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one TIFF below '{directory}', found {candidates}"
        )
    return candidates[0]


def _grid_indices(stem: str) -> tuple[int, int]:
    match = _GRID_NAME.fullmatch(stem)
    if match is None:
        raise ValueError(
            f"WHU tile '{stem}' is not named with two numeric fields"
        )
    return int(match.group("first")), int(match.group("second"))


def _natural_stem_key(stem: str) -> tuple[int, int, str]:
    first, second = _grid_indices(stem)
    return first, second, stem


def _candidate_grid_shapes(
    tile_count: int,
    width: int,
    height: int,
    tile_size: int,
    maximum: int = 4,
) -> list[tuple[int, int]]:
    """Return plausible (columns, rows) factors nearest the raster shape."""

    expected_columns = max(int(np.ceil(width / tile_size)), 1)
    expected_rows = max(int(np.ceil(height / tile_size)), 1)
    candidates: list[tuple[float, int, int]] = []
    for rows in range(1, tile_count + 1):
        if tile_count % rows:
            continue
        columns = tile_count // rows
        count_error = (
            abs(columns - expected_columns)
            + abs(rows - expected_rows)
        )
        aspect_error = abs(
            np.log(
                max(columns / rows, 1e-8)
                / max(width / height, 1e-8)
            )
        )
        candidates.append(
            (float(count_error + aspect_error), columns, rows)
        )
    candidates.sort()
    return [
        (columns, rows)
        for _, columns, rows in candidates[:maximum]
    ]


def _candidate_index_maps(
    stems: list[str],
) -> list[tuple[str, dict[str, int]]]:
    """Interpret WHU names as sequential IDs without assuming coordinates."""

    candidates: list[tuple[str, dict[str, int]]] = []
    natural = sorted(stems, key=_natural_stem_key)
    candidates.append(
        ("natural_rank", {stem: index for index, stem in enumerate(natural)})
    )
    numeric = {stem: _grid_indices(stem) for stem in stems}
    for field, name in ((0, "first_field"), (1, "second_field")):
        values = {stem: parts[field] for stem, parts in numeric.items()}
        unique = sorted(set(values.values()))
        if (
            len(unique) == len(stems)
            and unique[-1] - unique[0] + 1 == len(unique)
        ):
            minimum = unique[0]
            candidates.append(
                (
                    f"{name}_value",
                    {stem: value - minimum for stem, value in values.items()},
                )
            )

    deduplicated: list[tuple[str, dict[str, int]]] = []
    signatures: set[tuple[int, ...]] = set()
    ordered_stems = sorted(stems)
    for name, mapping in candidates:
        signature = tuple(mapping[stem] for stem in ordered_stems)
        if signature not in signatures:
            signatures.add(signature)
            deduplicated.append((name, mapping))
    return deduplicated


def _axis_offsets(
    length: int,
    tile_size: int,
    cells: int,
    mode: str,
) -> list[int]:
    maximum = max(length - tile_size, 0)
    if cells <= 1:
        return [0]
    if mode == "edge_clamped":
        return [
            min(index * tile_size, maximum)
            for index in range(cells)
        ]
    if mode == "evenly_spaced":
        return [
            int(round(value))
            for value in np.linspace(0, maximum, cells)
        ]
    raise ValueError(mode)


def _layout_coordinates(
    index_map: dict[str, int],
    columns: int,
    rows: int,
    traversal: str,
    position_mode: str,
    width: int,
    height: int,
    tile_size: int,
) -> dict[str, tuple[int, int]]:
    x_offsets = _axis_offsets(width, tile_size, columns, position_mode)
    y_offsets = _axis_offsets(height, tile_size, rows, position_mode)
    coordinates: dict[str, tuple[int, int]] = {}
    for stem, index in index_map.items():
        if traversal == "row_major":
            row, column = divmod(index, columns)
        elif traversal == "column_major":
            column, row = divmod(index, rows)
        else:
            raise ValueError(traversal)
        if row >= rows or column >= columns:
            raise ValueError(
                f"Index {index} exceeds {columns}x{rows} tile grid"
            )
        coordinates[stem] = x_offsets[column], y_offsets[row]
    return coordinates


def _binary_image(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        return _binary_array(
            np.asarray(source.convert("L"), dtype=np.uint8)
        )


def _binary_array(array: np.ndarray) -> np.ndarray:
    """Support both 0/1 and 0/255 binary-label encodings."""

    threshold = 0 if int(array.max()) <= 1 else 127
    return array > threshold


def _candidate_stems(stems: list[str], maximum: int = 32) -> list[str]:
    if len(stems) <= maximum:
        return stems
    indices = np.linspace(0, len(stems) - 1, maximum, dtype=np.int64)
    return [stems[int(index)] for index in indices]


def _alignment_score(
    whole_label: Image.Image,
    tile_arrays: dict[str, np.ndarray],
    stems: list[str],
    coordinates: dict[str, tuple[int, int]],
) -> dict[str, float]:
    intersection = 0
    union = 0
    equal = 0
    pixels = 0
    valid = 0
    width, height = whole_label.size
    for stem in stems:
        x, y = coordinates[stem]
        tile = tile_arrays[stem]
        tile_height, tile_width = tile.shape
        if x + tile_width > width or y + tile_height > height:
            continue
        crop = _binary_array(
            np.asarray(
                whole_label.crop(
                    (x, y, x + tile_width, y + tile_height)
                ),
                dtype=np.uint8,
            )
        )
        if tile.shape != crop.shape:
            continue
        intersection += int(np.logical_and(crop, tile).sum())
        union += int(np.logical_or(crop, tile).sum())
        equal += int((crop == tile).sum())
        pixels += int(crop.size)
        valid += 1
    return {
        "iou": intersection / max(union, 1),
        "pixel_agreement": equal / max(pixels, 1),
        "samples": float(valid),
    }


def _detect_tile_layout(
    root: Path,
    split: str,
    year: str,
    stems: list[str],
    tile_labels: dict[str, Path],
    tile_size: int,
) -> tuple[dict[str, tuple[int, int]], dict, tuple[int, int]]:
    whole_path = _find_tif(
        root / year / "whole_image" / split / "label",
        f"{year}_{split}.tif",
    )
    with Image.open(whole_path) as source:
        whole_label = source.convert("L")
    width, height = whole_label.size
    sampled_stems = _candidate_stems(
        sorted(stems, key=_natural_stem_key)
    )
    tile_arrays = {
        stem: _binary_image(tile_labels[stem])
        for stem in sampled_stems
    }
    candidates: list[dict] = []
    for columns, rows in _candidate_grid_shapes(
        len(stems),
        width,
        height,
        tile_size,
    ):
        for index_name, index_map in _candidate_index_maps(stems):
            for traversal in ("row_major", "column_major"):
                for position_mode in ("edge_clamped", "evenly_spaced"):
                    coordinates = _layout_coordinates(
                        index_map,
                        columns,
                        rows,
                        traversal,
                        position_mode,
                        width,
                        height,
                        tile_size,
                    )
                    score = _alignment_score(
                        whole_label,
                        tile_arrays,
                        sampled_stems,
                        coordinates,
                    )
                    candidates.append(
                        {
                            "name": (
                                f"{index_name}/{traversal}/"
                                f"{position_mode}/{columns}x{rows}"
                            ),
                            "index_source": index_name,
                            "traversal": traversal,
                            "position_mode": position_mode,
                            "columns": columns,
                            "rows": rows,
                            "score": score,
                            "coordinates": coordinates,
                        }
                    )
    whole_size = whole_label.size
    whole_label.close()
    candidates.sort(
        key=lambda candidate: (
            candidate["score"]["iou"],
            candidate["score"]["pixel_agreement"],
            candidate["score"]["samples"],
        ),
        reverse=True,
    )
    best = candidates[0]
    best_score = best["score"]
    if best_score["samples"] < 1 or (
        best_score["iou"] < 0.98
        and best_score["pixel_agreement"] < 0.995
    ):
        printable = [
            {
                key: value
                for key, value in candidate.items()
                if key != "coordinates"
            }
            for candidate in candidates[:8]
        ]
        raise RuntimeError(
            "Could not safely align WHU whole-image labels with split tiles. "
            f"Best candidates: {printable}"
        )
    metadata = {
        key: value
        for key, value in best.items()
        if key != "coordinates"
    }
    metadata["alternatives"] = [
        {
            key: value
            for key, value in candidate.items()
            if key != "coordinates"
        }
        for candidate in candidates[1:4]
    ]
    return best["coordinates"], metadata, whole_size


def prepare_split(
    root: Path,
    split: str,
    before_year: str,
    after_year: str,
    tile_size: int,
    overwrite: bool,
) -> dict:
    def split_directory(year: str, kind: str) -> Path:
        return root / year / "splited_images" / split / kind

    before_images = _paths_by_stem(
        split_directory(before_year, "image")
    )
    before_labels = _paths_by_stem(
        split_directory(before_year, "label")
    )
    after_images = _paths_by_stem(
        split_directory(after_year, "image")
    )
    after_labels = _paths_by_stem(
        split_directory(after_year, "label")
    )
    stems = sorted(before_images)
    reference = set(stems)
    for name, paths in (
        ("before labels", before_labels),
        ("after images", after_images),
        ("after labels", after_labels),
    ):
        if set(paths) != reference:
            raise RuntimeError(
                f"{split} {name} do not match before-image filenames"
            )

    before_coordinates, before_layout, before_whole_size = (
        _detect_tile_layout(
            root,
            split,
            before_year,
            stems,
            before_labels,
            tile_size,
        )
    )
    after_coordinates, after_layout, after_whole_size = (
        _detect_tile_layout(
            root,
            split,
            after_year,
            stems,
            after_labels,
            tile_size,
        )
    )
    if before_coordinates != after_coordinates:
        raise RuntimeError(
            "The before/after WHU split labels imply different tile "
            f"layouts: {before_year}={before_layout['name']}, "
            f"{after_year}={after_layout['name']}"
        )
    if before_whole_size != after_whole_size:
        raise RuntimeError(
            "The before/after WHU whole labels have different sizes: "
            f"{before_year}={before_whole_size}, "
            f"{after_year}={after_whole_size}"
        )
    tile_coordinates = before_coordinates
    detected_layouts = {
        before_year: before_layout,
        after_year: after_layout,
    }
    whole_change_path = _find_tif(
        root / "change_label" / split,
        "change_label.tif",
    )
    with Image.open(whole_change_path) as source:
        whole_change = source.convert("L")
    change_whole_size = whole_change.size
    if change_whole_size != before_whole_size:
        whole_change.close()
        raise RuntimeError(
            "The official change label and yearly labels have different "
            f"sizes: change={change_whole_size}, years={before_whole_size}"
        )
    output_dir = (
        root / "change_label" / "splited_images" / split / "label"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    official_positive = 0
    xor_positive = 0
    official_xor_intersection = 0
    official_xor_union = 0
    official_xor_disagreement = 0
    pixels = 0
    written = 0
    width, height = whole_change.size
    for stem in tqdm(stems, desc=f"Official change labels: {split}"):
        x, y = tile_coordinates[stem]
        if x + tile_size > width or y + tile_size > height:
            raise RuntimeError(
                f"Tile {stem} at ({x}, {y}) exceeds official change label "
                f"size {width}x{height}"
            )
        official = _binary_array(
            np.asarray(
                whole_change.crop((x, y, x + tile_size, y + tile_size)),
                dtype=np.uint8,
            )
        )
        before = _binary_image(before_labels[stem])
        after = _binary_image(after_labels[stem])
        xor = np.logical_xor(before, after)
        if official.shape != xor.shape:
            raise RuntimeError(
                f"Shape mismatch for {stem}: official={official.shape}, "
                f"xor={xor.shape}"
            )

        output_path = output_dir / f"{stem}.png"
        if overwrite or not output_path.is_file():
            Image.fromarray(official.astype(np.uint8) * 255).save(output_path)
            written += 1
        official_positive += int(official.sum())
        xor_positive += int(xor.sum())
        official_xor_intersection += int(
            np.logical_and(official, xor).sum()
        )
        official_xor_union += int(np.logical_or(official, xor).sum())
        official_xor_disagreement += int((official != xor).sum())
        pixels += int(official.size)
    whole_change.close()

    metrics = {
        "split": split,
        "tiles": len(stems),
        "written": written,
        "tile_size": tile_size,
        "tile_layout": before_layout["name"],
        "detected_layouts": detected_layouts,
        "whole_change_label": str(whole_change_path),
        "output_dir": str(output_dir),
        "official_positive_ratio": official_positive / max(pixels, 1),
        "xor_positive_ratio": xor_positive / max(pixels, 1),
        "official_xor_iou": (
            official_xor_intersection / max(official_xor_union, 1)
        ),
        "official_xor_disagreement_ratio": (
            official_xor_disagreement / max(pixels, 1)
        ),
    }
    with (output_dir.parent / "manifest.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=2)
    return metrics


def main() -> None:
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/uniflow_rs_512_2step_eval.yaml"),
    )
    parser.add_argument(
        "--split",
        choices=("train", "test", "all"),
        default="all",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    data_cfg = cfg["data"]
    change_cfg = data_cfg["change"]
    root = _resolve_whu_change_root(change_cfg["root"])
    splits = (
        ("train", "test") if args.split == "all" else (args.split,)
    )
    summaries = {}
    for split in splits:
        summaries[split] = prepare_split(
            root=root,
            split=split,
            before_year=str(change_cfg.get("before_year", "2012")),
            after_year=str(change_cfg.get("after_year", "2016")),
            tile_size=int(
                change_cfg.get(
                    "resolution",
                    data_cfg.get("resolution", 512),
                )
            ),
            overwrite=bool(args.overwrite),
        )
        print(json.dumps(summaries[split], indent=2))


if __name__ == "__main__":
    main()
