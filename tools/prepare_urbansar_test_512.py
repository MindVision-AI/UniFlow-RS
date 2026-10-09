"""Convert the official UrbanSARFloods full-frame test set to 512 tiles.

The resulting files use the same SAR/GT manifest contract as the regular
train/validation loader.  Edge tiles overlap only when a scene dimension is
not divisible by 512.  A VALID mask assigns every source pixel to exactly one
tile, so aggregate metrics neither omit nor double-count overlap pixels.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from tqdm.auto import tqdm


TILE_SIZE = 512


@dataclass(frozen=True)
class SceneSpec:
    name: str
    directory: str
    sar_filename: str
    gt_filename: str


SCENES = (
    SceneSpec(
        "20210727_Weihui",
        "20210727_Weihui",
        "20210727_Weihui_SAR.tif",
        "20210727_Weihui_GT.tif",
    ),
    SceneSpec(
        "20230609_NovaKakhovka",
        "20230609_NovaKakhovka",
        "20230609_NovaKakhovka_SAR.tif",
        "20230609_NovaKakhovka_GT.tif",
    ),
    SceneSpec(
        "20231201_Jubba_1",
        "20231201_Jubba_1",
        "20231201_Jubba_SAR.tif",
        "20231201_Jubba_1_GT.tif",
    ),
    SceneSpec(
        "20231201_Jubba_2",
        "20231201_Jubba_2",
        "20231201_Jubba_2_SAR.tif",
        "20231201_Jubba_2_GT.tif",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare official UrbanSARFloods test scenes as 512 tiles"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/UrbanSARFloods"),
        help=(
            "UrbanSARFloods root. It may directly contain Train_dataset.txt "
            "or contain the nested urban_sar_floods directory."
        ),
    )
    parser.add_argument(
        "--test-root",
        type=Path,
        default=None,
        help=(
            "Path to testing_case_orig. Defaults to "
            "<dataset-root>/testing_case_orig."
        ),
    )
    parser.add_argument(
        "--output-name",
        default="official_test_512",
        help="Output directory below the resolved training-data root.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing prepared test directory and manifest.",
    )
    return parser.parse_args()


def _resolve_training_root(configured: Path) -> Path:
    configured = configured.expanduser().resolve()
    for candidate in (configured, configured / "urban_sar_floods"):
        if (
            (candidate / "Train_dataset.txt").is_file()
            and (candidate / "Valid_dataset.txt").is_file()
            and (candidate / "data_norm.txt").is_file()
        ):
            return candidate
    raise FileNotFoundError(
        "Could not find UrbanSARFloods Train_dataset.txt, "
        f"Valid_dataset.txt and data_norm.txt below '{configured}'."
    )


def _resolve_test_root(dataset_root: Path, configured: Path | None) -> Path:
    if configured is not None:
        root = configured.expanduser().resolve()
    else:
        root = dataset_root.expanduser().resolve() / "testing_case_orig"
    if not root.is_dir():
        raise FileNotFoundError(
            f"UrbanSARFloods testing_case_orig not found: '{root}'"
        )
    return root


def _spatial_shape(array: np.ndarray) -> tuple[int, int]:
    if array.ndim != 3:
        raise ValueError(f"Expected three-dimensional SAR data, got {array.shape}")
    if array.shape[-1] == 8:
        return int(array.shape[0]), int(array.shape[1])
    if array.shape[0] == 8:
        return int(array.shape[1]), int(array.shape[2])
    raise ValueError(
        "Expected UrbanSARFloods SAR layout HxWx8 or 8xHxW, got "
        f"{array.shape}"
    )


def _starts(length: int) -> list[int]:
    if length < TILE_SIZE:
        raise ValueError(
            f"Official test dimension {length} is smaller than {TILE_SIZE}"
        )
    starts = list(range(0, max(length - TILE_SIZE + 1, 1), TILE_SIZE))
    last = length - TILE_SIZE
    if starts[-1] != last:
        starts.append(last)
    return starts


def _read_ground_truth(path: Path) -> np.ndarray:
    try:
        target = tifffile.imread(path)
    except (KeyError, RuntimeError, ValueError):
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(path) as image:
            target = np.asarray(image).copy()
    target = np.squeeze(np.asarray(target))
    if target.ndim != 2:
        raise ValueError(f"Expected a 2D ground truth, got {target.shape}: {path}")
    if not np.isfinite(target).all():
        raise ValueError(f"Ground truth contains NaN/Inf: '{path}'")
    target = target.astype(np.uint8, copy=False)
    if target.size and (target.min() < 0 or target.max() > 2):
        raise ValueError(
            f"UrbanSARFloods GT must contain IDs 0/1/2: '{path}'"
        )
    return target


def _sar_tile(
    source: np.ndarray,
    row: int,
    column: int,
) -> np.ndarray:
    if source.shape[-1] == 8:
        tile = source[
            row : row + TILE_SIZE,
            column : column + TILE_SIZE,
            :,
        ]
    else:
        tile = np.moveaxis(
            source[
                :,
                row : row + TILE_SIZE,
                column : column + TILE_SIZE,
            ],
            0,
            -1,
        )
    tile = np.asarray(tile)
    if tile.shape != (TILE_SIZE, TILE_SIZE, 8):
        raise RuntimeError(f"Unexpected SAR tile shape: {tile.shape}")
    return tile


def _valid_tile(
    row_index: int,
    column_index: int,
    rows: list[int],
    columns: list[int],
) -> np.ndarray:
    # The official full-frame evaluator writes windows in row-major order and
    # lets later windows own overlap pixels.  This mask reproduces that policy
    # while allowing each tile to be scored independently by the normal loader.
    valid_height = (
        TILE_SIZE
        if row_index == len(rows) - 1
        else rows[row_index + 1] - rows[row_index]
    )
    valid_width = (
        TILE_SIZE
        if column_index == len(columns) - 1
        else columns[column_index + 1] - columns[column_index]
    )
    valid = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)
    valid[:valid_height, :valid_width] = 1
    return valid


def _close_memmap(array: np.ndarray) -> None:
    current: object | None = array
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        memory_map = getattr(current, "_mmap", None)
        if memory_map is not None:
            memory_map.close()
            return
        current = getattr(current, "base", None)


def main() -> None:
    args = parse_args()
    configured_dataset_root = args.dataset_root.expanduser().resolve()
    training_root = _resolve_training_root(configured_dataset_root)
    test_root = _resolve_test_root(configured_dataset_root, args.test_root)
    output_root = training_root / str(args.output_name)
    manifest_path = training_root / "Test_dataset.txt"

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Prepared test directory already exists: '{output_root}'. "
                "Pass --overwrite to replace it."
            )
        shutil.rmtree(output_root)
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Test manifest already exists: '{manifest_path}'. "
            "Pass --overwrite to replace it."
        )

    manifest_rows: list[str] = []
    summary: dict[str, object] = {
        "source": str(test_root),
        "output": str(output_root),
        "tile_size": TILE_SIZE,
        "scenes": {},
    }
    for scene in SCENES:
        scene_root = test_root / scene.directory
        sar_path = scene_root / scene.sar_filename
        gt_path = scene_root / scene.gt_filename
        if not sar_path.is_file() or not gt_path.is_file():
            raise FileNotFoundError(
                f"Missing official test pair: '{sar_path}', '{gt_path}'"
            )
        sar = tifffile.memmap(sar_path, mode="r")
        target = _read_ground_truth(gt_path)
        try:
            height, width = _spatial_shape(sar)
            if target.shape != (height, width):
                raise ValueError(
                    f"SAR/GT mismatch for {scene.name}: "
                    f"SAR={(height, width)}, GT={target.shape}"
                )
            rows = _starts(height)
            columns = _starts(width)
            sar_dir = output_root / scene.name / "SAR"
            gt_dir = output_root / scene.name / "GT"
            valid_dir = output_root / scene.name / "VALID"
            sar_dir.mkdir(parents=True, exist_ok=True)
            gt_dir.mkdir(parents=True, exist_ok=True)
            valid_dir.mkdir(parents=True, exist_ok=True)

            positions = [
                (row_index, column_index, row, column)
                for row_index, row in enumerate(rows)
                for column_index, column in enumerate(columns)
            ]
            for row_index, column_index, row, column in tqdm(
                positions,
                desc=f"Tiling {scene.name}",
            ):
                stem = f"{scene.name}_r{row:06d}_c{column:06d}"
                sar_output = sar_dir / f"{stem}_SAR.tif"
                gt_output = gt_dir / f"{stem}_GT.tif"
                valid_output = valid_dir / f"{stem}_VALID.tif"
                tifffile.imwrite(
                    sar_output,
                    _sar_tile(sar, row, column),
                    compression="deflate",
                    metadata=None,
                )
                tifffile.imwrite(
                    gt_output,
                    target[
                        row : row + TILE_SIZE,
                        column : column + TILE_SIZE,
                    ],
                    compression="deflate",
                    metadata=None,
                )
                tifffile.imwrite(
                    valid_output,
                    _valid_tile(row_index, column_index, rows, columns),
                    compression="deflate",
                    metadata=None,
                )
                manifest_rows.append(
                    gt_output.relative_to(training_root).as_posix()
                )
            summary["scenes"][scene.name] = {
                "height": height,
                "width": width,
                "tiles": len(positions),
            }
        finally:
            _close_memmap(sar)

    manifest_path.write_text(
        "\n".join(manifest_rows) + "\n",
        encoding="utf-8",
    )
    summary["tiles"] = len(manifest_rows)
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
