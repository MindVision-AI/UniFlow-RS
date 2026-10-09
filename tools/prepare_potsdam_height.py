"""Split aligned Potsdam RGB/nDSM images into non-overlapping 512 tiles."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from PIL import Image
from tqdm.auto import tqdm


RGB_PATTERN = re.compile(r"top_potsdam_(\d+)_(\d+)_RGB$", re.IGNORECASE)
OWN_HEIGHT_PATTERN = re.compile(
    r"dsm_potsdam_(\d+)_(\d+)_normalized_ownapproach$",
    re.IGNORECASE,
)
LASTOOLS_HEIGHT_PATTERN = re.compile(
    r"dsm_potsdam_(\d+)_(\d+)_normalized_lastools$",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pad each complete Potsdam RGB/nDSM pair to a multiple of the "
            "tile size, then save aligned non-overlapping lossless tiles."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/Potsdam"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/potsdam_height_512_tiles"),
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument(
        "--height-source",
        choices=("lastools", "ownapproach", "auto"),
        default="lastools",
        help=(
            "Use one consistent complete nDSM source by default. 'auto' "
            "prefers a non-empty ownapproach file and falls back to lastools."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _tile_id(path: Path, pattern: re.Pattern[str]) -> str:
    match = pattern.fullmatch(path.stem)
    if match is None:
        raise ValueError(f"Unexpected Potsdam filename: '{path.name}'")
    return f"{int(match.group(1))}_{int(match.group(2))}"


def _discover(
    directory: Path,
    glob_pattern: str,
    filename_pattern: re.Pattern[str],
) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    paths = {
        _tile_id(path, filename_pattern): path
        for path in directory.glob(glob_pattern)
        if path.is_file()
    }
    if not paths:
        raise FileNotFoundError(
            f"No Potsdam files matching '{glob_pattern}' in '{directory}'"
        )
    return paths


def _save_tiles(
    rgb_path: Path,
    height_path: Path,
    output_root: Path,
    tile_id: str,
    tile_size: int,
    overwrite: bool,
) -> tuple[int, int, int, str | None]:
    image_dir = output_root / "images"
    height_dir = output_root / "heights"
    image_dir.mkdir(parents=True, exist_ok=True)
    height_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(rgb_path) as rgb_source, Image.open(
        height_path
    ) as height_source:
        rgb = rgb_source.convert("RGB")
        height = height_source.convert("L")
        alignment_adjustment = None
        if rgb.size != height.size:
            original_height_size = height.size
            width_difference = rgb.width - height.width
            height_difference = rgb.height - height.height
            if abs(width_difference) > 1 or abs(height_difference) > 1:
                raise RuntimeError(
                    "Potsdam RGB/nDSM size mismatch exceeds the supported "
                    f"one-pixel official-data correction: {rgb_path} "
                    f"{rgb.size}, {height_path} {height.size}"
                )
            copy_width = min(rgb.width, height.width)
            copy_height = min(rgb.height, height.height)
            height = height.crop((0, 0, copy_width, copy_height))
            aligned = Image.new("L", rgb.size)
            aligned.paste(height, (0, 0))
            if copy_width < rgb.width:
                right_edge = height.crop(
                    (copy_width - 1, 0, copy_width, copy_height)
                ).resize(
                    (rgb.width - copy_width, copy_height),
                    Image.Resampling.NEAREST,
                )
                aligned.paste(right_edge, (copy_width, 0))
            if copy_height < rgb.height:
                bottom_edge = aligned.crop(
                    (0, copy_height - 1, rgb.width, copy_height)
                ).resize(
                    (rgb.width, rgb.height - copy_height),
                    Image.Resampling.NEAREST,
                )
                aligned.paste(bottom_edge, (0, copy_height))
            height = aligned
            alignment_adjustment = (
                f"edge-aligned {original_height_size[0]}x"
                f"{original_height_size[1]} to "
                f"{rgb.width}x{rgb.height}"
            )
        source_width, source_height = rgb.size
        columns = math.ceil(source_width / tile_size)
        rows = math.ceil(source_height / tile_size)
        for row in range(rows):
            top = row * tile_size
            for column in range(columns):
                left = column * tile_size
                stem = f"{tile_id}__r{row:02d}_c{column:02d}"
                rgb_output = image_dir / f"{stem}.png"
                height_output = height_dir / f"{stem}.png"
                if (
                    not overwrite
                    and rgb_output.is_file()
                    and height_output.is_file()
                ):
                    continue
                box = (
                    left,
                    top,
                    left + tile_size,
                    top + tile_size,
                )
                # PIL pads crop regions outside the 6000x6000 source with
                # exact zeros. The dataset masks this padding from loss and
                # metrics, then crops it away when stitching predictions.
                rgb.crop(box).save(
                    rgb_output,
                    format="PNG",
                    optimize=False,
                )
                height.crop(box).save(
                    height_output,
                    format="PNG",
                    optimize=False,
                )
    return (
        source_width,
        source_height,
        rows * columns,
        alignment_adjustment,
    )


def main() -> None:
    args = parse_args()
    if args.resolution < 1:
        raise ValueError("--resolution must be positive")
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    rgb = _discover(
        input_root / "2_Ortho_RGB",
        "*_RGB.tif",
        RGB_PATTERN,
    )
    own_height = _discover(
        input_root / "1_DSM_normalisation",
        "*_normalized_ownapproach.jpg",
        OWN_HEIGHT_PATTERN,
    )
    lastools_height = _discover(
        input_root / "1_DSM_normalisation",
        "*_normalized_lastools.jpg",
        LASTOOLS_HEIGHT_PATTERN,
    )
    if set(rgb) != set(lastools_height):
        raise RuntimeError(
            "Potsdam RGB/lastools nDSM tiles are not paired: "
            f"missing heights={sorted(set(rgb) - set(lastools_height))}, "
            f"extra heights={sorted(set(lastools_height) - set(rgb))}"
        )
    selected_height: dict[str, Path] = {}
    height_source_by_tile: dict[str, str] = {}
    for tile_id in sorted(rgb):
        own_path = own_height.get(tile_id)
        if args.height_source == "lastools":
            selected_height[tile_id] = lastools_height[tile_id]
            height_source_by_tile[tile_id] = "lastools"
        elif args.height_source == "ownapproach":
            if own_path is None or own_path.stat().st_size == 0:
                raise RuntimeError(
                    f"Potsdam ownapproach nDSM is unavailable/empty for "
                    f"tile '{tile_id}'. Use --height-source lastools or auto."
                )
            selected_height[tile_id] = own_path
            height_source_by_tile[tile_id] = "ownapproach"
        elif own_path is not None and own_path.stat().st_size > 0:
            selected_height[tile_id] = own_path
            height_source_by_tile[tile_id] = "ownapproach"
        else:
            selected_height[tile_id] = lastools_height[tile_id]
            height_source_by_tile[tile_id] = "lastools_fallback"

    source_sizes: set[tuple[int, int]] = set()
    patches_per_image: set[int] = set()
    alignment_adjustments: dict[str, str] = {}
    for tile_id in tqdm(sorted(rgb), desc="Tiling Potsdam height"):
        (
            source_width,
            source_height,
            patch_count,
            alignment_adjustment,
        ) = _save_tiles(
            rgb[tile_id],
            selected_height[tile_id],
            output_root,
            tile_id,
            tile_size=args.resolution,
            overwrite=bool(args.overwrite),
        )
        source_sizes.add((source_width, source_height))
        patches_per_image.add(patch_count)
        if alignment_adjustment is not None:
            alignment_adjustments[tile_id] = alignment_adjustment
    if len(source_sizes) != 1 or len(patches_per_image) != 1:
        raise RuntimeError(
            "Every Potsdam source is expected to have the same dimensions, "
            f"got sizes={sorted(source_sizes)}, patches={patches_per_image}"
        )
    source_width, source_height = next(iter(source_sizes))
    patch_count = next(iter(patches_per_image))

    manifest = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "mode": "non_overlapping_tiles",
        "tile_size": int(args.resolution),
        "source_width": int(source_width),
        "source_height": int(source_height),
        "padded_width": int(
            math.ceil(source_width / args.resolution) * args.resolution
        ),
        "padded_height": int(
            math.ceil(source_height / args.resolution) * args.resolution
        ),
        "height_encoding": (
            "uint8 normalized DSM in decimeters; value 255 means >=25.5 m"
        ),
        "height_source": str(args.height_source),
        "height_source_counts": {
            source: list(height_source_by_tile.values()).count(source)
            for source in sorted(set(height_source_by_tile.values()))
        },
        "height_source_by_tile": height_source_by_tile,
        "alignment_adjustments": alignment_adjustments,
        "tiles": sorted(rgb),
        "images": len(rgb),
        "patches_per_image": int(patch_count),
        "patches": int(len(rgb) * patch_count),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
