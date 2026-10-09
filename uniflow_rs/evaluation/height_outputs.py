"""Evaluation components: height outputs."""

from __future__ import annotations

from pathlib import (
    Path,
)
import numpy as np
import torch
from PIL import (
    Image,
)
from tqdm.auto import (
    tqdm,
)
from uniflow_rs.evaluation.visualization import (
    _save_comparison,
    _uint8_rgb,
)


def _height_colormap(
    height_meters: np.ndarray,
    maximum: float,
) -> np.ndarray:
    normalized = np.clip(height_meters / max(float(maximum), 1.0e-6), 0, 1)
    stops = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    colors = np.asarray(
        [
            [32, 48, 160],
            [0, 190, 255],
            [40, 210, 90],
            [255, 225, 30],
            [210, 20, 30],
        ],
        dtype=np.float32,
    )
    return (
        np.stack(
            [np.interp(normalized, stops, colors[:, channel]) for channel in range(3)],
            axis=-1,
        )
        .round()
        .astype(np.uint8)
    )


def save_height_tile(
    output_dir: Path,
    name: str,
    prediction: torch.Tensor,
    max_height_meters: float,
) -> None:
    (output_dir / "tile_prediction_dm").mkdir(
        parents=True,
        exist_ok=True,
    )
    prediction_np = (prediction.detach().cpu().float().numpy()).clip(
        0.0, max_height_meters
    )
    Image.fromarray(
        np.rint(prediction_np * 10.0).astype(np.uint16),
    ).save(output_dir / "tile_prediction_dm" / f"{name}.png")


def save_height_comparison(
    output_dir: Path,
    name: str,
    sar: torch.Tensor,
    rgb: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    max_height_meters: float,
    raw_prediction: torch.Tensor | None = None,
) -> None:
    """Save one paired-modality height result without assuming a tile grid."""

    for directory in (
        "height_ground_truth_dm",
        "height_raw_prediction_dm",
        "height_prediction_dm",
        "height_raw_absolute_error_dm",
        "height_absolute_error_dm",
        "height_comparison",
    ):
        (output_dir / directory).mkdir(parents=True, exist_ok=True)
    sar_np = _uint8_rgb(sar)
    rgb_np = _uint8_rgb(rgb)
    prediction_np = (
        prediction.detach()
        .cpu()
        .float()
        .numpy()
        .clip(
            0.0,
            max_height_meters,
        )
    )
    raw_prediction_np = (
        prediction_np
        if raw_prediction is None
        else raw_prediction.detach()
        .cpu()
        .float()
        .numpy()
        .clip(
            0.0,
            max_height_meters,
        )
    )
    target_np = (
        target.detach()
        .cpu()
        .float()
        .numpy()
        .clip(
            0.0,
            max_height_meters,
        )
    )
    error_np = np.abs(prediction_np - target_np)
    raw_error_np = np.abs(raw_prediction_np - target_np)
    Image.fromarray(np.rint(target_np * 10.0).astype(np.uint16)).save(
        output_dir / "height_ground_truth_dm" / f"{name}.png"
    )
    Image.fromarray(np.rint(prediction_np * 10.0).astype(np.uint16)).save(
        output_dir / "height_prediction_dm" / f"{name}.png"
    )
    Image.fromarray(np.rint(raw_prediction_np * 10.0).astype(np.uint16)).save(
        output_dir / "height_raw_prediction_dm" / f"{name}.png"
    )
    Image.fromarray(np.rint(error_np * 10.0).astype(np.uint16)).save(
        output_dir / "height_absolute_error_dm" / f"{name}.png"
    )
    Image.fromarray(np.rint(raw_error_np * 10.0).astype(np.uint16)).save(
        output_dir / "height_raw_absolute_error_dm" / f"{name}.png"
    )
    target_color = _height_colormap(target_np, max_height_meters)
    raw_prediction_color = _height_colormap(
        raw_prediction_np,
        max_height_meters,
    )
    prediction_color = _height_colormap(prediction_np, max_height_meters)
    raw_error_color = _height_colormap(
        np.minimum(raw_error_np, 10.0),
        10.0,
    )
    error_color = _height_colormap(np.minimum(error_np, 10.0), 10.0)
    overlay = (
        (0.65 * rgb_np.astype(np.float32) + 0.35 * prediction_color.astype(np.float32))
        .clip(0, 255)
        .astype(np.uint8)
    )
    _save_comparison(
        output_dir / "height_comparison" / f"{name}.png",
        [
            ("SAR input", sar_np),
            ("RGB input", rgb_np),
            ("Ground-truth height", target_color),
            ("Raw prediction", raw_prediction_color),
            ("Postprocessed prediction", prediction_color),
            ("Raw absolute error (0-10 m)", raw_error_color),
            ("Post absolute error (0-10 m)", error_color),
            ("RGB + postprocessed", overlay),
        ],
    )


def _height_preview(
    decimeters: np.ndarray,
    preview_size: int,
) -> np.ndarray:
    image = Image.fromarray(decimeters.astype(np.float32))
    image = image.resize(
        (preview_size, preview_size),
        Image.Resampling.BILINEAR,
    )
    return np.asarray(image, dtype=np.float32) / 10.0


def stitch_height_predictions(
    output_dir: Path,
    dataset,
    max_height_meters: float,
    preview_size: int = 1024,
) -> None:
    """Stitch non-overlapping tile files back to each original 6000 image."""

    samples = getattr(dataset, "samples", None)
    if samples is None:
        raise TypeError("Height stitching requires dataset.samples metadata")
    grouped: dict[str, list[dict]] = {}
    for sample in samples:
        parent = str(sample["parent"])
        grouped.setdefault(parent, []).append(sample)
    directories = (
        "stitched_image",
        "stitched_prediction_dm",
        "stitched_ground_truth_dm",
        "stitched_comparison",
    )
    for directory in directories:
        (output_dir / directory).mkdir(parents=True, exist_ok=True)

    tile_size = int(getattr(dataset, "resolution", 512))
    source_size = int(getattr(dataset, "source_size", 6000))
    for parent, parent_samples in tqdm(
        sorted(grouped.items()),
        desc="Stitching height predictions",
    ):
        image_canvas = np.zeros(
            (source_size, source_size, 3),
            dtype=np.uint8,
        )
        prediction_dm = np.zeros(
            (source_size, source_size),
            dtype=np.uint16,
        )
        target_dm = np.zeros_like(prediction_dm)
        coverage = np.zeros(
            (source_size, source_size),
            dtype=np.uint8,
        )
        for sample in parent_samples:
            name = str(sample["tile_id"])
            row = int(sample["row"])
            column = int(sample["column"])
            valid_height = int(sample["valid_height"])
            valid_width = int(sample["valid_width"])
            top = row * tile_size
            left = column * tile_size
            image_path = Path(sample["image"])
            prediction_path = output_dir / "tile_prediction_dm" / f"{name}.png"
            target_path = Path(sample["height"])
            for path in (image_path, prediction_path, target_path):
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Missing height tile required for stitching: {path}"
                    )
            with Image.open(image_path) as source:
                image_tile = np.asarray(
                    source.convert("RGB"),
                    dtype=np.uint8,
                )
            with Image.open(prediction_path) as source:
                prediction_tile = np.asarray(
                    source,
                    dtype=np.uint16,
                )
            with Image.open(target_path) as source:
                target_tile = np.asarray(
                    source,
                    dtype=np.uint16,
                )
            rows = slice(top, top + valid_height)
            columns = slice(left, left + valid_width)
            image_canvas[rows, columns] = image_tile[
                :valid_height,
                :valid_width,
            ]
            prediction_dm[rows, columns] = prediction_tile[
                :valid_height,
                :valid_width,
            ]
            target_dm[rows, columns] = target_tile[
                :valid_height,
                :valid_width,
            ]
            coverage[rows, columns] += 1
        if not bool((coverage == 1).all()):
            raise RuntimeError(
                f"Height patches for '{parent}' do not cover every source "
                "pixel exactly once"
            )

        Image.fromarray(image_canvas).save(
            output_dir / "stitched_image" / f"{parent}.png"
        )
        Image.fromarray(prediction_dm).save(
            output_dir / "stitched_prediction_dm" / f"{parent}.png"
        )
        Image.fromarray(target_dm).save(
            output_dir / "stitched_ground_truth_dm" / f"{parent}.png"
        )

        image_preview = np.asarray(
            Image.fromarray(image_canvas).resize(
                (preview_size, preview_size),
                Image.Resampling.BILINEAR,
            ),
            dtype=np.uint8,
        )
        prediction_preview = _height_preview(
            prediction_dm,
            preview_size,
        )
        target_preview = _height_preview(target_dm, preview_size)
        error_preview = np.abs(prediction_preview - target_preview)
        prediction_color = _height_colormap(
            prediction_preview,
            max_height_meters,
        )
        target_color = _height_colormap(
            target_preview,
            max_height_meters,
        )
        error_color = _height_colormap(
            np.minimum(error_preview, 5.0),
            5.0,
        )
        overlay = (
            (
                0.65 * image_preview.astype(np.float32)
                + 0.35 * prediction_color.astype(np.float32)
            )
            .clip(0, 255)
            .astype(np.uint8)
        )
        _save_comparison(
            output_dir / "stitched_comparison" / f"{parent}.png",
            [
                ("Aerial image", image_preview),
                ("Ground truth height (0-25.5 m)", target_color),
                ("Predicted height (0-25.5 m)", prediction_color),
                ("Absolute error (0-5 m)", error_color),
                ("Prediction overlay", overlay),
            ],
        )
