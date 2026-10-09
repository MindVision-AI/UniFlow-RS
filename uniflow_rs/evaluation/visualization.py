"""Evaluation components: visualization."""

from __future__ import annotations

from pathlib import (
    Path,
)
import numpy as np
import torch
from PIL import (
    Image,
    ImageDraw,
)
from uniflow_rs._compat.multitask import (
    LOVEDA_PALETTE,
    URBANSAR_FLOOD_PALETTE,
)


def _uint8_rgb(image: torch.Tensor) -> np.ndarray:
    return (
        image.detach()
        .cpu()
        .permute(1, 2, 0)
        .add(1.0)
        .mul(127.5)
        .clamp(0, 255)
        .byte()
        .numpy()
    )


def _save_comparison(
    path: Path,
    panels: list[tuple[str, np.ndarray]],
) -> None:
    if not panels:
        raise ValueError("At least one comparison panel is required")
    height, width = panels[0][1].shape[:2]
    if any(panel.shape[:2] != (height, width) for _, panel in panels):
        raise ValueError("Every comparison panel must have the same size")
    header_height = 34
    canvas = Image.new(
        "RGB",
        (width * len(panels), height + header_height),
        color=(255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)
    for index, (title, panel) in enumerate(panels):
        left = index * width
        draw.text(
            (left + 10, 9),
            title,
            fill=(0, 0, 0),
        )
        canvas.paste(Image.fromarray(panel.astype(np.uint8)), (left, header_height))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def save_prediction(
    output_dir: Path,
    name: str,
    before: torch.Tensor,
    after: torch.Tensor,
    probability: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    prediction: torch.Tensor | None = None,
    raw_prediction: torch.Tensor | None = None,
) -> None:
    has_postprocessed_prediction = prediction is not None
    directories = [
        "before",
        "after",
        "probability",
        "prediction",
        "overlay",
        "ground_truth",
        "comparison",
    ]
    if has_postprocessed_prediction:
        directories.extend(("prediction_raw", "overlay_raw"))
    for directory in directories:
        (output_dir / directory).mkdir(parents=True, exist_ok=True)
    before_np = _uint8_rgb(before)
    after_np = _uint8_rgb(after)
    probability_np = probability.detach().cpu().clamp(0, 1).numpy()
    raw_prediction_np = (
        probability_np >= threshold
        if raw_prediction is None
        else raw_prediction.detach().cpu().numpy().astype(bool)
    )
    prediction_np = (
        raw_prediction_np
        if prediction is None
        else prediction.detach().cpu().numpy().astype(bool)
    )
    target_np = target.detach().cpu().numpy() >= 0.5
    overlay = after_np.copy()
    overlay[prediction_np] = (
        0.6 * overlay[prediction_np].astype(np.float32)
        + 0.4 * np.asarray([220, 20, 60], dtype=np.float32)
    ).astype(np.uint8)

    Image.fromarray(before_np).save(output_dir / "before" / f"{name}.png")
    Image.fromarray(after_np).save(output_dir / "after" / f"{name}.png")
    Image.fromarray(np.rint(probability_np * 255).astype(np.uint8)).save(
        output_dir / "probability" / f"{name}.png"
    )
    Image.fromarray((prediction_np * 255).astype(np.uint8)).save(
        output_dir / "prediction" / f"{name}.png"
    )
    if has_postprocessed_prediction:
        Image.fromarray((raw_prediction_np * 255).astype(np.uint8)).save(
            output_dir / "prediction_raw" / f"{name}.png"
        )
    Image.fromarray(overlay).save(output_dir / "overlay" / f"{name}.png")
    raw_overlay = after_np.copy()
    raw_overlay[raw_prediction_np] = (
        0.6 * raw_overlay[raw_prediction_np].astype(np.float32)
        + 0.4 * np.asarray([220, 20, 60], dtype=np.float32)
    ).astype(np.uint8)
    if has_postprocessed_prediction:
        Image.fromarray(raw_overlay).save(output_dir / "overlay_raw" / f"{name}.png")
    Image.fromarray((target_np * 255).astype(np.uint8)).save(
        output_dir / "ground_truth" / f"{name}.png"
    )
    prediction_rgb = np.repeat(
        (prediction_np * 255).astype(np.uint8)[..., None],
        3,
        axis=2,
    )
    target_rgb = np.repeat(
        (target_np * 255).astype(np.uint8)[..., None],
        3,
        axis=2,
    )
    raw_prediction_rgb = np.repeat(
        (raw_prediction_np * 255).astype(np.uint8)[..., None],
        3,
        axis=2,
    )
    prediction_panels = (
        [
            ("Raw prediction", raw_prediction_rgb),
            ("Postprocessed", prediction_rgb),
            ("Postprocessed overlay", overlay),
        ]
        if has_postprocessed_prediction
        else [
            ("Prediction", prediction_rgb),
            ("Prediction overlay", overlay),
        ]
    )
    _save_comparison(
        output_dir / "comparison" / f"{name}.png",
        [
            ("Before image", before_np),
            ("After image", after_np),
            ("Ground truth", target_rgb),
            *prediction_panels,
        ],
    )


def save_referring_prediction(
    output_dir: Path,
    name: str,
    image: torch.Tensor,
    probability: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    expression: str,
) -> None:
    directories = (
        "image",
        "probability",
        "prediction",
        "overlay",
        "ground_truth",
        "comparison",
        "expressions",
    )
    for directory in directories:
        (output_dir / directory).mkdir(parents=True, exist_ok=True)
    image_np = _uint8_rgb(image)
    probability_np = probability.detach().cpu().clamp(0, 1).numpy()
    prediction_np = probability_np >= threshold
    target_np = target.detach().cpu().numpy() >= 0.5
    overlay = image_np.copy()
    overlay[prediction_np] = (
        0.6 * overlay[prediction_np].astype(np.float32)
        + 0.4 * np.asarray([220, 20, 60], dtype=np.float32)
    ).astype(np.uint8)
    Image.fromarray(image_np).save(output_dir / "image" / f"{name}.png")
    Image.fromarray(np.rint(probability_np * 255).astype(np.uint8)).save(
        output_dir / "probability" / f"{name}.png"
    )
    Image.fromarray((prediction_np * 255).astype(np.uint8)).save(
        output_dir / "prediction" / f"{name}.png"
    )
    Image.fromarray(overlay).save(output_dir / "overlay" / f"{name}.png")
    Image.fromarray((target_np * 255).astype(np.uint8)).save(
        output_dir / "ground_truth" / f"{name}.png"
    )
    with (output_dir / "expressions" / f"{name}.txt").open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(str(expression).strip() + "\n")
    prediction_rgb = np.repeat(
        (prediction_np * 255).astype(np.uint8)[..., None],
        3,
        axis=2,
    )
    target_rgb = np.repeat(
        (target_np * 255).astype(np.uint8)[..., None],
        3,
        axis=2,
    )
    title = str(expression).strip()
    if len(title) > 72:
        title = title[:69] + "..."
    _save_comparison(
        output_dir / "comparison" / f"{name}.png",
        [
            (f"Image: {title}", image_np),
            ("Ground truth", target_rgb),
            ("Prediction", prediction_rgb),
            ("Prediction overlay", overlay),
        ],
    )


def save_semantic_prediction(
    output_dir: Path,
    name: str,
    image: torch.Tensor,
    decoded: torch.Tensor,
    raw_prediction: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    palette: np.ndarray = LOVEDA_PALETTE,
    ignore_zero: bool = True,
    zero_label_name: str = "ignore",
) -> None:
    directories = (
        "image",
        "decoded_rgb",
        "prediction_raw",
        "prediction",
        "ground_truth",
        "overlay_raw",
        "overlay",
        "comparison",
    )
    for directory in directories:
        (output_dir / directory).mkdir(parents=True, exist_ok=True)
    image_np = _uint8_rgb(image)
    decoded_np = _uint8_rgb(decoded)
    raw_prediction_np = raw_prediction.detach().cpu().numpy().astype(np.int64)
    prediction_np = prediction.detach().cpu().numpy().astype(np.int64)
    target_np = target.detach().cpu().numpy().astype(np.int64)
    palette = np.asarray(palette, dtype=np.uint8)
    raw_prediction_rgb = palette[np.clip(raw_prediction_np, 0, len(palette) - 1)].copy()
    prediction_rgb = palette[np.clip(prediction_np, 0, len(palette) - 1)].copy()
    target_rgb = palette[np.clip(target_np, 0, len(palette) - 1)]
    # LoveDA ID 0 is no-data/ignore and is hidden in its overlays. Other
    # datasets may use ID 0 as a real background class; in that case keep the
    # prediction visible so false positives are not concealed.
    ignored = (
        (target_np == 0)
        if ignore_zero
        else np.zeros_like(
            target_np,
            dtype=bool,
        )
    )
    raw_prediction_rgb[ignored] = palette[0]
    prediction_rgb[ignored] = palette[0]
    overlay = (
        (0.6 * image_np.astype(np.float32) + 0.4 * prediction_rgb.astype(np.float32))
        .clip(0, 255)
        .astype(np.uint8)
    )
    raw_overlay = (
        (
            0.6 * image_np.astype(np.float32)
            + 0.4 * raw_prediction_rgb.astype(np.float32)
        )
        .clip(0, 255)
        .astype(np.uint8)
    )
    overlay[ignored] = image_np[ignored]
    raw_overlay[ignored] = image_np[ignored]
    Image.fromarray(image_np).save(output_dir / "image" / f"{name}.png")
    Image.fromarray(decoded_np).save(output_dir / "decoded_rgb" / f"{name}.png")
    Image.fromarray(raw_prediction_rgb).save(
        output_dir / "prediction_raw" / f"{name}.png"
    )
    Image.fromarray(prediction_rgb).save(output_dir / "prediction" / f"{name}.png")
    Image.fromarray(target_rgb).save(output_dir / "ground_truth" / f"{name}.png")
    Image.fromarray(raw_overlay).save(output_dir / "overlay_raw" / f"{name}.png")
    Image.fromarray(overlay).save(output_dir / "overlay" / f"{name}.png")
    _save_comparison(
        output_dir / "comparison" / f"{name}.png",
        [
            ("Aerial image", image_np),
            (
                f"Ground truth (black = {zero_label_name})",
                target_rgb,
            ),
            ("Decoded RGB prediction", decoded_np),
            ("Raw nearest palette (ignore masked)", raw_prediction_rgb),
            ("Postprocessed (ignore masked)", prediction_rgb),
            ("Postprocessed overlay", overlay),
        ],
    )


def save_sar_change_prediction(
    output_dir: Path,
    name: str,
    before: torch.Tensor,
    after: torch.Tensor,
    decoded: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> None:
    """Save paired SAR conditions and three-class flood output together."""

    directories = (
        "before",
        "after",
        "decoded_rgb",
        "prediction",
        "ground_truth",
        "overlay",
        "comparison",
    )
    for directory in directories:
        (output_dir / directory).mkdir(parents=True, exist_ok=True)
    before_np = _uint8_rgb(before)
    after_np = _uint8_rgb(after)
    decoded_np = _uint8_rgb(decoded)
    prediction_np = prediction.detach().cpu().numpy().astype(np.int64)
    target_np = target.detach().cpu().numpy().astype(np.int64)
    prediction_rgb = URBANSAR_FLOOD_PALETTE[np.clip(prediction_np, 1, 3)]
    target_rgb = URBANSAR_FLOOD_PALETTE[np.clip(target_np, 1, 3)]
    overlay = (
        (0.6 * after_np.astype(np.float32) + 0.4 * prediction_rgb.astype(np.float32))
        .clip(0, 255)
        .astype(np.uint8)
    )
    for directory, image in (
        ("before", before_np),
        ("after", after_np),
        ("decoded_rgb", decoded_np),
        ("prediction", prediction_rgb),
        ("ground_truth", target_rgb),
        ("overlay", overlay),
    ):
        Image.fromarray(image).save(output_dir / directory / f"{name}.png")
    _save_comparison(
        output_dir / "comparison" / f"{name}.png",
        [
            ("Pre-event SAR composite", before_np),
            ("Co-event SAR composite", after_np),
            ("Ground truth: NF / FO / FU", target_rgb),
            ("Decoded RGB endpoint", decoded_np),
            ("Prediction: NF / FO / FU", prediction_rgb),
            ("Prediction overlay", overlay),
        ],
    )
