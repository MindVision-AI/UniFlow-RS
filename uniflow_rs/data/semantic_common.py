"""Data components: semantic common."""

from __future__ import annotations

from typing import (
    Sequence,
)
import numpy as np
import torch
from torch import (
    Tensor,
)


def _semantic_palette_distance(
    decoded: Tensor,
    palette: np.ndarray,
    dataset_name: str,
) -> Tensor:
    """Squared normalized-RGB distance to every evaluated class color."""

    if decoded.ndim != 4 or decoded.shape[1] != 3:
        raise ValueError(
            f"Expected decoded {dataset_name} RGB tensor with shape "
            f"[B,3,H,W], got {tuple(decoded.shape)}"
        )
    palette = np.asarray(palette, dtype=np.uint8)
    if palette.ndim != 2 or palette.shape[1] != 3 or len(palette) < 2:
        raise ValueError("Semantic palette must have shape [C+1, 3]")
    colors = (
        torch.as_tensor(
            palette[1:],
            device=decoded.device,
            dtype=torch.float32,
        )
        .div(127.5)
        .sub(1.0)
    )
    pixels = decoded.float().clamp(-1.0, 1.0).permute(0, 2, 3, 1)
    return (
        (pixels.unsqueeze(-2) - colors.view(1, 1, 1, len(colors), 3))
        .square()
        .sum(dim=-1)
    )


def _semantic_confusion(
    prediction: Tensor,
    target: Tensor,
    class_count: int,
) -> Tensor:
    prediction = prediction.long()
    target = target.long()
    valid = (target >= 1) & (target <= class_count)
    if not bool(valid.any()):
        return torch.zeros(
            class_count,
            class_count,
            device=target.device,
            dtype=torch.float64,
        )
    target_index = target[valid] - 1
    prediction_index = prediction[valid].clamp(1, class_count) - 1
    bins = torch.bincount(
        target_index * class_count + prediction_index,
        minlength=class_count * class_count,
    )
    return bins.reshape(class_count, class_count).to(torch.float64)


def _semantic_metrics(
    confusion: Tensor,
    prefix: str,
    class_names: Sequence[str],
) -> dict[str, float]:
    confusion = confusion.detach().cpu().double()
    true_positive = confusion.diag()
    ground_truth = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)
    union = ground_truth + predicted - true_positive
    class_iou = true_positive / union.clamp_min(1.0)
    class_dice = 2.0 * true_positive / (ground_truth + predicted).clamp_min(1.0)
    class_accuracy = true_positive / ground_truth.clamp_min(1.0)
    valid_classes = ground_truth > 0
    mean_iou = float(class_iou[valid_classes].mean())
    mean_dice = float(class_dice[valid_classes].mean())
    pixel_accuracy = float(true_positive.sum() / confusion.sum().clamp_min(1.0))
    mean_accuracy = float(class_accuracy[valid_classes].mean())
    metrics = {
        f"{prefix}_iou": mean_iou,
        f"{prefix}_dice": mean_dice,
        f"{prefix}_miou": mean_iou,
        f"{prefix}_mdice": mean_dice,
        f"{prefix}_pixel_accuracy": pixel_accuracy,
        f"{prefix}_mean_accuracy": mean_accuracy,
    }
    for index, name in enumerate(class_names):
        metrics[f"{prefix}_{name}_iou"] = float(class_iou[index])
        metrics[f"{prefix}_{name}_dice"] = float(class_dice[index])
    return metrics
