"""Checkpoint-compatible LoRA layers authored for UniFlow-RS."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


CHECKPOINT_FORMATS = {"uniflow_rs_lora_v1", "uniflow_rs_lora_v2"}


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if int(rank) <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))
        self.lora_A = nn.Parameter(
            torch.empty(self.rank, base.in_features, device=base.weight.device, dtype=torch.float32)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(base.out_features, self.rank, device=base.weight.device, dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, inputs: Tensor) -> Tensor:
        frozen = self.base(inputs)
        low_rank = F.linear(F.linear(self.dropout(inputs).float(), self.lora_A), self.lora_B)
        return frozen + low_rank.to(frozen.dtype) * self.scaling


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 32
    alpha: float = 32.0
    dropout: float = 0.0


def inject_lora(
    module: nn.Module,
    rank: int = 32,
    alpha: float = 32.0,
    dropout: float = 0.0,
) -> list[str]:
    names: list[str] = []

    def visit(parent: nn.Module, prefix: str) -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, LoRALinear):
                continue
            if isinstance(child, nn.Linear):
                setattr(parent, child_name, LoRALinear(child, rank, alpha, dropout))
                names.append(full_name)
            else:
                visit(child, full_name)

    visit(module, "")
    if not names:
        raise RuntimeError("No linear layers were available for LoRA injection")
    return names


def lora_state_dict(module: nn.Module) -> dict[str, Tensor]:
    return {
        name: tensor.detach().cpu()
        for name, tensor in module.state_dict().items()
        if name.endswith(("lora_A", "lora_B"))
    }


def save_lora_checkpoint(
    path: str | Path,
    module: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler,
    epoch: int,
    global_step: int,
    best_building_iou: float,
    config: LoRAConfig,
    metrics: dict | None = None,
    base_model: str | None = None,
    model_variant: str = "uniflow-rs-flux2-klein-base-4b-multitask",
    best_primary_metrics: dict[str, float] | None = None,
    training_state: dict | None = None,
    **best_values,
) -> None:
    payload: dict[str, object] = {
        "format": "uniflow_rs_lora_v2",
        "base_model": base_model,
        "model_variant": str(model_variant),
        "lora": {
            "rank": int(config.rank),
            "alpha": float(config.alpha),
            "dropout": float(config.dropout),
        },
        "state_dict": lora_state_dict(module),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_building_iou": float(best_building_iou),
        "metrics": metrics,
    }
    for name, value in best_values.items():
        if value is not None:
            payload[str(name)] = float(value)
    if best_primary_metrics is not None:
        payload["best_primary_metrics"] = {
            str(name): float(value) for name, value in best_primary_metrics.items()
        }
    if training_state is not None:
        payload["training_state"] = training_state
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, Path(path))


def _expanded_lora_state(
    module: nn.Module,
    incoming: dict[str, Tensor],
    source: LoRAConfig,
    target: LoRAConfig,
) -> dict[str, Tensor]:
    if target.rank < source.rank:
        raise ValueError("A LoRA checkpoint cannot be losslessly reduced in rank")
    if target.rank == source.rank and target.alpha == source.alpha:
        return incoming
    result: dict[str, Tensor] = {}
    current = module.state_dict()
    rescale_b = (source.alpha / source.rank) / (target.alpha / target.rank)
    for name, tensor in incoming.items():
        destination = current.get(name)
        if destination is None or tensor.ndim != 2:
            result[name] = tensor
            continue
        expanded = destination.detach().cpu().clone()
        if name.endswith("lora_A"):
            expanded[: source.rank].copy_(tensor)
        elif name.endswith("lora_B"):
            expanded[:, : source.rank].copy_(tensor * rescale_b)
        else:
            expanded = tensor
        result[name] = expanded
    return result


def load_lora_checkpoint(
    path: str | Path,
    module: nn.Module,
    inject: bool = True,
    expected_base_model: str | None = None,
    expected_model_variant: str | None = None,
    adapt_single_image_condition: bool = False,
    target_config: LoRAConfig | None = None,
    allow_rank_expansion: bool = False,
    checkpoint_state: dict | None = None,
) -> dict:
    if adapt_single_image_condition:
        raise ValueError("The FLUX.2 checkpoint already uses the final three-stream condition layout")
    checkpoint = checkpoint_state or torch.load(Path(path), map_location="cpu", weights_only=False)
    if checkpoint.get("format") not in CHECKPOINT_FORMATS:
        raise ValueError("Use the anonymized UniFlow-RS checkpoint supplied with this release")
    if expected_base_model is not None and checkpoint.get("base_model") != expected_base_model:
        raise ValueError("Checkpoint base-model metadata does not match the configured FLUX.2 model")
    if expected_model_variant is not None and checkpoint.get("model_variant") != expected_model_variant:
        raise ValueError("Checkpoint model-variant metadata does not match the configured UniFlow-RS model")
    metadata = checkpoint["lora"]
    source = LoRAConfig(
        rank=int(metadata["rank"]),
        alpha=float(metadata["alpha"]),
        dropout=float(metadata.get("dropout", 0.0)),
    )
    target = target_config or source
    if target.rank != source.rank and not allow_rank_expansion:
        raise ValueError("LoRA rank mismatch; enable explicit rank expansion to continue")
    if inject:
        inject_lora(module, target.rank, target.alpha, target.dropout)
    state = _expanded_lora_state(module, dict(checkpoint["state_dict"]), source, target)
    incompatible = module.load_state_dict(state, strict=False)
    missing = [name for name in incompatible.missing_keys if name.endswith(("lora_A", "lora_B"))]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"LoRA state mismatch: missing={missing}, unexpected={incompatible.unexpected_keys}"
        )
    return checkpoint
