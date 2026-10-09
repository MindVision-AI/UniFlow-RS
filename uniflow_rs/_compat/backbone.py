"""FLUX.2-only backbone assembly for the source-isolated release."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from uniflow_rs._compat.flux2_compat import (
    FLUX2_AUTOENCODER_MODEL,
    FLUX2_KLEIN_BASE_4B_FILENAME,
    FLUX2_KLEIN_BASE_4B_MODEL,
    FLUX2_MULTITASK_VARIANT,
    Flux2Files,
    load_flux2_autoencoder,
    load_flux2_klein_base_for_uniflow_rs,
    resolve_flux2_files,
)

FLUX2_BACKBONE = "flux2-klein-base-4b"


@dataclass(frozen=True)
class BackboneRuntime:
    name: str
    base_model: str
    model_variant: str
    text_backend: str
    files: Flux2Files


def backbone_name(cfg: dict[str, Any]) -> str:
    raw = str(cfg.get("model", {}).get("backbone", FLUX2_BACKBONE)).casefold()
    aliases = {
        "flux2": FLUX2_BACKBONE,
        "flux.2-klein-base-4b": FLUX2_BACKBONE,
        "klein-base-4b": FLUX2_BACKBONE,
    }
    value = aliases.get(raw, raw)
    if value != FLUX2_BACKBONE:
        raise ValueError(
            "The source-isolated release supports only the paper backbone "
            f"{FLUX2_BACKBONE!r}; got {raw!r}."
        )
    return value


def checkpoint_identity(cfg: dict[str, Any]) -> tuple[str, str]:
    backbone_name(cfg)
    pretrained = cfg.get("pretrained", {})
    model = cfg.get("model", {})
    return (
        str(pretrained.get("flux_repo_id", FLUX2_KLEIN_BASE_4B_MODEL)),
        str(model.get("variant", FLUX2_MULTITASK_VARIANT)),
    )


def text_backend(cfg: dict[str, Any]) -> str:
    backbone_name(cfg)
    return "qwen3"


def resolve_backbone(
    cfg: dict[str, Any],
    root_model_dir: str | Path | None = None,
) -> BackboneRuntime:
    name = backbone_name(cfg)
    pretrained = cfg.get("pretrained", {})
    base_model, variant = checkpoint_identity(cfg)
    model_root = (
        root_model_dir if root_model_dir is not None else pretrained.get("local_dir")
    )
    if pretrained.get("adapted_autoencoder_checkpoint") not in (
        None,
        "",
        False,
    ) or cfg.get("vae", {}).get("decoder_checkpoint") not in (None, "", False):
        raise ValueError(
            "This reviewer release uses the frozen pretrained VAE; adapted VAE checkpoints are not supported"
        )
    files = resolve_flux2_files(
        root_model_dir=model_root,
        transformer_repo_id=base_model,
        autoencoder_repo_id=str(
            pretrained.get("autoencoder_repo_id", FLUX2_AUTOENCODER_MODEL)
        ),
        transformer_filename=str(
            pretrained.get("transformer_filename", FLUX2_KLEIN_BASE_4B_FILENAME)
        ),
        autoencoder_filename=str(
            pretrained.get("autoencoder_filename", "ae.safetensors")
        ),
    )
    return BackboneRuntime(
        name=name,
        base_model=base_model,
        model_variant=variant,
        text_backend="qwen3",
        files=files,
    )


def load_backbone_autoencoder(
    runtime: BackboneRuntime,
    device: str | torch.device,
):
    """Load the unchanged pretrained VAE used by the released adapter."""
    return load_flux2_autoencoder(runtime.files.autoencoder, device=device)


def load_backbone_transformer(runtime: BackboneRuntime, device: str | torch.device):
    return load_flux2_klein_base_for_uniflow_rs(
        runtime.files.transformer,
        device=device,
        visual_streams=3,
    )
