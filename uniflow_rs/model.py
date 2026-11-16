"""UniFlow-RS: one shared prediction process for six task families.

Architecture (the same transformer and VAE serve every task):
    image(s) -> frozen VAE encoder -> source state + visual conditions
    prompt   -> frozen text encoder / cached text conditions
    source + conditions -> FLUX.2 with shared LoRA -> target latent
    target latent -> frozen VAE decoder -> structured RGB target
    structured target -> deterministic task recovery -> native prediction

UniFlowRS groups the model components in a dataclass. Checkpoints are loaded
into the transformer. Training-time TMAS lives in training/tmas.py; dataset
transforms, task recovery, and metrics are handled by the data and evaluation
modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from uniflow_rs._compat.backbone import (
    BackboneRuntime,
    load_backbone_autoencoder,
    load_backbone_transformer,
    resolve_backbone,
)
from uniflow_rs._compat.deventer_flow import (
    predict_multitask_decoded,
    predict_multitask_probability,
    select_prompt_condition,
)
from uniflow_rs._compat.lora import load_lora_checkpoint
from uniflow_rs.text_conditions import load_prompt_conditions


@dataclass
class UniFlowRS:
    """Shared transformer, VAE, and text conditions.

    Prediction methods expect preprocessed image tensors and encoded text.
    The caller supplies task source/conditioning options and inference
    settings, and manages the no_grad and autocast contexts.
    """

    transformer: Any
    autoencoder: Any
    prompt_bank: Any
    backbone: BackboneRuntime

    @classmethod
    def from_checkpoint(
        cls,
        cfg: dict,
        checkpoint: str | Path,
        accelerator,
        *,
        flux_model_dir: str | Path | None = None,
        text_model_dir: str | Path | None = None,
    ) -> UniFlowRS:
        """Load the pretrained components, text conditions, and LoRA adapter.

        The evaluation entry point resolves CLI overrides. The adapter loader
        checks the checkpoint's base-model and model-variant identifiers.
        """
        with accelerator.main_process_first():
            backbone = resolve_backbone(cfg, root_model_dir=flux_model_dir)
        checkpoint = Path(checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        prompt_bank = load_prompt_conditions(accelerator, cfg, text_model_dir)
        autoencoder = (
            load_backbone_autoencoder(backbone, device=accelerator.device)
            .eval()
            .requires_grad_(False)
        )
        checkpoint_state = torch.load(
            checkpoint, map_location="cpu", weights_only=False,
        )
        transformer = load_backbone_transformer(backbone, device=accelerator.device)
        load_lora_checkpoint(
            checkpoint,
            transformer,
            inject=True,
            expected_base_model=backbone.base_model,
            expected_model_variant=backbone.model_variant,
            checkpoint_state=checkpoint_state,
        )
        del checkpoint_state
        transformer.eval().requires_grad_(False)
        return cls(transformer, autoencoder, prompt_bank, backbone)

    def text_condition(self, prompt_keys, device, *, prompt_indices=None):
        """Select fixed task prompts or per-expression cached embeddings."""
        return select_prompt_condition(
            self.prompt_bank, prompt_keys, device, prompt_indices=prompt_indices,
        )

    def predict_target(self, **inputs):
        """Encode images, construct source, integrate flow, decode RGB target.

        Flow integration uses a Beta time grid and Euler updates.
        """
        return predict_multitask_decoded(
            transformer=self.transformer, autoencoder=self.autoencoder, **inputs,
        )

    def predict_probability(self, **inputs):
        """Predict a structured target and recover binary probabilities."""
        return predict_multitask_probability(
            transformer=self.transformer, autoencoder=self.autoencoder, **inputs,
        )

    def predict_openearthmap(self, **inputs):
        """Predict OpenEarthMap labels with sliding windows and palette recovery."""
        from uniflow_rs.task_runtime import predict_sar_semantic_output

        return predict_sar_semantic_output(
            transformer=self.transformer, autoencoder=self.autoencoder, **inputs,
        )
