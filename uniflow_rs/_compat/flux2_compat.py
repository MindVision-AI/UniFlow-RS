"""Compatibility layer for the lightweight FLUX.2 Klein Base backbone.

The multitask pipeline keeps its existing public tensor contract:

* images and dense targets are encoded as [B, 32, H/8, W/8] maps;
* the latent packer turns those maps into 128-D tokens;
* MMDiT receives [state, before, after] (384 channels) and predicts one
  128-D velocity token.

The native FLUX.2 VAE stores the same information as
[B, 128, H/16, W/16]. Flux2AutoEncoderAdapter only performs an invertible
2x2 space/depth rearrangement around the official VAE, so no interpolation
or learned conversion is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from einops import rearrange
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import Tensor, nn

from uniflow_rs._compat.vendor.flux2 import (
    AutoEncoder as NativeFlux2AutoEncoder,
    AutoEncoderParams,
    Flux2,
    Klein4BParams,
)


FLUX2_KLEIN_BASE_4B_MODEL = "black-forest-labs/FLUX.2-klein-base-4B"
FLUX2_AUTOENCODER_MODEL = "black-forest-labs/FLUX.2-dev"
FLUX2_KLEIN_BASE_4B_FILENAME = "flux-2-klein-base-4b.safetensors"
FLUX2_AUTOENCODER_FILENAME = "ae.safetensors"
FLUX2_QWEN3_4B_FP8_MODEL = "Qwen/Qwen3-4B-FP8"
FLUX2_MULTITASK_VARIANT = "uniflow-rs-flux2-klein-base-4b-multitask"


@dataclass(frozen=True)
class Flux2Files:
    transformer: Path
    autoencoder: Path


def resolve_flux2_files(
    root_model_dir: str | Path | None,
    transformer_repo_id: str = FLUX2_KLEIN_BASE_4B_MODEL,
    autoencoder_repo_id: str = FLUX2_AUTOENCODER_MODEL,
    transformer_filename: str = FLUX2_KLEIN_BASE_4B_FILENAME,
    autoencoder_filename: str = FLUX2_AUTOENCODER_FILENAME,
) -> Flux2Files:
    """Resolve the native Klein Base transformer and FLUX.2 VAE files."""

    if root_model_dir is not None:
        root = Path(root_model_dir).expanduser().resolve()
        transformer = root / str(transformer_filename)
        autoencoder = root / str(autoencoder_filename)
        missing = [
            str(path)
            for path in (transformer, autoencoder)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing FLUX.2 Klein file(s): " + ", ".join(missing)
            )
        return Flux2Files(transformer=transformer, autoencoder=autoencoder)

    try:
        transformer = Path(
            hf_hub_download(
                repo_id=str(transformer_repo_id),
                filename=str(transformer_filename),
            )
        )
        autoencoder = Path(
            hf_hub_download(
                repo_id=str(autoencoder_repo_id),
                filename=str(autoencoder_filename),
            )
        )
    except Exception as error:
        raise RuntimeError(
            "Could not resolve FLUX.2 Klein Base 4B. Download the two native "
            "safetensors files and set pretrained.local_dir. Required files: "
            f"{transformer_filename!r} and {autoencoder_filename!r}."
        ) from error
    return Flux2Files(transformer=transformer, autoencoder=autoencoder)


class Flux2AutoEncoderAdapter(nn.Module):
    """Expose FLUX.2's VAE through the latent-map API used by UniFlow-RS."""

    def __init__(self, autoencoder: NativeFlux2AutoEncoder):
        super().__init__()
        self.autoencoder = autoencoder

    def encode(self, image: Tensor) -> Tensor:
        native = self.autoencoder.encode(image)
        if native.ndim != 4 or native.shape[1] != 128:
            raise RuntimeError(
                "Unexpected FLUX.2 latent shape: "
                f"{tuple(native.shape)}; expected [B,128,H/16,W/16]"
            )
        return rearrange(
            native,
            "b (c ph pw) h w -> b c (h ph) (w pw)",
            ph=2,
            pw=2,
        )

    def decoder_input(self, latent: Tensor) -> Tensor:
        """Convert UniFlow-RS pseudo-latents to native decoder inputs."""

        if latent.ndim != 4 or latent.shape[1] != 32:
            raise ValueError(
                "UniFlow-RS FLUX.2 latent must be [B,32,H/8,W/8], got "
                f"{tuple(latent.shape)}"
            )
        if latent.shape[-2] % 2 or latent.shape[-1] % 2:
            raise ValueError(
                "FLUX.2 pseudo-latent height and width must be even, got "
                f"{tuple(latent.shape[-2:])}"
            )
        native = rearrange(
            latent,
            "b c (h ph) (w pw) -> b (c ph pw) h w",
            ph=2,
            pw=2,
        )
        native = self.autoencoder.inv_normalize(native)
        return rearrange(
            native,
            "... (c ph pw) h w -> ... c (h ph) (w pw)",
            ph=self.autoencoder.ps[0],
            pw=self.autoencoder.ps[1],
        )

    def decode_with_decoder(self, latent: Tensor, decoder: nn.Module) -> Tensor:
        """Decode through an explicit student or frozen reference decoder."""

        return decoder(self.decoder_input(latent))

    def decode(self, latent: Tensor) -> Tensor:
        decoder = getattr(self.autoencoder, "decoder", None)
        if decoder is not None:
            return self.decode_with_decoder(latent, decoder)
        # Retain compatibility with lightweight native-AE test doubles and
        # older wrappers that expose only a monolithic decode method.
        native = rearrange(
            latent,
            "b c (h ph) (w pw) -> b (c ph pw) h w",
            ph=2,
            pw=2,
        )
        return self.autoencoder.decode(native)

    def forward(self, image: Tensor) -> Tensor:
        return self.decode(self.encode(image))


def _image_ids_4d(ids: Tensor) -> Tensor:
    if ids.ndim != 3:
        raise ValueError(f"Expected token IDs [B,L,A], got {tuple(ids.shape)}")
    if ids.shape[-1] == 4:
        return ids
    if ids.shape[-1] != 3:
        raise ValueError(
            f"FLUX.2 requires three or four incoming axes, got {ids.shape[-1]}"
        )
    return torch.cat((ids, torch.zeros_like(ids[..., :1])), dim=-1)


def _text_ids_4d(txt: Tensor, ids: Tensor | None) -> Tensor:
    batch, length = txt.shape[:2]
    device = txt.device
    dtype = ids.dtype if ids is not None else torch.float32
    output = torch.zeros(batch, length, 4, device=device, dtype=dtype)
    output[..., 3] = torch.arange(length, device=device, dtype=dtype)
    return output


class Flux2UniFlowRSAdapter(nn.Module):
    """Map the existing UniFlow-RS call signature to native FLUX.2."""

    def __init__(self, model: Flux2, visual_streams: int = 3):
        super().__init__()
        self.model = model
        self.visual_streams = int(visual_streams)
        self.in_channels = 128 * self.visual_streams
        self.out_channels = 128

    def enable_gradient_checkpointing(self) -> None:
        self.model.gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self.model.gradient_checkpointing = False

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor | None = None,
        guidance: Tensor | None = None,
    ) -> Tensor:
        del y
        if img.shape[-1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} visual channels from "
                f"[state,before,after], got {img.shape[-1]}"
            )
        return self.model(
            x=img,
            x_ids=_image_ids_4d(img_ids),
            timesteps=timesteps,
            ctx=txt,
            ctx_ids=_text_ids_4d(txt, txt_ids),
            guidance=guidance,
        )


def expand_flux2_input_projection(
    model: Flux2,
    visual_streams: int = 3,
) -> None:
    """Widen 128-D native input to [state,before,after] without perturbation."""

    visual_streams = int(visual_streams)
    if visual_streams < 1:
        raise ValueError("visual_streams must be positive")
    source = model.img_in
    if source.in_features != 128:
        raise ValueError(
            f"Expected native FLUX.2 img_in width 128, got {source.in_features}"
        )
    target_features = 128 * visual_streams
    expanded = nn.Linear(
        target_features,
        source.out_features,
        bias=False,
        device=source.weight.device,
        dtype=source.weight.dtype,
    )
    with torch.no_grad():
        expanded.weight.zero_()
        expanded.weight[:, :128].copy_(source.weight)
    model.img_in = expanded
    model.in_channels = target_features


def load_flux2_klein_base_for_uniflow_rs(
    model_path: str | Path,
    device: str | torch.device = "cuda",
    visual_streams: int = 3,
) -> Flux2UniFlowRSAdapter:
    """Load official Base-4B weights and add zero-init condition columns."""

    with torch.device("meta"):
        model = Flux2(Klein4BParams()).to(dtype=torch.bfloat16)
    state_dict = load_file(str(model_path), device="cpu")
    incompatible = model.load_state_dict(state_dict, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "FLUX.2 Klein Base checkpoint mismatch: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    del state_dict
    expand_flux2_input_projection(model, visual_streams=visual_streams)
    model = model.to(device=device, dtype=torch.bfloat16)
    return Flux2UniFlowRSAdapter(model, visual_streams=visual_streams).eval()


def load_flux2_autoencoder(
    model_path: str | Path,
    device: str | torch.device = "cuda",
) -> Flux2AutoEncoderAdapter:
    with torch.device("meta"):
        autoencoder = NativeFlux2AutoEncoder(AutoEncoderParams())
    state_dict = load_file(str(model_path), device="cpu")
    incompatible = autoencoder.load_state_dict(
        state_dict,
        strict=True,
        assign=True,
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "FLUX.2 autoencoder checkpoint mismatch: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    autoencoder = autoencoder.to(device=device, dtype=torch.bfloat16)
    return Flux2AutoEncoderAdapter(autoencoder).eval()


class Qwen3Embedder(nn.Module):
    """Official FLUX.2 Klein Qwen3 hidden-state text conditioner."""

    output_layers = (9, 18, 27)

    def __init__(
        self,
        model_spec: str | Path = FLUX2_QWEN3_4B_FP8_MODEL,
        max_length: int = 128,
        device: str | torch.device = "cuda",
    ):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_spec = str(model_spec)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_spec,
            torch_dtype=None,
            device_map=str(device),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_spec)
        self.max_length = int(max_length)

    @torch.no_grad()
    def forward(self, prompts: list[str]) -> Tensor:
        input_ids = []
        attention_masks = []
        for prompt in prompts:
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": str(prompt)}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            encoded = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )
            input_ids.append(encoded["input_ids"])
            attention_masks.append(encoded["attention_mask"])
        ids = torch.cat(input_ids, dim=0).to(self.model.device)
        masks = torch.cat(attention_masks, dim=0).to(self.model.device)
        output = self.model(
            input_ids=ids,
            attention_mask=masks,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = torch.stack(
            [output.hidden_states[index] for index in self.output_layers],
            dim=1,
        )
        return rearrange(hidden, "b layers length dim -> b length (layers dim)")


def load_qwen3_embedder(
    model_spec: str | Path,
    max_length: int,
    device: str | torch.device = "cuda",
) -> Qwen3Embedder:
    return Qwen3Embedder(
        model_spec=model_spec,
        max_length=max_length,
        device=device,
    )
