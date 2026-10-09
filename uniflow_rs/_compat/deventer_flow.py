"""Training and inference primitives for UniFlow-RS."""

from __future__ import annotations

import gc
import json
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from scipy import stats
from torch import Tensor
from torch.nn import functional as F

from uniflow_rs._compat.deventer import (
    decoded_rgb_to_probability,
    make_image_ids,
    pack_latent,
    unpack_latent,
)
from uniflow_rs._compat.flux2_compat import (
    FLUX2_QWEN3_4B_FP8_MODEL,
    load_qwen3_embedder,
)


def beta_scheduler(
    num_timesteps: int,
    alpha: float = 2.5,
    beta: float = 1.0,
) -> list[float]:
    """Return the descending Beta-quantile integration grid used in the paper."""

    quantiles = torch.linspace(1.0, 0.0, int(num_timesteps) + 1)
    schedule = stats.beta.ppf(quantiles.numpy(), float(alpha), float(beta)).tolist()
    if schedule[-1] > 0.0:
        schedule.append(0.0)
    return [float(value) for value in schedule]


@contextmanager
def fixed_torch_seed(seed: int, device: torch.device):
    """Temporarily make stochastic VAE sampling reproducible.

    Restoring the incoming RNG state is important: validation must be
    deterministic without changing the randomness of subsequent training.
    """

    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(int(seed))
    torch.random.set_rng_state(cpu_generator.get_state())
    if device.type == "cuda":
        torch.cuda.manual_seed(int(seed))
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state, device)


@torch.no_grad()
def encode_prompt_bank(
    prompts: Mapping[str, str],
    device: torch.device,
    text_model_dir: str | Path | None = None,
    sequence_length: int = 512,
    text_backend: str = "qwen3",
    qwen_model_path: str | Path | None = None,
) -> dict[str, tuple[Tensor, Tensor]]:
    """Encode the fixed task-prompt bank with FLUX.2 Klein's Qwen3 encoder."""

    if not prompts:
        raise ValueError("At least one task prompt is required")
    sequence_length = int(sequence_length)
    if not 1 <= sequence_length <= 512:
        raise ValueError("sequence_length must be in [1, 512]")
    task_names = list(prompts)
    prompt_texts = [str(prompts[name]) for name in task_names]
    text_backend = str(text_backend).casefold()
    if text_backend != "qwen3":
        raise ValueError(
            "The source-isolated release supports only text_backend='qwen3'"
        )
    qwen = (
        load_qwen3_embedder(
            text_model_dir or qwen_model_path or FLUX2_QWEN3_4B_FP8_MODEL,
            max_length=sequence_length,
            device=device,
        )
        .eval()
        .requires_grad_(False)
    )
    txt_all = qwen(prompt_texts).detach().cpu()
    vec_all = torch.zeros(len(prompt_texts), 1, dtype=torch.bfloat16)
    del qwen
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        name: (txt_all[index : index + 1], vec_all[index : index + 1])
        for index, name in enumerate(task_names)
    }


class DiskPromptEmbeddingCache:
    """Read frozen Qwen features; retain legacy serialized-cache compatibility."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser()
        metadata_path = self.root / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Missing referring-expression text cache: {metadata_path}\n"
                "Run prepare_rrsisd_text_embeddings.py with the matching "
                "dataset config before training."
            )
        with metadata_path.open("r", encoding="utf-8") as file:
            self.metadata = json.load(file)
        supported_formats = {
            "rrsisd_text_embeddings_v1",
            "referring_text_embeddings_v1",
            "referring_text_embeddings_v2",
        }
        if self.metadata.get("format") not in supported_formats:
            raise ValueError(
                f"Unsupported text-cache format in '{metadata_path}': "
                f"{self.metadata.get('format')!r}"
            )
        self.count = int(self.metadata["count"])
        self.sequence_length = int(self.metadata["sequence_length"])
        self.is_qwen = self.metadata.get("format") == "referring_text_embeddings_v2"
        text_file = (
            self.metadata["text_file"] if self.is_qwen else self.metadata["t5_file"]
        )
        self.t5 = np.load(
            self.root / str(text_file),
            mmap_mode="r",
        )
        self.clip = (
            None
            if self.is_qwen
            else np.load(
                self.root / str(self.metadata["clip_file"]),
                mmap_mode="r",
            )
        )
        expected_t5 = (
            self.count,
            self.sequence_length,
            int(
                self.metadata["text_hidden_size" if self.is_qwen else "t5_hidden_size"]
            ),
        )
        if tuple(self.t5.shape) != expected_t5:
            raise ValueError(
                f"Cached text-feature shape {self.t5.shape} != {expected_t5}"
            )
        if self.clip is not None:
            expected_clip = (
                self.count,
                int(self.metadata["clip_hidden_size"]),
            )
            if tuple(self.clip.shape) != expected_clip:
                raise ValueError(
                    f"Cached pooled-feature shape {self.clip.shape} != {expected_clip}"
                )
        if self.t5.dtype != np.float16 or (
            self.clip is not None and self.clip.dtype != np.float16
        ):
            raise ValueError("Referring text caches must use float16")

    def rows(
        self,
        indices: Sequence[int],
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        values = [int(index) for index in indices]
        invalid = sorted({index for index in values if not 0 <= index < self.count})
        if invalid:
            raise IndexError(
                f"Referring prompt indices outside [0, {self.count}): " f"{invalid}"
            )
        # Explicit copies avoid undefined behavior from read-only NumPy maps.
        txt = torch.from_numpy(np.array(self.t5[values], dtype=np.float16, copy=True))
        vec = (
            torch.zeros(len(values), 1, dtype=torch.float16)
            if self.clip is None
            else torch.from_numpy(
                np.array(self.clip[values], dtype=np.float16, copy=True)
            )
        )
        return (
            txt.to(
                device=device,
                dtype=torch.bfloat16,
                non_blocking=True,
            ),
            vec.to(
                device=device,
                dtype=torch.bfloat16,
                non_blocking=True,
            ),
        )


class CompositeDiskPromptEmbeddingCache(DiskPromptEmbeddingCache):
    """Expose several compatible prompt caches through one index space.

    The referring datasets keep independent manifests and on-disk caches.
    Their dataset wrappers add a fixed prompt-index offset, while this class
    maps those global indices back to the corresponding memory map.  No text
    feature is copied or merged on disk.
    """

    def __init__(self, caches: Sequence[DiskPromptEmbeddingCache]):
        self.caches = tuple(caches)
        if not self.caches:
            raise ValueError("At least one prompt cache is required")
        sequence_lengths = {int(cache.sequence_length) for cache in self.caches}
        if len(sequence_lengths) != 1:
            raise ValueError(
                "All referring text caches must have the same sequence "
                f"length, got {sorted(sequence_lengths)}"
            )
        t5_hidden_sizes = {
            int(
                cache.metadata.get(
                    "text_hidden_size",
                    cache.metadata.get("t5_hidden_size"),
                )
            )
            for cache in self.caches
        }
        clip_hidden_sizes = {
            int(cache.metadata.get("clip_hidden_size", 1)) for cache in self.caches
        }
        if len(t5_hidden_sizes) != 1 or len(clip_hidden_sizes) != 1:
            raise ValueError(
                "All referring caches must use matching text and pooled feature dimensions"
            )
        self.sequence_length = sequence_lengths.pop()
        self.offsets: tuple[int, ...] = tuple(
            sum(cache.count for cache in self.caches[:index])
            for index in range(len(self.caches))
        )
        self.count = sum(cache.count for cache in self.caches)
        self.metadata = {
            "format": "composite_referring_text_embeddings_v1",
            "count": self.count,
            "sequence_length": self.sequence_length,
            "t5_hidden_size": t5_hidden_sizes.pop(),
            "clip_hidden_size": clip_hidden_sizes.pop(),
        }

    def rows(
        self,
        indices: Sequence[int],
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        values = [int(index) for index in indices]
        invalid = sorted({index for index in values if not 0 <= index < self.count})
        if invalid:
            raise IndexError(
                f"Composite prompt indices outside [0, {self.count}): " f"{invalid}"
            )

        txt_rows: list[Tensor | None] = [None] * len(values)
        vec_rows: list[Tensor | None] = [None] * len(values)
        for cache_index, (cache, offset) in enumerate(
            zip(self.caches, self.offsets, strict=True)
        ):
            upper = (
                self.offsets[cache_index + 1]
                if cache_index + 1 < len(self.offsets)
                else self.count
            )
            positions = [
                row for row, value in enumerate(values) if offset <= value < upper
            ]
            if not positions:
                continue
            cached_txt, cached_vec = cache.rows(
                [values[row] - offset for row in positions],
                device,
            )
            for cache_row, output_row in enumerate(positions):
                txt_rows[output_row] = cached_txt[cache_row : cache_row + 1]
                vec_rows[output_row] = cached_vec[cache_row : cache_row + 1]
        assert all(row is not None for row in txt_rows)
        assert all(row is not None for row in vec_rows)
        return (
            torch.cat([row for row in txt_rows if row is not None], dim=0),
            torch.cat([row for row in vec_rows if row is not None], dim=0),
        )


def select_prompt_condition(
    prompt_bank: Mapping[
        str,
        tuple[Tensor, Tensor] | DiskPromptEmbeddingCache,
    ],
    tasks: Sequence[str],
    device: torch.device,
    prompt_indices: Sequence[int] | Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Select the correct cached natural-language condition per sample."""

    unknown = sorted(set(tasks) - set(prompt_bank))
    if unknown:
        raise KeyError(f"No cached text prompt for task(s): {unknown}")
    if prompt_indices is None:
        indices = [-1] * len(tasks)
    elif isinstance(prompt_indices, Tensor):
        indices = [int(value) for value in prompt_indices.detach().cpu().tolist()]
    else:
        indices = [int(value) for value in prompt_indices]
    if len(indices) != len(tasks):
        raise ValueError(f"Got {len(indices)} prompt indices for {len(tasks)} tasks")

    txt_rows: list[Tensor | None] = [None] * len(tasks)
    vec_rows: list[Tensor | None] = [None] * len(tasks)
    cache_groups: dict[int, tuple[DiskPromptEmbeddingCache, list[int]]] = {}
    for row, (task, prompt_index) in enumerate(zip(tasks, indices)):
        condition = prompt_bank[str(task)]
        if isinstance(condition, DiskPromptEmbeddingCache):
            if prompt_index < 0:
                raise ValueError(f"Task '{task}' requires a non-negative prompt_index")
            identity = id(condition)
            if identity not in cache_groups:
                cache_groups[identity] = (condition, [])
            cache_groups[identity][1].append(row)
        else:
            txt_rows[row] = condition[0]
            vec_rows[row] = condition[1]
    for cache, rows in cache_groups.values():
        cached_txt, cached_vec = cache.rows(
            [indices[row] for row in rows],
            device,
        )
        for cache_row, output_row in enumerate(rows):
            txt_rows[output_row] = cached_txt[cache_row : cache_row + 1]
            vec_rows[output_row] = cached_vec[cache_row : cache_row + 1]
    assert all(row is not None for row in txt_rows)
    assert all(row is not None for row in vec_rows)
    sequence_lengths = {int(row.shape[1]) for row in txt_rows if row is not None}
    if len(sequence_lengths) != 1:
        raise ValueError(
            "Fixed task prompts and per-sample referring prompts must use "
            f"the same text sequence length, got {sorted(sequence_lengths)}"
        )
    # Fixed-task embeddings are loaded from the shared on-disk cache onto
    # CPU, whereas per-expression referring embeddings are copied directly
    # to this rank's device.  Mixed-task batches therefore have to normalize
    # every row *before* torch.cat; moving the concatenated result afterward
    # is too late because torch.cat requires a single device.
    txt = torch.cat(
        [
            row.to(
                device=device,
                dtype=torch.bfloat16,
                non_blocking=True,
            )
            for row in txt_rows
            if row is not None
        ],
        dim=0,
    )
    vec = torch.cat(
        [
            row.to(
                device=device,
                dtype=torch.bfloat16,
                non_blocking=True,
            )
            for row in vec_rows
            if row is not None
        ],
        dim=0,
    )
    txt_ids = torch.zeros(
        len(tasks),
        txt.shape[1],
        3,
        device=device,
        dtype=torch.float32,
    )
    return txt, vec, txt_ids


def _expand_multitask_choice(
    value: str | Sequence[str],
    batch_size: int,
    *,
    name: str,
    allowed: set[str],
) -> tuple[str, ...]:
    if isinstance(value, str):
        choices = (value,) * int(batch_size)
    else:
        choices = tuple(value)
        if len(choices) != int(batch_size):
            raise ValueError(
                f"The number of per-sample {name}s must match the batch "
                f"size: got {len(choices)} choices for {batch_size} samples"
            )
    choices = tuple(str(choice).strip().lower() for choice in choices)
    invalid = sorted(set(choices) - set(allowed))
    if invalid:
        expected = " or ".join(repr(choice) for choice in sorted(allowed))
        raise ValueError(f"{name} must be {expected}, got {invalid}")
    return choices


@torch.no_grad()
def _encode_multitask_maps(
    autoencoder,
    before: Tensor,
    after: Tensor,
    extra_images: tuple[Tensor, ...] = (),
    flow_source: str | Sequence[str] = "after",
    condition_mode: str | Sequence[str] = "before_after",
) -> tuple[Tensor, Tensor, tuple[Tensor, ...]]:
    """Encode only the condition images required by each batch sample."""

    if before.shape != after.shape:
        raise ValueError(
            "before and after must have the same shape, got "
            f"{tuple(before.shape)} and {tuple(after.shape)}"
        )
    batch_size = before.shape[0]
    for image in extra_images:
        if image.shape != before.shape:
            raise ValueError(
                "Every extra image must match the condition image shape, "
                f"got {tuple(image.shape)} versus {tuple(before.shape)}"
            )
    sources = _expand_multitask_choice(
        flow_source,
        batch_size,
        name="flow source",
        allowed={
            "after",
            "gaussian_noise",
            "latent_absolute_difference",
            "latent_difference",
        },
    )
    modes = _expand_multitask_choice(
        condition_mode,
        batch_size,
        name="condition mode",
        allowed={"single_after", "before_after"},
    )
    require_before = tuple(
        source in {"latent_absolute_difference", "latent_difference"}
        or mode == "before_after"
        for source, mode in zip(sources, modes)
    )

    if all(require_before):
        encoded = autoencoder.encode(
            torch.cat((before, after, *extra_images), dim=0).to(dtype=torch.bfloat16)
        )
        maps = encoded.split(batch_size, dim=0)
        return maps[0], maps[1], tuple(maps[2:])

    if not any(require_before):
        encoded = autoencoder.encode(
            torch.cat((after, *extra_images), dim=0).to(dtype=torch.bfloat16)
        )
        maps = encoded.split(batch_size, dim=0)
        after_map = maps[0]
        return (
            torch.zeros_like(after_map),
            after_map,
            tuple(maps[1:]),
        )

    before_indices = torch.tensor(
        [index for index, required in enumerate(require_before) if required],
        device=before.device,
        dtype=torch.long,
    )
    selected_before = before.index_select(0, before_indices)
    encoded = autoencoder.encode(
        torch.cat((after, *extra_images, selected_before), dim=0).to(
            dtype=torch.bfloat16
        )
    )
    split_sizes = (
        *([batch_size] * (1 + len(extra_images))),
        int(before_indices.numel()),
    )
    maps = encoded.split(split_sizes, dim=0)
    after_map = maps[0]
    before_map = torch.zeros_like(after_map)
    before_map.index_copy_(0, before_indices, maps[-1])
    return before_map, after_map, tuple(maps[1:-1])


@torch.no_grad()
def encode_multitask_inputs(
    autoencoder,
    before: Tensor,
    after: Tensor,
    target_rgb: Tensor,
    flow_source: str | Sequence[str] = "after",
    condition_mode: str | Sequence[str] = "before_after",
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Encode task conditions and the RGB target with no duplicate image."""

    before_map, after_map, extra_maps = _encode_multitask_maps(
        autoencoder,
        before,
        after,
        extra_images=(target_rgb,),
        flow_source=flow_source,
        condition_mode=condition_mode,
    )
    (target_map,) = extra_maps
    return (
        pack_latent(before_map),
        pack_latent(after_map),
        pack_latent(target_map),
        after_map,
    )


def multitask_flow_matching_output(
    transformer,
    before_tokens: Tensor,
    after_tokens: Tensor,
    mask_tokens: Tensor,
    image_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    beta_alpha: float = 2.5,
    flow_source: str | Sequence[str] = "after",
    latent_difference_scale: float = 1.0,
    condition_mode: str | Sequence[str] = "before_after",
    token_loss_weights: Tensor | None = None,
    fixed_timesteps: Tensor | None = None,
    return_per_sample_loss: bool = False,
) -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return flow loss, endpoint estimate, times, and optional sample MSE."""

    batch_size = after_tokens.shape[0]
    if fixed_timesteps is None:
        timesteps = torch.rand(batch_size, device=after_tokens.device).pow(
            1.0 / float(beta_alpha)
        )
    else:
        timesteps = fixed_timesteps.to(
            device=after_tokens.device,
            dtype=torch.float32,
        )
        if tuple(timesteps.shape) != (batch_size,):
            raise ValueError(
                "fixed_timesteps must have shape [B], got "
                f"{tuple(timesteps.shape)} for batch size {batch_size}"
            )
        if not bool(torch.isfinite(timesteps).all()):
            raise ValueError("fixed_timesteps must contain only finite values")
        if bool(((timesteps < 0.0) | (timesteps > 1.0)).any()):
            raise ValueError("fixed_timesteps must lie in the interval [0, 1]")
    interpolation_weight = timesteps[:, None, None]
    source_tokens = multitask_source_tokens(
        before_tokens,
        after_tokens,
        flow_source=flow_source,
        latent_difference_scale=latent_difference_scale,
    )
    condition_before, condition_after = multitask_condition_tokens(
        before_tokens,
        after_tokens,
        condition_mode=condition_mode,
    )
    state_t = (
        1.0 - interpolation_weight
    ) * mask_tokens + interpolation_weight * source_tokens
    target_velocity = source_tokens - mask_tokens
    predicted_velocity = transformer(
        img=torch.cat(
            (state_t, condition_before, condition_after),
            dim=-1,
        ),
        img_ids=image_ids,
        txt=txt,
        txt_ids=txt_ids,
        y=vec,
        timesteps=timesteps,
    )
    squared_error = (predicted_velocity.float() - target_velocity.float()).square()
    if token_loss_weights is None:
        per_sample_loss = squared_error.mean(dim=(1, 2))
        flow_loss = squared_error.mean()
    else:
        weights = token_loss_weights.float()
        if weights.ndim == 2:
            weights = weights.unsqueeze(-1)
        expected_shape = (*squared_error.shape[:2], 1)
        if tuple(weights.shape) != expected_shape:
            raise ValueError(
                "token_loss_weights must have shape [B,L] or [B,L,1], got "
                f"{tuple(token_loss_weights.shape)} for velocity "
                f"{tuple(squared_error.shape)}"
            )
        denominator = (weights.sum() * squared_error.shape[-1]).clamp_min(1.0)
        flow_loss = (squared_error * weights).sum() / denominator
        per_sample_denominator = (
            weights.sum(dim=(1, 2)) * squared_error.shape[-1]
        ).clamp_min(1.0)
        per_sample_loss = (squared_error * weights).sum(
            dim=(1, 2)
        ) / per_sample_denominator
    estimated_target = (
        state_t.float() - interpolation_weight.float() * predicted_velocity.float()
    )
    if return_per_sample_loss:
        return flow_loss, estimated_target, timesteps, per_sample_loss
    return flow_loss, estimated_target, timesteps


def pack_pixel_loss_mask(
    pixel_loss_mask: Tensor,
    latent_height: int,
    latent_width: int,
) -> Tensor:
    """Convert pixel validity to a conservative weight per packed token.

    A packed token is supervised only when every source pixel in its spatial
    cell is valid. This prevents any LoveDA ID-0 fill value from contributing
    to the flow objective.
    """

    if pixel_loss_mask.ndim != 4 or pixel_loss_mask.shape[1] != 1:
        raise ValueError(
            "pixel_loss_mask must have shape [B,1,H,W], got "
            f"{tuple(pixel_loss_mask.shape)}"
        )
    if latent_height % 2 or latent_width % 2:
        raise ValueError(
            "Latent height and width must be even before packing, got "
            f"{latent_height}x{latent_width}"
        )
    token_invalid = F.adaptive_max_pool2d(
        1.0 - pixel_loss_mask.float().clamp(0.0, 1.0),
        output_size=(latent_height // 2, latent_width // 2),
    )
    token_weights = 1.0 - token_invalid
    return token_weights.flatten(2).transpose(1, 2)


def multitask_source_tokens(
    before_tokens: Tensor,
    after_tokens: Tensor,
    flow_source: str | Sequence[str] = "after",
    latent_difference_scale: float = 1.0,
) -> Tensor:
    """Build each sample's t=1 endpoint without changing its conditions."""

    batch_size = int(after_tokens.shape[0])
    sources = _expand_multitask_choice(
        flow_source,
        batch_size,
        name="flow source",
        allowed={
            "after",
            "gaussian_noise",
            "latent_absolute_difference",
            "latent_difference",
        },
    )
    if all(source == "after" for source in sources):
        return after_tokens
    uses_difference = any(
        source in {"latent_absolute_difference", "latent_difference"}
        for source in sources
    )
    if uses_difference:
        scale = float(latent_difference_scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                "flow.latent_difference_scale must be finite and > 0, "
                f"got {latent_difference_scale!r}"
            )
        difference_tokens = (after_tokens - before_tokens) * scale
    else:
        difference_tokens = None
    if all(source == "latent_difference" for source in sources):
        assert difference_tokens is not None
        return difference_tokens
    if all(source == "latent_absolute_difference" for source in sources):
        assert difference_tokens is not None
        return difference_tokens.abs()
    if all(source == "gaussian_noise" for source in sources):
        return torch.randn_like(after_tokens)

    result = after_tokens
    if any(source == "latent_difference" for source in sources):
        assert difference_tokens is not None
        use_signed_difference = torch.tensor(
            [source == "latent_difference" for source in sources],
            device=after_tokens.device,
            dtype=torch.bool,
        ).view(batch_size, 1, 1)
        result = torch.where(
            use_signed_difference,
            difference_tokens,
            result,
        )
    if any(source == "latent_absolute_difference" for source in sources):
        assert difference_tokens is not None
        use_absolute_difference = torch.tensor(
            [source == "latent_absolute_difference" for source in sources],
            device=after_tokens.device,
            dtype=torch.bool,
        ).view(batch_size, 1, 1)
        result = torch.where(
            use_absolute_difference,
            difference_tokens.abs(),
            result,
        )
    if any(source == "gaussian_noise" for source in sources):
        use_noise = torch.tensor(
            [source == "gaussian_noise" for source in sources],
            device=after_tokens.device,
            dtype=torch.bool,
        ).view(batch_size, 1, 1)
        result = torch.where(
            use_noise,
            torch.randn_like(after_tokens),
            result,
        )
    return result


def multitask_condition_tokens(
    before_tokens: Tensor,
    after_tokens: Tensor,
    condition_mode: str | Sequence[str] = "before_after",
) -> tuple[Tensor, Tensor]:
    """Build fixed condition slots without duplicating single-image inputs.

    The returned slots always mean ``[before, after/current]``. A
    single-image task receives ``[zero, image]``; a bi-temporal task receives
    ``[before, after]``. The null slot is created after VAE encoding and is
    therefore an exact zero tensor, not the latent of a black RGB image.
    """

    batch_size = int(after_tokens.shape[0])
    modes = _expand_multitask_choice(
        condition_mode,
        batch_size,
        name="condition mode",
        allowed={"single_after", "before_after"},
    )
    if all(mode == "before_after" for mode in modes):
        return before_tokens, after_tokens
    if all(mode == "single_after" for mode in modes):
        return torch.zeros_like(before_tokens), after_tokens
    use_before = torch.tensor(
        [mode == "before_after" for mode in modes],
        device=before_tokens.device,
        dtype=torch.bool,
    ).view(batch_size, 1, 1)
    condition_before = torch.where(
        use_before,
        before_tokens,
        torch.zeros_like(before_tokens),
    )
    return condition_before, after_tokens


def task_flow_sources(
    tasks: Sequence[str],
    flow_config: Mapping,
) -> tuple[str, ...]:
    """Resolve a flow endpoint for each task sample in a mixed batch."""

    default_source = str(flow_config.get("source", "after"))
    source_by_task = flow_config.get("source_by_task", {})
    if not isinstance(source_by_task, Mapping):
        raise TypeError("flow.source_by_task must be a mapping")
    return tuple(str(source_by_task.get(str(task), default_source)) for task in tasks)


def task_condition_modes(
    tasks: Sequence[str],
    flow_config: Mapping,
) -> tuple[str, ...]:
    """Resolve single-image or bi-temporal condition slots per sample."""

    default_mode = str(flow_config.get("condition_mode", "before_after"))
    mode_by_task = flow_config.get("condition_by_task", {})
    if not isinstance(mode_by_task, Mapping):
        raise TypeError("flow.condition_by_task must be a mapping")
    return tuple(str(mode_by_task.get(str(task), default_mode)) for task in tasks)


@torch.no_grad()
def predict_multitask_decoded(
    transformer,
    autoencoder,
    before: Tensor,
    after: Tensor,
    txt: Tensor,
    vec: Tensor,
    num_steps: int,
    resolution: int,
    beta_alpha: float = 2.5,
    beta_beta: float = 1.0,
    flow_source: str | Sequence[str] = "after",
    latent_difference_scale: float = 1.0,
    condition_mode: str | Sequence[str] = "before_after",
) -> Tensor:
    """Generate a normalized RGB task target using its text condition."""

    batch_size = before.shape[0]
    before_map, after_map, _ = _encode_multitask_maps(
        autoencoder,
        before,
        after,
        flow_source=flow_source,
        condition_mode=condition_mode,
    )
    before_tokens = pack_latent(before_map)
    after_tokens = pack_latent(after_map)
    image_ids = make_image_ids(
        batch_size,
        after_map.shape[-2],
        after_map.shape[-1],
        after.device,
    )
    txt_ids = torch.zeros(
        batch_size,
        txt.shape[1],
        3,
        device=after.device,
        dtype=torch.float32,
    )

    state = multitask_source_tokens(
        before_tokens,
        after_tokens,
        flow_source=flow_source,
        latent_difference_scale=latent_difference_scale,
    )
    condition_before, condition_after = multitask_condition_tokens(
        before_tokens,
        after_tokens,
        condition_mode=condition_mode,
    )
    times = beta_scheduler(
        int(num_steps),
        alpha=float(beta_alpha),
        beta=float(beta_beta),
    )
    for current_time, next_time in zip(times[:-1], times[1:]):
        timestep = torch.full(
            (batch_size,),
            current_time,
            device=after.device,
            dtype=state.dtype,
        )
        velocity = transformer(
            img=torch.cat(
                (state, condition_before, condition_after),
                dim=-1,
            ),
            img_ids=image_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=timestep,
        )
        state = state + (next_time - current_time) * velocity

    target_latent_map = unpack_latent(state.float(), resolution, resolution)
    return autoencoder.decode(target_latent_map.to(dtype=torch.bfloat16)).float()


def sliding_window_starts(
    length: int,
    tile_size: int,
    stride: int,
) -> tuple[int, ...]:
    """Return starts that cover an axis and always include its far edge."""

    length = int(length)
    tile_size = int(tile_size)
    stride = int(stride)
    if length < 1 or tile_size < 1 or stride < 1:
        raise ValueError("length, tile_size, and stride must be positive")
    if tile_size > length:
        raise ValueError(f"tile_size {tile_size} exceeds axis length {length}")
    if tile_size == length:
        return (0,)
    starts = list(range(0, length - tile_size + 1, stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return tuple(starts)


@torch.no_grad()
def predict_multitask_palette_sliding(
    transformer,
    autoencoder,
    before: Tensor,
    after: Tensor,
    txt: Tensor,
    vec: Tensor,
    palette: Tensor | np.ndarray,
    num_steps: int,
    tile_size: int,
    stride: int,
    palette_temperature: float = 0.1,
    beta_alpha: float = 2.5,
    beta_beta: float = 1.0,
    flow_source: str | Sequence[str] = "after",
    latent_difference_scale: float = 1.0,
    condition_mode: str | Sequence[str] = "before_after",
) -> tuple[Tensor, Tensor]:
    """Predict a full image by blending overlapping palette probabilities.

    The model still sees its native ``tile_size`` input. Overlaps are fused
    before argmax, so seams are not baked into hard per-tile class maps.
    The first return value is the blended decoded RGB image for inspection;
    the second is a full-resolution soft class-probability tensor.
    """

    if before.shape != after.shape or before.ndim != 4:
        raise ValueError(
            "before and after must be matching BCHW tensors, got "
            f"{tuple(before.shape)} and {tuple(after.shape)}"
        )
    tile_size = int(tile_size)
    stride = int(stride)
    temperature = float(palette_temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("palette_temperature must be finite and positive")
    batch_size, _, height, width = after.shape
    row_starts = sliding_window_starts(height, tile_size, stride)
    column_starts = sliding_window_starts(width, tile_size, stride)
    colors = torch.as_tensor(
        palette,
        device=after.device,
        dtype=torch.float32,
    )
    if colors.ndim != 2 or colors.shape[1] != 3 or colors.shape[0] < 2:
        raise ValueError(
            "palette must have shape [num_classes, 3], got " f"{tuple(colors.shape)}"
        )
    colors = colors.div(127.5).sub(1.0)

    axis_weight = torch.hann_window(
        tile_size,
        periodic=False,
        device=after.device,
        dtype=torch.float32,
    ).clamp_min(1.0e-3)
    tile_weight = axis_weight[:, None] * axis_weight[None, :]
    tile_weight = tile_weight[None, None]
    decoded_sum = torch.zeros(
        batch_size,
        3,
        height,
        width,
        device=after.device,
        dtype=torch.float32,
    )
    probability_sum = torch.zeros(
        batch_size,
        colors.shape[0],
        height,
        width,
        device=after.device,
        dtype=torch.float32,
    )
    weight_sum = torch.zeros(
        batch_size,
        1,
        height,
        width,
        device=after.device,
        dtype=torch.float32,
    )

    def sample_choice(
        value: str | Sequence[str],
        index: int,
    ) -> str:
        return value if isinstance(value, str) else str(value[index])

    for sample_index in range(batch_size):
        for top in row_starts:
            for left in column_starts:
                rows = slice(top, top + tile_size)
                columns = slice(left, left + tile_size)
                decoded_tile = predict_multitask_decoded(
                    transformer=transformer,
                    autoencoder=autoencoder,
                    before=before[
                        sample_index : sample_index + 1,
                        :,
                        rows,
                        columns,
                    ],
                    after=after[
                        sample_index : sample_index + 1,
                        :,
                        rows,
                        columns,
                    ],
                    txt=txt[sample_index : sample_index + 1],
                    vec=vec[sample_index : sample_index + 1],
                    num_steps=num_steps,
                    resolution=tile_size,
                    beta_alpha=beta_alpha,
                    beta_beta=beta_beta,
                    flow_source=sample_choice(flow_source, sample_index),
                    latent_difference_scale=latent_difference_scale,
                    condition_mode=sample_choice(
                        condition_mode,
                        sample_index,
                    ),
                ).float()
                if decoded_tile.shape[-2:] != (tile_size, tile_size):
                    raise RuntimeError(
                        "Decoded sliding-window tile has unexpected size "
                        f"{tuple(decoded_tile.shape[-2:])}; expected "
                        f"{(tile_size, tile_size)}"
                    )
                pixels = decoded_tile.permute(0, 2, 3, 1).unsqueeze(-2)
                squared_distance = (
                    (pixels - colors.view(1, 1, 1, colors.shape[0], 3))
                    .square()
                    .sum(dim=-1)
                )
                probabilities = (
                    (-squared_distance / temperature).permute(0, 3, 1, 2).softmax(dim=1)
                )
                decoded_sum[
                    sample_index : sample_index + 1,
                    :,
                    rows,
                    columns,
                ] += (
                    decoded_tile * tile_weight
                )
                probability_sum[
                    sample_index : sample_index + 1,
                    :,
                    rows,
                    columns,
                ] += (
                    probabilities * tile_weight
                )
                weight_sum[
                    sample_index : sample_index + 1,
                    :,
                    rows,
                    columns,
                ] += tile_weight

    normalizer = weight_sum.clamp_min(1.0e-8)
    return decoded_sum / normalizer, probability_sum / normalizer


@torch.no_grad()
def predict_multitask_probability(
    transformer,
    autoencoder,
    before: Tensor,
    after: Tensor,
    txt: Tensor,
    vec: Tensor,
    num_steps: int,
    resolution: int,
    beta_alpha: float = 2.5,
    beta_beta: float = 1.0,
    flow_source: str | Sequence[str] = "after",
    latent_difference_scale: float = 1.0,
    condition_mode: str | Sequence[str] = "before_after",
) -> Tensor:
    """Backward-compatible binary probability prediction."""

    decoded = predict_multitask_decoded(
        transformer,
        autoencoder,
        before,
        after,
        txt,
        vec,
        num_steps=num_steps,
        resolution=resolution,
        beta_alpha=beta_alpha,
        beta_beta=beta_beta,
        flow_source=flow_source,
        condition_mode=condition_mode,
        latent_difference_scale=latent_difference_scale,
    )
    return decoded_rgb_to_probability(decoded).float()
