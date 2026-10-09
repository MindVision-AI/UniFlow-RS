"""Frozen fixed-prompt encoding and per-expression cache loading."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from accelerate import Accelerator

from uniflow_rs._compat.backbone import text_backend
from uniflow_rs.text_config import prompt_cache_path, resolve_text_model
from uniflow_rs._compat.deventer_flow import (
    CompositeDiskPromptEmbeddingCache,
    DiskPromptEmbeddingCache,
    encode_prompt_bank,
)
from uniflow_rs._compat.multitask import (
    REFERRING_TASK,
    active_tasks_from_config,
)

LOGGER = logging.getLogger("uniflow_rs.text_conditions")


def load_prompt_conditions(
    accelerator: Accelerator,
    cfg: dict,
    text_model_dir: Path | None = None,
) -> dict[
    str,
    tuple[torch.Tensor, torch.Tensor] | DiskPromptEmbeddingCache,
]:
    model_spec = resolve_text_model(cfg, text_model_dir)
    path = prompt_cache_path(cfg, text_model_dir)
    active_tasks = active_tasks_from_config(cfg)
    # Encode every configured fixed prompt. A shared task can select a
    # modality-specific prompt_key per data source while retaining the same
    # losses, output contract, MMDiT and LoRA weights.
    fixed_prompts = {
        str(task): str(prompt)
        for task, prompt in cfg["model"]["prompts"].items()
        if str(task) != REFERRING_TASK
    }
    if fixed_prompts and accelerator.is_main_process and not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Encoding fixed task prompts once: %s", fixed_prompts)
        prompt_bank = encode_prompt_bank(
            fixed_prompts,
            accelerator.device,
            sequence_length=int(cfg["model"].get("text_sequence_length", 512)),
            text_backend=text_backend(cfg),
            qwen_model_path=model_spec,
        )
        torch.save(prompt_bank, path)
    accelerator.wait_for_everyone()
    prompt_bank = (
        torch.load(path, map_location="cpu", weights_only=False)
        if fixed_prompts
        else {}
    )
    if REFERRING_TASK in active_tasks:
        referring_cfg = cfg["data"].get(REFERRING_TASK)
        if referring_cfg is None:
            raise KeyError("Active referring task requires data.referring")
        expected_length = int(cfg["model"].get("text_sequence_length", 512))
        configured_sources = referring_cfg.get("sources")
        if configured_sources is None:
            source_configs = {"referring": dict(referring_cfg)}
        else:
            if not isinstance(configured_sources, dict) or not configured_sources:
                raise ValueError("data.referring.sources must be a non-empty mapping")
            defaults = {
                key: value
                for key, value in referring_cfg.items()
                if key
                not in {
                    "sources",
                    "source_sampling_weights",
                    "samples_per_epoch",
                }
            }
            source_configs = {}
            for source_name, source_cfg in configured_sources.items():
                merged = dict(defaults)
                merged.update(dict(source_cfg))
                source_configs[str(source_name)] = merged

        referring_caches: list[DiskPromptEmbeddingCache] = []
        for source_name, source_cfg in source_configs.items():
            root = Path(source_cfg["root"]).expanduser()
            embeddings = root / str(
                source_cfg.get("prompt_embeddings", "text_embeddings")
            )
            referring_cache = DiskPromptEmbeddingCache(embeddings)
            manifest_path = root / str(source_cfg.get("manifest", "splits.json"))
            with manifest_path.open("r", encoding="utf-8") as file:
                referring_manifest = json.load(file)
            expected_identity = (
                int(referring_manifest["prompt_count"]),
                str(referring_manifest["prompt_sha256"]),
            )
            cache_identity = (
                referring_cache.count,
                str(referring_cache.metadata["prompt_sha256"]),
            )
            if cache_identity != expected_identity:
                raise ValueError(
                    f"Referring cache {source_name!r} does not match its "
                    f"manifest: cache={cache_identity}, "
                    f"manifest={expected_identity}"
                )
            if referring_cache.sequence_length != expected_length:
                raise ValueError(
                    f"Referring cache {source_name!r} sequence length "
                    f"{referring_cache.sequence_length} != model "
                    f"text_sequence_length {expected_length}. Regenerate "
                    "that cache with prepare_rrsisd_text_embeddings.py "
                    f"--sequence-length {expected_length} --overwrite."
                )
            referring_caches.append(referring_cache)
            LOGGER.info(
                "Loaded referring prompt cache %s: %d prompts",
                source_name,
                referring_cache.count,
            )
        prompt_bank[REFERRING_TASK] = (
            referring_caches[0]
            if len(referring_caches) == 1
            else CompositeDiskPromptEmbeddingCache(referring_caches)
        )
    return prompt_bank
