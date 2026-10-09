"""Shared Qwen model selection and fixed-prompt cache identity."""

import hashlib
import json
from pathlib import Path

from uniflow_rs._compat.backbone import text_backend
from uniflow_rs._compat.flux2_compat import FLUX2_QWEN3_4B_FP8_MODEL


def resolve_text_model(cfg: dict, text_model_dir: str | Path | None = None) -> str:
    """CLI/local override takes precedence over the configured Qwen asset."""
    pretrained = cfg.get("pretrained", {})
    obsolete = [
        key for key in ("t5_model_path", "clip_model_path", "t5_model_id", "clip_model_id")
        if pretrained.get(key) is not None
    ]
    if obsolete:
        raise ValueError(
            "Unsupported text-encoder settings: " + ", ".join(obsolete)
            + ". Use pretrained.qwen_model_path or --text-model-dir."
        )
    return str(
        text_model_dir
        or pretrained.get("text_model_dir")
        or pretrained.get("qwen_model_path")
        or pretrained.get("qwen_model_id")
        or FLUX2_QWEN3_4B_FP8_MODEL
    )


def prompt_cache_path(cfg: dict, text_model_dir: str | Path | None = None) -> Path:
    """Hash the effective encoder as well as prompts and sequence length.

    v2 deliberately does not reuse caches created when the CLI override was
    ignored. This is a cache-key change, not a text-conditioning change.
    """
    identity = json.dumps(
        {
            "cache_version": 2,
            "prompts": cfg["model"]["prompts"],
            "text_backend": text_backend(cfg),
            "qwen": resolve_text_model(cfg, text_model_dir),
            "sequence_length": int(cfg["model"].get("text_sequence_length", 512)),
        },
        sort_keys=True,
    )
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:12]
    return Path(cfg["output"]["root"]) / "_prompt_cache" / f"multitask_text_{digest}.pt"
