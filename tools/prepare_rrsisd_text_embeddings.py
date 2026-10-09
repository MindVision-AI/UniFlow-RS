"""Precompute frozen Qwen3 features for RRSIS-D or LaSeRS expressions."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

from uniflow_rs._compat.backbone import text_backend
from uniflow_rs._compat.flux2_compat import (
    FLUX2_QWEN3_4B_FP8_MODEL,
    load_qwen3_embedder,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/uniflow_rs_512_2step_eval.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help=(
            "Referring source name when data.referring.sources is used, "
            "for example rrsisd or lasers."
        ),
    )
    parser.add_argument("--qwen-model-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def _ordered_prompts(manifest: dict) -> list[str]:
    required_manifest_keys = {"splits", "prompt_count", "prompt_sha256"}
    missing_manifest_keys = required_manifest_keys.difference(manifest)
    if missing_manifest_keys:
        raise ValueError(
            "Incomplete referring manifest; missing keys: "
            f"{sorted(missing_manifest_keys)!r}"
        )
    # LaSeRS benchmark views overlap. The same prompt_index can therefore be
    # present in ``test`` and several test_* splits; encode it exactly once.
    by_index: dict[int, str] = {}
    for split, samples in manifest["splits"].items():
        for sample in samples:
            index = int(sample["prompt_index"])
            expression = str(sample["expression"])
            previous = by_index.setdefault(index, expression)
            if previous != expression:
                raise ValueError(
                    f"prompt_index {index} has conflicting expressions "
                    f"while reading split '{split}'"
                )
    indices = sorted(by_index)
    prompt_count = int(manifest["prompt_count"])
    if indices != list(range(prompt_count)):
        raise ValueError(
            "Referring prompt_index values must be contiguous from zero. "
            "Do not build text features from a --max-samples diagnostic set."
        )
    prompts = [by_index[index] for index in indices]
    prompt_hasher = hashlib.sha256()
    for index, expression in enumerate(prompts):
        prompt_hasher.update(f"{index}\t{expression}\n".encode("utf-8"))
    if prompt_hasher.hexdigest() != str(manifest["prompt_sha256"]):
        raise ValueError(
            "Referring manifest prompt_sha256 does not match its expressions"
        )
    return prompts


def _open_partial(
    path: Path,
    shape: tuple[int, ...],
    overwrite: bool,
) -> np.memmap:
    if path.is_file() and not overwrite:
        array = np.load(path, mmap_mode="r+")
        if tuple(array.shape) != shape or array.dtype != np.float16:
            raise ValueError(
                f"Partial cache shape/dtype mismatch for '{path}': "
                f"{array.shape} {array.dtype}, expected {shape} float16"
            )
        return array
    return np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float16,
        shape=shape,
    )


def _prepare_qwen_cache(
    args: argparse.Namespace,
    cfg: dict,
    prompts: list[str],
    manifest: dict,
    manifest_path: Path,
    output_dir: Path,
    sequence_length: int,
    device: torch.device,
) -> None:
    """Write the 7680-D Qwen3 features expected by Klein Base 4B."""

    pretrained = cfg.get("pretrained", {})
    model_spec = (
        args.qwen_model_path
        or pretrained.get("qwen_model_path")
        or pretrained.get("qwen_model_id")
        or FLUX2_QWEN3_4B_FP8_MODEL
    )
    if isinstance(model_spec, Path):
        model_spec = model_spec.expanduser().resolve()
        if not model_spec.is_dir():
            raise FileNotFoundError(model_spec)
    model_spec = str(model_spec)
    progress_path = output_dir / "progress.json"
    progress = {
        "backend": "qwen3",
        "prompt_sha256": str(manifest["prompt_sha256"]),
        "count": len(prompts),
        "sequence_length": sequence_length,
        "text_completed": 0,
    }
    if progress_path.is_file() and not args.overwrite:
        with progress_path.open("r", encoding="utf-8") as file:
            saved = json.load(file)
        expected = (
            "qwen3",
            progress["prompt_sha256"],
            progress["count"],
            progress["sequence_length"],
        )
        actual = (
            saved.get("backend"),
            saved.get("prompt_sha256"),
            int(saved.get("count", -1)),
            int(saved.get("sequence_length", -1)),
        )
        if actual != expected:
            raise RuntimeError(
                "Partial Qwen cache belongs to different settings; rerun "
                "with --overwrite."
            )
        progress.update(saved)

    partial = output_dir / "qwen3.float16.partial.npy"
    text_array = _open_partial(
        partial,
        (len(prompts), sequence_length, 7680),
        overwrite=args.overwrite,
    )
    qwen = load_qwen3_embedder(
        model_spec,
        max_length=sequence_length,
        device=device,
    ).eval().requires_grad_(False)
    start = 0 if args.overwrite else int(progress["text_completed"])
    for offset in tqdm(
        range(start, len(prompts), args.batch_size),
        desc="Encoding referring prompts with Qwen3",
    ):
        texts = prompts[offset : offset + args.batch_size]
        with torch.inference_mode():
            encoded = qwen(texts)
        values = encoded.float().cpu().numpy().astype(np.float16)
        text_array[offset : offset + len(texts)] = values
        text_array.flush()
        progress["text_completed"] = offset + len(texts)
        _atomic_json(progress, progress_path)
    del qwen, text_array
    gc.collect()
    torch.cuda.empty_cache()

    final = output_dir / "qwen3.float16.npy"
    os.replace(partial, final)
    metadata = {
        "format": "referring_text_embeddings_v2",
        "text_backend": "qwen3",
        "dataset_name": str(manifest.get("dataset_name", "RRSIS-D")),
        "count": len(prompts),
        "sequence_length": sequence_length,
        "text_hidden_size": 7680,
        "dtype": "float16",
        "prompt_sha256": str(manifest["prompt_sha256"]),
        "manifest": str(manifest_path),
        "text_model": model_spec,
        "text_file": final.name,
    }
    _atomic_json(metadata, output_dir / "metadata.json")
    progress_path.unlink(missing_ok=True)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"Saved: {output_dir}")


def main() -> None:
    args = _parse_args()
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")
    with args.config.open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    referring_cfg = dict(cfg["data"]["referring"])
    configured_sources = referring_cfg.get("sources")
    if configured_sources is not None:
        if args.source is None:
            raise ValueError(
                "--source is required for this multi-source config; choose "
                f"one of {sorted(configured_sources)}"
            )
        if args.source not in configured_sources:
            raise KeyError(
                f"Unknown referring source {args.source!r}; expected one of "
                f"{sorted(configured_sources)}"
            )
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
        defaults.update(dict(configured_sources[args.source]))
        referring_cfg = defaults
    elif args.source is not None:
        raise ValueError(
            "--source was provided but this config has one referring source"
        )
    root = Path(referring_cfg["root"]).expanduser().resolve()
    manifest_path = root / str(referring_cfg.get("manifest", "splits.json"))
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    prompts = _ordered_prompts(manifest)
    sequence_length = int(
        args.sequence_length
        if args.sequence_length is not None
        else cfg["model"].get("text_sequence_length", 64)
    )
    if not 1 <= sequence_length <= 512:
        raise ValueError("--sequence-length must be in [1, 512]")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / str(
            referring_cfg.get("prompt_embeddings", "text_embeddings")
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    backend = text_backend(cfg)
    if metadata_path.is_file() and not args.overwrite:
        with metadata_path.open("r", encoding="utf-8") as file:
            existing = json.load(file)
        expected = (
            len(prompts),
            sequence_length,
            str(manifest["prompt_sha256"]),
        )
        actual = (
            int(existing["count"]),
            int(existing["sequence_length"]),
            str(existing["prompt_sha256"]),
        )
        expected_format = "referring_text_embeddings_v2"
        if actual == expected and existing.get("format") == expected_format:
            print(f"Referring text cache is already complete: {output_dir}")
            return
        raise RuntimeError(
            f"Existing text cache metadata {actual} != expected {expected}; "
            "rerun with --overwrite."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError(
            "Referring text-cache preparation requires one CUDA GPU."
        )

    if backend != "qwen3":
        raise RuntimeError(
            "This source-isolated FLUX.2 release supports only Qwen3 text "
            f"conditioning; resolved backend={backend!r}."
        )
    _prepare_qwen_cache(
        args,
        cfg,
        prompts,
        manifest,
        manifest_path,
        output_dir,
        sequence_length,
        device,
    )


if __name__ == "__main__":
    main()
