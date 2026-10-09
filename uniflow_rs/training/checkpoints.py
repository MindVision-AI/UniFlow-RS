"""Run directories, resume resolution, and compatible adapter saving."""

from __future__ import annotations
import argparse
from accelerate import Accelerator
from uniflow_rs._compat.lora import LoRAConfig
from datetime import datetime
from pathlib import Path
from accelerate.utils import broadcast_object_list
from uniflow_rs._compat.lora import save_lora_checkpoint


def make_run_dir(accelerator: Accelerator, cfg: dict) -> Path:
    if accelerator.is_main_process:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        value = str(Path(cfg["output"]["root"]) / cfg["output"]["run_name"] / timestamp)
    else:
        value = None
    values = [value]
    broadcast_object_list(values)
    run_dir = Path(values[0])
    run_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    return run_dir


def latest_checkpoint(run_root: Path) -> Path:
    checkpoints = sorted(
        run_root.glob("*/checkpoints/checkpoint_epoch_*.pt"),
        key=lambda path: path.stat().st_mtime,
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found below '{run_root}'")
    return checkpoints[-1]


def resolve_resume(args: argparse.Namespace, cfg: dict) -> Path | None:
    value = args.resume
    if value is None:
        value = cfg["training"].get("resume")
    if value in (None, "", False):
        return None
    if str(value).lower() == "latest":
        root = Path(cfg["output"]["root"]) / cfg["output"]["run_name"]
        return latest_checkpoint(root)
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def prune_checkpoints(directory: Path, keep: int) -> None:
    checkpoints = sorted(
        directory.glob("checkpoint_epoch_*.pt"),
        key=lambda path: path.stat().st_mtime,
    )
    for path in checkpoints[: -max(int(keep), 1)]:
        path.unlink()


def _save_adapter(
    path: Path,
    accelerator: Accelerator,
    transformer,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    lora_cfg: LoRAConfig,
    training_state: dict | None,
    *,
    base_model: str,
    model_variant: str,
) -> None:
    save_lora_checkpoint(
        path,
        accelerator.unwrap_model(transformer),
        optimizer,
        scheduler,
        epoch,
        global_step,
        -1.0,  # Legacy serialization slot; no score-based selection.
        lora_cfg,
        None,
        base_model=base_model,
        model_variant=model_variant,
        training_state=training_state,
    )
