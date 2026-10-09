"""Training data loaders and frozen text-condition caches."""

from __future__ import annotations
from uniflow_rs.text_conditions import load_prompt_conditions as load_prompt_conditions
from uniflow_rs.text_config import prompt_cache_path as prompt_cache_path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from uniflow_rs._compat.multitask import (
    REFERRING_TASK,
    PairPreservingBatchSampler,
    WeightedMultitaskDataset,
    multitask_train_collate,
)
import logging

LOGGER = logging.getLogger("uniflow_rs.train")

__all__ = [
    "build_dataloaders",
    "load_prompt_conditions",
    "prompt_cache_path",
    "required_fixed_prompt_keys",
    "worker_init_fn",
]


def required_fixed_prompt_keys(
    cfg: dict,
    active_tasks: tuple[str, ...],
) -> set[str]:
    """Resolve task, source, and modality-specific fixed prompt keys."""

    required: set[str] = set()
    reserved = {
        "sources",
        "source_sampling_weights",
        "samples_per_epoch",
    }
    for task in active_tasks:
        if task == REFERRING_TASK:
            continue
        task_cfg = dict(cfg["data"].get(task, {}))
        sources = task_cfg.get("sources")
        if sources is None:
            source_configs = (task_cfg,)
        else:
            defaults = {
                key: value for key, value in task_cfg.items() if key not in reserved
            }
            source_configs = tuple(
                {**defaults, **dict(source_cfg)} for source_cfg in sources.values()
            )
        task_keys = {
            str(source_cfg[field])
            for source_cfg in source_configs
            for field in ("prompt_key", "rgb_prompt_key", "sar_prompt_key")
            if source_cfg.get(field) is not None
        }
        required.update(task_keys or {task})
    return required


def worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _data_loader(
    dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    drop_last: bool,
    collate_fn=None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=True,
        drop_last=bool(drop_last),
        persistent_workers=int(num_workers) > 0,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn,
    )


def build_dataloaders(
    cfg: dict,
    active_tasks: tuple[str, ...],
) -> tuple[DataLoader, dict[str, int]]:
    data_cfg = cfg["data"]
    training_cfg = cfg["training"]
    from uniflow_rs.training_splits import build_training_task_datasets

    train_tasks, _, _ = build_training_task_datasets(
        data_cfg,
        active_tasks,
        augment=bool(data_cfg.get("augment", True)),
        validation_enabled=False,
    )
    if len(active_tasks) == 1:
        train_dataset = train_tasks[active_tasks[0]]
    else:
        configured_weights = training_cfg.get("task_sampling_weights")
        weights = (
            {task: int(configured_weights[task]) for task in active_tasks}
            if configured_weights is not None
            else {task: 1 for task in active_tasks}
        )
        train_dataset = WeightedMultitaskDataset(
            train_tasks,
            task_weights=weights,
        )
    default_batch_size = int(training_cfg["batch_size_per_gpu"])
    configured_batch_sizes = training_cfg.get(
        "batch_size_per_gpu_by_task",
        {},
    )
    if not isinstance(configured_batch_sizes, dict):
        raise TypeError("training.batch_size_per_gpu_by_task must be a mapping")
    unknown_batch_tasks = sorted(set(configured_batch_sizes) - set(active_tasks))
    if unknown_batch_tasks:
        raise ValueError(
            "Per-task batch sizes contain inactive tasks: " f"{unknown_batch_tasks}"
        )
    batch_sizes_by_task = {
        task: int(configured_batch_sizes.get(task, default_batch_size))
        for task in active_tasks
    }
    invalid_batch_sizes = {
        task: value for task, value in batch_sizes_by_task.items() if value < 1
    }
    if invalid_batch_sizes:
        raise ValueError(
            "Every per-task batch size must be positive: " f"{invalid_batch_sizes}"
        )
    task_resolutions = {
        task: int(
            data_cfg.get(task, {}).get(
                "resolution",
                data_cfg.get("resolution", 512),
            )
        )
        for task in active_tasks
    }
    uses_multiple_resolutions = len(set(task_resolutions.values())) > 1
    task_homogeneous_batches = (
        bool(training_cfg.get("pair_modalities_in_batch", False))
        or bool(training_cfg.get("task_homogeneous_batches", False))
        or uses_multiple_resolutions
    )
    if task_homogeneous_batches and len(active_tasks) > 1:
        batch_sampler = PairPreservingBatchSampler(
            train_dataset,
            batch_size=default_batch_size,
            drop_last=True,
            seed=int(training_cfg.get("seed", 0)),
            batch_size_by_task=batch_sizes_by_task,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=int(training_cfg["num_workers"]),
            pin_memory=True,
            persistent_workers=int(training_cfg["num_workers"]) > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=multitask_train_collate,
        )
    else:
        batch_size = (
            batch_sizes_by_task[active_tasks[0]]
            if len(active_tasks) == 1
            else default_batch_size
        )
        train_loader = _data_loader(
            train_dataset,
            batch_size=batch_size,
            num_workers=int(training_cfg["num_workers"]),
            shuffle=True,
            drop_last=True,
            collate_fn=multitask_train_collate,
        )
    sizes = {
        **{f"train_{task}": len(train_tasks[task]) for task in active_tasks},
        "effective_train": len(train_dataset),
    }
    return train_loader, sizes
