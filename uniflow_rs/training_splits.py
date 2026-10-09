"""Fixed-budget training datasets and partition safety checks.

Never hold out optimization samples or fall back to the benchmark test split.
"""

from __future__ import annotations

import json
from pathlib import Path


def group_key(sample):
    if isinstance(sample, str):
        return sample
    if isinstance(sample, tuple):  # COCO image record plus crop coordinates
        sample = sample[0]
    if not isinstance(sample, dict):
        raise TypeError("No supported physical-image metadata for split checks")
    if "parent" in sample:
        return str(sample["parent"])
    for field in ("image", "file_name", "before_image"):
        if field in sample:
            return str(sample.get("domain", "")) + "/" + Path(sample[field]).name
    if "name" in sample:
        return str(sample["name"])
    raise ValueError("Split checks need parent, image, file_name, or name metadata")


def source_configs(task_cfg):
    if "sources" not in task_cfg:
        return {None: task_cfg}
    defaults = {
        k: v
        for k, v in task_cfg.items()
        if k not in ("sources", "source_sampling_weights", "samples_per_epoch")
    }
    return {name: {**defaults, **cfg} for name, cfg in task_cfg["sources"].items()}


def check_config(task_cfg):
    """Reject validation, test-as-training, and manifest partition overlap."""
    for source, cfg in source_configs(task_cfg).items():
        policy = cfg.get("validation_protocol", {})
        mode = policy.get("mode")
        train = cfg.get("train_splits", [cfg.get("train_split", "train")])
        train = [train] if isinstance(train, str) else train
        train = {str(v).casefold() for v in train}
        test = str(cfg.get("test_split", "test")).casefold()
        values = cfg.get("test_splits", [test])
        values = [values] if isinstance(values, str) else values
        if {str(v).casefold() for v in values} != {test}:
            raise ValueError(f"{source}: test_splits contradicts test_split")
        if test in train:
            raise ValueError(f"{source}: evaluation partition is in train_splits")
        if mode != "disabled":
            raise ValueError(
                f"{source}: fixed-budget training requires disabled validation"
            )
        if cfg.get("val_split") is not None or cfg.get("val_splits") is not None:
            raise ValueError(f"{source}: disabled validation requires val_split: null")
        # Different split names are insufficient if a prepared manifest aliases
        # the same physical images into both training and test.
        if "manifest" in cfg:
            path = Path(cfg["root"]) / cfg["manifest"]
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = json.loads(path.read_text())
            splits = payload.get("splits", payload)
            split_names = {str(name).casefold(): name for name in splits}
            train_rows = [row for name in train for row in splits[split_names[name]]]
            test_rows = splits[split_names[test]]
            if {group_key(v) for v in train_rows} & {group_key(v) for v in test_rows}:
                raise ValueError(f"{source}: manifest training overlaps test")


def leaves(dataset, source=None):
    if hasattr(dataset, "source_name"):
        source = dataset.source_name
    if hasattr(dataset, "dataset"):
        yield from leaves(dataset.dataset, source)
    elif hasattr(dataset, "datasets"):
        children = dataset.datasets
        for child in (children.values() if isinstance(children, dict) else children):
            yield from leaves(child, source)
    elif hasattr(dataset, "samples"):
        yield source, dataset
    else:
        raise TypeError(f"Unsupported training dataset: {type(dataset).__name__}")


def partition_task(train_dataset, val_dataset, task_cfg):
    """Count training groups without constructing or reserving validation data."""
    if val_dataset is not None:
        raise ValueError("Fixed-budget training does not accept a validation reader")
    policies = source_configs(task_cfg)
    train_leaves = list(leaves(train_dataset))
    report = {}
    for source, cfg in policies.items():
        tr = [d for key, d in train_leaves if key == source]
        if not tr:
            raise ValueError(f"Missing training source: {source}")
        tg = {group_key(s) for d in tr for s in d.samples}
        policy = cfg["validation_protocol"]
        if policy["mode"] != "disabled":
            raise ValueError(
                f"{source}: fixed-budget training requires disabled validation"
            )
        report[source or "default"] = {
            **policy,
            "training_groups": len(tg),
            "validation_groups": 0,
            "validation_group_ids": [],
            "overlap": 0,
        }
    return report


def build_training_task_datasets(data, tasks, augment=True, validation_enabled=False):
    if validation_enabled:
        raise ValueError("Fixed-budget training does not support validation")
    # Lazy import keeps partition tests independent of CUDA/model dependencies.
    from ._compat.multitask import build_task_datasets

    for task in tasks:
        check_config(data[task])
    train = build_task_datasets(
        data, logical_split="train", augment=augment, tasks=tasks
    )
    audit = {task: partition_task(train[task], None, data[task]) for task in tasks}
    # Preserve the caller's return contract; the validation slot is always empty.
    return train, {}, audit
