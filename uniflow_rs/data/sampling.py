"""Data components: sampling."""

from __future__ import annotations

import math
import random
from typing import (
    Iterator,
    Mapping,
    Sequence,
)
import torch
from torch import (
    Tensor,
)
from torch.utils.data import (
    ConcatDataset,
    Dataset,
    Sampler,
    default_collate,
)
from uniflow_rs.data.contracts import (
    _TRAIN_BATCH_KEYS,
)


class SourceTaggedDataset(Dataset):
    """Tag a source and optionally select its fixed text condition."""

    def __init__(
        self,
        dataset: Dataset,
        source_name: str,
        prompt_offset: int = 0,
        prompt_key: str | None = None,
    ):
        self.dataset = dataset
        self.source_name = str(source_name)
        self.prompt_offset = int(prompt_offset)
        self.prompt_key = None if prompt_key is None else str(prompt_key)

    def __len__(self) -> int:
        return len(self.dataset)

    def paired_batch_indices(
        self,
        batch_size: int,
        rng: random.Random,
        occurrence: int = 0,
    ) -> list[tuple[str, object]]:
        child_indices = _dataset_batch_indices(
            self.dataset,
            batch_size,
            rng,
            occurrence,
        )
        return [("__source_tagged__", index) for index in child_indices]

    def __getitem__(self, index: int | tuple[str, object]) -> dict:
        if isinstance(index, tuple):
            if len(index) != 2 or index[0] != "__source_tagged__":
                raise TypeError(f"Unsupported source-tagged index: {index!r}")
            index = index[1]
        sample = dict(self.dataset[index])
        sample["dataset_source"] = self.source_name
        sample["name"] = f"{self.source_name}__{sample['name']}"
        if self.prompt_key is not None:
            sample["prompt_key"] = self.prompt_key
        if "prompt_index" in sample:
            sample["prompt_index"] = int(sample["prompt_index"]) + self.prompt_offset
        return sample


class PairPreservingConcatDataset(ConcatDataset):
    """Concatenate splits without breaking registered paired batches.

    PyTorch's ordinary :class:`ConcatDataset` only supports integer lookup.
    That silently bypasses ``paired_batch_indices`` on child datasets, which
    would separate co-registered RGB/SAR views after combining train and test
    splits.  This wrapper chooses one physical split per microbatch and keeps
    the child's structured indices intact.
    """

    _INDEX_TAG = "__pair_preserving_concat__"

    def paired_batch_indices(
        self,
        batch_size: int,
        rng: random.Random,
        occurrence: int = 0,
    ) -> list[tuple[str, int, object]]:
        # Sample splits in proportion to their number of exposed examples.
        # For RGB/SAR datasets the exposed length already includes both views.
        dataset_index = rng.choices(
            range(len(self.datasets)),
            weights=[len(dataset) for dataset in self.datasets],
            k=1,
        )[0]
        child_indices = _dataset_batch_indices(
            self.datasets[dataset_index],
            batch_size,
            rng,
            occurrence,
        )
        return [(self._INDEX_TAG, dataset_index, index) for index in child_indices]

    def __getitem__(
        self,
        index: int | tuple[str, int, object],
    ) -> dict:
        if isinstance(index, tuple):
            if len(index) != 3 or index[0] != self._INDEX_TAG:
                raise TypeError(f"Unsupported pair-preserving concat index: {index!r}")
            return self.datasets[int(index[1])][index[2]]
        return super().__getitem__(index)


class BalancedSourceDataset(Dataset):
    """Sample several datasets of one task with explicit source weights.

    ``samples_per_epoch`` decouples the optimizer schedule from raw dataset
    sizes. This is important for LaSeRS, whose many expressions would
    otherwise make one multitask epoch several times longer than before.
    Source selection follows an exact integer cycle; the sample inside each
    source is randomized by the DataLoader worker RNG every epoch.
    """

    def __init__(
        self,
        datasets: Mapping[str, Dataset],
        source_weights: Mapping[str, int] | None = None,
        samples_per_epoch: int | None = None,
    ):
        if not datasets:
            raise ValueError("At least one source dataset is required")
        self.datasets = dict(datasets)
        if any(len(dataset) < 1 for dataset in self.datasets.values()):
            raise ValueError("Every source dataset must be non-empty")
        configured = source_weights or {name: 1 for name in self.datasets}
        missing = sorted(set(self.datasets) - set(configured))
        extra = sorted(set(configured) - set(self.datasets))
        if missing or extra:
            raise ValueError(
                "source_sampling_weights must match configured sources: "
                f"missing={missing}, extra={extra}"
            )
        self.weights = {name: int(configured[name]) for name in self.datasets}
        if any(weight < 1 for weight in self.weights.values()):
            raise ValueError("Every source sampling weight must be >= 1")
        self.source_cycle = tuple(
            name for name in self.datasets for _ in range(self.weights[name])
        )
        default_size = sum(len(dataset) for dataset in self.datasets.values())
        self.samples_per_epoch = int(
            default_size if samples_per_epoch is None else samples_per_epoch
        )
        if self.samples_per_epoch < 1:
            raise ValueError("samples_per_epoch must be positive")

    def __len__(self) -> int:
        return self.samples_per_epoch

    def paired_batch_indices(
        self,
        batch_size: int,
        rng: random.Random,
        occurrence: int = 0,
    ) -> list[tuple[str, str, object]]:
        source_name = self.source_cycle[occurrence % len(self.source_cycle)]
        child_indices = _dataset_batch_indices(
            self.datasets[source_name],
            batch_size,
            rng,
            occurrence // len(self.source_cycle),
        )
        return [("__balanced_source__", source_name, index) for index in child_indices]

    def __getitem__(
        self,
        index: int | tuple[str, str, object],
    ) -> dict:
        if isinstance(index, tuple):
            if len(index) != 3 or index[0] != "__balanced_source__":
                raise TypeError(f"Unsupported balanced-source index: {index!r}")
            return self.datasets[str(index[1])][index[2]]
        source_name = self.source_cycle[index % len(self.source_cycle)]
        dataset = self.datasets[source_name]
        return dataset[random.randrange(len(dataset))]


class WeightedMultitaskDataset(Dataset):
    """Repeat tasks according to integer sampling weights in every cycle."""

    def __init__(
        self,
        datasets: Mapping[str, Dataset],
        task_weights: Mapping[str, int] | None = None,
    ):
        if not datasets:
            raise ValueError("At least one task dataset is required")
        self.datasets = dict(datasets)
        if any(len(dataset) < 1 for dataset in self.datasets.values()):
            raise ValueError("Every active task dataset must be non-empty")
        configured_weights = task_weights or {task: 1 for task in self.datasets}
        missing = sorted(set(self.datasets) - set(configured_weights))
        extra = sorted(set(configured_weights) - set(self.datasets))
        if missing or extra:
            raise ValueError(
                "task_sampling_weights must match active datasets: "
                f"missing={missing}, extra={extra}"
            )
        self.weights = {task: int(configured_weights[task]) for task in self.datasets}
        if any(weight < 1 for weight in self.weights.values()):
            raise ValueError("Every task sampling weight must be >= 1")
        self.task_cycle = tuple(
            task for task in self.datasets for _ in range(self.weights[task])
        )
        self.cycles = max(
            math.ceil(len(self.datasets[task]) / self.weights[task])
            for task in self.datasets
        )
        cycle_occurrence: dict[str, int] = {task: 0 for task in self.datasets}
        offsets: list[int] = []
        for task in self.task_cycle:
            offsets.append(cycle_occurrence[task])
            cycle_occurrence[task] += 1
        self.cycle_offsets = tuple(offsets)

    def __len__(self) -> int:
        return self.cycles * len(self.task_cycle)

    def paired_epoch_batches(
        self,
        number_of_batches: int,
        batch_size: int,
        rng: random.Random,
        batch_size_by_task: Mapping[str, int] | None = None,
    ) -> Iterator[list[tuple[str, str, object]]]:
        configured_batch_sizes = dict(batch_size_by_task or {})
        unknown_tasks = sorted(set(configured_batch_sizes) - set(self.datasets))
        if unknown_tasks:
            raise ValueError(
                "Per-task batch sizes contain unknown tasks: " f"{unknown_tasks}"
            )
        schedule = [
            self.task_cycle[index % len(self.task_cycle)]
            for index in range(number_of_batches)
        ]
        rng.shuffle(schedule)
        occurrences = {task: 0 for task in self.datasets}
        for task in schedule:
            occurrence = occurrences[task]
            occurrences[task] += 1
            task_batch_size = int(configured_batch_sizes.get(task, batch_size))
            if task_batch_size < 1:
                raise ValueError(f"Batch size for task {task!r} must be positive")
            child_indices = _dataset_batch_indices(
                self.datasets[task],
                task_batch_size,
                rng,
                occurrence,
            )
            yield [("__weighted_task__", task, index) for index in child_indices]

    def __getitem__(
        self,
        index: int | tuple[str, str, object],
    ) -> dict[str, Tensor | str]:
        if isinstance(index, tuple):
            if len(index) != 3 or index[0] != "__weighted_task__":
                raise TypeError(f"Unsupported weighted-task index: {index!r}")
            return self.datasets[str(index[1])][index[2]]
        cycle_index, cycle_position = divmod(index, len(self.task_cycle))
        task = self.task_cycle[cycle_position]
        source_index = (
            cycle_index * self.weights[task] + self.cycle_offsets[cycle_position]
        ) % len(self.datasets[task])
        return self.datasets[task][source_index]


def _dataset_batch_indices(
    dataset: Dataset,
    batch_size: int,
    rng: random.Random,
    occurrence: int,
) -> list[object]:
    pair_method = getattr(dataset, "paired_batch_indices", None)
    if callable(pair_method):
        indices = list(pair_method(batch_size, rng, occurrence))
    else:
        indices = [rng.randrange(len(dataset)) for _ in range(batch_size)]
    if len(indices) != batch_size:
        raise RuntimeError(
            f"Dataset {type(dataset).__name__} produced {len(indices)} "
            f"indices for a batch of {batch_size}"
        )
    return indices


class PairPreservingBatchSampler(Sampler[list[object]]):
    """Build task-homogeneous batches and keep registered RGB/SAR pairs.

    For paired datasets an even per-device microbatch contains complete
    co-registered RGB/SAR pairs.  Accelerate then assigns whole batches to
    ranks, so neither view can be separated onto another GPU.  Ordinary
    sources retain random sampling but are still task/source homogeneous.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        *,
        drop_last: bool = True,
        seed: int = 0,
        batch_size_by_task: Mapping[str, int] | None = None,
    ):
        self.dataset = dataset
        self.default_batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.batch_size_by_task = {
            str(task): int(value) for task, value in (batch_size_by_task or {}).items()
        }
        # Accelerate inspects ``batch_size`` to decide whether a yielded batch
        # is complete. Expose ``None`` when task-specific sizes vary so its
        # BatchSamplerShard preserves every dynamic batch on every rank.
        self.batch_size = (
            None
            if any(
                value != self.default_batch_size
                for value in self.batch_size_by_task.values()
            )
            else self.default_batch_size
        )
        self.epoch = 0
        if self.default_batch_size < 1:
            raise ValueError("batch_size must be positive")
        invalid = {
            task: value for task, value in self.batch_size_by_task.items() if value < 1
        }
        if invalid:
            raise ValueError(
                "Every per-task batch size must be positive: " f"{invalid}"
            )

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.default_batch_size
        return math.ceil(len(self.dataset) / self.default_batch_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[object]]:
        rng = random.Random(self.seed + 1_000_003 * self.epoch)
        number_of_batches = len(self)
        epoch_method = getattr(self.dataset, "paired_epoch_batches", None)
        if callable(epoch_method):
            yield from epoch_method(
                number_of_batches,
                self.default_batch_size,
                rng,
                batch_size_by_task=self.batch_size_by_task,
            )
            return
        for occurrence in range(number_of_batches):
            yield _dataset_batch_indices(
                self.dataset,
                self.default_batch_size,
                rng,
                occurrence,
            )


def multitask_train_collate(
    samples: Sequence[Mapping[str, Tensor | str | int]],
) -> dict:
    """Collate only fields shared by every task during mixed-task training.

    Some datasets expose evaluation-only metadata. Potsdam height patches,
    for example, carry parent tile coordinates used to stitch predictions.
    PyTorch's default mapping collate assumes that every sample contains the
    keys of the first sample, so a shuffled batch containing both a height
    sample and another task can otherwise fail with ``KeyError:
    'parent_name'``. Training consumes only the common fields below; each
    validation loader remains task-homogeneous and retains its full metadata.
    """

    if not samples:
        raise ValueError("Cannot collate an empty multitask batch")
    missing = {
        index: [key for key in _TRAIN_BATCH_KEYS if key not in sample]
        for index, sample in enumerate(samples)
    }
    missing = {index: keys for index, keys in missing.items() if keys}
    if missing:
        raise KeyError(f"Multitask training sample fields missing: {missing}")
    shared_samples = [
        {key: sample[key] for key in _TRAIN_BATCH_KEYS} for sample in samples
    ]
    batch = default_collate(shared_samples)
    batch["prompt_index"] = torch.tensor(
        [int(sample.get("prompt_index", -1)) for sample in samples],
        dtype=torch.long,
    )
    batch["prompt_key"] = [
        str(sample.get("prompt_key", sample["task"])) for sample in samples
    ]
    # Preserve the physical input modality for modality-aware timestep
    # sampling. Most legacy optical datasets do not expose this field, so
    # their single image is explicitly treated as RGB. Paired datasets such
    # as SpaceNet6, DFC23 and OpenEarthMap already return either ``rgb`` or
    # ``sar`` for every view.
    batch["input_modality"] = [
        str(sample.get("input_modality", "rgb")).strip().casefold()
        for sample in samples
    ]
    return batch
