"""Exclude Accelerate's tail padding without changing collective call counts."""

from collections.abc import Mapping

from accelerate import DataLoaderConfiguration
import torch


def evaluation_dataloader_config():
    # Equal iteration counts are needed by detection's per-batch collectives.
    # Padding is removed before inference, metrics and output writes instead.
    return DataLoaderConfiguration(
        split_batches=False, dispatch_batches=False, even_batches=True,
    )


def validate_evaluation_sharding(accelerator):
    if accelerator.num_processes > 1 and (
        accelerator.split_batches
        or accelerator.dispatch_batches is True
        or not accelerator.even_batches
    ):
        raise ValueError(
            "Evaluation requires whole-batch sharding with split_batches=False, "
            "dispatch_batches=False and even_batches=True; padding is filtered explicitly."
        )


def real_batch_size(dataset_size, batch_size, num_processes, process_index, batch_index):
    """Real prefix of a sequential, whole-batch, round-robin shard batch."""
    if dataset_size < 0 or batch_size < 1 or num_processes < 1 or batch_index < 0:
        raise ValueError("Invalid evaluation shard dimensions")
    if not 0 <= process_index < num_processes:
        raise ValueError("Invalid process index")
    start = (batch_index * num_processes + process_index) * batch_size
    return min(batch_size, max(0, dataset_size - start))


def trim_evaluation_batch(batch, size):
    """Trim default-collated tensors and strings, preserving nested metadata."""
    actual_size = int(batch["before"].shape[0])
    if not 0 <= size <= actual_size:
        raise ValueError("Real evaluation batch size exceeds the collated batch")
    if size == actual_size:
        return batch

    def trim(value):
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return value
            if value.shape[0] != actual_size:
                raise ValueError("Unexpected unbatched tensor in evaluation metadata")
            return value[:size]
        if isinstance(value, Mapping):
            return {key: trim(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            if all(isinstance(item, (str, bytes)) for item in value):
                return value[:size]
            return type(value)(trim(item) for item in value)
        return value

    return trim(batch)


def collect_stitched_batch(stitcher, records, ground_truth, gather):
    """All ranks, including padded-only ranks, enter the same collectives."""
    detections = stitcher.add(gather(records))
    targets = gather(ground_truth) if ground_truth is not None else []
    return detections, targets
