"""Data components: builders."""

from __future__ import annotations

from typing import (
    Mapping,
    Sequence,
)
from torch.utils.data import (
    ConcatDataset,
    Dataset,
)
from uniflow_rs._compat.deventer import (
    build_dataset_from_config,
)
from uniflow_rs._compat.dior import (
    DIORDetectionDataset,
    load_dior_code_palette,
)
from uniflow_rs._compat.dota import (
    DETECTION_TASK,
)
from uniflow_rs.data.building import (
    BuildingTaskDataset,
    SpaceNet6SARBuildingDataset,
)
from uniflow_rs.data.change import (
    LEVIRCDPlusDataset,
    UrbanSARFloodChangeDataset,
    WHUChangeDataset,
)
from uniflow_rs.data.contracts import (
    ALL_TASKS,
    BUILDING_TASK,
    CHANGE_TASK,
    HEIGHT_TASK,
    LOVEDA_CODE_PALETTE_NAME,
    OPENEARTHMAP_SAR_CODE_PALETTE_NAME,
    REFERRING_TASK,
    SAR_BUILDING_TASK,
    SAR_CHANGE_TASK,
    SAR_SEMANTIC_TASK,
    SEMANTIC_TASK,
)
from uniflow_rs.data.height import (
    DFC23Track2HeightDataset,
    PotsdamHeightDataset,
)
from uniflow_rs.data.loveda import (
    LoveDASemanticDataset,
)
from uniflow_rs.data.openearthmap import (
    OpenEarthMapOpticalSemanticDataset,
    OpenEarthMapSARSemanticDataset,
)
from uniflow_rs.data.referring import (
    ReferringSegmentationDataset,
)
from uniflow_rs.data.sampling import (
    BalancedSourceDataset,
    PairPreservingConcatDataset,
    SourceTaggedDataset,
)


def _expanded_task_sources(task_cfg: Mapping) -> dict[str, dict]:
    """Return named source configs while inheriting task-level defaults."""

    sources = task_cfg.get("sources")
    if sources is None:
        return {}
    if not isinstance(sources, Mapping) or not sources:
        raise ValueError("task sources must be a non-empty mapping")
    reserved = {
        "sources",
        "source_sampling_weights",
        "samples_per_epoch",
    }
    defaults = {key: value for key, value in task_cfg.items() if key not in reserved}
    expanded: dict[str, dict] = {}
    for source_name, source_cfg in sources.items():
        if not isinstance(source_cfg, Mapping):
            raise TypeError(
                f"Source {source_name!r} must be a mapping, got "
                f"{type(source_cfg).__name__}"
            )
        merged = dict(defaults)
        merged.update(dict(source_cfg))
        expanded[str(source_name)] = merged
    return expanded


def _combine_task_sources(
    source_datasets: Mapping[str, Dataset],
    task_cfg: Mapping,
    logical_split: str,
) -> Dataset:
    if not source_datasets:
        raise ValueError("No source datasets were built")
    if len(source_datasets) == 1:
        return next(iter(source_datasets.values()))
    if logical_split == "train":
        return BalancedSourceDataset(
            source_datasets,
            source_weights=task_cfg.get("source_sampling_weights"),
            samples_per_epoch=task_cfg.get("samples_per_epoch"),
        )
    return ConcatDataset(list(source_datasets.values()))


def _build_change_source(
    change_cfg: Mapping,
    data: Mapping,
    logical_split: str,
    augment: bool,
) -> Dataset:
    change_dataset = str(change_cfg.get("dataset", "whu")).strip().casefold()
    resolution = int(change_cfg.get("resolution", data.get("resolution", 512)))
    configured_splits = change_cfg.get(f"{logical_split}_splits")
    if configured_splits is None:
        configured_splits = change_cfg.get(
            f"{logical_split}_split",
            logical_split,
        )
    if isinstance(configured_splits, str):
        splits = (configured_splits,)
    elif isinstance(configured_splits, Sequence):
        splits = tuple(str(split) for split in configured_splits)
    else:
        raise TypeError(
            "Change split configuration must be a string or sequence of "
            f"strings, got {type(configured_splits).__name__}"
        )
    if not splits or any(not split.strip() for split in splits):
        raise ValueError("Change split configuration cannot be empty")
    normalized_splits = tuple(split.strip().casefold() for split in splits)
    if len(set(normalized_splits)) != len(normalized_splits):
        raise ValueError(f"Change split configuration has duplicates: {splits}")

    split_datasets = []
    for split in splits:
        if change_dataset in {"levir", "levir_cd", "levir_cd_plus"}:
            split_dataset = LEVIRCDPlusDataset(
                root=change_cfg["root"],
                split=split,
                resolution=resolution,
                augment=augment,
                validation_fraction=float(change_cfg.get("validation_fraction", 0.1)),
                split_seed=int(change_cfg.get("split_seed", 12345)),
            )
        elif change_dataset == "whu":
            split_dataset = WHUChangeDataset(
                root=change_cfg["root"],
                split=split,
                resolution=resolution,
                before_year=str(change_cfg.get("before_year", "2012")),
                after_year=str(change_cfg.get("after_year", "2016")),
                augment=augment,
                target_source=str(change_cfg.get("target_source", "official")),
                official_label_dir=change_cfg.get("official_label_dir"),
            )
        else:
            raise ValueError(
                "data.change dataset must be 'whu' or 'levir_cd_plus', "
                f"got {change_dataset!r}"
            )
        split_datasets.append(split_dataset)

    return (
        split_datasets[0] if len(split_datasets) == 1 else ConcatDataset(split_datasets)
    )


def _build_height_source(
    height_cfg: Mapping,
    data: Mapping,
    logical_split: str,
    augment: bool,
    *,
    target_max_height_meters: float,
    target_height_codec: str,
) -> Dataset:
    configured_splits = height_cfg.get(f"{logical_split}_splits")
    if configured_splits is None:
        configured_splits = height_cfg.get(
            f"{logical_split}_split",
            logical_split,
        )
    if isinstance(configured_splits, str):
        splits = (configured_splits,)
    elif isinstance(configured_splits, Sequence):
        splits = tuple(str(split) for split in configured_splits)
    else:
        raise TypeError(
            "Height split configuration must be a string or sequence of "
            f"strings, got {type(configured_splits).__name__}"
        )
    if not splits or any(not split.strip() for split in splits):
        raise ValueError("Height split configuration cannot be empty")
    normalized_splits = tuple(split.strip().casefold() for split in splits)
    if len(set(normalized_splits)) != len(normalized_splits):
        raise ValueError(f"Height split configuration has duplicates: {splits}")

    height_format = str(
        height_cfg.get("dataset_format", height_cfg.get("source", ""))
    ).casefold()
    resolution = int(height_cfg.get("resolution", data.get("resolution", 512)))
    split_datasets: list[Dataset] = []
    for split in splits:
        if height_format == "dfc23_track2":
            split_dataset = DFC23Track2HeightDataset(
                root=height_cfg["root"],
                split=split,
                resolution=resolution,
                augment=augment,
                manifest=str(height_cfg.get("manifest", "splits.json")),
                max_height_meters=target_max_height_meters,
                height_codec=target_height_codec,
                input_modality=str(
                    height_cfg.get(
                        f"{logical_split}_input_modality",
                        height_cfg.get("input_modality", "rgb_or_sar"),
                    )
                ),
                rgb_prompt_key=height_cfg.get("rgb_prompt_key"),
                sar_prompt_key=height_cfg.get("sar_prompt_key"),
            )
        else:
            split_tiles = height_cfg.get(f"{split}_tiles")
            if split_tiles is None:
                split_tiles = height_cfg.get(f"{logical_split}_tiles")
            split_dataset = PotsdamHeightDataset(
                root=height_cfg["root"],
                split=split,
                resolution=resolution,
                tile_ids=split_tiles,
                augment=augment,
                image_dir=str(height_cfg.get("image_dir", "images")),
                height_dir=str(height_cfg.get("height_dir", "heights")),
                max_height_meters=target_max_height_meters,
                source_max_height_meters=float(
                    height_cfg.get(
                        "source_max_height_meters",
                        target_max_height_meters,
                    )
                ),
                height_codec=target_height_codec,
                source_size=int(height_cfg.get("source_size", 6000)),
            )
        split_datasets.append(split_dataset)

    return (
        split_datasets[0]
        if len(split_datasets) == 1
        else PairPreservingConcatDataset(split_datasets)
    )


def _build_referring_source(
    referring_cfg: Mapping,
    data: Mapping,
    logical_split: str,
    augment: bool,
) -> Dataset:
    configured_splits = referring_cfg.get(f"{logical_split}_splits")
    if configured_splits is None:
        configured_splits = referring_cfg.get(
            f"{logical_split}_split",
            logical_split,
        )
    if isinstance(configured_splits, str):
        splits = (configured_splits,)
    elif isinstance(configured_splits, Sequence):
        splits = tuple(str(split) for split in configured_splits)
    else:
        raise TypeError(
            "Referring split configuration must be a string or sequence of "
            f"strings, got {type(configured_splits).__name__}"
        )
    if not splits or any(not split.strip() for split in splits):
        raise ValueError("Referring split configuration cannot be empty")
    normalized_splits = tuple(split.strip().casefold() for split in splits)
    if len(set(normalized_splits)) != len(normalized_splits):
        raise ValueError(f"Referring split configuration has duplicates: {splits}")

    split_datasets = [
        ReferringSegmentationDataset(
            root=referring_cfg["root"],
            split=split,
            resolution=int(
                referring_cfg.get(
                    "resolution",
                    data.get("resolution", 512),
                )
            ),
            manifest=str(referring_cfg.get("manifest", "splits.json")),
            augment=augment and bool(referring_cfg.get("augment", False)),
        )
        for split in splits
    ]
    if len(split_datasets) == 1:
        return split_datasets[0]

    prompt_identities = {
        (dataset.prompt_count, dataset.prompt_sha256) for dataset in split_datasets
    }
    if len(prompt_identities) != 1:
        raise RuntimeError(
            "Referring splits from one source do not share one prompt bank: "
            f"{sorted(prompt_identities)}"
        )
    combined = PairPreservingConcatDataset(split_datasets)
    # SourceTaggedDataset uses this source-level count to offset the next
    # referring dataset (LaSeRS after RRSIS-D). Prompt indices are global
    # within each prepared manifest, so concatenating splits must not add a
    # second per-split offset.
    combined.prompt_count = split_datasets[0].prompt_count
    combined.prompt_sha256 = split_datasets[0].prompt_sha256
    return combined


def _build_sar_semantic_source(
    semantic_cfg: Mapping,
    data: Mapping,
    logical_split: str,
    augment: bool,
) -> Dataset:
    """Build either SAR or optical OpenEarthMap with one task contract."""

    if semantic_cfg.get("teacher_cache_dir") not in (None, ""):
        raise ValueError(
            "teacher_cache_dir is no longer supported. Remove this setting; "
            "UniFlow-RS does not consume teacher probability caches."
        )

    configured_splits = semantic_cfg.get(f"{logical_split}_splits")
    if configured_splits is None:
        configured_splits = semantic_cfg.get(
            f"{logical_split}_split",
            logical_split,
        )
    if isinstance(configured_splits, str):
        splits = (configured_splits,)
    elif isinstance(configured_splits, Sequence):
        splits = tuple(str(split) for split in configured_splits)
    else:
        raise TypeError(
            "OpenEarthMap split configuration must be a string or sequence "
            f"of strings, got {type(configured_splits).__name__}"
        )
    if not splits or any(not split.strip() for split in splits):
        raise ValueError("OpenEarthMap split configuration cannot be empty")
    if len(set(splits)) != len(splits):
        raise ValueError(f"OpenEarthMap split configuration has duplicates: {splits}")

    dataset_format = (
        str(semantic_cfg.get("dataset_format", "openearthmap_sar")).strip().casefold()
    )
    resolution = int(semantic_cfg.get("resolution", data.get("resolution", 512)))
    use_augmentation = augment and bool(semantic_cfg.get("augment", True))
    configured_modalities = semantic_cfg.get(
        f"{logical_split}_input_modality_by_split",
        semantic_cfg.get("input_modality_by_split", {}),
    )
    if configured_modalities is None:
        configured_modalities = {}
    if not isinstance(configured_modalities, Mapping):
        raise TypeError(
            "OpenEarthMap input_modality_by_split must be a mapping, got "
            f"{type(configured_modalities).__name__}"
        )
    modality_by_split = {
        str(split_name).strip().casefold(): str(modality)
        for split_name, modality in configured_modalities.items()
    }
    default_input_modality = str(
        semantic_cfg.get(
            f"{logical_split}_input_modality",
            semantic_cfg.get("input_modality", "sar"),
        )
    )

    def input_modality_for_split(split: str) -> str:
        return modality_by_split.get(
            split.strip().casefold(),
            default_input_modality,
        )

    if dataset_format in {
        "openearthmap",
        "openearthmap_optical",
        "openearthmap_rgb",
    }:
        split_datasets = [
            OpenEarthMapOpticalSemanticDataset(
                root=semantic_cfg["root"],
                split=split,
                resolution=resolution,
                augment=use_augmentation,
                code_palette=str(
                    semantic_cfg.get(
                        "code_palette",
                        OPENEARTHMAP_SAR_CODE_PALETTE_NAME,
                    )
                ),
            )
            for split in splits
        ]
    elif dataset_format in {
        "openearthmap_sar",
        "dfc25",
        "dfc25_track1",
    }:
        split_datasets = [
            OpenEarthMapSARSemanticDataset(
                root=semantic_cfg["root"],
                split=split,
                resolution=resolution,
                source_size=int(semantic_cfg.get("source_size", 1024)),
                trainval_dir=str(
                    semantic_cfg.get(
                        "trainval_dir",
                        "dfc25_track1_trainval",
                    )
                ),
                augment=use_augmentation,
                input_modality=input_modality_for_split(split),
                subset=str(
                    semantic_cfg.get(
                        f"{logical_split}_subset",
                        "all",
                    )
                ),
                holdout_fraction=float(semantic_cfg.get("holdout_fraction", 0.0)),
                holdout_seed=int(semantic_cfg.get("holdout_seed", 12345)),
                code_palette=str(
                    semantic_cfg.get(
                        "code_palette",
                        OPENEARTHMAP_SAR_CODE_PALETTE_NAME,
                    )
                ),
                rgb_prompt_key=semantic_cfg.get("rgb_prompt_key"),
                sar_prompt_key=semantic_cfg.get("sar_prompt_key"),
            )
            for split in splits
        ]
    else:
        raise ValueError(
            "data.sar_semantic.dataset_format must be openearthmap_sar or "
            f"openearthmap_optical, got {dataset_format!r}"
        )
    return (
        split_datasets[0]
        if len(split_datasets) == 1
        else PairPreservingConcatDataset(split_datasets)
    )


def build_task_datasets(
    data: dict,
    logical_split: str,
    augment: bool,
    tasks: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Dataset]:
    """Build only the requested datasets from the nested task config."""

    selected = ALL_TASKS if tasks is None else tuple(tasks)
    unknown = sorted(set(selected) - set(ALL_TASKS))
    if unknown:
        raise ValueError(f"Unsupported task(s): {unknown}")
    datasets: dict[str, Dataset] = {}
    if BUILDING_TASK in selected:
        if BUILDING_TASK not in data:
            raise KeyError("Active building task requires data.building")
        building_cfg = dict(data[BUILDING_TASK])
        building_cfg.setdefault(
            "resolution",
            int(data.get("resolution", 512)),
        )
        configured_splits = building_cfg.get(f"{logical_split}_splits")
        if configured_splits is None:
            configured_splits = building_cfg.get(
                f"{logical_split}_split",
                logical_split,
            )
        if isinstance(configured_splits, str):
            building_splits = (configured_splits,)
        elif isinstance(configured_splits, Sequence):
            building_splits = tuple(str(split) for split in configured_splits)
        else:
            raise TypeError(
                "Building split configuration must be a string or sequence "
                f"of strings, got {type(configured_splits).__name__}"
            )
        if not building_splits or any(not split.strip() for split in building_splits):
            raise ValueError("Building split configuration cannot be empty")
        if len({split.strip().casefold() for split in building_splits}) != len(
            building_splits
        ):
            raise ValueError(
                "Building split configuration has duplicates: " f"{building_splits}"
            )
        building_datasets = []
        for split in building_splits:
            split_cfg = dict(building_cfg)
            split_cfg[f"{logical_split}_split"] = split
            building_datasets.append(
                BuildingTaskDataset(
                    build_dataset_from_config(
                        split_cfg,
                        logical_split=logical_split,
                        augment=augment,
                    )
                )
            )
        datasets[BUILDING_TASK] = (
            building_datasets[0]
            if len(building_datasets) == 1
            else ConcatDataset(building_datasets)
        )

    if CHANGE_TASK in selected:
        if CHANGE_TASK not in data:
            raise KeyError("Active change task requires data.change")
        change_cfg = dict(data[CHANGE_TASK])
        source_cfgs = _expanded_task_sources(change_cfg)
        if source_cfgs:
            source_datasets = {
                source_name: SourceTaggedDataset(
                    _build_change_source(
                        source_cfg,
                        data,
                        logical_split,
                        augment,
                    ),
                    source_name,
                    prompt_key=source_cfg.get("prompt_key"),
                )
                for source_name, source_cfg in source_cfgs.items()
            }
            datasets[CHANGE_TASK] = _combine_task_sources(
                source_datasets,
                change_cfg,
                logical_split,
            )
        else:
            datasets[CHANGE_TASK] = _build_change_source(
                change_cfg,
                data,
                logical_split,
                augment,
            )
    if SEMANTIC_TASK in selected:
        if SEMANTIC_TASK not in data:
            raise KeyError("Active semantic task requires data.semantic")
        semantic_cfg = dict(data[SEMANTIC_TASK])
        configured_splits = semantic_cfg.get(f"{logical_split}_splits")
        if configured_splits is None:
            configured_splits = semantic_cfg.get(
                f"{logical_split}_split",
                logical_split,
            )
        if isinstance(configured_splits, str):
            semantic_splits = (configured_splits,)
        elif isinstance(configured_splits, Sequence):
            semantic_splits = tuple(str(split) for split in configured_splits)
        else:
            raise TypeError(
                "Semantic split configuration must be a string or sequence "
                f"of strings, got {type(configured_splits).__name__}"
            )
        if not semantic_splits or any(not split.strip() for split in semantic_splits):
            raise ValueError("Semantic split configuration cannot be empty")
        if len({split.strip().casefold() for split in semantic_splits}) != len(
            semantic_splits
        ):
            raise ValueError(
                "Semantic split configuration has duplicates: " f"{semantic_splits}"
            )
        semantic_datasets = [
            LoveDASemanticDataset(
                root=semantic_cfg["root"],
                split=split,
                resolution=int(
                    semantic_cfg.get(
                        "resolution",
                        data.get("resolution", 512),
                    )
                ),
                domains=tuple(semantic_cfg.get("domains", ("Urban", "Rural"))),
                augment=augment,
                source_size=int(semantic_cfg.get("source_size", 1024)),
                code_palette=str(
                    semantic_cfg.get(
                        "code_palette",
                        LOVEDA_CODE_PALETTE_NAME,
                    )
                ),
            )
            for split in semantic_splits
        ]
        datasets[SEMANTIC_TASK] = (
            semantic_datasets[0]
            if len(semantic_datasets) == 1
            else ConcatDataset(semantic_datasets)
        )
    if HEIGHT_TASK in selected:
        if HEIGHT_TASK not in data:
            raise KeyError("Active height task requires data.height")
        height_cfg = dict(data[HEIGHT_TASK])
        source_cfgs = _expanded_task_sources(height_cfg)
        target_max_height_meters = float(height_cfg.get("max_height_meters", 25.5))
        target_height_codec = str(height_cfg.get("height_codec", "linear"))
        if source_cfgs:
            source_datasets = {
                source_name: SourceTaggedDataset(
                    _build_height_source(
                        source_cfg,
                        data,
                        logical_split,
                        augment,
                        target_max_height_meters=target_max_height_meters,
                        target_height_codec=target_height_codec,
                    ),
                    source_name,
                    prompt_key=source_cfg.get("prompt_key"),
                )
                for source_name, source_cfg in source_cfgs.items()
            }
            datasets[HEIGHT_TASK] = _combine_task_sources(
                source_datasets,
                height_cfg,
                logical_split,
            )
        else:
            datasets[HEIGHT_TASK] = _build_height_source(
                height_cfg,
                data,
                logical_split,
                augment,
                target_max_height_meters=target_max_height_meters,
                target_height_codec=target_height_codec,
            )
    if SAR_BUILDING_TASK in selected:
        if SAR_BUILDING_TASK not in data:
            raise KeyError("Active SAR building task requires data.sar_building")
        sar_cfg = dict(data[SAR_BUILDING_TASK])
        split = str(sar_cfg.get(f"{logical_split}_split", logical_split))
        datasets[SAR_BUILDING_TASK] = SpaceNet6SARBuildingDataset(
            root=sar_cfg["root"],
            split=split,
            resolution=int(
                sar_cfg.get(
                    "resolution",
                    data.get("resolution", 512),
                )
            ),
            manifest=str(sar_cfg.get("manifest", "splits.json")),
            # Preparation canonicalizes north/south sensor looks. Arbitrary
            # quarter turns and vertical flips would undo that alignment.
            augment=augment and bool(sar_cfg.get("augment", False)),
            input_modality=str(
                sar_cfg.get(
                    f"{logical_split}_input_modality",
                    sar_cfg.get("input_modality", "sar"),
                )
            ),
            rgb_prompt_key=sar_cfg.get("rgb_prompt_key"),
            sar_prompt_key=sar_cfg.get("sar_prompt_key"),
        )
    if SAR_SEMANTIC_TASK in selected:
        if SAR_SEMANTIC_TASK not in data:
            raise KeyError("Active SAR semantic task requires data.sar_semantic")
        sar_semantic_cfg = dict(data[SAR_SEMANTIC_TASK])
        source_cfgs = _expanded_task_sources(sar_semantic_cfg)
        if source_cfgs:
            source_datasets = {
                source_name: SourceTaggedDataset(
                    _build_sar_semantic_source(
                        source_cfg,
                        data,
                        logical_split,
                        augment,
                    ),
                    source_name,
                    prompt_key=source_cfg.get("prompt_key"),
                )
                for source_name, source_cfg in source_cfgs.items()
            }
            datasets[SAR_SEMANTIC_TASK] = _combine_task_sources(
                source_datasets,
                sar_semantic_cfg,
                logical_split,
            )
        else:
            datasets[SAR_SEMANTIC_TASK] = _build_sar_semantic_source(
                sar_semantic_cfg,
                data,
                logical_split,
                augment,
            )
    if SAR_CHANGE_TASK in selected:
        if SAR_CHANGE_TASK not in data:
            raise KeyError("Active SAR change task requires data.sar_change")
        sar_change_cfg = dict(data[SAR_CHANGE_TASK])
        configured_splits = sar_change_cfg.get(f"{logical_split}_splits")
        if configured_splits is None:
            configured_splits = sar_change_cfg.get(
                f"{logical_split}_split", logical_split
            )
        if isinstance(configured_splits, str):
            sar_change_splits = (configured_splits,)
        elif isinstance(configured_splits, Sequence):
            sar_change_splits = tuple(str(split) for split in configured_splits)
        else:
            raise TypeError(
                "SAR change split configuration must be a string or "
                f"sequence of strings, got {type(configured_splits).__name__}"
            )
        if not sar_change_splits or any(
            not split.strip() for split in sar_change_splits
        ):
            raise ValueError("SAR change split configuration cannot be empty")
        normalized_splits = tuple(
            split.strip().casefold() for split in sar_change_splits
        )
        if len(set(normalized_splits)) != len(normalized_splits):
            raise ValueError(
                "SAR change split configuration has duplicates: " f"{sar_change_splits}"
            )
        split_datasets = [
            UrbanSARFloodChangeDataset(
                root=sar_change_cfg["root"],
                split=split,
                resolution=int(
                    sar_change_cfg.get(
                        "resolution",
                        data.get("resolution", 512),
                    )
                ),
                augment=augment and bool(sar_change_cfg.get("augment", True)),
                # The dataset balances only its physical ``train`` split;
                # the labeled test split is added once without oversampling.
                balance_categories=(
                    logical_split == "train"
                    and bool(sar_change_cfg.get("balance_categories", True))
                ),
                normalization_file=str(
                    sar_change_cfg.get(
                        "normalization_file",
                        "data_norm.txt",
                    )
                ),
            )
            for split in sar_change_splits
        ]
        datasets[SAR_CHANGE_TASK] = (
            split_datasets[0]
            if len(split_datasets) == 1
            else ConcatDataset(split_datasets)
        )
    if REFERRING_TASK in selected:
        if REFERRING_TASK not in data:
            raise KeyError("Active referring task requires data.referring")
        referring_cfg = dict(data[REFERRING_TASK])
        source_cfgs = _expanded_task_sources(referring_cfg)
        if source_cfgs:
            source_datasets: dict[str, Dataset] = {}
            prompt_offset = 0
            for source_name, source_cfg in source_cfgs.items():
                source_dataset = _build_referring_source(
                    source_cfg,
                    data,
                    logical_split,
                    augment,
                )
                source_datasets[source_name] = SourceTaggedDataset(
                    source_dataset,
                    source_name,
                    prompt_offset=prompt_offset,
                    prompt_key=source_cfg.get("prompt_key"),
                )
                prompt_offset += source_dataset.prompt_count
            datasets[REFERRING_TASK] = _combine_task_sources(
                source_datasets,
                referring_cfg,
                logical_split,
            )
        else:
            datasets[REFERRING_TASK] = _build_referring_source(
                referring_cfg,
                data,
                logical_split,
                augment,
            )
    if DETECTION_TASK in selected:
        if DETECTION_TASK not in data:
            raise KeyError("Active oriented-detection task requires data.detection")
        detection_cfg = dict(data[DETECTION_TASK])
        detection_format = (
            str(detection_cfg.get("dataset_format", "dior")).strip().casefold()
        )
        if detection_format != "dior":
            raise ValueError(
                "This release supports data.detection.dataset_format='dior' only, "
                f"got {detection_format!r}"
            )
        detection_code_palette = (
            load_dior_code_palette(detection_cfg["code_palette_file"])
            if detection_cfg.get("code_palette_file")
            else str(detection_cfg.get("code_palette", "dior_detection_v1"))
        )
        configured_splits = detection_cfg.get(f"{logical_split}_splits")
        if configured_splits is None:
            configured_splits = detection_cfg.get(
                f"{logical_split}_split",
                logical_split,
            )
        if isinstance(configured_splits, str):
            detection_splits = (configured_splits,)
        elif isinstance(configured_splits, Sequence):
            detection_splits = tuple(str(value) for value in configured_splits)
        else:
            raise TypeError(
                "Detection split configuration must be a string or sequence"
            )
        split_datasets = [
            DIORDetectionDataset(
                root=detection_cfg["root"],
                split=split,
                resolution=int(
                    detection_cfg.get("resolution", data.get("resolution", 512))
                ),
                manifest=str(detection_cfg.get("manifest", "splits.json")),
                augment=augment and bool(detection_cfg.get("augment", True)),
                prompt_key=detection_cfg.get("prompt_key"),
                negative_query_probability=float(
                    detection_cfg.get("negative_query_probability", 0.5)
                ),
                evaluation_query_mode=str(
                    detection_cfg.get("evaluation_query_mode", "all")
                ),
                target_mode=str(detection_cfg.get("target_mode", "binary")),
                code_palette=detection_code_palette,
                training=logical_split == "train",
            )
            for split in detection_splits
        ]
        datasets[DETECTION_TASK] = (
            split_datasets[0]
            if len(split_datasets) == 1
            else PairPreservingConcatDataset(split_datasets)
        )
    return datasets
