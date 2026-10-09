"""Data components: contracts."""

from __future__ import annotations

import numpy as np
from uniflow_rs._compat.dota import (
    DETECTION_TASK,
)

_RASTER_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


BUILDING_TASK = "building"


CHANGE_TASK = "change"


SEMANTIC_TASK = "semantic"


HEIGHT_TASK = "height"


HEIGHT_METRIC_COUNT_SIZE = 13


SAR_BUILDING_TASK = "sar_building"


SAR_SEMANTIC_TASK = "sar_semantic"


SAR_CHANGE_TASK = "sar_change"


REFERRING_TASK = "referring"


URBANSAR_FLOOD_REGIONS = ("Weihui", "Jubba", "Nova")


ALL_TASKS = (
    BUILDING_TASK,
    CHANGE_TASK,
    SEMANTIC_TASK,
    HEIGHT_TASK,
    SAR_BUILDING_TASK,
    SAR_SEMANTIC_TASK,
    SAR_CHANGE_TASK,
    REFERRING_TASK,
    DETECTION_TASK,
)


# Paper-facing task families used only to group TMAS statistics. The internal
# ``sar_*`` identities remain unchanged everywhere else: they still select
# their existing datasets, prompts, losses, metrics, and sampling weights.
CANONICAL_TASK_FAMILY = {
    BUILDING_TASK: BUILDING_TASK,
    SAR_BUILDING_TASK: BUILDING_TASK,
    CHANGE_TASK: CHANGE_TASK,
    SAR_CHANGE_TASK: CHANGE_TASK,
    SEMANTIC_TASK: SEMANTIC_TASK,
    SAR_SEMANTIC_TASK: SEMANTIC_TASK,
    HEIGHT_TASK: HEIGHT_TASK,
    REFERRING_TASK: REFERRING_TASK,
    DETECTION_TASK: DETECTION_TASK,
}


CANONICAL_TASK_FAMILIES = tuple(dict.fromkeys(CANONICAL_TASK_FAMILY.values()))


def canonicalize_task_family(task_name: str) -> str:
    """Map an internal optimization-task identity to one of six families."""

    normalized = str(task_name).strip().casefold().replace("-", "_")
    try:
        return CANONICAL_TASK_FAMILY[normalized]
    except KeyError as error:
        raise ValueError(
            "TMAS requires a known internal training-task identifier, got "
            f"{task_name!r}"
        ) from error


_TRAIN_BATCH_KEYS = (
    "before",
    "after",
    "mask",
    "loss_mask",
    "target_rgb",
    "task",
    "name",
)


LOVEDA_CLASS_NAMES = (
    "background",
    "building",
    "road",
    "water",
    "barren",
    "forest",
    "agriculture",
)


# Official LoveDA palette for saved ground truth, predictions, and overlays.
# Index 0 is no-data/ignore, while original label IDs 1..7 correspond to the
# seven evaluated classes. The VAE uses the shared semantic codebook below;
# native IDs and official display colors remain dataset-specific.
LOVEDA_PALETTE = np.asarray(
    (
        (0, 0, 0),
        (255, 255, 255),
        (255, 0, 0),
        (255, 255, 0),
        (0, 0, 255),
        (159, 129, 183),
        (0, 255, 0),
        (255, 195, 128),
    ),
    dtype=np.uint8,
)


# Canonical semantic ontology shared by LoveDA and OpenEarthMap. The values
# are deliberately well separated in RGB space because they are encoded by a
# frozen natural-image VAE. Dataset-specific palettes below are indexed by
# native label IDs, but point to the same canonical colors for shared
# concepts.
CANONICAL_SEMANTIC_CLASS_NAMES = (
    "background",
    "building",
    "road",
    "water",
    "barren",
    "forest",
    "agriculture",
    "rangeland",
    "developed_space",
)


CANONICAL_SEMANTIC_PALETTE = np.asarray(
    (
        (0, 0, 0),  # background
        (255, 0, 0),  # building
        (255, 255, 0),  # road
        (0, 0, 255),  # water
        (255, 128, 0),  # barren / bareland
        (0, 255, 0),  # forest / tree
        (0, 255, 255),  # agriculture / agriculture_land
        (255, 0, 255),  # rangeland
        (255, 255, 255),  # developed space
    ),
    dtype=np.uint8,
)


# Empirically VAE-friendly shared codebook used by the new unified protocol.
# This is intentionally kept beside canonical_v1 so older checkpoints remain
# reproducible. It follows the validated OpenEarthMap magenta_cyan_v1 colors
# while mapping LoveDA's semantically corresponding classes to the same RGB.
CANONICAL_SEMANTIC_V2_PALETTE = np.asarray(
    (
        (0, 0, 0),  # background
        (222, 31, 7),  # building
        (255, 255, 255),  # road
        (0, 69, 255),  # water
        (255, 0, 255),  # barren / bareland
        (34, 97, 38),  # forest / tree
        (0, 255, 255),  # agriculture / agriculture_land
        (0, 255, 36),  # rangeland
        (148, 148, 148),  # developed space
    ),
    dtype=np.uint8,
)


# Native label IDs -> canonical semantic IDs. LoveDA ID 0 is ignore, while
# ID 1 is its evaluated background class; both use the black background code
# in the VAE target, with ID 0 still excluded by loss_mask and metrics.
LOVEDA_CANONICAL_IDS = np.asarray(
    (0, 0, 1, 2, 3, 4, 5, 6),
    dtype=np.int64,
)


LOVEDA_CANONICAL_PALETTE = CANONICAL_SEMANTIC_PALETTE[LOVEDA_CANONICAL_IDS]


LOVEDA_CANONICAL_V2_PALETTE = CANONICAL_SEMANTIC_V2_PALETTE[LOVEDA_CANONICAL_IDS]


# Keep canonical codebooks and the official display palette. The paper
# configuration explicitly selects the VAE-aware canonical_v2 codebook.
LOVEDA_CODE_PALETTE_NAME = "canonical_v1"


LOVEDA_CODE_PALETTES = {
    "canonical_v1": LOVEDA_CANONICAL_PALETTE,
    "canonical_v2": LOVEDA_CANONICAL_V2_PALETTE,
    "official_v1": LOVEDA_PALETTE,
}


LOVEDA_CODE_PALETTE = LOVEDA_CODE_PALETTES[LOVEDA_CODE_PALETTE_NAME]


# Official OpenEarthMap / DFC 2025 Track 1 class order. Label ID 0 is
# background and is trained explicitly. The official mIoU averages IDs 1..8
# only, while predictions on ID-0 pixels still count as false positives for
# the corresponding evaluated class.
OPENEARTHMAP_SAR_CLASS_NAMES = (
    "bareland",
    "rangeland",
    "developed_space",
    "road",
    "tree",
    "water",
    "agriculture_land",
    "building",
)


# OpenEarthMap native IDs -> canonical semantic IDs. This makes shared
# concepts (building, road, water, vegetation, and agriculture) use the same
# VAE RGB code as LoveDA while preserving OpenEarthMap's native metric IDs.
OPENEARTHMAP_SAR_CANONICAL_IDS = np.asarray(
    (0, 4, 7, 8, 2, 5, 3, 6, 1),
    dtype=np.int64,
)


OPENEARTHMAP_SAR_CANONICAL_PALETTE = CANONICAL_SEMANTIC_PALETTE[
    OPENEARTHMAP_SAR_CANONICAL_IDS
]


OPENEARTHMAP_SAR_CANONICAL_V2_PALETTE = CANONICAL_SEMANTIC_V2_PALETTE[
    OPENEARTHMAP_SAR_CANONICAL_IDS
]


# Keep the benchmark's official palette strictly for exported visualizations.
# The model never has to reconstruct these closely spaced colors.
OPENEARTHMAP_SAR_DISPLAY_PALETTE = np.asarray(
    (
        (0, 0, 0),  # background
        (128, 0, 0),  # bareland
        (0, 255, 36),  # rangeland
        (148, 148, 148),  # developed_space
        (255, 255, 255),  # road
        (34, 97, 38),  # tree
        (0, 69, 255),  # water
        (75, 181, 73),  # agriculture_land
        (222, 31, 7),  # building
    ),
    dtype=np.uint8,
)


OPENEARTHMAP_SAR_CODE_PALETTE_NAME = "canonical_v1"


OPENEARTHMAP_SAR_CODE_PALETTES = {
    "canonical_v1": OPENEARTHMAP_SAR_CANONICAL_PALETTE,
    "canonical_v2": OPENEARTHMAP_SAR_CANONICAL_V2_PALETTE,
    "official_v1": OPENEARTHMAP_SAR_DISPLAY_PALETTE,
}


# Public semantic codebook follows the canonical training palette. Exported
# visualizations explicitly use OPENEARTHMAP_SAR_DISPLAY_PALETTE.
OPENEARTHMAP_SAR_PALETTE = OPENEARTHMAP_SAR_CANONICAL_PALETTE


def resolve_openearthmap_sar_code_palette(name: str) -> np.ndarray:
    normalized = str(name).strip().casefold()
    try:
        return OPENEARTHMAP_SAR_CODE_PALETTES[normalized]
    except KeyError as error:
        raise ValueError(
            f"Unknown OpenEarthMap code palette {name!r}; expected one of "
            f"{tuple(OPENEARTHMAP_SAR_CODE_PALETTES)}"
        ) from error


# UrbanSARFloods native labels are 0=non-flood, 1=flooded open area and
# 2=flooded urban area. Internally they are shifted to IDs 1..3 so they can
# share the existing semantic palette helpers, where ID 0 consistently means
# ignore/no-data. Index 0 below is therefore a dummy entry; the three actual
# VAE code colours are deliberately maximally separated RGB-cube vertices.
URBANSAR_FLOOD_CLASS_NAMES = (
    "non_flood",
    "open_flood",
    "urban_flood",
)


URBANSAR_FLOOD_PALETTE = np.asarray(
    (
        (0, 0, 0),
        (0, 0, 0),
        (0, 0, 255),
        (255, 0, 0),
    ),
    dtype=np.uint8,
)


def resolve_loveda_code_palette(name: str) -> np.ndarray:
    """Return a semantic RGB codebook by its stable configuration name."""

    key = str(name).strip().casefold()
    if key not in LOVEDA_CODE_PALETTES:
        raise ValueError(
            f"Unknown LoveDA code palette {name!r}; expected one of "
            f"{tuple(LOVEDA_CODE_PALETTES)}"
        )
    return LOVEDA_CODE_PALETTES[key]


def active_tasks_from_config(cfg: dict) -> tuple[str, ...]:
    """Return the explicitly enabled tasks in a stable order."""

    configured = cfg.get("training", {}).get(
        "active_tasks",
        cfg.get("evaluation", {}).get(
            "active_tasks",
            (BUILDING_TASK, CHANGE_TASK),
        ),
    )
    if isinstance(configured, str):
        configured = [configured]
    tasks = tuple(str(task).casefold() for task in configured)
    if not tasks:
        raise ValueError(
            "training/evaluation active_tasks must contain at least one task"
        )
    unknown = sorted(set(tasks) - set(ALL_TASKS))
    if unknown:
        raise ValueError(f"Unsupported active task(s): {unknown}; expected {ALL_TASKS}")
    if len(set(tasks)) != len(tasks):
        raise ValueError(f"active_tasks contains duplicates: {tasks}")
    return tasks


_PAIRED_VIEW_INDEX = "__paired_modal_view__"
