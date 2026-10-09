"""Task--Modality Adaptive Timestep Sampling (TMAS)."""

from __future__ import annotations
import math
import torch
from accelerate import Accelerator
from uniflow_rs._compat.multitask import (
    HEIGHT_TASK,
    SAR_BUILDING_TASK,
    SAR_CHANGE_TASK,
    SAR_SEMANTIC_TASK,
    canonicalize_task_family,
)

TMAS_GROUPING_PROTOCOL = "canonical_task_family_x_modality_v1"

_ADAPTIVE_MODALITY_ALIASES = {
    "rgb": "rgb",
    "optical": "rgb",
    "optical_rgb": "rgb",
    "sar": "sar",
    "synthetic_aperture_radar": "sar",
}


def _clamped_training_progress(progress: float) -> float:
    value = float(progress)
    if not math.isfinite(value):
        raise ValueError("adaptive curriculum progress must be finite")
    return min(max(value, 0.0), 1.0)


def normalize_adaptive_modality(modality: str) -> str:
    """Map dataset modality labels to the two physical input protocols."""

    normalized = str(modality).strip().casefold().replace("-", "_")
    try:
        return _ADAPTIVE_MODALITY_ALIASES[normalized]
    except KeyError as error:
        raise ValueError(
            "Task-adaptive sampling requires one physical input modality "
            f"('rgb' or 'sar'), got {modality!r}"
        ) from error


def make_tmas_group_key(task_name: str, modality: str) -> str:
    """Return the canonical task-family by physical-modality TMAS key."""

    canonical_task = canonicalize_task_family(task_name)
    physical_modality = normalize_adaptive_modality(modality)
    return f"{canonical_task}/{physical_modality}"


def task_modality_adaptive_groups(
    active_tasks: tuple[str, ...],
    adaptive_cfg: dict,
) -> tuple[str, ...]:
    """Resolve fixed canonical-family-by-modality rows on every DDP rank."""

    configured = adaptive_cfg.get("modalities_by_task", {})
    if configured is None:
        configured = {}
    if not isinstance(configured, dict):
        raise TypeError(
            "flow.timestep_sampling.task_adaptive.modalities_by_task "
            "must be a mapping"
        )
    multimodal_defaults = {
        HEIGHT_TASK,
        SAR_BUILDING_TASK,
        SAR_SEMANTIC_TASK,
    }
    family_order: list[str] = []
    modalities_by_family: dict[str, list[str]] = {}
    for task in active_tasks:
        canonical_task = canonicalize_task_family(task)
        if canonical_task not in modalities_by_family:
            family_order.append(canonical_task)
            modalities_by_family[canonical_task] = []
        task_modalities = configured.get(task)
        if task_modalities is None:
            task_modalities = (
                ("rgb", "sar")
                if task in multimodal_defaults
                else (("sar",) if task == SAR_CHANGE_TASK else ("rgb",))
            )
        if isinstance(task_modalities, str):
            task_modalities = (task_modalities,)
        modalities = tuple(
            dict.fromkeys(
                normalize_adaptive_modality(value) for value in task_modalities
            )
        )
        if not modalities:
            raise ValueError(f"Task {task!r} has no task-adaptive input modalities")
        for modality in modalities:
            if modality not in modalities_by_family[canonical_task]:
                modalities_by_family[canonical_task].append(modality)
    return tuple(
        make_tmas_group_key(canonical_task, modality)
        for canonical_task in family_order
        for modality in modalities_by_family[canonical_task]
    )


def batch_task_modality_groups(
    batch_tasks: list[str],
    batch_modalities: list[str],
) -> list[str]:
    """Create one adaptive-table key for each training sample."""

    if len(batch_tasks) != len(batch_modalities):
        raise ValueError(
            "Task and input_modality batch lengths differ: "
            f"{len(batch_tasks)} != {len(batch_modalities)}"
        )
    return [
        make_tmas_group_key(task, modality)
        for task, modality in zip(batch_tasks, batch_modalities)
    ]


class ProgressAwareTaskAdaptiveTimestepSampler:
    """Canonical task/modality allocation driven by difficulty and progress.

    All state is kept on CPU in float64 and is updated from the same
    distributed per-bin loss aggregates as the original sampler.
    """

    def __init__(
        self,
        tasks: tuple[str, ...],
        num_bins: int = 10,
        beta_fast: float = 0.90,
        beta_slow: float = 0.99,
        gamma_difficulty: float = 1.0,
        gamma_progress: float = 1.0,
        progress_mode: str = "absolute",
        progress_floor: float = 0.1,
        progress_clip: float = 1.0,
        uniform_mix: float = 0.2,
        warmup_ratio: float = 0.1,
        epsilon: float = 1.0e-6,
    ) -> None:
        if not tasks:
            raise ValueError(
                "progress-aware task sampling requires task/modality groups"
            )
        if len(set(tasks)) != len(tasks):
            raise ValueError("progress-aware group names must be unique")
        if num_bins < 2:
            raise ValueError("progress-aware timestep num_bins must be at least 2")
        if not 0.0 <= beta_fast < 1.0:
            raise ValueError("beta_fast must be in [0, 1)")
        if not 0.0 <= beta_slow < 1.0:
            raise ValueError("beta_slow must be in [0, 1)")
        if gamma_difficulty <= 0.0 or gamma_progress <= 0.0:
            raise ValueError("progress-aware utility exponents must be positive")
        mode = str(progress_mode).strip().casefold()
        if mode not in {"absolute", "positive"}:
            raise ValueError(
                "progress_mode must be 'absolute' or 'positive', "
                f"got {progress_mode!r}"
            )
        if progress_floor <= 0.0 or not math.isfinite(progress_floor):
            raise ValueError("progress_floor must be finite and positive")
        if progress_clip <= 0.0 or not math.isfinite(progress_clip):
            raise ValueError("progress_clip must be finite and positive")
        if not 0.0 <= warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if not 0.0 <= uniform_mix <= 1.0:
            raise ValueError("uniform_mix must be in [0, 1]")
        if epsilon <= 0.0 or not math.isfinite(epsilon):
            raise ValueError("epsilon must be finite and positive")
        self.tasks = tuple(str(task) for task in tasks)
        self.task_to_index = {task: index for index, task in enumerate(self.tasks)}
        self.num_bins = int(num_bins)
        self.beta_fast = float(beta_fast)
        self.beta_slow = float(beta_slow)
        self.gamma_difficulty = float(gamma_difficulty)
        self.gamma_progress = float(gamma_progress)
        self.progress_mode = mode
        self.progress_floor = float(progress_floor)
        self.progress_clip = float(progress_clip)
        self.uniform_mix = float(uniform_mix)
        self.warmup_ratio = float(warmup_ratio)
        self.epsilon = float(epsilon)
        shape = (len(self.tasks), self.num_bins)
        self.difficulty_fast = torch.zeros(shape, dtype=torch.float64)
        self.difficulty_slow = torch.zeros(shape, dtype=torch.float64)
        self.update_count = torch.zeros(shape, dtype=torch.float64)

    def _filled_dual_difficulty(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Fill unseen bins without inventing progress for them."""

        fast = self.difficulty_fast.clone()
        slow = self.difficulty_slow.clone()
        for task_index in range(len(self.tasks)):
            seen = self.update_count[task_index] > 0
            if bool(seen.any()):
                # Use one stable group fallback for both EMAs. This keeps an
                # unseen bin's signed progress at zero rather than creating a
                # false fast/slow difference from two different means.
                fallback = slow[task_index, seen].mean()
                fast[task_index, ~seen] = fallback
                slow[task_index, ~seen] = fallback
            else:
                fast[task_index].fill_(1.0)
                slow[task_index].fill_(1.0)
        return (
            fast.clamp_min(self.epsilon),
            slow.clamp_min(self.epsilon),
        )

    def statistics(self) -> dict[str, torch.Tensor]:
        fast, slow = self._filled_dual_difficulty()
        signed = (slow - fast) / slow.clamp_min(self.epsilon)
        signed = torch.nan_to_num(signed, nan=0.0, posinf=0.0, neginf=0.0)
        if self.progress_mode == "positive":
            progress = signed.clamp_min(0.0)
        else:
            progress = signed.abs()
        progress = progress.clamp(0.0, self.progress_clip)
        difficulty_relative = slow / slow.mean(
            dim=-1,
            keepdim=True,
        ).clamp_min(self.epsilon)
        utility = (difficulty_relative + self.epsilon).pow(self.gamma_difficulty) * (
            self.progress_floor + progress
        ).pow(self.gamma_progress)
        utility = torch.nan_to_num(
            utility,
            nan=1.0,
            posinf=1.0,
            neginf=1.0,
        ).clamp_min(self.epsilon)
        return {
            "difficulty_fast": fast,
            "difficulty_slow": slow,
            "signed_progress": signed,
            "learning_progress": progress,
            "difficulty_relative": difficulty_relative,
            "utility": utility,
        }

    def probabilities(self, progress: float = 1.0) -> torch.Tensor:
        normalized_progress = _clamped_training_progress(progress)
        uniform = torch.full(
            (len(self.tasks), self.num_bins),
            1.0 / self.num_bins,
            dtype=torch.float64,
        )
        if normalized_progress <= self.warmup_ratio:
            return uniform
        stats = self.statistics()
        utility = stats["utility"]
        q = utility / utility.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(self.epsilon)
        probabilities = self.uniform_mix * uniform + (1.0 - self.uniform_mix) * q
        return probabilities / probabilities.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(self.epsilon)

    def progress_snapshot(self, progress: float) -> dict[str, float | str]:
        normalized_progress = _clamped_training_progress(progress)
        return {
            "mode": "progress_aware",
            "phase": (
                "uniform"
                if normalized_progress <= self.warmup_ratio
                else "progress_aware"
            ),
            "progress": normalized_progress,
            "warmup_ratio": self.warmup_ratio,
            "progress_mode": self.progress_mode,
        }

    def sample(
        self,
        batch_groups: list[str],
        device: torch.device,
        progress: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        try:
            task_indices = torch.tensor(
                [self.task_to_index[str(group)] for group in batch_groups],
                device=device,
                dtype=torch.long,
            )
        except KeyError as error:
            raise ValueError(
                "Progress-aware sampler received unknown task/modality "
                f"group {error.args[0]!r}"
            ) from error
        all_probabilities = self.probabilities(progress).to(
            device=device,
            dtype=torch.float32,
        )
        sample_probabilities = all_probabilities.index_select(
            0,
            task_indices,
        )
        bin_ids = torch.multinomial(
            sample_probabilities,
            num_samples=1,
            replacement=True,
        ).squeeze(1)
        offsets = torch.rand(len(batch_groups), device=device)
        timesteps = (bin_ids.float() + offsets) / self.num_bins
        return timesteps, bin_ids, task_indices

    def update_from_reduced(
        self,
        loss_sums: torch.Tensor,
        counts: torch.Tensor,
    ) -> None:
        loss_sums = loss_sums.detach().double().cpu()
        counts = counts.detach().double().cpu()
        expected_shape = (len(self.tasks), self.num_bins)
        if tuple(loss_sums.shape) != expected_shape:
            raise ValueError("progress-aware loss_sums has wrong shape")
        if tuple(counts.shape) != expected_shape:
            raise ValueError("progress-aware counts have wrong shape")
        observed = counts > 0
        means = loss_sums / counts.clamp_min(1.0)
        first = observed & (self.update_count == 0)
        repeated = observed & ~first
        self.difficulty_fast[first] = means[first]
        self.difficulty_slow[first] = means[first]
        self.difficulty_fast[repeated] = (
            self.beta_fast * self.difficulty_fast[repeated]
            + (1.0 - self.beta_fast) * means[repeated]
        )
        self.difficulty_slow[repeated] = (
            self.beta_slow * self.difficulty_slow[repeated]
            + (1.0 - self.beta_slow) * means[repeated]
        )
        self.update_count[observed] += counts[observed]

    def state_dict(self) -> dict:
        return {
            "tasks": list(self.tasks),
            "grouping": TMAS_GROUPING_PROTOCOL,
            "sampler": "progress_aware_dual_ema",
            "num_bins": self.num_bins,
            "beta_fast": self.beta_fast,
            "beta_slow": self.beta_slow,
            "gamma_difficulty": self.gamma_difficulty,
            "gamma_progress": self.gamma_progress,
            "progress_mode": self.progress_mode,
            "progress_floor": self.progress_floor,
            "progress_clip": self.progress_clip,
            "uniform_mix": self.uniform_mix,
            "warmup_ratio": self.warmup_ratio,
            "epsilon": self.epsilon,
            "difficulty_fast": self.difficulty_fast.tolist(),
            "difficulty_slow": self.difficulty_slow.tolist(),
            "update_count": self.update_count.tolist(),
        }

    def load_state_dict(self, state: dict) -> None:
        saved_tasks = tuple(str(task) for task in state["tasks"])
        saved_grouping = state.get("grouping")
        if saved_grouping != TMAS_GROUPING_PROTOCOL:
            raise ValueError(
                "TMAS checkpoint grouping protocol is incompatible: "
                f"saved={saved_grouping!r}, "
                f"current={TMAS_GROUPING_PROTOCOL!r}. Internal-task x "
                "modality statistics cannot be silently merged into "
                "canonical task-family x modality statistics. Use "
                "--init-adapter for model-only initialization, or start a "
                "new run with a clean TMAS state."
            )
        if int(state["num_bins"]) != self.num_bins:
            raise ValueError(
                "Progress-aware checkpoint num_bins does not match config: "
                f"{state['num_bins']} != {self.num_bins}"
            )
        # A legacy task-adaptive checkpoint has one EMA. Treat it as both
        # fast and slow so resume remains safe without inventing progress.
        if "difficulty_fast" in state and "difficulty_slow" in state:
            fast = torch.as_tensor(state["difficulty_fast"], dtype=torch.float64)
            slow = torch.as_tensor(state["difficulty_slow"], dtype=torch.float64)
        else:
            legacy = torch.as_tensor(state["ema_loss"], dtype=torch.float64)
            fast = legacy
            slow = legacy.clone()
        counts = torch.as_tensor(state["update_count"], dtype=torch.float64)
        saved_shape = (len(saved_tasks), self.num_bins)
        if tuple(fast.shape) != saved_shape or tuple(slow.shape) != saved_shape:
            raise ValueError("Invalid progress-aware EMA state shape")
        if tuple(counts.shape) != saved_shape:
            raise ValueError("Invalid progress-aware count state shape")
        if saved_tasks != self.tasks:
            raise ValueError(
                "Canonical TMAS checkpoint group namespace does not match "
                f"the current config: {saved_tasks} != {self.tasks}. "
                "Profiles are not reordered or merged implicitly."
            )
        self.difficulty_fast.copy_(fast)
        self.difficulty_slow.copy_(slow)
        self.update_count.copy_(counts)


def update_task_adaptive_sampler(
    sampler: ProgressAwareTaskAdaptiveTimestepSampler,
    task_indices: torch.Tensor,
    bin_ids: torch.Tensor,
    per_sample_loss: torch.Tensor,
    include_mask: torch.Tensor,
    accelerator: Accelerator,
) -> None:
    """Synchronize every task/bin statistic across ranks in one collective."""

    valid = include_mask.bool() & torch.isfinite(per_sample_loss.detach())
    selected_tasks = task_indices.detach()[valid]
    selected_bins = bin_ids.detach()[valid]
    selected_losses = per_sample_loss.detach().float()[valid]
    table_size = len(sampler.tasks) * sampler.num_bins
    loss_sums = torch.zeros(
        table_size,
        device=per_sample_loss.device,
        dtype=torch.float32,
    )
    counts = torch.zeros_like(loss_sums)
    if selected_bins.numel() > 0:
        flat_indices = selected_tasks * sampler.num_bins + selected_bins
        loss_sums.scatter_add_(0, flat_indices, selected_losses)
        counts.scatter_add_(
            0,
            flat_indices,
            torch.ones_like(selected_losses),
        )
    reduced = accelerator.reduce(
        torch.cat((loss_sums, counts)),
        reduction="sum",
    )
    loss_sums, counts = reduced.split(table_size)
    shape = (len(sampler.tasks), sampler.num_bins)
    sampler.update_from_reduced(
        loss_sums.reshape(shape),
        counts.reshape(shape),
    )
