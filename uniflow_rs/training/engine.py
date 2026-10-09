"""Shared-LoRA training loop with frozen backbone, VAE, and text encoder."""

from __future__ import annotations
import json
import logging
import math
import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from uniflow_rs._compat.deventer import make_image_ids
from uniflow_rs._compat.backbone import (
    load_backbone_autoencoder,
    load_backbone_transformer,
    resolve_backbone,
)
from uniflow_rs._compat.deventer_flow import (
    encode_multitask_inputs,
    multitask_flow_matching_output,
    pack_pixel_loss_mask,
    select_prompt_condition,
    task_condition_modes,
    task_flow_sources,
)
from uniflow_rs._compat.lora import LoRAConfig, inject_lora, load_lora_checkpoint
from uniflow_rs._compat.multitask import (
    OPENEARTHMAP_SAR_CODE_PALETTE_NAME,
    SAR_SEMANTIC_TASK,
    active_tasks_from_config,
    resolve_openearthmap_sar_code_palette,
)
from .tmas import (
    ProgressAwareTaskAdaptiveTimestepSampler,
    TMAS_GROUPING_PROTOCOL,
    batch_task_modality_groups,
    task_modality_adaptive_groups,
    update_task_adaptive_sampler,
)
from .data import (
    build_dataloaders,
    load_prompt_conditions,
    required_fixed_prompt_keys,
)
from .checkpoints import (
    _save_adapter,
    make_run_dir,
    prune_checkpoints,
    resolve_resume,
)
from .cli import (
    _resolved_optional_path,
    apply_experiment_overrides,
    load_config,
    parse_args,
    validate_method_config,
)

LOGGER = logging.getLogger("uniflow_rs.train")


def main() -> None:
    args = parse_args()
    cfg = apply_experiment_overrides(load_config(args.config), args)
    training_cfg = cfg["training"]
    validate_method_config(cfg)
    active_tasks = active_tasks_from_config(cfg)
    model_cfg = cfg["model"]
    sar_semantic_code_palette = str(
        cfg["data"]
        .get(SAR_SEMANTIC_TASK, {})
        .get("code_palette", OPENEARTHMAP_SAR_CODE_PALETTE_NAME)
    )
    resolve_openearthmap_sar_code_palette(sar_semantic_code_palette)
    required_prompts = required_fixed_prompt_keys(cfg, active_tasks)
    missing_prompts = sorted(required_prompts - set(cfg["model"]["prompts"]))
    if missing_prompts:
        raise KeyError(
            f"Missing model.prompts entries required by active task sources: {missing_prompts}"
        )
    accelerator_kwargs = {}
    accelerator = Accelerator(
        gradient_accumulation_steps=int(
            training_cfg.get("gradient_accumulation_steps", 1)
        ),
        mixed_precision=str(training_cfg.get("mixed_precision", "bf16")),
        **accelerator_kwargs,
    )
    logging.basicConfig(
        level=logging.INFO if accelerator.is_main_process else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    set_seed(int(training_cfg.get("seed", 0)), device_specific=True)
    torch.backends.cuda.matmul.allow_tf32 = bool(training_cfg.get("allow_tf32", True))
    pretrained_cfg = cfg.get("pretrained", {})
    flux_model_dir = _resolved_optional_path(
        args.flux_model_dir, pretrained_cfg.get("local_dir")
    )
    text_model_dir = _resolved_optional_path(
        args.text_model_dir, pretrained_cfg.get("text_model_dir")
    )
    with accelerator.main_process_first():
        backbone = resolve_backbone(cfg, root_model_dir=flux_model_dir)
    resume_path = resolve_resume(args, cfg)
    if resume_path is not None and args.init_adapter is not None:
        raise ValueError("Use either --resume or --init-adapter, not both")
    run_dir = make_run_dir(accelerator, cfg)
    checkpoint_dir = run_dir / "checkpoints"
    if accelerator.is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "config.yaml").open("w", encoding="utf-8") as file:
            yaml.safe_dump(cfg, file, sort_keys=False, allow_unicode=True)
        LOGGER.info("Run directory: %s", run_dir)
        LOGGER.info("Active training tasks: %s", active_tasks)
        LOGGER.info(
            "Backbone: %s | transformer: %s", backbone.name, backbone.files.transformer
        )
        LOGGER.info("Backbone autoencoder: %s", backbone.files.autoencoder)
    prompt_bank = load_prompt_conditions(
        accelerator, cfg, text_model_dir
    )
    autoencoder = (
        load_backbone_autoencoder(backbone, device=accelerator.device)
        .eval()
        .requires_grad_(False)
    )
    if any((parameter.requires_grad for parameter in autoencoder.parameters())):
        raise RuntimeError("The VAE must remain frozen during shared-LoRA training")
    if accelerator.is_main_process:
        LOGGER.info("VAE encoder and decoder: frozen pretrained")
    transformer = load_backbone_transformer(backbone, device=accelerator.device)
    transformer.requires_grad_(False)
    if accelerator.is_main_process:
        LOGGER.info("MMDiT backbone: frozen")
    if str(model_cfg.get("train_mode", "lora")).casefold() != "lora":
        raise ValueError(
            "The shared FLUX MMDiT entry point supports LoRA training. Full fine-tuning requires a separate FSDP/ZeRO-3 setup."
        )
    configured_lora_cfg = LoRAConfig(
        rank=int(model_cfg.get("lora_rank", 32)),
        alpha=float(model_cfg.get("lora_alpha", 32.0)),
        dropout=float(model_cfg.get("lora_dropout", 0.0)),
    )
    lora_cfg = configured_lora_cfg
    resume_state = None
    adapter_path = (
        resume_path
        if resume_path is not None
        else (
            args.init_adapter.expanduser().resolve()
            if args.init_adapter is not None
            else None
        )
    )
    if adapter_path is None:
        replaced = inject_lora(
            transformer,
            rank=lora_cfg.rank,
            alpha=lora_cfg.alpha,
            dropout=lora_cfg.dropout,
        )
    else:
        raw_adapter = torch.load(adapter_path, map_location="cpu", weights_only=False)
        saved_training_state = raw_adapter.get("training_state", {})
        if (
            isinstance(saved_training_state, dict)
            and saved_training_state.get("task_family_heads") is not None
        ):
            raise ValueError(
                "This training entry point only supports the shared prediction head, not task-family-head checkpoints"
            )
        saved_vae_decoder = (
            saved_training_state.get("vae_decoder")
            if isinstance(saved_training_state, dict)
            else None
        )
        if saved_vae_decoder is not None:
            if resume_path is not None:
                raise ValueError(
                    "Checkpoint contains a trainable VAE decoder but the current config disables it. Use --init-adapter to load only LoRA weights or use a training implementation supporting decoder fine-tuning."
                )
            if accelerator.is_main_process:
                LOGGER.warning(
                    "Ignoring tuned VAE decoder in initialization checkpoint; this release supports only the frozen pretrained VAE"
                )
        saved_lora = raw_adapter["lora"]
        saved_lora_cfg = LoRAConfig(
            rank=int(saved_lora["rank"]),
            alpha=float(saved_lora["alpha"]),
            dropout=float(saved_lora.get("dropout", 0.0)),
        )
        if resume_path is not None:
            lora_cfg = saved_lora_cfg
            target_lora_cfg = saved_lora_cfg
            allow_rank_expansion = False
        else:
            lora_cfg = configured_lora_cfg
            target_lora_cfg = configured_lora_cfg
            allow_rank_expansion = configured_lora_cfg.rank > saved_lora_cfg.rank
        source_variant = str(raw_adapter.get("model_variant", "uniflow-rs"))
        load_lora_checkpoint(
            adapter_path,
            transformer,
            inject=True,
            expected_base_model=backbone.base_model,
            expected_model_variant=backbone.model_variant,
            adapt_single_image_condition=args.init_adapter is not None
            and source_variant == "uniflow-rs",
            target_config=target_lora_cfg,
            allow_rank_expansion=allow_rank_expansion,
            checkpoint_state=raw_adapter,
        )
        replaced = [
            name
            for (name, module) in transformer.named_modules()
            if module.__class__.__name__ == "LoRALinear"
        ]
        if resume_path is not None:
            resume_state = raw_adapter
        elif accelerator.is_main_process:
            LOGGER.info(
                "Initialized from %s adapter without resuming optimizer state",
                source_variant,
            )
            if configured_lora_cfg.rank > saved_lora_cfg.rank:
                LOGGER.info(
                    "Losslessly expanded LoRA rank %d -> %d (alpha %.6g -> %.6g)",
                    saved_lora_cfg.rank,
                    configured_lora_cfg.rank,
                    saved_lora_cfg.alpha,
                    configured_lora_cfg.alpha,
                )
        if resume_path is None:
            del raw_adapter
    if bool(model_cfg.get("gradient_checkpointing", True)):
        transformer.enable_gradient_checkpointing()
    transformer.train()
    lora_trainable_parameters = [
        parameter
        for (name, parameter) in transformer.named_parameters()
        if parameter.requires_grad
        and (name.endswith("lora_A") or name.endswith("lora_B"))
    ]
    if not lora_trainable_parameters:
        raise RuntimeError("No trainable LoRA parameters")
    unexpected_trainable = [
        name
        for (name, parameter) in transformer.named_parameters()
        if parameter.requires_grad
        and (not (name.endswith("lora_A") or name.endswith("lora_B")))
    ]
    if unexpected_trainable:
        raise RuntimeError(
            f"Unexpected trainable MMDiT parameters outside shared LoRA: {unexpected_trainable[:10]}"
        )
    trainable_parameters = lora_trainable_parameters
    if accelerator.is_main_process:
        LOGGER.info("LoRA wrapped %d linear layers", len(replaced))
        LOGGER.info(
            "Trainable shared LoRA parameters: %s",
            f"{sum((parameter.numel() for parameter in lora_trainable_parameters)):,}",
        )
        LOGGER.info(
            "Total trainable parameters: %s",
            f"{sum((parameter.numel() for parameter in trainable_parameters)):,}",
        )
        LOGGER.info("Shared LoRA: trainable")
    if len(trainable_parameters) != len(lora_trainable_parameters) or any(
        (
            id(left) != id(right)
            for (left, right) in zip(
                trainable_parameters, lora_trainable_parameters, strict=True
            )
        )
    ):
        raise RuntimeError(
            "Frozen-VAE Stage 2 optimizer must contain only LoRA parameters"
        )
    optimizer_parameter_groups = [
        {
            "params": lora_trainable_parameters,
            "lr": float(training_cfg["learning_rate"]),
            "weight_decay": float(training_cfg.get("weight_decay", 0.01)),
            "name": "lora",
        }
    ]
    optimizer = AdamW(optimizer_parameter_groups)
    scheduler = MultiStepLR(
        optimizer,
        milestones=[int(step) for step in training_cfg.get("lr_milestones", [])],
        gamma=float(training_cfg.get("lr_gamma", 0.5)),
    )
    train_loader, sizes = build_dataloaders(cfg, active_tasks)
    transformer, optimizer, train_loader = accelerator.prepare(
        transformer, optimizer, train_loader
    )
    if accelerator.is_main_process:
        configured_resolutions = {
            task: int(
                cfg["data"]
                .get(task, {})
                .get("resolution", cfg["data"].get("resolution", 512))
            )
            for task in active_tasks
        }
        LOGGER.info("Dataset sizes: %s", sizes)
        LOGGER.info(
            "Task-homogeneous microbatches: %s | resolutions: %s | per-device batch sizes: %s",
            bool(training_cfg.get("pair_modalities_in_batch", False))
            or bool(training_cfg.get("task_homogeneous_batches", False))
            or len(set(configured_resolutions.values())) > 1,
            configured_resolutions,
            {
                task: int(
                    training_cfg.get("batch_size_per_gpu_by_task", {}).get(
                        task, training_cfg["batch_size_per_gpu"]
                    )
                )
                for task in active_tasks
            },
        )
    start_epoch = 0
    global_step = 0
    if resume_state is not None:
        start_epoch = int(resume_state.get("epoch", -1)) + 1
        global_step = int(resume_state.get("global_step", 0))
        if "optimizer" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer"])
        if "scheduler" in resume_state:
            scheduler.load_state_dict(resume_state["scheduler"])
    writer = (
        SummaryWriter(run_dir / "tensorboard") if accelerator.is_main_process else None
    )
    epochs = int(training_cfg["epochs"])
    configured_max_steps = training_cfg.get("max_optimizer_steps")
    max_optimizer_steps = (
        int(configured_max_steps)
        if configured_max_steps not in (None, "", False)
        else None
    )
    gradient_accumulation_steps = int(
        training_cfg.get("gradient_accumulation_steps", 1)
    )
    optimizer_steps_per_epoch = max(
        1, math.ceil(len(train_loader) / gradient_accumulation_steps)
    )
    planned_optimizer_steps = (
        max_optimizer_steps
        if max_optimizer_steps is not None
        else epochs * optimizer_steps_per_epoch
    )
    save_every = int(training_cfg.get("save_every_epochs", 5))
    beta_alpha = float(cfg["flow"]["beta_alpha"])
    beta_beta = float(cfg["flow"].get("beta_beta", 1.0))
    timestep_sampling_cfg = dict(cfg["flow"].get("timestep_sampling", {}))
    timestep_sampling_mode = str(timestep_sampling_cfg.get("mode", "beta")).lower()
    if timestep_sampling_mode not in {"beta", "uniform", "task_adaptive_progress"}:
        raise ValueError(
            f"flow.timestep_sampling.mode must be beta, uniform, or task_adaptive_progress; got {timestep_sampling_mode!r}"
        )
    progress_task_adaptive_sampler = None
    if timestep_sampling_mode == "task_adaptive_progress":
        progress_cfg = dict(timestep_sampling_cfg.get("task_adaptive_progress", {}))
        progress_task_adaptive_sampler = ProgressAwareTaskAdaptiveTimestepSampler(
            tasks=task_modality_adaptive_groups(active_tasks, progress_cfg),
            num_bins=int(progress_cfg.get("num_bins", 10)),
            beta_fast=float(progress_cfg.get("beta_fast", 0.9)),
            beta_slow=float(progress_cfg.get("beta_slow", 0.99)),
            gamma_difficulty=float(progress_cfg.get("gamma_difficulty", 1.0)),
            gamma_progress=float(progress_cfg.get("gamma_progress", 1.0)),
            progress_mode=str(progress_cfg.get("progress_mode", "absolute")),
            progress_floor=float(progress_cfg.get("progress_floor", 0.1)),
            progress_clip=float(progress_cfg.get("progress_clip", 1.0)),
            uniform_mix=float(progress_cfg.get("uniform_mix", 0.2)),
            warmup_ratio=float(progress_cfg.get("warmup_ratio", 0.1)),
            epsilon=float(progress_cfg.get("epsilon", 1e-06)),
        )
        if resume_state is not None:
            saved_training_state = resume_state.get("training_state", {})
            saved_sampler_state = saved_training_state.get(
                "task_adaptive_progress_timestep"
            )
            legacy_state = saved_training_state.get("task_adaptive_timestep")
            if saved_sampler_state is None:
                saved_sampler_state = legacy_state
            if saved_sampler_state is not None:
                progress_task_adaptive_sampler.load_state_dict(saved_sampler_state)
                if accelerator.is_main_process:
                    LOGGER.info("Restored progress-aware task-adaptive timestep state")
            elif accelerator.is_main_process:
                LOGGER.warning(
                    "Resume checkpoint has no progress-aware timestep state; starting every task table from uniform"
                )
    latent_difference_scale = float(cfg["flow"].get("latent_difference_scale", 1.0))
    max_grad_norm = float(training_cfg.get("max_grad_norm", 1.0))
    if accelerator.is_main_process:
        LOGGER.info(
            "TMAS: %s",
            (
                "enabled (progress-aware canonical task-family x modality adaptive)"
                if progress_task_adaptive_sampler is not None
                else "disabled"
            ),
        )
        grouped_sampler = progress_task_adaptive_sampler
        if grouped_sampler is not None:
            LOGGER.info(
                "TMAS canonical group namespace (%d): %s",
                len(grouped_sampler.tasks),
                ", ".join(grouped_sampler.tasks),
            )
        if progress_task_adaptive_sampler is None:
            LOGGER.info(
                "Random Flow timestep sampling: %s (alpha=%.6g, beta=%.6g)",
                (
                    "Uniform(0,1)"
                    if timestep_sampling_mode == "uniform" or beta_alpha == 1.0
                    else "Beta(alpha,1)"
                ),
                beta_alpha,
                beta_beta,
            )
        elif progress_task_adaptive_sampler is not None:
            progress_cfg = progress_task_adaptive_sampler
            LOGGER.info(
                "Random Flow timestep sampling: Progress-aware Task x Modality Adaptive (%d bins, beta_fast %.4f, beta_slow %.4f, gamma_d %.4g, gamma_p %.4g, progress %s, uniform mix %.3f, warmup %.3f); flow-loss samples and all sampled timesteps contribute to updates",
                progress_cfg.num_bins,
                progress_cfg.beta_fast,
                progress_cfg.beta_slow,
                progress_cfg.gamma_difficulty,
                progress_cfg.gamma_progress,
                progress_cfg.progress_mode,
                progress_cfg.uniform_mix,
                progress_cfg.warmup_ratio,
            )
        LOGGER.info(
            "Flow sources by task: %s; condition modes: %s (latent difference scale %.6g)",
            {task: task_flow_sources((task,), cfg["flow"])[0] for task in active_tasks},
            {
                task: task_condition_modes((task,), cfg["flow"])[0]
                for task in active_tasks
            },
            latent_difference_scale,
        )
    try:
        stop_training = (
            max_optimizer_steps is not None and global_step >= max_optimizer_steps
        )
        for epoch in range(start_epoch, epochs):
            set_loader_epoch = getattr(train_loader, "set_epoch", None)
            if callable(set_loader_epoch):
                set_loader_epoch(epoch)
            else:
                set_batch_epoch = getattr(
                    getattr(train_loader, "batch_sampler", None), "set_epoch", None
                )
                if callable(set_batch_epoch):
                    set_batch_epoch(epoch)
            epoch_loss = torch.zeros(3, device=accelerator.device, dtype=torch.float64)
            progress = tqdm(
                train_loader,
                desc=(
                    f"joint({len(active_tasks)} tasks)"
                    if len(active_tasks) > 3
                    else "+".join(active_tasks)
                )
                + f" epoch {epoch + 1}/{epochs}",
                disable=not accelerator.is_local_main_process,
            )
            for batch in progress:
                if stop_training:
                    break
                before = batch["before"].to(accelerator.device, non_blocking=True)
                after = batch["after"].to(accelerator.device, non_blocking=True)
                target_rgb = batch["target_rgb"].to(
                    accelerator.device, non_blocking=True
                )
                pixel_loss_mask = batch["loss_mask"].to(
                    accelerator.device, non_blocking=True
                )
                batch_size = before.shape[0]
                batch_tasks = list(batch["task"])
                batch_adaptive_groups = (
                    batch_task_modality_groups(
                        batch_tasks,
                        list(batch.get("input_modality", ["rgb"] * batch_size)),
                    )
                    if progress_task_adaptive_sampler is not None
                    else None
                )
                batch_flow_sources = task_flow_sources(batch_tasks, cfg["flow"])
                batch_condition_modes = task_condition_modes(batch_tasks, cfg["flow"])
                with accelerator.accumulate(transformer):
                    with torch.no_grad(), accelerator.autocast():
                        before_tokens, after_tokens, mask_tokens, after_latent_map = (
                            encode_multitask_inputs(
                                autoencoder,
                                before,
                                after,
                                target_rgb,
                                flow_source=batch_flow_sources,
                                condition_mode=batch_condition_modes,
                            )
                        )
                        image_ids = make_image_ids(
                            batch_size,
                            after_latent_map.shape[-2],
                            after_latent_map.shape[-1],
                            accelerator.device,
                        )
                        txt, vec, txt_ids = select_prompt_condition(
                            prompt_bank,
                            list(batch.get("prompt_key", batch_tasks)),
                            accelerator.device,
                            prompt_indices=batch.get("prompt_index"),
                        )
                    with accelerator.autocast():
                        flow_arguments = dict(
                            transformer=transformer,
                            before_tokens=before_tokens,
                            after_tokens=after_tokens,
                            mask_tokens=mask_tokens,
                            image_ids=image_ids,
                            txt=txt,
                            txt_ids=txt_ids,
                            vec=vec,
                            beta_alpha=beta_alpha,
                            flow_source=batch_flow_sources,
                            condition_mode=batch_condition_modes,
                            latent_difference_scale=latent_difference_scale,
                            token_loss_weights=pack_pixel_loss_mask(
                                pixel_loss_mask,
                                after_latent_map.shape[-2],
                                after_latent_map.shape[-1],
                            ),
                        )
                        if progress_task_adaptive_sampler is None:
                            flow_loss, estimated_target_tokens, _ = (
                                multitask_flow_matching_output(**flow_arguments)
                            )
                            adaptive_bin_ids = None
                            adaptive_task_indices = None
                            adaptive_per_sample_loss = None
                        else:
                            adaptive_curriculum_progress = min(
                                max(global_step / max(planned_optimizer_steps, 1), 0.0),
                                1.0,
                            )
                            assert progress_task_adaptive_sampler is not None
                            assert batch_adaptive_groups is not None
                            (
                                sampled_timesteps,
                                adaptive_bin_ids,
                                adaptive_task_indices,
                            ) = progress_task_adaptive_sampler.sample(
                                batch_adaptive_groups,
                                accelerator.device,
                                progress=adaptive_curriculum_progress,
                            )
                            (
                                flow_loss,
                                estimated_target_tokens,
                                _,
                                adaptive_per_sample_loss,
                            ) = multitask_flow_matching_output(
                                **flow_arguments,
                                fixed_timesteps=sampled_timesteps,
                                return_per_sample_loss=True,
                            )
                        loss = flow_loss
                    if progress_task_adaptive_sampler is not None:
                        assert adaptive_task_indices is not None
                        assert adaptive_bin_ids is not None
                        assert adaptive_per_sample_loss is not None
                        update_task_adaptive_sampler(
                            progress_task_adaptive_sampler,
                            adaptive_task_indices,
                            adaptive_bin_ids,
                            adaptive_per_sample_loss,
                            torch.ones_like(adaptive_per_sample_loss, dtype=torch.bool),
                            accelerator,
                        )
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(trainable_parameters, max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if accelerator.sync_gradients:
                        scheduler.step()
                        global_step += 1
                        if (
                            max_optimizer_steps is not None
                            and global_step >= max_optimizer_steps
                        ):
                            stop_training = True
                epoch_loss[0] += loss.detach().double() * batch_size
                epoch_loss[1] += flow_loss.detach().double() * batch_size
                epoch_loss[2] += batch_size
                progress_values = {
                    "loss": f"{loss.item():.5f}",
                    "flow": f"{flow_loss.item():.5f}",
                    "step": global_step,
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                }
                progress.set_postfix(**progress_values)
            epoch_loss = accelerator.reduce(epoch_loss, reduction="sum")
            mean_total_loss = float(epoch_loss[0] / epoch_loss[2].clamp_min(1.0))
            mean_flow_loss = float(epoch_loss[1] / epoch_loss[2].clamp_min(1.0))
            if writer is not None:
                writer.add_scalar(
                    "train/latent_velocity_loss", mean_flow_loss, global_step
                )
                writer.add_scalar("train/total_loss", mean_total_loss, global_step)
                writer.add_scalar(
                    "train/learning_rate", optimizer.param_groups[0]["lr"], global_step
                )
            if accelerator.is_main_process:
                LOGGER.info(
                    "Epoch %d/%d | total %.8f flow %.8f | optimizer step %d | lr %.3e",
                    epoch + 1,
                    epochs,
                    mean_total_loss,
                    mean_flow_loss,
                    global_step,
                    optimizer.param_groups[0]["lr"],
                )
            training_state = None
            adaptive_curriculum_progress = min(
                max(global_step / max(planned_optimizer_steps, 1), 0.0), 1.0
            )
            if progress_task_adaptive_sampler is not None:
                progress_sampler = progress_task_adaptive_sampler
                progress_stats = progress_sampler.statistics()
                adaptive_probabilities = progress_sampler.probabilities(
                    adaptive_curriculum_progress
                )
                progress_snapshot = progress_sampler.progress_snapshot(
                    adaptive_curriculum_progress
                )
                training_state = {
                    "task_adaptive_progress_timestep": progress_sampler.state_dict()
                }
                if writer is not None:
                    writer.add_scalar(
                        "timestep_sampling/progress",
                        float(progress_snapshot["progress"]),
                        global_step,
                    )
                    for task_index, group in enumerate(progress_sampler.tasks):
                        safe_group = group.replace("/", "_")
                        for bin_index in range(progress_sampler.num_bins):
                            prefix = f"timestep_sampling/progress/{safe_group}"
                            writer.add_scalar(
                                f"{prefix}/difficulty_fast_bin_{bin_index:02d}",
                                float(
                                    progress_stats["difficulty_fast"][
                                        task_index, bin_index
                                    ]
                                ),
                                global_step,
                            )
                            writer.add_scalar(
                                f"{prefix}/difficulty_slow_bin_{bin_index:02d}",
                                float(
                                    progress_stats["difficulty_slow"][
                                        task_index, bin_index
                                    ]
                                ),
                                global_step,
                            )
                            writer.add_scalar(
                                f"{prefix}/signed_progress_bin_{bin_index:02d}",
                                float(
                                    progress_stats["signed_progress"][
                                        task_index, bin_index
                                    ]
                                ),
                                global_step,
                            )
                            writer.add_scalar(
                                f"{prefix}/progress_bin_{bin_index:02d}",
                                float(
                                    progress_stats["learning_progress"][
                                        task_index, bin_index
                                    ]
                                ),
                                global_step,
                            )
                            writer.add_scalar(
                                f"{prefix}/difficulty_relative_bin_{bin_index:02d}",
                                float(
                                    progress_stats["difficulty_relative"][
                                        task_index, bin_index
                                    ]
                                ),
                                global_step,
                            )
                            writer.add_scalar(
                                f"{prefix}/utility_bin_{bin_index:02d}",
                                float(progress_stats["utility"][task_index, bin_index]),
                                global_step,
                            )
                            writer.add_scalar(
                                f"{prefix}/prob_bin_{bin_index:02d}",
                                float(adaptive_probabilities[task_index, bin_index]),
                                global_step,
                            )
                            writer.add_scalar(
                                f"{prefix}/sample_count_bin_{bin_index:02d}",
                                float(
                                    progress_sampler.update_count[task_index, bin_index]
                                ),
                                global_step,
                            )
                if accelerator.is_main_process:
                    probability_snapshot = {}
                    statistics_snapshot = {}
                    for task_index, group in enumerate(progress_sampler.tasks):
                        probability_snapshot[group] = adaptive_probabilities[
                            task_index
                        ].tolist()
                        statistics_snapshot[group] = {
                            name: value[task_index].tolist()
                            for (name, value) in progress_stats.items()
                        }
                        statistics_snapshot[group]["sample_count"] = (
                            progress_sampler.update_count[task_index].tolist()
                        )
                        LOGGER.info(
                            "[P-TMAS] %s probability=[%s] progress=[%s]",
                            group,
                            ", ".join(
                                (
                                    f"{value:.4f}"
                                    for value in probability_snapshot[group]
                                )
                            ),
                            ", ".join(
                                (
                                    f"{value:.4f}"
                                    for value in statistics_snapshot[group][
                                        "learning_progress"
                                    ]
                                )
                            ),
                        )
                    snapshot = {
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "grouping": TMAS_GROUPING_PROTOCOL,
                        "bin_edges": [
                            index / progress_sampler.num_bins
                            for index in range(progress_sampler.num_bins + 1)
                        ],
                        "probabilities": probability_snapshot,
                        "statistics": statistics_snapshot,
                        "progress": progress_snapshot,
                        "sampler": progress_sampler.state_dict(),
                    }
                    (run_dir / "task_adaptive_progress_timestep.json").write_text(
                        json.dumps(snapshot, indent=2), encoding="utf-8"
                    )
            training_state = dict(training_state or {})
            if (epoch + 1) % save_every == 0 or epoch + 1 == epochs or stop_training:
                if accelerator.is_main_process:
                    path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt"
                    _save_adapter(
                        path,
                        accelerator,
                        transformer,
                        optimizer,
                        scheduler,
                        epoch,
                        global_step,
                        lora_cfg,
                        training_state,
                        base_model=backbone.base_model,
                        model_variant=backbone.model_variant,
                    )
                    prune_checkpoints(
                        checkpoint_dir, int(training_cfg.get("max_checkpoints", 3))
                    )
                    LOGGER.info("Saved %s", path.name)
            accelerator.wait_for_everyone()
            if stop_training:
                if accelerator.is_main_process:
                    LOGGER.info(
                        "Reached max_optimizer_steps=%d; training complete",
                        max_optimizer_steps,
                    )
                break
    finally:
        if writer is not None:
            writer.close()
