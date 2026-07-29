from __future__ import annotations

import argparse
import bisect
import math
import multiprocessing as mp
import json
import random
import re
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from evaluation.harmony_similarity import (
    generic_dtw_alignment,
    infer_harmony,
    parse_bar_notes,
    pitch_class_similarity,
    token_similarity,
)
from evaluation.rewards import (
    _extract_header_context,
    _extract_stream_line_features,
    _stream_line_local_metrics,
    score_candidate_text_with_local_metrics,
)
from notagen_runtime.notagen_cached_generation_batch import sample_completions_cached_batch
from notagen_runtime.notagen_replay import (
    PATCH_SIZE,
    _encoded_last_patch,
    autocast_context,
    batched_tail_encoded_targets,
    char_patch_token_logprobs_dists_and_counts,
    normalize_patch_for_context,
    split_tensor_by_counts,
    tail_encoded_targets,
)
from scripts.custom_grpo_notagen import (
    PATCH_STREAM,
    GoldbergRewardConfig,
    ModelShape,
    RolloutSample,
    SimilarityReference,
    SimilarityRewardWeights,
    _rollout_seed,
    build_model,
    build_rollout_prefix,
    count_stream_lines,
    disable_dropout_modules,
    generated_token_slots,
    grpo_kl_term,
    infer_model_shape,
    load_policy_checkpoint,
    load_prompt_rows,
    load_similarity_reference,
    load_structural_target,
    prompt_row_name,
    sample_completion,
    score_prompt_completion_pair,
    score_similarity_reward,
    select_device,
    set_seed,
)
from scripts.notagen_ppo_diagnostics import (
    aggregate_component_sums,
    advantage_distribution_summary,
    component_group_rewards,
    component_group_sums,
    component_lambda_return_tensors,
    component_prefix_totals,
    component_reward_sums,
    component_reward_tensors,
    logprob_advantage_diagnostics,
    masked_tensor_mean,
    per_patch_diagnostic_records,
    prefix_totals,
    tensor_correlation,
    value_prediction_metrics,
)
from utils import NotaGenLMHeadModel, Patchilizer


@dataclass
class PatchReplayChunk:
    logprobs: torch.Tensor
    values: torch.Tensor
    token_logprobs: torch.Tensor
    token_log_dists: torch.Tensor
    token_counts: torch.Tensor


@dataclass
class TokenDistributionReplay:
    token_log_dists: torch.Tensor
    token_counts: torch.Tensor


@dataclass
class PPOLossPayload:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    raw_value_loss: torch.Tensor
    value_loss_scale: torch.Tensor
    entropy_loss: torch.Tensor
    reference_kl_loss: torch.Tensor
    approx_kl: torch.Tensor
    old_policy_exact_kl: torch.Tensor
    reference_exact_kl: torch.Tensor
    clip_fraction: torch.Tensor
    advantages_mean: torch.Tensor
    advantages_std: torch.Tensor


@dataclass
class RewardScore:
    total: float
    breakdown: dict


@dataclass(frozen=True)
class RewardEvent:
    start: int
    end: int
    value: float
    name: str


@dataclass
class PatchRewardTrace:
    rewards: list[float]
    prefix_totals: list[float]
    final_score: RewardScore
    component_rewards: dict[str, list[float]]
    component_prefix_totals: dict[str, list[float]]


@dataclass
class ScoredRolloutBatch:
    trajectory_logs: list[dict]
    reward_traces: list[PatchRewardTrace]
    reward_summary: dict


@dataclass(frozen=True)
class PPORewardScoringOptions:
    similarity_chroma_bins: int
    similarity_band_ratio: float
    similarity_timeout_s: float
    max_similarity_reward: float
    patch_reward_attribution: str = "single_pass"
    reward_mode: str = "goldberg"
    simple_reward_note: str = "G"
    simple_reward_max_count: float = 64.0
    simple_reward_length_unit: str = "patches"
    simple_reward_length_target: float = 160.0
    simple_reward_scale: float = 1.0
    rollout_failure_terminal_reward: float = -1.0


@dataclass
class PPORolloutPayload:
    trajectory_index: int
    rollout_seed: int
    full_text: str
    generated_patches: list[list[int]]
    meta: dict
    prompt_idx: int = 0
    prompt_name: str = ""
    prompt: str = ""
    prompt_target: PromptStructuralTarget | None = None
    target: object | None = None
    target_stream_lines: int = 0
    prompt_schedule: PromptScheduleSelection | None = None


@dataclass(frozen=True)
class PromptStructuralTarget:
    target: object
    structure_path: str
    source_key: str


@dataclass(frozen=True)
class PromptScheduleSelection:
    prompt_idx: int
    selection: str
    slot_index: int
    cycle: int
    cycle_position: int
    cycle_length: int
    cycle_order: list[int]


@dataclass(frozen=True)
class PromptBatchItem:
    trajectory_index: int
    prompt_idx: int
    prompt_name: str
    prompt: str
    prompt_target: PromptStructuralTarget
    target: object
    target_stream_lines: int
    schedule: PromptScheduleSelection


@dataclass
class PPOBatchTensors:
    patch_rewards: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    value_targets: torch.Tensor


@dataclass
class PPOReplayEpochResult:
    loss_payload: PPOLossPayload
    new_replays: list[PatchReplayChunk]
    new_logprobs: torch.Tensor
    new_token_logprobs: torch.Tensor
    new_values: torch.Tensor
    grad_norm: float | None
    microbatch_count: int
    microbatch_size: int


class PatchValueHead(torch.nn.Module):
    def __init__(self, hidden_size: int, value_hidden_size: int = 512, dropout: float = 0.0) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.value_hidden_size = int(value_hidden_size)
        self.dropout = float(dropout)
        if value_hidden_size > 0:
            self.net = torch.nn.Sequential(
                torch.nn.LayerNorm(hidden_size),
                torch.nn.Linear(hidden_size, value_hidden_size),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(value_hidden_size, 1),
            )
        else:
            self.net = torch.nn.Sequential(
                torch.nn.LayerNorm(hidden_size),
                torch.nn.Linear(hidden_size, 1),
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states.float()).squeeze(-1)

    def config(self) -> dict:
        return {
            "hidden_size": self.hidden_size,
            "value_hidden_size": self.value_hidden_size,
            "dropout": self.dropout,
        }


def save_value_head_checkpoint(value_head: PatchValueHead, path: str | Path) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": value_head.config(),
            "state_dict": value_head.state_dict(),
        },
        checkpoint_path,
    )


def save_full_policy_checkpoint(model: NotaGenLMHeadModel, checkpoint_dir: str | Path, step_idx: int) -> dict:
    checkpoint_root = Path(checkpoint_dir)
    step_dir = checkpoint_root / f"step_{step_idx:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = step_dir / "current.pth"
    start = time.perf_counter()
    torch.save(
        {
            "step": int(step_idx),
            "checkpoint_type": "full_policy_state_dict",
            "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        },
        checkpoint_path,
    )
    return {
        "step": int(step_idx),
        "path": str(checkpoint_path),
        "checkpoint_type": "full_policy_state_dict",
        "elapsed_s": time.perf_counter() - start,
    }


def save_lora_policy_checkpoint(model: NotaGenLMHeadModel, checkpoint_dir: str | Path, step_idx: int) -> dict:
    checkpoint_root = Path(checkpoint_dir)
    step_dir = checkpoint_root / f"step_{step_idx:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    saved_parts: dict[str, str] = {}
    for name, module in (
        ("patch_level_decoder_base", model.patch_level_decoder.base),
        ("char_level_decoder_base", model.char_level_decoder.base),
    ):
        part_dir = step_dir / name
        if not hasattr(module, "save_pretrained"):
            raise RuntimeError(
                "LoRA checkpoint requested, but "
                f"{name} does not expose save_pretrained(); was --lora-r set?"
            )
        module.save_pretrained(part_dir)
        saved_parts[name] = str(part_dir)
    return {
        "step": int(step_idx),
        "path": str(step_dir),
        "checkpoint_type": "lora_adapter",
        "parts": saved_parts,
        "elapsed_s": time.perf_counter() - start,
    }


def save_ppo_policy_checkpoint(
    model: NotaGenLMHeadModel,
    checkpoint_dir: str | Path,
    step_idx: int,
    *,
    lora_r: int,
) -> dict:
    if lora_r > 0:
        return save_lora_policy_checkpoint(model, checkpoint_dir, step_idx)
    return save_full_policy_checkpoint(model, checkpoint_dir, step_idx)


def load_value_head_checkpoint(value_head: PatchValueHead, path: str | Path, device: torch.device) -> dict:
    payload = torch.load(Path(path), map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload:
        config = payload.get("config", {})
        if config and int(config.get("hidden_size", value_head.hidden_size)) != value_head.hidden_size:
            raise RuntimeError(
                f"value head hidden size mismatch: checkpoint={config.get('hidden_size')} "
                f"current={value_head.hidden_size}"
            )
        state_dict = payload["state_dict"]
    elif isinstance(payload, dict):
        config = {}
        state_dict = payload
    else:
        raise RuntimeError(f"unsupported value head checkpoint payload type: {type(payload)!r}")
    value_head.load_state_dict(state_dict)
    return {"path": str(path), "config": config}


def build_value_head(policy_shape: ModelShape, args, device: torch.device) -> tuple[PatchValueHead, dict | None]:
    checkpoint_payload = None
    checkpoint_config = {}
    if args.value_head_weights:
        checkpoint_payload = torch.load(Path(args.value_head_weights), map_location=device)
        if isinstance(checkpoint_payload, dict) and "state_dict" in checkpoint_payload:
            checkpoint_config = checkpoint_payload.get("config", {}) or {}

    value_head = PatchValueHead(
        policy_shape.hidden_size,
        value_hidden_size=int(checkpoint_config.get("value_hidden_size", args.value_head_hidden_size)),
        dropout=float(checkpoint_config.get("dropout", args.value_head_dropout)),
    ).to(device)
    if checkpoint_payload is None:
        return value_head, None

    loaded = load_value_head_checkpoint(value_head, args.value_head_weights, device)
    return value_head, loaded


def _resolve_prompt_path(raw_path: str | Path, *, base_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return base_dir / path


def _prompt_structure_path(row: dict, prompt_idx: int, args) -> tuple[Path, str]:
    base_dir = Path(args.prompts_jsonl).resolve().parent
    for key in ("target_structure_abc", "source", "continuation"):
        value = row.get(key)
        if not value:
            continue
        path = _resolve_prompt_path(value, base_dir=base_dir)
        if not path.exists():
            raise FileNotFoundError(
                f"prompt {prompt_idx} has {key}={value!r}, but the target structure file was not found at {path}"
            )
        return path, key

    fallback = _resolve_prompt_path(args.target_structure_abc, base_dir=base_dir)
    if not fallback.exists():
        raise FileNotFoundError(f"fallback --target-structure-abc was not found at {fallback}")
    return fallback, "fallback_target_structure_abc"


def load_prompt_structural_targets(prompts: list[dict], args) -> list[PromptStructuralTarget]:
    cache: dict[str, object] = {}
    prompt_targets: list[PromptStructuralTarget] = []
    for prompt_idx, row in enumerate(prompts):
        structure_path, source_key = _prompt_structure_path(row, prompt_idx, args)
        cache_key = str(structure_path.resolve())
        target = cache.get(cache_key)
        if target is None:
            target = load_structural_target(args.target_json, structure_path=structure_path)
            cache[cache_key] = target
        prompt_targets.append(
            PromptStructuralTarget(
                target=target,
                structure_path=str(structure_path),
                source_key=source_key,
            )
        )
    return prompt_targets


def prompt_structural_target_metadata(prompt_targets: list[PromptStructuralTarget]) -> list[dict]:
    return [
        {
            "structure_path": item.structure_path,
            "source_key": item.source_key,
            "expected_bars": int(item.target.expected_bars),
            "expected_reward_bars": int(item.target.expected_reward_bars),
        }
        for item in prompt_targets
    ]


def prompt_cycle_order(
    prompt_count: int,
    *,
    selection: str,
    seed: int,
    cycle: int,
) -> list[int]:
    if prompt_count <= 0:
        raise ValueError(f"prompt_count must be positive, got {prompt_count}")
    if cycle < 0:
        raise ValueError(f"cycle must be non-negative, got {cycle}")

    order = list(range(prompt_count))
    if selection == "ordered":
        return order
    if selection == "random":
        cycle_seed = int(seed) + (int(cycle) + 1) * 1_000_003
        random.Random(cycle_seed).shuffle(order)
        return order
    raise ValueError(f"unsupported prompt selection mode: {selection!r}")


def select_prompt_for_slot(
    *,
    slot_index: int,
    prompt_count: int,
    selection: str,
    seed: int,
) -> PromptScheduleSelection:
    if slot_index < 0:
        raise ValueError(f"slot_index must be non-negative, got {slot_index}")
    cycle = slot_index // prompt_count
    cycle_position = slot_index % prompt_count
    order = prompt_cycle_order(prompt_count, selection=selection, seed=seed, cycle=cycle)
    return PromptScheduleSelection(
        prompt_idx=int(order[cycle_position]),
        selection=selection,
        slot_index=int(slot_index),
        cycle=int(cycle),
        cycle_position=int(cycle_position),
        cycle_length=int(prompt_count),
        cycle_order=order,
    )


def select_prompt_for_update(
    *,
    update_index: int,
    prompt_count: int,
    selection: str,
    seed: int,
) -> PromptScheduleSelection:
    return select_prompt_for_slot(
        slot_index=update_index,
        prompt_count=prompt_count,
        selection=selection,
        seed=seed,
    )


def prompt_schedule_metadata(selection: PromptScheduleSelection) -> dict:
    return {
        "prompt_selection": selection.selection,
        "prompt_schedule_slot": selection.slot_index,
        "prompt_cycle": selection.cycle,
        "prompt_cycle_position": selection.cycle_position,
        "prompt_cycle_length": selection.cycle_length,
        "prompt_cycle_order": selection.cycle_order,
    }


def build_prompt_batch_for_slots(
    *,
    prompts: list[dict],
    prompt_targets: list[PromptStructuralTarget],
    selection: str,
    seed: int,
    start_slot: int,
    count: int,
) -> list[PromptBatchItem]:
    if count < 0:
        raise ValueError(f"prompt batch count must be non-negative, got {count}")
    if len(prompt_targets) != len(prompts):
        raise ValueError(f"prompt target count mismatch: prompts={len(prompts)} targets={len(prompt_targets)}")
    items: list[PromptBatchItem] = []
    for trajectory_index in range(count):
        schedule = select_prompt_for_slot(
            slot_index=start_slot + trajectory_index,
            prompt_count=len(prompts),
            selection=selection,
            seed=seed,
        )
        row = prompts[schedule.prompt_idx]
        prompt_target = prompt_targets[schedule.prompt_idx]
        target = prompt_target.target
        items.append(
            PromptBatchItem(
                trajectory_index=trajectory_index,
                prompt_idx=schedule.prompt_idx,
                prompt_name=prompt_row_name(row, schedule.prompt_idx),
                prompt=row["prompt"],
                prompt_target=prompt_target,
                target=target,
                target_stream_lines=int(target.expected_reward_bars),
                schedule=schedule,
            )
        )
    return items


def build_prompt_batch_for_repeated_slot(
    *,
    prompts: list[dict],
    prompt_targets: list[PromptStructuralTarget],
    selection: str,
    seed: int,
    slot_index: int,
    count: int,
) -> list[PromptBatchItem]:
    if count < 0:
        raise ValueError(f"prompt batch count must be non-negative, got {count}")
    if len(prompt_targets) != len(prompts):
        raise ValueError(f"prompt target count mismatch: prompts={len(prompts)} targets={len(prompt_targets)}")
    if count == 0:
        return []
    schedule = select_prompt_for_slot(
        slot_index=slot_index,
        prompt_count=len(prompts),
        selection=selection,
        seed=seed,
    )
    row = prompts[schedule.prompt_idx]
    prompt_target = prompt_targets[schedule.prompt_idx]
    target = prompt_target.target
    return [
        PromptBatchItem(
            trajectory_index=trajectory_index,
            prompt_idx=schedule.prompt_idx,
            prompt_name=prompt_row_name(row, schedule.prompt_idx),
            prompt=row["prompt"],
            prompt_target=prompt_target,
            target=target,
            target_stream_lines=int(target.expected_reward_bars),
            schedule=schedule,
        )
        for trajectory_index in range(count)
    ]


def build_prompt_batch_for_step(
    *,
    prompts: list[dict],
    prompt_targets: list[PromptStructuralTarget],
    args,
    step_idx: int,
    trajectories_per_step: int | None = None,
) -> list[PromptBatchItem]:
    trajectory_count = args.trajectories_per_step if trajectories_per_step is None else int(trajectories_per_step)
    prompt_batch_mode = str(getattr(args, "prompt_batch_mode", "trajectory"))
    if prompt_batch_mode == "step":
        return build_prompt_batch_for_repeated_slot(
            prompts=prompts,
            prompt_targets=prompt_targets,
            selection=args.prompt_selection,
            seed=args.seed,
            slot_index=int(step_idx) - 1,
            count=trajectory_count,
        )
    if prompt_batch_mode != "trajectory":
        raise ValueError(f"unsupported prompt_batch_mode: {prompt_batch_mode!r}")
    start_slot = (int(step_idx) - 1) * int(args.trajectories_per_step)
    return build_prompt_batch_for_slots(
        prompts=prompts,
        prompt_targets=prompt_targets,
        selection=args.prompt_selection,
        seed=args.seed,
        start_slot=start_slot,
        count=trajectory_count,
    )


def prompt_batch_metadata(prompt_batch: list[PromptBatchItem]) -> dict:
    if not prompt_batch:
        return {
            "prompt_selection": None,
            "prompt_batch_size": 0,
            "prompt_batch_multiple_prompts": False,
            "prompt_batch_prompt_indices": [],
            "prompt_batch_prompt_names": [],
            "prompt_batch_target_stream_lines": [],
            "prompt_batch_assignments": [],
        }
    first = prompt_batch[0]
    prompt_indices = [int(item.prompt_idx) for item in prompt_batch]
    prompt_names = [item.prompt_name for item in prompt_batch]
    target_stream_lines = [int(item.target_stream_lines) for item in prompt_batch]
    return {
        "prompt_selection": first.schedule.selection,
        "prompt_batch_size": len(prompt_batch),
        "prompt_batch_multiple_prompts": len(set(prompt_indices)) > 1,
        "prompt_batch_unique_prompt_indices": sorted(set(prompt_indices)),
        "prompt_batch_prompt_indices": prompt_indices,
        "prompt_batch_prompt_names": prompt_names,
        "prompt_batch_target_stream_lines": target_stream_lines,
        "prompt_batch_target_stream_lines_min": min(target_stream_lines),
        "prompt_batch_target_stream_lines_max": max(target_stream_lines),
        "prompt_batch_schedule_start_slot": int(first.schedule.slot_index),
        "prompt_batch_schedule_end_slot": int(prompt_batch[-1].schedule.slot_index),
        "prompt_batch_cycles": sorted({int(item.schedule.cycle) for item in prompt_batch}),
        "prompt_batch_assignments": [
            {
                "trajectory_index": int(item.trajectory_index),
                "prompt_index": int(item.prompt_idx),
                "prompt_name": item.prompt_name,
                "target_stream_lines": int(item.target_stream_lines),
                **prompt_schedule_metadata(item.schedule),
            }
            for item in prompt_batch
        ],
    }


def resolve_fixed_eval_prompt_selection(args) -> str:
    selection = str(getattr(args, "fixed_eval_prompt_selection", "same"))
    if selection == "same":
        return str(args.prompt_selection)
    if selection in {"ordered", "random"}:
        return selection
    raise ValueError(f"unsupported fixed_eval_prompt_selection: {selection!r}")


def resolve_fixed_eval_prompt_batch_mode(args) -> str:
    mode = str(getattr(args, "fixed_eval_prompt_batch_mode", "same"))
    if mode == "same":
        training_mode = str(getattr(args, "prompt_batch_mode", "trajectory"))
        if training_mode == "step":
            return "event"
        if training_mode == "trajectory":
            return "trajectory"
        raise ValueError(f"unsupported prompt_batch_mode: {training_mode!r}")
    if mode in {"trajectory", "event"}:
        return mode
    raise ValueError(f"unsupported fixed_eval_prompt_batch_mode: {mode!r}")


def fixed_eval_should_run_after_step(args, step_idx: int) -> bool:
    if int(args.fixed_eval_trajectories) <= 0:
        return False
    every_steps = int(getattr(args, "fixed_eval_every_steps", 1))
    if every_steps <= 0:
        return False
    return int(step_idx) > 0 and int(step_idx) % every_steps == 0


def fixed_eval_event_index_after_step(args, step_idx: int) -> int:
    every_steps = int(getattr(args, "fixed_eval_every_steps", 1))
    if every_steps <= 0:
        raise ValueError("fixed_eval_event_index_after_step requires fixed_eval_every_steps > 0")
    if not fixed_eval_should_run_after_step(args, step_idx):
        raise ValueError(f"step {step_idx} is not a fixed-eval step for every_steps={every_steps}")
    return int(step_idx) // every_steps - 1


def fixed_eval_event_index_before_training(args) -> int:
    every_steps = int(getattr(args, "fixed_eval_every_steps", 1))
    if every_steps <= 0:
        return 0
    return max(0, int(args.step_offset) // every_steps)


def build_fixed_eval_prompt_batch(
    *,
    prompts: list[dict],
    prompt_targets: list[PromptStructuralTarget],
    args,
    event_index: int,
) -> list[PromptBatchItem]:
    if int(args.fixed_eval_trajectories) <= 0:
        return []
    if event_index < 0:
        raise ValueError(f"fixed eval event index must be non-negative, got {event_index}")
    prompt_batch_mode = resolve_fixed_eval_prompt_batch_mode(args)
    if prompt_batch_mode == "event":
        return build_prompt_batch_for_repeated_slot(
            prompts=prompts,
            prompt_targets=prompt_targets,
            selection=resolve_fixed_eval_prompt_selection(args),
            seed=int(args.seed) + int(getattr(args, "fixed_eval_prompt_seed_offset", 2_000_000)),
            slot_index=int(event_index),
            count=int(args.fixed_eval_trajectories),
        )
    if prompt_batch_mode != "trajectory":
        raise ValueError(f"unsupported fixed eval prompt batch mode: {prompt_batch_mode!r}")
    start_slot = int(event_index) * int(args.fixed_eval_trajectories)
    return build_prompt_batch_for_slots(
        prompts=prompts,
        prompt_targets=prompt_targets,
        selection=resolve_fixed_eval_prompt_selection(args),
        seed=int(args.seed) + int(getattr(args, "fixed_eval_prompt_seed_offset", 2_000_000)),
        start_slot=start_slot,
        count=int(args.fixed_eval_trajectories),
    )


def prompt_context_from_payload(payload: PPORolloutPayload) -> PromptBatchItem:
    if (
        not payload.prompt
        or payload.prompt_target is None
        or payload.target is None
        or payload.target_stream_lines <= 0
        or payload.prompt_schedule is None
    ):
        raise RuntimeError(
            f"rollout payload {payload.trajectory_index} is missing prompt context for multi-prompt PPO"
        )
    return PromptBatchItem(
        trajectory_index=payload.trajectory_index,
        prompt_idx=payload.prompt_idx,
        prompt_name=payload.prompt_name,
        prompt=payload.prompt,
        prompt_target=payload.prompt_target,
        target=payload.target,
        target_stream_lines=payload.target_stream_lines,
        schedule=payload.prompt_schedule,
    )


def value_from_last_patch(
    model: NotaGenLMHeadModel,
    value_head: PatchValueHead,
    flat_ids: list[int],
    precision: str,
    replay_context_patches: int | None = None,
    detach_policy: bool = False,
) -> torch.Tensor:
    device = next(model.parameters()).device
    context = torch.no_grad() if detach_policy else nullcontext()
    with context:
        encoded_patch, _tokens = _encoded_last_patch(
            model,
            flat_ids,
            device,
            precision,
            replay_context_patches=replay_context_patches,
        )
    if detach_policy:
        encoded_patch = encoded_patch.detach()
    return value_head(encoded_patch)


def hidden_state_from_last_patch(
    model: NotaGenLMHeadModel,
    flat_ids: list[int],
    precision: str,
    replay_context_patches: int | None = None,
    detach_policy: bool = True,
) -> torch.Tensor:
    device = next(model.parameters()).device
    context = torch.no_grad() if detach_policy else nullcontext()
    with context:
        encoded_patch, _tokens = _encoded_last_patch(
            model,
            flat_ids,
            device,
            precision,
            replay_context_patches=replay_context_patches,
        )
    if detach_policy:
        encoded_patch = encoded_patch.detach()
    return encoded_patch


def patch_logprob_sum_and_value(
    model: NotaGenLMHeadModel,
    value_head: PatchValueHead,
    flat_prompt_ids: list[int],
    patch: list[int],
    precision: str,
    replay_context_patches: int | None = None,
) -> PatchReplayChunk:
    device = next(model.parameters()).device
    encoded_patch, tokens = _encoded_last_patch(
        model,
        flat_prompt_ids,
        device,
        precision,
        replay_context_patches=replay_context_patches,
    )
    logprobs: list[torch.Tensor] = []
    log_dists: list[torch.Tensor] = []
    for tok in patch:
        token_embeddings = torch.nn.functional.embedding(
            tokens.reshape(1, -1),
            model.char_level_decoder.base.transformer.wte.weight,
        )
        inputs_embeds = torch.cat((encoded_patch.reshape(1, 1, -1), token_embeddings[:, 1:, :]), dim=1)
        with autocast_context(device, precision):
            outputs = model.char_level_decoder.base(inputs_embeds=inputs_embeds)
            logits = outputs.logits[0, -1]
        token_log_dist = torch.log_softmax(logits.float(), dim=-1)
        logprobs.append(token_log_dist[tok])
        log_dists.append(token_log_dist)
        if len(tokens) >= PATCH_SIZE:
            break
        tokens = torch.cat((tokens, torch.tensor([tok], device=device, dtype=torch.long)), dim=0)

    if not logprobs:
        raise RuntimeError("cannot score an empty generated patch")
    token_logprobs = torch.stack(logprobs)
    token_log_dists = torch.stack(log_dists)
    return PatchReplayChunk(
        logprobs=token_logprobs.sum().reshape(1),
        values=value_head(encoded_patch).reshape(1),
        token_logprobs=token_logprobs,
        token_log_dists=token_log_dists,
        token_counts=torch.tensor([token_logprobs.numel()], device=device, dtype=torch.long),
    )


def patch_token_log_dists(
    model: NotaGenLMHeadModel,
    flat_prompt_ids: list[int],
    patch: list[int],
    precision: str,
    replay_context_patches: int | None = None,
) -> TokenDistributionReplay:
    device = next(model.parameters()).device
    encoded_patch, tokens = _encoded_last_patch(
        model,
        flat_prompt_ids,
        device,
        precision,
        replay_context_patches=replay_context_patches,
    )
    log_dists: list[torch.Tensor] = []
    for tok in patch:
        token_embeddings = torch.nn.functional.embedding(
            tokens.reshape(1, -1),
            model.char_level_decoder.base.transformer.wte.weight,
        )
        inputs_embeds = torch.cat((encoded_patch.reshape(1, 1, -1), token_embeddings[:, 1:, :]), dim=1)
        with autocast_context(device, precision):
            outputs = model.char_level_decoder.base(inputs_embeds=inputs_embeds)
            logits = outputs.logits[0, -1]
        log_dists.append(torch.log_softmax(logits.float(), dim=-1))
        if len(tokens) >= PATCH_SIZE:
            break
        tokens = torch.cat((tokens, torch.tensor([tok], device=device, dtype=torch.long)), dim=0)

    if not log_dists:
        raise RuntimeError("cannot score an empty generated patch")
    token_log_dists = torch.stack(log_dists)
    return TokenDistributionReplay(
        token_log_dists=token_log_dists,
        token_counts=torch.tensor([token_log_dists.shape[0]], device=device, dtype=torch.long),
    )


def tail_patch_logprob_value_chunk(
    model: NotaGenLMHeadModel,
    value_head: PatchValueHead,
    current_ids: list[int],
    remaining_patches: list[list[int]],
    chunk_start: int,
    chunk_end: int,
    precision: str,
    replay_context_patches: int | None = None,
) -> PatchReplayChunk:
    encoded_targets, target_patches = tail_encoded_targets(
        model=model,
        current_ids=current_ids,
        remaining_patches=remaining_patches,
        chunk_start=chunk_start,
        chunk_end=chunk_end,
        precision=precision,
        replay_context_patches=replay_context_patches,
        error_context="PPO replay",
    )
    token_logprobs, token_log_dists, token_counts = char_patch_token_logprobs_dists_and_counts(
        model,
        encoded_targets,
        target_patches,
        precision,
    )
    per_patch_logprobs = split_tensor_by_counts(token_logprobs, [int(item) for item in token_counts.detach().cpu()])
    return PatchReplayChunk(
        logprobs=torch.stack(
            [
                item.sum()
                if item.numel() > 0
                else torch.zeros((), device=encoded_targets.device, dtype=token_logprobs.dtype)
                for item in per_patch_logprobs
            ]
        ),
        values=value_head(encoded_targets),
        token_logprobs=token_logprobs,
        token_log_dists=token_log_dists,
        token_counts=token_counts,
    )


def batched_tail_patch_logprob_value_chunk(
    model: NotaGenLMHeadModel,
    value_head: PatchValueHead,
    current_ids_batch: list[list[int]],
    remaining_patches_batch: list[list[list[int]]],
    chunk_start: int,
    target_chunk_patches: int,
    precision: str,
    replay_context_patches: int | None = None,
    replay_batch_size: int = 0,
    detach_policy: bool = False,
) -> dict[int, PatchReplayChunk]:
    payload = batched_tail_encoded_targets(
        model=model,
        current_ids_batch=current_ids_batch,
        remaining_patches_batch=remaining_patches_batch,
        chunk_start=chunk_start,
        target_chunk_patches=target_chunk_patches,
        precision=precision,
        replay_context_patches=replay_context_patches,
        replay_batch_size=replay_batch_size,
        detach_policy=detach_policy,
        error_context="batched PPO replay",
    )
    if not payload:
        return {}

    sample_indices: list[int] = []
    encoded_targets: list[torch.Tensor] = []
    target_patches: list[list[int]] = []
    patch_counts: list[int] = []
    for sample_idx, (encoded, targets) in payload.items():
        sample_indices.append(sample_idx)
        encoded_targets.append(encoded)
        target_patches.extend(targets)
        patch_counts.append(len(targets))

    encoded_target_tensor = torch.cat(encoded_targets, dim=0)
    token_logprobs, token_log_dists, token_counts = char_patch_token_logprobs_dists_and_counts(
        model,
        encoded_target_tensor,
        target_patches,
        precision,
    )
    per_patch_logprobs = split_tensor_by_counts(token_logprobs, [int(item) for item in token_counts.detach().cpu()])
    logprob_sums = torch.stack(
        [
            item.sum() if item.numel() > 0 else torch.zeros((), device=encoded_target_tensor.device, dtype=token_logprobs.dtype)
            for item in per_patch_logprobs
        ]
    )
    values = value_head(encoded_target_tensor)
    split_logprobs = split_tensor_by_counts(logprob_sums, patch_counts)
    split_values = split_tensor_by_counts(values, patch_counts)
    split_token_counts = split_tensor_by_counts(token_counts, patch_counts)
    token_sample_counts = [int(counts.detach().sum().cpu()) for counts in split_token_counts]
    split_token_logprobs = split_tensor_by_counts(token_logprobs, token_sample_counts)
    split_token_log_dists = split_tensor_by_counts(token_log_dists, token_sample_counts)
    return {
        sample_idx: PatchReplayChunk(
            logprobs=logprobs,
            values=sample_values,
            token_logprobs=sample_token_logprobs,
            token_log_dists=sample_token_log_dists,
            token_counts=sample_token_counts,
        )
        for sample_idx, logprobs, sample_values, sample_token_logprobs, sample_token_log_dists, sample_token_counts in zip(
            sample_indices,
            split_logprobs,
            split_values,
            split_token_logprobs,
            split_token_log_dists,
            split_token_counts,
            strict=True,
        )
    }


def batched_tail_token_log_dist_chunk(
    model: NotaGenLMHeadModel,
    current_ids_batch: list[list[int]],
    remaining_patches_batch: list[list[list[int]]],
    chunk_start: int,
    target_chunk_patches: int,
    precision: str,
    replay_context_patches: int | None = None,
    replay_batch_size: int = 0,
) -> dict[int, TokenDistributionReplay]:
    payload = batched_tail_encoded_targets(
        model=model,
        current_ids_batch=current_ids_batch,
        remaining_patches_batch=remaining_patches_batch,
        chunk_start=chunk_start,
        target_chunk_patches=target_chunk_patches,
        precision=precision,
        replay_context_patches=replay_context_patches,
        replay_batch_size=replay_batch_size,
        detach_policy=True,
        error_context="batched PPO distribution replay",
    )
    if not payload:
        return {}

    sample_indices: list[int] = []
    encoded_targets: list[torch.Tensor] = []
    target_patches: list[list[int]] = []
    patch_counts: list[int] = []
    for sample_idx, (encoded, targets) in payload.items():
        sample_indices.append(sample_idx)
        encoded_targets.append(encoded)
        target_patches.extend(targets)
        patch_counts.append(len(targets))

    encoded_target_tensor = torch.cat(encoded_targets, dim=0)
    _token_logprobs, token_log_dists, token_counts = char_patch_token_logprobs_dists_and_counts(
        model,
        encoded_target_tensor,
        target_patches,
        precision,
    )
    split_token_counts = split_tensor_by_counts(token_counts, patch_counts)
    token_sample_counts = [int(counts.detach().sum().cpu()) for counts in split_token_counts]
    split_token_log_dists = split_tensor_by_counts(token_log_dists, token_sample_counts)
    return {
        sample_idx: TokenDistributionReplay(
            token_log_dists=sample_token_log_dists,
            token_counts=sample_token_counts,
        )
        for sample_idx, sample_token_log_dists, sample_token_counts in zip(
            sample_indices,
            split_token_log_dists,
            split_token_counts,
            strict=True,
        )
    }


def batched_tail_patch_value_chunk(
    model: NotaGenLMHeadModel,
    value_head: PatchValueHead,
    current_ids_batch: list[list[int]],
    remaining_patches_batch: list[list[list[int]]],
    chunk_start: int,
    target_chunk_patches: int,
    precision: str,
    replay_context_patches: int | None = None,
    replay_batch_size: int = 0,
    detach_policy: bool = True,
) -> dict[int, torch.Tensor]:
    payload = batched_tail_encoded_targets(
        model=model,
        current_ids_batch=current_ids_batch,
        remaining_patches_batch=remaining_patches_batch,
        chunk_start=chunk_start,
        target_chunk_patches=target_chunk_patches,
        precision=precision,
        replay_context_patches=replay_context_patches,
        replay_batch_size=replay_batch_size,
        detach_policy=detach_policy,
        error_context="batched PPO replay",
    )
    if not payload:
        return {}
    return {
        sample_idx: value_head(encoded)
        for sample_idx, (encoded, _targets) in payload.items()
    }


def tail_patch_value_chunk(
    model: NotaGenLMHeadModel,
    value_head: PatchValueHead,
    current_ids: list[int],
    remaining_patches: list[list[int]],
    chunk_start: int,
    chunk_end: int,
    precision: str,
    replay_context_patches: int | None = None,
    detach_policy: bool = True,
) -> torch.Tensor:
    encoded_targets, _target_patches = tail_encoded_targets(
        model=model,
        current_ids=current_ids,
        remaining_patches=remaining_patches,
        chunk_start=chunk_start,
        chunk_end=chunk_end,
        precision=precision,
        replay_context_patches=replay_context_patches,
        detach_policy=detach_policy,
        error_context="PPO value replay",
    )
    return value_head(encoded_targets)


def tail_patch_hidden_state_chunk(
    model: NotaGenLMHeadModel,
    current_ids: list[int],
    remaining_patches: list[list[int]],
    chunk_start: int,
    chunk_end: int,
    precision: str,
    replay_context_patches: int | None = None,
    detach_policy: bool = True,
) -> torch.Tensor:
    encoded_targets, _target_patches = tail_encoded_targets(
        model=model,
        current_ids=current_ids,
        remaining_patches=remaining_patches,
        chunk_start=chunk_start,
        chunk_end=chunk_end,
        precision=precision,
        replay_context_patches=replay_context_patches,
        detach_policy=detach_policy,
        error_context="PPO hidden-state replay",
    )
    return encoded_targets


def trajectory_patch_logprob_value_chunks(
    model: NotaGenLMHeadModel,
    value_head: PatchValueHead,
    flat_prompt_ids: list[int],
    generated_patches: list[list[int]],
    precision: str,
    replay_context_patches: int | None = None,
    target_chunk_patches: int = 0,
):
    if not generated_patches:
        return

    current_ids = list(flat_prompt_ids)
    start_idx = 0
    while start_idx < len(generated_patches) and len(current_ids) % PATCH_SIZE != 0:
        patch = generated_patches[start_idx]
        yield patch_logprob_sum_and_value(
            model,
            value_head,
            current_ids,
            patch,
            precision,
            replay_context_patches=replay_context_patches,
        )
        current_ids.extend(
            normalize_patch_for_context(
                patch,
                eos_token_id=model.eos_token_id,
                special_token_id=model.special_token_id,
            )
        )
        start_idx += 1

    if start_idx >= len(generated_patches):
        return

    remaining_patches = generated_patches[start_idx:]
    chunk_size = len(remaining_patches) if target_chunk_patches <= 0 else target_chunk_patches
    for chunk_start in range(0, len(remaining_patches), chunk_size):
        chunk_end = min(len(remaining_patches), chunk_start + chunk_size)
        yield tail_patch_logprob_value_chunk(
            model,
            value_head,
            current_ids,
            remaining_patches,
            chunk_start,
            chunk_end,
            precision,
            replay_context_patches=replay_context_patches,
        )


def trajectory_patch_logprobs_values(
    model: NotaGenLMHeadModel,
    value_head: PatchValueHead,
    flat_prompt_ids: list[int],
    generated_patches: list[list[int]],
    precision: str,
    replay_context_patches: int | None = None,
    target_chunk_patches: int = 0,
) -> PatchReplayChunk:
    chunks = list(
        trajectory_patch_logprob_value_chunks(
            model,
            value_head,
            flat_prompt_ids,
            generated_patches,
            precision,
            replay_context_patches=replay_context_patches,
            target_chunk_patches=target_chunk_patches,
        )
    )
    device = next(model.parameters()).device
    vocab_size = model.char_level_decoder.base.transformer.wte.weight.shape[0]
    if not chunks:
        return PatchReplayChunk(
            logprobs=torch.empty(0, device=device),
            values=torch.empty(0, device=device),
            token_logprobs=torch.empty(0, device=device),
            token_log_dists=torch.empty((0, vocab_size), device=device),
            token_counts=torch.empty(0, device=device, dtype=torch.long),
        )
    return PatchReplayChunk(
        logprobs=torch.cat([chunk.logprobs for chunk in chunks]),
        values=torch.cat([chunk.values for chunk in chunks]),
        token_logprobs=torch.cat([chunk.token_logprobs for chunk in chunks]),
        token_log_dists=torch.cat([chunk.token_log_dists for chunk in chunks]),
        token_counts=torch.cat([chunk.token_counts for chunk in chunks]),
    )


def batched_trajectory_patch_logprobs_values(
    model: NotaGenLMHeadModel,
    value_head: PatchValueHead,
    flat_prompt_ids: list[int],
    generated_patches_batch: list[list[list[int]]],
    precision: str,
    replay_context_patches: int | None = None,
    target_chunk_patches: int = 0,
    replay_batch_size: int = 0,
) -> list[PatchReplayChunk]:
    device = next(model.parameters()).device
    vocab_size = model.char_level_decoder.base.transformer.wte.weight.shape[0]
    current_ids_batch = [list(flat_prompt_ids) for _idx in generated_patches_batch]
    remaining_batch: list[list[list[int]]] = []
    outputs: list[list[PatchReplayChunk]] = [[] for _idx in generated_patches_batch]

    for sample_idx, generated_patches in enumerate(generated_patches_batch):
        current_ids = current_ids_batch[sample_idx]
        start_idx = 0
        while start_idx < len(generated_patches) and len(current_ids) % PATCH_SIZE != 0:
            patch = generated_patches[start_idx]
            outputs[sample_idx].append(
                patch_logprob_sum_and_value(
                    model,
                    value_head,
                    current_ids,
                    patch,
                    precision,
                    replay_context_patches=replay_context_patches,
                )
            )
            current_ids.extend(
                normalize_patch_for_context(
                    patch,
                    eos_token_id=model.eos_token_id,
                    special_token_id=model.special_token_id,
                )
            )
            start_idx += 1
        current_ids_batch[sample_idx] = current_ids
        remaining_batch.append(generated_patches[start_idx:])

    max_remaining = max((len(remaining) for remaining in remaining_batch), default=0)
    chunk_size = max_remaining if target_chunk_patches <= 0 else target_chunk_patches
    if chunk_size > 0:
        for chunk_start in range(0, max_remaining, chunk_size):
            chunk_payload = batched_tail_patch_logprob_value_chunk(
                model,
                value_head,
                current_ids_batch,
                remaining_batch,
                chunk_start,
                target_chunk_patches,
                precision,
                replay_context_patches=replay_context_patches,
                replay_batch_size=replay_batch_size,
            )
            for sample_idx, replay_chunk in chunk_payload.items():
                if replay_chunk.logprobs.numel() > 0:
                    outputs[sample_idx].append(replay_chunk)

    result: list[PatchReplayChunk] = []
    for chunks in outputs:
        if chunks:
            result.append(
                PatchReplayChunk(
                    logprobs=torch.cat([chunk.logprobs for chunk in chunks]),
                    values=torch.cat([chunk.values for chunk in chunks]),
                    token_logprobs=torch.cat([chunk.token_logprobs for chunk in chunks]),
                    token_log_dists=torch.cat([chunk.token_log_dists for chunk in chunks]),
                    token_counts=torch.cat([chunk.token_counts for chunk in chunks]),
                )
            )
        else:
            result.append(
                PatchReplayChunk(
                    logprobs=torch.empty(0, device=device),
                    values=torch.empty(0, device=device),
                    token_logprobs=torch.empty(0, device=device),
                    token_log_dists=torch.empty((0, vocab_size), device=device),
                    token_counts=torch.empty(0, device=device, dtype=torch.long),
                )
            )
    return result


def batched_trajectory_token_log_dists(
    model: NotaGenLMHeadModel,
    flat_prompt_ids: list[int],
    generated_patches_batch: list[list[list[int]]],
    precision: str,
    replay_context_patches: int | None = None,
    target_chunk_patches: int = 0,
    replay_batch_size: int = 0,
) -> list[TokenDistributionReplay]:
    device = next(model.parameters()).device
    vocab_size = model.char_level_decoder.base.transformer.wte.weight.shape[0]
    current_ids_batch = [list(flat_prompt_ids) for _idx in generated_patches_batch]
    remaining_batch: list[list[list[int]]] = []
    outputs: list[list[TokenDistributionReplay]] = [[] for _idx in generated_patches_batch]

    for sample_idx, generated_patches in enumerate(generated_patches_batch):
        current_ids = current_ids_batch[sample_idx]
        start_idx = 0
        while start_idx < len(generated_patches) and len(current_ids) % PATCH_SIZE != 0:
            patch = generated_patches[start_idx]
            outputs[sample_idx].append(
                patch_token_log_dists(
                    model,
                    current_ids,
                    patch,
                    precision,
                    replay_context_patches=replay_context_patches,
                )
            )
            current_ids.extend(
                normalize_patch_for_context(
                    patch,
                    eos_token_id=model.eos_token_id,
                    special_token_id=model.special_token_id,
                )
            )
            start_idx += 1
        current_ids_batch[sample_idx] = current_ids
        remaining_batch.append(generated_patches[start_idx:])

    max_remaining = max((len(remaining) for remaining in remaining_batch), default=0)
    chunk_size = max_remaining if target_chunk_patches <= 0 else target_chunk_patches
    if chunk_size > 0:
        for chunk_start in range(0, max_remaining, chunk_size):
            chunk_payload = batched_tail_token_log_dist_chunk(
                model,
                current_ids_batch,
                remaining_batch,
                chunk_start,
                target_chunk_patches,
                precision,
                replay_context_patches=replay_context_patches,
                replay_batch_size=replay_batch_size,
            )
            for sample_idx, replay_chunk in chunk_payload.items():
                if replay_chunk.token_counts.numel() > 0:
                    outputs[sample_idx].append(replay_chunk)

    result: list[TokenDistributionReplay] = []
    for chunks in outputs:
        if chunks:
            result.append(
                TokenDistributionReplay(
                    token_log_dists=torch.cat([chunk.token_log_dists for chunk in chunks]),
                    token_counts=torch.cat([chunk.token_counts for chunk in chunks]),
                )
            )
        else:
            result.append(
                TokenDistributionReplay(
                    token_log_dists=torch.empty((0, vocab_size), device=device),
                    token_counts=torch.empty(0, device=device, dtype=torch.long),
                )
            )
    return result


def trajectory_patch_value_chunks(
    model: NotaGenLMHeadModel,
    value_head: PatchValueHead,
    flat_prompt_ids: list[int],
    generated_patches: list[list[int]],
    precision: str,
    replay_context_patches: int | None = None,
    target_chunk_patches: int = 0,
    detach_policy: bool = True,
):
    if not generated_patches:
        return

    current_ids = list(flat_prompt_ids)
    start_idx = 0
    while start_idx < len(generated_patches) and len(current_ids) % PATCH_SIZE != 0:
        patch = generated_patches[start_idx]
        yield value_from_last_patch(
            model,
            value_head,
            current_ids,
            precision,
            replay_context_patches=replay_context_patches,
            detach_policy=detach_policy,
        ).reshape(1)
        current_ids.extend(
            normalize_patch_for_context(
                patch,
                eos_token_id=model.eos_token_id,
                special_token_id=model.special_token_id,
            )
        )
        start_idx += 1

    if start_idx >= len(generated_patches):
        return

    remaining_patches = generated_patches[start_idx:]
    chunk_size = len(remaining_patches) if target_chunk_patches <= 0 else target_chunk_patches
    for chunk_start in range(0, len(remaining_patches), chunk_size):
        chunk_end = min(len(remaining_patches), chunk_start + chunk_size)
        yield tail_patch_value_chunk(
            model,
            value_head,
            current_ids,
            remaining_patches,
            chunk_start,
            chunk_end,
            precision,
            replay_context_patches=replay_context_patches,
            detach_policy=detach_policy,
        )


def trajectory_patch_values(
    model: NotaGenLMHeadModel,
    value_head: PatchValueHead,
    flat_prompt_ids: list[int],
    generated_patches: list[list[int]],
    precision: str,
    replay_context_patches: int | None = None,
    target_chunk_patches: int = 0,
    detach_policy: bool = True,
) -> torch.Tensor:
    chunks = list(
        trajectory_patch_value_chunks(
            model,
            value_head,
            flat_prompt_ids,
            generated_patches,
            precision,
            replay_context_patches=replay_context_patches,
            target_chunk_patches=target_chunk_patches,
            detach_policy=detach_policy,
        )
    )
    device = next(model.parameters()).device
    if not chunks:
        return torch.empty(0, device=device)
    return torch.cat(chunks)


def batched_trajectory_patch_values(
    model: NotaGenLMHeadModel,
    value_head: PatchValueHead,
    flat_prompt_ids: list[int],
    generated_patches_batch: list[list[list[int]]],
    precision: str,
    replay_context_patches: int | None = None,
    target_chunk_patches: int = 0,
    replay_batch_size: int = 0,
    detach_policy: bool = True,
) -> list[torch.Tensor]:
    device = next(model.parameters()).device
    current_ids_batch = [list(flat_prompt_ids) for _idx in generated_patches_batch]
    remaining_batch: list[list[list[int]]] = []
    outputs: list[list[torch.Tensor]] = [[] for _idx in generated_patches_batch]

    for sample_idx, generated_patches in enumerate(generated_patches_batch):
        current_ids = current_ids_batch[sample_idx]
        start_idx = 0
        while start_idx < len(generated_patches) and len(current_ids) % PATCH_SIZE != 0:
            patch = generated_patches[start_idx]
            outputs[sample_idx].append(
                value_from_last_patch(
                    model,
                    value_head,
                    current_ids,
                    precision,
                    replay_context_patches=replay_context_patches,
                    detach_policy=detach_policy,
                ).reshape(1)
            )
            current_ids.extend(
                normalize_patch_for_context(
                    patch,
                    eos_token_id=model.eos_token_id,
                    special_token_id=model.special_token_id,
                )
            )
            start_idx += 1
        current_ids_batch[sample_idx] = current_ids
        remaining_batch.append(generated_patches[start_idx:])

    max_remaining = max((len(remaining) for remaining in remaining_batch), default=0)
    chunk_size = max_remaining if target_chunk_patches <= 0 else target_chunk_patches
    if chunk_size > 0:
        for chunk_start in range(0, max_remaining, chunk_size):
            chunk_payload = batched_tail_patch_value_chunk(
                model,
                value_head,
                current_ids_batch,
                remaining_batch,
                chunk_start,
                target_chunk_patches,
                precision,
                replay_context_patches=replay_context_patches,
                replay_batch_size=replay_batch_size,
                detach_policy=detach_policy,
            )
            for sample_idx, values in chunk_payload.items():
                if values.numel() > 0:
                    outputs[sample_idx].append(values)

    result: list[torch.Tensor] = []
    for chunks in outputs:
        if chunks:
            result.append(torch.cat(chunks))
        else:
            result.append(torch.empty(0, device=device))
    return result


def flat_prompt_ids_for_payload(payload: PPORolloutPayload, patchilizer: Patchilizer) -> list[int]:
    if not payload.prompt or payload.target_stream_lines <= 0:
        raise RuntimeError(
            f"rollout payload {payload.trajectory_index} is missing prompt text/target length for replay"
        )
    rollout_prompt = build_rollout_prefix(payload.prompt, payload.target_stream_lines)
    return [item for sublist in patchilizer.encode_generate(rollout_prompt) for item in sublist]


def _rollout_prompt_groups(rollout_payloads: list[PPORolloutPayload]) -> dict[tuple[int, int, str], list[int]]:
    groups: dict[tuple[int, int, str], list[int]] = {}
    for payload_idx, payload in enumerate(rollout_payloads):
        if not payload.prompt or payload.target_stream_lines <= 0:
            raise RuntimeError(
                f"rollout payload {payload.trajectory_index} is missing prompt context for grouped replay"
            )
        key = (int(payload.prompt_idx), int(payload.target_stream_lines), payload.prompt)
        groups.setdefault(key, []).append(payload_idx)
    return groups


def batched_trajectory_patch_logprobs_values_by_prompt(
    model: NotaGenLMHeadModel,
    value_head: PatchValueHead,
    rollout_payloads: list[PPORolloutPayload],
    precision: str,
    replay_context_patches: int | None = None,
    target_chunk_patches: int = 0,
    replay_batch_size: int = 0,
) -> list[PatchReplayChunk]:
    patchilizer = Patchilizer(stream=PATCH_STREAM)
    outputs: list[PatchReplayChunk | None] = [None] * len(rollout_payloads)
    for indices in _rollout_prompt_groups(rollout_payloads).values():
        group_payloads = [rollout_payloads[idx] for idx in indices]
        flat_prompt_ids = flat_prompt_ids_for_payload(group_payloads[0], patchilizer)
        group_outputs = batched_trajectory_patch_logprobs_values(
            model,
            value_head,
            flat_prompt_ids,
            [payload.generated_patches for payload in group_payloads],
            precision,
            replay_context_patches=replay_context_patches,
            target_chunk_patches=target_chunk_patches,
            replay_batch_size=replay_batch_size,
        )
        for output_idx, replay in zip(indices, group_outputs, strict=True):
            outputs[output_idx] = replay
    if any(item is None for item in outputs):
        raise RuntimeError("grouped PPO replay did not produce an output for every trajectory")
    return [item for item in outputs if item is not None]


def batched_trajectory_token_log_dists_by_prompt(
    model: NotaGenLMHeadModel,
    rollout_payloads: list[PPORolloutPayload],
    precision: str,
    replay_context_patches: int | None = None,
    target_chunk_patches: int = 0,
    replay_batch_size: int = 0,
) -> list[TokenDistributionReplay]:
    patchilizer = Patchilizer(stream=PATCH_STREAM)
    outputs: list[TokenDistributionReplay | None] = [None] * len(rollout_payloads)
    for indices in _rollout_prompt_groups(rollout_payloads).values():
        group_payloads = [rollout_payloads[idx] for idx in indices]
        flat_prompt_ids = flat_prompt_ids_for_payload(group_payloads[0], patchilizer)
        group_outputs = batched_trajectory_token_log_dists(
            model,
            flat_prompt_ids,
            [payload.generated_patches for payload in group_payloads],
            precision,
            replay_context_patches=replay_context_patches,
            target_chunk_patches=target_chunk_patches,
            replay_batch_size=replay_batch_size,
        )
        for output_idx, replay in zip(indices, group_outputs, strict=True):
            outputs[output_idx] = replay
    if any(item is None for item in outputs):
        raise RuntimeError("grouped PPO token-distribution replay did not produce an output for every trajectory")
    return [item for item in outputs if item is not None]


def batched_trajectory_patch_values_by_prompt(
    model: NotaGenLMHeadModel,
    value_head: PatchValueHead,
    rollout_payloads: list[PPORolloutPayload],
    precision: str,
    replay_context_patches: int | None = None,
    target_chunk_patches: int = 0,
    replay_batch_size: int = 0,
    detach_policy: bool = True,
) -> list[torch.Tensor]:
    patchilizer = Patchilizer(stream=PATCH_STREAM)
    outputs: list[torch.Tensor | None] = [None] * len(rollout_payloads)
    for indices in _rollout_prompt_groups(rollout_payloads).values():
        group_payloads = [rollout_payloads[idx] for idx in indices]
        flat_prompt_ids = flat_prompt_ids_for_payload(group_payloads[0], patchilizer)
        group_outputs = batched_trajectory_patch_values(
            model,
            value_head,
            flat_prompt_ids,
            [payload.generated_patches for payload in group_payloads],
            precision,
            replay_context_patches=replay_context_patches,
            target_chunk_patches=target_chunk_patches,
            replay_batch_size=replay_batch_size,
            detach_policy=detach_policy,
        )
        for output_idx, values in zip(indices, group_outputs, strict=True):
            outputs[output_idx] = values
    if any(item is None for item in outputs):
        raise RuntimeError("grouped PPO value replay did not produce an output for every trajectory")
    return [item for item in outputs if item is not None]


def trajectory_patch_hidden_state_chunks(
    model: NotaGenLMHeadModel,
    flat_prompt_ids: list[int],
    generated_patches: list[list[int]],
    precision: str,
    replay_context_patches: int | None = None,
    target_chunk_patches: int = 0,
    detach_policy: bool = True,
):
    if not generated_patches:
        return

    current_ids = list(flat_prompt_ids)
    start_idx = 0
    while start_idx < len(generated_patches) and len(current_ids) % PATCH_SIZE != 0:
        patch = generated_patches[start_idx]
        yield hidden_state_from_last_patch(
            model,
            current_ids,
            precision,
            replay_context_patches=replay_context_patches,
            detach_policy=detach_policy,
        ).reshape(1, -1)
        current_ids.extend(
            normalize_patch_for_context(
                patch,
                eos_token_id=model.eos_token_id,
                special_token_id=model.special_token_id,
            )
        )
        start_idx += 1

    if start_idx >= len(generated_patches):
        return

    remaining_patches = generated_patches[start_idx:]
    chunk_size = len(remaining_patches) if target_chunk_patches <= 0 else target_chunk_patches
    for chunk_start in range(0, len(remaining_patches), chunk_size):
        chunk_end = min(len(remaining_patches), chunk_start + chunk_size)
        yield tail_patch_hidden_state_chunk(
            model,
            current_ids,
            remaining_patches,
            chunk_start,
            chunk_end,
            precision,
            replay_context_patches=replay_context_patches,
            detach_policy=detach_policy,
        )


def trajectory_patch_hidden_states(
    model: NotaGenLMHeadModel,
    flat_prompt_ids: list[int],
    generated_patches: list[list[int]],
    precision: str,
    replay_context_patches: int | None = None,
    target_chunk_patches: int = 0,
    detach_policy: bool = True,
) -> torch.Tensor:
    chunks = list(
        trajectory_patch_hidden_state_chunks(
            model,
            flat_prompt_ids,
            generated_patches,
            precision,
            replay_context_patches=replay_context_patches,
            target_chunk_patches=target_chunk_patches,
            detach_policy=detach_policy,
        )
    )
    device = next(model.parameters()).device
    if not chunks:
        hidden_size = getattr(model.patch_level_decoder.base.config, "n_embd", None)
        if hidden_size is None:
            hidden_size = getattr(model.patch_level_decoder.base.config, "hidden_size")
        return torch.empty((0, int(hidden_size)), device=device)
    return torch.cat(chunks, dim=0)


def terminal_returns(final_reward: float, length: int, gamma: float, device: torch.device) -> torch.Tensor:
    if length <= 0:
        return torch.empty(0, device=device)
    steps = torch.arange(length, device=device, dtype=torch.float32)
    discounts = torch.pow(torch.tensor(float(gamma), device=device, dtype=torch.float32), length - 1 - steps)
    return float(final_reward) * discounts


def discounted_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    returns = torch.empty_like(rewards, dtype=torch.float32)
    running = torch.zeros((), device=rewards.device, dtype=torch.float32)
    discount = torch.tensor(float(gamma), device=rewards.device, dtype=torch.float32)
    for idx in range(rewards.numel() - 1, -1, -1):
        running = rewards[idx].float() + discount * running
        returns[idx] = running
    return returns


def generalized_advantage_estimates(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rewards.shape != values.shape:
        raise RuntimeError(f"GAE tensor shape mismatch: rewards={tuple(rewards.shape)} values={tuple(values.shape)}")
    advantages = torch.empty_like(rewards, dtype=torch.float32)
    running = torch.zeros((), device=rewards.device, dtype=torch.float32)
    discount = torch.tensor(float(gamma), device=rewards.device, dtype=torch.float32)
    trace_decay = torch.tensor(float(gae_lambda), device=rewards.device, dtype=torch.float32)
    baseline = values.detach().float()
    for idx in range(rewards.numel() - 1, -1, -1):
        next_value = baseline[idx + 1] if idx + 1 < rewards.numel() else torch.zeros((), device=rewards.device)
        delta = rewards[idx].float() + discount * next_value - baseline[idx]
        running = delta + discount * trace_decay * running
        advantages[idx] = running
    value_targets = advantages + baseline
    return advantages, value_targets


def batch_trajectory_returns_advantages(
    reward_tensors: list[torch.Tensor],
    value_tensors: list[torch.Tensor],
    gamma: float,
    gae_lambda: float,
) -> PPOBatchTensors:
    if len(reward_tensors) != len(value_tensors):
        raise RuntimeError(
            f"PPO trajectory tensor count mismatch: rewards={len(reward_tensors)} values={len(value_tensors)}"
        )
    if not reward_tensors:
        raise RuntimeError("PPO batch must contain at least one trajectory")

    returns: list[torch.Tensor] = []
    advantages: list[torch.Tensor] = []
    value_targets: list[torch.Tensor] = []
    for rewards, values in zip(reward_tensors, value_tensors, strict=True):
        if rewards.shape != values.shape:
            raise RuntimeError(
                f"PPO trajectory tensor shape mismatch: rewards={tuple(rewards.shape)} values={tuple(values.shape)}"
            )
        returns.append(discounted_returns(rewards, gamma))
        trajectory_advantages, trajectory_value_targets = generalized_advantage_estimates(
            rewards=rewards,
            values=values.detach().float(),
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        advantages.append(trajectory_advantages)
        value_targets.append(trajectory_value_targets)

    return PPOBatchTensors(
        patch_rewards=torch.cat(reward_tensors),
        returns=torch.cat(returns),
        advantages=torch.cat(advantages),
        value_targets=torch.cat(value_targets),
    )


def score_total_reward(
    *,
    prompt_text: str,
    completion_text: str,
    target,
    reward_config: GoldbergRewardConfig,
    candidate_name: str,
    similarity_weights: SimilarityRewardWeights,
    aria_similarity_ref: SimilarityReference | None,
    similarity_chroma_bins: int,
    similarity_band_ratio: float,
    similarity_timeout_s: float,
    max_similarity_reward: float,
) -> RewardScore:
    breakdown = score_prompt_completion_pair(
        prompt_text=prompt_text,
        completion_text=completion_text,
        target=target,
        config=reward_config,
        candidate_name=candidate_name,
    )
    return _score_total_reward_from_structural_breakdown(
        prompt_text=prompt_text,
        completion_text=completion_text,
        structural_breakdown=breakdown,
        similarity_weights=similarity_weights,
        aria_similarity_ref=aria_similarity_ref,
        similarity_chroma_bins=similarity_chroma_bins,
        similarity_band_ratio=similarity_band_ratio,
        similarity_timeout_s=similarity_timeout_s,
        max_similarity_reward=max_similarity_reward,
    )


def _score_total_reward_from_structural_breakdown(
    *,
    prompt_text: str,
    completion_text: str,
    structural_breakdown,
    similarity_weights: SimilarityRewardWeights,
    aria_similarity_ref: SimilarityReference | None,
    similarity_chroma_bins: int,
    similarity_band_ratio: float,
    similarity_timeout_s: float,
    max_similarity_reward: float,
) -> RewardScore:
    breakdown = structural_breakdown
    reward_breakdown = breakdown.to_json()
    structural_total_reward = breakdown.total_reward
    similarity_payload = score_similarity_reward(
        prompt_text=prompt_text,
        completion_text=completion_text,
        weights=similarity_weights,
        aria=aria_similarity_ref,
        variation=None,
        bins=similarity_chroma_bins,
        band_ratio=similarity_band_ratio,
        timeout_s=similarity_timeout_s,
    )
    reward_breakdown.update(similarity_payload)
    raw_similarity_reward = float(similarity_payload.get("similarity_reward", 0.0))
    clipped_similarity_reward = raw_similarity_reward
    if max_similarity_reward > 0:
        clipped_similarity_reward = max(-max_similarity_reward, min(max_similarity_reward, raw_similarity_reward))
    similarity_validity_gate = 1.0 if reward_breakdown.get("parse_valid") else 0.0
    effective_similarity_reward = clipped_similarity_reward * similarity_validity_gate
    total_reward = structural_total_reward + effective_similarity_reward
    reward_breakdown["structural_total_reward"] = structural_total_reward
    reward_breakdown["raw_similarity_reward"] = raw_similarity_reward
    reward_breakdown["clipped_similarity_reward"] = clipped_similarity_reward
    reward_breakdown["similarity_validity_gate"] = similarity_validity_gate
    reward_breakdown["effective_similarity_reward"] = effective_similarity_reward
    reward_breakdown["total_reward"] = total_reward
    return RewardScore(total=total_reward, breakdown=reward_breakdown)


def generated_patch_completion_prefixes(generated_patches: list[list[int]]) -> list[str]:
    patchilizer = Patchilizer(stream=PATCH_STREAM)
    chars: list[str] = []
    prefixes: list[str] = []
    for patch in generated_patches:
        chars.extend(patchilizer.decode([patch]))
        prefixes.append("".join(chars))
    return prefixes


def _generated_patch_texts(generated_patches: list[list[int]]) -> list[str]:
    patchilizer = Patchilizer(stream=PATCH_STREAM)
    return ["".join(patchilizer.decode([patch])) for patch in generated_patches]


def _patch_char_spans(patch_texts: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    offset = 0
    for patch_text in patch_texts:
        start = offset
        offset += len(patch_text)
        spans.append((start, offset))
    return spans


def _stream_line_spans(completion_text: str) -> list[tuple[int, int]]:
    starts = [match.start() for match in re.finditer(r"\[r:\d+/\d+\]", completion_text)]
    if not starts:
        return []
    return [(start, end) for start, end in zip(starts, starts[1:] + [len(completion_text)], strict=True) if end > start]


def _safe_float_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value)


def _rollout_length_diagnostics(
    *,
    full_text: str,
    completion_text: str,
    generated_patch_count: int,
    prompt_stream_lines: int,
    target_stream_lines: int,
    stop_reason: str | None,
) -> dict:
    completion_spans = _stream_line_spans(completion_text)
    completion_line_chars = [end - start for start, end in completion_spans]
    completion_stream_lines = len(completion_spans)
    full_stream_lines = count_stream_lines(full_text)
    target_generated_stream_lines = max(0, int(target_stream_lines) - int(prompt_stream_lines))
    patches_per_stream_line = (
        float(generated_patch_count) / float(completion_stream_lines)
        if completion_stream_lines > 0
        else None
    )
    chars_per_stream_line_mean = (
        float(np.mean(completion_line_chars))
        if completion_line_chars
        else None
    )
    chars_per_stream_line_max = (
        int(max(completion_line_chars))
        if completion_line_chars
        else None
    )
    chars_per_stream_line_p95 = (
        float(np.percentile(completion_line_chars, 95))
        if completion_line_chars
        else None
    )
    return {
        "stop_reason": stop_reason,
        "full_stream_lines": int(full_stream_lines),
        "completion_stream_lines": int(completion_stream_lines),
        "prompt_stream_lines": int(prompt_stream_lines),
        "target_stream_lines": int(target_stream_lines),
        "target_generated_stream_lines": int(target_generated_stream_lines),
        "target_stream_lines_reached": bool(full_stream_lines >= target_stream_lines),
        "extra_stream_lines_after_target": int(max(0, full_stream_lines - target_stream_lines)),
        "missing_stream_lines_to_target": int(max(0, target_stream_lines - full_stream_lines)),
        "completion_missing_stream_lines_to_target": int(
            max(0, target_generated_stream_lines - completion_stream_lines)
        ),
        "generated_patches": int(generated_patch_count),
        "completion_chars": int(len(completion_text)),
        "patches_per_stream_line": _safe_float_or_none(patches_per_stream_line),
        "chars_per_stream_line_mean": _safe_float_or_none(chars_per_stream_line_mean),
        "chars_per_stream_line_max": chars_per_stream_line_max,
        "chars_per_stream_line_p95": _safe_float_or_none(chars_per_stream_line_p95),
        "max_generated_patches_hit": stop_reason == "max_generated_patches",
    }


def _rollout_length_summary(trajectory_logs: list[dict]) -> dict:
    if not trajectory_logs:
        return {}

    diagnostics = [
        log.get("rollout_length_diagnostics")
        or log.get("reward_breakdown", {})
        for log in trajectory_logs
    ]

    def values(key: str) -> list[float]:
        return [
            float(item[key])
            for item in diagnostics
            if isinstance(item, dict) and item.get(key) is not None
        ]

    def summary(key: str) -> dict | None:
        vals = values(key)
        if not vals:
            return None
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    return {
        "target_reached_count": int(
            sum(1 for item in diagnostics if isinstance(item, dict) and item.get("target_stream_lines_reached"))
        ),
        "max_generated_patches_hit_count": int(
            sum(1 for item in diagnostics if isinstance(item, dict) and item.get("max_generated_patches_hit"))
        ),
        "stop_reasons": {
            str(reason): int(
                sum(
                    1
                    for item in diagnostics
                    if isinstance(item, dict) and item.get("stop_reason") == reason
                )
            )
            for reason in sorted(
                {
                    item.get("stop_reason")
                    for item in diagnostics
                    if isinstance(item, dict) and item.get("stop_reason") is not None
                },
                key=str,
            )
        },
        "generated_patch_count": summary("generated_patches"),
        "completion_stream_lines": summary("completion_stream_lines"),
        "missing_stream_lines_to_target": summary("missing_stream_lines_to_target"),
        "patches_per_stream_line": summary("patches_per_stream_line"),
        "chars_per_stream_line_mean": summary("chars_per_stream_line_mean"),
        "chars_per_stream_line_max": summary("chars_per_stream_line_max"),
    }


def _rollout_sampling_summary(rollout_payloads: list[PPORolloutPayload]) -> dict:
    if not rollout_payloads:
        return {
            "sampled_candidates": 0,
            "kept_trajectories": 0,
            "success_candidates": 0,
            "failed_candidates": 0,
            "dropped_candidates": 0,
            "dropped_success_candidates": 0,
        }

    metas = [payload.meta or {} for payload in rollout_payloads]

    def max_meta_int(key: str, default: int) -> int:
        values = [int(meta[key]) for meta in metas if meta.get(key) is not None]
        return max(values) if values else default

    def max_meta_float(key: str, default: float) -> float:
        values = [float(meta[key]) for meta in metas if meta.get(key) is not None]
        return max(values) if values else default

    failed_kept = int(sum(1 for meta in metas if meta.get("rollout_failed")))
    sampled_candidates = max_meta_int("rollout_sampled_candidates", len(rollout_payloads))
    success_candidates = max_meta_int("rollout_success_candidates", len(rollout_payloads) - failed_kept)
    failed_candidates = max_meta_int("rollout_failed_candidates", failed_kept)
    dropped_candidates = max_meta_int("rollout_dropped_candidates", sampled_candidates - len(rollout_payloads))
    dropped_success_candidates = max_meta_int("rollout_dropped_success_candidates", 0)
    effective_batch_size = max_meta_int(
        "rollout_effective_batch_size",
        max_meta_int("rollout_batch_size", len(rollout_payloads)),
    )
    requested_batch_size = max_meta_int("rollout_requested_batch_size", effective_batch_size)
    failure_policy = next(
        (str(meta["rollout_failure_policy"]) for meta in metas if meta.get("rollout_failure_policy") is not None),
        None,
    )
    return {
        "sampled_candidates": int(sampled_candidates),
        "kept_trajectories": int(len(rollout_payloads)),
        "success_candidates": int(success_candidates),
        "failed_candidates": int(failed_candidates),
        "dropped_candidates": int(dropped_candidates),
        "dropped_success_candidates": int(dropped_success_candidates),
        "spares_percent": max_meta_float("rollout_spares_percent", 0.0),
        "effective_batch_size": int(effective_batch_size),
        "requested_batch_size": int(requested_batch_size),
        "failure_policy": failure_policy,
    }


def _stream_line_end_patch_indices(completion_text: str, patch_texts: list[str]) -> list[int]:
    patch_spans = _patch_char_spans(patch_texts)
    if not patch_spans:
        return []

    spans = _stream_line_spans(completion_text)
    if not spans:
        return []
    cumulative_offsets = [end for _start, end in patch_spans]
    ends = [end for _start, end in spans]
    return [min(bisect.bisect_left(cumulative_offsets, end), len(patch_texts) - 1) for end in ends]


def _line_reward_events(completion_text: str, line_rewards: list[float], *, name: str = "structural_line") -> list[RewardEvent]:
    spans = _stream_line_spans(completion_text)
    return [
        RewardEvent(start=start, end=end, value=float(value), name=name)
        for (start, end), value in zip(spans, line_rewards, strict=False)
        if value != 0.0
    ]


def _completion_harmony_tokens(completion_text: str) -> tuple[list[dict], list[tuple[int, int]]]:
    spans = _stream_line_spans(completion_text)
    return [infer_harmony(parse_bar_notes(completion_text[start:end])) for start, end in spans], spans


def _effective_similarity_component(raw_component: float, final_score: RewardScore) -> float:
    breakdown = final_score.breakdown
    raw_total = float(breakdown.get("raw_similarity_reward", 0.0))
    if raw_total == 0.0 or raw_component == 0.0:
        return 0.0
    clipped_total = float(breakdown.get("clipped_similarity_reward", raw_total))
    gate = float(breakdown.get("similarity_validity_gate", 1.0))
    return raw_component * (clipped_total / raw_total) * gate


def _dtw_metric_reward_events(
    *,
    name: str,
    reference: list,
    candidate: list,
    candidate_spans: list[tuple[int, int]],
    similarity_fn,
    total_value: float,
    band_ratio: float,
) -> list[RewardEvent]:
    if total_value == 0.0 or not reference or not candidate or not candidate_spans:
        return []
    alignment = generic_dtw_alignment(reference, candidate, similarity_fn, band_ratio=band_ratio)
    if not alignment.path:
        return []

    credits = [0.0 for _ in candidate_spans]
    for (_ref_idx, candidate_idx), local_similarity in zip(
        alignment.path,
        alignment.local_similarities,
        strict=True,
    ):
        if 0 <= candidate_idx < len(credits):
            credits[candidate_idx] += max(0.0, float(local_similarity))

    total_credit = sum(credits)
    if total_credit <= 0.0:
        return []

    return [
        RewardEvent(
            start=start,
            end=end,
            value=total_value * (credit / total_credit),
            name=name,
        )
        for credit, (start, end) in zip(credits, candidate_spans, strict=True)
        if credit > 0.0 and end > start
    ]


def _harmony_reward_events(
    *,
    completion_text: str,
    similarity_weights: SimilarityRewardWeights,
    aria_similarity_ref: SimilarityReference | None,
    final_score: RewardScore,
    band_ratio: float,
) -> list[RewardEvent]:
    if (
        similarity_weights.aria_harmony == 0.0
        or aria_similarity_ref is None
        or aria_similarity_ref.harmony is None
        or not final_score.breakdown.get("similarity_harmony_valid")
    ):
        return []

    candidate_harmony, candidate_spans = _completion_harmony_tokens(completion_text)
    if not candidate_harmony:
        return []

    weight_per_metric = similarity_weights.aria_harmony / 3.0
    metric_specs = [
        (
            "aria_harmony_harmony_dtw",
            aria_similarity_ref.harmony,
            candidate_harmony,
            token_similarity,
        ),
        (
            "aria_harmony_root_dtw",
            [item["root"] for item in aria_similarity_ref.harmony],
            [item["root"] for item in candidate_harmony],
            pitch_class_similarity,
        ),
        (
            "aria_harmony_bass_dtw",
            [item["bass"] for item in aria_similarity_ref.harmony],
            [item["bass"] for item in candidate_harmony],
            pitch_class_similarity,
        ),
    ]

    events: list[RewardEvent] = []
    for metric_name, reference, candidate, similarity_fn in metric_specs:
        metric_score = float(final_score.breakdown.get(metric_name, 0.0))
        total_value = _effective_similarity_component(weight_per_metric * metric_score, final_score)
        events.extend(
            _dtw_metric_reward_events(
                name=f"{metric_name}_effective",
                reference=reference,
                candidate=candidate,
                candidate_spans=candidate_spans,
                similarity_fn=similarity_fn,
                total_value=total_value,
                band_ratio=band_ratio,
            )
        )
    return events


def _project_reward_events_to_patches(events: list[RewardEvent], patch_texts: list[str]) -> list[float]:
    patch_spans = _patch_char_spans(patch_texts)
    rewards = [0.0 for _ in patch_spans]
    if not patch_spans:
        return rewards

    completion_len = patch_spans[-1][1]
    for event in events:
        start = max(0, min(completion_len, event.start))
        end = max(start, min(completion_len, event.end))
        if end <= start or event.value == 0.0:
            continue

        overlaps: list[tuple[int, int]] = []
        for patch_idx, (patch_start, patch_end) in enumerate(patch_spans):
            overlap = max(0, min(end, patch_end) - max(start, patch_start))
            if overlap > 0:
                overlaps.append((patch_idx, overlap))
        total_overlap = sum(overlap for _patch_idx, overlap in overlaps)
        if total_overlap <= 0:
            continue
        for patch_idx, overlap in overlaps:
            rewards[patch_idx] += event.value * (overlap / total_overlap)
    return rewards


def _project_reward_events_by_name_to_patches(
    events: list[RewardEvent],
    patch_texts: list[str],
) -> dict[str, list[float]]:
    event_names = sorted({event.name for event in events})
    return {
        name: _project_reward_events_to_patches(
            [event for event in events if event.name == name],
            patch_texts,
        )
        for name in event_names
    }


def _terminal_patch_rewards(patch_count: int, value: float) -> list[float]:
    rewards = [0.0 for _idx in range(patch_count)]
    if rewards and value != 0.0:
        rewards[-1] = float(value)
    return rewards


def _terminal_structural_component_rewards(
    *,
    final_score: RewardScore,
    reward_config: GoldbergRewardConfig,
    patch_count: int,
) -> dict[str, list[float]]:
    breakdown = final_score.breakdown
    component_weights = {
        "parse_reward": reward_config.parse_weight,
        "countdown_reward": reward_config.countdown_weight,
        "line_closure_reward": reward_config.line_closure_weight,
        "bar_token_reward": reward_config.bar_token_weight,
        "meter_alignment_reward": reward_config.meter_alignment_weight,
        "meter_duration_closeness_reward": reward_config.meter_duration_closeness_weight,
        "bar_meter_consistency_reward": reward_config.bar_meter_consistency_weight,
        "bar_count_reward": reward_config.bar_count_weight,
        "voice_declaration_reward": reward_config.voice_declaration_weight,
        "score_voice_reward": reward_config.score_voice_weight,
    }
    component_rewards: dict[str, list[float]] = {}
    for component_name, weight in component_weights.items():
        value = float(weight) * float(breakdown.get(component_name, 0.0))
        if value != 0.0:
            component_rewards[component_name] = _terminal_patch_rewards(patch_count, value)

    gate_adjustment = float(breakdown.get("structural_validity_gate_adjustment", 0.0))
    if gate_adjustment != 0.0:
        component_rewards["structural_validity_gate_adjustment"] = _terminal_patch_rewards(
            patch_count,
            gate_adjustment,
        )
    return component_rewards


def _terminal_similarity_component_rewards(
    *,
    final_score: RewardScore,
    similarity_weights: SimilarityRewardWeights,
    patch_count: int,
) -> dict[str, list[float]]:
    component_rewards: dict[str, list[float]] = {}
    if similarity_weights.aria_chroma != 0.0:
        chroma_component = _effective_similarity_component(
            similarity_weights.aria_chroma * float(final_score.breakdown.get("aria_chroma_harmonic_hist", 0.0)),
            final_score,
        )
        if chroma_component != 0.0:
            component_rewards["aria_chroma_harmonic_hist_effective"] = _terminal_patch_rewards(
                patch_count,
                chroma_component,
            )

    if similarity_weights.aria_harmony != 0.0:
        weight_per_metric = similarity_weights.aria_harmony / 3.0
        for metric_name in (
            "aria_harmony_harmony_dtw",
            "aria_harmony_root_dtw",
            "aria_harmony_bass_dtw",
        ):
            component = _effective_similarity_component(
                weight_per_metric * float(final_score.breakdown.get(metric_name, 0.0)),
                final_score,
            )
            if component != 0.0:
                component_rewards[f"{metric_name}_effective"] = _terminal_patch_rewards(patch_count, component)
    return component_rewards


def _patch_reward_trace_from_terminal_components(
    *,
    final_score: RewardScore,
    component_rewards: dict[str, list[float]],
    patch_count: int,
) -> PatchRewardTrace:
    rewards = [
        float(sum(component_rewards[name][idx] for name in component_rewards))
        for idx in range(patch_count)
    ]
    terminal_residual = final_score.total - sum(rewards)
    if rewards and terminal_residual != 0.0:
        component_rewards["other_residual"] = _terminal_patch_rewards(patch_count, terminal_residual)
        rewards[-1] += terminal_residual
    else:
        component_rewards["other_residual"] = [0.0 for _idx in range(patch_count)]

    return PatchRewardTrace(
        rewards=rewards,
        prefix_totals=prefix_totals(rewards),
        final_score=final_score,
        component_rewards=component_rewards,
        component_prefix_totals=component_prefix_totals(component_rewards),
    )


def _strip_notagen_control_tags(text: str) -> str:
    return re.sub(r"\[(?:r:[^\]]*|V:[^\]]*|I:[^\]]*)\]", "", text)


def _count_abc_note_letter(text: str, note: str) -> int:
    note_letter = note.strip()
    if len(note_letter) != 1 or note_letter.upper() not in {"A", "B", "C", "D", "E", "F", "G"}:
        raise ValueError(f"simple note-count reward note must be one ABC pitch letter A-G, got {note!r}")
    stripped = _strip_notagen_control_tags(text)
    target = note_letter.lower()
    return sum(1 for char in stripped if char.lower() == target)


def _count_abc_note_letters(text: str) -> int:
    stripped = _strip_notagen_control_tags(text)
    return sum(1 for char in stripped if char.lower() in {"a", "b", "c", "d", "e", "f", "g"})


def _clipped_increment_rewards(values: list[float], *, target: float, scale: float) -> list[float]:
    if target <= 0:
        raise ValueError(f"simple reward target must be positive, got {target}")
    rewards: list[float] = []
    running = 0.0
    previous_clipped = 0.0
    for value in values:
        running += max(0.0, float(value))
        clipped = min(float(target), running)
        rewards.append(float(scale) * (clipped - previous_clipped) / float(target))
        previous_clipped = clipped
    return rewards


def patch_rewards_simple_test(
    *,
    generated_patches: list[list[int]],
    scoring_options: PPORewardScoringOptions,
) -> PatchRewardTrace:
    patch_texts = _generated_patch_texts(generated_patches)
    completion_text = "".join(patch_texts)
    mode = scoring_options.reward_mode
    component_name = f"simple_{mode}_reward"
    component_rewards: dict[str, list[float]]

    if mode == "note_count":
        patch_values = [
            float(_count_abc_note_letter(patch_text, scoring_options.simple_reward_note))
            for patch_text in patch_texts
        ]
        metric_value = float(sum(patch_values))
        metric_target = float(scoring_options.simple_reward_max_count)
        detail_key = "simple_reward_note_count"
        detail_value = metric_value
    elif mode == "note_fraction":
        if scoring_options.patch_reward_attribution != "terminal":
            raise RuntimeError("note_fraction simple reward requires terminal patch_reward_attribution")
        note_count = float(_count_abc_note_letter(completion_text, scoring_options.simple_reward_note))
        total_note_count = float(_count_abc_note_letters(completion_text))
        fraction = note_count / total_note_count if total_note_count > 0 else 0.0
        total_reward = float(scoring_options.simple_reward_scale) * fraction
        breakdown = {
            "reward_mode": mode,
            "parse_valid": True,
            "parse_reward": 1.0,
            "structural_total_reward": 0.0,
            "raw_similarity_reward": 0.0,
            "clipped_similarity_reward": 0.0,
            "similarity_validity_gate": 1.0,
            "effective_similarity_reward": 0.0,
            "simple_reward_component": component_name,
            "simple_reward_scale": float(scoring_options.simple_reward_scale),
            "simple_reward_note": scoring_options.simple_reward_note,
            "simple_reward_note_count": note_count,
            "simple_reward_total_note_count": total_note_count,
            "simple_reward_fraction": fraction,
            "simple_reward_normalized": fraction,
            "total_reward": total_reward,
        }
        final_score = RewardScore(total=total_reward, breakdown=breakdown)
        component_rewards = {component_name: _terminal_patch_rewards(len(patch_texts), total_reward)}
        return _patch_reward_trace_from_terminal_components(
            final_score=final_score,
            component_rewards=component_rewards,
            patch_count=len(patch_texts),
        )
    elif mode == "length":
        unit = scoring_options.simple_reward_length_unit
        if unit == "patches":
            patch_values = [1.0 for _patch_text in patch_texts]
        elif unit == "chars":
            patch_values = [float(len(patch_text)) for patch_text in patch_texts]
        elif unit == "stream_lines":
            cumulative_values: list[float] = []
            prefix = ""
            for patch_text in patch_texts:
                prefix += patch_text
                cumulative_values.append(float(count_stream_lines(prefix)))
            previous = 0.0
            patch_values = []
            for cumulative in cumulative_values:
                patch_values.append(max(0.0, cumulative - previous))
                previous = cumulative
        else:
            raise RuntimeError(f"unsupported simple length unit: {unit!r}")
        metric_value = float(sum(patch_values))
        metric_target = float(scoring_options.simple_reward_length_target)
        detail_key = f"simple_reward_length_{unit}"
        detail_value = metric_value
    else:
        raise RuntimeError(f"unsupported simple reward mode: {mode!r}")

    total_reward = float(scoring_options.simple_reward_scale) * min(metric_value, metric_target) / metric_target
    breakdown = {
        "reward_mode": mode,
        "parse_valid": True,
        "parse_reward": 1.0,
        "structural_total_reward": 0.0,
        "raw_similarity_reward": 0.0,
        "clipped_similarity_reward": 0.0,
        "similarity_validity_gate": 1.0,
        "effective_similarity_reward": 0.0,
        "simple_reward_component": component_name,
        "simple_reward_scale": float(scoring_options.simple_reward_scale),
        "simple_reward_target": metric_target,
        "simple_reward_metric_value": metric_value,
        "simple_reward_normalized": min(metric_value, metric_target) / metric_target,
        detail_key: detail_value,
        "total_reward": total_reward,
    }
    if mode == "note_count":
        breakdown["simple_reward_note"] = scoring_options.simple_reward_note
    if mode == "length":
        breakdown["simple_reward_length_unit"] = scoring_options.simple_reward_length_unit

    final_score = RewardScore(total=total_reward, breakdown=breakdown)
    if not patch_texts:
        return PatchRewardTrace(
            rewards=[],
            prefix_totals=[],
            final_score=final_score,
            component_rewards={},
            component_prefix_totals={},
        )

    if scoring_options.patch_reward_attribution == "terminal":
        component_rewards = {component_name: _terminal_patch_rewards(len(patch_texts), total_reward)}
        return _patch_reward_trace_from_terminal_components(
            final_score=final_score,
            component_rewards=component_rewards,
            patch_count=len(patch_texts),
        )

    component_rewards = {
        component_name: _clipped_increment_rewards(
            patch_values,
            target=metric_target,
            scale=float(scoring_options.simple_reward_scale),
        )
    }
    rewards = component_rewards[component_name][:]
    residual = total_reward - sum(rewards)
    if residual != 0.0:
        rewards[-1] += residual
        component_rewards["other_residual"] = _terminal_patch_rewards(len(patch_texts), residual)
    else:
        component_rewards["other_residual"] = [0.0 for _idx in patch_texts]
    return PatchRewardTrace(
        rewards=rewards,
        prefix_totals=prefix_totals(rewards),
        final_score=final_score,
        component_rewards=component_rewards,
        component_prefix_totals=component_prefix_totals(component_rewards),
    )


def _countdown_local_rewards(stream_lines) -> np.ndarray:
    if not stream_lines:
        return np.zeros(0, dtype=np.float32)
    rewards = np.zeros(len(stream_lines), dtype=np.float32)
    if stream_lines[0].index == 0:
        rewards[0] += 1.0
    for idx, (prev_line, curr_line) in enumerate(zip(stream_lines, stream_lines[1:]), start=1):
        if curr_line.index == prev_line.index + 1 and curr_line.tag_marker == prev_line.tag_marker - 1:
            rewards[idx] += 1.0
    if stream_lines[-1].tag_marker == 0:
        rewards[-1] += 1.0
    return rewards


def _single_pass_line_reward_components(
    *,
    full_text: str,
    target,
    reward_config: GoldbergRewardConfig,
) -> dict[str, list[float]]:
    stream_lines = _extract_stream_line_features(full_text)
    if not stream_lines:
        return {}

    header = _extract_header_context(full_text)
    local_metrics = _stream_line_local_metrics(stream_lines, header)
    return _line_reward_components_from_metrics(
        stream_lines=stream_lines,
        local_metrics=local_metrics,
        target=target,
        reward_config=reward_config,
    )


def _line_reward_components_from_metrics(
    *,
    stream_lines,
    local_metrics,
    target,
    reward_config: GoldbergRewardConfig,
) -> dict[str, list[float]]:
    if not stream_lines:
        return {}

    n = len(stream_lines)
    closure = np.array([1.0 if line.closed else 0.0 for line in stream_lines], dtype=np.float32)
    bar_token = np.array([1.0 if line.has_bar_token else 0.0 for line in stream_lines], dtype=np.float32)
    countdown = _countdown_local_rewards(stream_lines)
    meter_alignment = np.array(local_metrics.meter_alignment_reward, dtype=np.float32)
    meter_duration = np.array(local_metrics.meter_duration_closeness_reward, dtype=np.float32)
    bar_meter = np.array(local_metrics.bar_meter_consistency_reward, dtype=np.float32)
    voice_decl = np.array(local_metrics.voice_declaration_reward, dtype=np.float32)
    score_voice = np.array(local_metrics.score_voice_reward, dtype=np.float32)

    line_denominator = float(max(1, n))
    components: dict[str, np.ndarray] = {}

    def add_weighted_component(name: str, weight: float, values: np.ndarray) -> None:
        if weight != 0.0:
            components[name] = weight * values / line_denominator

    add_weighted_component("countdown_reward", reward_config.countdown_weight, countdown)
    add_weighted_component("line_closure_reward", reward_config.line_closure_weight, closure)
    add_weighted_component("bar_token_reward", reward_config.bar_token_weight, bar_token)
    add_weighted_component("meter_alignment_reward", reward_config.meter_alignment_weight, meter_alignment)
    add_weighted_component(
        "meter_duration_closeness_reward",
        reward_config.meter_duration_closeness_weight,
        meter_duration,
    )
    add_weighted_component("bar_meter_consistency_reward", reward_config.bar_meter_consistency_weight, bar_meter)
    add_weighted_component("voice_declaration_reward", reward_config.voice_declaration_weight, voice_decl)
    add_weighted_component("score_voice_reward", reward_config.score_voice_weight, score_voice)

    counts = np.arange(1, n + 1, dtype=np.float32)
    previous_counts = np.arange(0, n, dtype=np.float32)
    expected = float(target.expected_reward_bars)
    if expected > 0 and reward_config.bar_count_weight != 0.0:
        bar_count = np.maximum(0.0, 1.0 - np.abs(counts - expected) / expected)
        previous_bar_count = np.maximum(0.0, 1.0 - np.abs(previous_counts - expected) / expected)
        components["bar_count_reward"] = reward_config.bar_count_weight * (bar_count - previous_bar_count)

    return {name: [float(item) for item in values] for name, values in components.items()}


def _single_pass_line_rewards(
    *,
    full_text: str,
    target,
    reward_config: GoldbergRewardConfig,
) -> list[float]:
    components = _single_pass_line_reward_components(
        full_text=full_text,
        target=target,
        reward_config=reward_config,
    )
    if not components:
        return []
    component_values = list(components.values())
    return [
        float(sum(values[idx] for values in component_values))
        for idx in range(len(component_values[0]))
    ]


def patch_rewards_single_pass(
    *,
    prompt_text: str,
    generated_patches: list[list[int]],
    target,
    reward_config: GoldbergRewardConfig,
    candidate_name: str,
    similarity_weights: SimilarityRewardWeights,
    aria_similarity_ref: SimilarityReference | None,
    similarity_chroma_bins: int,
    similarity_band_ratio: float,
    similarity_timeout_s: float,
    max_similarity_reward: float,
) -> PatchRewardTrace:
    patch_texts = _generated_patch_texts(generated_patches)
    completion_text = "".join(patch_texts)
    if generated_patches:
        structural_score = score_candidate_text_with_local_metrics(
            abc_text=prompt_text + completion_text,
            target=target,
            config=reward_config,
            candidate_name=f"{candidate_name}_final",
        )
        final_score = _score_total_reward_from_structural_breakdown(
            prompt_text=prompt_text,
            completion_text=completion_text,
            structural_breakdown=structural_score.breakdown,
            similarity_weights=similarity_weights,
            aria_similarity_ref=aria_similarity_ref,
            similarity_chroma_bins=similarity_chroma_bins,
            similarity_band_ratio=similarity_band_ratio,
            similarity_timeout_s=similarity_timeout_s,
            max_similarity_reward=max_similarity_reward,
        )
    else:
        final_score = score_total_reward(
            prompt_text=prompt_text,
            completion_text="",
            target=target,
            reward_config=reward_config,
            candidate_name=f"{candidate_name}_empty",
            similarity_weights=similarity_weights,
            aria_similarity_ref=aria_similarity_ref,
            similarity_chroma_bins=similarity_chroma_bins,
            similarity_band_ratio=similarity_band_ratio,
            similarity_timeout_s=similarity_timeout_s,
            max_similarity_reward=max_similarity_reward,
        )
        return PatchRewardTrace(
            rewards=[],
            prefix_totals=[],
            final_score=final_score,
            component_rewards={},
            component_prefix_totals={},
        )

    line_reward_components = _line_reward_components_from_metrics(
        stream_lines=structural_score.stream_lines,
        local_metrics=structural_score.local_metrics,
        target=target,
        reward_config=reward_config,
    )
    component_rewards: dict[str, list[float]] = {}
    for component_name, line_rewards in line_reward_components.items():
        component_rewards[component_name] = _project_reward_events_to_patches(
            _line_reward_events(completion_text, line_rewards, name=component_name),
            patch_texts,
        )

    harmony_events = _harmony_reward_events(
        completion_text=completion_text,
        similarity_weights=similarity_weights,
        aria_similarity_ref=aria_similarity_ref,
        final_score=final_score,
        band_ratio=similarity_band_ratio,
    )
    component_rewards.update(_project_reward_events_by_name_to_patches(harmony_events, patch_texts))

    if reward_config.parse_weight != 0.0:
        parse_component = reward_config.parse_weight * float(final_score.breakdown.get("parse_reward", 0.0))
        component_rewards["parse_reward"] = _terminal_patch_rewards(len(patch_texts), parse_component)

    structural_gate_adjustment = float(final_score.breakdown.get("structural_validity_gate_adjustment", 0.0))
    if structural_gate_adjustment != 0.0:
        component_rewards["structural_validity_gate_adjustment"] = _terminal_patch_rewards(
            len(patch_texts),
            structural_gate_adjustment,
        )

    if similarity_weights.aria_chroma != 0.0:
        chroma_component = _effective_similarity_component(
            similarity_weights.aria_chroma * float(final_score.breakdown.get("aria_chroma_harmonic_hist", 0.0)),
            final_score,
        )
        component_rewards["aria_chroma_harmonic_hist_effective"] = _terminal_patch_rewards(
            len(patch_texts),
            chroma_component,
        )

    rewards = [
        float(sum(component_rewards[name][idx] for name in component_rewards))
        for idx in range(len(patch_texts))
    ]
    terminal_residual = final_score.total - sum(rewards)
    if terminal_residual != 0.0:
        component_rewards["other_residual"] = _terminal_patch_rewards(len(patch_texts), terminal_residual)
        rewards[-1] += terminal_residual
    else:
        component_rewards["other_residual"] = [0.0 for _idx in patch_texts]

    reward_prefix_totals = prefix_totals(rewards)
    return PatchRewardTrace(
        rewards=rewards,
        prefix_totals=reward_prefix_totals,
        final_score=final_score,
        component_rewards=component_rewards,
        component_prefix_totals=component_prefix_totals(component_rewards),
    )


def patch_rewards_terminal(
    *,
    prompt_text: str,
    generated_patches: list[list[int]],
    target,
    reward_config: GoldbergRewardConfig,
    candidate_name: str,
    similarity_weights: SimilarityRewardWeights,
    aria_similarity_ref: SimilarityReference | None,
    similarity_chroma_bins: int,
    similarity_band_ratio: float,
    similarity_timeout_s: float,
    max_similarity_reward: float,
) -> PatchRewardTrace:
    patch_texts = _generated_patch_texts(generated_patches)
    completion_text = "".join(patch_texts)
    final_score = score_total_reward(
        prompt_text=prompt_text,
        completion_text=completion_text,
        target=target,
        reward_config=reward_config,
        candidate_name=f"{candidate_name}_terminal",
        similarity_weights=similarity_weights,
        aria_similarity_ref=aria_similarity_ref,
        similarity_chroma_bins=similarity_chroma_bins,
        similarity_band_ratio=similarity_band_ratio,
        similarity_timeout_s=similarity_timeout_s,
        max_similarity_reward=max_similarity_reward,
    )
    if not generated_patches:
        return PatchRewardTrace(
            rewards=[],
            prefix_totals=[],
            final_score=final_score,
            component_rewards={},
            component_prefix_totals={},
        )

    component_rewards = _terminal_structural_component_rewards(
        final_score=final_score,
        reward_config=reward_config,
        patch_count=len(patch_texts),
    )
    component_rewards.update(
        _terminal_similarity_component_rewards(
            final_score=final_score,
            similarity_weights=similarity_weights,
            patch_count=len(patch_texts),
        )
    )
    return _patch_reward_trace_from_terminal_components(
        final_score=final_score,
        component_rewards=component_rewards,
        patch_count=len(patch_texts),
    )


def patch_rewards_from_prefix_deltas(**kwargs) -> PatchRewardTrace:
    return patch_rewards_single_pass(**kwargs)


def normalize_advantages(advantages: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = advantages.mean()
    std = advantages.std(unbiased=False)
    if advantages.numel() <= 1 or std <= eps:
        return advantages - mean, mean, std
    return (advantages - mean) / (std + eps), mean, std


def normalize_advantages_token_weighted(
    advantages: torch.Tensor,
    token_counts: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if advantages.shape != token_counts.shape:
        raise RuntimeError(
            "token-weighted advantage shape mismatch: "
            f"advantages={tuple(advantages.shape)} token_counts={tuple(token_counts.shape)}"
        )
    weights = token_counts.detach().float().to(advantages.device)
    total_weight = weights.sum()
    if advantages.numel() <= 1 or total_weight <= 0:
        return normalize_advantages(advantages, eps=eps)
    mean = (advantages.float() * weights).sum() / total_weight
    variance = (((advantages.float() - mean) ** 2) * weights).sum() / total_weight
    std = torch.sqrt(variance)
    if std <= eps:
        return advantages.float() - mean, mean, std
    return (advantages.float() - mean) / (std + eps), mean, std


def weighted_mean(values: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if weights is None:
        return values.float().mean()
    weights = weights.detach().float().to(values.device)
    if weights.shape != values.shape:
        raise RuntimeError(f"weighted mean shape mismatch: values={tuple(values.shape)} weights={tuple(weights.shape)}")
    total_weight = weights.sum()
    if total_weight <= 0:
        raise RuntimeError("weighted mean requires positive total weight")
    return (values.float() * weights).sum() / total_weight


def weighted_std(values: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if weights is None:
        return values.float().std(unbiased=False)
    mean = weighted_mean(values.float(), weights)
    return torch.sqrt(weighted_mean((values.float() - mean) ** 2, weights))


def exact_categorical_kl(policy_log_dists: torch.Tensor, reference_log_dists: torch.Tensor) -> torch.Tensor:
    if policy_log_dists.shape != reference_log_dists.shape:
        raise RuntimeError(
            "exact categorical KL shape mismatch: "
            f"policy={tuple(policy_log_dists.shape)} reference={tuple(reference_log_dists.shape)}"
        )
    if policy_log_dists.ndim != 2:
        raise RuntimeError(f"exact categorical KL expects [tokens, vocab], got {tuple(policy_log_dists.shape)}")
    if policy_log_dists.shape[0] == 0:
        raise RuntimeError("exact categorical KL needs at least one generated token")
    policy_log_dists = policy_log_dists.float()
    reference_log_dists = reference_log_dists.detach().float()
    policy_probs = policy_log_dists.exp()
    return (policy_probs * (policy_log_dists - reference_log_dists)).sum(dim=-1).mean()


def token_patch_indices_from_counts(token_counts: torch.Tensor) -> torch.Tensor:
    token_counts = token_counts.detach().long()
    return torch.repeat_interleave(
        torch.arange(token_counts.numel(), device=token_counts.device, dtype=torch.long),
        token_counts,
    )


def value_mse_loss(
    values: torch.Tensor,
    value_targets: torch.Tensor,
    *,
    normalize_value_loss: bool = False,
    eps: float = 1e-6,
    scale_min: float = 1e-6,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    squared_error = (values.float() - value_targets.detach().float()) ** 2
    raw_value_loss = weighted_mean(squared_error, weights)
    if not normalize_value_loss:
        return raw_value_loss, raw_value_loss, torch.ones((), device=values.device, dtype=torch.float32)

    target_std = weighted_std(value_targets.detach().float(), weights)
    scale = torch.clamp(target_std, min=max(float(eps), float(scale_min)))
    scaled_loss = weighted_mean(((values.float() / scale) - (value_targets.detach().float() / scale)) ** 2, weights)
    return scaled_loss, raw_value_loss, scale


def ppo_clipped_loss(
    *,
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    values: torch.Tensor,
    old_values: torch.Tensor,
    advantages: torch.Tensor,
    value_targets: torch.Tensor,
    clip_range: float,
    value_loss_coef: float,
    entropy_bonus_coef: float = 0.0,
    normalize_advantage: bool = True,
    normalize_value_loss: bool = False,
    value_loss_eps: float = 1e-6,
    value_loss_scale_min: float = 1e-6,
    normalized_advantages: torch.Tensor | None = None,
    advantages_mean: torch.Tensor | None = None,
    advantages_std: torch.Tensor | None = None,
    fixed_value_loss_scale: torch.Tensor | None = None,
    policy_patch_indices: torch.Tensor | None = None,
    value_token_counts: torch.Tensor | None = None,
    new_log_dists: torch.Tensor | None = None,
    old_log_dists: torch.Tensor | None = None,
    reference_log_dists: torch.Tensor | None = None,
    reference_kl_coef: float = 0.0,
) -> PPOLossPayload:
    if not (values.shape == old_values.shape == advantages.shape == value_targets.shape):
        raise RuntimeError(
            "PPO tensor shape mismatch: "
            f"values={tuple(values.shape)} old_values={tuple(old_values.shape)} "
            f"advantages={tuple(advantages.shape)} value_targets={tuple(value_targets.shape)}"
        )
    if new_logprobs.shape != old_logprobs.shape:
        raise RuntimeError(
            "PPO policy logprob shape mismatch: "
            f"new={tuple(new_logprobs.shape)} old={tuple(old_logprobs.shape)}"
        )
    if reference_kl_coef != 0.0 and reference_log_dists is None:
        raise RuntimeError("--reference-kl-coef requires reference token log-distributions")
    if (new_log_dists is None) and (old_log_dists is not None or reference_log_dists is not None):
        raise RuntimeError("PPO exact KL diagnostics require current token log-distributions")
    if new_log_dists is not None and new_log_dists.shape[0] != new_logprobs.numel():
        raise RuntimeError(
            "PPO current log-distribution/token shape mismatch: "
            f"log_dists={tuple(new_log_dists.shape)} logprobs={tuple(new_logprobs.shape)}"
        )
    if policy_patch_indices is None:
        raise RuntimeError("PPO loss is token-level and requires policy_patch_indices")
    raw_advantages = advantages.detach().float()
    policy_indices = None
    policy_indices = policy_patch_indices.detach().long().to(raw_advantages.device)
    if policy_indices.ndim != 1 or policy_indices.numel() != new_logprobs.numel():
        raise RuntimeError(
            "PPO token policy index shape mismatch: "
            f"indices={tuple(policy_indices.shape)} logprobs={tuple(new_logprobs.shape)}"
        )
    if policy_indices.numel() == 0:
        raise RuntimeError("PPO token-level policy loss needs at least one generated token")
    if int(policy_indices.min().detach().cpu()) < 0 or int(policy_indices.max().detach().cpu()) >= raw_advantages.numel():
        raise RuntimeError(
            "PPO token policy index out of range: "
            f"indices=({int(policy_indices.min().detach().cpu())}, {int(policy_indices.max().detach().cpu())}) "
            f"patches={raw_advantages.numel()}"
        )

    if value_token_counts is None:
        value_token_counts = torch.bincount(policy_indices, minlength=raw_advantages.numel()).to(raw_advantages.device)
    elif value_token_counts is not None:
        value_token_counts = value_token_counts.detach().long().to(raw_advantages.device)
        if value_token_counts.shape != values.shape:
            raise RuntimeError(
                "PPO value token-count shape mismatch: "
                f"token_counts={tuple(value_token_counts.shape)} values={tuple(values.shape)}"
            )
        if value_token_counts.sum() <= 0:
            raise RuntimeError("PPO token-level value loss needs at least one generated token")

    if normalized_advantages is not None:
        if normalized_advantages.shape != raw_advantages.shape:
            raise RuntimeError(
                "PPO normalized advantage shape mismatch: "
                f"normalized={tuple(normalized_advantages.shape)} raw={tuple(raw_advantages.shape)}"
            )
        advantages = normalized_advantages.detach().float()
        adv_mean = advantages_mean.detach().float() if advantages_mean is not None else raw_advantages.mean()
        adv_std = advantages_std.detach().float() if advantages_std is not None else raw_advantages.std(unbiased=False)
    elif normalize_advantage:
        if policy_indices is None:
            advantages, adv_mean, adv_std = normalize_advantages(raw_advantages)
        else:
            token_counts = torch.bincount(policy_indices, minlength=raw_advantages.numel()).to(raw_advantages.device)
            advantages, adv_mean, adv_std = normalize_advantages_token_weighted(raw_advantages, token_counts)
    else:
        advantages = raw_advantages
        adv_mean = raw_advantages.mean()
        adv_std = raw_advantages.std(unbiased=False)

    policy_advantages = advantages[policy_indices] if policy_indices is not None else advantages
    log_ratio = new_logprobs - old_logprobs.detach()
    ratio = torch.exp(log_ratio)
    unclipped = ratio * policy_advantages.detach()
    clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * policy_advantages.detach()
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    if fixed_value_loss_scale is not None and normalize_value_loss:
        value_loss_scale = fixed_value_loss_scale.detach().float().to(values.device)
        raw_value_loss = weighted_mean(
            (values.float() - value_targets.detach().float()) ** 2,
            value_token_counts,
        )
        value_loss = weighted_mean(
            ((values.float() / value_loss_scale) - (value_targets.detach().float() / value_loss_scale)) ** 2,
            value_token_counts,
        )
    else:
        value_loss, raw_value_loss, value_loss_scale = value_mse_loss(
            values,
            value_targets,
            normalize_value_loss=normalize_value_loss,
            eps=value_loss_eps,
            scale_min=value_loss_scale_min,
            weights=value_token_counts,
        )
    if new_log_dists is not None:
        entropy = -(new_log_dists.float().exp() * new_log_dists.float()).sum(dim=-1).mean()
    else:
        entropy = (-new_logprobs).mean()
    entropy_loss = -entropy_bonus_coef * entropy
    zero = torch.zeros((), device=new_logprobs.device, dtype=torch.float32)
    old_policy_exact_kl = (
        exact_categorical_kl(new_log_dists, old_log_dists)
        if new_log_dists is not None and old_log_dists is not None
        else zero
    )
    reference_exact_kl = (
        exact_categorical_kl(new_log_dists, reference_log_dists)
        if new_log_dists is not None and reference_log_dists is not None
        else zero
    )
    reference_kl_loss = float(reference_kl_coef) * reference_exact_kl
    loss = policy_loss + value_loss_coef * value_loss + entropy_loss + reference_kl_loss
    approx_kl = ((old_logprobs.detach() - new_logprobs) ** 2).mean() * 0.5
    clip_fraction = ((ratio - 1.0).abs() > clip_range).float().mean()
    return PPOLossPayload(
        loss=loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        raw_value_loss=raw_value_loss,
        value_loss_scale=value_loss_scale,
        entropy_loss=entropy_loss,
        reference_kl_loss=reference_kl_loss,
        approx_kl=approx_kl,
        old_policy_exact_kl=old_policy_exact_kl,
        reference_exact_kl=reference_exact_kl,
        clip_fraction=clip_fraction,
        advantages_mean=adv_mean,
        advantages_std=adv_std,
    )


def _ppo_loss_constants(
    batch_tensors: PPOBatchTensors,
    args,
    token_counts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_advantages = batch_tensors.advantages.detach().float()
    if args.no_advantage_normalization:
        normalized_advantages = raw_advantages
        advantages_mean = raw_advantages.mean()
        advantages_std = raw_advantages.std(unbiased=False)
    elif token_counts is not None:
        normalized_advantages, advantages_mean, advantages_std = normalize_advantages_token_weighted(
            raw_advantages,
            token_counts.detach().long().to(raw_advantages.device),
        )
    else:
        normalized_advantages, advantages_mean, advantages_std = normalize_advantages(raw_advantages)

    if args.normalize_value_loss:
        target_std = weighted_std(
            batch_tensors.value_targets.detach().float(),
            None if token_counts is None else token_counts.detach().long().to(batch_tensors.value_targets.device),
        )
        value_loss_scale = torch.clamp(
            target_std,
            min=max(float(args.value_loss_eps), float(args.value_loss_scale_min)),
        )
    else:
        value_loss_scale = torch.ones((), device=batch_tensors.value_targets.device, dtype=torch.float32)
    return normalized_advantages, advantages_mean, advantages_std, value_loss_scale


def _loss_payload_weighted_sum(
    payloads: list[tuple[PPOLossPayload, float]],
    *,
    value_loss_coef: float,
) -> PPOLossPayload:
    if not payloads:
        raise RuntimeError("cannot aggregate empty PPO loss payloads")

    def weighted_tensor(name: str) -> torch.Tensor:
        total = None
        for payload, weight in payloads:
            weighted_item = getattr(payload, name).detach().float() * float(weight)
            total = weighted_item if total is None else total + weighted_item
        if total is None:
            raise RuntimeError(f"missing PPO loss metric {name}")
        return total

    policy_loss = weighted_tensor("policy_loss")
    value_loss = weighted_tensor("value_loss")
    entropy_loss = weighted_tensor("entropy_loss")
    reference_kl_loss = weighted_tensor("reference_kl_loss")
    return PPOLossPayload(
        loss=policy_loss + float(value_loss_coef) * value_loss + entropy_loss + reference_kl_loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        raw_value_loss=weighted_tensor("raw_value_loss"),
        value_loss_scale=weighted_tensor("value_loss_scale"),
        entropy_loss=entropy_loss,
        reference_kl_loss=reference_kl_loss,
        approx_kl=weighted_tensor("approx_kl"),
        old_policy_exact_kl=weighted_tensor("old_policy_exact_kl"),
        reference_exact_kl=weighted_tensor("reference_exact_kl"),
        clip_fraction=weighted_tensor("clip_fraction"),
        advantages_mean=weighted_tensor("advantages_mean"),
        advantages_std=weighted_tensor("advantages_std"),
    )


def _trajectory_patch_offsets(lengths: list[int]) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for length in lengths:
        if length < 0:
            raise RuntimeError(f"negative trajectory patch count: {length}")
        offsets.append((cursor, cursor + length))
        cursor += length
    return offsets


def _effective_microbatch_size(requested: int, total_items: int) -> int:
    if requested <= 0:
        return max(1, total_items)
    return max(1, int(requested))


def compact_logprob_advantage_diagnostics(
    *,
    old_logprobs: torch.Tensor,
    current_logprobs: torch.Tensor,
    raw_advantages: torch.Tensor,
    normalized_advantages: torch.Tensor,
    patch_rewards: torch.Tensor,
    clip_range: float,
) -> dict:
    old_logprobs_f = old_logprobs.detach().float()
    current_logprobs_f = current_logprobs.detach().float()
    raw_advantages_f = raw_advantages.detach().float()
    normalized_advantages_f = normalized_advantages.detach().float()
    patch_rewards_f = patch_rewards.detach().float()
    if not (
        old_logprobs_f.shape
        == current_logprobs_f.shape
        == raw_advantages_f.shape
        == normalized_advantages_f.shape
        == patch_rewards_f.shape
    ):
        raise RuntimeError(
            "compact PPO diagnostic shape mismatch: "
            f"old={tuple(old_logprobs_f.shape)} current={tuple(current_logprobs_f.shape)} "
            f"raw_adv={tuple(raw_advantages_f.shape)} norm_adv={tuple(normalized_advantages_f.shape)} "
            f"patch_reward={tuple(patch_rewards_f.shape)}"
        )

    log_ratio = current_logprobs_f - old_logprobs_f
    ratio = torch.exp(log_ratio)
    positive_advantage = raw_advantages_f > 0
    negative_advantage = raw_advantages_f < 0
    nonzero_advantage = raw_advantages_f != 0
    sign_aligned = (log_ratio * raw_advantages_f) > 0
    upper_clipped = ratio > (1.0 + float(clip_range))
    lower_clipped = ratio < (1.0 - float(clip_range))
    any_clipped = upper_clipped | lower_clipped
    ppo_active_clipped = (positive_advantage & upper_clipped) | (negative_advantage & lower_clipped)
    return {
        "post_epoch_available": True,
        "patch_count": int(log_ratio.numel()),
        "advantage_summary": advantage_distribution_summary(raw_advantages_f, normalized_advantages_f),
        "approx_kl": float((((old_logprobs_f - current_logprobs_f) ** 2).mean() * 0.5).detach().cpu()),
        "clip_fraction": float(any_clipped.float().mean().detach().cpu()),
        "active_clip_fraction_nonzero_advantage": masked_tensor_mean(
            ppo_active_clipped.float(),
            nonzero_advantage,
        ),
        "log_ratio_mean": float(log_ratio.mean().detach().cpu()),
        "log_ratio_std": float(log_ratio.std(unbiased=False).detach().cpu()),
        "log_ratio_max_abs": float(log_ratio.abs().max().detach().cpu()),
        "log_ratio_mean_positive_advantage": masked_tensor_mean(log_ratio, positive_advantage),
        "log_ratio_mean_negative_advantage": masked_tensor_mean(log_ratio, negative_advantage),
        "positive_advantage_positive_log_ratio_fraction": masked_tensor_mean(
            (log_ratio > 0).float(),
            positive_advantage,
        ),
        "negative_advantage_negative_log_ratio_fraction": masked_tensor_mean(
            (log_ratio < 0).float(),
            negative_advantage,
        ),
        "advantage_log_ratio_correlation": tensor_correlation(raw_advantages_f, log_ratio),
        "normalized_advantage_log_ratio_correlation": tensor_correlation(normalized_advantages_f, log_ratio),
        "patch_reward_log_ratio_correlation": tensor_correlation(patch_rewards_f, log_ratio),
        "sign_alignment_fraction": masked_tensor_mean(sign_aligned.float(), nonzero_advantage),
    }


def run_ppo_replay_epoch_microbatched(
    *,
    policy_model: NotaGenLMHeadModel,
    value_head: PatchValueHead,
    optimizer: torch.optim.Optimizer,
    flat_prompt_ids: list[int],
    rollout_payloads: list[PPORolloutPayload],
    trajectory_lengths: list[int],
    old_logprobs: torch.Tensor,
    old_token_logprobs: torch.Tensor,
    old_token_log_dists: torch.Tensor,
    old_token_counts: torch.Tensor,
    reference_token_log_dists: torch.Tensor | None,
    old_values: torch.Tensor,
    batch_tensors: PPOBatchTensors,
    normalized_advantages: torch.Tensor,
    advantages_mean: torch.Tensor,
    advantages_std: torch.Tensor,
    value_loss_scale: torch.Tensor,
    args,
) -> PPOReplayEpochResult:
    if len(rollout_payloads) != len(trajectory_lengths):
        raise RuntimeError(
            f"PPO microbatch trajectory count mismatch: rollouts={len(rollout_payloads)} "
            f"lengths={len(trajectory_lengths)}"
        )
    total_patches = int(sum(trajectory_lengths))
    if total_patches <= 0:
        raise RuntimeError("PPO microbatch replay needs at least one scored patch")
    for name, tensor in (
        ("old_logprobs", old_logprobs),
        ("old_token_counts", old_token_counts),
        ("old_values", old_values),
        ("advantages", normalized_advantages),
        ("value_targets", batch_tensors.value_targets),
    ):
        if tensor.numel() != total_patches:
            raise RuntimeError(f"PPO microbatch tensor length mismatch for {name}: {tensor.numel()} != {total_patches}")
    offsets = _trajectory_patch_offsets(trajectory_lengths)
    trajectory_token_lengths = [
        int(old_token_counts[start:end].detach().sum().cpu())
        for start, end in offsets
    ]
    total_tokens = int(sum(trajectory_token_lengths))
    if total_tokens <= 0:
        raise RuntimeError("PPO microbatch replay needs at least one generated token")
    if old_token_logprobs.numel() != total_tokens:
        raise RuntimeError(
            f"PPO microbatch tensor length mismatch for old_token_logprobs: {old_token_logprobs.numel()} != {total_tokens}"
        )
    if old_token_log_dists.shape[0] != total_tokens:
        raise RuntimeError(
            f"PPO microbatch tensor length mismatch for old_token_log_dists: {old_token_log_dists.shape[0]} != {total_tokens}"
        )
    if reference_token_log_dists is not None and reference_token_log_dists.shape != old_token_log_dists.shape:
        raise RuntimeError(
            "PPO reference KL tensor shape mismatch: "
            f"reference={tuple(reference_token_log_dists.shape)} old={tuple(old_token_log_dists.shape)}"
        )

    microbatch_size = _effective_microbatch_size(args.ppo_replay_microbatch_size, len(rollout_payloads))
    token_offsets = _trajectory_patch_offsets(trajectory_token_lengths)
    optimizer.zero_grad(set_to_none=True)
    payloads_for_metrics: list[tuple[PPOLossPayload, float]] = []
    new_replays: list[PatchReplayChunk] = []
    microbatch_count = 0

    for trajectory_start in range(0, len(rollout_payloads), microbatch_size):
        trajectory_end = min(len(rollout_payloads), trajectory_start + microbatch_size)
        patch_start = offsets[trajectory_start][0]
        patch_end = offsets[trajectory_end - 1][1]
        token_start = token_offsets[trajectory_start][0]
        token_end = token_offsets[trajectory_end - 1][1]
        expected_patches = patch_end - patch_start
        expected_tokens = token_end - token_start
        trajectory_batch = rollout_payloads[trajectory_start:trajectory_end]
        chunk_replays = batched_trajectory_patch_logprobs_values_by_prompt(
            policy_model,
            value_head,
            trajectory_batch,
            args.precision,
            replay_context_patches=args.replay_context_patches,
            target_chunk_patches=args.score_chunk_patches,
            replay_batch_size=0,
        )
        new_logprobs = torch.cat([replay.logprobs.float() for replay in chunk_replays])
        new_token_logprobs = torch.cat([replay.token_logprobs.float() for replay in chunk_replays])
        new_token_log_dists = torch.cat([replay.token_log_dists.float() for replay in chunk_replays])
        new_token_counts = torch.cat([replay.token_counts.long() for replay in chunk_replays])
        new_values = torch.cat([replay.values.float() for replay in chunk_replays])
        if new_logprobs.numel() != expected_patches:
            raise RuntimeError(
                "PPO replay microbatch patch count mismatch: "
                f"trajectories={trajectory_start}:{trajectory_end} "
                f"new={new_logprobs.numel()} expected={expected_patches}"
            )
        if new_token_logprobs.numel() != expected_tokens:
            raise RuntimeError(
                "PPO replay microbatch token count mismatch: "
                f"trajectories={trajectory_start}:{trajectory_end} "
                f"new={new_token_logprobs.numel()} expected={expected_tokens}"
            )
        if not torch.equal(new_token_counts.detach().cpu(), old_token_counts[patch_start:patch_end].detach().cpu()):
            raise RuntimeError(
                "PPO replay microbatch token-count mismatch for scored patches: "
                f"trajectories={trajectory_start}:{trajectory_end}"
            )

        loss_payload = ppo_clipped_loss(
            new_logprobs=new_token_logprobs,
            old_logprobs=old_token_logprobs[token_start:token_end],
            values=new_values,
            old_values=old_values[patch_start:patch_end],
            advantages=batch_tensors.advantages[patch_start:patch_end],
            value_targets=batch_tensors.value_targets[patch_start:patch_end],
            clip_range=args.ppo_clip_range,
            value_loss_coef=args.value_loss_coef,
            entropy_bonus_coef=args.entropy_bonus_coef,
            normalize_advantage=False,
            normalize_value_loss=args.normalize_value_loss,
            value_loss_eps=args.value_loss_eps,
            value_loss_scale_min=args.value_loss_scale_min,
            normalized_advantages=normalized_advantages[patch_start:patch_end],
            advantages_mean=advantages_mean,
            advantages_std=advantages_std,
            fixed_value_loss_scale=value_loss_scale,
            policy_patch_indices=token_patch_indices_from_counts(new_token_counts),
            value_token_counts=new_token_counts,
            new_log_dists=new_token_log_dists,
            old_log_dists=old_token_log_dists[token_start:token_end],
            reference_log_dists=(
                None if reference_token_log_dists is None else reference_token_log_dists[token_start:token_end]
            ),
            reference_kl_coef=args.reference_kl_coef,
        )
        token_weight = expected_tokens / total_tokens
        payloads_for_metrics.append((loss_payload, token_weight))
        if not args.no_step:
            weighted_loss = (
                loss_payload.policy_loss * token_weight
                + float(args.value_loss_coef) * loss_payload.value_loss * token_weight
                + loss_payload.entropy_loss * token_weight
                + loss_payload.reference_kl_loss * token_weight
            )
            weighted_loss.backward()

        new_replays.extend(
            PatchReplayChunk(
                logprobs=replay.logprobs.detach().float(),
                values=replay.values.detach().float(),
                token_logprobs=replay.token_logprobs.detach().float(),
                token_log_dists=replay.token_log_dists.detach().float(),
                token_counts=replay.token_counts.detach().long(),
            )
            for replay in chunk_replays
        )
        del chunk_replays, new_logprobs, new_token_logprobs, new_token_log_dists, new_token_counts, new_values, loss_payload
        if next(policy_model.parameters()).device.type == "cuda":
            torch.cuda.empty_cache()
        microbatch_count += 1

    if not args.no_step:
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
            [
                param
                for param in list(policy_model.parameters()) + list(value_head.parameters())
                if param.requires_grad
            ],
            args.max_grad_norm,
        )
        optimizer.step()
        grad_norm = float(grad_norm_tensor.detach().cpu() if torch.is_tensor(grad_norm_tensor) else grad_norm_tensor)
    else:
        grad_norm = None

    new_logprobs = torch.cat([replay.logprobs for replay in new_replays])
    new_token_logprobs = torch.cat([replay.token_logprobs for replay in new_replays])
    new_values = torch.cat([replay.values for replay in new_replays])
    return PPOReplayEpochResult(
        loss_payload=_loss_payload_weighted_sum(payloads_for_metrics, value_loss_coef=args.value_loss_coef),
        new_replays=new_replays,
        new_logprobs=new_logprobs,
        new_token_logprobs=new_token_logprobs,
        new_values=new_values,
        grad_norm=grad_norm,
        microbatch_count=microbatch_count,
        microbatch_size=microbatch_size,
    )


def post_step_replay_microbatched(
    *,
    policy_model: NotaGenLMHeadModel,
    value_head: PatchValueHead,
    flat_prompt_ids: list[int],
    rollout_payloads: list[PPORolloutPayload],
    args,
) -> PatchReplayChunk:
    microbatch_size = _effective_microbatch_size(args.ppo_replay_microbatch_size, len(rollout_payloads))
    logprobs: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    token_logprobs: list[torch.Tensor] = []
    token_log_dists: list[torch.Tensor] = []
    token_counts: list[torch.Tensor] = []
    with torch.no_grad():
        for trajectory_start in range(0, len(rollout_payloads), microbatch_size):
            trajectory_end = min(len(rollout_payloads), trajectory_start + microbatch_size)
            trajectory_batch = rollout_payloads[trajectory_start:trajectory_end]
            replay_batch = batched_trajectory_patch_logprobs_values_by_prompt(
                policy_model,
                value_head,
                trajectory_batch,
                args.precision,
                replay_context_patches=args.replay_context_patches,
                target_chunk_patches=args.score_chunk_patches,
                replay_batch_size=0,
            )
            for replay in replay_batch:
                logprobs.append(replay.logprobs.detach().float())
                values.append(replay.values.detach().float())
                token_logprobs.append(replay.token_logprobs.detach().float())
                token_log_dists.append(replay.token_log_dists.detach().float())
                token_counts.append(replay.token_counts.detach().long())
            if next(policy_model.parameters()).device.type == "cuda":
                torch.cuda.empty_cache()
    device = next(policy_model.parameters()).device
    vocab_size = policy_model.char_level_decoder.base.transformer.wte.weight.shape[0]
    if not logprobs:
        return PatchReplayChunk(
            logprobs=torch.empty(0, device=device),
            values=torch.empty(0, device=device),
            token_logprobs=torch.empty(0, device=device),
            token_log_dists=torch.empty((0, vocab_size), device=device),
            token_counts=torch.empty(0, device=device, dtype=torch.long),
        )
    return PatchReplayChunk(
        logprobs=torch.cat(logprobs),
        values=torch.cat(values),
        token_logprobs=torch.cat(token_logprobs),
        token_log_dists=torch.cat(token_log_dists),
        token_counts=torch.cat(token_counts),
    )


def post_step_replay_logprobs_microbatched(
    *,
    policy_model: NotaGenLMHeadModel,
    value_head: PatchValueHead,
    flat_prompt_ids: list[int],
    rollout_payloads: list[PPORolloutPayload],
    args,
) -> torch.Tensor:
    return post_step_replay_microbatched(
        policy_model=policy_model,
        value_head=value_head,
        flat_prompt_ids=flat_prompt_ids,
        rollout_payloads=rollout_payloads,
        args=args,
    ).logprobs


def sample_ppo_rollouts(
    *,
    policy_model: NotaGenLMHeadModel,
    policy_shape: ModelShape,
    step_idx: int,
    args,
    prompt: str | None = None,
    target_stream_lines: int | None = None,
    prompt_batch: list[PromptBatchItem] | None = None,
) -> list[PPORolloutPayload]:
    if args.trajectories_per_step <= 0:
        raise ValueError(f"trajectories_per_step must be positive, got {args.trajectories_per_step}")
    if args.rollout_batch_size <= 0:
        raise ValueError(f"rollout_batch_size must be positive, got {args.rollout_batch_size}")
    if prompt_batch is None:
        if prompt is None or target_stream_lines is None:
            raise ValueError("sample_ppo_rollouts requires either prompt_batch or prompt plus target_stream_lines")
        prompt_batch = [
            PromptBatchItem(
                trajectory_index=trajectory_idx,
                prompt_idx=0,
                prompt_name="prompt_0",
                prompt=prompt,
                prompt_target=None,
                target=None,
                target_stream_lines=int(target_stream_lines),
                schedule=PromptScheduleSelection(
                    prompt_idx=0,
                    selection="ordered",
                    slot_index=trajectory_idx,
                    cycle=0,
                    cycle_position=0,
                    cycle_length=1,
                    cycle_order=[0],
                ),
            )
            for trajectory_idx in range(args.trajectories_per_step)
        ]
    if len(prompt_batch) != args.trajectories_per_step:
        raise ValueError(
            f"prompt_batch length must match trajectories_per_step: "
            f"{len(prompt_batch)} != {args.trajectories_per_step}"
        )

    failure_policy = getattr(args, "rollout_failure_policy", "error")
    if failure_policy not in {"error", "zero", "spares"}:
        raise ValueError(f"unknown rollout_failure_policy: {failure_policy}")
    spares_percent = float(getattr(args, "rollout_spares_percent", 10.0))
    if spares_percent < 0:
        raise ValueError(f"rollout_spares_percent must be non-negative, got {spares_percent}")
    max_attempts = 1 if failure_policy in {"zero", "spares"} else args.rollout_retries
    rollout_seed_scope = str(getattr(args, "rollout_seed_scope", "step"))
    if rollout_seed_scope == "step":
        rollout_seed_step_idx = step_idx
    elif rollout_seed_scope == "run":
        rollout_seed_step_idx = 1
    else:
        raise ValueError(f"unsupported rollout_seed_scope: {rollout_seed_scope!r}")

    def payload_prompt_meta(spec: PromptBatchItem) -> dict:
        return {
            "prompt_index": int(spec.prompt_idx),
            "prompt_name": spec.prompt_name,
            "rollout_target_stream_lines": int(spec.target_stream_lines),
            **prompt_schedule_metadata(spec.schedule),
        }

    def build_payload(
        spec: PromptBatchItem,
        rollout_seed: int,
        full_text: str,
        generated_patches: list[list[int]],
        *,
        batched_rollout: bool,
        extra_meta: dict | None = None,
    ) -> PPORolloutPayload:
        return PPORolloutPayload(
            trajectory_index=spec.trajectory_index,
            rollout_seed=rollout_seed,
            full_text=full_text,
            generated_patches=generated_patches,
            meta={
                "cached_rollout": bool(args.cached_rollout),
                "batched_rollout": bool(batched_rollout),
                "rollout_batch_size": args.rollout_batch_size if batched_rollout else 1,
                "rollout_failure_policy": failure_policy,
                "rollout_seed_scope": rollout_seed_scope,
                "rollout_seed_step_idx": int(rollout_seed_step_idx),
                **payload_prompt_meta(spec),
                **(extra_meta or {}),
            },
            prompt_idx=spec.prompt_idx,
            prompt_name=spec.prompt_name,
            prompt=spec.prompt,
            prompt_target=spec.prompt_target,
            target=spec.target,
            target_stream_lines=spec.target_stream_lines,
            prompt_schedule=spec.schedule,
        )

    def failed_payload(
        spec: PromptBatchItem,
        rollout_seed: int,
        error: str,
        *,
        batched_rollout: bool,
        full_text: str | None = None,
        generated_patches: list[list[int]] | None = None,
        result_meta: dict | None = None,
    ) -> PPORolloutPayload:
        failed_patches = generated_patches or []
        failure_meta = dict(result_meta or {})
        failure_meta.update(
            {
                "rollout_failed": True,
                "zero_contribution_rollout": len(failed_patches) == 0,
                "stop_reason": failure_meta.get("stop_reason", "rollout_failed"),
                "error": error,
                "failed_generated_patches": len(failed_patches),
            }
        )
        return build_payload(
            spec,
            rollout_seed=rollout_seed,
            full_text=full_text if full_text is not None else spec.prompt,
            generated_patches=failed_patches,
            batched_rollout=batched_rollout,
            extra_meta=failure_meta,
        )

    if failure_policy == "spares":
        if args.rollout_batch_size <= 1 or not args.cached_rollout:
            raise RuntimeError("--rollout-failure-policy spares requires cached batched rollout")
        requested_successes = len(prompt_batch)
        extra_candidates = int(math.ceil(requested_successes * spares_percent / 100.0))
        candidate_specs: list[tuple[PromptBatchItem, int]] = [(spec, 0) for spec in prompt_batch]
        candidate_specs.extend(
            (prompt_batch[extra_idx % requested_successes], 1 + extra_idx // requested_successes)
            for extra_idx in range(extra_candidates)
        )
        candidate_count = len(candidate_specs)
        effective_batch_size = args.rollout_batch_size
        if args.rollout_batch_size == requested_successes:
            effective_batch_size = candidate_count

        successes_by_trajectory: dict[int, PPORolloutPayload] = {}
        success_candidate_count = 0
        candidate_errors: dict[int, str] = {}
        for batch_start in range(0, candidate_count, effective_batch_size):
            batch_indices = list(range(batch_start, min(candidate_count, batch_start + effective_batch_size)))
            batch_specs = [candidate_specs[candidate_idx] for candidate_idx in batch_indices]
            seeds = [
                _rollout_seed(args.seed, rollout_seed_step_idx, spec.trajectory_index, attempt_idx)
                for spec, attempt_idx in batch_specs
            ]
            try:
                batch_results = sample_completions_cached_batch(
                    model=policy_model,
                    model_shape=policy_shape,
                    prompts=[spec.prompt for spec, _attempt_idx in batch_specs],
                    seeds=seeds,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    target_stream_lines=[spec.target_stream_lines for spec, _attempt_idx in batch_specs],
                    target_new_stream_lines=False,
                    max_chars=args.max_chars,
                    max_generated_patches=args.max_generated_patches,
                    timeout_s=args.timeout_s,
                    precision=args.precision,
                )
            except RuntimeError as exc:
                for candidate_idx in batch_indices:
                    candidate_errors[candidate_idx] = str(exc)
                continue
            for candidate_idx, (spec, attempt_idx), rollout_seed, result in zip(
                batch_indices,
                batch_specs,
                seeds,
                batch_results,
                strict=True,
            ):
                if result.ok and result.full_text is not None and result.generated_patches is not None:
                    success_candidate_count += 1
                    successes_by_trajectory.setdefault(
                        spec.trajectory_index,
                        build_payload(
                            spec,
                            rollout_seed=rollout_seed,
                            full_text=result.full_text,
                            generated_patches=result.generated_patches,
                            batched_rollout=True,
                            extra_meta={
                                "rollout_batch_size": effective_batch_size,
                                "rollout_requested_batch_size": args.rollout_batch_size,
                                "rollout_candidate_index": candidate_idx,
                                "rollout_spare_attempt": attempt_idx,
                                "rollout_sampled_candidates": candidate_count,
                                "rollout_spares_percent": spares_percent,
                                **(result.meta or {}),
                            },
                        ),
                    )
                else:
                    candidate_errors[candidate_idx] = result.error or "unknown batch rollout error"

        if len(successes_by_trajectory) < requested_successes:
            raise RuntimeError(
                "failed to fill PPO rollout batch with spares: "
                f"requested_successes={requested_successes} successes={len(successes_by_trajectory)} "
                f"sampled_candidates={candidate_count} failures={len(candidate_errors)} "
                f"errors={candidate_errors}"
            )

        kept_payloads = [successes_by_trajectory[spec.trajectory_index] for spec in prompt_batch]
        rollout_meta = {
            "rollout_failure_policy": "spares",
            "rollout_sampled_candidates": candidate_count,
            "rollout_success_candidates": success_candidate_count,
            "rollout_failed_candidates": candidate_count - success_candidate_count,
            "rollout_dropped_success_candidates": success_candidate_count - requested_successes,
            "rollout_dropped_candidates": candidate_count - requested_successes,
            "rollout_spares_percent": spares_percent,
            "rollout_effective_batch_size": effective_batch_size,
            "rollout_requested_batch_size": args.rollout_batch_size,
        }
        for payload in kept_payloads:
            payload.meta.update(rollout_meta)
        return kept_payloads

    rollout_payloads: list[PPORolloutPayload] = []
    if args.rollout_batch_size > 1:
        if not args.cached_rollout:
            raise RuntimeError("--rollout-batch-size > 1 requires --cached-rollout")

        pending = list(prompt_batch)
        last_errors: dict[int, str] = {}
        for retry_idx in range(max_attempts):
            next_pending: list[PromptBatchItem] = []
            for batch_start in range(0, len(pending), args.rollout_batch_size):
                batch_specs = pending[batch_start : batch_start + args.rollout_batch_size]
                seeds = [
                    _rollout_seed(args.seed, rollout_seed_step_idx, spec.trajectory_index, retry_idx)
                    for spec in batch_specs
                ]
                try:
                    batch_results = sample_completions_cached_batch(
                        model=policy_model,
                        model_shape=policy_shape,
                        prompts=[spec.prompt for spec in batch_specs],
                        seeds=seeds,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        target_stream_lines=[spec.target_stream_lines for spec in batch_specs],
                        target_new_stream_lines=False,
                        max_chars=args.max_chars,
                        max_generated_patches=args.max_generated_patches,
                        timeout_s=args.timeout_s,
                        precision=args.precision,
                )
                except RuntimeError as exc:
                    if failure_policy != "zero":
                        raise
                    for spec, rollout_seed in zip(batch_specs, seeds, strict=True):
                        rollout_payloads.append(
                            failed_payload(
                                spec,
                                rollout_seed,
                                str(exc),
                                batched_rollout=True,
                            )
                        )
                    continue
                for spec, rollout_seed, result in zip(batch_specs, seeds, batch_results, strict=True):
                    if result.ok and result.full_text is not None and result.generated_patches is not None:
                        rollout_payloads.append(
                            build_payload(
                                spec,
                                rollout_seed=rollout_seed,
                                full_text=result.full_text,
                                generated_patches=result.generated_patches,
                                batched_rollout=True,
                                extra_meta={
                                    **(result.meta or {}),
                                },
                            )
                        )
                    else:
                        last_errors[spec.trajectory_index] = result.error or "unknown batch rollout error"
                        if failure_policy == "zero":
                            rollout_payloads.append(
                                failed_payload(
                                    spec,
                                    rollout_seed,
                                    last_errors[spec.trajectory_index],
                                    batched_rollout=True,
                                    full_text=result.full_text,
                                    generated_patches=result.generated_patches,
                                    result_meta=result.meta,
                                )
                            )
                        else:
                            next_pending.append(spec)
            if not next_pending:
                pending = []
                break
            pending = next_pending
        if pending:
            raise RuntimeError(f"failed to sample PPO rollouts after retries: {last_errors}")
    else:
        for spec in prompt_batch:
            sample_built = False
            last_error: Exception | None = None
            last_rollout_seed = _rollout_seed(args.seed, rollout_seed_step_idx, spec.trajectory_index, 0)
            for retry_idx in range(max_attempts):
                rollout_seed = _rollout_seed(args.seed, rollout_seed_step_idx, spec.trajectory_index, retry_idx)
                last_rollout_seed = rollout_seed
                set_seed(rollout_seed)
                try:
                    full_text, generated_patches = sample_completion(
                        model=policy_model,
                        model_shape=policy_shape,
                        prompt=spec.prompt,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        target_stream_lines=spec.target_stream_lines,
                        max_chars=args.max_chars,
                        max_generated_patches=args.max_generated_patches,
                        timeout_s=args.timeout_s,
                        precision=args.precision,
                        cached_rollout=args.cached_rollout,
                    )
                    rollout_payloads.append(
                        build_payload(
                            spec,
                            rollout_seed=rollout_seed,
                            full_text=full_text,
                            generated_patches=generated_patches,
                            batched_rollout=False,
                        )
                    )
                    sample_built = True
                    break
                except RuntimeError as exc:
                    last_error = exc
                    continue
            if not sample_built:
                if failure_policy == "zero":
                    rollout_payloads.append(
                        failed_payload(
                            spec,
                            last_rollout_seed,
                            str(last_error) if last_error is not None else "unknown rollout error",
                            batched_rollout=False,
                        )
                    )
                    continue
                raise RuntimeError(f"failed to sample PPO rollout {spec.trajectory_index} after retries: {last_error}")

    rollout_payloads.sort(key=lambda item: item.trajectory_index)
    if len(rollout_payloads) != args.trajectories_per_step:
        raise RuntimeError(
            f"PPO rollout count mismatch: expected {args.trajectories_per_step}, got {len(rollout_payloads)}"
        )
    return rollout_payloads


def _reward_summary_from_logs(trajectory_logs: list[dict]) -> dict:
    if not trajectory_logs:
        return {
            "reward_mean": None,
            "reward_std": None,
            "reward_min": None,
            "reward_max": None,
            "reward_sum": 0.0,
            "sample_rewards": [],
        }
    sample_rewards = np.array([float(log["reward"]) for log in trajectory_logs], dtype=np.float32)
    return {
        "reward_mean": float(sample_rewards.mean()),
        "reward_std": float(sample_rewards.std()),
        "reward_min": float(sample_rewards.min()),
        "reward_max": float(sample_rewards.max()),
        "reward_sum": float(sample_rewards.sum()),
        "sample_rewards": sample_rewards.astype(float).tolist(),
    }


_PPO_REWARD_WORKER_CONTEXT: dict | None = None


def _init_ppo_reward_worker(context: dict) -> None:
    global _PPO_REWARD_WORKER_CONTEXT
    _PPO_REWARD_WORKER_CONTEXT = context


def _score_ppo_rollout_payload_worker(payload: PPORolloutPayload) -> tuple[PatchRewardTrace, dict]:
    if _PPO_REWARD_WORKER_CONTEXT is None:
        raise RuntimeError("PPO reward worker context was not initialized")
    return _score_ppo_rollout_payload_from_context(payload, _PPO_REWARD_WORKER_CONTEXT)


def _ppo_reward_scoring_options_from_args(args) -> PPORewardScoringOptions:
    return PPORewardScoringOptions(
        similarity_chroma_bins=int(args.similarity_chroma_bins),
        similarity_band_ratio=float(args.similarity_band_ratio),
        similarity_timeout_s=float(args.similarity_timeout_s),
        max_similarity_reward=float(args.max_similarity_reward),
        patch_reward_attribution=str(getattr(args, "patch_reward_attribution", "single_pass")),
        reward_mode=str(getattr(args, "reward_mode", "goldberg")),
        simple_reward_note=str(getattr(args, "simple_reward_note", "G")),
        simple_reward_max_count=float(getattr(args, "simple_reward_max_count", 64.0)),
        simple_reward_length_unit=str(getattr(args, "simple_reward_length_unit", "patches")),
        simple_reward_length_target=float(getattr(args, "simple_reward_length_target", 160.0)),
        simple_reward_scale=float(getattr(args, "simple_reward_scale", 1.0)),
        rollout_failure_terminal_reward=float(getattr(args, "rollout_failure_terminal_reward", -1.0)),
    )


def _score_ppo_rollout_payload_from_context(
    payload: PPORolloutPayload,
    context: dict,
) -> tuple[PatchRewardTrace, dict]:
    if context.get("use_payload_prompt_context"):
        prompt_context = prompt_context_from_payload(payload)
        prompt_stream_lines = count_stream_lines(
            build_rollout_prefix(prompt_context.prompt, prompt_context.target_stream_lines)
        )
        return _score_ppo_rollout_payload(
            prompt=prompt_context.prompt,
            prompt_idx=prompt_context.prompt_idx,
            prompt_name=prompt_context.prompt_name,
            prompt_target=prompt_context.prompt_target,
            target=prompt_context.target,
            target_stream_lines=prompt_context.target_stream_lines,
            payload=payload,
            prompt_stream_lines=prompt_stream_lines,
            reward_config=context["reward_config"],
            similarity_weights=context["similarity_weights"],
            aria_similarity_ref=context["aria_similarity_ref"],
            scoring_options=context["scoring_options"],
            candidate_name_prefix=context["candidate_name_prefix"],
        )

    return _score_ppo_rollout_payload(payload=payload, **context)


def _score_ppo_rollout_payload(
    *,
    prompt: str,
    prompt_idx: int,
    prompt_name: str,
    prompt_target: PromptStructuralTarget,
    target,
    target_stream_lines: int,
    payload: PPORolloutPayload,
    prompt_stream_lines: int,
    reward_config: GoldbergRewardConfig,
    similarity_weights: SimilarityRewardWeights,
    aria_similarity_ref: SimilarityReference | None,
    scoring_options: PPORewardScoringOptions,
    candidate_name_prefix: str,
) -> tuple[PatchRewardTrace, dict]:
    patchilizer = Patchilizer(stream=PATCH_STREAM)
    if (payload.meta or {}).get("rollout_failed"):
        completion_text = "".join(patchilizer.decode(payload.generated_patches))
        generated_patch_count = len(payload.generated_patches)
        generated_tokens = generated_token_slots(payload.generated_patches)
        stop_reason = (payload.meta or {}).get("stop_reason", "rollout_failed")
        length_diagnostics = _rollout_length_diagnostics(
            full_text=payload.full_text,
            completion_text=completion_text,
            generated_patch_count=generated_patch_count,
            prompt_stream_lines=prompt_stream_lines,
            target_stream_lines=target_stream_lines,
            stop_reason=stop_reason,
        )
        failure_reward = float(scoring_options.rollout_failure_terminal_reward)
        component_rewards = (
            {"rollout_failure_terminal_reward": _terminal_patch_rewards(generated_patch_count, failure_reward)}
            if generated_patch_count > 0
            else {}
        )
        patch_rewards = (
            component_rewards["rollout_failure_terminal_reward"][:]
            if generated_patch_count > 0
            else []
        )
        patch_reward_component_sums = component_reward_sums(component_rewards)
        patch_reward_groups = component_group_rewards(component_rewards, len(patch_rewards))
        zero_contribution_rollout = generated_patch_count == 0
        reward_breakdown = {
            "reward": failure_reward,
            "total_reward": failure_reward,
            "parse_valid": False,
            "parse_reward": 0.0,
            "structural_total_reward": failure_reward,
            "raw_similarity_reward": 0.0,
            "clipped_similarity_reward": 0.0,
            "similarity_validity_gate": 0.0,
            "effective_similarity_reward": 0.0,
            "rollout_failure_terminal_reward": failure_reward,
            "rollout_failed": True,
            "zero_contribution_rollout": zero_contribution_rollout,
            "generated_patches": generated_patch_count,
            "generated_token_slots": generated_tokens,
            "prompt_index": prompt_idx,
            "prompt_name": prompt_name,
            "target_structure_path": prompt_target.structure_path,
            "target_structure_source_key": prompt_target.source_key,
            "target_expected_reward_bars": int(target.expected_reward_bars),
            "target_stream_lines": target_stream_lines,
            "trajectory_index": payload.trajectory_index,
            "rollout_seed": payload.rollout_seed,
            "rollout_prefix_stream_lines": prompt_stream_lines,
            "patch_reward_mode": "terminal_failed_rollout_reward",
            "patch_reward_count": len(patch_rewards),
            "patch_reward_sum": float(sum(patch_rewards)),
            "patch_reward_component_sums": patch_reward_component_sums,
            "patch_reward_group_sums": component_group_sums(patch_reward_component_sums),
        }
        reward_breakdown.update(payload.meta)
        reward_breakdown["zero_contribution_rollout"] = zero_contribution_rollout
        reward_breakdown.update(length_diagnostics)
        reward_trace = PatchRewardTrace(
            rewards=patch_rewards,
            prefix_totals=prefix_totals(patch_rewards),
            final_score=RewardScore(total=failure_reward, breakdown=reward_breakdown),
            component_rewards=component_rewards,
            component_prefix_totals=component_prefix_totals(component_rewards),
        )
        trajectory_log = {
            "trajectory_index": payload.trajectory_index,
            "rollout_seed": payload.rollout_seed,
            "reward": failure_reward,
            "full_text": payload.full_text,
            "completion_text": completion_text,
            "generated_patches": payload.generated_patches,
            "generated_patch_count": generated_patch_count,
            "generated_token_slots": generated_tokens,
            "rollout_length_diagnostics": length_diagnostics,
            "patch_reward_mean": float(np.mean(patch_rewards)) if patch_rewards else 0.0,
            "patch_reward_std": float(np.std(patch_rewards)) if patch_rewards else 0.0,
            "patch_rewards": patch_rewards,
            "patch_reward_prefix_totals": reward_trace.prefix_totals,
            "patch_reward_components": component_rewards,
            "patch_reward_component_prefix_totals": reward_trace.component_prefix_totals,
            "patch_reward_component_sums": patch_reward_component_sums,
            "patch_reward_groups": patch_reward_groups,
            "patch_reward_group_prefix_totals": component_prefix_totals(patch_reward_groups),
            "patch_reward_group_sums": component_group_sums(patch_reward_component_sums),
            "reward_breakdown": reward_breakdown,
        }
        return reward_trace, trajectory_log

    if scoring_options.reward_mode != "goldberg":
        reward_trace = patch_rewards_simple_test(
            generated_patches=payload.generated_patches,
            scoring_options=scoring_options,
        )
        patch_reward_mode = f"simple_{scoring_options.reward_mode}_{scoring_options.patch_reward_attribution}"
    elif scoring_options.patch_reward_attribution == "single_pass":
        reward_trace = patch_rewards_from_prefix_deltas(
            prompt_text=prompt,
            generated_patches=payload.generated_patches,
            target=target,
            reward_config=reward_config,
            candidate_name=f"{candidate_name_prefix}_sample{payload.trajectory_index}",
            similarity_weights=similarity_weights,
            aria_similarity_ref=aria_similarity_ref,
            similarity_chroma_bins=scoring_options.similarity_chroma_bins,
            similarity_band_ratio=scoring_options.similarity_band_ratio,
            similarity_timeout_s=scoring_options.similarity_timeout_s,
            max_similarity_reward=scoring_options.max_similarity_reward,
        )
        patch_reward_mode = "single_pass_events_plus_terminal_residual"
    elif scoring_options.patch_reward_attribution == "terminal":
        reward_trace = patch_rewards_terminal(
            prompt_text=prompt,
            generated_patches=payload.generated_patches,
            target=target,
            reward_config=reward_config,
            candidate_name=f"{candidate_name_prefix}_sample{payload.trajectory_index}",
            similarity_weights=similarity_weights,
            aria_similarity_ref=aria_similarity_ref,
            similarity_chroma_bins=scoring_options.similarity_chroma_bins,
            similarity_band_ratio=scoring_options.similarity_band_ratio,
            similarity_timeout_s=scoring_options.similarity_timeout_s,
            max_similarity_reward=scoring_options.max_similarity_reward,
        )
        patch_reward_mode = "terminal_total_reward"
    else:
        raise RuntimeError(f"unsupported patch reward attribution: {scoring_options.patch_reward_attribution!r}")
    total_reward = reward_trace.final_score.total
    reward_breakdown = reward_trace.final_score.breakdown
    completion_text = "".join(patchilizer.decode(payload.generated_patches))
    stop_reason = (payload.meta or {}).get("stop_reason")
    length_diagnostics = _rollout_length_diagnostics(
        full_text=payload.full_text,
        completion_text=completion_text,
        generated_patch_count=len(payload.generated_patches),
        prompt_stream_lines=prompt_stream_lines,
        target_stream_lines=target_stream_lines,
        stop_reason=stop_reason,
    )
    reward_breakdown["generated_patches"] = len(payload.generated_patches)
    reward_breakdown["generated_token_slots"] = generated_token_slots(payload.generated_patches)
    reward_breakdown["prompt_index"] = prompt_idx
    reward_breakdown["prompt_name"] = prompt_name
    reward_breakdown["target_structure_path"] = prompt_target.structure_path
    reward_breakdown["target_structure_source_key"] = prompt_target.source_key
    reward_breakdown["target_expected_reward_bars"] = int(target.expected_reward_bars)
    reward_breakdown["target_stream_lines"] = target_stream_lines
    reward_breakdown["trajectory_index"] = payload.trajectory_index
    reward_breakdown["rollout_seed"] = payload.rollout_seed
    reward_breakdown["rollout_prefix_stream_lines"] = prompt_stream_lines
    reward_breakdown.update(payload.meta)
    reward_breakdown.update(length_diagnostics)
    reward_breakdown["patch_reward_mode"] = patch_reward_mode
    reward_breakdown["patch_reward_count"] = len(reward_trace.rewards)
    reward_breakdown["patch_reward_sum"] = float(sum(reward_trace.rewards))
    patch_reward_component_sums = component_reward_sums(reward_trace.component_rewards)
    patch_reward_groups = component_group_rewards(reward_trace.component_rewards, len(reward_trace.rewards))
    reward_breakdown["patch_reward_component_sums"] = patch_reward_component_sums
    reward_breakdown["patch_reward_group_sums"] = component_group_sums(patch_reward_component_sums)
    trajectory_log = {
        "trajectory_index": payload.trajectory_index,
        "rollout_seed": payload.rollout_seed,
        "reward": total_reward,
        "full_text": payload.full_text,
        "completion_text": completion_text,
        "generated_patches": payload.generated_patches,
        "generated_patch_count": len(payload.generated_patches),
        "generated_token_slots": generated_token_slots(payload.generated_patches),
        "rollout_length_diagnostics": length_diagnostics,
        "patch_reward_mean": float(np.mean(reward_trace.rewards)) if reward_trace.rewards else 0.0,
        "patch_reward_std": float(np.std(reward_trace.rewards)) if reward_trace.rewards else 0.0,
        "patch_rewards": reward_trace.rewards,
        "patch_reward_prefix_totals": reward_trace.prefix_totals,
        "patch_reward_components": reward_trace.component_rewards,
        "patch_reward_component_prefix_totals": reward_trace.component_prefix_totals,
        "patch_reward_component_sums": patch_reward_component_sums,
        "patch_reward_groups": patch_reward_groups,
        "patch_reward_group_prefix_totals": component_prefix_totals(patch_reward_groups),
        "patch_reward_group_sums": component_group_sums(patch_reward_component_sums),
        "reward_breakdown": reward_breakdown,
    }
    return reward_trace, trajectory_log


def _default_reward_worker_start_method() -> str:
    available = mp.get_all_start_methods()
    if "forkserver" in available:
        return "forkserver"
    if "spawn" in available:
        return "spawn"
    return available[0]


def _score_ppo_rollouts_parallel(
    *,
    rollout_payloads: list[PPORolloutPayload],
    max_workers: int,
    start_method: str,
    context: dict,
) -> list[tuple[PatchRewardTrace, dict]]:
    if start_method not in mp.get_all_start_methods():
        raise RuntimeError(f"unsupported reward worker start method: {start_method}")
    worker_count = min(max_workers, len(rollout_payloads))
    mp_context = mp.get_context(start_method)
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=mp_context,
        initializer=_init_ppo_reward_worker,
        initargs=(context,),
    ) as executor:
        return list(executor.map(_score_ppo_rollout_payload_worker, rollout_payloads, chunksize=1))


def score_ppo_rollout_payloads(
    *,
    prompt: str,
    prompt_idx: int,
    prompt_name: str,
    prompt_target: PromptStructuralTarget,
    target,
    target_stream_lines: int,
    rollout_payloads: list[PPORolloutPayload],
    reward_config: GoldbergRewardConfig,
    similarity_weights: SimilarityRewardWeights,
    aria_similarity_ref: SimilarityReference | None,
    args,
    step_idx: int,
    candidate_name_prefix: str,
) -> ScoredRolloutBatch:
    prompt_stream_lines = count_stream_lines(build_rollout_prefix(prompt, target_stream_lines))
    scoring_options = _ppo_reward_scoring_options_from_args(args)
    context = {
        "prompt": prompt,
        "prompt_idx": prompt_idx,
        "prompt_name": prompt_name,
        "prompt_target": prompt_target,
        "target": target,
        "target_stream_lines": target_stream_lines,
        "prompt_stream_lines": prompt_stream_lines,
        "reward_config": reward_config,
        "similarity_weights": similarity_weights,
        "aria_similarity_ref": aria_similarity_ref,
        "scoring_options": scoring_options,
        "candidate_name_prefix": candidate_name_prefix,
    }

    reward_workers = int(getattr(args, "reward_workers", 0) or 0)
    if reward_workers > 1 and len(rollout_payloads) > 1:
        start_method = getattr(args, "reward_worker_start_method", None) or _default_reward_worker_start_method()
        scored_items = _score_ppo_rollouts_parallel(
            rollout_payloads=rollout_payloads,
            max_workers=reward_workers,
            start_method=start_method,
            context=context,
        )
    else:
        scored_items = [
            _score_ppo_rollout_payload(payload=payload, **context)
            for payload in rollout_payloads
        ]

    reward_traces = [reward_trace for reward_trace, _trajectory_log in scored_items]
    trajectory_logs = [trajectory_log for _reward_trace, trajectory_log in scored_items]
    return ScoredRolloutBatch(
        trajectory_logs=trajectory_logs,
        reward_traces=reward_traces,
        reward_summary=_reward_summary_from_logs(trajectory_logs),
    )


def score_ppo_rollout_payloads_from_payload_context(
    *,
    rollout_payloads: list[PPORolloutPayload],
    reward_config: GoldbergRewardConfig,
    similarity_weights: SimilarityRewardWeights,
    aria_similarity_ref: SimilarityReference | None,
    args,
    step_idx: int,
    candidate_name_prefix: str,
) -> ScoredRolloutBatch:
    scoring_options = _ppo_reward_scoring_options_from_args(args)
    context = {
        "use_payload_prompt_context": True,
        "reward_config": reward_config,
        "similarity_weights": similarity_weights,
        "aria_similarity_ref": aria_similarity_ref,
        "scoring_options": scoring_options,
        "candidate_name_prefix": candidate_name_prefix,
    }

    reward_workers = int(getattr(args, "reward_workers", 0) or 0)
    if reward_workers > 1 and len(rollout_payloads) > 1:
        start_method = getattr(args, "reward_worker_start_method", None) or _default_reward_worker_start_method()
        scored_items = _score_ppo_rollouts_parallel(
            rollout_payloads=rollout_payloads,
            max_workers=reward_workers,
            start_method=start_method,
            context=context,
        )
    else:
        scored_items = [
            _score_ppo_rollout_payload_from_context(payload, context)
            for payload in rollout_payloads
        ]

    reward_traces = [reward_trace for reward_trace, _trajectory_log in scored_items]
    trajectory_logs = [trajectory_log for _reward_trace, trajectory_log in scored_items]
    return ScoredRolloutBatch(
        trajectory_logs=trajectory_logs,
        reward_traces=reward_traces,
        reward_summary=_reward_summary_from_logs(trajectory_logs),
    )


def fixed_eval_output_path(args) -> Path:
    if args.fixed_eval_output_jsonl:
        return Path(args.fixed_eval_output_jsonl)
    return Path(args.output_json).with_name("fixed_eval.jsonl")


def compact_eval_trajectory_log(trajectory_log: dict, *, include_trajectories: bool) -> dict:
    record = {
        "trajectory_index": trajectory_log["trajectory_index"],
        "rollout_seed": trajectory_log["rollout_seed"],
        "reward": trajectory_log["reward"],
        "generated_patch_count": trajectory_log["generated_patch_count"],
        "generated_token_slots": trajectory_log["generated_token_slots"],
        "patch_reward_mean": trajectory_log["patch_reward_mean"],
        "patch_reward_std": trajectory_log["patch_reward_std"],
        "reward_breakdown": trajectory_log["reward_breakdown"],
    }
    if include_trajectories:
        record.update(
            {
                "full_text": trajectory_log["full_text"],
                "completion_text": trajectory_log["completion_text"],
                "generated_patches": trajectory_log["generated_patches"],
            }
        )
    return record


def fixed_eval_reference_kl_diagnostics(
    *,
    policy_model: NotaGenLMHeadModel,
    reference_policy_model: NotaGenLMHeadModel,
    rollout_payloads: list[PPORolloutPayload],
    args,
) -> dict:
    eval_payloads = [
        payload
        for payload in rollout_payloads
        if not (payload.meta or {}).get("rollout_failed") and payload.generated_patches
    ]
    if not eval_payloads:
        return {
            "ok": False,
            "reason": "no_generated_tokens",
            "trajectory_count": 0,
            "patch_count": 0,
            "token_count": 0,
            "reference_exact_kl": None,
        }

    microbatch_size = _effective_microbatch_size(
        int(getattr(args, "fixed_eval_kl_replay_microbatch_size", 0)),
        len(eval_payloads),
    )
    policy_replays: list[TokenDistributionReplay] = []
    reference_replays: list[TokenDistributionReplay] = []
    with torch.no_grad():
        for trajectory_start in range(0, len(eval_payloads), microbatch_size):
            trajectory_end = min(len(eval_payloads), trajectory_start + microbatch_size)
            trajectory_batch = eval_payloads[trajectory_start:trajectory_end]
            policy_replays.extend(
                batched_trajectory_token_log_dists_by_prompt(
                    policy_model,
                    trajectory_batch,
                    args.precision,
                    replay_context_patches=args.replay_context_patches,
                    target_chunk_patches=args.score_chunk_patches,
                    replay_batch_size=0,
                )
            )
            reference_replays.extend(
                batched_trajectory_token_log_dists_by_prompt(
                    reference_policy_model,
                    trajectory_batch,
                    args.precision,
                    replay_context_patches=args.replay_context_patches,
                    target_chunk_patches=args.score_chunk_patches,
                    replay_batch_size=0,
                )
            )
            if next(policy_model.parameters()).device.type == "cuda":
                torch.cuda.empty_cache()

    per_trajectory: list[dict] = []
    for payload, policy_replay, reference_replay in zip(
        eval_payloads,
        policy_replays,
        reference_replays,
        strict=True,
    ):
        if policy_replay.token_log_dists.shape != reference_replay.token_log_dists.shape:
            raise RuntimeError(
                "fixed-eval reference KL replay shape mismatch: "
                f"trajectory={payload.trajectory_index} "
                f"policy={tuple(policy_replay.token_log_dists.shape)} "
                f"reference={tuple(reference_replay.token_log_dists.shape)}"
            )
        if not torch.equal(
            policy_replay.token_counts.detach().cpu(),
            reference_replay.token_counts.detach().cpu(),
        ):
            raise RuntimeError(f"fixed-eval reference KL token-count mismatch: trajectory={payload.trajectory_index}")
        per_trajectory.append(
            {
                "trajectory_index": int(payload.trajectory_index),
                "prompt_index": int(payload.prompt_idx),
                "prompt_name": payload.prompt_name,
                "patch_count": int(policy_replay.token_counts.numel()),
                "token_count": int(policy_replay.token_log_dists.shape[0]),
                "reference_exact_kl": float(
                    exact_categorical_kl(
                        policy_replay.token_log_dists,
                        reference_replay.token_log_dists,
                    )
                    .detach()
                    .cpu()
                ),
            }
        )

    policy_log_dists = torch.cat([replay.token_log_dists.detach().float() for replay in policy_replays])
    reference_log_dists = torch.cat([replay.token_log_dists.detach().float() for replay in reference_replays])
    token_count = int(policy_log_dists.shape[0])
    patch_count = int(sum(replay.token_counts.numel() for replay in policy_replays))
    return {
        "ok": True,
        "trajectory_count": len(eval_payloads),
        "patch_count": patch_count,
        "token_count": token_count,
        "reference_exact_kl": float(exact_categorical_kl(policy_log_dists, reference_log_dists).detach().cpu()),
        "per_trajectory": per_trajectory,
        "microbatch_size": microbatch_size,
    }


def run_fixed_eval_batch(
    *,
    policy_model: NotaGenLMHeadModel,
    policy_shape: ModelShape,
    prompt_batch: list[PromptBatchItem],
    reward_config: GoldbergRewardConfig,
    similarity_weights: SimilarityRewardWeights,
    aria_similarity_ref: SimilarityReference | None,
    args,
    step_idx: int,
    label: str,
    event_index: int,
    reference_policy_model: NotaGenLMHeadModel | None = None,
) -> dict | None:
    if args.fixed_eval_trajectories <= 0:
        return None
    if not prompt_batch:
        raise ValueError("fixed eval prompt batch is empty")
    prompt_batch_log = prompt_batch_metadata(prompt_batch)
    first_prompt = prompt_batch[0]
    eval_args = argparse.Namespace(**vars(args))
    eval_args.trajectories_per_step = args.fixed_eval_trajectories
    eval_args.rollout_batch_size = (
        args.fixed_eval_rollout_batch_size
        if args.fixed_eval_rollout_batch_size > 0
        else min(args.rollout_batch_size, args.fixed_eval_trajectories)
    )
    eval_args.rollout_retries = args.fixed_eval_rollout_retries
    eval_args.seed = args.seed + args.fixed_eval_seed_offset

    eval_start = time.perf_counter()
    rollout_start = time.perf_counter()
    try:
        with torch.no_grad():
            rollout_payloads = sample_ppo_rollouts(
                policy_model=policy_model,
                policy_shape=policy_shape,
                step_idx=args.fixed_eval_seed_step,
                args=eval_args,
                prompt_batch=prompt_batch,
            )
    except RuntimeError as exc:
        summary = {
            "event": "ppo_fixed_eval_complete",
            "label": label,
            "step": step_idx,
            "fixed_eval_event_index": event_index,
            "prompt_index": first_prompt.prompt_idx,
            "prompt_name": first_prompt.prompt_name,
            **prompt_batch_log,
            "ok": False,
            "error": str(exc),
            "trajectory_count": args.fixed_eval_trajectories,
            "rollout_batch_size": eval_args.rollout_batch_size,
            "fixed_eval_seed_offset": args.fixed_eval_seed_offset,
            "fixed_eval_seed_step": args.fixed_eval_seed_step,
            "timings": {
                "fixed_eval_total_s": time.perf_counter() - eval_start,
                "fixed_eval_rollout_s": time.perf_counter() - rollout_start,
                "fixed_eval_reward_s": 0.0,
                "fixed_eval_exact_kl_s": 0.0,
            },
        }
        output_path = fixed_eval_output_path(args)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")
        print(json.dumps(summary), flush=True)
        return summary
    rollout_s = time.perf_counter() - rollout_start
    rollout_sampling = _rollout_sampling_summary(rollout_payloads)

    reward_start = time.perf_counter()
    scored = score_ppo_rollout_payloads_from_payload_context(
        rollout_payloads=rollout_payloads,
        reward_config=reward_config,
        similarity_weights=similarity_weights,
        aria_similarity_ref=aria_similarity_ref,
        args=args,
        step_idx=step_idx,
        candidate_name_prefix=f"fixed_eval_{label}_step{step_idx}",
    )
    reward_s = time.perf_counter() - reward_start
    exact_kl_s = 0.0
    exact_kl_summary = None
    if bool(getattr(args, "fixed_eval_reference_kl_check", False)):
        if reference_policy_model is None:
            raise RuntimeError("--fixed-eval-reference-kl-check requires a loaded reference policy model")
        exact_kl_start = time.perf_counter()
        exact_kl_summary = fixed_eval_reference_kl_diagnostics(
            policy_model=policy_model,
            reference_policy_model=reference_policy_model,
            rollout_payloads=rollout_payloads,
            args=args,
        )
        exact_kl_s = time.perf_counter() - exact_kl_start
    patch_reward_component_sums = aggregate_component_sums(scored.reward_traces)
    rollout_length = _rollout_length_summary(scored.trajectory_logs)
    summary = {
        "event": "ppo_fixed_eval_complete",
        "label": label,
        "step": step_idx,
        "fixed_eval_event_index": event_index,
        "ok": True,
        "prompt_index": first_prompt.prompt_idx,
        "prompt_name": first_prompt.prompt_name,
        **prompt_batch_log,
        "target_structure_path": first_prompt.prompt_target.structure_path,
        "target_structure_source_key": first_prompt.prompt_target.source_key,
        "target_expected_reward_bars": int(first_prompt.target.expected_reward_bars),
        "target_stream_lines": first_prompt.target_stream_lines,
        "trajectory_count": len(rollout_payloads),
        "rollout_batch_size": eval_args.rollout_batch_size,
        "rollout_sampling": rollout_sampling,
        "fixed_eval_seed_offset": args.fixed_eval_seed_offset,
        "fixed_eval_seed_step": args.fixed_eval_seed_step,
        "patch_reward_attribution": args.patch_reward_attribution,
        "reward_mean": scored.reward_summary["reward_mean"],
        "reward_std": scored.reward_summary["reward_std"],
        "reward_min": scored.reward_summary["reward_min"],
        "reward_max": scored.reward_summary["reward_max"],
        "reward_sum": scored.reward_summary["reward_sum"],
        "sample_rewards": scored.reward_summary["sample_rewards"],
        "fixed_eval_reference_kl_check": bool(getattr(args, "fixed_eval_reference_kl_check", False)),
        "fixed_eval_exact_kl": exact_kl_summary,
        "reference_exact_kl": (
            None
            if exact_kl_summary is None or not exact_kl_summary.get("ok")
            else exact_kl_summary.get("reference_exact_kl")
        ),
        "patch_reward_component_sums": patch_reward_component_sums,
        "patch_reward_group_sums": component_group_sums(patch_reward_component_sums),
        "rollout_length": rollout_length,
        "generated_patch_count_mean": float(
            np.mean([log["generated_patch_count"] for log in scored.trajectory_logs])
        ),
        "generated_patch_count_min": int(min(log["generated_patch_count"] for log in scored.trajectory_logs)),
        "generated_patch_count_max": int(max(log["generated_patch_count"] for log in scored.trajectory_logs)),
        "generated_token_slots_mean": float(
            np.mean([log["generated_token_slots"] for log in scored.trajectory_logs])
        ),
        "timings": {
            "fixed_eval_total_s": time.perf_counter() - eval_start,
            "fixed_eval_rollout_s": rollout_s,
            "fixed_eval_reward_s": reward_s,
            "fixed_eval_exact_kl_s": exact_kl_s,
            "fixed_eval_rollout_per_trajectory_s": rollout_s / max(1, len(rollout_payloads)),
            "fixed_eval_reward_per_trajectory_s": reward_s / max(1, len(rollout_payloads)),
        },
    }
    record = dict(summary)
    if args.fixed_eval_save_trajectories:
        record["trajectories"] = [
            compact_eval_trajectory_log(
                trajectory_log,
                include_trajectories=True,
            )
            for trajectory_log in scored.trajectory_logs
        ]
    output_path = fixed_eval_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps(summary), flush=True)
    return summary


def train_value_head_on_returns(
    *,
    policy_model: NotaGenLMHeadModel,
    value_head: PatchValueHead,
    value_optimizer: torch.optim.Optimizer,
    rollout_payloads: list[PPORolloutPayload],
    return_tensors: list[torch.Tensor],
    args,
) -> dict:
    if args.value_warmup_epochs <= 0:
        return {"epochs": 0, "epoch_logs": []}
    if len(rollout_payloads) != len(return_tensors):
        raise RuntimeError(
            f"value warmup tensor count mismatch: rollouts={len(rollout_payloads)} returns={len(return_tensors)}"
        )

    logs: list[dict] = []
    start = time.perf_counter()
    targets = torch.cat([item.detach().float() for item in return_tensors])

    def collect_values() -> list[torch.Tensor]:
        values_by_trajectory: list[torch.Tensor] = []
        microbatch_size = _effective_microbatch_size(args.ppo_replay_microbatch_size, len(rollout_payloads))
        for trajectory_start in range(0, len(rollout_payloads), microbatch_size):
            trajectory_end = min(len(rollout_payloads), trajectory_start + microbatch_size)
            trajectory_batch = rollout_payloads[trajectory_start:trajectory_end]
            return_batch = return_tensors[trajectory_start:trajectory_end]
            value_batch = batched_trajectory_patch_values_by_prompt(
                policy_model,
                value_head,
                trajectory_batch,
                args.precision,
                replay_context_patches=args.replay_context_patches,
                target_chunk_patches=args.score_chunk_patches,
                replay_batch_size=0,
                detach_policy=True,
            )
            for payload, returns, values in zip(trajectory_batch, return_batch, value_batch, strict=True):
                if values.shape != returns.shape:
                    raise RuntimeError(
                        "value warmup shape mismatch: "
                        f"trajectory={payload.trajectory_index} values={tuple(values.shape)} "
                        f"returns={tuple(returns.shape)}"
                    )
                values_by_trajectory.append(values)
        return values_by_trajectory

    for epoch_idx in range(1, args.value_warmup_epochs + 1):
        epoch_start = time.perf_counter()
        value_optimizer.zero_grad(set_to_none=True)
        values_by_trajectory = collect_values()
        values = torch.cat(values_by_trajectory)
        before_metrics = value_prediction_metrics(values, targets)
        loss, raw_loss, scale = value_mse_loss(
            values,
            targets,
            normalize_value_loss=args.normalize_value_loss,
            eps=args.value_loss_eps,
            scale_min=args.value_loss_scale_min,
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [param for param in value_head.parameters() if param.requires_grad],
            args.max_grad_norm,
        )
        value_optimizer.step()
        with torch.no_grad():
            after_values = torch.cat(collect_values())
        logs.append(
            {
                "epoch": epoch_idx,
                "loss": float(loss.detach().cpu()),
                "raw_value_loss": float(raw_loss.detach().cpu()),
                "value_loss_scale": float(scale.detach().cpu()),
                "value_mean": float(values.detach().mean().cpu()),
                "value_std": float(values.detach().std(unbiased=False).cpu()),
                "grad_norm": float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm),
                "before_metrics": before_metrics,
                "after_metrics": value_prediction_metrics(after_values, targets),
                "duration_s": time.perf_counter() - epoch_start,
            }
        )

    return {
        "epochs": args.value_warmup_epochs,
        "duration_s": time.perf_counter() - start,
        "epoch_logs": logs,
    }


def run_ppo_smoke(
    policy_model: NotaGenLMHeadModel,
    policy_shape: ModelShape,
    value_head: PatchValueHead,
    prompts: list[dict],
    prompt_targets: list[PromptStructuralTarget],
    reward_config: GoldbergRewardConfig,
    args,
    behavior_policy_model: NotaGenLMHeadModel | None = None,
    reference_policy_model: NotaGenLMHeadModel | None = None,
) -> dict:
    patchilizer = Patchilizer(stream=PATCH_STREAM)
    device = next(policy_model.parameters()).device
    rollout_model = behavior_policy_model or policy_model
    old_logprob_model = behavior_policy_model or policy_model
    training_reference_kl_enabled = bool(
        reference_policy_model is not None
        and (args.reference_kl_coef != 0.0 or args.reference_kl_check)
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": [param for param in policy_model.parameters() if param.requires_grad], "lr": args.learning_rate},
            {"params": value_head.parameters(), "lr": args.value_learning_rate},
        ]
    )
    value_optimizer = torch.optim.AdamW(value_head.parameters(), lr=args.value_learning_rate)
    policy_model.eval()
    if behavior_policy_model is not None:
        behavior_policy_model.eval()
        for param in behavior_policy_model.parameters():
            param.requires_grad_(False)
    if reference_policy_model is not None:
        reference_policy_model.eval()
        for param in reference_policy_model.parameters():
            param.requires_grad_(False)
    value_head.train()
    dropout_modules_disabled = disable_dropout_modules(policy_model)
    behavior_dropout_modules_disabled = (
        disable_dropout_modules(behavior_policy_model) if behavior_policy_model is not None else []
    )
    reference_dropout_modules_disabled = (
        disable_dropout_modules(reference_policy_model) if reference_policy_model is not None else []
    )

    if args.reward_mode == "goldberg":
        similarity_weights = SimilarityRewardWeights(
            aria_chroma=args.aria_chroma_reward_weight,
            aria_harmony=args.aria_harmony_reward_weight,
        )
    else:
        similarity_weights = SimilarityRewardWeights()
    aria_similarity_ref: SimilarityReference | None = None
    if similarity_weights.enabled:
        aria_similarity_ref = load_similarity_reference(
            args.aria_reference_abc,
            load_chroma=similarity_weights.aria_chroma != 0.0,
            load_harmony=similarity_weights.aria_harmony != 0.0,
            bins=args.similarity_chroma_bins,
        )
    if not prompts:
        raise ValueError("no prompt rows loaded")
    if len(prompt_targets) != len(prompts):
        raise ValueError(f"prompt target count mismatch: prompts={len(prompts)} targets={len(prompt_targets)}")
    if args.prompt_selection not in {"ordered", "random"}:
        raise ValueError(f"unsupported prompt_selection: {args.prompt_selection!r}")
    if args.prompt_batch_mode not in {"trajectory", "step"}:
        raise ValueError(f"unsupported prompt_batch_mode: {args.prompt_batch_mode!r}")
    if args.rollout_seed_scope not in {"step", "run"}:
        raise ValueError(f"unsupported rollout_seed_scope: {args.rollout_seed_scope!r}")
    if not 0.0 <= args.gae_lambda <= 1.0:
        raise ValueError(f"gae_lambda must be in [0, 1], got {args.gae_lambda}")
    if args.patch_reward_attribution not in {"single_pass", "terminal"}:
        raise ValueError(
            "patch_reward_attribution must be one of {'single_pass', 'terminal'}, "
            f"got {args.patch_reward_attribution!r}"
        )
    if args.reward_mode not in {"goldberg", "note_count", "note_fraction", "length"}:
        raise ValueError(f"unsupported reward_mode: {args.reward_mode!r}")
    if args.reward_mode in {"note_count", "note_fraction"}:
        note = str(args.simple_reward_note).strip()
        if len(note) != 1 or note.upper() not in {"A", "B", "C", "D", "E", "F", "G"}:
            raise ValueError(f"simple_reward_note must be one ABC pitch letter A-G, got {args.simple_reward_note!r}")
    if args.reward_mode == "note_count":
        if args.simple_reward_max_count <= 0:
            raise ValueError(f"simple_reward_max_count must be positive, got {args.simple_reward_max_count}")
    if args.reward_mode == "note_fraction" and args.patch_reward_attribution != "terminal":
        raise ValueError("reward_mode note_fraction requires --patch-reward-attribution terminal")
    if args.reward_mode == "length" and args.simple_reward_length_target <= 0:
        raise ValueError(f"simple_reward_length_target must be positive, got {args.simple_reward_length_target}")
    if args.simple_reward_scale < 0:
        raise ValueError(f"simple_reward_scale must be non-negative, got {args.simple_reward_scale}")
    if args.rollout_retries <= 0:
        raise ValueError(f"rollout_retries must be positive, got {args.rollout_retries}")
    if args.rollout_spares_percent < 0:
        raise ValueError(f"rollout_spares_percent must be non-negative, got {args.rollout_spares_percent}")
    if args.ppo_epochs <= 0:
        raise ValueError(f"ppo_epochs must be positive, got {args.ppo_epochs}")
    if args.reference_kl_coef != 0.0 and reference_policy_model is None:
        raise ValueError("--reference-kl-coef requires a loaded reference policy model")
    if args.value_warmup_epochs < 0:
        raise ValueError(f"value_warmup_epochs must be non-negative, got {args.value_warmup_epochs}")
    if args.value_loss_eps <= 0:
        raise ValueError(f"value_loss_eps must be positive, got {args.value_loss_eps}")
    if args.value_loss_scale_min <= 0:
        raise ValueError(f"value_loss_scale_min must be positive, got {args.value_loss_scale_min}")
    if args.fixed_eval_trajectories < 0:
        raise ValueError(f"fixed_eval_trajectories must be non-negative, got {args.fixed_eval_trajectories}")
    if args.fixed_eval_every_steps < 0:
        raise ValueError(f"fixed_eval_every_steps must be non-negative, got {args.fixed_eval_every_steps}")
    if args.fixed_eval_prompt_selection not in {"same", "ordered", "random"}:
        raise ValueError(f"unsupported fixed_eval_prompt_selection: {args.fixed_eval_prompt_selection!r}")
    if args.fixed_eval_prompt_batch_mode not in {"same", "trajectory", "event"}:
        raise ValueError(
            f"unsupported fixed_eval_prompt_batch_mode: {args.fixed_eval_prompt_batch_mode!r}"
        )
    if args.fixed_eval_rollout_batch_size < 0:
        raise ValueError(
            f"fixed_eval_rollout_batch_size must be non-negative, got {args.fixed_eval_rollout_batch_size}"
        )
    if args.fixed_eval_rollout_retries <= 0:
        raise ValueError(f"fixed_eval_rollout_retries must be positive, got {args.fixed_eval_rollout_retries}")
    if args.fixed_eval_kl_replay_microbatch_size < 0:
        raise ValueError(
            "fixed_eval_kl_replay_microbatch_size must be non-negative, "
            f"got {args.fixed_eval_kl_replay_microbatch_size}"
        )
    if args.fixed_eval_reference_kl_check and reference_policy_model is None:
        raise ValueError("--fixed-eval-reference-kl-check requires a loaded reference policy model")

    logs: list[dict] = []
    fixed_eval_logs: list[dict] = []
    fixed_eval_event_cursor = fixed_eval_event_index_before_training(args)
    if args.fixed_eval_trajectories > 0 and args.fixed_eval_before_training:
        fixed_eval_event_index = fixed_eval_event_cursor
        fixed_eval_event_cursor += 1
        fixed_eval_prompt_batch = build_fixed_eval_prompt_batch(
            prompts=prompts,
            prompt_targets=prompt_targets,
            args=args,
            event_index=fixed_eval_event_index,
        )
        fixed_eval_log = run_fixed_eval_batch(
            policy_model=policy_model,
            policy_shape=policy_shape,
            prompt_batch=fixed_eval_prompt_batch,
            reward_config=reward_config,
            similarity_weights=similarity_weights,
            aria_similarity_ref=aria_similarity_ref,
            args=args,
            step_idx=args.step_offset,
            label="before_training",
            event_index=fixed_eval_event_index,
            reference_policy_model=reference_policy_model,
        )
        if fixed_eval_log is not None:
            fixed_eval_logs.append(fixed_eval_log)
    for local_step_idx in range(1, args.max_steps + 1):
        step_start = time.perf_counter()
        timings: dict[str, float] = {}
        step_idx = args.step_offset + local_step_idx
        prompt_batch = build_prompt_batch_for_step(
            prompts=prompts,
            prompt_targets=prompt_targets,
            args=args,
            step_idx=step_idx,
        )
        prompt_batch_log = prompt_batch_metadata(prompt_batch)
        first_prompt = prompt_batch[0]
        prompt_idx = first_prompt.prompt_idx
        prompt_name = first_prompt.prompt_name
        prompt_target = first_prompt.prompt_target
        target = first_prompt.target
        target_stream_lines = first_prompt.target_stream_lines
        prompt = first_prompt.prompt

        rollout_start = time.perf_counter()
        rollout_payloads = sample_ppo_rollouts(
            policy_model=rollout_model,
            policy_shape=policy_shape,
            step_idx=step_idx,
            args=args,
            prompt_batch=prompt_batch,
        )
        timings["rollout_s"] = time.perf_counter() - rollout_start
        timings["rollout_per_trajectory_s"] = timings["rollout_s"] / max(1, len(rollout_payloads))
        rollout_sampling = _rollout_sampling_summary(rollout_payloads)

        reward_start = time.perf_counter()
        scored_rollouts = score_ppo_rollout_payloads_from_payload_context(
            rollout_payloads=rollout_payloads,
            reward_config=reward_config,
            similarity_weights=similarity_weights,
            aria_similarity_ref=aria_similarity_ref,
            args=args,
            step_idx=step_idx,
            candidate_name_prefix=f"step{step_idx}",
        )
        trajectory_logs = scored_rollouts.trajectory_logs
        reward_traces = scored_rollouts.reward_traces
        timings["reward_s"] = time.perf_counter() - reward_start
        timings["reward_per_trajectory_s"] = timings["reward_s"] / max(1, len(rollout_payloads))
        update_items = [
            (payload, reward_trace, trajectory_log)
            for payload, reward_trace, trajectory_log in zip(
                rollout_payloads,
                reward_traces,
                trajectory_logs,
                strict=True,
            )
            if len(reward_trace.rewards) > 0
        ]
        skipped_update_logs = [
            trajectory_log
            for payload, reward_trace, trajectory_log in zip(
                rollout_payloads,
                reward_traces,
                trajectory_logs,
                strict=True,
            )
            if len(reward_trace.rewards) == 0
        ]
        failed_rollout_count = int(
            sum(1 for payload in rollout_payloads if (payload.meta or {}).get("rollout_failed"))
        )
        zero_contribution_count = len(skipped_update_logs)

        if args.rollout_only:
            timings["total_step_s"] = time.perf_counter() - step_start
            sample_rewards = [float(log["reward"]) for log in trajectory_logs]
            sample_rewards_array = np.array(sample_rewards, dtype=np.float32)
            flattened_patch_rewards = [
                reward
                for trajectory_log in trajectory_logs
                for reward in trajectory_log["patch_rewards"]
            ]
            patch_reward_component_sums = aggregate_component_sums(reward_traces)
            step_log = {
                "step": step_idx,
                "prompt_index": prompt_idx,
                "prompt_name": prompt_name,
                **prompt_batch_log,
                "target_structure_path": prompt_target.structure_path,
                "target_structure_source_key": prompt_target.source_key,
                "target_expected_reward_bars": int(target.expected_reward_bars),
                "target_stream_lines": target_stream_lines,
                "trajectories_per_step": len(rollout_payloads),
                "ppo_update_trajectories": len(update_items),
                "zero_contribution_trajectories": zero_contribution_count,
                "failed_rollout_count": failed_rollout_count,
                "rollout_batch_size": args.rollout_batch_size,
                "rollout_sampling": rollout_sampling,
                "rollout_failure_policy": args.rollout_failure_policy,
                "rollout_only": True,
                "patch_reward_mean": float(np.mean(flattened_patch_rewards)) if flattened_patch_rewards else 0.0,
                "patch_reward_std": float(np.std(flattened_patch_rewards)) if flattened_patch_rewards else 0.0,
                "patch_reward_component_sums": patch_reward_component_sums,
                "patch_reward_group_sums": component_group_sums(patch_reward_component_sums),
                "scored_patches": int(sum(len(log["patch_rewards"]) for log in trajectory_logs)),
                "reward": float(sample_rewards_array.mean()),
                "reward_mean": float(sample_rewards_array.mean()),
                "reward_std": float(sample_rewards_array.std()),
                "reward_min": float(sample_rewards_array.min()),
                "reward_max": float(sample_rewards_array.max()),
                "reward_sum": float(sample_rewards_array.sum()),
                "sample_rewards": sample_rewards,
                "reward_breakdown": trajectory_logs[0]["reward_breakdown"] if len(trajectory_logs) == 1 else None,
                "trajectories": trajectory_logs,
                "timings": timings,
            }
            print(json.dumps({"event": "ppo_rollout_only_step_complete", **step_log}), flush=True)
            logs.append(step_log)
            continue

        if not update_items:
            raise RuntimeError(
                "PPO step has no successful scorable rollouts; "
                f"failed_rollouts={failed_rollout_count} zero_contribution={zero_contribution_count}"
            )
        update_rollout_payloads = [item[0] for item in update_items]
        update_reward_traces = [item[1] for item in update_items]
        update_trajectory_logs = [item[2] for item in update_items]

        replay_start = time.perf_counter()
        old_replay_start = time.perf_counter()
        old_replays: list[PatchReplayChunk] = []
        reference_replays: list[TokenDistributionReplay] = []
        reward_tensors: list[torch.Tensor] = []
        old_replay_only_s = 0.0
        reference_replay_s = 0.0
        with torch.no_grad():
            microbatch_size = _effective_microbatch_size(args.ppo_replay_microbatch_size, len(update_rollout_payloads))
            for trajectory_start in range(0, len(update_rollout_payloads), microbatch_size):
                trajectory_end = min(len(update_rollout_payloads), trajectory_start + microbatch_size)
                trajectory_batch = update_rollout_payloads[trajectory_start:trajectory_end]
                reward_trace_batch = update_reward_traces[trajectory_start:trajectory_end]
                old_replay_batch_start = time.perf_counter()
                old_replay_batch = batched_trajectory_patch_logprobs_values_by_prompt(
                    old_logprob_model,
                    value_head,
                    trajectory_batch,
                    args.precision,
                    replay_context_patches=args.replay_context_patches,
                    target_chunk_patches=args.score_chunk_patches,
                    replay_batch_size=0,
                )
                old_replay_only_s += time.perf_counter() - old_replay_batch_start
                reference_replay_batch = None
                if training_reference_kl_enabled:
                    reference_replay_batch_start = time.perf_counter()
                    reference_replay_batch = batched_trajectory_token_log_dists_by_prompt(
                        reference_policy_model,
                        trajectory_batch,
                        args.precision,
                        replay_context_patches=args.replay_context_patches,
                        target_chunk_patches=args.score_chunk_patches,
                        replay_batch_size=0,
                    )
                    reference_replay_s += time.perf_counter() - reference_replay_batch_start
                for payload, reward_trace, old_replay in zip(
                    trajectory_batch,
                    reward_trace_batch,
                    old_replay_batch,
                    strict=True,
                ):
                    if old_replay.logprobs.numel() == 0:
                        raise RuntimeError(f"PPO rollout {payload.trajectory_index} produced no scorable patches")
                    if len(reward_trace.rewards) != old_replay.logprobs.numel():
                        raise RuntimeError(
                            "PPO patch reward/logprob count mismatch: "
                            f"trajectory={payload.trajectory_index} rewards={len(reward_trace.rewards)} "
                            f"logprobs={old_replay.logprobs.numel()}"
                        )
                    if old_replay.token_counts.numel() != old_replay.logprobs.numel():
                        raise RuntimeError(
                            "PPO replay token-count/logprob count mismatch: "
                            f"trajectory={payload.trajectory_index} token_counts={old_replay.token_counts.numel()} "
                            f"logprobs={old_replay.logprobs.numel()}"
                        )
                    old_replays.append(old_replay)
                    reward_tensors.append(torch.tensor(reward_trace.rewards, device=device, dtype=torch.float32))
                if reference_replay_batch is not None:
                    for payload, old_replay, reference_replay in zip(
                        trajectory_batch,
                        old_replay_batch,
                        reference_replay_batch,
                        strict=True,
                    ):
                        if reference_replay.token_log_dists.shape != old_replay.token_log_dists.shape:
                            raise RuntimeError(
                                "PPO reference replay token distribution shape mismatch: "
                                f"trajectory={payload.trajectory_index} "
                                f"reference={tuple(reference_replay.token_log_dists.shape)} "
                                f"old={tuple(old_replay.token_log_dists.shape)}"
                            )
                        if not torch.equal(
                            reference_replay.token_counts.detach().cpu(),
                            old_replay.token_counts.detach().cpu(),
                        ):
                            raise RuntimeError(
                                "PPO reference replay token-count mismatch: "
                                f"trajectory={payload.trajectory_index}"
                            )
                        reference_replays.append(reference_replay)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        timings["old_replay_s"] = old_replay_only_s
        timings["reference_replay_s"] = reference_replay_s
        timings["old_reference_replay_total_s"] = time.perf_counter() - old_replay_start
        old_logprobs = torch.cat([replay.logprobs.detach().float() for replay in old_replays])
        old_token_logprobs = torch.cat([replay.token_logprobs.detach().float() for replay in old_replays])
        old_token_log_dists = torch.cat([replay.token_log_dists.detach().float() for replay in old_replays])
        old_token_counts = torch.cat([replay.token_counts.detach().long() for replay in old_replays])
        reference_token_log_dists = (
            torch.cat([replay.token_log_dists.detach().float() for replay in reference_replays])
            if reference_replays
            else None
        )
        initial_old_value_tensors = [replay.values.detach().float() for replay in old_replays]
        trajectory_lengths = [int(replay.logprobs.numel()) for replay in old_replays]
        trajectory_token_lengths = [int(replay.token_logprobs.numel()) for replay in old_replays]
        return_tensors = [discounted_returns(rewards, args.gamma).detach() for rewards in reward_tensors]
        returns_for_metrics = torch.cat([item.detach().float() for item in return_tensors])
        initial_value_return_metrics = value_prediction_metrics(
            torch.cat(initial_old_value_tensors),
            returns_for_metrics,
        )

        value_warmup_start = time.perf_counter()
        value_warmup_log = train_value_head_on_returns(
            policy_model=policy_model,
            value_head=value_head,
            value_optimizer=value_optimizer,
            rollout_payloads=update_rollout_payloads,
            return_tensors=return_tensors,
            args=args,
        )
        timings["value_warmup_s"] = time.perf_counter() - value_warmup_start
        if args.value_warmup_epochs > 0:
            old_value_refresh_start = time.perf_counter()
            with torch.no_grad():
                old_value_tensors = []
                microbatch_size = _effective_microbatch_size(
                    args.ppo_replay_microbatch_size,
                    len(update_rollout_payloads),
                )
                for trajectory_start in range(0, len(update_rollout_payloads), microbatch_size):
                    trajectory_end = min(len(update_rollout_payloads), trajectory_start + microbatch_size)
                    trajectory_batch = update_rollout_payloads[trajectory_start:trajectory_end]
                    old_value_tensors.extend(
                        batched_trajectory_patch_values_by_prompt(
                            policy_model,
                            value_head,
                            trajectory_batch,
                            args.precision,
                            replay_context_patches=args.replay_context_patches,
                            target_chunk_patches=args.score_chunk_patches,
                            replay_batch_size=0,
                            detach_policy=True,
                        )
                    )
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                old_value_tensors = [values.detach().float() for values in old_value_tensors]
            timings["old_value_refresh_s"] = time.perf_counter() - old_value_refresh_start
        else:
            old_value_tensors = initial_old_value_tensors

        batch_tensors = batch_trajectory_returns_advantages(
            reward_tensors=reward_tensors,
            value_tensors=old_value_tensors,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )
        old_values = torch.cat([values.detach().float() for values in old_value_tensors])
        post_warmup_value_return_metrics = value_prediction_metrics(old_values, batch_tensors.returns)
        post_warmup_value_target_metrics = value_prediction_metrics(old_values, batch_tensors.value_targets)

        new_replay_start = time.perf_counter()
        ppo_epoch_logs: list[dict] = []
        new_logprobs = torch.empty(0, device=device)
        new_token_logprobs = torch.empty(0, device=device)
        new_values = torch.empty(0, device=device)
        loss_payload: PPOLossPayload | None = None
        new_replays: list[PatchReplayChunk] = []
        normalized_advantages, advantages_mean, advantages_std, value_loss_scale = _ppo_loss_constants(
            batch_tensors,
            args,
            token_counts=old_token_counts,
        )
        advantage_summary = advantage_distribution_summary(
            batch_tensors.advantages,
            normalized_advantages,
            trajectory_lengths=trajectory_lengths,
        )
        for ppo_epoch_idx in range(1, args.ppo_epochs + 1):
            ppo_epoch_start = time.perf_counter()
            epoch_result = run_ppo_replay_epoch_microbatched(
                policy_model=policy_model,
                value_head=value_head,
                optimizer=optimizer,
                flat_prompt_ids=[],
                rollout_payloads=update_rollout_payloads,
                trajectory_lengths=trajectory_lengths,
                old_logprobs=old_logprobs,
                old_token_logprobs=old_token_logprobs,
                old_token_log_dists=old_token_log_dists,
                old_token_counts=old_token_counts,
                reference_token_log_dists=reference_token_log_dists,
                old_values=old_values,
                batch_tensors=batch_tensors,
                normalized_advantages=normalized_advantages,
                advantages_mean=advantages_mean,
                advantages_std=advantages_std,
                value_loss_scale=value_loss_scale,
                args=args,
            )
            loss_payload = epoch_result.loss_payload
            new_replays = epoch_result.new_replays
            new_logprobs = epoch_result.new_logprobs
            new_token_logprobs = epoch_result.new_token_logprobs
            new_values = epoch_result.new_values
            value_return_metrics = value_prediction_metrics(new_values, batch_tensors.returns)
            value_target_metrics = value_prediction_metrics(new_values, batch_tensors.value_targets)
            post_epoch_logprob_advantage_diag = None
            if not args.no_step and args.post_epoch_kl_check:
                post_epoch_start = time.perf_counter()
                post_epoch_replay = post_step_replay_microbatched(
                    policy_model=policy_model,
                    value_head=value_head,
                    flat_prompt_ids=[],
                    rollout_payloads=update_rollout_payloads,
                    args=args,
                )
                post_epoch_logprob_advantage_diag = compact_logprob_advantage_diagnostics(
                    old_logprobs=old_logprobs,
                    current_logprobs=post_epoch_replay.logprobs,
                    raw_advantages=batch_tensors.advantages,
                    normalized_advantages=normalized_advantages,
                    patch_rewards=batch_tensors.patch_rewards,
                    clip_range=args.ppo_clip_range,
                )
                post_epoch_token_log_ratio = post_epoch_replay.token_logprobs - old_token_logprobs
                post_epoch_token_ratio = torch.exp(post_epoch_token_log_ratio)
                post_epoch_logprob_advantage_diag["token_approx_kl"] = float(
                    (((old_token_logprobs - post_epoch_replay.token_logprobs) ** 2).mean() * 0.5).detach().cpu()
                )
                post_epoch_logprob_advantage_diag["old_policy_exact_kl"] = float(
                    exact_categorical_kl(post_epoch_replay.token_log_dists, old_token_log_dists).detach().cpu()
                )
                post_epoch_logprob_advantage_diag["reference_exact_kl"] = (
                    None
                    if reference_token_log_dists is None
                    else float(
                        exact_categorical_kl(
                            post_epoch_replay.token_log_dists,
                            reference_token_log_dists,
                        )
                        .detach()
                        .cpu()
                    )
                )
                post_epoch_logprob_advantage_diag["token_clip_fraction"] = float(
                    ((post_epoch_token_ratio - 1.0).abs() > args.ppo_clip_range).float().mean().detach().cpu()
                )
                post_epoch_logprob_advantage_diag["token_log_ratio_mean"] = float(
                    post_epoch_token_log_ratio.mean().detach().cpu()
                )
                post_epoch_logprob_advantage_diag["duration_s"] = time.perf_counter() - post_epoch_start
            ppo_epoch_logs.append(
                {
                    "epoch": ppo_epoch_idx,
                    "loss": float(loss_payload.loss.detach().cpu()),
                    "policy_loss": float(loss_payload.policy_loss.detach().cpu()),
                    "value_loss": float(loss_payload.value_loss.detach().cpu()),
                    "raw_value_loss": float(loss_payload.raw_value_loss.detach().cpu()),
                    "value_loss_scale": float(loss_payload.value_loss_scale.detach().cpu()),
                    "entropy_loss": float(loss_payload.entropy_loss.detach().cpu()),
                    "reference_kl_loss": float(loss_payload.reference_kl_loss.detach().cpu()),
                    "approx_kl": float(loss_payload.approx_kl.detach().cpu()),
                    "old_policy_exact_kl": float(loss_payload.old_policy_exact_kl.detach().cpu()),
                    "reference_exact_kl": float(loss_payload.reference_exact_kl.detach().cpu()),
                    "reference_kl_coef": float(args.reference_kl_coef),
                    "clip_fraction": float(loss_payload.clip_fraction.detach().cpu()),
                    "policy_granularity": "token",
                    "policy_reduction": "token_mean",
                    "advantage_granularity": "patch_repeated_per_token",
                    "advantage_normalization_granularity": (
                        "none" if args.no_advantage_normalization else "token"
                    ),
                    "value_function_granularity": "patch",
                    "value_target_granularity": "patch",
                    "value_loss_granularity": "token",
                    "value_prediction_granularity": "patch",
                    "value_reduction": "token_mean",
                    "policy_token_count": int(new_token_logprobs.numel()),
                    "value_token_count": int(new_token_logprobs.numel()),
                    "value_prediction_patch_count": int(new_logprobs.numel()),
                    "value_return_metrics": value_return_metrics,
                    "value_target_metrics": value_target_metrics,
                    "grad_norm": epoch_result.grad_norm,
                    "replay_microbatch_size": epoch_result.microbatch_size,
                    "replay_microbatch_count": epoch_result.microbatch_count,
                    "advantage_summary": advantage_summary,
                    "post_epoch_logprob_advantage_diagnostics": post_epoch_logprob_advantage_diag,
                    "duration_s": time.perf_counter() - ppo_epoch_start,
                }
            )
            if args.print_epoch_logs:
                epoch_log = ppo_epoch_logs[-1]
                print(
                    json.dumps(
                        {
                            "event": "ppo_epoch_complete",
                            "step": step_idx,
                            "epoch": ppo_epoch_idx,
                            "prompt_index": prompt_idx,
                            "prompt_name": prompt_name,
                            **prompt_batch_log,
                            "loss": epoch_log["loss"],
                            "policy_loss": epoch_log["policy_loss"],
                            "approx_kl": epoch_log["approx_kl"],
                            "old_policy_exact_kl": epoch_log["old_policy_exact_kl"],
                            "reference_exact_kl": epoch_log["reference_exact_kl"],
                            "reference_kl_loss": epoch_log["reference_kl_loss"],
                            "clip_fraction": epoch_log["clip_fraction"],
                            "grad_norm": epoch_log["grad_norm"],
                            "duration_s": epoch_log["duration_s"],
                            "advantage_summary": epoch_log["advantage_summary"],
                            "post_epoch_logprob_advantage_diagnostics": (
                                epoch_log["post_epoch_logprob_advantage_diagnostics"]
                            ),
                        }
                    ),
                    flush=True,
                )
        if loss_payload is None:
            raise RuntimeError("PPO update produced no loss payload")

        for trajectory_log, new_replay in zip(update_trajectory_logs, new_replays, strict=True):
            trajectory_log["value_mean"] = float(new_replay.values.mean().detach().cpu())
            trajectory_log["value_std"] = float(new_replay.values.std(unbiased=False).detach().cpu())
            trajectory_log["scored_patches"] = int(new_replay.logprobs.numel())
        for trajectory_log in skipped_update_logs:
            trajectory_log["value_mean"] = None
            trajectory_log["value_std"] = None
            trajectory_log["scored_patches"] = 0

        if not args.no_step and args.post_step_kl_check:
            post_step_kl_start = time.perf_counter()
            post_step_replay = post_step_replay_microbatched(
                policy_model=policy_model,
                value_head=value_head,
                flat_prompt_ids=[],
                rollout_payloads=update_rollout_payloads,
                args=args,
            )
            post_step_logprobs = post_step_replay.logprobs
            post_step_token_logprobs = post_step_replay.token_logprobs
            post_step_log_ratio = post_step_logprobs - old_logprobs
            post_step_ratio = torch.exp(post_step_log_ratio)
            post_step_token_log_ratio = post_step_token_logprobs - old_token_logprobs
            post_step_token_ratio = torch.exp(post_step_token_log_ratio)
            post_step_approx_kl = ((old_token_logprobs - post_step_token_logprobs) ** 2).mean() * 0.5
            post_step_old_policy_exact_kl = exact_categorical_kl(
                post_step_replay.token_log_dists,
                old_token_log_dists,
            )
            post_step_reference_exact_kl = (
                None
                if reference_token_log_dists is None
                else exact_categorical_kl(post_step_replay.token_log_dists, reference_token_log_dists)
            )
            post_step_clip_fraction = ((post_step_token_ratio - 1.0).abs() > args.ppo_clip_range).float().mean()
            timings["post_step_kl_check_s"] = time.perf_counter() - post_step_kl_start
        else:
            post_step_approx_kl = None
            post_step_old_policy_exact_kl = None
            post_step_reference_exact_kl = None
            post_step_clip_fraction = None
            post_step_logprobs = None
            post_step_log_ratio = None
            post_step_token_logprobs = None
            post_step_token_log_ratio = None
        timings["new_replay_backward_s"] = time.perf_counter() - new_replay_start

        checkpoint_payload = None
        checkpoint_start = time.perf_counter()
        if (
            not args.no_step
            and args.checkpoint_dir
            and args.checkpoint_every_steps > 0
            and step_idx % args.checkpoint_every_steps == 0
        ):
            checkpoint_payload = save_ppo_policy_checkpoint(
                policy_model,
                args.checkpoint_dir,
                step_idx,
                lora_r=args.lora_r,
        )
        timings["checkpoint_s"] = time.perf_counter() - checkpoint_start

        timings["ppo_replay_backward_s"] = time.perf_counter() - replay_start
        fixed_eval_log = None
        if fixed_eval_should_run_after_step(args, step_idx):
            fixed_eval_event_index = fixed_eval_event_cursor
            fixed_eval_event_cursor += 1
            fixed_eval_prompt_batch = build_fixed_eval_prompt_batch(
                prompts=prompts,
                prompt_targets=prompt_targets,
                args=args,
                event_index=fixed_eval_event_index,
            )
            fixed_eval_log = run_fixed_eval_batch(
                policy_model=policy_model,
                policy_shape=policy_shape,
                prompt_batch=fixed_eval_prompt_batch,
                reward_config=reward_config,
                similarity_weights=similarity_weights,
                aria_similarity_ref=aria_similarity_ref,
                args=args,
                step_idx=step_idx,
                label="after_step",
                event_index=fixed_eval_event_index,
                reference_policy_model=reference_policy_model,
            )
            if fixed_eval_log is not None:
                fixed_eval_logs.append(fixed_eval_log)
                timings["fixed_eval_s"] = fixed_eval_log["timings"]["fixed_eval_total_s"]
        else:
            timings["fixed_eval_s"] = 0.0
        timings["total_step_s"] = time.perf_counter() - step_start

        sample_rewards = [float(log["reward"]) for log in trajectory_logs]
        sample_rewards_array = np.array(sample_rewards, dtype=np.float32)
        patch_rewards = batch_tensors.patch_rewards
        returns = batch_tensors.returns
        value_targets = batch_tensors.value_targets
        patch_reward_component_sums = aggregate_component_sums(update_reward_traces)
        all_patch_reward_component_sums = aggregate_component_sums(reward_traces)
        rollout_length = _rollout_length_summary(trajectory_logs)
        logprob_advantage_diag = logprob_advantage_diagnostics(
            old_logprobs=old_logprobs,
            post_step_logprobs=post_step_logprobs,
            raw_advantages=batch_tensors.advantages,
            normalized_advantages=normalized_advantages,
            patch_rewards=patch_rewards,
            returns=returns,
            value_targets=value_targets,
            old_values=old_values,
            trajectory_lengths=trajectory_lengths,
            trajectory_logs=update_trajectory_logs,
            clip_range=args.ppo_clip_range,
            position_bins=args.position_diagnostic_bins,
        )
        step_log = {
            "step": step_idx,
            "prompt_index": prompt_idx,
            "prompt_name": prompt_name,
            **prompt_batch_log,
            "target_structure_path": prompt_target.structure_path,
            "target_structure_source_key": prompt_target.source_key,
            "target_expected_reward_bars": int(target.expected_reward_bars),
            "target_stream_lines": target_stream_lines,
            "trajectories_per_step": len(rollout_payloads),
            "ppo_update_trajectories": len(update_rollout_payloads),
            "zero_contribution_trajectories": zero_contribution_count,
            "failed_rollout_count": failed_rollout_count,
            "policy_granularity": "token",
            "policy_reduction": "token_mean",
            "advantage_granularity": "patch_repeated_per_token",
            "advantage_normalization_granularity": (
                "none" if args.no_advantage_normalization else "token"
            ),
            "value_function_granularity": "patch",
            "value_target_granularity": "patch",
            "value_loss_granularity": "token",
            "value_prediction_granularity": "patch",
            "value_reduction": "token_mean",
            "rollout_batch_size": args.rollout_batch_size,
            "rollout_sampling": rollout_sampling,
            "rollout_failure_policy": args.rollout_failure_policy,
            "loss": float(loss_payload.loss.detach().cpu()),
            "policy_loss": float(loss_payload.policy_loss.detach().cpu()),
            "value_loss": float(loss_payload.value_loss.detach().cpu()),
            "raw_value_loss": float(loss_payload.raw_value_loss.detach().cpu()),
            "value_loss_scale": float(loss_payload.value_loss_scale.detach().cpu()),
            "entropy_loss": float(loss_payload.entropy_loss.detach().cpu()),
            "reference_kl_loss": float(loss_payload.reference_kl_loss.detach().cpu()),
            "approx_kl": float(loss_payload.approx_kl.detach().cpu()),
            "old_policy_exact_kl": float(loss_payload.old_policy_exact_kl.detach().cpu()),
            "reference_exact_kl": float(loss_payload.reference_exact_kl.detach().cpu()),
            "reference_kl_coef": float(args.reference_kl_coef),
            "reference_kl_enabled": bool(reference_policy_model is not None),
            "clip_fraction": float(loss_payload.clip_fraction.detach().cpu()),
            "post_step_approx_kl": (
                None if post_step_approx_kl is None else float(post_step_approx_kl.detach().cpu())
            ),
            "post_step_old_policy_exact_kl": (
                None
                if post_step_old_policy_exact_kl is None
                else float(post_step_old_policy_exact_kl.detach().cpu())
            ),
            "post_step_reference_exact_kl": (
                None
                if post_step_reference_exact_kl is None
                else float(post_step_reference_exact_kl.detach().cpu())
            ),
            "post_step_clip_fraction": (
                None if post_step_clip_fraction is None else float(post_step_clip_fraction.detach().cpu())
            ),
            "post_step_log_ratio_mean": (
                None if post_step_token_log_ratio is None else float(post_step_token_log_ratio.mean().detach().cpu())
            ),
            "post_step_log_ratio_max_abs": (
                None if post_step_token_log_ratio is None else float(post_step_token_log_ratio.abs().max().detach().cpu())
            ),
            "post_step_patch_log_ratio_mean": (
                None if post_step_log_ratio is None else float(post_step_log_ratio.mean().detach().cpu())
            ),
            "post_step_patch_log_ratio_max_abs": (
                None if post_step_log_ratio is None else float(post_step_log_ratio.abs().max().detach().cpu())
            ),
            "logprob_advantage_diagnostics": logprob_advantage_diag,
            "advantage_summary": advantage_summary,
            "advantages_mean": float(loss_payload.advantages_mean.detach().cpu()),
            "advantages_std": float(loss_payload.advantages_std.detach().cpu()),
            "return_mean": float(returns.mean().detach().cpu()),
            "return_std": float(returns.std(unbiased=False).detach().cpu()),
            "value_target_mean": float(value_targets.mean().detach().cpu()),
            "value_target_std": float(value_targets.std(unbiased=False).detach().cpu()),
            "gae_lambda": args.gae_lambda,
            "patch_reward_attribution": args.patch_reward_attribution,
            "ppo_epochs": args.ppo_epochs,
            "ppo_replay_microbatch_size": _effective_microbatch_size(
                args.ppo_replay_microbatch_size,
                len(update_rollout_payloads),
            ),
            "frozen_behavior_policy": bool(behavior_policy_model is not None),
            "reference_policy": bool(reference_policy_model is not None),
            "ppo_epoch_logs": ppo_epoch_logs,
            "value_warmup": value_warmup_log,
            "normalize_value_loss": args.normalize_value_loss,
            "value_loss_scale_min": args.value_loss_scale_min,
            "initial_value_return_metrics": initial_value_return_metrics,
            "post_warmup_value_return_metrics": post_warmup_value_return_metrics,
            "post_warmup_value_target_metrics": post_warmup_value_target_metrics,
            "final_value_return_metrics": value_prediction_metrics(new_values, returns),
            "final_value_target_metrics": value_prediction_metrics(new_values, value_targets),
            "patch_reward_mean": float(patch_rewards.mean().detach().cpu()),
            "patch_reward_std": float(patch_rewards.std(unbiased=False).detach().cpu()),
            "patch_rewards": patch_rewards.detach().cpu().tolist(),
            "patch_reward_component_sums": patch_reward_component_sums,
            "all_patch_reward_component_sums": all_patch_reward_component_sums,
            "patch_reward_group_sums": component_group_sums(patch_reward_component_sums),
            "all_patch_reward_group_sums": component_group_sums(all_patch_reward_component_sums),
            "rollout_length": rollout_length,
            "patch_reward_prefix_totals": (
                update_trajectory_logs[0]["patch_reward_prefix_totals"]
                if len(update_trajectory_logs) == 1
                else None
            ),
            "value_mean": float(new_values.mean().detach().cpu()),
            "value_std": float(new_values.std(unbiased=False).detach().cpu()),
            "scored_patches": int(new_logprobs.numel()),
            "scored_tokens": int(new_token_logprobs.numel()),
            "generated_tokens_per_patch_mean": float(old_token_counts.float().mean().detach().cpu()),
            "generated_tokens_per_patch_min": int(old_token_counts.min().detach().cpu()),
            "generated_tokens_per_patch_max": int(old_token_counts.max().detach().cpu()),
            "trajectory_token_lengths": trajectory_token_lengths,
            "reward": float(sample_rewards_array.mean()),
            "reward_mean": float(sample_rewards_array.mean()),
            "reward_std": float(sample_rewards_array.std()),
            "reward_min": float(sample_rewards_array.min()),
            "reward_max": float(sample_rewards_array.max()),
            "reward_sum": float(sample_rewards_array.sum()),
            "sample_rewards": sample_rewards,
            "update_sample_rewards": [float(log["reward"]) for log in update_trajectory_logs],
            "skipped_update_trajectory_indices": [
                int(log["trajectory_index"]) for log in skipped_update_logs
            ],
            "reward_breakdown": trajectory_logs[0]["reward_breakdown"] if len(trajectory_logs) == 1 else None,
            "fixed_eval": fixed_eval_log,
            "checkpoint": checkpoint_payload,
            "trajectories": trajectory_logs,
            "timings": timings,
        }
        if args.save_patch_diagnostics:
            diagnostic_component_rewards = component_reward_tensors(update_reward_traces, device=device)
            diagnostic_component_lambda_returns = component_lambda_return_tensors(
                update_reward_traces,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                device=device,
            )
            step_log["patch_diagnostics"] = per_patch_diagnostic_records(
                old_logprobs=old_logprobs,
                post_step_logprobs=post_step_logprobs,
                raw_advantages=batch_tensors.advantages,
                normalized_advantages=normalized_advantages,
                patch_rewards=patch_rewards,
                returns=returns,
                value_targets=value_targets,
                old_values=old_values,
                trajectory_lengths=trajectory_lengths,
                component_rewards=diagnostic_component_rewards,
                component_lambda_returns=diagnostic_component_lambda_returns,
            )
        print(json.dumps({"event": "ppo_step_complete", **step_log}), flush=True)
        logs.append(step_log)
        del (
            old_replays,
            reward_tensors,
            old_logprobs,
            old_token_logprobs,
            old_token_log_dists,
            old_token_counts,
            reference_replays,
            reference_token_log_dists,
            initial_old_value_tensors,
            return_tensors,
            returns_for_metrics,
            old_value_tensors,
            batch_tensors,
            old_values,
            new_replays,
            new_logprobs,
            new_token_logprobs,
            new_values,
            loss_payload,
            patch_rewards,
            returns,
            value_targets,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "steps": logs,
        "fixed_eval_logs": fixed_eval_logs,
        "policy_dropout_modules_disabled": dropout_modules_disabled,
        "behavior_policy_dropout_modules_disabled": behavior_dropout_modules_disabled,
        "reference_policy_dropout_modules_disabled": reference_dropout_modules_disabled,
        "value_head": {
            **value_head.config(),
            "trainable_params": sum(param.numel() for param in value_head.parameters() if param.requires_grad),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal PPO smoke runner for NotaGen Goldberg experiments.")
    parser.add_argument("--policy-weights", required=True)
    parser.add_argument("--prompts-jsonl", default=str(Path("data/processed/notagen/goldberg_grpo_prompts.jsonl")))
    parser.add_argument("--target-json", default=str(Path("data/processed/goldberg/structure/aria_bar_skeleton.json")))
    parser.add_argument("--target-structure-abc", required=True)
    parser.add_argument("--aria-reference-abc", default=str(Path("data/processed/goldberg/abc/aria-bwv-988.abc")))
    parser.add_argument("--aria-chroma-reward-weight", type=float, default=1.0)
    parser.add_argument("--aria-harmony-reward-weight", type=float, default=1.0)
    parser.add_argument("--max-similarity-reward", type=float, default=2.0)
    parser.add_argument("--similarity-chroma-bins", type=int, default=128)
    parser.add_argument("--similarity-band-ratio", type=float, default=0.25)
    parser.add_argument("--similarity-timeout-s", type=float, default=20.0)
    parser.add_argument(
        "--parse-validation-mode",
        choices=("music21", "abc-tokenize", "none"),
        default="music21",
        help=(
            "Parse-validity check used by the structural parse reward. "
            "music21 is exact but slow; abc-tokenize is much faster but only checks ABC tokenization; "
            "none treats parse validity as true for speed/debug runs."
        ),
    )
    parser.add_argument("--music21-parse-timeout-s", type=float, default=5.0)
    parser.add_argument("--parse-reward-weight", type=float, default=1.0)
    parser.add_argument("--countdown-reward-weight", type=float, default=0.25)
    parser.add_argument("--line-closure-reward-weight", type=float, default=0.25)
    parser.add_argument("--bar-token-reward-weight", type=float, default=0.10)
    parser.add_argument("--meter-alignment-reward-weight", type=float, default=0.75)
    parser.add_argument("--meter-duration-closeness-reward-weight", type=float, default=0.75)
    parser.add_argument("--bar-meter-consistency-reward-weight", type=float, default=0.75)
    parser.add_argument("--bar-count-reward-weight", type=float, default=3.0)
    parser.add_argument("--voice-declaration-reward-weight", type=float, default=1.0)
    parser.add_argument("--score-voice-reward-weight", type=float, default=0.5)
    parser.add_argument(
        "--rollout-failure-terminal-reward",
        type=float,
        default=-1.0,
        help=(
            "Terminal reward assigned to inference-failed PPO trajectories. If the failed trajectory "
            "has generated patches, this is placed on the final generated patch; empty failures are "
            "logged with this reward but cannot contribute a PPO gradient."
        ),
    )
    parser.add_argument(
        "--reward-mode",
        choices=("goldberg", "note_count", "note_fraction", "length"),
        default="goldberg",
        help=(
            "Reward objective. goldberg uses the normal structural/similarity reward. "
            "note_count, note_fraction, and length are simple PPO sanity-test rewards."
        ),
    )
    parser.add_argument(
        "--simple-reward-note",
        default="G",
        help="ABC pitch letter A-G counted by --reward-mode note_count or note_fraction, case-insensitive.",
    )
    parser.add_argument(
        "--simple-reward-max-count",
        type=float,
        default=64.0,
        help="Count at which --reward-mode note_count reaches --simple-reward-scale.",
    )
    parser.add_argument(
        "--simple-reward-length-unit",
        choices=("patches", "chars", "stream_lines"),
        default="patches",
        help="Length unit for --reward-mode length.",
    )
    parser.add_argument(
        "--simple-reward-length-target",
        type=float,
        default=160.0,
        help="Length at which --reward-mode length reaches --simple-reward-scale.",
    )
    parser.add_argument(
        "--simple-reward-scale",
        type=float,
        default=1.0,
        help="Simple reward multiplier; for note_count/length this is the reward after clipping to the target.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--prompt-limit",
        type=int,
        default=1,
        help="Number of prompt rows to load from --prompts-jsonl. Use 30 for the full Goldberg PPO prompt set.",
    )
    parser.add_argument(
        "--prompt-selection",
        choices=("ordered", "random"),
        default="ordered",
        help=(
            "How PPO selects one prompt per update step. ordered preserves modulo order. "
            "random uses a deterministic seeded shuffle cycle: every loaded prompt appears "
            "once per cycle, then the next cycle is reshuffled."
        ),
    )
    parser.add_argument(
        "--prompt-batch-mode",
        choices=("trajectory", "step"),
        default="trajectory",
        help=(
            "How prompts are assigned within a PPO step. trajectory consumes one prompt-schedule "
            "slot per sampled trajectory, so a step may contain multiple prompts. step consumes "
            "one prompt-schedule slot per PPO step and repeats that prompt for every trajectory "
            "in the step; use this for controlled prompt-effect sweeps."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument(
        "--trajectories-per-step",
        type=int,
        default=1,
        help="Number of completions sampled from the selected prompt for each PPO step.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--target-stream-lines", type=int, default=32)
    parser.add_argument("--max-chars", type=int, default=40000)
    parser.add_argument("--max-generated-patches", type=int, default=256)
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--rollout-retries", type=int, default=1)
    parser.add_argument(
        "--rollout-failure-policy",
        choices=("error", "zero", "spares"),
        default="error",
        help=(
            "How PPO handles rollout sampling failures. 'error' preserves strict retry/abort "
            "behavior. 'zero' records failed trajectory slots after one attempt and scores "
            "them with --rollout-failure-terminal-reward; partial failed generations can "
            "contribute to the PPO loss, while empty failures are logged but have no tokens "
            "to update. 'spares' oversamples candidates in the batched rollout and keeps the "
            "first successful trajectories_per_step candidates."
        ),
    )
    parser.add_argument(
        "--rollout-spares-percent",
        type=float,
        default=10.0,
        help=(
            "Extra rollout candidates to sample in --rollout-failure-policy spares mode, "
            "as a percentage of --trajectories-per-step. For example, 10 with 16 "
            "trajectories samples 18 candidates and keeps 16 successes."
        ),
    )
    parser.add_argument(
        "--reward-workers",
        "--workers",
        dest="reward_workers",
        type=int,
        default=0,
        help=(
            "Number of CPU worker processes used to score rollout rewards. "
            "Use 0 or 1 for serial scoring."
        ),
    )
    parser.add_argument(
        "--reward-worker-start-method",
        choices=tuple(mp.get_all_start_methods()),
        default=_default_reward_worker_start_method(),
        help=(
            "Multiprocessing start method for --reward-workers. forkserver/spawn avoid forking "
            "a live CUDA process; fork is useful for fast local tests without CUDA."
        ),
    )
    parser.add_argument(
        "--rollout-batch-size",
        type=int,
        default=1,
        help="Generate cached rollouts in batches. Values >1 require --cached-rollout.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--value-learning-rate", type=float, default=1e-5)
    parser.add_argument("--value-head-hidden-size", type=int, default=512)
    parser.add_argument("--value-head-dropout", type=float, default=0.0)
    parser.add_argument("--value-head-weights")
    parser.add_argument("--save-value-head-weights")
    parser.add_argument("--value-warmup-epochs", type=int, default=0)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--ppo-clip-range", type=float, default=0.2)
    parser.add_argument("--value-loss-coef", type=float, default=0.5)
    parser.add_argument("--normalize-value-loss", action="store_true")
    parser.add_argument("--value-loss-eps", type=float, default=1e-6)
    parser.add_argument(
        "--value-loss-scale-min",
        type=float,
        default=1e-6,
        help="Minimum denominator used only when --normalize-value-loss is enabled.",
    )
    parser.add_argument("--entropy-bonus-coef", type=float, default=0.0)
    parser.add_argument(
        "--reference-kl-coef",
        type=float,
        default=0.0,
        help=(
            "Coefficient for exact categorical KL(pi_current || pi_reference) over generated character tokens. "
            "The reference defaults to --policy-weights without PPO LoRA/checkpoint state."
        ),
    )
    parser.add_argument(
        "--reference-kl-check",
        action="store_true",
        help="Load the reference/SFT model and log exact reference KL even when --reference-kl-coef is 0.",
    )
    parser.add_argument(
        "--reference-policy-weights",
        help="Optional full-model weights for the reference/SFT model. Defaults to --policy-weights.",
    )
    parser.add_argument(
        "--reference-checkpoint-dir",
        help="Optional LoRA checkpoint to load into the reference model. Leave unset to anchor to raw SFT/base weights.",
    )
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument(
        "--patch-reward-attribution",
        choices=("single_pass", "terminal"),
        default="single_pass",
        help=(
            "How to assign final trajectory reward to generated patches. single_pass uses "
            "line/harmony events plus terminal residual; terminal puts every reward component "
            "on the final generated patch and relies on returns/GAE to propagate backward."
        ),
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--replay-context-patches", type=int, default=128)
    parser.add_argument("--score-chunk-patches", type=int, default=64)
    parser.add_argument(
        "--ppo-replay-microbatch-size",
        type=int,
        default=0,
        help=(
            "Number of trajectories to replay/backprop at once inside a PPO epoch. "
            "Use 0 to replay all trajectories together."
        ),
    )
    parser.add_argument("--lora-r", type=int, default=0)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--resume-checkpoint-dir")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--checkpoint-every-steps", type=int, default=0)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--rollout-seed-scope",
        choices=("step", "run"),
        default="step",
        help=(
            "How rollout RNG seeds vary across PPO steps. step is normal PPO behavior. "
            "run reuses the same trajectory-index seed set every step, useful for frozen-policy "
            "prompt sweeps where the prompt should be the only changing factor."
        ),
    )
    parser.add_argument("--step-offset", type=int, default=0)
    parser.add_argument("--cached-rollout", action="store_true")
    parser.add_argument(
        "--rollout-only",
        action="store_true",
        help="Only sample and score rollouts. Saves generated patches and patch rewards, but skips PPO replay/update.",
    )
    parser.add_argument("--no-step", action="store_true")
    parser.add_argument("--no-advantage-normalization", action="store_true")
    parser.add_argument(
        "--post-step-kl-check",
        action="store_true",
        help="After optimizer.step(), replay the same trajectories and log post-update KL/clip diagnostics.",
    )
    parser.add_argument(
        "--post-epoch-kl-check",
        action="store_true",
        help=(
            "After each PPO epoch optimizer step, replay the same trajectories and log compact "
            "post-update advantage/log-ratio diagnostics against the fixed old logprobs."
        ),
    )
    parser.add_argument(
        "--print-epoch-logs",
        action="store_true",
        help="Print one compact JSON event after each PPO epoch. Useful for long fixed-behavior diagnostics.",
    )
    parser.add_argument(
        "--frozen-behavior-policy",
        action="store_true",
        help=(
            "Build a separate frozen copy of the initial policy and use it for rollouts and old logprobs "
            "across all PPO steps. This samples fresh trajectories from a fixed pi_old while training "
            "the main policy."
        ),
    )
    parser.add_argument(
        "--position-diagnostic-bins",
        type=int,
        default=5,
        help="Number of relative patch-position bins to summarize in PPO logprob/advantage diagnostics. Use 0 to disable.",
    )
    parser.add_argument(
        "--save-patch-diagnostics",
        action="store_true",
        help=(
            "Persist one row per generated patch with logprobs, advantages, rewards, returns, values, "
            "and relative position. Useful for local slicing, but can make result.json large."
        ),
    )
    parser.add_argument(
        "--fixed-eval-trajectories",
        type=int,
        default=0,
        help=(
            "When fixed eval runs, sample this many trajectories from the independent eval prompt schedule "
            "and score them with the same reward path. Use 0 to disable fixed-policy evaluation."
        ),
    )
    parser.add_argument(
        "--fixed-eval-every-steps",
        type=int,
        default=1,
        help=(
            "Run fixed eval after every N PPO steps. Use 1 for every step, or 0 to disable "
            "after-step fixed eval while still allowing --fixed-eval-before-training."
        ),
    )
    parser.add_argument(
        "--fixed-eval-prompt-selection",
        choices=("same", "ordered", "random"),
        default="same",
        help=(
            "Prompt schedule used only for fixed eval. 'same' reuses --prompt-selection mode "
            "with an independent seed/slot counter; ordered/random force a separate eval mode."
        ),
    )
    parser.add_argument(
        "--fixed-eval-prompt-batch-mode",
        choices=("same", "trajectory", "event"),
        default="same",
        help=(
            "Prompt assignment within each fixed-eval event. same mirrors --prompt-batch-mode "
            "(step becomes event). trajectory consumes one prompt slot per eval trajectory. "
            "event repeats one selected prompt for the whole eval event."
        ),
    )
    parser.add_argument(
        "--fixed-eval-prompt-seed-offset",
        type=int,
        default=2_000_000,
        help="Offset added to --seed for the independent fixed-eval prompt shuffle.",
    )
    parser.add_argument(
        "--fixed-eval-rollout-batch-size",
        type=int,
        default=0,
        help=(
            "Rollout batch size for fixed evaluation. Use 0 to reuse the smaller of --rollout-batch-size "
            "and --fixed-eval-trajectories."
        ),
    )
    parser.add_argument(
        "--fixed-eval-rollout-retries",
        type=int,
        default=1,
        help="Retry count for fixed-eval sampling failures. Fixed-eval failures are logged and do not abort PPO.",
    )
    parser.add_argument(
        "--fixed-eval-seed-offset",
        type=int,
        default=1_000_000,
        help="Offset added to --seed for the fixed eval batch so train and eval rollouts use distinct seeds.",
    )
    parser.add_argument(
        "--fixed-eval-seed-step",
        type=int,
        default=0,
        help="Synthetic step index used for fixed-eval rollout seeds. Keep constant for repeated eval comparability.",
    )
    parser.add_argument(
        "--fixed-eval-reference-kl-check",
        action="store_true",
        help=(
            "For fixed eval, replay generated tokens through the current policy and reference/SFT policy "
            "and log exact full-character-vocabulary KL(pi_current || pi_reference)."
        ),
    )
    parser.add_argument(
        "--fixed-eval-kl-replay-microbatch-size",
        type=int,
        default=0,
        help=(
            "Trajectory microbatch size for fixed-eval exact KL replay. Use 0 to replay the full "
            "fixed-eval batch per prompt group."
        ),
    )
    parser.add_argument(
        "--fixed-eval-before-training",
        action="store_true",
        help="Run the fixed eval batch once before the first PPO update to establish a baseline.",
    )
    parser.add_argument(
        "--fixed-eval-output-jsonl",
        help="Optional JSONL path for fixed-eval records. Defaults to fixed_eval.jsonl beside --output-json.",
    )
    parser.add_argument(
        "--fixed-eval-save-trajectories",
        action="store_true",
        help="Persist fixed-eval generated text and patches in the fixed-eval JSONL sidecar.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = select_device()
    policy_weights = Path(args.policy_weights)
    policy_shape = infer_model_shape(policy_weights)
    policy_model = build_model(
        policy_weights,
        device,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        precision=args.precision,
    )
    resume_payload = None
    if args.resume_checkpoint_dir:
        resume_payload = load_policy_checkpoint(policy_model, Path(args.resume_checkpoint_dir))
        print(f"Resumed policy LoRA checkpoint from {args.resume_checkpoint_dir}")
    behavior_policy_model = None
    if args.frozen_behavior_policy:
        behavior_policy_model = build_model(
            policy_weights,
            device,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            precision=args.precision,
            freeze_before_precision_cast=True,
        )
        if args.resume_checkpoint_dir:
            load_policy_checkpoint(behavior_policy_model, Path(args.resume_checkpoint_dir))
            print(f"Loaded frozen behavior policy LoRA checkpoint from {args.resume_checkpoint_dir}")
        print("Frozen behavior policy enabled for rollouts and old logprobs")
    reference_policy_model = None
    reference_payload = None
    reference_policy_weights = None
    reference_policy_requested = (
        args.reference_kl_coef != 0.0
        or args.reference_kl_check
        or args.fixed_eval_reference_kl_check
    )
    if reference_policy_requested:
        reference_policy_weights = Path(args.reference_policy_weights) if args.reference_policy_weights else policy_weights
        reference_policy_model = build_model(
            reference_policy_weights,
            device,
            lora_r=args.lora_r if args.reference_checkpoint_dir else 0,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            precision=args.precision,
            freeze_before_precision_cast=True,
        )
        if args.reference_checkpoint_dir:
            reference_payload = load_policy_checkpoint(reference_policy_model, Path(args.reference_checkpoint_dir))
            print(f"Loaded reference policy LoRA checkpoint from {args.reference_checkpoint_dir}")
        print(f"Reference KL model enabled from {reference_policy_weights}")
    value_head, value_head_load = build_value_head(policy_shape, args, device)
    prompts = load_prompt_rows(args.prompts_jsonl, limit=args.prompt_limit)
    prompt_targets = load_prompt_structural_targets(prompts, args)
    reward_config = GoldbergRewardConfig(
        parse_weight=args.parse_reward_weight,
        countdown_weight=args.countdown_reward_weight,
        line_closure_weight=args.line_closure_reward_weight,
        bar_token_weight=args.bar_token_reward_weight,
        meter_alignment_weight=args.meter_alignment_reward_weight,
        meter_duration_closeness_weight=args.meter_duration_closeness_reward_weight,
        bar_meter_consistency_weight=args.bar_meter_consistency_reward_weight,
        bar_count_weight=args.bar_count_reward_weight,
        voice_declaration_weight=args.voice_declaration_reward_weight,
        score_voice_weight=args.score_voice_reward_weight,
        parse_validation_mode=args.parse_validation_mode,
        music21_parse_timeout_s=args.music21_parse_timeout_s,
    )
    payload = run_ppo_smoke(
        policy_model=policy_model,
        policy_shape=policy_shape,
        value_head=value_head,
        prompts=prompts,
        prompt_targets=prompt_targets,
        reward_config=reward_config,
        args=args,
        behavior_policy_model=behavior_policy_model,
        reference_policy_model=reference_policy_model,
    )
    if args.save_value_head_weights:
        save_value_head_checkpoint(value_head, args.save_value_head_weights)
        payload["saved_value_head_weights"] = str(args.save_value_head_weights)
    if value_head_load:
        payload["loaded_value_head_weights"] = value_head_load
    if resume_payload:
        payload["resume_checkpoint"] = resume_payload
    if reference_payload:
        payload["reference_checkpoint"] = reference_payload
    payload["run_config"] = {
        "args": vars(args),
        "policy_shape": asdict(policy_shape),
        "reward_config": asdict(reward_config),
        "policy_weights": str(policy_weights),
        "reference_policy_weights": None if reference_policy_weights is None else str(reference_policy_weights),
        "prompt_structural_targets": prompt_structural_target_metadata(prompt_targets),
        "reward_mode": {
            "mode": args.reward_mode,
            "simple_reward_note": args.simple_reward_note,
            "simple_reward_max_count": args.simple_reward_max_count,
            "simple_reward_length_unit": args.simple_reward_length_unit,
            "simple_reward_length_target": args.simple_reward_length_target,
            "simple_reward_scale": args.simple_reward_scale,
            "rollout_failure_terminal_reward": args.rollout_failure_terminal_reward,
        },
        "prompt_schedule": {
            "prompt_limit": args.prompt_limit,
            "loaded_prompt_count": len(prompts),
            "prompt_selection": args.prompt_selection,
            "prompt_batch_mode": args.prompt_batch_mode,
            "consumption_granularity": args.prompt_batch_mode,
            "one_prompt_per_trajectory": args.prompt_batch_mode == "trajectory",
            "one_prompt_per_step": args.prompt_batch_mode == "step",
            "seed": args.seed,
            "rollout_seed_scope": args.rollout_seed_scope,
            "step_offset": args.step_offset,
        },
        "fixed_eval": {
            "trajectories": args.fixed_eval_trajectories,
            "every_steps": args.fixed_eval_every_steps,
            "before_training": args.fixed_eval_before_training,
            "prompt_selection": args.fixed_eval_prompt_selection,
            "resolved_prompt_selection": resolve_fixed_eval_prompt_selection(args),
            "prompt_seed": args.seed + args.fixed_eval_prompt_seed_offset,
            "prompt_seed_offset": args.fixed_eval_prompt_seed_offset,
            "prompt_batch_mode": args.fixed_eval_prompt_batch_mode,
            "resolved_prompt_batch_mode": resolve_fixed_eval_prompt_batch_mode(args),
            "consumption_granularity": resolve_fixed_eval_prompt_batch_mode(args),
            "rollout_batch_size": args.fixed_eval_rollout_batch_size,
            "rollout_retries": args.fixed_eval_rollout_retries,
            "seed_offset": args.fixed_eval_seed_offset,
            "seed_step": args.fixed_eval_seed_step,
            "reference_kl_check": args.fixed_eval_reference_kl_check,
            "kl_replay_microbatch_size": args.fixed_eval_kl_replay_microbatch_size,
        },
        "ppo": {
            "clip_range": args.ppo_clip_range,
            "value_loss_coef": args.value_loss_coef,
            "entropy_bonus_coef": args.entropy_bonus_coef,
            "reference_kl_coef": args.reference_kl_coef,
            "reference_kl_check": args.reference_kl_check,
            "reference_policy": bool(reference_policy_model is not None),
            "reference_checkpoint_dir": args.reference_checkpoint_dir,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "ppo_epochs": args.ppo_epochs,
            "frozen_behavior_policy": args.frozen_behavior_policy,
            "value_warmup_epochs": args.value_warmup_epochs,
            "normalize_value_loss": args.normalize_value_loss,
            "value_loss_scale_min": args.value_loss_scale_min,
            "reward_assignment": (
                "terminal_total_reward"
                if args.patch_reward_attribution == "terminal"
                else "single_pass_events_plus_terminal_residual"
            ),
            "patch_reward_attribution": args.patch_reward_attribution,
            "rollout_only": args.rollout_only,
        },
    }
    Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
