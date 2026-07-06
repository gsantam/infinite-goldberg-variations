from __future__ import annotations

import argparse
import bisect
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from evaluation.rewards import (
    _abc_grammar_metrics,
    _extract_header_context,
    _extract_stream_line_features,
    _validated_bar_metrics,
)
from scripts.custom_grpo_notagen import (
    PATCH_SIZE,
    PATCH_STREAM,
    GoldbergRewardConfig,
    ModelShape,
    RolloutSample,
    SimilarityReference,
    SimilarityRewardWeights,
    _encoded_last_patch,
    _pad_generated_patch,
    _replay_start_patch,
    _rollout_seed,
    _split_flat_logprobs,
    autocast_context,
    build_model,
    build_rollout_prefix,
    char_patch_logprobs,
    count_stream_lines,
    disable_dropout_modules,
    generated_token_slots,
    grpo_kl_term,
    infer_model_shape,
    load_prompt_rows,
    load_similarity_reference,
    load_structural_target,
    normalize_patch_for_context,
    prompt_row_name,
    sample_completion,
    score_prompt_completion_pair,
    score_similarity_reward,
    select_device,
    set_seed,
)
from utils import NotaGenLMHeadModel, Patchilizer


@dataclass
class PatchReplayChunk:
    logprobs: torch.Tensor
    values: torch.Tensor


@dataclass
class PPOLossPayload:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy_loss: torch.Tensor
    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor
    advantages_mean: torch.Tensor
    advantages_std: torch.Tensor


@dataclass
class RewardScore:
    total: float
    breakdown: dict


@dataclass
class PatchRewardTrace:
    rewards: list[float]
    prefix_totals: list[float]
    final_score: RewardScore


class PatchValueHead(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states.float()).squeeze(-1)


def value_from_last_patch(
    model: NotaGenLMHeadModel,
    value_head: PatchValueHead,
    flat_ids: list[int],
    precision: str,
    replay_context_patches: int | None = None,
) -> torch.Tensor:
    device = next(model.parameters()).device
    encoded_patch, _tokens = _encoded_last_patch(
        model,
        flat_ids,
        device,
        precision,
        replay_context_patches=replay_context_patches,
    )
    return value_head(encoded_patch)


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
    for tok in patch:
        token_embeddings = torch.nn.functional.embedding(
            tokens.reshape(1, -1),
            model.char_level_decoder.base.transformer.wte.weight,
        )
        inputs_embeds = torch.cat((encoded_patch.reshape(1, 1, -1), token_embeddings[:, 1:, :]), dim=1)
        with autocast_context(device, precision):
            outputs = model.char_level_decoder.base(inputs_embeds=inputs_embeds)
            logits = outputs.logits[0, -1]
        logprobs.append(torch.log_softmax(logits.float(), dim=-1)[tok])
        if len(tokens) >= PATCH_SIZE:
            break
        tokens = torch.cat((tokens, torch.tensor([tok], device=device, dtype=torch.long)), dim=0)

    if not logprobs:
        raise RuntimeError("cannot score an empty generated patch")
    return PatchReplayChunk(
        logprobs=torch.stack(logprobs).sum().reshape(1),
        values=value_head(encoded_patch).reshape(1),
    )


def char_patch_logprob_sums(
    model: NotaGenLMHeadModel,
    encoded_patches: torch.Tensor,
    target_patches: list[list[int]],
    precision: str,
) -> torch.Tensor:
    special_token_id = model.special_token_id
    target_tensor = torch.tensor(
        [_pad_generated_patch(patch, special_token_id) for patch in target_patches],
        device=encoded_patches.device,
        dtype=torch.long,
    )
    token_counts = [sum(1 for token in patch if token != special_token_id) for patch in target_tensor.tolist()]
    flat_logprobs = char_patch_logprobs(model, encoded_patches, target_tensor, precision)
    per_patch = _split_flat_logprobs(flat_logprobs, token_counts)
    return torch.stack(
        [
            item.sum() if item.numel() > 0 else torch.zeros((), device=encoded_patches.device, dtype=flat_logprobs.dtype)
            for item in per_patch
        ]
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
    normalized_prefix = [
        normalize_patch_for_context(
            patch,
            eos_token_id=model.eos_token_id,
            special_token_id=model.special_token_id,
        )
        for patch in remaining_patches[:chunk_end]
    ]
    all_ids = current_ids[:]
    for patch in normalized_prefix:
        all_ids.extend(patch)

    if len(all_ids) % PATCH_SIZE != 0:
        raise RuntimeError("PPO replay expected full-patch alignment before chunked tail scoring")

    total_patches = len(all_ids) // PATCH_SIZE
    context_patch_count = len(current_ids) // PATCH_SIZE
    start_patch = _replay_start_patch(total_patches, context_patch_count, replay_context_patches)

    trimmed_ids = all_ids[start_patch * PATCH_SIZE :]
    device = next(model.parameters()).device
    patches_tensor = torch.tensor(trimmed_ids, device=device, dtype=torch.long).reshape(1, -1, PATCH_SIZE)
    first_target_local = context_patch_count - start_patch
    if first_target_local <= 0:
        raise RuntimeError("PPO replay window dropped all context before generated target patches")

    with autocast_context(device, precision):
        encoded = model.patch_level_decoder(patches_tensor)["last_hidden_state"][0]
    encoded_start = first_target_local + chunk_start - 1
    encoded_end = first_target_local + chunk_end - 1
    encoded_targets = encoded[encoded_start:encoded_end]
    target_patches = remaining_patches[chunk_start:chunk_end]
    return PatchReplayChunk(
        logprobs=char_patch_logprob_sums(model, encoded_targets, target_patches, precision),
        values=value_head(encoded_targets),
    )


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
    if not chunks:
        return PatchReplayChunk(
            logprobs=torch.empty(0, device=device),
            values=torch.empty(0, device=device),
        )
    return PatchReplayChunk(
        logprobs=torch.cat([chunk.logprobs for chunk in chunks]),
        values=torch.cat([chunk.values for chunk in chunks]),
    )


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


def _stream_line_end_patch_indices(completion_text: str, patch_texts: list[str]) -> list[int]:
    cumulative_offsets: list[int] = []
    offset = 0
    for patch_text in patch_texts:
        offset += len(patch_text)
        cumulative_offsets.append(offset)

    starts = [match.start() for match in re.finditer(r"\[r:\d+/\d+\]", completion_text)]
    if not starts:
        return []
    ends = starts[1:] + [len(completion_text)]
    return [min(bisect.bisect_left(cumulative_offsets, end), len(patch_texts) - 1) for end in ends]


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


def _single_pass_line_rewards(
    *,
    full_text: str,
    target,
    reward_config: GoldbergRewardConfig,
) -> list[float]:
    stream_lines = _extract_stream_line_features(full_text)
    if not stream_lines:
        return []

    n = len(stream_lines)
    header = _extract_header_context(full_text)
    closure = np.array([1.0 if line.closed else 0.0 for line in stream_lines], dtype=np.float32)
    bar_token = np.array([1.0 if line.has_bar_token else 0.0 for line in stream_lines], dtype=np.float32)
    countdown = _countdown_local_rewards(stream_lines)
    meter_alignment = np.zeros(n, dtype=np.float32)
    meter_duration = np.zeros(n, dtype=np.float32)
    bar_meter = np.zeros(n, dtype=np.float32)
    voice_decl = np.zeros(n, dtype=np.float32)
    score_voice = np.zeros(n, dtype=np.float32)

    for idx, line in enumerate(stream_lines):
        meter_metrics = _validated_bar_metrics([line], header)
        grammar_metrics = _abc_grammar_metrics([line], header)
        meter_alignment[idx] = meter_metrics.meter_alignment_reward
        meter_duration[idx] = meter_metrics.meter_duration_closeness_reward
        bar_meter[idx] = meter_metrics.bar_meter_consistency_reward
        voice_decl[idx] = grammar_metrics.voice_declaration_reward
        score_voice[idx] = grammar_metrics.score_voice_reward

    line_denominator = float(max(1, n))
    line_rewards = (
        reward_config.countdown_weight * countdown / line_denominator
        + reward_config.line_closure_weight * closure / line_denominator
        + reward_config.bar_token_weight * bar_token / line_denominator
        + reward_config.meter_alignment_weight * meter_alignment / line_denominator
        + reward_config.meter_duration_closeness_weight * meter_duration / line_denominator
        + reward_config.bar_meter_consistency_weight * bar_meter / line_denominator
        + reward_config.voice_declaration_weight * voice_decl / line_denominator
        + reward_config.score_voice_weight * score_voice / line_denominator
    )

    counts = np.arange(1, n + 1, dtype=np.float32)
    previous_counts = np.arange(0, n, dtype=np.float32)
    expected = float(target.expected_reward_bars)
    if expected > 0:
        bar_count = np.maximum(0.0, 1.0 - np.abs(counts - expected) / expected)
        previous_bar_count = np.maximum(0.0, 1.0 - np.abs(previous_counts - expected) / expected)
        line_rewards += reward_config.bar_count_weight * (bar_count - previous_bar_count)

    return [float(item) for item in line_rewards]


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
        final_score = score_total_reward(
            prompt_text=prompt_text,
            completion_text=completion_text,
            target=target,
            reward_config=reward_config,
            candidate_name=f"{candidate_name}_final",
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
        return PatchRewardTrace(rewards=[], prefix_totals=[], final_score=final_score)

    rewards = [0.0 for _ in generated_patches]
    line_rewards = _single_pass_line_rewards(
        full_text=prompt_text + completion_text,
        target=target,
        reward_config=reward_config,
    )
    line_end_patch_indices = _stream_line_end_patch_indices(completion_text, patch_texts)
    for line_reward, patch_idx in zip(line_rewards, line_end_patch_indices, strict=False):
        rewards[patch_idx] += line_reward

    terminal_residual = final_score.total - sum(rewards)
    rewards[-1] += terminal_residual
    prefix_totals: list[float] = []
    running = 0.0
    for reward in rewards:
        running += reward
        prefix_totals.append(running)
    return PatchRewardTrace(rewards=rewards, prefix_totals=prefix_totals, final_score=final_score)


def patch_rewards_from_prefix_deltas(**kwargs) -> PatchRewardTrace:
    return patch_rewards_single_pass(**kwargs)


def normalize_advantages(advantages: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = advantages.mean()
    std = advantages.std(unbiased=False)
    if advantages.numel() <= 1 or std <= eps:
        return advantages - mean, mean, std
    return (advantages - mean) / (std + eps), mean, std


def ppo_clipped_loss(
    *,
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    clip_range: float,
    value_loss_coef: float,
    entropy_bonus_coef: float = 0.0,
    normalize_advantage: bool = True,
) -> PPOLossPayload:
    if not (new_logprobs.shape == old_logprobs.shape == values.shape == old_values.shape == returns.shape):
        raise RuntimeError(
            "PPO tensor shape mismatch: "
            f"new={tuple(new_logprobs.shape)} old={tuple(old_logprobs.shape)} "
            f"values={tuple(values.shape)} old_values={tuple(old_values.shape)} returns={tuple(returns.shape)}"
        )
    raw_advantages = returns - old_values.detach()
    if normalize_advantage:
        advantages, adv_mean, adv_std = normalize_advantages(raw_advantages)
    else:
        advantages = raw_advantages
        adv_mean = raw_advantages.mean()
        adv_std = raw_advantages.std(unbiased=False)

    log_ratio = new_logprobs - old_logprobs.detach()
    ratio = torch.exp(log_ratio)
    unclipped = ratio * advantages.detach()
    clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages.detach()
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    value_loss = torch.nn.functional.mse_loss(values.float(), returns.float())
    entropy_loss = -entropy_bonus_coef * (-new_logprobs).mean()
    loss = policy_loss + value_loss_coef * value_loss + entropy_loss
    approx_kl = ((old_logprobs.detach() - new_logprobs) ** 2).mean() * 0.5
    clip_fraction = ((ratio - 1.0).abs() > clip_range).float().mean()
    return PPOLossPayload(
        loss=loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy_loss=entropy_loss,
        approx_kl=approx_kl,
        clip_fraction=clip_fraction,
        advantages_mean=adv_mean,
        advantages_std=adv_std,
    )


def run_ppo_smoke(
    policy_model: NotaGenLMHeadModel,
    policy_shape: ModelShape,
    value_head: PatchValueHead,
    prompts: list[dict],
    target,
    reward_config: GoldbergRewardConfig,
    args,
) -> dict:
    patchilizer = Patchilizer(stream=PATCH_STREAM)
    device = next(policy_model.parameters()).device
    optimizer = torch.optim.AdamW(
        [
            {"params": [param for param in policy_model.parameters() if param.requires_grad], "lr": args.learning_rate},
            {"params": value_head.parameters(), "lr": args.value_learning_rate},
        ]
    )
    policy_model.eval()
    value_head.train()
    dropout_modules_disabled = disable_dropout_modules(policy_model)

    similarity_weights = SimilarityRewardWeights(
        aria_chroma=args.aria_chroma_reward_weight,
        aria_harmony=args.aria_harmony_reward_weight,
    )
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

    logs: list[dict] = []
    for local_step_idx in range(1, args.max_steps + 1):
        step_start = time.perf_counter()
        timings: dict[str, float] = {}
        step_idx = args.step_offset + local_step_idx
        prompt_idx = (step_idx - 1) % len(prompts)
        row = prompts[prompt_idx]
        prompt_name = prompt_row_name(row, prompt_idx)
        prompt = row["prompt"]

        rollout_start = time.perf_counter()
        rollout_seed = _rollout_seed(args.seed, step_idx, 0, 0)
        set_seed(rollout_seed)
        full_text, generated_patches = sample_completion(
            model=policy_model,
            model_shape=policy_shape,
            prompt=prompt,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            target_stream_lines=args.target_stream_lines,
            max_chars=args.max_chars,
            max_generated_patches=args.max_generated_patches,
            timeout_s=args.timeout_s,
            precision=args.precision,
            cached_rollout=args.cached_rollout,
        )
        timings["rollout_s"] = time.perf_counter() - rollout_start

        reward_start = time.perf_counter()
        reward_trace = patch_rewards_from_prefix_deltas(
            prompt_text=prompt,
            generated_patches=generated_patches,
            target=target,
            reward_config=reward_config,
            candidate_name=f"step{step_idx}_sample0",
            weights=similarity_weights,
            aria_similarity_ref=aria_similarity_ref,
            similarity_chroma_bins=args.similarity_chroma_bins,
            similarity_band_ratio=args.similarity_band_ratio,
            similarity_timeout_s=args.similarity_timeout_s,
            max_similarity_reward=args.max_similarity_reward,
        )
        total_reward = reward_trace.final_score.total
        reward_breakdown = reward_trace.final_score.breakdown
        reward_breakdown["generated_patches"] = len(generated_patches)
        reward_breakdown["generated_token_slots"] = generated_token_slots(generated_patches)
        reward_breakdown["prompt_index"] = prompt_idx
        reward_breakdown["prompt_name"] = prompt_name
        reward_breakdown["rollout_seed"] = rollout_seed
        reward_breakdown["rollout_prefix_stream_lines"] = count_stream_lines(
            build_rollout_prefix(prompt, args.target_stream_lines)
        )
        reward_breakdown["patch_reward_mode"] = "prefix_delta"
        reward_breakdown["patch_reward_count"] = len(reward_trace.rewards)
        reward_breakdown["patch_reward_sum"] = float(sum(reward_trace.rewards))
        timings["reward_s"] = time.perf_counter() - reward_start

        replay_start = time.perf_counter()
        rollout_prompt = build_rollout_prefix(prompt, args.target_stream_lines)
        prompt_flat = [item for sublist in patchilizer.encode_generate(rollout_prompt) for item in sublist]
        with torch.no_grad():
            old_replay = trajectory_patch_logprobs_values(
                policy_model,
                value_head,
                prompt_flat,
                generated_patches,
                args.precision,
                replay_context_patches=args.replay_context_patches,
                target_chunk_patches=args.score_chunk_patches,
            )
        if old_replay.logprobs.numel() == 0:
            raise RuntimeError("PPO rollout produced no scorable patches")
        if len(reward_trace.rewards) != old_replay.logprobs.numel():
            raise RuntimeError(
                "PPO patch reward/logprob count mismatch: "
                f"rewards={len(reward_trace.rewards)} logprobs={old_replay.logprobs.numel()}"
            )
        patch_rewards = torch.tensor(reward_trace.rewards, device=device, dtype=torch.float32)
        returns = discounted_returns(patch_rewards, args.gamma)

        optimizer.zero_grad(set_to_none=True)
        new_replay = trajectory_patch_logprobs_values(
            policy_model,
            value_head,
            prompt_flat,
            generated_patches,
            args.precision,
            replay_context_patches=args.replay_context_patches,
            target_chunk_patches=args.score_chunk_patches,
        )
        loss_payload = ppo_clipped_loss(
            new_logprobs=new_replay.logprobs.float(),
            old_logprobs=old_replay.logprobs.detach().float(),
            values=new_replay.values.float(),
            old_values=old_replay.values.detach().float(),
            returns=returns.float(),
            clip_range=args.ppo_clip_range,
            value_loss_coef=args.value_loss_coef,
            entropy_bonus_coef=args.entropy_bonus_coef,
            normalize_advantage=not args.no_advantage_normalization,
        )
        if not args.no_step:
            loss_payload.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [param for param in list(policy_model.parameters()) + list(value_head.parameters()) if param.requires_grad],
                args.max_grad_norm,
            )
            optimizer.step()
        timings["ppo_replay_backward_s"] = time.perf_counter() - replay_start
        timings["total_step_s"] = time.perf_counter() - step_start

        step_log = {
            "step": step_idx,
            "prompt_index": prompt_idx,
            "prompt_name": prompt_name,
            "loss": float(loss_payload.loss.detach().cpu()),
            "policy_loss": float(loss_payload.policy_loss.detach().cpu()),
            "value_loss": float(loss_payload.value_loss.detach().cpu()),
            "entropy_loss": float(loss_payload.entropy_loss.detach().cpu()),
            "approx_kl": float(loss_payload.approx_kl.detach().cpu()),
            "clip_fraction": float(loss_payload.clip_fraction.detach().cpu()),
            "advantages_mean": float(loss_payload.advantages_mean.detach().cpu()),
            "advantages_std": float(loss_payload.advantages_std.detach().cpu()),
            "return_mean": float(returns.mean().detach().cpu()),
            "return_std": float(returns.std(unbiased=False).detach().cpu()),
            "patch_reward_mean": float(patch_rewards.mean().detach().cpu()),
            "patch_reward_std": float(patch_rewards.std(unbiased=False).detach().cpu()),
            "patch_rewards": reward_trace.rewards,
            "patch_reward_prefix_totals": reward_trace.prefix_totals,
            "value_mean": float(new_replay.values.mean().detach().cpu()),
            "value_std": float(new_replay.values.std(unbiased=False).detach().cpu()),
            "scored_patches": int(new_replay.logprobs.numel()),
            "reward": total_reward,
            "reward_breakdown": reward_breakdown,
            "timings": timings,
        }
        print(json.dumps({"event": "ppo_step_complete", **step_log}), flush=True)
        logs.append(step_log)

    return {
        "steps": logs,
        "policy_dropout_modules_disabled": dropout_modules_disabled,
        "value_head": {
            "hidden_size": value_head.proj.in_features,
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
    parser.add_argument("--aria-chroma-reward-weight", type=float, default=0.0)
    parser.add_argument("--aria-harmony-reward-weight", type=float, default=0.0)
    parser.add_argument("--max-similarity-reward", type=float, default=1.0)
    parser.add_argument("--similarity-chroma-bins", type=int, default=128)
    parser.add_argument("--similarity-band-ratio", type=float, default=0.25)
    parser.add_argument("--similarity-timeout-s", type=float, default=20.0)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--prompt-limit", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--target-stream-lines", type=int, default=32)
    parser.add_argument("--max-chars", type=int, default=40000)
    parser.add_argument("--max-generated-patches", type=int, default=256)
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--value-learning-rate", type=float, default=1e-5)
    parser.add_argument("--ppo-clip-range", type=float, default=0.2)
    parser.add_argument("--value-loss-coef", type=float, default=0.5)
    parser.add_argument("--entropy-bonus-coef", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--replay-context-patches", type=int, default=128)
    parser.add_argument("--score-chunk-patches", type=int, default=16)
    parser.add_argument("--lora-r", type=int, default=0)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step-offset", type=int, default=0)
    parser.add_argument("--cached-rollout", action="store_true")
    parser.add_argument("--no-step", action="store_true")
    parser.add_argument("--no-advantage-normalization", action="store_true")
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
    value_head = PatchValueHead(policy_shape.hidden_size).to(device)
    prompts = load_prompt_rows(args.prompts_jsonl, limit=args.prompt_limit)
    target = load_structural_target(args.target_json, structure_path=args.target_structure_abc)
    reward_config = GoldbergRewardConfig()
    payload = run_ppo_smoke(
        policy_model=policy_model,
        policy_shape=policy_shape,
        value_head=value_head,
        prompts=prompts,
        target=target,
        reward_config=reward_config,
        args=args,
    )
    payload["run_config"] = {
        "args": vars(args),
        "policy_shape": asdict(policy_shape),
        "reward_config": asdict(reward_config),
        "policy_weights": str(policy_weights),
        "ppo": {
            "clip_range": args.ppo_clip_range,
            "value_loss_coef": args.value_loss_coef,
            "entropy_bonus_coef": args.entropy_bonus_coef,
            "gamma": args.gamma,
            "reward_assignment": "prefix_delta_same_reward_logic_per_generated_patch",
        },
    }
    Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
