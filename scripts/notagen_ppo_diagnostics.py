from __future__ import annotations

from typing import Any, Sequence

import torch


def _safe_float(value: torch.Tensor | float | None) -> float | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        if not torch.isfinite(value).item():
            return None
        return float(value.detach().cpu())
    return float(value)


def _trajectory_patch_offsets(lengths: list[int]) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for length in lengths:
        if length < 0:
            raise RuntimeError(f"negative trajectory length: {length}")
        offsets.append((cursor, cursor + length))
        cursor += length
    return offsets


STRUCTURAL_REWARD_COMPONENTS = (
    "parse_reward",
    "countdown_reward",
    "line_closure_reward",
    "bar_token_reward",
    "meter_alignment_reward",
    "meter_duration_closeness_reward",
    "bar_meter_consistency_reward",
    "bar_count_reward",
    "voice_declaration_reward",
    "score_voice_reward",
    "structural_validity_gate_adjustment",
)


HARMONY_REWARD_COMPONENTS = (
    "aria_harmony_harmony_dtw_effective",
    "aria_harmony_root_dtw_effective",
    "aria_harmony_bass_dtw_effective",
)


def prefix_totals(rewards: list[float]) -> list[float]:
    totals: list[float] = []
    running = 0.0
    for reward in rewards:
        running += reward
        totals.append(running)
    return totals


def component_prefix_totals(component_rewards: dict[str, list[float]]) -> dict[str, list[float]]:
    return {name: prefix_totals(rewards) for name, rewards in component_rewards.items()}


def component_reward_sums(component_rewards: dict[str, list[float]]) -> dict[str, float]:
    return {name: float(sum(rewards)) for name, rewards in sorted(component_rewards.items())}


def component_group_sums(component_sums: dict[str, float]) -> dict[str, float]:
    structural_total = sum(component_sums.get(name, 0.0) for name in STRUCTURAL_REWARD_COMPONENTS)
    harmony_total = sum(component_sums.get(name, 0.0) for name in HARMONY_REWARD_COMPONENTS)
    chroma_total = component_sums.get("aria_chroma_harmonic_hist_effective", 0.0)
    residual = component_sums.get("other_residual", 0.0)
    return {
        "structural_total_reward": float(structural_total),
        "aria_chroma_harmonic_hist_effective": float(chroma_total),
        "aria_harmony_dtw_effective": float(harmony_total),
        "effective_similarity_reward": float(chroma_total + harmony_total),
        "other_residual": float(residual),
        "total_reward": float(structural_total + chroma_total + harmony_total + residual),
    }


def _sum_component_vectors(
    component_rewards: dict[str, list[float]],
    names: tuple[str, ...],
    patch_count: int,
) -> list[float]:
    values = [0.0 for _idx in range(patch_count)]
    for name in names:
        rewards = component_rewards.get(name)
        if rewards is None:
            continue
        if len(rewards) != patch_count:
            raise RuntimeError(f"component reward length mismatch for {name}: {len(rewards)} != {patch_count}")
        for idx, reward in enumerate(rewards):
            values[idx] += float(reward)
    return values


def component_group_rewards(component_rewards: dict[str, list[float]], patch_count: int) -> dict[str, list[float]]:
    structural_total = _sum_component_vectors(component_rewards, STRUCTURAL_REWARD_COMPONENTS, patch_count)
    harmony_total = _sum_component_vectors(component_rewards, HARMONY_REWARD_COMPONENTS, patch_count)
    chroma = list(component_rewards.get("aria_chroma_harmonic_hist_effective", [0.0 for _idx in range(patch_count)]))
    residual = list(component_rewards.get("other_residual", [0.0 for _idx in range(patch_count)]))
    effective_similarity = [
        float(chroma_value + harmony_value)
        for chroma_value, harmony_value in zip(chroma, harmony_total, strict=True)
    ]
    total = [
        float(structural_value + similarity_value + residual_value)
        for structural_value, similarity_value, residual_value in zip(
            structural_total,
            effective_similarity,
            residual,
            strict=True,
        )
    ]
    return {
        "structural_total_reward": structural_total,
        "aria_harmony_dtw_effective": harmony_total,
        "effective_similarity_reward": effective_similarity,
        "total_reward": total,
    }


def aggregate_component_sums(reward_traces: Sequence[Any]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for trace in reward_traces:
        for name, rewards in trace.component_rewards.items():
            totals[name] = totals.get(name, 0.0) + float(sum(rewards))
    return dict(sorted(totals.items()))


def component_reward_tensors(
    reward_traces: Sequence[Any],
    *,
    device: torch.device,
    include_groups: bool = True,
) -> dict[str, torch.Tensor]:
    component_names = sorted(
        {
            name
            for trace in reward_traces
            for name in trace.component_rewards
        }
    )
    tensors: dict[str, torch.Tensor] = {}
    for name in component_names:
        flat_rewards: list[float] = []
        for trace in reward_traces:
            rewards = trace.component_rewards.get(name)
            if rewards is None:
                rewards = [0.0 for _idx in trace.rewards]
            if len(rewards) != len(trace.rewards):
                raise RuntimeError(
                    f"component reward length mismatch for {name}: "
                    f"{len(rewards)} != {len(trace.rewards)}"
                )
            flat_rewards.extend(float(reward) for reward in rewards)
        tensors[name] = torch.tensor(flat_rewards, device=device, dtype=torch.float32)
    if include_groups:
        group_names = sorted(
            {
                name
                for trace in reward_traces
                for name in component_group_rewards(trace.component_rewards, len(trace.rewards))
            }
        )
        for name in group_names:
            flat_rewards = []
            for trace in reward_traces:
                groups = component_group_rewards(trace.component_rewards, len(trace.rewards))
                rewards = groups.get(name, [0.0 for _idx in trace.rewards])
                flat_rewards.extend(float(reward) for reward in rewards)
            tensors[name] = torch.tensor(flat_rewards, device=device, dtype=torch.float32)
    return tensors


def _discounted_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    returns = torch.empty_like(rewards, dtype=torch.float32)
    running = torch.zeros((), device=rewards.device, dtype=torch.float32)
    discount = torch.tensor(float(gamma), device=rewards.device, dtype=torch.float32)
    for idx in range(rewards.numel() - 1, -1, -1):
        running = rewards[idx].float() + discount * running
        returns[idx] = running
    return returns


def component_lambda_return_tensors(
    reward_traces: Sequence[Any],
    *,
    gamma: float,
    gae_lambda: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    discount = gamma * gae_lambda
    raw_tensors = component_reward_tensors(reward_traces, device=device)
    if not raw_tensors:
        return {}
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for trace in reward_traces:
        offsets.append((cursor, cursor + len(trace.rewards)))
        cursor += len(trace.rewards)

    lambda_returns: dict[str, torch.Tensor] = {}
    for name, tensor in raw_tensors.items():
        chunks = [_discounted_returns(tensor[start:end], discount) for start, end in offsets]
        lambda_returns[name] = torch.cat(chunks) if chunks else torch.empty(0, device=device)
    return lambda_returns


def value_prediction_metrics(
    values: torch.Tensor,
    targets: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> dict:
    values_f = values.detach().float().reshape(-1)
    targets_f = targets.detach().float().reshape(-1)
    if values_f.shape != targets_f.shape:
        raise RuntimeError(
            f"value metric shape mismatch: values={tuple(values_f.shape)} targets={tuple(targets_f.shape)}"
        )
    if values_f.numel() == 0:
        return {
            "count": 0,
            "mse": None,
            "mae": None,
            "bias": None,
            "explained_variance": None,
            "correlation": None,
            "value_mean": None,
            "value_std": None,
            "target_mean": None,
            "target_std": None,
            "residual_mean": None,
            "residual_std": None,
        }

    residual = targets_f - values_f
    mse = torch.mean(residual.square())
    mae = torch.mean(residual.abs())
    bias = torch.mean(values_f - targets_f)
    value_std = values_f.std(unbiased=False)
    target_std = targets_f.std(unbiased=False)
    residual_std = residual.std(unbiased=False)
    target_var = target_std.square()
    residual_var = residual_std.square()
    explained_variance = None
    correlation = None
    if target_var > eps:
        explained_variance = 1.0 - residual_var / target_var
    if values_f.numel() > 1 and value_std > eps and target_std > eps:
        centered_values = values_f - values_f.mean()
        centered_targets = targets_f - targets_f.mean()
        correlation = torch.mean(centered_values * centered_targets) / (value_std * target_std)

    return {
        "count": int(values_f.numel()),
        "mse": _safe_float(mse),
        "mae": _safe_float(mae),
        "bias": _safe_float(bias),
        "explained_variance": _safe_float(explained_variance),
        "correlation": _safe_float(correlation),
        "value_mean": _safe_float(values_f.mean()),
        "value_std": _safe_float(value_std),
        "target_mean": _safe_float(targets_f.mean()),
        "target_std": _safe_float(target_std),
        "residual_mean": _safe_float(residual.mean()),
        "residual_std": _safe_float(residual_std),
    }


def tensor_distribution_summary(values: torch.Tensor) -> dict:
    values_f = values.detach().float().reshape(-1).cpu()
    values_f = values_f[torch.isfinite(values_f)]
    if values_f.numel() == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p05": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "max": None,
        }

    quantiles = torch.quantile(
        values_f,
        torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95], dtype=torch.float32),
    )
    return {
        "count": int(values_f.numel()),
        "mean": _safe_float(values_f.mean()),
        "std": _safe_float(values_f.std(unbiased=False)),
        "min": _safe_float(values_f.min()),
        "p05": _safe_float(quantiles[0]),
        "p25": _safe_float(quantiles[1]),
        "p50": _safe_float(quantiles[2]),
        "p75": _safe_float(quantiles[3]),
        "p95": _safe_float(quantiles[4]),
        "max": _safe_float(values_f.max()),
    }


def advantage_distribution_summary(
    raw_advantages: torch.Tensor,
    normalized_advantages: torch.Tensor,
    *,
    trajectory_lengths: list[int] | None = None,
) -> dict:
    raw_advantages_f = raw_advantages.detach().float().reshape(-1)
    normalized_advantages_f = normalized_advantages.detach().float().reshape(-1)
    if raw_advantages_f.shape != normalized_advantages_f.shape:
        raise RuntimeError(
            "advantage summary shape mismatch: "
            f"raw={tuple(raw_advantages_f.shape)} normalized={tuple(normalized_advantages_f.shape)}"
        )

    finite = torch.isfinite(raw_advantages_f) & torch.isfinite(normalized_advantages_f)
    raw_finite = raw_advantages_f[finite]
    normalized_finite = normalized_advantages_f[finite]
    positive = raw_finite > 0
    negative = raw_finite < 0
    zero = raw_finite == 0
    summary = {
        "raw": tensor_distribution_summary(raw_finite),
        "normalized": tensor_distribution_summary(normalized_finite),
        "positive_fraction": masked_tensor_mean(positive.float(), torch.ones_like(positive, dtype=torch.bool)),
        "negative_fraction": masked_tensor_mean(negative.float(), torch.ones_like(negative, dtype=torch.bool)),
        "zero_fraction": masked_tensor_mean(zero.float(), torch.ones_like(zero, dtype=torch.bool)),
        "positive_mean": masked_tensor_mean(raw_finite, positive),
        "negative_mean": masked_tensor_mean(raw_finite, negative),
        "abs_mean": _safe_float(raw_finite.abs().mean()) if raw_finite.numel() else None,
    }
    if trajectory_lengths is None:
        return summary

    total_patches = int(sum(trajectory_lengths))
    if raw_advantages_f.numel() != total_patches:
        raise RuntimeError(
            "advantage trajectory summary length mismatch: "
            f"advantages={raw_advantages_f.numel()} trajectory_patches={total_patches}"
        )

    raw_means: list[torch.Tensor] = []
    raw_sums: list[torch.Tensor] = []
    normalized_means: list[torch.Tensor] = []
    normalized_sums: list[torch.Tensor] = []
    for start, end in _trajectory_patch_offsets(trajectory_lengths):
        raw_slice = raw_advantages_f[start:end]
        normalized_slice = normalized_advantages_f[start:end]
        if raw_slice.numel() == 0:
            continue
        raw_means.append(raw_slice.mean())
        raw_sums.append(raw_slice.sum())
        normalized_means.append(normalized_slice.mean())
        normalized_sums.append(normalized_slice.sum())

    if raw_means:
        summary["by_trajectory"] = {
            "raw_mean": tensor_distribution_summary(torch.stack(raw_means)),
            "raw_sum": tensor_distribution_summary(torch.stack(raw_sums)),
            "normalized_mean": tensor_distribution_summary(torch.stack(normalized_means)),
            "normalized_sum": tensor_distribution_summary(torch.stack(normalized_sums)),
        }
    else:
        empty = tensor_distribution_summary(torch.empty(0))
        summary["by_trajectory"] = {
            "raw_mean": empty,
            "raw_sum": empty,
            "normalized_mean": empty,
            "normalized_sum": empty,
        }
    return summary


def tensor_correlation(left: torch.Tensor, right: torch.Tensor, *, eps: float = 1e-8) -> float | None:
    left_f = left.detach().float().reshape(-1)
    right_f = right.detach().float().reshape(-1)
    if left_f.shape != right_f.shape:
        raise RuntimeError(f"correlation shape mismatch: left={tuple(left_f.shape)} right={tuple(right_f.shape)}")
    finite = torch.isfinite(left_f) & torch.isfinite(right_f)
    left_f = left_f[finite]
    right_f = right_f[finite]
    if left_f.numel() <= 1:
        return None
    left_std = left_f.std(unbiased=False)
    right_std = right_f.std(unbiased=False)
    if left_std <= eps or right_std <= eps:
        return None
    centered_left = left_f - left_f.mean()
    centered_right = right_f - right_f.mean()
    return _safe_float(torch.mean(centered_left * centered_right) / (left_std * right_std))


def masked_tensor_mean(values: torch.Tensor, mask: torch.Tensor) -> float | None:
    if values.shape != mask.shape:
        raise RuntimeError(f"masked mean shape mismatch: values={tuple(values.shape)} mask={tuple(mask.shape)}")
    selected = values.detach().float()[mask]
    if selected.numel() == 0:
        return None
    return _safe_float(selected.mean())


def patch_position_tensors(
    trajectory_lengths: list[int],
    *,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    trajectory_indices: list[int] = []
    absolute_positions: list[int] = []
    relative_positions: list[float] = []
    for trajectory_index, length in enumerate(trajectory_lengths):
        if length < 0:
            raise RuntimeError(f"negative trajectory length: {length}")
        denominator = max(1, length - 1)
        for patch_index in range(length):
            trajectory_indices.append(trajectory_index)
            absolute_positions.append(patch_index)
            relative_positions.append(float(patch_index) / float(denominator))
    return (
        torch.tensor(trajectory_indices, dtype=torch.long, device=device),
        torch.tensor(absolute_positions, dtype=torch.long, device=device),
        torch.tensor(relative_positions, dtype=torch.float32, device=device),
    )


def patch_position_diagnostics(
    *,
    raw_advantages: torch.Tensor,
    normalized_advantages: torch.Tensor,
    patch_rewards: torch.Tensor,
    returns: torch.Tensor,
    value_targets: torch.Tensor,
    old_values: torch.Tensor,
    trajectory_lengths: list[int],
    old_logprobs: torch.Tensor | None = None,
    post_step_logprobs: torch.Tensor | None = None,
    position_bins: int = 5,
) -> list[dict]:
    if position_bins <= 0:
        return []
    raw_advantages_f = raw_advantages.detach().float()
    normalized_advantages_f = normalized_advantages.detach().float()
    patch_rewards_f = patch_rewards.detach().float()
    returns_f = returns.detach().float()
    value_targets_f = value_targets.detach().float()
    old_values_f = old_values.detach().float()
    total_patches = int(sum(trajectory_lengths))
    tensors = {
        "raw_advantages": raw_advantages_f,
        "normalized_advantages": normalized_advantages_f,
        "patch_rewards": patch_rewards_f,
        "returns": returns_f,
        "value_targets": value_targets_f,
        "old_values": old_values_f,
    }
    if old_logprobs is not None:
        tensors["old_logprobs"] = old_logprobs.detach().float()
    if post_step_logprobs is not None:
        tensors["post_step_logprobs"] = post_step_logprobs.detach().float()
    for name, tensor in tensors.items():
        if tensor.numel() != total_patches:
            raise RuntimeError(f"position diagnostic tensor length mismatch for {name}: {tensor.numel()} != {total_patches}")

    _, absolute_positions, relative_positions = patch_position_tensors(
        trajectory_lengths,
        device=raw_advantages_f.device,
    )
    old_logprobs_f = tensors.get("old_logprobs")
    post_step_logprobs_f = tensors.get("post_step_logprobs")
    log_ratio = None
    if old_logprobs_f is not None and post_step_logprobs_f is not None:
        log_ratio = post_step_logprobs_f - old_logprobs_f

    bins: list[dict] = []
    positive_advantage = raw_advantages_f > 0
    negative_advantage = raw_advantages_f < 0
    for bin_index in range(position_bins):
        start = float(bin_index) / float(position_bins)
        end = float(bin_index + 1) / float(position_bins)
        if bin_index == position_bins - 1:
            mask = (relative_positions >= start) & (relative_positions <= end)
        else:
            mask = (relative_positions >= start) & (relative_positions < end)
        count = int(mask.sum().detach().cpu())
        row = {
            "bin": bin_index,
            "relative_start": start,
            "relative_end": end,
            "count": count,
            "absolute_patch_position": tensor_distribution_summary(absolute_positions.float()[mask]),
            "raw_advantage": tensor_distribution_summary(raw_advantages_f[mask]),
            "normalized_advantage": tensor_distribution_summary(normalized_advantages_f[mask]),
            "patch_reward": tensor_distribution_summary(patch_rewards_f[mask]),
            "return": tensor_distribution_summary(returns_f[mask]),
            "value_target": tensor_distribution_summary(value_targets_f[mask]),
            "old_value": tensor_distribution_summary(old_values_f[mask]),
            "positive_advantage_fraction": masked_tensor_mean(positive_advantage.float(), mask),
            "negative_advantage_fraction": masked_tensor_mean(negative_advantage.float(), mask),
        }
        if old_logprobs_f is not None:
            row["old_logprob"] = tensor_distribution_summary(old_logprobs_f[mask])
        if log_ratio is not None:
            row.update(
                {
                    "post_step_logprob": tensor_distribution_summary(post_step_logprobs_f[mask]),
                    "post_step_log_ratio": tensor_distribution_summary(log_ratio[mask]),
                    "positive_advantage_positive_log_ratio_fraction": masked_tensor_mean(
                        (log_ratio > 0).float(),
                        mask & positive_advantage,
                    ),
                    "negative_advantage_negative_log_ratio_fraction": masked_tensor_mean(
                        (log_ratio < 0).float(),
                        mask & negative_advantage,
                    ),
                    "advantage_log_ratio_correlation": tensor_correlation(raw_advantages_f[mask], log_ratio[mask]),
                    "sign_alignment_fraction": masked_tensor_mean(
                        ((log_ratio * raw_advantages_f) > 0).float(),
                        mask & (raw_advantages_f != 0),
                    ),
                }
            )
        bins.append(row)
    return bins


def per_patch_diagnostic_records(
    *,
    old_logprobs: torch.Tensor,
    post_step_logprobs: torch.Tensor | None,
    raw_advantages: torch.Tensor,
    normalized_advantages: torch.Tensor,
    patch_rewards: torch.Tensor,
    returns: torch.Tensor,
    value_targets: torch.Tensor,
    old_values: torch.Tensor,
    trajectory_lengths: list[int],
    component_rewards: dict[str, torch.Tensor] | None = None,
    component_lambda_returns: dict[str, torch.Tensor] | None = None,
) -> list[dict]:
    old_logprobs_f = old_logprobs.detach().float().cpu()
    raw_advantages_f = raw_advantages.detach().float().cpu()
    normalized_advantages_f = normalized_advantages.detach().float().cpu()
    patch_rewards_f = patch_rewards.detach().float().cpu()
    returns_f = returns.detach().float().cpu()
    value_targets_f = value_targets.detach().float().cpu()
    old_values_f = old_values.detach().float().cpu()
    post_step_logprobs_f = None if post_step_logprobs is None else post_step_logprobs.detach().float().cpu()
    total_patches = int(sum(trajectory_lengths))
    tensors = {
        "old_logprobs": old_logprobs_f,
        "raw_advantages": raw_advantages_f,
        "normalized_advantages": normalized_advantages_f,
        "patch_rewards": patch_rewards_f,
        "returns": returns_f,
        "value_targets": value_targets_f,
        "old_values": old_values_f,
    }
    if post_step_logprobs_f is not None:
        tensors["post_step_logprobs"] = post_step_logprobs_f
    component_rewards_f = {
        name: tensor.detach().float().cpu()
        for name, tensor in (component_rewards or {}).items()
    }
    component_lambda_returns_f = {
        name: tensor.detach().float().cpu()
        for name, tensor in (component_lambda_returns or {}).items()
    }
    for name, tensor in component_rewards_f.items():
        tensors[f"component_reward:{name}"] = tensor
    for name, tensor in component_lambda_returns_f.items():
        tensors[f"component_lambda_return:{name}"] = tensor
    for name, tensor in tensors.items():
        if tensor.numel() != total_patches:
            raise RuntimeError(f"per-patch diagnostic tensor length mismatch for {name}: {tensor.numel()} != {total_patches}")

    trajectory_indices, absolute_positions, relative_positions = patch_position_tensors(trajectory_lengths)
    records: list[dict] = []
    for patch_flat_index in range(total_patches):
        post_step_logprob = None
        log_ratio = None
        if post_step_logprobs_f is not None:
            post_step_logprob = _safe_float(post_step_logprobs_f[patch_flat_index])
            log_ratio = _safe_float(post_step_logprobs_f[patch_flat_index] - old_logprobs_f[patch_flat_index])
        record = {
            "flat_patch_index": patch_flat_index,
            "trajectory_index": int(trajectory_indices[patch_flat_index].item()),
            "trajectory_patch_index": int(absolute_positions[patch_flat_index].item()),
            "trajectory_relative_position": _safe_float(relative_positions[patch_flat_index]),
            "old_logprob": _safe_float(old_logprobs_f[patch_flat_index]),
            "post_step_logprob": post_step_logprob,
            "post_step_log_ratio": log_ratio,
            "raw_advantage": _safe_float(raw_advantages_f[patch_flat_index]),
            "normalized_advantage": _safe_float(normalized_advantages_f[patch_flat_index]),
            "patch_reward": _safe_float(patch_rewards_f[patch_flat_index]),
            "return": _safe_float(returns_f[patch_flat_index]),
            "value_target": _safe_float(value_targets_f[patch_flat_index]),
            "old_value": _safe_float(old_values_f[patch_flat_index]),
        }
        for name, tensor in component_rewards_f.items():
            record[f"{name}__reward"] = _safe_float(tensor[patch_flat_index])
        for name, tensor in component_lambda_returns_f.items():
            record[f"{name}__lambda_return"] = _safe_float(tensor[patch_flat_index])
        records.append(record)
    return records


def logprob_advantage_diagnostics(
    *,
    old_logprobs: torch.Tensor,
    post_step_logprobs: torch.Tensor | None,
    raw_advantages: torch.Tensor,
    normalized_advantages: torch.Tensor,
    patch_rewards: torch.Tensor,
    returns: torch.Tensor,
    value_targets: torch.Tensor,
    old_values: torch.Tensor,
    trajectory_lengths: list[int],
    trajectory_logs: list[dict],
    clip_range: float,
    position_bins: int = 5,
) -> dict:
    old_logprobs_f = old_logprobs.detach().float()
    raw_advantages_f = raw_advantages.detach().float()
    normalized_advantages_f = normalized_advantages.detach().float()
    patch_rewards_f = patch_rewards.detach().float()
    returns_f = returns.detach().float()
    value_targets_f = value_targets.detach().float()
    old_values_f = old_values.detach().float()
    total_patches = int(sum(trajectory_lengths))
    for name, tensor in (
        ("old_logprobs", old_logprobs_f),
        ("raw_advantages", raw_advantages_f),
        ("normalized_advantages", normalized_advantages_f),
        ("patch_rewards", patch_rewards_f),
        ("returns", returns_f),
        ("value_targets", value_targets_f),
        ("old_values", old_values_f),
    ):
        if tensor.numel() != total_patches:
            raise RuntimeError(f"PPO diagnostic tensor length mismatch for {name}: {tensor.numel()} != {total_patches}")

    diagnostics = {
        "old_logprob": tensor_distribution_summary(old_logprobs_f),
        "raw_advantage": tensor_distribution_summary(raw_advantages_f),
        "normalized_advantage": tensor_distribution_summary(normalized_advantages_f),
        "advantage_summary": advantage_distribution_summary(
            raw_advantages_f,
            normalized_advantages_f,
            trajectory_lengths=trajectory_lengths,
        ),
        "patch_reward": tensor_distribution_summary(patch_rewards_f),
        "return": tensor_distribution_summary(returns_f),
        "value_target": tensor_distribution_summary(value_targets_f),
        "old_value": tensor_distribution_summary(old_values_f),
        "by_relative_patch_position": patch_position_diagnostics(
            old_logprobs=old_logprobs_f,
            post_step_logprobs=post_step_logprobs,
            raw_advantages=raw_advantages_f,
            normalized_advantages=normalized_advantages_f,
            patch_rewards=patch_rewards_f,
            returns=returns_f,
            value_targets=value_targets_f,
            old_values=old_values_f,
            trajectory_lengths=trajectory_lengths,
            position_bins=position_bins,
        ),
    }

    if post_step_logprobs is None:
        diagnostics["post_step_available"] = False
        diagnostics["per_trajectory"] = []
        return diagnostics

    post_step_logprobs_f = post_step_logprobs.detach().float()
    if post_step_logprobs_f.numel() != total_patches:
        raise RuntimeError(
            f"PPO diagnostic post-step logprob length mismatch: {post_step_logprobs_f.numel()} != {total_patches}"
        )
    log_ratio = post_step_logprobs_f - old_logprobs_f
    ratio = torch.exp(log_ratio)
    positive_advantage = raw_advantages_f > 0
    negative_advantage = raw_advantages_f < 0
    nonzero_advantage = raw_advantages_f != 0
    sign_aligned = (log_ratio * raw_advantages_f) > 0
    upper_clipped = ratio > (1.0 + float(clip_range))
    lower_clipped = ratio < (1.0 - float(clip_range))
    any_clipped = upper_clipped | lower_clipped
    ppo_active_clipped = (positive_advantage & upper_clipped) | (negative_advantage & lower_clipped)

    top_k = max(1, int(0.10 * total_patches))
    top_advantage_indices = torch.topk(raw_advantages_f, k=top_k, largest=True).indices
    bottom_advantage_indices = torch.topk(raw_advantages_f, k=top_k, largest=False).indices
    diagnostics.update(
        {
            "post_step_available": True,
            "post_step_logprob": tensor_distribution_summary(post_step_logprobs_f),
            "post_step_log_ratio": tensor_distribution_summary(log_ratio),
            "advantage_counts": {
                "positive": int(positive_advantage.sum().detach().cpu()),
                "negative": int(negative_advantage.sum().detach().cpu()),
                "zero": int((~nonzero_advantage).sum().detach().cpu()),
            },
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
            "log_ratio_mean_top_advantage_decile": _safe_float(log_ratio[top_advantage_indices].mean()),
            "log_ratio_mean_bottom_advantage_decile": _safe_float(log_ratio[bottom_advantage_indices].mean()),
            "advantage_log_ratio_correlation": tensor_correlation(raw_advantages_f, log_ratio),
            "normalized_advantage_log_ratio_correlation": tensor_correlation(normalized_advantages_f, log_ratio),
            "patch_reward_log_ratio_correlation": tensor_correlation(patch_rewards_f, log_ratio),
            "sign_alignment_fraction": masked_tensor_mean(sign_aligned.float(), nonzero_advantage),
            "clip_by_advantage": {
                "clip_range": float(clip_range),
                "ratio_lower_bound": 1.0 - float(clip_range),
                "ratio_upper_bound": 1.0 + float(clip_range),
                "any_clip_fraction": masked_tensor_mean(any_clipped.float(), torch.ones_like(any_clipped, dtype=torch.bool)),
                "upper_clip_fraction": masked_tensor_mean(
                    upper_clipped.float(),
                    torch.ones_like(upper_clipped, dtype=torch.bool),
                ),
                "lower_clip_fraction": masked_tensor_mean(
                    lower_clipped.float(),
                    torch.ones_like(lower_clipped, dtype=torch.bool),
                ),
                "positive_advantage_any_clip_fraction": masked_tensor_mean(any_clipped.float(), positive_advantage),
                "positive_advantage_upper_clip_fraction": masked_tensor_mean(upper_clipped.float(), positive_advantage),
                "positive_advantage_lower_clip_fraction": masked_tensor_mean(lower_clipped.float(), positive_advantage),
                "positive_advantage_active_clip_fraction": masked_tensor_mean(
                    upper_clipped.float(),
                    positive_advantage,
                ),
                "negative_advantage_any_clip_fraction": masked_tensor_mean(any_clipped.float(), negative_advantage),
                "negative_advantage_upper_clip_fraction": masked_tensor_mean(upper_clipped.float(), negative_advantage),
                "negative_advantage_lower_clip_fraction": masked_tensor_mean(lower_clipped.float(), negative_advantage),
                "negative_advantage_active_clip_fraction": masked_tensor_mean(
                    lower_clipped.float(),
                    negative_advantage,
                ),
                "active_clip_fraction_nonzero_advantage": masked_tensor_mean(
                    ppo_active_clipped.float(),
                    nonzero_advantage,
                ),
            },
        }
    )

    offsets = _trajectory_patch_offsets(trajectory_lengths)
    per_trajectory: list[dict] = []
    for trajectory_log, (start, end) in zip(trajectory_logs, offsets, strict=True):
        trajectory_old = old_logprobs_f[start:end]
        trajectory_post = post_step_logprobs_f[start:end]
        trajectory_ratio = log_ratio[start:end]
        trajectory_advantages = raw_advantages_f[start:end]
        trajectory_normalized_advantages = normalized_advantages_f[start:end]
        trajectory_nonzero_advantage = trajectory_advantages != 0
        rollout_length = trajectory_log.get("rollout_length_diagnostics") or trajectory_log.get("reward_breakdown", {})
        per_trajectory.append(
            {
                "trajectory_index": trajectory_log.get("trajectory_index"),
                "reward": trajectory_log.get("reward"),
                "patch_count": int(end - start),
                "stop_reason": rollout_length.get("stop_reason") if isinstance(rollout_length, dict) else None,
                "target_stream_lines_reached": (
                    rollout_length.get("target_stream_lines_reached") if isinstance(rollout_length, dict) else None
                ),
                "completion_stream_lines": (
                    rollout_length.get("completion_stream_lines") if isinstance(rollout_length, dict) else None
                ),
                "missing_stream_lines_to_target": (
                    rollout_length.get("missing_stream_lines_to_target") if isinstance(rollout_length, dict) else None
                ),
                "patches_per_stream_line": (
                    rollout_length.get("patches_per_stream_line") if isinstance(rollout_length, dict) else None
                ),
                "chars_per_stream_line_max": (
                    rollout_length.get("chars_per_stream_line_max") if isinstance(rollout_length, dict) else None
                ),
                "old_logprob_sum": _safe_float(trajectory_old.sum()),
                "post_step_logprob_sum": _safe_float(trajectory_post.sum()),
                "log_ratio_sum": _safe_float(trajectory_ratio.sum()),
                "log_ratio_mean": _safe_float(trajectory_ratio.mean()),
                "log_ratio_std": _safe_float(trajectory_ratio.std(unbiased=False)),
                "raw_advantage_mean": _safe_float(trajectory_advantages.mean()),
                "raw_advantage_std": _safe_float(trajectory_advantages.std(unbiased=False)),
                "normalized_advantage_mean": _safe_float(trajectory_normalized_advantages.mean()),
                "positive_advantage_count": int((trajectory_advantages > 0).sum().detach().cpu()),
                "negative_advantage_count": int((trajectory_advantages < 0).sum().detach().cpu()),
                "positive_advantage_positive_log_ratio_fraction": masked_tensor_mean(
                    (trajectory_ratio > 0).float(),
                    trajectory_advantages > 0,
                ),
                "negative_advantage_negative_log_ratio_fraction": masked_tensor_mean(
                    (trajectory_ratio < 0).float(),
                    trajectory_advantages < 0,
                ),
                "advantage_log_ratio_correlation": tensor_correlation(trajectory_advantages, trajectory_ratio),
                "sign_alignment_fraction": masked_tensor_mean(
                    ((trajectory_ratio * trajectory_advantages) > 0).float(),
                    trajectory_nonzero_advantage,
                ),
            }
        )
    diagnostics["per_trajectory"] = per_trajectory
    return diagnostics
