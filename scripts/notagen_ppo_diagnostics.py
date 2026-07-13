from __future__ import annotations

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
        "patch_reward": tensor_distribution_summary(patch_rewards_f),
        "return": tensor_distribution_summary(returns_f),
        "value_target": tensor_distribution_summary(value_targets_f),
        "old_value": tensor_distribution_summary(old_values_f),
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
        per_trajectory.append(
            {
                "trajectory_index": trajectory_log.get("trajectory_index"),
                "reward": trajectory_log.get("reward"),
                "patch_count": int(end - start),
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
