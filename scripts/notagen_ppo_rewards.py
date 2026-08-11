from __future__ import annotations

import bisect
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from rewards.harmony_similarity import (
    generic_dtw_alignment,
    infer_harmony,
    parse_bar_notes,
    pitch_class_similarity,
    token_similarity,
)
from rewards.rewards import (
    GoldbergRewardConfig,
    _extract_header_context,
    _extract_stream_line_features,
    _stream_line_local_metrics,
    score_candidate_text_with_local_metrics,
    score_prompt_completion_pair,
)
from rewards.similarity_rewards import (
    SimilarityReference,
    SimilarityRewardWeights,
    finalize_similarity_reward_fields,
    score_similarity_reward,
)
from rewards.strict_similarity import (
    STRICT_SYMBOLIC_COMPONENT_WEIGHTS,
    STRICT_SYMBOLIC_COMPONENT_Z_KEY,
    written_harmony_reference,
)
from rewards.strict_similarity_baseline_norms import (
    STRICT_SIMILARITY_GLOBAL_NORMS,
    STRICT_SIMILARITY_Z_STD_FLOORS,
)
from scripts.custom_grpo_notagen import PATCH_STREAM, count_stream_lines
from scripts.notagen_ppo_diagnostics import component_prefix_totals, prefix_totals
from utils import Patchilizer


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
    reward_breakdown.update(
        finalize_similarity_reward_fields(
            similarity_payload=similarity_payload,
            structural_total_reward=structural_total_reward,
            completion_reward=reward_breakdown.get("completion_reward", 0.0),
            bar_count_reward=reward_breakdown.get("bar_count_reward", 0.0),
            max_similarity_reward=max_similarity_reward,
        )
    )
    return RewardScore(total=reward_breakdown["total_reward"], breakdown=reward_breakdown)


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


def _rollout_sampling_summary(rollout_payloads: list[Any]) -> dict:
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


def _active_similarity_component(raw_component: float, final_score: RewardScore) -> float:
    breakdown = final_score.breakdown
    raw_total = float(breakdown.get("raw_similarity_reward", 0.0))
    if raw_total == 0.0 or raw_component == 0.0:
        return 0.0
    clipped_total = float(breakdown.get("clipped_similarity_reward", raw_total))
    return raw_component * (clipped_total / raw_total)


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


def _same_bar_metric_reward_events(
    *,
    name: str,
    reference: list,
    candidate: list,
    candidate_spans: list[tuple[int, int]],
    similarity_fn,
    total_value: float,
) -> list[RewardEvent]:
    if total_value == 0.0 or not reference or not candidate or not candidate_spans:
        return []
    credits = [
        max(0.0, float(similarity_fn(left, right)))
        for left, right in zip(reference, candidate, strict=False)
    ]
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
        for credit, (start, end) in zip(credits, candidate_spans, strict=False)
        if credit > 0.0 and end > start
    ]


def _strict_metric_norm(metric_name: str) -> tuple[float, float]:
    norm = STRICT_SIMILARITY_GLOBAL_NORMS.get(metric_name)
    if norm is None:
        raise RuntimeError(f"missing strict similarity baseline norm for {metric_name!r}")
    std_floor = float(STRICT_SIMILARITY_Z_STD_FLOORS.get(metric_name, 1e-4))
    std = max(float(norm.get("std_safe", norm.get("std", 0.0))), std_floor)
    if std <= 0.0:
        raise RuntimeError(f"invalid strict similarity baseline std for {metric_name!r}: {std}")
    return float(norm["mean"]), std


def _span_for_candidate_index(candidate_spans: list[tuple[int, int]], candidate_idx: int) -> tuple[int, int] | None:
    if not candidate_spans:
        return None
    if 0 <= candidate_idx < len(candidate_spans):
        return candidate_spans[candidate_idx]
    return candidate_spans[-1]


def _terminal_metric_reward_event(
    *,
    name: str,
    candidate_spans: list[tuple[int, int]],
    total_value: float,
) -> list[RewardEvent]:
    if total_value == 0.0 or not candidate_spans:
        return []
    _start, end = candidate_spans[-1]
    start = max(_start, end - 1)
    if end <= start:
        return []
    return [RewardEvent(start=start, end=end, value=total_value, name=name)]


def _correct_event_total(events: list[RewardEvent], total_value: float) -> list[RewardEvent]:
    if not events:
        return events
    residual = total_value - sum(event.value for event in events)
    if abs(residual) <= 1e-12:
        return events
    last = events[-1]
    events[-1] = RewardEvent(
        start=last.start,
        end=last.end,
        value=last.value + residual,
        name=last.name,
    )
    return events


def _signed_local_similarity_reward_events(
    *,
    name: str,
    local_scores: list[tuple[tuple[int, int], float]],
    baseline_mean: float,
    total_value: float,
) -> list[RewardEvent]:
    if total_value == 0.0 or not local_scores:
        return []
    deltas = [float(score) - float(baseline_mean) for _span, score in local_scores]
    total_delta = sum(deltas)
    if abs(total_delta) <= 1e-12:
        values = [total_value / float(len(local_scores)) for _item in local_scores]
    else:
        values = [total_value * (delta / total_delta) for delta in deltas]

    events = [
        RewardEvent(start=start, end=end, value=float(value), name=name)
        for ((start, end), _score), value in zip(local_scores, values, strict=True)
        if end > start and value != 0.0
    ]
    return _correct_event_total(events, total_value)


def _signed_same_bar_metric_reward_events(
    *,
    name: str,
    reference: list,
    candidate: list,
    candidate_spans: list[tuple[int, int]],
    similarity_fn,
    metric_name: str,
    total_value: float,
) -> list[RewardEvent]:
    if total_value == 0.0 or not reference or not candidate_spans:
        return []
    baseline_mean, _std = _strict_metric_norm(metric_name)
    local_scores: list[tuple[tuple[int, int], float]] = []
    for idx, left in enumerate(reference):
        span = _span_for_candidate_index(candidate_spans, idx)
        if span is None:
            continue
        right = candidate[idx] if idx < len(candidate) else None
        score = 0.0 if right is None else max(0.0, float(similarity_fn(left, right)))
        local_scores.append((span, score))
    return _signed_local_similarity_reward_events(
        name=name,
        local_scores=local_scores,
        baseline_mean=baseline_mean,
        total_value=total_value,
    )


def _signed_dtw_metric_reward_events(
    *,
    name: str,
    reference: list,
    candidate: list,
    candidate_spans: list[tuple[int, int]],
    similarity_fn,
    metric_name: str,
    total_value: float,
    band_ratio: float,
) -> list[RewardEvent]:
    if total_value == 0.0 or not reference or not candidate or not candidate_spans:
        return []
    alignment = generic_dtw_alignment(reference, candidate, similarity_fn, band_ratio=band_ratio)
    if not alignment.path:
        return []
    baseline_mean, _std = _strict_metric_norm(metric_name)
    local_scores: list[tuple[tuple[int, int], float]] = []
    for (_ref_idx, candidate_idx), local_similarity in zip(
        alignment.path,
        alignment.local_similarities,
        strict=True,
    ):
        span = _span_for_candidate_index(candidate_spans, candidate_idx)
        if span is None:
            continue
        local_scores.append((span, max(0.0, float(local_similarity))))
    return _signed_local_similarity_reward_events(
        name=name,
        local_scores=local_scores,
        baseline_mean=baseline_mean,
        total_value=total_value,
    )


def _soft_root_bass_similarity(left: dict, right: dict) -> float:
    return 0.5 * pitch_class_similarity(left.get("root"), right.get("root")) + 0.5 * pitch_class_similarity(
        left.get("bass"),
        right.get("bass"),
    )


def _cadence_positions(reference_length: int) -> list[int]:
    if reference_length <= 0:
        return []
    positions = {reference_length - 1}
    for divisor in (4, 2):
        step = max(1, reference_length // divisor)
        for pos in range(step - 1, reference_length, step):
            positions.add(pos)
    return sorted(pos for pos in positions if 0 <= pos < reference_length)


def _strict_symbolic_component_value(
    *,
    metric_name: str,
    final_score: RewardScore,
    similarity_weights: SimilarityRewardWeights,
) -> float:
    metric_weight = float(STRICT_SYMBOLIC_COMPONENT_WEIGHTS.get(metric_name, 0.0))
    if metric_weight == 0.0 or similarity_weights.aria_strict_symbolic == 0.0:
        return 0.0
    raw_component = (
        float(similarity_weights.aria_strict_symbolic)
        * metric_weight
        * float(final_score.breakdown.get(f"aria_{metric_name}_global_base_z", 0.0))
    )
    return _active_similarity_component(raw_component, final_score)


def _strict_symbolic_dtw_subcomponent_value(
    *,
    metric_score: float,
    final_score: RewardScore,
    similarity_weights: SimilarityRewardWeights,
) -> float:
    metric_weight = float(STRICT_SYMBOLIC_COMPONENT_WEIGHTS.get("strict_dtw_combined_narrow", 0.0))
    if metric_weight == 0.0 or similarity_weights.aria_strict_symbolic == 0.0:
        return 0.0
    baseline_mean, baseline_std = _strict_metric_norm("strict_dtw_combined_narrow")
    raw_component = (
        float(similarity_weights.aria_strict_symbolic)
        * metric_weight
        * ((float(metric_score) / 3.0 - baseline_mean / 3.0) / baseline_std)
    )
    return _active_similarity_component(raw_component, final_score)


def _strict_symbolic_reward_events(
    *,
    reference_harmony: list[dict],
    candidate_harmony: list[dict],
    candidate_spans: list[tuple[int, int]],
    similarity_weights: SimilarityRewardWeights,
    final_score: RewardScore,
    band_ratio: float,
) -> list[RewardEvent]:
    if similarity_weights.aria_strict_symbolic == 0.0 or not reference_harmony or not candidate_harmony:
        return []

    events: list[RewardEvent] = []

    aligned_value = _strict_symbolic_component_value(
        metric_name="strict_aligned_root_bass",
        final_score=final_score,
        similarity_weights=similarity_weights,
    )
    events.extend(
        _signed_same_bar_metric_reward_events(
            name="aria_strict_aligned_root_bass_active",
            reference=reference_harmony,
            candidate=candidate_harmony,
            candidate_spans=candidate_spans,
            similarity_fn=_soft_root_bass_similarity,
            metric_name="strict_aligned_root_bass",
            total_value=aligned_value,
        )
    )

    dtw_metric_specs = [
        (
            "aria_strict_harmony_dtw_narrow_active",
            "aria_strict_harmony_dtw_narrow",
            reference_harmony,
            candidate_harmony,
            token_similarity,
        ),
        (
            "aria_strict_root_dtw_narrow_active",
            "aria_strict_root_dtw_narrow",
            [item.get("root") for item in reference_harmony],
            [item.get("root") for item in candidate_harmony],
            pitch_class_similarity,
        ),
        (
            "aria_strict_bass_dtw_narrow_active",
            "aria_strict_bass_dtw_narrow",
            [item.get("bass") for item in reference_harmony],
            [item.get("bass") for item in candidate_harmony],
            pitch_class_similarity,
        ),
    ]
    for name, score_key, reference, candidate, similarity_fn in dtw_metric_specs:
        metric_score = float(final_score.breakdown.get(score_key, 0.0))
        events.extend(
            _signed_dtw_metric_reward_events(
                name=name,
                reference=reference,
                candidate=candidate,
                candidate_spans=candidate_spans,
                similarity_fn=similarity_fn,
                metric_name="strict_dtw_combined_narrow",
                total_value=_strict_symbolic_dtw_subcomponent_value(
                    metric_score=metric_score,
                    final_score=final_score,
                    similarity_weights=similarity_weights,
                ),
                band_ratio=band_ratio,
            )
        )

    for metric_name in (
        "strict_root_bass_bigram_weighted_jaccard",
        "strict_root_bass_fourgram_weighted_jaccard",
    ):
        events.extend(
            _terminal_metric_reward_event(
                name=f"aria_{metric_name}_active",
                candidate_spans=candidate_spans,
                total_value=_strict_symbolic_component_value(
                    metric_name=metric_name,
                    final_score=final_score,
                    similarity_weights=similarity_weights,
                ),
            )
        )

    events.extend(
        _signed_same_bar_metric_reward_events(
            name="aria_strict_cadence_root_bass_active",
            reference=[reference_harmony[pos] for pos in _cadence_positions(len(reference_harmony))],
            candidate=[
                candidate_harmony[pos] if pos < len(candidate_harmony) else {}
                for pos in _cadence_positions(len(reference_harmony))
            ],
            candidate_spans=[
                _span_for_candidate_index(candidate_spans, pos) or candidate_spans[-1]
                for pos in _cadence_positions(len(reference_harmony))
            ],
            similarity_fn=_soft_root_bass_similarity,
            metric_name="strict_cadence_root_bass",
            total_value=_strict_symbolic_component_value(
                metric_name="strict_cadence_root_bass",
                final_score=final_score,
                similarity_weights=similarity_weights,
            ),
        )
    )
    return events


def _harmony_reward_events(
    *,
    completion_text: str,
    similarity_weights: SimilarityRewardWeights,
    aria_similarity_ref: SimilarityReference | None,
    final_score: RewardScore,
    band_ratio: float,
) -> list[RewardEvent]:
    if (
        not similarity_weights.needs_harmony
        or aria_similarity_ref is None
        or aria_similarity_ref.harmony is None
        or not final_score.breakdown.get("similarity_harmony_valid")
    ):
        return []

    candidate_harmony, candidate_spans = _completion_harmony_tokens(completion_text)
    if not candidate_harmony:
        return []

    reference_harmony = written_harmony_reference(aria_similarity_ref.harmony)
    events: list[RewardEvent] = []
    if similarity_weights.aria_harmony != 0.0:
        weight_per_metric = similarity_weights.aria_harmony / 3.0
        metric_specs = [
            (
                "aria_harmony_harmony_dtw",
                reference_harmony,
                candidate_harmony,
                token_similarity,
            ),
            (
                "aria_harmony_root_dtw",
                [item["root"] for item in reference_harmony],
                [item["root"] for item in candidate_harmony],
                pitch_class_similarity,
            ),
            (
                "aria_harmony_bass_dtw",
                [item["bass"] for item in reference_harmony],
                [item["bass"] for item in candidate_harmony],
                pitch_class_similarity,
            ),
        ]

        for metric_name, reference, candidate, similarity_fn in metric_specs:
            metric_score = float(final_score.breakdown.get(metric_name, 0.0))
            total_value = _active_similarity_component(weight_per_metric * metric_score, final_score)
            events.extend(
                _dtw_metric_reward_events(
                    name=f"{metric_name}_active",
                    reference=reference,
                    candidate=candidate,
                    candidate_spans=candidate_spans,
                    similarity_fn=similarity_fn,
                    total_value=total_value,
                    band_ratio=band_ratio,
                )
            )

    aligned_specs = [
        (
            "aria_harmony_aligned_root",
            similarity_weights.aria_harmony_aligned_root,
            [item["root"] for item in reference_harmony],
            [item["root"] for item in candidate_harmony],
            lambda left, right: 1.0 if left is not None and left == right else 0.0,
        ),
        (
            "aria_harmony_aligned_bass",
            similarity_weights.aria_harmony_aligned_bass,
            [item["bass"] for item in reference_harmony],
            [item["bass"] for item in candidate_harmony],
            lambda left, right: 1.0 if left is not None and left == right else 0.0,
        ),
        (
            "aria_harmony_aligned_top",
            similarity_weights.aria_harmony_aligned_top,
            [
                None if item.get("top_midi") is None else int(item["top_midi"]) % 12
                for item in reference_harmony
            ],
            [
                None if item.get("top_midi") is None else int(item["top_midi"]) % 12
                for item in candidate_harmony
            ],
            lambda left, right: 1.0 if left is not None and left == right else 0.0,
        ),
    ]
    for metric_name, weight, reference, candidate, similarity_fn in aligned_specs:
        if weight == 0.0:
            continue
        metric_score = float(final_score.breakdown.get(metric_name, 0.0))
        total_value = _active_similarity_component(weight * metric_score, final_score)
        events.extend(
            _same_bar_metric_reward_events(
                name=f"{metric_name}_active",
                reference=reference,
                candidate=candidate,
                candidate_spans=candidate_spans,
                similarity_fn=similarity_fn,
                total_value=total_value,
            )
        )
    events.extend(
        _strict_symbolic_reward_events(
            reference_harmony=reference_harmony,
            candidate_harmony=candidate_harmony,
            candidate_spans=candidate_spans,
            similarity_weights=similarity_weights,
            final_score=final_score,
            band_ratio=0.05,
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
        "completion_reward": reward_config.completion_weight,
        "expanded_completion_reward": reward_config.expanded_completion_weight,
        "parse_reward": reward_config.parse_weight,
        "syntax_penalty_reward": reward_config.syntax_penalty_weight,
        "termination_penalty_reward": reward_config.termination_penalty_weight,
        "countdown_reward": reward_config.countdown_weight,
        "line_closure_reward": reward_config.line_closure_weight,
        "bar_token_reward": reward_config.bar_token_weight,
        "note_bearing_line_reward": reward_config.note_bearing_line_weight,
        "meter_alignment_reward": reward_config.meter_alignment_weight,
        "meter_duration_closeness_reward": reward_config.meter_duration_closeness_weight,
        "bar_meter_consistency_reward": reward_config.bar_meter_consistency_weight,
        "bar_count_reward": reward_config.bar_count_weight,
        "expanded_bar_count_reward": reward_config.expanded_bar_count_weight,
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
        chroma_component = _active_similarity_component(
            similarity_weights.aria_chroma * float(final_score.breakdown.get("aria_chroma_harmonic_hist", 0.0)),
            final_score,
        )
        if chroma_component != 0.0:
            component_rewards["aria_chroma_harmonic_hist_active"] = _terminal_patch_rewards(
                patch_count,
                chroma_component,
            )
    if similarity_weights.aria_chroma_top != 0.0:
        top_component = _active_similarity_component(
            similarity_weights.aria_chroma_top * float(final_score.breakdown.get("aria_chroma_top_hist", 0.0)),
            final_score,
        )
        if top_component != 0.0:
            component_rewards["aria_chroma_top_hist_active"] = _terminal_patch_rewards(
                patch_count,
                top_component,
            )

    if similarity_weights.aria_harmony != 0.0:
        weight_per_metric = similarity_weights.aria_harmony / 3.0
        for metric_name in (
            "aria_harmony_harmony_dtw",
            "aria_harmony_root_dtw",
            "aria_harmony_bass_dtw",
        ):
            component = _active_similarity_component(
                weight_per_metric * float(final_score.breakdown.get(metric_name, 0.0)),
                final_score,
            )
            if component != 0.0:
                component_rewards[f"{metric_name}_active"] = _terminal_patch_rewards(patch_count, component)
    for metric_name, weight in (
        ("aria_harmony_aligned_root", similarity_weights.aria_harmony_aligned_root),
        ("aria_harmony_aligned_bass", similarity_weights.aria_harmony_aligned_bass),
        ("aria_harmony_aligned_top", similarity_weights.aria_harmony_aligned_top),
        (
            f"aria_{STRICT_SYMBOLIC_COMPONENT_Z_KEY}",
            similarity_weights.aria_strict_symbolic,
        ),
    ):
        if weight == 0.0:
            continue
        component = _active_similarity_component(
            weight * float(final_score.breakdown.get(metric_name, 0.0)),
            final_score,
        )
        if component != 0.0:
            component_rewards[f"{metric_name}_active"] = _terminal_patch_rewards(patch_count, component)
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
            "active_similarity_reward": 0.0,
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
        "active_similarity_reward": 0.0,
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
    note_bearing = np.array(local_metrics.note_bearing_line_reward, dtype=np.float32)
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
    add_weighted_component("note_bearing_line_reward", reward_config.note_bearing_line_weight, note_bearing)
    add_weighted_component("meter_alignment_reward", reward_config.meter_alignment_weight, meter_alignment)
    add_weighted_component(
        "meter_duration_closeness_reward",
        reward_config.meter_duration_closeness_weight,
        meter_duration,
    )
    add_weighted_component("bar_meter_consistency_reward", reward_config.bar_meter_consistency_weight, bar_meter)
    add_weighted_component("voice_declaration_reward", reward_config.voice_declaration_weight, voice_decl)
    add_weighted_component("score_voice_reward", reward_config.score_voice_weight, score_voice)

    repeat_expanded_measure_increments = np.array(local_metrics.musical_bar_units, dtype=np.float32)
    written_measure_increments = np.array(local_metrics.written_bar_units, dtype=np.float32)
    written_counts = np.cumsum(written_measure_increments)
    expected = float(target.expected_bars)
    if expected > 0 and reward_config.bar_count_weight != 0.0:
        previous_written_counts = np.concatenate(([0.0], written_counts[:-1])).astype(np.float32)
        bar_count = np.maximum(0.0, 1.0 - np.abs(written_counts - expected) / expected)
        previous_bar_count = np.maximum(0.0, 1.0 - np.abs(previous_written_counts - expected) / expected)
        components["bar_count_reward"] = reward_config.bar_count_weight * (bar_count - previous_bar_count)

    expanded_expected = float(getattr(target, "expected_repeat_expanded_bars", float(target.expected_bars) * 2.0))
    if expanded_expected > 0 and reward_config.expanded_bar_count_weight != 0.0:
        expanded_counts = np.cumsum(repeat_expanded_measure_increments)
        previous_expanded_counts = np.concatenate(([0.0], expanded_counts[:-1])).astype(np.float32)
        expanded_bar_count = np.maximum(0.0, 1.0 - np.abs(expanded_counts - expanded_expected) / expanded_expected)
        previous_expanded_bar_count = np.maximum(
            0.0,
            1.0 - np.abs(previous_expanded_counts - expanded_expected) / expanded_expected,
        )
        components["expanded_bar_count_reward"] = reward_config.expanded_bar_count_weight * (
            expanded_bar_count - previous_expanded_bar_count
        )

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

    terminal_structural_components = {
        "completion_reward": reward_config.completion_weight,
        "expanded_completion_reward": reward_config.expanded_completion_weight,
        "parse_reward": reward_config.parse_weight,
        "syntax_penalty_reward": reward_config.syntax_penalty_weight,
        "termination_penalty_reward": reward_config.termination_penalty_weight,
    }
    for component_name, weight in terminal_structural_components.items():
        if weight == 0.0:
            continue
        component = weight * float(final_score.breakdown.get(component_name, 0.0))
        if component != 0.0:
            component_rewards[component_name] = _terminal_patch_rewards(len(patch_texts), component)

    structural_gate_adjustment = float(final_score.breakdown.get("structural_validity_gate_adjustment", 0.0))
    if structural_gate_adjustment != 0.0:
        component_rewards["structural_validity_gate_adjustment"] = _terminal_patch_rewards(
            len(patch_texts),
            structural_gate_adjustment,
        )

    if similarity_weights.aria_chroma != 0.0:
        chroma_component = _active_similarity_component(
            similarity_weights.aria_chroma * float(final_score.breakdown.get("aria_chroma_harmonic_hist", 0.0)),
            final_score,
        )
        component_rewards["aria_chroma_harmonic_hist_active"] = _terminal_patch_rewards(
            len(patch_texts),
            chroma_component,
        )
    if similarity_weights.aria_chroma_top != 0.0:
        top_component = _active_similarity_component(
            similarity_weights.aria_chroma_top * float(final_score.breakdown.get("aria_chroma_top_hist", 0.0)),
            final_score,
        )
        component_rewards["aria_chroma_top_hist_active"] = _terminal_patch_rewards(
            len(patch_texts),
            top_component,
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
