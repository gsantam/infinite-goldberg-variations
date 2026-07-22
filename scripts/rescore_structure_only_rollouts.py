from __future__ import annotations

import argparse
import bisect
import json
import math
import random
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.rewards import (
    GoldbergRewardConfig,
    _extract_header_context,
    _extract_stream_line_features,
    _stream_line_local_metrics,
    load_structural_target,
    score_candidate_text_with_local_metrics,
)


PATCH_STREAM = True
EOS_TOKEN_ID = 2


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _decode_patch(patch: list[int]) -> str:
    chars: list[str] = []
    for idx in patch:
        if int(idx) == EOS_TOKEN_ID:
            break
        chars.append(chr(int(idx)))
    return "".join(chars)


def _patch_texts(generated_patches: list[list[int]]) -> list[str]:
    return [_decode_patch(patch) for patch in generated_patches]


def _prefix_totals(rewards: list[float]) -> list[float]:
    totals: list[float] = []
    running = 0.0
    for reward in rewards:
        running += float(reward)
        totals.append(running)
    return totals


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
    return [
        (start, end)
        for start, end in zip(starts, starts[1:] + [len(completion_text)], strict=True)
        if end > start
    ]


def _project_line_rewards_to_patches(
    completion_text: str,
    line_rewards: list[float],
    patch_texts: list[str],
) -> list[float]:
    patch_spans = _patch_char_spans(patch_texts)
    rewards = [0.0 for _idx in patch_spans]
    if not patch_spans:
        return rewards

    completion_len = patch_spans[-1][1]
    line_spans = _stream_line_spans(completion_text)
    for (event_start, event_end), value in zip(line_spans, line_rewards, strict=False):
        value = float(value)
        if value == 0.0:
            continue
        start = max(0, min(completion_len, event_start))
        end = max(start, min(completion_len, event_end))
        if end <= start:
            continue

        overlaps: list[tuple[int, int]] = []
        for patch_idx, (patch_start, patch_end) in enumerate(patch_spans):
            overlap = max(0, min(end, patch_end) - max(start, patch_start))
            if overlap > 0:
                overlaps.append((patch_idx, overlap))
        total_overlap = sum(overlap for _idx, overlap in overlaps)
        if total_overlap <= 0:
            continue
        for patch_idx, overlap in overlaps:
            rewards[patch_idx] += value * (overlap / total_overlap)
    return rewards


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
    add_weighted_component("meter_duration_closeness_reward", reward_config.meter_duration_closeness_weight, meter_duration)
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


def _terminal_patch_rewards(patch_count: int, value: float) -> list[float]:
    rewards = [0.0 for _idx in range(patch_count)]
    if rewards and value != 0.0:
        rewards[-1] = float(value)
    return rewards


def _current_structure_patch_rewards(
    *,
    full_text: str,
    completion_text: str,
    generated_patches: list[list[int]],
    target,
    reward_config: GoldbergRewardConfig,
    candidate_name: str,
) -> tuple[list[float], dict, dict[str, float]]:
    patch_texts = _patch_texts(generated_patches)
    structural_score = score_candidate_text_with_local_metrics(
        abc_text=full_text,
        target=target,
        config=reward_config,
        candidate_name=candidate_name,
    )
    breakdown = structural_score.breakdown.to_json()
    final_total = float(structural_score.breakdown.total_reward)
    component_rewards: dict[str, list[float]] = {}
    line_components = _line_reward_components_from_metrics(
        stream_lines=structural_score.stream_lines,
        local_metrics=structural_score.local_metrics,
        target=target,
        reward_config=reward_config,
    )
    for component_name, line_rewards in line_components.items():
        component_rewards[component_name] = _project_line_rewards_to_patches(
            completion_text,
            line_rewards,
            patch_texts,
        )

    if reward_config.parse_weight != 0.0:
        parse_component = reward_config.parse_weight * float(breakdown.get("parse_reward", 0.0))
        component_rewards["parse_reward"] = _terminal_patch_rewards(len(patch_texts), parse_component)

    gate_adjustment = float(breakdown.get("structural_validity_gate_adjustment", 0.0))
    if gate_adjustment != 0.0:
        component_rewards["structural_validity_gate_adjustment"] = _terminal_patch_rewards(
            len(patch_texts),
            gate_adjustment,
        )

    rewards = [
        float(sum(component_rewards[name][idx] for name in component_rewards))
        for idx in range(len(patch_texts))
    ]
    terminal_residual = final_total - sum(rewards)
    if rewards and terminal_residual != 0.0:
        component_rewards["other_residual"] = _terminal_patch_rewards(len(patch_texts), terminal_residual)
        rewards[-1] += terminal_residual
    else:
        component_rewards["other_residual"] = [0.0 for _idx in patch_texts]

    component_sums = {name: float(sum(values)) for name, values in sorted(component_rewards.items())}
    breakdown.update(
        {
            "structural_total_reward": final_total,
            "raw_similarity_reward": 0.0,
            "clipped_similarity_reward": 0.0,
            "similarity_validity_gate": 1.0 if breakdown.get("parse_valid") else 0.0,
            "effective_similarity_reward": 0.0,
            "total_reward": final_total,
            "patch_reward_mode": "structure_only_single_pass_current_reward",
            "patch_reward_count": len(rewards),
            "patch_reward_sum": float(sum(rewards)),
            "patch_reward_component_sums": component_sums,
            "reward_setup": "structure_only_current_reward",
        }
    )
    if not math.isclose(sum(rewards), final_total, rel_tol=0.0, abs_tol=1e-5):
        raise RuntimeError(
            f"{candidate_name}: patch rewards sum to {sum(rewards)} but final reward is {final_total}"
        )
    return rewards, breakdown, component_sums


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _resolve_prompt_path(raw_path: str | Path, *, base_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return base_dir / path


def _load_prompt_targets(
    *,
    prompts: list[dict],
    prompts_jsonl: Path,
    target_json: Path,
    fallback_structure_abc: Path,
) -> list[tuple[Any, str, str]]:
    base_dir = prompts_jsonl.resolve().parent
    cache: dict[str, Any] = {}
    targets: list[tuple[Any, str, str]] = []
    for prompt_idx, row in enumerate(prompts):
        selected_path: Path | None = None
        selected_key = ""
        for key in ("target_structure_abc", "source", "continuation"):
            value = row.get(key)
            if not value:
                continue
            path = _resolve_prompt_path(value, base_dir=base_dir)
            if not path.exists():
                raise FileNotFoundError(
                    f"prompt {prompt_idx} has {key}={value!r}, but {path} does not exist"
                )
            selected_path = path
            selected_key = key
            break
        if selected_path is None:
            selected_path = _resolve_prompt_path(fallback_structure_abc, base_dir=base_dir)
            selected_key = "fallback_target_structure_abc"
        cache_key = str(selected_path.resolve())
        if cache_key not in cache:
            cache[cache_key] = load_structural_target(target_json, structure_path=selected_path)
        targets.append((cache[cache_key], str(selected_path), selected_key))
    return targets


def main() -> int:
    parser = argparse.ArgumentParser(description="Rescore saved PPO rollouts with current structural-only rewards.")
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--prompts-jsonl",
        type=Path,
        default=Path("data/processed/notagen/goldberg_ppo_prompts_e3_header_allvoices.jsonl"),
    )
    parser.add_argument(
        "--target-json",
        type=Path,
        default=Path("data/processed/goldberg/structure/aria_bar_skeleton.json"),
    )
    parser.add_argument(
        "--target-structure-abc",
        type=Path,
        default=Path("data/processed/notagen/goldberg_metadata_only_split2/augmented/G/variation-01_G.abc"),
    )
    parser.add_argument("--music21-parse-timeout-s", type=float, default=5.0)
    parser.add_argument(
        "--parse-reward-weight",
        type=float,
        default=None,
        help="Override the parse validity reward weight. Defaults to the input JSON config or GoldbergRewardConfig.",
    )
    args = parser.parse_args()

    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    prompts = [json.loads(line) for line in args.prompts_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    prompt_targets = _load_prompt_targets(
        prompts=prompts,
        prompts_jsonl=args.prompts_jsonl,
        target_json=args.target_json,
        fallback_structure_abc=args.target_structure_abc,
    )
    old_config = payload.get("run_config", {}).get("reward_config", {})
    config_keys = GoldbergRewardConfig.__dataclass_fields__.keys()
    reward_config = GoldbergRewardConfig(
        **{
            key: value
            for key, value in old_config.items()
            if key in config_keys and key != "music21_parse_timeout_s"
        },
        music21_parse_timeout_s=args.music21_parse_timeout_s,
    )
    if args.parse_reward_weight is not None:
        reward_config.parse_weight = float(args.parse_reward_weight)

    start = time.perf_counter()
    rows: list[dict] = []
    for step in payload.get("steps", []):
        prompt_idx = int(step.get("prompt_index", 0))
        target, target_path, target_source_key = prompt_targets[prompt_idx]
        step_rewards: list[float] = []
        for trajectory in step.get("trajectories", []):
            generated_patches = [
                [int(token) for token in patch]
                for patch in trajectory.get("generated_patches", [])
            ]
            completion_text = str(trajectory.get("completion_text") or "")
            full_text = str(trajectory.get("full_text") or completion_text)
            candidate_name = f"step{int(step['step'])}_trajectory{int(trajectory['trajectory_index'])}"
            patch_rewards, breakdown, component_sums = _current_structure_patch_rewards(
                full_text=full_text,
                completion_text=completion_text,
                generated_patches=generated_patches,
                target=target,
                reward_config=reward_config,
                candidate_name=candidate_name,
            )
            prefix_totals = _prefix_totals(patch_rewards)
            old_reward = trajectory.get("reward")
            trajectory["reward"] = float(breakdown["total_reward"])
            trajectory["patch_rewards"] = patch_rewards
            trajectory["patch_reward_prefix_totals"] = prefix_totals
            trajectory["patch_reward_mean"] = float(np.mean(patch_rewards)) if patch_rewards else 0.0
            trajectory["patch_reward_std"] = float(np.std(patch_rewards)) if patch_rewards else 0.0
            trajectory["reward_breakdown"] = {
                **breakdown,
                "generated_patches": int(len(generated_patches)),
                "generated_token_slots": int(trajectory.get("generated_token_slots", 0)),
                "prompt_index": prompt_idx,
                "prompt_name": step.get("prompt_name"),
                "trajectory_index": int(trajectory["trajectory_index"]),
                "rollout_seed": trajectory.get("rollout_seed"),
                "target_structure_path": target_path,
                "target_structure_source_key": target_source_key,
            }
            step_rewards.append(float(trajectory["reward"]))
            rows.append(
                {
                    "step": int(step["step"]),
                    "prompt_index": prompt_idx,
                    "prompt_name": step.get("prompt_name"),
                    "trajectory_index": int(trajectory["trajectory_index"]),
                    "old_reward": None if old_reward is None else float(old_reward),
                    "new_reward": float(trajectory["reward"]),
                    "delta": None if old_reward is None else float(trajectory["reward"]) - float(old_reward),
                    "parse_valid": bool(breakdown.get("parse_valid")),
                    "patch_count": len(patch_rewards),
                    "patch_reward_component_sums": component_sums,
                }
            )
        step["reward"] = float(np.mean(step_rewards)) if step_rewards else 0.0
        step["reward_mean"] = float(np.mean(step_rewards)) if step_rewards else 0.0
        step["reward_std"] = float(np.std(step_rewards)) if step_rewards else 0.0
        step["reward_min"] = float(np.min(step_rewards)) if step_rewards else 0.0
        step["reward_max"] = float(np.max(step_rewards)) if step_rewards else 0.0
        step["reward_sum"] = float(np.sum(step_rewards)) if step_rewards else 0.0
        step["sample_rewards"] = step_rewards

    new_rewards = [row["new_reward"] for row in rows]
    old_rewards = [row["old_reward"] for row in rows if row["old_reward"] is not None]
    deltas = [row["delta"] for row in rows if row["delta"] is not None]
    payload["structure_only_rescore_from"] = str(args.input_json)
    payload["structure_only_rescore_seconds"] = time.perf_counter() - start
    payload["structure_only_reward_summary"] = {
        "trajectory_count": len(rows),
        "old_reward": _summary(old_rewards),
        "new_reward": _summary(new_rewards),
        "delta": _summary(deltas),
        "parse_valid_count": int(sum(1 for row in rows if row["parse_valid"])),
        "parse_invalid_count": int(sum(1 for row in rows if not row["parse_valid"])),
        "largest_abs_deltas": sorted(
            rows,
            key=lambda item: abs(float(item["delta"] or 0.0)),
            reverse=True,
        )[:20],
        "reward_config": asdict(reward_config),
    }
    payload["run_config"] = {
        **payload.get("run_config", {}),
        "reward_config": asdict(reward_config),
        "reward_setup": "structure_only_current_reward",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    print(json.dumps(payload["structure_only_reward_summary"], indent=2))
    print(json.dumps({"event": "structure_only_rescore_complete", "output_json": str(args.output_json)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
