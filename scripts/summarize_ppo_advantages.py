from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _get(mapping: dict[str, Any] | None, path: str, default: Any = None) -> Any:
    value: Any = mapping
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _round_float(value: Any, digits: int = 6) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value, digits)
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value


def _advantage_summary(step: dict[str, Any]) -> dict[str, Any]:
    summary = step.get("advantage_summary")
    if isinstance(summary, dict):
        return summary

    diagnostics = step.get("logprob_advantage_diagnostics")
    if not isinstance(diagnostics, dict):
        return {}

    raw = diagnostics.get("raw_advantage") if isinstance(diagnostics.get("raw_advantage"), dict) else {}
    normalized = (
        diagnostics.get("normalized_advantage")
        if isinstance(diagnostics.get("normalized_advantage"), dict)
        else {}
    )
    counts = diagnostics.get("advantage_counts") if isinstance(diagnostics.get("advantage_counts"), dict) else {}
    count = raw.get("count") or 0
    return {
        "raw": raw,
        "normalized": normalized,
        "positive_fraction": (counts.get("positive") / count) if count else None,
        "negative_fraction": (counts.get("negative") / count) if count else None,
        "zero_fraction": (counts.get("zero") / count) if count else None,
        "positive_mean": None,
        "negative_mean": None,
        "abs_mean": None,
    }


def summarize_steps(result_json: Path) -> list[dict[str, Any]]:
    result = json.loads(result_json.read_text())
    steps = result.get("steps")
    if not isinstance(steps, list):
        raise ValueError(f"{result_json} does not contain a list at key 'steps'")

    rows: list[dict[str, Any]] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        diagnostics = step.get("logprob_advantage_diagnostics")
        diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
        summary = _advantage_summary(step)
        raw = summary.get("raw") if isinstance(summary.get("raw"), dict) else {}
        normalized = summary.get("normalized") if isinstance(summary.get("normalized"), dict) else {}
        by_trajectory = summary.get("by_trajectory") if isinstance(summary.get("by_trajectory"), dict) else {}
        trajectory_raw_mean = (
            by_trajectory.get("raw_mean") if isinstance(by_trajectory.get("raw_mean"), dict) else {}
        )
        trajectory_raw_sum = (
            by_trajectory.get("raw_sum") if isinstance(by_trajectory.get("raw_sum"), dict) else {}
        )
        fixed_eval = step.get("fixed_eval") if isinstance(step.get("fixed_eval"), dict) else {}
        row = {
            "step": step.get("step"),
            "train_reward_mean": step.get("reward_mean"),
            "fixed_reward_mean": fixed_eval.get("reward_mean"),
            "raw_advantage_mean": raw.get("mean"),
            "raw_advantage_std": raw.get("std"),
            "raw_advantage_min": raw.get("min"),
            "raw_advantage_p05": raw.get("p05"),
            "raw_advantage_p50": raw.get("p50"),
            "raw_advantage_p95": raw.get("p95"),
            "raw_advantage_max": raw.get("max"),
            "normalized_advantage_mean": normalized.get("mean"),
            "normalized_advantage_std": normalized.get("std"),
            "positive_advantage_fraction": summary.get("positive_fraction"),
            "negative_advantage_fraction": summary.get("negative_fraction"),
            "zero_advantage_fraction": summary.get("zero_fraction"),
            "positive_advantage_mean": summary.get("positive_mean"),
            "negative_advantage_mean": summary.get("negative_mean"),
            "abs_advantage_mean": summary.get("abs_mean"),
            "trajectory_raw_advantage_mean_mean": trajectory_raw_mean.get("mean"),
            "trajectory_raw_advantage_mean_std": trajectory_raw_mean.get("std"),
            "trajectory_raw_advantage_sum_mean": trajectory_raw_sum.get("mean"),
            "trajectory_raw_advantage_sum_std": trajectory_raw_sum.get("std"),
            "advantage_log_ratio_correlation": diagnostics.get("advantage_log_ratio_correlation"),
            "normalized_advantage_log_ratio_correlation": diagnostics.get(
                "normalized_advantage_log_ratio_correlation"
            ),
            "sign_alignment_fraction": diagnostics.get("sign_alignment_fraction"),
            "positive_advantage_positive_log_ratio_fraction": diagnostics.get(
                "positive_advantage_positive_log_ratio_fraction"
            ),
            "negative_advantage_negative_log_ratio_fraction": diagnostics.get(
                "negative_advantage_negative_log_ratio_fraction"
            ),
            "post_step_approx_kl": step.get("post_step_approx_kl"),
            "post_step_clip_fraction": step.get("post_step_clip_fraction"),
        }
        rows.append({key: _round_float(value) for key, value in row.items()})
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_markdown(rows: list[dict[str, Any]], *, limit: int | None = None) -> None:
    if limit is not None:
        rows = rows[:limit]
    columns = [
        "step",
        "train_reward_mean",
        "fixed_reward_mean",
        "raw_advantage_mean",
        "raw_advantage_std",
        "positive_advantage_fraction",
        "advantage_log_ratio_correlation",
        "post_step_approx_kl",
    ]
    print("| " + " | ".join(columns) + " |")
    print("|" + "|".join("---" for _ in columns) + "|")
    for row in rows:
        print("| " + " | ".join("" if row.get(column) is None else str(row.get(column)) for column in columns) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize PPO advantage diagnostics over training steps.")
    parser.add_argument("result_json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--markdown", action="store_true", help="Print a compact Markdown table to stdout.")
    parser.add_argument("--limit", type=int, default=None, help="Limit Markdown output rows.")
    args = parser.parse_args()

    rows = summarize_steps(args.result_json)
    if args.output_csv:
        write_csv(rows, args.output_csv)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(rows, indent=2) + "\n")
    if args.markdown or (not args.output_csv and not args.output_json):
        print_markdown(rows, limit=args.limit)


if __name__ == "__main__":
    main()
