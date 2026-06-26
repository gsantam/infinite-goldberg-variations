#!/usr/bin/env python3
"""Export compact GRPO reward samples from trajectory manifests."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


COMPONENT_KEYS = (
    "parse_reward",
    "countdown_reward",
    "line_closure_reward",
    "bar_token_reward",
    "meter_alignment_reward",
    "meter_duration_closeness_reward",
    "bar_meter_consistency_reward",
    "bar_count_reward",
    "root_similarity_reward",
    "bass_pitch_class_reward",
    "cadence_root_reward",
    "cadence_bass_reward",
)


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def compact_sample(record: dict[str, Any]) -> dict[str, Any]:
    breakdown = record.get("reward_breakdown") or {}
    sample: dict[str, Any] = {
        "step": as_int(record.get("step")),
        "sample": as_int(record.get("sample_index")),
        "reward": as_float(record.get("reward", breakdown.get("total_reward"))),
        "total_reward": as_float(breakdown.get("total_reward", record.get("reward"))),
        "observed_bars": as_int(breakdown.get("observed_bars")),
        "validated_bars": as_int(breakdown.get("validated_bars")),
        "scored_tokens": as_int(breakdown.get("scored_tokens")),
        "kl_mean": as_float(breakdown.get("kl_mean")),
    }
    for key in COMPONENT_KEYS:
        sample[key] = as_float(breakdown.get(key))
    return sample


def load_samples(trajectories_dir: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for manifest in sorted(trajectories_dir.glob("step_*/trajectories.jsonl")):
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            samples.append(compact_sample(json.loads(line)))
    samples.sort(key=lambda row: (row["step"], row["sample"]))
    return samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trajectories-dir", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir
    trajectories_dir = args.trajectories_dir or output_dir / "trajectories"
    samples = load_samples(trajectories_dir)
    if not samples:
        raise SystemExit(f"No trajectory samples found under {trajectories_dir}")

    latest_step = max(int(sample["step"]) for sample in samples)
    destination = args.output_json or output_dir / "exports" / "grpo_reward_samples_latest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "latest_step": latest_step,
        "sample_count": len(samples),
        "samples": samples,
        "source": str(trajectories_dir),
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("generated_at_utc", "latest_step", "sample_count", "source")}, indent=2))
    print(str(destination))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
