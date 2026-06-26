#!/usr/bin/env python3
"""Export top GRPO trajectories by reward and harmony contribution."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tarfile
import time
from pathlib import Path
from typing import Any


HARMONY_KEYS = (
    ("root_similarity_reward", 2.0),
    ("bass_pitch_class_reward", 2.0),
    ("cadence_root_reward", 3.0),
    ("cadence_bass_reward", 3.0),
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


def harmony_raw(reward_breakdown: dict[str, Any]) -> float:
    return sum(
        weight * as_float(reward_breakdown.get(key, 0.0))
        for key, weight in HARMONY_KEYS
    )


def harmony_contribution(reward_breakdown: dict[str, Any]) -> float:
    progress = max(0.0, min(1.0, as_int(reward_breakdown.get("validated_bars")) / 32.0))
    return progress * harmony_raw(reward_breakdown)


def copy_if_present(source: str | None, destination_dir: Path) -> str | None:
    if not source:
        return None
    source_path = Path(source)
    if not source_path.exists():
        return None
    destination = destination_dir / source_path.name
    shutil.copy2(source_path, destination)
    return str(destination)


def load_records(trajectories_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for manifest in sorted(trajectories_dir.glob("step_*/trajectories.jsonl")):
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            breakdown = record.get("reward_breakdown") or {}
            records.append(
                {
                    "step": as_int(record.get("step")),
                    "sample_index": as_int(record.get("sample_index")),
                    "reward": as_float(
                        record.get("reward", breakdown.get("total_reward", 0.0))
                    ),
                    "validated_bars": as_int(breakdown.get("validated_bars")),
                    "observed_bars": as_int(breakdown.get("observed_bars")),
                    "harmony_reward_contrib": harmony_contribution(breakdown),
                    "raw_harmony_weighted_max10": harmony_raw(breakdown),
                    "root_similarity_reward": as_float(
                        breakdown.get("root_similarity_reward")
                    ),
                    "bass_pitch_class_reward": as_float(
                        breakdown.get("bass_pitch_class_reward")
                    ),
                    "cadence_root_reward": as_float(breakdown.get("cadence_root_reward")),
                    "cadence_bass_reward": as_float(breakdown.get("cadence_bass_reward")),
                    "completion_abc_path": record.get("completion_abc_path"),
                    "full_abc_path": record.get("full_abc_path"),
                    "prompt_abc_path": record.get("prompt_abc_path"),
                    "manifest_path": str(manifest),
                }
            )
    return records


def write_category(
    export_dir: Path, name: str, selected: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    category_dir = export_dir / name
    category_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for rank, record in enumerate(selected, 1):
        trajectory_dir = category_dir / (
            f"rank_{rank:02d}_step_{record['step']:06d}_"
            f"sample_{record['sample_index']:02d}_"
            f"reward_{record['reward']:.4f}_"
            f"harmony_{record['harmony_reward_contrib']:.4f}"
        )
        trajectory_dir.mkdir(parents=True, exist_ok=True)

        copied_completion = copy_if_present(record.get("completion_abc_path"), trajectory_dir)
        copied_full = copy_if_present(record.get("full_abc_path"), trajectory_dir)
        copied_prompt = copy_if_present(record.get("prompt_abc_path"), trajectory_dir)
        manifest = Path(record["manifest_path"])
        if manifest.exists():
            shutil.copy2(manifest, trajectory_dir / manifest.name)

        compact = dict(record)
        compact.update(
            {
                "rank": rank,
                "local_export_dir": str(trajectory_dir),
                "copied_completion_abc": copied_completion,
                "copied_full_abc": copied_full,
                "copied_prompt_abc": copied_prompt,
            }
        )
        (trajectory_dir / "record.json").write_text(
            json.dumps(compact, indent=2, sort_keys=True), encoding="utf-8"
        )

        rows.append(
            {
                "rank": rank,
                "step": record["step"],
                "sample": record["sample_index"],
                "reward": record["reward"],
                "validated_bars": record["validated_bars"],
                "harmony_reward_contrib": record["harmony_reward_contrib"],
                "raw_harmony_weighted_max10": record["raw_harmony_weighted_max10"],
                "root_similarity_reward": record["root_similarity_reward"],
                "bass_pitch_class_reward": record["bass_pitch_class_reward"],
                "cadence_root_reward": record["cadence_root_reward"],
                "cadence_bass_reward": record["cadence_bass_reward"],
                "completion_abc_path": record.get("completion_abc_path"),
                "full_abc_path": record.get("full_abc_path"),
                "dir": str(trajectory_dir),
            }
        )

    if rows:
        with (category_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (category_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trajectories-dir", type=Path)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    output_dir = args.output_dir
    trajectories_dir = args.trajectories_dir or output_dir / "trajectories"
    records = load_records(trajectories_dir)
    if not records:
        raise SystemExit(f"No trajectory records found under {trajectories_dir}")

    export_root = output_dir / "exports"
    export_root.mkdir(parents=True, exist_ok=True)
    name = args.name or f"top_reward_harmony_sofar_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    export_dir = export_root / name
    export_dir.mkdir(parents=True, exist_ok=False)

    top_reward = sorted(
        records,
        key=lambda row: (
            row["reward"],
            row["harmony_reward_contrib"],
            row["validated_bars"],
        ),
        reverse=True,
    )[: args.top_k]
    top_harmony = sorted(
        records,
        key=lambda row: (
            row["harmony_reward_contrib"],
            row["reward"],
            row["validated_bars"],
        ),
        reverse=True,
    )[: args.top_k]

    reward_rows = write_category(export_dir, "top_by_reward", top_reward)
    harmony_rows = write_category(export_dir, "top_by_harmony", top_harmony)
    latest_step = max(row["step"] for row in records)
    summary = {
        "latest_step": latest_step,
        "total_trajectories": len(records),
        "unique_selected": len(
            {(row["step"], row["sample_index"]) for row in top_reward + top_harmony}
        ),
        "export_dir": str(export_dir),
        "top_reward": reward_rows[:5],
        "top_harmony": harmony_rows[:5],
    }
    tar_path = Path(str(export_dir) + ".tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(export_dir, arcname=export_dir.name)
    summary["tar_path"] = str(tar_path)
    (export_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
