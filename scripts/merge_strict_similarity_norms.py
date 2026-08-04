#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rewards.strict_similarity import STRICT_SIMILARITY_SCORE_KEYS

STRICT_METRIC_KEYS = list(STRICT_SIMILARITY_SCORE_KEYS)


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.suffix == ".csv":
        rows: list[dict[str, Any]] = []
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                parsed: dict[str, Any] = {}
                for key, value in row.items():
                    if value in ("", None):
                        parsed[key] = value
                        continue
                    if key == "ok":
                        parsed[key] = value == "True"
                        continue
                    try:
                        parsed[key] = float(value)
                    except ValueError:
                        parsed[key] = value
                rows.append(parsed)
        return rows
    raise ValueError(f"unsupported row file extension: {path}")


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        key = (str(row.get("prefix_name", "")), str(row.get("seed", "")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values)


def summarize_prompt_norms(rows: list[dict[str, Any]], *, eps: float) -> list[dict[str, Any]]:
    prompt_names = sorted({str(row["prefix_name"]) for row in rows if row.get("ok")})
    summaries: list[dict[str, Any]] = []
    for prefix_name in prompt_names:
        prompt_rows = [row for row in rows if row.get("ok") and row["prefix_name"] == prefix_name]
        summary: dict[str, Any] = {
            "prefix_name": prefix_name,
            "n": len(prompt_rows),
        }
        for key in STRICT_METRIC_KEYS:
            values = [float(row[key]) for row in prompt_rows if isinstance(row.get(key), (int, float))]
            mean = statistics.mean(values) if values else 0.0
            std = _std(values)
            summary[f"{key}_mean"] = mean
            summary[f"{key}_std"] = std
            summary[f"{key}_std_safe"] = max(std, eps)
        summaries.append(summary)
    return summaries


def write_norms_json(path: Path, rows: list[dict[str, Any]], *, eps: float, config: dict[str, Any]) -> None:
    prompt_summaries = summarize_prompt_norms(rows, eps=eps)
    payload = {
        "config": config,
        "metrics": STRICT_METRIC_KEYS,
        "prompts": {
            row["prefix_name"]: {
                key: {
                    "mean": row[f"{key}_mean"],
                    "std": row[f"{key}_std"],
                    "std_safe": row[f"{key}_std_safe"],
                    "n": row["n"],
                }
                for key in STRICT_METRIC_KEYS
            }
            for row in prompt_summaries
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def global_summary(rows: list[dict[str, Any]], *, eps: float) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("ok")]
    summary: dict[str, Any] = {
        "n": len(ok_rows),
        "failed": len(rows) - len(ok_rows),
    }
    for key in STRICT_METRIC_KEYS:
        values = [float(row[key]) for row in ok_rows if isinstance(row.get(key), (int, float))]
        mean = statistics.mean(values) if values else 0.0
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        summary[key] = {"mean": mean, "std": std, "std_safe": max(std, eps), "n": len(values)}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("row_files", nargs="+", type=Path, help="sample_rows.jsonl or sample_rows.csv files to merge.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--std-eps", type=float, default=1e-4)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for path in args.row_files:
        rows.extend(read_rows(path))
    rows = dedupe_rows(rows)

    write_csv(args.out_dir / "sample_rows.csv", rows)
    (args.out_dir / "sample_rows.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )

    prompt_norm_rows = summarize_prompt_norms(rows, eps=args.std_eps)
    write_csv(args.out_dir / "prompt_strict_similarity_norms.csv", prompt_norm_rows)
    config = {
        "source_row_files": [str(path) for path in args.row_files],
        "std_eps": args.std_eps,
    }
    write_norms_json(args.out_dir / "prompt_strict_similarity_norms.json", rows, eps=args.std_eps, config=config)
    (args.out_dir / "global_strict_similarity_norms.json").write_text(
        json.dumps(global_summary(rows, eps=args.std_eps), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "rows": len(rows),
                "ok": sum(1 for row in rows if row.get("ok")),
                "failed": sum(1 for row in rows if not row.get("ok")),
                "prompts": len(prompt_norm_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
